/**
 * @file        audio/switch/switch_audio_system.cpp
 * @brief       Audio output for Nintendo Switch through libnx audout
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/audio/switch/switch_audio_system.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include <rex/assert.h>
#include <rex/audio/conversion.h>
#include <rex/audio/flags.h>
#include <rex/cvar.h>
#include <rex/logging.h>

#include <switch.h>

// The output hold (windowed_app_context_switch.cpp): a startup video has the device.
extern "C" bool RexSwitchOutputHeld(void);

// Passive counters: the audio thread neither writes reports nor pauses the game.
extern "C" void RexSwitchPerfAdd(unsigned id, u64 value);
extern "C" void RexSwitchPerfMax(unsigned id, u64 value);

REXCVAR_DEFINE_BOOL(audio_mute, false, "Audio", "Mute audio output");
REXCVAR_DEFINE_BOOL(audio_switch_pump, true, "Audio",
                    "Ask the game for one frame every 5.333 ms (187.5 Hz pump, like "
                    "MarathonRecomp-NX) instead of 4 in a row when each audout buffer is consumed, which "
                    "left the game's audio server voice without a packet (robotic audio); "
                    "read when the audio is opened");
REXCVAR_DEFINE_INT32(audio_switch_frames_in_queue, 10, "Audio",
                     "With audio_switch_pump: game frames kept queued before audout "
                     "(cushion; 10 = 53 ms, 6 up to build 105). A longer queue lets the game wait without cutting the "
                     "output, at the cost of latency: sa2_audio_wait_server_ms waits up to 30 ms; read "
                     "when the audio is opened");

namespace rex::audio::nx {

namespace {

// Guest frames per audout buffer, and buffers kept queued in the device.
// 4 frames are 1024 samples (21.3 ms), the block size MarathonRecomp-NX found
// to be safe on the shared cores; three buffers cap the latency at ~64 ms.
constexpr uint32_t kFramesPerBuffer = 4;
constexpr uint32_t kBufferCount = 3;
constexpr uint32_t kBufferSamples = kFramesPerBuffer * SwitchAudioDriver::kChannelSamples;
constexpr uint32_t kBufferBytes = kBufferSamples * 2 * sizeof(s16);
// audout wants page-aligned sample memory.
constexpr uint32_t kBufferAllocBytes = (kBufferBytes + 0xFFF) & ~0xFFFu;

// Above guest threads (0x3B) so the device is never starved, like the audio
// pump of MarathonRecomp-NX (0x2B).
constexpr int kOutputThreadPriority = 0x2B;
constexpr size_t kOutputThreadStack = 0x10000;
// Bounded so the thread notices shutdown even if the device stops releasing.
constexpr u64 kWaitTimeoutNs = 100'000'000;
// audio_switch_pump: one frame (256 samples) every 5.333 ms with absolute deadlines. With this many
// frames queued or fewer, another one is requested on each tick, and below the minimum one more is
// requested to catch up (the system clock and the audio clock do not match exactly).
constexpr u64 kPumpIntervalNs = 1'000'000'000ull * SwitchAudioDriver::kChannelSamples / 48000;
constexpr size_t kTargetQueuedFrames = 6;
constexpr size_t kMinQueuedFrames = 2;
// mk9-nx (2026-10-21): pump ticks with the queue above its target before the oldest frame is dropped to make room
// (~0.34 s). Without it, an audout that stops releasing buffers (seen in Ryujinx) stops the game's audio engine
// altogether, and MK9 waits forever for a movie's XAudio2 voice to drain (main-menu freeze, sub_82CE9400).
constexpr uint32_t kStalledTicksBeforeDrop = 64;

}  // namespace

/* --- driver ------------------------------------------------------------------ */

SwitchAudioDriver::SwitchAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
    : AudioDriver(memory), semaphore_(semaphore) {}

SwitchAudioDriver::~SwitchAudioDriver() {
  std::lock_guard<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  }
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  }
}

void SwitchAudioDriver::SubmitFrame(uint32_t frame_ptr) {
  // Runs inside the guest callback while it holds the global critical region:
  // copy and return, never block on the device.
  const auto input_frame = memory_->TranslateVirtual<float*>(frame_ptr);
  float* output_frame;
  {
    std::lock_guard<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      output_frame = new float[kFrameSamples];
    } else {
      output_frame = frames_unused_.top();
      frames_unused_.pop();
    }
  }
  std::memcpy(output_frame, input_frame, kFrameSamples * sizeof(float));
  std::lock_guard<std::mutex> guard(frames_mutex_);
  frames_queued_.push(output_frame);
}

bool SwitchAudioDriver::MixFrameInto(float* stereo_out, const StereoFold& fold, float gain,
                                     bool release) {
  float* frame;
  {
    std::lock_guard<std::mutex> guard(frames_mutex_);
    if (frames_queued_.empty()) {
      return false;
    }
    frame = frames_queued_.front();
    frames_queued_.pop();
  }
  std::array<float, kChannelSamples * 2> folded;
  conversion::sequential_6_BE_to_interleaved_2_LE(folded.data(), frame, kChannelSamples, fold,
                                                  gain);
  for (size_t i = 0; i < folded.size(); ++i) {
    stereo_out[i] += folded[i];
  }
  {
    std::lock_guard<std::mutex> guard(frames_mutex_);
    frames_unused_.push(frame);
  }
  // Exactly one release per frame really consumed: that is what lets the guest
  // render the next one. With audio_switch_pump, PumpTick requests them at a fixed rate.
  if (release) {
    auto released = semaphore_->Release(1, nullptr);
    assert_true(released);
  }
  return true;
}

size_t SwitchAudioDriver::QueuedFrames() {
  std::lock_guard<std::mutex> guard(frames_mutex_);
  return frames_queued_.size();
}

bool SwitchAudioDriver::DropOldestFrame() {
  std::lock_guard<std::mutex> guard(frames_mutex_);
  if (frames_queued_.empty()) {
    return false;
  }
  frames_unused_.push(frames_queued_.front());
  frames_queued_.pop();
  return true;
}

void SwitchAudioDriver::RequestFrame() {
  // The semaphore has a maximum: if the game does not serve the requests, the excess ones are lost.
  semaphore_->Release(1, nullptr);
}

/* --- system ------------------------------------------------------------------ */

struct SwitchAudioSystem::Output {
  Thread thread{};
  bool thread_created = false;
  std::atomic<bool> running{false};
  bool pump = false;  // audio_switch_pump, read when audio is opened
  uint32_t stalled_ticks = 0;   // mk9-nx: PumpTick ticks with the queue above frames_in_queue
  uint64_t dropped_frames = 0;  // mk9-nx: frames dropped because audout stopped consuming
  size_t frames_in_queue = kTargetQueuedFrames;  // audio_switch_frames_in_queue, read when audio is opened
  bool audout_initialized = false;
  bool audout_started = false;
  std::array<AudioOutBuffer, kBufferCount> buffers{};
  std::array<s16*, kBufferCount> samples{};
  std::array<float, kBufferSamples * 2> mix{};
};

std::unique_ptr<AudioSystem> SwitchAudioSystem::Create(
    runtime::FunctionDispatcher* function_dispatcher) {
  return std::make_unique<SwitchAudioSystem>(function_dispatcher);
}

SwitchAudioSystem::SwitchAudioSystem(runtime::FunctionDispatcher* function_dispatcher)
    : AudioSystem(function_dispatcher) {}

SwitchAudioSystem::~SwitchAudioSystem() {
  StopOutput();
}

X_STATUS SwitchAudioSystem::CreateDriver([[maybe_unused]] size_t index,
                                         rex::thread::Semaphore* semaphore,
                                         AudioDriver** out_driver) {
  assert_not_null(out_driver);
  // The device opens with the first client and then stays open, playing
  // silence while nobody renders.
  if (!output_ && !StartOutput()) {
    return X_STATUS_UNSUCCESSFUL;
  }
  auto driver = new SwitchAudioDriver(memory_, semaphore);
  {
    std::lock_guard<std::mutex> guard(drivers_mutex_);
    drivers_.push_back(driver);
  }
  *out_driver = driver;
  return X_STATUS_SUCCESS;
}

void SwitchAudioSystem::DestroyDriver(AudioDriver* driver) {
  assert_not_null(driver);
  auto switch_driver = static_cast<SwitchAudioDriver*>(driver);
  {
    std::lock_guard<std::mutex> guard(drivers_mutex_);
    drivers_.erase(std::remove(drivers_.begin(), drivers_.end(), switch_driver), drivers_.end());
  }
  delete switch_driver;
}

bool SwitchAudioSystem::StartOutput() {
  auto output = std::make_unique<Output>();
  output->pump = REXCVAR_GET(audio_switch_pump);
  output->frames_in_queue = static_cast<size_t>(std::clamp(REXCVAR_GET(audio_switch_frames_in_queue), 2, 32));
  REXLOG_INFO("SwitchAudioSystem: 187.5 Hz pump {}, {} frames queued", output->pump ? "active" : "off",
              output->frames_in_queue);
  output_ = std::move(output);

  for (size_t i = 0; i < kBufferCount; ++i) {
    output_->samples[i] = static_cast<s16*>(std::aligned_alloc(0x1000, kBufferAllocBytes));
    if (!output_->samples[i]) {
      REXLOG_ERROR("SwitchAudioSystem: out of memory for audio buffers");
      StopOutput();
      return false;
    }
    std::memset(output_->samples[i], 0, kBufferAllocBytes);
    AudioOutBuffer& buffer = output_->buffers[i];
    buffer.next = nullptr;
    buffer.buffer = output_->samples[i];
    buffer.buffer_size = kBufferAllocBytes;
    buffer.data_size = kBufferBytes;
    buffer.data_offset = 0;
  }

  // While the output is held the device is opened later, by the output thread.
  if (RexSwitchOutputHeld()) {
    REXLOG_INFO("SwitchAudioSystem: output held (startup video): the game is not heard until it ends");
  } else if (!OpenDevice()) {
    StopOutput();
    return false;
  }

  output_->running = true;
  Result rc = threadCreate(
      &output_->thread,
      [](void* arg) { static_cast<SwitchAudioSystem*>(arg)->OutputThreadMain(); }, this, nullptr,
      kOutputThreadStack, kOutputThreadPriority, -2);
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchAudioSystem: threadCreate failed: 0x{:08X}", rc);
    output_->running = false;
    StopOutput();
    return false;
  }
  output_->thread_created = true;
  rc = threadStart(&output_->thread);
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchAudioSystem: threadStart failed: 0x{:08X}", rc);
    output_->running = false;
    StopOutput();
    return false;
  }
  return true;
}

bool SwitchAudioSystem::OpenDevice() {
  Output& output = *output_;
  Result rc = audoutInitialize();
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchAudioSystem: audoutInitialize failed: 0x{:08X}", rc);
    return false;
  }
  output.audout_initialized = true;
  REXLOG_INFO("SwitchAudioSystem: audout {} Hz, {} channels, PCM format {}",
              audoutGetSampleRate(), audoutGetChannelCount(), int(audoutGetPcmFormat()));
  if (audoutGetSampleRate() != 48000 || audoutGetChannelCount() != 2 ||
      audoutGetPcmFormat() != PcmFormat_Int16) {
    REXLOG_ERROR("SwitchAudioSystem: unexpected audout format");
    return false;
  }
  rc = audoutStartAudioOut();
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchAudioSystem: audoutStartAudioOut failed: 0x{:08X}", rc);
    return false;
  }
  output.audout_started = true;
  return true;
}

void SwitchAudioSystem::StopOutput() {
  if (!output_) {
    return;
  }
  output_->running = false;
  if (output_->thread_created) {
    threadWaitForExit(&output_->thread);
    threadClose(&output_->thread);
  }
  if (output_->audout_started) {
    audoutStopAudioOut();
  }
  if (output_->audout_initialized) {
    audoutExit();
  }
  for (s16* samples : output_->samples) {
    std::free(samples);
  }
  output_.reset();
}

void SwitchAudioSystem::FillAndAppend(size_t buffer_index, bool append) {
  Output& output = *output_;
  output.mix.fill(0.0f);
  // Snapshot once, so a change mid-buffer cannot split it across two mixes.
  const StereoFold fold = GetStereoFold();
  const float gain = GetOutputGain();
  uint32_t frames_mixed = 0;
  uint32_t frames_missing = 0;
  {
    std::lock_guard<std::mutex> guard(drivers_mutex_);
    for (uint32_t frame = 0; frame < kFramesPerBuffer; ++frame) {
      float* out = output.mix.data() + frame * SwitchAudioDriver::kChannelSamples * 2;
      for (SwitchAudioDriver* driver : drivers_) {
        // A client with nothing queued contributes silence and gets no
        // release, exactly as the SDL driver does on underrun.
        if (driver->MixFrameInto(out, fold, gain, !output.pump)) {
          ++frames_mixed;
        } else {
          ++frames_missing;
        }
      }
    }
  }
  s16* samples = output.samples[buffer_index];
  bool pcm_nonzero = false;
  // How much the mix saturates before it is clipped (the crackling in handheld mode). Counting only.
  float pico = 0.0f;
  uint32_t saturated = 0;
  if (REXCVAR_GET(audio_mute)) {
    std::memset(samples, 0, kBufferBytes);
  } else {
    for (size_t i = 0; i < output.mix.size(); ++i) {
      const float value = std::isfinite(output.mix[i]) ? output.mix[i] : 0.0f;
      const float magnitude = std::fabs(value);
      pico = std::max(pico, magnitude);
      saturated += magnitude > 1.0f ? 1 : 0;
      samples[i] = static_cast<s16>(std::lrint(std::clamp(value, -1.0f, 1.0f) * 32767.0f));
      pcm_nonzero |= samples[i] != 0;
    }
  }
  if (!append) {
    return;
  }
  RexSwitchPerfAdd(21, saturated);
  RexSwitchPerfAdd(22, pico >= 0.98f ? 1 : 0);
  RexSwitchPerfMax(23, static_cast<u64>(std::min(pico, 100.0f) * 10000.0f));
  output.buffers[buffer_index].data_size = kBufferBytes;
  output.buffers[buffer_index].data_offset = 0;
  Result rc = audoutAppendAudioOutBuffer(&output.buffers[buffer_index]);
  RexSwitchPerfAdd(24, frames_mixed);
  RexSwitchPerfAdd(25, frames_missing);
  if (R_SUCCEEDED(rc)) {
    RexSwitchPerfAdd(26, pcm_nonzero ? 1 : 0);
    RexSwitchPerfAdd(27, 1);
  }
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchAudioSystem: audoutAppendAudioOutBuffer failed: 0x{:08X}", rc);
  }
}

void SwitchAudioSystem::PumpTick() {
  std::lock_guard<std::mutex> guard(drivers_mutex_);
  for (SwitchAudioDriver* driver : drivers_) {
    const size_t queued = driver->QueuedFrames();
    if (queued <= output_->frames_in_queue) {
      driver->RequestFrame();
      output_->stalled_ticks = 0;
    } else if (++output_->stalled_ticks >= kStalledTicksBeforeDrop && driver->DropOldestFrame()) {
      // mk9-nx: audout is not consuming; keep the game's audio clock running (see kStalledTicksBeforeDrop).
      driver->RequestFrame();
      if (output_->dropped_frames++ == 0) {
        REXLOG_WARN("SwitchAudioSystem: audout stopped consuming frames; dropping the oldest to keep the game's "
                    "audio running");
      }
    }
    if (queued < kMinQueuedFrames) {
      driver->RequestFrame();  // the queue has emptied: one more to catch up
    }
  }
}

// Without a device: the clients are served at the device's rate (the pump's ticks, and one buffer of frames consumed
// every 21.3 ms), so the game runs as it would with sound. Until the output hold ends, or for good.
void SwitchAudioSystem::RunSilent(bool until_released) {
  Output& output = *output_;
  const u64 pump_interval = armNsToTicks(kPumpIntervalNs);
  const u64 buffer_interval = armNsToTicks(1'000'000'000ull * kBufferSamples / 48000);
  u64 pump_deadline = armGetSystemTick() + pump_interval;
  u64 buffer_deadline = armGetSystemTick() + buffer_interval;
  while (output.running.load(std::memory_order_relaxed) && (!until_released || RexSwitchOutputHeld())) {
    const u64 now = armGetSystemTick();
    if (output.pump && now >= pump_deadline) {
      PumpTick();
      pump_deadline = std::max(pump_deadline + pump_interval, now);
    }
    if (now >= buffer_deadline) {
      FillAndAppend(0, false);
      buffer_deadline = std::max(buffer_deadline + buffer_interval, now);
    }
    const u64 wake = output.pump ? std::min(pump_deadline, buffer_deadline) : buffer_deadline;
    const u64 after = armGetSystemTick();
    if (wake > after) {
      svcSleepThread(s64(armTicksToNs(wake - after)));
    }
  }
}

void SwitchAudioSystem::OutputThreadMain() {
  Output& output = *output_;
  if (!output.audout_started) {
    RunSilent(true);
    if (!output.running.load(std::memory_order_relaxed)) {
      return;
    }
    if (!OpenDevice()) {
      REXLOG_ERROR("SwitchAudioSystem: no audio after the startup video");
      RunSilent(false);
      return;
    }
    REXLOG_INFO("SwitchAudioSystem: output released: audio on");
  }
  for (size_t i = 0; i < kBufferCount; ++i) {
    FillAndAppend(i);
  }
  const u64 interval = armNsToTicks(kPumpIntervalNs);
  u64 deadline = armGetSystemTick() + interval;
  while (output.running.load(std::memory_order_relaxed)) {
    u64 timeout_ns = kWaitTimeoutNs;
    if (output.pump) {
      const u64 now = armGetSystemTick();
      if (now >= deadline) {
        PumpTick();
        deadline += interval;
        if (now >= deadline) {
          deadline = now + interval;  // more than one tick late: no catch-up burst
        }
        continue;
      }
      timeout_ns = armTicksToNs(deadline - now);  // at most until the next deadline
    }
    AudioOutBuffer* released = nullptr;
    u32 released_count = 0;
    Result rc = audoutWaitPlayFinish(&released, &released_count, timeout_ns);
    // libnx hands back one released buffer per call; drain the rest so none is
    // left out of the rotation.
    while (R_SUCCEEDED(rc) && released && released_count) {
      const ptrdiff_t index = released - output.buffers.data();
      if (index >= 0 && index < ptrdiff_t(kBufferCount)) {
        FillAndAppend(size_t(index));
      }
      released = nullptr;
      released_count = 0;
      rc = audoutGetReleasedAudioOutBuffer(&released, &released_count);
    }
  }
}

}  // namespace rex::audio::nx
