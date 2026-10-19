/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <thread>
#include <vector>

#include <rex/assert.h>
#include <rex/audio/conversion.h>
#include <rex/audio/downmix.h>
#include <rex/audio/flags.h>
#include <rex/audio/sdl/sdl_audio_driver.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/perf/counter.h>
#include <SDL3/SDL.h>

REXCVAR_DEFINE_BOOL(audio_mute, false, "Audio", "Mute audio output");
REXCVAR_DEFINE_INT32(audio_sdl_burst_frames, 0, "Audio",
                     "Diagnostic: the SDL driver pulls the game's frames N at a time, with N "
                     "releases in a row, like the Switch driver with 4-frame buffers; "
                     "0 or 1 = one by one, as usual");
REXCVAR_DEFINE_BOOL(audio_sdl_pump, false, "Audio",
                    "Diagnostic: a thread requests a game frame every 5.333 ms (like the pump of the "
                    "Switch driver) and SDL stops releasing the semaphore on consumption");
REXCVAR_DEFINE_INT32(audio_dump_output_s, 0, "Audio",
                     "Diagnostic: seconds of what the SDL driver delivers to the device (including the "
                     "silences from missing frames) that are saved to audio_output.wav; 0 = none");
REXCVAR_DEFINE_INT32(audio_dump_output_from_s, 0, "Audio",
                     "Diagnostic: seconds of SDL output skipped before the dump");

namespace rex::audio::sdl {

namespace {

// Diagnostic audio_dump_output_s and the silence report: only SDL's audio thread uses them.
struct DumpOutput {
  std::vector<float> samples;
  uint64_t skipped = 0;  // samples per channel skipped before starting
  bool written = false;
};
DumpOutput dump_output;
uint64_t frames_with_data = 0;
uint64_t frames_of_silence = 0;
std::chrono::steady_clock::time_point last_report_output{};

void RecordOutput(const float* data, int bytes, uint32_t channels) {
  const int32_t seconds = REXCVAR_GET(audio_dump_output_s);
  if (seconds <= 0 || dump_output.written || !channels || bytes <= 0) {
    return;
  }
  const size_t samples = size_t(bytes) / sizeof(float);
  const uint64_t skip = uint64_t(std::max(REXCVAR_GET(audio_dump_output_from_s), 0)) * 48000;
  if (dump_output.skipped < skip) {
    dump_output.skipped += samples / channels;
    return;
  }
  dump_output.samples.insert(dump_output.samples.end(), data, data + samples);
  if (dump_output.samples.size() < size_t(seconds) * 48000 * channels) {
    return;
  }
  dump_output.written = true;
  const auto path = rex::filesystem::GetExecutableFolder() / "audio_output.wav";
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (file) {
    const uint32_t bytes_data = uint32_t(dump_output.samples.size() * sizeof(float));
    const auto u32 = [&](uint32_t v) { file.write(reinterpret_cast<const char*>(&v), 4); };
    const auto u16 = [&](uint16_t v) { file.write(reinterpret_cast<const char*>(&v), 2); };
    file.write("RIFF", 4);
    u32(36 + bytes_data);
    file.write("WAVEfmt ", 8);
    u32(16);
    u16(3);  // IEEE float
    u16(uint16_t(channels));
    u32(48000);
    u32(48000 * channels * 4);
    u16(uint16_t(channels * 4));
    u16(32);
    file.write("data", 4);
    u32(bytes_data);
    file.write(reinterpret_cast<const char*>(dump_output.samples.data()), bytes_data);
    REXAPU_INFO("[audio] SDL output dump: {} samples per channel, {} channels, in {}",
                dump_output.samples.size() / channels, channels, path.string());
  }
  dump_output.samples = std::vector<float>();
}

void CountFrame(bool silence) {
  ++(silence ? frames_of_silence : frames_with_data);
  const auto now = std::chrono::steady_clock::now();
  if (now - last_report_output < std::chrono::seconds(10)) {
    return;
  }
  if (last_report_output.time_since_epoch().count() != 0) {
    REXAPU_INFO("[audio] SDL in 10 s: {} frames with data and {} of silence from missing frames",
                frames_with_data, frames_of_silence);
  }
  frames_with_data = 0;
  frames_of_silence = 0;
  last_report_output = now;
}

}  // namespace

SDLAudioDriver::SDLAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore)
    : AudioDriver(memory), semaphore_(semaphore) {}

SDLAudioDriver::~SDLAudioDriver() {
  if (pump_.joinable()) {
    active_pump_ = false;
    pump_.join();
  }
  assert_true(frames_queued_.empty());
  assert_true(frames_unused_.empty());
}

bool SDLAudioDriver::Initialize() {
  // Set audio category for proper OS audio handling
  SDL_SetHint(SDL_HINT_AUDIO_CATEGORY, "playback");

  // Set app name for audio device identification
  SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, "rexglue");

  if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
    REXAPU_ERROR("SDL_InitSubSystem(SDL_INIT_AUDIO) failed: {}", SDL_GetError());
    return false;
  }
  sdl_initialized_ = true;

  SDL_AudioSpec desired_spec = {};
  SDL_AudioSpec obtained_spec = {};
  desired_spec.freq = frame_frequency_;
  desired_spec.format = SDL_AUDIO_F32LE;
  desired_spec.channels = frame_channels_;
  sdl_device_channels_ = frame_channels_;
  sdl_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired_spec,
                                          SDLCallback, this);
  if (!sdl_stream_) {
    REXAPU_ERROR("SDL_OpenAudioDeviceStream() failed: {}", SDL_GetError());
    return false;
  }

  SDL_AudioDeviceID sdl_device = SDL_GetAudioStreamDevice(sdl_stream_);
  if (!sdl_device) {
    REXAPU_ERROR("SDL_GetAudioStreamDevice() failed: {}", SDL_GetError());
    return false;
  }

  if (!SDL_GetAudioDeviceFormat(sdl_device, &obtained_spec, NULL)) {
    REXAPU_WARN("SDL_GetAudioDeviceFormat() failed: {}", SDL_GetError());
    obtained_spec = desired_spec;
  }

  // A 1-channel device gets the stereo fold too, then SDL collapses to mono.
  // Handing it a 6ch stream instead would use SDL's own downmix.
  if (obtained_spec.channels <= 2) {
    SDL_DestroyAudioStream(sdl_stream_);
    sdl_stream_ = nullptr;
    desired_spec.channels = 2;
    sdl_device_channels_ = 2;
    sdl_stream_ = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &desired_spec,
                                            SDLCallback, this);
    if (!sdl_stream_) {
      REXAPU_ERROR("SDL_OpenAudioDeviceStream() stereo fallback failed: {}", SDL_GetError());
      return false;
    }
    sdl_device = SDL_GetAudioStreamDevice(sdl_stream_);
    if (!sdl_device) {
      REXAPU_ERROR("SDL_GetAudioStreamDevice() failed after stereo fallback: {}", SDL_GetError());
      return false;
    }
  }

  // The endpoint layout decides which mix the callback runs, and it is the
  // first thing worth knowing when a report says the balance is wrong on one
  // speaker setup and right on another.
  const char* device_name = SDL_GetAudioDeviceName(sdl_device);
  REXAPU_INFO("audio endpoint '{}': {} ch, {} Hz, format 0x{:04X}; submitting {} ch",
              device_name ? device_name : "?", obtained_spec.channels, obtained_spec.freq,
              static_cast<uint32_t>(obtained_spec.format), static_cast<int>(sdl_device_channels_));

  if (!SDL_ResumeAudioDevice(sdl_device)) {
    REXAPU_ERROR("SDL_ResumeAudioDevice() failed: {}", SDL_GetError());
    return false;
  }

  if (REXCVAR_GET(audio_sdl_pump)) {
    active_pump_ = true;
    pump_ = std::thread([this]() { Pump(); });
    REXAPU_INFO("audio: diagnostic pump at 187.5 Hz active");
  }

  return true;
}

void SDLAudioDriver::Pump() {
  using Clock = std::chrono::steady_clock;
  const auto interval =
      std::chrono::nanoseconds(1'000'000'000ll * channel_samples_ / frame_frequency_);
  auto deadline = Clock::now() + interval;
  while (active_pump_.load(std::memory_order_relaxed)) {
    const auto now = Clock::now();
    if (now < deadline) {
      std::this_thread::yield();  // no sleep: on Windows it can overshoot by several ms
      continue;
    }
    size_t in_queue;
    {
      std::unique_lock<std::mutex> guard(frames_mutex_);
      in_queue = frames_queued_.size();
    }
    if (in_queue <= 6) {
      semaphore_->Release(1, nullptr);
    }
    if (in_queue < 2) {
      semaphore_->Release(1, nullptr);
    }
    deadline += interval;
    if (now >= deadline) {
      deadline = now + interval;
    }
  }
}

void SDLAudioDriver::SubmitFrame(uint32_t frame_ptr) {
  const auto input_frame = memory_->TranslateVirtual<float*>(frame_ptr);
  float* output_frame;
  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    if (frames_unused_.empty()) {
      output_frame = new float[frame_samples_];
    } else {
      output_frame = frames_unused_.top();
      frames_unused_.pop();
    }
  }

  std::memcpy(output_frame, input_frame, frame_samples_ * sizeof(float));

  static uint32_t sdl_submit_count = 0;
  if (sdl_submit_count < 10) {
    REXAPU_DEBUG("SDLAudioDriver::SubmitFrame: frame_ptr={:08X} queued_count={}", frame_ptr,
                 frames_queued_.size() + 1);
    sdl_submit_count++;
  }

  {
    std::unique_lock<std::mutex> guard(frames_mutex_);
    frames_queued_.push(output_frame);
    PROFILE_BUFFER_QUEUE_DEPTH(static_cast<int64_t>(frames_queued_.size()));
  }
}

void SDLAudioDriver::Shutdown() {
  if (pump_.joinable()) {
    active_pump_ = false;
    pump_.join();
  }
  if (sdl_stream_) {
    SDL_DestroyAudioStream(sdl_stream_);
    sdl_stream_ = nullptr;
  }
  if (sdl_initialized_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    sdl_initialized_ = false;
  }
  std::unique_lock<std::mutex> guard(frames_mutex_);
  while (!frames_unused_.empty()) {
    delete[] frames_unused_.top();
    frames_unused_.pop();
  }
  while (!frames_queued_.empty()) {
    delete[] frames_queued_.front();
    frames_queued_.pop();
  }
}

void SDLAudioDriver::SDLCallback(void* userdata, SDL_AudioStream* stream, int additional_amount,
                                 [[maybe_unused]] int total_amount) {
  SCOPE_profile_cpu_f("apu");
  if (!userdata || !stream) {
    REXAPU_ERROR("SDLAudioDriver::SDLCallback called with nullptr.");
    return;
  }
  const auto driver = static_cast<SDLAudioDriver*>(userdata);
  const int sample_count =
      static_cast<int>(channel_samples_ * std::max<uint8_t>(driver->sdl_device_channels_, 1));
  const int len = static_cast<int>(sizeof(float) * sample_count);
  float* data = SDL_stack_alloc(float, sample_count);
  if (!data) {
    REXAPU_ERROR("SDLAudioDriver::SDLCallback failed to allocate {} samples", sample_count);
    return;
  }
  // Snapshot once. A change mid-callback would split the frame across two mixes.
  const StereoFold fold = GetStereoFold();
  const SurroundMix mix = GetSurroundMix();
  const float gain = GetOutputGain();
  const int32_t burst = REXCVAR_GET(audio_sdl_burst_frames);
  if (burst > 1) {
    // Diagnostic: up to N frames at once, each with its release, into a private buffer SDL feeds from.
    // The game then gets the calls in bursts, as on the Switch.
    while (additional_amount > 0) {
      if (driver->read_burst_ >= driver->burst_.size()) {
        driver->burst_.clear();
        driver->read_burst_ = 0;
        std::unique_lock<std::mutex> guard(driver->frames_mutex_);
        for (int32_t k = 0; k < burst && !driver->frames_queued_.empty(); ++k) {
          float* buffer = driver->frames_queued_.front();
          driver->frames_queued_.pop();
          const size_t start = driver->burst_.size();
          driver->burst_.resize(start + size_t(sample_count));
          if (REXCVAR_GET(audio_mute)) {
            std::fill(driver->burst_.begin() + start, driver->burst_.end(), 0.0f);
          } else if (driver->sdl_device_channels_ == 2) {
            conversion::sequential_6_BE_to_interleaved_2_LE(driver->burst_.data() + start, buffer,
                                                            channel_samples_, fold, gain);
          } else {
            conversion::sequential_6_BE_to_interleaved_6_LE(driver->burst_.data() + start, buffer,
                                                            channel_samples_, mix, gain);
          }
          driver->frames_unused_.push(buffer);
          if (!driver->active_pump_.load(std::memory_order_relaxed)) {
            auto ret = driver->semaphore_->Release(1, nullptr);
            assert_true(ret);
          }
        }
      }
      if (driver->read_burst_ >= driver->burst_.size()) {
        // Nothing queued: silence without a release, as in the normal path.
        std::memset(data, 0, len);
        if (!SDL_PutAudioStreamData(stream, data, len)) {
          break;
        }
        RecordOutput(data, len, driver->sdl_device_channels_);
        CountFrame(true);
        additional_amount -= len;
        continue;
      }
      if (!SDL_PutAudioStreamData(stream, driver->burst_.data() + driver->read_burst_, len)) {
        break;
      }
      RecordOutput(driver->burst_.data() + driver->read_burst_, len, driver->sdl_device_channels_);
      CountFrame(false);
      driver->read_burst_ += size_t(sample_count);
      additional_amount -= len;
    }
    SDL_stack_free(data);
    return;
  }
  while (additional_amount > 0) {
    static uint32_t sdl_callback_count = 0;
    std::unique_lock<std::mutex> guard(driver->frames_mutex_);
    if (driver->frames_queued_.empty()) {
      if (sdl_callback_count < 10) {
        REXAPU_DEBUG("SDLCallback: no frames queued (silence)");
        sdl_callback_count++;
      }
      std::memset(data, 0, len);
      if (!SDL_PutAudioStreamData(stream, data, len)) {
        REXAPU_ERROR("SDL_PutAudioStreamData() failed while filling silence: {}", SDL_GetError());
        break;
      }
      RecordOutput(data, len, driver->sdl_device_channels_);
      CountFrame(true);
      additional_amount -= len;
    } else {
      auto buffer = driver->frames_queued_.front();
      driver->frames_queued_.pop();
      if (REXCVAR_GET(audio_mute)) {
        std::memset(data, 0, len);
      } else {
        switch (driver->sdl_device_channels_) {
          case 2:
            conversion::sequential_6_BE_to_interleaved_2_LE(data, buffer, channel_samples_, fold,
                                                            gain);
            break;
          case 6:
            conversion::sequential_6_BE_to_interleaved_6_LE(data, buffer, channel_samples_, mix,
                                                            gain);
            break;
          default:
            assert_unhandled_case(driver->sdl_device_channels_);
            break;
        }
      }
      if (!SDL_PutAudioStreamData(stream, data, len)) {
        REXAPU_ERROR("SDL_PutAudioStreamData() failed: {}", SDL_GetError());
        driver->frames_unused_.push(buffer);
        break;
      }
      RecordOutput(data, len, driver->sdl_device_channels_);
      CountFrame(false);
      driver->frames_unused_.push(buffer);

      if (!driver->active_pump_.load(std::memory_order_relaxed)) {
        auto ret = driver->semaphore_->Release(1, nullptr);
        assert_true(ret);
      }
      additional_amount -= len;
    }
  }
  SDL_stack_free(data);
}

}  // namespace rex::audio::sdl
