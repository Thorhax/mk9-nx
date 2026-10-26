// Thorhax intro (thorhax_intro.h): the startup video, played while the game boots behind it.
//
// HOW
//   Start holds the SDK's output and returns; the game boots meanwhile (unseen, unheard, controllers untouched). The
//   video is MPEG-1 (thorhax_intro.mpg, decoded by pl_mpeg) and the sound raw 48 kHz stereo PCM (thorhax_intro.pcm),
//   both linked into the executable; make_intro.sh makes them from the source video. A decode thread on core 1 keeps a
//   few frames ahead in a small queue. The player thread (core 2) shows the frame due at the audio clock (samples
//   audout has played), converting it to RGBA straight into a linear libnx framebuffer; a frame that is already late
//   is dropped rather than shown late. Afterwards the framebuffer and audout are closed and the output is released:
//   the game's window gets its Vulkan swapchain and its audio output opens the device.

#include "thorhax_intro.h"

#if defined(__SWITCH__)

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <rex/logging.h>

#include <switch.h>

#include "pl_mpeg.h"

// The SDK's output hold (windowed_app_context_switch.h).
extern "C" void RexSwitchHoldOutput(void);
extern "C" void RexSwitchReleaseOutput(void);

// The video and its sound, linked in (THORHAX_INTRO_DIR comes from thorhax_intro.cmake; this file is built without
// LTO so the symbols defined here stay visible to the linker).
__asm__(
    ".pushsection .rodata.thorhax_intro, \"a\"\n"
    ".balign 16\n"
    ".global thorhax_intro_video\n"
    "thorhax_intro_video:\n"
    ".incbin \"" THORHAX_INTRO_DIR "/thorhax_intro.mpg\"\n"
    ".global thorhax_intro_video_end\n"
    "thorhax_intro_video_end:\n"
    ".balign 16\n"
    ".global thorhax_intro_audio\n"
    "thorhax_intro_audio:\n"
    ".incbin \"" THORHAX_INTRO_DIR "/thorhax_intro.pcm\"\n"
    ".global thorhax_intro_audio_end\n"
    "thorhax_intro_audio_end:\n"
    ".popsection\n");

extern "C" const uint8_t thorhax_intro_video[];
extern "C" const uint8_t thorhax_intro_video_end[];
extern "C" const uint8_t thorhax_intro_audio[];
extern "C" const uint8_t thorhax_intro_audio_end[];

namespace thorhax::intro {

namespace {

constexpr double kUnskippableSeconds = 3.0;
// Held together, at any time.
constexpr u64 kSecretSkip = HidNpadButton_Minus | HidNpadButton_Plus;
constexpr u64 kSkipButtons = HidNpadButton_A | HidNpadButton_B | HidNpadButton_X | HidNpadButton_Y |
                             HidNpadButton_Plus | HidNpadButton_Minus | HidNpadButton_L | HidNpadButton_R |
                             HidNpadButton_ZL | HidNpadButton_ZR;
// The framebuffer is the size of the default window; the system scales it to the docked output.
constexpr u32 kWidth = 1280;
constexpr u32 kHeight = 720;
// Decoded frames kept ahead of the display: room for the slower intra frames.
constexpr uint32_t kSlots = 6;
// Above the game's threads (0x3B), so its loading does not make the video stutter.
constexpr int kDecodeCore = 1;
constexpr int kPlayerCore = 2;
constexpr int kThreadPriority = 0x2C;
constexpr u32 kSampleRate = 48000;
constexpr u32 kBytesPerSample = 4;  // s16 stereo
// Four 100 ms audout buffers, refilled as they come back.
constexpr u32 kAudioBufferSamples = 4800;
constexpr u32 kAudioBufferBytes = kAudioBufferSamples * kBytesPerSample;
constexpr u32 kAudioBufferAlloc = (kAudioBufferBytes + 0xFFF) & ~0xFFFu;
constexpr size_t kAudioBufferCount = 4;
// Never longer than this, whatever the clocks say.
constexpr u64 kMaxSeconds = 20;

struct Slot {
  std::unique_ptr<uint8_t[]> y, cb, cr;
  plm_frame_t frame{};
};

struct Decoder {
  plm_t* plm = nullptr;
  Slot slots[kSlots];
  std::atomic<uint32_t> written{0};
  std::atomic<uint32_t> read{0};
  std::atomic<bool> ended{false};
  std::atomic<bool> stop{false};
  u64 max_decode_ticks = 0;
  u64 total_decode_ticks = 0;
  Thread thread{};
};

void CopyPlane(plm_plane_t& to, uint8_t* storage, const plm_plane_t& from) {
  to = from;
  to.data = storage;
  std::memcpy(storage, from.data, size_t(from.width) * from.height);
}

void DecodeMain(void* arg) {
  auto& d = *static_cast<Decoder*>(arg);
  for (uint32_t n = 0;; ++n) {
    while (n - d.read.load(std::memory_order_acquire) >= kSlots) {
      if (d.stop.load(std::memory_order_relaxed)) {
        break;
      }
      svcSleepThread(1'000'000);
    }
    if (d.stop.load(std::memory_order_relaxed)) {
      break;
    }
    const u64 start = armGetSystemTick();
    plm_frame_t* frame = plm_decode_video(d.plm);
    if (!frame) {
      break;
    }
    Slot& slot = d.slots[n % kSlots];
    slot.frame = *frame;
    CopyPlane(slot.frame.y, slot.y.get(), frame->y);
    CopyPlane(slot.frame.cb, slot.cb.get(), frame->cb);
    CopyPlane(slot.frame.cr, slot.cr.get(), frame->cr);
    const u64 ticks = armGetSystemTick() - start;
    d.max_decode_ticks = std::max(d.max_decode_ticks, ticks);
    d.total_decode_ticks += ticks;
    d.written.store(n + 1, std::memory_order_release);
  }
  d.ended.store(true, std::memory_order_release);
}

struct Audio {
  bool open = false;
  AudioOutBuffer buffers[kAudioBufferCount]{};
  void* memory[kAudioBufferCount]{};
  size_t next_byte = 0;
  u64 played_base = 0;

  static size_t TotalBytes() { return size_t(thorhax_intro_audio_end - thorhax_intro_audio); }

  // Fills a buffer with the next part of the sound and queues it; false when there is none left.
  bool Queue(AudioOutBuffer& buffer) {
    const size_t bytes = std::min<size_t>(kAudioBufferBytes, TotalBytes() - next_byte);
    if (bytes == 0) {
      return false;
    }
    std::memcpy(buffer.buffer, thorhax_intro_audio + next_byte, bytes);
    next_byte += bytes;
    buffer.next = nullptr;
    buffer.data_size = bytes;
    buffer.data_offset = 0;
    return R_SUCCEEDED(audoutAppendAudioOutBuffer(&buffer));
  }

  void Open() {
    Result rc = audoutInitialize();
    if (R_FAILED(rc)) {
      REXLOG_WARN("intro: audoutInitialize failed: 0x{:08X} (the video plays without sound)", rc);
      return;
    }
    rc = audoutStartAudioOut();
    if (R_FAILED(rc)) {
      REXLOG_WARN("intro: audoutStartAudioOut failed: 0x{:08X} (the video plays without sound)", rc);
      audoutExit();
      return;
    }
    open = true;
    if (R_FAILED(audoutGetAudioOutPlayedSampleCount(&played_base))) {
      played_base = 0;
    }
    for (size_t i = 0; i < kAudioBufferCount; ++i) {
      memory[i] = std::aligned_alloc(0x1000, kAudioBufferAlloc);
      buffers[i].buffer = memory[i];
      buffers[i].buffer_size = kAudioBufferAlloc;
      Queue(buffers[i]);
    }
  }

  void Pump() {
    if (!open) {
      return;
    }
    AudioOutBuffer* released = nullptr;
    u32 count = 0;
    while (R_SUCCEEDED(audoutGetReleasedAudioOutBuffer(&released, &count)) && count > 0 && released) {
      Queue(*released);
    }
  }

  // Seconds of sound played, or a negative value when the device cannot say.
  double PlayedSeconds() const {
    u64 played = 0;
    if (!open || R_FAILED(audoutGetAudioOutPlayedSampleCount(&played))) {
      return -1.0;
    }
    return double(played - played_base) / kSampleRate;
  }

  // Without a played-sample count, all of it queued is as close as it gets.
  bool Done() const {
    const double played = PlayedSeconds();
    return !open || (played < 0.0 ? next_byte >= TotalBytes()
                                  : played >= double(TotalBytes() / kBytesPerSample) / kSampleRate);
  }

  void Close() {
    if (!open) {
      return;
    }
    audoutStopAudioOut();
    bool flushed = false;
    audoutFlushAudioOutBuffers(&flushed);
    audoutExit();
    for (void* p : memory) {
      std::free(p);
    }
    open = false;
  }
};

// Fills the whole (linear) framebuffer with opaque black and shows it. pl_mpeg's RGBA conversion leaves the alpha
// bytes alone, so this also makes every later frame opaque.
void ShowBlack(Framebuffer& fb) {
  u32 stride = 0;
  auto* pixels = static_cast<u8*>(framebufferBegin(&fb, &stride));
  for (u32 row = 0; row < kHeight; ++row) {
    auto* line = reinterpret_cast<u32*>(pixels + size_t(row) * stride);
    std::fill(line, line + kWidth, 0xFF000000u);
  }
  framebufferEnd(&fb);
}

// The intro's own thread: everything from opening the framebuffer to giving the output back.
void PlayerMain(void*) {
  const u64 started = armGetSystemTick();
  auto seconds_since = [](u64 tick) { return double(armTicksToNs(armGetSystemTick() - tick)) / 1e9; };

  auto decoder = std::make_unique<Decoder>();
  decoder->plm = plm_create_with_memory(const_cast<uint8_t*>(thorhax_intro_video),
                                        size_t(thorhax_intro_video_end - thorhax_intro_video), 0);
  if (!decoder->plm) {
    REXLOG_WARN("intro: the video could not be opened");
    RexSwitchReleaseOutput();
    return;
  }
  plm_set_audio_enabled(decoder->plm, 0);
  const int width = plm_get_width(decoder->plm);
  const int height = plm_get_height(decoder->plm);
  const double fps = plm_get_framerate(decoder->plm);
  if (width != int(kWidth) || height != int(kHeight) || fps <= 0.0) {
    REXLOG_WARN("intro: the video is {}x{} at {} fps, not {}x{}", width, height, fps, kWidth, kHeight);
    plm_destroy(decoder->plm);
    RexSwitchReleaseOutput();
    return;
  }
  const size_t luma = size_t((width + 15) & ~15) * ((height + 15) & ~15);
  for (Slot& slot : decoder->slots) {
    slot.y = std::make_unique<uint8_t[]>(luma);
    slot.cb = std::make_unique<uint8_t[]>(luma / 4);
    slot.cr = std::make_unique<uint8_t[]>(luma / 4);
  }

  Framebuffer fb{};
  Result rc = framebufferCreate(&fb, nwindowGetDefault(), kWidth, kHeight, PIXEL_FORMAT_RGBA_8888, 2);
  if (R_SUCCEEDED(rc)) {
    rc = framebufferMakeLinear(&fb);
    if (R_FAILED(rc)) {
      framebufferClose(&fb);
    }
  }
  if (R_FAILED(rc)) {
    REXLOG_WARN("intro: no framebuffer: 0x{:08X}", rc);
    plm_destroy(decoder->plm);
    RexSwitchReleaseOutput();
    return;
  }
  ShowBlack(fb);

  hidInitialize();
  padConfigureInput(8, HidNpadStyleSet_NpadStandard);
  PadState pad{};
  padInitializeAny(&pad);

  rc = threadCreate(&decoder->thread, DecodeMain, decoder.get(), nullptr, 0x20000, kThreadPriority, kDecodeCore);
  if (R_SUCCEEDED(rc)) {
    rc = threadStart(&decoder->thread);
    if (R_FAILED(rc)) {
      threadClose(&decoder->thread);
    }
  }
  if (R_FAILED(rc)) {
    REXLOG_WARN("intro: no decode thread: 0x{:08X}", rc);
    framebufferClose(&fb);
    plm_destroy(decoder->plm);
    RexSwitchReleaseOutput();
    return;
  }
  // A full queue before the sound starts.
  while (decoder->written.load(std::memory_order_acquire) < kSlots &&
         !decoder->ended.load(std::memory_order_acquire)) {
    svcSleepThread(2'000'000);
  }

  Audio audio;
  audio.Open();
  const u64 playback_started = armGetSystemTick();
  auto clock = [&]() {
    const double played = audio.PlayedSeconds();
    return played >= 0.0 ? played : seconds_since(playback_started);
  };

  uint32_t shown = 0;
  uint32_t dropped = 0;
  const char* ending = "ended";
  double ended_at = 0.0;
  for (;;) {
    padUpdate(&pad);
    const double now = clock();
    ended_at = now;
    if ((padGetButtons(&pad) & kSecretSkip) == kSecretSkip) {
      ending = "skipped with - and +";
      break;
    }
    if (now >= kUnskippableSeconds && (padGetButtonsDown(&pad) & kSkipButtons)) {
      ending = "skipped";
      break;
    }
    audio.Pump();

    // The newest decoded frame that is due. In the background (HOME, sleep) nothing is shown; the system suspends
    // the process soon after anyway.
    const uint32_t next = decoder->read.load(std::memory_order_relaxed);
    const uint32_t ready = decoder->written.load(std::memory_order_acquire);
    const auto due = uint32_t(now * fps);
    if (next < ready && next <= due && appletGetFocusState() == AppletFocusState_InFocus) {
      const uint32_t pick = std::min(due, ready - 1);
      dropped += pick - next;
      u32 stride = 0;
      auto* pixels = static_cast<u8*>(framebufferBegin(&fb, &stride));
      plm_frame_to_rgba(&decoder->slots[pick % kSlots].frame, pixels, int(stride));
      framebufferEnd(&fb);
      ++shown;
      decoder->read.store(pick + 1, std::memory_order_release);
      continue;
    }
    if (decoder->ended.load(std::memory_order_acquire) && next == ready && audio.Done()) {
      break;
    }
    if (seconds_since(started) > double(kMaxSeconds)) {
      ending = "stopped by the time limit";
      break;
    }
    svcSleepThread(2'000'000);
  }

  decoder->stop.store(true, std::memory_order_relaxed);
  threadWaitForExit(&decoder->thread);
  threadClose(&decoder->thread);
  audio.Close();
  ShowBlack(fb);
  framebufferClose(&fb);
  plm_destroy(decoder->plm);
  // The game's swapchain and audio from here on.
  RexSwitchReleaseOutput();

  const uint32_t decoded = decoder->written.load(std::memory_order_acquire);
  REXLOG_INFO("intro: {} of {} frames shown ({} late), decode {:.1f} ms average / {:.1f} ms slowest; {} at {:.1f} s "
              "({:.1f} s after the start)",
              shown, decoded, dropped,
              decoded ? double(armTicksToNs(decoder->total_decode_ticks / decoded)) / 1e6 : 0.0,
              double(armTicksToNs(decoder->max_decode_ticks)) / 1e6, ending, ended_at, seconds_since(started));
}

Thread g_player{};

}  // namespace

void Start() {
  // Before anything can open the window or the audio device.
  RexSwitchHoldOutput();
  Result rc = threadCreate(&g_player, PlayerMain, nullptr, nullptr, 0x20000, kThreadPriority, kPlayerCore);
  if (R_SUCCEEDED(rc)) {
    rc = threadStart(&g_player);
  }
  if (R_FAILED(rc)) {
    REXLOG_WARN("intro: no player thread: 0x{:08X}", rc);
    RexSwitchReleaseOutput();
  }
}

}  // namespace thorhax::intro

#else

namespace thorhax::intro {

void Start() {}

}  // namespace thorhax::intro

#endif
