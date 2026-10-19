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

#pragma once

#include <atomic>
#include <mutex>
#include <queue>
#include <stack>
#include <thread>
#include <vector>

#include <rex/audio/audio_driver.h>
#include <rex/thread.h>

#include <SDL3/SDL.h>

namespace rex::audio::sdl {

class SDLAudioDriver : public AudioDriver {
 public:
  SDLAudioDriver(memory::Memory* memory, rex::thread::Semaphore* semaphore);
  ~SDLAudioDriver() override;

  bool Initialize();
  void SubmitFrame(uint32_t frame_ptr) override;
  void Shutdown();

 protected:
  static void SDLCallback(void* userdata, SDL_AudioStream* stream, int additional_amount,
                          int total_amount);

  rex::thread::Semaphore* semaphore_ = nullptr;

  SDL_AudioStream* sdl_stream_ = nullptr;
  bool sdl_initialized_ = false;
  uint8_t sdl_device_channels_ = 0;

  static const uint32_t frame_frequency_ = 48000;
  static const uint32_t frame_channels_ = 6;
  static const uint32_t channel_samples_ = 256;
  static const uint32_t frame_samples_ = frame_channels_ * channel_samples_;
  static const uint32_t frame_size_ = sizeof(float) * frame_samples_;
  std::queue<float*> frames_queued_ = {};
  std::stack<float*> frames_unused_ = {};
  std::mutex frames_mutex_ = {};
  // Diagnostic audio_sdl_burst_frames: already converted samples from the last burst still to be
  // delivered to SDL. Only SDL's audio thread uses them.
  std::vector<float> burst_ = {};
  size_t read_burst_ = 0;
  // Diagnostic audio_sdl_pump: a thread that requests a frame every 5.333 ms while the queue holds at
  // most 6, like the Switch driver's pump; with it, SDL does not release the semaphore when consuming.
  void Pump();
  std::thread pump_ = {};
  std::atomic<bool> active_pump_{false};
};

}  // namespace rex::audio::sdl
