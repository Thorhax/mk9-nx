// Counts the frames the game presents and logs the rate every 5 s ("[fps] ..."), so test runs measure speed.
//
// sub_82BF6D58 is the game's Direct3D swap: the only caller of the kernel's VdSwap (import thunk 0x83376194 in
// generated/default/mk9_init.cpp; found 2026-10-20), so it runs once per presented frame. Its callers: sub_82BF73E0,
// sub_826B57F8, sub_82BF1D18, sub_82C0A640.
//
// From usf4-nx's usf4_frame_counter.cpp, without its "[hitch]" breakdown (that needs USF4's GPU/job/limiter wait
// hooks, which MK9 does not have yet): a frame longer than mk9_hitch_log_ms is only reported with its length.

#include <atomic>
#include <chrono>
#include <cstdint>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>

#include "sa2_capture.h"

REXCVAR_DEFINE_BOOL(mk9_fps_log, true, "MK9", "Log the game's presented frames per second every 5 s");
REXCVAR_DEFINE_INT32(mk9_hitch_log_ms, 0, "MK9", "Frames longer than this (ms) get a \"[hitch]\" line. 0 = off");

std::atomic<uint64_t> g_mk9_game_frames{0};

extern "C" void RexSwitchPerfHitch(uint64_t start, uint64_t end);  // sdk switch_perf.cpp (PC: me_pc_stubs.cpp)

namespace {

void Report() {
  using clock = std::chrono::steady_clock;
  static clock::time_point window_start = clock::now();
  static uint64_t window_frames = 0;
  static uint64_t worst_us = 0;
  static clock::time_point last = clock::now();
  const auto now = clock::now();
  const uint64_t dt = uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - last).count());
  last = now;
  if (dt > worst_us) worst_us = dt;
  static const uint64_t hitch_us = uint64_t(REXCVAR_GET(mk9_hitch_log_ms)) * 1000;
  // usf4-nx (0.7.4): game frames over 30 ms are stack-sampling windows too (profiling builds).
  if (dt > 30000) {
#if defined(__aarch64__)
    uint64_t end = 0;
    asm volatile("mrs %0, cntpct_el0" : "=r"(end));  // libnx's armGetSystemTick, 19.2 MHz
    const uint64_t length = dt * 192 / 10;
    RexSwitchPerfHitch(end > length ? end - length : 0, end);
#endif
  }
  if (hitch_us && dt >= hitch_us) {
    REXLOG_INFO("[hitch] frame {} took {:.1f} ms", g_mk9_game_frames.load(std::memory_order_relaxed), dt / 1000.0);
  }
  ++window_frames;
  const double elapsed = std::chrono::duration<double>(now - window_start).count();
  if (elapsed >= 5.0) {
    REXLOG_INFO("[fps] {:.1f} fps over {:.1f} s ({} frames), longest frame {:.1f} ms, total {}", window_frames / elapsed,
                elapsed, window_frames, worst_us / 1000.0, g_mk9_game_frames.load(std::memory_order_relaxed));
    window_start = now;
    window_frames = 0;
    worst_us = 0;
  }
}

}  // namespace

extern "C" REX_FUNC(__imp__sub_82BF6D58);
REX_HOOK_RAW(sub_82BF6D58) {
  sa2::capture::BeforeFlip();  // sa2_video_lockstep: the recorder takes every frame (sa2-nx's recorder)
  __imp__sub_82BF6D58(ctx, base);
  g_mk9_game_frames.fetch_add(1, std::memory_order_relaxed);
  g_sa2_game_frames.fetch_add(1, std::memory_order_relaxed);
  if (REXCVAR_GET(mk9_fps_log)) Report();
}
