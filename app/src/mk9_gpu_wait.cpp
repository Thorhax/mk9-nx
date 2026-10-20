// Mortal Kombat Komplete Edition: the game's Direct3D GPU wait sleeps instead of spinning (CPU enhancement
// mk9_cpu_gpu_wait_sleep). usf4-nx's usf4_gpu_wait.cpp (itself sa2-nx's sa2_gpu_wait.cpp) moved to MK9's copy of the
// same XDK code: sub_82BF5680 has USF4's sub_821F54E0 shape and device offsets (hung bit at +11069, read-pointer word
// at +11024, 5000-tick timeout; docs/mk9-notes.md). The text below is usf4-nx's, with USF4's addresses.
//
// WHAT THE GAME DOES
//   Direct3D's waits for the GPU (fences; ring space, sub_821F8FA8) loop on one iteration function, sub_821F54E0
//   (r3 = the wait record: [0] device, [8] the read pointer seen last time, [12] the tick of its last change). One
//   iteration is a short delay loop, then: GPU already declared hung (bit 0x02 of [device+11069]) -> 0 (stop); the
//   read pointer ([[device+11024]]) moved -> remember it and its tick; no movement for 5000 ticks -> the hang handler;
//   else 1 (keep waiting). The caller re-checks its condition after each iteration.
//   Switch 0.5.1, one-tile fight at 53 fps (2026-10-13): this loop and its caller were ~22 % of the main thread, on a
//   console whose three game cores were 85-95 % busy (the ring thread and the Vulkan recording worker need them).
//
// WHAT THIS DOES
//   Before an iteration in which nothing can have changed - the same wait record as the previous iteration on this
//   thread, the read pointer still the one the record saw, and no ring progress since the previous iteration returned
//   (NotifyRingProgress in me_native_system.cpp: after each ring segment, each fence write and each interrupt) - it
//   sleeps until the ring thread reports progress, at most usf4_gpu_wait_max_us. Then the original iteration runs as
//   always. The counter is read AFTER the original returns, i.e. before the caller re-checks its condition, so a change
//   the caller could have missed always shows as progress and the next iteration does not sleep.
//
// WHY IT IS EXACT: the game reads the same memory and takes the same decisions; it only reaches them later, by at most
//   one wake-up latency (or max_us when the wake-up comes from something the ring does not report).
// RISK: a wait whose condition changes without ring progress or an interrupt gains up to max_us per iteration.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/platform.h>
#include <rex/ppc.h>

#include "cpu/sa2_enhancements.h"
#include "me/me_native_system.h"

SA2_ENHANCEMENT(g_gpu_wait_sleep, mk9_cpu_gpu_wait_sleep, sa2::enh::kCpu, true,
                "The game's Direct3D GPU wait (sub_82BF5680) sleeps until the ring thread makes progress instead of "
                "spinning");
REXCVAR_DEFINE_INT32(mk9_gpu_wait_max_us, 1000, "MK9",
                     "mk9_cpu_gpu_wait_sleep: longest sleep per iteration of the game's GPU wait, in microseconds")
    .range(50, 100000)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

constexpr uint32_t kIteration = 0x82BF5680;
constexpr uint32_t kIterationBytes = 0xD8;  // to the final branch at 0x82BF5754 (tools/guest_code_hash.py 82BF5680)
constexpr uint64_t kIterationHash = 0x853791F298FEE8EDull;

constexpr uint32_t kDeviceReadPointerWord = 11024;  // address of the word the GPU writes the read pointer back to
constexpr uint32_t kDeviceState = 11069;            // bit 0x02: GPU declared hung

inline uint32_t PhysOff(uint32_t a) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  return a >= 0xE0000000u ? 0x1000u : 0u;
#else
  (void)a;
  return 0u;
#endif
}
inline uint32_t Ld32(const uint8_t* base, uint32_t a) {
  uint32_t v;
  std::memcpy(&v, base + a + PhysOff(a), 4);
  return __builtin_bswap32(v);
}
inline uint8_t Ld8(const uint8_t* base, uint32_t a) { return base[a + PhysOff(a)]; }

// The previous iteration on this thread: its record, the read pointer it had seen, and the progress counter when it
// returned. (SA2's record also holds the wait's start time; USF4's iteration does not use one, so the read pointer
// [8] tells a new wait in the same stack slot apart instead. MK9: the same.)
thread_local uint32_t t_record = 0;
thread_local uint32_t t_seen = 0;
thread_local uint32_t t_progress = 0;

std::atomic<int> g_state{0};  // 0 unknown, 1 on, -1 off (code check failed or key off); set by the first call

}  // namespace

REX_EXTERN(__imp__sub_82BF5680);
REX_HOOK_RAW(sub_82BF5680) {
  int state = g_state.load(std::memory_order_relaxed);
  if (state == 0) {
    state = (g_gpu_wait_sleep.active &&
             sa2::enh::GuestCodeMatches(base, kIteration, kIterationBytes, kIterationHash, "mk9_cpu_gpu_wait_sleep"))
                ? 1
                : -1;
    g_state.store(state, std::memory_order_relaxed);
  }
  if (state < 0) {
    __imp__sub_82BF5680(ctx, base);
    return;
  }
  const uint32_t record = ctx.r3.u32;
  const uint32_t seen = Ld32(base, record + 8);
  if (record == t_record && seen == t_seen) {
    const uint32_t device = Ld32(base, record);
    if (device && !(Ld8(base, device + kDeviceState) & 0x02)) {
      const uint32_t word = Ld32(base, device + kDeviceReadPointerWord);
      if (word && Ld32(base, word) == seen && me::native::RingProgress() == t_progress) {
        static const uint32_t max_us = uint32_t(REXCVAR_GET(mk9_gpu_wait_max_us));
        const auto t0 = std::chrono::steady_clock::now();
        me::native::WaitRingProgress(t_progress, max_us);
        g_gpu_wait_sleep.Use();
        g_gpu_wait_sleep.Add(uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - t0)
                                          .count()));
      }
    }
  }
  __imp__sub_82BF5680(ctx, base);
  t_record = record;
  t_seen = Ld32(base, record + 8);
  t_progress = me::native::RingProgress();
}

namespace {
const bool g_unit = [] {
  g_gpu_wait_sleep.amount_unit = "us asleep";
  return true;
}();
}  // namespace
