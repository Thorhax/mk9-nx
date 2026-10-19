// SA2 - the game's D3D wait for the GPU without spinning (native renderer).
//
// WHY (Switch stack profile)
//   sub_8222FA98 is the D3D check "has the GPU read pointer moved since the last iteration":
//   32 db16cyc delays, then it compares the word the GPU writes the
//   read pointer back to ([[device+10768]]) with the value seen last time ([r3+8]), and after 5000 ticks
//   without change declares the GPU hung. Its callers loop on it. On the console it was ~22 % of the busy
//   game thread (plus ~6 % in the caller), on a 3-core machine where the ring thread needs that core.
//
// WHAT IT DOES
//   If the read pointer is unchanged, it waits at most sa2_wait_ring_max_us for the ring thread
//   to write it back again (NotifyRingProgress), then calls the original, which decides as before. The game
//   sees the same values, only later. With bit 0x04 of [device+10813] (GPU already declared hung) or
//   without the native renderer, it does not wait.

#include "me_native_system.h"

#include "cpu/sa2_enhancements.h"

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>

REXCVAR_DEFINE_BOOL(sa2_wait_blocking_ring, false, "SA2 renderer",
                    "Native renderer: the game's D3D sleeps while waiting for the ring thread to advance "
                    "instead of spinning in sub_8222FA98")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_notify_no_lock, false, "SA2 renderer",
                    "Ring thread: NotifyRingProgress takes the progress mutex only when the game's D3D wait is "
                    "registered as sleeping (sequentially consistent handshake on both sides) instead of on every call "
                    "(one per fence write and per ring segment). Same wake-ups")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_wait_ring_max_us, 2000, "SA2 renderer",
                     "Native renderer: longest sleep per iteration of the game's D3D GPU wait, in microseconds")
    .range(100, 100000)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// sa2-nx: the ring thread's progress notification without the mutex when nobody sleeps on it (the same handshake as
// sa2_notify_no_lock, as an SA2 enhancement key). Exact: same wake-ups, the counter is only an event count.
SA2_ENHANCEMENT(g_notify_lockfree, sa2_cpu_ring_notify_lockfree, sa2::enh::kCpu, true,
                "Ring thread: wake the game's GPU wait without taking a mutex when nobody is sleeping on it");

namespace me::native {
namespace {

std::mutex g_progress_mutex;
std::condition_variable g_progress_cv;
std::atomic<uint32_t> g_progress{0};
std::atomic<int> g_waiting{0};

}  // namespace

uint32_t RingProgress() { return g_progress.load(std::memory_order_seq_cst); }

// Sleeps until the ring thread reports progress after `seen` (NotifyRingProgress), or max_us. Registers as a waiter
// under the mutex and checks the counter after registering (the other half of the handshake in NotifyRingProgress).
bool WaitRingProgress(uint32_t seen, uint32_t max_us) {
  std::unique_lock<std::mutex> lock(g_progress_mutex);
  g_waiting.fetch_add(1, std::memory_order_seq_cst);
  const bool progressed = g_progress_cv.wait_for(lock, std::chrono::microseconds(max_us), [seen] {
    return g_progress.load(std::memory_order_seq_cst) != seen;
  });
  g_waiting.fetch_sub(1, std::memory_order_seq_cst);
  return progressed;
}

// The ring thread bumps the counter after writing the read pointer back, and checks for waiters under the
// lock (an unlocked Dekker handshake lost wake-ups on ARM).
void NotifyRingProgress() {
  static const bool no_lock = REXCVAR_GET(sa2_notify_no_lock) || g_notify_lockfree.active;
  if (no_lock) g_notify_lockfree.Use();
  if (no_lock) {
    // Dekker handshake, sequentially consistent on both sides: the waiter registers in g_waiting and then checks
    // g_progress (hook below); this side bumps g_progress and then reads g_waiting. One of the two sees the other's
    // write. If a waiter is registered, the lock/unlock below waits for it to be inside the condition variable (it holds
    // the mutex from registering until wait_for releases it), so the notify cannot be lost.
    g_progress.fetch_add(1, std::memory_order_seq_cst);
    if (g_waiting.load(std::memory_order_seq_cst) <= 0) return;
    { std::lock_guard<std::mutex> lock(g_progress_mutex); }
    g_progress_cv.notify_all();
    return;
  }
  g_progress.fetch_add(1, std::memory_order_acq_rel);
  bool notify;
  {
    std::lock_guard<std::mutex> lock(g_progress_mutex);
    notify = g_waiting.load(std::memory_order_acquire) > 0;
  }
  if (notify) g_progress_cv.notify_all();
}

}  // namespace me::native

// banjo-nx: Mass Effect's hooks on its D3D busy-waits (sub_8222FA98 ring space, sub_822FE760 render-thread
// handoff, sub_82811750 occlusion poll) were here, with its device offsets (+10768 read-pointer word, +10813
// state). Banjo's equivalents are still to be identified (its device layout is 280 bytes further on for the
// shader fields: +12696 vs +12416), so only NotifyRingProgress above is kept.
