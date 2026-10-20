/**
 * @file    gpu/guest/callbacks.cpp
 */
#include "renut_engine/gpu/guest/callbacks.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <vector>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/ppc.h>

#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/backend/render_thread.h"
#include "renut_engine/gpu/render/textures.h"
#include "renut_engine/gpu/banjo_renut_checks.h"  // banjo-nx (package 27c)

// sa2-nx (2026-10-20): see g_waiting.
REXCVAR_DEFINE_BOOL(sa2_renut_callbacks_fast, true, "SA2/Renderer",
                    "Renut's renderer: every draw, clear and resolve hook first runs the guest's GPU callbacks the "
                    "render thread has reached; with none waiting (nearly always: SA2 inserts few) it returns without "
                    "taking the two locks it took to find that out. Off: the locks every time")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace renut::gpu {

namespace {

struct Callback {
  uint64_t seq = 0;
  u32 function = 0;
  u32 context = 0;
};

std::mutex g_mutex;
std::deque<Callback> g_callbacks;  // inserted, not run yet, in order
uint64_t g_last_seq = 0;
// Set while a thread runs callbacks: they run one at a time, in order.
bool g_running = false;
// sa2-nx (2026-10-20, sa2_renut_callbacks_fast): g_callbacks.size(), kept with it under g_mutex and read without it.
// A hook that reads 0 while another thread inserts one runs it at its next call; on the inserting thread the count is
// in program order.
std::atomic<size_t> g_waiting{0};

} // namespace

void InsertCallback(u32 function, u32 context) {
  uint64_t seq;
  {
    std::lock_guard lock(g_mutex);
    seq = ++g_last_seq;
    g_callbacks.push_back({seq, function, context});
    g_waiting.store(g_callbacks.size(), std::memory_order_release);
  }
  // The linear textures of the work before it are read now, not when the
  // command list is submitted: after the callback the game may rewrite them
  // (the XMV player's frames).
  // banjo-nx (package 27c): the vertex fills too, for the same reason.
  banjo::renut_gpu::checks::QueueKickFlush();
  render_thread::QueueCallbackMarker(seq);
}

uint64_t LastCallback() {
  std::lock_guard lock(g_mutex);
  return g_last_seq;
}

std::atomic<int> g_hook_fast_paths{0};  // sa2-nx: callbacks.h

bool HookFastPathsInit() {
  const bool on = REXCVAR_GET(sa2_renut_callbacks_fast);
  g_hook_fast_paths.store(on ? 1 : -1, std::memory_order_relaxed);
  return on;
}

void RunReadyCallbacks() {
  if (HookFastPathsOn() && g_waiting.load(std::memory_order_acquire) == 0)
    return;  // sa2-nx: nothing inserted and not run
  const uint64_t reached = render_thread::CallbacksReached();
  std::vector<Callback> ready;
  {
    std::lock_guard lock(g_mutex);
    if (g_running || g_callbacks.empty() || g_callbacks.front().seq > reached)
      return;
    while (!g_callbacks.empty() && g_callbacks.front().seq <= reached) {
      ready.push_back(g_callbacks.front());
      g_callbacks.pop_front();
    }
    g_waiting.store(g_callbacks.size(), std::memory_order_release);
    g_running = true;
  }
  for (const Callback &callback : ready) {
    PPCFunc *function = rex::runtime::ResolveIndirectFunction(callback.function);
    if (!function) {
      RGPU_ERROR("InsertCallback: no guest function at {:08X}", callback.function);
      continue;
    }
    rex::ppc::GuestToHostFunction<void>(function, callback.context);
  }
  {
    std::lock_guard lock(g_mutex);
    g_running = false;
  }
  render_thread::CallbacksRun(ready.back().seq);
}

} // namespace renut::gpu
