/**
 * @file    gpu/guest/callbacks.h
 * @brief   D3DDevice_InsertCallback: guest functions the GPU calls when it
 *          reaches their point in the ring.
 *
 * Here the render thread reaches the point (render_thread.h), and a guest
 * thread runs the callback at its next D3D call (RunReadyCallbacks): the
 * callbacks are guest code, which the render thread cannot run. A fence issued
 * after a callback completes only once the callback has run, as on the
 * console, where the game may free what a callback uses once the fence has
 * passed (N&B's XMV player frees itself after its last frame's
 * IXMediaXmvPlayer_Resume). Called on the game's threads, without
 * State().mutex.
 */
#pragma once

#include <atomic>
#include <cstdint>

#include <rex/types.h>

namespace renut::gpu {

// Queues pCallback(Context) behind the work queued so far.
void InsertCallback(u32 function, u32 context);

// The last callback inserted: the fence of a KickOff issued now completes
// after it (render_thread::QueueFenceCompletion).
uint64_t LastCallback();

// Runs the callbacks the render thread has reached, in order, on this thread.
void RunReadyCallbacks();

// sa2-nx (2026-10-20, sa2_renut_callbacks_fast): the hooks' first calls (RunReadyCallbacks, FlushDeferredVertices,
// FlushRecordedVertices) return without their locks when nothing waits. A plain flag, set at the first call (1 on, -1
// off; the same value whichever thread sets it): an inline read instead of a call with a static's guard.
extern std::atomic<int> g_hook_fast_paths;
bool HookFastPathsInit();
inline bool HookFastPathsOn() {
  const int v = g_hook_fast_paths.load(std::memory_order_relaxed);
  return v ? v > 0 : HookFastPathsInit();
}

} // namespace renut::gpu
