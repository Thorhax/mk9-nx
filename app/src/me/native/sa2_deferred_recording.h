// Deferred Vulkan command recording (sa2_native_deferred_recording).
//
// The ring thread translates PM4 and records Vulkan commands; recording in NVK (state flushes, descriptor
// and constant buffer pushes, pushbuffer writes) was ~15 % of that thread. With this on, the ring thread
// only queues each vkCmd* call (arguments copied, arrays into an arena) and a worker thread replays the
// queue into the same command buffers, in the same order. The ring thread waits for the queue to empty
// before anything that needs the recorded state: beginning, ending or resetting a command buffer or pool,
// submitting, updating descriptor sets, destroying or freeing any object. The image is unchanged: the same
// commands reach the driver in the same order.
#pragma once

#include <cstdint>
#include <functional>

#include <rex/ui/vulkan/device.h>

namespace sa2::native::deferred {

using Functions = rex::ui::vulkan::VulkanDevice::Functions;

// The function table the renderer uses: the real one when deferral is off, the proxy when it is on.
const Functions& Table(const Functions& real_fns);

// vkGetDeviceProcAddr through the proxy: vkCmd* functions the renderer fetches by name get wrappers too.
PFN_vkVoidFunction Proc(PFN_vkGetDeviceProcAddr real, VkDevice device, const char* name);

// Ring thread: waits until the worker has recorded everything queued (no-op when off).
void Drain();

// Ring thread: tells the worker about commands the batched mode (sa2_deferred_native_fast) still holds
// back. Call before the ring thread idles; a no-op otherwise.
void Flush();

// usf4-nx (usf4_deferred_async_submit), ring thread:
// true when the ring thread is the producer and the frame's submission goes through the worker.
bool AsyncSubmit();
// The real function table (calls that must not go through the queue, after a WaitTicket).
const Functions& Real();
// Runs `job` on the worker after everything queued so far (takes ownership); returns its ticket. Off the producer
// thread it runs at once (ticket 0).
uint64_t RunOnWorker(std::function<void()>* job);
// Waits until the worker has run everything up to `ticket` (not the whole queue). 0 = nothing to wait for.
void WaitTicket(uint64_t ticket);

}  // namespace sa2::native::deferred
