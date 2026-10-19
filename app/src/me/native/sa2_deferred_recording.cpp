// Deferred Vulkan command recording: see sa2_deferred_recording.h.
#include "sa2_deferred_recording.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <functional>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>

#include <rex/platform.h>
#if REX_PLATFORM_SWITCH
#include <switch.h>
#else
#include <thread>
#endif

#include <rex/cvar.h>
#include <rex/logging.h>

REXCVAR_DEFINE_BOOL(sa2_native_deferred_recording, false, "SA2 renderer",
                    "Ring thread queues every vkCmd* call and a worker thread records them into the same command "
                    "buffers in the same order (moves NVK recording off the ring thread; same commands, same image)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_native_deferred_recording_parts, 3, "SA2 renderer",
                     "Diagnostic bisection of sa2_native_deferred_recording: 1 = proxy function table, "
                     "2 = wrappers for the functions fetched by name")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_native_deferred_recording_naps, 64, "SA2 renderer",
                     "Worker: 5 us naps on an empty queue before it blocks on the condition variable "
                     "(each nap preempts a game thread on its core)")
    .range(0, 1000)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(sa2_native_deferred_worker_core, -2, "SA2 renderer",
                     "Switch: the deferred-recording worker's core. -2 = only the process's default core (as before; the "
                     "worker's priority, the ring thread's, is above the game threads', so it preempts whatever runs "
                     "there); 0-2 = only that core; -1 = any of cores 0-2 (ideal core 2)")
    .range(-2, 2)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_deferred_native_fast, false, "SA2 renderer",
                    "Queue: the producer publishes commands in batches (sa2_deferred_native_batch) instead "
                    "of one seq_cst store + wake check per command, keeps its own copy of the worker's read index "
                    "and the worker stores it every 16 commands (less cache line ping-pong between the cores). "
                    "Same commands, same order; false = as before")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_deferred_native_batch, 16, "SA2 renderer",
                     "sa2_deferred_native_fast: commands queued before the worker is told (the queue is also "
                     "published before the ring thread waits, drains or idles)")
    .range(1, 256)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_deferred_native_update, false, "SA2 renderer",
                    "vkUpdateDescriptorSets is queued too (its write arrays deep-copied into the arena) instead of "
                    "draining the worker (~70 drains per frame, each a spin-yield of the ring thread). The update runs "
                    "on the worker at the same position of the command stream; every submit, destroy and reset still "
                    "drains first")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(usf4_deferred_slots_log2, 10, "SA2 renderer",
                     "Deferred recording queue length as a power of two (10 = 1024 calls, 128 KB; 12 = 4096, 512 KB). "
                     "A USF4 fight frame queues ~4,900 calls: a short queue makes the ring thread and the worker take "
                     "turns")
    .range(8, 14)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(usf4_deferred_async_submit, false, "SA2 renderer",
                    "usf4-nx: the end of the frame does not wait for the recording worker: the work submission and the "
                    "output paint run on the worker, in queue order, and reusing a slot waits only until the worker "
                    "has passed that slot's submission (its ticket). The ring thread starts the next frame while the "
                    "worker records the end of the previous one. Same commands, same order")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace sa2::native::deferred {
namespace {

// ---------------------------------------------------------------------------------------------------------------
// Queue: single producer (the ring thread), single consumer (the worker).
// usf4-nx: the queue length is a cvar (usf4_deferred_slots); 1024 before (128 KB: the process runs within a few MB
// of its limit). 1024 calls are ~1/5 of a USF4 fight frame (~4,900 calls): ring thread and worker took turns.
size_t g_slots = size_t(1) << 10;
constexpr size_t kLoad = 112;
struct alignas(16) Slot {
  void (*run)(void*) = nullptr;
  alignas(16) unsigned char load[kLoad];
};
Slot* g_queue = nullptr;
alignas(64) std::atomic<uint64_t> g_written{0};
alignas(64) std::atomic<uint64_t> g_read{0};
uint64_t g_written_producer = 0;
// sa2_deferred_native_fast: what the worker was last told (g_written), the producer's copy of the
// worker's read index (it only reloads the shared one when the queue looks full) and the batch size.
uint64_t g_published = 0;
uint64_t g_read_view = 0;
bool g_fast = false;
uint32_t g_batch = 1;
uint64_t g_batches = 0, g_updates = 0, g_direct_updates = 0;
std::atomic<bool> g_sleeping{false};
std::mutex g_mutex;
std::condition_variable g_cv;

std::atomic<bool> g_active{false};
std::atomic<bool> g_pinned_producer{false};
thread_local bool t_producer = false;  // not std::thread::id: SDK threads are not pthreads

// Arrays of the queued calls (barriers, regions, viewports...). Reset whenever the queue is empty.
constexpr size_t kArena = size_t(512) << 10;
uint8_t* g_arena = nullptr;
size_t g_arena_usage = 0;
// usf4-nx: the arena as a ring. With usf4_deferred_async_submit the queue rarely empties (the frame's submission no
// longer drains it), so space is given back as the worker passes the calls that used it: one marker per call with
// arrays (its queue index and where its block ends). g_arena_usage is unused then.
bool g_arena_ring = false;
uint64_t g_arena_head = 0, g_arena_tail = 0;  // virtual offsets (mod kArena)
struct ArenaMark { uint64_t index, end; };
ArenaMark* g_arena_marks = nullptr;  // g_slots entries: at most one per queued call
uint64_t g_marks_first = 0, g_marks_end = 0;
uint64_t g_arena_waits = 0;

// Report. usf4-nx: drains by cause (the call that needed the worker to finish), with their waits.
struct DrainCause { const char* what = nullptr; uint64_t drains = 0, waits = 0, ns = 0; };
DrainCause g_causes[32];
DrainCause& Cause(const char* what) {
  for (DrainCause& c : g_causes) {
    if (c.what == what) return c;
    if (!c.what) { c.what = what; return c; }
  }
  return g_causes[31];
}
uint64_t g_queued = 0, g_direct = 0, g_drains = 0, g_drains_with_wait = 0, g_ns_wait = 0;
auto g_report = std::chrono::steady_clock::now();

// A short pause of the queue's spin loops. On the Switch a sleep of a few microseconds is what it says. On Windows
// sleep_for rounds up to the timer period (1-15.6 ms), which made every nap, drain and full-queue wait a stall
// (sa2-nx PC runs: frames of 100-160 ms); there it yields instead. The console path is unchanged.
inline void ShortPause(std::chrono::microseconds us) {
#if REX_PLATFORM_SWITCH
  std::this_thread::sleep_for(us);
#else
  (void)us;
  std::this_thread::yield();
#endif
}

inline bool IsProducer() {
  return t_producer && g_active.load(std::memory_order_relaxed);
}

void Wake() {
  if (g_sleeping.load(std::memory_order_seq_cst)) {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_cv.notify_one();
  }
}

// Tells the worker about everything queued so far (the batched mode may hold a few commands back).
void PublishAlready() {
  g_published = g_written_producer;
  ++g_batches;
  g_written.store(g_written_producer, std::memory_order_seq_cst);
  Wake();
}

uint64_t g_full_waits = 0, g_ns_full = 0;
// usf4-nx: calls NVK records through its meta paths (render passes with temporary image views on the graphics queue).
uint64_t g_n_copy = 0, g_n_copy_regions = 0, g_n_blit = 0, g_n_resolve = 0, g_n_clear_color = 0, g_n_clear_depth = 0;

Slot& Reserve() {
  if (g_fast) {
    if (g_written_producer - g_read_view >= g_slots) {
      g_read_view = g_read.load(std::memory_order_acquire);
    }
    if (g_written_producer - g_read_view >= g_slots) {
      if (g_published != g_written_producer) PublishAlready();  // the worker cannot free slots it was not told about
      const auto t0 = std::chrono::steady_clock::now();
      ++g_full_waits;
      for (;;) {
        g_read_view = g_read.load(std::memory_order_acquire);
        if (g_written_producer - g_read_view < g_slots) break;
        Wake();
        ShortPause(std::chrono::microseconds(20));
      }
      g_ns_full += uint64_t(
          std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    }
    return g_queue[g_written_producer & (g_slots - 1)];
  }
  while (g_written_producer - g_read.load(std::memory_order_acquire) >= g_slots) {
    Wake();
    ShortPause(std::chrono::microseconds(20));
  }
  return g_queue[g_written_producer & (g_slots - 1)];
}

void Publish() {
  ++g_written_producer;
  ++g_queued;
  if (g_fast) {
    if (g_written_producer - g_published >= g_batch) PublishAlready();
    return;
  }
  g_written.store(g_written_producer, std::memory_order_seq_cst);
  Wake();
}

int32_t g_priority_producer = -1;

void Worker() {
  uint64_t read = 0;
  uint32_t empty = 0;
  const bool fast = g_fast;
  for (;;) {
    const uint64_t written = g_written.load(std::memory_order_acquire);
    if (read < written) {
      if (fast) {
        // The producer only needs the read index when the queue is full or to drain: every 16 commands and at the
        // end of the burst (drain waits for g_read == g_written_producer, which the final store provides).
        do {
          Slot& r = g_queue[read & (g_slots - 1)];
          r.run(r.load);
          ++read;
          if ((read & 15) == 0) g_read.store(read, std::memory_order_release);
        } while (read < written);
        g_read.store(read, std::memory_order_release);
      } else {
        do {
          Slot& r = g_queue[read & (g_slots - 1)];
          r.run(r.load);
          ++read;
          g_read.store(read, std::memory_order_release);
        } while (read < written);
      }
      empty = 0;
      continue;
    }
    // A short spin (the ring publishes in bursts), then sleep: the three cores are shared with the game.
    static const uint32_t naps = uint32_t(REXCVAR_GET(sa2_native_deferred_recording_naps));
    if (++empty < naps) {
      ShortPause(std::chrono::microseconds(5));
      continue;
    }
    std::unique_lock<std::mutex> lock(g_mutex);
    g_sleeping.store(true, std::memory_order_seq_cst);
    if (g_written.load(std::memory_order_seq_cst) == read) {
      g_cv.wait_for(lock, std::chrono::milliseconds(2));
    }
    g_sleeping.store(false, std::memory_order_seq_cst);
    empty = 0;
  }
}

void Report();

void DrainInternal(const char* cause) {
  static uint32_t traces = 0;
  if (traces < 6) {
    ++traces;
    REXLOG_INFO("[native] deferred recording: drain {}: written {} read {}", traces, g_written_producer,
                g_read.load(std::memory_order_acquire));
  }
  if (g_fast && g_published != g_written_producer) PublishAlready();
  if (g_read.load(std::memory_order_acquire) != g_written_producer) {
    const auto t0 = std::chrono::steady_clock::now();
    Wake();
    // Not yield(): on Horizon it only gives way to threads of the SAME priority on the SAME core, so a worker
    // on this core with a lower priority would never run (the first builds of this worker hung exactly like that).
    for (uint32_t laps = 0; g_read.load(std::memory_order_acquire) != g_written_producer; ++laps) {
      if (laps < 32) std::this_thread::yield();
      else ShortPause(std::chrono::microseconds(20));
    }
    const uint64_t ns = uint64_t(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    g_ns_wait += ns;
    ++g_drains_with_wait;
    DrainCause& c = Cause(cause);
    ++c.waits;
    c.ns += ns;
  }
  ++g_drains;
  ++Cause(cause).drains;
  g_read_view = g_written_producer;  // the queue is empty (the worker has executed everything)
  g_arena_usage = 0;
  g_arena_tail = g_arena_head;
  g_marks_first = g_marks_end;
  Report();
}

// Every 10 s, from the ring thread (drains, and Flush after each ring segment: with usf4_deferred_async_submit
// there may be no drain for minutes).
void Report() {
  const auto now = std::chrono::steady_clock::now();
  if (now - g_report >= std::chrono::seconds(10)) {
    g_report = now;
    REXLOG_INFO("[native] deferred recording (10 s): {} calls queued, {} direct on the ring thread; {} "
                "drains ({} waiting for the worker, {:.1f} ms in total); {} publishes; vkUpdateDescriptorSets "
                "{} queued, {} direct",
                g_queued, g_direct, g_drains, g_drains_with_wait, double(g_ns_wait) / 1e6, g_batches,
                g_updates, g_direct_updates);
    std::string causes;
    for (DrainCause& c : g_causes) {
      if (!c.what || !c.drains) continue;
      causes += fmt::format(" {} {} ({} waited, {:.1f} ms);", c.what, c.drains, c.waits, double(c.ns) / 1e6);
      c.drains = c.waits = c.ns = 0;
    }
    REXLOG_INFO("[native] deferred recording drains by cause (10 s):{} arena waits (ring) {}; queue full {} times "
                "({:.1f} ms waiting for the worker)", causes, g_arena_waits, g_full_waits, double(g_ns_full) / 1e6);
    g_arena_waits = g_full_waits = g_ns_full = 0;
    REXLOG_INFO("[native] deferred recording, meta-path calls (10 s): vkCmdCopyImage {} ({} regions), "
                "vkCmdBlitImage {}, vkCmdResolveImage {}, vkCmdClearColorImage {}, vkCmdClearDepthStencilImage {}",
                g_n_copy, g_n_copy_regions, g_n_blit, g_n_resolve, g_n_clear_color, g_n_clear_depth);
    g_n_copy = g_n_copy_regions = g_n_blit = g_n_resolve = g_n_clear_color = g_n_clear_depth = 0;
    g_queued = g_direct = g_drains = g_drains_with_wait = g_ns_wait = 0;
    g_batches = g_updates = g_direct_updates = 0;
  }
}

// One arena block for all arrays of one call (a drain inside would reuse memory already copied for it).
void ArenaRelease() {
  const uint64_t read = g_read.load(std::memory_order_acquire);
  while (g_marks_first != g_marks_end && g_arena_marks[g_marks_first % g_slots].index < read) {
    g_arena_tail = g_arena_marks[g_marks_first % g_slots].end;
    ++g_marks_first;
  }
  if (g_marks_first == g_marks_end && read == g_written_producer) g_arena_tail = g_arena_head;
}

uint8_t* ArenaRing(size_t bytes) {
  // A block never wraps: when it does not fit before the end, the rest of the lap is skipped.
  const auto fits = [&] {
    const uint64_t at = g_arena_head % kArena;
    const uint64_t need = at + bytes > kArena ? (kArena - at) + bytes : bytes;
    return g_arena_head + need - g_arena_tail <= kArena;
  };
  ArenaRelease();
  if (!fits()) {
    ++g_arena_waits;
    if (g_fast && g_published != g_written_producer) PublishAlready();
    for (uint32_t laps = 0; ArenaRelease(), !fits(); ++laps) {
      Wake();
      if (laps < 32) std::this_thread::yield();
      else ShortPause(std::chrono::microseconds(20));
    }
  }
  if (g_arena_head % kArena + bytes > kArena) g_arena_head += kArena - g_arena_head % kArena;
  uint8_t* p = g_arena + g_arena_head % kArena;
  g_arena_head += bytes;
  // The call that uses this block is the next one queued (index g_written_producer).
  g_arena_marks[g_marks_end % g_slots] = {g_written_producer, g_arena_head};
  ++g_marks_end;
  return p;
}

uint8_t* Arena(size_t bytes) {
  bytes = (bytes + 15) & ~size_t(15);
  if (bytes > kArena) return nullptr;
  if (g_arena_ring) return ArenaRing(bytes);
  if (g_arena_usage + bytes > kArena) DrainInternal("arena full");
  uint8_t* p = g_arena + g_arena_usage;
  g_arena_usage += bytes;
  return p;
}
template <typename T>
size_t Size(uint32_t n) {
  return (sizeof(T) * n + 15) & ~size_t(15);
}
template <typename T>
const T* Copy(uint8_t*& cursor, const T* src_data, uint32_t n) {
  if (!src_data || !n) return src_data;
  T* target = reinterpret_cast<T*>(cursor);
  std::memcpy(target, src_data, sizeof(T) * n);
  cursor += Size<T>(n);
  return target;
}
template <typename T>
bool WithPNext(const T* p, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    if (p[i].pNext) return true;
  return false;
}

bool g_no_enqueue = false;  // bisection bit 32: producer and worker set up, calls made at once

template <typename F, typename... A>
void Enqueue(F f, A... a) {
  if (g_no_enqueue) {
    f(a...);
    return;
  }
  struct Package {
    F f;
    std::tuple<A...> args;
  };
  static_assert(sizeof(Package) <= kLoad && alignof(Package) <= 16, "call too large for the queue");
  Slot& r = Reserve();
  new (r.load) Package{f, std::tuple<A...>(a...)};
  r.run = [](void* c) {
    Package* p = static_cast<Package*>(c);
    std::apply(p->f, p->args);
    p->~Package();
  };
  Publish();
}

// ---------------------------------------------------------------------------------------------------------------
// Generic wrappers, one per function of the table (and per name fetched with Proc).
template <typename T>
constexpr bool kPointerData = std::is_pointer_v<T> && std::is_const_v<std::remove_pointer_t<T>>;

enum Mode : uint8_t { kDirect = 0, kDrain = 1, kDefer = 2 };

template <size_t Id, typename PFN>
struct Wrapper;
template <size_t Id, typename R, typename... A>
struct Wrapper<Id, R (*)(A...)> {
  using Pointer = R (*)(A...);
  static constexpr bool kDeferrable = std::is_void_v<R> && !(kPointerData<A> || ...);
  static inline Pointer real = nullptr;
  static inline uint8_t mode = kDirect;
  static inline const char* name = "?";
  static R F(A... a) {
    if (IsProducer()) {
      if constexpr (kDeferrable) {
        if (mode == kDefer) {
          Enqueue(real, a...);
          return;
        }
      }
      if (mode != kDirect) {
        DrainInternal(name);
        ++g_direct;
      }
    }
    return real(a...);
  }
};

uint8_t Classify(std::string_view n, bool deferrable) {
  if (n.starts_with("vkCmd")) return deferrable ? kDefer : kDrain;
  static constexpr std::string_view kDrainNames[] = {
      "vkBeginCommandBuffer", "vkEndCommandBuffer", "vkResetCommandBuffer", "vkResetCommandPool",
      "vkAllocateCommandBuffers", "vkQueueSubmit", "vkQueueSubmit2", "vkQueueWaitIdle", "vkDeviceWaitIdle",
      "vkQueuePresentKHR", "vkUpdateDescriptorSets", "vkResetDescriptorPool"};
  for (auto k : kDrainNames)
    if (n == k) return kDrain;
  if (n.starts_with("vkDestroy") || n.starts_with("vkFree")) return kDrain;
  return kDirect;
}

template <size_t Id, typename PFN>
void Install(PFN real, PFN& target, const char* name) {
  using E = Wrapper<Id, PFN>;
  if (!real) return;
  E::real = real;
  E::name = name;
  E::mode = Classify(name, E::kDeferrable);
  if (E::mode != kDirect) target = &E::F;
}

Functions g_real;
Functions g_proxy;

// ---------------------------------------------------------------------------------------------------------------
// Calls with arrays or structures: their data is copied into the arena.
VkResult BeginCommands(VkCommandBuffer cb, const VkCommandBufferBeginInfo* info) {
  if (g_active.load(std::memory_order_relaxed) && !g_pinned_producer.load(std::memory_order_relaxed)) {
    REXLOG_INFO("[native] deferred recording: first command buffer, starting the worker");
    bool created = false;
#if REX_PLATFORM_SWITCH
    {
      s32 priority = 0;
      if (R_SUCCEEDED(svcGetThreadPriority(&priority, CUR_THREAD_HANDLE))) g_priority_producer = int32_t(priority);
    }
    // A libnx thread, not std::thread: the ring thread is a Horizon thread created by the SDK, and a pthread
    // created from it hung the whole process.
    static Thread thread;
    const int32_t core = REXCVAR_GET(sa2_native_deferred_worker_core);
    if (R_SUCCEEDED(threadCreate(&thread, [](void*) { Worker(); }, nullptr, nullptr, 256 * 1024,
                                 g_priority_producer >= 0 ? g_priority_producer : 0x2C,
                                 core >= 0 ? int(core) : core == -1 ? 2 : -2))) {
      if (core == -1) {  // any of cores 0-2, ideal core 2 (the SDK's own call for a thread's affinity)
        const Result r = svcSetThreadCoreMask(thread.handle, 2, 0x7);
        REXLOG_INFO("[native] deferred recording worker: cores 0-2 (ideal 2): {}", R_SUCCEEDED(r) ? "set" : "REFUSED");
      } else if (core >= 0) {
        REXLOG_INFO("[native] deferred recording worker: core {} only", core);
      }
      created = R_SUCCEEDED(threadStart(&thread));
    }
#else
    // banjo-nx: PC build, a plain thread (kept for the life of the process, like the Switch one).
    static std::thread thread([] { Worker(); });
    created = true;
#endif
    if (created) {
      t_producer = true;
      g_pinned_producer.store(true, std::memory_order_release);
      REXLOG_INFO("[native] deferred recording active: the thread that began the first command buffer queues, "
                  "a worker records");
    } else {
      g_active.store(false, std::memory_order_release);
      REXLOG_ERROR("[native] deferred recording DISABLED: could not create the worker");
    }
  }
  if (IsProducer()) DrainInternal("vkBeginCommandBuffer");
  return g_real.vkBeginCommandBuffer(cb, info);
}

void Barrier(VkCommandBuffer cb, VkPipelineStageFlags s, VkPipelineStageFlags d, VkDependencyFlags f, uint32_t nm,
             const VkMemoryBarrier* m, uint32_t nb, const VkBufferMemoryBarrier* b, uint32_t ni,
             const VkImageMemoryBarrier* im) {
  if (!IsProducer() || WithPNext(m, nm) || WithPNext(b, nb) || WithPNext(im, ni)) {
    if (IsProducer()) { DrainInternal("vkCmdPipelineBarrier with pNext"); ++g_direct; }
    g_real.vkCmdPipelineBarrier(cb, s, d, f, nm, m, nb, b, ni, im);
    return;
  }
  uint8_t* c = Arena(Size<VkMemoryBarrier>(nm) + Size<VkBufferMemoryBarrier>(nb) + Size<VkImageMemoryBarrier>(ni));
  const auto* m2 = Copy(c, m, nm);
  const auto* b2 = Copy(c, b, nb);
  const auto* i2 = Copy(c, im, ni);
  Enqueue(g_real.vkCmdPipelineBarrier, cb, s, d, f, nm, m2, nb, b2, ni, i2);
}

void BindSets(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout l, uint32_t first, uint32_t n,
                 const VkDescriptorSet* sets, uint32_t nd, const uint32_t* dynamic_offsets) {
  if (!IsProducer()) return g_real.vkCmdBindDescriptorSets(cb, bp, l, first, n, sets, nd, dynamic_offsets);
  uint8_t* c = Arena(Size<VkDescriptorSet>(n) + Size<uint32_t>(nd));
  const auto* s2 = Copy(c, sets, n);
  const auto* d2 = Copy(c, dynamic_offsets, nd);
  Enqueue(g_real.vkCmdBindDescriptorSets, cb, bp, l, first, n, s2, nd, d2);
}

void Constants(VkCommandBuffer cb, VkPipelineLayout l, VkShaderStageFlags st, uint32_t off, uint32_t size,
                const void* data) {
  if (!IsProducer()) return g_real.vkCmdPushConstants(cb, l, st, off, size, data);
  uint8_t* c = Arena(size);
  const auto* d2 = Copy(c, static_cast<const uint8_t*>(data), size);
  Enqueue(g_real.vkCmdPushConstants, cb, l, st, off, size, static_cast<const void*>(d2));
}

void BeginPass(VkCommandBuffer cb, const VkRenderPassBeginInfo* info, VkSubpassContents content) {
  if (!IsProducer() || info->pNext) {
    if (IsProducer()) { DrainInternal("vkCmdBeginRenderPass with pNext"); ++g_direct; }
    return g_real.vkCmdBeginRenderPass(cb, info, content);
  }
  uint8_t* c = Arena(Size<VkRenderPassBeginInfo>(1) + Size<VkClearValue>(info->clearValueCount));
  auto* i2 = const_cast<VkRenderPassBeginInfo*>(Copy(c, info, 1));
  i2->pClearValues = Copy(c, info->pClearValues, info->clearValueCount);
  Enqueue(g_real.vkCmdBeginRenderPass, cb, static_cast<const VkRenderPassBeginInfo*>(i2), content);
}

void Framings(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkViewport* v) {
  if (!IsProducer()) return g_real.vkCmdSetViewport(cb, first, n, v);
  uint8_t* c = Arena(Size<VkViewport>(n));
  Enqueue(g_real.vkCmdSetViewport, cb, first, n, Copy(c, v, n));
}
void Scissors(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkRect2D* r) {
  if (!IsProducer()) return g_real.vkCmdSetScissor(cb, first, n, r);
  uint8_t* c = Arena(Size<VkRect2D>(n));
  Enqueue(g_real.vkCmdSetScissor, cb, first, n, Copy(c, r, n));
}
void ConstantsBlend(VkCommandBuffer cb, const float k[4]) {
  if (!IsProducer()) return g_real.vkCmdSetBlendConstants(cb, k);
  uint8_t* c = Arena(Size<float>(4));
  Enqueue(g_real.vkCmdSetBlendConstants, cb, Copy(c, k, 4));
}
void CopyBuffer(VkCommandBuffer cb, VkBuffer a, VkBuffer b, uint32_t n, const VkBufferCopy* r) {
  if (!IsProducer()) return g_real.vkCmdCopyBuffer(cb, a, b, n, r);
  uint8_t* c = Arena(Size<VkBufferCopy>(n));
  Enqueue(g_real.vkCmdCopyBuffer, cb, a, b, n, Copy(c, r, n));
}
void BufferToImage(VkCommandBuffer cb, VkBuffer b, VkImage i, VkImageLayout l, uint32_t n,
                  const VkBufferImageCopy* r) {
  if (!IsProducer()) return g_real.vkCmdCopyBufferToImage(cb, b, i, l, n, r);
  uint8_t* c = Arena(Size<VkBufferImageCopy>(n));
  Enqueue(g_real.vkCmdCopyBufferToImage, cb, b, i, l, n, Copy(c, r, n));
}
void ImageToBuffer(VkCommandBuffer cb, VkImage i, VkImageLayout l, VkBuffer b, uint32_t n,
                  const VkBufferImageCopy* r) {
  if (!IsProducer()) return g_real.vkCmdCopyImageToBuffer(cb, i, l, b, n, r);
  uint8_t* c = Arena(Size<VkBufferImageCopy>(n));
  Enqueue(g_real.vkCmdCopyImageToBuffer, cb, i, l, b, n, Copy(c, r, n));
}
void ClearAttached(VkCommandBuffer cb, uint32_t na, const VkClearAttachment* a, uint32_t nr, const VkClearRect* r) {
  if (!IsProducer()) return g_real.vkCmdClearAttachments(cb, na, a, nr, r);
  uint8_t* c = Arena(Size<VkClearAttachment>(na) + Size<VkClearRect>(nr));
  const auto* a2 = Copy(c, a, na);
  const auto* r2 = Copy(c, r, nr);
  Enqueue(g_real.vkCmdClearAttachments, cb, na, a2, nr, r2);
}
void ClearColor(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearColorValue* v, uint32_t n,
                 const VkImageSubresourceRange* r) {
  ++g_n_clear_color;
  if (!IsProducer()) return g_real.vkCmdClearColorImage(cb, i, l, v, n, r);
  uint8_t* c = Arena(Size<VkClearColorValue>(1) + Size<VkImageSubresourceRange>(n));
  const auto* v2 = Copy(c, v, 1);
  const auto* r2 = Copy(c, r, n);
  Enqueue(g_real.vkCmdClearColorImage, cb, i, l, v2, n, r2);
}
void BindVertices(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBuffer* b, const VkDeviceSize* o) {
  if (!IsProducer()) return g_real.vkCmdBindVertexBuffers(cb, first, n, b, o);
  uint8_t* c = Arena(Size<VkBuffer>(n) + Size<VkDeviceSize>(n));
  const auto* b2 = Copy(c, b, n);
  const auto* o2 = Copy(c, o, n);
  Enqueue(g_real.vkCmdBindVertexBuffers, cb, first, n, b2, o2);
}

// vkUpdateDescriptorSets queued (sa2_deferred_native_update). Every array the call points to is copied into
// one arena block; the update then runs on the worker at the same position of the command stream, before the
// vkCmdBindDescriptorSets / draws recorded after it. Anything it cannot copy faithfully (pNext chains, a null array
// for a type that needs one, extension descriptor types) drains and runs on the ring thread as before.
void UpdateSets(VkDevice vulkan_device, uint32_t nw, const VkWriteDescriptorSet* w, uint32_t nc,
                    const VkCopyDescriptorSet* c) {
  if (!IsProducer()) return g_real.vkUpdateDescriptorSets(vulkan_device, nw, w, nc, c);
  size_t bytes = Size<VkWriteDescriptorSet>(nw) + Size<VkCopyDescriptorSet>(nc);
  bool simple = true;
  for (uint32_t i = 0; i < nw && simple; ++i) {
    const VkWriteDescriptorSet& e = w[i];
    if (e.pNext) {
      simple = false;
      break;
    }
    switch (e.descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        if (!e.pImageInfo) simple = false;
        else bytes += Size<VkDescriptorImageInfo>(e.descriptorCount);
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        if (!e.pTexelBufferView) simple = false;
        else bytes += Size<VkBufferView>(e.descriptorCount);
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC:
        if (!e.pBufferInfo) simple = false;
        else bytes += Size<VkDescriptorBufferInfo>(e.descriptorCount);
        break;
      default:
        simple = false;
        break;
    }
  }
  for (uint32_t i = 0; i < nc && simple; ++i)
    if (c[i].pNext) simple = false;
  if (!simple || bytes > (size_t(64) << 10)) {
    DrainInternal("vkUpdateDescriptorSets (not queueable)");
    ++g_direct;
    ++g_direct_updates;
    return g_real.vkUpdateDescriptorSets(vulkan_device, nw, w, nc, c);
  }
  uint8_t* cursor = Arena(bytes);
  auto* w2 = reinterpret_cast<VkWriteDescriptorSet*>(cursor);
  cursor += Size<VkWriteDescriptorSet>(nw);
  if (nw) std::memcpy(w2, w, sizeof(VkWriteDescriptorSet) * nw);
  const auto* c2 = reinterpret_cast<const VkCopyDescriptorSet*>(cursor);
  if (nc) std::memcpy(cursor, c, sizeof(VkCopyDescriptorSet) * nc);
  cursor += Size<VkCopyDescriptorSet>(nc);
  for (uint32_t i = 0; i < nw; ++i) {
    VkWriteDescriptorSet& e = w2[i];
    switch (e.descriptorType) {
      case VK_DESCRIPTOR_TYPE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
        e.pImageInfo = Copy(cursor, e.pImageInfo, e.descriptorCount);
        break;
      case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
      case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER:
        e.pTexelBufferView = Copy(cursor, e.pTexelBufferView, e.descriptorCount);
        break;
      default:
        e.pBufferInfo = Copy(cursor, e.pBufferInfo, e.descriptorCount);
        break;
    }
  }
  ++g_updates;
  Enqueue(g_real.vkUpdateDescriptorSets, vulkan_device, nw, static_cast<const VkWriteDescriptorSet*>(w2), nc,
          nc ? c2 : static_cast<const VkCopyDescriptorSet*>(nullptr));
}

// Functions the renderer fetches by name.
PFN_vkCmdCopyImage g_copy_image = nullptr;
PFN_vkCmdBlitImage g_blit = nullptr;
PFN_vkCmdResolveImage g_resolve_image = nullptr;  // sa2-nx: sa2_msaa (fetched only with MSAA on)
PFN_vkCmdClearDepthStencilImage g_clear_depth = nullptr;
PFN_vkCmdSetColorBlendEnableEXT g_active_blend = nullptr;
PFN_vkCmdSetColorBlendEquationEXT g_equation = nullptr;
PFN_vkCmdSetColorWriteMaskEXT g_mask = nullptr;
PFN_vkCmdInsertDebugUtilsLabelEXT g_label = nullptr;

void CopyImage(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n,
                  const VkImageCopy* r) {
  ++g_n_copy;
  g_n_copy_regions += n;
  if (!IsProducer()) return g_copy_image(cb, a, la, b, lb, n, r);
  uint8_t* c = Arena(Size<VkImageCopy>(n));
  Enqueue(g_copy_image, cb, a, la, b, lb, n, Copy(c, r, n));
}
void ResolveImage(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n,
                  const VkImageResolve* r) {
  ++g_n_resolve;
  if (!IsProducer()) return g_resolve_image(cb, a, la, b, lb, n, r);
  uint8_t* c = Arena(Size<VkImageResolve>(n));
  Enqueue(g_resolve_image, cb, a, la, b, lb, n, Copy(c, r, n));
}
void Blit(VkCommandBuffer cb, VkImage a, VkImageLayout la, VkImage b, VkImageLayout lb, uint32_t n,
          const VkImageBlit* r, VkFilter f) {
  ++g_n_blit;
  if (!IsProducer()) return g_blit(cb, a, la, b, lb, n, r, f);
  uint8_t* c = Arena(Size<VkImageBlit>(n));
  Enqueue(g_blit, cb, a, la, b, lb, n, Copy(c, r, n), f);
}
void ClearDepth(VkCommandBuffer cb, VkImage i, VkImageLayout l, const VkClearDepthStencilValue* v, uint32_t n,
                       const VkImageSubresourceRange* r) {
  ++g_n_clear_depth;
  if (!IsProducer()) return g_clear_depth(cb, i, l, v, n, r);
  uint8_t* c = Arena(Size<VkClearDepthStencilValue>(1) + Size<VkImageSubresourceRange>(n));
  const auto* v2 = Copy(c, v, 1);
  const auto* r2 = Copy(c, r, n);
  Enqueue(g_clear_depth, cb, i, l, v2, n, r2);
}
void ActiveBlend(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkBool32* v) {
  if (!IsProducer()) return g_active_blend(cb, first, n, v);
  uint8_t* c = Arena(Size<VkBool32>(n));
  Enqueue(g_active_blend, cb, first, n, Copy(c, v, n));
}
void Equation(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkColorBlendEquationEXT* v) {
  if (!IsProducer()) return g_equation(cb, first, n, v);
  uint8_t* c = Arena(Size<VkColorBlendEquationEXT>(n));
  Enqueue(g_equation, cb, first, n, Copy(c, v, n));
}
void Mask(VkCommandBuffer cb, uint32_t first, uint32_t n, const VkColorComponentFlags* v) {
  if (!IsProducer()) return g_mask(cb, first, n, v);
  uint8_t* c = Arena(Size<VkColorComponentFlags>(n));
  Enqueue(g_mask, cb, first, n, Copy(c, v, n));
}
void Label(VkCommandBuffer cb, const VkDebugUtilsLabelEXT* e) {
  if (!IsProducer() || e->pNext) {
    if (IsProducer()) { DrainInternal("debug label with pNext"); ++g_direct; }
    return g_label(cb, e);
  }
  const size_t long_value = e->pLabelName ? std::strlen(e->pLabelName) + 1 : 0;
  uint8_t* c = Arena(Size<VkDebugUtilsLabelEXT>(1) + Size<char>(uint32_t(long_value)));
  auto* e2 = const_cast<VkDebugUtilsLabelEXT*>(Copy(c, e, 1));
  if (long_value) e2->pLabelName = Copy(c, e->pLabelName, uint32_t(long_value));
  Enqueue(g_label, cb, static_cast<const VkDebugUtilsLabelEXT*>(e2));
}

template <size_t Id, typename PFN>
PFN_vkVoidFunction Generic(PFN_vkVoidFunction real) {
  using E = Wrapper<Id, PFN>;
  E::real = reinterpret_cast<PFN>(real);
  E::name = "generic by name";
  E::mode = E::kDeferrable ? kDefer : kDrain;
  return reinterpret_cast<PFN_vkVoidFunction>(&E::F);
}

}  // namespace

const Functions& Table(const Functions& real_fns) {
  if (!REXCVAR_GET(sa2_native_deferred_recording)) return real_fns;
  static const bool built = [&] {
    g_real = real_fns;
    g_proxy = real_fns;
    const int32_t parts = REXCVAR_GET(sa2_native_deferred_recording_parts);
    if (!(parts & 4)) {
#define XE_UI_VULKAN_FUNCTION(name) Install<__COUNTER__>(real_fns.name, g_proxy.name, #name);
#define XE_UI_VULKAN_FUNCTION_PROMOTED(extension_name, core_name) \
  Install<__COUNTER__>(real_fns.core_name, g_proxy.core_name, #core_name);
#include <rex/ui/vulkan/functions/device_1_0.inc>
#include <rex/ui/vulkan/functions/device_khr_swapchain.inc>
#include <rex/ui/vulkan/functions/device_1_1_khr_get_memory_requirements2.inc>
#include <rex/ui/vulkan/functions/device_1_1_khr_bind_memory2.inc>
#include <rex/ui/vulkan/functions/device_1_3_khr_maintenance4.inc>
#include <rex/ui/vulkan/functions/device_1_3_khr_dynamic_rendering.inc>
#undef XE_UI_VULKAN_FUNCTION_PROMOTED
#undef XE_UI_VULKAN_FUNCTION
    }
    if (!(parts & 8)) {
    // The calls with arrays: their own wrappers (the generic ones would only drain and call directly).
    if (!(parts & 16)) g_proxy.vkBeginCommandBuffer = &BeginCommands;
    g_no_enqueue = (parts & 32) != 0;
    g_proxy.vkCmdPipelineBarrier = &Barrier;
    g_proxy.vkCmdBindDescriptorSets = &BindSets;
    g_proxy.vkCmdPushConstants = &Constants;
    g_proxy.vkCmdBeginRenderPass = &BeginPass;
    g_proxy.vkCmdSetViewport = &Framings;
    g_proxy.vkCmdSetScissor = &Scissors;
    g_proxy.vkCmdSetBlendConstants = &ConstantsBlend;
    g_proxy.vkCmdCopyBuffer = &CopyBuffer;
    g_proxy.vkCmdCopyBufferToImage = &BufferToImage;
    g_proxy.vkCmdCopyImageToBuffer = &ImageToBuffer;
    g_proxy.vkCmdClearAttachments = &ClearAttached;
    g_proxy.vkCmdClearColorImage = &ClearColor;
    g_proxy.vkCmdBindVertexBuffers = &BindVertices;
    if (REXCVAR_GET(sa2_deferred_native_update)) g_proxy.vkUpdateDescriptorSets = &UpdateSets;
    }
    g_fast = REXCVAR_GET(sa2_deferred_native_fast);
    g_batch = g_fast ? uint32_t(REXCVAR_GET(sa2_deferred_native_batch)) : 1;
    g_slots = size_t(1) << std::clamp(REXCVAR_GET(usf4_deferred_slots_log2), 8, 14);
    g_queue = new (std::nothrow) Slot[g_slots];
    g_arena = new (std::nothrow) uint8_t[kArena];
    g_arena_ring = REXCVAR_GET(usf4_deferred_async_submit);
    if (g_arena_ring) g_arena_marks = new (std::nothrow) ArenaMark[g_slots];
    if (!g_queue || !g_arena || (g_arena_ring && !g_arena_marks)) {
      REXLOG_ERROR("[native] deferred recording DISABLED: out of memory for the queue ({} KB)",
                   (sizeof(Slot) * g_slots + kArena) / 1024);
      return false;
    }
    g_active.store(true, std::memory_order_release);
    REXLOG_INFO("[native] deferred recording ready: {} queue slots ({} KB), {} KB arena", g_slots,
                sizeof(Slot) * g_slots / 1024, kArena >> 10);
    return true;
  }();
  return built && (REXCVAR_GET(sa2_native_deferred_recording_parts) & 1) ? g_proxy : real_fns;
}

PFN_vkVoidFunction Proc(PFN_vkGetDeviceProcAddr real, VkDevice device, const char* name) {
  const PFN_vkVoidFunction f = real(device, name);
  if (!f || !g_active.load(std::memory_order_acquire) ||
      !(REXCVAR_GET(sa2_native_deferred_recording_parts) & 2))
    return f;
  const std::string_view n(name);
  if (n == "vkCmdCopyImage") { g_copy_image = reinterpret_cast<PFN_vkCmdCopyImage>(f); return reinterpret_cast<PFN_vkVoidFunction>(&CopyImage); }
  if (n == "vkCmdBlitImage") { g_blit = reinterpret_cast<PFN_vkCmdBlitImage>(f); return reinterpret_cast<PFN_vkVoidFunction>(&Blit); }
  if (n == "vkCmdResolveImage") { g_resolve_image = reinterpret_cast<PFN_vkCmdResolveImage>(f); return reinterpret_cast<PFN_vkVoidFunction>(&ResolveImage); }
  if (n == "vkCmdClearDepthStencilImage") { g_clear_depth = reinterpret_cast<PFN_vkCmdClearDepthStencilImage>(f); return reinterpret_cast<PFN_vkVoidFunction>(&ClearDepth); }
  if (n == "vkCmdSetColorBlendEnableEXT") { g_active_blend = reinterpret_cast<PFN_vkCmdSetColorBlendEnableEXT>(f); return reinterpret_cast<PFN_vkVoidFunction>(&ActiveBlend); }
  if (n == "vkCmdSetColorBlendEquationEXT") { g_equation = reinterpret_cast<PFN_vkCmdSetColorBlendEquationEXT>(f); return reinterpret_cast<PFN_vkVoidFunction>(&Equation); }
  if (n == "vkCmdSetColorWriteMaskEXT") { g_mask = reinterpret_cast<PFN_vkCmdSetColorWriteMaskEXT>(f); return reinterpret_cast<PFN_vkVoidFunction>(&Mask); }
  if (n == "vkCmdInsertDebugUtilsLabelEXT") { g_label = reinterpret_cast<PFN_vkCmdInsertDebugUtilsLabelEXT>(f); return reinterpret_cast<PFN_vkVoidFunction>(&Label); }
#define GENERIC(fn_name) \
  if (n == #fn_name) return Generic<__COUNTER__, PFN_##fn_name>(f);
  GENERIC(vkCmdWriteTimestamp)
  GENERIC(vkCmdSetCullMode)
  GENERIC(vkCmdSetFrontFace)
  GENERIC(vkCmdSetPrimitiveTopology)
  GENERIC(vkCmdSetPrimitiveRestartEnable)
  GENERIC(vkCmdSetDepthBiasEnable)
  GENERIC(vkCmdSetDepthTestEnable)
  GENERIC(vkCmdSetDepthWriteEnable)
  GENERIC(vkCmdSetDepthCompareOp)
  GENERIC(vkCmdSetStencilTestEnable)
  GENERIC(vkCmdSetStencilOp)
  GENERIC(vkGetQueryPoolResults)
#undef GENERIC
  if (n.starts_with("vkCmd")) {
    // A recording call this proxy does not know would run out of order: no deferral at all.
    g_active.store(false, std::memory_order_release);
    REXLOG_ERROR("[native] deferred recording DISABLED: no wrapper for {}", name);
  }
  return f;
}

void Drain() {
  if (IsProducer()) DrainInternal("Drain()");
}

void Flush() {
  if (g_fast && g_published != g_written_producer && IsProducer()) PublishAlready();
  if (IsProducer()) Report();
}

bool AsyncSubmit() {
  static const bool on = REXCVAR_GET(usf4_deferred_async_submit);
  return on && IsProducer();
}

const Functions& Real() { return g_real; }

uint64_t RunOnWorker(std::function<void()>* job) {
  if (!IsProducer()) {
    (*job)();
    delete job;
    return 0;
  }
  Enqueue(+[](std::function<void()>* j) {
    (*j)();
    delete j;
  }, job);
  PublishAlready();  // the worker should not wait for the next batch
  return g_written_producer;
}

void WaitTicket(uint64_t ticket) {
  if (!IsProducer() || !ticket) return;
  if (g_read.load(std::memory_order_acquire) >= ticket) return;
  if (g_fast && g_published < ticket) PublishAlready();
  const auto t0 = std::chrono::steady_clock::now();
  Wake();
  for (uint32_t laps = 0; g_read.load(std::memory_order_acquire) < ticket; ++laps) {
    if (laps < 32) std::this_thread::yield();
    else ShortPause(std::chrono::microseconds(20));
  }
  DrainCause& c = Cause("ticket");
  ++c.drains;
  ++c.waits;
  c.ns += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
}

}  // namespace sa2::native::deferred
