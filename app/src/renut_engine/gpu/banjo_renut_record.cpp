// banjo-nx (package 28b): banjo_renut_record_worker. See banjo_renut_record.h.
#include "renut_engine/gpu/banjo_renut_record.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

#include <plume_render_interface.h>
#include <plume_vulkan.h>

#include <rex/cvar.h>
#include <rex/logging.h>

#include "cpu/banjo_enhancements.h"
#include "me/me_platform_compat.h"  // RexSwitchPinCurrentThreadToCore and the priority helper (no-ops off the Switch)
#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/backend/render_thread.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"

REXCVAR_DEFINE_BOOL(banjo_renut_record_worker, true, "Banjo/Enhancements",
                    "Renut's renderer (package 28b): its command recording split across two threads, as banjo-nx's "
                    "renderer's deferred recording: the render thread records into a proxy (each plume call copied "
                    "into a command stream) and a worker on another core plays it on the real command list in the "
                    "same order, so plume's and NVK's recording work (a quarter of the render thread on the console) "
                    "leaves the render thread. The list is ended and submitted once the worker has played all of it. "
                    "Same commands, same order. CPU group; in-run A/B target 14");
REXCVAR_DEFINE_INT32(banjo_renut_record_worker_core, 0, "Banjo",
                     "Renut's renderer, the Switch: the core of banjo_renut_record_worker's thread (-1: not pinned; "
                     "the render thread has banjo_ring_core, the pipeline workers banjo_background_compile_core)")
    .range(-1, 2);
REXCVAR_DECLARE(bool, banjo_cpu_verify);

namespace banjo::renut_gpu::record {
namespace {

constexpr int kWorkerPriority = 0x2D;          // just below the render thread (0x2C), above the game's threads (0x3B)
constexpr size_t kChunkBytes = 256u << 10;     // a chunk of the stream (a command never spans two)
constexpr size_t kPublishBytes = 64u << 10;    // handed to the worker once this much is recorded...
constexpr uint32_t kPublishCommands = 384;     // ...or this many commands (so it starts early in each list; the PC
                                               // run at 96 woke the worker about 70 times a frame)

enum Op : uint16_t {
  kBegin, kEnd, kBarriers, kDispatch, kDraw, kDrawIndexed, kSetPipeline, kSetComputeLayout, kSetComputePush,
  kSetComputeSet, kSetGraphicsLayout, kSetGraphicsPush, kSetGraphicsSet, kSetGraphicsSetDynamic, kSetRootDescriptor,
  kSetIndexBuffer, kSetVertexBuffers, kSetViewports, kSetScissors, kSetFramebuffer, kSetDepthBias,
  kSetStencilReference, kClearColor, kClearDepthStencil, kCopyBufferRegion, kCopyTextureRegion, kCopyBuffer,
  kCopyTexture, kResolveTexture, kResolveTextureRegion, kDiscardTexture, kResetQueryPool, kWriteTimestamp,
  kBlendConstants, kOpCount
};

struct Header {
  uint16_t op;
  uint16_t pad;
  uint32_t size;  // the whole command, a multiple of 8
};

template <typename T>
constexpr bool kPod = std::is_trivially_copyable_v<T>;
static_assert(kPod<plume::RenderBufferBarrier> && kPod<plume::RenderTextureBarrier> &&
              kPod<plume::RenderTextureCopyLocation> && kPod<plume::RenderVertexBufferView> &&
              kPod<plume::RenderInputSlot> && kPod<plume::RenderIndexBufferView> && kPod<plume::RenderViewport> &&
              kPod<plume::RenderRect> && kPod<plume::RenderBox> && kPod<plume::RenderColor> &&
              kPod<plume::RenderBufferReference>, "the proxy copies these as bytes");

struct Chunk {
  std::unique_ptr<uint8_t[]> bytes{new uint8_t[kChunkBytes]};
  size_t used = 0;
  plume::RenderCommandList *target = nullptr;  // the real list the chunk's commands are for
};

// Render thread -> worker.
std::mutex g_mutex;
std::condition_variable g_work;   // a chunk queued, or stopping
std::condition_variable g_done;   // a chunk played
std::deque<Chunk *> g_queue;
std::vector<Chunk *> g_free;
bool g_worker_waiting = false;
bool g_stop = false;
bool g_started = false;
bool g_exited = false;  // the worker returned (after a stop with everything played)
uint64_t g_published = 0;  // chunks handed over (render thread, under g_mutex)
uint64_t g_played = 0;     // chunks played (worker, under g_mutex)

// Statistics.
std::atomic<uint64_t> g_commands{0}, g_chunks{0}, g_drains{0}, g_drain_wait_us{0}, g_worker_busy_us{0}, g_lists{0},
    g_direct_lists{0}, g_wakes{0};

bool On() {
  static const bool on = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_record_worker));
  return on && RecordWorkerAbOn();
}

void Play(plume::RenderCommandList *cmd, const uint8_t *p, const uint8_t *end);

void Worker() {
#if REX_PLATFORM_SWITCH
  const int core = REXCVAR_GET(banjo_renut_record_worker_core);
  const bool priority_ok = RexSwitchSetCurrentThreadPriorityOk(kWorkerPriority);
  const bool core_ok = core < 0 || RexSwitchPinCurrentThreadToCore(core);
  REXLOG_INFO("[renut] record worker at priority 0x{:X}{}, {}", kWorkerPriority, priority_ok ? "" : " (NOT accepted)",
              core < 0 ? std::string("not pinned") : fmt::format("core {}{}", core, core_ok ? "" : " (NOT accepted)"));
#endif
  for (;;) {
    Chunk *c = nullptr;
    {
      std::unique_lock lock(g_mutex);
      while (g_queue.empty() && !g_stop) {
        g_worker_waiting = true;
        g_work.wait(lock);
        g_worker_waiting = false;
      }
      if (g_queue.empty()) {
        g_exited = true;  // stopping, and everything is played
        g_done.notify_all();
        return;
      }
      c = g_queue.front();
      g_queue.pop_front();
    }
    const auto t0 = std::chrono::steady_clock::now();
    Play(c->target, c->bytes.get(), c->bytes.get() + c->used);
    g_worker_busy_us.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
        std::memory_order_relaxed);
    {
      std::lock_guard lock(g_mutex);
      c->used = 0;
      g_free.push_back(c);
      ++g_played;
    }
    g_done.notify_all();
  }
}

template <typename T>
const T &At(const uint8_t *&p) {
  const T &v = *reinterpret_cast<const T *>(p);
  p += (sizeof(T) + 7) & ~size_t(7);
  return v;
}

template <typename T>
const T *Array(const uint8_t *&p, uint32_t count) {
  const T *v = count ? reinterpret_cast<const T *>(p) : nullptr;
  p += (sizeof(T) * count + 7) & ~size_t(7);
  return v;
}

struct Barriers { plume::RenderBarrierStages stages; uint32_t buffers, textures; };
struct U32x4 { uint32_t a, b, c, d; };
struct DrawIndexed { uint32_t count, instances, first; int32_t base; uint32_t first_instance; };
struct Ptr { const void *p; };
struct PushHead { uint32_t range, offset, size; };
struct SetHead { void *set; uint32_t index, count; };
struct RootHead { plume::RenderBufferReference ref; uint32_t index; };
struct VbHead { uint32_t start, count, has_slots; };
struct CountHead { uint32_t count; };
struct DepthBias { float bias, clamp, slope; };
struct ClearColorHead { uint32_t attachment; plume::RenderColor color; uint32_t rects; };
struct ClearDepthHead { uint32_t depth, stencil; float value; uint32_t stencil_value, rects; };
struct CopyBufferRegion { plume::RenderBufferReference dst, src; uint64_t size; };
struct CopyTextureRegionHead { plume::RenderTextureCopyLocation dst, src; uint32_t x, y, z, has_box; };
struct TwoPtr { const void *a, *b; };
struct ResolveRegionHead { const plume::RenderTexture *dst; uint32_t x, y; const plume::RenderTexture *src;
                           uint32_t has_rect; plume::RenderResolveMode mode; };
struct QueryHead { const plume::RenderQueryPool *pool; uint32_t first, count; };
struct Floats4 { float v[4]; };

void Play(plume::RenderCommandList *cmd, const uint8_t *p, const uint8_t *end) {
  while (p < end) {
    const Header &h = *reinterpret_cast<const Header *>(p);
    const uint8_t *next = p + h.size;
    p += sizeof(Header);
    switch (Op(h.op)) {
    case kBegin: cmd->begin(); break;
    case kEnd: cmd->end(); break;
    case kBarriers: {
      const Barriers &b = At<Barriers>(p);
      const auto *bb = Array<plume::RenderBufferBarrier>(p, b.buffers);
      const auto *tb = Array<plume::RenderTextureBarrier>(p, b.textures);
      cmd->barriers(b.stages, bb, b.buffers, tb, b.textures);
      break;
    }
    case kDispatch: { const U32x4 &d = At<U32x4>(p); cmd->dispatch(d.a, d.b, d.c); break; }
    case kDraw: { const U32x4 &d = At<U32x4>(p); cmd->drawInstanced(d.a, d.b, d.c, d.d); break; }
    case kDrawIndexed: {
      const DrawIndexed &d = At<DrawIndexed>(p);
      cmd->drawIndexedInstanced(d.count, d.instances, d.first, d.base, d.first_instance);
      break;
    }
    case kSetPipeline: {
      auto *pipeline = static_cast<const plume::RenderPipeline *>(At<Ptr>(p).p);
      // banjo_renut_tuning's NVK prefetch hint (the Switch), on the core that binds.
      banjo::renut_gpu::BeforePipelineBind(const_cast<plume::RenderPipeline *>(pipeline));
      cmd->setPipeline(pipeline);
      break;
    }
    case kSetComputeLayout:
      cmd->setComputePipelineLayout(static_cast<const plume::RenderPipelineLayout *>(At<Ptr>(p).p));
      break;
    case kSetGraphicsLayout:
      cmd->setGraphicsPipelineLayout(static_cast<const plume::RenderPipelineLayout *>(At<Ptr>(p).p));
      break;
    case kSetComputePush:
    case kSetGraphicsPush: {
      const PushHead &ph = At<PushHead>(p);
      const uint8_t *data = Array<uint8_t>(p, ph.size);
      if (h.op == kSetComputePush)
        cmd->setComputePushConstants(ph.range, data, ph.offset, ph.size);
      else
        cmd->setGraphicsPushConstants(ph.range, data, ph.offset, ph.size);
      break;
    }
    case kSetComputeSet:
    case kSetGraphicsSet: {
      const SetHead &s = At<SetHead>(p);
      auto *set = static_cast<plume::RenderDescriptorSet *>(s.set);
      if (h.op == kSetComputeSet)
        cmd->setComputeDescriptorSet(set, s.index);
      else
        cmd->setGraphicsDescriptorSet(set, s.index);
      break;
    }
    case kSetGraphicsSetDynamic: {
      const SetHead &s = At<SetHead>(p);
      const uint32_t *offsets = Array<uint32_t>(p, s.count);
      cmd->setGraphicsDescriptorSetDynamic(static_cast<plume::RenderDescriptorSet *>(s.set), s.index, offsets, s.count);
      break;
    }
    case kSetRootDescriptor: { const RootHead &r = At<RootHead>(p); cmd->setGraphicsRootDescriptor(r.ref, r.index); break; }
    case kSetIndexBuffer: {
      const auto &v = At<plume::RenderIndexBufferView>(p);
      cmd->setIndexBuffer(&v);
      break;
    }
    case kSetVertexBuffers: {
      const VbHead &v = At<VbHead>(p);
      const auto *views = Array<plume::RenderVertexBufferView>(p, v.count);
      const auto *slots = v.has_slots ? Array<plume::RenderInputSlot>(p, v.count) : nullptr;
      cmd->setVertexBuffers(v.start, views, v.count, slots);
      break;
    }
    case kSetViewports: {
      const uint32_t n = At<CountHead>(p).count;
      cmd->setViewports(Array<plume::RenderViewport>(p, n), n);
      break;
    }
    case kSetScissors: {
      const uint32_t n = At<CountHead>(p).count;
      cmd->setScissors(Array<plume::RenderRect>(p, n), n);
      break;
    }
    case kSetFramebuffer: cmd->setFramebuffer(static_cast<const plume::RenderFramebuffer *>(At<Ptr>(p).p)); break;
    case kSetDepthBias: { const DepthBias &d = At<DepthBias>(p); cmd->setDepthBias(d.bias, d.clamp, d.slope); break; }
    case kSetStencilReference: cmd->setStencilReference(At<CountHead>(p).count); break;
    case kClearColor: {
      const ClearColorHead &c = At<ClearColorHead>(p);
      cmd->clearColor(c.attachment, c.color, Array<plume::RenderRect>(p, c.rects), c.rects);
      break;
    }
    case kClearDepthStencil: {
      const ClearDepthHead &c = At<ClearDepthHead>(p);
      cmd->clearDepthStencil(c.depth != 0, c.stencil != 0, c.value, c.stencil_value,
                             Array<plume::RenderRect>(p, c.rects), c.rects);
      break;
    }
    case kCopyBufferRegion: {
      const CopyBufferRegion &c = At<CopyBufferRegion>(p);
      cmd->copyBufferRegion(c.dst, c.src, c.size);
      break;
    }
    case kCopyTextureRegion: {
      const CopyTextureRegionHead &c = At<CopyTextureRegionHead>(p);
      const plume::RenderBox *box = c.has_box ? &At<plume::RenderBox>(p) : nullptr;
      cmd->copyTextureRegion(c.dst, c.src, c.x, c.y, c.z, box);
      break;
    }
    case kCopyBuffer: {
      const TwoPtr &t = At<TwoPtr>(p);
      cmd->copyBuffer(static_cast<const plume::RenderBuffer *>(t.a), static_cast<const plume::RenderBuffer *>(t.b));
      break;
    }
    case kCopyTexture:
    case kResolveTexture: {
      const TwoPtr &t = At<TwoPtr>(p);
      const auto *a = static_cast<const plume::RenderTexture *>(t.a);
      const auto *b = static_cast<const plume::RenderTexture *>(t.b);
      if (h.op == kCopyTexture)
        cmd->copyTexture(a, b);
      else
        cmd->resolveTexture(a, b);
      break;
    }
    case kResolveTextureRegion: {
      const ResolveRegionHead &r = At<ResolveRegionHead>(p);
      const plume::RenderRect *rect = r.has_rect ? &At<plume::RenderRect>(p) : nullptr;
      cmd->resolveTextureRegion(r.dst, r.x, r.y, r.src, rect, r.mode);
      break;
    }
    case kDiscardTexture: cmd->discardTexture(static_cast<const plume::RenderTexture *>(At<Ptr>(p).p)); break;
    case kResetQueryPool: { const QueryHead &q = At<QueryHead>(p); cmd->resetQueryPool(q.pool, q.first, q.count); break; }
    case kWriteTimestamp: { const QueryHead &q = At<QueryHead>(p); cmd->writeTimestamp(q.pool, q.first); break; }
    case kBlendConstants: {
      const Floats4 &f = At<Floats4>(p);
      VkCommandBuffer vk = static_cast<plume::VulkanCommandList *>(cmd)->vk;
      if (vk && vkCmdSetBlendConstants)
        vkCmdSetBlendConstants(vk, f.v);
      break;
    }
    default:
      REXLOG_ERROR("[renut] record worker: unknown command {}", h.op);
      return;
    }
    p = next;
  }
}

// The proxy. Render thread only.
class Recorder final : public plume::RenderCommandList {
public:
  plume::RenderCommandList *real = nullptr;

  // Makes room for a command of 'payload' bytes and returns where its payload goes.
  uint8_t *Emit(Op op, size_t payload) {
    const size_t size = sizeof(Header) + ((payload + 7) & ~size_t(7));
    if (!cur_ || cur_->used + size > kChunkBytes)
      Publish(true);
    uint8_t *p = cur_->bytes.get() + cur_->used;
    Header &h = *reinterpret_cast<Header *>(p);
    h.op = op;
    h.pad = 0;
    h.size = uint32_t(size);
    cur_->used += size;
    ++commands_;
    return p + sizeof(Header);
  }

  template <typename T>
  static uint8_t *Put(uint8_t *p, const T &v) {
    std::memcpy(p, &v, sizeof(T));
    return p + ((sizeof(T) + 7) & ~size_t(7));
  }
  template <typename T>
  static uint8_t *PutArray(uint8_t *p, const T *v, uint32_t n) {
    if (n)
      std::memcpy(p, v, sizeof(T) * n);
    return p + ((sizeof(T) * n + 7) & ~size_t(7));
  }
  template <typename T>
  static constexpr size_t Size(uint32_t n = 1) {
    return (sizeof(T) * n + 7) & ~size_t(7);
  }

  // Hands the recorded chunk to the worker (always when 'full'; else once enough is recorded).
  void Publish(bool full) {
    if (cur_ && cur_->used && (full || cur_->used >= kPublishBytes || commands_ >= kPublishCommands)) {
      Chunk *c = cur_;
      cur_ = nullptr;
      c->target = real;  // every command since the last hand-over is for the list open now (Wrap)
      bool wake;
      {
        std::lock_guard lock(g_mutex);
        g_queue.push_back(c);
        ++g_published;
        wake = g_worker_waiting;
      }
      if (wake) {
        g_work.notify_one();
        g_wakes.fetch_add(1, std::memory_order_relaxed);
      }
      g_chunks.fetch_add(1, std::memory_order_relaxed);
      g_commands.fetch_add(commands_, std::memory_order_relaxed);
      commands_ = 0;
    }
    if (!cur_) {
      std::lock_guard lock(g_mutex);
      if (!g_free.empty()) {
        cur_ = g_free.back();
        g_free.pop_back();
      }
    }
    if (!cur_)
      cur_ = new Chunk();
  }

  void Drain() {
    Publish(true);
    const auto t0 = std::chrono::steady_clock::now();
    {
      std::unique_lock lock(g_mutex);
      if (g_played != g_published) {
        ::renut::gpu::render_thread::WaitScope wait;
        g_done.wait(lock, [] { return g_played == g_published; });
      }
    }
    g_drains.fetch_add(1, std::memory_order_relaxed);
    g_drain_wait_us.fetch_add(
        uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t0).count()),
        std::memory_order_relaxed);
  }

  // A command ends; it goes out once the chunk has enough (the worker starts early in a list).
  void Done() { Publish(false); }

  void begin() override { Emit(kBegin, 0); Done(); }
  void end() override { Emit(kEnd, 0); Done(); }
  void barriers(plume::RenderBarrierStages stages, const plume::RenderBufferBarrier *bb, uint32_t bn,
                const plume::RenderTextureBarrier *tb, uint32_t tn) override {
    uint8_t *p = Emit(kBarriers, Size<Barriers>() + Size<plume::RenderBufferBarrier>(bn) +
                                     Size<plume::RenderTextureBarrier>(tn));
    p = Put(p, Barriers{stages, bn, tn});
    p = PutArray(p, bb, bn);
    PutArray(p, tb, tn);
    Done();
  }
  void dispatch(uint32_t x, uint32_t y, uint32_t z) override {
    Put(Emit(kDispatch, Size<U32x4>()), U32x4{x, y, z, 0});
    Done();
  }
  void traceRays(uint32_t, uint32_t, uint32_t, plume::RenderBufferReference,
                 const plume::RenderShaderBindingGroupsInfo &) override {
    Unsupported("traceRays");
  }
  void drawInstanced(uint32_t a, uint32_t b, uint32_t c, uint32_t d) override {
    Put(Emit(kDraw, Size<U32x4>()), U32x4{a, b, c, d});
    Done();
  }
  void drawIndexedInstanced(uint32_t a, uint32_t b, uint32_t c, int32_t d, uint32_t e) override {
    Put(Emit(kDrawIndexed, Size<DrawIndexed>()), DrawIndexed{a, b, c, d, e});
    Done();
  }
  void setPipeline(const plume::RenderPipeline *pipeline) override { PutPtr(kSetPipeline, pipeline); }
  void setComputePipelineLayout(const plume::RenderPipelineLayout *l) override { PutPtr(kSetComputeLayout, l); }
  void setComputePushConstants(uint32_t range, const void *data, uint32_t offset, uint32_t size) override {
    Push(kSetComputePush, range, data, offset, size);
  }
  void setComputeDescriptorSet(plume::RenderDescriptorSet *set, uint32_t index) override {
    Put(Emit(kSetComputeSet, Size<SetHead>()), SetHead{set, index, 0});
    Done();
  }
  void setGraphicsPipelineLayout(const plume::RenderPipelineLayout *l) override { PutPtr(kSetGraphicsLayout, l); }
  void setGraphicsPushConstants(uint32_t range, const void *data, uint32_t offset, uint32_t size) override {
    Push(kSetGraphicsPush, range, data, offset, size);
  }
  void setGraphicsDescriptorSet(plume::RenderDescriptorSet *set, uint32_t index) override {
    Put(Emit(kSetGraphicsSet, Size<SetHead>()), SetHead{set, index, 0});
    Done();
  }
  void setGraphicsDescriptorSetDynamic(plume::RenderDescriptorSet *set, uint32_t index, const uint32_t *offsets,
                                       uint32_t count) override {
    uint8_t *p = Emit(kSetGraphicsSetDynamic, Size<SetHead>() + Size<uint32_t>(count));
    p = Put(p, SetHead{set, index, count});
    PutArray(p, offsets, count);
    Done();
  }
  void setGraphicsRootDescriptor(plume::RenderBufferReference ref, uint32_t index) override {
    Put(Emit(kSetRootDescriptor, Size<RootHead>()), RootHead{ref, index});
    Done();
  }
  void setRaytracingPipelineLayout(const plume::RenderPipelineLayout *) override { Unsupported("raytracing"); }
  void setRaytracingPushConstants(uint32_t, const void *, uint32_t, uint32_t) override { Unsupported("raytracing"); }
  void setRaytracingDescriptorSet(plume::RenderDescriptorSet *, uint32_t) override { Unsupported("raytracing"); }
  void setIndexBuffer(const plume::RenderIndexBufferView *view) override {
    Put(Emit(kSetIndexBuffer, Size<plume::RenderIndexBufferView>()), *view);
    Done();
  }
  void setVertexBuffers(uint32_t start, const plume::RenderVertexBufferView *views, uint32_t count,
                        const plume::RenderInputSlot *slots) override {
    uint8_t *p = Emit(kSetVertexBuffers, Size<VbHead>() + Size<plume::RenderVertexBufferView>(count) +
                                             (slots ? Size<plume::RenderInputSlot>(count) : 0));
    p = Put(p, VbHead{start, count, slots ? 1u : 0u});
    p = PutArray(p, views, count);
    if (slots)
      PutArray(p, slots, count);
    Done();
  }
  void setViewports(const plume::RenderViewport *v, uint32_t count) override {
    uint8_t *p = Emit(kSetViewports, Size<CountHead>() + Size<plume::RenderViewport>(count));
    PutArray(Put(p, CountHead{count}), v, count);
    Done();
  }
  void setScissors(const plume::RenderRect *r, uint32_t count) override {
    uint8_t *p = Emit(kSetScissors, Size<CountHead>() + Size<plume::RenderRect>(count));
    PutArray(Put(p, CountHead{count}), r, count);
    Done();
  }
  void setFramebuffer(const plume::RenderFramebuffer *fb) override { PutPtr(kSetFramebuffer, fb); }
  void setDepthBias(float a, float b, float c) override {
    Put(Emit(kSetDepthBias, Size<DepthBias>()), DepthBias{a, b, c});
    Done();
  }
  void setStencilReference(uint32_t reference) override {
    Put(Emit(kSetStencilReference, Size<CountHead>()), CountHead{reference});
    Done();
  }
  void clearColor(uint32_t attachment, plume::RenderColor color, const plume::RenderRect *rects,
                  uint32_t count) override {
    uint8_t *p = Emit(kClearColor, Size<ClearColorHead>() + Size<plume::RenderRect>(count));
    PutArray(Put(p, ClearColorHead{attachment, color, count}), rects, count);
    Done();
  }
  void clearDepthStencil(bool depth, bool stencil, float value, uint32_t stencil_value,
                         const plume::RenderRect *rects, uint32_t count) override {
    uint8_t *p = Emit(kClearDepthStencil, Size<ClearDepthHead>() + Size<plume::RenderRect>(count));
    PutArray(Put(p, ClearDepthHead{depth ? 1u : 0u, stencil ? 1u : 0u, value, stencil_value, count}), rects, count);
    Done();
  }
  void copyBufferRegion(plume::RenderBufferReference dst, plume::RenderBufferReference src, uint64_t size) override {
    Put(Emit(kCopyBufferRegion, Size<CopyBufferRegion>()), CopyBufferRegion{dst, src, size});
    Done();
  }
  void copyTextureRegion(const plume::RenderTextureCopyLocation &dst, const plume::RenderTextureCopyLocation &src,
                         uint32_t x, uint32_t y, uint32_t z, const plume::RenderBox *box) override {
    uint8_t *p = Emit(kCopyTextureRegion, Size<CopyTextureRegionHead>() + (box ? Size<plume::RenderBox>() : 0));
    p = Put(p, CopyTextureRegionHead{dst, src, x, y, z, box ? 1u : 0u});
    if (box)
      Put(p, *box);
    Done();
  }
  void copyBuffer(const plume::RenderBuffer *dst, const plume::RenderBuffer *src) override {
    Put(Emit(kCopyBuffer, Size<TwoPtr>()), TwoPtr{dst, src});
    Done();
  }
  void copyTexture(const plume::RenderTexture *dst, const plume::RenderTexture *src) override {
    Put(Emit(kCopyTexture, Size<TwoPtr>()), TwoPtr{dst, src});
    Done();
  }
  void resolveTexture(const plume::RenderTexture *dst, const plume::RenderTexture *src) override {
    Put(Emit(kResolveTexture, Size<TwoPtr>()), TwoPtr{dst, src});
    Done();
  }
  void resolveTextureRegion(const plume::RenderTexture *dst, uint32_t x, uint32_t y, const plume::RenderTexture *src,
                            const plume::RenderRect *rect, plume::RenderResolveMode mode) override {
    uint8_t *p = Emit(kResolveTextureRegion, Size<ResolveRegionHead>() + (rect ? Size<plume::RenderRect>() : 0));
    p = Put(p, ResolveRegionHead{dst, x, y, src, rect ? 1u : 0u, mode});
    if (rect)
      Put(p, *rect);
    Done();
  }
  void buildBottomLevelAS(const plume::RenderAccelerationStructure *, plume::RenderBufferReference,
                          const plume::RenderBottomLevelASBuildInfo &) override {
    Unsupported("acceleration structures");
  }
  void buildTopLevelAS(const plume::RenderAccelerationStructure *, plume::RenderBufferReference,
                       plume::RenderBufferReference, const plume::RenderTopLevelASBuildInfo &) override {
    Unsupported("acceleration structures");
  }
  void discardTexture(const plume::RenderTexture *texture) override { PutPtr(kDiscardTexture, texture); }
  void resetQueryPool(const plume::RenderQueryPool *pool, uint32_t first, uint32_t count) override {
    Put(Emit(kResetQueryPool, Size<QueryHead>()), QueryHead{pool, first, count});
    Done();
  }
  void writeTimestamp(const plume::RenderQueryPool *pool, uint32_t index) override {
    Put(Emit(kWriteTimestamp, Size<QueryHead>()), QueryHead{pool, index, 0});
    Done();
  }
  void BlendConstants(const float rgba[4]) {
    Floats4 f;
    std::memcpy(f.v, rgba, sizeof(f.v));
    Put(Emit(kBlendConstants, Size<Floats4>()), f);
    Done();
  }

private:
  Chunk *cur_ = nullptr;
  uint32_t commands_ = 0;

  void PutPtr(Op op, const void *ptr) {
    Put(Emit(op, Size<Ptr>()), Ptr{ptr});
    Done();
  }
  void Push(Op op, uint32_t range, const void *data, uint32_t offset, uint32_t size) {
    if (size == 0) {
      // plume reads the whole range then, whose size the proxy does not know: every caller gives it (package 28b).
      static bool logged = false;
      if (!logged) {
        logged = true;
        REXLOG_ERROR("[renut] record worker: push constants without a size (range {}); not recorded", range);
      }
      return;
    }
    uint8_t *p = Emit(op, Size<PushHead>() + Size<uint8_t>(size));
    PutArray(Put(p, PushHead{range, offset, size}), static_cast<const uint8_t *>(data), size);
    Done();
  }
  static void Unsupported(const char *what) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      REXLOG_ERROR("[renut] record worker: {} is not recorded (Renut's renderer does not use it)", what);
    }
  }
};

Recorder g_recorder;

}  // namespace

plume::RenderCommandList *Wrap(plume::RenderCommandList *real) {
  // Vulkan only: the blend constant is played as a Vulkan command (SetBlendConstant's own call).
  if (!real || !::renut::gpu::IsVulkan() || !On()) {
    g_direct_lists.fetch_add(1, std::memory_order_relaxed);
    return real;
  }
  {
    std::lock_guard lock(g_mutex);
    if (!g_started) {
      g_started = true;
      StartLongLivedThread(&Worker, 256u << 10);
    }
  }
  // Every earlier list was drained before its submit: the stream is empty here, so the target switches cleanly.
  g_recorder.real = real;
  g_lists.fetch_add(1, std::memory_order_relaxed);
  return &g_recorder;
}

void Drain() {
  if (g_started)
    g_recorder.Drain();
}

bool SetBlendConstants(plume::RenderCommandList *cmd, const float rgba[4]) {
  if (cmd != &g_recorder)
    return false;
  g_recorder.BlendConstants(rgba);
  return true;
}

bool IsProxy(const plume::RenderCommandList *cmd) {
  return cmd == &g_recorder;
}

void Shutdown() {
  if (!g_started)
    return;
  g_recorder.Drain();
  std::unique_lock lock(g_mutex);
  g_stop = true;
  g_work.notify_all();
  g_done.wait(lock, [] { return g_exited; });
}

void Report(int seconds) {
  (void)seconds;
  const uint64_t lists = g_lists.exchange(0, std::memory_order_relaxed);
  const uint64_t direct = g_direct_lists.exchange(0, std::memory_order_relaxed);
  if (!lists && !g_started)
    return;
  const uint64_t commands = g_commands.exchange(0, std::memory_order_relaxed);
  const uint64_t chunks = g_chunks.exchange(0, std::memory_order_relaxed);
  const uint64_t drains = g_drains.exchange(0, std::memory_order_relaxed);
  const uint64_t wait_us = g_drain_wait_us.exchange(0, std::memory_order_relaxed);
  const uint64_t busy_us = g_worker_busy_us.exchange(0, std::memory_order_relaxed);
  const uint64_t wakes = g_wakes.exchange(0, std::memory_order_relaxed);
  REXLOG_INFO("[renut] record worker (banjo_renut_record_worker): {} lists through it, {} direct; {} commands in {} "
              "chunks ({} wakes); worker playing {:.1f} ms; the render thread waited {:.1f} ms in {} drains",
              lists, direct, commands, chunks, wakes, double(busy_us) / 1000.0, double(wait_us) / 1000.0, drains);
}

}  // namespace banjo::renut_gpu::record
