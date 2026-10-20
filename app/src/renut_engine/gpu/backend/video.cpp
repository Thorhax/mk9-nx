/**
 * @file    gpu/backend/video.cpp
 * @brief   Device creation, bindless tables, resource retirement and command
 *          list management for the native renderer.
 */
#include "renut_engine/gpu/backend/video.h"

#include <cstdio>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <cstring>
#include <filesystem>
#include <system_error>
#include <vector>
#include <atomic>
#include <chrono>  // banjo-nx (2026-10-20): banjo_deko3d_latency_trim
#include <string>
#include <fmt/format.h>

#if defined(_WIN32)
#include <io.h>
#include <share.h>
#endif

#include <plume_render_interface.h>
#include <plume_render_interface_builders.h>

#include "renut_engine/perf/frame_stats.h"
#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/backend/gpu_profile.h"
#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/backend/native_window.h"
#include "renut_engine/gpu/render/pipeline.h"
#include "renut_engine/gpu/backend/render_thread.h"
#include "renut_engine/gpu/backend/video_internal.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"
#include "renut_engine/gpu/banjo_renut_record.h"  // banjo-nx (package 28b): banjo_renut_record_worker
#include <rex/frame_latency.h>  // banjo-nx (2026-10-20): the latency probe's "input -> GPU done"
#include <rex/cvar.h>  // sa2-nx (2026-10-20): sa2_renut_double_buffer
#if defined(PLUME_DEKO3D)
#include <plume_deko3d.h>  // banjo-nx (2026-10-20, banjo_deko3d_latency_trim): fence copies
#endif

#include "host_shaders/depth_resolve_ms_ps.spirv.h"
#include "host_shaders/depth_resolve_ps.spirv.h"
#include "host_shaders/present_ps.spirv.h"
#include "host_shaders/present_vs.spirv.h"
#include "host_shaders/rectangle_list_gs.spirv.h"
#include "host_shaders/point_sprite_gs.spirv.h"
#include "host_shaders/stream_convert_cs.spirv.h"
#include "host_shaders/target_readback_cs.spirv.h"
#include "host_shaders/texture_mip_ps.spirv.h"
#include "host_shaders/resolve_copy_ps.spirv.h"
#if defined(PLUME_DEKO3D)  // banjo-nx (2026-10-17): the deko3d back end's (tools/deko3d_host_shader.py)
#include "host_shaders/depth_resolve_ms_ps.deko3d.h"
#include "host_shaders/depth_resolve_ps.deko3d.h"
#include "host_shaders/present_ps.deko3d.h"
#include "host_shaders/present_vs.deko3d.h"
#include "host_shaders/rectangle_list_gs.deko3d.h"
#include "host_shaders/point_sprite_gs.deko3d.h"
#include "host_shaders/stream_convert_cs.deko3d.h"
#include "host_shaders/target_readback_cs.deko3d.h"
#include "host_shaders/texture_mip_ps.deko3d.h"
#include "host_shaders/resolve_copy_ps.deko3d.h"
#endif
#if defined(_WIN32)
#include "host_shaders/depth_resolve_ms_ps.dxil.h"
#include "host_shaders/depth_resolve_ps.dxil.h"
#include "host_shaders/present_ps.dxil.h"
#include "host_shaders/present_vs.dxil.h"
#include "host_shaders/rectangle_list_gs.dxil.h"
#include "host_shaders/point_sprite_gs.dxil.h"
#include "host_shaders/stream_convert_cs.dxil.h"
#include "host_shaders/target_readback_cs.dxil.h"
#include "host_shaders/texture_mip_ps.dxil.h"
#include "host_shaders/resolve_copy_ps.dxil.h"
#endif

REXCVAR_DEFINE_BOOL(sa2_renut_double_buffer, false, "SA2/Renderer",
                    "Renut's renderer: a swapchain of 2 images instead of 3 (input latency: a finished frame is shown at "
                    "the next refresh instead of queueing behind one already waiting). Off by default: a frame that "
                    "misses a refresh then waits a whole one, which hurts where frames are close to the budget (split "
                    "screen). Takes effect at start")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace renut::gpu {

// banjo-nx (2026-10-19, CPU proposal 5): State() is inline (video.h): a load of this pointer, set by the first call.
VideoState *g_state = nullptr;

VideoState &StateSlow() {
  static VideoState s;
  g_state = &s;
  return s;
}

namespace {

constexpr plume::RenderFormat kSwapChainFormat = plume::RenderFormat::B8G8R8A8_UNORM;

bool BuildUtilityLayout(VideoState &s) {
  plume::RenderDescriptorSetBuilder textures;
  textures.begin();
  textures.addTexture(0, kTextureDescriptorCount);
  textures.end(true, kTextureDescriptorCount);

  plume::RenderDescriptorSetBuilder samplers;
  samplers.begin();
  samplers.addSampler(0, kSamplerDescriptorCount);
  samplers.end(true, kSamplerDescriptorCount);

  // The two bindless tables are created once and shared by every layout that
  // includes them (plume's D3D12 backend draws sets from one global heap).
  s.texture_set = textures.create(s.device.get());
  s.sampler_set = samplers.create(s.device.get());
  if (!s.texture_set || !s.sampler_set) {
    RGPU_ERROR("Bindless descriptor set creation failed");
    return false;
  }
  s.texture_slot_used.assign(kTextureDescriptorCount, false);
  s.sampler_slot_used.assign(kSamplerDescriptorCount, false);
  s.texture_slot_used[kNullTextureSlot] = true;
  s.sampler_slot_used[kLinearClampSamplerSlot] = true;
  s.sampler_slot_used[kPointClampSamplerSlot] = true;

  plume::RenderPipelineLayoutBuilder layout;
  layout.begin(false, true);
  layout.addDescriptorSet(textures);  // set 0 -> t0, space0
  layout.addDescriptorSet(samplers);  // set 1 -> s0, space1
  // banjo-nx (2026-10-18): the bindless table again, read as uint (the depth resolve's stencil views).
  layout.addDescriptorSet(textures);  // set 2 -> t0, space2
  layout.addPushConstant(0, 2, sizeof(UtilityPush),
                         plume::RenderShaderStageFlag::VERTEX |
                             plume::RenderShaderStageFlag::PIXEL);  // b0, space2
  layout.end();
  s.utility_layout = layout.create(s.device.get());
  if (!s.utility_layout) {
    RGPU_ERROR("Utility pipeline layout creation failed");
    return false;
  }
  return true;
}

bool BuildSamplers(VideoState &s) {
  plume::RenderSamplerDesc linear;
  linear.minFilter = plume::RenderFilter::LINEAR;
  linear.magFilter = plume::RenderFilter::LINEAR;
  linear.mipmapMode = plume::RenderMipmapMode::LINEAR;
  linear.addressU = linear.addressV = linear.addressW = plume::RenderTextureAddressMode::CLAMP;
  s.linear_clamp_sampler = s.device->createSampler(linear);

  plume::RenderSamplerDesc point = linear;
  point.minFilter = plume::RenderFilter::NEAREST;
  point.magFilter = plume::RenderFilter::NEAREST;
  point.mipmapMode = plume::RenderMipmapMode::NEAREST;
  s.point_clamp_sampler = s.device->createSampler(point);

  if (!s.linear_clamp_sampler || !s.point_clamp_sampler) {
    RGPU_ERROR("Sampler creation failed");
    return false;
  }
  s.sampler_set->setSampler(kLinearClampSamplerSlot, s.linear_clamp_sampler.get());
  s.sampler_set->setSampler(kPointClampSamplerSlot, s.point_clamp_sampler.get());
  return true;
}

bool BuildNullTexture(VideoState &s) {
  constexpr auto kFormat = plume::RenderFormat::R8G8B8A8_UNORM;
  s.null_texture = s.device->createTexture(plume::RenderTextureDesc::Texture2D(1, 1, 1, kFormat));
  s.null_texture_3d =
      s.device->createTexture(plume::RenderTextureDesc::Texture3D(1, 1, 1, 1, kFormat));
  s.null_texture_cube = s.device->createTexture(plume::RenderTextureDesc::Texture(
      plume::RenderTextureDimension::TEXTURE_2D, 1, 1, 1, 1, 6, kFormat,
      plume::RenderTextureFlag::CUBE));
  if (!s.null_texture || !s.null_texture_3d || !s.null_texture_cube) {
    RGPU_ERROR("Null texture creation failed");
    return false;
  }
  s.null_texture->setName("renut null texture");
  s.null_texture_3d->setName("renut null texture 3d");
  s.null_texture_cube->setName("renut null texture cube");
  s.null_texture_view =
      s.null_texture->createTextureView(plume::RenderTextureViewDesc::Texture2D(kFormat));
  s.null_texture_3d_view =
      s.null_texture_3d->createTextureView(plume::RenderTextureViewDesc::Texture3D(kFormat));
  s.null_texture_cube_view =
      s.null_texture_cube->createTextureView(plume::RenderTextureViewDesc::TextureCube(kFormat));
  if (!s.null_texture_view || !s.null_texture_3d_view || !s.null_texture_cube_view)
    return false;
  // Every slot starts pointing at the 2D one, so a stale index still reads a
  // live (zero) descriptor. Contents are uploaded on the first command list.
  for (uint32_t i = 0; i < kTextureDescriptorCount; ++i) {
    s.texture_set->setTexture(i, s.null_texture.get(), plume::RenderTextureLayout::SHADER_READ,
                              s.null_texture_view.get());
  }
  s.texture_set->setTexture(kNullTexture3DSlot, s.null_texture_3d.get(),
                            plume::RenderTextureLayout::SHADER_READ, s.null_texture_3d_view.get());
  s.texture_set->setTexture(kNullTextureCubeSlot, s.null_texture_cube.get(),
                            plume::RenderTextureLayout::SHADER_READ,
                            s.null_texture_cube_view.get());
  s.texture_slot_used[kNullTexture3DSlot] = true;
  s.texture_slot_used[kNullTextureCubeSlot] = true;
  // banjo-nx (2026-10-17): banjo_renut_diag_poison_slots, a test of the eviction rules.
  if (banjo::renut_gpu::PoisonSlotsOn()) {
    s.poison_texture = s.device->createTexture(plume::RenderTextureDesc::Texture2D(1, 1, 1, kFormat));
    if (s.poison_texture) {
      s.poison_texture->setName("banjo-nx poison texture");
      s.poison_texture_view =
          s.poison_texture->createTextureView(plume::RenderTextureViewDesc::Texture2D(kFormat));
    }
    if (!s.poison_texture_view)
      s.poison_texture.reset();
  }
  return true;
}

// Zero-fills the null textures (every subresource is a single texel).
bool InitializeNullTextures(VideoState &s) {
  UploadAllocation zero = s.uploads[s.frame].Allocate(256, 512);
  if (!zero)
    return false;
  std::memset(zero.cpu, 0, 256);
  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderTexture *textures[] = {s.null_texture.get(), s.null_texture_3d.get(),
                                      s.null_texture_cube.get()};
  const uint32_t slices[] = {1, 1, 6};
  for (uint32_t t = 0; t < 3; ++t) {
    cmd->barriers(plume::RenderBarrierStage::COPY,
                  plume::RenderTextureBarrier(textures[t], plume::RenderTextureLayout::COPY_DEST));
    for (uint32_t slice = 0; slice < slices[t]; ++slice) {
      cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(textures[t], 0, slice),
                             plume::RenderTextureCopyLocation::PlacedFootprint(
                                 zero.ref.ref, plume::RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64,
                                 zero.ref.offset));
    }
    cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                  plume::RenderTextureBarrier(textures[t], plume::RenderTextureLayout::SHADER_READ));
  }
  // banjo-nx: the poison texture's one texel, magenta and opaque.
  if (s.poison_texture) {
    UploadAllocation magenta = s.uploads[s.frame].Allocate(256, 512);
    if (!magenta)
      return false;
    const uint8_t texel[4] = {0xFF, 0x00, 0xFF, 0xFF};
    std::memset(magenta.cpu, 0, 256);
    std::memcpy(magenta.cpu, texel, sizeof(texel));
    cmd->barriers(plume::RenderBarrierStage::COPY,
                  plume::RenderTextureBarrier(s.poison_texture.get(), plume::RenderTextureLayout::COPY_DEST));
    cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(s.poison_texture.get(), 0, 0),
                           plume::RenderTextureCopyLocation::PlacedFootprint(
                               magenta.ref.ref, plume::RenderFormat::R8G8B8A8_UNORM, 1, 1, 1, 64,
                               magenta.ref.offset));
    cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                  plume::RenderTextureBarrier(s.poison_texture.get(), plume::RenderTextureLayout::SHADER_READ));
  }
  return true;
}

bool BuildPresentPipeline(VideoState &s) {
  s.present_vs = CreateHostShader(s.device.get(), RENUT_HOST_SHADER(present_vs), "present_vs");
  s.present_ps = CreateHostShader(s.device.get(), RENUT_HOST_SHADER(present_ps), "present_ps");
  if (!s.present_vs || !s.present_ps)
    return false;

  plume::RenderGraphicsPipelineDesc desc;
  desc.pipelineLayout = s.utility_layout.get();
  desc.vertexShader = s.present_vs.get();
  desc.pixelShader = s.present_ps.get();
  desc.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  desc.cullMode = plume::RenderCullMode::NONE;
  desc.renderTargetCount = 1;
  desc.renderTargetFormat[0] = kSwapChainFormat;
  desc.renderTargetBlend[0] = plume::RenderBlendDesc::Copy();
  desc.depthTargetFormat = plume::RenderFormat::UNKNOWN;
  s.present_pipeline = s.device->createGraphicsPipeline(desc);
  if (!s.present_pipeline) {
    RGPU_ERROR("Present pipeline creation failed");
    return false;
  }
  s.present_pipeline->setName("renut present");
  return true;
}

// See depth_resolve.hlsli; present_vs draws the fullscreen triangle.
bool BuildDepthResolvePipelines(VideoState &s) {
  // banjo-nx (2026-10-18): the colour resolves drawn on deko3d (DrawResolveCopy); optional.
  s.resolve_copy_ps = CreateHostShader(s.device.get(), RENUT_HOST_SHADER(resolve_copy_ps), "resolve_copy_ps");
  s.depth_resolve_ps[0] =
      CreateHostShader(s.device.get(), RENUT_HOST_SHADER(depth_resolve_ps), "depth_resolve_ps");
  s.depth_resolve_ps[1] = CreateHostShader(s.device.get(), RENUT_HOST_SHADER(depth_resolve_ms_ps),
                                           "depth_resolve_ms_ps");
  for (uint32_t i = 0; i < 2; ++i) {
    if (!s.depth_resolve_ps[i])
      return false;
    plume::RenderGraphicsPipelineDesc desc;
    desc.pipelineLayout = s.utility_layout.get();
    desc.vertexShader = s.present_vs.get();
    desc.pixelShader = s.depth_resolve_ps[i].get();
    desc.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
    desc.cullMode = plume::RenderCullMode::NONE;
    desc.renderTargetCount = 2;
    desc.renderTargetFormat[0] = kDepthResolveFormat;
    desc.renderTargetFormat[1] = kDepthResolveWordsFormat;
    desc.renderTargetBlend[0] = plume::RenderBlendDesc::Copy();
    desc.renderTargetBlend[1] = plume::RenderBlendDesc::Copy();
    desc.depthTargetFormat = plume::RenderFormat::UNKNOWN;
    s.depth_resolve_pipeline[i] = s.device->createGraphicsPipeline(desc);
    if (!s.depth_resolve_pipeline[i]) {
      RGPU_ERROR("Depth resolve pipeline creation failed");
      return false;
    }
    s.depth_resolve_pipeline[i]->setName(i ? "renut depth resolve (msaa)" : "renut depth resolve");
    // banjo-nx (2026-10-19, banjo_renut_depth_words_lazy): the depth only (the words target bound, not written).
    desc.renderTargetBlend[1].renderTargetWriteMask = 0;
    s.depth_resolve_float_pipeline[i] = s.device->createGraphicsPipeline(desc);
    if (s.depth_resolve_float_pipeline[i])
      s.depth_resolve_float_pipeline[i]->setName(i ? "renut depth resolve, depth only (msaa)"
                                                   : "renut depth resolve, depth only");
  }
  return true;
}

// Optional: without geometry shaders rectangle lists stay triangle lists and
// point sprites single pixels.
void BuildRectangleListShader(VideoState &s) {
  if (!s.device->getCapabilities().geometryShader) {
    RGPU_WARN("No geometry shader support; rectangle lists are drawn as triangle lists and "
              "point sprites as points");
    return;
  }
  s.rectangle_list_gs =
      CreateHostShader(s.device.get(), RENUT_HOST_SHADER(rectangle_list_gs), "rectangle_list_gs");
  s.point_sprite_gs =
      CreateHostShader(s.device.get(), RENUT_HOST_SHADER(point_sprite_gs), "point_sprite_gs");
}

// See shaders/host/stream_convert_cs.hlsl.
plume::RenderDescriptorSetBuilder StreamConvertSetBuilder() {
  plume::RenderDescriptorSetBuilder buffers;
  buffers.begin();
  buffers.addReadWriteByteAddressBuffer(0, kStreamConvertBuffers);  // u0, space0
  buffers.end();
  return buffers;
}

// Optional: without it, draws that read memexport results as vertices wait
// for them on the CPU.
void BuildStreamConvertPipeline(VideoState &s) {
  plume::RenderPipelineLayoutBuilder layout;
  layout.begin(false, false);
  layout.addDescriptorSet(StreamConvertSetBuilder());
  layout.addPushConstant(0, 1, sizeof(StreamConvertPush), plume::RenderShaderStageFlag::COMPUTE);
  layout.end();
  s.stream_convert_layout = layout.create(s.device.get());
  s.stream_convert_cs =
      CreateHostShader(s.device.get(), RENUT_HOST_SHADER(stream_convert_cs), "stream_convert_cs");
  if (s.stream_convert_layout && s.stream_convert_cs) {
    s.stream_convert_pipeline = s.device->createComputePipeline(plume::RenderComputePipelineDesc(
        s.stream_convert_layout.get(), s.stream_convert_cs.get(), 64, 1, 1));
  }
  if (!s.stream_convert_pipeline) {
    RGPU_WARN("Stream conversion pipeline creation failed; draws reading memexport results as "
              "vertices wait for them");
    return;
  }
  s.stream_convert_pipeline->setName("renut stream convert");
}

// See shaders/host/target_readback_cs.hlsl.
plume::RenderDescriptorSetBuilder TargetReadbackSetBuilder() {
  plume::RenderDescriptorSetBuilder buffers;
  buffers.begin();
  buffers.addReadWriteByteAddressBuffer(0, 1);  // u0, space1
  buffers.end();
  return buffers;
}

// Optional: without it, render targets the game reads back on the CPU (photo
// mode's photos, guest/photo_capture.cpp) are not read back.
void BuildTargetReadbackPipeline(VideoState &s) {
  plume::RenderDescriptorSetBuilder textures;  // the bindless table, as BuildUtilityLayout's
  textures.begin();
  textures.addTexture(0, kTextureDescriptorCount);
  textures.end(true, kTextureDescriptorCount);

  plume::RenderPipelineLayoutBuilder layout;
  layout.begin(false, false);
  layout.addDescriptorSet(textures);                    // set 0 -> t0, space0
  layout.addDescriptorSet(TargetReadbackSetBuilder());  // set 1 -> u0, space1
  layout.addPushConstant(0, 2, sizeof(TargetReadbackPush),
                         plume::RenderShaderStageFlag::COMPUTE);  // b0, space2
  layout.end();
  s.target_readback_layout = layout.create(s.device.get());
  s.target_readback_cs = CreateHostShader(s.device.get(), RENUT_HOST_SHADER(target_readback_cs),
                                          "target_readback_cs");
  if (s.target_readback_layout && s.target_readback_cs) {
    s.target_readback_pipeline = s.device->createComputePipeline(plume::RenderComputePipelineDesc(
        s.target_readback_layout.get(), s.target_readback_cs.get(), 8, 8, 1));
  }
  if (!s.target_readback_pipeline) {
    RGPU_WARN("Render target readback pipeline creation failed; photos stay black");
    return;
  }
  s.target_readback_pipeline->setName("renut target readback");
}

} // namespace

namespace detail {

std::unique_ptr<plume::RenderDescriptorSet> CreateStreamConvertSet(VideoState &s) {
  return StreamConvertSetBuilder().create(s.device.get());
}

std::unique_ptr<plume::RenderDescriptorSet> CreateTargetReadbackSet(VideoState &s) {
  return TargetReadbackSetBuilder().create(s.device.get());
}

} // namespace detail

namespace detail {

bool BuildSwapChainTargets(VideoState &s) {
  s.swap_framebuffers.clear();
  s.present_semaphores.clear();
  const uint32_t count = s.swap_chain->getTextureCount();
  for (uint32_t i = 0; i < count; ++i) {
    const plume::RenderTexture *texture = s.swap_chain->getTexture(i);
    plume::RenderFramebufferDesc desc(&texture, 1);
    auto framebuffer = s.device->createFramebuffer(desc);
    if (!framebuffer) {
      RGPU_ERROR("Swapchain framebuffer {} creation failed", i);
      s.swap_framebuffers.clear();
      return false;
    }
    s.swap_framebuffers.push_back(std::move(framebuffer));
    s.present_semaphores.push_back(s.device->createCommandSemaphore());
  }
  return count != 0;
}

void DrainSlot(VideoState &s, uint32_t slot) {
  s.retired_framebuffers[slot].clear();
  s.retired_views[slot].clear();
  s.retired_textures[slot].clear();
  s.retired_buffers[slot].clear();
  s.retired_sets[slot].clear();
  ReleaseCacheCopiesLocked(slot);
  s.exported_streams_used[slot] = 0;
  s.gpu_fetch_data_used[slot] = 0;
  s.memexport_scratch_used[slot] = 0;
  for (uint32_t index : s.retired_texture_slots[slot]) {
    if (index >= kFirstDynamicTextureSlot && index < kTextureDescriptorCount) {
      // banjo-nx: banjo_renut_diag_poison_slots points it at magenta instead.
      const bool poison = s.poison_texture != nullptr;
      s.texture_set->setTexture(index, poison ? s.poison_texture.get() : s.null_texture.get(),
                                plume::RenderTextureLayout::SHADER_READ,
                                poison ? s.poison_texture_view.get() : s.null_texture_view.get());
      if (s.texture_slot_used[index] && s.texture_slots_in_use)
        --s.texture_slots_in_use;
      s.texture_slot_used[index] = false;
    }
  }
  s.retired_texture_slots[slot].clear();
  s.uploads[slot].Reset();
}

// ---------------------------------------------------------------------------
// banjo-nx (2026-10-19): the ring cleaner (render-thread proposal 23) and the kick submits (input latency)
// ---------------------------------------------------------------------------
// The upload ring is CPU-cached memory: what the CPU wrote is cleaned to memory before the GPU reads it
// (UploadRing::Publish at a submit). At each kick, once the kick's fills are written, the ranges written since the last
// hand-off go to a thread on the background core (a clean by address reaches the line in any core's cache); a submit
// waits for the ranges its lists use, and takes over any the thread has not started (so a busy background core never
// holds the render thread up). Nothing writes a range after it is handed off: the ring only grows, and the fills of a
// kick are written before its hand-off.
// The kick submits: the list recorded up to a kick ends there (once it holds kKickSubmitDraws draws, at most
// kKickLists times a frame) and goes to the GPU at the next kick (its ranges are cleaned by then), or at the present,
// before the frame's last list; the GPU runs the commands in the order they were recorded either way.
namespace {

constexpr uint32_t kKickSubmitDraws = 200;
constexpr uint32_t kKickListsAFrame = 3;
// banjo-nx (2026-10-20, banjo_deko3d_eager_kicks): smaller lists, more of them, each submitted at its own kick.
constexpr uint32_t kEagerKickDraws = 100;
static_assert(kKickListsAFrame <= VideoState::kKickLists);

class RingCleaner {
public:
  // The ranges, cleaned in order; the ticket that Wait takes for them.
  uint64_t Enqueue(std::vector<UploadRing::Range> &&ranges) {
    if (ranges.empty())
      return 0;
    std::lock_guard lock(mutex_);
    Start();
    for (const UploadRing::Range &r : ranges)
      queue_.push_back({r, ++last_ticket_});
    const uint64_t ticket = last_ticket_;
    cv_.notify_one();
    return ticket;
  }

  // Every range up to 'ticket' (all of them for UINT64_MAX) is clean: those not started are cleaned here, and one the
  // thread is cleaning is waited for. (Tickets are in order: a range up to 'ticket' is queued, being cleaned, or done.)
  void Wait(uint64_t ticket) {
    std::unique_lock lock(mutex_);
    if (ticket == UINT64_MAX)
      ticket = last_ticket_;
    for (;;) {
      while (!queue_.empty() && queue_.front().ticket <= ticket) {
        const Job job = queue_.front();
        queue_.pop_front();
        lock.unlock();
        Clean(job.range);
        banjo::renut_gpu::CountKickSubmit(2);
        lock.lock();
      }
      if (working_ == 0 || working_ > ticket)
        return;
      render_thread::WaitScope wait;
      done_cv_.wait(lock, [&] { return working_ == 0 || working_ > ticket; });
    }
  }

private:
  struct Job {
    UploadRing::Range range;
    uint64_t ticket;
  };

  static void Clean(const UploadRing::Range &r) {
    const plume::RenderRange range(r.begin, r.end);
    r.buffer->unmap(0, &range);  // plume: cleans the range of a persistently mapped buffer
  }

  void Start() {
    if (started_)
      return;
    started_ = true;
    banjo::renut_gpu::StartLongLivedThread(
        [] { Instance().Run(); }, 64 * 1024);
  }

  void Run() {
    banjo::renut_gpu::OnBackgroundThreadStart("ring cleaner");
    std::unique_lock lock(mutex_);
    for (;;) {
      cv_.wait(lock, [this] { return !queue_.empty(); });
      const Job job = queue_.front();
      queue_.pop_front();
      working_ = job.ticket;
      lock.unlock();
      Clean(job.range);
      banjo::renut_gpu::CountKickSubmit(1);
      lock.lock();
      working_ = 0;
      done_cv_.notify_all();
    }
  }

public:
  static RingCleaner &Instance() {
    static RingCleaner cleaner;
    return cleaner;
  }

private:
  std::mutex mutex_;
  std::condition_variable cv_, done_cv_;
  std::deque<Job> queue_;
  uint64_t last_ticket_ = 0, working_ = 0;  // working_: the ticket of the range being cleaned, 0 for none
  bool started_ = false;
};

bool UseRingCleaner() {
  return IsDeko3D() && banjo::renut_gpu::RingCleanerOn();
}

// Every range written so far is clean (handed to the cleaner and waited for, the rest by Publish).
void PublishAll(VideoState &s, uint32_t slot) {
  if (UseRingCleaner()) {
    std::vector<UploadRing::Range> ranges;
    s.uploads[slot].HandOff(ranges);
    RingCleaner::Instance().Enqueue(std::move(ranges));
    RingCleaner::Instance().Wait(UINT64_MAX);
  }
  s.uploads[slot].Publish();
}

plume::RenderCommandList *SlotList(VideoState &s, uint32_t slot) {
  return s.slot_list[slot] ? s.slot_list[slot] : s.command_lists[slot].get();
}

} // namespace

namespace {

// After the slot's fence has been waited: its frame's GPU time, the sum of its
// lists' spans (so the gaps where the GPU waited on the CPU are left out).
void CollectGpuTime(VideoState &s, uint32_t slot) {
  const uint32_t lists = s.timed_lists[slot];
  s.timed_lists[slot] = 0;
  plume::RenderQueryPool *pool = s.timestamps[slot].get();
  if (!pool || !lists)
    return;
  pool->queryResults();
  const uint64_t *ns = pool->getResults();
  uint64_t total = 0;
  for (uint32_t i = 0; i < lists; ++i) {
    if (ns[2 * i + 1] > ns[2 * i])
      total += ns[2 * i + 1] - ns[2 * i];
  }
  frame_stats::OnGpuFrame(double(total) / 1e6);
}

} // namespace

void BeginListTiming(VideoState &s) {
  gpu_profile::BeginList(s);
  s.timing_open = false;
  plume::RenderQueryPool *pool = s.timestamps[s.frame].get();
  uint32_t &lists = s.timed_lists[s.frame];
  if (!pool || lists >= VideoState::kTimedLists)
    return;
  // Vulkan queries are reset before they are written again, outside any
  // render pass: at the frame's first list, all of them.
  if (lists == 0)
    s.command_list->resetQueryPool(pool, 0, 2 * VideoState::kTimedLists);
  s.command_list->writeTimestamp(pool, 2 * lists);
  s.timing_open = true;
}

void EndListTiming(VideoState &s, bool last_in_frame) {
  gpu_profile::EndList(s, last_in_frame);
  plume::RenderQueryPool *pool = s.timestamps[s.frame].get();
  if (!pool || !s.timing_open)
    return;
  uint32_t &lists = s.timed_lists[s.frame];
  s.command_list->writeTimestamp(pool, 2 * lists + 1);
  ++lists;
  s.timing_open = false;
  // Vulkan returns no results while any query of the pool is unwritten.
  if (last_in_frame && IsVulkan()) {
    for (uint32_t i = 2 * lists; i < 2 * VideoState::kTimedLists; ++i)
      s.command_list->writeTimestamp(pool, i);
  }
}

// banjo-nx (2026-10-20, input latency, banjo_deko3d_latency_trim). Where the GPU is the limit, the render thread used to
// finish recording a frame long before the GPU could start it (the frame waited in the queue behind the one the GPU
// was running: about the GPU's frame time less the recording time, 30 ms and more in the heaviest scenes), and the game
// thread, released by the render thread, read its controller that much earlier than it needed to. With the key on,
// each kick submission signals a fence, and after a frame is submitted and presented the render thread waits until
// the GPU has finished all but the frame's last part (its last kick list): the next frame is recorded, and the game
// thread released, only then, while the GPU still has that last part to run. If the GPU then finishes the frame before
// the next frame sends it any work (it would sit idle), holding back stops for kTrimBackoffFrames frames.
namespace {
#if defined(PLUME_DEKO3D)
struct LatencyTrim {
  std::unique_ptr<plume::RenderCommandFence> kick_fences[kFramesInFlight][VideoState::kKickLists];
  uint32_t kick_submits[kFramesInFlight] = {};  // fenced kick submissions of the slot's frame
  plume::Deko3DFenceCopy last_kick[kFramesInFlight];  // the last of them
  plume::Deko3DFenceCopy frame_end[kFramesInFlight];  // the slot's frame's final submission
  bool sent[kFramesInFlight] = {};  // the slot's frame has sent the GPU work
  bool held_back = false;           // the frame before the open one was held back
  uint32_t backoff = 0;
  // The 10 s report's (read by the game thread without the lock).
  std::atomic<uint64_t> frames{0}, held{0}, held_us{0}, starved{0};
};
LatencyTrim g_trim;
constexpr uint32_t kTrimBackoffFrames = 60;
#endif
uint64_t g_presenting_swap = 0;                    // the guest Swap the frame being presented ends
uint64_t g_slot_swap[kFramesInFlight] = {};        // the guest Swap each slot's frame ends (0: none)
bool g_gpu_done_noted[kFramesInFlight] = {true, true};
}  // namespace

// The slot's frame's GPU work is done, the first time that is seen: the latency probe's "input -> GPU done". Seen at
// the kicks and the presents (a few ms late at most) or at the wait for the slot (exact when the wait blocked).
void NoteGpuDoneSeenLocked(uint32_t slot, bool waited) {
  if (g_gpu_done_noted[slot])
    return;
#if defined(PLUME_DEKO3D)
  if (!waited && !plume::Deko3DFenceCopyDone(g_trim.frame_end[slot]))
    return;
#else
  if (!waited)
    return;
#endif
  g_gpu_done_noted[slot] = true;
  rex::latency::NoteGpuDone(g_slot_swap[slot]);
}

#if defined(PLUME_DEKO3D)
// The slot's frame sends the GPU its first work. If the frame before was held back and the GPU has finished it, the GPU
// sat idle: no holding back for a while.
void FirstGpuWorkLocked(uint32_t slot) {
  LatencyTrim &t = g_trim;
  if (t.sent[slot])
    return;
  t.sent[slot] = true;
  const uint32_t prev = (slot + kFramesInFlight - 1) % kFramesInFlight;
  if (t.held_back && plume::Deko3DFenceCopyDone(t.frame_end[prev])) {
    t.starved.fetch_add(1, std::memory_order_relaxed);
    t.backoff = kTrimBackoffFrames;
  }
  t.held_back = false;
}
#endif

void LatencyTrimLocked(VideoState &s) {
#if defined(PLUME_DEKO3D)
  LatencyTrim &t = g_trim;
  t.frames.fetch_add(1, std::memory_order_relaxed);
  const uint32_t prev = (s.frame + kFramesInFlight - 1) % kFramesInFlight;  // the frame just submitted
  NoteGpuDoneSeenLocked((prev + kFramesInFlight - 1) % kFramesInFlight, false);
  if (!banjo::renut_gpu::LatencyTrimOn())
    return;
  if (t.backoff) {
    --t.backoff;
    return;
  }
  if (!t.kick_submits[prev])
    return;  // no kick list in the frame: nothing to wait for short of the whole frame
  const auto start = std::chrono::steady_clock::now();
  {
    render_thread::WaitScope wait;
    plume::Deko3DFenceCopyWait(t.last_kick[prev], s.device.get());
  }
  const uint64_t us = uint64_t(
      std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - start).count());
  if (us >= 500) {
    t.held_back = true;
    t.held.fetch_add(1, std::memory_order_relaxed);
    t.held_us.fetch_add(us, std::memory_order_relaxed);
  }
#else
  (void)s;
#endif
}

std::string LatencyTrimReport() {
#if defined(PLUME_DEKO3D)
  LatencyTrim &t = g_trim;
  if (!banjo::renut_gpu::LatencyTrimOn())
    return {};
  const uint64_t frames = t.frames.exchange(0, std::memory_order_relaxed);
  const uint64_t held = t.held.exchange(0, std::memory_order_relaxed);
  const uint64_t held_us = t.held_us.exchange(0, std::memory_order_relaxed);
  const uint64_t starved = t.starved.exchange(0, std::memory_order_relaxed);
  return fmt::format("latency trim (banjo_deko3d_latency_trim): {} of {} frames held back {:.1f} ms on average; the "
                     "GPU ran dry after one {} times (held back no more for {} frames each)",
                     held, frames, held ? double(held_us) / 1000.0 / double(held) : 0.0, starved, kTrimBackoffFrames);
#else
  return {};
#endif
}

void WaitSlotLocked(VideoState &s, uint32_t slot) {
  if (!s.submitted[slot])
    return;
  NoteGpuDoneSeenLocked(slot, false);  // banjo-nx (2026-10-20): done already (seen now)
  {
    render_thread::WaitScope wait;
    s.queue->waitForCommandFence(s.fences[slot].get());
  }
  NoteGpuDoneSeenLocked(slot, true);
  s.submitted[slot] = false;
  CollectGpuTime(s, slot);
  gpu_profile::Collect(s, slot);
  // banjo-nx (2026-10-19, banjo_deko3d_kick_submit): every list of the slot's frame has run: its kick lists are free.
  s.kick_lists_used[slot] = 0;
  s.slot_list[slot] = nullptr;
}

void WaitAllSubmitted(VideoState &s) {
  // Oldest slot first, so its memexport results land before newer ones.
  for (uint32_t k = 1; k <= kFramesInFlight; ++k) {
    const uint32_t i = (s.frame + k) % kFramesInFlight;
    if (s.submitted[i]) {
      WaitSlotLocked(s, i);
      CompleteMemExportsLocked(i);
      if (i != s.frame)
        DrainSlot(s, i);
    }
  }
}

void SubmitAndWaitLocked(VideoState &s) {
  if (!s.list_open)
    return;
  const uint32_t cur = s.frame;
  FinishRecordingLocked(s);
  EndListTiming(s, false);
  banjo::renut_gpu::record::Drain();  // banjo-nx (banjo_renut_record_worker): played on the real list first
  plume::RenderCommandList *current = SlotList(s, cur);
  current->end();
  PublishAll(s, cur);  // banjo-nx: banjo_renut_upload_cached (and the cleaner's ranges, 2026-10-19)
  // banjo-nx (2026-10-19, banjo_deko3d_kick_submit): the list ended at the last kick first.
  const plume::RenderCommandList *lists[] = {s.pending_list ? s.pending_list : current, current};
  const uint32_t count = s.pending_list ? 2 : 1;
  s.pending_list = nullptr;
  s.queue->executeCommandLists(lists, count, nullptr, 0, nullptr, 0, s.fences[cur].get());
  {
    render_thread::WaitScope wait;
    s.queue->waitForCommandFence(s.fences[cur].get());
  }
  s.submitted[cur] = false;
  // The slot's upload ring and retired objects stay: the rest of the frame
  // records into the same slot. banjo-nx: every list of it has run, so its first list is free again.
  s.kick_lists_used[cur] = 0;
  s.slot_list[cur] = nullptr;
  s.command_list = banjo::renut_gpu::record::Wrap(s.command_lists[cur].get());
  s.command_list->begin();
  s.list_draws = 0;
  ++s.list_generation;
  BeginListTiming(s);
}

void SubmitAndAdvance(VideoState &s, plume::RenderCommandSemaphore *wait,
                      plume::RenderCommandSemaphore *signal) {
  const uint32_t cur = s.frame;
  if (s.list_open) {
    FinishRecordingLocked(s);
    EndListTiming(s, true);
    banjo::renut_gpu::record::Drain();  // banjo-nx (banjo_renut_record_worker)
    plume::RenderCommandList *current = SlotList(s, cur);
    current->end();
    s.list_open = false;
    s.command_list = nullptr;
    PublishAll(s, cur);  // banjo-nx: banjo_renut_upload_cached (and the cleaner's ranges, 2026-10-19)
    // banjo-nx (2026-10-19, banjo_deko3d_kick_submit): the list ended at the last kick first.
    const plume::RenderCommandList *lists[] = {s.pending_list ? s.pending_list : current, current};
    const uint32_t count = s.pending_list ? 2 : 1;
    s.pending_list = nullptr;
    plume::RenderCommandSemaphore *waits[] = {wait};
    plume::RenderCommandSemaphore *signals[] = {signal};
    s.queue->executeCommandLists(lists, count, wait ? waits : nullptr, wait ? 1 : 0,
                                 signal ? signals : nullptr, signal ? 1 : 0,
                                 s.fences[cur].get());
    s.submitted[cur] = true;
    // banjo-nx (2026-10-20, input latency): the guest Swap this frame ends, for the probe's "input -> GPU done".
    g_slot_swap[cur] = g_presenting_swap;
    g_gpu_done_noted[cur] = g_presenting_swap == 0;
#if defined(PLUME_DEKO3D)
    g_trim.frame_end[cur] = plume::Deko3DCopyFence(s.fences[cur].get());
    FirstGpuWorkLocked(cur);
#endif
  }
  g_presenting_swap = 0;

  // Record into the other slot next; the GPU keeps working on this one.
  s.frame = (cur + 1) % kFramesInFlight;
  WaitSlotLocked(s, s.frame);
#if defined(PLUME_DEKO3D)
  g_trim.kick_submits[s.frame] = 0;  // banjo-nx (2026-10-20, banjo_deko3d_latency_trim): the slot's new frame
  g_trim.last_kick[s.frame] = {};
  g_trim.sent[s.frame] = false;
#endif
  CompleteMemExportsLocked(s.frame);
  s.timed_lists[s.frame] = 0;
  DrainSlot(s, s.frame);
}

} // namespace detail

bool Video::CreateHostDevice() {
  auto &s = State();
  std::lock_guard lock(s.mutex);
  if (s.host_ready)
    return true;

  plume::RenderWindow render_window{};
  if (!GetNativeRenderWindow(render_window))
    return false;

#if defined(_WIN32)
  // plume reports its failures (vkQueueSubmit, swapchain creation, ...) on
  // stderr, which a Windows GUI process without a console discards.
  // Opened shared (freopen_s denies other readers), so it can be read while
  // the game runs.
  if (_fileno(stderr) < 0) {
    std::error_code ec;
    std::filesystem::create_directories("logs", ec);
    FILE *file = nullptr;
    if (freopen_s(&file, "NUL", "w", stderr) == 0) {
      if (FILE *log = _fsopen("logs/plume_stderr.log", "w", _SH_DENYNO)) {
        _dup2(_fileno(log), _fileno(stderr));
        setvbuf(stderr, nullptr, _IONBF, 0);
        RGPU_INFO("plume's error messages go to logs/plume_stderr.log");
      }
    }
  }
#endif

  banjo::renut_gpu::BeforeDeviceCreated();  // banjo-nx: plume's pipeline cache and barriers (banjo_renut_tuning.h)
  s.render_interface = CreateRenderInterface();
  if (!s.render_interface) {
    RGPU_ERROR("{} render interface creation failed", BackendName(ActiveBackend()));
    return false;
  }
  s.device = s.render_interface->createDevice();
  if (!s.device) {
    RGPU_ERROR("{} device creation failed", BackendName(ActiveBackend()));
    return false;
  }
  banjo::renut_gpu::OnDeviceCreated(s.device.get());
  s.backend_description =
      std::string(BackendName(ActiveBackend())) + " on " + s.device->getDescription().name;
  RGPU_INFO("Renderer: {}", s.backend_description);

  s.queue = s.device->createCommandQueue(plume::RenderCommandListType::DIRECT);
  if (!s.queue) {
    RGPU_ERROR("Command queue creation failed");
    return false;
  }
  for (uint32_t i = 0; i < kFramesInFlight; ++i) {
    s.command_lists[i] = s.queue->createCommandList();
    s.fences[i] = s.device->createCommandFence();
    s.acquire_semaphores[i] = s.device->createCommandSemaphore();
    if (!s.command_lists[i] || !s.fences[i] || !s.acquire_semaphores[i]) {
      RGPU_ERROR("Frame {} command objects creation failed", i);
      return false;
    }
    // banjo-nx: on Vulkan each chunk carries the draws' uniform buffer descriptor sets (upload_ring.h).
    if (!s.uploads[i].Init(s.device.get(), i == 0 ? "upload ring 0" : "upload ring 1", UsesVulkanBindings(),
                           banjo::renut_gpu::UploadCached()))
      return false;
    if (s.device->getCapabilities().queryPools)
      s.timestamps[i] = s.device->createQueryPool(2 * VideoState::kTimedLists);
  }

  // One image beyond the frames in flight, so acquiring never waits on scanout.
  // sa2-nx (2026-10-20, sa2_renut_double_buffer): two images, so a finished frame is shown at the next refresh instead
  // of queueing behind one already waiting for it; a frame that misses a refresh then waits a whole one.
  static const bool double_buffer = REXCVAR_GET(sa2_renut_double_buffer);
  const uint32_t images = double_buffer ? 2u : kFramesInFlight + 1;
  plume::RenderSwapChainDesc swap_desc(render_window, kSwapChainFormat, images, false, kFramesInFlight);
  RGPU_INFO("Swapchain: {} images{}", images, double_buffer ? " (sa2_renut_double_buffer)" : "");
  s.swap_chain = s.queue->createSwapChain(swap_desc);
  // plume's Vulkan swapchain is created lazily by resize().
  if (s.swap_chain && IsVulkan())
    s.swap_chain->resize();
  if (!s.swap_chain) {
    RGPU_ERROR("Swapchain creation failed");
    return false;
  }
  // banjo-nx: a window that starts minimized (window_start_minimized, the automated tests) has no
  // surface extent yet: PresentLocked builds the swapchain's targets once it has one (needsResize),
  // presenting nothing until then, as it does while minimized.
  if (s.swap_chain->isEmpty() || !detail::BuildSwapChainTargets(s)) {
    if (s.swap_chain->getWidth() && s.swap_chain->getHeight()) {
      RGPU_ERROR("Swapchain creation failed");
      return false;
    }
    RGPU_INFO("Swapchain: the window has no size yet (minimized); presenting starts when it has one");
  }

  if (!BuildUtilityLayout(s) || !BuildSamplers(s) || !BuildNullTexture(s) ||
      !BuildPresentPipeline(s) || !BuildDepthResolvePipelines(s) || !BuildDrawLayout(s)) {
    return false;
  }
  BuildRectangleListShader(s);
  BuildStreamConvertPipeline(s);
  BuildTargetReadbackPipeline(s);
  // Optional: without it, scaled resolve destinations keep one level.
  s.texture_mip_ps =
      CreateHostShader(s.device.get(), RENUT_HOST_SHADER(texture_mip_ps), "texture_mip_ps");
  StartPipelinePrebuild();

  s.host_ready = true;
  return true;
}

bool Video::HostReady() {
  auto &s = State();
  std::lock_guard lock(s.mutex);
  return s.host_ready;
}

void Video::BeginShutdown() {
  auto &s = State();
  s.shutting_down.store(true, std::memory_order_release);
  StopPipelinePrebuild();
  render_thread::Stop();
  std::lock_guard lock(s.mutex);
  if (s.queue)
    detail::WaitAllSubmitted(s);
  banjo::renut_gpu::BeforeDeviceDestroyed();  // banjo-nx: the pipeline cache's last save
}

plume::RenderDevice *Video::Device() {
  return State().device.get();
}

void Video::SetGuestDevice(uint32_t device_va) {
  auto &s = State();
  std::lock_guard lock(s.mutex);
  s.guest_device.store(device_va, std::memory_order_release);
  s.guest_ready = device_va != 0;
}

uint32_t Video::GuestDevice() {
  return State().guest_device.load(std::memory_order_acquire);
}

void Video::RequestResize() {
  State().resize_requested.store(true, std::memory_order_release);
}

void Video::SetOverlayHook(OverlayHook hook) {
  auto &s = State();
  std::lock_guard lock(s.mutex);
  s.overlay_hook = std::move(hook);
}

bool Video::EnsureCommandListLocked() {
  auto &s = State();
  if (s.list_open)
    return true;
  if (!s.host_ready || s.shutting_down.load(std::memory_order_acquire))
    return false;

  // banjo-nx (banjo_renut_record_worker, package 28b): the proxy over the slot's list, played by the worker.
  s.command_list = banjo::renut_gpu::record::Wrap(detail::SlotList(s, s.frame));
  s.command_list->begin();
  s.list_open = true;
  s.list_draws = 0;  // banjo-nx (2026-10-19, banjo_deko3d_kick_submit)
  ++s.list_generation;
  detail::BeginListTiming(s);

  if (!s.null_texture_initialized)
    s.null_texture_initialized = InitializeNullTextures(s);
  return true;
}

void Video::NoteDrawLocked() {
  ++State().list_draws;
}

void Video::NotePresentingSwap(uint64_t swap) {
  detail::g_presenting_swap = swap;  // the render thread, right before the present that submits the frame
}

std::string Video::LatencyTrimReport() {
  return detail::LatencyTrimReport();  // atomics: no lock (the game thread's report)
}

void Video::KickLocked() {
  auto &s = State();
  if (!s.list_open || !IsDeko3D())
    return;
  const uint32_t cur = s.frame;
#if defined(PLUME_DEKO3D)
  // banjo-nx (2026-10-20): the frame before done? Seen at every kick (package 36i noted it only where a list went out
  // at the next kick, so with banjo_deko3d_eager_kicks "input -> GPU done" was seen later, not done later).
  detail::NoteGpuDoneSeenLocked((cur + kFramesInFlight - 1) % kFramesInFlight, false);
#endif
  // The ring written up to this kick (its fills included) to the cleaner.
  uint64_t ticket = 0;
  if (detail::UseRingCleaner()) {
    std::vector<UploadRing::Range> ranges;
    s.uploads[cur].HandOff(ranges);
    ticket = detail::RingCleaner::Instance().Enqueue(std::move(ranges));
  }
  // The list ended at the last kick: its memory is clean (its ranges were handed off then).
  if (s.pending_list) {
    if (detail::UseRingCleaner())
      detail::RingCleaner::Instance().Wait(s.pending_ticket);
    else
      s.uploads[cur].Publish();
    const plume::RenderCommandList *lists[] = {s.pending_list};
    // banjo-nx (2026-10-20, banjo_deko3d_latency_trim): a fence for each kick submission, how far the GPU has got.
    plume::RenderCommandFence *fence = nullptr;
#if defined(PLUME_DEKO3D)
    detail::LatencyTrim &t = detail::g_trim;
    if (banjo::renut_gpu::LatencyTrimOn() && t.kick_submits[cur] < VideoState::kKickLists) {
      std::unique_ptr<plume::RenderCommandFence> &f = t.kick_fences[cur][t.kick_submits[cur]];
      if (!f)
        f = s.device->createCommandFence();
      fence = f.get();
    }
#endif
    s.queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, fence);
#if defined(PLUME_DEKO3D)
    if (fence) {
      t.last_kick[cur] = plume::Deko3DCopyFence(fence);
      ++t.kick_submits[cur];
    }
    detail::FirstGpuWorkLocked(cur);
    detail::NoteGpuDoneSeenLocked((cur + kFramesInFlight - 1) % kFramesInFlight, false);
#endif
    s.pending_list = nullptr;
    banjo::renut_gpu::CountKickSubmit(0);
  }
  // The open list ends here when it holds enough draws; the frame goes on in a kick list of the slot.
  const bool eager = banjo::renut_gpu::EagerKicksOn();
  if (!banjo::renut_gpu::KickSubmitOn() ||
      s.list_draws < (eager ? detail::kEagerKickDraws : detail::kKickSubmitDraws) ||
      s.kick_lists_used[cur] >= (eager ? VideoState::kKickLists : detail::kKickListsAFrame))
    return;
  std::unique_ptr<plume::RenderCommandList> &next = s.kick_lists[cur][s.kick_lists_used[cur]];
  if (!next) {
    next = s.queue->createCommandList();
    if (!next)
      return;
  }
  detail::EndListTiming(s, false);
  banjo::renut_gpu::record::Drain();
  plume::RenderCommandList *current = detail::SlotList(s, cur);
  current->end();
  s.pending_list = current;
  s.pending_ticket = detail::UseRingCleaner() ? ticket : 0;
  ++s.kick_lists_used[cur];
  s.slot_list[cur] = next.get();
  s.command_list = banjo::renut_gpu::record::Wrap(next.get());
  s.command_list->begin();
  s.list_draws = 0;
  ++s.list_generation;
  detail::BeginListTiming(s);
#if defined(PLUME_DEKO3D)
  // banjo-nx (2026-10-20, banjo_deko3d_latency_trim): the frame's first list goes to the GPU at its own kick, not at
  // the next one (its memory cleaned first: the cleaner is waited for): after a frame held back, the GPU then has the
  // next frame's first part as soon as it is recorded.
  // banjo-nx (2026-10-20, banjo_deko3d_eager_kicks): every list at its own kick, so the GPU works while the render
  // thread is still replaying the frame.
  if (((banjo::renut_gpu::LatencyTrimOn() && !detail::g_trim.sent[cur]) || eager) && s.pending_list) {
    if (detail::UseRingCleaner())
      detail::RingCleaner::Instance().Wait(s.pending_ticket);
    else
      s.uploads[cur].Publish();
    detail::LatencyTrim &t = detail::g_trim;
    plume::RenderCommandFence *fence = nullptr;
    if (t.kick_submits[cur] < VideoState::kKickLists) {
      std::unique_ptr<plume::RenderCommandFence> &f = t.kick_fences[cur][t.kick_submits[cur]];
      if (!f)
        f = s.device->createCommandFence();
      fence = f.get();
    }
    const plume::RenderCommandList *lists[] = {s.pending_list};
    s.queue->executeCommandLists(lists, 1, nullptr, 0, nullptr, 0, fence);
    if (fence) {
      t.last_kick[cur] = plume::Deko3DCopyFence(fence);
      ++t.kick_submits[cur];
    }
    detail::FirstGpuWorkLocked(cur);
    s.pending_list = nullptr;
    banjo::renut_gpu::CountKickSubmit(0);
  }
#endif
}

plume::RenderCommandList *Video::CommandListLocked() {
  return State().command_list;
}

UploadRing &Video::UploadsLocked() {
  auto &s = State();
  return s.uploads[s.frame];
}

uint32_t Video::AllocateTextureSlotLocked() {
  auto &s = State();
  for (uint32_t i = kFirstDynamicTextureSlot; i < kTextureDescriptorCount; ++i) {
    if (!s.texture_slot_used[i]) {
      s.texture_slot_used[i] = true;
      ++s.texture_slots_in_use;
      return i;
    }
  }
  RGPU_ERROR("Bindless texture table full ({} slots)", kTextureDescriptorCount);
  return kInvalidDescriptor;
}

void Video::SetTextureSlotLocked(uint32_t slot, plume::RenderTexture *texture,
                                 plume::RenderTextureView *view) {
  if (slot >= kTextureDescriptorCount)
    return;
  State().texture_set->setTexture(slot, texture, plume::RenderTextureLayout::SHADER_READ, view);
}

uint32_t Video::TextureSlotsInUseLocked() {
  return State().texture_slots_in_use;
}

void Video::FreeTextureSlotLocked(uint32_t slot) {
  auto &s = State();
  if (slot >= kFirstDynamicTextureSlot && slot < kTextureDescriptorCount)
    s.retired_texture_slots[s.frame].push_back(slot);
}

void Video::RetireLocked(std::unique_ptr<plume::RenderTexture> texture) {
  if (texture) {
    auto &s = State();
    s.retired_textures[s.frame].push_back(std::move(texture));
  }
}

void Video::RetireLocked(std::unique_ptr<plume::RenderTextureView> view) {
  if (view) {
    auto &s = State();
    s.retired_views[s.frame].push_back(std::move(view));
  }
}

void Video::RetireLocked(std::unique_ptr<plume::RenderBuffer> buffer) {
  if (buffer) {
    auto &s = State();
    s.retired_buffers[s.frame].push_back(std::move(buffer));
  }
}

void Video::RetireLocked(std::unique_ptr<plume::RenderFramebuffer> framebuffer) {
  if (framebuffer) {
    auto &s = State();
    s.retired_framebuffers[s.frame].push_back(std::move(framebuffer));
  }
}

void Video::RetireLocked(std::unique_ptr<plume::RenderDescriptorSet> set) {
  if (set) {
    auto &s = State();
    s.retired_sets[s.frame].push_back(std::move(set));
  }
}

bool Video::UploadRGBA8Locked(plume::RenderTexture *texture, uint32_t width, uint32_t height,
                              const uint8_t *rgba) {
  auto &s = State();
  if (!s.list_open || !texture || !rgba || width == 0 || height == 0)
    return false;

  // D3D12 requires 256-byte aligned row pitches for buffer->texture copies.
  const uint32_t row_bytes = width * 4;
  const uint32_t row_pitch = (row_bytes + 255u) & ~255u;
  UploadAllocation staging = s.uploads[s.frame].Allocate(uint64_t(row_pitch) * height, 512);
  if (!staging)
    return false;
  auto *dst = static_cast<uint8_t *>(staging.cpu);
  for (uint32_t y = 0; y < height; ++y)
    std::memcpy(dst + size_t(y) * row_pitch, rgba + size_t(y) * row_bytes, row_bytes);

  s.command_list->barriers(plume::RenderBarrierStage::COPY,
                           plume::RenderTextureBarrier(texture, plume::RenderTextureLayout::COPY_DEST));
  s.command_list->copyTextureRegion(
      plume::RenderTextureCopyLocation::Subresource(texture, 0),
      plume::RenderTextureCopyLocation::PlacedFootprint(staging.ref.ref,
                                                        plume::RenderFormat::R8G8B8A8_UNORM, width,
                                                        height, 1, row_pitch / 4,
                                                        staging.ref.offset));
  s.command_list->barriers(plume::RenderBarrierStage::GRAPHICS,
                           plume::RenderTextureBarrier(texture, plume::RenderTextureLayout::SHADER_READ));
  return true;
}

} // namespace renut::gpu
