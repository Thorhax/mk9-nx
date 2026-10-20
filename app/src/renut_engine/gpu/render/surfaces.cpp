/**
 * @file    gpu/render/surfaces.cpp
 */
#include "renut_engine/gpu/render/surfaces.h"

#include <algorithm>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/cvar.h>

#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"
#include "renut_engine/gpu/d3d/d3d.h"
#include "renut_engine/gpu/render/format.h"
#include "renut_engine/gpu/render/graphics_options.h"
#include "renut_engine/gpu/render/textures.h"
#include "renut_engine/gpu/guest/guest_memory.h"
#include "renut_engine/gpu/guest/widescreen.h"
#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/backend/video.h"

REXCVAR_DEFINE_STRING(renut_resolution, "720p", "Nuts&Bolts/Graphics",
                      "Internal render resolution, by its height: 480p, 720p (the game's own), "
                      "1080p, 1440p or 4K (2160 lines). Applies from the next frame.")
    .allowed({"480p", "720p", "1080p", "1440p", "4K"});
// sa2-nx (2026-10-20, the user's request: MSAA for the final build, 2x): SA2 makes every render target single-sample,
// so Renut's multisampling (N&B's targets are multisampled: banjo-nx's resolves, depth resolves and readbacks of them)
// never ran. The game's single-sample targets get this many samples on the host, as the native renderer's sa2_msaa did;
// the game's resolves downsample them. Its 4x surfaces (Direct3D's fast Clear aliases) stay as they are.
REXCVAR_DEFINE_INT32(sa2_renut_msaa, 2, "SA2/Renderer",
                     "Renut's renderer: multisample anti-aliasing of the game's single-sample render targets: 0 = off "
                     "(the game's own), 2 = 2x (default), 4 = 4x, 8 = 8x. Smoother edges; each step costs more GPU time "
                     "and memory (8x: about 30 MB more for each 1280x720 target). Read at start")
    .range(0, 8)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace renut::gpu {

namespace {

// ResolutionScale, once latched.
RenderScale g_resolution_scale;
bool g_resolution_latched = false;

struct FramebufferKey {
  const HostSurface *colors[d3d::kMaxRenderTargets] = {};
  uint64_t color_generations[d3d::kMaxRenderTargets] = {};
  const HostSurface *depth = nullptr;
  uint64_t depth_generation = 0;

  bool operator==(const FramebufferKey &o) const {
    for (uint32_t i = 0; i < d3d::kMaxRenderTargets; ++i) {
      if (colors[i] != o.colors[i] || color_generations[i] != o.color_generations[i])
        return false;
    }
    return depth == o.depth && depth_generation == o.depth_generation;
  }
};

struct CachedFramebuffer {
  FramebufferKey key;
  std::unique_ptr<plume::RenderFramebuffer> framebuffer;
};

// A D3DSurface is a view of EDRAM: its base tile (RB_COLOR_INFO /
// RB_DEPTH_INFO bits 0-11), pitch (RB_SURFACE_INFO bits 0-13) and sample
// count. Surface objects over the same EDRAM with the same layout see the same
// pixels on the GPU, and N&B relies on it: the depth surface its after-effects
// pass binds is a second object over the scene's depth buffer (the character
// holograms test against it), and each hologram renders through surfaces of
// its own at the same EDRAM bases. So host targets belong to the EDRAM view,
// shared by every surface object over it, as long as they also agree on the
// host format and render scale: graphics_options renders some surfaces at
// another scale (the post-processing buffers at 1x), and a view used by
// surfaces of both kinds would otherwise be rebuilt at every switch.
uint64_t TargetKey(const d3d::SurfaceDesc &desc, bool depth, plume::RenderFormat format,
                   RenderScale scale) {
  const u32 info = desc.SurfaceInfo;
  return uint64_t(u32(desc.ColorInfo) & 0xFFF) | uint64_t(info & 0x3FFF) << 12 |
         uint64_t((info >> 16) & 3) << 26 | uint64_t(depth) << 28 |
         uint64_t(scale.code & 7) << 29 | uint64_t(u32(format)) << 32;
}

std::unordered_map<uint64_t, std::unique_ptr<HostSurface>> g_targets;
// Surface VA -> its target's key, and how many surfaces use each target.
std::unordered_map<u32, uint64_t> g_surface_keys;
std::unordered_map<uint64_t, uint32_t> g_target_users;
std::vector<CachedFramebuffer> g_framebuffers;
// banjo-nx (2026-10-19, render-thread proposal 8): the last framebuffers looked up, checked before the search of every
// framebuffer made (a linear search with an 80-byte compare at every draw's BindTargets); emptied whenever a cached
// framebuffer is retired (g_framebuffer_epoch).
struct RecentFramebuffer {
  FramebufferKey key;
  plume::RenderFramebuffer *framebuffer = nullptr;
  uint64_t epoch = 0;
};
RecentFramebuffer g_recent_framebuffers[4];
uint64_t g_framebuffer_epoch = 1;
uint64_t g_next_generation = 1;
// banjo-nx (2026-10-19, banjo_deko3d_depth24): the D3D depth formats each EDRAM depth view has been bound as (1 D24S8,
// 2 D24FS8). The host format is part of a target's key, so a view used as both would get two targets where the console
// has one memory: once a view has been bound as D24FS8, every surface over it takes the 32-bit float format.
std::unordered_map<uint64_t, uint8_t> g_depth_view_formats;
// SetTilingExtentLocked; 0 x 0 when not tiling.
uint32_t g_tiling_width = 0;
uint32_t g_tiling_height = 0;

// Drops cached framebuffers that reference 'surface'.
void ForgetFramebuffers(const HostSurface *surface) {
  for (size_t i = 0; i < g_framebuffers.size();) {
    const FramebufferKey &key = g_framebuffers[i].key;
    bool uses = key.depth == surface;
    for (const HostSurface *color : key.colors)
      uses = uses || color == surface;
    if (uses) {
      Video::RetireLocked(std::move(g_framebuffers[i].framebuffer));
      g_framebuffers[i] = std::move(g_framebuffers.back());
      g_framebuffers.pop_back();
      ++g_framebuffer_epoch;
    } else {
      ++i;
    }
  }
}

// Retires the target and everything made over it (deferred past in-flight
// frames).
// banjo-nx (2026-10-19, CPU proposal 8, banjo_renut_cpu_memos): GetHostSurfaceLocked's result per surface VA and the
// three descriptor words it reads, valid while nothing it reads has changed (g_surface_epoch: bumped by every target
// built, retired, unused or rebound, the resolution scale latched and the tiling extent set).
struct SurfaceMemo {
  u32 va = 0, size = 0, info = 0, format = 0;
  uint64_t epoch = 0;
  HostSurface *surface = nullptr;
};
SurfaceMemo g_surface_memo[64];  // banjo-nx (2026-10-19, render-thread proposal 8): 64, indexed by a hash of the VA
uint64_t g_surface_epoch = 1;

void FreeDepthViews(std::vector<DepthImage::View> &views) {
  for (DepthImage::View &v : views) {
    if (v.slot != UINT32_MAX && v.slot != kInvalidDescriptor)
      Video::FreeTextureSlotLocked(v.slot);
    Video::RetireLocked(std::move(v.view));
  }
  views.clear();
}

void RetireTarget(HostSurface &surface) {
  if (surface.depth && surface.handed)
    DepthHandOverBeforeRetireLocked(&surface);  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover)
  FreeDepthViews(surface.handover_views);
  surface.stencil_used = false;
  ++g_surface_epoch;
  ForgetFramebuffers(&surface);
  if (surface.sample_slot != UINT32_MAX)
    Video::FreeTextureSlotLocked(surface.sample_slot);
  surface.sample_slot = UINT32_MAX;
  Video::RetireLocked(std::move(surface.sample_view));
  if (surface.stencil_slot != UINT32_MAX && surface.stencil_slot != kInvalidDescriptor)
    Video::FreeTextureSlotLocked(surface.stencil_slot);
  surface.stencil_slot = UINT32_MAX;
  Video::RetireLocked(std::move(surface.stencil_view));
  Video::RetireLocked(std::move(surface.texture));
}

plume::RenderSampleCounts SampleCount(u32 info_word) {
  switch ((info_word >> 16) & 3) {
  case 1:
    return plume::RenderSampleCount::COUNT_2;
  case 2:
    return plume::RenderSampleCount::COUNT_4;
  default:
    return plume::RenderSampleCount::COUNT_1;
  }
}

// At least 'min_width' x 'min_height' (see SetTilingExtentLocked), at 'scale'.
bool BuildTarget(HostSurface &surface, const d3d::SurfaceDesc &desc, uint32_t min_width,
                 uint32_t min_height, RenderScale scale) {
  surface.size_word = desc.Size;
  surface.info_word = desc.SurfaceInfo;
  surface.format_word = desc.Format;
  surface.width = std::max(d3d::SurfaceWidth(surface.size_word), min_width);
  surface.height = std::max(d3d::SurfaceHeight(surface.size_word), min_height);
  surface.scale = scale;
  std::fill(std::begin(surface.profile_labels), std::end(surface.profile_labels), UINT32_MAX);
  const uint32_t host_width = surface.scale.Size(surface.width);
  const uint32_t host_height = surface.scale.Size(surface.height);
  surface.depth = d3d::IsDepthFormat(surface.format_word);
  surface.samples = SampleCount(surface.info_word);
  if (surface.samples == plume::RenderSampleCount::COUNT_1) {
    // sa2-nx (2026-10-20, sa2_renut_msaa): the game's single-sample targets multisampled on the host.
    static const plume::RenderSampleCounts forced = [] {
      const int32_t n = REXCVAR_GET(sa2_renut_msaa);
      return n >= 8 ? plume::RenderSampleCounts(plume::RenderSampleCount::COUNT_8)
             : n >= 4 ? plume::RenderSampleCounts(plume::RenderSampleCount::COUNT_4)
             : n >= 2 ? plume::RenderSampleCounts(plume::RenderSampleCount::COUNT_2)
                      : plume::RenderSampleCounts(plume::RenderSampleCount::COUNT_1);
    }();
    surface.samples = forced;
  }
  if (surface.depth) {
    surface.format = DepthTargetFormat(surface.format_word);  // banjo-nx (2026-10-19, banjo_deko3d_depth24)
    if (banjo::renut_gpu::Depth24On()) {
      // As GetHostSurfaceSlow keys it: a view ever bound as D24FS8 is 32-bit float for every surface over it.
      const auto it = g_depth_view_formats.find(TargetKey(desc, true, plume::RenderFormat::UNKNOWN, scale));
      if (it != g_depth_view_formats.end() && (it->second & 2))
        surface.format = DepthTargetFormat();
    }
    if (surface.format != DepthTargetFormat()) {
      static uint32_t logged = 0;
      if (logged++ < 16)
        RGPU_INFO("Depth target {:08X}: {}x{} x{} samples, 24-bit unorm depth with stencil (banjo_deko3d_depth24)",
                  surface.guest_va, surface.width, surface.height, u32(surface.samples));
    }
  } else {
    surface.format = SurfaceColorFormat(desc);
  }

  plume::RenderDevice *device = Video::Device();
  const plume::RenderMultisampling multisampling(surface.samples);
  // Sampled too: resolves read them, and so does scan-out of a bound target.
  if (surface.depth) {
    surface.texture = device->createTexture(plume::RenderTextureDesc::DepthTarget(
        host_width, host_height, surface.format, multisampling));
  } else {
    surface.texture = device->createTexture(plume::RenderTextureDesc::ColorTarget(
        host_width, host_height, surface.format, multisampling));
  }
  surface.layout = plume::RenderTextureLayout::UNKNOWN;
  surface.generation = g_next_generation++;
  if (!TextureCreated(surface.texture.get())) {
    surface.texture.reset();
    if (!ReportDeviceLost(device)) {
      RGPU_ERROR("Surface {:08X}: {}x{} {} target creation failed", surface.guest_va,
                 host_width, host_height, surface.depth ? "depth" : "color");
    }
    return false;
  }
  surface.texture->setName(surface.depth ? "renut depth surface" : "renut color surface");
  RGPU_DEBUG("Surface {:08X}: {}x{} (x{:.3g} scale) {} format {} x{} samples", surface.guest_va,
             surface.width, surface.height, surface.scale.Factor(), surface.depth ? "depth" : "color",
             u32(surface.format), u32(surface.samples));
  return true;
}

} // namespace

uint64_t SurfaceEpochLocked() { return g_surface_epoch; }  // sa2-nx (2026-10-20, sa2_renut_chain_reuse)
uint64_t FramebufferEpochLocked() { return g_framebuffer_epoch; }  // sa2-nx (2026-10-20, sa2_renut_chain_reuse)

RenderScale ResolutionScale() {
  return g_resolution_scale;
}

void LatchResolutionScaleLocked() {
  LatchGraphicsOptionsLocked();
  const std::string resolution = REXCVAR_GET(renut_resolution);
  // Codes of RenderScale.
  RenderScale scale{resolution == "480p"    ? 1u
                    : resolution == "1080p" ? 2u
                    : resolution == "1440p" ? 3u
                    : resolution == "4K"    ? 4u
                                            : 0u};
  // banjo-nx (2026-10-18): the deko3d renderer draws at the console's resolution only; its shaders are made for it
  // (tools/deko3d_glsl_opt.py, pass scale1: no per-texture scale divisor, resolution_scale 1).
  if (IsDeko3D() && scale.code != 0) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      RGPU_WARN("renut_resolution = \"{}\" is not taken on deko3d: it draws at the console's 1280x720", resolution);
    }
    scale = {};
  }
  if (!g_resolution_latched || scale != g_resolution_scale) {
    RGPU_INFO("Render resolution {}x{} ({:.3g}x)", scale.Size(GuestScreenWidth()), scale.Size(720),
              scale.Factor());
    g_resolution_scale = scale;
    ++g_surface_epoch;
    g_resolution_latched = true;
  }
}

void SetTilingExtentLocked(uint32_t width, uint32_t height) {
  if (g_tiling_width != width || g_tiling_height != height)
    ++g_surface_epoch;
  g_tiling_width = width;
  g_tiling_height = height;
}

// A surface stops using its target (destroyed, or its header now describes
// another view); the last user retires it.
void UnuseTarget(uint64_t key) {
  auto users = g_target_users.find(key);
  if (users == g_target_users.end() || --users->second)
    return;
  g_target_users.erase(users);
  ++g_surface_epoch;
  if (auto it = g_targets.find(key); it != g_targets.end()) {
    RetireTarget(*it->second);
    g_targets.erase(it);
  }
}

HostSurface *FindHostSurfaceLocked(u32 surface_va) {
  auto key = g_surface_keys.find(surface_va);
  if (key == g_surface_keys.end())
    return nullptr;
  auto it = g_targets.find(key->second);
  return it != g_targets.end() ? it->second.get() : nullptr;
}

void ColorAliasesLocked(const HostSurface *surface, std::vector<HostSurface *> &out) {
  out.clear();
  if (!surface || surface->depth)
    return;
  // TargetKey: the view in the low 32 bits, the host format above.
  constexpr uint64_t kViewMask = 0xFFFFFFFFull;
  uint64_t own = 0;
  bool found = false;
  for (const auto &[key, target] : g_targets) {
    if (target.get() == surface) {
      own = key;
      found = true;
      break;
    }
  }
  if (!found)
    return;
  for (const auto &[key, target] : g_targets) {
    if (target && target.get() != surface && target->texture && !target->depth &&
        (key & kViewMask) == (own & kViewMask))
      out.push_back(target.get());
  }
}

HostSurface *GetHostSurfaceSlow(u32 surface_va, const d3d::SurfaceDesc *desc);

HostSurface *GetHostSurfaceLocked(u32 surface_va) {
  if (!surface_va)
    return nullptr;
  // Targets made before the first Swap are of the configured scale too.
  if (!g_resolution_latched)
    LatchResolutionScaleLocked();
  const auto *desc = GuestPtr<const d3d::SurfaceDesc>(surface_va + sizeof(d3d::Resource));
  if (!desc)
    return nullptr;

  const bool memo = banjo::renut_gpu::CpuMemosOn();
  SurfaceMemo &m = g_surface_memo[(surface_va * 0x9E3779B1u) >> 26];
  if (memo && m.epoch == g_surface_epoch && m.va == surface_va && m.size == desc->Size &&
      m.info == desc->SurfaceInfo && m.format == desc->Format) {
    // the same as the lookup's: the surface bound now
    m.surface->size_word = desc->Size;
    m.surface->info_word = desc->SurfaceInfo;
    m.surface->format_word = desc->Format;
    return m.surface;
  }
  HostSurface *result = GetHostSurfaceSlow(surface_va, desc);
  if (memo && result) {
    m.va = surface_va;
    m.size = desc->Size;
    m.info = desc->SurfaceInfo;
    m.format = desc->Format;
    m.epoch = g_surface_epoch;
    m.surface = result;
  }
  return result;
}

HostSurface *GetHostSurfaceSlow(u32 surface_va, const d3d::SurfaceDesc *desc) {
  const bool depth = d3d::IsDepthFormat(desc->Format);
  plume::RenderFormat format = depth ? DepthTargetFormat(desc->Format) : SurfaceColorFormat(*desc);
  // The render resolution, or the finer one graphics_options gives it.
  const RenderScale scale = SurfaceRenderScale(*desc, g_resolution_scale);
  if (depth && banjo::renut_gpu::Depth24On()) {
    // banjo-nx (2026-10-19, banjo_deko3d_depth24): one view, one format (g_depth_view_formats).
    uint8_t &seen = g_depth_view_formats[TargetKey(*desc, true, plume::RenderFormat::UNKNOWN, scale)];
    const uint8_t bit = (desc->Format & 0x3F) == 23 ? 2 : 1;
    if ((seen | bit) == 3 && seen != 3) {
      static uint32_t logged = 0;
      if (logged++ < 8)
        RGPU_WARN("Depth view of surface {:08X} bound as both D24S8 and D24FS8: 32-bit float from now on "
                  "(banjo_deko3d_depth24)", surface_va);
    }
    seen |= bit;
    if (seen & 2)
      format = DepthTargetFormat();
  }
  const uint64_t key = TargetKey(*desc, depth, format, scale);
  auto [bound, first_use] = g_surface_keys.try_emplace(surface_va, key);
  if (first_use) {
    ++g_target_users[key];
    ++g_surface_epoch;
  } else if (bound->second != key) {
    ++g_surface_epoch;
    // The header was rewritten (or the VA reused without Destroy reaching
    // us): the surface is a view of other EDRAM now.
    UnuseTarget(bound->second);
    bound->second = key;
    ++g_target_users[key];
  }

  auto &slot = g_targets[key];
  // The view's target covers every surface over it, and every tile while
  // tiling (SetTilingExtentLocked).
  uint32_t min_width = std::max(g_tiling_width, d3d::SurfaceWidth(desc->Size));
  uint32_t min_height = std::max(g_tiling_height, d3d::SurfaceHeight(desc->Size));
  if (slot && slot->texture) {
    // The surface bound now: its guest size (Targets::guest_width) and D3D
    // format (D24S8 or D24FS8 for a depth resolve).
    slot->size_word = desc->Size;
    slot->info_word = desc->SurfaceInfo;
    slot->format_word = desc->Format;
    if (slot->width >= min_width && slot->height >= min_height && slot->scale == scale)
      return slot.get();
    // A larger surface over the view, or a larger tiling: grow it, keeping
    // the size the others gave it (what it held is lost). (Another scale is
    // another target: TargetKey.)
    min_width = std::max(min_width, slot->width);
    min_height = std::max(min_height, slot->height);
    RetireTarget(*slot);
  } else if (!slot) {
    slot = std::make_unique<HostSurface>();
    slot->guest_va = surface_va;
  }
  ++g_surface_epoch;
  if (!BuildTarget(*slot, *desc, min_width, min_height, scale))
    return nullptr;
  // No defined content yet; D3D12 wants a new target discarded (or cleared)
  // before anything else uses it, and a resolve may read it before any draw.
  if (plume::RenderCommandList *cmd = Video::CommandListLocked()) {
    TransitionSurfaceLocked(slot.get(), slot->depth ? plume::RenderTextureLayout::DEPTH_WRITE
                                                    : plume::RenderTextureLayout::COLOR_WRITE);
    cmd->discardTexture(slot->texture.get());
  }
  return slot.get();
}

HostSurface *SurfaceByGenerationLocked(uint64_t generation) {
  for (const auto &[key, surface] : g_targets)
    if (surface && surface->texture && surface->generation == generation)
      return surface.get();
  return nullptr;
}

void SurfaceMemoryLocked(size_t &targets, uint64_t &bytes) {
  targets = 0;
  bytes = 0;
  for (const auto &[key, surface] : g_targets) {
    if (!surface || !surface->texture)
      continue;
    ++targets;
    const uint64_t texel = plume::RenderFormatSize(surface->format);
    bytes += uint64_t(surface->scale.Size(surface->width)) * surface->scale.Size(surface->height) *
             texel * surface->samples;
  }
}

void ReleaseHostSurfaceLocked(u32 surface_va) {
  auto it = g_surface_keys.find(surface_va);
  if (it == g_surface_keys.end())
    return;
  const uint64_t key = it->second;
  g_surface_keys.erase(it);
  ++g_surface_epoch;
  UnuseTarget(key);
}

uint32_t SurfaceSampleSlotLocked(HostSurface *surface) {
  if (!surface || !surface->texture)
    return kInvalidDescriptor;
  if (surface->sample_slot != UINT32_MAX)
    return surface->sample_slot;
  // The depth format views the depth plane.
  surface->sample_view = surface->texture->createTextureView(
      plume::RenderTextureViewDesc::Texture2D(surface->format));
  if (!surface->sample_view)
    return kInvalidDescriptor;
  const uint32_t slot = Video::AllocateTextureSlotLocked();
  if (slot == kInvalidDescriptor)
    return kInvalidDescriptor;
  Video::SetTextureSlotLocked(slot, surface->texture.get(), surface->sample_view.get());
  surface->sample_slot = slot;
  return slot;
}

// banjo-nx (2026-10-18): the bindless slot of a depth/stencil target's stencil view (read as uint by the depth
// resolve), or kInvalidDescriptor (no stencil, D3D12, or no view). Tried once a surface.
uint32_t SurfaceStencilSlotLocked(HostSurface *surface) {
  if (!surface || !surface->texture || !UsesVulkanBindings() || !plume::RenderFormatIsStencil(surface->format))
    return kInvalidDescriptor;
  if (surface->stencil_slot != UINT32_MAX)
    return surface->stencil_slot;
  surface->stencil_slot = kInvalidDescriptor;
  surface->stencil_view = surface->texture->createTextureView(
      plume::RenderTextureViewDesc::Texture2D(plume::RenderFormat::S8_UINT));
  if (!surface->stencil_view)
    return kInvalidDescriptor;
  const uint32_t slot = Video::AllocateTextureSlotLocked();
  if (slot == kInvalidDescriptor)
    return kInvalidDescriptor;
  Video::SetTextureSlotLocked(slot, surface->texture.get(), surface->stencil_view.get());
  surface->stencil_slot = slot;
  return slot;
}

void TransitionSurfaceLocked(HostSurface *surface, plume::RenderTextureLayout layout) {
  if (!surface || !surface->texture || surface->layout == layout)
    return;
  // banjo-nx (2026-10-19, banjo_deko3d_barrier_lean): from one read layout to another: no barrier (textures.cpp).
  if (IsDeko3D() && banjo::renut_gpu::Deko3DBarrierLeanOn() && ReadOnlyLayout(surface->layout) &&
      ReadOnlyLayout(layout)) {
    surface->layout = layout;
    return;
  }
  plume::RenderCommandList *cmd = Video::CommandListLocked();
  if (!cmd)
    return;
  cmd->barriers(layout == plume::RenderTextureLayout::COPY_SOURCE ||
                        layout == plume::RenderTextureLayout::COPY_DEST ||
                        layout == plume::RenderTextureLayout::RESOLVE_SOURCE ||
                        layout == plume::RenderTextureLayout::RESOLVE_DEST
                    ? plume::RenderBarrierStage::COPY
                    : plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(surface->texture.get(), layout));
  surface->layout = layout;
}

void SwapDepthImageLocked(HostSurface &surface, DepthImage &image) {
  ++g_surface_epoch;
  std::swap(surface.texture, image.texture);
  std::swap(surface.layout, image.layout);
  std::swap(surface.generation, image.generation);
  std::swap(surface.sample_view, image.sample_view);
  std::swap(surface.sample_slot, image.sample_slot);
  std::swap(surface.stencil_view, image.stencil_view);
  std::swap(surface.stencil_slot, image.stencil_slot);
  std::swap(surface.handover_views, image.views);
  image.host_width = surface.scale.Size(surface.width);
  image.host_height = surface.scale.Size(surface.height);
  image.format = surface.format;
  image.samples = surface.samples;
  image.format_word = surface.format_word;
}

bool MakeDepthImageLocked(const HostSurface &surface, DepthImage &image) {
  plume::RenderDevice *device = Video::Device();
  const uint32_t w = surface.scale.Size(surface.width), h = surface.scale.Size(surface.height);
  image.texture = device->createTexture(
      plume::RenderTextureDesc::DepthTarget(w, h, surface.format, plume::RenderMultisampling(surface.samples)));
  if (!TextureCreated(image.texture.get())) {
    image.texture.reset();
    ReportDeviceLost(device);
    return false;
  }
  image.texture->setName("renut depth surface (hand-over)");
  image.layout = plume::RenderTextureLayout::UNKNOWN;
  image.generation = g_next_generation++;
  image.host_width = w;
  image.host_height = h;
  image.format = surface.format;
  image.samples = surface.samples;
  image.format_word = surface.format_word;
  return true;
}

void RetireDepthImageLocked(DepthImage &image) {
  // The framebuffers a surface made over this image (cached by its generation).
  for (size_t i = 0; i < g_framebuffers.size();) {
    if (g_framebuffers[i].key.depth && g_framebuffers[i].key.depth_generation == image.generation) {
      Video::RetireLocked(std::move(g_framebuffers[i].framebuffer));
      g_framebuffers[i] = std::move(g_framebuffers.back());
      g_framebuffers.pop_back();
      ++g_framebuffer_epoch;
    } else {
      ++i;
    }
  }
  FreeDepthViews(image.views);
  if (image.sample_slot != UINT32_MAX && image.sample_slot != kInvalidDescriptor)
    Video::FreeTextureSlotLocked(image.sample_slot);
  image.sample_slot = UINT32_MAX;
  Video::RetireLocked(std::move(image.sample_view));
  if (image.stencil_slot != UINT32_MAX && image.stencil_slot != kInvalidDescriptor)
    Video::FreeTextureSlotLocked(image.stencil_slot);
  image.stencil_slot = UINT32_MAX;
  Video::RetireLocked(std::move(image.stencil_view));
  Video::RetireLocked(std::move(image.texture));
}

void TransitionDepthImageLocked(DepthImage &image, plume::RenderTextureLayout layout) {
  if (!image.texture || image.layout == layout)
    return;
  if (IsDeko3D() && banjo::renut_gpu::Deko3DBarrierLeanOn() && ReadOnlyLayout(image.layout) && ReadOnlyLayout(layout)) {
    image.layout = layout;
    return;
  }
  plume::RenderCommandList *cmd = Video::CommandListLocked();
  if (!cmd)
    return;
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS, plume::RenderTextureBarrier(image.texture.get(), layout));
  image.layout = layout;
}

uint32_t DepthImageViewSlotLocked(plume::RenderTexture *texture, plume::RenderFormat format,
                                  std::vector<DepthImage::View> &views, const plume::RenderComponentMapping &mapping) {
  if (!texture)
    return kInvalidDescriptor;
  for (const DepthImage::View &v : views) {
    if (v.mapping.r == mapping.r && v.mapping.g == mapping.g && v.mapping.b == mapping.b && v.mapping.a == mapping.a)
      return v.slot;
  }
  // The depth format views the depth plane (deko3d samples it in the red channel), with the destination's mapping.
  plume::RenderTextureViewDesc desc = plume::RenderTextureViewDesc::Texture2D(format);
  desc.componentMapping = mapping;
  DepthImage::View v;
  v.mapping = mapping;
  v.view = texture->createTextureView(desc);
  if (!v.view)
    return kInvalidDescriptor;
  v.slot = Video::AllocateTextureSlotLocked();
  if (v.slot == kInvalidDescriptor) {
    Video::RetireLocked(std::move(v.view));
    return kInvalidDescriptor;
  }
  Video::SetTextureSlotLocked(v.slot, texture, v.view.get());
  views.push_back(std::move(v));
  return views.back().slot;
}

plume::RenderFramebuffer *GetFramebufferLocked(HostSurface *const *colors, uint32_t color_count,
                                               HostSurface *depth) {
  FramebufferKey key;
  for (uint32_t i = 0; i < color_count && i < d3d::kMaxRenderTargets; ++i) {
    key.colors[i] = colors[i];
    key.color_generations[i] = colors[i] ? colors[i]->generation : 0;
  }
  key.depth = depth;
  key.depth_generation = depth ? depth->generation : 0;
  // banjo-nx (2026-10-19, render-thread proposal 8): the recent ones first, the found one moved to the front.
  for (uint32_t i = 0; i < 4; ++i) {
    RecentFramebuffer &r = g_recent_framebuffers[i];
    if (r.epoch == g_framebuffer_epoch && r.framebuffer && r.key == key) {
      plume::RenderFramebuffer *found = r.framebuffer;
      if (i) {
        const RecentFramebuffer hit = r;
        for (uint32_t j = i; j > 0; --j)
          g_recent_framebuffers[j] = g_recent_framebuffers[j - 1];
        g_recent_framebuffers[0] = hit;
      }
      return found;
    }
  }
  auto remember = [&key](plume::RenderFramebuffer *framebuffer) {
    for (uint32_t j = 3; j > 0; --j)
      g_recent_framebuffers[j] = g_recent_framebuffers[j - 1];
    g_recent_framebuffers[0] = {key, framebuffer, g_framebuffer_epoch};
    return framebuffer;
  };
  for (auto &cached : g_framebuffers) {
    if (cached.key == key)
      return remember(cached.framebuffer.get());
  }

  const plume::RenderTexture *attachments[d3d::kMaxRenderTargets] = {};
  uint32_t count = 0;
  for (uint32_t i = 0; i < color_count && i < d3d::kMaxRenderTargets; ++i) {
    if (!colors[i] || !colors[i]->texture)
      break;  // host framebuffers need contiguous color attachments
    attachments[count++] = colors[i]->texture.get();
  }
  plume::RenderFramebufferDesc desc(attachments, count,
                                    depth && depth->texture ? depth->texture.get() : nullptr);
  auto framebuffer = Video::Device()->createFramebuffer(desc);
  if (!framebuffer) {
    RGPU_ERROR("Framebuffer creation failed ({} color targets, depth {})", count, depth != nullptr);
    return nullptr;
  }
  g_framebuffers.push_back({key, std::move(framebuffer)});
  return remember(g_framebuffers.back().framebuffer.get());
}

} // namespace renut::gpu
