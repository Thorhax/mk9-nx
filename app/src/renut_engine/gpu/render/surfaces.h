/**
 * @file    gpu/render/surfaces.h
 * @brief   Host render targets behind guest D3DSurfaces.
 *
 * N&B's own D3DDevice_CreateSurface still builds the surface (and does its
 * EDRAM bookkeeping); the renderer attaches a host color or depth target to
 * the EDRAM view the surface describes (base, pitch, samples, format) the
 * first time a surface over it is bound, sized and formatted from the surface
 * header. Surfaces over the same view share the target, as they share EDRAM
 * on the GPU; content lives in it while any of them does, which is what the
 * game relies on between Clear, draws and Resolve.
 * All functions require State().mutex.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <plume_render_interface.h>

#include <rex/types.h>

#include "renut_engine/gpu/render/render_scale.h"

namespace renut::gpu {

// banjo-nx (2026-10-19, banjo_deko3d_depth_handover): a depth target's image and what is made over it (its generation,
// layout, sample and stencil views), which moves between a surface and a depth resolve destination that holds the
// surface's content (render/draw.cpp, "Depth hand-over"). 'views' are a destination's views of it, by component
// mapping (the fetch swizzle as the destination's own R32 views take it).
struct DepthImage {
  std::unique_ptr<plume::RenderTexture> texture;
  plume::RenderTextureLayout layout = plume::RenderTextureLayout::UNKNOWN;
  uint64_t generation = 0;
  std::unique_ptr<plume::RenderTextureView> sample_view;
  uint32_t sample_slot = UINT32_MAX;
  std::unique_ptr<plume::RenderTextureView> stencil_view;
  uint32_t stencil_slot = UINT32_MAX;
  struct View {
    plume::RenderComponentMapping mapping;
    std::unique_ptr<plume::RenderTextureView> view;
    uint32_t slot = UINT32_MAX;
  };
  std::vector<View> views;
  // The target it was made for or last taken from: a surface takes it back only when these are its own.
  uint32_t host_width = 0, host_height = 0;
  plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
  plume::RenderSampleCounts samples = plume::RenderSampleCount::COUNT_1;
  u32 format_word = 0;
};

struct HostSurface {
  u32 guest_va = 0;  // the first surface over the view
  // Header words of the surface over the view bound most recently.
  u32 size_word = 0;
  u32 info_word = 0;
  u32 format_word = 0;

  // In guest pixels; the host target is 'scale' times that on each axis
  // (ResolutionScale when it was built).
  uint32_t width = 0;
  uint32_t height = 0;
  RenderScale scale;
  bool depth = false;
  plume::RenderFormat format = plume::RenderFormat::UNKNOWN;
  plume::RenderSampleCounts samples = plume::RenderSampleCount::COUNT_1;

  std::unique_ptr<plume::RenderTexture> texture;
  plume::RenderTextureLayout layout = plume::RenderTextureLayout::UNKNOWN;
  uint64_t generation = 0;  // bumped on rebuild, part of framebuffer keys
  // banjo-nx (2026-10-19, banjo_renut_resolve_skip): bumped by every draw that can write it (a colour write mask; for
  // depth, a depth write or stencil writes) and every clear of it; part of a resolve's signature.
  uint64_t writes = 0;

  // The view and bindless slot shaders read the target through (depth
  // resolves); made on first use, see SurfaceSampleSlotLocked.
  std::unique_ptr<plume::RenderTextureView> sample_view;
  uint32_t sample_slot = UINT32_MAX;
  // banjo-nx (2026-10-18): a depth/stencil target's stencil view and its slot (SurfaceStencilSlotLocked).
  std::unique_ptr<plume::RenderTextureView> stencil_view;
  uint32_t stencil_slot = UINT32_MAX;
  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover): the current image's views for destinations that sample it
  // (DepthImage::views; they go with the image), how many destinations hold its content, and whether anything has
  // used its stencil (a draw with the stencil test, a stencil clear): a clear of the depth alone hands the image over
  // only when nothing has. A stencil nothing writes is undefined either way (a depth resolve's words read it as is).
  std::vector<DepthImage::View> handover_views;
  uint32_t handed = 0;
  bool stencil_used = false;

  // gpu_profile labels of draws with it as the first target: color only,
  // color with depth, depth only. Made on first use.
  uint32_t profile_labels[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
};

// Render resolution: every host render target, and every texture a resolve
// writes, is this many times its guest size on each axis (1 = the game's own
// 720p, 3 = 2160p, 2/3 = 480p). The renderer works in guest pixels and scales
// where it records host viewports, scissors, clears and copies; shaders see a
// scaled texture's guest size (the scale is in its descriptor index, see
// kResolutionScaleShift) and the guest pixel position.
// From the renut_resolution cvar, latched once per frame by
// LatchResolutionScaleLocked (at Swap) so a frame never mixes scales. Targets
// and resolve destinations of another scale are rebuilt on their next use.
RenderScale ResolutionScale();
void LatchResolutionScaleLocked();

// The host target for the D3DSurface at 'surface_va', created or rebuilt as
// needed. nullptr for 0 or when the target cannot be created.
HostSurface *GetHostSurfaceLocked(u32 surface_va);
// sa2-nx (2026-10-20, sa2_renut_chain_reuse): banjo-nx's surface epoch (bumped by every target built, retired,
// unused or rebound, the resolution scale latched and the tiling extent set).
uint64_t SurfaceEpochLocked();
// sa2-nx (2026-10-20): banjo-nx's framebuffer epoch (bumped when a framebuffer is retired).
uint64_t FramebufferEpochLocked();

// Only if one exists already.
HostSurface *FindHostSurfaceLocked(u32 surface_va);

// The other color targets over the same EDRAM view as 'surface' (base tile,
// pitch, samples and scale), in another host format: on the console one
// memory, so what a clear writes through one is what the others then hold.
void ColorAliasesLocked(const HostSurface *surface, std::vector<HostSurface *> &out);

// Predicated tiling (D3DDevice_BeginTiling .. EndTiling): the game draws the
// frame into surfaces the size of one EDRAM tile, and the GPU replays the
// draws once per tile with the tile's screen offset, resolving each tile into
// its part of the destination. The host has no EDRAM to fit, so a surface
// used while tiling is backed by a target covering every tile (and keeps that
// size), which the draws and the per-tile resolves address in screen space.
// 'width' x 'height' is the bottom-right corner of the tile rects; 0 x 0 ends
// tiling.
void SetTilingExtentLocked(uint32_t width, uint32_t height);

// The surface no longer uses its target (guest D3DResource_Destroy); the
// target is retired when no surface over its view remains.
void ReleaseHostSurfaceLocked(u32 surface_va);

// Bindless slot of the surface's target (the depth plane for depth targets,
// multisampled when the target is), or kInvalidDescriptor.
uint32_t SurfaceSampleSlotLocked(HostSurface *surface);
// banjo-nx (2026-10-18): its stencil view's slot (uint), or kInvalidDescriptor.
uint32_t SurfaceStencilSlotLocked(HostSurface *surface);

// Moves 'surface' into 'layout' with a barrier on the open command list.
void TransitionSurfaceLocked(HostSurface *surface, plume::RenderTextureLayout layout);

// For the memory report: host render targets and the video memory they take
// (estimated from format, size and samples).
void SurfaceMemoryLocked(size_t &targets, uint64_t &bytes);

// banjo-nx (2026-10-19): the live surface with this generation, or nullptr (rebuilt or gone since).
HostSurface *SurfaceByGenerationLocked(uint64_t generation);

// banjo-nx (2026-10-19, banjo_deko3d_depth_handover): exchanges the surface's image (its texture, generation, layout
// and views) with 'image'. Framebuffers are cached by surface and generation, so each image keeps its own.
void SwapDepthImageLocked(HostSurface &surface, DepthImage &image);
// A new image of the surface's size, format and samples for SwapDepthImageLocked; false when it cannot be made.
bool MakeDepthImageLocked(const HostSurface &surface, DepthImage &image);
// Retires the image, its views and the framebuffers made over it (deferred past in-flight frames).
void RetireDepthImageLocked(DepthImage &image);
// Moves the image into 'layout' with a barrier on the open command list.
void TransitionDepthImageLocked(DepthImage &image, plume::RenderTextureLayout layout);
// The slot of a view of the image with 'mapping' (made on first use, kept in 'views'), or kInvalidDescriptor.
uint32_t DepthImageViewSlotLocked(plume::RenderTexture *texture, plume::RenderFormat format,
                                  std::vector<DepthImage::View> &views, const plume::RenderComponentMapping &mapping);
// render/draw.cpp: a depth target is about to go; the destinations that still hold its content get it first.
void DepthHandOverBeforeRetireLocked(HostSurface *surface);

// Framebuffer over up to 4 color targets plus depth; cached per target set.
plume::RenderFramebuffer *GetFramebufferLocked(HostSurface *const *colors, uint32_t color_count,
                                               HostSurface *depth);

} // namespace renut::gpu
