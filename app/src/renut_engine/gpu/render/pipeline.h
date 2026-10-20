/**
 * @file    gpu/render/pipeline.h
 * @brief   The pipeline layout of the recompiled guest shaders and the
 *          pipeline cache.
 *
 * Binding contract (shader_common.h, RENUT_RECOMP):
 *   set 0/1/2  bindless Texture2D / Texture3D / TextureCube (one shared table)
 *   set 3      bindless samplers
 *   set 4      the texture table again, as Texture2DArray (stacked textures)
 *   D3D12      b0/b1/b2 space4 root CBVs: VS constants, PS constants,
 *              SharedConstants; t0 space5 root SRV: this draw's vertex fetch
 *              data (ByteAddressBuffer)
 *   Vulkan     push constants: the three constant buffers' device addresses;
 *              vertex fetch data is read through its address in
 *              SharedConstants
 */
#pragma once

#include <cstdint>

#include <plume_render_interface.h>

#include "renut_engine/gpu/d3d/d3d.h"
#include "renut_engine/gpu/backend/video.h"

struct ShaderCacheEntry;

namespace renut::gpu {

constexpr uint32_t kDrawRootVertexConstants = 0;
constexpr uint32_t kDrawRootPixelConstants = 1;
constexpr uint32_t kDrawRootSharedConstants = 2;
constexpr uint32_t kDrawRootVertexFetchData = 3;
constexpr uint32_t kDrawRootMemExport = 4;  // u0 space6, the memexport buffer

struct DrawPushConstants {
  uint64_t vertex_constants;
  uint64_t pixel_constants;
  uint64_t shared_constants;
};

// banjo-nx: sets 5-7 of the draw layout on Vulkan, one dynamic uniform buffer each (render/upload_ring.h): the
// vertex shader's float constants, the pixel shader's, SharedConstants.
constexpr uint32_t kVertexConstantsSet = 5;
constexpr uint32_t kPixelConstantsSet = 6;
constexpr uint32_t kSharedConstantsSet = 7;

bool BuildDrawLayout(VideoState &s);

// Everything a pipeline is built from. Compared and hashed as raw bytes, so
// it is always zero-initialised before filling.
struct PipelineKey {
  const void *vertex_shader;  // ShaderCacheEntry (static, unlike the host shader objects)
  const void *pixel_shader;   // ShaderCacheEntry, nullptr without a pixel shader
  uint64_t layout_hash;       // the declaration's host input layout
  plume::RenderPrimitiveTopology topology;
  plume::RenderFormat color_formats[d3d::kMaxRenderTargets];
  plume::RenderFormat depth_format;
  uint32_t color_count;
  uint32_t samples;
  uint32_t blend_control[d3d::kMaxRenderTargets];
  uint32_t color_mask;
  uint32_t depth_control;
  uint32_t stencil_ref_mask;
  uint32_t stencil_ref_mask_bf;
  uint32_t mode_control;
  uint32_t alpha_to_coverage;
  uint32_t per_instance_streams;  // tessellated draws: vertex streams are per instance
  uint32_t rectangle_list;        // triangle list expanded by rectangle_list_gs
  uint32_t point_sprite;          // point list expanded by point_sprite_gs
  // banjo-nx (banjo_gpu_bool_spec, package 27): the bool constants the two shaders test, as their specialization
  // constants 1000 + N (N the unified index: vertex shader bools 0-127, pixel shader bools 128-255; banjo_bool_spec.h in
  // the translator), and the draw's values of them. All zero: the generic pipeline, whose shaders read every bool at run
  // time (as before). A specialised key not built yet draws with the generic one while a prebuild worker builds it.
  uint32_t bool_mask[8];
  uint32_t bool_values[8];
  // banjo-nx (banjo_gpu_blend_skip, package 27): the pixel shader's specialization constant 900 (BlendSkipMode).
  uint32_t blend_skip;
  // banjo-nx (banjo_gpu_output_trim, package 28): the pixel shader's specialization constant 901 (OutputTrimMask).
  uint32_t output_trim;
  // banjo-nx (banjo_deko3d_alpha_spec, 2026-10-18): the alpha test's compare function + 1 (SharedConstants.
  // alpha_test_func; 8 = always, the test off), the pixel shader's specialization constant 903 (tools/deko3d_glsl_opt.py,
  // pass alpha); 0 = read at run time. Like the bools: a key with it not built yet draws with the generic one (0).
  uint32_t alpha_func;
  uint32_t reserved;  // 0 (the key is compared and hashed as bytes)
};

// banjo-nx (banjo_gpu_blend_skip): for the key's other fields, 1 when its one target keeps its value for a source alpha
// of 0 or less, 2 when it does for a whole source of 0 or less, 0 otherwise (shader_common.h, banjoBlendSkip).
uint32_t BlendSkipMode(const PipelineKey &key);

// banjo-nx (banjo_gpu_output_trim): for the key's other fields, the colour channels (4 bits a target, RGBA) the
// pixel shader may leave 0: unwritten (write mask, a target's format, no target), and the alpha only when no written
// colour channel blends with it and alpha to coverage is off (shader_common.h, banjoTrim).
uint32_t OutputTrimMask(const PipelineKey &key);

struct HostVertexLayout;

// The live objects a key was made from, used only when the pipeline is built.
struct PipelineInputs {
  plume::RenderShader *vertex_shader = nullptr;
  plume::RenderShader *pixel_shader = nullptr;
  const HostVertexLayout *layout = nullptr;  // the declaration's (key.layout_hash)
};

// Built pipeline for the key, compiling it on first use unless the pipeline
// list (below) built it ahead. nullptr on failure (logged once per key).
plume::RenderPipeline *GetPipelineLocked(const PipelineKey &key, const PipelineInputs &inputs);
// sa2-nx (2026-10-20, sa2_renut_chain_reuse): GetPipelineLocked's calls so far, and the prebuild epoch.
uint64_t PipelineLookupsLocked();
uint64_t PrebuildEpochLocked();

// banjo-nx (2026-10-20): the pipeline list's last records synced to the card once they are a second old (the render
// thread, once a frame: a game closed with HOME keeps them).
void SyncPipelineListLocked();

// The pipeline list: every pipeline a draw needed is written to a file
// (renut_pipeline_list), and at the next start worker threads build the ones
// listed there before the game draws with them, so a scene the game has
// shown before costs no compile on the render thread. A listed pipeline is
// built once the game has made its shaders (PipelineShaderLoaded), so only
// what the levels played this session use is built. Start once the draw
// layout and host shaders exist; Stop at shutdown.
void StartPipelinePrebuild();
void StopPipelinePrebuild();

// The game made a shader object of this cache entry (any thread, any time).
void PipelineShaderLoaded(const ShaderCacheEntry *entry);

// For the memory report: pipelines the draws use, pipelines built ahead from
// the list that no draw has asked for yet (all of them stay built), and listed
// pipelines whose shaders the game has not made.
struct PipelineCounts {
  size_t built = 0;
  size_t prebuilt_unused = 0;
  size_t waiting = 0;
  size_t listed = 0;
};
PipelineCounts PipelineCountsLocked();

} // namespace renut::gpu
