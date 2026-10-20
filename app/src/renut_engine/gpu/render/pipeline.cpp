/**
 * @file    gpu/render/pipeline.cpp
 */
#include "renut_engine/gpu/render/pipeline.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <plume_render_interface_builders.h>
#include <rex/cvar.h>

// banjo-nx (2026-10-17, deko3d renderer): every graphics pipeline's shaders and specialization constants, one line each,
// for tools/deko3d_variant_cache.py (the DKSH cache of the deko3d back end, compiled on the PC). Any back end: the
// constants are the renderer's. Written next to the executable.
REXCVAR_DEFINE_BOOL(banjo_deko3d_variant_log, false, "Banjo/Renut",
                    "Write deko3d_variants.txt next to the executable: each graphics pipeline's vertex and pixel shader "
                    "and specialization constants (tools/deko3d_variant_cache.py builds the deko3d DKSH cache from it)");
#include <xxhash.h>
#include <cstdio>  // banjo-nx (2026-10-20): the pipeline list file
#if defined(_WIN32)
#include <io.h>  // _commit
#else
#include <unistd.h>  // fsync
#endif

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/render/format.h"
#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/guest/resources.h"
#include "renut_engine/gpu/render/shader_library.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"

REXCVAR_DEFINE_STRING(renut_pipeline_list, "pipeline_list.bin", "Nuts&Bolts/Graphics",
                      "The pipelines the renderer has built, which it builds ahead of use at the "
                      "next start (delete it to start over)");

namespace renut::gpu {

namespace {

struct CachedPipeline {
  PipelineKey key;
  std::unique_ptr<plume::RenderPipeline> pipeline;
};

// Render thread (State().mutex).
std::unordered_map<uint64_t, std::vector<CachedPipeline>> g_pipelines;
// The last lookup's: most draws use the pipeline the draw before them did,
// which a compare of the key finds without hashing it.
PipelineKey g_last_key{};
plume::RenderPipeline *g_last_pipeline = nullptr;

// banjo-nx (2026-10-19, CPU proposal 6, banjo_renut_pipeline_memo): what GetPipelineLocked returned for a key, in a
// 64-entry direct-mapped table before the XXH3 hash and the map: the built pipeline; for a specialised key still being
// built, the generic one it draws with until the prebuild epoch moves (kind 1); for one that failed, its generic one
// (kind 2; failures stay failed). Pipelines are never destroyed, so an entry cannot dangle.
struct PipelineMemo {
  PipelineKey key{};
  plume::RenderPipeline *pipeline = nullptr;
  uint64_t epoch = 0;
  uint8_t kind = 0;
};
// banjo-nx (2026-10-19, render-thread proposal 6): 512 sets of 2 ways (was 64 entries direct-mapped: the console binds
// 450-1,050 pipelines a frame out of a few hundred keys, and the memo thrashed). Way 0 is the most recent.
constexpr uint32_t kPipelineMemoSets = 512;
PipelineMemo g_pipeline_memo[kPipelineMemoSets][2];

inline uint32_t PipelineMemoIndex(const PipelineKey &key) {
  uint64_t w[8];
  std::memcpy(w, &key, sizeof(w));  // the shaders, the layout hash, topology, formats
  uint64_t h = w[0] * 0x9E3779B97F4A7C15ull ^ w[1] * 0xC2B2AE3D27D4EB4Full ^ w[2] ^ (w[3] << 17) ^ (w[4] >> 3) ^
               (w[5] << 29) ^ w[6] ^ (w[7] >> 11);
  uint32_t b;
  std::memcpy(&b, reinterpret_cast<const char *>(&key) + offsetof(PipelineKey, bool_values), sizeof(b));
  h ^= uint64_t(b) * 0xFF51AFD7ED558CCDull ^ uint64_t(key.alpha_func) << 40 ^ uint64_t(key.blend_control[0]) << 7 ^
       uint64_t(key.depth_control) << 23;
  return uint32_t((h ^ (h >> 31) ^ (h >> 47)) & (kPipelineMemoSets - 1));
}
// banjo-nx (2026-10-18): bumped each time a prebuild job is done (any thread).
std::atomic<uint64_t> g_prebuild_epoch{0};
// banjo-nx (2026-10-18), render thread: specialised keys a worker is building, with the generic pipeline their draws
// take and the g_prebuild_epoch they were last asked at. Until a job is done, a draw of such a key takes the generic
// pipeline here instead of locking the prebuild queue and searching it again (package 33's first run: hundreds of
// draws a frame waited so for their alpha-specialised pipelines).
struct SpecPending {
  PipelineKey key;
  uint64_t epoch = 0;
  plume::RenderPipeline *generic = nullptr;
};
std::unordered_map<uint64_t, SpecPending> g_spec_pending;

// Every draw looks its pipeline up: a byte-at-a-time hash of the whole key
// cost as much as the rest of the lookup.
uint64_t HashKey(const PipelineKey &key) {
  return XXH3_64bits(&key, sizeof(key));
}

plume::RenderBlendDesc BlendDesc(u32 control, u32 write_mask) {
  // RB_BLENDCONTROL: color src 0-4, op 5-7, dst 8-12; alpha src 16-20,
  // op 21-23, dst 24-28. Xenos has no enable bit: ONE/ZERO/ADD is no blending.
  plume::RenderBlendDesc desc;
  const u32 color_src = control & 0x1F, color_op = (control >> 5) & 7,
            color_dst = (control >> 8) & 0x1F;
  const u32 alpha_src = (control >> 16) & 0x1F, alpha_op = (control >> 21) & 7,
            alpha_dst = (control >> 24) & 0x1F;
  desc.srcBlend = ConvertBlendFactor(color_src);
  desc.dstBlend = ConvertBlendFactor(color_dst);
  desc.blendOp = ConvertBlendOp(color_op);
  desc.srcBlendAlpha = ConvertBlendFactor(alpha_src);
  desc.dstBlendAlpha = ConvertBlendFactor(alpha_dst);
  desc.blendOpAlpha = ConvertBlendOp(alpha_op);
  const bool color_passthrough = color_src == 1 && color_dst == 0 && color_op == 0;
  const bool alpha_passthrough = alpha_src == 1 && alpha_dst == 0 && alpha_op == 0;
  desc.blendEnabled = !(color_passthrough && alpha_passthrough);
  // Alpha factors cannot reference color; D3D12 rejects them.
  auto alpha_only = [](plume::RenderBlend b) {
    switch (b) {
    case plume::RenderBlend::SRC_COLOR:
      return plume::RenderBlend::SRC_ALPHA;
    case plume::RenderBlend::INV_SRC_COLOR:
      return plume::RenderBlend::INV_SRC_ALPHA;
    case plume::RenderBlend::DEST_COLOR:
      return plume::RenderBlend::DEST_ALPHA;
    case plume::RenderBlend::INV_DEST_COLOR:
      return plume::RenderBlend::INV_DEST_ALPHA;
    default:
      return b;
    }
  };
  desc.srcBlendAlpha = alpha_only(desc.srcBlendAlpha);
  desc.dstBlendAlpha = alpha_only(desc.dstBlendAlpha);
  desc.renderTargetWriteMask = uint8_t(write_mask & 0xF);
  return desc;
}

} // namespace

// banjo-nx (banjo_gpu_blend_skip). Vulkan clamps a UNORM target's source and factors to [0, 1] before the blend, so a
// source alpha of 0 or less is 0 there, and a whole source of 0 or less is 0. The target keeps its value where the
// source term is 0 and the destination factor is 1 at that source, on every channel written, with ADD.
uint32_t BlendSkipMode(const PipelineKey &key) {
  if (!key.pixel_shader || key.color_count != 1 || key.alpha_to_coverage)
    return 0;
  // Depth or stencil writes would differ (the discard writes neither).
  if (key.depth_format != plume::RenderFormat::UNKNOWN && (key.depth_control & (1u | 4u)))
    return 0;
  switch (key.color_formats[0]) {
  case plume::RenderFormat::R8G8B8A8_UNORM:
  case plume::RenderFormat::B8G8R8A8_UNORM:
  case plume::RenderFormat::R16G16B16A16_UNORM:
  case plume::RenderFormat::R8G8_UNORM:
  case plume::RenderFormat::R8_UNORM:
  case plume::RenderFormat::R16G16_UNORM:
  case plume::RenderFormat::R16_UNORM:
    break;
  default:
    return 0;  // float targets do not clamp; sRGB ones convert
  }
  const plume::RenderBlendDesc b = BlendDesc(key.blend_control[0], key.color_mask);
  if (!b.blendEnabled || (b.renderTargetWriteMask & 0xF) == 0)
    return 0;
  using B = plume::RenderBlend;
  using Op = plume::RenderBlendOperation;
  const bool rgb = (b.renderTargetWriteMask & 7) != 0, alpha = (b.renderTargetWriteMask & 8) != 0;
  // Alpha: the source alpha is 0 in both modes, so its term is 0 whatever its factor; the destination's must be 1.
  if (alpha && !(b.blendOpAlpha == Op::ADD && (b.dstBlendAlpha == B::ONE || b.dstBlendAlpha == B::INV_SRC_ALPHA)))
    return 0;
  if (!rgb)
    return 1;
  if (b.blendOp != Op::ADD)
    return 0;
  // Mode 1: only the alpha is 0 or less: the colour's source factor must be 0 at alpha 0.
  const bool src_zero_at_alpha_zero =
      b.srcBlend == B::ZERO || b.srcBlend == B::SRC_ALPHA || b.srcBlend == B::SRC_ALPHA_SAT;
  const bool dst_one_at_alpha_zero = b.dstBlend == B::ONE || b.dstBlend == B::INV_SRC_ALPHA;
  if (src_zero_at_alpha_zero && dst_one_at_alpha_zero)
    return 1;
  // Mode 2: the whole source is 0: its term is 0 for any factor; the destination's must be 1 at a zero source.
  if (dst_one_at_alpha_zero || b.dstBlend == B::INV_SRC_COLOR)
    return 2;
  return 0;
}

namespace {

// The channels a render target format has (RGBA bits, as the colour write mask). Vulkan drops a component the format
// lacks. Formats not listed: all four (nothing trimmed).
uint32_t FormatChannels(plume::RenderFormat format) {
  using F = plume::RenderFormat;
  switch (format) {
  case F::R8_UNORM:
  case F::R16_UNORM:
  case F::R16_FLOAT:
  case F::R32_FLOAT:
    return 0x1;
  case F::R8G8_UNORM:
  case F::R16G16_UNORM:
  case F::R16G16_FLOAT:
  case F::R32G32_FLOAT:
    return 0x3;
  default:
    return 0xF;
  }
}

bool BlendReadsSourceAlpha(plume::RenderBlend b) {
  using B = plume::RenderBlend;
  return b == B::SRC_ALPHA || b == B::INV_SRC_ALPHA || b == B::SRC_ALPHA_SAT;
}

bool BlendReadsSecondSource(plume::RenderBlend b) {
  using B = plume::RenderBlend;
  return b == B::SRC1_COLOR || b == B::INV_SRC1_COLOR || b == B::SRC1_ALPHA || b == B::INV_SRC1_ALPHA;
}

} // namespace

// banjo-nx (banjo_gpu_output_trim). Vulkan writes a channel only where the write mask has it and the format has it;
// blending reads the source per channel (its own value, SRC_COLOR) and the source alpha (the alpha factors), never
// another colour channel; the alpha factors of the alpha channel read only alphas.
uint32_t OutputTrimMask(const PipelineKey &key) {
  if (!key.pixel_shader)
    return 0;
  uint32_t trim = 0;
  for (uint32_t i = 0; i < d3d::kMaxRenderTargets; ++i) {
    if (i >= key.color_count) {
      trim |= 0xFu << (4 * i);  // no target: an output nothing receives
      continue;
    }
    const uint32_t write = (key.color_mask >> (4 * i)) & 0xF & FormatChannels(key.color_formats[i]);
    uint32_t t = ~write & 0xF;
    const plume::RenderBlendDesc b = BlendDesc(key.blend_control[i], key.color_mask >> (4 * i));
    if (b.blendEnabled) {
      if (BlendReadsSecondSource(b.srcBlend) || BlendReadsSecondSource(b.dstBlend) ||
          BlendReadsSecondSource(b.srcBlendAlpha) || BlendReadsSecondSource(b.dstBlendAlpha))
        t = 0;
      // A written colour channel whose factors read the source alpha.
      if ((write & 7) && (BlendReadsSourceAlpha(b.srcBlend) || BlendReadsSourceAlpha(b.dstBlend)))
        t &= 7;
    }
    if (i == 0 && key.alpha_to_coverage)
      t &= 7;
    trim |= t << (4 * i);
  }
  return trim;
}

namespace {

// Any thread: reads only what device creation set up.
std::unique_ptr<plume::RenderPipeline> Build(const PipelineKey &key, const PipelineInputs &inputs) {
  // banjo-nx (a test, banjo_renut_diag_build_delay_ms): console-slow compiles on the PC.
  if (const int delay = banjo::renut_gpu::DiagBuildDelayMs())
    std::this_thread::sleep_for(std::chrono::milliseconds(delay));
  banjo::renut_gpu::NotePipelineBuilt();  // banjo-nx (banjo_renut_cache_save_quiet)
  auto &s = State();
  const HostVertexLayout &layout = *inputs.layout;

  plume::RenderGraphicsPipelineDesc desc;
  desc.pipelineLayout = s.draw_layout.get();
  desc.vertexShader = inputs.vertex_shader;
  desc.pixelShader = inputs.pixel_shader;
  if (key.rectangle_list)
    desc.geometryShader = s.rectangle_list_gs.get();
  if (key.point_sprite)
    desc.geometryShader = s.point_sprite_gs.get();
  desc.primitiveTopology = key.topology;

  // PA_SU_SC_MODE_CNTL: cull front 0, cull back 1, front face clockwise 2.
  // Points have no facing; their sprites are never culled.
  const bool cull_front = !key.point_sprite && (key.mode_control & 1);
  const bool cull_back = !key.point_sprite && (key.mode_control & 2);
  desc.cullMode = cull_front && !cull_back   ? plume::RenderCullMode::FRONT
                  : cull_back && !cull_front ? plume::RenderCullMode::BACK
                                             : plume::RenderCullMode::NONE;
  desc.frontFace = (key.mode_control & 4) ? plume::RenderFrontFace::CLOCKWISE
                                          : plume::RenderFrontFace::COUNTER_CLOCKWISE;
  desc.depthClipEnabled = true;

  // RB_DEPTHCONTROL.
  const u32 dc = key.depth_control;
  if (key.depth_format != plume::RenderFormat::UNKNOWN) {
    desc.depthEnabled = dc & 2;
    desc.depthWriteEnabled = dc & 4;
    // banjo-nx (2026-10-18): a test that always passes and writes nothing has no effect; off, the GPU does not read the
    // depth target for it (full-screen passes over a bound depth target).
    if (desc.depthEnabled && !desc.depthWriteEnabled && ((dc >> 4) & 7) == 7)
      desc.depthEnabled = false;
    desc.depthFunction = ConvertCompareFunction((dc >> 4) & 7);
    desc.stencilEnabled = dc & 1;
    desc.stencilReference = key.stencil_ref_mask & 0xFF;
    desc.stencilReadMask = (key.stencil_ref_mask >> 8) & 0xFF;
    desc.stencilWriteMask = (key.stencil_ref_mask >> 16) & 0xFF;
    desc.stencilFrontFace.compareFunction = ConvertCompareFunction((dc >> 8) & 7);
    desc.stencilFrontFace.failOp = ConvertStencilOp((dc >> 11) & 7);
    desc.stencilFrontFace.passOp = ConvertStencilOp((dc >> 14) & 7);
    desc.stencilFrontFace.depthFailOp = ConvertStencilOp((dc >> 17) & 7);
    if (dc & 0x80) {
      desc.stencilBackFace.compareFunction = ConvertCompareFunction((dc >> 20) & 7);
      desc.stencilBackFace.failOp = ConvertStencilOp((dc >> 23) & 7);
      desc.stencilBackFace.passOp = ConvertStencilOp((dc >> 26) & 7);
      desc.stencilBackFace.depthFailOp = ConvertStencilOp((dc >> 29) & 7);
    } else {
      desc.stencilBackFace = desc.stencilFrontFace;
    }
  }
  desc.depthTargetFormat = key.depth_format;

  desc.renderTargetCount = key.color_count;
  for (uint32_t i = 0; i < key.color_count; ++i) {
    desc.renderTargetFormat[i] = key.color_formats[i];
    desc.renderTargetBlend[i] = BlendDesc(key.blend_control[i], key.color_mask >> (i * 4));
  }
  desc.multisampling = plume::RenderMultisampling(plume::RenderSampleCounts(key.samples));
  // banjo-nx (banjo_renut_alpha_to_mask_test, 2026-10-18): not on a single-sample target, where the draw's shared
  // constants make it an alpha test at 0.5 (draw.cpp). The key keeps the bit: the same keys and shader variants.
  desc.alphaToCoverageEnabled = key.alpha_to_coverage != 0 &&
                                !(key.samples == uint32_t(plume::RenderSampleCount::COUNT_1) &&
                                  banjo::renut_gpu::AlphaToMaskTestOn());

  // banjo-nx (banjo_gpu_bool_spec): the key's bools as the shaders' specialization constants (banjo_bool_spec.h). One
  // list for every stage: a stage takes the ids it declares (vertex and pixel shader bools have distinct ones).
  plume::RenderSpecConstant spec[272];  // 256 bools and 900-908
  uint32_t spec_count = 0;
  for (uint32_t w = 0; w < 8; ++w) {
    for (uint32_t m = key.bool_mask[w]; m; m &= m - 1) {
      const uint32_t b = uint32_t(std::countr_zero(m));
      spec[spec_count++] = plume::RenderSpecConstant(1000 + w * 32 + b, (key.bool_values[w] >> b) & 1u);
    }
  }
  // banjo-nx (banjo_gpu_blend_skip): the pixel shader's discard of pixels the blend would not change.
  if (key.blend_skip)
    spec[spec_count++] = plume::RenderSpecConstant(900, key.blend_skip);
  // banjo-nx (banjo_gpu_output_trim): the colour channels the pixel shader leaves 0.
  if (key.output_trim) {
    spec[spec_count++] = plume::RenderSpecConstant(901, key.output_trim);
    if (banjo::renut_gpu::DiagTrimPoison())
      spec[spec_count++] = plume::RenderSpecConstant(902, 1);  // the test: trimmed channels 60000
  }
  // banjo-nx (banjo_deko3d_alpha_spec, 2026-10-18): the alpha test's function (+1), constant 903 of the deko3d GLSL.
  if (key.alpha_func)
    spec[spec_count++] = plume::RenderSpecConstant(903, key.alpha_func);
  // banjo-nx (banjo_deko3d_tess_spec, 2026-10-18): not a tessellated draw: the vertex shader's patch kind is 0
  // (SharedConstants.tessellation[0], constant 904 + 1). Decided by the key: no new pipelines.
  if (!key.per_instance_streams && banjo::renut_gpu::TessSpecOn())
    spec[spec_count++] = plume::RenderSpecConstant(904, 1);
  // banjo-nx (banjo_deko3d_early_hoist, 2026-10-18): the main body's first texture region ahead of the early alpha test
  // (constant 907, 1 by default in the GLSL); off: after it, as Renut's shaders have it.
  if (!banjo::renut_gpu::EarlyHoistOn())
    spec[spec_count++] = plume::RenderSpecConstant(907, 0);
  // banjo-nx (2026-10-19, GPU): texture sizes from the size buffer (constant 908, 1 by default in the GLSL).
  if (const uint32_t txq = banjo::renut_gpu::TxqMode(); txq != 1u)
    spec[spec_count++] = plume::RenderSpecConstant(908, txq);
  // banjo-nx (2026-10-19, GPU): depth tested before a discarding pixel shader where the draw writes neither depth nor
  // stencil (RB_DEPTHCONTROL bit 2 depth writes; as BlendSkipMode): constant 909 (tools/deko3d_glsl_opt.py earlyz;
  // shaders without a discard do not have it and get no define). 2026-10-20: a stencil test that writes nothing (write
  // mask 0, or every op of both faces KEEP: bits 11-19 front, 23-31 back with bit 7) counts as no stencil writes; the
  // early tests only skip the shader for pixels the tests would have dropped after it (no occlusion queries here).
  if (key.pixel_shader && banjo::renut_gpu::EarlyZOn() && !(dc & 4u)) {
    const bool stencil_ops = ((dc >> 11) & 0x1FF) != 0 || ((dc & 0x80) && ((dc >> 23) & 0x1FF) != 0);
    const bool stencil_writes = (dc & 1u) && ((key.stencil_ref_mask >> 16) & 0xFF) != 0 && stencil_ops;
    if (!stencil_writes)
      spec[spec_count++] = plume::RenderSpecConstant(909, 1);
  }
  // banjo-nx (2026-10-19, GPU, A/B): the listed pixel shaders compiled with 32 GPRs at most (constant 910).
  // (key.reserved bit 1: the off half of the in-run A/B, banjo_renut_ab_target 16.)
  if (key.pixel_shader && banjo::renut_gpu::Gpr32On() && !(key.reserved & 2u))
    spec[spec_count++] = plume::RenderSpecConstant(910, 1);
  // banjo-nx (2026-10-20, GPU A/B): the listed lit shaders' lights in branches (constant 911, pass lightbranch).
  // (key.reserved bit 0: the off half of the in-run A/B, banjo_renut_ab_target 15.)
  if (key.pixel_shader && banjo::renut_gpu::LightBranchOn() && !(key.reserved & 1u))
    spec[spec_count++] = plume::RenderSpecConstant(911, 1);
  // banjo-nx (2026-10-20, diagnostic): the fur family without its pixel work (constant 912, pass nullps).
  if (key.pixel_shader && banjo::renut_gpu::NullPsOn())
    spec[spec_count++] = plume::RenderSpecConstant(912, 1);
  desc.specConstants = spec_count ? spec : nullptr;
  desc.specConstantsCount = spec_count;
  if (REXCVAR_GET(banjo_deko3d_variant_log)) {
    static std::mutex log_mutex;
    static std::unordered_set<std::string> logged;
    static std::ofstream log_file(banjo::renut_gpu::NextToExecutable("deko3d_variants.txt"), std::ios::app);
    const auto *vs_entry = static_cast<const ShaderCacheEntry *>(key.vertex_shader);
    const auto *ps_entry = static_cast<const ShaderCacheEntry *>(key.pixel_shader);
    std::string line = fmt::format("vs {:016X} ps {:016X}", vs_entry ? vs_entry->hash : 0, ps_entry ? ps_entry->hash : 0);
    for (uint32_t i = 0; i < spec_count; ++i)
      line += fmt::format(" {}={}", spec[i].index, spec[i].value);
    // The state the blend skip and the output trim are decided from (tools/deko3d_variant_cache.py skips these).
    line += fmt::format(" | targets {} format0 {} blend0 {:08X} mask {:04X} depth {:08X} a2c {}", key.color_count,
                        int(key.color_formats[0]), key.blend_control[0], key.color_mask, key.depth_control,
                        key.alpha_to_coverage);
    std::lock_guard<std::mutex> lock(log_mutex);
    if (log_file && logged.insert(line).second) {
      log_file << line << '\n';
      log_file.flush();
    }
  }

  plume::RenderInputSlot slots[kMaxVertexStreams + 1];
  uint32_t slot_count = 0;
  for (uint32_t stream = 0; stream < kMaxVertexStreams; ++stream) {
    const HostVertexStream &hs = layout.streams[stream];
    if (hs.used) {
      slots[slot_count++] = plume::RenderInputSlot(
          stream, hs.host_stride,
          key.per_instance_streams ? plume::RenderInputSlotClassification::PER_INSTANCE_DATA
                                   : plume::RenderInputSlotClassification::PER_VERTEX_DATA);
    }
  }
  slots[slot_count++] = plume::RenderInputSlot(kZeroStreamSlot, 0);
  desc.inputSlots = slots;
  desc.inputSlotsCount = slot_count;
  desc.inputElements = layout.input_elements.data();
  desc.inputElementsCount = uint32_t(layout.input_elements.size());

  auto pipeline = s.device->createGraphicsPipeline(desc);
  if (GraphicsPipelineCreated(pipeline.get())) {
    // Names the guest shaders in captures.
    const auto *vs = static_cast<const ShaderCacheEntry *>(key.vertex_shader);
    const auto *ps = static_cast<const ShaderCacheEntry *>(key.pixel_shader);
    pipeline->setName(fmt::format("vs_{:016x} ps_{:016x}", vs ? vs->hash : 0, ps ? ps->hash : 0));
    return pipeline;
  }
  return nullptr;
}

// ---------------------------------------------------------------------------
// The pipeline list file
// ---------------------------------------------------------------------------
//
// A header, then records: u32 byte size of the rest of the record, the key
// with the two shader pointers replaced by the shaders' cache hashes, the
// layout's streams (u8 used, u32 host stride, 16 of them), u32 input element
// count, and each element (u8 name length, the name, u32 semantic index,
// location, format, slot, offset). A build whose PipelineKey or layout differs
// bumps kListVersion; a file of another version is started over.

constexpr char kListMagic[4] = {'R', 'N', 'P', 'L'};
// banjo-nx: 2, the key's bool words (banjo_gpu_bool_spec); 3, output_trim; 4, alpha_func (a version 3 list is kept,
// LoadList).
constexpr u32 kListVersion = 4;
constexpr u32 kListV3KeySize = 184;  // the key up to output_trim

struct ListHeader {
  char magic[4];
  u32 version;
  u32 key_size;
  u32 streams;
};

class ByteWriter {
public:
  template <typename T>
  void Put(const T &value) {
    const auto *p = reinterpret_cast<const char *>(&value);
    bytes_.insert(bytes_.end(), p, p + sizeof(T));
  }
  void PutBytes(const void *data, size_t size) {
    const auto *p = static_cast<const char *>(data);
    bytes_.insert(bytes_.end(), p, p + size);
  }
  const std::vector<char> &Bytes() const { return bytes_; }

private:
  std::vector<char> bytes_;
};

class ByteReader {
public:
  ByteReader(const char *data, size_t size) : data_(data), size_(size) {}
  template <typename T>
  bool Get(T &value) {
    if (size_ - at_ < sizeof(T))
      return false;
    std::memcpy(&value, data_ + at_, sizeof(T));
    at_ += sizeof(T);
    return true;
  }
  bool GetBytes(void *out, size_t size) {
    if (size_ - at_ < size)
      return false;
    std::memcpy(out, data_ + at_, size);
    at_ += size;
    return true;
  }

private:
  const char *data_;
  size_t size_;
  size_t at_ = 0;
};

// The key as stored: the shader pointers as their cache hashes.
PipelineKey StoredKey(const PipelineKey &key) {
  PipelineKey stored = key;
  const auto *vs = static_cast<const ShaderCacheEntry *>(key.vertex_shader);
  const auto *ps = static_cast<const ShaderCacheEntry *>(key.pixel_shader);
  const uint64_t vs_hash = vs ? vs->hash : 0, ps_hash = ps ? ps->hash : 0;
  static_assert(sizeof(stored.vertex_shader) == sizeof(uint64_t));
  std::memcpy(&stored.vertex_shader, &vs_hash, sizeof(vs_hash));
  std::memcpy(&stored.pixel_shader, &ps_hash, sizeof(ps_hash));
  return stored;
}

std::vector<char> EncodeRecord(const PipelineKey &key, const HostVertexLayout &layout) {
  ByteWriter w;
  w.Put(StoredKey(key));
  for (u32 s = 0; s < kMaxVertexStreams; ++s) {
    w.Put(u8(layout.streams[s].used ? 1 : 0));
    w.Put(u32(layout.streams[s].host_stride));
  }
  w.Put(u32(layout.input_elements.size()));
  for (const plume::RenderInputElement &e : layout.input_elements) {
    const std::string name = e.semanticName ? e.semanticName : "";
    w.Put(u8(name.size()));
    w.PutBytes(name.data(), name.size());
    w.Put(u32(e.semanticIndex));
    w.Put(u32(e.location));
    w.Put(u32(e.format));
    w.Put(u32(e.slotIndex));
    w.Put(u32(e.alignedByteOffset));
  }
  return w.Bytes();
}

// A listed pipeline, as the worker threads build it.
struct Job {
  PipelineKey key{};
  uint64_t hash = 0;
  HostVertexLayout layout;
  std::deque<std::string> names;  // the elements' semantic names point here
  enum class Stage { Pending, Building, Done } stage = Stage::Pending;
  std::unique_ptr<plume::RenderPipeline> pipeline;  // Done: nullptr if taken or failed
  // Its shaders the game has not made yet (PipelineShaderLoaded); queued at 0.
  u32 shaders_missing = 0;
};

bool DecodeRecord(const char *data, size_t size, Job &job) {
  ByteReader r(data, size);
  PipelineKey stored;
  if (!r.Get(stored))
    return false;
  uint64_t vs_hash, ps_hash;
  std::memcpy(&vs_hash, &stored.vertex_shader, sizeof(vs_hash));
  std::memcpy(&ps_hash, &stored.pixel_shader, sizeof(ps_hash));
  const ShaderCacheEntry *vs = FindShaderEntry(vs_hash);
  const ShaderCacheEntry *ps = ps_hash ? FindShaderEntry(ps_hash) : nullptr;
  if (!vs || (ps_hash && !ps))
    return false;  // a shader this build lacks
  job.key = stored;
  job.key.vertex_shader = vs;
  job.key.pixel_shader = ps;
  // banjo-nx (2026-10-20): the key as this configuration makes it. A record written with other keys on (Renut's blend
  // skip before package 30, an A/B half) would be built ahead for no draw, and its shader variants compiled at run time
  // (the DKSH cache leaves them out): emulator runs compiled three blend-skip variants at every start.
  if (!banjo::renut_gpu::BlendSkipOn())
    job.key.blend_skip = 0;
  if (!banjo::renut_gpu::OutputTrimOn())
    job.key.output_trim = 0;
  if (!banjo::renut_gpu::AlphaSpecOn())
    job.key.alpha_func = 0;
  if (!banjo::renut_gpu::BoolSpecOn()) {
    std::memset(job.key.bool_mask, 0, sizeof(job.key.bool_mask));
    std::memset(job.key.bool_values, 0, sizeof(job.key.bool_values));
  }
  job.key.reserved = 0;
  job.hash = HashKey(job.key);
  for (u32 s = 0; s < kMaxVertexStreams; ++s) {
    u8 used;
    u32 stride;
    if (!r.Get(used) || !r.Get(stride))
      return false;
    job.layout.streams[s].used = used != 0;
    job.layout.streams[s].host_stride = stride;
  }
  u32 count;
  if (!r.Get(count) || count > 64)
    return false;
  for (u32 i = 0; i < count; ++i) {
    u8 length;
    std::string name;
    u32 index, location, format, slot, offset;
    if (!r.Get(length))
      return false;
    name.resize(length);
    if (!r.GetBytes(name.data(), length) || !r.Get(index) || !r.Get(location) || !r.Get(format) ||
        !r.Get(slot) || !r.Get(offset)) {
      return false;
    }
    job.names.push_back(std::move(name));
    job.layout.input_elements.emplace_back(job.names.back().c_str(), index, location,
                                           plume::RenderFormat(format), slot, offset);
  }
  return true;
}

// The list's records, and the file the new ones are appended to. Render
// thread, except the load at the start.
// banjo-nx (2026-10-20, Peggle 2's report on the DKSH cache): a flushed stream only reaches the file system service,
// which may keep it buffered; a game closed with HOME (or stopped in the emulator) lost the records of the run, so the
// same pipelines (and their shader variants) were built again at every start. The file is synced to the card at most
// once a second (kListSyncNs: the records of the last second may still be lost), not per record (a new area appends
// hundreds).
struct PipelineList {
  std::FILE *out = nullptr;
  bool failed = false;
  std::unordered_set<uint64_t> listed;  // key hashes in the file
  std::chrono::steady_clock::time_point last_sync{};
  bool unsynced = false;
  static constexpr auto kListSync = std::chrono::seconds(1);

  void Sync() {
    if (!out || !unsynced)
      return;
    std::fflush(out);
#if defined(_WIN32)
    _commit(_fileno(out));
#else
    fsync(fileno(out));
#endif
    unsynced = false;
    last_sync = std::chrono::steady_clock::now();
  }

  void Append(const PipelineKey &key, uint64_t hash, const HostVertexLayout &layout) {
    if (failed || !listed.insert(hash).second)
      return;
    if (!out) {
      const std::filesystem::path path = banjo::renut_gpu::NextToExecutable(REXCVAR_GET(renut_pipeline_list));
      std::error_code ec;
      const bool fresh = !std::filesystem::exists(path, ec) || std::filesystem::file_size(path, ec) == 0;
      out = std::fopen(path.string().c_str(), "ab");
      if (!out) {
        failed = true;
        RGPU_WARN("Pipeline list {} cannot be written; pipelines are not built ahead next time",
                  path.string());
        return;
      }
      if (fresh) {
        ListHeader header{};
        std::memcpy(header.magic, kListMagic, 4);
        header.version = kListVersion;
        header.key_size = sizeof(PipelineKey);
        header.streams = kMaxVertexStreams;
        std::fwrite(&header, sizeof(header), 1, out);
      }
    }
    const std::vector<char> record = EncodeRecord(key, layout);
    const u32 size = u32(record.size());
    std::fwrite(&size, sizeof(size), 1, out);
    std::fwrite(record.data(), 1, record.size(), out);
    std::fflush(out);
    unsynced = true;
    if (std::chrono::steady_clock::now() - last_sync >= kListSync)
      Sync();
  }
};
PipelineList g_list;

}  // namespace

// banjo-nx (2026-10-20): the list's last records synced to the card a second after they were written, if no record
// came after them (the render thread, once a frame).
void SyncPipelineListLocked() {
  if (g_list.unsynced && std::chrono::steady_clock::now() - g_list.last_sync >= PipelineList::kListSync)
    g_list.Sync();
}

namespace {

// The records of the list file; empty (and the file started over) when it
// is missing or from another version.
std::vector<std::unique_ptr<Job>> LoadList() {
  std::vector<std::unique_ptr<Job>> jobs;
  const std::filesystem::path path = banjo::renut_gpu::NextToExecutable(REXCVAR_GET(renut_pipeline_list));
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return jobs;
  const std::vector<char> bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  in.close();
  ListHeader header{};
  if (bytes.size() >= sizeof(header))
    std::memcpy(&header, bytes.data(), sizeof(header));
  const bool header_ok = bytes.size() >= sizeof(header) && std::memcmp(header.magic, kListMagic, 4) == 0 &&
                         header.streams == kMaxVertexStreams;
  const bool valid = header_ok && header.version == kListVersion && header.key_size == sizeof(PipelineKey);
  // banjo-nx (2026-10-18): a version 3 list (before alpha_func): its keys get the fields after output_trim as 0 (the
  // generic pipelines, which the specialised ones draw with until they are built), and the file is written again as
  // version 4.
  const bool v3 = header_ok && header.version == 3 && header.key_size == kListV3KeySize;
  if (!valid && !v3) {
    std::error_code ec;
    std::filesystem::remove(path, ec);
    return jobs;
  }
  std::unordered_set<uint64_t> seen;
  std::vector<char> converted;
  if (v3) {
    ListHeader h4 = header;
    h4.version = kListVersion;
    h4.key_size = sizeof(PipelineKey);
    converted.resize(sizeof(h4));
    std::memcpy(converted.data(), &h4, sizeof(h4));
  }
  size_t at = sizeof(header);
  std::vector<char> record;
  while (bytes.size() - at >= sizeof(u32)) {
    u32 size;
    std::memcpy(&size, bytes.data() + at, sizeof(size));
    at += sizeof(size);
    if (bytes.size() - at < size)
      break;  // cut short (the game closed while writing)
    const char *data = bytes.data() + at;
    u32 data_size = size;
    if (v3) {
      if (size < kListV3KeySize)
        break;
      record.assign(data, data + kListV3KeySize);
      record.resize(sizeof(PipelineKey), 0);
      record.insert(record.end(), data + kListV3KeySize, data + size);
      data = record.data();
      data_size = u32(record.size());
      const size_t o = converted.size();
      converted.resize(o + sizeof(u32) + data_size);
      std::memcpy(converted.data() + o, &data_size, sizeof(u32));
      std::memcpy(converted.data() + o + sizeof(u32), data, data_size);
    }
    auto job = std::make_unique<Job>();
    if (DecodeRecord(data, data_size, *job) && seen.insert(job->hash).second)
      jobs.push_back(std::move(job));
    at += size;
  }
  if (v3) {
    std::ofstream rewrite(path, std::ios::binary | std::ios::trunc);
    if (rewrite)
      rewrite.write(converted.data(), std::streamsize(converted.size()));
    RGPU_INFO("Pipeline list {}: version 3, {} pipelines kept as version 4 (generic alpha test)", path.string(),
              jobs.size());
  }
  // Records of shaders this build lacks stay out of 'listed': a later build
  // that has them lists the pipelines again.
  g_list.listed = std::move(seen);
  return jobs;
}

// ---------------------------------------------------------------------------
// Building the listed pipelines ahead
// ---------------------------------------------------------------------------
//
// A listed pipeline is built once the game has made both of its shaders (a
// level's shaders come with its load), not at the start: the list holds every
// level ever played, and the driver keeps each built pipeline's shaders and
// code in host memory. What is built stays built.

struct Prebuild {
  std::mutex mutex;
  std::condition_variable work;  // a job queued, or stopping
  std::condition_variable done;  // a job reached Done, or a worker ended
  std::vector<std::unique_ptr<Job>> jobs;
  std::unordered_map<uint64_t, std::vector<Job *>> by_hash;
  std::deque<Job *> queue;       // shaders all made; may have been claimed since
  // The shaders the game has made, and the jobs waiting for each of the others.
  std::unordered_set<const ShaderCacheEntry *> loaded;
  std::unordered_map<const ShaderCacheEntry *, std::vector<Job *>> waiting;
  u32 workers = 0;               // still running
  u32 building = 0;              // workers between taking a job and finishing it
  bool stop = false;
  bool started = false;          // banjo-nx: the workers were started (StartWorkersLocked)
  // Host shaders for the queued jobs, shared by the workers; freed whenever the
  // queue runs dry (the pipelines keep their own compiled copies).
  std::unordered_map<const ShaderCacheEntry *, std::unique_ptr<plume::RenderShader>> shaders;
};
Prebuild g_prebuild;

plume::RenderShader *PrebuildShader(const ShaderCacheEntry *entry) {
  if (!entry)
    return nullptr;
  auto &slot = g_prebuild.shaders[entry];
  if (!slot)
    slot = CreateCachedShader(State().device.get(), *entry);
  return slot.get();
}

void PrebuildWorker() {
#if defined(_WIN32)
  SetThreadDescription(GetCurrentThread(), L"renut pipeline prebuild");
  SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#endif
  banjo::renut_gpu::OnPrebuildWorkerStart();  // banjo-nx: its core and priority on the Switch
  std::unique_lock lock(g_prebuild.mutex);
  for (;;) {
    g_prebuild.work.wait(lock, [] { return g_prebuild.stop || !g_prebuild.queue.empty(); });
    if (g_prebuild.stop)
      break;
    Job &job = *g_prebuild.queue.front();
    g_prebuild.queue.pop_front();
    if (job.stage != Job::Stage::Pending)
      continue;  // the render thread claimed it (TakePrebuilt)
    job.stage = Job::Stage::Building;
    ++g_prebuild.building;
    PipelineInputs inputs;
    inputs.vertex_shader = PrebuildShader(static_cast<const ShaderCacheEntry *>(job.key.vertex_shader));
    inputs.pixel_shader = PrebuildShader(static_cast<const ShaderCacheEntry *>(job.key.pixel_shader));
    inputs.layout = &job.layout;
    const bool shaders = inputs.vertex_shader && (!job.key.pixel_shader || inputs.pixel_shader);
    lock.unlock();
    std::unique_ptr<plume::RenderPipeline> pipeline = shaders ? Build(job.key, inputs) : nullptr;
    lock.lock();
    job.pipeline = std::move(pipeline);
    job.stage = Job::Stage::Done;
    g_prebuild_epoch.fetch_add(1, std::memory_order_release);  // banjo-nx: g_spec_pending asks again
    if (--g_prebuild.building == 0 && g_prebuild.queue.empty())
      g_prebuild.shaders.clear();
    g_prebuild.done.notify_all();
  }
  if (--g_prebuild.workers == 0)
    g_prebuild.shaders.clear();
  g_prebuild.done.notify_all();
}

// Requires g_prebuild.mutex. Queues the job once the game has made its shaders,
// or waits for the ones it has not.
void QueueOrWait(Job &job) {
  for (const void *shader : {job.key.vertex_shader, job.key.pixel_shader}) {
    const auto *entry = static_cast<const ShaderCacheEntry *>(shader);
    if (entry && !g_prebuild.loaded.contains(entry)) {
      ++job.shaders_missing;
      g_prebuild.waiting[entry].push_back(&job);
    }
  }
  if (job.shaders_missing == 0)
    g_prebuild.queue.push_back(&job);
}

// banjo-nx: the prebuild workers, started once (by the list at the start, or by the first specialised pipeline).
// Requires g_prebuild.mutex.
void StartWorkersLocked() {
  if (g_prebuild.workers != 0 || g_prebuild.stop || g_prebuild.started)
    return;
  g_prebuild.started = true;
  // Leave the render and game threads their cores.
  const u32 cores = std::max(1u, std::thread::hardware_concurrency());
  const u32 workers = banjo::renut_gpu::PrebuildWorkers(std::clamp(cores / 2, 1u, 4u));  // banjo-nx
  g_prebuild.workers = workers;
  // banjo-nx: not std::thread::detach, which throws on the Switch (package 24's crash); a thread that is never
  // joined, with a stack of its own size there (banjo_renut_tuning.h). StopPipelinePrebuild waits for the workers.
  for (u32 i = 0; i < workers; ++i)
    banjo::renut_gpu::StartLongLivedThread(PrebuildWorker, 512 * 1024);
}

// banjo-nx (banjo_gpu_bool_spec): a specialised pipeline, without waiting. Ready: built (moved to 'out'); Pending:
// queued or building (a job made now if there was none); Failed: its build failed. Render thread.
enum class SpecState { Ready, Pending, Failed };
SpecState SpecialisedPipeline(const PipelineKey &key, uint64_t hash, const HostVertexLayout &layout,
                              std::unique_ptr<plume::RenderPipeline> &out) {
  std::lock_guard lock(g_prebuild.mutex);
  if (g_prebuild.stop)
    return SpecState::Failed;
  auto &jobs = g_prebuild.by_hash[hash];
  for (Job *j : jobs) {
    if (std::memcmp(&j->key, &key, sizeof(key)) != 0)
      continue;
    if (j->stage != Job::Stage::Done)
      return SpecState::Pending;
    if (!j->pipeline)
      return SpecState::Failed;
    out = std::move(j->pipeline);
    return SpecState::Ready;
  }
  // A job of its own, self-contained (the record form copies the layout's names), as the list's are.
  auto job = std::make_unique<Job>();
  const std::vector<char> record = EncodeRecord(key, layout);
  if (!DecodeRecord(record.data(), record.size(), *job))
    return SpecState::Failed;
  Job *j = job.get();
  g_prebuild.jobs.push_back(std::move(job));
  jobs.push_back(j);
  QueueOrWait(*j);
  StartWorkersLocked();
  g_prebuild.work.notify_one();
  return SpecState::Pending;
}

bool Specialised(const PipelineKey &key) {
  if (key.alpha_func)
    return true;
  for (const uint32_t w : key.bool_mask) {
    if (w)
      return true;
  }
  return false;
}

// Render thread: specialised keys whose build failed (they draw with the generic pipeline).
std::unordered_set<uint64_t> g_spec_failed;

// The listed pipeline for the key, when the list has it: built by a worker
// (waited for when one is building it), or nullptr with 'claimed' set when no
// worker has started it, which the caller then builds itself. Render thread.
plume::RenderPipeline *TakePrebuilt(const PipelineKey &key, uint64_t hash,
                                    std::unique_ptr<plume::RenderPipeline> &out, bool &claimed) {
  claimed = false;
  std::unique_lock lock(g_prebuild.mutex);
  auto it = g_prebuild.by_hash.find(hash);
  if (it == g_prebuild.by_hash.end())
    return nullptr;
  Job *job = nullptr;
  for (Job *j : it->second) {
    if (std::memcmp(&j->key, &key, sizeof(key)) == 0)
      job = j;
  }
  if (!job)
    return nullptr;
  if (job->stage == Job::Stage::Pending) {
    job->stage = Job::Stage::Done;  // the render thread's now
    claimed = true;
    return nullptr;
  }
  g_prebuild.done.wait(lock, [job] { return job->stage == Job::Stage::Done; });
  out = std::move(job->pipeline);
  return out.get();
}

} // namespace

bool BuildDrawLayout(VideoState &s) {
  plume::RenderDescriptorSetBuilder textures;
  textures.begin();
  textures.addTexture(0, kTextureDescriptorCount);
  textures.end(true, kTextureDescriptorCount);

  plume::RenderDescriptorSetBuilder samplers;
  samplers.begin();
  samplers.addSampler(0, kSamplerDescriptorCount);
  samplers.end(true, kSamplerDescriptorCount);

  plume::RenderPipelineLayoutBuilder layout;
  layout.begin(false, true);
  layout.addDescriptorSet(textures);  // space0 Texture2D[]
  layout.addDescriptorSet(textures);  // space1 Texture3D[]
  layout.addDescriptorSet(textures);  // space2 TextureCube[]
  layout.addDescriptorSet(samplers);  // space3 SamplerState[]
  layout.addDescriptorSet(textures);  // space4 Texture2DArray[] (stacked textures)
  plume::RenderDescriptorSetBuilder constants;
  constants.begin();
  constants.addConstantBufferDynamic(0);
  constants.end();
  if (UsesVulkanBindings()) {  // banjo-nx: and deko3d
    // banjo-nx: the guest shaders' constants (sets 5-7, kVertexConstantsSet...).
    layout.addDescriptorSet(constants);
    layout.addDescriptorSet(constants);
    layout.addDescriptorSet(constants);
    // Geometry too: point_sprite_gs reads SharedConstants.
    layout.addPushConstant(0, 4, sizeof(DrawPushConstants),
                           plume::RenderShaderStageFlag::VERTEX | plume::RenderShaderStageFlag::GEOMETRY |
                               plume::RenderShaderStageFlag::PIXEL);
  } else {
    layout.addRootDescriptor(0, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
    layout.addRootDescriptor(1, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
    layout.addRootDescriptor(2, 4, plume::RenderRootDescriptorType::CONSTANT_BUFFER);
    layout.addRootDescriptor(0, 5, plume::RenderRootDescriptorType::SHADER_RESOURCE);
    layout.addRootDescriptor(0, 6, plume::RenderRootDescriptorType::UNORDERED_ACCESS);
  }
  layout.end();
  s.draw_layout = layout.create(s.device.get());
  if (!s.draw_layout) {
    RGPU_ERROR("Draw pipeline layout creation failed");
    return false;
  }
  return true;
}

static plume::RenderPipeline *GetPipelineSlow(const PipelineKey &key, const PipelineInputs &inputs, uint8_t &kind,
                                              uint64_t &memo_epoch);

// sa2-nx (2026-10-20, sa2_renut_chain_reuse): GetPipelineLocked's calls counted, and the prebuild epoch: a draw whose key
// is the previous draw's takes that draw's answer when no lookup was made since and no prebuilt pipeline came in
// (DrawStateLocked). The answer depends on nothing else: the last key and the memo change only in the lookups, and a
// memo entry of a generic pipeline is held only for its prebuild epoch.
uint64_t g_pipeline_lookups = 0;
uint64_t PipelineLookupsLocked() { return g_pipeline_lookups; }
uint64_t PrebuildEpochLocked() { return g_prebuild_epoch.load(std::memory_order_acquire); }

plume::RenderPipeline *GetPipelineLocked(const PipelineKey &key, const PipelineInputs &inputs) {
  ++g_pipeline_lookups;  // sa2-nx
  if (g_last_pipeline && std::memcmp(&g_last_key, &key, sizeof(key)) == 0)
    return g_last_pipeline;
  const bool memo_on = banjo::renut_gpu::PipelineMemoOn();
  const uint32_t set = PipelineMemoIndex(key);
  PipelineMemo *ways = g_pipeline_memo[set];
  if (memo_on) {
    for (uint32_t w = 0; w < 2; ++w) {
      PipelineMemo &m = ways[w];
      if (m.pipeline && std::memcmp(&m.key, &key, sizeof(key)) == 0 &&
          (m.kind != 1 || m.epoch == g_prebuild_epoch.load(std::memory_order_acquire))) {
        if (m.kind)
          banjo::renut_gpu::CountBoolSpec(banjo::renut_gpu::BoolSpecEvent::kGeneric);
        plume::RenderPipeline *hit = m.pipeline;
        if (w)
          std::swap(ways[0], ways[1]);  // the most recent in way 0
        return hit;
      }
    }
  }
  uint8_t kind = 0;
  uint64_t memo_epoch = 0;
  plume::RenderPipeline *result = GetPipelineSlow(key, inputs, kind, memo_epoch);
  if (memo_on && result) {
    // The least recent (way 1) goes; the new one takes way 0. (GetPipelineSlow may have filled this set for a generic
    // key: both are kept.)
    ways = g_pipeline_memo[set];
    ways[1] = ways[0];
    ways[0].key = key;
    ways[0].pipeline = result;
    ways[0].kind = kind;
    ways[0].epoch = memo_epoch;
  }
  return result;
}

static plume::RenderPipeline *GetPipelineSlow(const PipelineKey &key, const PipelineInputs &inputs, uint8_t &kind,
                                              uint64_t &memo_epoch) {
  const uint64_t hash = HashKey(key);
  auto &bucket = g_pipelines[hash];
  for (auto &cached : bucket) {
    if (std::memcmp(&cached.key, &key, sizeof(key)) == 0) {
      if (cached.pipeline) {
        g_last_key = key;
        g_last_pipeline = cached.pipeline.get();
      }
      return cached.pipeline.get();
    }
  }
  // banjo-nx (banjo_gpu_bool_spec): a specialised pipeline not built yet is built by a prebuild worker while the draws
  // take the generic one (every bool read at run time: the same result), from the first draw after it is done.
  if (Specialised(key)) {
    const uint64_t epoch = g_prebuild_epoch.load(std::memory_order_acquire);
    if (auto pend = g_spec_pending.find(hash); pend != g_spec_pending.end() && pend->second.epoch == epoch &&
                                               pend->second.generic &&
                                               std::memcmp(&pend->second.key, &key, sizeof(key)) == 0) {
      banjo::renut_gpu::CountBoolSpec(banjo::renut_gpu::BoolSpecEvent::kGeneric);
      kind = 1;
      memo_epoch = epoch;
      return pend->second.generic;
    }
    std::unique_ptr<plume::RenderPipeline> ready;
    const SpecState state = g_spec_failed.contains(hash) ? SpecState::Failed
                                                         : SpecialisedPipeline(key, hash, *inputs.layout, ready);
    if (state != SpecState::Pending)
      g_spec_pending.erase(hash);
    if (state == SpecState::Ready && ready) {
      g_list.Append(key, hash, *inputs.layout);
      bucket.push_back({key, std::move(ready)});
      g_last_key = key;
      g_last_pipeline = bucket.back().pipeline.get();
      banjo::renut_gpu::CountBoolSpec(banjo::renut_gpu::BoolSpecEvent::kBuilt);
      return g_last_pipeline;
    }
    if (state == SpecState::Failed)
      g_spec_failed.insert(hash);
    banjo::renut_gpu::CountBoolSpec(banjo::renut_gpu::BoolSpecEvent::kGeneric);
    PipelineKey generic = key;
    std::memset(generic.bool_mask, 0, sizeof(generic.bool_mask));
    std::memset(generic.bool_values, 0, sizeof(generic.bool_values));
    generic.alpha_func = 0;  // banjo-nx (banjo_deko3d_alpha_spec): the function read at run time
    plume::RenderPipeline *pipeline = GetPipelineLocked(generic, inputs);
    if (state == SpecState::Pending) {
      SpecPending &p = g_spec_pending[hash];
      p.key = key;
      p.epoch = epoch;
      p.generic = pipeline;
      kind = 1;
      memo_epoch = epoch;
    } else {
      kind = 2;  // failed: the generic one from now on
    }
    return pipeline;
  }
  std::unique_ptr<plume::RenderPipeline> pipeline;
  bool claimed = false;
  if (!TakePrebuilt(key, hash, pipeline, claimed)) {
    pipeline = Build(key, inputs);
    if (pipeline && !claimed)
      g_list.Append(key, hash, *inputs.layout);
  }
  if (!pipeline) {
    static uint32_t logged = 0;
    if (!ReportDeviceLost(State().device.get()) && logged++ < 32) {
      const auto *vs = static_cast<const ShaderCacheEntry *>(key.vertex_shader);
      const auto *ps = static_cast<const ShaderCacheEntry *>(key.pixel_shader);
      RGPU_ERROR("Pipeline creation failed (key {:016X}): VS {:016X} (spec constants {:X}), "
                 "PS {:016X} (spec constants {:X}), {} targets [{} {} {} {}], depth format {}, "
                 "{} samples, {} input elements, topology {}",
                 hash, vs ? vs->hash : 0, vs ? vs->specConstantsMask : 0, ps ? ps->hash : 0,
                 ps ? ps->specConstantsMask : 0, key.color_count, u32(key.color_formats[0]),
                 u32(key.color_formats[1]), u32(key.color_formats[2]), u32(key.color_formats[3]),
                 u32(key.depth_format), key.samples, inputs.layout->input_elements.size(),
                 u32(key.topology));
    }
  }
  bucket.push_back({key, std::move(pipeline)});
  plume::RenderPipeline *result = bucket.back().pipeline.get();
  if (result) {
    g_last_key = key;
    g_last_pipeline = result;
  }
  return result;
}

void StartPipelinePrebuild() {
  std::vector<std::unique_ptr<Job>> jobs = LoadList();
  if (jobs.empty())
    return;
  std::lock_guard lock(g_prebuild.mutex);
  g_prebuild.jobs = std::move(jobs);
  for (const auto &job : g_prebuild.jobs) {
    g_prebuild.by_hash[job->hash].push_back(job.get());
    QueueOrWait(*job);
  }
  StartWorkersLocked();  // banjo-nx: shared with the specialised pipelines'
}

void StopPipelinePrebuild() {
  // The workers finish the pipeline each is building (they use the device).
  std::unique_lock lock(g_prebuild.mutex);
  g_prebuild.stop = true;
  g_prebuild.work.notify_all();
  g_prebuild.done.wait(lock, [] { return g_prebuild.workers == 0; });
}

void PipelineShaderLoaded(const ShaderCacheEntry *entry) {
  if (!entry)
    return;
  std::lock_guard lock(g_prebuild.mutex);
  if (!g_prebuild.loaded.insert(entry).second)
    return;
  auto it = g_prebuild.waiting.find(entry);
  if (it == g_prebuild.waiting.end())
    return;
  bool queued = false;
  for (Job *job : it->second) {
    // A job the render thread claimed meanwhile is done with.
    if (--job->shaders_missing == 0 && job->stage == Job::Stage::Pending) {
      g_prebuild.queue.push_back(job);
      queued = true;
    }
  }
  g_prebuild.waiting.erase(it);
  if (queued)
    g_prebuild.work.notify_all();
}

PipelineCounts PipelineCountsLocked() {
  PipelineCounts counts;
  for (const auto &[hash, bucket] : g_pipelines) {
    for (const CachedPipeline &cached : bucket)
      counts.built += cached.pipeline ? 1 : 0;
  }
  std::lock_guard lock(g_prebuild.mutex);
  counts.listed = g_prebuild.jobs.size();
  for (const auto &job : g_prebuild.jobs) {
    counts.prebuilt_unused += job->pipeline ? 1 : 0;
    counts.waiting += job->stage == Job::Stage::Pending && job->shaders_missing ? 1 : 0;
  }
  return counts;
}

} // namespace renut::gpu
