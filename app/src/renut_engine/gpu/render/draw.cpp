/**
 * @file    gpu/render/draw.cpp
 */
#include "renut_engine/gpu/render/draw.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <new>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <cstdlib>
#include <vector>

#include <plume_render_interface.h>

#include <fmt/format.h>
#include <xxhash.h>

#include <rex/system/xmemory.h>

#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/backend/gpu_profile.h"
#include "renut_engine/gpu/guest/command_buffers.h"
#include "renut_engine/gpu/d3d/d3d.h"
#include "renut_engine/gpu/render/format.h"
#include "renut_engine/gpu/guest/guest_memory.h"
#include "renut_engine/gpu/sa2_line_compare.h"
#include "renut_engine/gpu/guest/guest_records.h"
#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/render/pipeline.h"
#include "renut_engine/gpu/guest/resources.h"
#include "renut_engine/gpu/render/shader_library.h"  // banjo-nx: ShaderBoolSpecMask
#include "renut_engine/gpu/render/surfaces.h"
#include "renut_engine/gpu/render/textures.h"
#include "renut_engine/gpu/render/vertex_format.h"
#include "renut_engine/gpu/backend/video_internal.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"
#if defined(PLUME_DEKO3D)
#include <plume_deko3d.h>  // banjo-nx (2026-10-17): Deko3DBindMemExport
#endif
#include "renut_engine/gpu/banjo_renut_record.h"  // banjo-nx (package 28b)
#include <rex/cvar.h>

// banjo-nx (2026-10-17, a diagnostic for the deko3d renderer's missing river water): draws that read memexport results
// as vertices take them through the CPU path (wait for the export, read it back) instead of stream_convert_cs.
REXCVAR_DEFINE_BOOL(banjo_renut_diag_no_stream_convert, false, "Banjo/Renut",
                    "Diagnostic: exported streams reach their draws through the CPU (no stream_convert_cs)");
// banjo-nx (2026-10-17, a diagnostic for the depth of field that banjo_gpu_blend_skip breaks): pixel shaders (their
// 16-digit hashes, comma separated) left out of the blend skip.
REXCVAR_DEFINE_STRING(banjo_renut_diag_stream_dump_vs, "", "Banjo/Renut",
                      "Diagnostic: log the vertex streams of draws with this vertex shader (hex hash): guest and "
                      "converted words of the first vertices ([streamdump], first 16 binds)");
REXCVAR_DEFINE_INT32(banjo_renut_diag_draw_log_s, 0, "Banjo/Renut",
                     "Diagnostic: this many seconds after the first draw, log every draw of the next two frames "
                     "([drawlog]: shaders, counts, target, state; 0 = off)");
REXCVAR_DEFINE_STRING(banjo_renut_diag_blend_skip_exclude, "", "Banjo/Renut",
                      "Diagnostic: pixel shader hashes (hex, comma separated) banjo_gpu_blend_skip leaves alone");

// sa2-nx (2026-10-20): see DrawStateLocked.
REXCVAR_DEFINE_BOOL(sa2_renut_chain_reuse, true, "SA2/Renderer",
                    "Renut's renderer (with sa2_renut_snapshot_deltas): a draw of the delta chain takes the previous "
                    "draw's pipeline key when nothing the key is made from changed (the device's render state chunks, "
                    "the shaders, the declaration, the targets, the primitive) instead of making it again")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_chain_reuse_verify, false, "SA2/Renderer",
                    "Tests (sa2_renut_chain_reuse): the key is made anyway and compared with the one taken; a "
                    "difference is logged and the made one used")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_inline_line_compare, true, "SA2/Renderer",
                    "Renut's renderer on deko3d: the pushed shared constants' 64-byte lines are compared with what the "
                    "GPU holds by inline loads instead of a memcmp call per line (21 a draw). Off: memcmp")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_stream_mask, true, "SA2/Renderer",
                    "Renut's renderer: a draw's vertex streams are walked by the layout's mask of used streams instead "
                    "of a check of all 16")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_texture_defaults_copy, true, "SA2/Renderer",
                    "Renut's renderer: a draw's default texture and sampler words are one copy of a block made once")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_fetch_slots_skip, true, "SA2/Renderer",
                    "Renut's renderer on deko3d: a draw without vertex fetches skips the compare of the shared "
                    "constants' fetch slots when the last push had none either (both zero)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_fetch_data_unread, true, "SA2/Renderer",
                    "Renut's renderer on deko3d: a draw whose shaders fetch nothing by address gets fetch data address "
                    "and size 0 (they never read them) instead of a new allocation's, and no fetch binding")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_shared_partial_clear, true, "SA2/Renderer",
                    "Renut's renderer: a draw's shared constants are zeroed from the booleans on (848 of 1,360 bytes); "
                    "the texture and sampler words before them are all written by FillTextures. Off: all zeroed")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renut_lean_diagnostics, true, "SA2/Renderer",
                    "Renut's renderer: the per-draw counts only the 10 s log lines read (the draw kinds, the fetch "
                    "alignment census) are kept only while those lines are logged (log_level info or lower), and the "
                    "draw and stream dumps' checks are made once instead of a call per draw")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DECLARE(bool, sa2_renut_neon_compare);  // sa2-nx (2026-10-20): backend/queued.cpp

namespace renut::gpu {

ChainHint g_chain_hint;  // sa2-nx (2026-10-20): draw.h

namespace {

// sa2-nx (2026-10-20, sa2_renut_lean_diagnostics): whether the per-draw diagnostic counts are kept (their lines are
// logged), once.
bool DiagCountsOn() {
  static const bool on = [] {
    if (!REXCVAR_GET(sa2_renut_lean_diagnostics))
      return true;
    auto *logger = ::rex::GetLoggerRaw(::rex::log::core());
    return logger && logger->should_log(spdlog::level::info);
  }();
  return on;
}

// banjo_renut_diag_blend_skip_exclude: the pixel shader's hash is in the list (read once).
bool BlendSkipExcluded(const void *pixel_shader) {
  static const std::unordered_set<uint64_t> excluded = [] {
    std::unordered_set<uint64_t> set;
    const std::string list = REXCVAR_GET(banjo_renut_diag_blend_skip_exclude);
    size_t at = 0;
    while (at < list.size()) {
      const size_t comma = list.find(',', at);
      const std::string item = list.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
      if (!item.empty())
        set.insert(std::strtoull(item.c_str(), nullptr, 16));
      if (comma == std::string::npos)
        break;
      at = comma + 1;
    }
    return set;
  }();
  const auto *entry = static_cast<const ShaderCacheEntry *>(pixel_shader);
  return entry && !excluded.empty() && excluded.count(entry->hash);
}

constexpr u32 kMaxDrawVertices = 1u << 21;
constexpr u32 kMaxDrawIndices = 1u << 22;
constexpr u32 kMaxFetchCopyBytes = 32u << 20;
constexpr u32 kUpEndian = 2;  // Begin*Vertices fetch constants use 8in32

// Xenos primitive types (D3DPRIMITIVETYPE).
enum Primitive : u32 {
  kPointList = 1,
  kLineList = 2,
  kLineStrip = 3,
  kTriangleList = 4,
  kTriangleFan = 5,
  kTriangleStrip = 6,
  kRectangleList = 8,
  kLineLoop = 12,
  kQuadList = 13,
  kTrianglePatch = 17,  // tessellated draws only
  kQuadPatch = 18,
};

// SharedConstants as shader_common.h lays it out for RENUT_RECOMP (both the
// SPIR-V raw loads and the DXIL cbuffer packoffsets).
struct SharedConstants {
  u32 texture_indices[3][32];  // 0: [2D, 3D, cube][fetch constant]
  u32 sampler_indices[32];     // 384
  u32 booleans[8];             // 512: VS bools 0-127, PS bools 128-255
  // 544 / 568: screen-space draws (screen_space): what the Xenos viewport
  // transform left of scale and offset, per axis, in pixels (see
  // ComputeViewport).
  float screen_offset[2];      // 544
  float alpha_threshold;       // 552
  u32 alpha_test_func;         // 556: CompareFunction, 7 (always) when disabled
  uint64_t vertex_fetch_data;  // 560: device address of this draw's fetch copies (SPIR-V)
  float screen_scale[2];       // 568
  u32 vertex_fetch_slots[96];  // 576: per fetch index, copy offset | endian
  uint64_t memexport_address;  // 960: device address of the export buffer (SPIR-V)
  // 968: 1 when the vertex shader's positions are in pixels (the viewport
  // transform off) and it maps them to clip space itself; 0 when the host
  // viewport does it.
  u32 screen_space;
  // 972, banjo-nx (package 27; was padding): the byte size of the fetch data from vertex_fetch_data, the bound of
  // vfetchMemoryClamped (shader_common.h): the predicated-block pass runs bone fetches unconditionally in that form.
  u32 vertex_fetch_bytes;
  // 976: memexport windows: guest byte address, size, byte offset in the
  // export buffer. None yet, so memExport stores nothing.
  u32 memexport_streams[4][4];
  // 1040: tessellated draws: patch kind (0 none, 1 triangles, 2 quads),
  // segments per edge, first patch index; other draws: [3] what the host vertex
  // ID lacks of the guest vertex index (renutVertexRegisters).
  u32 tessellation[4];
  // 1056: per vertex input the shader reads with vfetchBound, its element as the
  // bound declaration describes it: stream fetch index, offset and stride in
  // dwords, D3DDECLTYPE (see DescribeBoundFetches).
  u32 bound_fetch[16][4];
  // 1312: point sprites (point_sprite_gs.hlsl): the render state's point width
  // and height, minimum and maximum size, in pixels (PA_SU_POINT_SIZE holds
  // half the size in 12.4 fixed point, PA_SU_POINT_MINMAX the size).
  float point_state[4];
  // 1328: the size in guest pixels of what clip space spans (the host
  // viewport, which is 'resolution_scale' times that), and padding.
  float viewport_size[2];
  float resolution_scale;  // 1336: of the render targets (ResolutionScale)
  // 1340, banjo-nx: FLT_MAX, the Xenos clamps' bound the SPIR-V shaders read here (shader_common.h; was padding).
  float flt_max;
  // 1344 (Renut upstream 574b488, Beta-Nut 4: "fixed an issue where jinjo bingo would show a black square instead of
  // the silhouette"; banjo-nx 2026-10-19): in guest pixels, what the pixel position is relative to: a draw predicated
  // to one tile has the tile's corner (ViewportState::pixel_origin).
  float pixel_origin[2];
  float pad_1352[2];
};
static_assert(offsetof(SharedConstants, sampler_indices) == 384);
static_assert(offsetof(SharedConstants, booleans) == 512);
static_assert(offsetof(SharedConstants, screen_offset) == 544);
static_assert(offsetof(SharedConstants, screen_scale) == 568);
static_assert(offsetof(SharedConstants, screen_space) == 968);
static_assert(offsetof(SharedConstants, alpha_test_func) == 556);
static_assert(offsetof(SharedConstants, vertex_fetch_data) == 560);
static_assert(offsetof(SharedConstants, vertex_fetch_slots) == 576);
static_assert(offsetof(SharedConstants, memexport_address) == 960);
static_assert(offsetof(SharedConstants, memexport_streams) == 976);
static_assert(offsetof(SharedConstants, tessellation) == 1040);
static_assert(offsetof(SharedConstants, bound_fetch) == 1056);
static_assert(offsetof(SharedConstants, point_state) == 1312);
static_assert(offsetof(SharedConstants, viewport_size) == 1328);
static_assert(offsetof(SharedConstants, resolution_scale) == 1336);
static_assert(offsetof(SharedConstants, pixel_origin) == 1344);
static_assert(sizeof(SharedConstants) == 1360);
static_assert(sizeof(SharedConstants) == kSharedConstantsUboBytes);

// The guest D3DDevice a draw reads its state from: the live device, or a
// recorded copy of its first kDeviceSnapshotSize bytes.
class GuestDevice {
public:
  explicit GuestDevice(const u8 *bytes) : bytes_(bytes) {}
  bool valid() const { return bytes_ != nullptr; }
  const d3d::Device &dev() const { return *reinterpret_cast<const d3d::Device *>(bytes_); }
  u32 U32(u32 offset) const { return *reinterpret_cast<const be_u32 *>(bytes_ + offset); }
  u8 U8(u32 offset) const { return bytes_[offset]; }
  float F32(u32 offset) const {
    const u32 bits = U32(offset);
    float value;
    std::memcpy(&value, &bits, 4);
    return value;
  }
  // Xenos fetch constant 'index' (0-95) word 0/1 as a vertex fetch constant.
  u32 VertexFetchWord(u32 index, u32 word) const {
    return U32(u32(offsetof(d3d::Device, m_Constants)) + index * 8 + word * 4);
  }

private:
  const u8 *bytes_;
};

// A shader stage's last float constant upload: the guest bytes of the registers
// it holds, reused while the next draw's shader reads only registers it holds
// with the same bytes.
struct ConstantUpload {
  UploadAllocation alloc;
  u32 valid[8] = {};  // registers written to alloc, bit i = register i
  u8 bytes[256 * 16];  // their guest (big-endian) bytes
  // The replay (ReplayConstants::replay) whose draw last made 'valid' match
  // its constants, since when that replay's dirty groups are the only ones
  // that can differ; 0 for none.
  uint64_t replay = 0;
  // banjo-nx (banjo_gpu_literals_cbuf): the shader whose literal registers alloc holds (UploadFloatConstants).
  const ShaderCacheEntry *literals = nullptr;
  // banjo-nx (2026-10-19, banjo_deko3d_push_constants): the registers of the pushed uniform buffer that hold
  // 'literals''s literal values (cleared where a group update overwrites one).
  u32 literal_regs[8] = {};
};

// State bound on the current command list, reset whenever a new one begins.
struct Bound {
  uint64_t generation = UINT64_MAX;
  const plume::RenderFramebuffer *framebuffer = nullptr;
  const plume::RenderPipeline *pipeline = nullptr;
  bool layout = false;
  UploadAllocation zero_stream;
  ConstantUpload vs_constants, ps_constants;
  // banjo-nx (2026-10-19, banjo_deko3d_push_constants): the models of the pushed uniform buffers, apart from the ring
  // uploads' (point-sprite draws keep those, and the two must not share one model: emulator verify run 60).
  ConstantUpload pushed_vs, pushed_ps;
  bool blend_constant_set = false;
  float blend_constant[4] = {};

  // State as the draws last set it on the list, which most draws in a row
  // share (the viewport, the zero stream, the memexport buffer): setting it
  // again cost a driver call each. Both backends keep all of it across
  // pipeline and framebuffer changes; D3D12 root descriptors are dropped with
  // the root signature, so they go with 'layout'.
  bool viewport_set = false;
  plume::RenderViewport viewport;
  plume::RenderRect scissor;
  uint32_t viewport_count = 0;  // viewports and scissors set (SetViewport)
  bool root_set[kDrawRootMemExport + 1] = {};
  plume::RenderBufferReference roots[kDrawRootMemExport + 1];
  struct Stream {
    bool set = false;
    plume::RenderVertexBufferView view;
    plume::RenderInputSlot slot;
  };
  Stream streams[kZeroStreamSlot + 1];
  bool index_set = false;
  plume::RenderIndexBufferView index;
  // banjo-nx (banjo_renut_shared_dedupe): the last SharedConstants upload and its bytes, which a draw with the same
  // bytes reuses (no upload, and set 7 stays bound).
  UploadAllocation shared_alloc;
  bool shared_valid = false;
  SharedConstants shared_bytes;
  // banjo-nx (2026-10-19, banjo_deko3d_push_constants): what the pushed shared uniform buffer holds (valid once the
  // list's first pushed draw sent it whole), and whether plume reads bound textures from it.
  bool pushed_shared_valid = false;
  bool shared_words_set = false;
  SharedConstants pushed_shared;
  // banjo-nx: the uniform buffer sets 5-7 last bound (Vulkan), and their offsets.
  const plume::RenderDescriptorSet *constant_sets[3] = {};
  uint32_t constant_offsets[3] = {};
};
Bound g_bound;

// banjo-nx: binds a constant upload's uniform buffer set at its offset (Vulkan), unless it is bound there already:
// most draws in a row reuse the vertex shader's constants, and many the pixel shader's.
void BindConstantSet(plume::RenderCommandList *cmd, uint32_t set_index, plume::RenderDescriptorSet *set,
                     uint64_t offset) {
  const uint32_t slot = set_index - kVertexConstantsSet;
  const uint32_t offset32 = uint32_t(offset);
  if (g_bound.constant_sets[slot] == set && g_bound.constant_offsets[slot] == offset32)
    return;
  cmd->setGraphicsDescriptorSetDynamic(set, set_index, &offset32, 1);
  g_bound.constant_sets[slot] = set;
  g_bound.constant_offsets[slot] = offset32;
}

// The last declaration a draw looked up by VA (otherwise a registry lookup
// under its lock per draw): draws in a row mostly use the same one. Cleared
// when a declaration goes (ForgetVertexDeclarationLocked).
u32 g_last_declaration_va = 0;
VertexDeclarationRecord *g_last_declaration = nullptr;

bool SameReference(const plume::RenderBufferReference &a, const plume::RenderBufferReference &b) {
  return a.ref == b.ref && a.offset == b.offset;
}

// plume's Vulkan pipelines have one viewport and scissor per color target
// (all dynamic), each of which a draw must have set; they all get the same.
void SetViewport(plume::RenderCommandList *cmd, const plume::RenderViewport &viewport,
                 const plume::RenderRect &scissor, uint32_t color_count) {
  // Every index set so far is rewritten, so none keeps an older value.
  uint32_t count = std::clamp(color_count, 1u, uint32_t(d3d::kMaxRenderTargets));
  if (g_bound.viewport_set)
    count = std::max(count, g_bound.viewport_count);
  const plume::RenderViewport &v = g_bound.viewport;
  const plume::RenderRect &r = g_bound.scissor;
  const bool more = count > g_bound.viewport_count;
  if (!g_bound.viewport_set || more || v.x != viewport.x || v.y != viewport.y ||
      v.width != viewport.width || v.height != viewport.height ||
      v.minDepth != viewport.minDepth || v.maxDepth != viewport.maxDepth) {
    const plume::RenderViewport viewports[d3d::kMaxRenderTargets] = {viewport, viewport, viewport,
                                                                     viewport};
    cmd->setViewports(viewports, count);
    g_bound.viewport = viewport;
  }
  if (!g_bound.viewport_set || more || r.left != scissor.left || r.top != scissor.top ||
      r.right != scissor.right || r.bottom != scissor.bottom) {
    const plume::RenderRect scissors[d3d::kMaxRenderTargets] = {scissor, scissor, scissor, scissor};
    cmd->setScissors(scissors, count);
    g_bound.scissor = scissor;
  }
  if (!g_bound.viewport_set || more)
    g_bound.viewport_count = count;
  g_bound.viewport_set = true;
}

void SetRootDescriptor(plume::RenderCommandList *cmd, const plume::RenderBufferReference &ref,
                       uint32_t index) {
  if (g_bound.root_set[index] && SameReference(g_bound.roots[index], ref))
    return;
  cmd->setGraphicsRootDescriptor(ref, index);
  g_bound.roots[index] = ref;
  g_bound.root_set[index] = true;
}

void SetVertexBuffer(plume::RenderCommandList *cmd, const plume::RenderVertexBufferView &view,
                     const plume::RenderInputSlot &slot) {
  Bound::Stream &b = g_bound.streams[slot.index];
  if (b.set && SameReference(b.view.buffer, view.buffer) && b.view.size == view.size &&
      b.slot.stride == slot.stride && b.slot.classification == slot.classification)
    return;
  cmd->setVertexBuffers(slot.index, &view, 1, &slot);
  b.set = true;
  b.view = view;
  b.slot = slot;
}

void SetIndexBuffer(plume::RenderCommandList *cmd, const plume::RenderIndexBufferView &view) {
  const plume::RenderIndexBufferView &b = g_bound.index;
  if (g_bound.index_set && SameReference(b.buffer, view.buffer) && b.size == view.size &&
      b.format == view.format)
    return;
  cmd->setIndexBuffer(&view);
  g_bound.index = view;
  g_bound.index_set = true;
}

// The blend constant (RB_BLEND_RED..ALPHA) the draw's blend controls read, or
// false if none reads it. The host has a single constant for BLEND_FACTOR, so a
// colour factor that reads the constant alpha gets the alpha in rgb as well.
bool DrawBlendConstant(const d3d::Device &dev, const PipelineKey &key, float rgba[4]) {
  constexpr u32 kConstantColor = u32(D3DBlend::BlendFactor);
  constexpr u32 kOneMinusConstantAlpha = u32(D3DBlend::InvConstantAlpha);
  constexpr u32 kConstantAlpha = u32(D3DBlend::ConstantAlpha);
  bool used = false, color_reads_color = false, color_reads_alpha = false;
  for (u32 i = 0; i < key.color_count; ++i) {
    // RB_BLENDCONTROL factors: colour src 0, colour dst 8, alpha src 16, alpha dst 24.
    for (u32 shift : {0u, 8u, 16u, 24u}) {
      const u32 factor = (key.blend_control[i] >> shift) & 0x1F;
      if (factor < kConstantColor || factor > kOneMinusConstantAlpha)
        continue;
      used = true;
      if (shift < 16)
        (factor >= kConstantAlpha ? color_reads_alpha : color_reads_color) = true;
    }
  }
  if (!used)
    return false;
  const d3d::ValuesPacket &v = dev.m_ValuesPacket;
  const float alpha = v.BlendAlpha;
  const bool replicate_alpha = color_reads_alpha && !color_reads_color;
  if (color_reads_alpha && color_reads_color) {
    static bool logged = false;
    if (!logged) {
      logged = true;
      RGPU_WARN("Blend control {:08X} reads both the constant colour and alpha; the host uses the colour",
                key.blend_control[0]);
    }
  }
  rgba[0] = replicate_alpha ? alpha : float(v.BlendRed);
  rgba[1] = replicate_alpha ? alpha : float(v.BlendGreen);
  rgba[2] = replicate_alpha ? alpha : float(v.BlendBlue);
  rgba[3] = alpha;
  return true;
}

std::vector<u32> g_indices;  // scratch for index reads

// SetTilesLocked / SetReplayPredicationLocked.
std::vector<ClearRect> g_tiles;
u32 g_replay_predication = ~0u;

// A vertex stream as a draw reads it: guest vertices through a host layout.
struct StreamKey {
  const u8 *guest;
  uint64_t layout_hash;
  u32 stream, stride, count, endian;
  bool operator==(const StreamKey &) const = default;
};
struct StreamKeyHash {
  size_t operator()(const StreamKey &k) const {
    uint64_t h = uint64_t(reinterpret_cast<uintptr_t>(k.guest)) * 0x9E3779B97F4A7C15ull;
    h ^= k.layout_hash + 0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
    h ^= (uint64_t(k.stream) << 48 | uint64_t(k.endian) << 40 | k.stride) + (h << 6) + (h >> 2);
    h ^= uint64_t(k.count) * 0xFF51AFD7ED558CCDull + (h << 6) + (h >> 2);
    return size_t(h);
  }
};

// Vertex data a draw reads from guest buffers, read when the ring is kicked
// off (FlushVertexFillsLocked) rather than at the draw call: the GPU starts on
// a draw only after the kick, and the game may still write the buffers the
// draw uses until then.
struct PendingStreamFill {
  HostVertexStream stream;
  const u8 *guest = nullptr;
  u32 stride = 0, count = 0, endian = 0;
  u8 *host = nullptr;
  StreamKey key{};   // cacheable: the g_stream_cache entry
  bool cacheable = false;
  uint64_t vs_hash = 0, ps_hash = 0;  // mk9-nx: mk9_diag_bad_vertices names the draw
  u32 primitive = 0;
};
struct PendingCopy {
  const u8 *src = nullptr;
  u8 *dst = nullptr;
  u32 size = 0;
};
std::vector<PendingStreamFill> g_stream_fills;
std::vector<PendingCopy> g_copy_fills;

// Fetch data copies (UploadVertexFetchData) queued since the last flush, by
// guest bytes: they all read memory at the flush, so a later draw that reads
// the same bytes takes the queued copy rather than another. Where in its
// upload chunk the copy is (the draw's fetch slots count from the chunk's
// start). A generation, bumped at each flush, retires them all at once.
struct FetchCopy {
  const u8 *src = nullptr;
  u32 size = 0;
  uint64_t generation = 0;
  const plume::RenderBuffer *buffer = nullptr;
  u32 offset = 0;
};
constexpr u32 kFetchCopyBits = 8;
FetchCopy g_fetch_copies[1u << kFetchCopyBits];
uint64_t g_fetch_copy_generation = 1;

FetchCopy &FetchCopyFor(const u8 *src, u32 size) {
  const uint64_t key = uint64_t(reinterpret_cast<uintptr_t>(src)) ^ (uint64_t(size) << 40);
  return g_fetch_copies[(key * 0x9E3779B97F4A7C15ull) >> (64 - kFetchCopyBits)];
}

// Streams queued in g_stream_fills since the last flush. The fills all read
// guest memory at the flush, so a later draw that reads the same guest
// vertices through the same host layout draws from the upload already queued.
// banjo-nx (2026-10-19, render-thread proposal 20): an open-addressed table with a generation (no node allocated per
// stream and freed at each flush); a stream that finds no free slot within kQueuedProbe is converted again (as without
// the table: only slower).
struct QueuedStream {
  StreamKey key{};
  UploadAllocation alloc;
  uint64_t generation = 0;
};
constexpr uint32_t kQueuedStreamSlots = 2048;  // a power of two
constexpr uint32_t kQueuedProbe = 16;
std::vector<QueuedStream> g_queued_streams(kQueuedStreamSlots);
uint64_t g_queued_generation = 1;

const UploadAllocation *FindQueuedStream(const StreamKey &key) {
  const size_t h = StreamKeyHash{}(key);
  for (uint32_t i = 0; i < kQueuedProbe; ++i) {
    const QueuedStream &q = g_queued_streams[(h + i) & (kQueuedStreamSlots - 1)];
    if (q.generation != g_queued_generation)
      return nullptr;
    if (q.key == key)
      return &q.alloc;
  }
  return nullptr;
}

void AddQueuedStream(const StreamKey &key, const UploadAllocation &alloc) {
  const size_t h = StreamKeyHash{}(key);
  for (uint32_t i = 0; i < kQueuedProbe; ++i) {
    QueuedStream &q = g_queued_streams[(h + i) & (kQueuedStreamSlots - 1)];
    if (q.generation != g_queued_generation || q.key == key) {
      q.key = key;
      q.alloc = alloc;
      q.generation = g_queued_generation;
      return;
    }
  }
}

// Converted streams (and index ranges, below) kept across frames. Most are
// geometry the game never rewrites, which every draw would otherwise convert
// again. A stream whose guest bytes hash the same at kStableFlushes flushes in
// a row gets a persistent converted copy, which draws bind while the bytes
// still hash the same, checked off the render thread once every
// kCacheVerifyFrames frames a draw reads it: hashing every copy every frame
// cost more than any other part of a draw. A stream the game rewrites
// kMaxCacheDrops times is never cached again. A copy no draw has read for
// kStreamCacheIdleFrames goes: turning back to a scene within that converts
// nothing again.
constexpr u32 kStableFlushes = 3;
constexpr uint64_t kCacheVerifyFrames = 4;
constexpr u32 kMaxCacheDrops = 3;
constexpr uint64_t kStreamCacheIdleFrames = 3600;
constexpr uint64_t kStreamCacheMaxBytes = 768ull << 20;
uint64_t g_stream_cache_bytes = 0;
uint64_t g_stream_cache_frame = 0;
// banjo-nx (2026-10-17): FlushVertexFillsLocked's calls (every kick, and the end of every command list): the points
// after which the game may learn the GPU is done with what the draws before read, and write those buffers again.
// banjo_renut_exact_streams proves a cached copy once per interval between two of them.
uint64_t g_flush_serial = 0;
// banjo-nx (package 28, banjo_renut_diag_resolve_trace): draws since the last change of targets.
uint32_t g_trace_draws = 0;

// A cached copy's memory: part of one of the pool's buffers.
struct PoolSpan {
  plume::RenderBuffer *buffer = nullptr;
  u8 *cpu = nullptr;
  u32 chunk = 0;
  uint64_t offset = 0;
  u32 size = 0;
  explicit operator bool() const { return buffer != nullptr; }
  plume::RenderBufferReference Ref() const { return plume::RenderBufferReference(buffer, offset); }
};

// Where the cached copies live: a few large upload buffers, mapped once, that
// copies are carved from. A buffer of its own per copy cost a driver
// allocation each, and a scene coming into view made hundreds of copies in a
// frame. A freed span is reused once the frame slot it was freed in has
// finished on the GPU (DrainSlot), as retired resources are. A chunk left
// with nothing in it is released (the first one stays): the copies of a
// scene the game has left go after kStreamCacheIdleFrames, and their memory
// with them.
class CopyPool {
public:
  PoolSpan Allocate(u32 size) {
    const uint64_t need = (uint64_t(size) + kAlign - 1) & ~(kAlign - 1);
    for (u32 i = 0; i < chunks_.size(); ++i) {
      if (PoolSpan span = Take(i, need); span)
        return span;
    }
    const u32 index = AddChunk(need);
    if (index == UINT32_MAX)
      return {};
    return Take(index, need);
  }

  void Free(const PoolSpan &span) {
    if (span)
      deferred_[State().frame].push_back(span);
  }

  // The slot's lists have finished: what was freed while it recorded is free.
  void ReleaseSlot(uint32_t slot) {
    for (const PoolSpan &span : deferred_[slot]) {
      Chunk &chunk = chunks_[span.chunk];
      Insert(chunk, span.offset, span.size);
      // Spans hold their chunk's index, so the entry stays for AddChunk to
      // reuse. The other slot may still draw from it: retired, not freed.
      if (span.chunk != 0 && chunk.free.size() == 1 && chunk.free.begin()->second == chunk.size) {
        Video::RetireLocked(std::move(chunk.buffer));
        chunk.cpu = nullptr;
        chunk.free.clear();
        chunk.size = 0;
      }
    }
    deferred_[slot].clear();
  }

  // Host memory the chunks hold.
  uint64_t Bytes() const {
    uint64_t bytes = 0;
    for (const Chunk &chunk : chunks_)
      bytes += chunk.size;
    return bytes;
  }

private:
  static constexpr uint64_t kChunkSize = 64ull << 20;
  static constexpr uint64_t kAlign = 256;

  struct Chunk {
    std::unique_ptr<plume::RenderBuffer> buffer;  // null once released
    u8 *cpu = nullptr;
    uint64_t size = 0;
    std::map<uint64_t, uint64_t> free;  // offset -> size, adjacent ranges merged
  };

  // The index of a new chunk of at least 'min_size' bytes, or UINT32_MAX.
  u32 AddChunk(uint64_t min_size) {
    const uint64_t size = std::max(kChunkSize, min_size);
    auto buffer = State().device->createBuffer(plume::RenderBufferDesc::UploadBuffer(
        size, plume::RenderBufferFlag::VERTEX | plume::RenderBufferFlag::INDEX));
    void *mapped = buffer ? buffer->map() : nullptr;
    if (!mapped)
      return UINT32_MAX;
    u32 index = 0;
    while (index < chunks_.size() && chunks_[index].buffer)
      ++index;
    if (index == chunks_.size())
      chunks_.emplace_back();
    buffer->setName("renut stream cache #" + std::to_string(index));
    Chunk &chunk = chunks_[index];
    chunk.buffer = std::move(buffer);
    chunk.cpu = static_cast<u8 *>(mapped);
    chunk.size = size;
    chunk.free.emplace(0, size);
    return index;
  }

  // First fit in chunk 'index'.
  PoolSpan Take(u32 index, uint64_t size) {
    Chunk &chunk = chunks_[index];
    for (auto it = chunk.free.begin(); it != chunk.free.end(); ++it) {
      if (it->second < size)
        continue;
      const uint64_t offset = it->first, left = it->second - size;
      chunk.free.erase(it);
      if (left)
        chunk.free.emplace(offset + size, left);
      PoolSpan span;
      span.buffer = chunk.buffer.get();
      span.cpu = chunk.cpu + offset;
      span.chunk = index;
      span.offset = offset;
      span.size = u32(size);
      return span;
    }
    return {};
  }

  static void Insert(Chunk &chunk, uint64_t offset, uint64_t size) {
    auto next = chunk.free.lower_bound(offset);
    if (next != chunk.free.begin()) {
      auto prev = std::prev(next);
      if (prev->first + prev->second == offset) {
        offset = prev->first;
        size += prev->second;
        chunk.free.erase(prev);
      }
    }
    if (next != chunk.free.end() && offset + size == next->first) {
      size += next->second;
      chunk.free.erase(next);
    }
    chunk.free.emplace(offset, size);
  }

  std::vector<Chunk> chunks_;
  std::vector<PoolSpan> deferred_[kFramesInFlight];
};
CopyPool g_copy_pool;

// A direct-mapped memo of a node map's entries by key hash, in front of the
// stream and index caches: draws look up the same keys every frame, and the
// map's bucket and node chain were cache misses each time (a tenth of the
// render thread). Here a repeat is one line and the entry itself, which the
// draw reads anyway. The entries stay where they are until erased (the map is
// node-based, a rehash moves none), so the memo is cleared whenever the map
// erases any.
template <typename Map, typename Hash, u32 kBits>
class EntryMemo {
public:
  using Entry = typename Map::value_type;

  // banjo-nx (2026-10-19, render-thread proposal 9): the slot holds the key itself, so a hit compares it in the slot's
  // line (no read of the node's key, a second cache miss: the node is read after for its fields only).
  Entry *Find(const typename Map::key_type &key, size_t hash) const {
    const Slot &slot = slots_[Index(hash)];
    return slot.entry && slot.hash == hash && slot.key == key ? slot.entry : nullptr;
  }

  // The map's entry for 'key', made if 'make' (else null when it has none).
  Entry *Get(Map &map, const typename Map::key_type &key, bool make) {
    const size_t hash = Hash{}(key);
    if (Entry *entry = Find(key, hash))
      return entry;
    auto it = map.find(key);
    if (it == map.end()) {
      if (!make)
        return nullptr;
      it = map.try_emplace(key).first;
    }
    slots_[Index(hash)] = {hash, &*it, key};
    return &*it;
  }

  void Clear() { std::fill(std::begin(slots_), std::end(slots_), Slot{}); }

private:
  struct Slot {
    size_t hash = 0;
    Entry *entry = nullptr;
    typename Map::key_type key{};
  };
  static u32 Index(size_t hash) {
    return u32((uint64_t(hash) * 0x9E3779B97F4A7C15ull) >> (64 - kBits));
  }
  Slot slots_[1u << kBits];
};

struct CachedStream {
  uint64_t hash = 0;  // guest bytes at the last conversion, or of the copy
  u32 stable = 0;     // flushes in a row that converted the same bytes
  u32 drops = 0;      // copies dropped because the guest bytes changed
  bool dynamic = false;
  PoolSpan copy;
  u32 bytes = 0;
  uint64_t last_used = 0;               // g_stream_cache_frame
  uint64_t checked_frame = UINT64_MAX;  // g_stream_cache_frame the copy was last checked in
  uint64_t proven_flush = UINT64_MAX;   // banjo-nx: g_flush_serial the copy was last hashed equal in
  uint64_t exact_frame = UINT64_MAX;    // banjo-nx: g_stream_cache_frame of its last exact check (repeat count)
};
using StreamCache = std::unordered_map<StreamKey, CachedStream, StreamKeyHash>;
StreamCache g_stream_cache;
EntryMemo<StreamCache, StreamKeyHash, 12> g_stream_memo;  // banjo-nx: 4,096 slots of a key each (proposal 9)

// Whether a cached copy last checked in 'checked_frame' is due a check now.
bool CacheCheckDue(uint64_t checked_frame) {
  return checked_frame == UINT64_MAX || g_stream_cache_frame - checked_frame >= kCacheVerifyFrames;
}

// banjo-nx (package 27c): banjo_renut_exact_streams' checks, through a memo of the ranges hashed in this interval
// (between two kicks): a buffer cached under several stream keys (layouts, stream slots) or index keys is read once.
// Exact as the checks themselves: they hash at the first use in the interval, and so does the memo's first.
struct IntervalHash {
  const void *data = nullptr;
  size_t size = 0;
  uint64_t serial = UINT64_MAX;
  uint64_t hash = 0;
};
IntervalHash g_interval_hashes[4096];

// banjo-nx (2026-10-19, CPU proposal 1, banjo_renut_store_cards): the store cards (rex/system/xmemory.h,
// tools/store_cards.py). Every guest store outside the stack zeroes the card of its page (the byte at its guest page
// number in the first MB of the guest window), and so does every host writer of guest memory. A copy's check reads the
// cards of its pages, sets those it finds written back to 1 and records the interval (g_flush_serial) it found them in;
// the copy is current when no page of it was found written in or after the interval it was last proven in. A write
// whose card the check reads before its data is visible is found in that interval, so the copy is hashed again in the
// next one, when the data has long landed: the cards never answer for an interval in which they were cleared.
//
// A physical page's cards are those of the views the guest writes it through: 0xA0000000, 0xC0000000, 0xE0000000
// (4 KB lower: guest 0xE0000000 is physical 0x1000) and, in the first 16 MB, 0x7F000000. A store marks its first
// byte's page only, so an unaligned store that runs into the next page leaves that page's cards clean: a page also
// counts as written when the page before it was.
uint32_t g_page_serial[0x20000];

// banjo_renut_store_cards_verify (PC): what the checks found. [0] pages read, [1] found written, [2..5] by the view
// 0xA0000000 / 0xC0000000 / 0xE0000000 / 0x7F000000, [6] copies written only through the page before theirs,
// [7] copies hashed after a written page whose bytes had not changed, [8] copies not in physical memory.
uint64_t g_card_diag[9];
uint32_t g_card_samples[8];
uint32_t g_card_sample_count;

bool LogLimited(uint32_t &counter, uint32_t limit);

// The cards: the guest window's first MB (rex::memory::g_store_cards, a runtime variable this file cannot link to on
// Windows, where the runtime is a DLL).
u8 *StoreCards() {
  static u8 *const cards = REX_KERNEL_MEMORY()->virtual_membase();
  return cards;
}

// The physical address of the guest memory at 'data' (a pointer into the raw physical view, or into the guest window at
// a guest address of one of the views of physical memory: a vertex or index buffer's CPU address, where the game wrote
// it), or UINT32_MAX for anything else.
u32 PhysicalOf(const u8 *data) {
  static const u8 *const physical = GuestPhysicalPtr<const u8>(0);
  const uintptr_t raw = uintptr_t(data) - uintptr_t(physical);
  if (raw < 0x20000000u)
    return u32(raw);
  const uint64_t window = uint64_t(uintptr_t(data) - uintptr_t(StoreCards()));
  if (window >= 0x100000000ull)
    return UINT32_MAX;
  u32 va = u32(window);
  if (rex::memory::detail::PhysicalHostOffset(va) && va >= 0xE0001000u)
    va -= 0x1000u;  // the PC's host offset of the 0xE0000000 view
  if (va >= 0xA0000000u)
    return (va & 0x1FFFFFFFu) + (va >= 0xE0000000u ? 0x1000u : 0u);
  if (va >= 0x7F000000u && va < 0x80000000u)
    return va - 0x7F000000u;
  return UINT32_MAX;
}

bool PagesClean(const u8 *data, size_t bytes, uint64_t proven) {
  u8 *cards = StoreCards();
  const u32 offset = cards && bytes ? PhysicalOf(data) : UINT32_MAX;
  const bool diag = banjo::renut_gpu::StoreCardsVerifyOn();
  if (offset == UINT32_MAX || bytes > 0x20000000u - offset) {
    if (diag)
      ++g_card_diag[7 + 1];  // not physical memory
    return false;
  }
  const u32 serial = u32(g_flush_serial);
  const u32 last = u32((uint64_t(offset) + bytes - 1) >> 12);
  u32 page = offset >> 12;
  page -= page != 0;
  bool clean = true;
  const u32 own_first = offset >> 12;
  bool own_written = false;
  for (; page <= last; ++page) {
    u8 *a = cards + 0xA0000u + page, *c = cards + 0xC0000u + page;
    u8 *e = page ? cards + 0xE0000u + page - 1 : a, *f = page < 0x1000u ? cards + 0x7F000u + page : a;
    if (diag) {
      ++g_card_diag[0];
      const bool va = !*a, vc = !*c, ve = !*e, vf = !*f;
      if (va || vc || ve || vf) {
        ++g_card_diag[1];
        g_card_diag[2] += va;
        g_card_diag[3] += vc;
        g_card_diag[4] += ve;
        g_card_diag[5] += vf;
        own_written |= page >= own_first;
        if (g_card_sample_count < 8)
          g_card_samples[g_card_sample_count++] = page | (va ? 1u << 28 : 0) | (vc ? 1u << 29 : 0) |
                                                  (ve ? 1u << 30 : 0) | (vf ? 1u << 31 : 0);
      }
    }
    if (!__atomic_load_n(a, __ATOMIC_RELAXED) || !__atomic_load_n(c, __ATOMIC_RELAXED) ||
        !__atomic_load_n(e, __ATOMIC_RELAXED) || !__atomic_load_n(f, __ATOMIC_RELAXED)) {
      __atomic_store_n(a, u8(1), __ATOMIC_RELAXED);
      __atomic_store_n(c, u8(1), __ATOMIC_RELAXED);
      __atomic_store_n(e, u8(1), __ATOMIC_RELAXED);
      __atomic_store_n(f, u8(1), __ATOMIC_RELAXED);
      g_page_serial[page] = serial;
    }
    if (g_page_serial[page] >= u32(proven))
      clean = false;
  }
  if (diag && !clean && !own_written && own_first != 0 && g_page_serial[own_first - 1] >= u32(proven))
    ++g_card_diag[6];
  return clean;
}

// The diagnostics' 10 s line (banjo_renut_tuning.cpp calls it in verify mode).
extern "C" uint64_t RexStoreCardsHostBytes(int which);

// Whether the cards prove a cached copy last proven in interval 'proven' (UINT64_MAX: never, made at a flush) whose
// guest bytes hashed 'hash' then. The cards are read, and cleared, before any hash of the range in this interval.
bool CardsProve(const u8 *data, size_t bytes, uint64_t proven, uint64_t hash) {
  if (!banjo::renut_gpu::StoreCardsOn())
    return false;
  const bool clean = PagesClean(data, bytes, proven) && proven != UINT64_MAX;
  if (clean && banjo::renut_gpu::StoreCardsVerifyOn() && banjo::renut_gpu::ContentHash64(data, bytes) != hash) {
    static uint32_t logged = 0;
    if (LogLimited(logged, 32)) {
      static const u8 *const physical = GuestPhysicalPtr<const u8>(0);
      RGPU_WARN("[cards] physical {:08X}-{:08X} ({} bytes) changed under clean cards: a writer of guest memory that "
                "marks none (banjo_renut_store_cards_verify)",
                u32(data - physical), u32(data - physical + bytes - 1), bytes);
    }
    banjo::renut_gpu::CountCardsMissed(bytes);
    return false;
  }
  banjo::renut_gpu::CountCards(bytes, clean);
  return clean;
}

uint64_t ExactHash(const void *data, size_t size) {
  const uintptr_t k = (reinterpret_cast<uintptr_t>(data) >> 4) ^ (uintptr_t(size) * 0x9E3779B1u);
  IntervalHash &e = g_interval_hashes[(k ^ (k >> 12)) & 4095];
  if (e.serial == g_flush_serial && e.data == data && e.size == size) {
    banjo::renut_gpu::CountExactMemoHit(size);
    return e.hash;
  }
  e = {data, size, g_flush_serial, banjo::renut_gpu::ContentHash64(data, size)};
  return e.hash;
}

uint64_t HashGuestStream(const StreamKey &key) {
  // banjo-nx: banjo_renut_pmull_hash (the PMULL CRC-64 on ARM).
  return banjo::renut_gpu::ContentHash64(key.guest, size_t(key.count) * key.stride);
}

// The cached copies' checks, hashed off the render thread: every copy's guest
// bytes each kCacheVerifyFrames frames was a twentieth of its time. A copy
// found stale is dropped when the render thread takes the results, at the next
// frame (AdvanceStreamCacheFrameLocked), so a rewritten range can draw from
// its old copy for about a frame longer than the check interval alone allows.
template <typename Key, uint64_t (*Hash)(const Key &)>
class ContentVerifier {
public:
  struct Check {
    Key key;
    uint64_t hash;  // the copy's hash when submitted
  };

  ~ContentVerifier() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    wake_.notify_one();
    if (thread_.joinable())
      thread_.join();
  }

  // Render thread. Collected until the frame's Flush: waking the worker for
  // each check was a system call per draw.
  void Submit(const Key &key, uint64_t hash) { submitted_.push_back({key, hash}); }

  // Render thread, once a frame: hands the frame's checks to the worker.
  void Flush() {
    if (submitted_.empty())
      return;
    {
      std::lock_guard lock(mutex_);
      if (!thread_.joinable())
        thread_ = std::thread([this] { Run(); });
      pending_.insert(pending_.end(), submitted_.begin(), submitted_.end());
    }
    submitted_.clear();
    wake_.notify_one();
  }

  // Render thread: the checks whose bytes no longer hash as submitted.
  void TakeStale(std::vector<Check> &out) {
    std::lock_guard lock(mutex_);
    out.swap(stale_);
    stale_.clear();
  }

private:
  void Run() {
    // banjo-nx: the background core and priority on the Switch (banjo_renut_tuning.h).
    banjo::renut_gpu::OnBackgroundThreadStart("content verifier");
    std::vector<Check> batch;
    std::unique_lock lock(mutex_);
    for (;;) {
      wake_.wait(lock, [this] { return stop_ || !pending_.empty(); });
      if (stop_)
        return;
      batch.swap(pending_);
      lock.unlock();
      // Guest memory stays mapped; a read torn by a game write only fails
      // the check, which converts the range again.
      std::erase_if(batch, [](const Check &c) { return Hash(c.key) == c.hash; });
      lock.lock();
      stale_.insert(stale_.end(), batch.begin(), batch.end());
      batch.clear();
    }
  }

  std::vector<Check> submitted_;  // render thread only
  std::mutex mutex_;
  std::condition_variable wake_;
  std::vector<Check> pending_, stale_;
  bool stop_ = false;
  std::thread thread_;
};
ContentVerifier<StreamKey, HashGuestStream> g_stream_verifier;

void DropCachedCopy(CachedStream &c) {
  if (!c.copy)
    return;
  g_stream_cache_bytes -= c.bytes;
  g_copy_pool.Free(c.copy);
  c.copy = {};
  c.bytes = 0;
}

// After a flush converted 'f': count how long its bytes have held, and give a
// stream that has held long enough its persistent copy.
void TrackConvertedStream(const PendingStreamFill &f) {
  auto it = g_stream_cache.find(f.key);
  if (it == g_stream_cache.end())
    return;
  CachedStream &c = it->second;
  if (c.dynamic || c.copy)
    return;
  const uint64_t hash = HashGuestStream(f.key);
  c.stable = hash == c.hash ? c.stable + 1 : 0;
  c.hash = hash;
  const u32 bytes = f.count * f.stream.host_stride;
  if (c.stable + 1 < kStableFlushes ||
      g_stream_cache_bytes + bytes > banjo::renut_gpu::StreamCacheMaxBytes(kStreamCacheMaxBytes))  // banjo-nx
    return;
  const PoolSpan copy = g_copy_pool.Allocate(bytes);
  if (!copy)
    return;
  // Converted again rather than copied from f.host: upload memory is
  // write-combined, slow to read back.
  ConvertVertices(f.stream, f.guest, f.stride, f.count, f.endian, copy.cpu);
  c.copy = copy;
  c.bytes = bytes;
  c.checked_frame = g_stream_cache_frame;
  c.proven_flush = UINT64_MAX;  // banjo-nx: made at a flush; proven again at its first use after it
  g_stream_cache_bytes += bytes;
}

void SyncBound(VideoState &s) {
  if (g_bound.generation != s.list_generation) {
    g_bound = Bound{};
    g_bound.generation = s.list_generation;
  }
}

bool LogLimited(uint32_t &counter, uint32_t limit = 8) {
  return counter++ < limit;
}

// Guest VA or GPU physical address to a host pointer.
const u8 *AddressPtr(u32 address) {
  if (address < 0x20000000u)
    return GuestPhysicalPtr<const u8>(address);
  return GuestPtr<const u8>(address);
}

plume::RenderPrimitiveTopology Topology(u32 primitive) {
  switch (primitive) {
  case kPointList:
    return plume::RenderPrimitiveTopology::POINT_LIST;
  case kLineList:
    return plume::RenderPrimitiveTopology::LINE_LIST;
  case kLineStrip:
  case kLineLoop:
    return plume::RenderPrimitiveTopology::LINE_STRIP;
  case kTriangleStrip:
    return plume::RenderPrimitiveTopology::TRIANGLE_STRIP;
  default:  // lists, and fans / quads expanded to lists
    return plume::RenderPrimitiveTopology::TRIANGLE_LIST;
  }
}

// ---------------------------------------------------------------------------
// Render targets
// ---------------------------------------------------------------------------

struct Targets {
  HostSurface *colors[d3d::kMaxRenderTargets] = {};
  uint32_t color_count = 0;
  HostSurface *depth = nullptr;
  // In guest pixels, as everything the draws and clears work out; host
  // viewports, scissors and rects are 'scale' times that (HostSurface::scale).
  uint32_t width = 0, height = 0;
  RenderScale scale;
  // The size the guest surface header gives; smaller than width x height for
  // a target grown for tiling (SetTilingExtentLocked).
  uint32_t guest_width = 0, guest_height = 0;
};

// banjo-nx (2026-10-19, banjo_deko3d_depth_handover; "Depth hand-over" below): the targets of the draw whose textures
// FillTextures binds (a destination is not sampled from the image of a target of its own draw), and what is done
// before a depth target is bound again.
const Targets *g_handover_targets = nullptr;
void DepthHandOverBeforeBindLocked(HostSurface *surface, const plume::RenderRect *clear_rect, u32 clear_flags);

// A rect in guest pixels as host pixels of targets 'scale' times the size.
plume::RenderRect ScaleRect(const plume::RenderRect &rect, RenderScale scale) {
  return plume::RenderRect(scale.Coord(rect.left), scale.Coord(rect.top), scale.Coord(rect.right),
                           scale.Coord(rect.bottom));
}

bool GatherTargets(const GuestDevice &device, Targets &t) {
  for (u32 i = 0; i < d3d::kMaxRenderTargets; ++i) {
    const u32 va = device.U32(d3d::kDeviceOffsetRenderTargets + i * 4);
    if (!va)
      break;
    HostSurface *surface = GetHostSurfaceLocked(va);
    if (!surface || surface->depth)
      break;
    t.colors[t.color_count++] = surface;
  }
  HostSurface *depth = GetHostSurfaceLocked(device.U32(d3d::kDeviceOffsetDepthStencil));
  if (depth && depth->depth)
    t.depth = depth;
  if (!t.color_count && !t.depth)
    return false;
  const HostSurface *first = t.color_count ? t.colors[0] : t.depth;
  t.width = first->width;
  t.height = first->height;
  t.scale = first->scale;
  // A pass over targets rendered at different scales (graphics_options) would
  // address them alike; the options pick only targets drawn on their own. The
  // post-processing buffers render at 1x with the scene's depth still bound,
  // untested (on the console too its pitch is not theirs), so only colors count.
  bool mixed = false;
  for (uint32_t i = 1; i < t.color_count; ++i)
    mixed = mixed || t.colors[i]->scale != t.scale;
  if (mixed) {
    static uint32_t logged = 0;
    if (LogLimited(logged, 8)) {
      RGPU_WARN("Draw binds color targets of different render scales ({}x{} at {:.3g}x)",
                first->width, first->height, t.scale.Factor());
    }
  }
  t.guest_width = d3d::SurfaceWidth(first->size_word);
  t.guest_height = d3d::SurfaceHeight(first->size_word);
  // sa2-nx (2026-10-20, diagnostic): a depth target smaller than the colour targets. The draw area and scissor follow
  // the first colour target; a pixel past the depth target's edge is undefined on D3D12 and a "ZT width violation"
  // GPU exception on Maxwell (the PC's GPU reset at 04:45 with it, from some process). Counted for the report.
  if (t.depth && t.color_count && (t.depth->width < t.width || t.depth->height < t.height)) {
    static uint32_t logged = 0;
    if (logged++ < 8)
      RGPU_WARN("Draw binds a depth target smaller than its colour target: colour {:08X} {}x{}, depth {:08X} {}x{}",
                t.colors[0]->guest_va, t.width, t.height, t.depth->guest_va, t.depth->width, t.depth->height);
  }
  return true;
}

// gpu_profile: draws into 't', by their host size and attachments.
uint32_t TargetsProfileLabel(const Targets &t) {
  HostSurface *first = t.color_count ? t.colors[0] : t.depth;
  const uint32_t variant = t.color_count ? (t.depth ? 1 : 0) : 2;
  uint32_t &label = first->profile_labels[variant];
  if (label == UINT32_MAX) {
    static const char *const kKinds[3] = {"color", "color+depth", "depth"};
    label = gpu_profile::Label(fmt::format("draws {}x{} {}", first->scale.Size(first->width),
                                           first->scale.Size(first->height), kKinds[variant]));
  }
  return label;
}

// banjo-nx (banjo_renut_diag_gpu_shaders, package 28b): the draw's pass split by its pixel shader and blending.
uint32_t ShaderProfileLabel(const Targets &t, const ShaderRecord *ps, const PipelineKey &key) {
  const uint32_t target = TargetsProfileLabel(t);
  // RB_BLENDCONTROL ONE / ZERO / ADD for colour and alpha (BlendDesc's no-blend case).
  const bool blended = key.color_count && (key.blend_control[0] & 0x1FFF1FFF) != 0x00010001;
  const uint64_t k = (uint64_t(target) << 44) ^ (ps ? ps->hash : 0x5A5A5A5Au) ^ (blended ? (1ull << 63) : 0);
  static std::unordered_map<uint64_t, uint32_t> labels;
  if (auto it = labels.find(k); it != labels.end())
    return it->second;
  const std::string name = ps ? fmt::format("{} | PS {:016X}{}", gpu_profile::Name(target), ps->hash,
                                            blended ? " blended" : "")
                              : fmt::format("{} | no pixel shader", gpu_profile::Name(target));
  const uint32_t label = gpu_profile::Label(name);
  labels.emplace(k, label);
  return label;
}

bool BindTargets(const Targets &t) {
  if (t.depth && t.depth->handed)
    DepthHandOverBeforeBindLocked(t.depth, nullptr, 0);  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover)
  // banjo-nx: with banjo_renut_diag_gpu_shaders the draw marks its own label (ShaderProfileLabel), the pass's name
  // in it, which saves a timestamp a draw.
  if (!banjo::renut_gpu::DiagGpuShadersOn() && gpu_profile::Timed())
    gpu_profile::Mark(State(), TargetsProfileLabel(t));
  for (uint32_t i = 0; i < t.color_count; ++i)
    TransitionSurfaceLocked(t.colors[i], plume::RenderTextureLayout::COLOR_WRITE);
  if (t.depth)
    TransitionSurfaceLocked(t.depth, plume::RenderTextureLayout::DEPTH_WRITE);
  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): the last bind's framebuffer again for the same surfaces while no target
  // was built, retired or rebound (their generations with it) and no framebuffer retired: GetFramebufferLocked would
  // find that one.
  struct FramebufferReuse {
    HostSurface *colors[d3d::kMaxRenderTargets] = {};
    uint32_t count = 0;
    HostSurface *depth = nullptr;
    uint64_t surface_epoch = 0, framebuffer_epoch = 0;
    plume::RenderFramebuffer *framebuffer = nullptr;
  };
  static FramebufferReuse last;
  static const bool fb_reuse = REXCVAR_GET(sa2_renut_chain_reuse);
  static const bool fb_verify = fb_reuse && REXCVAR_GET(sa2_renut_chain_reuse_verify);
  const uint64_t surface_epoch = SurfaceEpochLocked(), framebuffer_epoch = FramebufferEpochLocked();
  bool same = fb_reuse && last.framebuffer && last.count == t.color_count && last.depth == t.depth &&
              last.surface_epoch == surface_epoch && last.framebuffer_epoch == framebuffer_epoch;
  for (uint32_t i = 0; same && i < t.color_count; ++i)
    same = last.colors[i] == t.colors[i];
  plume::RenderFramebuffer *framebuffer = nullptr;
  if (same) {
    framebuffer = last.framebuffer;
    if (fb_verify) {
      plume::RenderFramebuffer *made = GetFramebufferLocked(t.colors, t.color_count, t.depth);
      if (made != framebuffer) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused framebuffer differs from the one looked up");
        framebuffer = made;
      }
    }
  } else {
    framebuffer = GetFramebufferLocked(t.colors, t.color_count, t.depth);
    if (fb_reuse) {
      for (uint32_t i = 0; i < d3d::kMaxRenderTargets; ++i)
        last.colors[i] = i < t.color_count ? t.colors[i] : nullptr;
      last.count = t.color_count;
      last.depth = t.depth;
      last.surface_epoch = SurfaceEpochLocked();
      last.framebuffer_epoch = FramebufferEpochLocked();
      last.framebuffer = framebuffer;
    }
  }
  if (!framebuffer)
    return false;
  if (banjo::renut_gpu::ResolveTraceOn() && g_bound.framebuffer != framebuffer) {
    RGPU_INFO("[trace] targets: {} draws before; now {} colour {:08X} {:08X} ({}x{} fmt {} samples {}), depth {:08X} "
              "({}x{} samples {})",
              g_trace_draws, t.color_count, t.color_count ? t.colors[0]->guest_va : 0,
              t.color_count > 1 ? t.colors[1]->guest_va : 0, t.color_count ? t.colors[0]->width : 0,
              t.color_count ? t.colors[0]->height : 0, t.color_count ? u32(t.colors[0]->format) : 0,
              t.color_count ? u32(t.colors[0]->samples) : 0, t.depth ? t.depth->guest_va : 0,
              t.depth ? t.depth->width : 0, t.depth ? t.depth->height : 0, t.depth ? u32(t.depth->samples) : 0);
    if (const auto *desc = t.color_count ? GuestPtr<const d3d::SurfaceDesc>(t.colors[0]->guest_va + 0x18) : nullptr)
      RGPU_INFO("[trace]   colour EDRAM base {} pitch {} msaa {} format {:08X}", u32(desc->ColorInfo) & 0xFFF,
                u32(desc->SurfaceInfo) & 0x3FFF, (u32(desc->SurfaceInfo) >> 16) & 3, u32(desc->Format));
    g_trace_draws = 0;
  }
  if (g_bound.framebuffer != framebuffer) {
    State().command_list->setFramebuffer(framebuffer);
    g_bound.framebuffer = framebuffer;
  }
  return true;
}

// A draw's host viewport and scissor. The game's D3D sets the viewport one of
// two ways (N&B database):
// - D3DDevice_SetViewport (0x82222F20) writes scale w/2, -h/2, maxZ - minZ
//   and offset x + w/2, y + h/2, minZ, all enabled in PA_CL_VTE_CNTL: the
//   host viewport is that rectangle, and the shaders pass positions through.
// - D3DRS_VIEWPORTENABLE false (SetRenderState_ViewportEnable, 0x82205A80)
//   disables the transform (and clipping): the vertices are in pixels. The
//   host viewport then covers the target, and the vertex shader maps the
//   pixels to clip space itself (screen_space; any enabled part of the
//   transform comes along as screen_scale / screen_offset).
struct ViewportState {
  // Host pixels (Targets::scale).
  plume::RenderViewport viewport;
  plume::RenderRect scissor;
  // The viewport's size in guest pixels, which screen-space positions and
  // point sizes are in.
  float guest_size[2] = {};
  bool screen_space = false;
  float screen_scale[2] = {1.0f, 1.0f};
  float screen_offset[2] = {0.0f, 0.0f};
  // Guest pixels (Renut upstream 574b488): the pixel position (VPOS) the GPU gives the pixel shader is in the
  // coordinates of the EDRAM it renders to. While tiling that is one tile, and N&B adds the tile's corner back from a
  // per-tile constant (the deferred shadow pass, vs_f6965212d74a7552 / ps_d86a35e4f2b956b3: c0.x = 384 in the lower
  // tile). The host renders every tile into one target, so a draw predicated to one tile has its position made
  // relative to that tile.
  float pixel_origin[2] = {0.0f, 0.0f};
};

// The screen rect of the one tile a draw is predicated to while tiling
// (SetTilesLocked); false for a draw that runs in more tiles, or none.
bool PredicatedTile(const GuestDevice &device, ClearRect &rect) {
  if (g_tiles.size() < 2)
    return false;
  u32 mask = device.U32(d3d::kDeviceOffsetPredication);
  mask = (mask ? mask : ~0u) & g_replay_predication;
  int32_t tile = -1;
  for (size_t i = 0; i < g_tiles.size() && i < 16; ++i) {
    if (!(mask & (3u << (2 * i))))
      continue;
    if (tile >= 0)
      return false;
    tile = int32_t(i);
  }
  if (tile < 0)
    return false;
  rect = g_tiles[tile];
  return true;
}

ViewportState ComputeViewportSlow(const GuestDevice &device, const Targets &t);

// banjo-nx (2026-10-19, render-thread proposal 8): the last draw's viewport again when everything ComputeViewportSlow
// reads is the same: the device words (VteControl, the six viewport values, ModeControl, WindowOffset, the window
// scissor, 0x29C0), the predicated tile (worked out first: it reads the replay's predication and the tiles), the
// targets' sizes and scale, and the half-pixel switch.
ViewportState ComputeViewport(const GuestDevice &device, const Targets &t) {
  const d3d::Device &dev = device.dev();
  const d3d::ValuesPacket &v = dev.m_ValuesPacket;
  ClearRect tile{};
  const bool one_tile = PredicatedTile(device, tile);
  u32 key[24];
  key[0] = dev.m_ControlPacket.VteControl;
  std::memcpy(&key[1], &v.VportXScale, 4);
  std::memcpy(&key[2], &v.VportXOffset, 4);
  std::memcpy(&key[3], &v.VportYScale, 4);
  std::memcpy(&key[4], &v.VportYOffset, 4);
  std::memcpy(&key[5], &v.VportZScale, 4);
  std::memcpy(&key[6], &v.VportZOffset, 4);
  key[7] = u32(dev.m_ControlPacket.ModeControl);
  key[8] = dev.m_WindowPacket.WindowOffset;
  key[9] = dev.m_WindowPacket.WindowScissorTL;
  key[10] = dev.m_WindowPacket.WindowScissorBR;
  key[11] = device.U32(0x29C0);
  key[12] = u32(one_tile) | (banjo::renut_gpu::HalfPixelOffsetOn() ? 2u : 0u);
  key[13] = one_tile ? u32(tile.x1) : 0;
  key[14] = one_tile ? u32(tile.y1) : 0;
  key[15] = one_tile ? u32(tile.x2) : 0;
  key[16] = one_tile ? u32(tile.y2) : 0;
  key[17] = t.width;
  key[18] = t.height;
  key[19] = t.guest_width;
  key[20] = t.guest_height;
  key[21] = t.scale.code;
  key[22] = 0;
  key[23] = 0;
  static u32 last_key[24];
  static ViewportState last;
  static bool valid = false;
  if (valid && std::memcmp(key, last_key, sizeof(key)) == 0)
    return last;
  last = ComputeViewportSlow(device, t);
  std::memcpy(last_key, key, sizeof(key));
  valid = true;
  return last;
}

ViewportState ComputeViewportSlow(const GuestDevice &device, const Targets &t) {
  const d3d::Device &dev = device.dev();
  const u32 vte = dev.m_ControlPacket.VteControl;
  const d3d::ValuesPacket &v = dev.m_ValuesPacket;
  const float xs = (vte & 0x01) ? float(v.VportXScale) : 1.0f;
  float xo = (vte & 0x02) ? float(v.VportXOffset) : 0.0f;
  const float ys = (vte & 0x04) ? float(v.VportYScale) : 1.0f;
  float yo = (vte & 0x08) ? float(v.VportYOffset) : 0.0f;
  const float zs = (vte & 0x10) ? float(v.VportZScale) : 1.0f;
  const float zo = (vte & 0x20) ? float(v.VportZOffset) : 0.0f;

  // PA_SU_SC_MODE_CNTL.vtx_window_offset_enable adds PA_SC_WINDOW_OFFSET.
  int32_t wx = 0, wy = 0;
  if (u32(dev.m_ControlPacket.ModeControl) & 0x10000) {
    const u32 offset = dev.m_WindowPacket.WindowOffset;
    wx = int32_t(offset << 17) >> 17;
    wy = int32_t(offset << 1) >> 17;
    xo += float(wx);
    yo += float(wy);
  }
  // A draw predicated to one tile renders into that tile's EDRAM only, which
  // the GPU resolves to the tile's rect of the screen: it is cut to the rect.
  // Most such draws are in the tile's own coordinates, a viewport the size of
  // the tile-sized surface (N&B's per-tile scene command buffers: the vehicle
  // parts), which the host offsets to the rect. A viewport larger than that
  // surface addresses the whole screen (N&B's water, sub_823BA3A8, drawn once
  // per tile with a full-screen viewport and that tile's constants): it stays.
  ClearRect tile_rect{};
  const bool one_tile = PredicatedTile(device, tile_rect);
  const bool screen_wide = one_tile && (vte & 0x0F) == 0x0F && t.guest_width && t.guest_height &&
                           (2.0f * xs > float(t.guest_width) + 0.5f ||
                            -2.0f * ys > float(t.guest_height) + 0.5f);
  const bool tile_local = one_tile && !screen_wide;
  int32_t tx = 0, ty = 0;
  if (tile_local) {
    tx = tile_rect.x1;
    ty = tile_rect.y1;
  }
  xo += float(tx);
  yo += float(ty);

  ViewportState vs;
  if (one_tile) {  // Renut upstream 574b488 (the Jinjo Bingo silhouette)
    vs.pixel_origin[0] = float(tile_rect.x1);
    vs.pixel_origin[1] = float(tile_rect.y1);
  }
  const float min_z = std::clamp(zo, 0.0f, 1.0f), max_z = std::clamp(zo + zs, 0.0f, 1.0f);
  // banjo-nx (2026-10-18, banjo_renut_half_pixel_offset): PA_SU_VTX_CNTL.pix_center (device 0x29C0 bit 0, which
  // D3DRS_HALFPIXELOFFSET sets) 0 puts the pixel centres at integer window coordinates (D3D9's), so the GPU's window
  // positions are the host's plus half a pixel, as Xenia offsets the viewport (draw_util). Without it everything drew
  // half a pixel up-left; the depth of field's 320x180 passes, whose blur taps are one-sided, shifted their result a
  // whole texel: the orange halo above Kazooie's crest. Not in the scissor (whole pixels).
  const float pixel_center = (banjo::renut_gpu::HalfPixelOffsetOn() && !(device.U32(0x29C0) & 1)) ? 0.5f : 0.0f;
  // SetViewport's rectangle back from its scale and offset: x = offset - scale
  // and width = 2 * scale; y = offset + scale and height = -2 * scale (the
  // scale is negative: y goes down).
  vs.screen_space = (vte & 0x0F) != 0x0F || !(xs > 0.0f) || !(ys < 0.0f);
  if (!vs.screen_space) {
    vs.viewport = plume::RenderViewport(xo + pixel_center - xs, yo + pixel_center + ys, 2.0f * xs, -2.0f * ys,
                                        min_z, max_z);
  } else {
    vs.viewport = plume::RenderViewport(0.0f, 0.0f, float(t.width), float(t.height), min_z, max_z);
    vs.screen_scale[0] = xs;
    vs.screen_scale[1] = ys;
    vs.screen_offset[0] = xo + pixel_center;
    vs.screen_offset[1] = yo + pixel_center;
  }

  // PA_SC_WINDOW_SCISSOR_TL/BR: x 0-14, y 16-30.
  const u32 tl = dev.m_WindowPacket.WindowScissorTL;
  const u32 br = dev.m_WindowPacket.WindowScissorBR;
  int32_t sx1 = int32_t(tl & 0x7FFF) + wx + tx, sy1 = int32_t((tl >> 16) & 0x7FFF) + wy + ty;
  int32_t sx2 = int32_t(br & 0x7FFF) + wx + tx, sy2 = int32_t((br >> 16) & 0x7FFF) + wy + ty;
  // A target grown for tiling: D3D bounds the window scissor by the tile-sized
  // surface, and the GPU applies it per tile in window coordinates, so across
  // the tiles that bound covers the whole target (a one-tile draw is cut to
  // its tile below; one in tile coordinates has the bound offset there).
  if (!tile_local && t.guest_width && t.width > t.guest_width && sx2 >= int32_t(t.guest_width))
    sx2 = int32_t(t.width);
  if (!tile_local && t.guest_height && t.height > t.guest_height &&
      sy2 >= int32_t(t.guest_height))
    sy2 = int32_t(t.height);
  // A screen-space draw's clip space spans the whole target: what a partly
  // enabled transform would clip is cut by the scissor instead.
  if (vs.screen_space && (vte & 0x0F) == 0x0F) {
    sx1 = std::max(sx1, int32_t(std::floor(xo - std::fabs(xs))));
    sx2 = std::min(sx2, int32_t(std::ceil(xo + std::fabs(xs))));
    sy1 = std::max(sy1, int32_t(std::floor(yo - std::fabs(ys))));
    sy2 = std::min(sy2, int32_t(std::ceil(yo + std::fabs(ys))));
  }
  if (one_tile) {
    sx1 = std::max(sx1, tile_rect.x1);
    sy1 = std::max(sy1, tile_rect.y1);
    sx2 = std::min(sx2, tile_rect.x2);
    sy2 = std::min(sy2, tile_rect.y2);
  }
  sx1 = std::clamp(sx1, 0, int32_t(t.width));
  sy1 = std::clamp(sy1, 0, int32_t(t.height));
  sx2 = std::clamp(sx2, sx1, int32_t(t.width));
  sy2 = std::clamp(sy2, sy1, int32_t(t.height));

  // All of the above is in guest pixels; the host targets are t.scale times
  // that. Depth is not scaled.
  vs.guest_size[0] = vs.viewport.width;
  vs.guest_size[1] = vs.viewport.height;
  const float scale = t.scale.Factor();
  vs.viewport.x *= scale;
  vs.viewport.y *= scale;
  vs.viewport.width *= scale;
  vs.viewport.height *= scale;
  vs.scissor = ScaleRect(plume::RenderRect(sx1, sy1, sx2, sy2), t.scale);
  return vs;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

// Where a draw's float constant registers are (ReplayConstants).
struct FloatConstants {
  const u8 *state = nullptr;
  const u8 *running = nullptr;  // null: all in 'state'
  uint64_t set = 0;
  u32 offset = 0;  // of VertexShaderF or PixelShaderF in the device layout
  uint64_t *dirty = nullptr;    // ReplayConstants::vs_dirty / ps_dirty
  uint64_t replay = 0;

  const u8 *Register(u32 r) const {
    const bool own = !running || ((set >> (63 - r / 4)) & 1);
    return (own ? state : running) + offset + r * 16;
  }
};

FloatConstants DrawFloatConstants(const u8 *state, const ReplayConstants *replay, bool pixel) {
  FloatConstants c;
  c.state = state;
  c.offset = u32(offsetof(d3d::Device, m_Constants)) +
             u32(pixel ? offsetof(d3d::Constants, PixelShaderF) : offsetof(d3d::Constants, VertexShaderF));
  if (replay) {
    c.running = replay->running;
    c.set = pixel ? replay->ps_set : replay->vs_set;
    c.dirty = pixel ? replay->ps_dirty : replay->vs_dirty;
    c.replay = c.dirty ? replay->replay : 0;
  }
  return c;
}

// Float constants go by groups of four registers (64 bytes): the unit m_Pending
// marks dirty, and the one a replayed draw takes from its snapshot or from the
// running device (FloatConstants::Register), so a group's bytes are contiguous.
// Bit 63 - group, as in m_Pending.
constexpr u32 kConstantGroupBytes = 4 * 16;

// The groups holding a register the shader reads.
uint64_t UsedGroups(const u32 used[8]) {
  uint64_t groups = 0;
  for (u32 word = 0; word < 8; ++word) {
    for (u32 bits = used[word]; bits;) {
      const u32 k = u32(std::countr_zero(bits)) / 4;
      groups |= uint64_t(1) << (63 - (word * 8 + k));
      bits &= ~(0xFu << (k * 4));
    }
  }
  return groups;
}

// banjo-nx (2026-10-19, render-thread proposal 12): UsedGroups and the registers of those groups (an upload's 'valid')
// once per shader: 'used' is a shader's ShaderCacheEntry::floatConstants (static) or one of the two static masks, so
// its address names it.
struct GroupSummary {
  const u32 *used = nullptr;
  uint64_t groups = 0;
  u32 valid[8] = {};
  // banjo-nx (2026-10-19, render-thread proposal 10): the bytes an upload must hold for the shader: all 257 registers
  // (with the zero register 256 that indexed reads past the end reach) when it indexes (every register marked used),
  // else up to its last group.
  uint32_t bytes = 0;
};
GroupSummary g_group_summaries[512];

const GroupSummary &SummaryOf(const u32 used[8]) {
  const uintptr_t a = reinterpret_cast<uintptr_t>(used);
  GroupSummary &g = g_group_summaries[((a >> 3) ^ (a >> 12)) & 511];
  if (g.used != used) {
    g.groups = UsedGroups(used);
    for (u32 word = 0; word < 8; ++word) {
      const u32 word_groups = u32(g.groups >> (56 - word * 8)) & 0xFFu;  // group word * 8 at bit 7
      u32 valid = 0;
      for (u32 k = 0; k < 8; ++k) {
        if ((word_groups >> (7 - k)) & 1)
          valid |= 0xFu << (k * 4);
      }
      g.valid[word] = valid;
    }
    bool all = true;
    for (u32 word = 0; word < 8; ++word)
      all &= used[word] == ~0u;
    g.bytes = all ? kFloatConstantsUboBytes
          : g.groups ? (64u - u32(std::countr_zero(g.groups))) * u32(kConstantGroupBytes)  // last group + 1
                     : 16u;
    g.used = used;
  }
  return g;
}

// The registers of a group the shader reads, bit i = the group's register i.
u32 GroupRegisters(const u32 used[8], u32 group) {
  return (used[group / 8] >> (group % 8 * 4)) & 0xFu;
}

// The registers a shader reads ('used', ShaderCacheEntry::floatConstants) as a
// little-endian upload of all 256, reusing the last upload on this command list
// when it holds them with the same bytes. Only the groups holding a register
// the shader reads are written, whole: fixed-size copies the compiler keeps
// inline, where a run of registers at a time cost a branch per register.
UploadAllocation UploadFloatConstants(const FloatConstants &constants, const u32 used[8],
                                      ConstantUpload &last, const ShaderCacheEntry *entry) {
  // banjo-nx (banjo_gpu_literals_cbuf): the shader's literal registers it reads from the constant buffer hold its
  // literal values, as the console's Direct3D loads them at SetShader. An upload holding another shader's (or none) is
  // not reused for it.
  const ShaderCacheEntry *literals = entry && entry->literalCount ? entry : nullptr;
  bool reuse = bool(last.alloc) && (!literals || last.literals == literals);
  for (u32 word = 0; reuse && word < 8; ++word)
    reuse = !(used[word] & ~last.valid[word]);
  // Within the replay that last matched 'last', only groups written since can
  // differ (the running device is the same throughout a replay).
  const bool tracked = constants.replay && last.replay == constants.replay;
  const uint64_t dirty = tracked ? *constants.dirty : ~uint64_t(0);
  const GroupSummary &summary = SummaryOf(used);  // banjo-nx (2026-10-19, render-thread proposal 12)
  const uint64_t groups = summary.groups;
  // Only the registers the shader reads count: the rest of a group may be
  // another shader's, and differ.
  for (uint64_t check = reuse ? groups & dirty : 0; reuse && check; check &= check - 1) {
    const u32 group = 63 - u32(std::countr_zero(check));
    const u32 registers = GroupRegisters(used, group);
    const u8 *cached = last.bytes + group * kConstantGroupBytes;
    const u8 *guest = constants.Register(group * 4);
    uint64_t diff = 0;
    for (u32 i = 0; i < 4; ++i) {
      if (!((registers >> i) & 1))
        continue;
      uint64_t a[2], b[2];
      std::memcpy(a, cached + i * 16, 16);
      std::memcpy(b, guest + i * 16, 16);
      diff |= (a[0] ^ b[0]) | (a[1] ^ b[1]);
    }
    reuse = diff == 0;
  }
  if (reuse) {
    if (constants.replay && !tracked) {
      // Every register the draw reads now matches: keep just those, and
      // track the replay's writes from here.
      std::memcpy(last.valid, used, sizeof(last.valid));
      last.replay = constants.replay;
      *constants.dirty = 0;
    }
    return last.alloc;
  }

  // banjo-nx (2026-10-19, render-thread proposal 10): only the bytes the shader reads (its groups, its literal
  // registers; all of them when it indexes) are allocated, so only those are cleaned at the submit; the uniform buffer
  // stays bound at its full size, which the allocation keeps room for in the chunk. The registers past what is
  // allocated are never read by this shader (it reads only those in floatConstants, all of them when it indexes).
  uint32_t need = summary.bytes;
  if (literals) {
    for (u32 i = 0; i < literals->literalCount; ++i) {
      const ShaderLiteral &l = g_shaderLiterals[literals->literalOffset + i];
      if (l.reg < 256)
        need = std::max(need, (l.reg + 1) * 16u);
    }
  }
  UploadAllocation alloc = Video::UploadsLocked().AllocateWithRoom(need, kFloatConstantsUboBytes, 256);
  if (!alloc)
    return alloc;
  auto *out = static_cast<u8 *>(alloc.cpu);
  // banjo-nx (2026-10-18): register 256, zero (pass bonerow of tools/deko3d_glsl_opt.py), for a shader that indexes.
  if (need > 256 * 16)
    std::memset(out + 256 * 16, 0, kFloatConstantsUboBytes - 256 * 16);
  // The upload memory is write-combined: written in order, never read.
  for (uint64_t remaining = groups; remaining;) {
    const u32 group = u32(std::countl_zero(remaining));
    remaining &= ~(uint64_t(1) << (63 - group));
    u32 lanes[kConstantGroupBytes / 4];
    std::memcpy(lanes, constants.Register(group * 4), kConstantGroupBytes);
    std::memcpy(last.bytes + group * kConstantGroupBytes, lanes, kConstantGroupBytes);
    for (u32 &lane : lanes)
      lane = std::byteswap(lane);
    std::memcpy(out + group * kConstantGroupBytes, lanes, kConstantGroupBytes);
  }
  // Every register of those groups is in the upload, with the bytes it has
  // now.
  std::memcpy(last.valid, summary.valid, sizeof(last.valid));
  // banjo-nx (banjo_gpu_literals_cbuf): the literal registers, after the groups (a group may hold one). They are not
  // the device's values: out of 'valid', so a shader reading them as registers takes a new upload.
  last.literals = literals;
  if (literals) {
    for (u32 i = 0; i < literals->literalCount; ++i) {
      const ShaderLiteral &l = g_shaderLiterals[literals->literalOffset + i];
      if (l.reg >= 256)
        continue;
      std::memcpy(out + l.reg * 16, l.value, 16);
      last.valid[l.reg / 32] &= ~(1u << (l.reg % 32));
    }
  }
  last.alloc = alloc;
  last.replay = constants.replay;
  if (constants.replay)
    *constants.dirty = 0;
  return alloc;
}

// ---------------------------------------------------------------------------
// Pushed constants (banjo-nx, 2026-10-19, banjo_deko3d_push_constants; render-thread proposals 2 and 5)
// ---------------------------------------------------------------------------
// On deko3d the vertex shader's float constants, the pixel shader's and SharedConstants live in three fixed uniform
// buffers (zeroed once, so register 256 stays zero), bound at the bindings the game shaders read (0, 1, 2) and updated
// with pipelined constant-buffer updates (plume::Deko3DPushUniform: dkCmdBufPushConstants; a draw reads the values
// pushed before it in command order). ConstantUpload is then a model of what the GPU's buffer holds after the list's
// updates so far (valid: registers holding the device's bytes in 'bytes'; literal_regs: registers holding 'literals''s
// literal values), reset with each list (Bound). A draw pushes each 64-byte group holding a register the shader reads
// that is not valid or whose guest bytes differ; within a command buffer replay, a valid group the replay has not
// written since (its dirty mask) needs no compare. Then the shader's literal registers, unless they hold them already.
// Every register a shader reads holds what UploadFloatConstants would have written: the device's bytes, byte-swapped,
// or the literal. SharedConstants: the 64-byte lines that differ from what the buffer holds; plume reads bound
// textures from the renderer's copy (Deko3DSetSharedWords). Draws whose point_sprite_gs reads SharedConstants through
// its address keep the ring uploads (an update is ordered for constant-bank reads, not for memory reads).
#if defined(PLUME_DEKO3D)
struct PushedBuffers {
  std::unique_ptr<plume::RenderBuffer> buffer;
  uint64_t address = 0;
};
PushedBuffers g_pushed_buffers;
constexpr uint64_t kPushedFloatStride = (kFloatConstantsUboBytes + 255u) & ~uint64_t(255);
constexpr uint64_t kPushedSharedOffset = 2 * kPushedFloatStride;

bool EnsurePushedBuffers() {
  if (g_pushed_buffers.address)
    return true;
  auto &s = State();
  const uint64_t size = kPushedSharedOffset + ((sizeof(SharedConstants) + 255u) & ~uint64_t(255));
  g_pushed_buffers.buffer = s.device->createBuffer(plume::RenderBufferDesc::UploadBuffer(size));
  if (!g_pushed_buffers.buffer)
    return false;
  if (void *cpu = g_pushed_buffers.buffer->map()) {
    std::memset(cpu, 0, size);  // register 256 (and everything not pushed yet) zero
    g_pushed_buffers.buffer->unmap();
  }
  g_pushed_buffers.address = g_pushed_buffers.buffer->getDeviceAddress();
  return g_pushed_buffers.address != 0;
}

// The registers of the groups in a byte of a group mask (group word * 8 + k at bit 7 - k, as SummaryOf takes them),
// as a 'valid' word.
constexpr std::array<u32, 256> kGroupByteRegisters = [] {
  std::array<u32, 256> t{};
  for (u32 b = 0; b < 256; ++b) {
    for (u32 k = 0; k < 8; ++k) {
      if ((b >> (7 - k)) & 1)
        t[b] |= 0xFu << (k * 4);
    }
  }
  return t;
}();

// One stage's float constants (see above); 'ubo' its buffer's address.
void PushFloatConstants(plume::RenderCommandList *cmd, const FloatConstants &constants, const u32 used[8],
                        ConstantUpload &last, const ShaderCacheEntry *entry, uint64_t ubo) {
  const GroupSummary &summary = SummaryOf(used);
  const ShaderCacheEntry *literals = entry && entry->literalCount ? entry : nullptr;
  // The shader's literal registers, when the buffer holds them (literal_regs is exactly one shader's): a shader that
  // indexes reads every register, its literal ones included, and those hold the literal, not the device's bytes.
  const u32 *own_literals = literals && last.literals == literals ? last.literal_regs : nullptr;
  const bool tracked = constants.replay && last.replay == constants.replay;
  const uint64_t dirty = tracked ? *constants.dirty : ~uint64_t(0);
  // The groups to update (bit 63 - group, as UsedGroups).
  uint64_t push = 0;
  // sa2-nx (2026-10-20, sa2_renut_snapshot_deltas_constants): within a tracked replay (the chain's draws are one) with
  // none of the shader's groups dirty, a group is pushed only for a register the buffer does not hold: when every
  // register the shader reads (but its own literal ones) is held, the loop below would push nothing. Word by word.
  bool all_held = tracked && (dirty & summary.groups) == 0;
  for (u32 w = 0; all_held && w < 8; ++w)
    all_held = (used[w] & ~(own_literals ? own_literals[w] : 0u) & ~last.valid[w]) == 0;
  for (uint64_t check = all_held ? 0 : summary.groups; check; check &= check - 1) {
    const u32 bit = u32(std::countr_zero(check));
    const u32 group = 63 - bit;
    const u32 lits = own_literals ? (own_literals[group / 8] >> (group % 8 * 4)) & 0xFu : 0u;
    const u32 registers = GroupRegisters(used, group) & ~lits;
    const u32 held = (last.valid[group / 8] >> (group % 8 * 4)) & 0xFu;
    if (registers & ~held) {
      push |= uint64_t(1) << bit;
      continue;
    }
    if (!((dirty >> bit) & 1))
      continue;  // valid, and the replay has not written it since
    const u8 *cached = last.bytes + group * kConstantGroupBytes;
    const u8 *guest = constants.Register(group * 4);
    uint64_t diff = 0;
    for (u32 i = 0; i < 4; ++i) {
      if (!((registers >> i) & 1))
        continue;
      uint64_t a[2], b[2];
      std::memcpy(a, cached + i * 16, 16);
      std::memcpy(b, guest + i * 16, 16);
      diff |= (a[0] ^ b[0]) | (a[1] ^ b[1]);
    }
    if (diff)
      push |= uint64_t(1) << bit;
  }
  // The updates, a run of consecutive groups at a time.
  uint64_t pushed_bytes = 0;
  bool overwrote_literal = false;
  for (uint64_t remaining = push; remaining;) {
    const u32 first = u32(std::countl_zero(remaining));  // the lowest group left
    u32 count = 0;
    while (first + count < 64 && ((remaining >> (63 - (first + count))) & 1))
      ++count;
    alignas(16) u32 lanes[64 * kConstantGroupBytes / 4];
    for (u32 k = 0; k < count; ++k) {
      const u32 group = first + k;
      const u8 *guest = constants.Register(group * 4);
      std::memcpy(last.bytes + group * kConstantGroupBytes, guest, kConstantGroupBytes);
      std::memcpy(lanes + k * (kConstantGroupBytes / 4), guest, kConstantGroupBytes);
      last.valid[group / 8] |= 0xFu << (group % 8 * 4);
      const u32 lits = (last.literal_regs[group / 8] >> (group % 8 * 4)) & 0xFu;
      if (lits) {
        overwrote_literal = true;
        last.literal_regs[group / 8] &= ~(0xFu << (group % 8 * 4));
      }
      remaining &= ~(uint64_t(1) << (63 - group));
    }
    for (u32 k = 0; k < count * (kConstantGroupBytes / 4); ++k)
      lanes[k] = std::byteswap(lanes[k]);
    plume::Deko3DPushUniform(cmd, ubo, kFloatConstantsUboBytes, first * kConstantGroupBytes, lanes,
                             count * kConstantGroupBytes);
    pushed_bytes += count * kConstantGroupBytes;
  }
  // The literal registers: after the groups (a group may hold one, and its update overwrote it).
  if (overwrote_literal && last.literals != literals)
    last.literals = nullptr;  // another shader's literals, no longer all in place
  if (literals && (last.literals != literals || overwrote_literal)) {
    if (last.literals != literals) {
      // Another shader's literal registers left in the buffer are out of 'valid' already; literal_regs is this one's.
      for (u32 &word : last.literal_regs)
        word = 0;
    }
    for (u32 i = 0; i < literals->literalCount; ++i) {
      const ShaderLiteral &l = g_shaderLiterals[literals->literalOffset + i];
      if (l.reg >= 256)
        continue;
      plume::Deko3DPushUniform(cmd, ubo, kFloatConstantsUboBytes, l.reg * 16, l.value, 16);
      pushed_bytes += 16;
      last.valid[l.reg / 32] &= ~(1u << (l.reg % 32));
      last.literal_regs[l.reg / 32] |= 1u << (l.reg % 32);
    }
    last.literals = literals;
  }
  // The replay's writes from here: the next draw of this replay skips a valid group the replay has not written since,
  // so 'valid' keeps only registers that hold the device's bytes now: those of the groups pushed, those this draw
  // compared, and (tracked) those of groups the replay has not written since this replay's last draw. A register of a
  // group compared for another (a shader reads some of a group's registers) is not known (emulator verify run 61: 8 %
  // of the draws read one stale).
  if (constants.replay) {
    const uint64_t unsure = (tracked ? dirty : ~uint64_t(0)) & ~push;
    // sa2-nx (2026-10-20): with no group unsure (a chain draw whose groups are all clean) the loop changes nothing.
    for (u32 word = 0; unsure && word < 8; ++word)
      last.valid[word] &= ~kGroupByteRegisters[u32(unsure >> (56 - word * 8)) & 0xFFu] | used[word];
    last.replay = constants.replay;
    *constants.dirty = 0;
  } else {
    last.replay = 0;
  }
  banjo::renut_gpu::CountPushedConstants(pushed_bytes, 0);
  if (banjo::renut_gpu::PushConstantsVerifyOn()) {
    bool ok = true;
    const u32 *literal_regs = literals && last.literals == literals ? last.literal_regs : nullptr;
    for (u32 r = 0; r < 256 && ok; ++r) {
      if (!((used[r / 32] >> (r % 32)) & 1))
        continue;
      if (literal_regs && ((literal_regs[r / 32] >> (r % 32)) & 1))
        continue;  // the literal (checked below)
      ok = ((last.valid[r / 32] >> (r % 32)) & 1) &&
           std::memcmp(last.bytes + r * 16, constants.Register(r), 16) == 0;
    }
    if (literals) {
      for (u32 i = 0; i < literals->literalCount && ok; ++i) {
        const u32 reg = g_shaderLiterals[literals->literalOffset + i].reg;
        ok = reg >= 256 || (last.literals == literals && ((last.literal_regs[reg / 32] >> (reg % 32)) & 1));
      }
    }
    if (!ok) {
      static uint32_t logged = 0;
      if (logged++ < 16)
        RGPU_WARN("[pushed] a register the shader reads is not what the model says the GPU holds (banjo_deko3d_push_"
                  "constants_verify)");
      banjo::renut_gpu::CountPushedConstants(1, 3);
    }
  }
}

// SharedConstants: the lines that differ from what the buffer holds.
// sa2-nx (2026-10-20, sa2_renut_inline_line_compare): whether a line of the shared constants differs from the held
// one: a whole 64-byte line as fixed-size loads the compiler keeps inline (a memcmp call per line before: 21 calls a
// draw, the render thread's memcmp of the first console profile); the 16-byte tail by memcmp.
inline bool SharedLineDiffers(const u8 *a, const u8 *b, u32 n) {
  static const bool inline_on = REXCVAR_GET(sa2_renut_inline_line_compare);
  static const bool neon = REXCVAR_GET(sa2_renut_neon_compare);  // sa2-nx (2026-10-20): sa2_line_compare.h
  if (n != 64 || !inline_on)
    return std::memcmp(a, b, n) != 0;
  return sa2::renut::Line64Differs(a, b, neon);
}

// sa2-nx (2026-10-20, sa2_renut_chain_reuse): the pushes counted; 'equal_lines' (bit i = 64-byte line i) are lines the
// buffer holds already: the texture words of a chain draw that are the previous draw's when it pushed last, and the
// vertex fetch slots a draw without fetches leaves zero when the last push's were zero too.
uint64_t g_shared_pushes = 0;
bool g_held_fetch_slots_zero = false;  // the buffer's vertex fetch slots (lines 9-14) are zero
constexpr u32 kTextureLines = 0xFFu;      // lines 0-7: the texture and sampler words
constexpr u32 kFetchSlotLines = 0x7E00u;  // lines 9-14: vertex_fetch_slots
static_assert(offsetof(SharedConstants, vertex_fetch_slots) == 9 * 64 &&
              sizeof(SharedConstants::vertex_fetch_slots) == 6 * 64);

void PushSharedConstants(plume::RenderCommandList *cmd, const SharedConstants &sc, u32 equal_lines = 0) {
  constexpr u32 kLine = 64;
  constexpr u32 kSize = u32(sizeof(SharedConstants));
  ++g_shared_pushes;
  const uint64_t ubo = g_pushed_buffers.address + kPushedSharedOffset;
  const u8 *now = reinterpret_cast<const u8 *>(&sc);
  u8 *held = reinterpret_cast<u8 *>(&g_bound.pushed_shared);
  uint64_t pushed_bytes = 0;
  if (!g_bound.pushed_shared_valid) {
    plume::Deko3DPushUniform(cmd, ubo, kSize, 0, now, kSize);
    std::memcpy(held, now, kSize);
    g_bound.pushed_shared_valid = true;
    pushed_bytes = kSize;
  } else {
    for (u32 at = 0; at < kSize;) {
      const u32 n = std::min(kLine, kSize - at);
      if (((equal_lines >> (at / kLine)) & 1u) || !SharedLineDiffers(held + at, now + at, n)) {
        at += n;
        continue;
      }
      // A run of lines that differ.
      u32 end = at + n;
      while (end < kSize) {
        const u32 m = std::min(kLine, kSize - end);
        if (((equal_lines >> (end / kLine)) & 1u) || !SharedLineDiffers(held + end, now + end, m))
          break;
        end += m;
      }
      plume::Deko3DPushUniform(cmd, ubo, kSize, at, now + at, end - at);
      std::memcpy(held + at, now + at, end - at);
      pushed_bytes += end - at;
      at = end;
    }
  }
  banjo::renut_gpu::CountPushedConstants(pushed_bytes, 1);
}
#endif  // PLUME_DEKO3D

TextureFetch DeviceTextureFetch(const d3d::Device &dev, u32 r) {
  return TextureFetch::FromGuest(dev.m_Constants.TextureFetch[r]);
}

// banjo-nx (deko3d renderer, docs/deko3d-renderer.md): the sampler words hold the sampler id where a texture handle has
// it (bit 20 up), so the deko3d shaders make the handle as (texture word & 0xFFFFF) | (sampler word & 0xFFF00000): no
// shift, and the form Ryujinx's shader translator follows back to the constant buffer (the emulator tests).
inline u32 SamplerWord(u32 slot) { return IsDeko3D() ? slot << 20 : slot; }

// banjo-nx (2026-10-19, CPU proposal 2, banjo_renut_cpu_memos): FillTextures' answer per fetch slot (the shared-constant
// index and sampler words), keyed by the texture address, the device's fetch constant and the guest header's words,
// valid for the frame (the first bind of a frame runs the content check) while g_texture_epoch is unchanged. A hit does
// the bind's transitions and counters again.
struct TextureSlotMemo {
  u32 va = 0;
  u32 device[6] = {};
  u32 header[6] = {};
  uint64_t frame = 0;
  uint64_t epoch = 0;
  uint64_t depth_epoch = 0;  // banjo-nx (2026-10-19, render-thread proposal 3): for binds through depth words
  u32 dim = 0;
  u32 index = 0;
  u32 sampler = 0;
  BindEffects fx;
};
// banjo-nx (2026-10-19, render-thread proposal 3): by texture (its header address and the fetch constant's first
// words), not by fetch slot: a texture bound earlier in the frame, through any slot, is a hit.
constexpr uint32_t kTextureMemoEntries = 512;
TextureSlotMemo g_texture_slot_memo[kTextureMemoEntries];

// sa2-nx (2026-10-20, sa2_renut_chain_reuse): 'slot_fx' (32 entries, or null) receives each bound slot's effects (its
// memo entry's on a hit, the bind's otherwise) and 'all_memo' whether every bound slot went through the memo (a
// texture header to key it), for the chain's reuse of the next draw.
void FillTextures(const GuestDevice &device, u32 fetch_mask, SharedConstants &sc, BindEffects *slot_fx = nullptr,
                  bool *all_memo = nullptr) {
  if (all_memo)
    *all_memo = banjo::renut_gpu::CpuMemosOn();
  // sa2-nx (2026-10-20, sa2_renut_texture_defaults_copy): the null words as one copy of a block made once (160 stores a
  // draw before).
  struct DefaultWords {
    u32 indices[3][32];
    u32 samplers[32];
  };
  static const DefaultWords kDefaults = [] {
    DefaultWords w{};
    for (u32 r = 0; r < 32; ++r) {
      w.indices[0][r] = kNullTextureSlot;
      w.indices[1][r] = kNullTexture3DSlot;
      w.indices[2][r] = kNullTextureCubeSlot;
      w.samplers[r] = SamplerWord(kLinearClampSamplerSlot);
    }
    return w;
  }();
  static const bool defaults_copy = REXCVAR_GET(sa2_renut_texture_defaults_copy);
  if (defaults_copy) {
    std::memcpy(sc.texture_indices, kDefaults.indices, sizeof(sc.texture_indices));
    std::memcpy(sc.sampler_indices, kDefaults.samplers, sizeof(sc.sampler_indices));
  } else {
    for (u32 r = 0; r < 32; ++r) {
      sc.texture_indices[0][r] = kNullTextureSlot;
      sc.texture_indices[1][r] = kNullTexture3DSlot;
      sc.texture_indices[2][r] = kNullTextureCubeSlot;
      sc.sampler_indices[r] = SamplerWord(kLinearClampSamplerSlot);
    }
  }
  const d3d::Device &dev = device.dev();
  for (u32 r = 0; r < d3d::kTextureFetchSlots; ++r) {
    if (!(fetch_mask & (1u << r)))
      continue;
    const TextureFetch fetch = DeviceTextureFetch(dev, r);
    const u32 texture_va = device.U32(d3d::kDeviceOffsetTextures + r * 4);
    const auto *header = texture_va ? GuestPtr<const d3d::BaseTexture>(texture_va) : nullptr;
    TextureSlotMemo &m =
        g_texture_slot_memo[((texture_va ^ (fetch.dword[0] * 0x85EBCA6Bu) ^ fetch.dword[1]) * 0x9E3779B1u) >> 23];
    const bool memo = banjo::renut_gpu::CpuMemosOn() && header;
    if (all_memo && !memo)
      *all_memo = false;
    if (memo && m.frame == TextureFrameLocked() && m.epoch == g_texture_epoch && m.va == texture_va &&
        (!m.fx.depth_source || m.depth_epoch == g_depth_epoch) &&
        std::memcmp(m.device, fetch.dword, sizeof(m.device)) == 0 &&
        std::memcmp(m.header, &header->Format, sizeof(m.header)) == 0 &&
        !(m.fx.handover_surface && g_handover_targets && g_handover_targets->depth == m.fx.handover_surface)) {
      for (HostTexture *t : m.fx.transition)
        if (t)
          TransitionTextureLocked(t, plume::RenderTextureLayout::SHADER_READ);
      // banjo-nx (2026-10-19, banjo_deko3d_depth_handover): the target image it was sampled through.
      if (m.fx.handover_surface)
        TransitionSurfaceLocked(m.fx.handover_surface, plume::RenderTextureLayout::SHADER_READ);
      if (m.fx.handover_image)
        TransitionDepthImageLocked(*m.fx.handover_image, plume::RenderTextureLayout::SHADER_READ);
      if (m.fx.float_sampled)
        banjo::renut_gpu::CountDepthFloatSampled();
      if (m.fx.words_viewed)
        banjo::renut_gpu::CountDepthWords(true);
      sc.texture_indices[m.dim][r] = m.index;
      sc.sampler_indices[r] = m.sampler;
      if (slot_fx)
        slot_fx[r] = m.fx;
      continue;
    }
    BindEffects fx;
    const TextureBinding binding = BindTextureLocked(texture_va, fetch, &fx);
    if (slot_fx)
      slot_fx[r] = fx;
    const uint32_t dim = binding.dimension == TextureDimension::Tex3D  ? 1
                         : binding.dimension == TextureDimension::Cube ? 2
                                                                       : 0;
    // Stacked textures go through the 3D fetches, flagged (tfetchStacked in
    // shader_common.h), with the fetch constant's filter between layers and
    // its Z address mode.
    u32 index = binding.slot | binding.scale.code << kResolutionScaleShift;
    if (binding.stacked) {
      index |= kStackedTextureBit;
      if (fetch.VolumeMagFilter() != TextureFilter::Point ||
          fetch.VolumeMinFilter() != TextureFilter::Point) {
        index |= kStackedLayerLinearBit;
      }
      if (fetch.Address(2) == TextureAddress::Wrap)
        index |= kStackedLayerWrapBit;
    }
    sc.texture_indices[dim][r] = index;
    sc.sampler_indices[r] = SamplerWord(SamplerSlotLocked(fetch, binding.lod_scale));
    if (memo) {
      m.va = texture_va;
      std::memcpy(m.device, fetch.dword, sizeof(m.device));
      std::memcpy(m.header, &header->Format, sizeof(m.header));
      m.frame = TextureFrameLocked();
      m.epoch = g_texture_epoch;  // after the bind (its own bumps)
      m.depth_epoch = g_depth_epoch;
      m.dim = dim;
      m.index = index;
      m.sampler = sc.sampler_indices[r];
      m.fx = fx;
    }
  }
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

struct Geometry {
  bool indexed = false;
  u32 index_count = 0;
  plume::RenderIndexBufferView index_view;  // indices from first_vertex
  u32 vertex_count = 0;  // non-indexed host draw count
  u32 first_vertex = 0;  // first guest vertex converted
  u32 vertex_range = 0;  // vertices converted per stream
  // Tessellated draws: one instance per patch, the streams per instance.
  u32 instance_count = 1;
  bool per_instance = false;
};

// Reads 'count' guest indices into g_indices.
bool ReadIndices(const u8 *src, u32 count, bool is32, u32 endian) {
  if (!src || count > kMaxDrawIndices)
    return false;
  g_indices.resize(count);
  if (is32) {
    SwapEndian(endian, src, g_indices.data(), size_t(count) * 4);
  } else {
    const bool swap = endian == 1 || endian == 2;
    for (u32 i = 0; i < count; ++i) {
      uint16_t v;
      std::memcpy(&v, src + i * 2, 2);
      g_indices[i] = swap ? uint16_t((v >> 8) | (v << 8)) : v;
    }
  }
  return true;
}

// Rewrites g_indices for fans and quads as triangle lists.
void ExpandIndices(u32 primitive) {
  // banjo-nx (2026-10-19, render-thread proposal 20): a scratch vector kept between draws (swapped with g_indices, so
  // both keep their capacity), not a new one per fan or quad draw.
  static std::vector<u32> out;
  if (primitive == kTriangleFan) {
    const size_t n = g_indices.size() > 2 ? g_indices.size() - 2 : 0;
    out.resize(n * 3);
    for (size_t i = 2, o = 0; i < g_indices.size(); ++i, o += 3) {
      out[o] = g_indices[0];
      out[o + 1] = g_indices[i - 1];
      out[o + 2] = g_indices[i];
    }
    g_indices.swap(out);
  } else if (primitive == kQuadList) {
    const size_t n = g_indices.size() / 4;
    out.resize(n * 6);
    for (size_t i = 0, o = 0; i + 3 < g_indices.size(); i += 4, o += 6) {
      const u32 a = g_indices[i], b = g_indices[i + 1], c = g_indices[i + 2], d = g_indices[i + 3];
      out[o] = a;
      out[o + 1] = b;
      out[o + 2] = c;
      out[o + 3] = a;
      out[o + 4] = c;
      out[o + 5] = d;
    }
    g_indices.swap(out);
  }
}

// Guest index ranges kept across frames, rebased and in the host format, like
// the vertex streams (g_stream_cache): most draws read index buffers the game
// never rewrites. A range that hashes the same in kStableFlushes frames in a
// row gets a persistent copy, which draws use while the range still hashes the
// same, checked like the streams' (CacheCheckDue, off the render thread), and
// a range rewritten kMaxCacheDrops times is never cached again.
struct IndexKey {
  const u8 *src;
  u32 count, primitive, endian;
  bool is32;
  bool operator==(const IndexKey &) const = default;
};
struct IndexKeyHash {
  size_t operator()(const IndexKey &k) const {
    uint64_t h = uint64_t(reinterpret_cast<uintptr_t>(k.src)) * 0x9E3779B97F4A7C15ull;
    h ^= (uint64_t(k.count) << 16 | k.primitive << 3 | k.endian << 1 | u32(k.is32)) +
         0x632BE59BD9B4E019ull + (h << 6) + (h >> 2);
    return size_t(h);
  }
};
struct CachedIndices {
  uint64_t hash = 0;
  u32 stable = 0;  // frames in a row the range hashed the same
  u32 drops = 0;   // copies dropped because the guest range changed
  bool dynamic = false;
  PoolSpan copy;
  u32 bytes = 0, index_count = 0, lo = 0, max_index = 0;
  bool wide = false;
  uint64_t last_used = 0;               // g_stream_cache_frame
  uint64_t checked_frame = UINT64_MAX;  // frame the range was last hashed or checked in
  uint64_t proven_flush = UINT64_MAX;   // banjo-nx: g_flush_serial the copy was last hashed equal in
  uint64_t exact_frame = UINT64_MAX;    // banjo-nx: g_stream_cache_frame of its last exact check (repeat count)
};
using IndexCache = std::unordered_map<IndexKey, CachedIndices, IndexKeyHash>;
IndexCache g_index_cache;
EntryMemo<IndexCache, IndexKeyHash, 11> g_index_memo;  // banjo-nx: 2,048 slots of a key each (proposal 9)

uint64_t HashIndices(const IndexKey &key) {
  return banjo::renut_gpu::ContentHash64(key.src, size_t(key.count) * (key.is32 ? 4 : 2));  // banjo-nx
}

ContentVerifier<IndexKey, HashIndices> g_index_verifier;

void DropCachedIndices(CachedIndices &c) {
  if (!c.copy)
    return;
  g_stream_cache_bytes -= c.bytes;
  g_copy_pool.Free(c.copy);
  c.copy = {};
  c.bytes = 0;
}

// The draw's indices from the cache, if it has a copy of the guest range.
bool CachedIndexDraw(const IndexKey &key, const DrawCall &call, Geometry &g) {
  auto *entry = g_index_memo.Get(g_index_cache, key, false);
  if (!entry || !entry->second.copy)
    return false;
  CachedIndices &c = entry->second;
  c.last_used = g_stream_cache_frame;
  if (banjo::renut_gpu::ExactStreamsOn()) {
    // banjo-nx (2026-10-17): banjo_renut_exact_streams, as for streams. A changed range is dropped and the draw reads
    // it itself (TrackIndices hashes it again).
    if (c.proven_flush != g_flush_serial &&
        CardsProve(key.src, size_t(key.count) * (key.is32 ? 4 : 2), c.proven_flush, c.hash)) {
      c.proven_flush = g_flush_serial;  // banjo-nx: banjo_renut_store_cards
    } else if (c.proven_flush != g_flush_serial) {
      const bool same = ExactHash(key.src, size_t(key.count) * (key.is32 ? 4 : 2)) == c.hash;
      g_card_diag[7] += same && c.proven_flush != UINT64_MAX && banjo::renut_gpu::StoreCardsVerifyOn();
      banjo::renut_gpu::CountExactCheck(uint64_t(key.count) * (key.is32 ? 4 : 2), !same,
                                        c.exact_frame == g_stream_cache_frame);
      c.exact_frame = g_stream_cache_frame;
      if (!same) {
        DropCachedIndices(c);
        c.stable = 0;
        c.checked_frame = UINT64_MAX;
        c.dynamic = ++c.drops >= kMaxCacheDrops;
        return false;
      }
      c.proven_flush = g_flush_serial;
    }
  } else if (CacheCheckDue(c.checked_frame)) {
    c.checked_frame = g_stream_cache_frame;
    g_index_verifier.Submit(key, c.hash);
  }
  g.first_vertex = u32(int64_t(call.base_vertex) + c.lo);
  g.vertex_range = c.max_index + 1;
  g.indexed = true;
  g.index_count = c.index_count;
  g.index_view = plume::RenderIndexBufferView(
      c.copy.Ref(), c.bytes,
      c.wide ? plume::RenderFormat::R32_UINT : plume::RenderFormat::R16_UINT);
  return true;
}

// After a draw read the range into g_indices: count how long it has held, and
// give a range that has held long enough its persistent copy.
void TrackIndices(const IndexKey &key, u32 lo, u32 max_index) {
  CachedIndices &c = g_index_memo.Get(g_index_cache, key, true)->second;
  c.last_used = g_stream_cache_frame;
  if (c.dynamic || c.copy || c.checked_frame == g_stream_cache_frame)
    return;
  c.checked_frame = g_stream_cache_frame;
  const uint64_t hash = HashIndices(key);
  c.stable = hash == c.hash ? c.stable + 1 : 0;
  c.hash = hash;
  const bool wide = max_index > 0xFFFE;
  const u32 bytes = u32(g_indices.size()) * (wide ? 4 : 2);
  if (c.stable + 1 < kStableFlushes ||
      g_stream_cache_bytes + bytes > banjo::renut_gpu::StreamCacheMaxBytes(kStreamCacheMaxBytes))  // banjo-nx
    return;
  const PoolSpan copy = g_copy_pool.Allocate(bytes);
  if (!copy)
    return;
  if (wide) {
    auto *out = reinterpret_cast<u32 *>(copy.cpu);
    for (size_t i = 0; i < g_indices.size(); ++i)
      out[i] = g_indices[i] - lo;
  } else {
    auto *out = reinterpret_cast<uint16_t *>(copy.cpu);
    for (size_t i = 0; i < g_indices.size(); ++i)
      out[i] = uint16_t(g_indices[i] - lo);
  }
  c.copy = copy;
  c.bytes = bytes;
  c.index_count = u32(g_indices.size());
  c.lo = lo;
  c.max_index = max_index;
  c.wide = wide;
  g_stream_cache_bytes += bytes;
}

bool PrepareGeometry(const GuestDevice &device, const DrawCall &call, Geometry &g) {
  const bool expand = call.primitive == kTriangleFan || call.primitive == kQuadList;
  static uint32_t rect_logged = 0;
  if (call.primitive == kRectangleList && !State().rectangle_list_gs && LogLimited(rect_logged, 1))
    RGPU_WARN("Rectangle lists are drawn as triangle lists");

  u32 lo = 0, max_index = 0;  // host indices are g_indices - lo, the largest max_index
  IndexKey index_key{};
  bool cacheable_indices = false;
  if (call.indexed) {
    const u8 *src = nullptr;
    bool is32 = false;
    u32 endian = 0;
    if (call.up_indices) {
      is32 = call.up_indices_32;
      endian = is32 ? 2 : 1;
      src = call.up_indices + size_t(call.start_index) * (is32 ? 4 : 2);
    } else {
      const auto *ib = GuestPtr<const d3d::IndexBuffer>(device.U32(d3d::kDeviceOffsetIndexBuffer));
      if (!ib)
        return false;
      const u32 common = ib->resource.Common;
      is32 = (common & d3d::kIndexBufferCommon32Bit) != 0;
      endian = d3d::IndexBufferEndian(common);
      const u8 *base = AddressPtr(ib->Address);
      if (!base)
        return false;
      src = base + size_t(call.start_index) * (is32 ? 4 : 2);
      // Draw*IndexedPrimitiveUP indices are the game's own memory, only valid
      // during the call; these are an index buffer's.
      index_key = IndexKey{src, call.count, call.primitive, endian, is32};
      cacheable_indices = call.count <= kMaxDrawIndices;
      if (cacheable_indices && CachedIndexDraw(index_key, call, g))
        return true;
    }
    if (!ReadIndices(src, call.count, is32, endian))
      return false;
    if (expand)
      ExpandIndices(call.primitive);
    if (g_indices.empty())
      return false;
    lo = UINT32_MAX;
    u32 hi = 0;
    for (u32 index : g_indices) {
      lo = std::min(lo, index);
      hi = std::max(hi, index);
    }
    if (hi - lo >= kMaxDrawVertices)
      return false;
    g.first_vertex = u32(int64_t(call.base_vertex) + lo);
    g.vertex_range = hi - lo + 1;
    max_index = hi - lo;
  } else {
    if (!call.count || call.count > kMaxDrawVertices)
      return false;
    g.first_vertex = call.start_vertex;
    g.vertex_range = call.count;
    if (expand) {
      g_indices.resize(call.count);
      for (u32 i = 0; i < call.count; ++i)
        g_indices[i] = i;
      ExpandIndices(call.primitive);
      if (g_indices.empty())
        return false;
      max_index = call.count - 1;
    } else {
      g.vertex_count = call.count;
    }
  }

  if (call.indexed || expand) {
    // The host indices are rebased to the range rather than drawn with a base
    // vertex of -lo: D3D12's SV_VertexID leaves the base vertex out (Vulkan's
    // vertex index includes it), and vertex shaders that fetch by address
    // take the guest index from the vertex ID (renutVertexRegisters).
    // 16-bit when every index fits, like the guest's own format mostly is.
    const bool wide = max_index > 0xFFFE;
    const u32 bytes = u32(g_indices.size()) * (wide ? 4 : 2);
    UploadAllocation alloc = Video::UploadsLocked().Allocate(bytes, 16);
    if (!alloc)
      return false;
    if (wide) {
      auto *out = static_cast<u32 *>(alloc.cpu);
      for (size_t i = 0; i < g_indices.size(); ++i)
        out[i] = g_indices[i] - lo;
    } else {
      auto *out = static_cast<uint16_t *>(alloc.cpu);
      for (size_t i = 0; i < g_indices.size(); ++i)
        out[i] = uint16_t(g_indices[i] - lo);
    }
    g.indexed = true;
    g.index_count = u32(g_indices.size());
    g.index_view = plume::RenderIndexBufferView(
        alloc.ref, bytes, wide ? plume::RenderFormat::R32_UINT : plume::RenderFormat::R16_UINT);
    if (cacheable_indices)
      TrackIndices(index_key, lo, max_index);
  }
  return true;
}

// banjo-nx (banjo_gpu_tess_grid, package 27): the index list of a tessellation grid, made once per kind and level: the
// triangles renutVertexRegisters' kinds 1 and 2 place, in their order, with their corners, each corner as its grid
// vertex's number (kinds 3 and 4 there). Render thread; the buffers live as long as the device.
struct TessellationGrid {
  std::unique_ptr<plume::RenderBuffer> buffer;
  u32 index_count = 0;
  bool wide = false;
};
std::map<u32, TessellationGrid> g_tess_grids;  // key: triangles << 8 | level

const TessellationGrid *TessellationGridFor(bool triangles, u32 level) {
  TessellationGrid &grid = g_tess_grids[(triangles ? 1u : 0u) << 8 | level];
  if (grid.buffer)
    return &grid;
  std::vector<u32> indices;
  if (triangles) {
    auto number = [](u32 a, u32 b) { return a * (a + 1) / 2 + b; };
    for (u32 t = 0; t < level * level; ++t) {
      u32 row = u32(std::sqrt(double(t)));
      while (row * row > t)
        --row;
      while ((row + 1) * (row + 1) <= t)
        ++row;
      const u32 k = t - row * row, b = k / 2;
      if ((k & 1) == 0) {
        indices.insert(indices.end(), {number(row, b), number(row + 1, b), number(row + 1, b + 1)});
      } else {
        indices.insert(indices.end(), {number(row, b), number(row + 1, b + 1), number(row, b + 1)});
      }
    }
  } else {
    auto number = [level](u32 u, u32 v) { return v * (level + 1) + u; };
    for (u32 t = 0; t < 2 * level * level; ++t) {
      const u32 cell = t / 2, u = cell % level, v = cell / level;
      if ((t & 1) == 0)
        indices.insert(indices.end(), {number(u, v), number(u + 1, v), number(u + 1, v + 1)});
      else
        indices.insert(indices.end(), {number(u, v), number(u + 1, v + 1), number(u, v + 1)});
    }
  }
  const u32 points = triangles ? (level + 1) * (level + 2) / 2 : (level + 1) * (level + 1);
  grid.wide = points > 0xFFFF;
  const size_t bytes = indices.size() * (grid.wide ? 4 : 2);
  auto buffer = State().device->createBuffer(
      plume::RenderBufferDesc::UploadBuffer(bytes, plume::RenderBufferFlag::INDEX));
  void *mapped = buffer ? buffer->map() : nullptr;
  if (!mapped)
    return nullptr;
  if (grid.wide) {
    std::memcpy(mapped, indices.data(), bytes);
  } else {
    auto *out = static_cast<uint16_t *>(mapped);
    for (size_t i = 0; i < indices.size(); ++i)
      out[i] = uint16_t(indices[i]);
  }
  buffer->unmap();
  buffer->setName(fmt::format("banjo-nx tessellation grid {} {}", triangles ? "triangles" : "quads", level));
  grid.buffer = std::move(buffer);
  grid.index_count = u32(indices.size());
  return &grid;
}

// Tessellated draws (DrawCall::tessellation): the patches become instances of a
// grid the vertex shader places (renutVertexRegisters in shader_common.h), at
// one level for the whole draw, the highest any patch edge asks for.
bool PrepareTessellation(const GuestDevice &device, const DrawCall &call, Geometry &g,
                         u32 tessellation[4]) {
  const bool triangles = call.primitive == kTrianglePatch;
  if (!triangles && call.primitive != kQuadPatch) {
    static uint32_t logged = 0;
    if (LogLimited(logged, 1))
      RGPU_WARN("Tessellated primitive {} not drawn", call.primitive);
    return false;
  }
  // Segments per edge are the tessellation level plus one.
  const float min_level = std::max(device.F32(d3d::kDeviceOffsetMinTessellationLevel), 0.0f) + 1.0f;
  const float max_level =
      std::max(device.F32(d3d::kDeviceOffsetMaxTessellationLevel) + 1.0f, min_level);
  float level = max_level;
  // The count is the patch count. (N&B's water: each section's patch stream is
  // sized 36 bytes x the count it draws, and its factor pass exports one
  // patch's edge factors per invocation.)
  const u32 patches = call.count;
  if ((device.U32(d3d::kDeviceOffsetTessellationControl) & 3) == 2) {
    // Adaptive: the index buffer holds each patch's edge factors (floats), from
    // the start index on. When a memexport draw in flight writes them, these
    // are from a frame or two before (DrawNeedsExportFlush does not wait for
    // them).
    const u32 edges = triangles ? 3 : 4;
    // Without them, the maximum level.
    const auto *ib = GuestPtr<const d3d::IndexBuffer>(device.U32(d3d::kDeviceOffsetIndexBuffer));
    const u8 *base = ib ? AddressPtr(ib->Address) : nullptr;
    if (base && ReadIndices(base + size_t(call.start_index) * 4, patches * edges, true,
                            d3d::IndexBufferEndian(ib->resource.Common))) {
      float highest = 0.0f;
      for (u32 bits : g_indices) {
        float factor;
        std::memcpy(&factor, &bits, 4);
        if (factor > highest && factor < 1024.0f)
          highest = factor;
      }
      level = std::clamp(highest + 1.0f, min_level, max_level);
    }
  }
  if (!patches || patches > kMaxDrawVertices)
    return false;
  constexpr u32 kMaxTessellatedVertices = 1u << 22;
  u32 segments = std::clamp(u32(std::ceil(level)), 1u, 64u);
  const u32 per_segment = triangles ? 3 : 6;  // vertices per segment^2
  while (segments > 1 &&
         uint64_t(patches) * segments * segments * per_segment > kMaxTessellatedVertices)
    --segments;

  g.vertex_count = segments * segments * per_segment;
  g.instance_count = patches;
  g.per_instance = true;
  g.first_vertex = u32(call.base_vertex);
  g.vertex_range = patches;
  tessellation[0] = triangles ? 1 : 2;
  tessellation[1] = segments;
  tessellation[2] = u32(call.base_vertex);
  // banjo-nx (banjo_gpu_tess_grid): the same triangles, indexed by grid vertex.
  if (banjo::renut_gpu::TessGridOn()) {
    if (const TessellationGrid *grid = TessellationGridFor(triangles, segments)) {
      g.indexed = true;
      g.index_count = grid->index_count;
      g.index_view = plume::RenderIndexBufferView(plume::RenderBufferReference(grid->buffer.get()),
                                                  grid->index_count * (grid->wide ? 4 : 2),
                                                  grid->wide ? plume::RenderFormat::R32_UINT
                                                             : plume::RenderFormat::R16_UINT);
      tessellation[0] = triangles ? 3 : 4;
      banjo::renut_gpu::CountTessGrid(uint64_t(patches) * g.vertex_count,
                                      uint64_t(patches) * (triangles ? (segments + 1) * (segments + 2) / 2
                                                                     : (segments + 1) * (segments + 1)));
    }
  }
  return true;
}

struct StreamSource {
  const u8 *data = nullptr;  // guest vertex 0 of the stream
  u32 stride = 0;
  u32 endian = 0;
  u32 vertex_count = 0;      // vertices available from data
  u32 cpu_address = 0;       // guest VA of data, when read through the vertex buffer
};

// 'shader_strides': the strides in the vertex shader's microcode when the draw
// fetches as the shader was bound (or baked), with no declaration set on the
// device. The GPU uses those, whatever SetStreamSource was given; D3D patches
// the stream strides in only when it binds the shader to a device
// declaration.
StreamSource GetStreamSource(const GuestDevice &device, const DrawCall &call,
                             const ShaderRecord &vs, const u32 *shader_strides, u32 stream) {
  StreamSource src;
  if (stream == 0 && call.up_vertices) {
    src.data = call.up_vertices;
    src.stride = call.up_stride;
    src.endian = kUpEndian;
    src.vertex_count = call.up_vertex_count;
    return src;
  }
  const u32 fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(stream);
  const u32 w0 = device.VertexFetchWord(fetch_index, 0);
  const u32 w1 = device.VertexFetchWord(fetch_index, 1);
  if (shader_strides)
    src.stride = shader_strides[stream];
  if (!src.stride)
    src.stride = u32(device.U8(d3d::kDeviceOffsetStreamStrides + stream)) * 4;
  if (!src.stride)
    src.stride = vs.bound_strides[stream];  // SetStreamSource(..., 0): the Bind stride
  if (!src.stride)
    src.stride = vs.fetch_strides[stream];  // or the one the shader's vfetch encodes
  src.endian = w1 & 3;
  const u32 size = w1 & 0x3FFFFFCu;
  const u32 gpu_address = w0 & ~3u;
  src.data = gpu_address ? GuestPhysicalPtr<const u8>(gpu_address) : nullptr;
  // Read through the vertex buffer's CPU address, where the game wrote it,
  // when the fetch constant is the one SetStreamSource made from that buffer
  // (its GPU address plus the stream offset).
  if (const auto *vb = GuestPtr<const d3d::VertexBuffer>(
          device.U32(d3d::kDeviceOffsetStreamSources + stream * 4))) {
    const u32 cpu_base = u32(vb->Format.dword[0]) & ~3u;
    const u32 gpu_base = d3d::CpuToGpuAddress(cpu_base);
    if (cpu_base >= 0x20000000u && gpu_address >= gpu_base &&
        gpu_address - gpu_base < (u32(vb->Format.dword[1]) & 0x3FFFFFCu)) {
      src.cpu_address = cpu_base + (gpu_address - gpu_base);
      src.data = GuestPtr<const u8>(src.cpu_address);
    }
  }
  src.vertex_count = src.stride ? size / src.stride : (size ? 1 : 0);
  return src;
}

// Host vertices of the draw's streams built on the GPU from export windows
// (ConvertExportedStreamsLocked), from its first vertex; 'bytes' 0 for the
// others.
struct GpuStreams {
  plume::RenderBufferReference ref[kMaxVertexStreams];
  u32 bytes[kMaxVertexStreams] = {};
};

// mk9-nx (2026-10-21): a test for the fights' huge flat-coloured triangles on the Switch (and Ryujinx; the PC is
// clean): every converted stream's float positions are checked, and a draw with a NaN, an infinity or a coordinate
// beyond 1e5 is logged with its shaders. Off by default (it reads back every converted position).
REXCVAR_DEFINE_BOOL(mk9_diag_bad_vertices, false, "MK9/Renderer",
                    "Test: log draws whose converted float positions hold a NaN, an infinity or a coordinate beyond "
                    "1e5 (shaders, primitive, vertex, value; at most 64 lines)");

// The guest word behind each logged bad position, read again at the next flushes: a word that becomes a sane float
// later was copied before the game had finished writing it (not a decode error).
struct DiagRecheck {
  const u8 *guest = nullptr;
  u32 serial = 0, first_bits = 0, endian = 0;
};
DiagRecheck g_diag_rechecks[16];
u32 g_diag_serial = 0;

void DiagRecheckPositions() {
  ++g_diag_serial;
  for (DiagRecheck &r : g_diag_rechecks) {
    if (!r.guest || g_diag_serial - r.serial < 2)
      continue;
    u32 bits = *reinterpret_cast<const be_u32 *>(r.guest);
    float f;
    std::memcpy(&f, &bits, 4);
    RGPU_WARN("[badvert] recheck {} flushes later: word {:08X} -> {:08X} ({}){}", g_diag_serial - r.serial,
              r.first_bits, bits, f, bits != r.first_bits ? ": changed after the copy" : ": unchanged");
    r.guest = nullptr;
  }
}

void DiagCheckPositions(const HostVertexStream &stream, const u8 *host, u32 count, uint64_t vs_hash, uint64_t ps_hash,
                        u32 primitive, const u8 *guest, const char *where, u32 guest_stride = 0) {
  static uint32_t logged = 0;
  if (logged >= 64)
    return;
  for (const HostVertexElement &e : stream.elements) {
    if (e.usage != 0)  // D3DDECLUSAGE_POSITION
      continue;
    const u32 n = e.host_format == plume::RenderFormat::R32G32B32A32_FLOAT ? 4
                  : e.host_format == plume::RenderFormat::R32G32B32_FLOAT  ? 3
                  : e.host_format == plume::RenderFormat::R32G32_FLOAT     ? 2
                                                                            : 0;
    if (!n)
      continue;
    for (u32 v = 0; v < count; ++v) {
      float f[4] = {};
      std::memcpy(f, host + size_t(v) * stream.host_stride + e.host_offset, n * 4);
      for (u32 c = 0; c < n; ++c) {
        if (std::isfinite(f[c]) && std::fabs(f[c]) <= 1e5f)
          continue;
        ++logged;
        RGPU_WARN("[badvert] {}: vs {:016X} ps {:016X} prim {} vertex {} of {} position.{} = {} (physical {:08X}, "
                  "format {}, usage index {})",
                  where, vs_hash, ps_hash, primitive, v, count, "xyzw"[c], f[c],
                  guest ? u32(guest - detail::GuestPhysicalBase()) : 0,
                  int(e.host_format), e.usage_index);
        if (guest && guest_stride && e.conversion == ElementConversion::Copy32) {
          for (DiagRecheck &r : g_diag_rechecks) {
            if (r.guest)
              continue;
            r.guest = guest + size_t(v) * guest_stride + e.guest_offset + c * 4;
            r.serial = g_diag_serial;
            r.first_bits = *reinterpret_cast<const be_u32 *>(r.guest);
            break;
          }
        }
        return;
      }
    }
  }
}

// banjo-nx (2026-10-18, the deko3d water): banjo_renut_diag_stream_dump_vs names a vertex shader (hex hash); its draws'
// streams are logged (first 16): the guest source, the first two vertices as the guest has them and as converted for
// the host, and whether a cached copy is bound.
bool StreamDumpLog(const ShaderRecord &vs, u32 stream, const StreamSource &src, const HostVertexStream &hs,
                   const u8 *guest, u32 first, u32 count, const DrawCall &call) {
  static const uint64_t want = [] {
    const std::string v = REXCVAR_GET(banjo_renut_diag_stream_dump_vs);
    return v.empty() ? 0ull : uint64_t(std::strtoull(v.c_str(), nullptr, 16));
  }();
  if (!want || vs.hash != want)
    return false;
  static uint32_t seen = 0;
  if (seen++ >= 16 && seen % 64 != 0)  // the first 16, then every 64th
    return false;
  std::string guest_words, host_words;
  if (guest && count) {
    const u32 n = std::min<u32>(src.stride / 4, 16);
    for (u32 v = 0; v < std::min<u32>(count, 2); ++v) {
      for (u32 i = 0; i < n; ++i) {
        u32 w;
        std::memcpy(&w, guest + size_t(v) * src.stride + i * 4, 4);
        guest_words += fmt::format("{:08X} ", w);
      }
      guest_words += "| ";
    }
    std::vector<u8> host(size_t(hs.host_stride) * 2 + 16);
    ConvertVertices(hs, guest, src.stride, std::min<u32>(count, 2), src.endian, host.data());
    for (u32 i = 0; i < std::min<u32>(hs.host_stride * 2 / 4, 32); ++i) {
      u32 w;
      std::memcpy(&w, host.data() + i * 4, 4);
      host_words += fmt::format("{:08X} ", w);
    }
  }
  RGPU_INFO("[streamdump] vs {:016X} stream {} cpu {:08X} stride {} host stride {} endian {} first {} count {} prim {} "
            "draw count {}: guest {}host {}",
            vs.hash, stream, src.cpu_address, src.stride, hs.host_stride, src.endian, first, count, call.primitive,
            call.count, guest_words, host_words);
  return true;
}

bool BindStreams(const GuestDevice &device, const DrawCall &call, const ShaderRecord &vs,
                 const u32 *shader_strides, const HostVertexLayout &layout, uint64_t layout_hash,
                 const Geometry &g, const GpuStreams &gpu_streams) {
  plume::RenderCommandList *cmd = State().command_list;
  // sa2-nx (2026-10-20, sa2_renut_stream_mask): the used streams by their mask, in the same order (16 checks a draw
  // before, 1.3 % of the render thread in the second console run).
  static const bool stream_mask = REXCVAR_GET(sa2_renut_stream_mask);
  for (u32 left = stream_mask ? layout.used_mask : (1u << kMaxVertexStreams) - 1u; left; left &= left - 1) {
    const u32 stream = u32(std::countr_zero(left));
    const HostVertexStream &hs = layout.streams[stream];
    if (!hs.used)
      continue;
    if (gpu_streams.bytes[stream]) {
      {
        static const uint64_t want = [] {
          const std::string v = REXCVAR_GET(banjo_renut_diag_stream_dump_vs);
          return v.empty() ? 0ull : uint64_t(std::strtoull(v.c_str(), nullptr, 16));
        }();
        static uint32_t seen = 0;
        if (want && vs.hash == want && (seen++ < 16 || seen % 64 == 0))
          RGPU_INFO("[streamdump] vs {:016X} stream {}: the GPU-converted exported stream ({} bytes)", vs.hash, stream,
                    gpu_streams.bytes[stream]);
      }
      const plume::RenderVertexBufferView view(gpu_streams.ref[stream], gpu_streams.bytes[stream]);
      const plume::RenderInputSlot slot(
          stream, hs.host_stride,
          g.per_instance ? plume::RenderInputSlotClassification::PER_INSTANCE_DATA
                         : plume::RenderInputSlotClassification::PER_VERTEX_DATA);
      SetVertexBuffer(cmd, view, slot);
      continue;
    }
    const StreamSource src = GetStreamSource(device, call, vs, shader_strides, stream);
    // A zero stride repeats one vertex for every index.
    const bool constant = src.stride == 0;
    const u32 first = constant ? 0 : g.first_vertex;
    u32 count = constant ? 1 : g.vertex_range;
    if (first < src.vertex_count)
      count = std::min(count, src.vertex_count - first);
    else
      count = src.data ? 0 : count;
    const u32 bytes = std::max<u32>(count, 1) * hs.host_stride;
    const bool fills = src.data && count;
    const u8 *guest = fills ? src.data + size_t(first) * src.stride : nullptr;
    const bool queued = fills && !(stream == 0 && call.up_vertices);
    const StreamKey key{guest, layout_hash, stream, src.stride, count, src.endian};
    // sa2-nx (2026-10-20, sa2_renut_lean_diagnostics): the dump's check once, not a call per stream.
    static const bool stream_dump = !REXCVAR_GET(sa2_renut_lean_diagnostics) ||
                                    !REXCVAR_GET(banjo_renut_diag_stream_dump_vs).empty();
    const bool dumped = stream_dump && StreamDumpLog(vs, stream, src, hs, guest, first, count, call);
    CachedStream *cached = nullptr;
    if (queued && src.stride) {
      cached = &g_stream_memo.Get(g_stream_cache, key, true)->second;
      cached->last_used = g_stream_cache_frame;
      if (cached->copy && banjo::renut_gpu::ExactStreamsOn()) {
        // banjo-nx (2026-10-17): banjo_renut_exact_streams. The copy's guest bytes hashed whole on its first use
        // after each kick; changed, it is dropped and the stream converted below, as an uncached one.
        if (cached->proven_flush != g_flush_serial &&
            CardsProve(key.guest, size_t(key.count) * key.stride, cached->proven_flush, cached->hash)) {
          cached->proven_flush = g_flush_serial;  // banjo-nx: banjo_renut_store_cards
        } else if (cached->proven_flush != g_flush_serial) {
          const bool same = ExactHash(key.guest, size_t(key.count) * key.stride) == cached->hash;
          g_card_diag[7] += same && cached->proven_flush != UINT64_MAX && banjo::renut_gpu::StoreCardsVerifyOn();
          banjo::renut_gpu::CountExactCheck(uint64_t(key.count) * key.stride, !same,
                                            cached->exact_frame == g_stream_cache_frame);
          cached->exact_frame = g_stream_cache_frame;
          if (same) {
            cached->proven_flush = g_flush_serial;
          } else {
            DropCachedCopy(*cached);
            cached->stable = 0;
            cached->checked_frame = UINT64_MAX;
            cached->dynamic = ++cached->drops >= kMaxCacheDrops;
          }
        }
      } else if (cached->copy && CacheCheckDue(cached->checked_frame)) {
        cached->checked_frame = g_stream_cache_frame;
        g_stream_verifier.Submit(key, cached->hash);
      }
      if (cached->copy) {
        if (dumped)
          RGPU_INFO("[streamdump] stream {}: bound the cached copy ({} bytes)", stream, cached->bytes);
        const plume::RenderVertexBufferView view(cached->copy.Ref(), cached->bytes);
        const plume::RenderInputSlot slot(
            stream, hs.host_stride,
            g.per_instance ? plume::RenderInputSlotClassification::PER_INSTANCE_DATA
                           : plume::RenderInputSlotClassification::PER_VERTEX_DATA);
        SetVertexBuffer(cmd, view, slot);
        continue;
      }
    }
    const UploadAllocation *queued_alloc = queued ? FindQueuedStream(key) : nullptr;
    const bool reused = queued_alloc != nullptr;
    UploadAllocation alloc = reused ? *queued_alloc : Video::UploadsLocked().Allocate(bytes, 16);
    if (!alloc)
      return false;
    // Conversion writes every vertex; only an unfilled buffer needs zeros.
    if (!fills)
      std::memset(alloc.cpu, 0, bytes);
    if (fills && !reused) {
      if (call.up_vertices && stream == 0) {
        // A copy taken at the call (queued.cpp).
        ConvertVertices(hs, guest, src.stride, count, src.endian, static_cast<u8 *>(alloc.cpu));
        static const bool diag = REXCVAR_GET(mk9_diag_bad_vertices);
        if (diag)
          DiagCheckPositions(hs, static_cast<const u8 *>(alloc.cpu), count, vs.hash, 0, call.primitive,
                             nullptr, "UP");
      } else {
        PendingStreamFill fill;
        fill.stream = hs;
        fill.guest = guest;
        fill.stride = src.stride;
        fill.count = count;
        fill.endian = src.endian;
        fill.host = static_cast<u8 *>(alloc.cpu);
        fill.key = key;
        fill.cacheable = cached && !cached->dynamic;
        fill.vs_hash = vs.hash;  // mk9-nx: mk9_diag_bad_vertices
        fill.primitive = call.primitive;
        g_stream_fills.push_back(std::move(fill));
        AddQueuedStream(key, alloc);
      }
    }
    const plume::RenderVertexBufferView view(alloc.ref, bytes);
    const plume::RenderInputSlot slot(stream, constant ? 0 : hs.host_stride,
                                      g.per_instance
                                          ? plume::RenderInputSlotClassification::PER_INSTANCE_DATA
                                          : plume::RenderInputSlotClassification::PER_VERTEX_DATA);
    SetVertexBuffer(cmd, view, slot);
  }

  if (!g_bound.zero_stream) {
    g_bound.zero_stream = Video::UploadsLocked().Allocate(kZeroStreamSize, 16);
    if (!g_bound.zero_stream)
      return false;
    std::memset(g_bound.zero_stream.cpu, 0, kZeroStreamSize);
  }
  const plume::RenderVertexBufferView zero_view(g_bound.zero_stream.ref, kZeroStreamSize);
  const plume::RenderInputSlot zero_slot(kZeroStreamSlot, 0);
  SetVertexBuffer(cmd, zero_view, zero_slot);
  return true;
}

// The vertex inputs the shader reads from its declaration by an index other
// than the vertex index (vfetchBound): each described from the declaration's
// element for it. Returns the fetch constants (bit i = index 64 + i) whose
// buffers those reads need copied.
u32 DescribeBoundFetches(const GuestDevice &device, const DrawCall &call, const ShaderRecord &vs,
                         const VertexDeclarationRecord &declaration, const u32 *shader_strides,
                         SharedConstants &sc) {
  if (!vs.entry)
    return 0;
  const ShaderVertexInput *inputs = g_shaderVertexInputs + vs.entry->vertexInputOffset;
  const u32 count = std::min<u32>(vs.entry->vertexInputCount, kMaxBoundFetches);
  // banjo-nx (2026-10-19, render-thread proposal 21): whether the shader has any input fetched by another index, once
  // per shader entry (static): most have none, and their draws skip the scan of cold input records.
  struct BoundSummary {
    const ShaderCacheEntry *entry = nullptr;
    bool any = false;
  };
  static BoundSummary summaries[256];
  const uintptr_t a = reinterpret_cast<uintptr_t>(vs.entry);
  BoundSummary &summary = summaries[((a >> 3) ^ (a >> 11)) & 255];
  if (summary.entry != vs.entry) {
    summary.any = false;
    for (u32 i = 0; i < count; ++i)
      summary.any |= (inputs[i].flags & kVertexInputBoundFetch) != 0;
    summary.entry = vs.entry;
  }
  if (!summary.any)
    return 0;
  u32 mask = 0;
  for (u32 i = 0; i < count; ++i) {
    if (!(inputs[i].flags & kVertexInputBoundFetch))
      continue;
    for (const GuestVertexElement &e : declaration.elements) {
      if (e.Usage != inputs[i].usage || e.UsageIndex != inputs[i].usageIndex)
        continue;
      const u32 stream = e.Stream;
      if (stream >= kMaxVertexStreams)
        break;
      const u32 fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(stream);
      const StreamSource src = GetStreamSource(device, call, vs, shader_strides, stream);
      sc.bound_fetch[i][0] = fetch_index;
      sc.bound_fetch[i][1] = u32(e.Offset) / 4;
      sc.bound_fetch[i][2] = src.stride / 4;
      sc.bound_fetch[i][3] = BoundDeclType(u32(e.Type)) & 0x3FFFFFu;
      mask |= 1u << (fetch_index - 64);
      break;
    }
  }
  return mask;
}

bool FetchFromExportWindow(const GuestDevice &device, const DrawCall &call, u32 fetch_index,
                           u32 &slot, uint64_t &offset);
bool EnsureBuffer(std::unique_ptr<plume::RenderBuffer> &buffer, uint64_t &capacity, uint64_t size,
                  const plume::RenderBufferDesc &desc_template, const char *name);

// Copies the buffers of the fetch constants the vertex shader reads by address
// into one allocation and fills their SharedConstants slots. A buffer whose
// newest data is in a pending export's window comes from there instead: then
// the whole fetch data is put together on the GPU, in the slot's
// gpu_fetch_data, and the returned allocation's ref and gpu_address point
// there. Records copies: before the framebuffer is bound.
// banjo-nx (2026-10-18, the deko3d water): set by DrawStateLocked for draws of banjo_renut_diag_stream_dump_vs's vertex
// shader; UploadVertexFetchData then logs its fetch copies.
bool g_dump_fetch = false;

UploadAllocation UploadVertexFetchData(const GuestDevice &device, const DrawCall &call, u32 mask,
                                       SharedConstants &sc) {
  struct Copy {
    u32 index;
    const u8 *src;
    u32 size;
    u32 endian;
    bool exported;        // from the export window below
    u32 export_slot;
    uint64_t export_offset;
  };
  Copy copies[32];
  u32 copy_count = 0;
  u32 total = 16;
  bool any_exported = false;
  const u32 up_fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(0);
  for (u32 bit = 0; bit < 32; ++bit) {
    if (!(mask & (1u << bit)))
      continue;
    const u32 index = 64 + bit;
    Copy c{index, nullptr, 0, 0, false, 0, 0};
    if (index == up_fetch_index && call.up_vertices) {
      c.src = call.up_vertices;
      c.size = call.up_vertex_count * call.up_stride;
      c.endian = kUpEndian;
    } else {
      const u32 w0 = device.VertexFetchWord(index, 0);
      const u32 w1 = device.VertexFetchWord(index, 1);
      c.src = (w0 & ~3u) ? GuestPhysicalPtr<const u8>(w0 & ~3u) : nullptr;
      c.size = w1 & 0x3FFFFFCu;
      c.endian = w1 & 3;
    }
    if (!c.src)
      c.size = 0;
    c.size = std::min(c.size, kMaxFetchCopyBytes);
    if (c.size && FetchFromExportWindow(device, call, index, c.export_slot, c.export_offset))
      any_exported = c.exported = true;
    copies[copy_count++] = c;
    total += (c.size + 15u) & ~15u;
  }

  // Without exports the slots count from the start of the upload chunk, so a
  // buffer an earlier draw has queued a copy of in the same chunk is read
  // there (FetchCopy) and not copied again.
  const bool shared = !any_exported;
  // banjo-nx (2026-10-19, render-thread proposal 10): room only for the copies to make: those already queued in the
  // chunk the allocation goes to are read there (before, their room was taken anyway, and cleaned at the submit). If
  // the allocation lands in another chunk, none is (that chunk has none of them): then room for all.
  bool reused[32] = {};
  u32 needed = total;
  const plume::RenderBuffer *chunk = shared ? Video::UploadsLocked().CurrentBuffer() : nullptr;
  if (shared && chunk) {
    needed = 16;
    for (u32 i = 0; i < copy_count; ++i) {
      const Copy &c = copies[i];
      if (c.size && !c.exported && !(c.index == up_fetch_index && call.up_vertices)) {
        const FetchCopy &known = FetchCopyFor(c.src, c.size);
        reused[i] = known.generation == g_fetch_copy_generation && known.src == c.src && known.size == c.size &&
                    known.buffer == chunk;
      }
      if (!reused[i])
        needed += (c.size + 15u) & ~15u;
    }
  }
  UploadAllocation alloc = Video::UploadsLocked().Allocate(needed, 256);
  if (alloc && shared && chunk && alloc.ref.ref != chunk) {
    std::fill(std::begin(reused), std::end(reused), false);
    if (needed < total)
      alloc = Video::UploadsLocked().Allocate(total, 256);
  }
  if (!alloc)
    return alloc;
  const uint64_t chunk_offset = shared ? alloc.ref.offset : 0;
  u32 offset = 0;
  for (u32 i = 0; i < copy_count; ++i) {
    const Copy &c = copies[i];
    u8 *dst = static_cast<u8 *>(alloc.cpu) + offset;
    u32 at = u32(chunk_offset) + offset;
    if (reused[i]) {
      // Queued in this chunk by an earlier draw: read there, no room here.
      sc.vertex_fetch_slots[c.index] = FetchCopyFor(c.src, c.size).offset | c.endian;
      continue;
    }
    if (c.size && !c.exported) {
      if (c.index == up_fetch_index && call.up_vertices) {
        std::memcpy(dst, c.src, c.size);
      } else if (shared) {
        FetchCopy &known = FetchCopyFor(c.src, c.size);
        if (known.generation == g_fetch_copy_generation && known.src == c.src &&
            known.size == c.size && known.buffer == alloc.ref.ref) {
          at = known.offset;
        } else {
          g_copy_fills.push_back({c.src, dst, c.size});
          known = {c.src, c.size, g_fetch_copy_generation, alloc.ref.ref, at};
        }
      } else {
        g_copy_fills.push_back({c.src, dst, c.size});
      }
    }
    sc.vertex_fetch_slots[c.index] = at | c.endian;
    if (g_dump_fetch) {
      static uint32_t logged = 0;
      if (logged++ < 48) {
        u32 w[4] = {};
        if (c.src && c.size >= 16)
          std::memcpy(w, c.src, 16);
        RGPU_INFO("[fetchdump] fetch {} src {} size {} endian {} exported {} (slot {} offset {}) shared {} at {} chunk "
                  "offset {} guest {:08X} {:08X} {:08X} {:08X}",
                  c.index, fmt::ptr(c.src), c.size, c.endian, c.exported, c.export_slot, c.export_offset, shared, at,
                  chunk_offset, w[0], w[1], w[2], w[3]);
      }
    }
    offset += (c.size + 15u) & ~15u;
  }
  if (shared) {
    alloc.ref.offset = 0;
    alloc.gpu_address = alloc.gpu_address ? alloc.gpu_address - chunk_offset : 0;
    sc.vertex_fetch_data = alloc.gpu_address;
    // banjo-nx: the whole chunk from its start (the base), all of it allocated memory: the same for every draw of the
    // chunk, so draws keep the same SharedConstants (banjo_renut_shared_dedupe). 0 for a draw with no copies (its
    // shaders fetch nothing from memory).
    sc.vertex_fetch_bytes = copy_count ? u32(std::min<uint64_t>(alloc.chunk_size, 0xFFFFFFF0ull)) : 0u;
    return alloc;
  }
  sc.vertex_fetch_data = alloc.gpu_address;
  sc.vertex_fetch_bytes = copy_count ? total : 0u;  // banjo-nx: the region below, from its start

  // The guest buffers' copies from the upload, the exported ones from their
  // windows (raw bytes, as guest memory would hold them), each into its own
  // part of the region.
  auto &s = State();
  uint64_t &used = s.gpu_fetch_data_used[s.frame];
  if (!s.gpu_fetch_data[s.frame] || used + total > s.gpu_fetch_data_capacity[s.frame]) {
    // The draws before still read the old buffer, retired with the slot.
    if (!EnsureBuffer(s.gpu_fetch_data[s.frame], s.gpu_fetch_data_capacity[s.frame], used + total,
                      plume::RenderBufferDesc::DefaultBuffer(0, AddressableBufferFlags()),
                      "renut gpu fetch data")) {
      return UploadAllocation{};
    }
    used = 0;
  }
  plume::RenderBuffer *dest = s.gpu_fetch_data[s.frame].get();
  const uint64_t region = used;
  used = (used + total + 255u) & ~uint64_t(255);
  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderBufferBarrier to_copy[kFramesInFlight + 1];
  u32 barrier_count = 0;
  to_copy[barrier_count++] = plume::RenderBufferBarrier(dest, plume::RenderBufferAccess::WRITE);
  for (u32 slot = 0; slot < kFramesInFlight; ++slot) {
    if (s.memexport_buffer[slot]) {
      to_copy[barrier_count++] =
          plume::RenderBufferBarrier(s.memexport_buffer[slot].get(), plume::RenderBufferAccess::READ);
    }
  }
  cmd->barriers(plume::RenderBarrierStage::COPY, to_copy, barrier_count);
  offset = 0;
  for (u32 i = 0; i < copy_count; ++i) {
    const Copy &c = copies[i];
    if (c.size) {
      const plume::RenderBufferReference to(dest, region + offset);
      if (c.exported) {
        cmd->copyBufferRegion(
            to, plume::RenderBufferReference(s.memexport_buffer[c.export_slot].get(), c.export_offset),
            c.size);
      } else {
        cmd->copyBufferRegion(
            to, plume::RenderBufferReference(alloc.ref.ref, alloc.ref.offset + offset), c.size);
      }
    }
    offset += (c.size + 15u) & ~15u;
  }
  // Read by the vertex shader (as a root SRV on D3D12).
  to_copy[0] = plume::RenderBufferBarrier(dest, plume::RenderBufferAccess::READ);
  for (u32 i = 1; i < barrier_count; ++i)
    to_copy[i].accessBits = plume::RenderBufferAccess::WRITE;
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS_AND_COMPUTE, to_copy, 1);
  if (barrier_count > 1)
    cmd->barriers(plume::RenderBarrierStage::GRAPHICS, to_copy + 1, barrier_count - 1);
  g_bound.framebuffer = nullptr;  // the copies end any render pass
  alloc.ref = plume::RenderBufferReference(dest, region);
  alloc.gpu_address = UsesVulkanBindings() ? dest->getDeviceAddress() + region : 0;  // banjo-nx: and deko3d
  sc.vertex_fetch_data = alloc.gpu_address;
  return alloc;
}

// ---------------------------------------------------------------------------

// banjo-nx (banjo_gpu_bool_spec): the bools a shader's SPIR-V specialises, read once.
const u32 *BoolSpecMask(const ShaderRecord *r) {
  if (!r->bool_spec_read) {
    r->bool_spec_read = true;
    if (r->entry)
      ShaderBoolSpecMask(*r->entry, r->bool_spec_mask);
  }
  return r->bool_spec_mask;
}

REXCVAR_DEFINE_BOOL(mk9_bound_fetch_spec, true, "MK9/Renderer",
                    "Renut's renderer: the vertex shaders' bound fetches (vfetchBound) take their element type as a "
                    "specialization constant from the pipeline key, so a shader keeps one format's decode instead of "
                    "all of them (MK9's skinned vertex shaders: 800 KB of GLSL, 45 s of uam on a PC). Off: read at run "
                    "time, as before")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

PipelineKey MakeKey(const GuestDevice &device, const ShaderRecord *vs, const ShaderRecord *ps,
                    const VertexDeclarationRecord *declaration, const Targets &t, u32 primitive,
                    bool per_instance, const u32 *booleans, u32 alpha_test_func) {
  PipelineKey key;
  std::memset(&key, 0, sizeof(key));
  // banjo-nx (banjo_gpu_bool_spec): the draw's values of the bools its shaders specialise (the same words the shaders
  // read at run time, SharedConstants.booleans).
  if (banjo::renut_gpu::BoolSpecOn()) {
    const u32 *vm = BoolSpecMask(vs);
    const u32 *pm = ps ? BoolSpecMask(ps) : nullptr;
    bool any = false;
    for (u32 w = 0; w < 8; ++w) {
      key.bool_mask[w] = vm[w] | (pm ? pm[w] : 0u);
      key.bool_values[w] = booleans[w] & key.bool_mask[w];
      any |= key.bool_mask[w] != 0;
    }
    if (any)
      banjo::renut_gpu::CountBoolSpec(banjo::renut_gpu::BoolSpecEvent::kDraw);
  }
  const d3d::Device &dev = device.dev();
  key.vertex_shader = vs->entry;
  key.pixel_shader = ps ? ps->entry : nullptr;
  key.layout_hash = declaration->layout_hash;
  // mk9-nx (2026-10-21, mk9_bound_fetch_spec): the bound elements' types, as DescribeBoundFetches gives the shader.
  if (REXCVAR_GET(mk9_bound_fetch_spec) && vs->entry) {
    const ShaderVertexInput *inputs = g_shaderVertexInputs + vs->entry->vertexInputOffset;
    const u32 count = std::min<u32>(vs->entry->vertexInputCount, kMaxBoundFetches);
    for (u32 i = 0; i < count && i < 16; ++i) {
      if (!(inputs[i].flags & kVertexInputBoundFetch))
        continue;
      for (const GuestVertexElement &e : declaration->elements) {
        if (e.Usage == inputs[i].usage && e.UsageIndex == inputs[i].usageIndex && e.Stream < kMaxVertexStreams) {
          key.bound_types[i] = (BoundDeclType(u32(e.Type)) & 0x3FFFFFu) + 1;
          break;
        }
      }
    }
  }
  key.topology = Topology(primitive);
  key.color_count = t.color_count;
  for (uint32_t i = 0; i < t.color_count; ++i)
    key.color_formats[i] = t.colors[i]->format;
  key.depth_format = t.depth ? t.depth->format : plume::RenderFormat::UNKNOWN;
  const HostSurface *first = t.color_count ? t.colors[0] : t.depth;
  key.samples = uint32_t(first->samples);
  const d3d::ControlPacket &cp = dev.m_ControlPacket;
  key.blend_control[0] = cp.BlendControl0;
  key.blend_control[1] = cp.BlendControl1;
  key.blend_control[2] = cp.BlendControl2;
  key.blend_control[3] = cp.BlendControl3;
  key.color_mask = u32(dev.m_ValuesPacket.ColorMask) & 0xFFFF;
  key.depth_control = t.depth ? u32(cp.DepthControl) : 0;
  key.stencil_ref_mask = t.depth ? u32(dev.m_ValuesPacket.StencilRefMask) & 0xFFFFFF : 0;
  key.stencil_ref_mask_bf = t.depth ? u32(dev.m_ValuesPacket.StencilRefMaskBF) & 0xFFFFFF : 0;
  key.mode_control = u32(cp.ModeControl) & 0x7;
  key.alpha_to_coverage = (u32(cp.ColorControl) >> 4) & 1;
  if (primitive == kRectangleList) {
    if (State().rectangle_list_gs)
      key.rectangle_list = 1;
    // Rectangles are not culled. N&B's D3D clears through rectangles
    // (ClearRects_D3D, 0x822043D8) whose corners run clockwise on screen, under
    // whatever cull mode the game left set: with back faces culled and
    // counterclockwise front faces (after hitting some enemies) the clear of
    // the post-process surface was culled, so its resolve took the previous
    // frame's final image back into the bloom, which grew to white.
    key.mode_control &= ~3u;
  }
  if (primitive == kPointList && State().point_sprite_gs)
    key.point_sprite = 1;
  if (per_instance) {
    key.per_instance_streams = 1;
    // Which way the tessellated grid winds depends on how the shader weights
    // the control points, so it is not culled.
    key.mode_control &= ~3u;
  }
  // banjo-nx (banjo_gpu_blend_skip): from the fields above.
  // banjo-nx (2026-10-19, render-thread proposal 6): both masks again for the same fields they read (a one-entry memo:
  // OutputTrimMask built a BlendDesc per target at every draw). The fields: whether there is a pixel shader, the
  // targets' count and formats, their blend controls and write mask, alpha to coverage, the depth format and control.
  {
    struct MaskInputs {
      u32 has_ps, color_count, color_mask, alpha_to_coverage, depth_control;
      plume::RenderFormat color_formats[d3d::kMaxRenderTargets];
      plume::RenderFormat depth_format;
      u32 blend_control[d3d::kMaxRenderTargets];
    };
    MaskInputs in;
    std::memset(&in, 0, sizeof(in));
    in.has_ps = key.pixel_shader != nullptr;
    in.color_count = key.color_count;
    in.color_mask = key.color_mask;
    in.alpha_to_coverage = key.alpha_to_coverage;
    in.depth_control = key.depth_control;
    std::memcpy(in.color_formats, key.color_formats, sizeof(in.color_formats));
    in.depth_format = key.depth_format;
    std::memcpy(in.blend_control, key.blend_control, sizeof(in.blend_control));
    static MaskInputs last_in;
    static u32 last_skip = 0, last_trim = 0;
    static bool last_valid = false;
    if (!last_valid || std::memcmp(&in, &last_in, sizeof(in)) != 0) {
      last_skip = BlendSkipMode(key);
      last_trim = OutputTrimMask(key);
      last_in = in;
      last_valid = true;
    }
    if (banjo::renut_gpu::BlendSkipOn() && !BlendSkipExcluded(key.pixel_shader))
      key.blend_skip = last_skip;
    // banjo-nx (banjo_gpu_output_trim, package 28): from the fields above too (the same pipelines: they differ by them).
    if (banjo::renut_gpu::OutputTrimOn())
      key.output_trim = last_trim;
  }
  // banjo-nx (banjo_deko3d_alpha_spec, 2026-10-18): the alpha test's function as the pixel shader's constant 903 (+1):
  // without the test (7) no compare and no discard, and the GPU tests depth before the pixel shader.
  if (ps && banjo::renut_gpu::AlphaSpecOn())
    key.alpha_func = alpha_test_func + 1;
  // banjo-nx (2026-10-20, in-run A/B targets 15 / 16): in the off half of banjo_deko3d_light_branch's or
  // banjo_deko3d_gpr32's A/B the pipelines are built without its constant (911 / 910): keys of their own (reserved
  // bits 0 / 1), so both halves' pipelines stay built. 0 otherwise (the keys as before).
  if (ps) {
    const int ab_off = banjo::renut_gpu::g_ab_off.load(std::memory_order_relaxed);
    if (ab_off == 15 && banjo::renut_gpu::LightBranchOn())
      key.reserved |= 1u;
    else if (ab_off == 16 && banjo::renut_gpu::Gpr32On())
      key.reserved |= 2u;
  }
  return key;
}

// ---------------------------------------------------------------------------
// Memexport. N&B fills vertex buffers (and other data) from vertex shaders that
// write to memory: `mad eA, index, const0100, stream` then eM# (see memExport in
// shader_common.h). A draw whose shaders do gets a window in the export buffer
// per stream constant it uses, seeded from guest memory so unwritten bytes keep
// their value; after the draw the windows are copied to the readback buffer,
// and before anything reads that memory (a draw that reads it, a kickoff or
// present) the list is submitted and waited, and the
// results are written to guest memory, where every other path reads them from.
// Waiting at every draw instead cost a GPU round trip per export draw.

struct MemExportStream {
  u32 base = 0;  // guest physical byte address
  u32 size = 0;  // bytes
  u32 format = 0;      // color format (MemExportElementSize)
  u32 endian = 0;      // Endian128
  u32 num_format = 0;  // SurfaceNumberFormat
};

// Bytes per element of a memexport format: the game's bits per pixel for a
// format of single texels in whole dwords, which is what the shaders write
// (memExport in shader_common.h); 0 for any other.
u32 MemExportElementSize(u32 format) {
  const GpuTextureFormat f = GpuTextureFormat(format & 0x3F);
  u32 block_width, block_height;
  FormatBlockSize(f, block_width, block_height);
  const u32 bpp = FormatBitsPerPixel(f);
  return block_width == 1 && block_height == 1 && bpp % 32 == 0 && bpp <= 128 && bpp != 96
             ? bpp / 8
             : 0;
}

constexpr u32 kMaxMemExportStreams = 4;
constexpr u32 kMaxMemExportBytes = 64u << 20;

// The streams the shader's stream constants describe, laid out as N&B's own
// export passes build them (sub_82317020 before D3DDevice_EndExport): x the
// GPU address in dwords (bits 0-29) tagged 01 in bits 30-31, z the endian in
// bits 0-2, the format in 8-13 and the number format in 16-18, w the element
// count in bits 0-22 (y, z and w carry 0x4B high bits to stay floats).
void GatherMemExportStreams(const ShaderCacheEntry *entry, const FloatConstants &constants,
                            MemExportStream *streams, u32 &count) {
  if (!entry)
    return;
  // banjo-nx (2026-10-19, render-thread proposal 4): whether the shader exports at all, once per entry (static): the
  // 99 % of shaders without exports no longer read their eight cold memExportConstants words at every draw.
  struct ExportSummary {
    const ShaderCacheEntry *entry = nullptr;
    bool any = false;
  };
  static ExportSummary summaries[512];
  const uintptr_t a = reinterpret_cast<uintptr_t>(entry);
  ExportSummary &summary = summaries[((a >> 3) ^ (a >> 12)) & 511];
  if (summary.entry != entry) {
    u32 any = 0;
    for (u32 word = 0; word < 8; ++word)
      any |= entry->memExportConstants[word];
    summary.any = any != 0;
    summary.entry = entry;
  }
  if (!summary.any)
    return;
  // Only the registers the shader exports through (most draws have none).
  for (u32 word = 0; word < 8; ++word) {
    for (u32 bits = entry->memExportConstants[word]; bits; bits &= bits - 1) {
      const u32 r = word * 32 + u32(std::countr_zero(bits));
      const auto *words = reinterpret_cast<const be_u32 *>(constants.Register(r));
      const u32 x = words[0], z = words[2], w = words[3];
      const u32 element = MemExportElementSize((z >> 8) & 0x3F);
      const u32 elements = w & 0x7FFFFF;
      if ((x >> 30) != 1 || !element || !elements) {
        static uint32_t logged = 0;
        if ((x >> 30) == 1 && LogLimited(logged, 8)) {
          RGPU_WARN("Memexport stream in c{} not exported: {:08X} {:08X} {:08X} {:08X} (format "
                    "{}, {} elements)",
                    r, x, u32(words[1]), z, w, (z >> 8) & 0x3F, elements);
        }
        continue;
      }
      MemExportStream stream;
      stream.base = (x & 0x3FFFFFFF) * 4;
      stream.size = std::min(elements * element, kMaxMemExportBytes);
      stream.format = (z >> 8) & 0x3F;
      stream.endian = z & 7;
      stream.num_format = (z >> 16) & 7;
      if (stream.base >= 0x20000000u)
        continue;
      stream.size = std::min(stream.size, 0x20000000u - stream.base);
      bool known = false;
      for (u32 i = 0; i < count; ++i)
        known |= streams[i].base == stream.base && streams[i].size == stream.size;
      if (!known && count < kMaxMemExportStreams)
        streams[count++] = stream;
    }
  }
}

struct PendingExport {
  u32 base = 0;
  u32 size = 0;
  // Its window in the slot's export buffer, and its copy in the slot's
  // readback buffer (the same offset in both).
  uint64_t offset = 0;
};
// The exports a slot's lists made. They reach guest memory once the slot's
// fence is waited (detail::CompleteMemExportsLocked), a frame later; until
// then their windows stay in the slot's export buffer, where the next frame's
// draws seed from them on the GPU (SeedFromPendingExportsLocked).
struct SlotExports {
  std::vector<PendingExport> pending;
  uint64_t used = 0;
};
SlotExports g_exports[kFramesInFlight];

// The 64 KiB granules of physical memory any pending export writes: most
// ranges a draw reads touch none, and skip the lists (PendingExportSlots runs
// for every draw's streams and textures while exports are pending, which is
// always).
constexpr u32 kExportGranuleShift = 16;
constexpr u32 kExportGranules = 0x20000000u >> kExportGranuleShift;
uint64_t g_export_granules[kExportGranules / 64];

void MarkExportGranules(u32 base, u32 size) {
  if (!size)
    return;
  const u32 first = (base & 0x1FFFFFFFu) >> kExportGranuleShift;
  const u32 last = std::min((((base & 0x1FFFFFFFu) + size - 1) >> kExportGranuleShift),
                            kExportGranules - 1);
  for (u32 g = first; g <= last; ++g)
    g_export_granules[g / 64] |= uint64_t(1) << (g % 64);
}

void RebuildExportGranules() {
  std::memset(g_export_granules, 0, sizeof(g_export_granules));
  for (const SlotExports &slot : g_exports) {
    for (const PendingExport &p : slot.pending)
      MarkExportGranules(p.base, p.size);
  }
}

// A word of the bitmap at a time: the bits of the first and last words that
// are in the range, the words between whole.
bool ExportGranulesTouched(u32 base, u32 size) {
  const u32 first = (base & 0x1FFFFFFFu) >> kExportGranuleShift;
  const u32 last = std::min((((base & 0x1FFFFFFFu) + size - 1) >> kExportGranuleShift),
                            kExportGranules - 1);
  if (last < first)
    return false;
  const u32 first_word = first / 64, last_word = last / 64;
  const uint64_t first_mask = ~uint64_t(0) << (first % 64);
  const uint64_t last_mask = ~uint64_t(0) >> (63 - last % 64);
  if (first_word == last_word)
    return (g_export_granules[first_word] & first_mask & last_mask) != 0;
  if (g_export_granules[first_word] & first_mask)
    return true;
  for (u32 w = first_word + 1; w < last_word; ++w) {
    if (g_export_granules[w])
      return true;
  }
  return (g_export_granules[last_word] & last_mask) != 0;
}

bool AnyPendingExports() {
  for (const SlotExports &slot : g_exports) {
    if (!slot.pending.empty())
      return true;
  }
  return false;
}

// A GPU physical address, or a guest VA in the physical windows, as a physical
// address (other VAs come out as unrelated addresses, which at worst flush the
// exports early).
u32 PhysicalAddress(u32 address) {
  return address >= 0x20000000u ? d3d::CpuToGpuAddress(address) & 0x1FFFFFFFu : address;
}

// The slots (a mask) with a pending export that will write memory overlapping
// [address, address + size).
u32 PendingExportSlots(u32 address, u32 size) {
  if (!size)
    return 0;
  const u32 base = PhysicalAddress(address);
  if (!ExportGranulesTouched(base, size))
    return 0;
  u32 slots = 0;
  for (u32 slot = 0; slot < kFramesInFlight; ++slot) {
    for (const PendingExport &p : g_exports[slot].pending) {
      if (base < p.base + p.size && p.base < base + size) {
        slots |= 1u << slot;
        break;
      }
    }
  }
  return slots;
}

// The pending export window holding the newest data of all of [base, base +
// size) (physical): the newest pending export overlapping it, if it covers it
// whole. Its slot and the offset of 'base' in that slot's export buffer.
bool FindExportWindow(u32 base, u32 size, u32 &slot, uint64_t &offset) {
  const auto &s = State();
  // Oldest slot first, and a slot's exports in the order they were drawn.
  const PendingExport *newest = nullptr;
  u32 newest_slot = 0;
  for (u32 k = 1; k <= kFramesInFlight; ++k) {
    const u32 at = (s.frame + k) % kFramesInFlight;
    for (const PendingExport &p : g_exports[at].pending) {
      if (base < p.base + p.size && p.base < base + size) {
        newest = &p;
        newest_slot = at;
      }
    }
  }
  if (!newest || base < newest->base || base + size > newest->base + newest->size)
    return false;
  slot = newest_slot;
  offset = newest->offset + (base - newest->base);
  return true;
}

// Whether UploadVertexFetchData copies the buffer of fetch constant
// 'fetch_index' (read by vfetchMemory / vfetchBound) from a pending export's
// window on the GPU, which it does when the window holds all of it.
bool FetchFromExportWindow(const GuestDevice &device, const DrawCall &call, u32 fetch_index,
                           u32 &slot, uint64_t &offset) {
  if (call.up_vertices && fetch_index == d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(0))
    return false;
  const u32 base = device.VertexFetchWord(fetch_index, 0) & ~3u;
  const u32 size = std::min(device.VertexFetchWord(fetch_index, 1) & 0x3FFFFFCu, kMaxFetchCopyBytes);
  return base && size && PendingExportSlots(base, size) && FindExportWindow(base, size, slot, offset);
}

// A vertex stream built on the GPU from the export window that holds its
// newest data (ConvertExportedStreamsLocked).
struct ExportedStream {
  u32 slot = 0;          // whose export buffer holds the window
  uint64_t offset = 0;   // of the stream's first byte there
  u32 stride = 0;        // guest
  u32 host_stride = 0;
  u32 vertex_count = 0;  // in the stream's buffer
  u32 element_count = 0;
  u32 elements[2 * kStreamConvertMaxElements] = {};  // StreamConvertPush::elements
};

// Whether a draw can read vertex stream 'stream' from a pending export's
// window instead of guest memory: the window holds the newest data of all of
// the stream's buffer (FindExportWindow), and stream_convert_cs can build its
// host layout (word-aligned elements, few enough of them). Shader reads
// through vfetchBound get the window's bytes too (UploadVertexFetchData).
bool FindExportedStream(const GuestDevice &device, const DrawCall &call, const ShaderRecord &vs,
                        const u32 *shader_strides, const VertexDeclarationRecord &declaration,
                        u32 stream, ExportedStream &out) {
  const auto &s = State();
  if (!s.stream_convert_pipeline || stream >= kMaxVertexStreams || (stream == 0 && call.up_vertices))
    return false;
  const HostVertexStream &hs = declaration.layout.streams[stream];
  if (!hs.used)
    return false;
  const u32 fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(stream);
  const u32 base = device.VertexFetchWord(fetch_index, 0) & ~3u;
  const u32 size = device.VertexFetchWord(fetch_index, 1) & 0x3FFFFFCu;
  if (!base || !size || !PendingExportSlots(base, size))
    return false;
  const StreamSource src = GetStreamSource(device, call, vs, shader_strides, stream);
  if (!src.stride || src.stride % 4 || src.stride / 4 > kStreamConvertMaxStrideWords ||
      hs.host_stride % 4 || hs.elements.size() > kStreamConvertMaxElements) {
    return false;
  }
  u32 elements[2 * kStreamConvertMaxElements] = {};
  for (size_t i = 0; i < hs.elements.size(); ++i) {
    const HostVertexElement &e = hs.elements[i];
    if (e.guest_offset % 4 || e.guest_offset + 4 > src.stride)
      return false;
    const bool expand = e.conversion == ElementConversion::ExpandFloat;
    // As ConvertVertices: swapping the halves too is the endian mode ^ 3.
    const u32 endian = !expand && e.swap_halves ? src.endian ^ 3 : src.endian;
    u32 type = e.type.format | (e.type.is_signed ? 1u << 8 : 0u) | (e.type.is_integer ? 1u << 9 : 0u);
    for (u32 c = 0; c < 4; ++c)
      type |= u32(e.type.swizzle[c] & 7) << (10 + 3 * c);
    elements[2 * i] = (e.guest_offset / 4) | (e.host_offset / 4) << 8 | endian << 16 |
                      (expand ? 1u : 0u) << 18 | (e.host_size / 4 & 7) << 19;
    elements[2 * i + 1] = type;
  }
  if (!FindExportWindow(base, size, out.slot, out.offset))
    return false;
  out.stride = src.stride;
  out.host_stride = hs.host_stride;
  out.vertex_count = size / src.stride;
  out.element_count = u32(hs.elements.size());
  std::memcpy(out.elements, elements, sizeof(elements));
  return true;
}

// Why a draw needs the pending exports in guest memory first (see
// DrawNeedsExportFlush).
enum class ExportRead : u32 {
  None,
  Stream,    // a declared vertex stream
  Fetch,     // a vertex fetch by address (vfetchMemory)
  Index,     // the index buffer
  Texture,
  Capacity,  // its exports would not fit the readback buffer beside theirs
};

// Whether pending exports must reach guest memory before a draw, and whose
// ('slots', a mask): it reads memory they write (its vertex streams, bound or
// read by address, indices or textures), or its exports would not fit this
// slot's buffers beside theirs (growing them would lose their windows). Any
// other draw can be recorded while they are still on the GPU, and so can one
// whose streams are built from the export windows (FindExportedStream) or
// whose own exports are seeded from them (SeedFromPendingExportsLocked).
ExportRead DrawNeedsExportFlush(const GuestDevice &device, const DrawCall &call,
                                const ShaderRecord &vs, const ShaderRecord *ps,
                                const VertexDeclarationRecord &declaration,
                                const u32 *shader_strides, const MemExportStream *exports, u32 count,
                                u32 &slots) {
  const u32 current = 1u << State().frame;
  auto fetch_slots = [&](u32 fetch_index) {
    const u32 w0 = device.VertexFetchWord(fetch_index, 0);
    const u32 w1 = device.VertexFetchWord(fetch_index, 1);
    return PendingExportSlots(w0 & ~3u, w1 & 0x3FFFFFCu);
  };
  auto found = [&](ExportRead read, u32 found_slots) {
    slots = found_slots;
    return read;
  };
  u32 streams_checked = 0;
  for (const GuestVertexElement &e : declaration.elements) {
    const u32 stream = u16(e.Stream);
    if (stream >= kMaxVertexStreams || (streams_checked & (1u << stream)))
      continue;
    streams_checked |= 1u << stream;
    // A Begin/DrawVerticesUP draw's stream 0 is its own vertices, whatever
    // buffer the fetch constant still names: nothing of it is read from
    // memory.
    if (stream == 0 && call.up_vertices)
      continue;
    const u32 fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(stream);
    if (const u32 found_slots = fetch_slots(fetch_index)) {
      // A stream the host layout reads is built from the export window; one it
      // does not (no host input) is read only by vfetchBound, whose copy the
      // window supplies (UploadVertexFetchData).
      ExportedStream exported;
      u32 slot;
      uint64_t offset;
      const bool on_gpu =
          declaration.layout.streams[stream].used
              ? FindExportedStream(device, call, vs, shader_strides, declaration, stream, exported)
              : FetchFromExportWindow(device, call, fetch_index, slot, offset);
      if (!on_gpu)
        return found(ExportRead::Stream, found_slots);
    }
  }
  if (vs.entry) {
    for (u32 bits = vs.entry->vertexFetchMemoryMask; bits; bits &= bits - 1) {
      const u32 fetch_index = 64 + u32(std::countr_zero(bits));
      if (const u32 found_slots = fetch_slots(fetch_index)) {
        u32 slot;
        uint64_t offset;
        if (!FetchFromExportWindow(device, call, fetch_index, slot, offset))
          return found(ExportRead::Fetch, found_slots);
      }
    }
  }
  // A tessellated draw's "indices" are only adaptive edge factors, which set
  // its level (PrepareTessellation): the ones in guest memory, from a frame
  // before, set it as well. N&B's water computes them with memexport before
  // each draw, and waiting for them was two GPU round trips a frame.
  if (call.indexed && !call.up_indices && !call.tessellation) {
    const auto *ib = GuestPtr<const d3d::IndexBuffer>(device.U32(d3d::kDeviceOffsetIndexBuffer));
    if (ib) {
      if (const u32 found_slots = PendingExportSlots(ib->Address, ib->Size))
        return found(ExportRead::Index, found_slots);
    }
  }
  const u32 texture_mask =
      (vs.entry ? vs.entry->textureFetchMask : 0) | (ps && ps->entry ? ps->entry->textureFetchMask : 0);
  for (u32 bits = texture_mask; bits; bits &= bits - 1) {
    u32 pages[2], sizes[2];
    TextureGuestExtent(DeviceTextureFetch(device.dev(), u32(std::countr_zero(bits))), pages, sizes);
    for (u32 i = 0; i < 2; ++i) {
      if (!pages[i])
        continue;
      if (const u32 found_slots = PendingExportSlots(pages[i] << 12, sizes[i]))
        return found(ExportRead::Texture, found_slots);
    }
  }
  // banjo-nx (2026-10-19, render-thread proposal 4): the draw's exports, gathered once (DrawStateLocked).
  uint64_t bytes = 0;
  for (u32 i = 0; i < count; ++i)
    bytes += (exports[i].size + 255u) & ~255u;
  const auto &s = State();
  const uint64_t used = g_exports[s.frame].used;
  if (bytes && used &&
      used + bytes > std::min(s.memexport_capacity[s.frame], s.memexport_readback_capacity[s.frame])) {
    return found(ExportRead::Capacity, current);
  }
  return ExportRead::None;
}

// Recreates 'buffer' with at least 'size' bytes when it is smaller.
bool EnsureBuffer(std::unique_ptr<plume::RenderBuffer> &buffer, uint64_t &capacity, uint64_t size,
                  const plume::RenderBufferDesc &desc_template, const char *name) {
  if (buffer && capacity >= size)
    return true;
  uint64_t grown = std::max<uint64_t>(capacity * 2, 4u << 20);
  while (grown < size)
    grown *= 2;
  plume::RenderBufferDesc desc = desc_template;
  desc.size = grown;
  if (buffer)
    Video::RetireLocked(std::move(buffer));
  buffer = State().device->createBuffer(desc);
  capacity = buffer ? grown : 0;
  if (buffer)
    buffer->setName(name);
  return buffer != nullptr;
}

// The current slot's export buffer, in its unordered-access state (created on
// first use).
plume::RenderBuffer *MemExportBufferLocked(VideoState &s, uint64_t size = 256) {
  auto &buffer = s.memexport_buffer[s.frame];
  const bool had = buffer && s.memexport_capacity[s.frame] >= size;
  if (!EnsureBuffer(buffer, s.memexport_capacity[s.frame], size,
                    plume::RenderBufferDesc::DefaultBuffer(
                        0, StorageBufferFlags() | AddressableBufferFlags()),
                    "renut memexport buffer")) {
    return nullptr;
  }
  if (!had) {
    s.command_list->barriers(plume::RenderBarrierStage::GRAPHICS,
                             plume::RenderBufferBarrier(buffer.get(),
                                                        plume::RenderBufferAccess::WRITE));
  }
  return buffer.get();
}

// A part of a draw's export window whose seed comes from a pending export's
// window instead of guest memory.
struct SeedPiece {
  u32 slot = 0;
  uint64_t source = 0;  // in the slot's export buffer
  uint64_t dest = 0;    // in the current slot's export buffer
  u32 size = 0;
};

// Guest memory lacks what the pending exports will write there, the older
// slots' and this slot's own earlier draws': copies their windows over the
// parts of the draw's windows they overlap, each byte from the newest export
// of it (the copies of one batch are unordered, so no two overlap). Waiting
// for this slot's instead cost a GPU round trip per such draw, the whole
// frame's GPU work so far. A slot's own windows pass through its scratch
// buffer, as a buffer cannot be copied into itself. The copies wait for the
// guest seed through the buffer's unordered-access state in between.
bool SeedFromPendingExportsLocked(VideoState &s, plume::RenderBuffer *buffer,
                                  const MemExportStream *streams, u32 count, const u32 *offsets) {
  static std::vector<SeedPiece> pieces;
  static std::vector<std::pair<u32, u32>> covered;  // [begin, end), newer exports
  static std::vector<std::pair<u32, u32>> parts;
  pieces.clear();
  for (u32 i = 0; i < count; ++i) {
    const u32 stream_begin = streams[i].base, stream_end = streams[i].base + streams[i].size;
    covered.clear();
    // Newest first: this slot's latest draw, back to the oldest slot's first.
    for (u32 k = kFramesInFlight; k >= 1; --k) {
      const u32 slot = (s.frame + k) % kFramesInFlight;
      if (!s.memexport_buffer[slot])
        continue;
      const auto &pending = g_exports[slot].pending;
      for (auto p = pending.rbegin(); p != pending.rend(); ++p) {
        const u32 begin = std::max(p->base, stream_begin);
        const u32 end = std::min(p->base + p->size, stream_end);
        if (begin >= end)
          continue;
        // What of [begin, end) no newer export covers.
        parts.assign(1, {begin, end});
        for (const auto &[cb, ce] : covered) {
          for (size_t j = 0; j < parts.size();) {
            const auto [pb, pe] = parts[j];
            if (ce <= pb || pe <= cb) {
              ++j;
              continue;
            }
            parts.erase(parts.begin() + j);
            if (pb < cb)
              parts.insert(parts.begin() + j++, {pb, cb});
            if (ce < pe)
              parts.insert(parts.begin() + j++, {ce, pe});
          }
        }
        for (const auto &[pb, pe] : parts) {
          pieces.push_back({slot, p->offset + (pb - p->base),
                            offsets[i] + uint64_t(pb - stream_begin), pe - pb});
        }
        covered.push_back({begin, end});
      }
    }
  }
  if (pieces.empty())
    return true;

  // This slot's own pieces go to the scratch buffer first.
  uint64_t own_bytes = 0;
  for (const SeedPiece &piece : pieces) {
    if (piece.slot == s.frame)
      own_bytes += piece.size;
  }
  plume::RenderBuffer *scratch = nullptr;
  uint64_t scratch_offset = 0;
  if (own_bytes) {
    uint64_t &used = s.memexport_scratch_used[s.frame];
    if (!s.memexport_scratch[s.frame] || used + own_bytes > s.memexport_scratch_capacity[s.frame]) {
      // Copies recorded before still use the old one, retired with the slot.
      if (!EnsureBuffer(s.memexport_scratch[s.frame], s.memexport_scratch_capacity[s.frame],
                        used + own_bytes, plume::RenderBufferDesc::DefaultBuffer(0),
                        "renut memexport scratch")) {
        return false;
      }
      used = 0;
    }
    scratch = s.memexport_scratch[s.frame].get();
    scratch_offset = used;
    used += own_bytes;
  }

  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderBufferBarrier barriers[kFramesInFlight + 1];
  u32 barrier_count = 0;
  if (scratch) {
    barriers[0] = plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::READ);
    barriers[1] = plume::RenderBufferBarrier(scratch, plume::RenderBufferAccess::WRITE);
    cmd->barriers(plume::RenderBarrierStage::COPY, barriers, 2);
    uint64_t at = scratch_offset;
    for (const SeedPiece &piece : pieces) {
      if (piece.slot != s.frame)
        continue;
      cmd->copyBufferRegion(plume::RenderBufferReference(scratch, at),
                            plume::RenderBufferReference(buffer, piece.source), piece.size);
      at += piece.size;
    }
    barriers[barrier_count++] = plume::RenderBufferBarrier(scratch, plume::RenderBufferAccess::READ);
  }
  barriers[barrier_count++] = plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::WRITE);
  for (u32 k = 1; k < kFramesInFlight; ++k) {
    const u32 slot = (s.frame + k) % kFramesInFlight;
    if (s.memexport_buffer[slot]) {
      barriers[barrier_count++] =
          plume::RenderBufferBarrier(s.memexport_buffer[slot].get(), plume::RenderBufferAccess::READ);
    }
  }
  cmd->barriers(plume::RenderBarrierStage::COPY, barriers, barrier_count);
  uint64_t at = scratch_offset;
  for (const SeedPiece &piece : pieces) {
    if (piece.slot == s.frame) {
      cmd->copyBufferRegion(plume::RenderBufferReference(buffer, piece.dest),
                            plume::RenderBufferReference(scratch, at), piece.size);
      at += piece.size;
    } else {
      cmd->copyBufferRegion(plume::RenderBufferReference(buffer, piece.dest),
                            plume::RenderBufferReference(s.memexport_buffer[piece.slot].get(),
                                                         piece.source),
                            piece.size);
    }
  }
  // Back to unordered access, as the other export buffers were.
  barrier_count = 0;
  for (u32 slot = 0; slot < kFramesInFlight; ++slot) {
    if (s.memexport_buffer[slot]) {
      barriers[barrier_count++] =
          plume::RenderBufferBarrier(s.memexport_buffer[slot].get(), plume::RenderBufferAccess::WRITE);
    }
  }
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS, barriers, barrier_count);
  return true;
}

// Seeds the draw's export windows from guest memory (and the older slots'
// pending exports) and fills its stream table. The windows follow the slot's
// earlier ones in its export buffer, which keeps them for later draws' seeds;
// returns their offsets.
bool PrepareMemExportLocked(VideoState &s, const MemExportStream *streams, u32 count,
                            SharedConstants &sc, u32 *offsets) {
  const uint64_t used = g_exports[s.frame].used;
  uint64_t total = used;
  for (u32 i = 0; i < count; ++i) {
    offsets[i] = u32(total);
    total += (streams[i].size + 255u) & ~255u;
  }
  plume::RenderBuffer *buffer = MemExportBufferLocked(s, total);
  if (!buffer || !EnsureBuffer(s.memexport_readback[s.frame], s.memexport_readback_capacity[s.frame],
                               total, plume::RenderBufferDesc::ReadbackBuffer(0),
                               "renut memexport readback")) {
    return false;
  }
  plume::RenderCommandList *cmd = s.command_list;
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::WRITE));
  for (u32 i = 0; i < count; ++i) {
    UploadAllocation seed = Video::UploadsLocked().Allocate(streams[i].size, 256);
    const u8 *guest = GuestPhysicalPtr<const u8>(streams[i].base);
    if (!seed || !guest)
      return false;
    std::memcpy(seed.cpu, guest, streams[i].size);
    cmd->copyBufferRegion(plume::RenderBufferReference(buffer, offsets[i]), seed.ref,
                          streams[i].size);
    sc.memexport_streams[i][0] = streams[i].base;
    sc.memexport_streams[i][1] = streams[i].size;
    sc.memexport_streams[i][2] = offsets[i];
  }
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::WRITE));
  return SeedFromPendingExportsLocked(s, buffer, streams, count, offsets);
}

// After the draw: copies its windows to the slot's readback buffer, pending
// until the slot's fence is waited.
void FinishMemExportLocked(VideoState &s, const MemExportStream *streams, u32 count,
                           const u32 *offsets) {
  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderBuffer *buffer = s.memexport_buffer[s.frame].get();
  plume::RenderBuffer *readback = s.memexport_readback[s.frame].get();
  SlotExports &slot = g_exports[s.frame];
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::READ));
  for (u32 i = 0; i < count; ++i) {
    cmd->copyBufferRegion(plume::RenderBufferReference(readback, offsets[i]),
                          plume::RenderBufferReference(buffer, offsets[i]), streams[i].size);
    slot.pending.push_back({streams[i].base, streams[i].size, offsets[i]});
    MarkExportGranules(streams[i].base, streams[i].size);
    slot.used = offsets[i] + ((streams[i].size + 255u) & ~255u);
  }
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderBufferBarrier(buffer, plume::RenderBufferAccess::WRITE));
}

// Brings the pending exports of 'slots' to guest memory, with those of every
// older slot (so a later export of the same memory lands last): waits for the
// older slots' fences, and submits the open list and waits for it when the
// current slot is one of them.
void WaitForExportsLocked(u32 slots) {
  auto &s = State();
  u32 last = 0;  // in age order: 1 is the oldest slot, kFramesInFlight the current one
  for (u32 k = 1; k <= kFramesInFlight; ++k) {
    if (slots & (1u << ((s.frame + k) % kFramesInFlight)))
      last = k;
  }
  for (u32 k = 1; k <= last; ++k) {
    const u32 slot = (s.frame + k) % kFramesInFlight;
    if (g_exports[slot].pending.empty())
      continue;
    if (slot == s.frame)
      detail::SubmitAndWaitLocked(s);
    else
      detail::WaitSlotLocked(s, slot);
    detail::CompleteMemExportsLocked(slot);
  }
}

// Builds the host vertices of the draw's streams whose newest data is in a
// pending export's window (FindExportedStream; DrawNeedsExportFlush did not
// wait for those) with stream_convert_cs, into the slot's stream buffer.
// Records dispatches: before the framebuffer is bound.
bool ConvertExportedStreamsLocked(VideoState &s, const GuestDevice &device, const DrawCall &call,
                                  const ShaderRecord &vs, const u32 *shader_strides,
                                  const VertexDeclarationRecord &declaration, const Geometry &g,
                                  GpuStreams &out) {
  if (!s.stream_convert_pipeline || !AnyPendingExports() || REXCVAR_GET(banjo_renut_diag_no_stream_convert))
    return true;
  struct Job {
    u32 stream;
    ExportedStream src;
    u32 first, count;
  };
  // banjo-nx (2026-10-19, render-thread proposal 4): storage only (ExportedStream's member initialisers zeroed all 16
  // jobs, about 1.8 KB, at every draw while exports were pending); a job is made for each stream that has one.
  alignas(Job) unsigned char job_storage[sizeof(Job) * kMaxVertexStreams];
  Job *jobs = reinterpret_cast<Job *>(job_storage);
  u32 job_count = 0;
  uint64_t total = 0;
  for (u32 stream = 0; stream < kMaxVertexStreams; ++stream) {
    if (!declaration.layout.streams[stream].used)
      continue;
    const u32 fetch_index = d3d::kFirstVertexFetchIndex + d3d::StreamFetchIndex(stream);
    const u32 base = device.VertexFetchWord(fetch_index, 0) & ~3u;
    const u32 size = device.VertexFetchWord(fetch_index, 1) & 0x3FFFFFCu;
    if (!PendingExportSlots(base, size))
      continue;
    Job &job = *new (&jobs[job_count]) Job{stream};  // made in place, kept by the ++job_count below
    if (!FindExportedStream(device, call, vs, shader_strides, declaration, stream, job.src))
      continue;
    // The vertices BindStreams would convert.
    job.first = g.first_vertex;
    if (job.first >= job.src.vertex_count)
      continue;
    job.count = std::min(g.vertex_range, job.src.vertex_count - job.first);
    if (!job.count)
      continue;
    total += (uint64_t(job.count) * job.src.host_stride + 255u) & ~uint64_t(255);
    ++job_count;
  }
  if (!job_count)
    return true;

  // A buffer too small is replaced; the draws before still read the old one,
  // retired with the slot.
  uint64_t &used = s.exported_streams_used[s.frame];
  if (!s.exported_streams[s.frame] || used + total > s.exported_streams_capacity[s.frame]) {
    if (!EnsureBuffer(s.exported_streams[s.frame], s.exported_streams_capacity[s.frame],
                      used + total,
                      plume::RenderBufferDesc::DefaultBuffer(
                          0, StorageBufferFlags() | plume::RenderBufferFlag::VERTEX),
                      "renut exported streams")) {
      return false;
    }
    used = 0;
  }
  plume::RenderBuffer *dest = s.exported_streams[s.frame].get();
  const plume::RenderBuffer *buffers[kStreamConvertBuffers];
  for (u32 slot = 0; slot < kFramesInFlight; ++slot)
    buffers[slot] = s.memexport_buffer[slot] ? s.memexport_buffer[slot].get() : dest;
  buffers[kFramesInFlight] = dest;
  if (!s.stream_convert_set ||
      std::memcmp(buffers, s.stream_convert_set_buffers, sizeof(buffers)) != 0) {
    // Dispatches recorded before still use the old set.
    Video::RetireLocked(std::move(s.stream_convert_set));
    s.stream_convert_set = detail::CreateStreamConvertSet(s);
    if (!s.stream_convert_set)
      return false;
    for (u32 i = 0; i < kStreamConvertBuffers; ++i)
      s.stream_convert_set->setBuffer(i, buffers[i]);
    std::memcpy(s.stream_convert_set_buffers, buffers, sizeof(buffers));
  }

  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderBufferBarrier barriers[kFramesInFlight + 1];
  u32 barrier_count = 0;
  barriers[barrier_count++] = plume::RenderBufferBarrier(dest, plume::RenderBufferAccess::WRITE);
  for (u32 slot = 0; slot < kFramesInFlight; ++slot) {
    if (s.memexport_buffer[slot]) {
      barriers[barrier_count++] =
          plume::RenderBufferBarrier(s.memexport_buffer[slot].get(), plume::RenderBufferAccess::WRITE);
    }
  }
  static const uint32_t kProfileLabel = gpu_profile::Label("exported stream conversion");
  gpu_profile::Mark(s, kProfileLabel);
  cmd->barriers(plume::RenderBarrierStage::COMPUTE, barriers, barrier_count);
  cmd->setComputePipelineLayout(s.stream_convert_layout.get());
  cmd->setPipeline(s.stream_convert_pipeline.get());
  cmd->setComputeDescriptorSet(s.stream_convert_set.get(), 0);
  for (u32 i = 0; i < job_count; ++i) {
    const Job &job = jobs[i];
    const u32 bytes = job.count * job.src.host_stride;
    StreamConvertPush push{};
    push.source = job.src.slot;
    push.source_offset = u32(job.src.offset + uint64_t(job.first) * job.src.stride);
    push.dest_offset = u32(used);
    push.vertices = job.count;
    push.guest_stride_words = job.src.stride / 4;
    push.host_stride_words = job.src.host_stride / 4;
    push.element_count = job.src.element_count;
    std::memcpy(push.elements, job.src.elements, sizeof(push.elements));
    cmd->setComputePushConstants(0, &push, 0, sizeof(push));  // banjo-nx: the size (the record worker copies it)
    cmd->dispatch((job.count * job.src.element_count + 63) / 64, 1, 1);
    out.ref[job.stream] = plume::RenderBufferReference(dest, used);
    out.bytes[job.stream] = bytes;
    used += (uint64_t(bytes) + 255u) & ~uint64_t(255);
  }
  barriers[0] = plume::RenderBufferBarrier(dest, plume::RenderBufferAccess::READ);
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS, barriers, barrier_count);
  // The compute pipeline replaced the graphics one (D3D12), and the barriers
  // end any render pass.
  g_bound.pipeline = nullptr;
  g_bound.framebuffer = nullptr;
  return true;
}

// Stands in for a null vertex declaration: the vertex shader fetches its data
// by address (vfetchMemory), and any input it declares reads the zero stream.
VertexDeclarationRecord &NoDeclaration() {
  static VertexDeclarationRecord record = [] {
    VertexDeclarationRecord r;
    r.layout = BuildHostVertexLayout(nullptr, 0);
    r.layout_hash = 0x4E4F44454300ull;  // any value no FNV hash of a real layout hits
    return r;
  }();
  return record;
}

} // namespace

namespace detail {

void FinishRecordingLocked(VideoState &) {
  FlushVertexFillsLocked();
  FlushTextureFillsLocked();
}

} // namespace detail

void FlushVertexFillsLocked() {
  ++g_flush_serial;
  static const bool diag = REXCVAR_GET(mk9_diag_bad_vertices);
  if (diag)
    DiagRecheckPositions();  // banjo-nx: banjo_renut_exact_streams' intervals
  banjo::renut_gpu::CountKick();
  for (const PendingStreamFill &f : g_stream_fills) {
    ConvertVertices(f.stream, f.guest, f.stride, f.count, f.endian, f.host);
    static const bool diag = REXCVAR_GET(mk9_diag_bad_vertices);
    if (diag)
      DiagCheckPositions(f.stream, f.host, f.count, f.vs_hash, f.ps_hash, f.primitive, f.guest, "stream", f.stride);
    if (f.cacheable)
      TrackConvertedStream(f);
  }
  g_stream_fills.clear();
  ++g_queued_generation;  // banjo-nx (2026-10-19): the table emptied
  for (const PendingCopy &c : g_copy_fills)
    std::memcpy(c.dst, c.src, c.size);
  g_copy_fills.clear();
  ++g_fetch_copy_generation;
}

void SetTilesLocked(const ClearRect *rects, u32 count) {
  g_tiles.assign(rects, rects + (rects ? count : 0));
}

u32 SetReplayPredicationLocked(u32 mask) {
  const u32 previous = g_replay_predication;
  g_replay_predication = mask;
  return previous;
}

// banjo-nx (banjo_gpu_depth_only, package 28; Unleashed / Marathon's depth-only pipelines): a draw that writes no
// colour, into a depth target, without the alpha test or alpha to coverage, keeps nothing of its pixel shader unless the
// shader kills pixels, writes depth or exports to memory (ShaderCacheEntry::shaderFlags, which the translator checks
// against the shader's own text, and memExportConstants): it is drawn without it. Depth and stencil come from the
// rasterizer either way (no SV_Depth), so the targets get the same values; the shader's constants and textures are not
// set up, and its draws share one pipeline per vertex shader and state, whose vertex outputs NVK drops (linked stages).
bool DrawsWithoutPixelShader(const GuestDevice &device, const Targets &t, const ShaderRecord &ps) {
  if (!t.depth || !ps.entry || !banjo::renut_gpu::DepthOnlyOn())
    return false;
  // sa2-nx (2026-10-20): the colour mask first (every SA2 draw writes colour: one word instead of the shader's export
  // constants); all the checks only read, so their order does not change the answer.
  const d3d::Device &dev = device.dev();
  const u32 color_mask = u32(dev.m_ValuesPacket.ColorMask) & 0xFFFF;  // MakeKey's; 4 bits per target
  if (t.color_count && (color_mask & ((1u << (4 * t.color_count)) - 1u)))
    return false;
  const ShaderCacheEntry &e = *ps.entry;
  if (e.shaderFlags & (kShaderFlagPixelKill | kShaderFlagPixelDepth))
    return false;
  for (const u32 m : e.memExportConstants) {
    if (m)
      return false;
  }
  const u32 color_control = dev.m_ControlPacket.ColorControl;
  if ((color_control & 8) && (color_control & 7) != 7)  // the alpha test, as sc.alpha_test_func
    return false;
  return ((color_control >> 4) & 1) == 0;  // alpha to coverage, as MakeKey's
}

// banjo-nx (2026-10-18, matching RenderDoc captures of Ryujinx to the game's shaders): banjo_renut_diag_draw_log_s
// seconds after the first draw, every draw of the next two frames: its shaders, counts, target and state, in order.
void DrawLog(const DrawCall &call, const Geometry &geometry, const ShaderRecord *vs, const ShaderRecord *ps,
             const Targets &targets, const PipelineKey &key, const SharedConstants &sc, u32 export_count) {
  static const int32_t at = REXCVAR_GET(banjo_renut_diag_draw_log_s);
  if (at <= 0)
    return;
  static const auto start = std::chrono::steady_clock::now();
  static uint64_t first_frame = 0;
  static uint32_t ordinal = 0;
  static uint64_t ordinal_frame = 0;
  const uint64_t frame = TextureFrameLocked();
  if (!first_frame) {
    if (std::chrono::steady_clock::now() - start < std::chrono::seconds(at))
      return;
    first_frame = frame + 1;  // from the next whole frame
  }
  if (frame < first_frame || frame > first_frame + 1)
    return;
  if (frame != ordinal_frame) {
    ordinal_frame = frame;
    ordinal = 0;
  }
  const HostSurface *c0 = targets.color_count ? targets.colors[0] : nullptr;
  RGPU_INFO("[drawlog] frame {} draw {}: vs {:016X} ps {:016X} prim {} count {} -> {} {} inst {}{} target {}x{} fmt {} "
            "x{} depth {} dc {:08X} blend {:08X} mask {:X} a2c {} vfetch {} bytes, exports {}",
            frame, ordinal++, vs ? vs->hash : 0, ps ? ps->hash : 0, call.primitive, call.count,
            geometry.indexed ? "indices" : "vertices", geometry.indexed ? geometry.index_count : geometry.vertex_count,
            geometry.instance_count, call.tessellation ? " tess" : "", c0 ? c0->width : 0, c0 ? c0->height : 0,
            c0 ? int(c0->format) : -1, targets.color_count, targets.depth ? int(targets.depth->format) : -1,
            key.depth_control, key.blend_control[0], key.color_mask, key.alpha_to_coverage, sc.vertex_fetch_bytes,
            export_count);
}

// sa2-nx (2026-10-20, sa2_renut_chain_reuse): the previous chain draw's targets and viewport (DrawStateLocked).
struct TargetsReuse {
  uint64_t seq = 0, epoch = 0;
  u32 va[d3d::kMaxRenderTargets + 1] = {};       // the device's render target words and its depth word
  u32 desc[d3d::kMaxRenderTargets + 1][3] = {};  // each one's Size, SurfaceInfo, Format (GetHostSurfaceLocked's)
  Targets targets;
};
TargetsReuse g_targets_reuse;
struct ViewportReuse {
  uint64_t seq = 0;
  u32 width = 0, height = 0, guest_width = 0, guest_height = 0, scale = 0, predication = 0;
  size_t tiles = 0;
  ViewportState viewport;
};
ViewportReuse g_viewport_reuse;

// The descriptor words GetHostSurfaceLocked reads for a surface word of the device (0 0 0 for none).
void SurfaceDescWords(u32 va, u32 out[3]) {
  out[0] = out[1] = out[2] = 0;
  if (!va)
    return;
  if (const auto *desc = GuestPtr<const d3d::SurfaceDesc>(va + sizeof(d3d::Resource))) {
    out[0] = desc->Size;
    out[1] = desc->SurfaceInfo;
    out[2] = desc->Format;
  }
}

bool SameTargets(const Targets &a, const Targets &b) {
  if (a.color_count != b.color_count || a.depth != b.depth || a.width != b.width || a.height != b.height ||
      a.scale.code != b.scale.code || a.guest_width != b.guest_width || a.guest_height != b.guest_height)
    return false;
  for (u32 i = 0; i < a.color_count; ++i)
    if (a.colors[i] != b.colors[i])
      return false;
  return true;
}

bool DrawStateLocked(const u8 *device_state, const DrawCall &call, const ReplayConstants *replay) {
  ++g_trace_draws;
  auto &s = State();
  const GuestDevice device(device_state);
  if (!device.valid())
    return false;

  const u32 vs_va = device.U32(d3d::kDeviceOffsetVertexShader);
  auto *vs = FindShaderLocked(vs_va, RecordKind::VertexShader);
  const u32 declaration_va = device.U32(d3d::kDeviceOffsetVertexDeclaration);
  auto *ps = FindShaderLocked(device.U32(d3d::kDeviceOffsetPixelShader), RecordKind::PixelShader);
  // Without a declaration set, the shader fetches as D3DVertexShader_Bind
  // bound it, or else as its own vfetches describe (bundle meshes).
  const VertexDeclarationRecord *declaration = nullptr;
  const u32 *shader_strides = nullptr;  // see GetStreamSource
  if (declaration_va) {
    if (declaration_va != g_last_declaration_va) {
      g_last_declaration =
          FindRecord<VertexDeclarationRecord>(declaration_va, RecordKind::VertexDeclaration);
      // sa2-nx: or one the game's own Direct3D created (shader.cpp).
      if (!g_last_declaration)
        g_last_declaration = const_cast<VertexDeclarationRecord *>(FindGuestDeclarationLocked(declaration_va));
      g_last_declaration_va = g_last_declaration ? declaration_va : 0;
    }
    declaration = g_last_declaration_va == declaration_va ? g_last_declaration : nullptr;
  } else if (vs && vs->bound_declaration) {
    declaration = vs->bound_declaration.get();
    shader_strides = vs->bound_strides;
  } else if (const VertexDeclarationRecord *fetch = vs ? ShaderFetchDeclarationLocked(*vs) : nullptr) {
    declaration = fetch;
    shader_strides = vs->fetch_strides;
  } else {
    declaration = &NoDeclaration();
  }
  // banjo-nx (2026-10-19, render-thread proposal 16): the records this draw reads next, fetched while the code between
  // runs (the shader entries' first lines, the declaration): first touches were most of the draw's sampled loads.
  if (vs && vs->entry) {
    __builtin_prefetch(vs->entry);
    __builtin_prefetch(reinterpret_cast<const char *>(vs->entry) + 64);
  }
  if (ps && ps->entry) {
    __builtin_prefetch(ps->entry);
    __builtin_prefetch(reinterpret_cast<const char *>(ps->entry) + 64);
  }
  if (declaration)
    __builtin_prefetch(declaration);
  if (!vs || !declaration) {
    static uint32_t logged = 0;
    if (LogLimited(logged, 16)) {
      const u32 va = vs ? declaration_va : vs_va;
      const auto *words = GuestPtr<const be_u32>(va);
      RGPU_WARN("Draw dropped: {} {:08X} is not one the renderer knows (words {:08X} {:08X})",
                vs ? "vertex declaration" : "vertex shader", va, words ? u32(words[0]) : 0,
                words ? u32(words[1]) : 0);
    }
    return false;
  }
  // Earlier memexport results reach guest memory before this draw reads them
  // (WaitForExportsLocked may submit the list: before anything is recorded).
  const FloatConstants vs_constants = DrawFloatConstants(device_state, replay, false);
  const FloatConstants ps_constants = DrawFloatConstants(device_state, replay, true);
  // banjo-nx (2026-10-19, render-thread proposal 4): the draw's memexport streams, gathered once here (they were
  // gathered for each pass of the loop below and again for the draw: the same constants each time).
  MemExportStream exports[kMaxMemExportStreams];
  u32 export_count = 0;
  GatherMemExportStreams(vs->entry, vs_constants, exports, export_count);
  if (ps)
    GatherMemExportStreams(ps->entry, ps_constants, exports, export_count);
  // Waiting for an older slot's results leaves the current slot's, which the
  // draw may need as well: another pass checks (none are left after it).
  while (AnyPendingExports()) {
    u32 slots = 0;
    const ExportRead read = DrawNeedsExportFlush(device, call, *vs, ps, *declaration,
                                                 shader_strides, exports, export_count, slots);
    if (read == ExportRead::None)
      break;
    static uint32_t wait_logged = 0;
    if (LogLimited(wait_logged)) {
      static constexpr const char *kReadNames[] = {"none",    "vertex stream", "vertex fetch", "indices",
                                                   "texture", "capacity"};
      RGPU_WARN("Draw waits for memexport results ({}; VS {:016X}, PS {:016X}{})",
                kReadNames[u32(read)], vs->hash, ps ? ps->hash : 0,
                (slots & (1u << s.frame)) ? ", this frame's: GPU round trip" : "");
    }
    WaitForExportsLocked(slots);
  }
  if (!Video::EnsureCommandListLocked())
    return false;
  SyncBound(s);

  plume::RenderShader *host_vs = GetHostShader(vs);
  plume::RenderShader *host_ps = ps ? GetHostShader(ps) : nullptr;
  if (!host_vs || (ps && !host_ps))
    return false;  // missing from the shader cache (captured for the next build)

  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): the previous chain draw's targets when the device's target words (bytes
  // 12816-12835) did not change, no target was built, retired or rebound since (the surface epoch), and the targets'
  // descriptor words (which GetHostSurfaceLocked reads in guest memory) are the same.
  static const bool targets_reuse_on = REXCVAR_GET(sa2_renut_chain_reuse);
  static const bool targets_verify = targets_reuse_on && REXCVAR_GET(sa2_renut_chain_reuse_verify);
  const ChainHint &chain = g_chain_hint;
  Targets targets;
  bool targets_reused = false;
  if (targets_reuse_on && chain.seq && chain.seq == g_targets_reuse.seq + 1 && !ChainChanged(chain, d3d::kDeviceOffsetRenderTargets, d3d::kDeviceOffsetDepthStencil + 3) &&
      g_targets_reuse.epoch == SurfaceEpochLocked()) {
    targets_reused = true;
    for (u32 i = 0; targets_reused && i <= d3d::kMaxRenderTargets; ++i) {
      u32 words[3];
      SurfaceDescWords(g_targets_reuse.va[i], words);
      targets_reused = std::memcmp(words, g_targets_reuse.desc[i], sizeof(words)) == 0;
    }
  }
  if (targets_reused) {
    targets = g_targets_reuse.targets;
    if (targets_verify) {
      Targets made;
      if (!GatherTargets(device, made) || !SameTargets(made, targets)) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused targets differ from those gathered");
        targets = made;
      }
    }
  } else {
    if (!GatherTargets(device, targets)) {
      g_targets_reuse.seq = 0;
      return false;
    }
  }
  if (targets_reuse_on) {
    g_targets_reuse.seq = chain.seq;
    g_targets_reuse.epoch = SurfaceEpochLocked();
    if (!targets_reused) {
      for (u32 i = 0; i <= d3d::kMaxRenderTargets; ++i) {
        g_targets_reuse.va[i] = device.U32(i < d3d::kMaxRenderTargets ? d3d::kDeviceOffsetRenderTargets + i * 4
                                                                       : d3d::kDeviceOffsetDepthStencil);
        SurfaceDescWords(g_targets_reuse.va[i], g_targets_reuse.desc[i]);
      }
      g_targets_reuse.targets = targets;
    }
  }
  // banjo-nx (2026-10-20, NOT exact, banjo_renut_fur_shells = N > 1, the user's call): the fur family's meshes are drawn
  // once a shell, shell by shell (each shell draws every part of the body: RenderDoc of Ryujinx shows the same four
  // parts repeating); of each part's draws in a frame (the same shaders and index range) every Nth is drawn, the first
  // always (tools/nullps_shaders.txt: Banjo's fur and the dialogue portrait's).
  // banjo_renut_portrait_fur_shells: the same for draws into a target of 128 x 128 or less (the dialogue portrait,
  // 96 x 96: 21 shells there cost ~2.5 ms of the heaviest frames, mostly vertex work).
  const bool portrait = targets.width <= 128 && targets.height <= 128;
  if (const uint32_t every = portrait ? banjo::renut_gpu::PortraitFurShellEvery() : banjo::renut_gpu::FurShellEvery();
      every > 1 && ps) {
    const uint64_t h = ps->hash;
    if (h == 0x2F36770769087F99ull || h == 0xC92BA80672FB9774ull || h == 0xB3C10F3D80EF61E2ull ||
        h == 0xD40EBF0336F89333ull) {
      struct ShellPart {
        uint64_t frame = 0;
        const void *vs = nullptr, *ps = nullptr;
        u32 start_index = 0, start_vertex = 0, count = 0;
        i32 base_vertex = 0;
        bool indexed = false;
        bool portrait = false;
        u32 n = 0;
      };
      static ShellPart parts[64];
      const uint64_t frame = TextureFrameLocked();
      ShellPart *part = nullptr, *free_part = nullptr;
      for (ShellPart &p : parts) {
        if (p.frame != frame) {
          if (!free_part)
            free_part = &p;
          continue;
        }
        if (p.vs == vs && p.ps == ps && p.start_index == call.start_index && p.start_vertex == call.start_vertex &&
            p.count == call.count && p.base_vertex == call.base_vertex && p.indexed == call.indexed &&
            p.portrait == portrait) {
          part = &p;
          break;
        }
      }
      bool draw = true;
      if (part) {
        draw = ++part->n % every == 0;
      } else if (free_part) {
        *free_part = {frame, vs, ps, call.start_index, call.start_vertex, call.count, call.base_vertex, call.indexed,
                      portrait, 0};
      }
      banjo::renut_gpu::CountFurShell(draw);
      if (!draw)
        return true;
    }
  }

  // banjo-nx (banjo_gpu_depth_only): before anything of the pixel shader is set up.
  if (ps && DrawsWithoutPixelShader(device, targets, *ps)) {
    ps = nullptr;
    host_ps = nullptr;
    banjo::renut_gpu::CountDepthOnly();
  }

  // Everything that records copies or barriers comes before the framebuffer.
  SharedConstants sc;
  // sa2-nx (2026-10-20, sa2_renut_shared_partial_clear): FillTextures writes all 128 texture and 32 sampler words (its
  // first loop, or the chain's copy of a previous draw's), so only what follows them is zeroed (848 of 1,360 bytes).
  static_assert(offsetof(SharedConstants, booleans) == sizeof(sc.texture_indices) + sizeof(sc.sampler_indices));
  static const bool partial_clear = REXCVAR_GET(sa2_renut_shared_partial_clear);
  if (partial_clear)
    std::memset(&sc.booleans, 0, sizeof(sc) - offsetof(SharedConstants, booleans));
  else
    std::memset(&sc, 0, sizeof(sc));
  sc.flt_max = std::numeric_limits<float>::max();  // banjo-nx: 0x7F7FFFFF, the clamps' bound (shader_common.h)
  const u32 texture_mask =
      (vs->entry ? vs->entry->textureFetchMask : 0) | (ps && ps->entry ? ps->entry->textureFetchMask : 0);
  g_handover_targets = &targets;  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover)
  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): the previous chain draw's texture and sampler words when FillTextures'
  // memo would answer every bound slot as it did then: the device's fetch constants and texture words unchanged
  // (bytes 1152-1775, 12920-13023), the same slots, frame, texture and depth epochs and depth target, and each bound
  // texture's header words the same. The memo hit's work is done again for each slot (its transitions and counters).
  struct TexturesReuse {
    uint64_t seq = 0, frame = 0, epoch = 0, depth_epoch = 0;
    u32 mask = 0;
    const HostSurface *depth = nullptr;
    u32 header[32][6] = {};
    BindEffects fx[32];
    u32 indices[3][32] = {};
    u32 samplers[32] = {};
  };
  static TexturesReuse treuse;
  bool textures_reused = targets_reuse_on && chain.seq && chain.seq == treuse.seq + 1 && treuse.mask == texture_mask &&
                         !ChainChanged(chain, 1152, 1775) && !ChainChanged(chain, d3d::kDeviceOffsetTextures, d3d::kDeviceOffsetTextures + 4 * d3d::kTextureFetchSlots - 1) &&
                         treuse.frame == TextureFrameLocked() && treuse.epoch == g_texture_epoch &&
                         treuse.depth_epoch == g_depth_epoch && treuse.depth == targets.depth;
  for (u32 r = 0; textures_reused && r < d3d::kTextureFetchSlots; ++r) {
    if (!(texture_mask & (1u << r)))
      continue;
    const u32 va = device.U32(d3d::kDeviceOffsetTextures + r * 4);
    const auto *header = va ? GuestPtr<const d3d::BaseTexture>(va) : nullptr;
    textures_reused = header && std::memcmp(treuse.header[r], &header->Format, sizeof(treuse.header[r])) == 0;
  }
  if (textures_reused) {
    for (u32 r = 0; r < 32; ++r) {
      sc.texture_indices[0][r] = treuse.indices[0][r];
      sc.texture_indices[1][r] = treuse.indices[1][r];
      sc.texture_indices[2][r] = treuse.indices[2][r];
      sc.sampler_indices[r] = treuse.samplers[r];
    }
    for (u32 r = 0; r < d3d::kTextureFetchSlots; ++r) {
      if (!(texture_mask & (1u << r)))
        continue;
      const BindEffects &fx = treuse.fx[r];
      for (HostTexture *tex : fx.transition)
        if (tex)
          TransitionTextureLocked(tex, plume::RenderTextureLayout::SHADER_READ);
      if (fx.handover_surface)
        TransitionSurfaceLocked(fx.handover_surface, plume::RenderTextureLayout::SHADER_READ);
      if (fx.handover_image)
        TransitionDepthImageLocked(*fx.handover_image, plume::RenderTextureLayout::SHADER_READ);
      if (fx.float_sampled)
        banjo::renut_gpu::CountDepthFloatSampled();
      if (fx.words_viewed)
        banjo::renut_gpu::CountDepthWords(true);
    }
    if (targets_verify) {
      SharedConstants made{};
      FillTextures(device, texture_mask, made);
      if (std::memcmp(made.texture_indices, sc.texture_indices, sizeof(sc.texture_indices)) != 0 ||
          std::memcmp(made.sampler_indices, sc.sampler_indices, sizeof(sc.sampler_indices)) != 0) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused texture words differ from those filled");
        std::memcpy(sc.texture_indices, made.texture_indices, sizeof(sc.texture_indices));
        std::memcpy(sc.sampler_indices, made.sampler_indices, sizeof(sc.sampler_indices));
      }
    }
  } else {
    bool all_memo = false;
    FillTextures(device, texture_mask, sc, targets_reuse_on ? treuse.fx : nullptr,
                 targets_reuse_on ? &all_memo : nullptr);
    if (targets_reuse_on) {
      // Kept for the next draw only when every slot is one the memo keys (a header), as it then answers them all.
      treuse.seq = all_memo ? chain.seq : 0;
      treuse.mask = texture_mask;
      treuse.frame = TextureFrameLocked();
      treuse.epoch = g_texture_epoch;
      treuse.depth_epoch = g_depth_epoch;
      treuse.depth = targets.depth;
      for (u32 r = 0; r < d3d::kTextureFetchSlots; ++r) {
        if (!(texture_mask & (1u << r)))
          continue;
        const u32 va = device.U32(d3d::kDeviceOffsetTextures + r * 4);
        const auto *header = va ? GuestPtr<const d3d::BaseTexture>(va) : nullptr;
        if (header)
          std::memcpy(treuse.header[r], &header->Format, sizeof(treuse.header[r]));
      }
      for (u32 r = 0; r < 32; ++r) {
        treuse.indices[0][r] = sc.texture_indices[0][r];
        treuse.indices[1][r] = sc.texture_indices[1][r];
        treuse.indices[2][r] = sc.texture_indices[2][r];
        treuse.samplers[r] = sc.sampler_indices[r];
      }
    }
  }
  if (textures_reused && targets_reuse_on)
    treuse.seq = chain.seq;
  g_handover_targets = nullptr;

  const d3d::Device &dev = device.dev();
  const u8 *bools = reinterpret_cast<const u8 *>(&dev.m_Constants.VertexShaderB[0]);
  // banjo-nx (2026-10-19, render-thread proposal 22): SwapEndian(2, ...) (8in32) inline: eight byte swaps, vectorised.
  for (u32 i = 0; i < 8; ++i) {
    u32 word;
    std::memcpy(&word, bools + 4 * i, 4);
    sc.booleans[i] = __builtin_bswap32(word);
  }
  const u32 color_control = dev.m_ControlPacket.ColorControl;
  sc.alpha_test_func = (color_control & 8) ? (color_control & 7) : 7;
  sc.alpha_threshold = dev.m_ValuesPacket.AlphaRef;
  if (sc.alpha_test_func == 7 && ((color_control >> 4) & 1) && banjo::renut_gpu::AlphaToMaskTestOn()) {
    // banjo-nx (banjo_renut_alpha_to_mask_test, 2026-10-18): alpha to coverage on a single-sample target is an alpha
    // test at 0.5 (MakeKey leaves the pipeline's alpha to coverage off for it). NVIDIA GPUs dither alpha to coverage
    // without samples: the checker-patterned grass, fur and foliage edges of the console and Ryujinx runs.
    const HostSurface *first = targets.color_count ? targets.colors[0] : targets.depth;
    if (ps && first && first->samples == plume::RenderSampleCount::COUNT_1) {
      sc.alpha_test_func = 6;  // >=
      sc.alpha_threshold = 0.5f;
    }
  }
  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): the previous chain draw's viewport when nothing ComputeViewport reads
  // changed: the window, values and control packets (bytes 10432-10599), the word at 0x29C0, the predication, the
  // targets' sizes and scale, the tiles and the replay's predication.
  ViewportState viewport;
  const bool viewport_reused =
      targets_reuse_on && chain.seq && chain.seq == g_viewport_reuse.seq + 1 && !ChainChanged(chain, 10432, 10599) &&
      !ChainChanged(chain, 0x29C0, 0x29C3) && !ChainChanged(chain, d3d::kDeviceOffsetPredication,
                                                          d3d::kDeviceOffsetPredication + 3) &&
      g_viewport_reuse.width == targets.width && g_viewport_reuse.height == targets.height &&
      g_viewport_reuse.guest_width == targets.guest_width && g_viewport_reuse.guest_height == targets.guest_height &&
      g_viewport_reuse.scale == targets.scale.code && g_viewport_reuse.tiles == g_tiles.size() &&
      g_viewport_reuse.predication == g_replay_predication;
  if (viewport_reused) {
    viewport = g_viewport_reuse.viewport;
    if (targets_verify) {
      const ViewportState made = ComputeViewport(device, targets);
      if (std::memcmp(&made, &viewport, sizeof(made)) != 0) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused viewport differs from the one computed");
        viewport = made;
      }
    }
  } else {
    viewport = ComputeViewport(device, targets);
  }
  if (targets_reuse_on) {
    g_viewport_reuse.seq = chain.seq;
    g_viewport_reuse.width = targets.width;
    g_viewport_reuse.height = targets.height;
    g_viewport_reuse.guest_width = targets.guest_width;
    g_viewport_reuse.guest_height = targets.guest_height;
    g_viewport_reuse.scale = targets.scale.code;
    g_viewport_reuse.tiles = g_tiles.size();
    g_viewport_reuse.predication = g_replay_predication;
    g_viewport_reuse.viewport = viewport;
  }
  sc.screen_space = viewport.screen_space ? 1 : 0;
  std::memcpy(sc.screen_scale, viewport.screen_scale, sizeof(sc.screen_scale));
  std::memcpy(sc.screen_offset, viewport.screen_offset, sizeof(sc.screen_offset));
  sc.viewport_size[0] = viewport.guest_size[0];
  sc.viewport_size[1] = viewport.guest_size[1];
  sc.resolution_scale = targets.scale.Factor();
  std::memcpy(sc.pixel_origin, viewport.pixel_origin, sizeof(sc.pixel_origin));  // Renut upstream 574b488
  if (call.primitive == kPointList) {
    // PA_SU_POINT_SIZE: height 0-15, width 16-31, half the size in 12.4 fixed
    // point (SetRenderState_PointSize 0x8264C6C0 stores the size * 8);
    // PA_SU_POINT_MINMAX: minimum 0-15, maximum 16-31, the size in 12.4
    // (SetRenderState_PointSizeMin 0x8264C720 stores the minimum * 16).
    const auto *tessellator = reinterpret_cast<const be_u32 *>(dev.m_TessellatorPacket);
    const u32 point_size = tessellator[0], point_min_max = tessellator[1];
    sc.point_state[0] = float(point_size >> 16) / 8.0f;
    sc.point_state[1] = float(point_size & 0xFFFF) / 8.0f;
    // banjo-nx (2026-10-18, the depth of field's bokeh sprites): the minimum and maximum are radii in 12.4 as well, as
    // the hardware (and Xenia, the emulated renderer: point_vertex_diameter_min/max = value * 2 / 16) reads them; / 16
    // halved the largest sprites.
    const float minmax_scale = banjo::renut_gpu::PointMinMaxRadiusOn() ? 8.0f : 16.0f;
    sc.point_state[2] = float(point_min_max & 0xFFFF) / minmax_scale;
    sc.point_state[3] = float(point_min_max >> 16) / minmax_scale;
  }

  Geometry geometry;
  if (call.tessellation ? !PrepareTessellation(device, call, geometry, sc.tessellation)
                        : !PrepareGeometry(device, call, geometry)) {
    return false;
  }
  // The host draws the vertex range in use from vertex 0 (indices rebased to
  // it in PrepareGeometry); the guest's r0.x is the vertex index itself.
  if (!call.tessellation)
    sc.tessellation[3] = geometry.first_vertex;
  GpuStreams gpu_streams;
  if (!ConvertExportedStreamsLocked(s, device, call, *vs, shader_strides, *declaration, geometry,
                                    gpu_streams)) {
    return false;
  }

  const u32 fetch_mask = (vs->entry ? vs->entry->vertexFetchMemoryMask : 0) |
                         DescribeBoundFetches(device, call, *vs, *declaration, shader_strides, sc);
  {
    static const uint64_t want = [] {
      const std::string v = REXCVAR_GET(banjo_renut_diag_stream_dump_vs);
      return v.empty() ? 0ull : uint64_t(std::strtoull(v.c_str(), nullptr, 16));
    }();
    g_dump_fetch = want && vs->hash == want;
  }
  UploadAllocation fetch_alloc = UploadVertexFetchData(device, call, fetch_mask, sc);
  if (g_dump_fetch) {
    static uint32_t logged = 0;
    if (logged++ < 24)
      RGPU_INFO("[fetchdump] draw: mask {:08X} fetch data {:X} bytes {} prim {} count {} tess {}", fetch_mask,
                sc.vertex_fetch_data, sc.vertex_fetch_bytes, call.primitive, call.count, call.tessellation);
  }
  g_dump_fetch = false;
#if defined(PLUME_DEKO3D)
  // banjo-nx (deko3d renderer): the vertex shaders read the fetch data through a storage binding of its range alone,
  // indexed by the address's low word (shader_common.h, g_DkAddressWindow): the binding starts at the address rounded
  // down to 256 bytes, and the word becomes the offset from there. One binding over every buffer (runs 14-15) cost
  // Ryujinx ~12 s a frame: it brings a storage binding's whole range up to date with guest memory at each draw.
  // sa2-nx (2026-10-20, sa2_renut_fetch_data_unread): a draw whose shaders fetch nothing by address (fetch_mask 0:
  // the vertex shader's memory fetches and its bound fetches) never reads the fetch data's address or size: they stay
  // 0 instead of this draw's new allocation, so the shared constants' line with them does not change at every draw,
  // and no binding is made.
  static const bool fetch_data_unread = REXCVAR_GET(sa2_renut_fetch_data_unread);
  if (IsDeko3D() && fetch_mask == 0 && fetch_data_unread) {
    sc.vertex_fetch_data = 0;
    sc.vertex_fetch_bytes = 0;
  } else if (IsDeko3D() && sc.vertex_fetch_data) {
    const uint64_t address = sc.vertex_fetch_data;
    const uint64_t base = address & ~uint64_t(0xFF);
    plume::Deko3DBindFetchData(s.command_list, base, u32(address - base) + sc.vertex_fetch_bytes);
    sc.vertex_fetch_data = address - base;
    // banjo-nx (2026-10-20): the 16-byte multiples tools/deko3d_glsl_opt.py's aw4 pass relies on.
    if (DiagCountsOn()) {  // sa2-nx (2026-10-20, sa2_renut_lean_diagnostics)
      u32 bits = u32(sc.vertex_fetch_data) | sc.vertex_fetch_bytes;
      for (u32 m = fetch_mask; m; m &= m - 1)
        bits |= sc.vertex_fetch_slots[64 + std::countr_zero(m)] & ~3u;
      banjo::renut_gpu::CountFetchAlignment((bits & 15u) == 0);
    }
  }
#endif
  static constexpr u32 kAllRegisters[8] = {~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u};
  static constexpr u32 kNoRegisters[8] = {};
  const u32 *vs_used = vs->entry ? vs->entry->floatConstants : kAllRegisters;
  const u32 *ps_used = !ps ? kNoRegisters : ps->entry ? ps->entry->floatConstants : kAllRegisters;
  // banjo-nx (2026-10-19, banjo_deko3d_push_constants): pushed, except for draws point_sprite_gs expands (it reads
  // SharedConstants through its address).
#if defined(PLUME_DEKO3D)
  const bool pushed = IsDeko3D() && banjo::renut_gpu::PushConstantsOn() &&
                      !(call.primitive == kPointList && s.point_sprite_gs) && EnsurePushedBuffers();
#else
  constexpr bool pushed = false;
#endif
  UploadAllocation vs_alloc, ps_alloc;
  if (pushed) {
#if defined(PLUME_DEKO3D)
    plume::RenderCommandList *cmd = s.command_list;
    PushFloatConstants(cmd, vs_constants, vs_used, g_bound.pushed_vs, vs->entry, g_pushed_buffers.address);
    PushFloatConstants(cmd, ps_constants, ps_used, g_bound.pushed_ps, ps ? ps->entry : nullptr,
                       g_pushed_buffers.address + kPushedFloatStride);
    // The replay's dirty masks were cleared for the pushed models: the ring uploads' tracking of them no longer holds.
    g_bound.vs_constants.replay = 0;
    g_bound.ps_constants.replay = 0;
    banjo::renut_gpu::CountPushedConstants(1, 2);
#endif
  } else {
    vs_alloc = UploadFloatConstants(vs_constants, vs_used, g_bound.vs_constants, vs->entry);
    ps_alloc = UploadFloatConstants(ps_constants, ps_used, g_bound.ps_constants, ps ? ps->entry : nullptr);
    // banjo-nx (2026-10-19): and the other way round: the uploads cleared the dirty masks the pushed models track.
    g_bound.pushed_vs.replay = 0;
    g_bound.pushed_ps.replay = 0;
  }
  u32 export_offsets[kMaxMemExportStreams] = {};
  if (export_count &&!PrepareMemExportLocked(s, exports, export_count, sc, export_offsets)) {
    export_count = 0;
    std::memset(sc.memexport_streams, 0, sizeof(sc.memexport_streams));
  }
  plume::RenderBuffer *memexport = MemExportBufferLocked(s);
  if (memexport && UsesVulkanBindings())  // banjo-nx: and deko3d
    sc.memexport_address = memexport->getDeviceAddress();
#if defined(PLUME_DEKO3D)
  // banjo-nx (deko3d renderer): the deko3d shaders write memexport through a binding of the buffer itself.
  if (memexport && IsDeko3D())
    plume::Deko3DBindMemExport(s.command_list, memexport);
#endif
  UploadAllocation sc_alloc;
  if (pushed) {
#if defined(PLUME_DEKO3D)
    // sa2-nx (2026-10-20, sa2_renut_chain_reuse): a chain draw whose texture words are the previous draw's, when that
    // draw pushed last (no push since, the same command list's buffer): its first 512 bytes are held already.
    static uint64_t texture_push_seq = 0, texture_push_count = 0;
    const bool texture_lines_held = textures_reused && !targets_verify && chain.seq &&
                                    texture_push_seq + 1 == chain.seq && texture_push_count == g_shared_pushes &&
                                    g_bound.pushed_shared_valid;
    // sa2-nx (2026-10-20, sa2_renut_fetch_slots_skip): no fetches in this draw and none in the last push: both have
    // the fetch slots zero (sc is zeroed there; only UploadVertexFetchData writes them, for the fetch mask's bits).
    static const bool fetch_slots_skip = REXCVAR_GET(sa2_renut_fetch_slots_skip);
    const bool fetch_slots_equal = fetch_slots_skip && fetch_mask == 0 && g_held_fetch_slots_zero &&
                                   g_bound.pushed_shared_valid;
    PushSharedConstants(s.command_list, sc,
                        (texture_lines_held ? kTextureLines : 0u) | (fetch_slots_equal ? kFetchSlotLines : 0u));
    g_held_fetch_slots_zero = fetch_mask == 0;  // the buffer holds this draw's now
    texture_push_seq = chain.seq;
    texture_push_count = g_shared_pushes;
#endif
    // The 10 s line's draw count is the uploads plus the reuses: a pushed draw counts as an upload (package 35 logged
    // about 2,800 draws where there were 250,000).
    banjo::renut_gpu::CountSharedConstants(false);
  } else if (g_bound.shared_valid && banjo::renut_gpu::SharedDedupeOn() &&
      std::memcmp(&g_bound.shared_bytes, &sc, sizeof(sc)) == 0) {
    sc_alloc = g_bound.shared_alloc;  // banjo-nx: the same bytes are in this list's last upload
    banjo::renut_gpu::CountSharedConstants(true);
  } else {
    sc_alloc = Video::UploadsLocked().Upload(&sc, sizeof(sc), 256);
    if (sc_alloc) {
      g_bound.shared_alloc = sc_alloc;
      std::memcpy(&g_bound.shared_bytes, &sc, sizeof(sc));
      g_bound.shared_valid = true;
    }
    banjo::renut_gpu::CountSharedConstants(false);
  }
  if (!fetch_alloc || (!pushed && (!vs_alloc || !ps_alloc || !sc_alloc)) || !memexport)
    return false;

  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): a chain draw takes the previous draw's key when nothing MakeKey reads
  // changed: the device's bool constants and render state packets (bytes 10112-10599: VertexShaderB / PixelShaderB,
  // the clip planes, the destination, window, values and control packets), the shaders' cache entries, the
  // declaration's layout, the primitive, the instancing, the alpha test, the targets (their surfaces and the surface
  // epoch, which every target built, retired or rebound bumps) and the in-run A/B half.
  struct KeyReuse {
    uint64_t seq = 0;
    const ShaderCacheEntry *vs_entry = nullptr, *ps_entry = nullptr;
    uint64_t layout_hash = 0, surface_epoch = 0;
    u32 primitive = 0, alpha_test_func = 0, color_count = 0;
    bool per_instance = false;
    int ab_off = 0;
    HostSurface *colors[d3d::kMaxRenderTargets] = {};
    HostSurface *depth = nullptr;
    PipelineKey key{};
  };
  static KeyReuse reuse;
  static const bool reuse_on = REXCVAR_GET(sa2_renut_chain_reuse);
  static const bool reuse_verify = reuse_on && REXCVAR_GET(sa2_renut_chain_reuse_verify);
  const ChainHint &hint = chain;
  const ShaderCacheEntry *const ps_entry = ps ? ps->entry : nullptr;
  const int ab_off = banjo::renut_gpu::g_ab_off.load(std::memory_order_relaxed);
  const uint64_t surface_epoch = SurfaceEpochLocked();
  bool reused = reuse_on && hint.seq && hint.seq == reuse.seq + 1 && !ChainChanged(hint, 10112, 10599) &&
                reuse.vs_entry == vs->entry && reuse.ps_entry == ps_entry &&
                reuse.layout_hash == declaration->layout_hash && reuse.primitive == call.primitive &&
                reuse.per_instance == geometry.per_instance && reuse.alpha_test_func == sc.alpha_test_func &&
                reuse.color_count == targets.color_count && reuse.depth == targets.depth &&
                reuse.surface_epoch == surface_epoch && reuse.ab_off == ab_off;
  for (u32 i = 0; reused && i < targets.color_count; ++i)
    reused = reuse.colors[i] == targets.colors[i];
  PipelineKey key;
  if (reused) {
    key = reuse.key;
    if (reuse_verify) {
      const PipelineKey made = MakeKey(device, vs, ps, declaration, targets, call.primitive, geometry.per_instance,
                                       sc.booleans, sc.alpha_test_func);
      if (std::memcmp(&made, &key, sizeof(key)) != 0) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused pipeline key differs from the one made (VS {:016X})",
                    vs->hash);
        key = made;
      }
    }
  } else {
    key = MakeKey(device, vs, ps, declaration, targets, call.primitive, geometry.per_instance, sc.booleans,
                  sc.alpha_test_func);
  }
  if (reuse_on) {
    reuse.seq = hint.seq;
    reuse.vs_entry = vs->entry;
    reuse.ps_entry = ps_entry;
    reuse.layout_hash = declaration->layout_hash;
    reuse.surface_epoch = surface_epoch;
    reuse.primitive = call.primitive;
    reuse.alpha_test_func = sc.alpha_test_func;
    reuse.color_count = targets.color_count;
    reuse.per_instance = geometry.per_instance;
    reuse.ab_off = ab_off;
    for (u32 i = 0; i < d3d::kMaxRenderTargets; ++i)
      reuse.colors[i] = i < targets.color_count ? targets.colors[i] : nullptr;
    reuse.depth = targets.depth;
    if (!reused)
      reuse.key = key;
  }
  if (DiagCountsOn()) {  // sa2-nx (2026-10-20, sa2_renut_lean_diagnostics)
    // banjo-nx (package 28, a census): draws that write no colour, nothing at all, or part of a target's channels.
    const u32 used_mask = key.color_count ? key.color_mask & ((1u << (4 * key.color_count)) - 1u) : 0u;
    const bool no_colour = used_mask == 0;
    const u32 dc = key.depth_control;
    const bool depth = key.depth_format != plume::RenderFormat::UNKNOWN;
    const bool depth_writes = depth && (dc & 2) && (dc & 4);
    const u32 stencil_write_mask = (key.stencil_ref_mask >> 16) & 0xFF;
    const bool stencil_ops = ((dc >> 11) & 0x1FF) != 0 || ((dc >> 20) & 0x1FF) != 0;
    const bool stencil_writes = depth && (dc & 1) && stencil_write_mask && stencil_ops;
    bool partial = false;
    for (u32 i = 0; i < key.color_count; ++i) {
      const u32 m = (key.color_mask >> (4 * i)) & 0xF;
      partial |= m != 0 && m != 0xF;
    }
    const bool exports = (vs->entry && vs->entry->memExportConstants[0]) || (ps && ps->entry && ps->entry->memExportConstants[0]);
    banjo::renut_gpu::CountDrawKind(no_colour && depth, no_colour && depth && sc.alpha_test_func == 7 &&
                                    !key.alpha_to_coverage && ps != nullptr,
                                    no_colour && !depth_writes && !stencil_writes && !exports, partial);
  }
  // sa2-nx (2026-10-20, sa2_renut_chain_reuse): the previous draw's pipeline for its key again when no lookup was made
  // since and no prebuilt pipeline came in (PipelineLookupsLocked): GetPipelineLocked would answer the same.
  struct PipelineReuse {
    uint64_t seq = 0, lookups = 0, epoch = 0;
    plume::RenderPipeline *pipeline = nullptr;
  };
  static PipelineReuse preuse;
  plume::RenderPipeline *pipeline = nullptr;
  // The prebuild epoch read once, before the test and the lookup (found by banjo-nx's verify run): a prebuild worker
  // finishing a specialised pipeline between this draw's lookup (which returned the generic one) and a later read
  // would put its bump into the stored epoch, and the next draw would reuse the generic pipeline.
  const uint64_t prebuild_epoch = PrebuildEpochLocked();
  if (reused && preuse.pipeline && preuse.seq + 1 == hint.seq && preuse.lookups == PipelineLookupsLocked() &&
      preuse.epoch == prebuild_epoch) {
    pipeline = preuse.pipeline;
    if (reuse_verify) {
      plume::RenderPipeline *made = GetPipelineLocked(key, {host_vs, host_ps, &declaration->layout});
      if (made != pipeline) {
        static uint32_t logged = 0;
        if (logged++ < 8)
          RGPU_WARN("sa2_renut_chain_reuse_verify: the reused pipeline differs from the one looked up");
        pipeline = made;
      }
    }
  } else {
    pipeline = GetPipelineLocked(key, {host_vs, host_ps, &declaration->layout});
  }
  if (reuse_on)
    preuse = {hint.seq, PipelineLookupsLocked(), prebuild_epoch, pipeline};
  if (!pipeline || !BindTargets(targets))
    return false;
  plume::RenderCommandList *cmd = s.command_list;
  SetViewport(cmd, viewport.viewport, viewport.scissor, targets.color_count);

  if (!g_bound.layout) {
    cmd->setGraphicsPipelineLayout(s.draw_layout.get());
    cmd->setGraphicsDescriptorSet(s.texture_set.get(), 0);
    cmd->setGraphicsDescriptorSet(s.texture_set.get(), 1);
    cmd->setGraphicsDescriptorSet(s.texture_set.get(), 2);
    cmd->setGraphicsDescriptorSet(s.sampler_set.get(), 3);
    cmd->setGraphicsDescriptorSet(s.texture_set.get(), 4);
    g_bound.layout = true;
    g_bound.pipeline = nullptr;
    std::fill(std::begin(g_bound.root_set), std::end(g_bound.root_set), false);
    std::fill(std::begin(g_bound.constant_sets), std::end(g_bound.constant_sets), nullptr);
  }
  if (g_bound.pipeline != pipeline) {
    // banjo-nx: NVK's prefetch hint (the Switch); the record worker gives it itself, on the core that binds.
    if (!banjo::renut_gpu::record::IsProxy(cmd))
      banjo::renut_gpu::BeforePipelineBind(pipeline);
    cmd->setPipeline(pipeline);
    g_bound.pipeline = pipeline;
    // Vulkan: a pipeline without the constant as dynamic state overwrites it.
    if (IsVulkan())
      g_bound.blend_constant_set = false;
  }
  float blend_constant[4];
  if (DrawBlendConstant(dev, key, blend_constant) &&
      !(g_bound.blend_constant_set &&
        std::memcmp(blend_constant, g_bound.blend_constant, sizeof(blend_constant)) == 0)) {
    if (SetBlendConstant(cmd, blend_constant)) {
      std::memcpy(g_bound.blend_constant, blend_constant, sizeof(blend_constant));
      g_bound.blend_constant_set = true;
    } else {
      static bool logged = false;
      if (!logged) {
        logged = true;
        RGPU_WARN("Draws blend with the blend constant, which this backend cannot set");
      }
    }
  }
  if (UsesVulkanBindings()) {  // banjo-nx: and deko3d
    // banjo-nx (banjo_renut_push_point_only): with the constants in uniform buffers only point_sprite_gs reads the
    // push constant, so the other draws do not push it (NVK: a root-table upload a draw less).
    if (key.point_sprite || !banjo::renut_gpu::PushPointOnly()) {
      const DrawPushConstants push{vs_alloc.gpu_address, ps_alloc.gpu_address, sc_alloc.gpu_address};
      cmd->setGraphicsPushConstants(0, &push, 0, sizeof(push));
    }
    // banjo-nx: what the guest shaders read.
    if (pushed) {
#if defined(PLUME_DEKO3D)
      // banjo-nx (2026-10-19, banjo_deko3d_push_constants): the three fixed buffers (bindings 0, 1, 2), and bound
      // textures read from the renderer's copy of the shared constants.
      plume::Deko3DBindUniform(cmd, 0, g_pushed_buffers.address, kFloatConstantsUboBytes);
      plume::Deko3DBindUniform(cmd, 1, g_pushed_buffers.address + kPushedFloatStride, kFloatConstantsUboBytes);
      plume::Deko3DBindUniform(cmd, 2, g_pushed_buffers.address + kPushedSharedOffset, u32(sizeof(SharedConstants)));
      std::fill(std::begin(g_bound.constant_sets), std::end(g_bound.constant_sets), nullptr);  // rebind after
      if (!g_bound.shared_words_set) {
        plume::Deko3DSetSharedWords(cmd, &g_bound.pushed_shared, u32(sizeof(SharedConstants)));
        g_bound.shared_words_set = true;
      }
#endif
    } else {
#if defined(PLUME_DEKO3D)
      if (g_bound.shared_words_set) {
        plume::Deko3DSetSharedWords(cmd, nullptr, 0);
        g_bound.shared_words_set = false;
      }
#endif
      BindConstantSet(cmd, kVertexConstantsSet, vs_alloc.float_constants_set, vs_alloc.ref.offset);
      BindConstantSet(cmd, kPixelConstantsSet, ps_alloc.float_constants_set, ps_alloc.ref.offset);
      BindConstantSet(cmd, kSharedConstantsSet, sc_alloc.shared_constants_set, sc_alloc.ref.offset);
    }
  } else {
    SetRootDescriptor(cmd, vs_alloc.ref, kDrawRootVertexConstants);
    SetRootDescriptor(cmd, ps_alloc.ref, kDrawRootPixelConstants);
    SetRootDescriptor(cmd, sc_alloc.ref, kDrawRootSharedConstants);
    SetRootDescriptor(cmd, fetch_alloc.ref, kDrawRootVertexFetchData);
    SetRootDescriptor(cmd, plume::RenderBufferReference(memexport), kDrawRootMemExport);
  }

  if (!BindStreams(device, call, *vs, shader_strides, declaration->layout,
                   declaration->layout_hash, geometry, gpu_streams)) {
    return false;
  }
  if (gpu_profile::Timed()) {
    // banjo-nx (package 28b): the GPU profile's draws a frame, and with banjo_renut_diag_gpu_shaders its label.
    // banjo-nx (2026-10-19, render-thread proposal 7): only in the frames the profile times.
    const uint32_t label = banjo::renut_gpu::DiagGpuShadersOn() ? ShaderProfileLabel(targets, ps, key)
                                                                : TargetsProfileLabel(targets);
    if (banjo::renut_gpu::DiagGpuShadersOn())
      gpu_profile::Mark(s, label);
    gpu_profile::CountDraw(s, label);
  }
  // sa2-nx (2026-10-20, sa2_renut_lean_diagnostics): the draw log's check once, not a call per draw.
  static const bool draw_log = !REXCVAR_GET(sa2_renut_lean_diagnostics) ||
                               REXCVAR_GET(banjo_renut_diag_draw_log_s) > 0;
  if (draw_log)
    DrawLog(call, geometry, vs, ps, targets, key, sc, export_count);
    // mk9-nx (2026-10-21): bring-up diagnostic, with the draw log: the pixel shader's registers 0-5 as the draw has them.
    if (static const int32_t mk9_at = REXCVAR_GET(banjo_renut_diag_draw_log_s); mk9_at > 0) {
      static uint32_t mk9_n = 0;
      if (mk9_n < 400 && TextureFrameLocked() > 0) {
        auto f = [&](u32 r, u32 c) { return float(*reinterpret_cast<const be_f32 *>(ps_constants.Register(r) + c * 4)); };
        static auto start = std::chrono::steady_clock::now();
        if (std::chrono::steady_clock::now() - start > std::chrono::seconds(mk9_at)) {
          ++mk9_n;
          RGPU_INFO("[mk9] ps {:016X} c0 {} {} {} {} c3 {} {} {} {} c4 {} c5 {}", ps ? ps->hash : 0, f(0, 0), f(0, 1),
                    f(0, 2), f(0, 3), f(3, 0), f(3, 1), f(3, 2), f(3, 3), f(4, 0), f(5, 0));
        }
      }
    }
  // banjo-nx (2026-10-19, banjo_renut_resolve_skip): the targets this draw can write.
  for (u32 i = 0; i < targets.color_count && i < key.color_count; ++i)
    if ((key.color_mask >> (4 * i)) & 0xF)
      ++targets.colors[i]->writes;
  if (targets.depth) {
    const u32 dc = key.depth_control;
    const bool depth_writes = (dc & 2) && (dc & 4);
    const bool stencil_writes = (dc & 1) && ((key.stencil_ref_mask >> 16) & 0xFF) != 0;
    if (depth_writes || stencil_writes)
      ++targets.depth->writes;
    if (dc & 1)
      targets.depth->stencil_used = true;  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover)
  }
  Video::NoteDrawLocked();  // banjo-nx (2026-10-19, banjo_deko3d_kick_submit): the open list's draws
  if (geometry.indexed) {
    SetIndexBuffer(cmd, geometry.index_view);
    // banjo-nx: instanced for a tessellation grid (banjo_gpu_tess_grid); 1 for every other indexed draw.
    cmd->drawIndexedInstanced(geometry.index_count, geometry.instance_count, 0, 0, 0);
  } else {
    cmd->drawInstanced(geometry.vertex_count, geometry.instance_count, 0, 0);
  }
  if (export_count) {
    FinishMemExportLocked(s, exports, export_count, export_offsets);
    g_bound.framebuffer = nullptr;  // the copies end any render pass
  }
  return true;
}

namespace detail {

void ReleaseCacheCopiesLocked(uint32_t slot) {
  g_copy_pool.ReleaseSlot(slot);
}

void CompleteMemExportsLocked(uint32_t slot) {
  SlotExports &exports = g_exports[slot];
  if (exports.pending.empty())
    return;
  auto &s = State();
  plume::RenderBuffer *readback = s.memexport_readback[slot].get();
  if (const auto *data = readback ? static_cast<const u8 *>(readback->map()) : nullptr) {
    for (const PendingExport &p : exports.pending) {
      if (u8 *guest = GuestPhysicalPtr<u8>(p.base)) {
        std::memcpy(guest, data + p.offset, p.size);
        // banjo-nx: the store cards (sa2-nx: SA2's generated code marks them too since 2026-10-20).
        rex::memory::MarkStoreCardsAt(StoreCards(), 0xA0000000u + (p.base & 0x1FFFFFFFu), p.size);
      }
    }
    readback->unmap();
  }
  exports.pending.clear();
  exports.used = 0;
  RebuildExportGranules();
  // banjo-nx (2026-10-17): guest memory changed under the draws: cached copies are proven again
  // (banjo_renut_exact_streams), as after a kick.
  ++g_flush_serial;
}

} // namespace detail

void ForgetVertexDeclarationLocked(u32 declaration_va) {
  if (declaration_va == g_last_declaration_va) {
    g_last_declaration_va = 0;
    g_last_declaration = nullptr;
  }
}

void AdvanceStreamCacheFrameLocked() {
  banjo::renut_gpu::ResolveTraceFrameStart();
  // banjo-nx (2026-10-19): banjo_renut_store_cards_verify's diagnostics, every 10 s.
  if (banjo::renut_gpu::StoreCardsVerifyOn()) {
    static auto last = std::chrono::steady_clock::now();
    const auto now = std::chrono::steady_clock::now();
    if (now - last >= std::chrono::seconds(10)) {
      last = now;
      uint64_t host[3] = {};
#if !defined(__SWITCH__)
      for (int i = 0; i < 3; ++i)
        host[i] = RexStoreCardsHostBytes(i);
#endif
      // Written cards in each view's part of the table (0: written): A, C, E (512 MB each) and 7F (16 MB).
      uint32_t zero[4] = {};
      if (const u8 *cards = StoreCards()) {
        for (u32 i = 0; i < 0x20000u; ++i) {
          zero[0] += cards[0xA0000u + i] == 0;
          zero[1] += cards[0xC0000u + i] == 0;
          zero[2] += cards[0xE0000u + i] == 0;
        }
        for (u32 i = 0; i < 0x1000u; ++i)
          zero[3] += cards[0x7F000u + i] == 0;
      }
      std::string samples = fmt::format(" (written cards now: A {} C {} E {} 7F {} of 131072/131072/131072/4096)",
                                        zero[0], zero[1], zero[2], zero[3]);
      for (uint32_t i = 0; i < g_card_sample_count; ++i)
        samples += fmt::format(" {:05X}{}{}{}{}", g_card_samples[i] & 0xFFFFFFF, (g_card_samples[i] >> 28) & 1 ? "a" : "",
                               (g_card_samples[i] >> 29) & 1 ? "c" : "", (g_card_samples[i] >> 30) & 1 ? "e" : "",
                               (g_card_samples[i] >> 31) & 1 ? "f" : "");
      RGPU_INFO("[cards] 10 s: {} pages read, {} found written (view A {}, C {}, E {}, 7F {}); {} copies written only "
                "through the page before; {} hashed after a written page with their bytes unchanged; host writers "
                "marked {:.1f} MB of the physical views and {:.1f} MB elsewhere in {} calls; {} checks of memory outside the "
                "physical views; first pages:{}",
                g_card_diag[0], g_card_diag[1], g_card_diag[2], g_card_diag[3], g_card_diag[4], g_card_diag[5],
                g_card_diag[6], g_card_diag[7], double(host[0]) / double(1u << 20), double(host[1]) / double(1u << 20),
                host[2], g_card_diag[8], samples);
      std::fill(std::begin(g_card_diag), std::end(g_card_diag), 0);
      g_card_sample_count = 0;
    }
  }
  g_trace_draws = 0;
  // Copies whose guest bytes changed (ContentVerifier), unless made again
  // since they were checked.
  static std::vector<decltype(g_stream_verifier)::Check> stale;
  g_stream_verifier.Flush();
  g_stream_verifier.TakeStale(stale);
  for (const auto &check : stale) {
    auto it = g_stream_cache.find(check.key);
    if (it == g_stream_cache.end())
      continue;
    CachedStream &c = it->second;
    if (!c.copy || c.hash != check.hash)
      continue;
    DropCachedCopy(c);
    c.stable = 0;
    c.checked_frame = UINT64_MAX;
    c.dynamic = ++c.drops >= kMaxCacheDrops;
    banjo::renut_gpu::CountLateStale(1);  // banjo-nx: drawn from its old copy until now
  }
  stale.clear();

  static std::vector<decltype(g_index_verifier)::Check> stale_indices;
  g_index_verifier.Flush();
  g_index_verifier.TakeStale(stale_indices);
  for (const auto &check : stale_indices) {
    auto it = g_index_cache.find(check.key);
    if (it == g_index_cache.end())
      continue;
    CachedIndices &c = it->second;
    if (!c.copy || c.hash != check.hash)
      continue;
    DropCachedIndices(c);
    c.stable = 0;
    c.checked_frame = UINT64_MAX;  // hashed again by TrackIndices
    c.dynamic = ++c.drops >= kMaxCacheDrops;
    banjo::renut_gpu::CountLateStale(1);  // banjo-nx: drawn from its old copy until now
  }
  stale_indices.clear();

  if (++g_stream_cache_frame % 60)
    return;
  // The memos may hold entries about to go.
  g_stream_memo.Clear();
  g_index_memo.Clear();
  // banjo-nx: banjo_renut_memory_policy's idle time (Renut's kStreamCacheIdleFrames without it).
  const uint64_t idle_frames = banjo::renut_gpu::StreamCacheIdleFrames(kStreamCacheIdleFrames);
  std::erase_if(g_stream_cache, [idle_frames](auto &entry) {
    if (g_stream_cache_frame - entry.second.last_used < idle_frames)
      return false;
    DropCachedCopy(entry.second);
    return true;
  });
  std::erase_if(g_index_cache, [idle_frames](auto &entry) {
    if (g_stream_cache_frame - entry.second.last_used < idle_frames)
      return false;
    DropCachedIndices(entry.second);
    return true;
  });
}

StreamCacheMemory StreamCacheMemoryLocked() {
  StreamCacheMemory usage;
  usage.pool_bytes = g_copy_pool.Bytes();
  usage.cached_bytes = g_stream_cache_bytes;
  usage.streams = g_stream_cache.size();
  usage.index_ranges = g_index_cache.size();
  return usage;
}

namespace {

// Clears on explicit targets (framebuffer bound as for a draw). A color clear
// reaches the targets over the same EDRAM in other host formats too
// (ColorAliasesLocked): on the console the clear writes the memory, whatever
// format a surface reads it as. N&B's paint shop clears its glow buffer
// through the 8-bit view of that EDRAM, then adds the glow particles through a
// 16-bit float view; the float target kept the frames before, and the glow
// piled up into drips.
void ClearTargets(const Targets &t, u32 flags, const plume::RenderRect &rect, const float color[4],
                  float depth, u32 stencil) {
  if (banjo::renut_gpu::ResolveTraceOn()) {
    RGPU_INFO("[trace] clear {:X} rect ({},{})-({},{}) of {}x{}", flags, rect.left, rect.top, rect.right, rect.bottom,
              t.width, t.height);
  }
  if (rect.right <= rect.left || rect.bottom <= rect.top)
    return;
  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover): before the target is bound for the clear.
  if (t.depth && t.depth->handed)
    DepthHandOverBeforeBindLocked(t.depth, &rect, flags);
  if (t.depth && (flags & 0x20))
    t.depth->stencil_used = true;
  if (!BindTargets(t))
    return;
  plume::RenderCommandList *cmd = State().command_list;
  const plume::RenderRect host_rect = ScaleRect(rect, t.scale);
  const plume::RenderColor clear_color(color[0], color[1], color[2], color[3]);
  // banjo-nx (2026-10-19, render-thread proposal 20): scratch vectors kept between clears (render thread only).
  static std::vector<HostSurface *> aliases;
  static std::vector<HostSurface *> found;
  aliases.clear();
  for (uint32_t i = 0; i < t.color_count; ++i) {
    if (!(flags & (1u << i)))
      continue;
    ++t.colors[i]->writes;  // banjo-nx (banjo_renut_resolve_skip)
    cmd->clearColor(i, clear_color, &host_rect, 1);
    ColorAliasesLocked(t.colors[i], found);
    for (HostSurface *alias : found) {
      if (std::find(aliases.begin(), aliases.end(), alias) == aliases.end() &&
          std::find(t.colors, t.colors + t.color_count, alias) == t.colors + t.color_count)
        aliases.push_back(alias);
    }
  }
  if (t.depth && (flags & 0x30)) {
    ++t.depth->writes;  // banjo-nx (banjo_renut_resolve_skip)
    cmd->clearDepthStencil((flags & 0x10) != 0, (flags & 0x20) != 0, depth, stencil & 0xFF,
                           &host_rect, 1);
  }
  for (HostSurface *alias : aliases) {
    Targets a;
    a.colors[a.color_count++] = alias;
    a.width = alias->width;
    a.height = alias->height;
    a.scale = alias->scale;
    const plume::RenderRect alias_rect(
        std::min(rect.left, int32_t(alias->width)), std::min(rect.top, int32_t(alias->height)),
        std::min(rect.right, int32_t(alias->width)), std::min(rect.bottom, int32_t(alias->height)));
    if (alias_rect.right <= alias_rect.left || alias_rect.bottom <= alias_rect.top || !BindTargets(a))
      continue;
    const plume::RenderRect alias_host = ScaleRect(alias_rect, a.scale);
    ++alias->writes;  // banjo-nx (banjo_renut_resolve_skip)
    cmd->clearColor(0, clear_color, &alias_host, 1);
  }
}

// Depth resolve (shaders/host/depth_resolve.hlsli): 'src' of the depth target
// to ('dest_x', 'dest_y') of the destination and of its guest words, in guest
// pixels (the destination is of the surface's scale).
bool ResolveDepth(HostSurface *surface, HostTexture *dest, const ClearRect &src, i32 dest_x,
                  i32 dest_y, bool words = true) {
  const RenderScale scale = surface->scale;
  auto &s = State();
  const uint32_t variant = surface->samples != plume::RenderSampleCount::COUNT_1 ? 1 : 0;
  const uint32_t source_slot = SurfaceSampleSlotLocked(surface);
  if (!s.depth_resolve_float_pipeline[variant])
    words = true;  // banjo-nx (banjo_renut_depth_words_lazy): no depth-only pipeline
  if (source_slot == kInvalidDescriptor || !dest->depth_framebuffer ||
      !s.depth_resolve_pipeline[variant]) {
    return false;
  }
  plume::RenderCommandList *cmd = s.command_list;
  static const uint32_t kProfileLabel = gpu_profile::Label("depth resolves");
  gpu_profile::Mark(s, kProfileLabel);
  TransitionSurfaceLocked(surface, plume::RenderTextureLayout::SHADER_READ);
  TransitionTextureLocked(dest, plume::RenderTextureLayout::COLOR_WRITE);
  TransitionTextureLocked(dest->words.get(), plume::RenderTextureLayout::COLOR_WRITE);
  cmd->setFramebuffer(dest->depth_framebuffer.get());
  // Pixel coordinates are destination texels.
  cmd->setViewports(plume::RenderViewport(0.0f, 0.0f, float(dest->alloc_width),
                                          float(dest->alloc_height)));
  cmd->setScissors(ScaleRect(plume::RenderRect(dest_x, dest_y, dest_x + (src.x2 - src.x1),
                                               dest_y + (src.y2 - src.y1)),
                             surface->scale));
  cmd->setGraphicsPipelineLayout(s.utility_layout.get());
  cmd->setPipeline(words ? s.depth_resolve_pipeline[variant].get() : s.depth_resolve_float_pipeline[variant].get());
  cmd->setGraphicsDescriptorSet(s.texture_set.get(), 0);
  cmd->setGraphicsDescriptorSet(s.sampler_set.get(), 1);
  cmd->setGraphicsDescriptorSet(s.texture_set.get(), 2);  // banjo-nx: the stencil view, read as uint
  UtilityPush push{};
  push.scale[0] = push.scale[1] = 1.0f;
  push.translate[0] = float(scale.Coord(src.x1) - scale.Coord(dest_x));
  push.translate[1] = float(scale.Coord(src.y1) - scale.Coord(dest_y));
  push.texture_index = source_slot;
  push.sampler_index = kPointClampSamplerSlot;
  // banjo-nx (2026-10-18, banjo_renut_depth_words_stencil): the stencil in the words' low byte (depth_resolve.hlsli),
  // which N&B's depth of field reads to keep the characters sharp. The resolve reads no sampler: the index is the
  // stencil view's slot.
  if (banjo::renut_gpu::DepthWordsStencilOn() && words) {
    if (const uint32_t stencil_slot = SurfaceStencilSlotLocked(surface); stencil_slot != kInvalidDescriptor) {
      push.sampler_index = stencil_slot;
      push.param1 = 1.0f;
    }
  }
  // D3DFMT_D24FS8 (GPU texture format 24_8_FLOAT) stores 20e4 float depth.
  push.param0 = GpuTextureFormat(surface->format_word & 0x3F) == GpuTextureFormat::Fmt24_8_Float
                    ? 1.0f
                    : 0.0f;
  cmd->setGraphicsPushConstants(0, &push, 0, sizeof(push));
  cmd->drawInstanced(3, 1, 0, 0);
  TransitionTextureLocked(dest, plume::RenderTextureLayout::SHADER_READ);

  // The draw path binds its own layout, pipeline, targets and viewport again.
  g_bound.framebuffer = nullptr;
  g_bound.pipeline = nullptr;
  g_bound.layout = false;
  g_bound.viewport_set = false;
  return true;
}

// ---------------------------------------------------------------------------
// Depth hand-over (banjo-nx, 2026-10-19, banjo_deko3d_depth_handover)
// ---------------------------------------------------------------------------
// A depth resolve of a whole single-sampled depth target into a destination of its size, without the words
// (banjo_renut_depth_words_lazy), is not drawn: the destination's content stays in the target's image (handed_from,
// with the generation and writes it had), and a bind of the destination samples that image through a depth view with
// the destination's component mapping. The float the resolve writes is the depth it loads (depth_resolve.hlsli), so the
// shaders read the same values. Before the target is bound again (a draw or a clear: BindTargets), or goes, the
// resolve is drawn after all (MaterialiseLocked), from the image still unchanged; so is it for a draw that has the
// target as its own depth target. A clear of the target's whole depth (and its stencil, or of a target whose stencil
// nothing uses) instead gives the image to the destination (held) and the destination's previous one, or a new one,
// to the target, which the clear then fills. The shadow maps are resolved once a frame, sampled by the scene and
// cleared whole at the start of the next frame's shadow pass: they are never copied.

std::vector<HostTexture *> g_handed;  // destinations whose content is a target's current image

void UnhandLocked(HostTexture &d) {
  if (!d.handed_from)
    return;
  if (d.handed_from->handed)
    --d.handed_from->handed;
  d.handed_from = nullptr;
  std::erase(g_handed, &d);
}

// Draws the resolve left out into the destination's own targets, from where its content is.
bool MaterialiseLocked(HostTexture &d, bool words) {
  ++g_texture_epoch;
  const ClearRect src{d.last_src[0], d.last_src[1], d.last_src[2], d.last_src[3]};
  if (HostSurface *source = d.handed_from) {
    const bool same = source->texture && source->generation == d.handed_generation &&
                      source->writes == d.handed_writes;
    UnhandLocked(d);
    if (same && ResolveDepth(source, &d, src, d.last_dest_x, d.last_dest_y, words)) {
      d.words_current = words;
      banjo::renut_gpu::CountHandOver(3);
      return true;
    }
    banjo::renut_gpu::CountHandOver(5);
    return false;
  }
  if (!d.held || !d.held_current)
    return false;
  d.held_current = false;
  // The taken image as the target it was taken from, for ResolveDepth.
  HostSurface image;
  image.width = d.width;
  image.height = d.height;
  image.scale = d.scale;
  image.depth = true;
  image.format = d.held->format;
  image.samples = d.held->samples;
  image.format_word = d.held->format_word;
  SwapDepthImageLocked(image, *d.held);
  const bool ok = ResolveDepth(&image, &d, src, d.last_dest_x, d.last_dest_y, words);
  SwapDepthImageLocked(image, *d.held);
  if (ok) {
    d.words_current = words;
    banjo::renut_gpu::CountHandOver(3);
    banjo::renut_gpu::CountHandOver(4);
  } else {
    banjo::renut_gpu::CountHandOver(5);
  }
  return ok;
}

// A resolve into 'd' left in the target's image when it can be (true). Otherwise (false, the resolve is drawn) a
// destination still holding other content gets it first when the resolve does not cover it.
bool HandOverResolveLocked(HostSurface *s, HostTexture *d, const ClearRect &src, i32 dest_x, i32 dest_y,
                           bool words) {
  const bool whole = src.x1 == 0 && src.y1 == 0 && src.x2 == int32_t(s->width) && src.y2 == int32_t(s->height) &&
                     dest_x == 0 && dest_y == 0;
  const bool can = IsDeko3D() && banjo::renut_gpu::DepthHandOverOn() && !words && whole && s->texture &&
                   s->samples == plume::RenderSampleCount::COUNT_1 && d->width == s->width &&
                   d->height == s->height && d->scale == s->scale && d->alloc_width == s->scale.Size(s->width) &&
                   d->alloc_height == s->scale.Size(s->height) && d->dimension == TextureDimension::Tex2D &&
                   !d->stacked && d->extra_levels == 0 && d->mip_levels == 1;
  if (!can) {
    if (d->handed_from || d->held_current) {
      const bool covers = dest_x == 0 && dest_y == 0 && src.x2 - src.x1 >= int32_t(d->width) &&
                          src.y2 - src.y1 >= int32_t(d->height);
      if (covers) {
        UnhandLocked(*d);
        d->held_current = false;
      } else {
        MaterialiseLocked(*d, false);
      }
    }
    return false;
  }
  UnhandLocked(*d);  // what it held is overwritten
  d->held_current = false;
  d->handed_from = s;
  d->handed_generation = s->generation;
  d->handed_writes = s->writes;
  ++s->handed;
  g_handed.push_back(d);
  ++g_texture_epoch;
  banjo::renut_gpu::CountHandOver(0);
  return true;
}

// The destination takes the target's image; the target takes the destination's previous one (or a new one).
bool TakeImageLocked(HostTexture &d, HostSurface &s) {
  if (d.held && (d.held->host_width != s.scale.Size(s.width) || d.held->host_height != s.scale.Size(s.height) ||
                 d.held->format != s.format || d.held->samples != s.samples)) {
    RetireDepthImageLocked(*d.held);
    d.held.reset();
  }
  if (!d.held) {
    auto image = std::make_shared<DepthImage>();
    if (!MakeDepthImageLocked(s, *image))
      return false;
    d.held = std::move(image);
  }
  UnhandLocked(d);
  SwapDepthImageLocked(s, *d.held);
  d.held_current = true;
  ++g_texture_epoch;
  banjo::renut_gpu::CountHandOver(2);
  return true;
}

void DepthHandOverBeforeBindLocked(HostSurface *surface, const plume::RenderRect *clear_rect, u32 clear_flags) {
  if (!surface || !surface->handed)
    return;
  bool take = clear_rect && (clear_flags & 0x10) && ((clear_flags & 0x20) || !surface->stencil_used) &&
              clear_rect->left <= 0 && clear_rect->top <= 0 && clear_rect->right >= int32_t(surface->width) &&
              clear_rect->bottom >= int32_t(surface->height);
  std::vector<HostTexture *> holders;
  for (HostTexture *d : g_handed)
    if (d->handed_from == surface)
      holders.push_back(d);
  // The last resolve's destination takes the image; any other is drawn.
  for (size_t i = holders.size(); i-- > 0;) {
    HostTexture &d = *holders[i];
    if (take && surface->generation == d.handed_generation && surface->writes == d.handed_writes &&
        TakeImageLocked(d, *surface)) {
      take = false;
      continue;
    }
    MaterialiseLocked(d, false);
  }
}

// banjo-nx (2026-10-18, banjo_deko3d_draw_resolves): a single-sampled colour resolve drawn into the destination
// (shaders/host/resolve_copy_ps.hlsl) rather than copied. deko3d copies images on the copy engine, which plume_deko3d
// fences with a full barrier before and after (the 3D engine idle for the whole copy); a draw is ordered by the usual
// fragment barriers and overlaps the work around it. Same format both sides, 2D level 0, slice 0: every texel is written
// back bit for bit. 'rect' is the source rect and (host_x, host_y) its place in the destination, in host pixels.
bool DrawResolveCopy(HostSurface *surface, HostTexture *dest, const plume::RenderRect &rect, uint32_t host_x,
                     uint32_t host_y) {
  auto &s = State();
  if (!s.resolve_copy_ps || !dest->texture || !dest->render_target || dest->host_format != surface->format ||
      dest->dimension != TextureDimension::Tex2D || dest->stacked)
    return false;
  switch (surface->format) {  // formats a texel load gives back exactly when written as a colour
  case plume::RenderFormat::R8G8B8A8_UNORM:
  case plume::RenderFormat::B8G8R8A8_UNORM:
  case plume::RenderFormat::R16G16B16A16_UNORM:
  case plume::RenderFormat::R16G16B16A16_FLOAT:
  case plume::RenderFormat::R16G16_UNORM:
  case plume::RenderFormat::R8G8_UNORM:
  case plume::RenderFormat::R8_UNORM:
  case plume::RenderFormat::R16_UNORM:
  case plume::RenderFormat::R16_FLOAT:
  case plume::RenderFormat::R16G16_FLOAT:
  case plume::RenderFormat::R32_FLOAT:
  case plume::RenderFormat::R32G32_FLOAT:
  case plume::RenderFormat::R32G32B32A32_FLOAT:
    break;
  default:
    return false;
  }
  const uint32_t source_slot = SurfaceSampleSlotLocked(surface);
  if (source_slot == kInvalidDescriptor)
    return false;
  auto &pipeline = s.resolve_copy_pipelines[uint32_t(surface->format)];
  if (!pipeline) {
    plume::RenderGraphicsPipelineDesc desc;
    desc.pipelineLayout = s.utility_layout.get();
    desc.vertexShader = s.present_vs.get();
    desc.pixelShader = s.resolve_copy_ps.get();
    desc.primitiveTopology = plume::RenderPrimitiveTopology::TRIANGLE_LIST;
    desc.cullMode = plume::RenderCullMode::NONE;
    desc.renderTargetCount = 1;
    desc.renderTargetFormat[0] = surface->format;
    desc.renderTargetBlend[0] = plume::RenderBlendDesc::Copy();
    desc.depthTargetFormat = plume::RenderFormat::UNKNOWN;
    pipeline = s.device->createGraphicsPipeline(desc);
    if (!pipeline)
      return false;
    pipeline->setName("renut resolve copy");
  }
  if (!dest->copy_framebuffer) {
    const plume::RenderTexture *targets[] = {dest->texture.get()};
    dest->copy_framebuffer = s.device->createFramebuffer(plume::RenderFramebufferDesc(targets, 1));
    if (!dest->copy_framebuffer)
      return false;
  }
  plume::RenderCommandList *cmd = s.command_list;
  TransitionSurfaceLocked(surface, plume::RenderTextureLayout::SHADER_READ);
  TransitionTextureLocked(dest, plume::RenderTextureLayout::COLOR_WRITE);
  cmd->setFramebuffer(dest->copy_framebuffer.get());
  cmd->setViewports(plume::RenderViewport(0.0f, 0.0f, float(dest->alloc_width), float(dest->alloc_height)));
  cmd->setScissors(plume::RenderRect(int32_t(host_x), int32_t(host_y), int32_t(host_x) + (rect.right - rect.left),
                                     int32_t(host_y) + (rect.bottom - rect.top)));
  cmd->setGraphicsPipelineLayout(s.utility_layout.get());
  cmd->setPipeline(pipeline.get());
  cmd->setGraphicsDescriptorSet(s.texture_set.get(), 0);
  cmd->setGraphicsDescriptorSet(s.sampler_set.get(), 1);
  UtilityPush push{};
  push.scale[0] = push.scale[1] = 1.0f;
  push.translate[0] = float(rect.left - int32_t(host_x));
  push.translate[1] = float(rect.top - int32_t(host_y));
  push.texture_index = source_slot;
  push.sampler_index = kPointClampSamplerSlot;
  cmd->setGraphicsPushConstants(0, &push, 0, sizeof(push));
  cmd->drawInstanced(3, 1, 0, 0);
  banjo::renut_gpu::CountDrawnResolve();
  // The draw path binds its own layout, pipeline, targets and viewport again.
  g_bound.framebuffer = nullptr;
  g_bound.pipeline = nullptr;
  g_bound.layout = false;
  g_bound.viewport_set = false;
  return true;
}

plume::RenderRect ClampRect(const ClearRect *rect, uint32_t width, uint32_t height) {
  if (!rect)
    return plume::RenderRect(0, 0, int32_t(width), int32_t(height));
  const int32_t x1 = std::clamp(rect->x1, 0, int32_t(width));
  const int32_t y1 = std::clamp(rect->y1, 0, int32_t(height));
  const int32_t x2 = std::clamp(rect->x2, x1, int32_t(width));
  const int32_t y2 = std::clamp(rect->y2, y1, int32_t(height));
  return plume::RenderRect(x1, y1, x2, y2);
}

} // namespace

void ClearStateLocked(const u8 *device_state, u32 flags, const ClearRect *rect,
                      const float color[4], float depth, u32 stencil) {
  auto &s = State();
  if (!Video::EnsureCommandListLocked())
    return;
  SyncBound(s);
  const GuestDevice device(device_state);
  if (!device.valid())
    return;
  Targets targets;
  if (!GatherTargets(device, targets))
    return;

  // D3D clamps clears to the viewport (CDevice viewport floats at 0x3168).
  ClearRect r = rect ? *rect : ClearRect{0, 0, int32_t(targets.width), int32_t(targets.height)};
  const float vx = device.F32(0x3168), vy = device.F32(0x316C);
  const float vw = device.F32(0x3170), vh = device.F32(0x3174);
  if (vw > 0.0f && vh > 0.0f) {
    int32_t vx2 = int32_t(vx + vw), vy2 = int32_t(vy + vh);
    // A target grown for tiling: D3D bounds the viewport by the tile-sized
    // surface, which covers the whole target across the tiles (as for the
    // scissor in ComputeViewport).
    if (targets.width > targets.guest_width && vx2 >= int32_t(targets.guest_width))
      vx2 = int32_t(targets.width);
    if (targets.height > targets.guest_height && vy2 >= int32_t(targets.guest_height))
      vy2 = int32_t(targets.height);
    r.x1 = std::max(r.x1, int32_t(vx));
    r.y1 = std::max(r.y1, int32_t(vy));
    r.x2 = std::min(r.x2, vx2);
    r.y2 = std::min(r.y2, vy2);
  }
  const plume::RenderRect clear_rect = ClampRect(&r, targets.width, targets.height);
  if (clear_rect.right <= clear_rect.left || clear_rect.bottom <= clear_rect.top)
    return;
  ClearTargets(targets, flags, clear_rect, color, depth, stencil);
}

// banjo-nx (2026-10-19, banjo_renut_depth_words_lazy): a destination's first resolves write the words either way.
constexpr uint32_t kDepthWordsProbe = 300;

void EnsureDepthWordsLocked(HostTexture &dest) {
  if (!banjo::renut_gpu::DepthWordsLazyOn())
    return;
  dest.words_wanted = true;
  if (dest.words_current)
    return;
  // banjo-nx (2026-10-19, banjo_deko3d_depth_handover): left in a target's image: drawn now, with the words.
  if (dest.handed_from || dest.held_current) {
    banjo::renut_gpu::CountLazyWords(MaterialiseLocked(dest, true) ? 1 : 2);
    return;
  }
  HostSurface *surface = SurfaceByGenerationLocked(dest.last_source_generation);
  if (surface && surface->writes == dest.last_source_writes) {
    const ClearRect src{dest.last_src[0], dest.last_src[1], dest.last_src[2], dest.last_src[3]};
    if (ResolveDepth(surface, &dest, src, dest.last_dest_x, dest.last_dest_y, true)) {
      dest.words_current = true;
      banjo::renut_gpu::CountLazyWords(1);
      return;
    }
  }
  // The source has changed since: these words are older ones. From now on this destination writes them each time.
  banjo::renut_gpu::CountLazyWords(2);
}

// banjo-nx (2026-10-19, banjo_deko3d_depth_handover): see "Depth hand-over".
uint32_t DepthHandOverSlotLocked(HostTexture &d, const plume::RenderComponentMapping &mapping, BindEffects *fx) {
  if (HostSurface *source = d.handed_from) {
    if (!source->texture || source->generation != d.handed_generation || source->writes != d.handed_writes) {
      UnhandLocked(d);
      banjo::renut_gpu::CountHandOver(5);
      return UINT32_MAX;
    }
    // Not the image of a target of the draw: its resolve is drawn after all.
    if (g_handover_targets && g_handover_targets->depth == source) {
      MaterialiseLocked(d, false);
      return UINT32_MAX;
    }
    const uint32_t slot =
        DepthImageViewSlotLocked(source->texture.get(), source->format, source->handover_views, mapping);
    if (slot == kInvalidDescriptor) {
      MaterialiseLocked(d, false);
      return UINT32_MAX;
    }
    TransitionSurfaceLocked(source, plume::RenderTextureLayout::SHADER_READ);
    if (fx)
      fx->handover_surface = source;
    banjo::renut_gpu::CountHandOver(1);
    return slot;
  }
  if (!d.held || !d.held_current)
    return UINT32_MAX;
  DepthImage &image = *d.held;
  const uint32_t slot = DepthImageViewSlotLocked(image.texture.get(), image.format, image.views, mapping);
  if (slot == kInvalidDescriptor) {
    MaterialiseLocked(d, false);
    return UINT32_MAX;
  }
  TransitionDepthImageLocked(image, plume::RenderTextureLayout::SHADER_READ);
  if (fx)
    fx->handover_image = &image;
  banjo::renut_gpu::CountHandOver(1);
  return slot;
}

void DepthHandOverForgetLocked(HostTexture &d) {
  UnhandLocked(d);
  d.held_current = false;
  if (d.held) {
    RetireDepthImageLocked(*d.held);
    d.held.reset();
  }
}

void DepthHandOverBeforeRetireLocked(HostSurface *surface) {
  if (!surface || !surface->handed)
    return;
  const bool list = Video::EnsureCommandListLocked();
  std::vector<HostTexture *> holders;
  for (HostTexture *d : g_handed)
    if (d->handed_from == surface)
      holders.push_back(d);
  for (HostTexture *d : holders) {
    if (list) {
      MaterialiseLocked(*d, false);
    } else {
      UnhandLocked(*d);
      banjo::renut_gpu::CountHandOver(5);
    }
  }
}

// banjo-nx (2026-10-19, banjo_renut_resolve_skip): what a resolve wrote into its destination: the source surface (its
// identity, rebuilds and writes), the rectangle and the place. Never 0.
static uint64_t ResolveSignature(const HostSurface *surface, const ClearRect &src, i32 dest_x, i32 dest_y,
                                 u32 level, u32 slice) {
  uint64_t h = 0xcbf29ce484222325ull;
  auto mix = [&h](uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    h *= 0x100000001b3ull;
  };
  mix(uint64_t(uintptr_t(surface)));
  mix(surface->generation);
  mix(surface->writes);
  mix(surface->format_word);
  mix(uint64_t(uint32_t(src.x1)) | uint64_t(uint32_t(src.y1)) << 32);
  mix(uint64_t(uint32_t(src.x2)) | uint64_t(uint32_t(src.y2)) << 32);
  mix(uint64_t(uint32_t(dest_x)) | uint64_t(uint32_t(dest_y)) << 32);
  mix(uint64_t(level) | uint64_t(slice) << 32);
  return h ? h : 1;
}

void ResolveStateLocked(const u8 *device_state, u32 flags, const ClearRect *source_rect,
                        u32 dest_texture_va, const d3d::TextureFetchConstant *dest_format,
                        i32 dest_x, i32 dest_y, u32 dest_level, u32 dest_slice,
                        const float clear_color[4], float clear_depth, u32 clear_stencil) {
  auto &s = State();
  static const uint32_t kProfileLabel = gpu_profile::Label("resolves");
  gpu_profile::Mark(s, kProfileLabel);
  // Pending memexport results stay on the GPU: a resolve writes host textures
  // only, and a clear reads no memory.
  if (!Video::EnsureCommandListLocked())
    return;
  SyncBound(s);
  const GuestDevice device(device_state);
  if (!device.valid())
    return;

  const u32 source = flags & 7;
  const u32 source_va = source == 4 ? device.U32(d3d::kDeviceOffsetDepthStencil)
                                    : device.U32(d3d::kDeviceOffsetRenderTargets + source * 4);
  HostSurface *surface = GetHostSurfaceLocked(source_va);
  if (banjo::renut_gpu::ResolveTraceOn() && surface) {
    const ClearRect r = source_rect ? *source_rect : ClearRect{0, 0, -1, -1};
    RGPU_INFO("[trace] resolve {} {:08X} ({}x{} fmt {} samples {}) -> texture {:08X} at ({},{}) level {} slice {}, "
              "rect ({},{})-({},{}), flags {:X} ({}{})",
              surface->depth ? "depth" : "colour", source_va, surface->width, surface->height, u32(surface->format),
              u32(surface->samples), dest_texture_va, dest_x, dest_y, dest_level, dest_slice, r.x1, r.y1, r.x2, r.y2,
              flags, (flags & 0x100) ? "clear colour after " : "", (flags & 0x200) ? "clear depth after" : "");
    // mk9-nx (2026-10-21): the source's EDRAM view (TargetKey's fields), to see aliases.
    if (const auto *desc = GuestPtr<const d3d::SurfaceDesc>(source_va + 0x18))
      RGPU_INFO("[trace]   source EDRAM base {} pitch {} msaa {} format {:08X}", u32(desc->ColorInfo) & 0xFFF,
                u32(desc->SurfaceInfo) & 0x3FFF, (u32(desc->SurfaceInfo) >> 16) & 3, u32(desc->Format));
  }
  plume::RenderCommandList *cmd = s.command_list;
  static uint32_t depth_logged = 0;

  // The source rect within the target and the destination level.
  auto fit = [&](const HostTexture *dest, ClearRect &src) {
    const uint32_t level_w = std::max(1u, dest->width >> dest_level);
    const uint32_t level_h = std::max(1u, dest->height >> dest_level);
    src = source_rect ? *source_rect : ClearRect{0, 0, int32_t(level_w), int32_t(level_h)};
    src.x1 = std::clamp(src.x1, 0, int32_t(surface->width));
    src.y1 = std::clamp(src.y1, 0, int32_t(surface->height));
    src.x2 = std::clamp(src.x2, src.x1, int32_t(surface->width));
    src.y2 = std::clamp(src.y2, src.y1, int32_t(surface->height));
    dest_x = std::clamp(dest_x, 0, int32_t(level_w));
    dest_y = std::clamp(dest_y, 0, int32_t(level_h));
    src.x2 = std::min(src.x2, src.x1 + int32_t(level_w) - dest_x);
    src.y2 = std::min(src.y2, src.y1 + int32_t(level_h) - dest_y);
    return src.x2 > src.x1 && src.y2 > src.y1;
  };

  if (surface && dest_texture_va) {
    if (surface->depth) {
      HostTexture *dest =
          dest_level == 0 && dest_slice == 0
              ? GetDepthResolveTextureLocked(dest_texture_va, surface->scale, dest_format)
              : nullptr;
      ClearRect src;
      if (!dest) {
        if (LogLimited(depth_logged, 2)) {
          RGPU_WARN("Depth resolve into {:08X} (level {}, slice {}) not done", dest_texture_va,
                    dest_level, dest_slice);
        }
      } else if (fit(dest, src)) {
        // banjo-nx (2026-10-19, banjo_renut_resolve_skip): the same resolve of an unchanged source is in dest already.
        const uint64_t sig = ResolveSignature(surface, src, dest_x, dest_y, dest_level, dest_slice);
        // banjo-nx (2026-10-19, banjo_renut_depth_words_lazy): the words only where something reads them (or while
        // probing a new destination).
        const bool lazy = banjo::renut_gpu::DepthWordsLazyOn();
        const bool words = !lazy || dest->words_wanted || dest->words_probe < kDepthWordsProbe;
        if (banjo::renut_gpu::ResolveSkipOn() && dest->resolve_signature == sig) {
          banjo::renut_gpu::CountSkippedResolve(true);
        } else if (HandOverResolveLocked(surface, dest, src, dest_x, dest_y, words) ||
                   ResolveDepth(surface, dest, src, dest_x, dest_y, words)) {
          dest->resolve_signature = sig;
          dest->words_current = words;
          if (!dest->words_wanted && dest->words_probe < kDepthWordsProbe)
            ++dest->words_probe;
          if (!words)
            banjo::renut_gpu::CountLazyWords(0);
          dest->last_source_generation = surface->generation;
          dest->last_source_writes = surface->writes;
          dest->last_src[0] = src.x1;
          dest->last_src[1] = src.y1;
          dest->last_src[2] = src.x2;
          dest->last_src[3] = src.y2;
          dest->last_dest_x = dest_x;
          dest->last_dest_y = dest_y;
        } else {
          dest->resolve_signature = 0;
        }
      }
    } else if (HostTexture *dest = GetResolveTextureLocked(dest_texture_va, surface->format,
                                                           surface->scale, dest_format)) {
      ClearRect src;
      // A volume's slices are depth; stacked textures and cubes have layers.
      const bool volume = dest->dimension == TextureDimension::Tex3D && !dest->stacked;
      const uint32_t slice = volume ? 0 : dest_slice;
      const uint32_t z = volume ? dest_slice : 0;
      const bool fits = fit(dest, src);
      const uint64_t sig = fits ? ResolveSignature(surface, src, dest_x, dest_y, dest_level, dest_slice) : 0;
      if (fits && banjo::renut_gpu::ResolveSkipOn() && dest->resolve_signature == sig) {
        // banjo-nx (2026-10-19, banjo_renut_resolve_skip): the same resolve of an unchanged source is in dest already.
        banjo::renut_gpu::CountSkippedResolve(false);
      } else if (fits) {
        dest->resolve_signature = sig;
        g_bound.framebuffer = nullptr;
        // The destination is of the surface's scale.
        plume::RenderRect rect =
            ScaleRect(plume::RenderRect(src.x1, src.y1, src.x2, src.y2), surface->scale);
        const uint32_t host_x = uint32_t(surface->scale.Coord(dest_x));
        const uint32_t host_y = uint32_t(surface->scale.Coord(dest_y));
        // At a fractional scale the source rect, rounded at its own edges, can
        // pass the destination's host size by a texel.
        rect.right = std::min(rect.right, rect.left + int32_t(dest->alloc_width - host_x));
        rect.bottom = std::min(rect.bottom, rect.top + int32_t(dest->alloc_height - host_y));
        const plume::RenderRect dest_rect(int32_t(host_x), int32_t(host_y),
                                          int32_t(host_x) + (rect.right - rect.left),
                                          int32_t(host_y) + (rect.bottom - rect.top));
        const bool multisampled = surface->samples != plume::RenderSampleCount::COUNT_1;
        if (multisampled && dest->extra_levels && IsVulkan()) {
          // A destination with host levels takes it through a single-level
          // target (ResolveStagingLocked): plume's Vulkan resolve writes every
          // level of its destination (D3D12's writes level 0 only, and
          // resolves in place below). Without one (device lost) nothing of
          // the resolve lands.
          if (HostTexture *staging = ResolveStagingLocked(dest)) {
            TransitionSurfaceLocked(surface, plume::RenderTextureLayout::RESOLVE_SOURCE);
            TransitionTextureLocked(staging, plume::RenderTextureLayout::RESOLVE_DEST);
            cmd->resolveTextureRegion(staging->texture.get(), host_x, host_y,
                                      surface->texture.get(), &rect);
            TransitionTextureLocked(staging, plume::RenderTextureLayout::COPY_SOURCE);
            TransitionTextureLocked(dest, plume::RenderTextureLayout::COPY_DEST);
            const plume::RenderBox box(dest_rect.left, dest_rect.top, dest_rect.right,
                                       dest_rect.bottom);
            cmd->copyTextureRegion(
                plume::RenderTextureCopyLocation::Subresource(dest->texture.get(), 0, slice),
                plume::RenderTextureCopyLocation::Subresource(staging->texture.get()), host_x,
                host_y, z, &box);
          }
        } else if (multisampled) {
          TransitionSurfaceLocked(surface, plume::RenderTextureLayout::RESOLVE_SOURCE);
          TransitionTextureLocked(dest, plume::RenderTextureLayout::RESOLVE_DEST);
          cmd->resolveTextureRegion(dest->texture.get(), host_x, host_y, surface->texture.get(),
                                    &rect);
        } else if (IsDeko3D() && banjo::renut_gpu::DrawResolvesOn() && slice == 0 && z == 0 &&
                   DrawResolveCopy(surface, dest, rect, host_x, host_y)) {
          // banjo-nx (2026-10-18): drawn, not copied (deko3d).
        } else {
          TransitionSurfaceLocked(surface, plume::RenderTextureLayout::COPY_SOURCE);
          TransitionTextureLocked(dest, plume::RenderTextureLayout::COPY_DEST);
          const plume::RenderBox box(rect.left, rect.top, rect.right, rect.bottom);
          cmd->copyTextureRegion(
              plume::RenderTextureCopyLocation::Subresource(dest->texture.get(), 0, slice),
              plume::RenderTextureCopyLocation::Subresource(surface->texture.get()), host_x,
              host_y, z, &box);
        }
        TransitionTextureLocked(dest, plume::RenderTextureLayout::SHADER_READ);
        if (dest->extra_levels) {
          UpdateResolveMipsLocked(dest, dest_rect);
          // The draw path binds its own layout, pipeline and viewport again.
          g_bound.pipeline = nullptr;
          g_bound.layout = false;
          g_bound.viewport_set = false;
        }
      }
    }
  }

  // Resolve-and-clear.
  if ((flags & 0x300) && surface) {
    Targets t;
    if (surface->depth)
      t.depth = surface;
    else
      t.colors[t.color_count++] = surface;
    t.width = surface->width;
    t.height = surface->height;
    t.scale = surface->scale;
    const plume::RenderRect rect = ClampRect(source_rect, t.width, t.height);
    static const float kBlack[4] = {0, 0, 0, 0};
    const u32 clear_flags = ((flags & 0x100) ? 1u : 0u) | ((flags & 0x200) ? 0x30u : 0u);
    ClearTargets(t, clear_flags, rect, clear_color ? clear_color : kBlack, clear_depth,
                 clear_stencil);
  }
}

bool ReadRenderTargetLocked(const u8 *device_state, u32 source, u32 &width, u32 &height,
                            std::vector<u32> &words) {
  auto &s = State();
  if (!s.target_readback_pipeline || !Video::EnsureCommandListLocked())
    return false;
  SyncBound(s);
  const GuestDevice device(device_state);
  if (!device.valid() || source >= d3d::kMaxRenderTargets)
    return false;
  HostSurface *surface =
      GetHostSurfaceLocked(device.U32(d3d::kDeviceOffsetRenderTargets + source * 4));
  if (!surface || surface->depth || !surface->texture)
    return false;
  width = std::min(width, surface->width);
  height = std::min(height, surface->height);
  if (!width || !height)
    return false;
  static const uint32_t kProfileLabel = gpu_profile::Label("render target readback");
  gpu_profile::Mark(s, kProfileLabel);

  // A single-sampled copy of the rect, which the shader reads by its slot.
  const RenderScale scale = surface->scale;
  const plume::RenderRect rect(0, 0, int32_t(scale.Size(width)), int32_t(scale.Size(height)));
  plume::RenderTextureDesc desc = plume::RenderTextureDesc::Texture2D(
      uint32_t(rect.right), uint32_t(rect.bottom), 1, surface->format);
  desc.flags |= plume::RenderTextureFlag::RENDER_TARGET;
  std::unique_ptr<plume::RenderTexture> copy = s.device->createTexture(desc);
  if (!TextureCreated(copy.get()))
    return false;
  std::unique_ptr<plume::RenderTextureView> view =
      copy->createTextureView(plume::RenderTextureViewDesc::Texture2D(surface->format));
  const uint64_t bytes = uint64_t(width) * height * 4;
  std::unique_ptr<plume::RenderBuffer> output = s.device->createBuffer(
      plume::RenderBufferDesc::DefaultBuffer(bytes, StorageBufferFlags()));
  std::unique_ptr<plume::RenderBuffer> readback =
      s.device->createBuffer(plume::RenderBufferDesc::ReadbackBuffer(bytes));
  std::unique_ptr<plume::RenderDescriptorSet> set = detail::CreateTargetReadbackSet(s);
  const uint32_t slot = view ? Video::AllocateTextureSlotLocked() : kInvalidDescriptor;
  if (!view || !output || !readback || !set || slot == kInvalidDescriptor)
    return false;
  Video::SetTextureSlotLocked(slot, copy.get(), view.get());
  set->setBuffer(0, output.get());

  plume::RenderCommandList *cmd = s.command_list;
  g_bound.framebuffer = nullptr;
  if (surface->samples != plume::RenderSampleCount::COUNT_1) {
    TransitionSurfaceLocked(surface, plume::RenderTextureLayout::RESOLVE_SOURCE);
    cmd->barriers(plume::RenderBarrierStage::COPY,
                  plume::RenderTextureBarrier(copy.get(), plume::RenderTextureLayout::RESOLVE_DEST));
    cmd->resolveTextureRegion(copy.get(), 0, 0, surface->texture.get(), &rect);
  } else {
    TransitionSurfaceLocked(surface, plume::RenderTextureLayout::COPY_SOURCE);
    cmd->barriers(plume::RenderBarrierStage::COPY,
                  plume::RenderTextureBarrier(copy.get(), plume::RenderTextureLayout::COPY_DEST));
    const plume::RenderBox box(rect.left, rect.top, rect.right, rect.bottom);
    cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(copy.get()),
                           plume::RenderTextureCopyLocation::Subresource(surface->texture.get()),
                           0, 0, 0, &box);
  }
  const plume::RenderTextureBarrier read(copy.get(), plume::RenderTextureLayout::SHADER_READ);
  const plume::RenderBufferBarrier write(output.get(), plume::RenderBufferAccess::WRITE);
  cmd->barriers(plume::RenderBarrierStage::COMPUTE, &write, 1, &read, 1);
  cmd->setComputePipelineLayout(s.target_readback_layout.get());
  cmd->setPipeline(s.target_readback_pipeline.get());
  cmd->setComputeDescriptorSet(s.texture_set.get(), 0);
  cmd->setComputeDescriptorSet(set.get(), 1);
  TargetReadbackPush push{};
  push.texture_index = slot;
  push.width = width;
  push.height = height;
  push.scale_numerator = RenderScale::kNumerators[scale.code];
  push.scale_denominator = RenderScale::kDenominators[scale.code];
  cmd->setComputePushConstants(0, &push, 0, sizeof(push));  // banjo-nx: the size (the record worker copies it)
  cmd->dispatch((width + 7) / 8, (height + 7) / 8, 1);
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderBufferBarrier(output.get(), plume::RenderBufferAccess::READ));
  cmd->copyBufferRegion(plume::RenderBufferReference(readback.get()),
                        plume::RenderBufferReference(output.get()), bytes);
  // The compute pipeline replaced the graphics one (D3D12), and the barriers
  // ended any render pass.
  g_bound.pipeline = nullptr;
  g_bound.layout = false;

  detail::SubmitAndWaitLocked(s);
  bool read_back = false;
  if (const auto *data = static_cast<const u32 *>(readback->map())) {
    words.assign(data, data + size_t(width) * height);
    readback->unmap();
    read_back = true;
  }
  Video::FreeTextureSlotLocked(slot);
  Video::RetireLocked(std::move(view));
  Video::RetireLocked(std::move(copy));
  Video::RetireLocked(std::move(set));
  Video::RetireLocked(std::move(output));
  Video::RetireLocked(std::move(readback));
  return read_back;
}

bool FrontBufferWrittenLocked(u32 texture_va) {
  const HostTexture *texture = FindHostTextureLocked(texture_va);
  return texture && texture->texture && texture->gpu_written;
}

PresentImage FrontBufferImageLocked(u32 texture_va) {
  PresentImage image;
  HostTexture *texture = FindHostTextureLocked(texture_va);
  if (!texture || !texture->texture || texture->slot == UINT32_MAX)
    return image;
  image.texture = texture->texture.get();
  image.layout = &texture->layout;
  image.descriptor = texture->slot;
  image.width = texture->alloc_width;
  image.height = texture->alloc_height;
  image.source_width = texture->scale.Size(texture->width);
  image.source_height = texture->scale.Size(texture->height);
  return image;
}

// banjo-nx (2026-10-19, render-thread proposal 1): see draw.h.
bool GuestRangeCardsClean(const u8 *data, size_t bytes, uint64_t proven) {
  return PagesClean(data, bytes, proven);
}

uint64_t CardsInterval() {
  return g_flush_serial;
}

} // namespace renut::gpu
