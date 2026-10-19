#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string_view>

namespace me::native {

// Pure proof for replacing a stencil-only rectangle with a constant stencil
// clear in an existing canonical physical-word owner. This is intentionally NOT
// a permission to omit stencil writes or to publish a collapsed depth view.
// Bit positions below are SDK registers.h RB_DEPTHCONTROL/RB_STENCILREFMASK,
// RB_COLORCONTROL, PA_SU_SC_MODE_CNTL, PA_SU_VTX_CNTL and PA_CL_VTE_CNTL.
struct StencilClearOwnershipInput {
  uint32_t depth_control = 0, stencil_front = 0, stencil_back = 0;
  uint32_t color_control = 0, su_sc_mode = 0, vtx_cntl = 0, vte_cntl = 0;
  uint32_t window_offset = 0, clip_cntl = 0;
  uint32_t msaa_enum = UINT32_MAX, physical_pitch_tiles = 0;
  uint32_t scissor_x0 = 0, scissor_y0 = 0, scissor_x1 = 0, scissor_y1 = 0;
  std::array<std::array<float, 4>, 3> post_vs_position{};
  // sa2-nx: interpolator 0 of the three vertices (valid when all were exported) and whether the pixel shader is
  // exactly oC0 = r0 (ProvePixelColor0IsR0): together they give a clear's colour.
  std::array<std::array<float, 4>, 3> post_vs_interpolator0{};
  bool interpolator0_valid = false, ps_color0_is_r0 = false;
  bool xy_raster = false;  // x/y above already went through the viewport transform (estimator)
  // These are explicit caller proofs, never inferred from a shader number or
  // from an unrecognized raw sample-control register. The SDK exposes no
  // decoded RB_SAMPLE_CONTROL; its PA_SC_AA_MASK is not interpreted here.
  bool rectangle_proven = false, ps_identity_proven = false;
  bool ps_no_discard = false, ps_no_depth_export = false;
  bool sample_coverage_proven_full = false;
};

struct StencilClearOwnershipProof {
  bool eligible = false;
  std::string_view reason = "unproven";
  uint32_t stencil_reference = 0;
  uint32_t guest_width = 0, guest_height = 0;
  uint32_t physical_width = 0, physical_height = 0;
  uint32_t columns_tiles = 0, rows_tiles = 0, length_tiles = 0;
};

inline StencilClearOwnershipProof ProveStencilClearOwnership(const StencilClearOwnershipInput& in) {
  StencilClearOwnershipProof out;
  auto fail = [&](std::string_view reason) { out.reason = reason; return out; };
  if (!in.rectangle_proven || !in.ps_identity_proven || !in.ps_no_discard ||
      !in.ps_no_depth_export || !in.sample_coverage_proven_full) return fail("shader-or-coverage-unproven");
  const uint32_t dc = in.depth_control;
  if (!(dc & 1) || (dc & 6)) return fail("not-stencil-only");
  if (((dc >> 8) & 7) != 7 || ((dc >> 14) & 7) != 2)
    return fail("front-not-always-replace");
  if (((in.stencil_front >> 16) & 255) != 255) return fail("partial-stencil-mask");
  if (dc & 0x80) {
    if (((dc >> 20) & 7) != 7 || ((dc >> 26) & 7) != 2 ||
        ((in.stencil_back >> 16) & 255) != 255 ||
        (in.stencil_back & 0xFFFFFF) != (in.stencil_front & 0xFFFFFF))
      return fail("backface-not-equivalent");
  }
  if (in.color_control & 0x18) return fail("alpha-coverage");
  if (in.su_sc_mode & 3) return fail("face-culling");
  if ((in.su_sc_mode >> 3) & 3) return fail("non-default-polygon-mode");
  if (!(in.clip_cntl & 0x10000) || !(in.vte_cntl & 0x100) || (in.vte_cntl & 0xF))
    return fail("not-direct-unclipped-xy");
  // Initial profile requires the common D3D round-to-even / 1/16 quantization.
  if ((in.vtx_cntl & 0x3E) != 4) return fail("vertex-quantization");
  if (in.msaa_enum > 2 || !in.physical_pitch_tiles || in.physical_pitch_tiles > 2048)
    return fail("physical-layout");

  float minimum_x = INFINITY, minimum_y = INFINITY;
  float maximum_x = -INFINITY, maximum_y = -INFINITY;
  for (const auto& p : in.post_vs_position) {
    for (float v : p) if (!std::isfinite(v)) return fail("nonfinite-position");
    if (p[3] != 1.0f) return fail("non-unit-w");
    minimum_x = std::min(minimum_x, p[0]); maximum_x = std::max(maximum_x, p[0]);
    minimum_y = std::min(minimum_y, p[1]); maximum_y = std::max(maximum_y, p[1]);
  }
  // Check actual corners, rather than trusting a bounding box containing three
  // unrelated vertices. The caller's rectangle_proven additionally guarantees
  // native longest-edge/fourth-corner expansion does not extrapolate the box.
  uint32_t corner_mask = 0;
  for (const auto& p : in.post_vs_position) {
    if ((p[0] != minimum_x && p[0] != maximum_x) ||
        (p[1] != minimum_y && p[1] != maximum_y)) return fail("not-box-corners");
    const uint32_t bit = 1u << (uint32_t(p[0] == maximum_x) | (uint32_t(p[1] == maximum_y) << 1));
    if (corner_mask & bit) return fail("duplicate-corner");
    corner_mask |= bit;
  }
  const float center_adjust = (in.vtx_cntl & 1) ? 0.0f : 0.5f;
  const auto sign15 = [](uint32_t v) { return int32_t(v & 0x7FFF) - int32_t((v & 0x4000) << 1); };
  const int32_t window_x = (in.su_sc_mode & 0x10000) ? sign15(in.window_offset) : 0;
  const int32_t window_y = (in.su_sc_mode & 0x10000) ? sign15(in.window_offset >> 16) : 0;
  const double x0 = double(minimum_x) + center_adjust + window_x;
  const double y0 = double(minimum_y) + center_adjust + window_y;
  const double x1 = double(maximum_x) + center_adjust + window_x;
  const double y1 = double(maximum_y) + center_adjust + window_y;
  // Origin zero and exact integer edges avoid partial samples or preserved
  // borders. No epsilon, floor or ceil may turn uncertain coverage into proof.
  if (x0 != 0 || y0 != 0 || x1 <= 0 || y1 <= 0 || x1 > 8192 || y1 > 8192 ||
      x1 != std::floor(x1) || y1 != std::floor(y1)) return fail("non-integer-origin-box");
  out.guest_width = uint32_t(x1); out.guest_height = uint32_t(y1);
  if (in.scissor_x0 || in.scissor_y0 || in.scissor_x1 < out.guest_width ||
      in.scissor_y1 < out.guest_height) return fail("scissor-clips-box");
  out.physical_width = out.guest_width << uint32_t(in.msaa_enum == 2);
  out.physical_height = out.guest_height << uint32_t(in.msaa_enum != 0);
  if (out.physical_width % 80 || out.physical_height % 16) return fail("partial-physical-tile");
  out.columns_tiles = out.physical_width / 80;
  out.rows_tiles = out.physical_height / 16;
  // Restrict to complete pitch rows. A narrower rectangle needs segmented
  // publication; its contiguous span must not claim unmodified row-gap tiles.
  if (out.columns_tiles != in.physical_pitch_tiles) return fail("partial-pitch-row");
  const uint64_t tile_count = uint64_t(out.columns_tiles) * out.rows_tiles;
  if (!tile_count || tile_count > 2048) return fail("tile-count");
  out.length_tiles = uint32_t(tile_count);
  out.stencil_reference = in.stencil_front & 255;
  out.eligible = true;
  out.reason = "full-physical-tile-stencil-replace";
  return out;
}

}  // namespace me::native
