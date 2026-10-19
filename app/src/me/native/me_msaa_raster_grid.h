#pragma once
#include <cstdint>

namespace me::native {

struct MsaaRasterGrid {
  bool valid;
  // Multipliers, NOT log2 values. ImageNative.raster_grid_x stores log2(grid_x).
  uint32_t grid_x;
  uint32_t grid_y;
};

// Addressing policy only, not a claim of complete multisample rasterization.
// 4x Xenos has two horizontal samples. Preserve these for depth-only passes
// when explicitly enabled; Y sample positions remain a separate approximation.
constexpr MsaaRasterGrid GetMsaaRasterGrid(bool enabled, uint32_t msaa, bool depth_only) {
  if (msaa > 2) return {false, 0, 0};
  return {true, enabled && depth_only && msaa == 2 ? 2u : 1u, 1u};
}

}  // namespace me::native
