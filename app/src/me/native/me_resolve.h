#pragma once
#include <cstdint>
#include <algorithm>

namespace me::native {
struct ClearRectangle {
  int32_t x, y;
  uint32_t width, height;
  constexpr bool whole(uint32_t w, uint32_t h) const {
    return x == 0 && y == 0 && width == w && height == h;
  }
};
constexpr ClearRectangle ClipClearRectangle(int32_t x0, int32_t y0, int32_t x1,
                                            int32_t y1, uint32_t width, uint32_t height) {
  x0 = std::clamp(x0, 0, int32_t(width));
  y0 = std::clamp(y0, 0, int32_t(height));
  x1 = std::clamp(x1, x0, int32_t(width));
  y1 = std::clamp(y1, y0, int32_t(height));
  return {x0, y0, uint32_t(x1 - x0), uint32_t(y1 - y0)};
}
// RB_COPY_DEST_INFO.copy_dest_exp_bias, signed six-bit field [16:21].
constexpr int32_t ResolveExponentBias(uint32_t destination_info) {
  const int32_t bits = int32_t((destination_info >> 16) & 63);
  return bits >= 32 ? bits - 64 : bits;
}
// Match util/draw.cpp GetResolveInfo/GetCopyShader: both Raw and Convert
// requests use the full conversion path when the exponent bias is nonzero.
constexpr int32_t ColorResolveExponentBias(uint32_t command, uint32_t destination_info) {
  return command <= 1 ? ResolveExponentBias(destination_info) : 0;
}
}  // namespace me::native
