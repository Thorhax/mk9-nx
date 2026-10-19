// sa2-nx: render resolution scale of the native renderer (sa2_render_height_handheld / sa2_render_height_docked).
//
// Every host image of a render target, a depth target or a resolved texture is the guest image scaled by q / 4
// (q = render height / 180: 4 = 720p as the game, 6 = 1080p, 2..12 allowed). ImageNative::width / height stay the GUEST
// size, so every piece of guest logic (EDRAM tiles and their ownership, resolve rectangles, texture identity by address
// and size, 1/size of the texture constants) is unchanged; only the Vulkan commands use host coordinates, converted
// here. The EDRAM becomes a "virtual EDRAM" q/4 times larger in each axis: an 80x16-sample tile is 20q x 4q host
// samples, so every view of the same tiles (1x, 4x MSAA aliases, 32/64 bpp, depth or color) keeps its exact relation to
// the others and the transfer shaders run the same arithmetic with the scaled tile size.
//
// A guest pixel edge e maps to the host edge E(e) = ceil(e * q / 4 - 1/2): the first host pixel whose centre lies at or
// after guest coordinate e. Adjacent guest rectangles therefore cover disjoint host pixels with no gap, and with q = 4
// every function here is the identity (the default 720p path issues exactly the same commands as before).
#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace sa2::native {

constexpr uint32_t kScaleQuartersNative = 4;  // q of the game's own resolution (1280x720)

// Host edge of a guest pixel edge (any sign).
constexpr int32_t ScaleEdge(int32_t guest, uint32_t q) {
  if (q == kScaleQuartersNative) return guest;
  // floor((2 * guest * q + 3) / 8), with a floor division for negative values.
  const int64_t n = 2 * int64_t(guest) * int64_t(q) + 3;
  return int32_t(n >= 0 ? n / 8 : -((-n + 7) / 8));
}

// Host size of a guest extent that starts at 0 (image sizes).
constexpr uint32_t ScaleSize(uint32_t guest, uint32_t q) {
  return uint32_t(ScaleEdge(int32_t(guest), q));
}

// Host rectangle of a guest rectangle: both edges converted, so neighbours never overlap.
inline VkRect2D ScaleRect(const VkRect2D& guest, uint32_t q) {
  if (q == kScaleQuartersNative) return guest;
  const int32_t x0 = ScaleEdge(guest.offset.x, q), y0 = ScaleEdge(guest.offset.y, q);
  const int32_t x1 = ScaleEdge(guest.offset.x + int32_t(guest.extent.width), q);
  const int32_t y1 = ScaleEdge(guest.offset.y + int32_t(guest.extent.height), q);
  return VkRect2D{{x0, y0}, {uint32_t(x1 > x0 ? x1 - x0 : 0), uint32_t(y1 > y0 ? y1 - y0 : 0)}};
}

// The same for an offset + extent pair (VkImageCopy, VkImageBlit, VkBufferImageCopy, push constants).
inline void ScaleOffsetExtent(int32_t& x, int32_t& y, uint32_t& width, uint32_t& height, uint32_t q) {
  if (q == kScaleQuartersNative) return;
  const VkRect2D r = ScaleRect(VkRect2D{{x, y}, {width, height}}, q);
  x = r.offset.x;
  y = r.offset.y;
  width = r.extent.width;
  height = r.extent.height;
}

inline VkImageCopy ScaleCopy(VkImageCopy copy, uint32_t q) {
  if (q == kScaleQuartersNative) return copy;
  uint32_t width = copy.extent.width, height = copy.extent.height;
  int32_t sx = copy.srcOffset.x, sy = copy.srcOffset.y;
  ScaleOffsetExtent(sx, sy, width, height, q);
  uint32_t dw = copy.extent.width, dh = copy.extent.height;
  int32_t dx = copy.dstOffset.x, dy = copy.dstOffset.y;
  ScaleOffsetExtent(dx, dy, dw, dh, q);
  copy.srcOffset.x = sx;
  copy.srcOffset.y = sy;
  copy.dstOffset.x = dx;
  copy.dstOffset.y = dy;
  // Source and destination rectangles of the same guest size can differ by one host pixel when their guest
  // offsets differ in parity (q not a multiple of 4): copy what both have.
  copy.extent.width = width < dw ? width : dw;
  copy.extent.height = height < dh ? height : dh;
  return copy;
}

// The EDRAM tile in host samples: 80x16 guest samples scaled (20q x 4q).
constexpr uint32_t ScaledTileWidth(uint32_t q) { return 20u * q; }
constexpr uint32_t ScaledTileHeight(uint32_t q) { return 4u * q; }

}  // namespace sa2::native
