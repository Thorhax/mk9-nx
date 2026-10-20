// sa2-nx (2026-10-20): whether two 64-byte lines differ, for the device chunks of the delta chain (backend/queued.cpp)
// and the pushed shared constants' lines (render/draw.cpp). GCC left the eight 64-bit words of the plain version scalar
// (about 30 instructions a line: 9 % of the game thread in the second console run); four 128-bit loads a side, XORed,
// ORed and one horizontal max are about 10 (sa2_renut_neon_compare).
#pragma once

#include <cstdint>
#include <cstring>

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace sa2::renut {

inline bool Line64DiffersPlain(const uint8_t *a, const uint8_t *b) {
  uint64_t diff = 0;
  for (uint32_t k = 0; k < 64; k += 8) {
    uint64_t x, y;
    std::memcpy(&x, a + k, 8);
    std::memcpy(&y, b + k, 8);
    diff |= x ^ y;
  }
  return diff != 0;
}

inline bool Line64Differs(const uint8_t *a, const uint8_t *b, bool neon) {
#if defined(__aarch64__)
  if (neon) {
    const uint8x16x4_t x = vld1q_u8_x4(a);
    const uint8x16x4_t y = vld1q_u8_x4(b);
    const uint8x16_t d = vorrq_u8(vorrq_u8(veorq_u8(x.val[0], y.val[0]), veorq_u8(x.val[1], y.val[1])),
                                  vorrq_u8(veorq_u8(x.val[2], y.val[2]), veorq_u8(x.val[3], y.val[3])));
    return vmaxvq_u32(vreinterpretq_u32_u8(d)) != 0;
  }
#else
  (void)neon;
#endif
  return Line64DiffersPlain(a, b);
}

}  // namespace sa2::renut
