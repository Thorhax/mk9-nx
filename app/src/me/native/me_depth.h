#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>

namespace me::native {

// Xenos unsigned float24: 20 mantissa bits, 4 exponent bits, bias 15.
// All encodings are finite; 0xFFFFFF represents 2 - 2^-20, not NaN.
constexpr float DecodeDepthFloat20e4(uint32_t bits) {
  bits &= 0xFFFFFFu;
  if (!bits) return 0.0f;
  uint32_t mantissa = bits & 0xFFFFFu;
  int32_t exponent = int32_t(bits >> 20);
  if (!exponent) {
    const int32_t shift = int32_t(std::countl_zero(mantissa)) - 11;
    exponent = 1 - shift;
    mantissa = (mantissa << shift) & 0xFFFFFu;
  }
  return std::bit_cast<float>((uint32_t(exponent + 112) << 23) | (mantissa << 3));
}

// RB_DEPTH_CLEAR stores stencil in bits 0..7 and the guest depth in 8..31.
// Legacy modes store guest [0,1] directly. MODE4 FLOAT24 uses a coherent half
// range together with its viewport, shader variants, EDRAM transfers and resolves.
constexpr float NativeDepthToGuest(float depth, bool float24_half) {
  return float24_half ? depth * 2.0f : depth;
}

constexpr float GuestDepthToNative(float depth, bool float24_half) {
  return depth * (float24_half ? 0.5f : 1.0f);
}

constexpr float NativeDepthClearValue(uint32_t depth_info, uint32_t depth_clear,
                                       bool float24_half = false) {
  const uint32_t depth = depth_clear >> 8;
  const float decoded = ((depth_info >> 16) & 1u)
                            ? DecodeDepthFloat20e4(depth)
                            : float(depth) / 16777215.0f;
  return std::clamp(GuestDepthToNative(decoded, float24_half), 0.0f, 1.0f);
}

}  // namespace me::native
