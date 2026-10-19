#pragma once

// Derived from ReXGlue / Xenia draw-extent and render-target-height planning.
// Copyright (c) 2026, Tom Clay <tomc@tctechstuff.com>
// Copyright (c) 2022, Ben Vanik and Xenia project contributors
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>

namespace me::native {

// Raw guest register words, not native viewport/scissor state. No GPU dependency.
struct DrawExtentRegisters {
  uint32_t window_offset = 0;
  uint32_t window_scissor_tl = 0;
  uint32_t window_scissor_br = 0;
  uint32_t screen_scissor_br = 0;
  uint32_t clip_control = 0;
  uint32_t sc_mode_control = 0;
  uint32_t vertex_control = 0;
  uint32_t vte_control = 0;
  uint32_t viewport_y_scale = 0;
  uint32_t viewport_y_offset = 0;
};

constexpr int32_t DrawExtentSigned15(uint32_t word) {
  const uint32_t low = word & 0x7FFFu;
  return int32_t(low) - int32_t((low & 0x4000u) << 1);
}

// SDK DrawExtentEstimator::EstimateMaxY, with the optional (default-disabled)
// CPU vertex interpreter omitted. Unclipped draws therefore retain the SDK's
// conservative scissor bound. This is NOT a geometry coverage estimate.
inline uint32_t EstimateDrawMaxY(const DrawExtentRegisters& r) {
  const int32_t window_y = DrawExtentSigned15(r.window_offset >> 16);
  int32_t bottom = int32_t((r.window_scissor_br >> 16) & 0x3FFFu);
  if (!(r.window_scissor_tl & 0x80000000u)) bottom += window_y;
  bottom = std::min(bottom, DrawExtentSigned15(r.screen_scissor_br >> 16));
  uint32_t max_y = uint32_t(std::max(bottom, int32_t(0)));
  if (!(r.clip_control & 0x10000u)) {
    // Preserve SDK addition order and its first-argument NaN handling.
    float viewport_bottom = 0.0f;
    if (r.sc_mode_control & 0x10000u) viewport_bottom += float(window_y);
    if (!(r.vertex_control & 1u)) viewport_bottom += 0.5f;
    if (r.vte_control & 8u)
      viewport_bottom += std::bit_cast<float>(r.viewport_y_offset);
    viewport_bottom += (r.vte_control & 4u)
        ? std::abs(std::bit_cast<float>(r.viewport_y_scale)) : 1.0f;
    max_y = uint32_t(std::min(float(max_y), std::max(0.0f, viewport_bottom)));
  }
  return max_y;
}

// Exact valid-input SDK GetRenderTargetHeight formula: ceil physical tile rows,
// then FLOOR rows at the guest/host limit. Returned height is in guest pixels.
// 64bpp does not change the height allocation, only the bound tile range.
constexpr uint32_t DrawRenderTargetHeight(uint32_t pitch_tiles_at_32bpp,
                                          uint32_t msaa, uint32_t host_max_height,
                                          uint32_t resolution_scale_y = 1) {
  if (!pitch_tiles_at_32bpp || msaa > 2 || !resolution_scale_y) return 0;
  const uint64_t rows = (2048ull + pitch_tiles_at_32bpp - 1) / pitch_tiles_at_32bpp;
  const uint64_t max_scaled = std::min<uint64_t>(8192ull * resolution_scale_y,
                                                host_max_height);
  const uint32_t sample_y_log2 = uint32_t(msaa >= 1);
  const uint64_t limited_rows = (max_scaled << sample_y_log2) /
                                (16ull * resolution_scale_y);
  return uint32_t(std::min(rows, limited_rows) * (16u >> sample_y_log2));
}

inline uint32_t EstimateDrawUsedHeight(const DrawExtentRegisters& r,
                                       uint32_t pitch_pixels, uint32_t msaa,
                                       uint32_t host_max_height) {
  if (msaa > 2 || pitch_pixels > 8192) return 0;
  const uint32_t pitch_tiles = ((pitch_pixels << uint32_t(msaa >= 2)) + 79) / 80;
  return std::min(EstimateDrawMaxY(r),
                  DrawRenderTargetHeight(pitch_tiles, msaa, host_max_height));
}

}  // namespace me::native
