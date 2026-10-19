#pragma once

// Derived from ReXGlue / Xenia render-target range planning.
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

#include <array>
#include <cstdint>

namespace me::native {

// usf4-nx: 4096, twice the Xbox 360's 2048 tiles, so surfaces that run past tile 2048 (USF4's one-tile scene) do not
// wrap onto tile 0 (sa2_native_targets.cpp, kTilesEDRAM4).
inline constexpr uint32_t kOwnershipEdramTiles = 4096;

// CPU-only extraction of ReXGlue/Xenia RenderTargetCache::Update's bound-range
// planning (pipeline/render_target/cache.cpp). This is not an ownership map or
// a converter, and makes no claim that native draw/clear contents are correct.
// Xbox RB_SURFACE_INFO supplies one pitch and sample count for all five views.
struct EdramBoundView {
  uint32_t base = 0;
  bool wide = false;  // 64bpp color; depth is always 32bpp.
};

struct EdramBoundRequest {
  std::array<EdramBoundView, 5> views{};  // depth, color0..3
  uint32_t pitch_pixels = 0;
  uint32_t msaa = 0;  // guest enumeration: 0 = 1x, 1 = 2x, 2 = 4x
  // Guest pixel height ALREADY capped/estimated by the caller, not allocation
  // height. The SDK obtains this from GetRenderTargetHeight + EstimateMaxY.
  uint32_t used_height_pixels = 0;
  uint16_t normalized_color_mask = 0;  // four channel bits per color attachment
  bool depth_used = false;  // normalized depth/stencil use, including reads
  bool clamp_to_min = true;  // SDK mrt_edram_used_range_clamp_to_min policy
};

struct EdramBoundPlan {
  bool valid = false;
  uint32_t requested_mask = 0;
  uint32_t active_mask = 0;
  uint32_t suppressed_mask = 0;
  uint32_t pitch_tiles_at_32bpp = 0;
  std::array<uint32_t, 5> length_tiles{};  // indexed by attachment, NOT sorted rank
  std::array<uint32_t, 5> sorted_slots{};
  uint32_t count = 0;
};

constexpr EdramBoundPlan PlanEdramBoundRanges(const EdramBoundRequest& request) {
  EdramBoundPlan result;
  if (request.msaa > 2 || request.pitch_pixels > 8192) return result;
  if (request.depth_used) result.requested_mask |= 1;
  for (uint32_t color = 0; color < 4; ++color) {
    if ((request.normalized_color_mask >> (color * 4)) & 15)
      result.requested_mask |= 1u << (color + 1);
  }
  for (uint32_t slot = 0; slot < 5; ++slot) {
    if (!(result.requested_mask & (1u << slot))) continue;
    if (request.views[slot].base >= kOwnershipEdramTiles ||
        (slot == 0 && request.views[slot].wide)) return result;
  }
  result.valid = true;
  if (!request.pitch_pixels) return result;  // SDK disables rasterization.
  result.pitch_tiles_at_32bpp =
      ((request.pitch_pixels << uint32_t(request.msaa >= 2)) + 79) / 80;
  result.active_mask = result.requested_mask;
  // Same-base priority: earlier color attachment wins, and color beats depth.
  // This is the SDK rule, specifically documented for ME1's lighting pass.
  for (uint32_t slot = 1; slot < 5; ++slot) {
    if (!(result.active_mask & (1u << slot))) continue;
    if ((result.active_mask & 1) && request.views[0].base == request.views[slot].base)
      result.active_mask &= ~1u;
    for (uint32_t other = slot + 1; other < 5; ++other) {
      if (request.views[other].base == request.views[slot].base)
        result.active_mask &= ~(1u << other);
    }
  }
  result.suppressed_mask = result.requested_mask & ~result.active_mask;
  for (uint32_t slot = 0; slot < 5; ++slot) {
    if (!(result.active_mask & (1u << slot))) continue;
    uint32_t pos = result.count++;
    while (pos && request.views[result.sorted_slots[pos - 1]].base > request.views[slot].base) {
      result.sorted_slots[pos] = result.sorted_slots[pos - 1];
      --pos;
    }
    result.sorted_slots[pos] = slot;
  }
  if (!result.count) return result;
  uint32_t max_distance_at_64bpp = kOwnershipEdramTiles * 2;
  if (request.clamp_to_min && result.count >= 2) {
    for (uint32_t rank = 0; rank < result.count; ++rank) {
      const auto& current = request.views[result.sorted_slots[rank]];
      uint32_t next_base = rank + 1 < result.count
          ? request.views[result.sorted_slots[rank + 1]].base
          : kOwnershipEdramTiles + request.views[result.sorted_slots[0]].base;
      const uint32_t distance = (next_base - current.base) << uint32_t(!current.wide);
      if (distance < max_distance_at_64bpp) max_distance_at_64bpp = distance;
    }
  }
  // uint64_t prevents malformed caller heights from overflowing before the
  // physical-ring caps; hardware-normal input matches the SDK uint32 formula.
  const uint64_t used_at_32bpp =
      ((uint64_t(request.used_height_pixels) << uint32_t(request.msaa >= 1)) + 15) / 16 *
      result.pitch_tiles_at_32bpp;
  for (uint32_t rank = 0; rank < result.count; ++rank) {
    const uint32_t slot = result.sorted_slots[rank];
    const auto& current = request.views[slot];
    const uint32_t next_base = rank + 1 < result.count
        ? request.views[result.sorted_slots[rank + 1]].base
        : kOwnershipEdramTiles + request.views[result.sorted_slots[0]].base;
    uint64_t length = used_at_32bpp << uint32_t(current.wide);
    const uint32_t min_distance = max_distance_at_64bpp >> uint32_t(!current.wide);
    if (length > min_distance) length = min_distance;
    if (length > next_base - current.base) length = next_base - current.base;
    result.length_tiles[slot] = uint32_t(length);
  }
  return result;
}

// sa2-nx round 6 (sa2_cpu_ring_bound_plan_memo): field-wise equality of two requests and of two plans, for a memo of
// PlanEdramBoundRanges (a pure function of its request; consecutive draws usually ask the same).
constexpr bool SameBoundRequest(const EdramBoundRequest& a, const EdramBoundRequest& b) {
  for (uint32_t i = 0; i < 5; ++i)
    if (a.views[i].base != b.views[i].base || a.views[i].wide != b.views[i].wide) return false;
  return a.pitch_pixels == b.pitch_pixels && a.msaa == b.msaa && a.used_height_pixels == b.used_height_pixels &&
         a.normalized_color_mask == b.normalized_color_mask && a.depth_used == b.depth_used &&
         a.clamp_to_min == b.clamp_to_min;
}
constexpr bool SameBoundPlan(const EdramBoundPlan& a, const EdramBoundPlan& b) {
  return a.valid == b.valid && a.requested_mask == b.requested_mask && a.active_mask == b.active_mask &&
         a.suppressed_mask == b.suppressed_mask && a.pitch_tiles_at_32bpp == b.pitch_tiles_at_32bpp &&
         a.length_tiles == b.length_tiles && a.sorted_slots == b.sorted_slots && a.count == b.count;
}

struct EdramPhysicalSpan {
  uint32_t start = 0;
  uint32_t end = 0;  // exclusive physical tile; may equal kOwnershipEdramTiles
};

struct EdramWrappedRange {
  bool valid = false;
  std::array<EdramPhysicalSpan, 2> spans{};
  uint32_t count = 0;
};

// Split exactly as SDK ChangeOwnership: at most one full EDRAM period.
constexpr EdramWrappedRange WrapEdramRange(uint32_t base, uint32_t start_relative,
                                           uint32_t length) {
  EdramWrappedRange result;
  if (base >= kOwnershipEdramTiles || length > kOwnershipEdramTiles ||
      start_relative > kOwnershipEdramTiles - uint32_t(length != 0)) return result;
  result.valid = true;
  if (!length) return result;
  const uint32_t start = (base + start_relative) & (kOwnershipEdramTiles - 1);
  const uint32_t end = start + length;
  result.spans[result.count++] = {start, end < kOwnershipEdramTiles ? end : kOwnershipEdramTiles};
  if (end > kOwnershipEdramTiles)
    result.spans[result.count++] = {0, end & (kOwnershipEdramTiles - 1)};
  return result;
}

struct EdramRelativeSpan {
  uint32_t start = 0;
  uint32_t length = 0;
  constexpr uint32_t end() const { return start + length; }
  constexpr bool contains(uint32_t tile) const { return tile >= start && tile < end(); }
};

struct EdramClearRequest {
  uint32_t depth_base = 0, color_base = 0;
  uint32_t pitch_tiles_at_32bpp = 0;
  uint32_t msaa = 0;
  bool clear_depth = false, clear_color = false, color_wide = false;
  // The SDK PrepareHostRenderTargetsResolveClear's FINAL clear_rectangle:
  // includes ResolveInfo base-offset rows and offsets, already clamped to the
  // tile-rounded pitch / permitted guest RT height. Not raw copy vertices.
  uint32_t x_pixels = 0, y_pixels = 0, width_pixels = 0, height_pixels = 0;
};

struct EdramClearPlan {
  bool valid = false;
  EdramRelativeSpan depth{}, color{};
};

// Exact range step of SDK PrepareHostRenderTargetsResolveClear. Contiguous
// ranges include the gaps between short rectangle rows; import/clear-cutout
// handling must preserve those pixels before publishing the entire range.
// The caller must constrain ACTUAL clear writes as well as ownership writes:
// merely truncating the owner publication does not undo out-of-range clears.
constexpr EdramClearPlan PlanEdramClearRanges(const EdramClearRequest& request) {
  EdramClearPlan result;
  const uint32_t mx = uint32_t(request.msaa >= 2), my = uint32_t(request.msaa >= 1);
  if (request.msaa > 2 || !request.pitch_tiles_at_32bpp ||
      request.pitch_tiles_at_32bpp > ((8192u << mx) + 79) / 80 ||
      (request.clear_depth && request.depth_base >= kOwnershipEdramTiles) ||
      (request.clear_color && request.color_base >= kOwnershipEdramTiles)) return result;
  const uint32_t pitch_pixels = request.pitch_tiles_at_32bpp * (80u >> mx);
  if (request.x_pixels > pitch_pixels ||
      request.width_pixels > pitch_pixels - request.x_pixels) return result;
  result.valid = true;
  if (!request.width_pixels || !request.height_pixels ||
      (!request.clear_depth && !request.clear_color)) return result;
  const uint64_t start32 = ((uint64_t(request.y_pixels) << my) / 16) *
      request.pitch_tiles_at_32bpp + ((uint64_t(request.x_pixels) << mx) / 80);
  const uint64_t end32 =
      (((uint64_t(request.y_pixels) + request.height_pixels - 1) << my) / 16) *
          request.pitch_tiles_at_32bpp +
      (((uint64_t(request.x_pixels) + request.width_pixels - 1) << mx) / 80) + 1;
  const auto cap = [](uint64_t value) constexpr {
    return uint32_t(value < kOwnershipEdramTiles ? value : kOwnershipEdramTiles);
  };
  if (request.clear_depth) {
    result.depth.start = cap(start32);
    result.depth.length = cap(end32) - result.depth.start;
  }
  if (request.clear_color) {
    result.color.start = cap(start32 << uint32_t(request.color_wide));
    result.color.length = cap(end32 << uint32_t(request.color_wide)) - result.color.start;
  }
  if (result.depth.length && result.color.length) {
    const uint32_t ds = (request.depth_base + result.depth.start) & (kOwnershipEdramTiles - 1);
    const uint32_t cs = (request.color_base + result.color.start) & (kOwnershipEdramTiles - 1);
    const uint32_t depth_distance = (cs < ds ? kOwnershipEdramTiles : 0) + cs - ds;
    const uint32_t color_distance = (ds < cs ? kOwnershipEdramTiles : 0) + ds - cs;
    if (result.depth.length > depth_distance) result.depth.length = depth_distance;
    if (result.color.length > color_distance) result.color.length = color_distance;
  }
  return result;
}

struct EdramPixelRectangle { uint32_t x = 0, y = 0, width = 0, height = 0; };
struct EdramClearRectangles {
  bool valid = false;
  std::array<EdramPixelRectangle, 3> rectangles{};
  uint32_t count = 0;
  constexpr bool whole(uint32_t width, uint32_t height) const {
    return count == 1 && !rectangles[0].x && !rectangles[0].y &&
        rectangles[0].width == width && rectangles[0].height == height;
  }
};

// Project a contiguous relative physical span into first/middle/last tile rows,
// intersected with the actual clear cutout. Missing physical padding is rejected
// rather than claiming bits that don't have host backing storage.
constexpr EdramClearRectangles PlanEdramClearRectangles(EdramRelativeSpan span,
    uint32_t pitch_tiles, uint32_t msaa, bool wide, uint32_t view_width, uint32_t view_height,
    EdramPixelRectangle cutout) {
  EdramClearRectangles result;
  if (!pitch_tiles || msaa > 2 || span.start > 2048 || span.length > 2048 - span.start)
    return result;
  result.valid = true;
  if (!span.length) return result;
  const uint32_t tile_width = 80u >> (uint32_t(wide) + uint32_t(msaa >= 2));
  const uint32_t tile_height = 16u >> uint32_t(msaa >= 1);
  if (uint64_t(pitch_tiles) * tile_width > view_width ||
      uint64_t((span.end() - 1) / pitch_tiles + 1) * tile_height > view_height ||
      uint64_t(cutout.x) + cutout.width > view_width ||
      uint64_t(cutout.y) + cutout.height > view_height) {
    result.valid = false;
    return result;
  }
  const auto append = [&](uint32_t tile_start, uint32_t width_tiles, uint32_t height_rows) constexpr {
    const uint32_t x = (tile_start % pitch_tiles) * tile_width;
    const uint32_t y = (tile_start / pitch_tiles) * tile_height;
    const uint32_t x0 = x > cutout.x ? x : cutout.x;
    const uint32_t y0 = y > cutout.y ? y : cutout.y;
    const uint32_t x1 = x + width_tiles * tile_width < cutout.x + cutout.width
        ? x + width_tiles * tile_width : cutout.x + cutout.width;
    const uint32_t y1 = y + height_rows * tile_height < cutout.y + cutout.height
        ? y + height_rows * tile_height : cutout.y + cutout.height;
    if (x1 > x0 && y1 > y0)
      result.rectangles[result.count++] = {x0, y0, x1 - x0, y1 - y0};
  };
  uint32_t cursor = span.start;
  if (cursor % pitch_tiles) {
    const uint32_t remaining_row = pitch_tiles - cursor % pitch_tiles;
    const uint32_t first = span.length < remaining_row ? span.length : remaining_row;
    append(cursor, first, 1);
    cursor += first;
  }
  const uint32_t rows = (span.end() - cursor) / pitch_tiles;
  if (rows) {
    append(cursor, pitch_tiles, rows);
    cursor += rows * pitch_tiles;
  }
  if (cursor < span.end()) append(cursor, span.end() - cursor, 1);
  return result;
}

}  // namespace me::native
