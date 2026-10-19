#pragma once

#include <cstdint>
#include <limits>

namespace me::native {

enum class ResolvedSamplingMode { Direct, Crop, Unsupported };
enum class ResolvedSamplingFailure {
  None, UnprovenLayout, IncompatibleFormat, EmptyExtent, WrongPitch,
  ExtentOutsideBacking, InvalidTexelSize, Overflow, CopyLimit, SourceBounds,
};

// Compatibility means equal SDK GetBaseFormat, not equality of raw guest enums.
// The caller must prove the flags: this helper cannot infer them from an address.
struct ResolvedSamplingContract {
  uint32_t logical_width = 0;
  uint32_t logical_height = 0;
  uint32_t guest_pitch = 0;
  uint32_t backing_width = 0;
  uint32_t backing_height = 0;
  uint32_t bytes_per_texel = 0;
  bool base_formats_compatible = false;
  bool source_origin_zero = false;
  bool single_level_2d = false;  // No mips, arrays, stacked textures or volumes.
  bool scale_1x = false;
};

struct ResolvedSamplingPlan {
  ResolvedSamplingMode mode = ResolvedSamplingMode::Unsupported;
  ResolvedSamplingFailure failure = ResolvedSamplingFailure::None;
  uint64_t texel_count = 0;
  uint64_t copy_bytes = 0;  // Tight output footprint, including for Direct.
  uint64_t source_bytes_required = 0;  // Last used source row + logical width.
};

inline ResolvedSamplingPlan AnalyzeResolvedSampling(
    const ResolvedSamplingContract& c,
    uint64_t max_copy_bytes = std::numeric_limits<uint64_t>::max(),
    uint64_t source_payload_bytes = std::numeric_limits<uint64_t>::max()) {
  ResolvedSamplingPlan result;
  const auto reject = [&](ResolvedSamplingFailure reason) {
    result.failure = reason;
    return result;
  };
  if (!c.source_origin_zero || !c.single_level_2d || !c.scale_1x)
    return reject(ResolvedSamplingFailure::UnprovenLayout);
  if (!c.base_formats_compatible)
    return reject(ResolvedSamplingFailure::IncompatibleFormat);
  if (!c.logical_width || !c.logical_height || !c.backing_width || !c.backing_height)
    return reject(ResolvedSamplingFailure::EmptyExtent);
  if (c.guest_pitch != c.backing_width)
    return reject(ResolvedSamplingFailure::WrongPitch);
  if (c.logical_width > c.backing_width || c.logical_height > c.backing_height)
    return reject(ResolvedSamplingFailure::ExtentOutsideBacking);
  if (!c.bytes_per_texel)
    return reject(ResolvedSamplingFailure::InvalidTexelSize);
  const uint64_t texels = uint64_t(c.logical_width) * c.logical_height;
  const uint64_t source_texels = uint64_t(c.logical_height - 1) * c.backing_width + c.logical_width;
  constexpr uint64_t kMax = std::numeric_limits<uint64_t>::max();
  if (texels > kMax / c.bytes_per_texel || source_texels > kMax / c.bytes_per_texel)
    return reject(ResolvedSamplingFailure::Overflow);
  const uint64_t bytes = texels * c.bytes_per_texel;
  const uint64_t source_bytes = source_texels * c.bytes_per_texel;
  if (bytes > max_copy_bytes) return reject(ResolvedSamplingFailure::CopyLimit);
  if (source_bytes > source_payload_bytes) return reject(ResolvedSamplingFailure::SourceBounds);
  result.texel_count = texels;
  result.copy_bytes = bytes;
  result.source_bytes_required = source_bytes;
  result.mode = c.logical_width == c.backing_width && c.logical_height == c.backing_height
                    ? ResolvedSamplingMode::Direct : ResolvedSamplingMode::Crop;
  return result;
}

}  // namespace me::native
