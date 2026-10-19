#pragma once

#include <cstdint>
#include <span>

namespace me::native {

// Hardware VkSampleCountFlagBits values, without a Vulkan dependency. They are
// NOT guest MSAA enums (0/1/2), physical EDRAM X/Y shifts, or legacy raster grids.
// This deliberately bounded backend contract accepts 1x/2x/4x only. It does not
// establish format/device support, sample ordering, or physical-alias semantics.
constexpr bool IsSupportedHardwareSampleCount(uint32_t count) {
  return count == 1 || count == 2 || count == 4;
}

constexpr bool IsSingleSample(uint32_t count) {
  return count == 1;
}

// Zero means an absent attachment. All five slots absent, unknown counts, and
// mixed counts fail closed with zero; the caller must not use zero as VkSamples.
constexpr uint32_t CommonHardwareSampleCount(std::span<const uint32_t, 5> attachments) {
  uint32_t common = 0;
  for (uint32_t count : attachments) {
    if (!count) continue;
    if (!IsSupportedHardwareSampleCount(count) || (common && common != count)) return 0;
    common = count;
  }
  return common;
}

// Require the exact supported sample bit, not a numeric comparison or any bit
// overlap with a combined/invalid count such as 3. Other mask bits are allowed.
constexpr bool IsHardwareSampleCountSupportedByMask(uint32_t count, uint32_t mask) {
  return IsSupportedHardwareSampleCount(count) && (mask & count) == count;
}

// Command sample-count prerequisites only. Equal counts do not prove compatible
// formats/aspects, usage/layouts, bounds, ownership, or EDRAM sample mappings.
constexpr bool CanCopyImageSamples(uint32_t source, uint32_t destination) {
  return IsSupportedHardwareSampleCount(source) && source == destination;
}

constexpr bool CanBlitImageSamples(uint32_t source, uint32_t destination) {
  return IsSingleSample(source) && IsSingleSample(destination);
}

constexpr bool CanBufferImageTransferSamples(uint32_t image) {
  return IsSingleSample(image);
}

}  // namespace me::native
