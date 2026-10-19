// color render target formats beyond k_8_8_8_8 for the native renderer.
// Unreal Engine 3 draws the scene into k_2_10_10_10_FLOAT(_AS_16_16_16_16) and k_16_16_16_16_FLOAT
// targets. Host formats and clear values are the SDK's own for host render targets
// (the SDK's render target cache: GetColorVulkanFormat and the resolve clear).
#pragma once

#include <rex/graphics/xenos.h>
#include <rex/math.h>
#include <rex/ui/vulkan/api.h>

#include <cstdint>
#include <cstring>

namespace sa2::native {

// Host image format of a color render target (xenos::ColorRenderTargetFormat); VK_FORMAT_UNDEFINED
// for formats the native renderer does not draw into yet.
inline VkFormat HostFormatTargetColor(uint32_t format) {
  using F = rex::graphics::xenos::ColorRenderTargetFormat;
  switch (F(format)) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_GAMMA:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_10_10_10_10:
      // Host storage is FP16, but UNORM10 and 7e3 remain distinct guest storage classes.
      // Aliasing must reinterpret packed EDRAM words, not copy floating-point texels.
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case F::k_2_10_10_10_FLOAT:
    case F::k_2_10_10_10_FLOAT_AS_16_16_16_16:
    case F::k_16_16_16_16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case F::k_16_16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
    case F::k_32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    case F::k_32_32_FLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

// Host image format of a resolved texture (xenos::ColorFormat of RB_COPY_DEST_INFO).
inline VkFormat HostFormatResolved(uint32_t format) {
  using F = rex::graphics::xenos::ColorFormat;
  switch (F(format)) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_A:
    case F::k_8_8_8_8_AS_16_16_16_16:
      return VK_FORMAT_R8G8B8A8_UNORM;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_16_16_16_16:
      return VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    case F::k_16_16_16_16_FLOAT:
      return VK_FORMAT_R16G16B16A16_SFLOAT;
    case F::k_16_16_FLOAT:
      return VK_FORMAT_R16G16_SFLOAT;
    case F::k_32_FLOAT:
      return VK_FORMAT_R32_SFLOAT;
    case F::k_32_32_FLOAT:
      return VK_FORMAT_R32G32_SFLOAT;
    // banjo-nx: Nuts & Bolts also resolves into these (k_16_FLOAT: the one-channel luminance chain of its
    // auto-exposure). Same host formats as TextureFormatFor, so the resolved image is sampled as the texture.
    case F::k_8:
      return VK_FORMAT_R8_UNORM;
    case F::k_8_8:
      return VK_FORMAT_R8G8_UNORM;
    case F::k_16:
      return VK_FORMAT_R16_UNORM;
    case F::k_16_16:
      return VK_FORMAT_R16G16_UNORM;
    case F::k_16_16_16_16:
      return VK_FORMAT_R16G16B16A16_UNORM;
    case F::k_16_FLOAT:
      return VK_FORMAT_R16_SFLOAT;
    default:
      return VK_FORMAT_UNDEFINED;
  }
}

// RB_COLOR_CLEAR (low 32 bits) and RB_COLOR_CLEAR_LO (high) as a Vulkan clear color.
inline VkClearColorValue ClearValueColor(uint32_t format, uint64_t value) {
  using F = rex::graphics::xenos::ColorRenderTargetFormat;
  VkClearColorValue color{};
  switch (F(format)) {
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_10_10_10_10:
      for (uint32_t j = 0; j < 3; ++j) {
        color.float32[j] = float((value >> (j * 10)) & 0x3FF) * (1.0f / 0x3FF);
      }
      color.float32[3] = float((value >> 30) & 0x3) * (1.0f / 0x3);
      break;
    case F::k_2_10_10_10_FLOAT:
    case F::k_2_10_10_10_FLOAT_AS_16_16_16_16:
      for (uint32_t j = 0; j < 3; ++j) {
        color.float32[j] = rex::graphics::xenos::Float7e3To32(uint32_t(value >> (j * 10)) & 0x3FF);
      }
      color.float32[3] = float((value >> 30) & 0x3) * (1.0f / 0x3);
      break;
    case F::k_16_16_FLOAT:
    case F::k_16_16_16_16_FLOAT:
      for (uint32_t j = 0; j < 4; ++j) {
        color.float32[j] = rex::xenos_half_to_float(uint16_t(value >> (j * 16)));
      }
      break;
    case F::k_32_FLOAT:
    case F::k_32_32_FLOAT: {
      const uint32_t c[2] = {uint32_t(value), uint32_t(value >> 32)};
      std::memcpy(color.float32, c, sizeof(c));
      break;
    }
    default:
      for (uint32_t j = 0; j < 4; ++j) {
        color.float32[j] = float((value >> (j * 8)) & 0xFF) * (1.0f / 255.0f);
      }
      break;
  }
  return color;
}

}  // namespace sa2::native
