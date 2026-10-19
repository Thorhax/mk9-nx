#pragma once

#include <cstdint>
#include <optional>

namespace me::native {

// SDK draw.cpp SanitizeCopySampleSelect: depth never averages. Native guest2x
// folds sample2/3 onto0/1; k01 and the other combined selectors become k0.
// This accepts a raw THREE-BIT field, not an arbitrary RB_COPY_CONTROL word.
constexpr std::optional<uint32_t> SanitizeNative2xDepthCopySampleSelect(uint32_t raw) {
  if (raw > 7) return std::nullopt;
  return raw < 4 ? (raw & 1u) : 0u;
}

// Standard native2x only (not guest2x hosted as4x, which has a different map).
constexpr std::optional<uint32_t> Native2xDepthHostSample(uint32_t sanitized) {
  if (sanitized > 1) return std::nullopt;
  return sanitized ^ 1u;
}

}  // namespace me::native
