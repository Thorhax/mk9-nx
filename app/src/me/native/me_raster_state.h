#pragma once
#include <cstdint>

namespace me::native {
inline constexpr uint32_t kNativeDepthClamp = 0x20;
// Renderer-only safety identity, not a Vulkan raster state. Survives EDS keys.
inline constexpr uint32_t kNativeMsaa2PhaseProbe = 0x40;
// Phase2 translates Y in the opposite direction. Preserve it in module/prewarm
// identity; the generic phase bit alone cannot select an inverse FragCoord XY.
inline constexpr uint32_t kNativeMsaa2PhaseMode2 = 0x80;
inline constexpr uint32_t kNativeDynamicRasterState = 0x1F;
struct Msaa2PhaseProbe {
  bool valid;
  bool active;
  float geometry_x;
  float geometry_y;
};
// OFF-by-default single-sample experiments, NOT complete Xenos MSAA support.
// 1 emulates SDK native-2x guest sample0 at (-.25,-.25) from pixel center.
// 2 cancels the measured (-.25,+.25) viewport jitter in the failing ME chain;
//   this is a causal diagnostic, NOT an asserted universal sample pattern.
// Geometry translations are opposite to sampling offsets. Disabled clipping
// utility/clear rectangles are deliberately left unchanged.
constexpr Msaa2PhaseProbe GetMsaa2PhaseProbe(uint32_t mode, uint32_t msaa,
                                           bool clipping_disabled, uint32_t vte) {
  if (mode > 2 || msaa > 2) return {false, false, 0, 0};
  if (!mode || msaa != 1 || clipping_disabled) return {true, false, 0, 0};
  if ((vte & 0xF) != 0xF) return {false, false, 0, 0};
  return {true, true, .25f, mode == 1 ? .25f : -.25f};
}
constexpr bool XenosDepthClamp(uint32_t clip_control, bool supported, bool force = false) {
  return supported && (force || (clip_control & (1u << 16)));
}
// EDS1/2 covers cull/front/restart/bias, not depth clamp or safety identities.
constexpr uint32_t StaticRasterKey(uint32_t raster_state) {
  return raster_state & ~kNativeDynamicRasterState;
}
}
