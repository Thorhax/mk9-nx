/**
 * @file    gpu/guest/sa2_screen.cpp
 * @brief   sa2-nx (2026-10-20): the game's screen width for Renut's renderer.
 *
 * Renut's guest/widescreen.cpp widens N&B's screen through N&B's own code; SA2
 * has none of that code, so its screen is the console's 1280x720 (the
 * renderer's options read the width, graphics_options.cpp).
 */
#include "renut_engine/gpu/guest/widescreen.h"

namespace renut::gpu {

uint32_t GuestScreenWidth() {
  return 1280;
}

} // namespace renut::gpu
