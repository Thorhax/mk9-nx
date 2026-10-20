// sa2-nx (2026-10-20): the game's frame pacing with Renut's renderer (sa2_renut_pacing.cpp).
#pragma once

#include <cstdint>

namespace sa2::renut {

// Called by the Swap hook before the frame goes to the render thread: waits until the game's presentation interval
// since the previous Swap has passed (the console's vertical blank).
void PaceSwap(uint32_t device_va);

}  // namespace sa2::renut
