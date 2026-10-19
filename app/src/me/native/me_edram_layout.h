#pragma once
#include <cstdint>

namespace me::native {
// Xenos tiles contain 80 32-bit words x 16 physical sample rows.
// 2x MSAA expands Y, 4x expands X and Y. This describes addressing only.
struct EdramLayout {
  uint32_t x_log2, y_log2, words_log2;
  constexpr EdramLayout(uint32_t msaa, bool wide)
      : x_log2(msaa >= 2), y_log2(msaa >= 1), words_log2(wide) {}
  constexpr uint32_t pitch_tiles(uint32_t pitch) const {
    return (((pitch << x_log2) + 79) / 80) << words_log2;
  }
  constexpr uint32_t tile_width() const { return 80 >> (words_log2 + x_log2); }
  constexpr uint32_t tile_height() const { return 16 >> y_log2; }
};
}
