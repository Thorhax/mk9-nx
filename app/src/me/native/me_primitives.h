#pragma once

#include <cstdint>
#include <span>
#include <vector>

namespace me::native {

// Vulkan portability devices need triangle lists. Reset starts a new fan,
// preserving the winding of (center, previous, next) in each segment.
template <typename Allocator>
inline void ConvertFan(std::span<const uint32_t> input,
                            std::vector<uint32_t, Allocator>& output) {
  output.clear();
  uint32_t center = 0, previous = 0, count = 0;
  for (uint32_t index : input) {
    if (index == UINT32_MAX) { count = 0; continue; }
    if (count == 0) center = index;
    else if (count >= 2) {
      output.push_back(center);
      output.push_back(previous);
      output.push_back(index);
    }
    previous = index;
    if (count < 2) ++count;
  }
}

}  // namespace me::native
