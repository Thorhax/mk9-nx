#pragma once

#include "me_shader_identity.h"
#include <cstddef>
#include <cstdint>
#include <span>

namespace me::native {
enum class VertexFetchSelectionFailure { None, InvalidDeclaredFetch, NoEligibleFetch, Ambiguous };
struct VertexFetchSelection {
  size_t word_offset = SIZE_MAX;
  VertexFetchSelectionFailure failure = VertexFetchSelectionFailure::NoEligibleFetch;
  explicit operator bool() const { return failure == VertexFetchSelectionFailure::None; }
};
inline uint32_t VertexFetchWrittenMask(uint32_t swizzle) {
  uint32_t mask = 0;
  for (uint32_t i = 0; i < 4; ++i)
    if (((swizzle >> (i * 3)) & 7) != 7) mask |= 1u << i;
  return mask;
}

// TEMP registers may be reused at different execution points. Only a complete
// ordered identity proof makes the declaration's own instruction authoritative.
// Otherwise, accept one uniquely eligible declared instruction, never the first
// same-TEMP fetch. This does not prove arbitrary reordered shader execution.
template <class FetchRange, class Projection>
inline VertexFetchSelection SelectVertexFetch(std::span<const uint32_t> original,
                                               std::span<const uint32_t> loaded,
                                               size_t own_instruction, bool ordered_identity,
                                               const FetchRange& declarations,
                                               Projection instruction_index) {
  const auto in_bounds = [](size_t instruction, size_t count) {
    return count >= 3 && instruction <= (count - 3) / 3;
  };
  if (original.size() != loaded.size() || !in_bounds(own_instruction, original.size()))
    return {SIZE_MAX, VertexFetchSelectionFailure::InvalidDeclaredFetch};
  const size_t own = own_instruction * 3;
  if ((original[own] & 31) != 0)
    return {SIZE_MAX, VertexFetchSelectionFailure::InvalidDeclaredFetch};
  const uint32_t target = (original[own] >> 12) & 63, swizzle = original[own + 1] & 4095;
  const auto eligible = [&](size_t word) {
    return (loaded[word] & 31) == 0 && ((loaded[word] >> 12) & 63) == target &&
           VertexFetchWrittenMask(loaded[word + 1] & 4095) == VertexFetchWrittenMask(swizzle) &&
           VertexFetchSwizzleRepresentable(swizzle, loaded[word + 1] & 4095);
  };
  if (ordered_identity)
    return eligible(own) ? VertexFetchSelection{own, VertexFetchSelectionFailure::None}
                         : VertexFetchSelection{SIZE_MAX, VertexFetchSelectionFailure::NoEligibleFetch};
  size_t selected = SIZE_MAX;
  for (const auto& declaration : declarations) {
    const size_t instruction = size_t(instruction_index(declaration));
    if (!in_bounds(instruction, loaded.size()))
      return {SIZE_MAX, VertexFetchSelectionFailure::InvalidDeclaredFetch};
    const size_t word = instruction * 3;
    if (!eligible(word) || word == selected) continue;
    if (selected != SIZE_MAX) return {SIZE_MAX, VertexFetchSelectionFailure::Ambiguous};
    selected = word;
  }
  return selected == SIZE_MAX ? VertexFetchSelection{}
                              : VertexFetchSelection{selected, VertexFetchSelectionFailure::None};
}
}  // namespace me::native
