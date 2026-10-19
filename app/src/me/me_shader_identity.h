#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <span>

namespace me::native {
enum class ShaderIdentityStage : uint8_t { Vertex, Pixel };
struct ShaderIdentityView {
  ShaderIdentityStage stage;
  std::span<const uint32_t> words;  // host-order sequencer words, not container bytes
};

// Full containers with different DEF/ABI metadata may share these words. Keep
// their identity when compatible; this only rejects a different GPU program.
inline bool PixelShaderIdentityMatches(ShaderIdentityView loaded, ShaderIdentityView selected) {
  return loaded.stage == ShaderIdentityStage::Pixel &&
         selected.stage == ShaderIdentityStage::Pixel && !loaded.words.empty() &&
         loaded.words.size() == selected.words.size() &&
         std::equal(loaded.words.begin(), loaded.words.end(), selected.words.begin());
}

// Exactly the mappings RemapCode + remapInput + the original HLSL fetch
// can express. Original constants are emitted after the fetch, and preserved
// destinations are not assigned by that HLSL at all. Reserved selector 6 has
// no supported meaning. Repeated input selectors share one remap component.
inline bool VertexFetchSwizzleRepresentable(uint32_t original, uint32_t patched) {
  uint32_t replacements[4]{7, 7, 7, 7};
  for (uint32_t destination = 0; destination < 4; ++destination) {
    const uint32_t before = (original >> (destination * 3)) & 7;
    const uint32_t after = (patched >> (destination * 3)) & 7;
    if (before == 6 || after == 6 || (before == 7) != (after == 7)) return false;
    if (before < 4) {
      if (after > 5 || (replacements[before] != 7 && replacements[before] != after)) return false;
      replacements[before] = after;
    } else if (before != after) return false;
  }
  return true;
}

// Experimental tolerance (cvar sa2_native_vs_identity_tolerance, default 0 = exact, as before): how many
// words OUTSIDE the declared fetch triples (ALU/CF words) a loaded vertex shader may differ from the library
// candidate of the same size and still be accepted. Without it such a draw is dropped (no VS identified).
inline std::atomic<uint32_t> g_vs_identity_alu_tolerance{0};

// selected.words must be the library's normalized VS words. Only the declared
// fetch triples may differ in the SAME loader-masked fields or in a swizzle
// proven representable by the existing typed input ABI above.
// No fetch permutation or ALU/CF exception is allowed. Projection returns a
// three-word instruction index (not a word offset). Avoids allocation per draw.
template <class FetchRange, class Projection>
inline bool VertexShaderIdentityMatches(ShaderIdentityView loaded, ShaderIdentityView selected,
                                        const FetchRange& fetches, Projection instruction_index) {
  if (loaded.stage != ShaderIdentityStage::Vertex || selected.stage != ShaderIdentityStage::Vertex ||
      loaded.words.empty() || loaded.words.size() != selected.words.size()) return false;
  constexpr uint32_t masks[3]{0x0007FFFFu, 0x80000FFFu, 0x80000000u};
  for (const auto& fetch : fetches) {
    const uint64_t instruction = uint64_t(instruction_index(fetch));
    if (instruction > UINT32_MAX || instruction * 3 + 3 > selected.words.size()) return false;
    const size_t start = size_t(instruction) * 3;
    if ((selected.words[start] & 0x1Fu) != 0u) return false;  // Not a vertex fetch.
    for (size_t lane = 0; lane < 3; ++lane)
      if ((selected.words[start + lane] & masks[lane]) != selected.words[start + lane]) return false;
    if (!VertexFetchSwizzleRepresentable(selected.words[start + 1] & 0xFFFu,
                                         loaded.words[start + 1] & 0xFFFu)) return false;
  }
  uint32_t tolerance_left = g_vs_identity_alu_tolerance.load(std::memory_order_relaxed);
  for (size_t word = 0; word < loaded.words.size(); ++word) {
    if (loaded.words[word] == selected.words[word]) continue;
    bool declared_fetch = false;
    for (const auto& fetch : fetches) {
      const size_t start = size_t(instruction_index(fetch)) * 3;
      if (word >= start && word - start < 3) {
        declared_fetch = true;
        const uint32_t mask = word - start == 1 ? 0x80000000u : masks[word - start];
        if ((loaded.words[word] & mask) != (selected.words[word] & mask)) return false;
        break;
      }
    }
    if (!declared_fetch) {
      if (tolerance_left == 0) return false;
      --tolerance_left;
    }
  }
  return true;
}
}  // namespace me::native
