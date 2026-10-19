#pragma once

// sa2-nx (2026-10-07): vertex shaders whose declared fetches Direct3D moved to each other's slots.
//
// Direct3D's vertex fetch patcher (sub_82555E18) rewrites a vertex shader's fetches for the vertex declaration it is
// drawn with, and for some declarations it also moves them: the fetch that writes r10 sits in the slot the library
// container has r1's fetch in, and so on, inside one run of fetch-only slots; every ALU and control-flow word stays the
// same. VertexShaderIdentityMatches holds every fetch at its own slot, so such a load was not identified and its draws
// were dropped: the Chao garden's stats window (n0, loaded with IM_LOAD_IMMEDIATE) and garden objects (n21 at
// 1F349040). Until now each order was fixed on its own by a container repaired offline from a dump of the load
// (shaders/tools/repair_vertex_variant_declarations.cpp) and added to the package.
//
// ProveVertexFetchReorder is that tool's proof at run time, for any order. The loaded program is the library shader
// with its declared fetches permuted when
//   - every word outside the declared fetch slots is identical (ALU, control flow);
//   - the loaded program executes exactly the declared slots as vertex fetches, each once, in unconditional EXEC
//     clauses, every mini fetch after a full fetch;
//   - each declared fetch has exactly one slot in the loaded program with its destination register and write mask,
//     its opcode and source fields, its predicate bits, and a destination swizzle the typed input ABI can express (a
//     bijection), and the draw path's own selection (SelectVertexFetch without the ordered proof) picks that slot;
//   - each fetch moved only within an unconditional, unpredicated, unserialized fetch-only interval (one EXEC clause,
//     or two adjacent ones), between full-fetch bases that are equivalent (same constant selector, index source,
//     stride and rounding, no index register written in between).
// The library shader's SPIR-V then computes what the loaded program computes: each fetch writes its own register
// before any ALU instruction reads it, in either order, and the draw path reads each element's buffer, offset,
// format and swizzle from the slot the loaded program has it in (sa2_native_draws.cpp, ComputeEntry).
//
// The proof steps below are the offline tool's, one for one, without exceptions. shaders/tests/
// test_vertex_fetch_reorder.cpp holds both against the package's repaired variants.

#include "me_shader_identity.h"
#include "me_vertex_fetch_selection.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace me::native {

struct VertexFetchReorder {
  static constexpr uint32_t kMax = 16;
  // The declared slots of the library shader (each once, in declaration order) and, for each, the slot of the loaded
  // program that holds that fetch.
  std::array<uint16_t, kMax> from{}, to{};
  uint32_t count = 0;
  uint32_t moved = 0;    // slots whose fetch moved
  uint32_t remaps = 0;   // destination swizzle components that differ (the draw path remaps them)
  const char* failure = "not checked";  // nullptr = proven
  explicit operator bool() const { return failure == nullptr; }
};

namespace reorder_detail {

// One executed vertex fetch of the loaded program (FetchLocations of the offline tool).
struct Location {
  unsigned instruction, clause, cluster, clause_first, clause_count, opcode;
  bool serialized;
};
struct Locations {
  std::array<Location, 32> items{};
  size_t count = 0;
  const Location* Find(unsigned instruction) const {
    for (size_t i = 0; i < count; ++i)
      if (items[i].instruction == instruction) return &items[i];
    return nullptr;
  }
};

inline const char* FetchLocations(std::span<const uint32_t> w, Locations& result) {
  result.count = 0;
  size_t end = w.size() / 3;
  unsigned cluster = ~0u;
  for (size_t pair = 0; pair < end; ++pair) {
    const uint64_t cf[2]{uint64_t(w[pair * 3]) | uint64_t(w[pair * 3 + 1] & 65535) << 32,
                         uint64_t(w[pair * 3 + 1] >> 16) | uint64_t(w[pair * 3 + 2]) << 16};
    for (unsigned half = 0; half < 2; ++half) {
      const uint64_t c = cf[half];
      const unsigned op = (c >> 44) & 15;
      if (!((op >= 1 && op <= 6) || op == 13 || op == 14)) continue;
      const unsigned first = c & 4095, count = (c >> 12) & 7, seq = (c >> 16) & 4095;
      if (first) end = std::min(end, size_t(first));
      if (!count || uint64_t(first) + count > w.size() / 3) return "invalid EXEC extent";
      // A conditional EXEC can enter without its preceding full-fetch base.
      for (unsigned i = 0; i < count; ++i) {
        const unsigned ins = first + i;
        if (!(seq >> (i * 2) & 1)) continue;
        if (op != 1 && op != 2) return "conditional fetch EXEC unsupported for base continuity proof";
        const unsigned p = ins * 3;
        if ((w[p] & 31) != 0) return "non-vertex fetch in VS fetch-only proof";
        if (!(w[p + 1] & 0x40000000u)) cluster = ins;
        if (cluster == ~0u) return "mini fetch without full base";
        if (result.Find(ins)) return "multiply executed fetch unsupported";
        if (result.count == result.items.size()) return "too many executed fetches";
        result.items[result.count++] = {ins, unsigned(pair * 2 + half), cluster, first, count, op,
                                        bool(seq >> (i * 2 + 1) & 1)};
      }
    }
  }
  return nullptr;
}

inline bool UnconditionalFetchOnlyInterval(unsigned low, unsigned high, const Locations& locations,
                                           std::span<const uint32_t> words) {
  if (high < low || high - low >= 16) return false;
  for (unsigned i = low; i <= high; ++i) {
    const Location* it = locations.Find(i);
    if (!it || it->opcode != 1 || it->serialized || (words[i * 3 + 1] & 0x80000000u)) return false;
  }
  return true;
}

// SDK ParseVertexFetchInstruction inherits only index source, fetch constant, stride and index rounding. Format and
// offset belong to EACH operation. No ALU, serialization or source-index writes may occur between the two actual
// address computations.
inline bool EquivalentFullBases(unsigned a, unsigned b, const Locations& locations, std::span<const uint32_t> words) {
  if (a == b) return true;
  const unsigned low = std::min(a, b), high = std::max(a, b);
  if (!UnconditionalFetchOnlyInterval(low, high, locations, words)) return false;
  const unsigned p = a * 3, q = b * 3;
  constexpr uint32_t inherited0 = 0xC7F00FE0u;  // src/relative, constant selector, src swizzle
  if ((words[p] & inherited0) != (words[q] & inherited0) ||
      (words[p + 1] & 0x80008000u) != (words[q + 1] & 0x80008000u) ||
      (words[p + 2] & 0x800000FFu) != (words[q + 2] & 0x800000FFu)) return false;
  // Relative TEMP addressing needs a broader loop/aL proof; likewise a relative fetch destination could overwrite
  // the source.
  if (words[p] & 0x800u) return false;
  const unsigned source = (words[p] >> 5) & 63, component = (words[p] >> 30) & 3;
  for (unsigned i = low; i < high; ++i) {
    const unsigned r = i * 3;
    if (words[r] & 0x40000u) return false;
    if (((words[r] >> 12) & 63) == source && (VertexFetchWrittenMask(words[r + 1] & 4095) & (1u << component)))
      return false;
  }
  return true;
}

// Two immediately adjacent unconditional EXECs with consecutive instruction ranges: the whole moved interval is an
// unpredicated, unserialized fetch-only region inheriting proven equivalent bases.
inline bool AdjacentFetchOnlyRegion(unsigned old, unsigned next, const Locations& locations,
                                    std::span<const uint32_t> words) {
  const Location* a = locations.Find(old);
  const Location* b = locations.Find(next);
  if (!a || !b) return false;
  const Location& first = a->clause < b->clause ? *a : *b;
  const Location& second = a->clause < b->clause ? *b : *a;
  if (second.clause != first.clause + 1 || first.opcode != 1 || second.opcode != 1 ||
      first.clause_first + first.clause_count != second.clause_first) return false;
  const unsigned low = std::min(old, next), high = std::max(old, next);
  if (!UnconditionalFetchOnlyInterval(low, high, locations, words)) return false;
  for (unsigned i = low; i <= high; ++i) {
    const Location* it = locations.Find(i);
    if (!it || !EquivalentFullBases(it->cluster, a->cluster, locations, words) ||
        (it->clause != a->clause && it->clause != b->clause) || it->serialized ||
        (words[i * 3 + 1] & 0x80000000u)) return false;
  }
  return true;
}

}  // namespace reorder_detail

// `selected` = the library entry's normalized words (fetch triples masked as the library keeps them), `fetches` its
// declared fetches. Allocates nothing.
template <class FetchRange, class Projection>
inline VertexFetchReorder ProveVertexFetchReorder(std::span<const uint32_t> loaded, std::span<const uint32_t> selected,
                                                  const FetchRange& fetches, Projection instruction_index) {
  using namespace reorder_detail;
  VertexFetchReorder r;
  const auto fail = [&r](const char* why) {
    r.failure = why;
    return r;
  };
  if (loaded.empty() || loaded.size() != selected.size()) return fail("word count");
  constexpr uint32_t masks[3]{0x0007FFFFu, 0x80000FFFu, 0x80000000u};
  for (const auto& fetch : fetches) {
    const uint64_t instruction = uint64_t(instruction_index(fetch));
    if (instruction * 3 + 3 > selected.size()) return fail("DECL is not an in-bounds vertex fetch");
    const size_t p = size_t(instruction) * 3;
    if ((selected[p] & 31) != 0) return fail("DECL is not an in-bounds vertex fetch");
    for (size_t lane = 0; lane < 3; ++lane)
      if ((selected[p + lane] & masks[lane]) != selected[p + lane]) return fail("library entry not normalized");
    bool seen = false;
    for (uint32_t k = 0; k < r.count; ++k) seen |= r.from[k] == instruction;
    if (seen) continue;
    if (r.count == VertexFetchReorder::kMax) return fail("too many declared fetches");
    r.from[r.count++] = uint16_t(instruction);
  }
  if (r.count < 2) return fail("fewer than two declared fetches");
  const auto declared = [&r](size_t instruction) {
    for (uint32_t k = 0; k < r.count; ++k)
      if (r.from[k] == instruction) return true;
    return false;
  };
  for (size_t i = 0; i < loaded.size(); ++i)
    if (loaded[i] != selected[i] && !declared(i / 3)) return fail("NON-fetch ALU/CF differs");
  Locations locations;
  if (const char* why = FetchLocations(loaded, locations)) return fail(why);
  if (locations.count != r.count) return fail("undeclared or missing executed vertex fetch");
  for (size_t i = 0; i < locations.count; ++i)
    if (!declared(locations.items[i].instruction)) return fail("fetch address set changed");
  uint32_t used = 0;
  for (uint32_t k = 0; k < r.count; ++k) {
    const size_t p = size_t(r.from[k]) * 3;
    const uint32_t temp = (selected[p] >> 12) & 63, swizzle = selected[p + 1] & 4095;
    uint32_t matches = 0, match = 0;
    for (uint32_t m = 0; m < r.count; ++m) {
      const size_t q = size_t(r.from[m]) * 3;
      if (((loaded[q] >> 12) & 63) != temp ||
          VertexFetchWrittenMask(loaded[q + 1] & 4095) != VertexFetchWrittenMask(swizzle) ||
          !VertexFetchSwizzleRepresentable(swizzle, loaded[q + 1] & 4095)) continue;
      if ((loaded[q] & masks[0]) != selected[p] ||
          (loaded[q + 1] & 0x80000000u) != (selected[p + 1] & 0x80000000u) ||
          (loaded[q + 2] & masks[2]) != selected[p + 2]) continue;
      ++matches;
      match = m;
    }
    if (matches != 1 || ((used >> match) & 1)) return fail("fetch mapping is not unique bijection");
    used |= 1u << match;
    r.to[k] = r.from[match];
    r.moved += r.to[k] != r.from[k];
    const uint32_t patched = loaded[size_t(r.to[k]) * 3 + 1] & 4095;
    for (uint32_t c = 0; c < 4; ++c) r.remaps += ((swizzle >> (c * 3)) & 7) != ((patched >> (c * 3)) & 7);
    // The draw path finds each element's fetch with SelectVertexFetch, without the ordered proof: the same slot.
    const auto selection = SelectVertexFetch(selected, loaded, r.from[k], false, fetches, instruction_index);
    if (!selection || selection.word_offset != size_t(r.to[k]) * 3)
      return fail("the draw path would select another fetch");
  }
  if (!r.moved) return fail("no fetch moved");
  for (uint32_t k = 0; k < r.count; ++k) {
    const unsigned old = r.from[k], next = r.to[k];
    if (old != next &&
        !UnconditionalFetchOnlyInterval(std::min(old, next), std::max(old, next), locations, loaded))
      return fail("DECL permutation crosses non-fetch/predicate/serialized interval");
    const Location* a = locations.Find(old);
    const Location* b = locations.Find(next);
    if (!EquivalentFullBases(a->cluster, b->cluster, locations, loaded))
      return fail("DECL permutation crosses inequivalent actual full/mini bases");
    if (a->clause != b->clause && !AdjacentFetchOnlyRegion(old, next, locations, loaded))
      return fail("DECL permutation crosses unproven EXEC boundary");
  }
  r.failure = nullptr;
  return r;
}

// Self-check of a proof by brute force: the library words with their declared triples placed every possible way,
// each held against the loaded program by the ordered predicate (VertexShaderIdentityMatches). Exactly one placement
// must pass, and it must be the proven one. Up to 7 declared slots (5,040 placements); returns true beyond.
inline bool VertexFetchReorderAgreesWithPlacements(std::span<const uint32_t> loaded, std::span<const uint32_t> selected,
                                                   const VertexFetchReorder& proven, std::vector<uint32_t>& scratch) {
  if (!proven || proven.count > 7) return true;
  struct Slot {
    uint16_t instruction;
  };
  std::array<Slot, VertexFetchReorder::kMax> slots{};
  std::array<uint8_t, VertexFetchReorder::kMax> placement{};
  for (uint32_t k = 0; k < proven.count; ++k) placement[k] = uint8_t(k);
  uint32_t passes = 0;
  bool proven_passes = false;
  do {
    scratch.assign(selected.begin(), selected.end());
    for (uint32_t k = 0; k < proven.count; ++k) {
      const size_t source = size_t(proven.from[k]) * 3, target = size_t(proven.from[placement[k]]) * 3;
      for (size_t lane = 0; lane < 3; ++lane) scratch[target + lane] = selected[source + lane];
      slots[k].instruction = proven.from[placement[k]];
    }
    if (VertexShaderIdentityMatches({ShaderIdentityStage::Vertex, loaded},
                                    {ShaderIdentityStage::Vertex, std::span<const uint32_t>(scratch)},
                                    std::span<const Slot>(slots.data(), proven.count),
                                    [](const Slot& s) { return s.instruction; })) {
      ++passes;
      bool same = true;
      for (uint32_t k = 0; k < proven.count; ++k) same &= proven.from[placement[k]] == proven.to[k];
      proven_passes |= same;
    }
  } while (std::next_permutation(placement.begin(), placement.begin() + proven.count));
  return passes == 1 && proven_passes;
}

}  // namespace me::native
