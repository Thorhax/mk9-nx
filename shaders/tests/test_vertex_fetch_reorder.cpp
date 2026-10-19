// Run-time proof of reordered vertex fetches (app/src/me/me_vertex_fetch_reorder.h) against the offline repair
// (shaders/tools/repair_vertex_variant_declarations.cpp): for each pair given, a library container and a container
// the tool repaired from a dump of Direct3D's load, the run-time proof on the loaded words must hold and map every
// declared fetch to the slot the tool wrote into the repaired DECL; the brute-force self-check must agree; and
// mutated loads (an ALU word, a fetch's destination, the unmoved order) must be refused.
//   clang++ -std=c++20 -O2 -static this.cpp -o test_vertex_fetch_reorder
//   test_vertex_fetch_reorder template.bin repaired.bin [template.bin repaired.bin ...]
//   test_vertex_fetch_reorder --orders library.bin [...]
// --orders: every placement of a library shader's declared fetches over its declared slots, as Direct3D would load it
// (each fetch keeps its own opcode, registers, swizzle and predicate; each slot keeps its fetch constant, full/mini
// flag, format, offset and stride), held against the run-time proof: how many orders it covers, and why it refuses
// the others.
#define main RepairVertexVariantMain
#include "../tools/repair_vertex_variant_declarations.cpp"
#undef main
#include "../../app/src/me/me_vertex_fetch_reorder.h"

namespace {
struct Element {
  uint16_t instruction;
};
std::vector<Element> Declared(const Container& c) {
  std::vector<Element> elements;
  for (uint32_t e : c.entries) elements.push_back({uint16_t(e & 4095)});
  return elements;
}
std::vector<uint32_t> Normalized(const Container& c) {
  constexpr uint32_t keeps[3]{0x0007FFFFu, 0x80000FFFu, 0x80000000u};
  std::vector<uint32_t> words = c.words;
  for (uint32_t e : c.entries)
    for (size_t lane = 0; lane < 3; ++lane) words[size_t(e & 4095) * 3 + lane] &= keeps[lane];
  return words;
}
me::native::VertexFetchReorder Prove(const std::vector<uint32_t>& loaded, const std::vector<uint32_t>& selected,
                                     const std::vector<Element>& elements) {
  return me::native::ProveVertexFetchReorder(std::span<const uint32_t>(loaded), std::span<const uint32_t>(selected),
                                             elements, [](const Element& e) { return e.instruction; });
}
}  // namespace

int Orders(int argc, char** argv) {
  int failures = 0;
  std::vector<uint32_t> scratch;
  for (int i = 2; i < argc; ++i) {
    const Container library(argv[i]);
    const auto elements = Declared(library);
    const auto selected = Normalized(library);
    std::vector<unsigned> slots;
    for (const Element& e : elements)
      if (std::find(slots.begin(), slots.end(), e.instruction) == slots.end()) slots.push_back(e.instruction);
    if (slots.size() < 2) {
      std::printf("%s: %zu declared fetch(es), nothing to reorder\n", argv[i], slots.size());
      continue;
    }
    std::vector<unsigned> placement(slots.size());
    for (size_t k = 0; k < placement.size(); ++k) placement[k] = unsigned(k);
    unsigned orders = 0, proven = 0;
    std::map<std::string, unsigned> refused;
    do {
      bool moved = false;
      for (size_t k = 0; k < slots.size(); ++k) moved |= placement[k] != k;
      if (!moved) continue;
      ++orders;
      std::vector<uint32_t> loaded = library.words;
      for (size_t k = 0; k < slots.size(); ++k) {
        const size_t source = size_t(slots[k]) * 3, target = size_t(slots[placement[k]]) * 3;
        loaded[target] = (library.words[source] & 0x0007FFFFu) | (library.words[target] & ~0x0007FFFFu);
        loaded[target + 1] = (library.words[source + 1] & 0x80000FFFu) | (library.words[target + 1] & ~0x80000FFFu);
        loaded[target + 2] = (library.words[source + 2] & 0x80000000u) | (library.words[target + 2] & ~0x80000000u);
      }
      const auto proof = Prove(loaded, selected, elements);
      if (!proof) {
        ++refused[proof.failure];
        continue;
      }
      bool same = true;
      for (uint32_t k = 0; k < proof.count; ++k) {
        const size_t index = size_t(std::find(slots.begin(), slots.end(), proof.from[k]) - slots.begin());
        same &= proof.to[k] == slots[placement[index]];
      }
      if (!same || !me::native::VertexFetchReorderAgreesWithPlacements(loaded, selected, proof, scratch)) {
        std::printf("FAIL %s: an order proven with the wrong mapping\n", argv[i]);
        ++failures;
        continue;
      }
      ++proven;
    } while (std::next_permutation(placement.begin(), placement.end()));
    std::printf("%s: %zu declared fetches, %u reordered placements, %u proven", argv[i], slots.size(), orders, proven);
    for (const auto& [why, n] : refused) std::printf("; %u refused: %s", n, why.c_str());
    std::printf("\n");
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}

int main(int argc, char** argv) try {
  if (argc >= 3 && std::string(argv[1]) == "--orders") return Orders(argc, argv);
  if (argc < 3 || (argc - 1) % 2) {
    std::fprintf(stderr, "usage: %s template.bin repaired.bin [...]\n", argv[0]);
    return 2;
  }
  int failures = 0;
  std::vector<uint32_t> scratch;
  for (int i = 1; i + 1 < argc; i += 2) {
    const Container library(argv[i]), repaired(argv[i + 1]);
    const auto elements = Declared(library);
    const auto selected = Normalized(library);
    const std::vector<uint32_t>& loaded = repaired.words;  // the tool keeps the dumped code as it was loaded
    auto fail = [&](const std::string& why) {
      std::printf("FAIL %s / %s: %s\n", argv[i], argv[i + 1], why.c_str());
      ++failures;
    };
    if (library.entries.size() != repaired.entries.size()) {
      fail("DECL counts differ");
      continue;
    }
    const auto proof = Prove(loaded, selected, elements);
    if (!proof) {
      fail(std::string("proof refused: ") + proof.failure);
      continue;
    }
    std::string order;
    bool same = true;
    for (size_t e = 0; e < library.entries.size(); ++e) {
      const unsigned from = library.entries[e] & 4095, expected = repaired.entries[e] & 4095;
      unsigned to = ~0u;
      for (uint32_t k = 0; k < proof.count; ++k)
        if (proof.from[k] == from) to = proof.to[k];
      same &= to == expected;
      order += " " + std::to_string(from) + "->" + std::to_string(to);
    }
    if (!same) fail("mapping differs from the repaired DECL:" + order);
    if (!me::native::VertexFetchReorderAgreesWithPlacements(loaded, selected, proof, scratch))
      fail("brute-force placements disagree");
    // Mutations that must be refused.
    {
      auto alu = loaded;
      size_t word = 0;
      while (word < alu.size()) {
        bool fetch = false;
        for (const Element& e : elements) fetch |= word / 3 == e.instruction;
        if (!fetch && word >= 3) break;
        ++word;
      }
      alu[word] ^= 0x00010000u;
      if (Prove(alu, selected, elements)) fail("an ALU/CF word change was accepted");
    }
    {
      auto destination = loaded;
      destination[size_t(proof.to[0]) * 3] ^= 1u << 12;  // another temp register
      if (Prove(destination, selected, elements)) fail("a fetch destination change was accepted");
    }
    if (Prove(Normalized(library), selected, elements)) fail("the library's own order was accepted as reordered");
    if (!failures)
      std::printf("ok %s <- %s: %u fetches, %u moved, %u swizzle components remapped;%s\n", argv[i], argv[i + 1],
                  proof.count, proof.moved, proof.remaps, order.c_str());
  }
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "error: %s\n", e.what());
  return 1;
}
