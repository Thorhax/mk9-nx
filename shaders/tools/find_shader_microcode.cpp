#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <vector>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace fs = std::filesystem;

static uint32_t Be32(const std::vector<uint8_t>& b, size_t p) {
  return uint32_t(b[p]) << 24 | uint32_t(b[p + 1]) << 16 | uint32_t(b[p + 2]) << 8 | b[p + 3];
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  const uint64_t wanted = std::stoull(argv[2], nullptr, 16);
  for (const auto& e : fs::directory_iterator(argv[1])) {
    if (!e.is_regular_file() || e.path().extension() != ".bin") continue;
    std::ifstream in(e.path(), std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(in)), {});
    if (b.size() < 32) continue;
    const uint32_t virt = Be32(b, 4);
    size_t start = virt;
    uint32_t size = Be32(b, 8);
    if ((Be32(b, 0) & 0xFFFFFF00u) == 0x102A1100u) {
      const uint32_t header = Be32(b, 24);
      if (header + 8 > b.size()) continue;
      start += Be32(b, header);
      size = Be32(b, header + 4);
    }
    if (start + size > b.size()) continue;
    std::vector<uint32_t> words(size / 4);
    for (size_t i = 0; i < words.size(); ++i) words[i] = Be32(b, start + i * 4);
    if (e.path().filename().string().starts_with("vs_") &&
        (Be32(b, 0) & 0xFFFFFF00u) == 0x102A1100u) {
      const uint32_t header = Be32(b, 24);
      const uint32_t table = Be32(b, 16);
      const bool empty_table = !table || (size_t(table) + 20 <= b.size() && Be32(b, table + 16) == 0);
      static constexpr uint32_t keep[3] = {0x0007FFFF, 0x80000FFF, 0x80000000};
      if (empty_table) {
        uint32_t end = uint32_t(words.size() / 3);
        for (uint32_t pair = 0; pair < end && pair * 3 + 2 < words.size(); ++pair) {
          const uint64_t cf[2] = {
              uint64_t(words[pair * 3]) | (uint64_t(words[pair * 3 + 1] & 0xFFFF) << 32),
              uint64_t(words[pair * 3 + 1] >> 16) | (uint64_t(words[pair * 3 + 2]) << 16)};
          for (uint64_t c : cf) {
            const uint32_t opcode = uint32_t(c >> 44) & 0xF;
            if (!((opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14)) continue;
            const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 7;
            const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
            if (address) end = std::min(end, address);
            for (uint32_t i = 0; i < count; ++i) {
              const uint32_t instruction = address + i;
              if (!((sequence >> (i * 2)) & 1) || size_t(instruction) * 3 + 2 >= words.size()) continue;
              if ((words[size_t(instruction) * 3] & 0x1F) != 0) continue;
              for (size_t j = 0; j < 3; ++j) words[size_t(instruction) * 3 + j] &= keep[j];
            }
          }
        }
      } else {
      const uint32_t before = Be32(b, header + 24);
      const uint32_t count = Be32(b, header + 28);
      const size_t elements = size_t(header) + 36 + size_t(before) * 4;
      if (count <= 64 && elements + size_t(count) * 4 <= virt) {
        for (uint32_t i = 0; i < count; ++i) {
          const uint32_t instruction = Be32(b, elements + size_t(i) * 4) & 0xFFF;
          if (size_t(instruction) * 3 + 2 < words.size())
            for (size_t j = 0; j < 3; ++j) words[size_t(instruction) * 3 + j] &= keep[j];
        }
      }
      }
    }
    if (XXH3_64bits(words.data(), words.size() * 4) == wanted) std::puts(e.path().c_str());
  }
}
