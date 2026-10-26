// mk9-nx (2026-10-27): ue3_shader_locate, ue3_shader_scan's walk for the installer's recipe (tools/installer_recipe_mk9.py):
// where each shader container of MK9's packages is, as the page will read it. For every distinct container (named as
// Renut's translator names it: <vs|ps>_<XXH3 64 of the container>) one line:
//   name<TAB>package (relative to the root)<TAB>size<TAB>chunks<TAB>at
// chunks: the compressed chunks it lies in, "<file offset of the chunk header>:<its uncompressed size>" joined by ',';
// at: its offset in the first one's uncompressed data. For a package without compressed chunks: chunks is "-" and at
// the file offset. The first package (in the order given) that holds a container is the one listed.
//   ue3_shader_locate <root> <packages...>   > locations.tsv
//
// ue3_shader_scan's header follows.
// Offline shader collection from Unreal Engine 3 packages: decompress every compressed chunk
// of each package (tag 0x9E2A83C1; 128 KB blocks compressed with LZO1X, CompressionFlags = 2, which is
// what Mass Effect's packages use), scan the result for 2008 shader
// containers (0x102A11xx with coherent virtual/physical sizes) and write each distinct one
// as {vs,ps}_<fnv64>.bin, the same naming as app/src/me_shader_dump.cpp, so both sources merge.
//   usage: ue3_shader_scan <output folder> <package files...>
// No dependencies: the LZO1X decoder is below.
#define XXH_INLINE_ALL
#include "xxhash.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <vector>

namespace {

// LZO1X decompression (the lzo1x_decompress_safe algorithm), bounds-checked.
// Mass Effect's packages use COMPRESS_LZO (CompressionFlags = 2).
bool Lzo(const uint8_t* in, size_t in_len, uint8_t* out, size_t out_len) {
  const uint8_t* ip = in;
  const uint8_t* const ie = in + in_len;
  uint8_t* op = out;
  uint8_t* const oe = out + out_len;
  size_t t = 0;
  const uint8_t* m;
  auto need_in = [&](size_t n) { return size_t(ie - ip) >= n; };
  auto need_out = [&](size_t n) { return size_t(oe - op) >= n; };
  auto lit = [&](size_t n) {
    if (!need_in(n) || !need_out(n)) return false;
    std::memcpy(op, ip, n); op += n; ip += n; return true;
  };
  auto copy = [&](const uint8_t* from, size_t n) {
    if (from < out || !need_out(n)) return false;
    for (size_t i = 0; i < n; ++i) op[i] = from[i];
    op += n; return true;
  };
  auto run = [&](size_t base) -> bool {
    size_t v = base;
    while (need_in(1) && *ip == 0) { v += 255; ++ip; }
    if (!need_in(1)) return false;
    t = v + *ip++;
    return true;
  };
  if (!need_in(1)) return false;
  bool first = false;
  if (*ip > 17) {
    t = *ip++ - 17;
    if (t < 4) goto match_next;
    if (!lit(t)) return false;
    first = true;
  }
  for (;;) {
    if (!first) {
      if (!need_in(1)) return false;
      t = *ip++;
      if (t >= 16) goto match;
      if (t == 0 && !run(15)) return false;
      if (!lit(t + 3)) return false;
    }
    first = false;
    if (!need_in(1)) return false;
    t = *ip++;
    if (t < 16) {
      if (!need_in(1)) return false;
      m = op - (1 + 0x0800) - (t >> 2) - (size_t(*ip++) << 2);
      if (!copy(m, 3)) return false;
      goto match_done;
    }
  match:
    for (;;) {
      if (t >= 64) {
        if (!need_in(1)) return false;
        m = op - 1 - ((t >> 2) & 7) - (size_t(*ip++) << 3);
        t = (t >> 5) - 1;
        if (!copy(m, t + 2)) return false;
      } else if (t >= 32) {
        t &= 31;
        if (t == 0 && !run(31)) return false;
        if (!need_in(2)) return false;
        m = op - 1 - ((ip[0] >> 2) + (size_t(ip[1]) << 6));
        ip += 2;
        if (!copy(m, t + 2)) return false;
      } else if (t >= 16) {
        m = op - (size_t(t & 8) << 11);
        t &= 7;
        if (t == 0 && !run(7)) return false;
        if (!need_in(2)) return false;
        m -= (ip[0] >> 2) + (size_t(ip[1]) << 6);
        ip += 2;
        if (m == op) return op == oe;  // end of stream
        m -= 0x4000;
        if (!copy(m, t + 2)) return false;
      } else {
        if (!need_in(1)) return false;
        m = op - 1 - (t >> 2) - (size_t(*ip++) << 2);
        if (!copy(m, 2)) return false;
      }
    match_done:
      t = ip[-2] & 3;
      if (t == 0) break;
    match_next:
      if (!lit(t)) return false;
      if (!need_in(1)) return false;
      t = *ip++;
    }
  }
}

uint32_t BE(const uint8_t* p) { return uint32_t(p[0]) << 24 | p[1] << 16 | p[2] << 8 | p[3]; }

uint64_t Fnv(const uint8_t* p, size_t n) {
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ull;
  return h;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: ue3_shader_locate <root> <packages...>\n");
    return 1;
  }
  const std::filesystem::path root = argv[1];
  std::set<std::string> seen;
  size_t packages = 0, chunks = 0, bad_blocks = 0;
  for (int a = 2; a < argc; ++a) {
    std::ifstream f(argv[a], std::ios::binary);
    std::vector<uint8_t> d((std::istreambuf_iterator<char>(f)), {});
    ++packages;
    const std::string rel = std::filesystem::relative(argv[a], root).generic_string();
    std::vector<uint8_t> plain;
    struct Chunk { size_t header, plain_at, usize; };
    std::vector<Chunk> found_chunks;
    for (size_t off = 0; off + 16 <= d.size(); ++off) {
      if (BE(&d[off]) != 0x9E2A83C1u || (BE(&d[off + 4]) != 0x9E2A83C1u && BE(&d[off + 4]) != 0x20000u)) continue;
      const uint32_t csize = BE(&d[off + 8]), usize = BE(&d[off + 12]);
      if (usize == 0 || usize > 0x4000000 || csize == 0 || csize > d.size()) continue;
      const size_t nblocks = (usize + 0x1FFFF) / 0x20000;
      const size_t hdr = off + 16;
      size_t data = hdr + nblocks * 8;
      if (data > d.size()) continue;
      uint64_t sum_c = 0, sum_u = 0;
      for (size_t k = 0; k < nblocks; ++k) {
        sum_c += BE(&d[hdr + k * 8]);
        sum_u += BE(&d[hdr + k * 8 + 4]);
      }
      if (sum_c != csize || sum_u != usize || data + csize > d.size()) continue;
      std::vector<uint8_t> out(usize);
      size_t produced = 0;
      bool ok = true;
      for (size_t k = 0; k < nblocks && ok; ++k) {
        const uint32_t bc = BE(&d[hdr + k * 8]), bu = BE(&d[hdr + k * 8 + 4]);
        if (!Lzo(&d[data], bc, out.data() + produced, bu)) {
          ++bad_blocks;
          ok = false;
        }
        produced += bu;
        data += bc;
      }
      if (!ok) continue;
      ++chunks;
      found_chunks.push_back({off, plain.size(), usize});
      plain.insert(plain.end(), out.begin(), out.end());
      off = data - 1;
    }
    const bool raw = plain.empty();
    const std::vector<uint8_t>& b = raw ? d : plain;
    for (size_t i = 0; i + 24 <= b.size(); ++i) {
      const uint32_t sig = BE(&b[i]);
      if ((sig & 0xFFFFFF00u) != 0x102A1100u) continue;
      const uint32_t vs = BE(&b[i + 4]), ps = BE(&b[i + 8]);
      if (vs < 24 || vs > 0x40000 || ps == 0 || ps > 0x40000 || i + vs + ps > b.size()) continue;
      char name[64];
      std::snprintf(name, sizeof(name), "%s_%016llx", (sig & 1) ? "vs" : "ps",
                    static_cast<unsigned long long>(XXH3_64bits(&b[i], vs + ps)));
      if (!seen.insert(name).second) continue;
      std::string where;
      size_t at = i;
      if (raw) {
        where = "-";
      } else {
        for (const Chunk& c : found_chunks) {
          if (c.plain_at + c.usize <= i || c.plain_at >= i + vs + ps) continue;
          if (where.empty()) at = i - c.plain_at;
          if (!where.empty()) where += ',';
          where += std::to_string(c.header) + ":" + std::to_string(c.usize);
        }
      }
      std::printf("%s\t%s\t%u\t%s\t%zu\n", name, rel.c_str(), vs + ps, where.c_str(), at);
    }
  }
  std::fprintf(stderr, "%zu packages, %zu chunks, %zu bad blocks, %zu distinct containers\n", packages, chunks,
               bad_blocks, seen.size());
  return 0;
}
