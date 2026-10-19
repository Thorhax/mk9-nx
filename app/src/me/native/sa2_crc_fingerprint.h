// SA2 - native renderer: 64-bit content fingerprint from the ARMv8 CRC32 instructions.
//
// Why. The ring thread hashes guest vertex/index/texture samples with XXH3 (~7 % of the Switch ring thread: the Cortex-A57
// runs XXH3's NEON loop at about 3 bytes/cycle and every seeded call over 240 bytes first regenerates a 192-byte secret).
// The fingerprints only decide whether bytes seen again are the bytes seen before (same-frame vertex dedupe, same-frame
// index cache, sampled texture recheck); they are never stored on disk and need no avalanche for hash tables.
//
// What. Two interleaved lanes over the 8-byte words (even words -> lane 0, odd words -> lane 1). Each lane runs BOTH
// hardware polynomials over its words: CRC32C (Castagnoli, crc32cx) and CRC32 (IEEE, crc32x), i.e. 4 independent
// dependency chains, enough to keep the 1-per-cycle, 3-cycle-latency CRC pipe of the A57 busy (~4 bytes/cycle, no setup
// cost). The 4 CRCs are folded into 64 bits with an invertible mix per lane.
//
// Why the collision odds are 2^-64 (and not the 2^-32 of one CRC):
//  * The two polynomials are different degree-32 generators, so a lane's pair (CRC32C, CRC32) is the remainder modulo
//    their product, a degree-64 polynomial. An error pattern confined to one lane that is shorter than 64 bits (any single
//    changed word, any burst) is ALWAYS detected (a nonzero polynomial of degree < 64 is not a multiple of the product);
//    longer or scattered patterns escape with probability 2^-64 for unstructured changes.
//  * The lane pair is mapped to the lane's 64-bit value injectively (concatenation) and the two lane values are folded by
//    a bijection in each argument (multiply by an odd constant, xorshift), so a change confined to one lane changes the
//    result with certainty whenever the lane pair changes. Changes in both lanes collide with probability ~2^-64.
//  * The seed (chained from the previous block's fingerprint) is the initial CRC state, and the length enters the fold,
//    so equal prefixes of different length differ.
// The use is not adversarial: the bytes are a game's own vertex/index/texture data, compared within a frame.
//
// Falls back to XXH3 where the CRC instructions are not compiled in (the callers keep the XXH3 path behind a switch).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(__ARM_FEATURE_CRC32) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_acle.h>
#define SA2_CRC_FINGERPRINT 1
#else
#define SA2_CRC_FINGERPRINT 0
#endif

namespace sa2::native::crc_fingerprint {

constexpr bool kAvailable = SA2_CRC_FINGERPRINT != 0;

#if SA2_CRC_FINGERPRINT

inline uint64_t Blend(uint64_t v0, uint64_t v1, uint64_t bytes) {
  // Bijective in v0 for a fixed v1 and in v1 for a fixed v0.
  uint64_t h = v0 * 0x9E3779B97F4A7C15ull;
  h ^= h >> 32;
  h += v1 * 0xC2B2AE3D27D4EB4Full;
  h ^= h >> 29;
  h *= 0x165667B19E3779F9ull;
  h ^= h >> 32;
  h ^= bytes * 0xFF51AFD7ED558CCDull;
  h ^= h >> 31;
  return h;
}

// Fingerprint of [data, data + bytes) chained from `seed` (0 starts a new one).
inline uint64_t Fingerprint64(const uint8_t* data, size_t bytes, uint64_t seed) {
  uint32_t c0 = uint32_t(seed), i0 = uint32_t(seed >> 32);        // lane 0: CRC32C, CRC32
  uint32_t c1 = ~uint32_t(seed), i1 = ~uint32_t(seed >> 32);      // lane 1
  const uint8_t* p = data;
  size_t n = bytes;
  while (n >= 16) {
    uint64_t a, b;
    std::memcpy(&a, p, 8);
    std::memcpy(&b, p + 8, 8);
    c0 = __crc32cd(c0, a);
    i0 = __crc32d(i0, a);
    c1 = __crc32cd(c1, b);
    i1 = __crc32d(i1, b);
    p += 16;
    n -= 16;
  }
  if (n >= 8) {
    uint64_t a;
    std::memcpy(&a, p, 8);
    c0 = __crc32cd(c0, a);
    i0 = __crc32d(i0, a);
    p += 8;
    n -= 8;
  }
  if (n >= 4) {
    uint32_t a;
    std::memcpy(&a, p, 4);
    c1 = __crc32cw(c1, a);
    i1 = __crc32w(i1, a);
    p += 4;
    n -= 4;
  }
  while (n) {
    c1 = __crc32cb(c1, *p);
    i1 = __crc32b(i1, *p);
    ++p;
    --n;
  }
  return Blend(uint64_t(c0) | (uint64_t(i0) << 32), uint64_t(c1) | (uint64_t(i1) << 32), uint64_t(bytes));
}

#endif  // SA2_CRC_FINGERPRINT

}  // namespace sa2::native::crc_fingerprint
