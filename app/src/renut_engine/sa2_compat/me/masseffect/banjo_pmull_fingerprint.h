// banjo-nx (2026-10-15): a 64-bit CRC by carry-less multiplication (ARMv8 PMULL) for the whole-range fingerprints of
// the ring thread (banjo_ring_pmull_fingerprint).
//
// Why. The console's ring thread spent 8.8 % of its samples in RangeFingerprint (sixteenth package, intro cutscene,
// stock clocks): the persistent vertex arena fingerprints every window it cannot prove by interval or submission,
// about 1.35 GB per 10 s. masseffect_crc_fingerprint.h runs the two CRC32 polynomials over every 8-byte word: four CRC
// instructions per 16 bytes, all in the A57's single M pipeline (1 a cycle), so 4 bytes a cycle at best (the console
// measured about 1.5). Folding a CRC with PMULL needs two 64x64 carry-less multiplies per 16 bytes (the F0 pipeline, 1 a
// cycle, latency 3, hidden by four independent chains): 8 bytes a cycle at best, half the instructions per byte.
//
// What. The CRC of polynomial P(x) = x^64 + 0x42F0E1EBA9EA3693 (ECMA-182; the coefficient of x^i is bit i) of the
// message polynomial M = sum of its 16-byte blocks V_b * x^(128 * (n - 1 - b)) (a block's bit i, little-endian, is the
// coefficient of x^i; the last block is zero-padded): 4 chains fold every 4th block by x^512, are combined, the rest is
// folded block by block, and Barrett reduction gives the exact 64-bit residue M mod P. The seed is added to the first
// block (chaining); the length enters a bijective final mix, so equal residues of different lengths still differ.
//
// Collision odds, as for masseffect_crc_fingerprint.h: two different messages of the same length collide only if their
// difference is a multiple of P, a degree-64 polynomial with a nonzero constant term: never for a change spanning fewer
// than 64 bits of one block's bit order, otherwise 2^-64 for unstructured changes. The use is not adversarial.
// Checked against a bit-by-bit reference on the PC (sdk/tests/pmull_fingerprint_test.cpp, through simde's model of the
// same instructions) and on the console against the CRC32 pair (banjo_cpu_verify: both are computed and their verdicts
// on the same windows compared, see the caller).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#if defined(BANJO_PMULL_TEST)
#define BANJO_PMULL_FINGERPRINT 1  // the PC test includes simde/arm/neon.h with native aliases first
#elif defined(__aarch64__) && (defined(__ARM_FEATURE_CRYPTO) || defined(__ARM_FEATURE_AES))
#include <arm_neon.h>
#define BANJO_PMULL_FINGERPRINT 1
#else
#define BANJO_PMULL_FINGERPRINT 0
#endif

namespace banjo::pmull_fingerprint {

constexpr bool kAvailable = BANJO_PMULL_FINGERPRINT != 0;
constexpr uint64_t kPoly = 0x42F0E1EBA9EA3693ull;  // P(x) - x^64

// r * x^n mod P, bit by bit (compile time and the reference).
constexpr uint64_t MulXPow(uint64_t r, unsigned n) {
  for (unsigned i = 0; i < n; ++i) {
    const bool carry = (r >> 63) != 0;
    r <<= 1;
    if (carry) r ^= kPoly;
  }
  return r;
}
constexpr uint64_t XPow(unsigned n) { return MulXPow(1, n); }

// The low 64 coefficients of floor(x^128 / P) (its x^64 coefficient is 1): Barrett's constant.
constexpr uint64_t BarrettMu() {
  uint64_t q = 0, r = 0;
  for (int i = 128; i >= 0; --i) {
    const bool top = (r >> 63) != 0;
    r = (r << 1) | (i == 128 ? 1u : 0u);
    if (top) {
      r ^= kPoly;
      if (i < 64) q |= uint64_t(1) << i;
    }
  }
  return q;
}

constexpr uint64_t kMu = BarrettMu();
// Folding by k blocks: x^(128k) mod P for the low half, x^(128k + 64) mod P for the high half.
constexpr uint64_t kFold1Lo = XPow(128), kFold1Hi = XPow(192);
constexpr uint64_t kFold2Lo = XPow(256), kFold2Hi = XPow(320);
constexpr uint64_t kFold3Lo = XPow(384), kFold3Hi = XPow(448);
constexpr uint64_t kFold4Lo = XPow(512), kFold4Hi = XPow(576);

// The bijective final mix (fixed length): distinct residues give distinct fingerprints.
inline uint64_t Mix(uint64_t residue, uint64_t bytes) {
  uint64_t h = residue ^ (bytes * 0x9E3779B97F4A7C15ull);
  h *= 0xFF51AFD7ED558CCDull;
  h ^= h >> 33;
  h *= 0xC4CEB9FE1A85EC53ull;
  h ^= h >> 33;
  return h;
}

// The bit-by-bit reference (tests): the same residue, block by block.
inline uint64_t Reference(const uint8_t* data, size_t bytes, uint64_t seed) {
  uint64_t s = 0;
  for (size_t at = 0; at < bytes; at += 16) {
    uint8_t block[16] = {};
    std::memcpy(block, data + at, bytes - at < 16 ? bytes - at : 16);
    uint64_t lo, hi;
    std::memcpy(&lo, block, 8);
    std::memcpy(&hi, block + 8, 8);
    if (at == 0) lo ^= seed;
    s = MulXPow(s, 128) ^ lo ^ MulXPow(hi, 64);
  }
  if (bytes == 0) s = seed;
  return Mix(s, bytes);
}

#if BANJO_PMULL_FINGERPRINT

inline uint64x2_t Fold(uint64x2_t a, poly64x2_t k) {
  const poly64x2_t p = vreinterpretq_p64_u64(a);
  const uint64x2_t lo = vreinterpretq_u64_p128(vmull_p64(vgetq_lane_p64(p, 0), vgetq_lane_p64(k, 0)));
  const uint64x2_t hi = vreinterpretq_u64_p128(vmull_high_p64(p, k));
  return veorq_u64(lo, hi);
}

inline poly64x2_t Constants(uint64_t lo, uint64_t hi) {
  return vreinterpretq_p64_u64(vcombine_u64(vcreate_u64(lo), vcreate_u64(hi)));
}

inline uint64x2_t Load(const uint8_t* p) { return vreinterpretq_u64_u8(vld1q_u8(p)); }

// A 128-bit polynomial reduced mod P (Barrett, exact for polynomials over GF(2)).
inline uint64_t Reduce(uint64x2_t a) {
  const uint64_t a_lo = vgetq_lane_u64(a, 0), a_hi = vgetq_lane_u64(a, 1);
  const uint64x2_t t = vreinterpretq_u64_p128(vmull_p64(poly64_t(a_hi), poly64_t(kMu)));
  const uint64_t q = a_hi ^ vgetq_lane_u64(t, 1);
  const uint64x2_t u = vreinterpretq_u64_p128(vmull_p64(poly64_t(q), poly64_t(kPoly)));
  return a_lo ^ vgetq_lane_u64(u, 0);
}

inline uint64_t Fingerprint(const uint8_t* data, size_t bytes, uint64_t seed) {
  if (bytes == 0) return Mix(seed, 0);
  const uint8_t* p = data;
  size_t n = bytes;
  const uint64x2_t seed_block = vcombine_u64(vcreate_u64(seed), vcreate_u64(0));
  uint64x2_t s;
  if (n >= 64) {
    const poly64x2_t k4 = Constants(kFold4Lo, kFold4Hi);
    uint64x2_t a0 = veorq_u64(Load(p), seed_block), a1 = Load(p + 16), a2 = Load(p + 32), a3 = Load(p + 48);
    p += 64;
    n -= 64;
    while (n >= 64) {
      // Package 27c: the data is cold (guest vertex and index buffers, tens of MB a frame) and the A57's own
      // prefetcher does not keep up: this loop ran at about 1.5 bytes a cycle against the 8 its PMULLs allow.
      // A prefetch a few lines ahead; it changes no result.
      __builtin_prefetch(p + 512, 0, 0);
      a0 = veorq_u64(Fold(a0, k4), Load(p));
      a1 = veorq_u64(Fold(a1, k4), Load(p + 16));
      a2 = veorq_u64(Fold(a2, k4), Load(p + 32));
      a3 = veorq_u64(Fold(a3, k4), Load(p + 48));
      p += 64;
      n -= 64;
    }
    s = veorq_u64(veorq_u64(Fold(a0, Constants(kFold3Lo, kFold3Hi)), Fold(a1, Constants(kFold2Lo, kFold2Hi))),
                  veorq_u64(Fold(a2, Constants(kFold1Lo, kFold1Hi)), a3));
  } else {
    uint8_t block[16] = {};
    const size_t first = n < 16 ? n : 16;
    std::memcpy(block, p, first);
    s = veorq_u64(Load(block), seed_block);
    p += first;
    n -= first;
  }
  const poly64x2_t k1 = Constants(kFold1Lo, kFold1Hi);
  while (n >= 16) {
    s = veorq_u64(Fold(s, k1), Load(p));
    p += 16;
    n -= 16;
  }
  if (n) {
    uint8_t block[16] = {};
    std::memcpy(block, p, n);
    s = veorq_u64(Fold(s, k1), Load(block));
  }
  return Mix(Reduce(s), uint64_t(bytes));
}

#endif  // BANJO_PMULL_FINGERPRINT

}  // namespace banjo::pmull_fingerprint
