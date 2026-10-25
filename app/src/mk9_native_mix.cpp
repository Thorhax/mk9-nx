// Mortal Kombat Komplete Edition: the sound engine's mix-matrix routine, native (CPU enhancement mk9_cpu_native_mix).
//
// WHAT THE GAME DOES
//   sub_82E1E148(r3 = the voice's mix object, r4 = out, r5 = in, r6 = output channels, r7 = input channels,
//   r8 = frames) mixes one voice into the output: for each frame and output channel j,
//       out[j] += sum over i of in[i] * gain[j][i];   then gain[j][i] += step[j][i]
//   with the gains and their steps kept as Xenon float16 (vupkd3d128 / vpkd3d128 FLOAT16_4, through the stack).
//   Lines: gain[j] at [obj+64+4j], step[j] at [obj+88+4j], target[j] at [obj+40+4j] (big-endian halves). The ramp's
//   frames left (s16 at obj+120) drop by the frames mixed; at 0 every gain (rows obj+36 x columns obj+38, s16) becomes
//   target * scale (float at obj+124). Returns 0.
//   Switch 2026-10-25 (SOAK19P, profiled): the guest sound thread used 40 % of a core on the lava stage, 31 % of it in
//   this routine: each element goes through the recompiler's float16 emulation and four stack round trips.
//
// WHAT THIS DOES
//   The same loops in C++, with the recompiled code's arithmetic: fadds / fmuls rounded to float from a double,
//   fmadds a double fma rounded to float, the FPU's flush mode off (as the recompiled code sets it for them), and the
//   recompiler's float16 conversions bit for bit (no infinity or NaN: exponent 31 is a normal number; denormal halves
//   read as zero; packing truncates, clamps above 65504 and NaN to 0x7FFF, keeps the sign), and its flush mode as the
//   loads see it (a pack turns flush-to-zero on, the next arithmetic off: a denormal sample loaded after a pack reads
//   as zero; the first verify run differed in 209 of 262,144 calls, all mono voices, before this was copied).
//   mk9_native_mix_verify runs the original on the same memory first and compares every byte it wrote ([mk9] mix).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ppc.h>

#include "cpu/sa2_enhancements.h"

SA2_ENHANCEMENT(g_native_mix, mk9_cpu_native_mix, sa2::enh::kCpu, true,
                "The sound engine's mix-matrix routine (sub_82E1E148: float16 gain ramps) runs as native code");
REXCVAR_DEFINE_INT32(mk9_native_mix_verify, 0, "MK9",
                     "mk9_cpu_native_mix: 1 = each call also runs the original on a copy and compares the results "
                     "([mk9] mix lines; tests only, slower than the original); 2 = the original against itself (what "
                     "other threads change meanwhile: the baseline)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

constexpr uint32_t kMix = 0x82E1E148;
constexpr uint32_t kMixBytes = 816;  // tools/guest_code_hash.py 82E1E148
constexpr uint64_t kMixHash = 0x762253E54AF2BD24ull;

inline uint32_t PhysOff(uint32_t a) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  return a >= 0xE0000000u ? 0x1000u : 0u;
#else
  (void)a;
  return 0u;
#endif
}
inline uint8_t* At(uint8_t* base, uint32_t a) { return base + a + PhysOff(a); }
inline uint32_t Ld32(uint8_t* base, uint32_t a) {
  uint32_t v;
  std::memcpy(&v, At(base, a), 4);
  return __builtin_bswap32(v);
}
inline uint16_t Ld16(uint8_t* base, uint32_t a) {
  uint16_t v;
  std::memcpy(&v, At(base, a), 2);
  return __builtin_bswap16(v);
}
inline void St16(uint8_t* base, uint32_t a, uint16_t v) {
  v = __builtin_bswap16(v);
  std::memcpy(At(base, a), &v, 2);
}
inline float LdF(uint8_t* base, uint32_t a) {
  const uint32_t u = Ld32(base, a);
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline void StF(uint8_t* base, uint32_t a, float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u = __builtin_bswap32(u);
  std::memcpy(At(base, a), &u, 4);
}

// vupkd3d128 FLOAT16_4, one element (the recompiler's).
inline float HalfToFloat(uint16_t h) {
  const uint32_t t = h;
  uint32_t u = ((t & 0x8000) << 16) | (((t & 0x7C00) + 0x1C000) << 13) | ((t & 0x03FF) << 13);
  if ((t & 0x7C00) == 0) u = (t & 0x8000) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// vpkd3d128 FLOAT16_4, one element (the recompiler's).
inline uint16_t FloatToHalf(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t a = x & 0x7FFFFFFF;
  float af;
  std::memcpy(&af, &a, 4);
  uint16_t h;
  if (af != af || af > 65504.0f) {
    h = 0x7FFF;
  } else {
    const uint32_t e = (x & 0x7F800000) >> 23;
    const uint32_t m = (x & 0x7FE000) >> 13;
    if (e > 0x70)
      h = uint16_t(((e - 0x70) << 10) + m);
    else
      h = 0x71 - e > 31 ? 0 : uint16_t((0x400 + m) >> (0x71 - e));
  }
  return uint16_t(h | ((x & 0x80000000) >> 16));
}

// The recompiled code's single-precision operations.
inline double Fadds(double a, double b) { return double(float(a + b)); }
inline double Fmuls(double a, double b) { return double(float(a * b)); }
inline double Fmadds(double a, double b, double c) { return double(float(std::fma(a, b, c))); }

// A float load (lfs) as the recompiled code does it: converted under the flush mode in force, which a vpkd3d128 turns
// on and the next arithmetic turns off (codegen fp_moves_modeless), so after a pack a denormal sample reads as zero.
inline double LdFMode(uint8_t* base, uint32_t a, bool flush) {
  float f = LdF(base, a);
  if (flush && std::fpclassify(f) == FP_SUBNORMAL) f = std::copysign(0.0f, f);
  return f;
}

// 'flush': the mode at entry. Returns the mode the original leaves.
bool NativeMix(uint8_t* base, uint32_t obj, uint32_t out, uint32_t in, int32_t outs, int32_t ins, uint32_t frames,
               bool flush) {
  for (uint32_t f = 0; f < frames; ++f) {
    for (int32_t j = 0; j < outs; ++j) {
      const uint32_t gain = Ld32(base, obj + 64 + 4 * j);
      const uint32_t step = Ld32(base, obj + 88 + 4 * j);
      const double before = LdFMode(base, out, flush);
      double even = 0.0, odd = 0.0, last = 0.0;
      int32_t i = 0;
      if (ins >= 2) {
        do {
          const double x0 = LdFMode(base, in + 4 * i, flush);
          const float g0 = HalfToFloat(Ld16(base, gain + 2 * i));
          St16(base, gain + 2 * i, FloatToHalf(float(Fadds(HalfToFloat(Ld16(base, step + 2 * i)), g0))));
          const double x1 = LdFMode(base, in + 4 * i + 4, true);  // after the pack
          even = Fmadds(x0, g0, even);
          const float g1 = HalfToFloat(Ld16(base, gain + 2 * i + 2));
          St16(base, gain + 2 * i + 2, FloatToHalf(float(Fadds(HalfToFloat(Ld16(base, step + 2 * i + 2)), g1))));
          odd = Fmadds(x1, g1, odd);
          flush = true;  // the iteration ends with a pack
          i += 2;
        } while (i < ins - 1);
      }
      if (i < ins) {
        const double x = LdFMode(base, in + 4 * i, flush);
        const float g = HalfToFloat(Ld16(base, gain + 2 * i));
        St16(base, gain + 2 * i, FloatToHalf(float(Fadds(HalfToFloat(Ld16(base, step + 2 * i)), g))));
        last = Fmuls(x, g);
      }
      StF(base, out, float(Fadds(Fadds(Fadds(odd, even), last), before)));
      flush = false;  // the channel ends with fadds
      out += 4;
    }
    in += 4 * uint32_t(ins);
  }
  const int16_t left = int16_t(Ld16(base, obj + 120) - uint16_t(frames));
  St16(base, obj + 120, uint16_t(left));
  if (left != 0)
    return flush;
  const int16_t rows = int16_t(Ld16(base, obj + 36));
  for (int32_t r = 0; r < rows; ++r) {
    const uint32_t target = Ld32(base, obj + 40 + 4 * r);
    const uint32_t gain = Ld32(base, obj + 64 + 4 * r);
    for (int32_t c = 0; c < int16_t(Ld16(base, obj + 38)); ++c) {
      const double scale = LdFMode(base, obj + 124, flush);  // loaded again for each element
      St16(base, gain + 2 * c, FloatToHalf(float(Fmuls(scale, HalfToFloat(Ld16(base, target + 2 * c))))));
      flush = true;
    }
  }
  return flush;
}

// mk9_native_mix_verify: what a call may write, saved, so the original and the native code start from the same bytes.
struct Region {
  uint32_t addr, size;
};
std::vector<Region> Written(uint8_t* base, uint32_t obj, uint32_t out, int32_t outs, int32_t ins, uint32_t frames) {
  std::vector<Region> regions;
  if (frames && outs > 0) regions.push_back({out, frames * uint32_t(outs) * 4});
  regions.push_back({obj + 120, 2});
  const int32_t rows = std::max<int32_t>(outs, int16_t(Ld16(base, obj + 36)));
  const int32_t cols = std::max<int32_t>(ins, int16_t(Ld16(base, obj + 38)));
  for (int32_t r = 0; r < rows && r < 6 && cols > 0; ++r)
    regions.push_back({Ld32(base, obj + 64 + 4 * r), uint32_t(cols) * 2});
  return regions;
}
std::vector<uint8_t> Save(uint8_t* base, const std::vector<Region>& regions) {
  std::vector<uint8_t> bytes;
  for (const Region& r : regions) bytes.insert(bytes.end(), At(base, r.addr), At(base, r.addr) + r.size);
  return bytes;
}
void Restore(uint8_t* base, const std::vector<Region>& regions, const std::vector<uint8_t>& bytes) {
  size_t at = 0;
  for (const Region& r : regions) {
    std::memcpy(At(base, r.addr), bytes.data() + at, r.size);
    at += r.size;
  }
}

std::atomic<int> g_state{0};  // 0 unknown, 1 on, -1 off (code check failed or key off); set by the first call
std::atomic<uint64_t> g_calls{0}, g_differ{0};

}  // namespace

REX_EXTERN(__imp__sub_82E1E148);
REX_HOOK_RAW(sub_82E1E148) {
  int state = g_state.load(std::memory_order_relaxed);
  if (state == 0) {
    state = (g_native_mix.active && sa2::enh::GuestCodeMatches(base, kMix, kMixBytes, kMixHash, "mk9_cpu_native_mix"))
                ? 1
                : -1;
    g_state.store(state, std::memory_order_relaxed);
  }
  if (state < 0) {
    __imp__sub_82E1E148(ctx, base);
    return;
  }
  const uint32_t obj = ctx.r3.u32, out = ctx.r4.u32, in = ctx.r5.u32;
  const int32_t outs = ctx.r6.s32, ins = ctx.r7.s32;
  const uint32_t frames = ctx.r8.u32;
  static const int32_t verify = REXCVAR_GET(mk9_native_mix_verify);
  std::vector<Region> regions;
  std::vector<uint8_t> before, original;
  if (verify) {
    regions = Written(base, obj, out, outs, ins, frames);
    before = Save(base, regions);
    __imp__sub_82E1E148(ctx, base);
    original = Save(base, regions);
    Restore(base, regions, before);
  }
  if (verify == 2) {
    ctx.r3.u64 = obj, ctx.r4.u64 = out, ctx.r5.u64 = in;  // the first run used them up
    ctx.r6.s64 = outs, ctx.r7.s64 = ins, ctx.r8.u64 = frames;
    __imp__sub_82E1E148(ctx, base);
  } else {
    const bool entry_flush = (ctx.fpscr.csr & rex::ppc::FPSCRRegister::FlushMask) != 0;
    ctx.fpscr.disableFlushMode();
    if (NativeMix(base, obj, out, in, outs, ins, frames, entry_flush))
      ctx.fpscr.enableFlushMode();
  }
  ctx.r3.u64 = 0;
  g_native_mix.Use();
  if (verify) {
    const uint64_t calls = g_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const std::vector<uint8_t> native = Save(base, regions);
    if (native != original) {
      const uint64_t differ = g_differ.fetch_add(1, std::memory_order_relaxed) + 1;
      if (differ <= 8) {
        // The first differing word: which region (0 out, 1 the ramp counter, 2+ gain lines) and both values.
        size_t at = 0, k = 0;
        while (at < native.size() && native[at] == original[at]) ++at;
        size_t start = 0;
        while (k < regions.size() && start + regions[k].size <= at) start += regions[k++].size;
        const size_t word = start + (at - start) / 4 * 4;
        auto hex = [](const std::vector<uint8_t>& v, size_t o) {
          uint32_t w = 0;
          for (size_t b = 0; b < 4 && o + b < v.size(); ++b) w = (w << 8) | v[o + b];
          return w;
        };
        REXLOG_WARN("[mk9] mix: native result differs from the original (obj {:08X}, {} outs, {} ins, {} frames): "
                    "region {} byte {}: original {:08X} native {:08X} before {:08X}; in[0] {:08X}, ramp {}, gain0 {:04X} "
                    "step0 {:04X}",
                    obj, outs, ins, frames, k, at - start, hex(original, word), hex(native, word), hex(before, word),
                    Ld32(base, in), int16_t(Ld16(base, obj + 120)), Ld16(base, Ld32(base, obj + 64)),
                    Ld16(base, Ld32(base, obj + 88)));
      }
    }
    if ((calls & (calls - 1)) == 0 && calls >= 1024)
      REXLOG_INFO("[mk9] mix verify: {} calls, {} differ", calls, g_differ.load(std::memory_order_relaxed));
  }
}
