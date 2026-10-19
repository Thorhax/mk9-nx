/**
 * @file        ppc/diet_ops.h
 * @brief       Codegen "diet" helpers: cheaper but bit-identical lowerings of single PPC instructions.
 *
 * Included from context.h. Each helper reproduces exactly the value the previous generated expression produced
 * (including the upper word of the 64-bit register image), so it can be switched on per option.
 */

#pragma once

#include <bit>
#include <cstdint>

#include <rex/platform/fpscr.h>

namespace rex::ppc {

/// fctiwz: truncate toward zero to int32 with PPC saturation, NaN -> 0x80000000. Returned as the s64 register image
/// that the old expression produced: NaN -> 0x0000000080000000 (zero extended), everything else sign extended.
/// The old expression was `isnan ? 0x80000000U : v >= INT_MAX ? INT_MAX : simde_mm_cvttsd_si32(v)`, where simde's
/// portable version returns INT32_MIN for v <= INT32_MIN: that is exactly the saturating AArch64 fcvtzs (positive
/// overflow -> INT32_MAX, negative overflow -> INT32_MIN); only NaN (fcvtzs gives 0) needs the select.
inline int64_t fctiwz(double v) noexcept {
#if defined(__aarch64__)
  int32_t r;
  asm("fcvtzs %w0, %d1" : "=r"(r) : "w"(v));
  return __builtin_isnan(v) ? int64_t(0x80000000U) : int64_t(r);
#else
  if (__builtin_isnan(v))
    return int64_t(0x80000000U);
  if (v >= 2147483647.0)
    return INT32_MAX;
  return (v > -2147483648.0) ? int64_t(int32_t(v)) : int64_t(INT32_MIN);
#endif
}

/// sa2-nx round 7 (codegen option fp_moves_modeless): the lfs/stfs conversions with the IEEE result, i.e. the one
/// the conversion gives with the host's flush mode off, whatever the mode is. VMX code runs with the mode on (the
/// 360's VMX flushes denormals); the scalar float loads and stores between its instructions used to switch it off and
/// on again around each of them, and on the A57 every switch is an `msr fpcr` that costs as much as dozens of
/// instructions (SA2's VMX-heavy game thread spent 20-26 % of its time right after one).
/// The flush mode can change a single <-> double conversion only for a denormal single input (lfs), or a result in
/// the single denormal range or a denormal double input (stfs). Those convert in the cold helpers below with the mode
/// cleared for that one instruction (the volatile asm barriers keep the conversion between the two mode writes).
namespace detail {
#if defined(__aarch64__) || defined(_M_ARM64)
#define REX_DIET_FP_BARRIER(x) __asm__ __volatile__("" : "+w"(x))
#elif defined(__x86_64__) || defined(_M_X64)
#define REX_DIET_FP_BARRIER(x) __asm__ __volatile__("" : "+x"(x))
#else
#define REX_DIET_FP_BARRIER(x) ((void)0)
#endif

[[gnu::noinline, gnu::cold]] inline double SingleToDoubleSlow(uint32_t bits) noexcept {
  using P = rex::platform::FPSCRPlatform;
  const auto saved = P::getcsr();
  P::setcsr(decltype(saved)(saved & ~decltype(saved)(P::FlushMask)));
  float f = std::bit_cast<float>(bits);
  REX_DIET_FP_BARRIER(f);
  double d = double(f);
  REX_DIET_FP_BARRIER(d);
  P::setcsr(saved);
  return d;
}

[[gnu::noinline, gnu::cold]] inline float DoubleToSingleSlow(double d) noexcept {
  using P = rex::platform::FPSCRPlatform;
  const auto saved = P::getcsr();
  P::setcsr(decltype(saved)(saved & ~decltype(saved)(P::FlushMask)));
  REX_DIET_FP_BARRIER(d);
  float f = float(d);
  REX_DIET_FP_BARRIER(f);
  P::setcsr(saved);
  return f;
}
#undef REX_DIET_FP_BARRIER
}  // namespace detail

/// lfs: the single's bits to the double register value. A denormal single (exponent 0, fraction not 0) takes the
/// slow path; zero, normal, infinite and NaN inputs convert the same in either mode.
inline double SingleToDoubleAnyMode(uint32_t bits) noexcept {
  if (uint32_t((bits & 0x7FFFFFFFu) - 1u) < 0x007FFFFFu) [[unlikely]] {
    return detail::SingleToDoubleSlow(bits);
  }
  return double(std::bit_cast<float>(bits));
}

/// stfs: the double register value to the stored single. 0 < |d| < 2^-126 (FLT_MIN; this includes every denormal
/// double) takes the slow path: only there can the result be tiny. From FLT_MIN up, and for zero, infinity and NaN,
/// neither an input flush nor an output flush can apply, on AArch64 (tininess before rounding) or x86 (after).
inline float DoubleToSingleAnyMode(double d) noexcept {
  const uint64_t a = std::bit_cast<uint64_t>(d) & 0x7FFFFFFFFFFFFFFFull;
  if (a - 1u < 0x3810000000000000ull - 1u) [[unlikely]] {
    return detail::DoubleToSingleSlow(d);
  }
  return float(d);
}

}  // namespace rex::ppc
