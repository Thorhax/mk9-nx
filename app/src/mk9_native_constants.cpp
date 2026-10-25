// Mortal Kombat Komplete Edition: Direct3D's shader constant copy, native (CPU enhancement mk9_cpu_native_constants).
//
// WHAT THE GAME DOES
//   sub_82BE9828(r3 = the device, r4 = the first constant register, r5 = the source, r6 = the count of vec4 registers,
//   r7 = the dirty bits to set) copies r6 16-byte vectors from r5 (any alignment: lvlx128 + lvrx128 + vor128) to the
//   device's constant shadow at (r3 + ((r4 + 120) << 4)) & ~15 (stvx128), four at a time and then one at a time, then
//   ORs r7 into the device's 64-bit dirty word at r3. Eight callers (the vertex and pixel shader SetShaderConstantF
//   paths, some as tail calls). Switch 2026-10-26 (PROF_HELL2, LTO build): 5.6 % of the game's rendering thread on
//   Hell, every vector through the recompiler's lvlx/lvrx emulation.
//
// WHAT THIS DOES
//   The same bytes with one memmove (guest to guest: the byte order does not change), the dirty word's OR, and the
//   registers the routine leaves (r4-r6 and r8-r11, cr6) as it leaves them. Its stores mark their store cards, as the
//   recompiled stores do. mk9_native_constants_verify runs the original on the same memory first and compares.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ppc.h>
#include <rex/system/xmemory.h>

#include "cpu/sa2_enhancements.h"

SA2_ENHANCEMENT(g_native_constants, mk9_cpu_native_constants, sa2::enh::kCpu, true,
                "Direct3D's shader constant copy (sub_82BE9828: SetShaderConstantF's vec4 copy) runs as native code");
REXCVAR_DEFINE_BOOL(mk9_native_constants_verify, false, "MK9",
                    "mk9_cpu_native_constants: each call also runs the original first and compares what both wrote "
                    "([mk9] constants lines; tests only)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

constexpr uint32_t kCopy = 0x82BE9828;
constexpr uint32_t kCopyBytes = 216;  // tools/guest_code_hash.py 82BE9828
constexpr uint64_t kCopyHash = 0x9C7F63BCBF9EE982ull;

inline uint32_t PhysOff(uint32_t a) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  return a >= 0xE0000000u ? 0x1000u : 0u;
#else
  (void)a;
  return 0u;
#endif
}
inline uint8_t* At(uint8_t* base, uint32_t a) { return base + a + PhysOff(a); }

std::atomic<int> g_state{0};  // 0 unknown, 1 on, -1 off (code check failed or key off); set by the first call
std::atomic<uint64_t> g_calls{0}, g_differ{0};

}  // namespace

REX_EXTERN(__imp__sub_82BE9828);
REX_HOOK_RAW(sub_82BE9828) {
  int state = g_state.load(std::memory_order_relaxed);
  if (state == 0) {
    state = (g_native_constants.active &&
             sa2::enh::GuestCodeMatches(base, kCopy, kCopyBytes, kCopyHash, "mk9_cpu_native_constants"))
                ? 1
                : -1;
    g_state.store(state, std::memory_order_relaxed);
  }
  if (state < 0) {
    __imp__sub_82BE9828(ctx, base);
    return;
  }
  const uint32_t device = ctx.r3.u32, first = ctx.r4.u32, src = ctx.r5.u32, count = ctx.r6.u32;
  const uint64_t bits = ctx.r7.u64;
  const uint32_t dst = (device + ((first + 120u) << 4)) & ~15u;
  const uint32_t bytes = count * 16u;

  static const bool verify = REXCVAR_GET(mk9_native_constants_verify);
  std::vector<uint8_t> before, original;
  uint64_t word_before = 0, word_original = 0;
  if (verify) {
    before.assign(At(base, dst), At(base, dst) + bytes);
    std::memcpy(&word_before, At(base, device), 8);
    __imp__sub_82BE9828(ctx, base);
    original.assign(At(base, dst), At(base, dst) + bytes);
    std::memcpy(&word_original, At(base, device), 8);
    std::memcpy(At(base, dst), before.data(), bytes);
    std::memcpy(At(base, device), &word_before, 8);
  }

  if (bytes) {
    std::memmove(At(base, dst), At(base, src), bytes);
    rex::memory::MarkStoreCardsAt(base, dst, bytes);
  }
  uint64_t word;
  std::memcpy(&word, At(base, device), 8);
  word = __builtin_bswap64(__builtin_bswap64(word) | bits);
  std::memcpy(At(base, device), &word, 8);
  rex::memory::MarkStoreCardsAt(base, device, 8);

  // The registers as the original leaves them: the four-at-a-time loop (count > 3) sets r4 = 64, r6 = 32, r8 = 48 and
  // leaves r9 = count % 4; otherwise r8 = 128 and r9 = count. r5 = 16, r10 past the last store, r11 the dirty word.
  const bool blocks = count > 3;
  const uint32_t rest = blocks ? count - ((count - 4) / 4 + 1) * 4 : count;
  if (blocks) {
    ctx.r4.u64 = 64;
    ctx.r6.u64 = 32;
    ctx.r8.u64 = 48;
  } else {
    ctx.r4.u64 = first;
    ctx.r8.u64 = 128;
  }
  ctx.r5.u64 = 16;
  ctx.r9.u64 = rest;
  ctx.r10.u64 = uint32_t(device + ((first + 120u) << 4) + bytes);
  ctx.r11.u64 = __builtin_bswap64(word);
  ctx.cr6.compare<uint32_t>(ctx.r9.u32, 0, ctx.xer);
  g_native_constants.Use();

  if (verify) {
    const uint64_t calls = g_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    uint64_t word_native;
    std::memcpy(&word_native, At(base, device), 8);
    if (std::memcmp(At(base, dst), original.data(), bytes) != 0 || word_native != word_original) {
      if (g_differ.fetch_add(1, std::memory_order_relaxed) < 8)
        REXLOG_WARN("[mk9] constants: native result differs from the original (device {:08X}, register {}, {} "
                    "vectors from {:08X}, bits {:016X})",
                    device, first, count, src, bits);
    }
    if ((calls & (calls - 1)) == 0 && calls >= 1024)
      REXLOG_INFO("[mk9] constants verify: {} calls, {} differ", calls, g_differ.load(std::memory_order_relaxed));
  }
}
