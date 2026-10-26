// Mortal Kombat Komplete Edition: the skinning bone-matrix copy, native (CPU enhancement mk9_cpu_native_bones).
//
// WHAT THE GAME DOES
//   sub_82972F60(r3 = this, r4 = a pose object) on the game's rendering thread: releases the object at this+44
//   (sub_82590438, sub_825AFD70), stores the new one there and takes a reference (sub_82590410), calls its vtable[1],
//   then for each section of the mesh (this+12 -> +336, indexed by the pose's +76) copies the section's bone matrices:
//   bone j's 16-bit index from the section's table (+52), the pose's 4x4 matrix at pose+28 + index * 64, transposed to
//   three rows of four floats at the section's output buffer (+572) + 48 * j; when bit 30 of this+12 -> +396 is set,
//   again from pose+52 to output + 1728 (36 bones on). It sets the output's +588 to that bit and +584 to min(section
//   +72, 4), and this+28 to -1 at the end. Switch 2026-10-26 (HELLPROF, v19): 6.2 % of the game's rendering thread on
//   Hell, every float through lfs / stfs.
//
// WHAT THIS DOES
//   The same calls (into the guest code), then the copies as 32-bit words (lfs / stfs move single floats unchanged:
//   the words keep their byte order), the section's stores marking their store cards as the recompiled stores do.
//   mk9_native_bones_verify runs the original instead and compares what the native copies would write.
#include <atomic>
#include <cstdint>
#include <cstring>
#include <initializer_list>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/ppc.h>
#include <rex/runtime.h>
#include <rex/system/xmemory.h>

#include "cpu/sa2_enhancements.h"

SA2_ENHANCEMENT(g_native_bones, mk9_cpu_native_bones, sa2::enh::kCpu, true,
                "The skinning bone-matrix copy (sub_82972F60) runs as native code");
REXCVAR_DEFINE_BOOL(mk9_native_bones_verify, false, "MK9",
                    "mk9_cpu_native_bones: each call runs the original and compares what the native copies would "
                    "write ([mk9] bones lines; tests only)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

constexpr uint32_t kFunc = 0x82972F60;
constexpr uint32_t kFuncBytes = 1736;  // tools/guest_code_hash.py 82972F60
constexpr uint64_t kFuncHash = 0xBC5EFFD3FDF0B3CAull;
constexpr uint32_t kRelease = 0x82590438, kFree = 0x825AFD70, kAddRef = 0x82590410;
constexpr uint32_t kSecondSet = 1728;  // the second matrix set's offset in a section's output (36 bones of 48 bytes)

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
inline void St32(uint8_t* base, uint32_t a, uint32_t v) {
  v = __builtin_bswap32(v);
  std::memcpy(At(base, a), &v, 4);
  rex::memory::MarkStoreCardsAt(base, a, 4);
}

// A guest function called from the host (r3 its argument), with a frame of its own below the caller's stack pointer.
void CallGuest(PPCContext& ctx, uint8_t* base, uint32_t fn, uint32_t r3) {
  const uint64_t r1 = ctx.r1.u64, lr = ctx.lr;
  ctx.r1.u64 = (ctx.r1.u32 - 0x200) & ~0xFu;
  ctx.r3.u64 = r3;
  rex::runtime::ResolveIndirectFunction(fn)(ctx, base);
  ctx.r1.u64 = r1;
  ctx.lr = lr;
}

// One bone: the 4x4 matrix's first three columns as three rows of four (word order 0 4 8 12 / 1 5 9 13 / 2 6 10 14).
inline void CopyBone(const uint32_t* s, uint32_t* d) {
  d[0] = s[0], d[1] = s[4], d[2] = s[8], d[3] = s[12];
  d[4] = s[1], d[5] = s[5], d[6] = s[9], d[7] = s[13];
  d[8] = s[2], d[9] = s[6], d[10] = s[10], d[11] = s[14];
}

// A set of 'count' bones: indices at 'table', matrices at 'matrices', to 'out' (or to 'scratch' for the verify mode).
// The matrices are read in the index table's order, scattered over the pose: each one a cache miss on the console
// (HELLPROF22: the first load of a matrix took most of the copy's time). They are prefetched kAhead bones ahead.
constexpr int32_t kAhead = 4;
inline void PrefetchBone(uint8_t* base, uint32_t table, uint32_t matrices, int32_t j) {
  __builtin_prefetch(At(base, matrices + (uint32_t(Ld16(base, table + 2 * uint32_t(j))) << 6)));
}

void CopySet(uint8_t* base, uint32_t table, uint32_t matrices, int32_t count, uint32_t out, uint32_t* scratch) {
  for (int32_t j = 0; j < count && j < kAhead; ++j) PrefetchBone(base, table, matrices, j);
  for (int32_t j = 0; j < count; ++j) {
    if (j + kAhead < count) PrefetchBone(base, table, matrices, j + kAhead);
    const uint32_t src = matrices + (uint32_t(Ld16(base, table + 2 * uint32_t(j))) << 6);
    uint32_t s[16], d[12];
    std::memcpy(s, At(base, src), 64);
    CopyBone(s, d);
    std::memcpy(scratch ? scratch + 12 * j : reinterpret_cast<uint32_t*>(At(base, out + 48 * uint32_t(j))), d, 48);
  }
  if (!scratch && count > 0) rex::memory::MarkStoreCardsAt(base, out, 48 * uint32_t(count));
}

// The sections' copies (after the calls). verify: compares instead of writing; returns the number of differences.
uint32_t Sections(uint8_t* base, uint32_t self, bool verify) {
  const uint32_t pose = Ld32(base, self + 44), mesh = Ld32(base, self + 12);
  const uint32_t index = Ld32(base, pose + 76);
  const uint32_t outputs = index * 536 + Ld32(base, self + 16);
  const uint32_t second = (Ld32(base, mesh + 396) >> 30) & 1;
  const uint32_t list = Ld32(base, Ld32(base, mesh + 336) + index * 4);
  const int32_t sections = int32_t(Ld32(base, list + 16));
  uint32_t differ = 0;
  static uint32_t scratch[36 * 12];
  for (int32_t i = 0; i < sections; ++i) {
    const uint32_t section = Ld32(base, list + 12) + 76 * uint32_t(i);
    const uint32_t out = Ld32(base, Ld32(base, outputs + 524) + 4 * uint32_t(i));
    if (i + 1 < sections) {  // the next section's output object (its +572 was the sections' other miss)
      __builtin_prefetch(At(base, Ld32(base, Ld32(base, outputs + 524) + 4 * uint32_t(i + 1)) + 572));
      __builtin_prefetch(At(base, section + 76 + 52));
    }
    const int32_t count = int32_t(Ld32(base, section + 56));
    const uint32_t table = Ld32(base, section + 52), buffer = Ld32(base, out + 572);
    const int32_t c72 = int32_t(Ld32(base, section + 72));
    const uint32_t limit = uint32_t(c72 < 4 ? c72 : 4);
    if (!verify) {
      St32(base, out + 588, second);
      CopySet(base, table, Ld32(base, pose + 28), count, buffer, nullptr);
      if (second) CopySet(base, table, Ld32(base, pose + 52), count, buffer + kSecondSet, nullptr);
      St32(base, out + 584, limit);
      continue;
    }
    const int32_t n = count > 36 ? 36 : count;
    differ += Ld32(base, out + 588) != second;
    differ += Ld32(base, out + 584) != limit;
    if (n > 0) {
      CopySet(base, table, Ld32(base, pose + 28), n, buffer, scratch);
      differ += std::memcmp(scratch, At(base, buffer), 48 * uint32_t(n)) != 0;
      if (second) {
        CopySet(base, table, Ld32(base, pose + 52), n, buffer + kSecondSet, scratch);
        differ += std::memcmp(scratch, At(base, buffer + kSecondSet), 48 * uint32_t(n)) != 0;
      }
    }
  }
  return differ;
}

std::atomic<int> g_state{0};  // 0 unknown, 1 on, -1 off (code check failed or key off); set by the first call
std::atomic<uint64_t> g_calls{0}, g_differ{0};

}  // namespace

REX_EXTERN(__imp__sub_82972F60);
REX_HOOK_RAW(sub_82972F60) {
  int state = g_state.load(std::memory_order_relaxed);
  if (state == 0) {
    state = (g_native_bones.active && sa2::enh::GuestCodeMatches(base, kFunc, kFuncBytes, kFuncHash,
                                                                 "mk9_cpu_native_bones"))
                ? 1
                : -1;
    g_state.store(state, std::memory_order_relaxed);
  }
  if (state < 0) {
    __imp__sub_82972F60(ctx, base);
    return;
  }
  const uint32_t self = ctx.r3.u32, pose = ctx.r4.u32;

  static const bool verify = REXCVAR_GET(mk9_native_bones_verify);
  if (verify) {
    __imp__sub_82972F60(ctx, base);
    const uint64_t calls = g_calls.fetch_add(1, std::memory_order_relaxed) + 1;
    const uint32_t differ = Sections(base, self, true) + (Ld32(base, self + 28) != 0xFFFFFFFFu);
    if (differ && g_differ.fetch_add(1, std::memory_order_relaxed) < 8)
      REXLOG_WARN("[mk9] bones: native copies differ from the original's in {} places (this {:08X}, pose {:08X})",
                  differ, self, pose);
    if ((calls & (calls - 1)) == 0 && calls >= 1024)
      REXLOG_INFO("[mk9] bones verify: {} calls, {} differ", calls, g_differ.load(std::memory_order_relaxed));
    return;
  }

  // The calls, as the original makes them: the old pose released, the new one stored and referenced, its vtable[1].
  if (const uint32_t old = Ld32(base, self + 44)) {
    CallGuest(ctx, base, kRelease, old);
    CallGuest(ctx, base, kFree, Ld32(base, self + 44));
  }
  St32(base, self + 44, pose);
  CallGuest(ctx, base, kAddRef, pose);
  {
    const uint32_t object = Ld32(base, self + 44);
    CallGuest(ctx, base, Ld32(base, Ld32(base, object) + 4), object);
  }
  Sections(base, self, false);
  St32(base, self + 28, 0xFFFFFFFFu);
  ctx.r11.u64 = 0xFFFFFFFFFFFFFFFFull;  // as the original leaves it (li r11,-1)
  g_native_bones.Use();
}
