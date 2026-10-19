// Sonic Adventure 2: the native Direct3D layer's shared machinery (see sa2_api.h and docs/api-renderer.md): the
// PowerPC rotate-and-mask helpers the transliterated code uses, SA2's Direct3D device layout, guest memory access, the
// record ops, the draw-time state flush and shader load, the transport of records (marker NOPs while Direct3D code
// still writes PM4), and verify mode's shadow runs of the originals (on a device copy) with their comparisons.
#pragma once

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#include <fmt/format.h>
#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/platform.h>
#include <rex/ppc.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>
#include "cpu/sa2_enhancements.h"
#include "cpu/sa2_hot_copies.h"
#include "me/me_native_system.h"
#include "me/sa2_api.h"

// The enhancement keys of the native layer (one per group of entry points; defined in their files).
extern ::sa2::enh::Enhancement g_api_draws;
extern ::sa2::enh::Enhancement g_api_clears;

// Guest functions the native code calls: the pure shader-patching helpers, the command buffer's transport, and the
// originals verify mode runs on a device copy.
REX_EXTERN(sub_82555E18);
REX_EXTERN(sub_825565C8);
REX_EXTERN(__imp__sub_8254EBB8);
REX_EXTERN(__imp__sub_8254D9A8);
REX_EXTERN(__imp__sub_82565C18);
REX_EXTERN(__imp__sub_8254FCE8);

namespace sa2::api {
namespace detail {

// ---------------------------------------------------------------------------------------------------------------------
// PowerPC rotate-and-mask, IBM bit numbering (bit 0 = most significant).
constexpr uint32_t Mask32(int mb, int me) {
  const uint32_t a = 0xFFFFFFFFu >> mb, b = 0xFFFFFFFFu << (31 - me);
  return mb <= me ? (a & b) : (a | b);
}
constexpr uint32_t Rlwinm(uint32_t s, int sh, int mb, int me) { return std::rotl(s, sh) & Mask32(mb, me); }
constexpr uint32_t Rlwimi(uint32_t a, uint32_t s, int sh, int mb, int me) {
  const uint32_t m = Mask32(mb, me);
  return (std::rotl(s, sh) & m) | (a & ~m);
}
constexpr uint64_t Mask64(int mb, int me) {
  const uint64_t a = ~0ull >> mb, b = ~0ull << (63 - me);
  return mb <= me ? (a & b) : (a | b);
}
constexpr uint64_t Rldicr(uint64_t s, int sh, int me) { return std::rotl(s, sh) & Mask64(0, me); }
// cntlzw x, then rlwinm 27,31,31: 1 when x is 0.
constexpr uint32_t IsZero(uint32_t x) { return Rlwinm(uint32_t(std::countl_zero(x)), 27, 31, 31); }

// ---------------------------------------------------------------------------------------------------------------------
// SA2's Direct3D device (2012 XDK): the fields the draw path reads and writes.
namespace dev {
constexpr uint32_t kDirtyVsConsts = 0, kDirtyPsConsts = 8, kDirtyGroups = 16, kDirtyFetch = 24, kDirtyMisc = 32,
                   kDirtyRb = 40;  // groups re-sent with sub_825555A0
constexpr uint32_t kWptr = 48, kLimit = 52, kSoftLimit = 56;  // the command buffer (transport)
constexpr uint32_t kShadowFetch = 1152, kShadowVsConsts = 1920, kShadowPsConsts = 6016, kShadowBoolLoop = 10112;
constexpr uint32_t kShadow2388 = 10272;
constexpr uint32_t kShadow2000 = 10368, kShadow2100 = 10444, kShadow2180 = 10528, kShadow2200 = 10548,
                   kShadow2280 = 10596, kShadow2300 = 10680, kShadow2380 = 10832;
constexpr uint32_t kReg2007 = 10396, kReg2104 = 10460, kReg2180 = 10528, kReg2181 = 10532, kReg2182 = 10536,
                   kReg2203 = 10560, kReg2208 = 10580;
constexpr uint32_t kFenceCompleted = 11024;  // -> the GPU's last completed fence
constexpr uint32_t kFenceNext = 11036;       // the fence the next submission carries
constexpr uint32_t kFlags68 = 11068, kFlags69 = 11069, kFlags70 = 11070, kFlags71 = 11071, kFlags72 = 11072;
constexpr uint32_t kVertexDecl = 12240;
constexpr uint32_t kBoundStrides = 12264;  // 16 bytes: the strides the loaded VS binding was patched for
constexpr uint32_t kSurfaceFlags = 12284;
constexpr uint32_t kVizTable = 12498, kVizIndex = 12563;
constexpr uint32_t kStreamStrides = 12904;  // 16 bytes: stride / 4 of each vertex stream
constexpr uint32_t kBinMaskSave = 13084, kBinSelectSave = 13088;
constexpr uint32_t kReg2000Base = 13552;
constexpr uint32_t kPixelShader = 13068, kVertexShader = 13072;
constexpr uint32_t kDebugObject = 13904, kDebugMode = 13912;
constexpr uint32_t kSize = 0x3800;  // compared in verify mode (the fields above end below it)
}  // namespace dev

// Direct3D's globals the shader load uses.
constexpr uint32_t kGlobalInterpFlag = 0x83853878, kGlobalInterpMask = 0x83853874, kGlobalDeclIds = 0x8385387C;
// Vertices per primitive and overlap, per primitive type (8 bytes each), for draws of more than 65535 vertices.
constexpr uint32_t kPrimitiveTable = 0x820C65E8;

// ---------------------------------------------------------------------------------------------------------------------
// Guest memory (the native code's accessors; a template parameter, so another implementation can stand in).
inline uint8_t* HostOf(uint8_t* base, uint32_t a) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  return base + a + (a >= 0xE0000000u ? 0x1000u : 0u);  // the generated code's REX_PHYS_HOST_OFFSET
#else
  return base + a;
#endif
}

// A guest-to-guest copy of n bytes: one memcpy (the PC's physical-range offset split where a range crosses it).
inline void CopyGuest(uint8_t* base, uint32_t dst, uint32_t src, uint32_t n) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  auto crosses = [](uint32_t a, uint32_t len) { return a < 0xE0000000u && uint64_t(a) + len > 0xE0000000u; };
  if (crosses(dst, n) || crosses(src, n)) [[unlikely]] {
    for (uint32_t i = 0; i < n; ++i) *HostOf(base, dst + i) = *HostOf(base, src + i);
    return;
  }
#endif
  std::memcpy(HostOf(base, dst), HostOf(base, src), n);
}

struct Direct {
  uint8_t* base;
  const uint8_t* Host(uint32_t a) const { return HostOf(base, a); }
  uint8_t L8(uint32_t a) const { return *HostOf(base, a); }
  uint16_t L16(uint32_t a) const { return rex::memory::load_and_swap<uint16_t>(HostOf(base, a)); }
  uint32_t L32(uint32_t a) const { return rex::memory::load_and_swap<uint32_t>(HostOf(base, a)); }
  uint64_t L64(uint32_t a) const { return rex::memory::load_and_swap<uint64_t>(HostOf(base, a)); }
  void S8(uint32_t a, uint8_t v) { *HostOf(base, a) = v; }
  void S32(uint32_t a, uint32_t v) { rex::memory::store_and_swap<uint32_t>(HostOf(base, a), v); }
  void S64(uint32_t a, uint64_t v) { rex::memory::store_and_swap<uint64_t>(HostOf(base, a), v); }
  // Direct3D's interlocked increment of a guest word (lwarx/stwcx.): the new value.
  uint32_t AtomicIncrement(uint32_t a) {
    auto* p = reinterpret_cast<std::atomic<uint32_t>*>(HostOf(base, a));
    uint32_t old = p->load(std::memory_order_relaxed);
    while (!p->compare_exchange_weak(old, __builtin_bswap32(__builtin_bswap32(old) + 1), std::memory_order_acq_rel)) {
    }
    return __builtin_bswap32(old) + 1;
  }
};

// ---------------------------------------------------------------------------------------------------------------------
// Bulk copy of guest words (big-endian) to host order: the register runs' values from the device's shadows. ARM first:
// NEON's rev32 swaps four words an instruction (the A57 runs the loop near memcpy speed); the compilers vectorize the
// scalar form elsewhere. `src` is the host address of the first guest word.
inline void SwapCopy(uint32_t* __restrict dst, const uint8_t* __restrict src, uint32_t n) {
  uint32_t i = 0;
#if defined(__aarch64__)
  for (; i + 8 <= n; i += 8) {
    const uint8x16_t a = vld1q_u8(src + 4 * i), b = vld1q_u8(src + 4 * i + 16);
    vst1q_u8(reinterpret_cast<uint8_t*>(dst + i), vrev32q_u8(a));
    vst1q_u8(reinterpret_cast<uint8_t*>(dst + i + 4), vrev32q_u8(b));
  }
  if (i + 4 <= n) {
    vst1q_u8(reinterpret_cast<uint8_t*>(dst + i), vrev32q_u8(vld1q_u8(src + 4 * i)));
    i += 4;
  }
#endif
  for (; i < n; ++i) {
    uint32_t v;
    std::memcpy(&v, src + 4 * i, 4);
    dst[i] = __builtin_bswap32(v);
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Ops: the direct-state record format (me_native_system.cpp, namespace direct).
enum Op : uint32_t { kRun = 1, kType3 = 4, kPair = 5 };
constexpr uint32_t kOpsCapacity = 1u << 16;

// DRAW_INDX / DRAW_INDX_2: the packets whose address words SetPacketAddresses fills (the ring keys replays by them).
constexpr bool IsDrawHeader(uint32_t header) {
  const uint32_t opcode = (header >> 8) & 0x7F;
  return opcode == 0x22 || opcode == 0x36;
}

// A call's ops. Written through a cursor that is a pointer: a store of an op word (uint32_t) cannot alias it, so the
// compilers keep it in a register (with a uint32_t count, every store forced a reload and a store of the count: about
// 11 instructions an op word on the A57). The words SetPacketAddresses fills (kPair records, draw packets) are noted as
// they are written instead of found by a scan of the ops.
struct Ops {
  static constexpr uint32_t kMaxFix = 32;
  uint32_t* buf;
  uint32_t* cur;
  uint32_t* lim;
  bool overflow = false;
  bool in_ring = false;  // written in place in the ring of records (OpenOps): Publish commits instead of copying
  uint32_t nfix = 0;
  uint32_t fix[kMaxFix];  // word index of a kPair tag, or of a draw op's tag | 0x80000000
  explicit Ops(uint32_t* b) : buf(b), cur(b), lim(b + kOpsCapacity) {}
  uint32_t size() const { return uint32_t(cur - buf); }
  bool Room(uint32_t words) {
    if (uint32_t(lim - cur) >= words) return true;
    overflow = true;
    return false;
  }
  void Fix(uint32_t index) {
    if (nfix < kMaxFix) fix[nfix] = index;
    ++nfix;  // more than kMaxFix: SetPacketAddresses scans instead
  }
  void Push(uint32_t v) {
    if (Room(1)) *cur++ = v;
  }
  // A register run of `count` values from guest memory at `src` (a shadow), in host order.
  template <class M>
  void Run(const M& m, uint32_t reg, uint32_t count, uint32_t src) {
    if (!count || !Room(2 + count)) return;
    uint32_t* const p = cur;
    p[0] = kRun | (count << 8);
    p[1] = reg;
    SwapCopy(p + 2, m.Host(src), count);
    cur = p + 2 + count;
  }
  void Reg(uint32_t reg, uint32_t v) {
    if (!Room(3)) return;
    uint32_t* const p = cur;
    p[0] = kRun | (1u << 8);
    p[1] = reg;
    p[2] = v;
    cur = p + 3;
  }
  // A register run of the given values.
  void Regs(uint32_t reg, std::initializer_list<uint32_t> values) {
    const uint32_t count = uint32_t(values.size());
    if (!count || !Room(2 + count)) return;
    uint32_t* const p = cur;
    p[0] = kRun | (count << 8);
    p[1] = reg;
    uint32_t k = 2;
    for (uint32_t v : values) p[k++] = v;
    cur = p + k;
  }
  // A type-3 packet as the ring runs it (Type3), payload in guest byte order; `address` names the packet (the ring
  // keys the draw pairing by packet address).
  void Type3(uint32_t header, uint32_t address, std::initializer_list<uint32_t> words) {
    const uint32_t count = uint32_t(words.size());
    if (!Room(3 + count)) return;
    uint32_t* const p = cur;
    if (IsDrawHeader(header)) Fix(uint32_t(p - buf) | 0x80000000u);
    p[0] = kType3 | (count << 8);
    p[1] = header;
    p[2] = address;
    uint32_t k = 3;
    for (uint32_t v : words) p[k++] = __builtin_bswap32(v);
    cur = p + k;
  }
  // A type-3 packet whose payload is two values and then `count` words copied as they are from guest memory (already
  // in guest byte order: the immediate load's microcode).
  void Type3Guest(uint32_t header, uint32_t address, uint32_t w0, uint32_t w1, const uint8_t* guest, uint32_t count) {
    if (!Room(5 + count)) return;
    uint32_t* const p = cur;
    p[0] = kType3 | ((count + 2) << 8);
    p[1] = header;
    p[2] = address;
    p[3] = __builtin_bswap32(w0);
    p[4] = __builtin_bswap32(w1);
    std::memcpy(p + 5, guest, size_t(count) * 4);
    cur = p + 5 + count;
  }
  // The draw's pairing record [tag][packet type r5 r6 r7 vs ps packet -1 device] (words 1 and 8 are filled by
  // SetPacketAddresses where the packet address is not known yet).
  void Pair(uint32_t packet, uint32_t type, uint32_t r5, uint32_t r6, uint32_t r7, uint32_t vs, uint32_t ps,
            uint32_t d) {
    if (!Room(11)) return;
    uint32_t* const p = cur;
    Fix(uint32_t(p - buf));
    p[0] = kPair | (10u << 8);
    p[1] = packet;
    p[2] = type;
    p[3] = r5;
    p[4] = r6;
    p[5] = r7;
    p[6] = vs;
    p[7] = ps;
    p[8] = packet;
    p[9] = uint32_t(-1);
    p[10] = d;
    cur = p + 11;
  }
};

// The physical address Direct3D writes in a packet for a guest address (rlwinm 12,20,31; addi 0x200; rlwinm 0,19,19;
// clrlwi 3; add): the 0xE0000000 range is 4 KB further.
constexpr uint32_t Physical(uint32_t a) {
  return Rlwinm(Rlwinm(a, 12, 20, 31) + 0x200u, 0, 19, 19) + (a & 0x1FFFFFFFu);
}

// Why a call runs the original instead (checked before anything is changed; counted per reason in the [api] line).
enum Fallback : uint32_t { kFbNone = 0, kFbFlags, kFbAssert, kFbShader, kFbOps, kFbCount };
inline const char* const kFallbackNames[kFbCount] = {"none", "device flags", "assertion path", "shader size", "ops overflow"};

constexpr uint32_t kScratchSize = 0x11000;  // [0, 64 KB) a microcode copy, then the relink word
inline uint32_t g_scratch = 0;
inline uint32_t Scratch() {
  if (!g_scratch) g_scratch = rex::system::kernel_state()->memory()->SystemHeapAlloc(kScratchSize, 0x1000);
  return g_scratch;
}

// ---------------------------------------------------------------------------------------------------------------------
// Per-thread state, reached once per call (Thread()): on the Switch every thread_local access is a call of
// __aarch64_read_tp (one sat inside the inline shader copy's word loop in the first version).
struct InlineAlloc {  // verify mode: an inline allocation of the current call (address, words, contents written)
  uint32_t address, words;
  bool written;
};
struct ThreadData {
  uint32_t* ops = nullptr;     // the call's ops, kOpsCapacity words
  bool in_original = false;    // verify mode: the original runs (nested native hooks pass through)
  bool fence_snapshot = false;  // verify mode: the completed fence of this call (CompletedFence)
  uint32_t fence_value = 0;
  uint32_t shadow_bump = 0;    // verify mode: the shadow run's inline allocations
  std::vector<InlineAlloc> native_allocs, shadow_allocs;
  uint32_t guest_depth = 0;    // GuestCall contexts in use
  std::vector<std::unique_ptr<PPCContext>> guest;
  uint32_t finished_vs = 0, finished_ps = 0;  // NoteObjects: the last shader pair found finished (stays finished)
};
inline thread_local ThreadData* t_thread = nullptr;  // constant-initialized pointer: no TLS wrapper function
inline ThreadData& Thread() {
  ThreadData* t = t_thread;
  if (!t) [[unlikely]] {
    t = new ThreadData();
    t->ops = new uint32_t[kOpsCapacity];
    t_thread = t;
  }
  return *t;
}

// A call of a guest function from native code (the pure helpers, Direct3D's inline-data allocator and space function)
// on a context of this thread's own: the caller's stack pointer, r2 / r13 and floating-point state are copied in, the
// arguments are set by the caller. The first version copied the caller's whole context (2.7 KB, most of it the 128
// vector registers) for every call, which cost more than many of the helpers themselves; the recompiled callees read
// nothing else of it (their other registers are C++ locals, or written before they are read). A call made while
// another one runs on this thread (the allocator submitting, say) gets a context of its own.
class GuestCall {
 public:
  GuestCall(ThreadData& t, const PPCContext& caller) : t_(t) {
    if (t.guest_depth == t.guest.size()) t.guest.push_back(std::make_unique<PPCContext>());
    c_ = t.guest[t.guest_depth++].get();
    c_->r1 = caller.r1;
    c_->r2 = caller.r2;
    c_->r13 = caller.r13;
    c_->fpscr = caller.fpscr;
    c_->msr = caller.msr;
    c_->vscr_sat = caller.vscr_sat;
  }
  ~GuestCall() { --t_.guest_depth; }
  GuestCall(const GuestCall&) = delete;
  GuestCall& operator=(const GuestCall&) = delete;
  PPCContext& operator*() { return *c_; }
  PPCContext* operator->() { return c_; }

 private:
  ThreadData& t_;
  PPCContext* c_;
};

// Calls a recompiled pure helper with arguments (r3..r7, and v1 from 16 guest bytes when given).
inline void CallHelper(ThreadData& t, void (*fn)(PPCContext&, uint8_t*), const PPCContext& caller, uint8_t* base,
                       std::initializer_list<uint32_t> args, const uint8_t* v1_guest = nullptr) {
  GuestCall g(t, caller);
  PPCContext& c = *g;
  // v1 as the recompiled code holds a vector loaded from guest memory: bytes reversed (lvx's shuffle).
  if (v1_guest)
    for (int i = 0; i < 16; ++i) c.v1.u8[i] = v1_guest[15 - i];
  uint32_t i = 0;
  for (uint32_t a : args) {
    switch (i++) {
      case 0: c.r3.u64 = a; break;
      case 1: c.r4.u64 = a; break;
      case 2: c.r5.u64 = a; break;
      case 3: c.r6.u64 = a; break;
      case 4: c.r7.u64 = a; break;
      default: break;
    }
  }
  fn(c, base);
}

// A call of a guest function with up to eight register arguments (r3..r10) and stack arguments (the words a caller
// stores at r1+84, r1+92, ... before its `bl`), on a frame of its own below the caller's stack pointer: the native
// code stands where the original's frame would be, so that space is free.
inline void CallGuest(ThreadData& t, void (*fn)(PPCContext&, uint8_t*), const PPCContext& caller, uint8_t* base,
                      std::initializer_list<uint32_t> regs,
                      std::initializer_list<std::pair<uint32_t, uint32_t>> stack = {}) {
  GuestCall g(t, caller);
  PPCContext& c = *g;
  c.r1.u64 = (caller.r1.u32 - 0x400u) & ~0xFu;
  for (const auto& [offset, value] : stack)
    rex::memory::store_and_swap<uint32_t>(HostOf(base, c.r1.u32 + offset), value);
  PPCRegister* const r[8] = {&c.r3, &c.r4, &c.r5, &c.r6, &c.r7, &c.r8, &c.r9, &c.r10};
  uint32_t i = 0;
  for (uint32_t a : regs)
    if (i < 8) r[i++]->u64 = a;
  fn(c, base);
}

// One call's context for the native code.
template <class M>
struct Ctx {
  M& m;
  Ops& ops;
  uint8_t* base;
  const PPCContext& ppc;
  uint32_t d;        // the device
  uint32_t scratch;  // guest scratch for the pure helpers
  ThreadData& t;
};

// ---------------------------------------------------------------------------------------------------------------------
// FlushState pieces.

// sub_82555CC8 / sub_82555930 / sub_82555B70 / sub_825559F0's runs: one register run per run of dirty bits (bit 63
// first), `per_bit` registers a bit, from the shadow. (The fetch flush's 0x5000-0x5002 end marker is not sent: no
// renderer code reads those registers.)
template <class M>
inline void FlushRuns(Ctx<M>& c, uint64_t mask, uint32_t reg, uint32_t shadow, uint32_t per_bit) {
  for (uint64_t r = mask; r;) {
    const int lz = std::countl_zero(r);
    r <<= lz;
    reg += uint32_t(lz) * per_bit;
    shadow += uint32_t(lz) * per_bit * 4;
    const int run = std::countl_one(r);
    const uint32_t count = uint32_t(run) * per_bit;
    c.ops.Run(c.m, reg, count, shadow);
    reg += count;
    shadow += 4 * count;
    r = run == 64 ? 0 : r << run;
  }
}

// sub_825553A8 (group bit 0x20000): recomputes the low bits of register 0x2203 from the depth state, the pixel shader
// and the surface; may send event 15 (flag 0x04 of +11069). Returns mask | 0x100.
template <class M>
inline uint64_t Sub825553A8(Ctx<M>& c, uint64_t mask) {
  M& m = c.m;
  const uint32_t d = c.d;
  const uint32_t ps = m.L32(d + dev::kPixelShader);
  uint32_t r8 = 0;
  if (ps) r8 = Rlwinm(m.L32(ps + m.L32(ps + 64) + 48), 5, 31, 31);
  const uint32_t r10 = m.L32(d + dev::kShadow2200);
  const uint32_t r7 = Rlwinm(r10, 31, 1, 31);
  const uint32_t r6 = m.L32(d + dev::kReg2203);
  uint32_t r11 = Rlwinm(m.L8(d + dev::kFlags70), 27, 5, 31) & r7;
  uint32_t r9 = Rlwinm(m.L32(d + dev::kSurfaceFlags), 12, 20, 31);
  r11 = Rlwimi(r11, r6, 0, 0, 30);
  m.S32(d + dev::kReg2203, r11);
  if (!(r9 & 2)) {
    r11 = r9 & 1;
  } else {
    const uint32_t func = Rlwinm(r10, 28, 29, 31);
    uint32_t ok;
    if (!(r11 & 0x10)) ok = (func == 1 || func == 3 || func == 2) ? 1 : 0;
    else ok = (func == 4 || func == 6 || func == 2) ? 1 : 0;
    r11 = ((r11 & ~r8) & r7 & ok) & 1;
    if (r10 & 1) {
      r11 &= IsZero(r10 & Mask32(18, 20)) & IsZero(r10 & Mask32(12, 14));
      if (r10 & 0x80) r11 &= IsZero(r10 & Mask32(0, 2)) & IsZero(r10 & Mask32(6, 8));
    }
  }
  r9 = Rlwimi(m.L32(d + dev::kReg2203), r11, 1, 30, 30);
  m.S32(d + dev::kReg2203, r9);
  const uint8_t b69 = m.L8(d + dev::kFlags69);
  if (r9 & 2) {
    if ((b69 & 4) && (m.L32(d + dev::kDebugObject) == 0 || m.L32(d + dev::kDebugMode) == 1)) {
      m.S8(d + dev::kFlags69, uint8_t(b69 & 251));
      c.ops.Type3(0xC0004600u, 0, {15});  // EVENT_WRITE 15
    }
  } else if ((r7 & 1) && (r10 & 4)) {
    const uint32_t func = Rlwinm(r10, 28, 29, 31);
    bool set;
    if (func == 7 || func == 5) set = true;
    else if (!(r9 & 0x10)) set = func == 4 || func == 6;
    else set = func == 1 || func == 3;
    if (set) m.S8(d + dev::kFlags69, uint8_t(m.L8(d + dev::kFlags69) | 4));
  }
  return mask | 256;
}

// sub_825555A0 (when the group mask meets device+40): re-sends register 0x2203 (bit 0x100), and 0x2000 / 0x2104 when
// their bits are set, with the bin mask/select packets Direct3D wraps around the latter. (Its tiling-only part, flag
// 0x20 of +11071, is excluded by the caller's check.)
template <class M>
inline uint64_t Sub825555A0(Ctx<M>& c, uint64_t r26, uint32_t r28) {
  M& m = c.m;
  const uint32_t d = c.d;
  int32_t r23 = -1, r24 = -1, r22 = -1;
  if (r26 & 0x100) {
    uint32_t r10 = r28;
    if (!(m.L32(d + dev::kSurfaceFlags) & Mask32(12, 14))) r10 = Rlwinm(r28, 0, 0, 30);
    c.ops.Reg(0x2203, r10);
    if (r10 != r28 && (m.L8(d + dev::kFlags68) & 0x40)) r23 = int32_t(r28);
    r26 &= ~uint64_t(257 - 1);  // li r12,-257; and: clears 0x100
  }
  const uint64_t r10 = m.L64(d + dev::kDirtyRb) & r26;
  if (r10 & (1ull << 57)) {
    const uint32_t v = m.L32(d + dev::kShadow2000);
    c.ops.Reg(0x2000, v);
    r26 &= ~(1ull << 57);
    r24 = int32_t(Rlwimi(m.L32(d + dev::kReg2000Base), v, 0, 0, 17));
  }
  if (r10 & (1ull << 37)) {
    c.ops.Reg(0x2104, m.L32(d + dev::kReg2104));
    r22 = 0;
    r26 &= ~(1ull << 37);
  }
  if ((r22 & r24 & r23) != -1) {
    c.ops.Type3(0xC0006100u, 0, {0});            // SET_BIN_SELECT_LO? (0x61) 0
    c.ops.Type3(0xC0006000u, 0, {0x55555555u});  // SET_BIN_MASK_LO (0x60)
    if (r23 != -1) c.ops.Type3(0xC0015501u, 0, {0x2203, uint32_t(r23)});
    if (r24 != -1) c.ops.Type3(0xC0015501u, 0, {0x2000, uint32_t(r24)});
    if (r22 != -1) c.ops.Type3(0xC0015501u, 0, {0x2104, uint32_t(r22)});
    c.ops.Type3(0xC0006000u, 0, {m.L32(d + dev::kBinMaskSave)});
    c.ops.Type3(0xC0006100u, 0, {m.L32(d + dev::kBinSelectSave)});
  }
  return r26;
}

// sub_825559F0 (misc bits 49-54): register 0x2007, the texture cache coherency request (COHER_STATUS_HOST 0x10000,
// COHER_SIZE/BASE), a WAIT_REG_MEM on it, then 4 registers per bit from 0x2388.
template <class M>
inline void Sub825559F0(Ctx<M>& c, uint64_t mask) {
  c.ops.Reg(0x2007, c.m.L32(c.d + dev::kReg2007));
  c.ops.Reg(0x0A31, 0x00010000u);
  c.ops.Reg(0x0A2F, 0);
  c.ops.Reg(0x0A30, 0x1000);
  c.ops.Type3(0xC0043C00u, 0, {3, 0x0A31, 0, 0x80000000u, 8});
  FlushRuns(c, mask, 0x2388, c.d + dev::kShadow2388, 4);
}

// sub_8254F6A8 (misc bits 62-63 with flag 0x80/0x40 of +11072): WAIT_UNTIL, the RB/TC flushes, flag updates.
template <class M>
inline void Sub8254F6A8(Ctx<M>& c) {
  M& m = c.m;
  const uint32_t d = c.d;
  c.ops.Reg(0x05C8, 0x00020000u);
  if (m.L8(d + dev::kFlags72) & 0x40) {
    c.ops.Reg(0x0E00, 1);
    c.ops.Reg(0x0E40, 1);
  }
  m.S8(d + dev::kFlags72, uint8_t(m.L8(d + dev::kFlags72) & 0x3F));
  m.S8(d + dev::kFlags71, uint8_t(Rlwinm(m.L8(d + dev::kFlags71), 0, 0, 29)));
  m.S64(d + dev::kDirtyMisc, m.L64(d + dev::kDirtyMisc) & 0x3FFFFFFFFFFFFFFFull);
}

// Direct3D's partial flush, inlined by Clear and the resolve before their own packets: sub_825555A0 when the group
// mask meets device+40, the register groups 0x2200 / 0x2180 / 0x2000 / 0x2100 by group bits, 0x2280 by the fetch mask,
// 0x2300 by the misc mask, each clearing its bits.
template <class M>
inline void PartialFlush(Ctx<M>& c) {
  M& m = c.m;
  const uint32_t d = c.d;
  if (m.L64(d + dev::kDirtyGroups)) {
    if (m.L64(d + dev::kDirtyRb) & m.L64(d + dev::kDirtyGroups))
      m.S64(d + dev::kDirtyGroups, Sub825555A0(c, m.L64(d + dev::kDirtyGroups), m.L32(d + dev::kReg2203)));
    uint64_t g = m.L64(d + dev::kDirtyGroups);
    if (g & 0xFFF) {
      FlushRuns(c, Rldicr(g, 52, 11), 0x2200, d + dev::kShadow2200, 1);
      m.S64(d + dev::kDirtyGroups, Rldicr(m.L64(d + dev::kDirtyGroups), 0, 51));
    }
    g = m.L64(d + dev::kDirtyGroups);
    if (Rlwinm(uint32_t(g), 0, 15, 19)) {
      FlushRuns(c, Rldicr(g, 47, 4), 0x2180, d + dev::kShadow2180, 1);
      m.S64(d + dev::kDirtyGroups, m.L64(d + dev::kDirtyGroups) & 0xFFFFFFFFFFFE0FFFull);
    }
    g = m.L64(d + dev::kDirtyGroups);
    if (g & Rldicr(0xFFFF, 42, 21)) {
      FlushRuns(c, Rldicr(g, 6, 15), 0x2000, d + dev::kShadow2000, 1);
      m.S64(d + dev::kDirtyGroups, m.L64(d + dev::kDirtyGroups) & std::rotl(0xFFFFFFFFFFFF0000ull, 42));
    }
    g = m.L64(d + dev::kDirtyGroups);
    if (g & 0x000003FFFFE00000ull) {
      FlushRuns(c, Rldicr(g, 22, 20), 0x2100, d + dev::kShadow2100, 1);
      m.S64(d + dev::kDirtyGroups, m.L64(d + dev::kDirtyGroups) & std::rotl(0xFFFFFFFFFFE00000ull, 21));
    }
  }
  {
    const uint64_t f = m.L64(d + dev::kDirtyFetch);
    if (f && (f & Rldicr(0x1FFFFF, 34, 29))) {
      FlushRuns(c, Rldicr(f, 9, 20), 0x2280, d + dev::kShadow2280, 1);
      m.S64(d + dev::kDirtyFetch, m.L64(d + dev::kDirtyFetch) & std::rotl(0xFFFFFFFFFFE00000ull, 34));
    }
  }
  {
    const uint64_t x = m.L64(d + dev::kDirtyMisc);
    if (x && (x & 0x3FFFFFFFFFull)) {
      FlushRuns(c, Rldicr(x, 26, 37), 0x2300, d + dev::kShadow2300, 1);
      m.S64(d + dev::kDirtyMisc, Rldicr(m.L64(d + dev::kDirtyMisc), 0, 25));
    }
  }
}

// The window offset and scissor (registers 0x2080-0x2082) as sub_8254FCE8 writes them.
inline void WindowScissorOps(Ops& ops, uint32_t x0, uint32_t y0, uint32_t x1, uint32_t y1) {
  ops.Regs(0x2080, {0, Rlwinm(y0, 16, 2, 15) | Rlwinm(x0, 0, 18, 31), Rlwinm(y1, 16, 2, 15) | Rlwinm(x1, 0, 18, 31)});
}

// sub_82556870: a shader program's literal constants, one LOAD_ALU_CONSTANT per table entry.
template <class M>
inline void LiteralConstants(Ctx<M>& c, uint32_t prog, uint32_t base_address) {
  M& m = c.m;
  const uint32_t off = m.L32(prog + 20);
  if (!off) return;
  const uint32_t table = prog + off;
  uint32_t e = table + 20;
  const uint32_t end = m.L32(table + 16) + e;
  while (e < end) {
    const uint32_t index = m.L16(e), count = m.L16(e + 2);
    if (!count) break;
    const uint32_t address = Physical(m.L32(e + 4) + base_address);
    c.ops.Type3(0xC0022F00u, 0, {address, Rlwinm(index, 2, 0, 29), count});
    e += 8;
  }
}

// Vertex shader bindings: per binding (416 bytes from the object) +40 the declaration's id, +48/+56 the strides it was
// patched for, +64 its fence; its program descriptor (vs + vs[(i + 112) * 8]) has +872 the microcode offset from the
// object's microcode base (vs+32) and +876 the size in bytes.
inline uint32_t BindingState(uint32_t vs, uint32_t idx) { return vs + idx * 416; }
template <class M>
inline uint32_t BindingProgram(const M& m, uint32_t vs, uint32_t idx) { return vs + m.L32(vs + Rlwinm(idx + 112, 3, 0, 28)); }

// The GPU's last completed fence (the ring thread writes it): verify mode reads a snapshot taken for the call, which
// the original's device copy points at too, so both sides decide on the same value.
template <class M>
inline uint32_t CompletedFence(const ThreadData& t, const M& m, uint32_t d) {
  return t.fence_snapshot ? t.fence_value : m.L32(m.L32(d + dev::kFenceCompleted));
}

// sub_82556920: is binding `idx` patched for this declaration and these strides? If not and the GPU is done with it,
// it is patched in place (sub_82555E18 on its microcode in guest memory) and records what it was patched for. False:
// the GPU may still read it (the caller sends an inline patched copy instead).
template <class M>
inline bool BindingMatches(Ctx<M>& c, uint32_t vs, uint32_t decl, uint32_t idx) {
  M& m = c.m;
  const uint32_t d = c.d;
  const uint32_t st = BindingState(vs, idx);
  if (m.L32(st + 40) == m.L32(decl + 48)) {
    const uint64_t x = ((m.L64(st + 48) ^ m.L64(d + dev::kStreamStrides)) & m.L64(decl + 32)) |
                       ((m.L64(st + 56) ^ m.L64(d + dev::kStreamStrides + 8)) & m.L64(decl + 40));
    if (x == 0) {
      m.S64(d + dev::kBoundStrides, m.L64(d + dev::kStreamStrides));
      m.S64(d + dev::kBoundStrides + 8, m.L64(d + dev::kStreamStrides + 8));
      return true;
    }
  }
  const uint32_t fence = m.L32(st + 64);
  if (fence) {
    const uint32_t next = m.L32(d + dev::kFenceNext);
    const uint32_t completed = CompletedFence(c.t, m, d);
    if (next - fence < next - completed) return false;
  }
  const uint32_t prog = BindingProgram(m, vs, idx);
  const uint32_t ucode = m.L32(prog + 872) + m.L32(vs + 32);
  CallHelper(c.t, &sub_82555E18, c.ppc, c.base, {vs, ucode, decl, d + dev::kStreamStrides, idx});
  if (m.L32(decl + 48) == 0) {
    uint32_t id;
    do {
      id = m.AtomicIncrement(kGlobalDeclIds);
    } while (id + 1 < 2);  // 0 and -1 are skipped
    m.S32(decl + 48, id);
  }
  m.S32(st + 40, m.L32(decl + 48));
  m.S64(st + 48, m.L64(d + dev::kStreamStrides));
  m.S64(st + 56, m.L64(d + dev::kStreamStrides + 8));
  m.S64(d + dev::kBoundStrides, m.L64(d + dev::kStreamStrides));
  m.S64(d + dev::kBoundStrides + 8, m.L64(d + dev::kStreamStrides + 8));
  return true;
}

// sub_82556718: binding `idx` sent as an inline copy (IM_LOAD_IMMEDIATE), patched for the declaration, relinked to the
// pixel shader's inputs first when `relink` (which may change *loc80, the SQ_PROGRAM_CNTL word).
template <class M>
inline void BindInline(Ctx<M>& c, uint32_t relink, uint32_t vs, uint32_t& loc80, uint32_t decl, uint32_t vprog,
                uint32_t pprog, uint32_t idx) {
  M& m = c.m;
  const uint32_t d = c.d;
  const uint32_t prog = BindingProgram(m, vs, idx);
  const uint32_t size = m.L32(prog + 876);
  const uint32_t words = Rlwinm(size, 30, 2, 31);
  const uint32_t s = c.scratch;
  const uint32_t src = Physical(m.L32(prog + 872) + m.L32(vs + 32)) + 0xC0000000u;
  CopyGuest(c.base, s, src, size);  // the microcode, as it is (guest byte order)
  if (relink) {
    const uint32_t word = s + 0x10000;
    rex::memory::store_and_swap<uint32_t>(HostOf(c.base, word), loc80);
    CallHelper(c.t, &sub_825565C8, c.ppc, c.base, {vprog, s, word, pprog});
    loc80 = rex::memory::load_and_swap<uint32_t>(HostOf(c.base, word));
  }
  m.S8(d + dev::kFlags70, uint8_t(Rlwimi(m.L8(d + dev::kFlags70), relink != 0 ? 1u : 0u, 7, 24, 24)));
  CallHelper(c.t, &sub_82555E18, c.ppc, c.base, {vs, s, decl, d + dev::kStreamStrides, idx});
  // INVALIDATE_STATE 0x100 (no effect on the ring), then the immediate load: [type 0 = vertex][size][microcode], the
  // patched copy straight from the scratch area (guest byte order, as the payload is kept).
  c.ops.Type3Guest(Rlwimi(0xC0002B00u, words + 1, 16, 2, 15), 0, 0, Rlwinm(words, 0, 18, 31), HostOf(c.base, s),
                   words);
  m.S64(d + dev::kBoundStrides + 8, m.L64(d + dev::kStreamStrides + 8));
  m.S64(d + dev::kBoundStrides, m.L64(d + dev::kStreamStrides));
}

// sub_82556A60: the draw-time shader load (group bits 17-20). Returns the updated group mask. Its tiling-only parts
// (flag 0x40/0x80/0x01 of +11068, flags 0x02/0x08 of +11071) are excluded by the caller's check.
template <class M>
inline uint64_t ShaderLoad(Ctx<M>& c, uint64_t r4) {
  M& m = c.m;
  const uint32_t d = c.d;
  uint64_t r17 = r4;
  int32_t r15 = -1;
  uint32_t r16 = 0;
  if (Rlwinm(uint32_t(r4), 0, 14, 14)) r17 = Sub825553A8(c, r4);
  const uint32_t vs = m.L32(d + dev::kVertexShader);
  const uint32_t decl = m.L32(d + dev::kVertexDecl);
  const uint32_t ps = m.L32(d + dev::kPixelShader);
  if (!vs) return r17;
  uint32_t r25 = vs + m.L32(vs + 896) + 872;  // binding 0's program (+872)
  uint32_t r23 = 0;
  uint32_t r27 = 0, r18 = 0, r19 = 0, r28 = 0;
  if (!ps) {
    if (m.L32(vs + 872) & 0x20) {
      r23 = 1;
      r25 = vs + m.L32(vs + 904) + 872;
    }
    const uint32_t mc = m.L32(d + dev::kReg2208);
    if ((mc & 7) != 5) {
      m.S32(d + dev::kReg2208, Rlwimi(mc, 5, 0, 29, 31));
      r17 |= 0x80008;
    }
    r28 = (m.L32(r25 + 20) & 0x1F) != 0 ? 1 : 0;
  } else {
    const uint32_t pp = ps + m.L32(ps + 64);
    r27 = pp + 40;
    r18 = m.L32(pp + 48);
    r16 = m.L32(pp + 52);
    r19 = r18;
    const uint32_t mrt = Rlwinm(r18, 0, 1, 3) > 0x10000000u ? 1u : 0u;
    const uint8_t b70 = m.L8(d + dev::kFlags70), b71 = m.L8(d + dev::kFlags71);
    m.S8(d + dev::kFlags70, uint8_t(Rlwimi(b70, Rlwinm(mrt, 3, 21, 28) | b70, 0, 28, 28)));
    m.S8(d + dev::kFlags71, uint8_t(Rlwimi(b71, Rlwinm(mrt, 1, 23, 30) | b71, 0, 30, 30)));
    if (Rlwinm(uint32_t(r17), 0, 11, 11)) {  // the pixel shader changed
      LiteralConstants(c, ps + 40, m.L32(ps + 24));
      const uint32_t ucode = m.L32(pp + 40) + m.L32(ps + 24);
      const uint32_t address = (Rlwinm(Rlwinm(ucode, 12, 20, 31) + 0x200u, 0, 19, 19) + Rlwinm(ucode, 0, 3, 30)) | 1u;
      c.ops.Type3(0xC0012700u, 0, {address, Rlwinm(m.L32(pp + 44), 30, 2, 31)});
      r17 |= 16384;
      m.S32(d + dev::kReg2182, m.L32(r27 + 24));
      const uint32_t mc = m.L32(d + dev::kReg2208);
      if ((mc & 7) != 4) {
        m.S32(d + dev::kReg2208, Rlwimi(mc, 1, 2, 29, 31));
        r17 |= 0x80008;
      }
    }
    // The interpolator layout both programs use, under Direct3D's mask (initialized on its first use).
    uint32_t imask;
    const uint32_t flag = m.L32(kGlobalInterpFlag);
    if (!(flag & 1)) {
      m.S32(kGlobalInterpFlag, flag | 1);
      imask = 0x000603FFu;
      m.S32(kGlobalInterpMask, imask);
    } else {
      imask = m.L32(kGlobalInterpMask);
    }
    r28 = (m.L32(r27 + 20) & imask) != (m.L32(r25 + 20) & imask) ? 1 : 0;
  }
  // loc_82556D5C (the tiling block that follows is excluded by the caller's check)
  uint32_t loc80 = m.L32(r25 + 8);
  const uint32_t r26 = m.L32(d + dev::kReg2208);
  const uint32_t r24 = m.L32(r25 + 12);
  const uint32_t r29 = r26;
  // loc_82556E28
  {
    const uint32_t mrt = Rlwinm(r19, 0, 1, 3) > 0x10000000u ? 1u : 0u;
    const uint8_t b71 = m.L8(d + dev::kFlags71);
    m.S8(d + dev::kFlags71, uint8_t(Rlwimi(mrt | b71, b71, 0, 24, 30)));
  }
  const uint8_t b70 = m.L8(d + dev::kFlags70);
  if ((b70 & 0x80) || r28) {
    // loc_82556F84: an inline copy (relinked when the interpolators differ).
    if (Rlwinm(uint32_t(r17), 0, 11, 12)) {
      m.S8(d + dev::kFlags70, uint8_t(b70 & 191));
      LiteralConstants(c, vs + 872, m.L32(vs + 32));
      BindInline(c, r28, vs, loc80, decl, r25, r27, r23);
    }
  } else if (Rlwinm(uint32_t(r17), 0, 12, 12)) {  // the vertex shader changed
    LiteralConstants(c, vs + 872, m.L32(vs + 32));
    const uint8_t f = uint8_t(m.L8(d + dev::kFlags70) & 191);
    m.S8(d + dev::kFlags70, f);
    if (decl == 0 || (m.L32(vs + 872) & 0x40)) {
      // loc_82556F48: the binding as it is.
      const uint32_t st = BindingState(vs, r23);
      r15 = int32_t(r23);
      m.S64(d + dev::kBoundStrides, m.L64(st + 48));
      m.S64(d + dev::kBoundStrides + 8, m.L64(st + 56));
      if (((r29 ^ r26) & 7) && (m.L32(vs + 872) & 0x20)) m.S8(d + dev::kFlags70, uint8_t(f | 64));
    } else {
      if (BindingMatches(c, vs, decl, r23)) {
        r15 = int32_t(r23);
      } else {
        BindInline(c, 0, vs, loc80, decl, r25, r27, r23);
      }
      if (((r29 ^ r26) & 7) && (m.L32(vs + 872) & 0x20) && BindingMatches(c, vs, decl, 1))
        m.S8(d + dev::kFlags70, uint8_t(m.L8(d + dev::kFlags70) | 64));
    }
  }
  // loc_82556FCC
  if (Rlwinm(uint32_t(r17), 0, 11, 12)) {
    m.S32(d + dev::kReg2181, r24 | r16);
    r17 |= 0x10000;
    r17 |= 32768;
    m.S32(d + dev::kReg2180, loc80 | r18);
  }
  // loc_82557354 (no tiling): the vertex binding's load, and its fence.
  if (r15 != -1) {
    const uint32_t prog = BindingProgram(m, vs, uint32_t(r15));
    const uint32_t ucode = m.L32(prog + 872) + m.L32(vs + 32);
    const uint32_t address = Rlwinm(Rlwinm(ucode, 12, 20, 31) + 0x200u, 0, 19, 19) + (ucode & 0x1FFFFFFFu);
    c.ops.Type3(0xC0012700u, 0, {address, Rlwinm(m.L32(prog + 876), 30, 2, 31)});
    m.S32(BindingState(vs, uint32_t(r15)) + 64, m.L32(d + dev::kFenceNext));
  }
  return r17;
}

// Direct3D's FlushState, as DrawVertices inlines it.
template <class M>
inline void FlushState(Ctx<M>& c) {
  M& m = c.m;
  const uint32_t d = c.d;
  const uint64_t r4 = m.L64(d + dev::kDirtyVsConsts);
  const uint64_t r27 = m.L64(d + dev::kDirtyMisc);
  const uint64_t r28 = m.L64(d + dev::kDirtyFetch);
  uint64_t r30 = m.L64(d + dev::kDirtyGroups);
  const uint64_t r29 = m.L64(d + dev::kDirtyPsConsts);
  if (r4) {
    FlushRuns(c, r4, 0x4000, d + dev::kShadowVsConsts, 16);
    m.S64(d + dev::kDirtyVsConsts, 0);
  }
  if (r29) {
    FlushRuns(c, r29, 0x4400, d + dev::kShadowPsConsts, 16);
    m.S64(d + dev::kDirtyPsConsts, 0);
  }
  if (r30) {
    if (Rlwinm(uint32_t(r30), 0, 11, 14)) r30 = ShaderLoad(c, r30);
    if (m.L64(d + dev::kDirtyRb) & r30) r30 = Sub825555A0(c, r30, m.L32(d + dev::kReg2203));
    if (r30 & 0xFFF) FlushRuns(c, Rldicr(r30, 52, 11), 0x2200, d + dev::kShadow2200, 1);
    if (Rlwinm(uint32_t(r30), 0, 15, 19)) FlushRuns(c, Rldicr(r30, 47, 4), 0x2180, d + dev::kShadow2180, 1);
    if (r30 & Rldicr(0xFFFF, 42, 21)) FlushRuns(c, Rldicr(r30, 6, 15), 0x2000, d + dev::kShadow2000, 1);
    if (r30 & (0xFFFFFFFFFFE00000ull & 0x000003FFFFFFFFFFull))
      FlushRuns(c, Rldicr(r30, 22, 20), 0x2100, d + dev::kShadow2100, 1);
    m.S64(d + dev::kDirtyGroups, 0);
  }
  if (r28) {
    if (r28 & 0xFFFFFFFFull) FlushRuns(c, Rldicr(r28, 32, 31), 0x4800, d + dev::kShadowFetch, 6);
    if (r28 & Rldicr(0x1FFFFF, 34, 29)) FlushRuns(c, Rldicr(r28, 9, 20), 0x2280, d + dev::kShadow2280, 1);
    m.S64(d + dev::kDirtyFetch, 0);
  }
  if (r27) {
    if (r27 & Rldicr(255, 38, 25)) FlushRuns(c, Rldicr(r27, 18, 7), 0x2380, d + dev::kShadow2380, 1);
    if (r27 & Rldicr(63, 49, 14)) Sub825559F0(c, Rldicr(r27, 9, 5));
    if (r27 & Rldicr(1, 56, 63)) FlushRuns(c, 0xFFFFFFFFFF000000ull, 0x4900, d + dev::kShadowBoolLoop, 1);
    if (r27 & 0x3FFFFFFFFFull) FlushRuns(c, Rldicr(r27, 26, 37), 0x2300, d + dev::kShadow2300, 1);
    if (Rldicr(r27, 0, 1)) {
      const uint8_t b72 = m.L8(d + dev::kFlags72);
      if (Rlwinm(b72, 0, 0, 24) || Rlwinm(b72, 0, 25, 25)) Sub8254F6A8(c);
    }
    m.S64(d + dev::kDirtyMisc, 0);
  }
}

// The draw packet's first payload word (Direct3D's per-draw viz/initiator extension).
template <class M>
inline uint32_t VizWord(const M& m, uint32_t d) {
  const uint32_t b = m.L8(d + dev::kVizIndex);
  uint32_t v = m.L8(d + dev::kVizTable + b);
  v = Rlwimi(v, b != 0 ? 1u : 0u, 8, 23, 23);
  return Rlwinm(v & 0x1FFu, 0, 26, 23);
}

// D3DDevice_DrawVertices(device, type, start, count): FlushState, then per chunk of at most 65535 vertices the start
// register (VGT_INDX_OFFSET, 0x2102), this call's pairing record and the predicated DRAW_INDX (auto-index).
template <class M>
inline void DrawVerticesOps(Ctx<M>& c, uint32_t type, uint32_t start, uint32_t count, uint32_t address) {
  FlushState(c);
  M& m = c.m;
  const uint32_t d = c.d;
  const uint32_t vs = m.L32(d + dev::kVertexShader), ps = m.L32(d + dev::kPixelShader);
  uint32_t r24 = count, r23 = start;
  for (uint32_t chunk = 0;; ++chunk) {
    uint32_t r29 = r24;
    if (r24 > 65535) {
      const uint32_t per = m.L32(kPrimitiveTable + Rlwinm(type, 3, 0, 28));
      r29 = per ? Rlwinm(0xFFFFu / per, 0, 0, 30) * per : r24;
    }
    c.ops.Reg(0x2102, r23);
    const uint32_t packet = address + chunk * 16;
    c.ops.Pair(packet, type, start, count, 0, vs, ps, d);
    c.ops.Type3(0xC0012201u, packet, {VizWord(m, d), Rlwinm(r29, 16, 0, 15) | (type & 0x3F) | 128});
    if (r24 == r29) break;
    const uint32_t r10 = r24 - r29;
    const uint32_t overlap = m.L32(kPrimitiveTable + 4 + Rlwinm(type, 3, 0, 28));
    r24 = overlap + r10;
    r23 = (r29 - overlap) + r23;
  }
}

// Every state in which the native draw does not follow Direct3D, checked before anything is changed.
template <class M>
Fallback Check(const M& m, uint32_t d, uint64_t extra_groups = 0) {
  if (m.L8(d + dev::kFlags68) != 0) return kFbFlags;                // tiling, binned draws, forced inline loads
  if (m.L8(d + dev::kFlags71) & (0x02 | 0x08 | 0x20)) return kFbFlags;
  const uint64_t groups = m.L64(d + dev::kDirtyGroups) | extra_groups;
  if (Rlwinm(uint32_t(groups), 0, 11, 14)) {
    const uint32_t vs = m.L32(d + dev::kVertexShader), ps = m.L32(d + dev::kPixelShader);
    if (vs) {
      if (m.L32(d + dev::kDebugObject) == 0 && ((m.L32(vs + 872) & 0x10) || (ps && (m.L32(ps + 40) & 0x10))))
        return kFbAssert;
      for (uint32_t i = 0; i < 2; ++i)
        if (m.L32(BindingProgram(m, vs, i) + 876) > 0x10000) return kFbShader;
    }
  }
  return kFbNone;
}

// ---------------------------------------------------------------------------------------------------------------------
// Verify mode: canonical forms and the comparison.

struct Item {
  uint32_t kind;  // 0 = registers, 1 = type-3 packet
  std::map<uint32_t, uint32_t> regs;
  std::vector<uint32_t> words;  // header (predicate kept), payload
  bool operator==(const Item& o) const { return kind == o.kind && regs == o.regs && words == o.words; }
};
using Canon = std::vector<Item>;

inline bool Effectful(uint32_t opcode) { return opcode != 0x10 && opcode != 0x3B; }  // NOP, INVALIDATE_STATE

inline void AddReg(Canon& c, uint32_t reg, uint32_t v) {
  if (reg >= 0x5000) return;  // the fetch flush's end marker
  if (c.empty() || c.back().kind != 0) c.push_back({0, {}, {}});
  c.back().regs[reg] = v;
}

inline void AddType3(Canon& c, uint32_t header, const uint32_t* payload_host, uint32_t count) {
  if (!Effectful((header >> 8) & 0x7F)) return;
  Item it{1, {}, {}};
  it.words.push_back(header);
  it.words.insert(it.words.end(), payload_host, payload_host + count);
  c.push_back(std::move(it));
}

// The original's packets in [first, last] (guest addresses of words).
inline void CanonPm4(Canon& c, uint8_t* base, uint32_t first, uint32_t last) {
  std::vector<uint32_t> payload;
  for (uint32_t p = first; p <= last;) {
    const uint32_t h = rex::memory::load_and_swap<uint32_t>(HostOf(base, p));
    const uint32_t type = h >> 30;
    if (type == 2 || h == 0) {
      p += 4;
      continue;
    }
    if (type == 0) {
      const uint32_t count = ((h >> 16) & 0x3FFF) + 1, reg = h & 0x7FFF;
      const bool single = (h >> 15) & 1;
      for (uint32_t i = 0; i < count; ++i)
        AddReg(c, single ? reg : reg + i, rex::memory::load_and_swap<uint32_t>(HostOf(base, p + 4 + 4 * i)));
      p += 4 * (1 + count);
    } else if (type == 1) {
      AddReg(c, h & 0x7FF, rex::memory::load_and_swap<uint32_t>(HostOf(base, p + 4)));
      AddReg(c, (h >> 11) & 0x7FF, rex::memory::load_and_swap<uint32_t>(HostOf(base, p + 8)));
      p += 12;
    } else {
      const uint32_t count = ((h >> 16) & 0x3FFF) + 1;
      payload.resize(count);
      for (uint32_t i = 0; i < count; ++i)
        payload[i] = rex::memory::load_and_swap<uint32_t>(HostOf(base, p + 4 + 4 * i));
      AddType3(c, h, payload.data(), count);
      p += 4 * (1 + count);
    }
  }
}

inline void CanonOps(Canon& c, const uint32_t* op, uint32_t n) {
  std::vector<uint32_t> payload;
  for (uint32_t i = 0; i < n;) {
    const uint32_t tag = op[i], count = tag >> 8;
    switch (tag & 0xFF) {
      case kRun:
        for (uint32_t k = 0; k < count; ++k) AddReg(c, op[i + 1] + k, op[i + 2 + k]);
        i += 2 + count;
        break;
      case kType3:
        payload.resize(count);
        for (uint32_t k = 0; k < count; ++k) payload[k] = __builtin_bswap32(op[i + 3 + k]);
        AddType3(c, op[i + 1], payload.data(), count);
        i += 3 + count;
        break;
      case kPair:
        i += 1 + count;
        break;
      default:
        i = n;
        break;
    }
  }
}

inline std::string Describe(const Canon& c) {
  std::string s;
  for (const Item& it : c) {
    if (s.size() > 3000) {
      s += " ...";
      break;
    }
    if (it.kind == 0) {
      s += " {";
      for (const auto& [r, v] : it.regs) s += fmt::format(" {:04X}={:08X}", r, v);
      s += " }";
    } else {
      s += " [";
      for (size_t k = 0; k < it.words.size() && k < 8; ++k) s += fmt::format(" {:08X}", it.words[k]);
      if (it.words.size() > 8) s += fmt::format(" ...{} words", it.words.size());
      s += " ]";
    }
  }
  return s;
}

// ---------------------------------------------------------------------------------------------------------------------
// State and statistics.
inline std::atomic<bool> g_failed{false};  // a native draw could not finish: native draws off for the rest of the run
inline std::atomic<uint64_t> g_native{0}, g_original{0}, g_fallbacks[kFbCount]{}, g_ops_words{0}, g_checked{0},
    g_differences{0}, g_window{0};
inline int g_logged = 0;

inline void Bump(std::atomic<uint64_t>& c, uint64_t n = 1) {
  c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed);
}

// The layer's own cost on the drawing thread, for the [api] line (the console's measure of this code without the
// profiler): every 16th native call is timed with the monotonic clock (on the Switch the system counter, no syscall);
// the timed calls' average times all native calls gives the time per interval. One clock read per timed call is in it.
inline std::atomic<uint64_t> g_timed_ns{0}, g_timed_calls{0}, g_calls{0};
inline uint32_t g_time_tick = 0;  // the drawing thread's call counter (a lost count only moves which call is timed)
inline uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch()).count());
}
template <class F>
inline bool Timed(F&& f) {
  if ((++g_time_tick & 15) != 0) [[likely]] {
    const bool done = f();
    if (done) Bump(g_calls);
    return done;
  }
  const uint64_t t0 = NowNs();
  const bool done = f();
  if (done) {
    Bump(g_timed_ns, NowNs() - t0);
    Bump(g_timed_calls);
    Bump(g_calls);
  }
  return done;
}

inline void LogDifference(const std::string& what, const std::string& detail) {
  sa2::enh::VerifyDifference();
  Bump(g_differences);
  if (g_logged++ < 16) REXLOG_ERROR("[api] DIFFERENCE: {}: {}", what, detail);
}

// The shader objects as the old draw path notes them on every draw (NoteDrawCall): the ring identifies loads by them.
// "Finished" is permanent (NoteShaderObject's sets are never erased; for a finished object it returns without effect),
// so the pair last found finished is remembered here: a draw with the same pair skips the two loads of the shared
// table, which the record stream keeps evicting from the cache (round 10 console: a miss on most draws).
inline void NoteObjects(ThreadData& t, uint8_t* base, uint32_t d) {
  const uint32_t vs = rex::memory::load_and_swap<uint32_t>(HostOf(base, d + dev::kVertexShader));
  const uint32_t ps = rex::memory::load_and_swap<uint32_t>(HostOf(base, d + dev::kPixelShader));
  if (vs == t.finished_vs && ps == t.finished_ps && vs) [[likely]] return;
  if (!me::native::ShaderObjectsFinished(vs, ps)) {
    me::native::NoteShaderObject(base, vs, true);
    me::native::NoteShaderObject(base, ps, false);
    return;
  }
  t.finished_vs = vs;
  t.finished_ps = ps;
}

}  // namespace detail

namespace detail {

// Ops of D3DDevice_BeginVertices(device, type, count, stride) (sub_82565758): the stream-0 stride byte is the call's
// while FlushState runs (the vertex shader is reloaded when its binding was patched for another stride), fetch slot 31
// is left out of the flush (the draw sets it to the inline vertices) and marked dirty again after; then the start
// register, slot 31's six registers and the draw. `data` is the inline vertices' guest address (Direct3D's inline-data
// allocator, called by the caller), `words` their size.
template <class M>
inline void BeginVerticesState(Ctx<M>& c, uint32_t stride) {
  M& m = c.m;
  const uint32_t d = c.d;
  const uint32_t r11 = m.L8(d + dev::kBoundStrides);
  const uint32_t r10 = Rlwinm(stride, 30, 24, 31);
  m.S8(d + dev::kStreamStrides, uint8_t(r10));
  if (r10 != r11 && m.L32(d + dev::kVertexShader) != 0)
    m.S64(d + dev::kDirtyGroups, m.L64(d + dev::kDirtyGroups) | 0x80000ull);
  m.S64(d + dev::kDirtyFetch, Rldicr(m.L64(d + dev::kDirtyFetch), 0, 62));
}

template <class M>
inline void BeginVerticesDraw(Ctx<M>& c, uint32_t type, uint32_t count, uint32_t stride, uint32_t data, uint32_t words,
                       uint32_t packet) {
  M& m = c.m;
  const uint32_t d = c.d;
  c.ops.Reg(0x2102, 0);
  c.ops.Regs(0x48BA, {m.L32(d + 1896), m.L32(d + 1900), m.L32(d + 1904), m.L32(d + 1908), Physical(data) | 3u,
                      Rlwimi(0x10000002u, words, 2, 6, 29)});
  const uint32_t vs = m.L32(d + dev::kVertexShader), ps = m.L32(d + dev::kPixelShader);
  c.ops.Pair(packet, type, count, stride, 0, vs, ps, d);
  c.ops.Type3(0xC0012201u, packet, {VizWord(m, d), Rlwinm(count, 16, 0, 15) | (type & 0x3F) | 128});
  m.S64(d + dev::kDirtyFetch, m.L64(d + dev::kDirtyFetch) | 1);
  m.S32(d + 13848, data);
  m.S32(d + 13856, words);
}

// Direct3D's inline-data allocator (transport: its segment of the command buffer; replaced by a native arena once no
// Direct3D code writes PM4). In verify mode the call's inline allocations are noted in order (ThreadData::
// native_allocs: CompareShadow maps them onto the shadow run's), with whether the call itself writes their contents.
inline uint32_t AllocInline(ThreadData& t, uint8_t* base, const PPCContext& ctx, uint32_t d, uint32_t words,
                            uint32_t align = 16, bool written = false) {
  GuestCall g(t, ctx);
  g->r3.u64 = d;
  g->r4.u64 = words;
  g->r5.u64 = align;
  __imp__sub_8254D9A8(*g, base);
  const uint32_t a = g->r3.u32;
  if (a && sa2::enh::VerifyMode()) t.native_allocs.push_back({a, words, written});
  return a;
}

// The space for a marker (Direct3D's own check: the space function may submit and switch segments): the write pointer.
inline uint32_t MarkerSpace(ThreadData& t, uint8_t* base, const PPCContext& ctx, uint32_t d) {
  uint32_t w = rex::memory::load_and_swap<uint32_t>(HostOf(base, d + dev::kWptr));
  if (w > rex::memory::load_and_swap<uint32_t>(HostOf(base, d + dev::kSoftLimit))) [[unlikely]] {
    GuestCall g(t, ctx);
    g->r3.u64 = d;
    __imp__sub_8254EBB8(*g, base);
    w = g->r3.u32;
  }
  return w;
}

// The marker's three words after `w` (the command buffer's last written word): its last word's address.
inline uint32_t PutMarker(uint8_t* base, uint32_t w, uint32_t seq) {
  uint8_t* p = HostOf(base, w + 4);
  rex::memory::store_and_swap<uint32_t>(p, 0xC0011000u);
  rex::memory::store_and_swap<uint32_t>(p + 4, 0x53413244u);  // 'SA2D'
  rex::memory::store_and_swap<uint32_t>(p + 8, seq);
  return w + 12;
}

// The native code applies to this call (null: run the original). The thread's state is reached here once and handed
// down (Ctx::t).
inline ThreadData* Enabled(const ::sa2::enh::Enhancement& key, const PPCContext& ctx) {
  if (!key.active || g_failed.load(std::memory_order_relaxed) || !me::native::NativeActive()) return nullptr;
  ThreadData& t = Thread();
  if (t.in_original) return nullptr;  // verify mode: inside an original's shadow run
  const uint32_t d = ctx.r3.u32;
  if (d < 0x40000000u || d >= 0x7F000000u || !Scratch()) return nullptr;
  return &t;
}

// The call's ops, written in place in the ring of records when it has room (nothing to copy at the publish), else in
// the thread's buffer (PublishApiRecord copies them). Opened after verify mode's shadow run, published by Publish;
// nothing else publishes records in between (one drawing thread).
inline Ops OpenOps(ThreadData& t) {
  if (uint32_t* const p = me::native::ReserveApiRecord(kOpsCapacity)) [[likely]] {
    Ops o{p};
    o.in_ring = true;
    return o;
  }
  return Ops{t.ops};
}

inline bool Publish(uint32_t& seq, const Ops& ops) {
  if (ops.overflow) [[unlikely]] {
    REXLOG_ERROR("[api] sa2_api_draws: a call's ops did not fit ({} words); native draws off", kOpsCapacity);
    g_failed.store(true, std::memory_order_relaxed);
    return false;
  }
  seq = ops.in_ring ? me::native::CommitApiRecord(ops.buf, ops.size())
                    : me::native::PublishApiRecord(ops.buf, ops.size());
  if (!seq) [[unlikely]] {
    REXLOG_ERROR("[api] sa2_api_draws: {}; native draws off",
                 ops.in_ring ? "the record's reserved space was taken by another record (a second producer?)"
                             : "no room in the ring of records");
    g_failed.store(true, std::memory_order_relaxed);
    return false;
  }
  Bump(g_native);
  Bump(g_ops_words, ops.size());
  g_api_draws.UseSampled();
  return true;
}

// The draw packets' addresses (the ring keys the pairing by them), given once the marker's place is known: per draw,
// its kPair record (words 1 and 8) and the DRAW_INDX op's address word, 16 bytes apart. The positions were noted as
// the ops were written (Ops::fix); a call with more than Ops::kMaxFix of them is scanned.
inline void ScanPacketAddresses(Ops& ops, uint32_t first) {
  uint32_t packet = first;
  const uint32_t n = ops.size();
  for (uint32_t i = 0; i < n;) {
    const uint32_t tag = ops.buf[i], count = tag >> 8;
    switch (tag & 0xFF) {
      case kRun: i += 2 + count; break;
      case kPair:
        ops.buf[i + 1] = packet;
        ops.buf[i + 8] = packet;
        i += 1 + count;
        break;
      case kType3: {
        if (IsDrawHeader(ops.buf[i + 1])) {  // DRAW_INDX, DRAW_INDX_2: unique per frame (the ring keys replays)
          ops.buf[i + 2] = packet;
          packet += 16;
        }
        i += 3 + count;
        break;
      }
      default: i = n; break;
    }
  }
}
inline void SetPacketAddresses(Ops& ops, uint32_t first) {
  if (ops.nfix > Ops::kMaxFix) [[unlikely]] {
    ScanPacketAddresses(ops, first);
    return;
  }
  uint32_t packet = first;
  for (uint32_t k = 0; k < ops.nfix; ++k) {
    const uint32_t f = ops.fix[k];
    if (f & 0x80000000u) {
      ops.buf[(f & 0x7FFFFFFFu) + 2] = packet;
      packet += 16;
    } else {
      ops.buf[f + 1] = packet;
      ops.buf[f + 8] = packet;
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------------
// Verify mode: Direct3D's original runs first on a copy of the device (r3 = the copy) with a command buffer of its own,
// large enough that it never asks for space (so it never submits or switches segments); the guest regions outside the
// device it may write (the bound vertex shader's binding states and program descriptors, the two bindings' microcode,
// the declaration's id, Direct3D's globals) are captured after it and put back. Then the native code runs for real
// (the renderer draws what it produced), and the two are compared: the device (the command buffer and submission
// fields excepted, and the interrupt callback's counters), those regions, and the native ops against the original's
// packets (canonical forms).
constexpr uint32_t kShadowCopy = 0x6000, kShadowBuffer = 0x80000;
inline uint32_t g_shadow = 0;

struct Region {
  uint32_t a, n;
  std::vector<uint8_t> before, original;
};

inline std::vector<Region> ExternalRegions(uint8_t* base, uint32_t d) {
  std::vector<Region> r;
  auto add = [&](uint32_t a, uint32_t n) {
    if (a >= 0x40000000u && n && n <= 0x10000u) r.push_back({a, n, {}, {}});
  };
  const Direct m{base};
  const uint32_t vs = m.L32(d + dev::kVertexShader), decl = m.L32(d + dev::kVertexDecl);
  if (vs) {
    add(vs, 0x400);
    for (uint32_t i = 0; i < 2; ++i) {
      const uint32_t prog = BindingProgram(m, vs, i);
      add(prog + 872, 8);
      add(m.L32(prog + 872) + m.L32(vs + 32), m.L32(prog + 876));
    }
  }
  if (decl) add(decl + 48, 4);
  add(kGlobalInterpMask, 12);
  return r;
}

inline void Snapshot(uint8_t* base, std::vector<Region>& r, bool original) {
  for (Region& x : r) {
    auto& v = original ? x.original : x.before;
    v.resize(x.n);
    std::memcpy(v.data(), HostOf(base, x.a), x.n);
  }
}

// Device fields the command buffer transport (space, submission, segments, inline data), the interrupt callback and
// the GPU (the ring's memory writes: +11032) write, never compared.
inline bool TransportField(uint32_t i) {
  return (i >= 48 && i < 60) || (i >= dev::kFenceCompleted && i < dev::kFenceCompleted + 4) ||
         (i >= 11032 && i < 11036) || (i >= 11036 && i < 11040) || (i >= 11088 && i < 11092) || (i >= 11120 && i < 11128) ||
         (i >= 11152 && i < 11156) || (i >= 12256 && i < 12260) || (i >= 13600 && i < 13616) ||
         (i >= 13772 && i < 13776) || (i >= 13936 && i < 13948);
}

struct Shadow {
  uint32_t copy = 0;
  uint32_t first = 0;          // the shadow buffer's first word
  std::vector<Region> regions;
  bool ok = false;
};

// Runs `original` on the copy (the caller's arguments, r3 = the copy).
inline Shadow RunShadow(ThreadData& t, uint8_t* base, const PPCContext& ctx, uint32_t d,
                        void (*original)(PPCContext&, uint8_t*),
                        std::initializer_list<std::pair<uint32_t, uint32_t>> extra = {}) {
  Shadow s;
  if (!g_shadow) g_shadow = rex::system::kernel_state()->memory()->SystemHeapAlloc(kShadowCopy + kShadowBuffer, 0x1000);
  if (!g_shadow) return s;
  s.copy = g_shadow;
  const uint32_t buffer = g_shadow + kShadowCopy;
  s.regions = ExternalRegions(base, d);
  for (const auto& [a, n] : extra) s.regions.push_back({a, n, {}, {}});  // the call's own guest outputs
  Snapshot(base, s.regions, false);
  std::memcpy(HostOf(base, s.copy), HostOf(base, d), kShadowCopy);
  rex::memory::store_and_swap<uint32_t>(HostOf(base, s.copy + dev::kWptr), buffer);
  rex::memory::store_and_swap<uint32_t>(HostOf(base, s.copy + dev::kLimit), buffer + kShadowBuffer - 0x1000);
  rex::memory::store_and_swap<uint32_t>(HostOf(base, s.copy + dev::kSoftLimit), buffer + kShadowBuffer / 2);
  s.first = buffer + 4;
  // The completed fence, read once: the copy points at this snapshot (the last word of the shadow area) and the native
  // code reads it too (ThreadData::fence_snapshot, until CompareShadow).
  const uint32_t holder = rex::memory::load_and_swap<uint32_t>(HostOf(base, d + dev::kFenceCompleted));
  t.fence_value = holder ? rex::memory::load_and_swap<uint32_t>(HostOf(base, holder)) : 0;
  t.fence_snapshot = holder != 0;
  if (holder) {
    const uint32_t word = g_shadow + kShadowCopy + kShadowBuffer - 4;
    rex::memory::store_and_swap<uint32_t>(HostOf(base, word), t.fence_value);
    rex::memory::store_and_swap<uint32_t>(HostOf(base, s.copy + dev::kFenceCompleted), word);
  }
  t.shadow_allocs.clear();
  t.native_allocs.clear();
  t.shadow_bump = g_shadow + kShadowCopy + kShadowBuffer / 2;
  PPCContext c = ctx;  // verify mode only: the original gets the caller's whole context, as it would
  c.r3.u64 = s.copy;
  t.in_original = true;
  original(c, base);
  t.in_original = false;
  Snapshot(base, s.regions, true);
  for (Region& x : s.regions) std::memcpy(HostOf(base, x.a), x.before.data(), x.n);
  s.ok = true;
  Bump(g_original);
  return s;
}

// After the native code ran for real: the comparison. `last` = the shadow buffer's last packet word; `submitted` = the
// native code's own submission (marker space) changed the transport fields. `fix` patches the native canon (the inline
// vertices' address, which differs when the native side submitted before its allocation).
inline void CompareShadow(ThreadData& t, const char* what, uint8_t* base, uint32_t d, const Shadow& s, const Ops& ops,
                          uint32_t last, bool submitted, std::initializer_list<std::pair<uint32_t, uint32_t>> skip) {
  t.fence_snapshot = false;
  Bump(g_checked);
  sa2::enh::VerifyCheck();
  if (ops.overflow) LogDifference(what, "native ops overflow");
  // Inline allocations: the k-th native one stands for the k-th of the shadow run (same size). Contents the call itself
  // wrote are compared, and the native ops' references to it (a vertex fetch constant's address word, Physical(a) | 3)
  // are mapped onto the shadow's address before the canonical comparison.
  std::vector<uint32_t> mapped(ops.buf, ops.buf + ops.size());
  if (t.native_allocs.size() != t.shadow_allocs.size())
    LogDifference(what, fmt::format("{} inline allocations, the original {}", t.native_allocs.size(),
                                    t.shadow_allocs.size()));
  for (size_t k = 0; k < std::min(t.native_allocs.size(), t.shadow_allocs.size()); ++k) {
    const InlineAlloc& a = t.native_allocs[k];
    const InlineAlloc& b = t.shadow_allocs[k];
    if (a.words != b.words) {
      LogDifference(what, fmt::format("inline allocation {}: {} words, the original {}", k, a.words, b.words));
      continue;
    }
    if (a.written && std::memcmp(HostOf(base, a.address), HostOf(base, b.address), size_t(a.words) * 4) != 0) {
      uint32_t w = 0;
      while (w < a.words && std::memcmp(HostOf(base, a.address + 4 * w), HostOf(base, b.address + 4 * w), 4) == 0) ++w;
      LogDifference(what, fmt::format("inline data {} word {}: native {:08X} original {:08X}", k, w,
                                      rex::memory::load_and_swap<uint32_t>(HostOf(base, a.address + 4 * w)),
                                      rex::memory::load_and_swap<uint32_t>(HostOf(base, b.address + 4 * w))));
    }
    const uint32_t from = Physical(a.address) | 3u, to = Physical(b.address) | 3u;
    for (uint32_t i = 0; i < mapped.size();) {
      const uint32_t tag = mapped[i], count = tag >> 8;
      if ((tag & 0xFF) == kRun) {
        for (uint32_t j = 0; j < count; ++j)
          if (mapped[i + 2 + j] == from) mapped[i + 2 + j] = to;
        i += 2 + count;
      } else if ((tag & 0xFF) == kType3) {
        i += 3 + count;
      } else if ((tag & 0xFF) == kPair) {
        i += 1 + count;
      } else {
        break;
      }
    }
  }
  t.native_allocs.clear();
  t.shadow_allocs.clear();
  Canon want, got;
  CanonPm4(want, base, s.first, last);
  CanonOps(got, mapped.data(), uint32_t(mapped.size()));
  if (!(want == got)) LogDifference(what, fmt::format("original:{} | native:{}", Describe(want), Describe(got)));
  for (uint32_t i = 0; i < dev::kSize; ++i) {
    if (TransportField(i)) continue;
    // The native side submitted (its marker's space): the submission's bookkeeping, fences and flag bits move.
    if (submitted && ((i >= 11020 && i < 11160) || (i >= 13560 && i < 13950))) continue;
    bool skipped = false;
    for (const auto& [a, b] : skip) skipped |= i >= a && i < b;
    if (skipped) continue;
    const uint8_t want_b = *HostOf(base, s.copy + i), got_b = *HostOf(base, d + i);
    if (want_b != got_b) {
      LogDifference(what, fmt::format("device+{} native {:02X} original {:02X}", i, got_b, want_b));
      break;
    }
  }
  for (const Region& x : s.regions) {
    if (std::memcmp(HostOf(base, x.a), x.original.data(), x.n) != 0) {
      uint32_t k = 0;
      while (k < x.n && *HostOf(base, x.a + k) == x.original[k]) ++k;
      LogDifference(what, fmt::format("guest byte {:08X} native {:02X} original {:02X}", x.a + k,
                                      *HostOf(base, x.a + k), x.original[k]));
      break;
    }
  }
}

// The native BeginVertices: state and flush, the marker's space, the inline vertices (Direct3D's allocator), the draw,
// the record and the marker (+48 moves at EndVertices). Returns the inline vertices' address (0: no space).
inline uint32_t BeginVerticesNative(Ctx<Direct>& c, uint32_t type, uint32_t count, uint32_t stride, uint32_t& marker_end,
                             bool& ok, bool& submitted) {
  ok = true;
  const uint32_t words = Rlwinm(count * stride, 30, 2, 31);
  const uint8_t saved = c.m.L8(c.d + dev::kStreamStrides);
  BeginVerticesState(c, stride);
  FlushState(c);
  c.m.S8(c.d + dev::kStreamStrides, saved);
  const uint32_t submissions = c.m.L32(c.d + 11088);
  const uint32_t w = MarkerSpace(c.t, c.base, c.ppc, c.d);
  const uint32_t data = AllocInline(c.t, c.base, c.ppc, c.d, words);
  submitted = c.m.L32(c.d + 11088) != submissions;
  if (data) BeginVerticesDraw(c, type, count, stride, data, words, w + 16);
  uint32_t seq = 0;
  if (!Publish(seq, c.ops)) {
    ok = false;
    return 0;
  }
  marker_end = PutMarker(c.base, w, seq);
  if (!data) c.m.S32(c.d + dev::kWptr, marker_end);  // no inline space: the state goes out, no draw
  return data;
}

}  // namespace detail
}  // namespace sa2::api
