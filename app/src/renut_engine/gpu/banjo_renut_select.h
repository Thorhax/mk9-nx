// banjo-nx (2026-10-16): two renderers in one NRO. `banjo_renderer` chooses at boot:
//   "native": banjo-nx's own native renderer (app/src/me), the default and the exact reference;
//   "renut":  Renut's Direct3D-level renderer (github.com/masterspike52/reNut, imported by tools/renut_gpu_import.py
//             into app/src/renut_engine/gpu), which runs N&B's Direct3D natively through plume: no PM4, no EDRAM tiles.
// Every hook of either renderer is always linked; each runs its renderer's code only when that renderer is chosen and
// the game's original function otherwise. At the addresses both hook (banjo_shared_hooks.h) banjo-nx's hook calls
// Renut's side, renut_native_sub_X, when Renut's renderer is chosen.
#pragma once

#include <cstdint>

#include <rex/hook.h>
#include <rex/ppc.h>

namespace banjo::renut_gpu {

// banjo_renderer == "renut", read once (the choice holds for the session). banjo-nx (2026-10-19): inline, a byte load
// after the first call (every Direct3D hook asks).
namespace detail {
extern int8_t g_active;  // 1 renut, -1 not, 0 not read yet
bool LatchActive();
}  // namespace detail
inline bool Active() {
  const int8_t a = detail::g_active;
  return a ? a > 0 : detail::LatchActive();
}

}  // namespace banjo::renut_gpu

// Renut's hook at an address banjo-nx does not hook.
#define BANJO_RENUT_HOOK(subroutine, function)                                         \
  REX_EXTERN(__imp__##subroutine);                                                     \
  extern "C" REX_FUNC(subroutine) {                                                    \
    if (::banjo::renut_gpu::Active()) {                                                \
      ::rex::ppc::HostToGuestFunction<function>(ctx, base);                            \
      return;                                                                          \
    }                                                                                  \
    __imp__##subroutine(ctx, base);                                                    \
  }

// Renut's hook at an address banjo-nx hooks too: its side only, for banjo-nx's hook to call.
#define BANJO_RENUT_HOOK_SHARED(subroutine, function)                                  \
  extern "C" void renut_native_##subroutine(PPCContext& ctx, uint8_t* base) {          \
    ::rex::ppc::HostToGuestFunction<function>(ctx, base);                              \
  }

// Renut's raw hooks (the body follows the macro, as after REX_HOOK_RAW).
#define BANJO_RENUT_HOOK_RAW(subroutine)                                               \
  REX_EXTERN(__imp__##subroutine);                                                     \
  static void banjo_renut_raw_##subroutine(PPCContext& ctx, uint8_t* base);            \
  extern "C" REX_FUNC(subroutine) {                                                    \
    if (::banjo::renut_gpu::Active()) {                                                \
      banjo_renut_raw_##subroutine(ctx, base);                                         \
      return;                                                                          \
    }                                                                                  \
    __imp__##subroutine(ctx, base);                                                    \
  }                                                                                    \
  static void banjo_renut_raw_##subroutine([[maybe_unused]] PPCContext& __restrict ctx, uint8_t* base)

#define BANJO_RENUT_HOOK_RAW_SHARED(subroutine) \
  extern "C" void renut_native_##subroutine([[maybe_unused]] PPCContext& __restrict ctx, uint8_t* base)

// For banjo-nx's own renderer hooks, first thing in their body: with Renut's renderer chosen, Renut's side at a shared
// address (BANJO_RENUT_YIELD_SHARED) or the game's original (BANJO_RENUT_YIELD), and nothing of banjo-nx's renderer.
#define BANJO_RENUT_YIELD_SHARED(subroutine)                                           \
  if (::banjo::renut_gpu::Active()) {                                                  \
    renut_native_##subroutine(ctx, base);                                              \
    return;                                                                            \
  }
#define BANJO_RENUT_YIELD(subroutine)                                                  \
  if (::banjo::renut_gpu::Active()) {                                                  \
    __imp__##subroutine(ctx, base);                                                    \
    return;                                                                            \
  }
