// banjo-nx - lets the masseffect-nx renderer (written for aarch64: the Switch and an Apple Silicon Mac) build on an
// x86-64 Windows PC, for debugging the renderer on PC before the console.
//   - NEON: the real intrinsics on aarch64; SIMDe's portable implementation (SDK thirdparty) elsewhere, with the
//     same names, so the vectorised paths run unchanged (slower, but identical results).
//   - Switch thread helpers from sdk/src/core/threading_switch.h and the profiler counters: no-ops off the Switch.
#pragma once

#include <rex/platform.h>

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#else
#ifndef SIMDE_ENABLE_NATIVE_ALIASES
#define SIMDE_ENABLE_NATIVE_ALIASES
#endif
#include <simde/arm/neon.h>
#endif

#if REX_PLATFORM_SWITCH
#include <threading_switch.h>
// sa2-nx (2026-10-20): Renut's renderer (renut_engine, through banjo-nx) unpins its record worker when the key that
// pinned it is turned off; SA2's SDK has no unpin (banjo-nx's SDK does): the worker keeps its core then.
inline bool RexSwitchUnpinCurrentThread(int) { return false; }
#else
inline bool RexSwitchSetCurrentThreadPriorityOk(int) { return true; }
inline int RexSwitchCurrentCore() { return -1; }
inline bool RexSwitchSetCurrentThreadCore(int) { return true; }
inline bool RexSwitchPinCurrentThreadToCore(int) { return true; }
inline bool RexSwitchUnpinCurrentThread(int) { return true; }
#endif
