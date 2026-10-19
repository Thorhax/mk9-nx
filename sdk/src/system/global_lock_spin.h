#pragma once
/**
 * @file        system/global_lock_spin.h
 * @brief       Acquire the global critical region with a short bounded spin before blocking in the kernel.
 *
 * The region guards very short sections (a handle-table lookup, a retain). When another core holds it, a
 * blocking lock costs a kernel arbitration round trip (tens of microseconds on Horizon, plus a context
 * switch on both sides) for a wait of a few hundred nanoseconds. Spinning a bounded number of try-locks first
 * avoids that. The bound keeps the priority-inversion property of a real mutex: if the holder was preempted
 * on this core, the spinner gives up after a few microseconds and blocks.
 */

#include <cstdint>
#include <mutex>

#include <rex/cvar.h>
#include <rex/sys_counters.h>

REXCVAR_DECLARE(int32_t, global_lock_spin);

namespace rex::system {

inline void CpuRelax() {
#if defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#endif
}

// `contended_counter`: syscount id bumped when the first try failed (0 = none).
inline std::unique_lock<std::recursive_mutex> AcquireGlobalSpin(std::recursive_mutex& mutex,
                                                                unsigned contended_counter = 0) {
  std::unique_lock<std::recursive_mutex> lock(mutex, std::try_to_lock);
  if (lock.owns_lock()) {
    return lock;
  }
  if (contended_counter) {
    REX_SYS_COUNT(contended_counter, 1);
  }
  for (int32_t i = REXCVAR_GET(global_lock_spin); i > 0; --i) {
    CpuRelax();
    if (lock.try_lock()) {
      return lock;
    }
  }
  lock.lock();
  return lock;
}

}  // namespace rex::system
