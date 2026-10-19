// SA2 native renderer: runs of PM4 register writes into the constant/fetch register range (0x4000-0x48BF).
//
// ApplyRunReference is the original per-word loop of NativeGraphicsSystem::WriteConstantRun; ApplyRunRaw does the same
// from the guest's big-endian words in memory, four registers at a time with NEON (byte swap, compare, store). Both leave
// the same register values and report which generation group (VS constants 0x4000-0x43FF, PS constants 0x4400-0x47FF,
// fetch constants 0x4800-0x48BF) had at least one word change. Used by the ring thread behind
// sa2_native_pm4_fast and by tests/cpu/test_native_pm4_runs.cpp.
#pragma once

#include <algorithm>
#include <cstdint>

#if defined(__ARM_NEON)
#include "me_platform_compat.h"  // banjo-nx: NEON on aarch64, SIMDe elsewhere
#endif

namespace me::native {

struct RunChange {
  bool vs = false, ps = false, fetch = false;
  bool operator==(const RunChange&) const = default;
};

// `run` points at register `index`; the run must lie inside [0x4000, 0x48C0).
template <typename Next>
inline RunChange ApplyRunReference(uint32_t* run, uint32_t index, uint32_t count, Next next) {
  RunChange change;
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t r = index + i, v = next();
    if (run[i] != v) {
      run[i] = v;
      if (r < 0x4400) change.vs = true;
      else if (r < 0x4800) change.ps = true;
      else change.fetch = true;
    }
  }
  return change;
}

// Same result from `guest` (count big-endian words, possibly unaligned).
inline RunChange ApplyRunRaw(uint32_t* regs, uint32_t index, uint32_t count, const uint32_t* guest) {
  // One range [from, to) of registers; true if any word changed.
  const auto range = [&](uint32_t from, uint32_t to) {
    bool changed = false;
    uint32_t r = from;
#if defined(__ARM_NEON)
    uint32x4_t difference = vdupq_n_u32(0);
    for (; r + 4 <= to; r += 4) {
      const uint32x4_t incoming =
          vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(vld1q_u32(guest + (r - index)))));
      difference = vorrq_u32(difference, veorq_u32(incoming, vld1q_u32(regs + r)));
      vst1q_u32(regs + r, incoming);
    }
    changed = vmaxvq_u32(difference) != 0;
#endif
    for (; r < to; ++r) {
      const uint32_t v = __builtin_bswap32(guest[r - index]);
      if (regs[r] != v) {
        regs[r] = v;
        changed = true;
      }
    }
    return changed;
  };
  const uint32_t end = index + count;
  RunChange change;
  if (index < 0x4400) change.vs = range(index, std::min<uint32_t>(end, 0x4400));
  if (end > 0x4400 && index < 0x4800) change.ps = range(std::max<uint32_t>(index, 0x4400), std::min<uint32_t>(end, 0x4800));
  if (end > 0x4800) change.fetch = range(std::max<uint32_t>(index, 0x4800), end);
  return change;
}

}  // namespace me::native
