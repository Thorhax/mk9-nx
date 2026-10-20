// sa2-nx (2026-10-20): banjo-nx's enhancement switches, as Renut's renderer (through banjo-nx) asks for them, answered
// by SA2's (src/cpu/sa2_enhancements.h): a banjo_* key is active when it is on and its group's master is on
// (sa2_cpu_enhancements / sa2_noncpu_enhancements).
#pragma once

#include "cpu/sa2_enhancements.h"

namespace banjo {

enum class EnhancementGroup { kCpu, kNonCpu };

inline bool EnhancementActive(EnhancementGroup group, bool own_key) {
  return own_key && (group == EnhancementGroup::kCpu ? sa2::enh::CpuMaster() : sa2::enh::NonCpuMaster());
}

// The generated code's self-checks: SA2's own 10 s enhancement report gives them (sa2::enh::StartReport).
inline void CodegenChecksReport(int) {}

}  // namespace banjo
