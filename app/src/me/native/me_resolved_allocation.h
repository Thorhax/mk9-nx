#pragma once

#include <cstdint>

namespace me::native {

// A submission serial is proof only AFTER a successful fence wait. Raw fence
// handles cannot be cached as proof because work slots reset and reuse them.
constexpr uint64_t ObserveResolvedAllocationCompletion(uint64_t completed,
    uint64_t serial, bool wait_success) {
  return wait_success && serial > completed ? serial : completed;
}

constexpr bool ResolvedAllocationReady(uint64_t retired, uint64_t completed) {
  return retired != 0 && completed >= retired;
}

// Overflow-safe aggregate allocation bound. Incoming is the Vulkan memory
// requirement, not merely width*height*texel_size.
constexpr bool ResolvedAllocationFits(uint64_t count, uint64_t bytes,
    uint64_t incoming, uint64_t count_cap = 32, uint64_t byte_cap = 128ull << 20) {
  return count < count_cap && bytes <= byte_cap && incoming <= byte_cap - bytes;
}

}  // namespace me::native
