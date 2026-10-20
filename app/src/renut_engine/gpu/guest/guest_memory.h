/**
 * @file    gpu/guest/guest_memory.h
 * @brief   Guest virtual address translation for the renderer.
 */
#pragma once

#include <atomic>

#include <rex/system/kernel_state.h>
#include <rex/types.h>

namespace renut::gpu {

namespace detail {
// banjo-nx (2026-10-19, render-thread proposal 11): the guest memory bases, read once (they never change after
// Memory::Initialize): one load at a translation instead of the kernel state -> memory -> base chain (603 sites).
inline std::atomic<u8 *> g_guest_vbase{nullptr};
inline std::atomic<u8 *> g_guest_pbase{nullptr};
[[gnu::noinline, gnu::cold]] inline void LoadGuestBases() {
  rex::memory::Memory *memory = REX_KERNEL_MEMORY();
  g_guest_vbase.store(memory->virtual_membase(), std::memory_order_relaxed);
  g_guest_pbase.store(memory->physical_membase(), std::memory_order_relaxed);
}
inline u8 *GuestPhysicalBase() {
  u8 *base = g_guest_pbase.load(std::memory_order_relaxed);
  if (!base) [[unlikely]] {
    LoadGuestBases();
    base = g_guest_pbase.load(std::memory_order_relaxed);
  }
  return base;
}
inline u8 *GuestVirtualBase() {
  u8 *base = g_guest_vbase.load(std::memory_order_relaxed);
  if (!base) [[unlikely]] {
    LoadGuestBases();
    base = g_guest_vbase.load(std::memory_order_relaxed);
  }
  return base;
}
} // namespace detail

// Host pointer for a guest VA, or nullptr for VA 0.
template <typename T>
inline T *GuestPtr(u32 guest_address) {
  if (!guest_address)
    return nullptr;
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && defined(__aarch64__))
  return REX_KERNEL_MEMORY()->TranslateVirtual<T *>(guest_address);  // heaps with host offsets there
#else
  return reinterpret_cast<T *>(detail::GuestVirtualBase() + guest_address);  // TranslateVirtual's sum (no offsets)
#endif
}

// Host pointer for a GPU physical address (fetch constants, index buffer
// addresses): the low 29 bits index physical memory.
template <typename T>
inline T *GuestPhysicalPtr(u32 physical_address) {
  return reinterpret_cast<T *>(detail::GuestPhysicalBase() + (physical_address & 0x1FFFFFFF));  // TranslatePhysical
}

} // namespace renut::gpu
