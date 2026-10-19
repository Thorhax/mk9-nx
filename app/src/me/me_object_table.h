// SA2 native renderer: shader object address -> library entry, readable without a lock.
//
// The game thread learns which library entry a Direct3D shader object belongs to (NoteShaderObject) and the ring thread
// needs it for every draw. The map that holds it (g_entry_of_object) is guarded by a mutex that both threads take; this
// mirrors it in an open-addressing table of atomics for the ring (sa2_native_fast_pair).
//  * One writer at a time (the caller holds the map's mutex); keys are never removed or moved, a value may be replaced.
//  * A slot's value is stored before its key is published with release, so a reader that acquires the key sees the value.
//  * Keys are guest addresses >= 0x40000000, so 0 marks an empty slot.
//  * If a key cannot be placed within kProbes slots the table is marked full and readers must ask the map instead.
// Used by me_native_system.cpp and tests/cpu/test_native_object_table.cpp.
#pragma once

#include <atomic>
#include <cstdint>

namespace me::native {

template <class Value>  // a pointer type
class ObjectEntryTable {
 public:
  static constexpr uint32_t kSlots = 4096, kProbes = 64;

  void Publish(uint32_t object, Value entry) {
    uint32_t i = SlotOf(object);
    for (uint32_t probe = 0; probe < kProbes; ++probe, i = (i + 1) & (kSlots - 1)) {
      const uint32_t key = keys_[i].load(std::memory_order_relaxed);
      if (key == object) {
        values_[i].store(entry, std::memory_order_release);
        return;
      }
      if (key == 0) {
        values_[i].store(entry, std::memory_order_relaxed);
        keys_[i].store(object, std::memory_order_release);
        return;
      }
    }
    full_.store(true, std::memory_order_release);
  }

  // true = answered (entry is null for an unknown object); false = the table is full: ask the map.
  bool Lookup(uint32_t object, Value& entry) const {
    uint32_t i = SlotOf(object);
    for (uint32_t probe = 0; probe < kProbes; ++probe, i = (i + 1) & (kSlots - 1)) {
      const uint32_t key = keys_[i].load(std::memory_order_acquire);
      if (key == object) {
        entry = values_[i].load(std::memory_order_acquire);
        return true;
      }
      if (key == 0) {
        entry = nullptr;
        return !full_.load(std::memory_order_acquire);
      }
    }
    return false;
  }

 private:
  static uint32_t SlotOf(uint32_t object) { return (((object >> 4) * 0x9E3779B1u) >> 20) & (kSlots - 1); }

  std::atomic<uint32_t> keys_[kSlots]{};
  std::atomic<Value> values_[kSlots]{};
  std::atomic<bool> full_{false};
};

}  // namespace me::native
