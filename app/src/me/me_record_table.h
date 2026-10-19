// SA2 native renderer: the ring thread's table of draw records (game thread -> ring, see NoteDrawCall).
//
// Records are keyed by the command-buffer address the game was about to write (`start`, a physical address below 2^29).
// 8 KB address buckets of 8 slots; a DRAW_INDX finds the record whose start is the closest one at or below the packet,
// looking in the packet's bucket and the previous one. Two equivalent lookups:
//   FindScan  - the original: walks the slots (a cache line each) testing `valid` and `start`.
//   FindDense - the same selection over a dense array of the start addresses (empty = 0xFFFFFFFF, above any address):
//               two 32-byte rows instead of ~16 cold slots.
// Both structures are kept in sync by Insert/Consume whichever lookup is used, so the choice can change at any time.
// Used by me_native_system.cpp (sa2_native_fast_pair) and tests/cpu/test_native_record_table.cpp.
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace me::native {

template <class Record>
class DrawRecordTable {
 public:
  static constexpr uint32_t kBuckets = 4096, kPerBucket = 8, kEmpty = 0xFFFFFFFFu;
  struct Slot {
    Record record;
    uint32_t start = 0;
    bool valid = false;
  };

  // The insertion order of each slot lives in a dense array (orders_) like the starts: a full bucket evicts its oldest
  // record, and finding it must not touch eight cold slots.
  DrawRecordTable()
      : slots_(kBuckets * kPerBucket), starts_(kBuckets * kPerBucket, kEmpty), orders_(kBuckets * kPerBucket, 0) {}

  // Stores a record under `start`: replaces the one with the same start, else takes the first free slot of the
  // bucket, else evicts the oldest. `dense` selects how the first two are searched (same result).
  void Insert(Record&& record, uint32_t start, bool dense) {
    const uint32_t first = ((start >> 13) & (kBuckets - 1)) * kPerBucket;
    Slot* bucket = &slots_[first];
    Slot* slot = nullptr;
    if (dense) {
      const uint32_t* starts = &starts_[first];
      uint32_t same = kPerBucket, empty = kPerBucket;
      for (uint32_t i = 0; i < kPerBucket; ++i) {
        if (starts[i] == start && same == kPerBucket) same = i;
        if (starts[i] == kEmpty && empty == kPerBucket) empty = i;
      }
      if (same != kPerBucket) slot = &bucket[same];
      else if (empty != kPerBucket) slot = &bucket[empty];
    } else {
      for (uint32_t i = 0; i < kPerBucket && !slot; ++i)
        if (bucket[i].valid && bucket[i].start == start) slot = &bucket[i];
      for (uint32_t i = 0; i < kPerBucket && !slot; ++i)
        if (!bucket[i].valid) slot = &bucket[i];
    }
    if (!slot) {  // full: the oldest record of the bucket goes
      const uint64_t* orders = &orders_[first];
      uint32_t oldest = 0;
      for (uint32_t i = 1; i < kPerBucket; ++i)
        if (orders[i] < orders[oldest]) oldest = i;
      slot = &bucket[oldest];
    }
    slot->record = std::move(record);
    slot->start = start;
    slot->valid = true;
    const size_t index = size_t(slot - slots_.data());
    starts_[index] = start;
    orders_[index] = ++order_;
  }

  // The slot with the closest start at or below `address` (nullptr if none), scanning the slots.
  Slot* FindScan(uint32_t address) {
    Slot* best = nullptr;
    for (uint32_t k = 0; k < 2; ++k) {
      const uint32_t b = ((address >> 13) - k) & (kBuckets - 1);
      Slot* bucket = &slots_[b * kPerBucket];
      for (uint32_t i = 0; i < kPerBucket; ++i) {
        Slot& s = bucket[i];
        if (s.valid && s.start <= address && (!best || s.start > best->start)) best = &s;
      }
    }
    return best;
  }

  // The same over the dense start array.
  Slot* FindDense(uint32_t address) {
    uint32_t best_start = 0, best_index = UINT32_MAX;
    for (uint32_t k = 0; k < 2; ++k) {
      const uint32_t first = (((address >> 13) - k) & (kBuckets - 1)) * kPerBucket;
      const uint32_t* starts = &starts_[first];
      for (uint32_t i = 0; i < kPerBucket; ++i) {
        const uint32_t start = starts[i];
        if (start <= address && (best_index == UINT32_MAX || start > best_start)) {
          best_start = start;
          best_index = first + i;
        }
      }
    }
    return best_index == UINT32_MAX ? nullptr : &slots_[best_index];
  }

  // The record was taken: the slot is free again.
  void Consume(Slot* slot) {
    slot->valid = false;
    starts_[size_t(slot - slots_.data())] = kEmpty;
  }

  size_t IndexOf(const Slot* slot) const { return size_t(slot - slots_.data()); }

 private:
  std::vector<Slot> slots_;
  std::vector<uint32_t> starts_;
  std::vector<uint64_t> orders_;
  uint64_t order_ = 0;
};

}  // namespace me::native
