// Ultra Street Fighter IV - native renderer: vertex ranges kept across frames (usf4_vertex_cache).
//
// THE PROBLEM, MEASURED
//   Switch 0.6.1, fight at CPU 1785 / GPU 768 (2026-10-15): ~20,000 vertex-binding lookups per second miss the
//   same-frame dedupe (sa2_native_vertices_dedupe.h), and ~16,000 of them were "copied in an older work slot":
//   the very same range the previous frame copied, copied again because the upload buffer belongs to one work
//   slot and is reset when that slot is reused. Copying means byte-swapping every word into GPU-visible memory;
//   on the ring thread CopyVertices + memcpy + the NEON loads/stores were ~18 % of its samples.
//
// WHAT THIS DOES
//   A separate GPU buffer whose ranges survive the work slots. A binding that missed the same-frame dedupe is
//   looked up here by (physical address, byte order, bytes). The guest bytes are hashed (XXH3-64) and compared
//   with the hash taken when the range was copied: equal -> the copy already in this buffer is bound and
//   nothing is written; different (the game rewrote it) -> copied again to a new place. So every reuse is
//   checked against the guest's current bytes; only the byte-swap and the write into GPU memory are saved.
//   Within one frame, a range already checked is not hashed again until the guest synchronizes with the GPU
//   (the same rule as the same-frame dedupe: before that, the game cannot legally rewrite what it drew).
//
// RECHECK INTERVAL (0.6.4)
//   Hashing turned out to cost as much as the copy it saves: Switch 0.6.3, ~6 MB hashed per frame, the CRC
//   fingerprint ~16 % of the ring thread, ring work per Swap unchanged. So a range is trusted without a hash,
//   like a stable texture (sa2_native_texture_interval_max): after 2 checks in a row in different frames that found
//   the same bytes, it is checked again after 2, then 4, then up to usf4_vertex_cache_interval_max frames.
//   Only while it is drawn without a gap: a range not bound in the previous frame (a new round, a new stage, a
//   loading screen in between) is checked first. A range found changed after it had been trusted is checked every
//   frame from then on (late_change), as sa2_native_adaptive_texture does for textures.
//
// WHY THE GPU NEVER READS A RANGE WHILE IT IS REWRITTEN
//   A copy is never written in place: a changed range goes to a new allocation. Space is handed out from a
//   ring, oldest first, and an allocation is only reused when the last frame that bound it has finished on
//   the GPU (UseSlot of a slot means that slot's previous submission was waited for; a queue's fences signal
//   in submission order, so every earlier frame has finished too). An allocation still in flight at the front
//   of the ring is retired instead: it is no longer handed to new draws, so a few frames later it is free.
//
// Single thread (the ring thread).
#pragma once

#include <cstdint>
#include <deque>
#include <vector>

namespace usf4::native {

class VertexCache {
 public:
  static constexpr uint32_t kTable = 32768;  // direct-mapped entries (power of 2)

  void Init(uint64_t size) {
    size_ = size;
    table_.assign(kTable, Entry{});
    allocations_.clear();
    first_id_ = 0;
    head_ = 0;
    wrapped_ = false;
  }
  bool active() const { return size_ != 0; }
  uint64_t size() const { return size_; }

  struct Entry {
    uint64_t key = 0;
    uint64_t hash = 0;
    uint64_t id = 0;          // its allocation
    uint64_t offset = 0;
    uint64_t checked_frame = ~uint64_t(0);  // the frame and guest synchronization of the last hash check
    uint64_t skip_until = 0;  // a range that keeps changing is not looked up again before this frame
    uint32_t checked_sync = 0;
    uint32_t bytes = 0;
    uint32_t changes = 0;     // consecutive copies with a different hash
    uint64_t seen_frame = 0;  // the last frame that bound it
    uint64_t next_check = 0;  // trusted without a hash before this frame (while seen in the previous frame)
    uint32_t stable = 0;      // checks in a row, in different frames, that found the same bytes
    bool late_change = false; // changed after it had been trusted: checked every frame
    bool used = false;
  };

  struct Found {
    Entry* entry = nullptr;
    bool same = false;  // same key and bytes
  };

  // The entry for this range. `same` says whether it is really this range (the table is direct-mapped).
  Found Look(uint64_t address, uint32_t order, uint32_t bytes) {
    const uint64_t key = Key(address, order);
    Entry& e = table_[Index(key, bytes)];
    return {&e, e.used && e.key == key && e.bytes == bytes};
  }

  // Marks a reuse of a live entry in this frame (keeps its allocation from being reused while in flight).
  void Use(Entry& e, uint64_t frame) {
    Allocation* a = AllocationOf(e.id);
    if (a) a->last_use = frame;
  }

  bool Live(const Entry& e) const {
    const Allocation* a = const_cast<VertexCache*>(this)->AllocationOf(e.id);
    return a && a->live;
  }

  // Space for `bytes` at a multiple of `align` (a multiple of 4, not always a power of 2). false = no room right
  // now without touching memory the GPU may still read: the caller copies as before.
  bool Allocate(uint64_t bytes, uint64_t align, uint64_t frame_completed, uint64_t& offset) {
    if (!size_ || bytes == 0 || bytes + align > size_ / 4) return false;
    for (uint32_t guard = 0; guard < 4096; ++guard) {
      const uint64_t start = (head_ + align - 1) / align * align;
      if (allocations_.empty()) {
        wrapped_ = false;
        const uint64_t s = start + bytes <= size_ ? start : 0;
        return Take(s, bytes, offset);
      }
      if (!wrapped_) {
        if (start + bytes <= size_) return Take(start, bytes, offset);
        head_ = 0;  // the end of the buffer is too short: carry on from the start, behind the oldest
        wrapped_ = true;
        ++wraps_;
        continue;
      }
      const uint64_t tail = allocations_.front().offset;
      if (start + bytes <= tail) return Take(start, bytes, offset);
      // The oldest allocation is in the way.
      Allocation& front = allocations_.front();
      if (front.last_use > frame_completed) {
        // Still read by a frame in flight. Everything up to where this allocation would end is retired: no
        // new draw gets those ranges from now on (they are copied again elsewhere), so they free themselves
        // once the frames that bound them have finished.
        for (Allocation& a : allocations_) {
          if (a.offset < start) break;  // past the wrap: not in the way
          if (a.live) {
            a.live = false;
            ++retired_;
          }
          if (a.offset + a.bytes >= start + bytes) break;
        }
        ++blocked_;
        return false;
      }
      const uint64_t before = front.offset;
      Forget(front);
      allocations_.pop_front();
      ++first_id_;
      ++evicted_;
      if (allocations_.empty() || allocations_.front().offset < before) wrapped_ = false;
    }
    return false;
  }

  // Records the copy just placed at `offset` (from Allocate) for this range.
  void Note(Entry& e, uint64_t address, uint32_t order, uint32_t bytes, uint64_t hash, uint64_t frame,
            uint32_t sync) {
    const uint64_t key = Key(address, order);
    if (e.used && (e.key != key || e.bytes != bytes)) ++replaced_;  // another range loses its entry
    if (e.used && e.key == key && e.bytes == bytes) {
      e.changes = e.hash == hash ? 0 : e.changes + 1;
    } else {
      e.changes = 0;
      e.late_change = false;
    }
    e.stable = 0;
    e.next_check = 0;
    e.seen_frame = frame;
    if (Allocation* old = e.used ? AllocationOf(e.id) : nullptr) old->live = false;
    e.used = true;
    e.key = key;
    e.bytes = bytes;
    e.hash = hash;
    e.id = first_id_ + allocations_.size() - 1;  // Take() has just pushed it
    e.offset = allocations_.back().offset;
    e.checked_frame = frame;
    e.checked_sync = sync;
    allocations_.back().last_use = frame;
    allocations_.back().slot = uint32_t(&e - table_.data());
  }


  // A check in `frame` found the same bytes: when the next one is due. `max` = usf4_vertex_cache_interval_max.
  static void Stable(Entry& e, uint64_t frame, uint32_t max) {
    if (e.checked_frame != frame) ++e.stable;
    e.seen_frame = frame;
    if (e.late_change || max < 2 || e.stable < 2) {
      e.next_check = 0;
      return;
    }
    uint32_t interval = e.stable >= 6 ? 32u : 1u << (e.stable - 1);  // 2, 4, 8, 16, then 32
    interval = interval < max ? interval : max;
    // Spread: ranges that became stable together (a new stage) are not all checked in the same frame.
    const uint32_t jitter = interval >= 4 ? uint32_t(e.key >> 4) & (interval / 2 - 1) : 0;
    e.next_check = frame + interval - jitter;
  }

  // May this range be bound in `frame` without a hash?
  static bool Trusted(const Entry& e, uint64_t frame) {
    return frame < e.next_check && e.seen_frame + 1 >= frame;
  }

  // Counters for the report.
  uint64_t evicted() const { return evicted_; }
  uint64_t retired() const { return retired_; }
  uint64_t blocked() const { return blocked_; }
  uint64_t replaced() const { return replaced_; }
  uint64_t wraps() const { return wraps_; }
  uint64_t used_bytes() const {
    uint64_t total = 0;
    for (const Allocation& a : allocations_) total += a.bytes;
    return total;
  }

 private:
  struct Allocation {
    uint64_t offset = 0;
    uint64_t bytes = 0;
    uint64_t last_use = 0;
    uint32_t slot = 0;
    bool live = true;
  };

  bool Take(uint64_t start, uint64_t bytes, uint64_t& offset) {
    allocations_.push_back({start, bytes, 0, 0, true});
    head_ = start + bytes;
    offset = start;
    return true;
  }

  Allocation* AllocationOf(uint64_t id) {
    if (id < first_id_ || id - first_id_ >= allocations_.size()) return nullptr;
    return &allocations_[size_t(id - first_id_)];
  }

  void Forget(const Allocation& a) {
    Entry& e = table_[a.slot];
    if (e.used && e.id == first_id_) {  // the front is the allocation with id first_id_
      e.used = false;
    }
  }

  static uint64_t Key(uint64_t address, uint32_t order) { return (address << 2) | (uint64_t(order) & 0x3); }
  static uint32_t Index(uint64_t key, uint32_t bytes) {
    uint64_t x = key ^ (uint64_t(bytes) * 0x9E3779B97F4A7C15ull);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    return uint32_t(x) & (kTable - 1);
  }

  uint64_t size_ = 0;
  std::vector<Entry> table_;
  std::deque<Allocation> allocations_;  // oldest first, in ring order
  uint64_t first_id_ = 0;               // id of allocations_.front()
  uint64_t head_ = 0;
  bool wrapped_ = false;                // head_ is behind the oldest allocation
  uint64_t evicted_ = 0, retired_ = 0, blocked_ = 0, replaced_ = 0, wraps_ = 0;
};

}  // namespace usf4::native
