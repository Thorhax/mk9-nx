// SA2 - native renderer: deduplication of vertex uploads within a frame.
//
// The problem, measured
// ---------------------
// Every draw copies to the upload buffer the [vmin..vmax] range of each of its vertex bindings,
// with the byte order swapped. There is no cache: the same piece of geometry is copied again in
// full every time it is drawn.
//
//   Console, last 10 s stretch of gameplay (sa2_001.log):
//       vertices 2044.9 MB in 269 frames = 7.60 MB per frame = 204 MB/s
//       CPU write speed into the upload buffer's memory type (type 1, no CPU cache,
//       coherent) = 2462 MB/s  ->  3.09 ms per frame just writing
//
//   PC, with the sa2_native_diag_vertices_repeated diagnostic on, steady gameplay:
//       "5954.9 MB copied; 2584.2 MB repeated within the same frame and
//        3329.2 MB identical to an earlier frame"
//       -> 99.0-99.6 % of the bytes are byte-for-byte repeats, in eight reports in a row.
//          Of those, 41-49 % repeat within the same frame.
//
// What this piece does
// --------------------
// Only the safe half: the same-frame one. A table from (physical address, bytes, order) to the
// offset that copy already occupies in this frame's upload buffer. If a later draw asks for
// exactly the same range with the same order, the existing offset is bound and nothing is copied.
//
// Why it is exact and not a gamble
// --------------------------------
// On the Xbox 360 the GPU reads vertices from main memory when it executes the draw, not when
// the game records the packet. A title cannot rewrite a vertex range it has already referenced
// in this frame without first synchronizing with the GPU: if it did, the result on the original
// console would not be defined either. So between two draws of the same frame with no
// synchronization in between, the contents of that range are the same by construction.
//
// The points where the guest may have rewritten memory are the synchronizations. Forget() is
// called there and the table is cleared. With that, deduplication does not rely on any
// assumption.
//
// What it does not do (on purpose)
// --------------------------------
// The other half (the 51-59 % that is equal to an earlier frame) needs a persistent buffer and
// a content check (XXH3 of the guest bytes, as the texture cache does with raw_fingerprint). It is
// more expensive and more delicate than this half, which is free.
//
// How it is wired in (sa2_native_draws.cpp)
// ---------------------------------------------
//   - UseSlot(...)      : dedupe_.NewFrame(frame_);   // after ++frame_
//       This already covers SendAndWait: despite its name it does not wait for anything, it
//       switches work slots, and that goes through UseSlot, which resets used_upload_ and
//       bumps frame_. Old offsets die on their own.
//   - dedupe_.Forget() where the ring thread serves a guest wait for the GPU (a fence or a
//     WAIT_REG_MEM): the only point where the game can legally rewrite a range it already drew.
//
//   Beware of sa2_native_send_after_shadows: when true, the shadow pass ends in one slot and
//   the scene in the next, so the scene cannot reuse the shadow copies and half the savings
//   are lost.
//   - in the bindings loop of the draw path, before Reserve/EnqueueCopy:
//         VkDeviceSize offset;
//         if (dedupe_.Search(source.address, source.bytes, uint32_t(source.order), offset)) {
//           vertex_offsets[b] = offset;
//           continue;                       // already in this frame's upload buffer
//         }
//         Reserve(source.bytes, 4, offset);
//         ... (the usual copy) ...
//         dedupe_.Note(source.address, source.bytes, uint32_t(source.order), offset);
//
// The table has kSlots entries of 40 bytes (640 KB) and is cleared by generation, without
// memset.

#pragma once

#include <array>
#include <cstdint>

namespace sa2::native {

// Deduplication of vertex bindings within a frame. Single thread (the ring thread).
class DedupeVertices {
 public:
  // Number of slots. Power of 2. With ~1,244 draws per frame, each with 1 or 2 vertex bindings,
  // they fit easily in 4096 slots. The collisions() counter tells whether it falls short: if it
  // rises, double it here.
  /*
   * Raised to 16384 (640 KB). With 4096 there were 19,705-33,121 collisions per report; with
   * 16384 they drop to 5,423-6,725, 3.6x fewer.
   *
   * But the savings do not go up (30 % -> 31 %, 1.83 -> 1.84 MB per frame). Collisions were not
   * the limit. The real ceiling of the safe half is ~30-38 % of the bytes, not the 41-49 % the
   * diagnostic measures, because the table is cleared on every slot change (NewFrame,
   * which happens several times per frame with SendAndWait) and on every WAIT_REG_MEM. Those
   * clears are exactly what makes this exact rather than a gamble.
   *
   * It stays at 16384 because evicting live entries is wasted work, but enlarging it further
   * will not bring more savings.
   */
  static constexpr size_t kSlots = 16384;

  // A frame starts: everything recorded before becomes invalid (the upload buffer has been reset).
  void NewFrame(uint64_t frame) {
    // frame_ can never equal kEmpty, which marks an unused entry.
    frame_ = frame == kEmpty ? 0 : frame;
  }

  // The guest may have synchronized with the GPU and rewritten memory: what was recorded
  // becomes invalid even within the same frame. O(1): it only moves the generation.
  void Forget() {
    ++generation_;
  }

  // true if that range is already copied in this frame's upload buffer. 'offset' then holds
  // its offset and nothing needs to be copied.
  //
  // fingerprint: 0 = no check (the normal case). Non-zero = paranoid mode: the content fingerprint
  // is stored when recording and compared when looking up; on a mismatch it is counted in
  // discrepancies() and false is returned (the data is copied as usual). It proves in real
  // gameplay that deduplication never serves stale data, without risking a single pixel.
  bool Search(uint64_t address, uint32_t bytes, uint32_t order, uint64_t& offset,
              uint64_t fingerprint = 0) {
    ++queries_;
    const uint64_t key = Key(address, order);
    const Entry& e = table_[Index(key, bytes)];
    if (e.frame != frame_ || e.generation != generation_ || e.key != key ||
        e.bytes != bytes) {
      // usf4-nx: why it missed (report): another range or never seen, an older work slot, a guest wait since.
      ++misses_[e.key != key || e.bytes != bytes || e.frame == kEmpty ? 0 : e.frame != frame_ ? 1 : 2];
      return false;
    }
    // usf4-nx: only entries recorded with a fingerprint are checked (usf4_dedupe_audit_each records one in N).
    if (fingerprint != 0 && e.fingerprint != 0 && e.fingerprint != fingerprint) {
      ++discrepancies_;
      return false;
    }
    ++hits_;
    bytes_saved_ += bytes;
    offset = e.offset;
    return true;
  }

  // usf4-nx: the fingerprint of the live entry that Search would match (0 if none, or if it was recorded without
  // one): the caller computes the content fingerprint only when there is something to compare it with.
  uint64_t Fingerprint(uint64_t address, uint32_t bytes, uint32_t order) const {
    const uint64_t key = Key(address, order);
    const Entry& e = table_[Index(key, bytes)];
    return e.frame == frame_ && e.generation == generation_ && e.key == key && e.bytes == bytes ? e.fingerprint : 0;
  }
  // Records a copy just made. 'fingerprint' is only used in the paranoid mode of Search.
  void Note(uint64_t address, uint32_t bytes, uint32_t order, uint64_t offset,
              uint64_t fingerprint = 0) {
    const uint64_t key = Key(address, order);
    Entry& e = table_[Index(key, bytes)];
    if (e.frame == frame_ && e.generation == generation_ &&
        (e.key != key || e.bytes != bytes)) {
      ++collisions_;  // another live entry loses its slot: the table is too small
    }
    e.key = key;
    e.bytes = bytes;
    e.offset = offset;
    e.fingerprint = fingerprint;
    e.frame = frame_;
    e.generation = generation_;
  }

  // Counters for the periodic report (deltas between reports, like the others).
  uint64_t queries() const { return queries_; }
  uint64_t hits() const { return hits_; }
  uint64_t bytes_saved() const { return bytes_saved_; }
  uint64_t collisions() const { return collisions_; }
  uint64_t discrepancies() const { return discrepancies_; }
  // usf4-nx: misses by reason: 0 not in the table, 1 copied in an older work slot, 2 forgotten at a guest wait.
  const std::array<uint64_t, 3>& misses() const { return misses_; }

 private:
  static constexpr uint64_t kEmpty = ~uint64_t(0);

  struct Entry {
    uint64_t key = 0;           // (address << 2) | order
    uint64_t offset = 0;          // offset inside the upload buffer
    uint64_t fingerprint = 0;          // paranoid mode only
    uint64_t frame = kEmpty;  // kEmpty = never used
    uint32_t bytes = 0;
    uint32_t generation = 0;      // incremented by every Forget()
  };
  static_assert(sizeof(Entry) == 40, "an entry must fit in 40 bytes");

  static uint64_t Key(uint64_t address, uint32_t order) {
    return (address << 2) | (uint64_t(order) & 0x3);
  }

  // Multiply-and-shift mix (splitmix). Cheap, and it spreads aligned addresses well, which is
  // exactly what arrives here.
  static size_t Index(uint64_t key, uint32_t bytes) {
    uint64_t x = key ^ (uint64_t(bytes) * 0x9E3779B97F4A7C15ull);
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    return size_t(x) & (kSlots - 1);
  }

  std::array<Entry, kSlots> table_{};
  uint64_t frame_ = 0;
  uint32_t generation_ = 0;

  uint64_t queries_ = 0;
  uint64_t hits_ = 0;
  uint64_t bytes_saved_ = 0;
  uint64_t collisions_ = 0;
  uint64_t discrepancies_ = 0;
  std::array<uint64_t, 3> misses_{};
};

}  // namespace sa2::native
