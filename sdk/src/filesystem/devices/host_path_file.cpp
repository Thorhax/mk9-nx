/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/filesystem/devices/host_path_device.h>
#include <rex/filesystem/devices/host_path_entry.h>
#include <rex/filesystem/devices/host_path_file.h>
#include "startup_trace.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/string.h>

/*
 * Size of the read-ahead window, in KB. 0 turns it off.
 *
 * 256 KB rather than 64 KB. With 64 KB the entry threshold was 16 KB (window/4), and the game does
 * not do a single read of 16 KB or less: its average is 375 KB and the smallest one it repeats in a
 * loop is 64 KB (measured with 64 KB: 5 hits out of 1,232 reads). With 256 KB the threshold lands
 * exactly on those 64 KB. The RAM ceiling goes down, not up: 16 windows x 256 KB = 4 MB, against
 * 128 x 64 KB = 8 MB.
 */
REXCVAR_DEFINE_INT32(sa2_io_window_kb, 256, "Filesystem",
                     "Read-ahead per file, in KB (0 = off). Read-only data only.");

// How many windows can exist at once. A hard cap so that opening many files does not eat the RAM.
// 128 was an arbitrary number; in a whole measured session there were never more than 10 live windows.
REXCVAR_DEFINE_INT32(sa2_io_windows_max, 16, "Filesystem",
                     "Maximum number of files with read-ahead at the same time.");

/*
 * Split large direct reads into pieces of this many MB. 0 = a single call.
 *
 * The worst case measured is a pread() of 19,932,032 bytes that takes 261.5 ms (there are 7 reads
 * over 8 MB, 169.2 ms on average). During those 261 ms the thread passes no point where the
 * scheduler can cleanly switch it out, and the per-file lock of XFile (file_lock_) stays held the
 * whole time.
 *
 * The cost of splitting can be bounded with the same data: fitting time = fixed + bytes/rate over
 * the 4.00 MB reads (59.8 ms on average, 21 samples) and the 19.01 MB reads (275.6 ms, 2 samples)
 * gives rate = 69.5 MB/s and fixed = 2.3 ms per call. So splitting 19 MB into 4 MB pieces is 4
 * extra calls = +9 ms over 275 (+3 %); 1 MB pieces would be 18 extra = +41 ms (+15 %). Hence 4
 * and not 1.
 *
 * In exchange, the worst pread drops from 261 ms to about 60, and short reads from the file system
 * are no longer visible: the loop keeps asking from where it stopped and only stops when a piece
 * returns 0 bytes (real end of file). The unsplit path returned the short read as it was.
 *
 * Measured: this does not raise FPS. Reads run on a thread other than the render thread, and outside
 * loading screens there is no correlation between slow reads and stutters. It is on because it
 * bounds the worst case and fixes short reads, not because a gain is expected.
 */
REXCVAR_DEFINE_INT32(sa2_io_chunk_mb, 4, "Filesystem",
                     "Splits large direct reads into pieces of this many MB (0 = in a single call).");

/*
 * RAM read cache by blocks (sa2_io_cache_mb, off). The game rereads the same data from the SD
 * every lap.
 *
 * Measured over 5 minutes of play (line `[io] ranges:`):
 *     from minute 2 on, between 79 % and 95 % of the bytes the game reads from disk had already
 *     been read before, and the sizes repeat with the lap period (20.9 MB three times, 13.0-13.6 MB
 *     twice...).
 * The game streaming works per zone pack: on leaving a zone it releases the whole
 * pack and on entering again it asks for it again. We evict nothing (`0 evictions` in every
 * report): the one releasing it is the game's memory manager, and it cannot be told no.
 *
 * What can be done is make the second time not cost a trip to the SD. In a race that is 441
 * reads, 181.5 MB and 4,447 ms inside read(), with single reads of up to 132 ms: until the pack
 * arrives, the facade shows the fallback texture, which is the "finishes loading when I get
 * close" effect seen on the console.
 *
 * 256 KB blocks with LRU. Missing blocks are read grouped, in a single pread per contiguous range,
 * so the first read costs the same as without the cache (not 36 calls for one of 9 MB) and later
 * ones are memcpy. Only read-only devices and files opened without write access: the game data
 * never changes.
 *
 * Sized at 64 MB: the unique bytes of a whole race are ~36 MB (181 MB with 80 % rereads), so the
 * whole route fits. 0 turns it off.
 *
 * ================================================================================================
 *  Off: on the console it does the opposite of what it promises.
 *
 *  Its own report on the console:
 *      304.6 MB delivered, 318.3 read from the SD, 1,053 blocks evicted
 *  It reads more disk than it saves. On the PC it gave 254 delivered against 217 read, which was
 *  misleading. The difference: on the console the race asks for more unique data than fits, the LRU
 *  evicts nonstop, and each 256 KB block is brought in to serve a small request and thrown away
 *  before it is reused. The 256 KB amplification per request eats the savings and more.
 *
 *  It also blocks: `g_cache_mutex` is a single mutex for all files and is held during the disk
 *  read. Any other reading thread waits behind it. (Careful when measuring: that time does not
 *  show in `[io] 15 s`, which only times inside read(); that line showed 32 ms per 15 s while the
 *  game crawled.)
 * ================================================================================================
 */
REXCVAR_DEFINE_INT32(sa2_io_cache_mb, 0, "Filesystem",
                     "RAM cache of what was already read from disk, in MB (0 = off). Read-only data "
                     "only. The game rereads the same data over and over.");

/*
 * Read cache by exact range. This is the second attempt, not the block cache above.
 *
 * What was measured. Across two different builds the per-lap stutter pattern is identical: a period
 * of 62-68 s, three or four fixed places, always in the same order and with the same size. In those
 * frames the breakdown says GPU 27-29 ms (normal) and the `record` stage 0.2-0.4 ms: the time is
 * neither in the GPU nor in recording. What does go together with the big stutters are `[io] SLOW`
 * reads of a large streaming archive marked (REREAD), of 11 to 20 ms, and the worst one of the session
 * blocks for 132 to 140 ms. They are rereads of the zone pack: the game releases the pack on
 * leaving and asks for it again on entering, with the same offset and the same size.
 *
 * Why by range and not by blocks. The block cache above (sa2_io_cache_mb) failed on the console
 * because of the blocks: only 1 % of the offsets the game asks for are aligned to 256 KB, so it
 * brought 365 KB from the SD for every 110 KB requested (x3.3 amplification) and ended up reading
 * more disk than it saved. Storing the exact range (file, offset, bytes) makes the amplification
 * zero by construction: not a single byte the game did not ask for is read.
 *
 * 4 MB floor (first sizing; lowered below). Only requests of 4 MB or more got in. That leaves out
 * entirely the small requests, the ones that sank the previous attempt, and keeps exactly the
 * spikes the player notices.
 *
 * The numbers, simulated on the session logs: the slowest read of the race goes from 132.2 ms to
 * 29, and the 12 spikes of 80 ms or more go to 0. It costs 20.9 MB, 3 entries and 0 evictions.
 *
 * The cap cannot be lowered. With a 20 MB cap the result is zero spikes avoided: the three entries
 * do not fit, the LRU evicts them before they are reused and not a single hit is scored. It is a
 * cliff, not a slope, and it is exactly what sank the block cache. 32 MB and no less.
 *
 * Where that RAM comes from: the host heap (1,024 MB, runtime_switch.cpp), of which the guest
 * backing eats 506. Not the GPU budget, which is separate. That is why the first sizing stopped
 * at 32.
 *
 * The lock is not held across the disk read. The other failure of the block cache was that
 * g_cache_mutex, a single mutex for all files, stayed held inside the pread: any other reading
 * thread waited behind it, and that time did not show in `[io] 15 s` (which only times inside
 * read()). Here the lock is taken to look up the table and released before going to disk; the
 * data lives in a shared_ptr, so copying it needs no lock either, and an eviction cannot free the
 * buffer another thread is reading.
 */
/*
 * Resized from the race logs. The first version (cap 32, floor 4 MB) hit 25 % and saved 10 ms
 * per race. Two misreadings of the data:
 *
 * 1. The 34 evictions were not race data: they were the sequential sweep of the level load
 *    (34 blocks of a large streaming file, 4 MB at a time, 154 MB, each read once). Raising the cap from 32
 *    to 128 did not buy a single hit. What is needed is not storing that sweep (see SaveRange).
 * 2. The 4 MB floor did not see what repeats during a race, which is small: 62 distinct requests
 *    of 0.14 to 2.0 MB (24.7 MB in total), repeated 2 to 10 times each, almost all from
 *    a large streaming file. Simulated on the real trace: floor 4 MB -> 16.6 % hits and 10 ms; floor
 *    256 KB -> 52.7 % and 1,132 ms saved per race. Footprint with that floor: 56.6 MB, so 64 MB
 *    is the first cap with 0 evictions, and above it nothing is gained.
 *
 * Memory: it comes from the host heap (1,024 MB), not from the GPU heap. With ~506 MB for the guest
 * and the thread stacks, 64 MB leaves margin; 128 would not in the pessimistic case.
 */
REXCVAR_DEFINE_INT32(sa2_io_ranges_mb, 64, "Filesystem",
                     "Cache of reads by exact range, total cap in MB (0 = off). 64 = the measured "
                     "footprint of a session (56.6 MB) without evictions; more buys nothing.");

// Floor, in KB. At 4 MB it left out everything that is reread during a race (0.14-2.0 MB).
REXCVAR_DEFINE_INT32(sa2_io_ranges_min_kb, 256, "Filesystem",
                     "Only reads of this many KB or more are cached. 256 = the point where the simulation "
                     "stops gaining (with 128 evictions go up and hits go down).");

// Compatibility: the old cvar in MB. If a toml sets it to a value > 0, it overrides the KB one.
REXCVAR_DEFINE_INT32(sa2_io_ranges_min_mb, 0, "Filesystem",
                     "OBSOLETE (build 131): use sa2_io_ranges_min_kb. If it is > 0 it takes precedence over the KB one.");

// Per-entry ceiling: a single read cannot take more than this out of the total cap.
REXCVAR_DEFINE_INT32(sa2_io_ranges_max_mb, 12, "Filesystem",
                     "No entry of the range cache exceeds this many MB.");

namespace rex::filesystem {

namespace {
std::atomic<uint64_t> g_hits{0};
std::atomic<uint64_t> g_filled{0};
std::atomic<uint64_t> g_direct{0};
std::atomic<uint64_t> g_bytes_ram{0};
std::atomic<int64_t> g_live_windows{0};

// Above this it does not pay off: the request is already large and the window would only add an extra copy.
constexpr size_t kSubmissionMaxFraction = 4;  // request <= window/4

/* The read cache (see sa2_io_cache_mb). */
constexpr size_t kBlockCache = 256 * 1024;

struct BlockCache {
  std::vector<uint8_t> data;  // may be shorter than kBlockCache at end of file
  uint64_t usage = 0;            // for the LRU
};

std::mutex g_cache_mutex;
std::unordered_map<uint64_t, BlockCache> g_cache;   // key: file id << 32 | block
std::map<uint64_t, uint64_t> g_cache_lru;            // use -> key, oldest first
std::unordered_map<std::string, uint32_t> g_cache_ids;  // path -> id, so the key is exact
uint64_t g_cache_bytes = 0;
uint64_t g_cache_usage = 0;
uint64_t g_cache_hits = 0;
uint64_t g_cache_failures = 0;
uint64_t g_cache_bytes_served = 0;
uint64_t g_cache_bytes_disk = 0;
uint64_t g_cache_evicted = 0;
uint64_t g_cache_report = 0;
bool g_cache_no_memory = false;  // turned itself off for lack of RAM: not retried

// Stable id per path, assigned the first time it is opened. With it, the block key cannot
// collide between files (a 64-bit hash could, and a collision would serve another file's data).
uint32_t IdOfPath(const std::string& path) {
  std::lock_guard lock(g_cache_mutex);
  const auto [it, new_value] = g_cache_ids.try_emplace(path, uint32_t(g_cache_ids.size() + 1));
  return it->second;
}

// Marks a block as just used. Called with g_cache_mutex held.
void TouchBlock(uint64_t key, BlockCache& block) {
  if (block.usage) {
    g_cache_lru.erase(block.usage);
  }
  block.usage = ++g_cache_usage;
  g_cache_lru[block.usage] = key;
}

// Brings the cache under the limit by dropping the oldest blocks. Called with g_cache_mutex held.
void PruneCache(size_t limit) {
  while (g_cache_bytes > limit && !g_cache_lru.empty()) {
    const auto old = g_cache_lru.begin();
    const auto it = g_cache.find(old->second);
    if (it != g_cache.end()) {
      g_cache_bytes -= it->second.data.size();
      g_cache.erase(it);
      ++g_cache_evicted;
    }
    g_cache_lru.erase(old);
  }
}

/*
 * Serves the read from RAM, bringing from disk only the missing blocks (grouped, one pread per
 * contiguous range). Returns false if it cannot: then the usual path answers and nothing that
 * matters has been touched.
 */
bool ReadWithCache(FileHandle* fh, uint32_t id, std::span<uint8_t> buffer, size_t byte_offset,
                  size_t* out_bytes_read) {
  const size_t limit = static_cast<size_t>(REXCVAR_GET(sa2_io_cache_mb)) * 1024u * 1024u;
  const size_t request = buffer.size();
  if (!limit || !request || !id || g_cache_no_memory) {
    return false;
  }
  const uint64_t first = byte_offset / kBlockCache;
  const uint64_t last = (byte_offset + request - 1) / kBlockCache;
  // A request larger than half the cache would fill it on its own and evict everything useful.
  if ((last - first + 1) * kBlockCache > limit / 2) {
    return false;
  }

  std::lock_guard lock(g_cache_mutex);
  size_t copied = 0;
  for (uint64_t b = first; b <= last; ++b) {
    const uint64_t key = (static_cast<uint64_t>(id) << 32) | b;
    auto it = g_cache.find(key);
    if (it == g_cache.end()) {
      // Contiguous run of missing blocks: a single read for all of them.
      uint64_t end = b;
      while (end < last && !g_cache.count((static_cast<uint64_t>(id) << 32) | (end + 1))) {
        ++end;
      }
      const size_t span = static_cast<size_t>(end - b + 1) * kBlockCache;
      size_t read = 0;
      /*
       * If the console does not have that RAM, the cache turns itself off and does not try again:
       * the game carries on down the usual path. It can never bring the process down for lack of memory.
       */
      try {
        std::vector<uint8_t> tmp(span);
        /*
         * Same as DirectRead: in pieces, and a short read is not end of file (the file system
         * truncates large ones). The loop only stops when a piece returns 0 bytes. Without this,
         * half-filled blocks would be cached and the game would get incomplete data.
         */
        const int32_t chunk_mb = REXCVAR_GET(sa2_io_chunk_mb);
        const size_t chunk = chunk_mb > 0 ? static_cast<size_t>(chunk_mb) * 1024u * 1024u : span;
        while (read < span) {
          size_t n = 0;
          const size_t requests = std::min(chunk, span - read);
          if (!fh->Read(b * kBlockCache + read, tmp.data() + read, requests, &n)) {
            if (read == 0) {
              return false;  // real error: let the usual path answer
            }
            break;
          }
          if (n == 0) {
            break;  // end of file
          }
          read += n;
        }
        ++g_cache_failures;
        g_cache_bytes_disk += read;
        for (uint64_t k = b; k <= end; ++k) {
          const size_t from = static_cast<size_t>(k - b) * kBlockCache;
          if (from >= read) {
            break;  // end of file
          }
          const size_t n = std::min(kBlockCache, read - from);
          const uint64_t ck = (static_cast<uint64_t>(id) << 32) | k;
          BlockCache& new_value = g_cache[ck];
          new_value.data.assign(tmp.begin() + from, tmp.begin() + from + n);
          g_cache_bytes += n;
          TouchBlock(ck, new_value);
        }
      } catch (const std::bad_alloc&) {
        const uint64_t had = g_cache_bytes >> 20;
        g_cache_no_memory = true;
        g_cache.clear();
        g_cache_lru.clear();
        g_cache_bytes = 0;
        REXLOG_WARN("[io] RAM cache: out of memory, switching it off (it was at {} MB). The game keeps reading "
                    "from disk as before",
                    had);
        return false;
      }
      PruneCache(limit);
      it = g_cache.find(key);  // the insertion may have rebuilt the table
      if (it == g_cache.end()) {
        break;  // not even the first block had bytes: end of file
      }
    } else {
      ++g_cache_hits;
      TouchBlock(key, it->second);
    }
    const size_t inside = byte_offset + copied - static_cast<size_t>(b * kBlockCache);
    if (inside >= it->second.data.size()) {
      break;  // the block ends before the requested range: end of file
    }
    const size_t n = std::min(it->second.data.size() - inside, request - copied);
    std::memcpy(buffer.data() + copied, it->second.data.data() + inside, n);
    copied += n;
    if (it->second.data.size() < kBlockCache) {
      break;  // short block: no more file
    }
  }
  g_cache_bytes_served += copied;
  *out_bytes_read = copied;

  const uint64_t laps = g_cache_hits + g_cache_failures;
  if (laps >= g_cache_report + 200) {
    g_cache_report = laps;
    REXLOG_INFO("[io] RAM cache: {} blocks ({} MB of {}), {} served from RAM and {} from disk "
                "({:.1f} % hits); {:.1f} MB delivered, {:.1f} read from the SD, {} blocks "
                "evicted",
                g_cache.size(), g_cache_bytes >> 20, REXCVAR_GET(sa2_io_cache_mb), g_cache_hits,
                g_cache_failures, laps ? 100.0 * double(g_cache_hits) / double(laps) : 0.0,
                double(g_cache_bytes_served) / 1048576.0, double(g_cache_bytes_disk) / 1048576.0,
                g_cache_evicted);
  }
  return true;
}

/*
 * ================================================================================================
 * The exact-range cache. See sa2_io_ranges_mb above.
 *
 * The table is a vector and is scanned in full. This is deliberate: it was sized for a 32 MB cap
 * and a 4 MB floor per entry, where at most eight entries fit and the linear search is eight
 * integer comparisons (the current 64 MB cap and 256 KB floor allow up to 256). A hash map would
 * only add a hash function, a rehash that invalidates references and a possible collision that
 * would serve another file's data.
 * ================================================================================================
 */
struct RangeEntry {
  uint32_t id = 0;               // file (stable id per path, the same as IdOfPath)
  uint64_t displacement = 0;   // where the read starts
  uint32_t request = 0;           // how much the game asked for: part of the key, not a detail
  // The bytes the disk returned. It can be shorter than `request` if the file ends earlier; it is
  // stored as is and returned as is, which is what the disk would have answered.
  std::shared_ptr<const std::vector<uint8_t>> data;
  uint64_t usage = 0;              // for the LRU
};

std::mutex g_ranges_mutex;
std::vector<RangeEntry> g_ranges;
uint64_t g_ranges_usage = 0;
uint64_t g_ranges_bytes = 0;
bool g_ranges_no_memory = false;  // turned itself off for lack of RAM: not retried

// Counters. Atomic so the periodic summary can read them without taking the lock.
std::atomic<uint64_t> g_ranges_hits{0};
std::atomic<uint64_t> g_ranges_failures{0};
std::atomic<uint64_t> g_ranges_bytes_ram{0};
std::atomic<uint64_t> g_ranges_bytes_disk{0};
std::atomic<uint64_t> g_ranges_evictions{0};
std::atomic<uint64_t> g_ranges_inputs{0};
std::atomic<uint64_t> g_ranges_bytes_live{0};
// Why something is not cached. Without this the cache could not be tuned: the counter said how
// often it hit, but not whether the misses came from the floor, the ceiling or the load sweep.
std::atomic<uint64_t> g_low_ranges_floor{0};     // reads skipped for being too small
std::atomic<uint64_t> g_ranges_over_ceiling{0};    // skipped for being too large
std::atomic<uint64_t> g_sequential_ranges{0};   // rejected as part of the load sweep
std::atomic<uint64_t> g_sequential_ranges_mb{0};

// Last byte read for each file, to detect the sequential sweep. Guarded by g_ranges_mutex.
std::unordered_map<uint32_t, uint64_t> g_last_ranges_end;

int64_t FloorBytes() {
  const int64_t old_mb = REXCVAR_GET(sa2_io_ranges_min_mb);
  if (old_mb > 0) {
    return old_mb * 1024 * 1024;  // an old toml sets it: honor it
  }
  return int64_t(REXCVAR_GET(sa2_io_ranges_min_kb)) * 1024;
}

// Does this read go into the cache? Only the size decides; the constructor already took care of
// the file being read-only when it gave (or did not give) it an id.
bool EligibleRange(size_t request) {
  const int64_t cap = int64_t(REXCVAR_GET(sa2_io_ranges_mb)) * 1024 * 1024;
  if (cap <= 0) {
    return false;
  }
  const int64_t ceiling = int64_t(REXCVAR_GET(sa2_io_ranges_max_mb)) * 1024 * 1024;
  if (int64_t(request) < FloorBytes()) {
    g_low_ranges_floor.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (int64_t(request) > ceiling || int64_t(request) > cap) {
    g_ranges_over_ceiling.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

/*
 * The level load sweep: large reads whose offset is exactly the end of the previous read of the
 * same file. Measured: 31 consecutive blocks of a large streaming file of exactly 4 MB, each read only once,
 * filled the LRU with garbage before the race started (the first hit came 100 s after the start).
 *
 * Bounded so as not to drop anything that is reread: it only counts as a sweep if the read is also
 * 4 MB or more. What the game rereads during a race is between 0.14 and 2.0 MB, so it never falls
 * here. The end of every eligible read is recorded (including the first read of the sweep, which
 * is stored: it is 1 block out of 31 and gets evicted on its own).
 */
constexpr uint64_t kMinSweep = uint64_t(4) << 20;

bool IsSweepAndNote(uint32_t id, uint64_t displacement, uint32_t request) {
  // Call with g_ranges_mutex held.
  auto it = g_last_ranges_end.find(id);
  const bool followed = it != g_last_ranges_end.end() && it->second == displacement;
  g_last_ranges_end[id] = displacement + request;
  return followed && request >= kMinSweep;
}

/*
 * Looks up the exact range. Returns the buffer, or nullptr if it was not there.
 *
 * It returns a shared_ptr on purpose: the caller releases the lock before copying the megabytes,
 * and while it copies another thread may evict this entry. With the shared_ptr that only removes
 * the entry from the table; the bytes stay alive until the last reader releases them.
 */
std::shared_ptr<const std::vector<uint8_t>> SearchRange(uint32_t id, uint64_t displacement,
                                                        uint32_t request) {
  std::lock_guard lock(g_ranges_mutex);
  for (auto& e : g_ranges) {
    if (e.id == id && e.displacement == displacement && e.request == request) {
      e.usage = ++g_ranges_usage;
      return e.data;
    }
  }
  return nullptr;
}

/*
 * Stores what the disk just returned. Called after the read, with the lock released for the whole
 * trip to the SD.
 *
 * The copy is done outside the lock: it is up to 12 MB of memcpy plus a heap allocation, and
 * neither has any reason to block another thread that only wants to look at the table.
 */
void SaveRange(uint32_t id, uint64_t displacement, uint32_t request, const uint8_t* data,
                  size_t bytes) {
  if (!bytes) {
    return;
  }
  {
    std::lock_guard lock(g_ranges_mutex);
    if (g_ranges_no_memory) {
      return;
    }
    // The load sweep does not go in. Decided here, before copying the megabytes.
    if (IsSweepAndNote(id, displacement, request)) {
      g_sequential_ranges.fetch_add(1, std::memory_order_relaxed);
      g_sequential_ranges_mb.fetch_add(bytes, std::memory_order_relaxed);
      return;
    }
  }

  std::shared_ptr<std::vector<uint8_t>> copy;
  try {
    copy = std::make_shared<std::vector<uint8_t>>(data, data + bytes);
  } catch (const std::bad_alloc&) {
    // If the console does not have that RAM, the cache turns off and is not retried. The game keeps
    // reading from disk as usual: it can never bring the process down for lack of memory.
    uint64_t had = 0;
    {
      std::lock_guard lock(g_ranges_mutex);
      had = g_ranges_bytes >> 20;
      g_ranges_no_memory = true;
      g_ranges.clear();
      g_ranges_bytes = 0;
    }
    g_ranges_inputs.store(0, std::memory_order_relaxed);
    g_ranges_bytes_live.store(0, std::memory_order_relaxed);
    REXLOG_WARN("[io] range cache: out of memory, switching it off (it was at {} MB). The game keeps reading "
                "from disk as before",
                had);
    return;
  }

  const size_t cap = size_t(REXCVAR_GET(sa2_io_ranges_mb)) * 1024u * 1024u;
  std::lock_guard lock(g_ranges_mutex);
  for (const auto& e : g_ranges) {
    if (e.id == id && e.displacement == displacement && e.request == request) {
      return;  // another thread asked for the same range and got there first: its copy is just as good
    }
  }
  RangeEntry& new_entry = g_ranges.emplace_back();
  new_entry.id = id;
  new_entry.displacement = displacement;
  new_entry.request = request;
  new_entry.data = std::move(copy);
  new_entry.usage = ++g_ranges_usage;
  g_ranges_bytes += bytes;

  // LRU: evict the oldest until it fits. The entry just added has the highest use count, so it is
  // never the one evicted.
  while (g_ranges_bytes > cap && g_ranges.size() > 1) {
    size_t old = 0;
    for (size_t i = 1; i < g_ranges.size(); ++i) {
      if (g_ranges[i].usage < g_ranges[old].usage) {
        old = i;
      }
    }
    g_ranges_bytes -= g_ranges[old].data->size();
    g_ranges.erase(g_ranges.begin() + static_cast<ptrdiff_t>(old));
    g_ranges_evictions.fetch_add(1, std::memory_order_relaxed);
  }
  g_ranges_inputs.store(g_ranges.size(), std::memory_order_relaxed);
  g_ranges_bytes_live.store(g_ranges_bytes, std::memory_order_relaxed);
}
}  // namespace

StatsRanges ReadStatsRanges() {
  StatsRanges e;
  e.hits = g_ranges_hits.load(std::memory_order_relaxed);
  e.failures = g_ranges_failures.load(std::memory_order_relaxed);
  e.bytes_ram = g_ranges_bytes_ram.load(std::memory_order_relaxed);
  e.bytes_disk = g_ranges_bytes_disk.load(std::memory_order_relaxed);
  e.inputs = g_ranges_inputs.load(std::memory_order_relaxed);
  e.bytes_live = g_ranges_bytes_live.load(std::memory_order_relaxed);
  e.cap_drops = g_ranges_evictions.load(std::memory_order_relaxed);
  e.low_floor = g_low_ranges_floor.load(std::memory_order_relaxed);
  e.over_ceiling = g_ranges_over_ceiling.load(std::memory_order_relaxed);
  e.sequential = g_sequential_ranges.load(std::memory_order_relaxed);
  e.sequential_bytes = g_sequential_ranges_mb.load(std::memory_order_relaxed);
  e.floor_kb = uint64_t(FloorBytes() / 1024);
  const int32_t cap = REXCVAR_GET(sa2_io_ranges_mb);
  e.cap_mb = cap > 0 ? uint64_t(cap) : 0;
  {
    std::lock_guard lock(g_ranges_mutex);
    e.no_memory = g_ranges_no_memory;
  }
  return e;
}

StatsWindow ReadStatsWindow() {
  StatsWindow e;
  e.hits = g_hits.load(std::memory_order_relaxed);
  e.filled = g_filled.load(std::memory_order_relaxed);
  e.direct = g_direct.load(std::memory_order_relaxed);
  e.bytes_ram = g_bytes_ram.load(std::memory_order_relaxed);
  int64_t live = g_live_windows.load(std::memory_order_relaxed);
  e.live_windows = live > 0 ? static_cast<uint64_t>(live) : 0;
  return e;
}

HostPathFile::HostPathFile(uint32_t file_access, HostPathEntry* entry,
                           std::unique_ptr<rex::filesystem::FileHandle> file_handle)
    : File(file_access, entry), file_handle_(std::move(file_handle)) {
  // The window is decided once, at open time. Conditions: there is a handle (it is not a directory),
  // the device is read-only, the file was not opened for writing and the cvar allows it.
  const int32_t kb = REXCVAR_GET(sa2_io_window_kb);
  const bool wants_write =
      (file_access & (FileAccess::kGenericWrite | FileAccess::kFileWriteData |
                      FileAccess::kFileAppendData | FileAccess::kGenericAll)) != 0;
  /*
   * The RAM cache, with the same conditions as the window except that it has its own cvar. The id
   * is requested once, at open time, so the path map is not touched on every read.
   *
   * The same id serves both caches, the block one (off) and the range one. That the device is
   * read-only and the file was not opened for writing is checked here and only here: that is why
   * neither cache can ever touch a save or a profile, where the bytes change underneath us and
   * serving a stale copy would mean a corrupt save.
   */
  if (file_handle_ && entry && entry->is_read_only() && !wants_write &&
      (REXCVAR_GET(sa2_io_cache_mb) > 0 || REXCVAR_GET(sa2_io_ranges_mb) > 0)) {
    cache_id_ = IdOfPath(entry->path());
  }
  // Startup trace / preload (startup_trace.cpp): same read-only condition, so saves and profiles
  // are never recorded or served from the preload store.
  if (file_handle_ && entry && entry->is_read_only() && !wants_write) {
    trace_tag_ = startup_trace::OnOpen(static_cast<HostPathDevice*>(entry->device()), entry->path());
  }
  if (file_handle_ && kb > 0 && entry && entry->is_read_only() && !wants_write) {
    const int64_t max = REXCVAR_GET(sa2_io_windows_max);
    if (g_live_windows.fetch_add(1, std::memory_order_relaxed) < max) {
      window_size_ = static_cast<size_t>(kb) * 1024u;
      active_window_ = true;
      counted_window_ = true;
    } else {
      g_live_windows.fetch_sub(1, std::memory_order_relaxed);
    }
  }
}

HostPathFile::~HostPathFile() {
  if (counted_window_) {
    g_live_windows.fetch_sub(1, std::memory_order_relaxed);
  }
}

void HostPathFile::Destroy() {
  delete this;
}

// fsync on POSIX, FlushFileBuffers on Windows. Called by NtFlushBuffersFile, which the game uses right
// after saving; without it, the save could be left half written on the SD if the game was closed
// right afterwards.
X_STATUS HostPathFile::Flush() {
  if (!file_handle_) {
    return X_STATUS_SUCCESS;
  }
  file_handle_->Flush();
  return X_STATUS_SUCCESS;
}

/*
 * Horizon's file system refuses to rename a file that is still open, and games rename through the
 * open handle: Mass Effect writes Save_GamerProfile_*.tmp and renames it to GamerProfile.sav before
 * closing it. Measured on the console: "RenameEntryInternal: failed to rename ... .tmp to
 * GamerProfile.sav", then the game deleted the .tmp and the profile was never saved. On Switch the
 * host handle is closed around the rename and reopened on the new path with the same access.
 */
X_STATUS HostPathFile::Rename(const std::filesystem::path& file_path) {
#if REX_PLATFORM_SWITCH
  if (file_handle_) {
    file_handle_.reset();
    const X_STATUS status = entry()->Rename(file_path);
    auto* host_entry = static_cast<HostPathEntry*>(entry());
    file_handle_ = rex::filesystem::FileHandle::OpenExisting(
        host_entry->host_path(), file_access(),
        static_cast<HostPathDevice*>(host_entry->device())->allow_share_delete());
    if (!file_handle_) {
      REXFS_ERROR("HostPathFile::Rename: could not reopen '{}' after rename",
                  rex::path_to_utf8(host_entry->host_path()));
    }
    return status;
  }
#endif
  return entry()->Rename(file_path);
}

bool HostPathFile::FillWindow(size_t byte_offset, size_t request, std::span<uint8_t> buffer,
                                   size_t* out_bytes_read, X_STATUS* out_status) {
  if (window_.size() != window_size_) {
    window_.resize(window_size_);
  }

  size_t read = 0;
  ++filled_window_;
  g_filled.fetch_add(1, std::memory_order_relaxed);
  if (!file_handle_->Read(byte_offset, window_.data(), window_size_, &read)) {
    // Real error. The window is turned off and the normal path answers.
    window_bytes_ = 0;
    active_window_ = false;
    window_.clear();
    window_.shrink_to_fit();
    return false;
  }

  // What was read is valid whether or not it covers the request: they are real bytes of the file and
  // the file is read-only.
  window_start_ = byte_offset;
  window_bytes_ = read;

  if (read < request) {
    /*
     * The window does not cover the request. It can be the end of the file (and then the direct read
     * will return the same) or a short read from the file system, which cannot be told apart here. In
     * both cases the answer that cannot be wrong is the usual one: reply with the direct read. If this
     * happens several times in a row, the file system is truncating large reads, and the window would
     * only add an extra trip per read: it is turned off for this file.
     */
    if (++short_window_ >= 4) {
      active_window_ = false;
      window_bytes_ = 0;
      window_.clear();
      window_.shrink_to_fit();
    }
    return false;
  }
  short_window_ = 0;

  std::memcpy(buffer.data(), window_.data(), request);
  *out_bytes_read = request;
  *out_status = X_STATUS_SUCCESS;
  g_bytes_ram.fetch_add(request, std::memory_order_relaxed);
  return true;
}

/*
 * The direct read, split into pieces. See sa2_io_chunk_mb above.
 *
 * It must return exactly the same as a single pread in every case the game can see: success with
 * whatever count results, and X_STATUS_END_OF_FILE only if the first call really fails.
 */
X_STATUS HostPathFile::DirectRead(std::span<uint8_t> buffer, size_t byte_offset,
                                   size_t* out_bytes_read) {
  const size_t request = buffer.size();
  const int32_t chunk_mb = REXCVAR_GET(sa2_io_chunk_mb);
  const size_t chunk = chunk_mb > 0 ? static_cast<size_t>(chunk_mb) * 1024u * 1024u : 0;

  if (!chunk || request <= chunk) {
    if (file_handle_->Read(byte_offset, buffer.data(), request, out_bytes_read)) {
      return X_STATUS_SUCCESS;
    }
    *out_bytes_read = 0;
    return X_STATUS_END_OF_FILE;
  }

  size_t total = 0;
  while (total < request) {
    const size_t n = std::min(chunk, request - total);
    size_t read = 0;
    if (!file_handle_->Read(byte_offset + total, buffer.data() + total, n, &read)) {
      // Real error. If there were already good bytes they are delivered; if not, it is the usual failure.
      if (total == 0) {
        *out_bytes_read = 0;
        return X_STATUS_END_OF_FILE;
      }
      break;
    }
    if (read == 0) {
      // End of file. A short but non-empty piece does not stop the loop: it asks again from where it
      // stopped, which is better than a single pread (that returned the short read and nothing more).
      break;
    }
    total += read;
  }

  *out_bytes_read = total;
  return X_STATUS_SUCCESS;
}

X_STATUS HostPathFile::ReadSync(std::span<uint8_t> buffer, size_t byte_offset,
                                size_t* out_bytes_read) {
  // No handle means a directory (see HostPathEntry::Open). A directory is enumerated, not read.
  if (!file_handle_) {
    return X_STATUS_FILE_IS_A_DIRECTORY;
  }
  if (!(file_access_ & (FileAccess::kGenericRead | FileAccess::kFileReadData))) {
    return X_STATUS_ACCESS_DENIED;
  }

  const size_t request = buffer.size();
  // Records the read (record mode) or serves it from the preload store (preload mode).
  if (trace_tag_ && startup_trace::OnRead(trace_tag_, byte_offset, buffer, out_bytes_read)) {
    if (startup_trace::VerifyEnabled()) {
      std::vector<uint8_t> check(request);
      size_t n = 0;
      const bool read = file_handle_->Read(byte_offset, check.data(), request, &n);
      startup_trace::ReportVerify(trace_tag_, byte_offset, request,
                                  read && n == *out_bytes_read &&
                                      std::memcmp(check.data(), buffer.data(), n) == 0);
    }
    return X_STATUS_SUCCESS;
  }
  if (active_window_ && request && request <= window_size_ / kSubmissionMaxFraction) {
    // The request falls entirely within what is already in RAM.
    if (window_bytes_ && byte_offset >= window_start_ &&
        byte_offset + request <= window_start_ + window_bytes_) {
      std::memcpy(buffer.data(), window_.data() + (byte_offset - window_start_), request);
      *out_bytes_read = request;
      ++window_hits_;
      g_hits.fetch_add(1, std::memory_order_relaxed);
      g_bytes_ram.fetch_add(request, std::memory_order_relaxed);
      return X_STATUS_SUCCESS;
    }

    X_STATUS state = X_STATUS_SUCCESS;
    const bool served = FillWindow(byte_offset, request, buffer, out_bytes_read, &state);

    /*
     * Audit. Each fill is a 256 KB read from the SD that only pays off if hits follow. If after
     * 8 fills this file has not given at least one hit per fill, the game is not reading it
     * sequentially and the window is pure cost: it is turned off for this file. This prevents a
     * repeat of the measured case where 4.1 MB of SD reads served 0.3 MB without anyone noticing.
     */
    if (active_window_ && filled_window_ >= 8 && window_hits_ < filled_window_) {
      active_window_ = false;
      window_bytes_ = 0;
      window_.clear();
      window_.shrink_to_fit();
    }

    if (served) {
      return state;
    }
    // If it could not, it falls through to the usual path below.
  }

  /*
   * The exact-range cache, before going to disk. See sa2_io_ranges_mb.
   *
   * It is the only thing between the game and the SD for large reads: the zone pack that the game
   * releases on leaving and asks for again on entering, byte for byte, at the same place and with
   * the same size. Those rereads are the ones that block for 11 to 140 ms in the middle of a lap.
   */
  const bool eligible_range = cache_id_ && EligibleRange(request);
  if (eligible_range) {
    // The lock only lasts as long as the table lookup; the memcpy happens outside it, and the
    // shared_ptr keeps the data alive even if another thread evicts this entry while we copy.
    if (auto data = SearchRange(cache_id_, byte_offset, uint32_t(request))) {
      const size_t n = std::min(data->size(), request);
      std::memcpy(buffer.data(), data->data(), n);
      *out_bytes_read = n;
      g_ranges_hits.fetch_add(1, std::memory_order_relaxed);
      g_ranges_bytes_ram.fetch_add(n, std::memory_order_relaxed);
      return X_STATUS_SUCCESS;
    }
  }

  // The RAM cache, before going to disk. See sa2_io_cache_mb: the game rereads the same data every
  // lap, and those reads are what make a facade take a while to get its proper texture.
  if (cache_id_ && request) {
    size_t read = 0;
    if (ReadWithCache(file_handle_.get(), cache_id_, buffer, byte_offset, &read)) {
      *out_bytes_read = read;
      return X_STATUS_SUCCESS;
    }
  }

  g_direct.fetch_add(1, std::memory_order_relaxed);
  const X_STATUS state = DirectRead(buffer, byte_offset, out_bytes_read);

  /*
   * And here what just arrived is stored. None of this happens with a shared lock held: the read
   * above has already finished. That exact mistake (g_cache_mutex held inside the pread) is what
   * made the block cache block the other threads without showing in any counter.
   */
  if (eligible_range && state == X_STATUS_SUCCESS && *out_bytes_read) {
    g_ranges_failures.fetch_add(1, std::memory_order_relaxed);
    g_ranges_bytes_disk.fetch_add(*out_bytes_read, std::memory_order_relaxed);
    SaveRange(cache_id_, byte_offset, uint32_t(request), buffer.data(), *out_bytes_read);
  }
  return state;
}

X_STATUS HostPathFile::WriteSync(std::span<const uint8_t> buffer, size_t byte_offset,
                                 size_t* out_bytes_written) {
  if (!file_handle_) {
    return X_STATUS_FILE_IS_A_DIRECTORY;
  }
  if (!(file_access_ &
        (FileAccess::kGenericWrite | FileAccess::kFileWriteData | FileAccess::kFileAppendData))) {
    return X_STATUS_ACCESS_DENIED;
  }

  // Safety belt: a file being written cannot have a window. It should never get here with one (it is
  // only enabled on read-only devices and without write access), but if it did, the window is dropped.
  active_window_ = false;
  window_bytes_ = 0;

  if (file_handle_->Write(byte_offset, buffer.data(), buffer.size(), out_bytes_written)) {
    return X_STATUS_SUCCESS;
  } else {
    return X_STATUS_END_OF_FILE;
  }
}

X_STATUS HostPathFile::SetLength(size_t length) {
  if (!file_handle_) {
    return X_STATUS_FILE_IS_A_DIRECTORY;
  }
  if (!(file_access_ & (FileAccess::kGenericWrite | FileAccess::kFileWriteData))) {
    return X_STATUS_ACCESS_DENIED;
  }

  active_window_ = false;
  window_bytes_ = 0;

  if (file_handle_->SetLength(length)) {
    return X_STATUS_SUCCESS;
  } else {
    return X_STATUS_END_OF_FILE;
  }
}

}  // namespace rex::filesystem
