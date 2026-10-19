/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

// Disable warnings about unused parameters for kernel functions
#pragma GCC diagnostic ignored "-Wunused-parameter"

#include <atomic>
#include <chrono>
#include <thread>

#include <rex/cvar.h>
#include <rex/filesystem/device.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/filesystem/devices/host_path_file.h>
#include <rex/kernel/xboxkrnl/private.h>
#include <rex/logging.h>
#include <rex/memory.h>
#include <rex/hook.h>
#include <rex/types.h>
#include <rex/system/info/file.h>
#include <rex/system/kernel_state.h>
#include <rex/system/util/string_utils.h>
#include <rex/system/xevent.h>
#include <rex/system/xfile.h>
#include <rex/system/xiocompletion.h>
#include <rex/system/xsymboliclink.h>
#include <rex/system/xthread.h>
#include <rex/system/xtypes.h>
#include <rex/thread/mutex.h>

#include <cctype>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

/*
 * I/O measurement.
 *
 * The log only showed failures (they are WARN); everything else (how many opens there are, how long
 * each one takes, how much is read) went to TRACE and did not show. And that is where the time was:
 * failing opens cost 32 ms each. So every open and every read is timed, the ones over the limit are
 * reported, and a summary is written every few seconds.
 *
 * A monotonic clock read per call is a few nanoseconds next to a trip to the SD. It is on.
 */
REXCVAR_DEFINE_INT32(sa2_io_warning_ms, 8, "Filesystem",
                     "Logs a warning for every open or read that takes more than this many ms (0 = never).");
REXCVAR_DEFINE_INT32(sa2_io_us_per_kb, 0, "Filesystem",
                     "Diagnostic (Mac): every NtReadFile sleeps this many microseconds per KiB read after the read, to "
                     "emulate the Switch SD card (~100 = 13 ms per 128 KiB); 0 = off.");
REXCVAR_DEFINE_INT32(sa2_io_summary_s, 15, "Filesystem",
                     "How many seconds between [io] summaries (0 = never).");

namespace rex::kernel::xboxkrnl {
using namespace rex::system;

namespace {

struct CountersIo {
  std::atomic<uint64_t> opens_ok{0};
  std::atomic<uint64_t> opens_failure{0};
  std::atomic<uint64_t> us_opens{0};
  std::atomic<uint64_t> us_opening_worst{0};
  std::atomic<uint64_t> reads{0};
  std::atomic<uint64_t> bytes_read{0};
  std::atomic<uint64_t> us_reads{0};
  std::atomic<uint64_t> us_read_worst{0};
};

CountersIo g_io;
std::atomic<uint32_t> g_io_warnings{0};
std::atomic<uint64_t> g_io_next_summary{0};

/*
 * Which part of what is read had already been read, and whether it is read sequentially.
 *
 * Logging only how many bytes and from which file, not from where, the most that could be checked
 * was whether the (file, size) pair repeated, and it did: 270.2 MB of the 364.1 MB of slow reads
 * were repeats, and a large streaming file repeated the same sizes every ~90 s, which is one pass over the
 * route. But "same size" is not "same place": the same size also comes from a sequential sweep
 * (the 31 consecutive 4.00 MB reads at startup are exactly that). Without the offset the two cannot
 * be told apart, and whether a block cache would remove reads depends on it.
 *
 * So the exact range is recorded. Two fixed-size tables, with no allocation or growth:
 *  - ranges already seen (8,192 slots of 8 bytes = 64 KB), to count rereads of the same place;
 *  - end of the last read of each file (64 slots), to count the reads that start exactly where the
 *    previous one ended, which is what tells whether the read-ahead window has anything to do.
 *
 * It costs a hash of the name and a lock per read. That is 1,290 reads in 435 s: nothing.
 */
struct RangesState {
  static constexpr size_t kSlots = 8192;  // power of two
  static constexpr size_t kFiles = 64;
  static constexpr size_t kPolls = 8;

  std::mutex mutex;
  uint64_t key[kSlots] = {};
  uint64_t file_ids[kFiles] = {};
  uint64_t end[kFiles] = {};
  // Totals for the whole session; the summary reports the difference from the previous snapshot.
  uint64_t rereads = 0, bytes_reread = 0;
  uint64_t distinct = 0, bytes_distinct = 0;
  uint64_t consecutive = 0, evictions = 0;
};
RangesState g_ranges;

inline uint64_t HashPath(const std::string_view s) {
  uint64_t h = 1469598103934665603ull;
  for (unsigned char c : s) {
    h ^= c;
    h *= 1099511628211ull;
  }
  return h;
}

// Records the range read. Returns true if that exact range had already been read before.
bool NoteRange(const std::string_view path, uint64_t displacement, uint32_t bytes) {
  if (!bytes) {
    return false;
  }
  const uint64_t hr = HashPath(path);
  uint64_t key = hr ^ (displacement * 0x9E3779B97F4A7C15ull) ^ (uint64_t(bytes) << 1);
  if (key == 0) {
    key = 1;  // 0 marks a free slot
  }

  std::lock_guard<std::mutex> lock(g_ranges.mutex);

  // Sequential read: it starts exactly where the previous read of this same file ended.
  const size_t fi = static_cast<size_t>(hr % RangesState::kFiles);
  if (g_ranges.file_ids[fi] == hr && g_ranges.end[fi] == displacement) {
    ++g_ranges.consecutive;
  }
  g_ranges.file_ids[fi] = hr;
  g_ranges.end[fi] = displacement + bytes;

  // Repeated range. Open addressing with a few probes: if it does not fit, a slot is overwritten
  // and that is counted.
  const size_t base = static_cast<size_t>((key >> 17) & (RangesState::kSlots - 1));
  for (size_t p = 0; p < RangesState::kPolls; ++p) {
    const size_t j = (base + p) & (RangesState::kSlots - 1);
    if (g_ranges.key[j] == key) {
      ++g_ranges.rereads;
      g_ranges.bytes_reread += bytes;
      return true;
    }
    if (g_ranges.key[j] == 0) {
      g_ranges.key[j] = key;
      ++g_ranges.distinct;
      g_ranges.bytes_distinct += bytes;
      return false;
    }
  }
  g_ranges.key[base] = key;
  ++g_ranges.distinct;
  g_ranges.bytes_distinct += bytes;
  ++g_ranges.evictions;  // if this grows, the table is too small and the figures fall short too
  return false;
}

// Snapshot of the previous summary, to report the interval and not the whole-session total.
struct SnapshotIo {
  uint64_t opens_ok = 0, opens_failure = 0, us_opens = 0;
  uint64_t reads = 0, bytes_read = 0, us_reads = 0;
  uint64_t misses_in_dry = 0, stats_in_sd = 0, swept_in_sd = 0;
  uint64_t hits = 0, filled = 0, direct = 0, bytes_ram = 0;
  uint64_t rereads = 0, bytes_reread = 0, distinct = 0, bytes_distinct = 0, consecutive = 0;
};
SnapshotIo g_io_snapshot;
std::mutex g_io_snapshot_mutex;

inline uint64_t NowUs() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

inline void MaxAtomic(std::atomic<uint64_t>& target, uint64_t value) {
  uint64_t seen = target.load(std::memory_order_relaxed);
  while (value > seen &&
         !target.compare_exchange_weak(seen, value, std::memory_order_relaxed)) {
  }
}

// A single thread writes the summary: whichever wins the exchange of the next deadline.
void MaybeSummaryIo(uint64_t now_us) {
  const int32_t period_s = REXCVAR_GET(sa2_io_summary_s);
  if (period_s <= 0) {
    return;
  }
  uint64_t next = g_io_next_summary.load(std::memory_order_relaxed);
  const uint64_t step = static_cast<uint64_t>(period_s) * 1000000ull;
  if (next == 0) {
    // First call: only the timer is armed, nothing is written yet.
    g_io_next_summary.compare_exchange_strong(next, now_us + step,
                                                 std::memory_order_relaxed);
    return;
  }
  if (now_us < next) {
    return;
  }
  if (!g_io_next_summary.compare_exchange_strong(next, now_us + step,
                                                    std::memory_order_relaxed)) {
    return;
  }

  const auto paths = rex::filesystem::ReadStatsPaths();
  const auto window = rex::filesystem::ReadStatsWindow();

  SnapshotIo now;
  now.opens_ok = g_io.opens_ok.load(std::memory_order_relaxed);
  now.opens_failure = g_io.opens_failure.load(std::memory_order_relaxed);
  now.us_opens = g_io.us_opens.load(std::memory_order_relaxed);
  now.reads = g_io.reads.load(std::memory_order_relaxed);
  now.bytes_read = g_io.bytes_read.load(std::memory_order_relaxed);
  now.us_reads = g_io.us_reads.load(std::memory_order_relaxed);
  now.misses_in_dry = paths.misses_in_dry;
  now.stats_in_sd = paths.stats_in_sd;
  now.swept_in_sd = paths.swept_in_sd;
  now.hits = window.hits;
  now.filled = window.filled;
  now.direct = window.direct;
  now.bytes_ram = window.bytes_ram;
  uint64_t evictions = 0;
  {
    std::lock_guard<std::mutex> lock(g_ranges.mutex);
    now.rereads = g_ranges.rereads;
    now.bytes_reread = g_ranges.bytes_reread;
    now.distinct = g_ranges.distinct;
    now.bytes_distinct = g_ranges.bytes_distinct;
    now.consecutive = g_ranges.consecutive;
    evictions = g_ranges.evictions;
  }

  SnapshotIo d;
  {
    std::lock_guard<std::mutex> lock(g_io_snapshot_mutex);
    d.opens_ok = now.opens_ok - g_io_snapshot.opens_ok;
    d.opens_failure = now.opens_failure - g_io_snapshot.opens_failure;
    d.us_opens = now.us_opens - g_io_snapshot.us_opens;
    d.reads = now.reads - g_io_snapshot.reads;
    d.bytes_read = now.bytes_read - g_io_snapshot.bytes_read;
    d.us_reads = now.us_reads - g_io_snapshot.us_reads;
    d.misses_in_dry = now.misses_in_dry - g_io_snapshot.misses_in_dry;
    d.stats_in_sd = now.stats_in_sd - g_io_snapshot.stats_in_sd;
    d.swept_in_sd = now.swept_in_sd - g_io_snapshot.swept_in_sd;
    d.hits = now.hits - g_io_snapshot.hits;
    d.filled = now.filled - g_io_snapshot.filled;
    d.direct = now.direct - g_io_snapshot.direct;
    d.bytes_ram = now.bytes_ram - g_io_snapshot.bytes_ram;
    d.rereads = now.rereads - g_io_snapshot.rereads;
    d.bytes_reread = now.bytes_reread - g_io_snapshot.bytes_reread;
    d.distinct = now.distinct - g_io_snapshot.distinct;
    d.bytes_distinct = now.bytes_distinct - g_io_snapshot.bytes_distinct;
    d.consecutive = now.consecutive - g_io_snapshot.consecutive;
    g_io_snapshot = now;
  }

  const uint64_t worst_open = g_io.us_opening_worst.exchange(0, std::memory_order_relaxed);
  const uint64_t worst_read = g_io.us_read_worst.exchange(0, std::memory_order_relaxed);
  const uint64_t opens = d.opens_ok + d.opens_failure;

  REXKRNL_INFO(
      "[io] {} s: {} opens ({} failed) {:.1f} ms in total, worst {:.1f} ms, mean {:.2f} ms; "
      "{} reads {:.1f} MB in {:.1f} ms, worst {:.1f} ms",
      period_s, opens, d.opens_failure, d.us_opens / 1000.0, worst_open / 1000.0,
      opens ? (d.us_opens / 1000.0) / double(opens) : 0.0, d.reads,
      d.bytes_read / (1024.0 * 1024.0), d.us_reads / 1000.0, worst_read / 1000.0);
  REXKRNL_INFO(
      "[io] paths: {} resolved without touching the SD, {} stats, {} directory sweeps ({} entries "
      "in the tree); window: {} hits / {} fills / {} direct, {:.1f} MB from RAM, {} "
      "files with a window",
      d.misses_in_dry, d.stats_in_sd, d.swept_in_sd, paths.inputs_in_tree, d.hits,
      d.filled, d.direct, d.bytes_ram / (1024.0 * 1024.0), window.live_windows);

  /*
   * The line that tells whether a block cache is worth it and whether the window has anything to do.
   *  - "rereads"   = reads whose exact range (file+offset+size) had already been read before.
   *                  A cache would remove those bytes entirely; not the rest.
   *  - "sequential" = reads that start exactly where the previous one of the same file ended. If
   *                  this is high, read-ahead helps; if it is low, the game jumps around and the
   *                  window only gets in the way.
   *  - "evictions" = times the range table ran out of room. If it grows, the figures fall short.
   */
  const uint64_t reads_span = d.rereads + d.distinct;
  REXKRNL_INFO(
      "[io] ranges: {} rereads of {} ({:.0f} %), {:.1f} MB already read before out of {:.1f} MB ({:.0f} %); {} "
      "sequential ({:.0f} %); {} evictions",
      d.rereads, reads_span,
      reads_span ? 100.0 * double(d.rereads) / double(reads_span) : 0.0,
      d.bytes_reread / (1024.0 * 1024.0),
      (d.bytes_reread + d.bytes_distinct) / (1024.0 * 1024.0),
      (d.bytes_reread + d.bytes_distinct)
          ? 100.0 * double(d.bytes_reread) / double(d.bytes_reread + d.bytes_distinct)
          : 0.0,
      d.consecutive, reads_span ? 100.0 * double(d.consecutive) / double(reads_span) : 0.0,
      evictions);

  /*
   * The cache of large reads by exact range (sa2_io_ranges_mb).
   *
   * It is accumulated since startup, not per interval: there are few events and what matters is
   * the session total. And it is always printed, whether it is on or not and whether there was
   * activity or not: there was already a counter that existed and was never printed, and without
   * the line there is no way to tell whether it works.
   *
   * How to read it:
   *  - "evictions" at 0 is what is expected. If it grows, the cap is too small and the LRU
   *    drops entries before reusing them, which is exactly how the block cache died.
   *  - "from the SD" against "from RAM" is the amplification. By exact range it must be 1:1 on
   *    the first lap and go down from there; if more SD were read than delivered, something is
   *    wrong (the block cache read 318 for 304 delivered, and that is why it was turned off).
   */
  const auto ranges = rex::filesystem::ReadStatsRanges();
  const uint64_t ranges_total = ranges.hits + ranges.failures;
  REXKRNL_INFO(
      "[io] range-cache: {} hits / {} misses ({:.0f} %), {:.1f} MB from RAM against {:.1f} MB "
      "from the SD; {} live entries ({:.1f} MB of {} cap), {} evictions; not cached: {} below the "
      "floor of {} KB, {} above the ceiling, {} from the load sweep ({:.1f} MB){}",
      ranges.hits, ranges.failures,
      ranges_total ? 100.0 * double(ranges.hits) / double(ranges_total) : 0.0,
      ranges.bytes_ram / (1024.0 * 1024.0), ranges.bytes_disk / (1024.0 * 1024.0), ranges.inputs,
      ranges.bytes_live / (1024.0 * 1024.0), ranges.cap_mb, ranges.cap_drops, ranges.low_floor,
      ranges.floor_kb, ranges.over_ceiling, ranges.sequential,
      ranges.sequential_bytes / (1024.0 * 1024.0), ranges.no_memory ? " [OFF: out of memory]" : "");
}

// Always measured: reading the monotonic clock on Horizon is a processor register read, not a system call.
inline void NoteOpening(uint64_t us, bool ok, const std::string_view path) {
  if (ok) {
    g_io.opens_ok.fetch_add(1, std::memory_order_relaxed);
  } else {
    g_io.opens_failure.fetch_add(1, std::memory_order_relaxed);
  }
  g_io.us_opens.fetch_add(us, std::memory_order_relaxed);
  MaxAtomic(g_io.us_opening_worst, us);

  const int32_t warning_ms = REXCVAR_GET(sa2_io_warning_ms);
  if (warning_ms > 0 && us >= static_cast<uint64_t>(warning_ms) * 1000ull &&
      g_io_warnings.fetch_add(1, std::memory_order_relaxed) < 300) {
    REXKRNL_WARN("[io] SLOW: open '{}' took {:.1f} ms ({})", path, us / 1000.0,
                 ok ? "opened" : "does not exist");
  }
}

inline void NoteRead(uint64_t us, uint32_t bytes, const std::string_view path,
                          uint64_t displacement) {
  g_io.reads.fetch_add(1, std::memory_order_relaxed);
  g_io.bytes_read.fetch_add(bytes, std::memory_order_relaxed);
  g_io.us_reads.fetch_add(us, std::memory_order_relaxed);
  MaxAtomic(g_io.us_read_worst, us);

  const bool reread = NoteRange(path, displacement, bytes);

  const int32_t warning_ms = REXCVAR_GET(sa2_io_warning_ms);
  if (warning_ms > 0 && us >= static_cast<uint64_t>(warning_ms) * 1000ull &&
      g_io_warnings.fetch_add(1, std::memory_order_relaxed) < 300) {
    // The offset is included. Without it a sequential sweep cannot be told apart from a reread.
    REXKRNL_WARN("[io] SLOW: read {} bytes of '{}' at {} took {:.1f} ms{}", bytes, path,
                 displacement, us / 1000.0, reread ? " (REREAD)" : "");
  }
}

}  // namespace

struct CreateOptions {
  // https://processhacker.sourceforge.io/doc/ntioapi_8h.html
  static const uint32_t FILE_DIRECTORY_FILE = 0x00000001;
  // Optimization - files access will be sequential, not random.
  static const uint32_t FILE_SEQUENTIAL_ONLY = 0x00000004;
  static const uint32_t FILE_SYNCHRONOUS_IO_ALERT = 0x00000010;
  static const uint32_t FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020;
  static const uint32_t FILE_NON_DIRECTORY_FILE = 0x00000040;
  // Optimization - file access will be random, not sequential.
  static const uint32_t FILE_RANDOM_ACCESS = 0x00000800;
};

static bool IsValidPath(const std::string_view s, bool is_pattern) {
  // TODO(gibbed): validate path components individually
  bool got_asterisk = false;
  for (const auto& c : s) {
    if (c <= 31 || c >= 127) {
      return false;
    }
    if (got_asterisk) {
      // * must be followed by a . (*.)
      //
      // 4D530819 has a bug in its game code where it attempts to
      // FindFirstFile() with filters of "Game:\\*_X3.rkv", "Game:\\m*_X3.rkv",
      // and "Game:\\w*_X3.rkv" and will infinite loop if the path filter is
      // allowed.
      if (c != '.') {
        return false;
      }
      got_asterisk = false;
    }
    switch (c) {
      case '"':
      // case '*':
      case '+':
      case ',':
      // case ':':
      // case ';':
      case '<':
      // case '=':
      case '>':
      // case '?':
      case '|': {
        return false;
      }
      case '*': {
        // Pattern-specific (for NtQueryDirectoryFile)
        if (!is_pattern) {
          return false;
        }
        got_asterisk = true;
        break;
      }
      case '?': {
        // Pattern-specific (for NtQueryDirectoryFile)
        if (!is_pattern) {
          return false;
        }
        break;
      }
      default: {
        break;
      }
    }
  }
  return true;
}

u32 NtCreateFile_entry(mapped_u32 handle_out, u32 desired_access,
                       ppc_ptr_t<X_OBJECT_ATTRIBUTES> object_attrs,
                       ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block, mapped_u64 allocation_size_ptr,
                       u32 file_attributes, u32 share_access, u32 creation_disposition,
                       u32 create_options) {
  // note used. maybe later
  // uint64_t allocation_size = 0;  // is this correct???
  // if (allocation_size_ptr) {
  //  allocation_size = *allocation_size_ptr;
  //}

  if (!object_attrs) {
    // ..? Some games do this. This parameter is not optional.
    return X_STATUS_INVALID_PARAMETER;
  }
  assert_not_null(handle_out);

  auto object_name = REX_KERNEL_MEMORY()->TranslateVirtual<X_ANSI_STRING*>(object_attrs->name_ptr);

  rex::filesystem::Entry* root_entry = nullptr;

  // Compute path, possibly attrs relative.
  auto target_path = util::TranslateAnsiPath(REX_KERNEL_MEMORY(), object_name);
  REXKRNL_IMPORT_TRACE(
      "NtCreateFile", "path={} access={:#x} attrs={:#x} share={:#x} disp={:#x} options={:#x}",
      target_path, (uint32_t)desired_access, (uint32_t)file_attributes, (uint32_t)share_access,
      (uint32_t)creation_disposition, (uint32_t)create_options);

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  if (object_attrs->root_directory != 0xFFFFFFFD &&  // ObDosDevices
      object_attrs->root_directory != 0) {
    auto root_file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(object_attrs->root_directory);
    assert_not_null(root_file);
    assert_true(root_file->type() == XObject::Type::File);

    root_entry = root_file->entry();
  }

  // Attempt open (or create).
  rex::filesystem::File* vfs_file;
  rex::filesystem::FileAction file_action;
  const uint64_t io_us_start = NowUs();
  X_STATUS result = REX_KERNEL_FS()->OpenFile(
      root_entry, target_path, rex::filesystem::FileDisposition((uint32_t)creation_disposition),
      desired_access, (create_options & CreateOptions::FILE_DIRECTORY_FILE) != 0,
      (create_options & CreateOptions::FILE_NON_DIRECTORY_FILE) != 0, &vfs_file, &file_action);
  const uint64_t io_us_end = NowUs();
  NoteOpening(io_us_end - io_us_start, XSUCCEEDED(result), target_path);
  MaybeSummaryIo(io_us_end);
  object_ref<XFile> file = nullptr;

  X_HANDLE handle = X_INVALID_HANDLE_VALUE;
  if (XSUCCEEDED(result)) {
    // If true, desired_access SYNCHRONIZE flag must be set.
    bool synchronous = (create_options & CreateOptions::FILE_SYNCHRONOUS_IO_ALERT) ||
                       (create_options & CreateOptions::FILE_SYNCHRONOUS_IO_NONALERT);
    file = object_ref<XFile>(new XFile(REX_KERNEL_STATE(), vfs_file, synchronous));

    // Handle ref is incremented, so return that.
    handle = file->handle();
  }

  if (io_status_block) {
    io_status_block->status = result;
    io_status_block->information = (uint32_t)file_action;
  }

  *handle_out = handle;
  if (XFAILED(result)) {
    REXKRNL_IMPORT_FAIL("NtCreateFile", "path='{}' -> {:#x}", target_path, result);
  } else {
    REXKRNL_IMPORT_RESULT("NtCreateFile", "{:#x} handle={:#x}", result, handle);
  }
  return result;
}

u32 NtOpenFile_entry(mapped_u32 handle_out, u32 desired_access,
                     ppc_ptr_t<X_OBJECT_ATTRIBUTES> object_attributes,
                     ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block, u32 open_options) {
  return NtCreateFile_entry(handle_out, desired_access, object_attributes, io_status_block, nullptr,
                            0, 0, static_cast<uint32_t>(rex::filesystem::FileDisposition::kOpen),
                            open_options);
}

u32 NtReadFile_entry(u32 file_handle, u32 event_handle, mapped_void apc_routine_ptr,
                     mapped_void apc_context, ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block,
                     mapped_void buffer, u32 buffer_length, mapped_u64 byte_offset_ptr) {
  uint64_t byte_offset = byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : 0;
  const bool apc_requested = (static_cast<uint32_t>(apc_routine_ptr) & ~1u) != 0;
  REXKRNL_IMPORT_TRACE(
      "NtReadFile",
      "handle={:#x} event={:#x} apc={:#x} apc_ctx={:#x} iosb={:#x} buf={:#x} len={:#x} offset={}",
      (uint32_t)file_handle, (uint32_t)event_handle, apc_routine_ptr.guest_address(),
      apc_context.guest_address(), io_status_block.guest_address(), buffer.guest_address(),
      (uint32_t)buffer_length, byte_offset_ptr ? (int64_t)byte_offset : -1);
  X_STATUS result = X_STATUS_SUCCESS;
  bool apc_queued = false;

  bool signal_event = false;
  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  auto file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  }

  if (XSUCCEEDED(result)) {
    /*
     * Why the "true ||" stays, and what happens if it is removed.
     *
     * It was not added by this port: it comes unchanged from Xenia and upstream ReXGlue, in
     * NtReadFile, NtReadFileScatter and NtWriteFile. The else branch below is Xenia's unfinished gap
     * ("TODO(benvanik): async."): it reads nothing, it only writes PENDING to the status block and
     * returns PENDING. Removing the "true ||" does not make the I/O asynchronous, it makes every
     * file opened without FILE_SYNCHRONOUS_IO return zero bytes.
     *
     * And games open files that way. In the game this was measured on, NtReadFile was called from three places:
     *   site A: event=0, apc=0, ctx=0, 64 KB per iteration in a loop. Right after, it reads
     *               information from the status block without waiting for anything. It expects an
     *               already completed answer.
     *   site B: event=0, apc=0, ctx=0. Compares the result with 259 (0x103 = PENDING) and has a
     *               path for that case.
     *   site C: event = [r31+16], apc=0, ctx, status block = r31, and before the call it writes
     *               259 to r31+0. That one is a real asynchronous path, OVERLAPPED style.
     *
     * Honoring the asynchrony (an I/O thread that, on completion, writes the status block, signals
     * the event and queues the APC) was evaluated and rejected, because measured it gains nothing:
     * reads already run on a thread other than the render thread, and outside loading screens there
     * is no correlation between slow reads and stutters (1 of 17 stutters over 150 ms falls within
     * 0.25 s of a slow read, against a base rate of 5 %). The only place it could gain is inside
     * loading, if the game overlaps; that is 29.7 s out of 483. Not worth the risk.
     */
    if (true || file->is_synchronous()) {
      // Synchronous.
      uint32_t bytes_read = 0;
      // Where the read starts. If no offset is given, the game reads from the file position.
      const uint64_t displacement =
          byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : file->position();
      const uint64_t io_us_start = NowUs();
      result = file->Read(buffer.guest_address(), buffer_length,
                          byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
                          &bytes_read, apc_context.guest_address());
      if (const int32_t us_kb = REXCVAR_GET(sa2_io_us_per_kb); us_kb > 0 && bytes_read) {
        std::this_thread::sleep_for(std::chrono::microseconds(uint64_t(us_kb) * bytes_read / 1024));
      }
      const uint64_t io_us_end = NowUs();
      NoteRead(io_us_end - io_us_start, bytes_read, file->path(), displacement);
      MaybeSummaryIo(io_us_end);
      if (io_status_block) {
        io_status_block->status = result;
        io_status_block->information = bytes_read;
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      if ((uint32_t)apc_routine_ptr & ~1) {
        if (apc_context && result == X_STATUS_SUCCESS) {
          auto thread = XThread::GetCurrentThread();
          uint32_t apc_routine = static_cast<uint32_t>(apc_routine_ptr) & ~1u;
          uint32_t apc_ctx = apc_context.guest_address();
          uint32_t apc_arg1 = io_status_block.guest_address();
          REXKRNL_IMPORT_TRACE("NtReadFile",
                               "queue_apc thid={} normal={:#x} ctx={:#x} arg1={:#x} arg2=0",
                               thread ? thread->thread_id() : 0, apc_routine, apc_ctx, apc_arg1);
          thread->EnqueueApc(apc_routine, apc_ctx, apc_arg1, 0);
          apc_queued = true;
        } else {
          REXKRNL_IMPORT_TRACE("NtReadFile", "skip_apc_queue (apc_ctx={:#x}, status={:#x})",
                               apc_context.guest_address(), result);
        }
      }

      if (!file->is_synchronous() && result != X_STATUS_END_OF_FILE) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;
    } else {
      // TODO(benvanik): async.

      // X_STATUS_PENDING if not returning immediately.
      // XFile is waitable and signalled after each async req completes.
      // reset the input event (->Reset())
      /*xeNtReadFileState* call_state = new xeNtReadFileState();
      XAsyncRequest* request = new XAsyncRequest(
      state, file,
      (XAsyncRequest::CompletionCallback)xeNtReadFileCompleted,
      call_state);*/
      // result = file->Read(buffer.guest_address(), buffer_length, byte_offset,
      //                     request);
      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }

      result = X_STATUS_PENDING;
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  // Log detailed completion info for debugging async IO issues
  if (file) {
    REXKRNL_IMPORT_RESULT(
        "NtReadFile",
        "{:#x} (sync={}, iosb_status={:#x}, iosb_info={}, ev_signaled={}, apc_requested={}, "
        "apc_queued={})",
        result, file->is_synchronous(),
        io_status_block ? (uint32_t)io_status_block->status : 0xDEAD,
        io_status_block ? (uint32_t)io_status_block->information : 0, ev && signal_event,
        apc_requested, apc_queued);
  } else {
    REXKRNL_IMPORT_RESULT("NtReadFile", "{:#x} (apc_requested={}, apc_queued={})", result,
                          apc_requested, apc_queued);
  }
  return result;
}

u32 NtReadFileScatter_entry(u32 file_handle, u32 event_handle, mapped_void apc_routine_ptr,
                            mapped_void apc_context, ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block,
                            mapped_u32 segment_array, u32 length, mapped_u64 byte_offset_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  bool signal_event = false;
  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  auto file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  }

  if (XSUCCEEDED(result)) {
    if (true || file->is_synchronous()) {
      // Synchronous.
      uint32_t bytes_read = 0;
      result = file->ReadScatter(segment_array.guest_address(), length,
                                 byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
                                 &bytes_read, apc_context.guest_address());
      if (io_status_block) {
        io_status_block->status = result;
        io_status_block->information = bytes_read;
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      if ((uint32_t)apc_routine_ptr & ~1) {
        if (apc_context) {
          auto thread = XThread::GetCurrentThread();
          thread->EnqueueApc(static_cast<uint32_t>(apc_routine_ptr) & ~1u,
                             apc_context.guest_address(), io_status_block.guest_address(), 0);
        }
      }

      if (!file->is_synchronous()) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;
    } else {
      // TODO(benvanik): async.

      // TODO: On Windows it might be worth trying to use Win32 ReadFileScatter
      // here instead of handling it ourselves

      // X_STATUS_PENDING if not returning immediately.
      // XFile is waitable and signalled after each async req completes.
      // reset the input event (->Reset())
      /*xeNtReadFileState* call_state = new xeNtReadFileState();
      XAsyncRequest* request = new XAsyncRequest(
      state, file,
      (XAsyncRequest::CompletionCallback)xeNtReadFileCompleted,
      call_state);*/
      // result = file->Read(buffer.guest_address(), buffer_length, byte_offset,
      //                     request);
      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }

      result = X_STATUS_PENDING;
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  return result;
}

u32 NtWriteFile_entry(u32 file_handle, u32 event_handle, u32 apc_routine, mapped_void apc_context,
                      ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block, mapped_void buffer,
                      u32 buffer_length, mapped_u64 byte_offset_ptr) {
  X_STATUS result = X_STATUS_SUCCESS;

  // Grab event to signal.
  bool signal_event = false;
  auto ev = REX_KERNEL_OBJECTS()->LookupObject<XEvent>(event_handle);
  if (event_handle && !ev) {
    result = X_STATUS_INVALID_HANDLE;
  }

  // Grab file.
  auto file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  }

  // Execute write.
  if (XSUCCEEDED(result)) {
    // TODO(benvanik): async path.
    if (true || file->is_synchronous()) {
      // Synchronous request.
      uint32_t bytes_written = 0;
      result = file->Write(buffer.guest_address(), buffer_length,
                           byte_offset_ptr ? static_cast<uint64_t>(*byte_offset_ptr) : -1,
                           &bytes_written, apc_context.guest_address());

      if (io_status_block) {
        io_status_block->status = result;
        io_status_block->information = static_cast<uint32_t>(bytes_written);
      }

      // Queue the APC callback. It must be delivered via the APC mechanism even
      // though were are completing immediately.
      // Low bit probably means do not queue to IO ports.
      if ((uint32_t)apc_routine & ~1) {
        if (apc_context) {
          auto thread = XThread::GetCurrentThread();
          thread->EnqueueApc(static_cast<uint32_t>(apc_routine) & ~1u, apc_context.guest_address(),
                             io_status_block.guest_address(), 0);
        }
      }

      if (!file->is_synchronous()) {
        result = X_STATUS_PENDING;
      }

      // Mark that we should signal the event now. We do this after
      // we have written the info out.
      signal_event = true;
    } else {
      // X_STATUS_PENDING if not returning immediately.
      result = X_STATUS_PENDING;

      if (io_status_block) {
        io_status_block->status = X_STATUS_PENDING;
        io_status_block->information = 0;
      }
    }
  }

  if (XFAILED(result) && io_status_block) {
    io_status_block->status = result;
    io_status_block->information = 0;
  }

  if (ev && signal_event) {
    ev->Set(0, false);
  }

  return result;
}

u32 NtCreateIoCompletion_entry(mapped_u32 out_handle, u32 desired_access,
                               mapped_void object_attribs, u32 num_concurrent_threads) {
  auto completion = new XIOCompletion(REX_KERNEL_STATE());
  if (out_handle) {
    *out_handle = completion->handle();
  }

  return X_STATUS_SUCCESS;
}

u32 NtSetIoCompletion_entry(u32 handle, u32 key_context, u32 apc_context, u32 completion_status,
                            u32 num_bytes) {
  auto port = REX_KERNEL_OBJECTS()->LookupObject<XIOCompletion>(handle);
  if (!port) {
    return X_STATUS_INVALID_HANDLE;
  }

  XIOCompletion::IONotification notification;
  notification.key_context = key_context;
  notification.apc_context = apc_context;
  notification.num_bytes = num_bytes;
  notification.status = completion_status;

  port->QueueNotification(notification);
  return X_STATUS_SUCCESS;
}

// Dequeues a packet from the completion port.
u32 NtRemoveIoCompletion_entry(u32 handle, mapped_u32 key_context, mapped_u32 apc_context,
                               ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block, mapped_u64 timeout) {
  X_STATUS status = X_STATUS_SUCCESS;
  // uint32_t info = 0;

  auto port = REX_KERNEL_OBJECTS()->LookupObject<XIOCompletion>(handle);
  if (!port) {
    status = X_STATUS_INVALID_HANDLE;
  }

  uint64_t timeout_ticks = timeout ? static_cast<uint32_t>(*timeout)
                                   : static_cast<uint64_t>(std::numeric_limits<int64_t>::min());
  XIOCompletion::IONotification notification;
  if (port->WaitForNotification(timeout_ticks, &notification)) {
    if (key_context) {
      *key_context = notification.key_context;
    }
    if (apc_context) {
      *apc_context = notification.apc_context;
    }

    if (io_status_block) {
      io_status_block->status = notification.status;
      io_status_block->information = notification.num_bytes;
    }
  } else {
    status = X_STATUS_TIMEOUT;
  }

  return status;
}

u32 NtQueryFullAttributesFile_entry(ppc_ptr_t<X_OBJECT_ATTRIBUTES> obj_attribs,
                                    ppc_ptr_t<X_FILE_NETWORK_OPEN_INFORMATION> file_info) {
  auto object_name = REX_KERNEL_MEMORY()->TranslateVirtual<X_ANSI_STRING*>(obj_attribs->name_ptr);
  auto path_str = util::TranslateAnsiPath(REX_KERNEL_MEMORY(), object_name);
  REXKRNL_IMPORT_TRACE("NtQueryFullAttributesFile", "path={}", path_str);

  object_ref<XFile> root_file;
  if (obj_attribs->root_directory != 0xFFFFFFFD &&  // ObDosDevices
      obj_attribs->root_directory != 0) {
    root_file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(obj_attribs->root_directory);
    assert_not_null(root_file);
    assert_true(root_file->type() == XObject::Type::File);
    assert_always();
  }

  auto target_path = util::TranslateAnsiPath(REX_KERNEL_MEMORY(), object_name);

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  // Resolve the file using the virtual file system.
  auto entry = REX_KERNEL_FS()->ResolvePath(target_path);
  if (entry) {
    // Found.
    file_info->creation_time = entry->create_timestamp();
    file_info->last_access_time = entry->access_timestamp();
    file_info->last_write_time = entry->write_timestamp();
    file_info->change_time = entry->write_timestamp();
    file_info->allocation_size = entry->allocation_size();
    file_info->end_of_file = entry->size();
    file_info->attributes = entry->attributes();

    REXKRNL_IMPORT_RESULT("NtQueryFullAttributesFile", "0x0 size={}", (uint64_t)entry->size());
    return X_STATUS_SUCCESS;
  }

  REXKRNL_IMPORT_RESULT("NtQueryFullAttributesFile", "X_STATUS_NO_SUCH_FILE");
  return X_STATUS_NO_SUCH_FILE;
}

u32 NtQueryDirectoryFile_entry(u32 file_handle, u32 event_handle, u32 apc_routine,
                               mapped_void apc_context,
                               ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block,
                               ppc_ptr_t<X_FILE_DIRECTORY_INFORMATION> file_info_ptr, u32 length,
                               ppc_ptr_t<X_ANSI_STRING> file_name, u32 restart_scan) {
  if (length < 72) {
    return X_STATUS_INFO_LENGTH_MISMATCH;
  }

  X_STATUS result = X_STATUS_UNSUCCESSFUL;
  uint32_t info = 0;

  auto file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(file_handle);
  auto name = util::TranslateAnsiPath(REX_KERNEL_MEMORY(), file_name);

  // Enforce that the path is ASCII.
  if (!IsValidPath(name, true)) {
    return X_STATUS_INVALID_PARAMETER;
  }

  if (file) {
    // X_FILE_DIRECTORY_INFORMATION dir_info = {0};
    result = file->QueryDirectory(file_info_ptr, length, name, restart_scan != 0);
    if (XSUCCEEDED(result)) {
      info = length;
    }
  } else {
    result = X_STATUS_NO_SUCH_FILE;
  }

  if (XFAILED(result)) {
    info = 0;
  }

  if (io_status_block) {
    io_status_block->status = result;
    io_status_block->information = info;
  }

  return result;
}

/*
 * This used to lie: it returned SUCCESS without even looking at the handle.
 *
 * The game calls it right after saving: it is its way of saying "this must be on disk now". If
 * nobody honors it, what was written is at the mercy of whenever the file system decides to flush
 * it, and on the Switch that means that leaving through the HOME menu or turning the console off
 * right after saving can leave the save half written on the SD. It shows up as the game's
 * damaged-profile message ("it looks like <profile> is damaged and cannot be used").
 */
u32 NtFlushBuffersFile_entry(u32 file_handle, ppc_ptr_t<X_IO_STATUS_BLOCK> io_status_block_ptr) {
  auto result = X_STATUS_SUCCESS;

  auto file = REX_KERNEL_OBJECTS()->LookupObject<XFile>(file_handle);
  if (!file) {
    result = X_STATUS_INVALID_HANDLE;
  } else {
    result = file->Flush();
    static std::atomic<uint32_t> warnings{0};
    if (warnings.fetch_add(1, std::memory_order_relaxed) < 16) {
      REXKRNL_INFO("[save] NtFlushBuffersFile: {} flushed to disk (result {:08X})",
                   file->path(), uint32_t(result));
    }
  }

  if (io_status_block_ptr) {
    io_status_block_ptr->status = result;
    io_status_block_ptr->information = 0;
  }

  return result;
}

// https://docs.microsoft.com/en-us/windows/win32/devnotes/ntopensymboliclinkobject
u32 NtOpenSymbolicLinkObject_entry(mapped_u32 handle_out,
                                   ppc_ptr_t<X_OBJECT_ATTRIBUTES> object_attrs) {
  if (!object_attrs) {
    return X_STATUS_INVALID_PARAMETER;
  }
  assert_not_null(handle_out);

  assert_true(object_attrs->attributes == 64);  // case insensitive

  auto object_name = REX_KERNEL_MEMORY()->TranslateVirtual<X_ANSI_STRING*>(object_attrs->name_ptr);

  auto target_path = util::TranslateAnsiPath(REX_KERNEL_MEMORY(), object_name);

  // Enforce that the path is ASCII.
  if (!IsValidPath(target_path, false)) {
    return X_STATUS_OBJECT_NAME_INVALID;
  }

  if (object_attrs->root_directory != 0) {
    assert_always();
  }

  if (rex::string::utf8_starts_with(target_path, "\\??\\")) {
    target_path = target_path.substr(4);  // Strip the full qualifier
  }

  std::string link_path;
  if (!REX_KERNEL_FS()->FindSymbolicLink(target_path, link_path)) {
    return X_STATUS_NO_SUCH_FILE;
  }

  object_ref<XSymbolicLink> symlink(new XSymbolicLink(REX_KERNEL_STATE()));
  symlink->Initialize(target_path, link_path);

  *handle_out = symlink->handle();

  return X_STATUS_SUCCESS;
}

// https://docs.microsoft.com/en-us/windows/win32/devnotes/ntquerysymboliclinkobject
u32 NtQuerySymbolicLinkObject_entry(u32 handle, ppc_ptr_t<X_ANSI_STRING> target) {
  auto symlink = REX_KERNEL_OBJECTS()->LookupObject<XSymbolicLink>(handle);
  if (!symlink) {
    return X_STATUS_NO_SUCH_FILE;
  }
  auto length = std::min(static_cast<size_t>(target->maximum_length), symlink->target().size());
  if (length > 0) {
    auto target_buf = REX_KERNEL_MEMORY()->TranslateVirtual(target->pointer);
    std::memcpy(target_buf, symlink->target().c_str(), length);
  }
  target->length = static_cast<uint16_t>(length);
  return X_STATUS_SUCCESS;
}

u32 FscGetCacheElementCount_entry(u32 r3) {
  return 0;
}

u32 FscSetCacheElementCount_entry(u32 unk_0, u32 unk_1) {
  // unk_0 = 0
  // unk_1 looks like a count? in what units? 256 is a common value
  return X_STATUS_SUCCESS;
}

u32 NtDeviceIoControlFile_entry(u32 handle, u32 event_handle, u32 apc_routine, u32 apc_context,
                                u32 io_status_block, u32 io_control_code, mapped_void input_buffer,
                                u32 input_buffer_len, mapped_void output_buffer,
                                u32 output_buffer_len) {
  // Called by XMountUtilityDrive cache-mounting code
  // (checks if the returned values look valid, values below seem to pass the
  // checks)
  const uint32_t cache_size = 0xFF000;

  const uint32_t X_IOCTL_DISK_GET_DRIVE_GEOMETRY = 0x70000;
  const uint32_t X_IOCTL_DISK_GET_PARTITION_INFO = 0x74004;

  if (io_control_code == X_IOCTL_DISK_GET_DRIVE_GEOMETRY) {
    if (output_buffer_len < 0x8) {
      assert_always();
      return X_STATUS_BUFFER_TOO_SMALL;
    }
    memory::store_and_swap<uint32_t>(output_buffer, cache_size / 512);
    memory::store_and_swap<uint32_t>(output_buffer + 4, 512);
  } else if (io_control_code == X_IOCTL_DISK_GET_PARTITION_INFO) {
    if (output_buffer_len < 0x10) {
      assert_always();
      return X_STATUS_BUFFER_TOO_SMALL;
    }
    memory::store_and_swap<uint64_t>(output_buffer, 0);
    memory::store_and_swap<uint64_t>(output_buffer + 8, cache_size);
  } else {
    REXKRNL_DEBUG("NtDeviceIoControlFile(0x{:X}) - unhandled IOCTL!", uint32_t(io_control_code));
    assert_always();
    return X_STATUS_INVALID_PARAMETER;
  }

  return X_STATUS_SUCCESS;
}

u32 IoCreateDevice_entry(u32 device_struct, u32 r4, u32 r5, u32 r6, u32 r7, mapped_u32 out_struct) {
  // Called from XMountUtilityDrive XAM-task code
  // That code tries writing things to a pointer at out_struct+0x18
  // We'll alloc some scratch space for it so it doesn't cause any exceptions

  // 0x24 is guessed size from accesses to out_struct - likely incorrect
  auto out_guest = REX_KERNEL_MEMORY()->SystemHeapAlloc(0x24);

  auto out = REX_KERNEL_MEMORY()->TranslateVirtual<uint8_t*>(out_guest);
  memset(out, 0, 0x24);

  // XMountUtilityDrive writes some kind of header here
  // 0x1000 bytes should be enough to store it
  auto out_guest2 = REX_KERNEL_MEMORY()->SystemHeapAlloc(0x1000);
  memory::store_and_swap(out + 0x18, out_guest2);

  *out_struct = out_guest;
  return X_STATUS_SUCCESS;
}

u32 IoDismountVolumeByFileHandle_entry(u32 handle) {
  REXKRNL_WARN("IoDismountVolumeByFileHandle({:#x}) - stub", (uint32_t)handle);
  return X_STATUS_SUCCESS;
}

u32 IoDismountVolumeByName_entry(ppc_ptr_t<X_ANSI_STRING> name) {
  REXKRNL_WARN("IoDismountVolumeByName - stub");
  return X_STATUS_SUCCESS;
}

u32 IoSynchronousDeviceIoControlRequest_entry(u32 ioctl, mapped_void device_object,
                                              mapped_void input_buffer, u32 input_length,
                                              mapped_void output_buffer, u32 output_length,
                                              mapped_u32 returned_length, u32 unk) {
  REXKRNL_WARN("IoSynchronousDeviceIoControlRequest({:#x}) - stub", (uint32_t)ioctl);
  return X_STATUS_SUCCESS;
}

u32 StfsCreateDevice_entry(mapped_void device_object, u32 flags, mapped_u32 out_device) {
  REXKRNL_WARN("StfsCreateDevice - stub");
  // if (out_device) *out_device = 0;
  return X_STATUS_SUCCESS;
}

u32 StfsControlDevice_entry(mapped_void device_object, u32 ioctl, mapped_void input_buffer,
                            u32 input_length, mapped_void output_buffer, u32 output_length) {
  REXKRNL_WARN("StfsControlDevice({:#x}) - stub", (uint32_t)ioctl);
  return X_STATUS_SUCCESS;
}

}  // namespace rex::kernel::xboxkrnl

REX_EXPORT(__imp__NtCreateFile, rex::kernel::xboxkrnl::NtCreateFile_entry)
REX_EXPORT(__imp__NtOpenFile, rex::kernel::xboxkrnl::NtOpenFile_entry)
REX_EXPORT(__imp__NtReadFile, rex::kernel::xboxkrnl::NtReadFile_entry)
REX_EXPORT(__imp__NtReadFileScatter, rex::kernel::xboxkrnl::NtReadFileScatter_entry)
REX_EXPORT(__imp__NtWriteFile, rex::kernel::xboxkrnl::NtWriteFile_entry)
REX_EXPORT(__imp__NtCreateIoCompletion, rex::kernel::xboxkrnl::NtCreateIoCompletion_entry)
REX_EXPORT(__imp__NtSetIoCompletion, rex::kernel::xboxkrnl::NtSetIoCompletion_entry)
REX_EXPORT(__imp__NtRemoveIoCompletion, rex::kernel::xboxkrnl::NtRemoveIoCompletion_entry)
REX_EXPORT(__imp__NtQueryFullAttributesFile, rex::kernel::xboxkrnl::NtQueryFullAttributesFile_entry)
REX_EXPORT(__imp__NtQueryDirectoryFile, rex::kernel::xboxkrnl::NtQueryDirectoryFile_entry)
REX_EXPORT(__imp__NtFlushBuffersFile, rex::kernel::xboxkrnl::NtFlushBuffersFile_entry)
REX_EXPORT(__imp__NtOpenSymbolicLinkObject, rex::kernel::xboxkrnl::NtOpenSymbolicLinkObject_entry)
REX_EXPORT(__imp__NtQuerySymbolicLinkObject, rex::kernel::xboxkrnl::NtQuerySymbolicLinkObject_entry)
REX_EXPORT(__imp__FscGetCacheElementCount, rex::kernel::xboxkrnl::FscGetCacheElementCount_entry)
REX_EXPORT(__imp__FscSetCacheElementCount, rex::kernel::xboxkrnl::FscSetCacheElementCount_entry)
REX_EXPORT(__imp__NtDeviceIoControlFile, rex::kernel::xboxkrnl::NtDeviceIoControlFile_entry)
REX_EXPORT(__imp__IoCreateDevice, rex::kernel::xboxkrnl::IoCreateDevice_entry)
REX_EXPORT(__imp__IoDismountVolumeByFileHandle,
           rex::kernel::xboxkrnl::IoDismountVolumeByFileHandle_entry)
REX_EXPORT(__imp__IoDismountVolumeByName, rex::kernel::xboxkrnl::IoDismountVolumeByName_entry)
REX_EXPORT(__imp__IoSynchronousDeviceIoControlRequest,
           rex::kernel::xboxkrnl::IoSynchronousDeviceIoControlRequest_entry)
REX_EXPORT(__imp__StfsCreateDevice, rex::kernel::xboxkrnl::StfsCreateDevice_entry)
REX_EXPORT(__imp__StfsControlDevice, rex::kernel::xboxkrnl::StfsControlDevice_entry)

REX_EXPORT_STUB(__imp__IoAcquireDeviceObjectLock);
REX_EXPORT_STUB(__imp__IoAllocateIrp);
REX_EXPORT_STUB(__imp__IoBuildAsynchronousFsdRequest);
REX_EXPORT_STUB(__imp__IoBuildDeviceIoControlRequest);
REX_EXPORT_STUB(__imp__IoBuildSynchronousFsdRequest);
REX_EXPORT_STUB(__imp__IoCallDriver);
REX_EXPORT_STUB(__imp__IoCheckShareAccess);
REX_EXPORT_STUB(__imp__IoCompleteRequest);
REX_EXPORT_STUB(__imp__IoCreateFile);
REX_EXPORT_STUB(__imp__IoDeleteDevice);
REX_EXPORT_STUB(__imp__IoDismountVolume);
REX_EXPORT_STUB(__imp__IoFreeIrp);
REX_EXPORT_STUB(__imp__IoInitializeIrp);
REX_EXPORT_STUB(__imp__IoInvalidDeviceRequest);
REX_EXPORT_STUB(__imp__IoQueueThreadIrp);
REX_EXPORT_STUB(__imp__IoReleaseDeviceObjectLock);
REX_EXPORT_STUB(__imp__IoRemoveShareAccess);
REX_EXPORT_STUB(__imp__IoSetIoCompletion);
REX_EXPORT_STUB(__imp__IoSetShareAccess);
REX_EXPORT_STUB(__imp__IoStartNextPacket);
REX_EXPORT_STUB(__imp__IoStartNextPacketByKey);
REX_EXPORT_STUB(__imp__IoStartPacket);
REX_EXPORT_STUB(__imp__IoSynchronousFsdRequest);
REX_EXPORT_STUB(__imp__NtDeleteFile);
REX_EXPORT_STUB(__imp__NtSetSystemTime);
REX_EXPORT_STUB(__imp__NtWriteFileGather);
REX_EXPORT_STUB(__imp__IoAcquireCancelSpinLock);
REX_EXPORT_STUB(__imp__IoReleaseCancelSpinLock);
REX_EXPORT_STUB(__imp__NtCancelIoFile);
REX_EXPORT_STUB(__imp__NtCancelIoFileEx);
REX_EXPORT_STUB(__imp__SvodCreateDevice);

REX_EXPORT_STUB(__imp__SataCdRomRecordReset);
