/**
 * Startup read trace and background preload (docs/startup-speedup.md, item 2).
 *
 * The problem. Between process start and the first menu the guest does ~600-800 blocking reads of
 * 128 KB (and ~60 opens) on the guest thread: about 7 s of read() and 1.2 s of open() on the Switch
 * SD, most of it per-call overhead (2.3 ms fixed per call + 69.5 MB/s). The guest reads the same
 * files in the same order on every start, so the sequence can be learned once and replayed ahead of
 * the guest on another thread.
 *
 * Two modes, both off by default:
 *
 *  RECORD (sa2_io_trace_record). Every guest read of a read-only game file is appended to an
 *  in-memory log (kind, thread, file, offset, length, time since mount); so is every open. A small
 *  thread rewrites the trace file every 15 s and once more at sa2_io_trace_record_s seconds after
 *  the mount, so a run that is killed early still leaves a usable trace.
 *
 *  PRELOAD (sa2_io_preload). Right after the mount the trace is parsed (one small read) and a
 *  low-priority thread walks it in order:
 *    - an "open" step opens the file ahead of the guest and parks the handle in a pool; the guest's
 *      HostPathEntry::Open takes it instead of paying the open itself;
 *    - reads of the same file that are adjacent in the trace are merged into one pread of up to
 *      sa2_io_preload_coalesce_kb (this is where most of the gain is: 8 x 128 KB reads become one
 *      1 MB read), then sliced back into the exact (offset, length) ranges the guest will ask for;
 *    - the slices live in a bounded store (sa2_io_preload_mb). When the guest asks for exactly
 *      one of them, the read is a memcpy; when its last recorded use is done the slice is freed.
 *  The store never holds more than the cap: the thread waits for the guest to consume, and drops
 *  slices that are far behind the guest's position in the trace (the guest skipped them).
 *
 * Safety:
 *  - Only the first read-only device that mounts (the game root) and only files the guest opens
 *    without write access. Saves and profiles live on writable devices and are never touched.
 *  - A trace entry is used only if the file still has the recorded size and write time (the write
 *    time check can be turned off). Anything else is skipped file by file.
 *  - A missing, truncated, corrupt (FNV-1a trailer) or foreign trace disables the preload with one
 *    log line. An SD error in the thread skips the range; after 4 failures in a row it stops.
 *  - A range is served only on an exact (file, offset, length) match, with the bytes the SD
 *    returned, so the guest sees the same data as without the preload. Ranges that are not ready
 *    when the guest asks are marked skipped and the guest reads the SD itself, exactly as before.
 *
 * Trace file format (little endian, packed, version 1):
 *   char     magic[8]  = "RXSTRC01"
 *   uint32   version   = 1
 *   uint32   file_count
 *   uint32   event_count
 *   uint32   flags     = 0
 *   file_count x { uint16 path_len; char path[path_len]; uint64 size; uint64 write_timestamp; }
 *       path is the guest-visible path inside the device (Entry::path()), files in first-open order.
 *   event_count x 24 bytes {
 *       uint8 kind (0 = read, 1 = open); uint8 pad; uint16 thread (hash of the thread id);
 *       uint32 file_index; uint64 offset; uint32 length; uint32 t_us (since mount) }
 *   uint64   fnv1a64 of every byte above
 */

#include "startup_trace.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <new>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/logging.h>
#include <rex/platform.h>
#include <rex/string.h>

#if REX_PLATFORM_SWITCH
extern "C" void RexSwitchSetCurrentThreadPriority(int priority);
extern "C" bool RexSwitchSetCurrentThreadCore(int core);
#endif

REXCVAR_DEFINE_BOOL(sa2_io_trace_record, false, "Filesystem",
                    "Record the guest's reads and opens of the read-only game files into the startup "
                    "trace file (see sa2_io_trace_file). Turn on for one run to refresh the trace; "
                    "when on, sa2_io_preload is ignored.");
REXCVAR_DEFINE_INT32(sa2_io_trace_record_s, 50, "Filesystem",
                     "Seconds after the VFS mount at which the trace recording stops and the final "
                     "trace file is written (it is also rewritten every 15 s).");
REXCVAR_DEFINE_STRING(sa2_io_trace_file, "", "Filesystem",
                      "Startup trace file. Empty = <parent of the game folder>/cache/startup_reads.bin.");
REXCVAR_DEFINE_BOOL(sa2_io_preload, false, "Filesystem",
                    "Replay the startup trace on a low-priority background thread: read the recorded "
                    "ranges ahead of the guest into a bounded RAM store so its reads are memcpy.");
REXCVAR_DEFINE_INT32(sa2_io_preload_mb, 64, "Filesystem",
                     "Hard cap of the preload store, in MB (resident slices plus the read in flight).");
REXCVAR_DEFINE_INT32(sa2_io_preload_max_range_kb, 4096, "Filesystem",
                     "Trace reads larger than this many KB are not preloaded.");
REXCVAR_DEFINE_INT32(sa2_io_preload_coalesce_kb, 1024, "Filesystem",
                     "Adjacent trace reads of one file are merged into a single SD read of up to this "
                     "many KB (0 = one SD read per trace read).");
REXCVAR_DEFINE_INT32(sa2_io_preload_opens, 32, "Filesystem",
                     "Max read handles the preload thread keeps pre-opened for the guest (0 = do not "
                     "pre-open files).");
REXCVAR_DEFINE_BOOL(sa2_io_preload_check_mtime, true, "Filesystem",
                    "Besides the size, require the file's write time to equal the recorded one.");
REXCVAR_DEFINE_BOOL(sa2_io_preload_verify, false, "Filesystem",
                    "Test aid: every read served from the preload store is also read from the SD and "
                    "compared; mismatches are logged. Slower than no preload; never use for timing.");
REXCVAR_DEFINE_INT32(sa2_io_preload_priority, 58, "Filesystem",
                     "Horizon thread priority of the preload thread (Switch only; 59 is the lowest "
                     "the process may use, the game's threads run at 44 and below).");
REXCVAR_DEFINE_INT32(sa2_io_preload_core, -1, "Filesystem",
                     "Preferred core of the preload thread (Switch only, -1 = leave it to the kernel).");
REXCVAR_DEFINE_STRING(sa2_io_trace_markers,
                      "coalesced.ini,engine.xxx,bioc_base,entrymenu,bioa_pro10", "Filesystem",
                      "Comma separated, case-insensitive substrings. The first guest read of the first "
                      "file whose path contains each one is logged as a '[io] timeline' line.");

namespace rex::filesystem::startup_trace {
namespace {

using Clock = std::chrono::steady_clock;

constexpr char kMagic[8] = {'R', 'X', 'S', 'T', 'R', 'C', '0', '1'};
constexpr uint32_t kVersion = 1;
constexpr size_t kMaxEvents = 400000;
constexpr uint32_t kFarBehindEvents = 256;  // slices this far behind the guest are dropped

#pragma pack(push, 1)
struct Event {
  uint8_t kind;  // 0 read, 1 open
  uint8_t pad;
  uint16_t thread;
  uint32_t file;
  uint64_t offset;
  uint32_t length;
  uint32_t t_us;
};
#pragma pack(pop)
static_assert(sizeof(Event) == 24, "trace event is 24 bytes");

struct TraceFile {
  std::string path;
  uint64_t size = 0;
  uint64_t mtime = 0;
};

enum RangeState : uint8_t { kPending, kReading, kReady, kDone, kSkipped, kDropped };

struct Range {
  uint32_t file = 0;
  uint64_t offset = 0;
  uint32_t length = 0;
  uint32_t first_event = 0;
  uint32_t uses_left = 0;
  RangeState state = kPending;
  std::shared_ptr<std::vector<uint8_t>> data;
};

struct Step {
  bool open = false;
  uint32_t file = 0;   // open step
  uint32_t first = 0;  // read step: indices [first, last) into `order`
  uint32_t last = 0;
};

struct FileState {
  std::string path;
  int64_t trace_file = -1;
  bool first_read_logged = false;
};

struct State {
  std::mutex mu;
  std::condition_variable cv;
  std::atomic<bool> active{false};  // anything to do on the guest's path
  bool primary_set = false;
  HostPathDevice* device = nullptr;
  std::filesystem::path root;
  std::filesystem::path trace_path;
  Clock::time_point t0;
  bool recording = false;
  bool preloading = false;

  std::vector<std::unique_ptr<FileState>> files;
  std::unordered_map<std::string, uint32_t> tags;  // path -> index in files (tag = index + 1)

  // Record.
  std::vector<Event> events;
  std::atomic<bool> record_overflow{false};

  // Preload (immutable after load, except what the comments mark).
  std::vector<TraceFile> trace_files;
  std::unordered_map<std::string, uint32_t> trace_index;  // path -> trace file
  std::vector<Event> trace_events;
  std::vector<Range> ranges;                               // state/data/uses_left mutable under mu
  std::map<std::tuple<uint32_t, uint64_t, uint32_t>, uint32_t> range_index;
  std::vector<uint32_t> order;  // range indices in trace order (first occurrences only)
  std::vector<Step> plan;
  // MSVC's std::deque has an unconstrained copy constructor and a throwing move constructor, so a std::vector of
  // deques of move-only handles tries to copy them when it grows. Deleting the copy makes it move instead.
  struct HandleQueue : std::deque<std::unique_ptr<FileHandle>> {
    HandleQueue() = default;
    HandleQueue(HandleQueue&&) = default;
    HandleQueue& operator=(HandleQueue&&) = default;
    HandleQueue(const HandleQueue&) = delete;
    HandleQueue& operator=(const HandleQueue&) = delete;
  };
  std::vector<HandleQueue> pool;  // per trace file, mutable under mu
  size_t pool_count = 0;
  uint64_t resident = 0;  // bytes in the store + reserved by the read in flight
  uint64_t peak = 0;
  uint32_t guest_pos = 0;  // highest first_event the guest has asked for
  std::atomic<bool> preload_done{false};
  uint64_t cap = 0;

  // Counters (under mu unless atomic).
  uint64_t planned_ranges = 0, planned_bytes = 0, skipped_big = 0, stale_files = 0;
  uint64_t hits = 0, hit_bytes = 0, late = 0, waited = 0, other_miss = 0, unmatched = 0;
  uint64_t guest_reads = 0, guest_bytes = 0;
  uint64_t verified = 0, mismatches = 0;
  uint64_t sd_calls = 0, sd_bytes = 0, sd_errors = 0, dropped = 0, opens_pooled = 0, opens_taken = 0;
  double sd_seconds = 0;

  std::vector<std::string> markers;
  std::vector<bool> marker_done;
};

// Never destroyed: the worker threads are detached and may outlive static destruction.
State& G() {
  static State* s = new State();
  return *s;
}

double SecondsSinceMount() {
  return std::chrono::duration<double>(Clock::now() - G().t0).count();
}

uint64_t Fnv1a(const uint8_t* p, size_t n, uint64_t h = 1469598103934665603ull) {
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 1099511628211ull;
  }
  return h;
}

std::string Lower(std::string s) {
  for (auto& c : s) {
    if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
  }
  return s;
}

// Entry::path() is the guest path ("Layer0\\MEInit\\Coalesced.ini"): turn it into a host path.
std::filesystem::path HostFile(const std::filesystem::path& root, std::string rel) {
  for (auto& c : rel) {
    if (c == '\\') c = '/';
  }
  return root / rex::to_path(rel);
}

// Starts a worker that lives until the process ends. NEVER detach() or join() it: on the Switch
// std::thread::detach() throws std::system_error (see audio/xma_context.cpp), which killed the game
// right after the "recording" log line. The std::thread object is leaked on purpose. Returns false
// (and logs) if the thread cannot be created, so the caller can turn the feature off.
bool StartWorker(const char* name, void (*fn)()) {
  try {
    REXFS_INFO("[io] trace: stage: creating thread '{}'", name);
    new std::thread([name, fn] {
      REXFS_INFO("[io] trace: stage: thread '{}' is running", name);
      fn();
      REXFS_INFO("[io] trace: stage: thread '{}' finished", name);
    });
    return true;
  } catch (const std::exception& e) {
    REXFS_WARN("[io] trace: could not create thread '{}': {}", name, e.what());
    return false;
  }
}

std::filesystem::path TracePath(const std::filesystem::path& host_root) {
  const std::string& configured = REXCVAR_GET(sa2_io_trace_file);
  if (!configured.empty()) {
    return rex::to_path(configured);
  }
  auto folder = host_root;
  if (!folder.has_filename()) {
    folder = folder.parent_path();
  }
  return folder.parent_path() / "cache" / "startup_reads.bin";
}

// ---------------------------------------------------------------------------------------------
// Trace file I/O
// ---------------------------------------------------------------------------------------------

template <typename T>
void Put(std::vector<uint8_t>& out, const T& v) {
  const auto* p = reinterpret_cast<const uint8_t*>(&v);
  out.insert(out.end(), p, p + sizeof(T));
}

struct Reader {
  const uint8_t* p;
  size_t n;
  size_t at = 0;
  bool ok = true;
  template <typename T>
  T Get() {
    T v{};
    if (at + sizeof(T) > n) {
      ok = false;
      return v;
    }
    std::memcpy(&v, p + at, sizeof(T));
    at += sizeof(T);
    return v;
  }
  std::string Str(size_t len) {
    if (at + len > n) {
      ok = false;
      return {};
    }
    std::string s(reinterpret_cast<const char*>(p + at), len);
    at += len;
    return s;
  }
};

bool LoadTrace(State& g, std::string* why) {
  FILE* f = rex::filesystem::OpenFile(g.trace_path, "rb");
  if (!f) {
    *why = "no trace file at " + rex::path_to_utf8(g.trace_path);
    return false;
  }
  std::vector<uint8_t> data;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    const long size = std::ftell(f);
    if (size > 0 && size < (256l << 20) && std::fseek(f, 0, SEEK_SET) == 0) {
      data.resize(size_t(size));
      if (std::fread(data.data(), 1, data.size(), f) != data.size()) {
        data.clear();
      }
    }
  }
  std::fclose(f);
  if (data.size() < 28 + 8) {
    *why = "trace file unreadable or too short";
    return false;
  }
  const size_t body = data.size() - 8;
  uint64_t stored = 0;
  std::memcpy(&stored, data.data() + body, 8);
  if (Fnv1a(data.data(), body) != stored) {
    *why = "trace checksum mismatch (truncated or corrupt)";
    return false;
  }
  Reader r{data.data(), body};
  char magic[8];
  for (auto& c : magic) c = char(r.Get<uint8_t>());
  if (std::memcmp(magic, kMagic, 8) != 0 || r.Get<uint32_t>() != kVersion) {
    *why = "not a version 1 startup trace";
    return false;
  }
  const uint32_t nfiles = r.Get<uint32_t>();
  const uint32_t nevents = r.Get<uint32_t>();
  (void)r.Get<uint32_t>();
  if (!r.ok || nfiles > 100000 || nevents > kMaxEvents) {
    *why = "trace header out of range";
    return false;
  }
  g.trace_files.resize(nfiles);
  for (auto& tf : g.trace_files) {
    const uint16_t len = r.Get<uint16_t>();
    tf.path = r.Str(len);
    tf.size = r.Get<uint64_t>();
    tf.mtime = r.Get<uint64_t>();
  }
  if (!r.ok || body - r.at != size_t(nevents) * sizeof(Event)) {
    *why = "trace layout does not match its header";
    return false;
  }
  g.trace_events.resize(nevents);
  if (nevents) {
    std::memcpy(g.trace_events.data(), data.data() + r.at, size_t(nevents) * sizeof(Event));
  }
  for (const auto& e : g.trace_events) {
    if (e.file >= nfiles) {
      *why = "trace event refers to a missing file";
      return false;
    }
  }
  for (uint32_t i = 0; i < nfiles; ++i) {
    g.trace_index.emplace(g.trace_files[i].path, i);
  }
  return true;
}

bool WriteTrace(State& g, const std::vector<Event>& events,
                const std::vector<std::string>& paths, std::string* why) {
  // Stat every file once so the stored size and write time are the real ones.
  std::vector<uint8_t> out;
  out.reserve(64 + events.size() * sizeof(Event) + paths.size() * 96);
  out.insert(out.end(), kMagic, kMagic + 8);
  Put<uint32_t>(out, kVersion);
  Put<uint32_t>(out, uint32_t(paths.size()));
  Put<uint32_t>(out, uint32_t(events.size()));
  Put<uint32_t>(out, 0);
  for (const auto& p : paths) {
    FileInfo info{};
    uint64_t size = 0, mtime = 0;
    if (GetInfo(HostFile(g.root, p), &info)) {
      size = info.total_size;
      mtime = info.write_timestamp;
    }
    Put<uint16_t>(out, uint16_t(p.size()));
    out.insert(out.end(), p.begin(), p.end());
    Put<uint64_t>(out, size);
    Put<uint64_t>(out, mtime);
  }
  const auto* ep = reinterpret_cast<const uint8_t*>(events.data());
  out.insert(out.end(), ep, ep + events.size() * sizeof(Event));
  Put<uint64_t>(out, Fnv1a(out.data(), out.size()));

  std::error_code ec;
  std::filesystem::create_directories(g.trace_path.parent_path(), ec);
  auto tmp = g.trace_path;
  tmp += ".tmp";
  FILE* f = rex::filesystem::OpenFile(tmp, "wb");
  if (!f) {
    *why = "cannot create " + rex::path_to_utf8(tmp);
    return false;
  }
  const bool wrote = std::fwrite(out.data(), 1, out.size(), f) == out.size();
  const bool closed = std::fclose(f) == 0;
  if (!wrote || !closed) {
    std::filesystem::remove(tmp, ec);
    *why = "write failed for " + rex::path_to_utf8(tmp);
    return false;
  }
  // Horizon cannot rename over an existing file, so remove the old one first (the index writer
  // does the same).
  std::filesystem::remove(g.trace_path, ec);
  std::filesystem::rename(tmp, g.trace_path, ec);
  if (ec) {
    *why = "rename failed: " + ec.message();
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------------------------
// Record
// ---------------------------------------------------------------------------------------------

uint16_t ThreadHash() {
  return uint16_t(std::hash<std::thread::id>{}(std::this_thread::get_id()) & 0xFFFF);
}

void PushEvent(State& g, uint8_t kind, uint32_t file, uint64_t offset, uint32_t length) {
  // g.mu held.
  if (g.events.size() >= kMaxEvents) {
    g.record_overflow.store(true, std::memory_order_relaxed);
    return;
  }
  Event e{};
  e.kind = kind;
  e.thread = ThreadHash();
  e.file = file;
  e.offset = offset;
  e.length = length;
  e.t_us = uint32_t(std::min<int64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - g.t0).count(),
      0xFFFFFFFFll));
  g.events.push_back(e);
}

void SnapshotTrace(State& g, bool final_write) {
  std::vector<Event> events;
  std::vector<std::string> paths;
  {
    std::lock_guard lock(g.mu);
    events = g.events;
    for (const auto& f : g.files) paths.push_back(f->path);
  }
  std::string why;
  const auto t = Clock::now();
  if (WriteTrace(g, events, paths, &why)) {
    REXFS_INFO("[io] trace: {} {} events, {} files -> {} ({:.1f} ms)",
               final_write ? "final:" : "snapshot:", events.size(), paths.size(),
               rex::path_to_utf8(g.trace_path),
               std::chrono::duration<double, std::milli>(Clock::now() - t).count());
  } else {
    REXFS_WARN("[io] trace: could not write the trace: {}", why);
  }
}

void RecorderThread() {
  State& g = G();
  const double stop_s = std::max(5, REXCVAR_GET(sa2_io_trace_record_s));
  double next_snapshot = 15.0;
  double next_beat = 5.0;
  for (;;) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    const double t = SecondsSinceMount();
    if (t >= stop_s) {
      break;
    }
    if (t >= next_beat) {
      next_beat += 5.0;
      size_t n;
      {
        std::lock_guard lock(g.mu);
        n = g.events.size();
      }
      REXFS_INFO("[io] trace: recording, t={:.1f} s, {} events so far", t, n);
    }
    if (t >= next_snapshot) {
      next_snapshot += 15.0;
      SnapshotTrace(g, false);
    }
  }
  g.active.store(false, std::memory_order_relaxed);  // stop collecting
  SnapshotTrace(g, true);
  if (g.record_overflow.load()) {
    REXFS_WARN("[io] trace: the event limit ({}) was reached, the trace is incomplete", kMaxEvents);
  }
}

// ---------------------------------------------------------------------------------------------
// Preload
// ---------------------------------------------------------------------------------------------

void BuildPlan(State& g) {
  const uint64_t max_range = uint64_t(std::max(0, REXCVAR_GET(sa2_io_preload_max_range_kb))) * 1024;
  const uint64_t coalesce = uint64_t(std::max(0, REXCVAR_GET(sa2_io_preload_coalesce_kb))) * 1024;
  const bool opens = REXCVAR_GET(sa2_io_preload_opens) > 0;

  // Distinct ranges, first occurrence order; count the uses.
  std::vector<uint32_t> first_of_event(g.trace_events.size(), UINT32_MAX);
  for (uint32_t i = 0; i < g.trace_events.size(); ++i) {
    const Event& e = g.trace_events[i];
    if (e.kind != 0 || e.length == 0) continue;
    const auto key = std::make_tuple(e.file, e.offset, e.length);
    auto it = g.range_index.find(key);
    if (it == g.range_index.end()) {
      Range r;
      r.file = e.file;
      r.offset = e.offset;
      r.length = e.length;
      r.first_event = i;
      r.uses_left = 1;
      if (e.length > max_range) {
        r.state = kSkipped;
        ++g.skipped_big;
      }
      const uint32_t idx = uint32_t(g.ranges.size());
      g.ranges.push_back(std::move(r));
      g.range_index.emplace(key, idx);
      first_of_event[i] = idx;
    } else {
      ++g.ranges[it->second].uses_left;
    }
  }

  std::vector<uint32_t> opened(g.trace_files.size(), 0);
  Step cur;
  bool have = false;
  uint64_t cur_end = 0, cur_bytes = 0;
  auto flush = [&] {
    if (have) {
      cur.last = uint32_t(g.order.size());
      g.plan.push_back(cur);
      have = false;
    }
  };
  for (uint32_t i = 0; i < g.trace_events.size(); ++i) {
    const Event& e = g.trace_events[i];
    if (e.kind == 1) {
      flush();
      if (opens && opened[e.file]++ < 2) {
        Step s;
        s.open = true;
        s.file = e.file;
        g.plan.push_back(s);
      }
      continue;
    }
    const uint32_t idx = first_of_event[i];
    if (idx == UINT32_MAX || g.ranges[idx].state == kSkipped) continue;
    const Range& r = g.ranges[idx];
    const bool joins = have && g.ranges[g.order.back()].file == r.file && r.offset == cur_end &&
                       cur_bytes + r.length <= coalesce;
    if (!joins) {
      flush();
      cur = Step{};
      cur.first = uint32_t(g.order.size());
      cur_bytes = 0;
      have = true;
    }
    g.order.push_back(idx);
    cur_end = r.offset + r.length;
    cur_bytes += r.length;
    ++g.planned_ranges;
    g.planned_bytes += r.length;
  }
  flush();
  g.pool.resize(g.trace_files.size());
}

void ReportLocked(State& g, const char* tag) {
  // g.mu held.
  const uint64_t asked = g.hits + g.late + g.waited + g.other_miss;
  REXFS_INFO("[io] preload {}: t={:.1f} s, SD: {} calls, {:.1f} MB in {:.2f} s ({} errors); "
             "guest: {} ranges matched ({} hits = {:.1f} MB from RAM, {} late, {} waited-then-hit/miss, "
             "{} other), {} reads not in the trace; store {:.1f} MB (peak {:.1f}, cap {} MB), "
             "{} dropped, handles pooled {} / taken {}, verified {} (mismatches {})",
             tag, SecondsSinceMount(), g.sd_calls, double(g.sd_bytes) / 1048576.0, g.sd_seconds,
             g.sd_errors, asked, g.hits, double(g.hit_bytes) / 1048576.0, g.late, g.waited,
             g.other_miss, g.unmatched, double(g.resident) / 1048576.0,
             double(g.peak) / 1048576.0, g.cap >> 20, g.dropped, g.opens_pooled, g.opens_taken, g.verified, g.mismatches);
}

void DropBehindLocked(State& g) {
  // g.mu held. Drop ready slices the guest has clearly passed without using.
  if (g.guest_pos <= kFarBehindEvents) return;
  const uint32_t limit = g.guest_pos - kFarBehindEvents;
  for (auto& r : g.ranges) {
    if (r.state == kReady && r.first_event < limit) {
      g.resident -= r.data ? r.data->size() : 0;
      r.data.reset();
      r.state = kDropped;
      ++g.dropped;
    }
  }
}

struct Handles {
  std::unordered_map<uint32_t, std::unique_ptr<FileHandle>> open;
  std::deque<uint32_t> lru;
  FileHandle* Get(State& g, uint32_t file) {
    auto it = open.find(file);
    if (it != open.end()) return it->second.get();
    auto h = FileHandle::OpenExisting(HostFile(g.root, g.trace_files[file].path),
                                      FileAccess::kGenericRead, false);
    if (!h) return nullptr;
    if (open.size() >= 8) {
      open.erase(lru.front());
      lru.pop_front();
    }
    lru.push_back(file);
    return (open[file] = std::move(h)).get();
  }
};

// 0 = unknown, 1 = valid, 2 = stale
bool FileIsValid(State& g, std::vector<uint8_t>& verdict, uint32_t file) {
  if (verdict[file]) return verdict[file] == 1;
  FileInfo info{};
  const TraceFile& tf = g.trace_files[file];
  bool ok = GetInfo(HostFile(g.root, tf.path), &info) && info.total_size == tf.size;
  if (ok && REXCVAR_GET(sa2_io_preload_check_mtime) && info.write_timestamp != tf.mtime) {
    ok = false;
  }
  verdict[file] = ok ? 1 : 2;
  if (!ok) {
    std::lock_guard lock(g.mu);
    ++g.stale_files;
    REXFS_INFO("[io] preload: '{}' changed since the trace was recorded (size or write time), skipped",
               tf.path);
  }
  return ok;
}

void SkipRange(State& g, uint32_t idx) {
  // g.mu held.
  Range& r = g.ranges[idx];
  if (r.state == kPending || r.state == kReading) r.state = kSkipped;
}

// Reads one contiguous run of ranges with as few SD calls as possible and publishes the slices.
// Returns false if the SD failed.
bool ReadRun(State& g, Handles& handles, const std::vector<uint32_t>& run) {
  const Range& first = g.ranges[run.front()];
  const Range& last = g.ranges[run.back()];
  const uint64_t base = first.offset;
  const uint64_t total = last.offset + last.length - base;

  FileHandle* h = handles.Get(g, first.file);
  size_t got = 0;
  std::vector<uint8_t> buf;
  bool ok = h != nullptr;
  const auto t = Clock::now();
  uint64_t calls = 0;
  if (ok) {
    try {
      buf.resize(size_t(total));
    } catch (const std::bad_alloc&) {
      ok = false;
    }
  }
  if (ok) {
    constexpr size_t kPiece = 4u << 20;
    while (got < total) {
      size_t n = 0;
      ++calls;
      if (!h->Read(size_t(base + got), buf.data() + got, std::min<size_t>(kPiece, total - got), &n)) {
        if (got == 0) ok = false;
        break;
      }
      if (n == 0) break;  // end of file
      got += n;
    }
  }
  const double secs = std::chrono::duration<double>(Clock::now() - t).count();

  std::vector<std::shared_ptr<std::vector<uint8_t>>> slices(run.size());
  if (ok) {
    try {
      for (size_t i = 0; i < run.size(); ++i) {
        const Range& r = g.ranges[run[i]];
        const uint64_t rel = r.offset - base;
        if (rel >= got) continue;
        const size_t n = size_t(std::min<uint64_t>(r.length, got - rel));
        slices[i] = std::make_shared<std::vector<uint8_t>>(buf.begin() + rel, buf.begin() + rel + n);
      }
    } catch (const std::bad_alloc&) {
      slices.assign(run.size(), nullptr);
      ok = false;
    }
  }

  std::lock_guard lock(g.mu);
  g.sd_calls += calls;
  g.sd_bytes += got;
  g.sd_seconds += secs;
  uint64_t kept = 0;
  for (size_t i = 0; i < run.size(); ++i) {
    Range& r = g.ranges[run[i]];
    if (r.state != kReading) continue;  // cannot happen, but never overwrite a decision
    if (ok && slices[i]) {
      kept += slices[i]->size();
      r.data = std::move(slices[i]);
      r.state = kReady;
    } else {
      r.state = kSkipped;
    }
  }
  g.resident = g.resident - total + kept;
  g.peak = std::max(g.peak, g.resident);
  if (!ok) ++g.sd_errors;
  g.cv.notify_all();
  return ok;
}

void PreloadThread() {
  State& g = G();
#if REX_PLATFORM_SWITCH
  RexSwitchSetCurrentThreadPriority(REXCVAR_GET(sa2_io_preload_priority));
  const int core = REXCVAR_GET(sa2_io_preload_core);
  if (core >= 0) RexSwitchSetCurrentThreadCore(core);
#endif
  Handles handles;
  std::vector<uint8_t> verdict(g.trace_files.size(), 0);
  const size_t max_pool = size_t(std::max(0, REXCVAR_GET(sa2_io_preload_opens)));
  uint32_t consecutive_errors = 0;
  double next_report = 5.0;
  bool alive = true;

  auto maybe_report = [&] {
    if (SecondsSinceMount() >= next_report) {
      next_report += 5.0;
      std::lock_guard lock(g.mu);
      ReportLocked(g, "progress");
    }
  };

  for (const Step& step : g.plan) {
    if (!alive) break;
    if (step.open) {
      {
        std::lock_guard lock(g.mu);
        if (g.pool_count >= max_pool || g.pool[step.file].size() >= 2) continue;
      }
      if (!FileIsValid(g, verdict, step.file)) continue;
      auto h = FileHandle::OpenExisting(HostFile(g.root, g.trace_files[step.file].path),
                                        FileAccess::kGenericRead, false);
      if (h) {
        std::lock_guard lock(g.mu);
        g.pool[step.file].push_back(std::move(h));
        ++g.pool_count;
        ++g.opens_pooled;
      }
      maybe_report();
      continue;
    }

    // Read step: keep only ranges still pending (the guest may have passed them), valid files only.
    std::vector<uint32_t> run;
    std::vector<std::vector<uint32_t>> runs;
    const uint32_t file = g.ranges[g.order[step.first]].file;
    const bool valid = FileIsValid(g, verdict, file);
    {
      std::unique_lock lock(g.mu);
      uint64_t want = 0;
      for (uint32_t i = step.first; i < step.last; ++i) want += g.ranges[g.order[i]].length;
      // Respect the hard cap: wait for the guest to consume, drop what it skipped.
      while (valid && g.resident > 0 && g.resident + want > g.cap && SecondsSinceMount() < 300.0) {
        DropBehindLocked(g);
        if (g.resident + want <= g.cap) break;
        g.cv.wait_for(lock, std::chrono::milliseconds(50));
        lock.unlock();
        maybe_report();
        lock.lock();
      }
      for (uint32_t i = step.first; i < step.last; ++i) {
        const uint32_t idx = g.order[i];
        Range& r = g.ranges[idx];
        if (r.state != kPending) continue;
        if (!valid) {
          r.state = kSkipped;
          continue;
        }
        if (!run.empty() && g.ranges[run.back()].offset + g.ranges[run.back()].length != r.offset) {
          runs.push_back(std::move(run));
          run.clear();
        }
        r.state = kReading;
        run.push_back(idx);
      }
      if (!run.empty()) runs.push_back(std::move(run));
      for (const auto& rn : runs) {
        g.resident += g.ranges[rn.back()].offset + g.ranges[rn.back()].length - g.ranges[rn.front()].offset;
      }
      g.peak = std::max(g.peak, g.resident);
    }
    for (const auto& rn : runs) {
      if (ReadRun(g, handles, rn)) {
        consecutive_errors = 0;
      } else if (++consecutive_errors >= 4) {
        REXFS_WARN("[io] preload: 4 SD errors in a row, stopping the preload (the guest reads the SD itself)");
        alive = false;
      }
    }
    maybe_report();
  }

  // Everything is queued or read. Keep the store and the pool alive while the guest still walks
  // through the trace, report, then release the memory.
  const double last_event_s =
      g.trace_events.empty() ? 0.0 : double(g.trace_events.back().t_us) / 1e6;
  const double end_s = std::max(last_event_s + 15.0, 20.0);
  while (SecondsSinceMount() < end_s && g.active.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    maybe_report();
  }
  std::lock_guard lock(g.mu);
  ReportLocked(g, "final");
  g.preload_done.store(true, std::memory_order_relaxed);
  for (auto& r : g.ranges) {
    r.data.reset();
    if (r.state == kPending || r.state == kReady || r.state == kReading) r.state = kDone;
  }
  for (auto& q : g.pool) q.clear();
  g.pool_count = 0;
  g.resident = 0;
  g.cv.notify_all();
}

}  // namespace

// ---------------------------------------------------------------------------------------------
// Public entry points
// ---------------------------------------------------------------------------------------------

void OnMount(HostPathDevice* device) {
  State& g = G();
  const bool want_record = REXCVAR_GET(sa2_io_trace_record);
  const bool want_preload = REXCVAR_GET(sa2_io_preload);
  if (!device || !device->is_read_only() || (!want_record && !want_preload)) {
    return;
  }
  {
    std::lock_guard lock(g.mu);
    if (g.primary_set) return;
    g.primary_set = true;
    g.device = device;
    g.root = device->host_path();
    g.t0 = Clock::now();
    g.trace_path = TracePath(g.root);
    g.cap = uint64_t(std::max(1, REXCVAR_GET(sa2_io_preload_mb))) << 20;
    std::string list = REXCVAR_GET(sa2_io_trace_markers);
    size_t at = 0;
    while (at <= list.size()) {
      size_t comma = list.find(',', at);
      if (comma == std::string::npos) comma = list.size();
      std::string m = Lower(list.substr(at, comma - at));
      if (!m.empty()) g.markers.push_back(std::move(m));
      at = comma + 1;
    }
    g.marker_done.assign(g.markers.size(), false);
  }

  if (want_record) {
    if (want_preload) {
      REXFS_WARN("[io] trace: sa2_io_trace_record is on, sa2_io_preload is ignored for this run");
    }
    g.recording = true;
    g.events.reserve(8192);
    g.active.store(true);
    REXFS_INFO("[io] trace: recording the reads of '{}' for {} s into {}",
               rex::path_to_utf8(g.root), std::max(5, REXCVAR_GET(sa2_io_trace_record_s)),
               rex::path_to_utf8(g.trace_path));
    if (!StartWorker("recorder", RecorderThread)) {
      g.recording = false;
      g.active.store(false);
    }
    return;
  }

  REXFS_INFO("[io] preload: stage: loading {}", rex::path_to_utf8(g.trace_path));
  const auto t = Clock::now();
  std::string why;
  if (!LoadTrace(g, &why)) {
    REXFS_INFO("[io] preload: disabled, {}", why);
    return;
  }
  try {
    BuildPlan(g);
  } catch (const std::bad_alloc&) {
    REXFS_WARN("[io] preload: disabled, out of memory building the plan");
    return;
  }
  REXFS_INFO("[io] preload: trace {} loaded in {:.1f} ms: {} events, {} files, {} distinct ranges "
             "({:.1f} MB planned, {} larger than {} KB left to the guest), {} SD steps, cap {} MB",
             rex::path_to_utf8(g.trace_path),
             std::chrono::duration<double, std::milli>(Clock::now() - t).count(),
             g.trace_events.size(), g.trace_files.size(), g.ranges.size(),
             double(g.planned_bytes) / 1048576.0, g.skipped_big,
             REXCVAR_GET(sa2_io_preload_max_range_kb), g.plan.size(), g.cap >> 20);
  g.preloading = true;
  g.active.store(true);
  if (!StartWorker("preload", PreloadThread)) {
    g.preloading = false;
    g.active.store(false);
  }
}

uint32_t OnOpen(HostPathDevice* device, const std::string& path) {
  State& g = G();
  if (!g.active.load(std::memory_order_relaxed) || device != g.device) {
    return 0;
  }
  std::lock_guard lock(g.mu);
  auto [it, fresh] = g.tags.try_emplace(path, uint32_t(g.files.size()));
  if (fresh) {
    auto fs = std::make_unique<FileState>();
    fs->path = path;
    if (g.preloading) {
      auto ti = g.trace_index.find(path);
      if (ti != g.trace_index.end()) fs->trace_file = ti->second;
    }
    g.files.push_back(std::move(fs));
  }
  const uint32_t file = it->second;
  if (g.recording) {
    PushEvent(g, 1, file, 0, 0);
  }
  return file + 1;
}

bool OnRead(uint32_t tag, uint64_t offset, std::span<uint8_t> buffer, size_t* out_bytes_read) {
  State& g = G();
  if (!tag || !g.active.load(std::memory_order_relaxed)) {
    return false;
  }
  const uint32_t length = uint32_t(buffer.size());
  std::shared_ptr<std::vector<uint8_t>> data;
  {
    std::unique_lock lock(g.mu);
    FileState& fs = *g.files[tag - 1];
    ++g.guest_reads;
    g.guest_bytes += length;

    if (!fs.first_read_logged) {
      fs.first_read_logged = true;
      const std::string lower = Lower(fs.path);
      for (size_t i = 0; i < g.markers.size(); ++i) {
        if (!g.marker_done[i] && lower.find(g.markers[i]) != std::string::npos) {
          g.marker_done[i] = true;
          REXFS_INFO("[io] timeline: {:.2f} s after the VFS mount, guest first reads '{}' "
                     "(marker '{}'; {} reads, {:.1f} MB read so far; preload {})",
                     SecondsSinceMount(), fs.path, g.markers[i], g.guest_reads,
                     double(g.guest_bytes) / 1048576.0,
                     g.preloading ? (g.preload_done.load() ? "finished" : "running")
                                  : (g.recording ? "off (recording)" : "off"));
          break;
        }
      }
    }

    if (g.recording) {
      PushEvent(g, 0, tag - 1, offset, length);
      return false;
    }
    if (!g.preloading || g.preload_done.load(std::memory_order_relaxed) || !length) {
      return false;
    }
    if (fs.trace_file < 0) {
      ++g.unmatched;
      return false;
    }
    auto it = g.range_index.find(std::make_tuple(uint32_t(fs.trace_file), offset, length));
    if (it == g.range_index.end()) {
      ++g.unmatched;
      return false;
    }
    Range& r = g.ranges[it->second];
    g.guest_pos = std::max(g.guest_pos, r.first_event);
    bool waited = false;
    if (r.state == kReading) {
      // The thread is reading it right now: waiting is cheaper than a second SD read.
      waited = true;
      g.cv.wait_for(lock, std::chrono::milliseconds(250), [&] { return r.state != kReading; });
    }
    switch (r.state) {
      case kReady: {
        data = r.data;
        g.hit_bytes += data ? data->size() : 0;
        ++g.hits;
        if (waited) ++g.waited;
        if (--r.uses_left == 0) {
          g.resident -= data ? data->size() : 0;
          r.data.reset();
          r.state = kDone;
        }
        g.cv.notify_all();
        break;
      }
      case kPending:
        r.state = kSkipped;  // the guest got there first: the thread will not read it
        ++g.late;
        return false;
      default:
        if (waited) {
          ++g.waited;
        } else {
          ++g.other_miss;
        }
        return false;
    }
  }
  if (!data) {
    return false;
  }
  const size_t n = std::min(data->size(), buffer.size());
  std::memcpy(buffer.data(), data->data(), n);
  *out_bytes_read = n;
  return true;
}

bool VerifyEnabled() {
  return REXCVAR_GET(sa2_io_preload_verify);
}

void ReportVerify(uint32_t tag, uint64_t offset, size_t length, bool equal) {
  State& g = G();
  std::lock_guard lock(g.mu);
  ++g.verified;
  if (!equal) {
    ++g.mismatches;
    if (g.mismatches <= 20) {
      REXFS_ERROR("[io] preload: VERIFY MISMATCH '{}' offset {} length {}",
                  g.files[tag - 1]->path, offset, length);
    }
  }
}

std::unique_ptr<FileHandle> TakePreopened(HostPathDevice* device, const std::string& path) {
  State& g = G();
  if (!g.active.load(std::memory_order_relaxed) || !g.preloading || device != g.device) {
    return nullptr;
  }
  std::lock_guard lock(g.mu);
  auto ti = g.trace_index.find(path);
  if (ti == g.trace_index.end() || ti->second >= g.pool.size() || g.pool[ti->second].empty()) {
    return nullptr;
  }
  auto h = std::move(g.pool[ti->second].front());
  g.pool[ti->second].pop_front();
  --g.pool_count;
  ++g.opens_taken;
  return h;
}

}  // namespace rex::filesystem::startup_trace
