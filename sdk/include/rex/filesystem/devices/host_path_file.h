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

#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <rex/filesystem.h>
#include <rex/filesystem/file.h>

namespace rex::filesystem {

class HostPathEntry;

/*
 * How much is being served from RAM instead of from the SD.
 *
 * hits      = game reads answered without going to disk.
 * fills     = times the disk had to be read to fill the window.
 * direct    = reads that bypassed the window (large ones, writes, save files).
 * bytes_ram = bytes the game received from the window.
 */
struct StatsWindow {
  uint64_t hits = 0;
  uint64_t filled = 0;
  uint64_t direct = 0;
  uint64_t bytes_ram = 0;
  uint64_t live_windows = 0;
};
StatsWindow ReadStatsWindow();

/*
 * Exact-range read cache. See sa2_io_ranges_mb in the .cpp.
 *
 * Everything is cumulative since startup, not per interval: there are few events and the total is
 * what matters.
 *
 * hits         = large reads answered from RAM, without going to the SD.
 * misses       = eligible large reads that did have to go to the SD.
 * bytes_ram    = bytes delivered to the game from the cache.
 * bytes_disk  = bytes those same reads brought from the SD.
 * entries      = ranges alive right now.
 * bytes_live  = what those ranges take in the host heap.
 * evictions    = ranges dropped because of the cap. If this grows, the cap is too small.
 * no_memory  = the cache turned itself off for lack of RAM.
 */
struct StatsRanges {
  uint64_t hits = 0;
  uint64_t failures = 0;
  uint64_t bytes_ram = 0;
  uint64_t bytes_disk = 0;
  uint64_t inputs = 0;
  uint64_t bytes_live = 0;
  uint64_t cap_drops = 0;
  uint64_t cap_mb = 0;  // the cvar, so the summary does not have to declare it on its own
  // Why something was not cached. Without this the cache could not be tuned.
  uint64_t low_floor = 0;          // reads that are too small
  uint64_t over_ceiling = 0;         // too large
  uint64_t sequential = 0;        // the level-load sweep, which is not kept
  uint64_t sequential_bytes = 0;
  uint64_t floor_kb = 0;
  bool no_memory = false;
};
StatsRanges ReadStatsRanges();

class HostPathFile : public File {
 public:
  HostPathFile(uint32_t file_access, HostPathEntry* entry,
               std::unique_ptr<rex::filesystem::FileHandle> file_handle);
  ~HostPathFile() override;

  void Destroy() override;

  X_STATUS ReadSync(std::span<uint8_t> buffer, size_t byte_offset, size_t* out_bytes_read) override;
  X_STATUS WriteSync(std::span<const uint8_t> buffer, size_t byte_offset,
                     size_t* out_bytes_written) override;
  X_STATUS SetLength(size_t length) override;
  X_STATUS Flush() override;
  X_STATUS Rename(const std::filesystem::path& file_path) override;

 private:
  /*
   * Read-ahead window.
   *
   * On Horizon every NtReadFile from the game became a pread() against the SD with no buffering in
   * between: the descriptor is opened with a bare open() (filesystem_posix.cpp), so there is not even the
   * stdio buffer. A file read header by header (the .gin, the .abk, the indices) means dozens of trips to
   * the card to read a few bytes each time.
   *
   * A window is read in one go and whatever falls inside is served from RAM. Files that fit whole in the
   * window end up entirely in memory, which is exactly what was wanted for the small, frequently opened
   * ones.
   *
   * Only on read-only devices. It is never enabled for saves and profiles: there the bytes change under
   * our feet and a stale window would mean a corrupted save.
   *
   * If a refill returns less than requested (end of file, or a short read from the file system, which
   * cannot be told apart from here), the request is answered with the usual direct read; after the
   * fourth in a row the window is turned off for this file. It never returns less than the path without
   * the window would have.
   *
   * The window used to be dead, and it cost something too. Measured (472.9 MB over 1,290 reads): 5 hits
   * out of 1,232 eligible reads = 0.41 %, 0.3 MB served from RAM out of 472.9 MB (0.06 %), and 66
   * refills x 64 KB = 4.1 MB of SD reads spent to serve those 0.3 MB. A net loss.
   *
   * The cause was the threshold: only requests of at most window/4 go into the window, i.e. 16 KB with a
   * 64 KB window. The game's average read is 375 KB; the smallest it does in a loop is 64 KB (it is in
   * the binary itself: a loop calls NtReadFile with r9 = 0x10000). Not a single game read
   * was under 16 KB, so all of them took the direct path.
   *
   * Fix: the window is now 256 KB, which puts the threshold at 64 KB and catches exactly the game's
   * 64 KB loop; and the cap on live windows drops from 128 to 16 (there were never more than 10), so the
   * RAM ceiling drops from 8 MB to 4 MB. It comes with an audit: if after 8 refills this file has not
   * given at least one hit per refill, the pattern is not sequential and the window is turned off for
   * it; that way it can never again be pure cost in silence.
   */
  bool FillWindow(size_t byte_offset, size_t request, std::span<uint8_t> buffer,
                       size_t* out_bytes_read, X_STATUS* out_status);

  // Direct read split into chunks. See the comment on sa2_io_chunk_mb in the .cpp.
  X_STATUS DirectRead(std::span<uint8_t> buffer, size_t byte_offset, size_t* out_bytes_read);

  std::unique_ptr<rex::filesystem::FileHandle> file_handle_;

  std::vector<uint8_t> window_;
  size_t window_start_ = 0;
  size_t window_bytes_ = 0;
  size_t window_size_ = 0;
  uint32_t short_window_ = 0;
  // Per-file audit: a window that does not hit turns itself off.
  uint32_t filled_window_ = 0;
  uint32_t window_hits_ = 0;
  bool active_window_ = false;
  bool counted_window_ = false;
  // File id inside the RAM cache (0 = this file is not cached). See
  // sa2_io_cache_mb.
  uint32_t cache_id_ = 0;
  // Tag in the startup trace (0 = not traced). See startup_trace.cpp.
  uint32_t trace_tag_ = 0;
};

}  // namespace rex::filesystem
