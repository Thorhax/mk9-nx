/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <rex/filesystem.h>
#include <rex/filesystem/device.h>

namespace rex::filesystem {

class HostPathEntry;

/*
 * How much SD work the path lookup saves.
 *
 * misses_in_dry  = times a missing path was resolved using only the in-memory tree (before, each of
 *                   these was a stat plus a full enumeration of the parent directory: ~32 ms).
 * stats_in_sd     = stats of the exact name that did reach the disk.
 * swept_in_sd  = full enumerations of a directory (the expensive part: one stat per entry).
 * inputs_in_tree = files and folders Initialize left loaded in memory.
 */
struct StatsPaths {
  uint64_t misses_in_dry = 0;
  uint64_t stats_in_sd = 0;
  uint64_t swept_in_sd = 0;
  uint64_t inputs_in_tree = 0;
};
StatsPaths ReadStatsPaths();

class HostPathDevice : public Device {
 public:
  HostPathDevice(const std::string_view mount_path, const std::filesystem::path& host_path,
                 bool read_only, bool allow_share_delete = false);
  ~HostPathDevice() override;

  bool Initialize() override;
  void Dump(string::StringBuffer* string_buffer) override;
  Entry* ResolvePath(const std::string_view path) override;

  bool is_read_only() const override { return read_only_; }
  // NOTE(tomc): When true, host file handles open with FILE_SHARE_DELETE so an open read
  // handle (e.g. a save-slot preview) does not block an overwrite's
  // delete+recreate. Opt-in per device: content/save devices set it; game-data
  // and read-only devices leave it off (see a5c3a963 ghost-file fix).
  bool allow_share_delete() const { return allow_share_delete_; }
  const std::filesystem::path& host_path() const { return host_path_; }

  const std::string& name() const override { return name_; }
  uint32_t attributes() const override { return 0; }
  uint32_t component_name_max_length() const override { return 255; }

  uint32_t total_allocation_units() const override { return 128 * 1024; }
  uint32_t available_allocation_units() const override { return 128 * 1024; }
  uint32_t sectors_per_allocation_unit() const override { return 1; }
  uint32_t bytes_per_sector() const override { return 0x200; }

 private:
  void PopulateEntry(HostPathEntry* parent_entry);
  std::vector<rex::filesystem::FileInfo> ListChildren(const std::filesystem::path& path);

  // On-disk index of the tree (backlog U2, host_path_index.cpp). Read-only devices only.
  bool ActiveIndex() const;
  std::filesystem::path PathIndex() const;
  bool LoadIndex(HostPathEntry* root, uint64_t* inputs, std::string* report);
  bool SaveIndex(HostPathEntry* root, uint64_t inputs, std::string* reason);

  std::string name_;
  std::filesystem::path host_path_;
  std::unique_ptr<Entry> root_entry_;
  bool read_only_;
  bool allow_share_delete_;
};

}  // namespace rex::filesystem
