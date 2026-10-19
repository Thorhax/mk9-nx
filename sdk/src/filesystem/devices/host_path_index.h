/**
 * Directory index of a read-only HostPathDevice, kept on disk between runs (backlog U2).
 *
 * Private to rexfilesystem. See host_path_index.cpp for the why and the validation rules.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rex::filesystem::vfs_index {

/*
 * One child of a directory as the directory listing itself reports it: name, type and size, with no
 * per-file stat. On Horizon this comes straight from fsDirRead (FsDirectoryEntry carries type and
 * size), so listing a directory of 1,000 files is a handful of IPC calls instead of thousands.
 */
struct LightEntry {
  std::string name;
  bool is_directory = false;
  uint64_t size = 0;  // 0 for directories, as ListFiles reports them
};

/*
 * Lists @p path without stat'ing each child. Returns false only when the directory could not be
 * opened; an empty directory returns true with @p output empty.
 *
 * Same type rule as ListFiles: only a real directory counts as one (ListFiles uses d_type ==
 * DT_DIR; anything else, symlinks included, is a file whose size is what stat says).
 */
bool ListLight(const std::string& path, std::vector<LightEntry>* output);

#if defined(__SWITCH__)
// The Horizon implementation lives in its own translation unit (host_path_listed_switch.cpp)
// so that switch.h does not leak into the rest of the filesystem code.
bool ListLightSwitch(const std::string& path, std::vector<LightEntry>* output);
#endif

}  // namespace rex::filesystem::vfs_index
