/**
 * Horizon directory listing for the VFS index validation (backlog U2).
 *
 * Kept apart on purpose: switch.h brings u8/Result/BIT and friends into the global namespace.
 */

#if defined(__SWITCH__)

#include <switch.h>

#include <cstring>
#include <string>
#include <vector>

#include "host_path_index.h"

namespace rex::filesystem::vfs_index {

/*
 * Why not readdir. newlib's readdir on libnx does get the type and the size from fsDirRead, but it
 * throws the size away (struct dirent has no room for it), so ListFiles stats every file to learn it.
 * That stat is what makes the full scan cost ~1.9 ms per file on the SD (4,523 entries = 8.7 s).
 * Here the same FsDirectoryEntry records are read directly, 64 per IPC, size included.
 */
bool ListLightSwitch(const std::string& path, std::vector<LightEntry>* output) {
  output->clear();
  FsFileSystem* fs = nullptr;
  char path_fs[FS_MAX_PATH];
  std::memset(path_fs, 0, sizeof(path_fs));
  if (fsdevTranslatePath(path.c_str(), &fs, path_fs) == -1 || !fs) {
    return false;
  }
  FsDir dir;
  if (R_FAILED(fsFsOpenDirectory(fs, path_fs, FsDirOpenMode_ReadDirs | FsDirOpenMode_ReadFiles,
                                 &dir))) {
    return false;
  }
  constexpr size_t kBatch = 64;
  std::vector<FsDirectoryEntry> batch(kBatch);
  bool ok = true;
  for (;;) {
    s64 read = 0;
    if (R_FAILED(fsDirRead(&dir, &read, kBatch, batch.data()))) {
      ok = false;
      break;
    }
    if (read <= 0) {
      break;
    }
    for (s64 i = 0; i < read; ++i) {
      const FsDirectoryEntry& e = batch[size_t(i)];
      LightEntry l;
      l.name.assign(e.name, strnlen(e.name, sizeof(e.name)));
      if (l.name == "." || l.name == "..") {
        continue;
      }
      l.is_directory = e.type == FsDirEntryType_Dir;
      l.size = l.is_directory ? 0 : uint64_t(e.file_size);
      output->push_back(std::move(l));
    }
  }
  fsDirClose(&dir);
  return ok;
}

}  // namespace rex::filesystem::vfs_index

#endif  // __SWITCH__
