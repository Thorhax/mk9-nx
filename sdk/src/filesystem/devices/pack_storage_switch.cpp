/**
 * Horizon storage of the asset pack (pack_device.h): an FsFile read at explicit offsets.
 *
 * fsFileRead takes the offset with each call, so readers on several threads (the game's loader, the ADX streams)
 * need no shared lock and no seek, unlike newlib's FILE*. Big reads go in 1 MB pieces so another reader's request
 * can be served between them. Kept apart because switch.h brings u8/Result into the global namespace.
 */

#if defined(__SWITCH__)

#include <switch.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <memory>

#include <rex/filesystem/devices/pack_device.h>
#include <rex/logging.h>
#include <rex/string.h>

namespace rex::filesystem {

namespace {

class PackStorageSwitch final : public PackStorage {
 public:
  explicit PackStorageSwitch(const FsFile& file) : file_(file) {}
  ~PackStorageSwitch() override { fsFileClose(&file_); }

  bool ReadAt(uint64_t offset, void* buffer, size_t size) override {
    auto* out = static_cast<uint8_t*>(buffer);
    while (size) {
      const size_t piece = std::min<size_t>(size, 1 << 20);
      u64 read = 0;
      Result rc = fsFileRead(&file_, s64(offset), out, piece, FsReadOption_None, &read);
      if (R_FAILED(rc)) {
        // The destination is guest memory, read into directly through IPC. Should the file system ever refuse a
        // range of it as a transfer buffer (memory mapped another way), read the piece into host memory and copy.
        static thread_local std::unique_ptr<uint8_t[]> bounce;
        if (!bounce) bounce = std::make_unique<uint8_t[]>(1 << 20);
        const Result direct = rc;
        rc = fsFileRead(&file_, s64(offset), bounce.get(), piece, FsReadOption_None, &read);
        if (R_SUCCEEDED(rc)) {
          std::memcpy(out, bounce.get(), size_t(read));
          static std::atomic<int> logged{0};
          if (logged.fetch_add(1, std::memory_order_relaxed) < 8) {
            REXFS_WARN("[pack] a direct read of {} bytes at {} failed (result {:08X}); read through a host buffer",
                       piece, offset, direct);
          }
        }
      }
      if (R_FAILED(rc) || read == 0) {
        REXFS_ERROR("[pack] read of {} bytes at {} failed (result {:08X}, {} bytes)", piece, offset, rc, read);
        return false;
      }
      // A short read is not an error: Horizon's file system returns fewer bytes than asked for some large reads
      // (host_path_file.cpp says the same for loose files). Ask again from where it stopped. Treating it as a
      // failure gave Sonic Adventure 2 zero bytes for the 16.5 MB gd/SOC/arasuji.pak (story mode synopsis), and the
      // game retried that read forever on its loading screen (console, 2026-10-06).
      out += read;
      offset += read;
      size -= read;
    }
    return true;
  }

 private:
  FsFile file_;
};

}  // namespace

std::unique_ptr<PackStorage> PackStorage::Open(const std::filesystem::path& path) {
  const std::string utf8 = rex::path_to_utf8(path);
  FsFileSystem* fs = nullptr;
  char path_fs[FS_MAX_PATH];
  std::memset(path_fs, 0, sizeof(path_fs));
  if (fsdevTranslatePath(utf8.c_str(), &fs, path_fs) == -1 || !fs) {
    REXFS_ERROR("[pack] cannot translate {}", utf8);
    return nullptr;
  }
  FsFile file;
  if (R_FAILED(fsFsOpenFile(fs, path_fs, FsOpenMode_Read, &file))) {
    REXFS_ERROR("[pack] cannot open {}", utf8);
    return nullptr;
  }
  return std::make_unique<PackStorageSwitch>(file);
}

}  // namespace rex::filesystem

#endif  // __SWITCH__
