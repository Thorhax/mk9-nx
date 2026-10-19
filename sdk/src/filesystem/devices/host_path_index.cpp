/**
 * On-disk index of a read-only HostPathDevice tree (backlog U2).
 *
 * The problem. HostPathDevice::Initialize walks the whole game folder with ListFiles, which does one
 * stat per entry. On the Switch SD that is ~1.9 ms per entry: the console log has "[io] mounted
 * '/switch/masseffect-nx/game_root' ...: 4523 entries in 8556.7 ms", and nothing else happens during
 * those 8.7 s of every start.
 *
 * What this does. After a full scan the tree (names, types, sizes, the three timestamps and the
 * "children complete" mark of every directory) is written to one file, by default
 * <parent of the game folder>/cache/vfs_index_<game folder name>.bin, i.e.
 * sdmc:/switch/masseffect-nx/cache/vfs_index_game_root.bin. On the next start it is read in one
 * sequential read and the same HostPathEntry objects are rebuilt with HostPathEntry::Create, in the
 * same order, so lookups, case handling, sizes and directory enumeration see exactly the tree a full
 * scan would have built.
 *
 * Validation, and why not mtimes. On FAT32/exFAT a directory's modification time is not a reliable
 * signal: the FAT spec only sets it at creation, Windows and macOS do not update it when a child is
 * added, removed or rewritten, and Linux vfat does only for create/delete in that same directory.
 * The game folder gets changed exactly that way (switch_cycle.sh re-uploads
 * Layer0/MEInit/Coalesced.ini over FTP on every test run, with a different size whenever the ini
 * changes), so an mtime check would serve a stale size. Instead every directory of the cached tree is
 * listed again *without* stat'ing its children: on Horizon fsDirRead returns name, type and size for
 * 64 entries per IPC call. The game tree has 12 directories and 4,523 entries, so that is ~12 opens
 * and ~80 reads instead of ~4,500 stats. The set of (name, type, size) of every directory must match
 * the index exactly, otherwise the index is thrown away and the full scan runs (and rewrites it).
 *
 * What is NOT validated: the three timestamps. They come from the index, so a file replaced by
 * another of exactly the same size keeps its old timestamps in the guest's view until the next
 * rescan. File contents are always read from the SD, never from the index. Game data is read-only
 * and the game does not look at those timestamps for its own files; vfs_redo_index forces a fresh
 * scan if that ever matters.
 *
 * Any read, parse, checksum or validation failure falls back to the full scan. Writable devices
 * (saves, profiles) never use the index: their contents change under us.
 *
 * Format (host byte order; the file is only ever read by the machine that wrote it):
 *   header  : "RXVFSIDX" | u32 version | u32 0 | u64 entry count | u64 body size | u64 FNV-1a(body)
 *   body    : str host path | dir(root)
 *   dir     : u8 children complete | u32 child count | child * count
 *   child   : u8 type (0 file, 1 dir) | str name | u64 size | u64 create | u64 access | u64 write
 *             | dir (only when type == 1)
 *   str     : u32 length | bytes
 */

#include <rex/filesystem/devices/host_path_device.h>
#include <rex/filesystem/devices/host_path_entry.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <system_error>
#include <tuple>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <fmt/format.h>

#if !defined(_WIN32) && !defined(__SWITCH__)
#include <dirent.h>
#include <sys/stat.h>
#endif

#include "host_path_index.h"

REXCVAR_DEFINE_BOOL(vfs_index, true, "Filesystem",
                    "Saves the game folder tree to a file and on the following starts "
                    "reads it from there instead of scanning the SD (read-only devices "
                    "only). false = always a full scan and nothing is written");
REXCVAR_DEFINE_BOOL(vfs_redo_index, false, "Filesystem",
                    "Ignores the saved index: full scan and the index is rewritten");
REXCVAR_DEFINE_BOOL(vfs_validate_index, true, "Filesystem",
                    "Before using the index, relists every directory (without a stat per file) and "
                    "compares names, types and sizes. false = trust the index without looking at the SD "
                    "(for measuring only; after changing game files vfs_redo_index "
                    "has to be used)");
REXCVAR_DEFINE_STRING(vfs_path_index, "", "Filesystem",
                      "Index file. Empty = <game parent folder>/cache/"
                      "vfs_index_<folder name>.bin");

namespace rex::filesystem {

namespace vfs_index {

bool ListLight(const std::string& path, std::vector<LightEntry>* output) {
#if defined(__SWITCH__)
  return ListLightSwitch(path, output);
#elif defined(_WIN32)
  // No cheap listing here; ListFiles cannot tell "empty" from "failed", so empty counts as opened.
  output->clear();
  for (const auto& info : rex::filesystem::ListFiles(rex::to_path(path))) {
    LightEntry l;
    l.name = rex::path_to_utf8(info.name);
    l.is_directory = info.type == FileInfo::Type::kDirectory;
    l.size = l.is_directory ? 0 : uint64_t(info.total_size);
    output->push_back(std::move(l));
  }
  return true;
#else
  // Same rules as ListFiles in filesystem_posix.cpp (d_type decides, stat gives the size), only
  // without the timestamps and telling a failed opendir apart from an empty directory.
  output->clear();
  DIR* dir = opendir(path.c_str());
  if (!dir) {
    return false;
  }
  while (auto* ent = readdir(dir)) {
    if (std::strcmp(ent->d_name, ".") == 0 || std::strcmp(ent->d_name, "..") == 0) {
      continue;
    }
    LightEntry l;
    l.name = ent->d_name;
    l.is_directory = ent->d_type == DT_DIR;
    if (!l.is_directory) {
      struct stat st;
      std::string complete = path;
      if (complete.empty() || complete.back() != '/') {
        complete.push_back('/');
      }
      complete += l.name;
      l.size = stat(complete.c_str(), &st) == 0 ? uint64_t(st.st_size) : ~uint64_t(0);
    }
    output->push_back(std::move(l));
  }
  closedir(dir);
  return true;
#endif
}

namespace {

constexpr char kMagic[8] = {'R', 'X', 'V', 'F', 'S', 'I', 'D', 'X'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeader = 8 + 4 + 4 + 8 + 8 + 8;
// Sanity limits for parsing a damaged file: nothing in a game folder comes close.
constexpr uint32_t kMaxName = 4096;
constexpr uint32_t kMaxChildren = 1u << 20;
constexpr int kMaxDepth = 256;

uint64_t Fnv1a(const uint8_t* p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

class Writer {
 public:
  template <typename T>
  void Pod(T v) {
    const auto* p = reinterpret_cast<const uint8_t*>(&v);
    data_.insert(data_.end(), p, p + sizeof(T));
  }
  void String(const std::string& s) {
    Pod<uint32_t>(uint32_t(s.size()));
    data_.insert(data_.end(), s.begin(), s.end());
  }
  std::vector<uint8_t>& data() { return data_; }

 private:
  std::vector<uint8_t> data_;
};

class Reader {
 public:
  Reader(const uint8_t* p, size_t n) : p_(p), end_(p + n) {}
  template <typename T>
  bool Pod(T* v) {
    if (size_t(end_ - p_) < sizeof(T)) return false;
    std::memcpy(v, p_, sizeof(T));
    p_ += sizeof(T);
    return true;
  }
  bool String(std::string* s, uint32_t max) {
    uint32_t n = 0;
    if (!Pod(&n) || n > max || size_t(end_ - p_) < n) return false;
    s->assign(reinterpret_cast<const char*>(p_), n);
    p_ += n;
    return true;
  }
  bool at_end() const { return p_ == end_; }

 private:
  const uint8_t* p_;
  const uint8_t* end_;
};

using Key = std::tuple<std::string, bool, uint64_t>;

}  // namespace
}  // namespace vfs_index

using namespace vfs_index;

std::filesystem::path HostPathDevice::PathIndex() const {
  const std::string& configured = REXCVAR_GET(vfs_path_index);
  if (!configured.empty()) {
    return rex::to_path(configured);
  }
  auto folder = host_path_;
  if (!folder.has_filename()) {
    folder = folder.parent_path();  // "game_root/" -> "game_root"
  }
  return folder.parent_path() / "cache" /
         rex::to_path("vfs_index_" + rex::path_to_utf8(folder.filename()) + ".bin");
}

bool HostPathDevice::ActiveIndex() const {
  return read_only_ && REXCVAR_GET(vfs_index);
}

namespace {

void SerializeDirectory(Writer& w, HostPathEntry* dir, uint64_t* count) {
  w.Pod<uint8_t>(dir->children_complete() ? 1 : 0);
  w.Pod<uint32_t>(uint32_t(dir->children().size()));
  for (const auto& child_ptr : dir->children()) {
    auto* child = static_cast<HostPathEntry*>(child_ptr.get());
    const bool is_dir = (child->attributes() & kFileAttributeDirectory) != 0;
    w.Pod<uint8_t>(is_dir ? 1 : 0);
    // The name exactly as the host listing returned it (host_path's last component), which is
    // what PopulateEntry fed to HostPathEntry::Create.
    w.String(rex::path_to_utf8(child->host_path().filename()));
    w.Pod<uint64_t>(uint64_t(child->size()));
    w.Pod<uint64_t>(child->create_timestamp());
    w.Pod<uint64_t>(child->access_timestamp());
    w.Pod<uint64_t>(child->write_timestamp());
    ++*count;
    if (is_dir) {
      SerializeDirectory(w, child, count);
    }
  }
}

}  // namespace

bool HostPathDevice::SaveIndex(HostPathEntry* root, uint64_t inputs, std::string* reason) {
  Writer w;
  // Header placeholder, filled in at the end.
  w.data().resize(kHeader);
  w.String(rex::path_to_utf8(host_path_));
  uint64_t count = 0;
  SerializeDirectory(w, root, &count);
  if (count != inputs) {
    *reason = fmt::format("the tree has {} entries and the scan counted {}", count, inputs);
    return false;
  }
  auto& data = w.data();
  const uint64_t size_body = data.size() - kHeader;
  const uint64_t hash = Fnv1a(data.data() + kHeader, size_t(size_body));
  uint8_t* c = data.data();
  std::memcpy(c, kMagic, 8);
  const uint32_t version = kVersion, zero = 0;
  std::memcpy(c + 8, &version, 4);
  std::memcpy(c + 12, &zero, 4);
  std::memcpy(c + 16, &count, 8);
  std::memcpy(c + 24, &size_body, 8);
  std::memcpy(c + 32, &hash, 8);

  const auto path = PathIndex();
  std::error_code ec;
  std::filesystem::create_directories(path.parent_path(), ec);
  auto temporal = path;
  temporal += ".tmp";
  FILE* f = rex::filesystem::OpenFile(temporal, "wb");
  if (!f) {
    *reason = "could not create " + rex::path_to_utf8(temporal);
    return false;
  }
  const bool written = std::fwrite(data.data(), 1, data.size(), f) == data.size();
  const bool closed = std::fclose(f) == 0;
  if (!written || !closed) {
    std::filesystem::remove(temporal, ec);
    *reason = "failed to write " + rex::path_to_utf8(temporal);
    return false;
  }
  /*
   * Temp file + rename, so a reader never sees a half-written index. Horizon's rename (like
   * Windows') refuses to replace an existing file, so on failure the old one is removed first and
   * the rename retried. In that window there is no index at all, which only means one more full scan.
   */
  // std::filesystem::rename takes the native path type (wide on Windows).
  std::filesystem::rename(temporal, path, ec);
  if (ec) {
    std::filesystem::remove(path, ec);
    std::filesystem::rename(temporal, path, ec);
    if (ec) {
      std::filesystem::remove(temporal, ec);
      *reason = "could not rename to " + rex::path_to_utf8(path);
      return false;
    }
  }
  return true;
}

namespace {

struct Context {
  HostPathDevice* device;
  uint64_t inputs = 0;
  std::string reason;
};

bool BuildDirectory(Context& ctx, Reader& r, HostPathEntry* dir, int depth,
                         bool* complete_out) {
  uint8_t complete = 0;
  uint32_t n = 0;
  if (!r.Pod(&complete) || !r.Pod(&n) || complete > 1 || n > kMaxChildren) {
    ctx.reason = "corrupt data";
    return false;
  }
  *complete_out = complete != 0;
  for (uint32_t i = 0; i < n; ++i) {
    uint8_t type = 0;
    FileInfo info;
    std::string name;
    uint64_t size = 0, ct = 0, at = 0, wt = 0;
    if (!r.Pod(&type) || type > 1 || !r.String(&name, kMaxName) || name.empty() ||
        !r.Pod(&size) || !r.Pod(&ct) || !r.Pod(&at) || !r.Pod(&wt)) {
      ctx.reason = "corrupt data";
      return false;
    }
    info.type = type ? FileInfo::Type::kDirectory : FileInfo::Type::kFile;
    info.name = rex::to_path(name);
    info.path = dir->host_path();
    info.total_size = size_t(size);
    info.create_timestamp = ct;
    info.access_timestamp = at;
    info.write_timestamp = wt;
    auto* child = HostPathEntry::Create(ctx.device, dir, dir->host_path() / info.name, info);
    dir->AddChildIndex(child);
    ++ctx.inputs;
    if (type) {
      if (depth >= kMaxDepth) {
        ctx.reason = "too deep";
        return false;
      }
      bool complete_child = false;
      if (!BuildDirectory(ctx, r, child, depth + 1, &complete_child)) {
        return false;
      }
      child->MarkCompleteIndex(complete_child);
    }
  }
  return true;
}

// Compares one directory of the rebuilt tree with what the SD lists now, then its subdirectories.
bool ValidateDirectory(HostPathEntry* dir, uint64_t* directories, std::string* reason) {
  ++*directories;
  std::vector<LightEntry> disk;
  const std::string path = rex::path_to_utf8(dir->host_path());
  if (!ListLight(path, &disk)) {
    // The full scan would also have got nothing here (and left the directory unmarked).
    if (dir->children().empty() && !dir->children_complete()) {
      return true;
    }
    *reason = "could not list " + path;
    return false;
  }
  if (disk.size() != dir->children().size()) {
    *reason = fmt::format("{}: {} entries on the SD and {} in the index", path, disk.size(),
                          dir->children().size());
    return false;
  }
  std::vector<Key> a, b;
  a.reserve(disk.size());
  b.reserve(disk.size());
  for (auto& e : disk) {
    a.emplace_back(std::move(e.name), e.is_directory, e.size);
  }
  for (const auto& h : dir->children()) {
    auto* child = static_cast<HostPathEntry*>(h.get());
    const bool is_dir = (child->attributes() & kFileAttributeDirectory) != 0;
    b.emplace_back(rex::path_to_utf8(child->host_path().filename()), is_dir,
                   is_dir ? 0 : uint64_t(child->size()));
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  if (a != b) {
    for (size_t i = 0; i < a.size(); ++i) {
      if (a[i] != b[i]) {
        *reason = fmt::format("{}: '{}' ({} bytes) on the SD, '{}' ({} bytes) in the index", path,
                              std::get<0>(a[i]), std::get<2>(a[i]), std::get<0>(b[i]),
                              std::get<2>(b[i]));
        break;
      }
    }
    return false;
  }
  // A non-empty directory the full scan read is always marked complete; keep that invariant.
  if (dir->children_complete() != !dir->children().empty()) {
    *reason = path + ": inconsistent 'complete' mark";
    return false;
  }
  for (const auto& h : dir->children()) {
    auto* child = static_cast<HostPathEntry*>(h.get());
    if ((child->attributes() & kFileAttributeDirectory) &&
        !ValidateDirectory(child, directories, reason)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool HostPathDevice::LoadIndex(HostPathEntry* root, uint64_t* inputs, std::string* report) {
  const auto t0 = std::chrono::steady_clock::now();
  const auto path = PathIndex();
  FILE* f = rex::filesystem::OpenFile(path, "rb");
  if (!f) {
    *report = "no index at " + rex::path_to_utf8(path);
    return false;
  }
  std::vector<uint8_t> data;
  bool read = false;
  if (std::fseek(f, 0, SEEK_END) == 0) {
    const long size = std::ftell(f);
    if (size >= long(kHeader) && std::fseek(f, 0, SEEK_SET) == 0) {
      data.resize(size_t(size));
      read = std::fread(data.data(), 1, data.size(), f) == data.size();
    }
  }
  std::fclose(f);
  if (!read) {
    *report = "could not read " + rex::path_to_utf8(path);
    return false;
  }
  const double ms_read =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  uint32_t version = 0;
  uint64_t count = 0, size_body = 0, hash = 0;
  std::memcpy(&version, data.data() + 8, 4);
  std::memcpy(&count, data.data() + 16, 8);
  std::memcpy(&size_body, data.data() + 24, 8);
  std::memcpy(&hash, data.data() + 32, 8);
  if (std::memcmp(data.data(), kMagic, 8) != 0 || version != kVersion) {
    *report = "index from another version";
    return false;
  }
  if (size_body != data.size() - kHeader ||
      Fnv1a(data.data() + kHeader, size_t(size_body)) != hash) {
    *report = "damaged index (size or checksum)";
    return false;
  }

  Reader r(data.data() + kHeader, size_t(size_body));
  std::string saved_path;
  if (!r.String(&saved_path, 1u << 16) || saved_path != rex::path_to_utf8(host_path_)) {
    *report = "the index is from another folder (" + saved_path + ")";
    return false;
  }

  Context ctx;
  ctx.device = this;
  bool complete_root = false;
  auto discard = [&](const std::string& why) {
    root->DiscardChildrenIndex();
    *report = why;
    return false;
  };
  if (!BuildDirectory(ctx, r, root, 0, &complete_root)) {
    return discard(ctx.reason);
  }
  root->MarkCompleteIndex(complete_root);
  if (!r.at_end() || ctx.inputs != count) {
    return discard("corrupt data");
  }
  const auto t1 = std::chrono::steady_clock::now();

  uint64_t directories = 0;
  std::string reason;
  if (REXCVAR_GET(vfs_validate_index)) {
    if (!ValidateDirectory(root, &directories, &reason)) {
      return discard("the SD does not match the index: " + reason);
    }
  }
  const auto t2 = std::chrono::steady_clock::now();
  *inputs = ctx.inputs;
  *report = fmt::format(
      "index {} ({} KB): read {:.1f} ms, tree {:.1f} ms, {}",
      rex::path_to_utf8(path), data.size() / 1024, ms_read,
      std::chrono::duration<double, std::milli>(t1 - t0).count() - ms_read,
      REXCVAR_GET(vfs_validate_index)
          ? fmt::format("validated {} directories in {:.1f} ms", directories,
                        std::chrono::duration<double, std::milli>(t2 - t1).count())
          : std::string("NOT validated (vfs_validate_index=false)"));
  return true;
}

}  // namespace rex::filesystem
