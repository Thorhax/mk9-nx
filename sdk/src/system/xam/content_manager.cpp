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

#include <array>
#include <cstring>
#include <queue>
#include <string>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/filesystem/devices/host_path_device.h>
#include <rex/filesystem/devices/stfs_container_device.h>
#include <rex/logging.h>
#include <rex/string.h>
#include <rex/string/utf8.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xam/content_device.h>
#include <rex/system/xam/content_manager.h>
#include <rex/system/xfile.h>
#include <rex/system/xobject.h>

#include "../../filesystem/devices/host_path_index.h"

/*
 * Readable copy of the saves and a trace of what happens to them.
 *
 * All of the game's saves go through here: XamContentCreate opens a container, the game writes inside it
 * as if it were a drive (SAV:, for example) and XamContentClose closes it. Where they end up is opaque
 * (content_root/xuid/title/type/name), so the player cannot see the saves and, when one comes out
 * "damaged", there is no way to tell whether the failure was in writing or in reading.
 *
 * Two additions:
 *   - A "[save]" trace in the log: every create, open and close, with the real path and the files
 *     inside with their sizes. A 0-byte save, or one that never gets written, shows at a glance.
 *   - content_backup_root: if it is not empty, when a container is closed it is copied whole to
 *     <content_backup_root>/<profile>/current, and whatever was there before moves to <profile>/previous.
 *     Besides giving readable copies of the saves, sorted by profile, it always keeps a previous
 *     version to recover a profile that got corrupted.
 */
REXCVAR_DEFINE_STRING(content_backup_root, "", "Kernel",
                      "Folder where a copy of each save is left, with one subfolder per profile. "
                      "Empty = nothing is copied");

namespace rex {
namespace system {
namespace xam {

static const char* kThumbnailFileName = "__thumbnail.png";

static const char* kGameUserContentDirName = "profile";

static const char* kGameContentHeaderDirName = "Headers";

static int content_device_id_ = 0;

namespace {

// A folder name the SD accepts, derived from the one the player sees.
std::string SafeName(const std::string_view original) {
  std::string clean;
  for (const char c : original) {
    const unsigned char u = static_cast<unsigned char>(c);
    if (u < 0x20 || u == 0x7F) {
      continue;
    }
    // 0x5C is the backslash.
    if (c == '/' || c == ':' || c == '*' || c == '?' || c == '"' || c == '<' || c == '>' ||
        c == '|' || u == 0x5C) {
      continue;
    }
    clean.push_back(c);
  }
  while (!clean.empty() && (clean.back() == ' ' || clean.back() == '.')) {
    clean.pop_back();
  }
  return clean.empty() ? std::string("no_name") : clean;
}

// What is inside a container, for the log. It is the line that says whether the save has data or not.
// Listed without a stat per file (ListLight: name, type and size come with the listing): with ListFiles
// this line alone cost ~350 ms per open and per close of USF4's 314-file save on the Switch SD.
std::string ThatHasInside(const std::filesystem::path& folder) {
  std::string text;
  uint64_t total = 0;
  uint32_t how_many = 0;
  std::vector<rex::filesystem::vfs_index::LightEntry> listed;
  rex::filesystem::vfs_index::ListLight(rex::path_to_utf8(folder), &listed);
  for (const auto& f : listed) {
    if (f.is_directory) {
      continue;
    }
    const uint64_t size = f.size == ~uint64_t(0) ? 0 : f.size;
    if (!text.empty()) {
      text += ", ";
    }
    text += fmt::format("{} ({} B)", f.name, size);
    total += size;
    ++how_many;
  }
  if (how_many == 0) {
    return "EMPTY";
  }
  return fmt::format("{} file(s), {} B: {}", how_many, total, text);
}

bool CopyFile(const std::filesystem::path& source, const std::filesystem::path& target) {
  FILE* entry = rex::filesystem::OpenFile(source, "rb");
  if (!entry) {
    return false;
  }
  FILE* output = rex::filesystem::OpenFile(target, "wb");
  if (!output) {
    fclose(entry);
    return false;
  }
  std::vector<uint8_t> buffer(64 * 1024);
  bool ok = true;
  while (true) {
    const size_t read = fread(buffer.data(), 1, buffer.size(), entry);
    if (read == 0) {
      break;
    }
    if (fwrite(buffer.data(), 1, read, output) != read) {
      ok = false;
      break;
    }
  }
  if (ferror(entry)) {
    ok = false;
  }
  fclose(entry);
  if (fclose(output) != 0) {
    ok = false;
  }
  return ok;
}

// Saves are small, so copying them by hand is cheaper than fighting std::filesystem on Horizon.
bool CopyFolder(const std::filesystem::path& source, const std::filesystem::path& target,
                   uint32_t& files) {
  std::error_code ec;
  std::filesystem::create_directories(target, ec);
  bool ok = true;
  for (const auto& f : rex::filesystem::ListFiles(source)) {
    if (f.type == rex::filesystem::FileInfo::Type::kDirectory) {
      ok = CopyFolder(source / f.name, target / f.name, files) && ok;
      continue;
    }
    if (CopyFile(source / f.name, target / f.name)) {
      ++files;
    } else {
      ok = false;
    }
  }
  return ok;
}

// The readable copy: <content_backup_root>/<profile>/current, and the previous one to <profile>/previous.
void CopyForTheUser(const std::string_view name, const std::filesystem::path& source) {
  const std::string root = REXCVAR_GET(content_backup_root);
  if (root.empty()) {
    return;
  }
  const std::string profile = SafeName(name);
  const std::filesystem::path base = std::filesystem::path(root) / profile;
  const std::filesystem::path actual = base / "current";
  const std::filesystem::path previous = base / "previous";

  std::error_code ec;
  if (std::filesystem::exists(actual, ec)) {
    std::filesystem::remove_all(previous, ec);
    ec.clear();
    std::filesystem::rename(actual, previous, ec);
    if (ec) {
      // If renaming is not allowed, copy and delete: what matters is not losing the previous version.
      uint32_t n = 0;
      CopyFolder(actual, previous, n);
      std::filesystem::remove_all(actual, ec);
      ec.clear();
    }
  }

  uint32_t files = 0;
  const bool ok = CopyFolder(source, actual, files);
  if (ok) {
    REXSYS_INFO("[save] copy of '{}': {} file(s) in {}", profile, files,
                rex::path_to_utf8(actual));
  } else {
    REXSYS_WARN("[save] the copy of '{}' failed halfway ({} file(s) in {})", profile,
                files, rex::path_to_utf8(actual));
  }
}

}  // namespace

ContentPackage::ContentPackage(KernelState* kernel_state, const std::string_view root_name,
                               const XCONTENT_AGGREGATE_DATA& data,
                               const std::filesystem::path& package_path)
    : kernel_state_(kernel_state), root_name_(root_name), package_path_(package_path), license_(0) {
  device_path_ = fmt::format("\\Device\\Content\\{0}\\", ++content_device_id_);
  content_data_ = data;

  auto fs = kernel_state_->file_system();
  auto device = std::make_unique<rex::filesystem::HostPathDevice>(device_path_, package_path, false,
                                                                  /*allow_share_delete=*/true);
  device->Initialize();
  fs->RegisterDevice(std::move(device));
  fs->RegisterSymbolicLink(root_name_ + ":", device_path_);
}

ContentPackage::~ContentPackage() {
  auto fs = kernel_state_->file_system();
  fs->UnregisterSymbolicLink(root_name_ + ":");
  fs->UnregisterDevice(device_path_);
}

void ContentPackage::LoadPackageLicenseMask(const std::filesystem::path header_path) {
  if (!std::filesystem::exists(header_path)) {
    return;
  }

  auto file = rex::filesystem::OpenFile(header_path, "rb");
  if (!file) {
    return;
  }

  auto file_size = std::filesystem::file_size(header_path);
  if (file_size < sizeof(XCONTENT_AGGREGATE_DATA) + sizeof(license_)) {
    fclose(file);
    return;
  }

  fseek(file, sizeof(XCONTENT_AGGREGATE_DATA), SEEK_SET);
  fread(&license_, 1, sizeof(license_), file);
  fclose(file);
}

ContentManager::ContentManager(KernelState* kernel_state, const std::filesystem::path& root_path)
    : kernel_state_(kernel_state), root_path_(root_path) {}

ContentManager::~ContentManager() = default;

std::filesystem::path ContentManager::ResolvePackageRoot(uint64_t xuid, XContentType content_type,
                                                         uint32_t title_id) {
  if (title_id == kCurrentlyRunningTitleId) {
    title_id = kernel_state_->title_id();
  }
  auto xuid_str = fmt::format("{:016X}", xuid);
  auto title_id_str = fmt::format("{:08X}", title_id);
  auto content_type_str = fmt::format("{:08X}", uint32_t(content_type));

  // Package root path:
  // content_root/xuid/title_id/content_type/
  return root_path_ / xuid_str / title_id_str / content_type_str;
}

std::filesystem::path ContentManager::ResolvePackagePath(uint64_t xuid,
                                                         const XCONTENT_AGGREGATE_DATA& data) {
  uint64_t used_xuid = (data.xuid != uint64_t(-1) && data.xuid != 0) ? uint64_t(data.xuid) : xuid;

  // DLCs are stored in common directory
  if (data.content_type == XContentType::kMarketplaceContent) {
    used_xuid = 0;
  }

  // Content path:
  // content_root/xuid/title_id/content_type/data_file_name/
  auto package_root = ResolvePackageRoot(used_xuid, data.content_type, data.title_id);
  return package_root / rex::to_path(data.file_name());
}

std::filesystem::path ContentManager::ResolvePackageHeaderPath(const std::string_view file_name,
                                                               uint64_t xuid, uint32_t title_id,
                                                               XContentType content_type) const {
  if (title_id == kCurrentlyRunningTitleId) {
    title_id = kernel_state_->title_id();
  }

  if (content_type == XContentType::kMarketplaceContent) {
    xuid = 0;
  }

  auto xuid_str = fmt::format("{:016X}", xuid);
  auto title_id_str = fmt::format("{:08X}", title_id);
  auto content_type_str = fmt::format("{:08X}", uint32_t(content_type));
  std::string final_name = std::string(file_name) + ".header";

  // Header root path:
  // content_root/xuid/title_id/Headers/content_type/filename.header
  return root_path_ / xuid_str / title_id_str / kGameContentHeaderDirName / content_type_str /
         final_name;
}

std::vector<XCONTENT_AGGREGATE_DATA> ContentManager::ListContent(uint32_t device_id, uint64_t xuid,
                                                                 XContentType content_type,
                                                                 uint32_t title_id) {
  std::vector<XCONTENT_AGGREGATE_DATA> result;

  if (title_id == kCurrentlyRunningTitleId) {
    title_id = kernel_state_->title_id();
  }

  // Search path:
  // content_root/xuid/title_id/type_name/*
  auto package_root = ResolvePackageRoot(xuid, content_type, title_id);
  auto file_infos = rex::filesystem::ListFiles(package_root);
  for (const auto& file_info : file_infos) {
    if (file_info.type != rex::filesystem::FileInfo::Type::kDirectory) {
      // Directories only.
      continue;
    }

    XCONTENT_AGGREGATE_DATA content_data;
    if (XSUCCEEDED(ReadContentHeaderFile(rex::path_to_utf8(file_info.name), xuid, title_id,
                                         content_type, content_data))) {
      result.emplace_back(std::move(content_data));
    } else {
      content_data.device_id = device_id;
      content_data.content_type = content_type;
      content_data.set_display_name(rex::path_to_utf16(file_info.name));
      content_data.set_file_name(rex::path_to_utf8(file_info.name));
      content_data.title_id = title_id;
      content_data.xuid = xuid;
      result.emplace_back(std::move(content_data));
    }
  }

  // What the game sees when it asks for the list of saves. If a profile that exists on disk does not show
  // up here, the problem is the enumeration; if it shows up and is then reported as damaged, the problem
  // is its contents.
  REXSYS_INFO("[save] list type {:08X} in {}: {} entry(ies)", uint32_t(content_type),
              rex::path_to_utf8(package_root), result.size());
  for (const auto& entry : result) {
    REXSYS_INFO("[save]   '{}' (displayed as '{}')", entry.file_name(),
                rex::string::to_utf8(entry.display_name()));
  }

  return result;
}

std::unique_ptr<ContentPackage> ContentManager::ResolvePackage(
    const std::string_view root_name, uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data) {
  auto package_path = ResolvePackagePath(xuid, data);
  if (!std::filesystem::exists(package_path)) {
    return nullptr;
  }
  auto package = std::make_unique<ContentPackage>(kernel_state_, root_name, data, package_path);
  return package;
}

bool ContentManager::ContentExists(uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data) {
  auto path = ResolvePackagePath(xuid, data);
  return std::filesystem::exists(path);
}

X_RESULT ContentManager::WriteContentHeaderFile(uint64_t xuid, XCONTENT_AGGREGATE_DATA data,
                                                uint32_t license_mask) {
  if (data.title_id == uint32_t(-1)) {
    data.title_id = kernel_state_->title_id();
  }
  if (data.xuid == uint64_t(-1)) {
    data.xuid = xuid;
  }
  uint64_t used_xuid = (data.xuid != uint64_t(-1) && data.xuid != 0) ? uint64_t(data.xuid) : xuid;

  auto header_path =
      ResolvePackageHeaderPath(data.file_name(), used_xuid, data.title_id, data.content_type);
  auto parent_path = header_path.parent_path();

  if (!std::filesystem::exists(parent_path)) {
    if (!std::filesystem::create_directories(parent_path)) {
      return X_ERROR_ACCESS_DENIED;
    }
  }

  rex::filesystem::CreateEmptyFile(header_path);

  auto file = rex::filesystem::OpenFile(header_path, "wb");
  if (!file) {
    return X_ERROR_FILE_NOT_FOUND;
  }
  fwrite(&data, 1, sizeof(XCONTENT_AGGREGATE_DATA), file);
  if (license_mask != 0) {
    fwrite(&license_mask, 1, sizeof(license_mask), file);
  }
  const bool ok = fclose(file) == 0;
  REXSYS_INFO("[save] header of '{}' {} in {}", data.file_name(),
              ok ? "written" : "FAILED on close", rex::path_to_utf8(header_path));
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::ReadContentHeaderFile(const std::string_view file_name, uint64_t xuid,
                                               uint32_t title_id, XContentType content_type,
                                               XCONTENT_AGGREGATE_DATA& data) const {
  auto header_file_path = ResolvePackageHeaderPath(file_name, xuid, title_id, content_type);
  constexpr uint32_t header_size = sizeof(XCONTENT_AGGREGATE_DATA);

  if (!std::filesystem::exists(header_file_path)) {
    return X_ERROR_FILE_NOT_FOUND;
  }

  auto file = rex::filesystem::OpenFile(header_file_path, "rb");
  if (!file) {
    return X_ERROR_FILE_NOT_FOUND;
  }

  auto file_size = std::filesystem::file_size(header_file_path);
  if (file_size < header_size) {
    fclose(file);
    return X_ERROR_FILE_NOT_FOUND;
  }

  std::array<uint8_t, header_size> buffer;
  size_t result = fread(buffer.data(), 1, header_size, file);
  fclose(file);

  if (result != header_size) {
    return X_ERROR_FILE_NOT_FOUND;
  }

  std::memcpy(&data, buffer.data(), buffer.size());
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::CreateContent(const std::string_view root_name, uint64_t xuid,
                                       const XCONTENT_AGGREGATE_DATA& data) {
  {
    auto global_lock = global_critical_region_.Acquire();
    if (open_packages_.count(string::string_key_case(root_name))) {
      return X_ERROR_ALREADY_EXISTS;
    }
  }

  auto package_path = ResolvePackagePath(xuid, data);
  REXSYS_INFO("[save] create '{}' (root {}) in {}", data.file_name(), root_name,
              rex::path_to_utf8(package_path));
  if (std::filesystem::exists(package_path)) {
    REXSYS_WARN("[save] create '{}': it already existed, returning ALREADY_EXISTS", data.file_name());
    return X_ERROR_ALREADY_EXISTS;
  }
  if (!std::filesystem::create_directories(package_path)) {
    REXSYS_ERROR("[save] create '{}': could NOT create {}", data.file_name(),
                 rex::path_to_utf8(package_path));
    return X_ERROR_ACCESS_DENIED;
  }
  auto package = ResolvePackage(root_name, xuid, data);
  assert_not_null(package);

  {
    auto global_lock = global_critical_region_.Acquire();
    if (open_packages_.count(string::string_key_case(root_name))) {
      return X_ERROR_ALREADY_EXISTS;
    }
    open_packages_.insert({string::string_key_case::create(root_name), package.release()});
  }
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::OpenContent(const std::string_view root_name, uint64_t xuid,
                                     const XCONTENT_AGGREGATE_DATA& data,
                                     uint32_t& content_license) {
  {
    auto global_lock = global_critical_region_.Acquire();
    if (open_packages_.count(string::string_key_case(root_name))) {
      return X_ERROR_ALREADY_EXISTS;
    }
  }

  auto package_path = ResolvePackagePath(xuid, data);
  if (!std::filesystem::exists(package_path)) {
    REXSYS_WARN("[save] open '{}': {} does not exist", data.file_name(),
                rex::path_to_utf8(package_path));
    return X_ERROR_FILE_NOT_FOUND;
  }
  // What the game will find inside. If this says EMPTY and the profile is then reported as damaged, the
  // failure was in saving; if it lists the files with their sizes, the failure is in reading them or in
  // the contents themselves.
  REXSYS_INFO("[save] open '{}' (root {}) in {} -> {}", data.file_name(), root_name,
              rex::path_to_utf8(package_path), ThatHasInside(package_path));
  auto package = ResolvePackage(root_name, xuid, data);
  assert_not_null(package);
  package->LoadPackageLicenseMask(ResolvePackageHeaderPath(
      data.file_name(), xuid, kernel_state_->title_id(), data.content_type));
  content_license = package->GetPackageLicense();

  {
    auto global_lock = global_critical_region_.Acquire();
    if (open_packages_.count(string::string_key_case(root_name))) {
      return X_ERROR_ALREADY_EXISTS;
    }
    open_packages_.insert({string::string_key_case::create(root_name), package.release()});
  }
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::CloseContent(const std::string_view root_name) {
  ContentPackage* package = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();
    // Some games use different casing between Create and Close (e.g. "save" vs "SAVE")
    auto it = open_packages_.find(string::string_key_case(root_name));
    if (it == open_packages_.end()) {
      return X_ERROR_FILE_NOT_FOUND;
    }
    package = DetachPackage(it);
  }
  // The path and name are recorded before destroying the package, which is what holds them.
  const std::filesystem::path path = package->package_path();
  const std::string name = package->GetPackageContentData().file_name();
  delete package;  // unmounts the guest drive: from here on the files are in place

  REXSYS_INFO("[save] close '{}' (root {}) in {} -> {}", name, root_name,
              rex::path_to_utf8(path), ThatHasInside(path));
  CopyForTheUser(name, path);
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::GetContentThumbnail(uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data,
                                             std::vector<uint8_t>* buffer) {
  auto global_lock = global_critical_region_.Acquire();
  auto package_path = ResolvePackagePath(xuid, data);
  auto thumb_path = package_path / kThumbnailFileName;
  if (std::filesystem::exists(thumb_path)) {
    auto file = rex::filesystem::OpenFile(thumb_path, "rb");
    fseek(file, 0, SEEK_END);
    size_t file_len = ftell(file);
    fseek(file, 0, SEEK_SET);
    buffer->resize(file_len);
    fread(const_cast<uint8_t*>(buffer->data()), 1, buffer->size(), file);
    fclose(file);
    return X_ERROR_SUCCESS;
  } else {
    return X_ERROR_FILE_NOT_FOUND;
  }
}

X_RESULT ContentManager::SetContentThumbnail(uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data,
                                             std::vector<uint8_t> buffer) {
  auto global_lock = global_critical_region_.Acquire();
  auto package_path = ResolvePackagePath(xuid, data);
  std::filesystem::create_directories(package_path);
  if (std::filesystem::exists(package_path)) {
    auto thumb_path = package_path / kThumbnailFileName;
    auto file = rex::filesystem::OpenFile(thumb_path, "wb");
    fwrite(buffer.data(), 1, buffer.size(), file);
    fclose(file);
    return X_ERROR_SUCCESS;
  } else {
    return X_ERROR_FILE_NOT_FOUND;
  }
}

X_RESULT ContentManager::DeleteContent(uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data) {
  auto global_lock = global_critical_region_.Acquire();

  if (IsContentOpen(data)) {
    // TODO(Gliniak): Get real error code for this case.
    return X_ERROR_ACCESS_DENIED;
  }

  auto package_path = ResolvePackagePath(xuid, data);
  std::error_code ec;
  auto dir_removed = std::filesystem::remove_all(package_path, ec);
  if (ec) {
    return X_ERROR_ACCESS_DENIED;
  }

  uint64_t used_xuid = (data.xuid != uint64_t(-1) && data.xuid != 0) ? uint64_t(data.xuid) : xuid;
  auto header_path =
      ResolvePackageHeaderPath(data.file_name(), used_xuid, data.title_id, data.content_type);
  std::error_code ec2;
  bool header_removed = std::filesystem::remove(header_path, ec2);

  if (dir_removed > 0 || header_removed) {
    return X_ERROR_SUCCESS;
  }
  return X_ERROR_FILE_NOT_FOUND;
}

X_RESULT ContentManager::UnmountContent(uint64_t xuid, const XCONTENT_AGGREGATE_DATA& data) {
  ContentPackage* package = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = FindOpenPackageByData(data);
    if (it == open_packages_.end()) {
      return X_ERROR_FILE_NOT_FOUND;
    }
    package = DetachPackage(it);
  }
  delete package;
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::UnmountAndDeleteContent(uint64_t xuid,
                                                 const XCONTENT_AGGREGATE_DATA& data) {
  // Unmount phase: tolerant of not-mounted state
  ContentPackage* package = nullptr;
  {
    auto global_lock = global_critical_region_.Acquire();
    auto it = FindOpenPackageByData(data);
    if (it != open_packages_.end()) {
      package = DetachPackage(it);
    }
  }
  delete package;

  // Delete phase: remove package directory and .header file
  auto package_path = ResolvePackagePath(xuid, data);

  uint64_t used_xuid = (data.xuid != uint64_t(-1) && data.xuid != 0) ? uint64_t(data.xuid) : xuid;
  auto header_path =
      ResolvePackageHeaderPath(data.file_name(), used_xuid, data.title_id, data.content_type);

  std::error_code ec;
  auto dir_removed = std::filesystem::remove_all(package_path, ec);
  if (ec) {
    return X_ERROR_ACCESS_DENIED;
  }

  std::error_code ec2;
  bool header_removed = std::filesystem::remove(header_path, ec2);

  if (dir_removed > 0 || header_removed) {
    return X_ERROR_SUCCESS;
  }
  return X_ERROR_FILE_NOT_FOUND;
}

std::filesystem::path ContentManager::ResolveGameUserContentPath() {
  auto title_id = fmt::format("{:08X}", kernel_state_->title_id());
  auto user_name = rex::to_path(kernel_state_->user_profile()->name());

  // Per-game per-profile data location:
  // content_root/title_id/profile/user_name
  return root_path_ / title_id / kGameUserContentDirName / user_name;
}

std::unordered_map<string::string_key_case, ContentPackage*,
                   string::string_key_case::Hash>::iterator
ContentManager::FindOpenPackageByData(const XCONTENT_AGGREGATE_DATA& data) {
  // Resolve kCurrentlyRunningTitleId so both sides compare actual title IDs.
  uint32_t query_title = data.title_id;
  if (query_title == kCurrentlyRunningTitleId) {
    query_title = kernel_state_->title_id();
  }

  for (auto it = open_packages_.begin(); it != open_packages_.end(); ++it) {
    const auto& pkg = it->second->GetPackageContentData();

    uint32_t pkg_title = pkg.title_id;
    if (pkg_title == kCurrentlyRunningTitleId) {
      pkg_title = kernel_state_->title_id();
    }

    // Match on content_type + file_name + resolved title_id.
    // device_id is a virtual storage selector, not a content identifier.
    if (data.content_type == pkg.content_type && data.file_name() == pkg.file_name() &&
        query_title == pkg_title) {
      return it;
    }
  }
  return open_packages_.end();
}

ContentPackage* ContentManager::DetachPackage(
    std::unordered_map<string::string_key_case, ContentPackage*,
                       string::string_key_case::Hash>::iterator it) {
  CloseOpenedFilesFromContent(it->first.view());
  ContentPackage* package = it->second;
  open_packages_.erase(it);
  return package;
}

bool ContentManager::IsContentOpen(const XCONTENT_AGGREGATE_DATA& data) const {
  return std::any_of(open_packages_.cbegin(), open_packages_.cend(), [&data](const auto& content) {
    return data == content.second->GetPackageContentData();
  });
}

std::filesystem::path ContentManager::GetOpenPackagePath(const std::string_view root_name) const {
  auto it = open_packages_.find(string::string_key_case(root_name));
  if (it == open_packages_.end()) {
    return {};
  }
  return it->second->package_path();
}

void ContentManager::CloseOpenedFilesFromContent(const std::string_view root_name) {
  // TODO(Gliniak): Cleanup this code to care only about handles
  // related to provided content
  const std::vector<object_ref<XFile>> all_files_handles =
      kernel_state_->object_table()->GetObjectsByType<XFile>(XObject::Type::File);

  std::string resolved_path = "";
  kernel_state_->file_system()->FindSymbolicLink(std::string(root_name) + ':', resolved_path);

  for (const object_ref<XFile>& file : all_files_handles) {
    std::string file_path = file->entry()->absolute_path();
    bool is_file_inside_content = rex::string::utf8_starts_with(file_path, resolved_path);

    if (is_file_inside_content) {
      file->ReleaseHandle();
    }
  }
}

static X_RESULT ExtractEntry(rex::filesystem::Entry* entry,
                             const std::filesystem::path& base_path) {
  auto dest_path = base_path / rex::to_path(rex::string::utf8_fix_path_separators(entry->path()));

  if (entry->attributes() & rex::filesystem::kFileAttributeDirectory) {
    std::error_code ec;
    std::filesystem::create_directories(dest_path, ec);
    if (ec) {
      return X_ERROR_ACCESS_DENIED;
    }
    return X_ERROR_SUCCESS;
  }

  // Ensure parent directory exists
  std::error_code ec;
  std::filesystem::create_directories(dest_path.parent_path(), ec);

  rex::filesystem::File* in_file = nullptr;
  X_STATUS status = entry->Open(rex::filesystem::FileAccess::kFileReadData, &in_file);
  if (status != X_STATUS_SUCCESS) {
    return X_ERROR_ACCESS_DENIED;
  }

  auto out_file = rex::filesystem::OpenFile(dest_path, "wb");
  if (!out_file) {
    in_file->Destroy();
    return X_ERROR_ACCESS_DENIED;
  }

  constexpr size_t kBufferSize = 4 * 1024 * 1024;  // 4 MiB
  auto buffer = std::make_unique<uint8_t[]>(kBufferSize);
  size_t remaining = entry->size();
  size_t offset = 0;

  while (remaining > 0) {
    size_t bytes_read = 0;
    size_t to_read = std::min(remaining, kBufferSize);
    in_file->ReadSync(std::span<uint8_t>(buffer.get(), to_read), offset, &bytes_read);
    if (bytes_read == 0) {
      break;
    }
    fwrite(buffer.get(), 1, bytes_read, out_file);
    offset += bytes_read;
    remaining -= bytes_read;
  }

  fclose(out_file);
  in_file->Destroy();
  return X_ERROR_SUCCESS;
}

X_RESULT ContentManager::InstallContent(const std::filesystem::path& package_path) {
  if (!std::filesystem::exists(package_path)) {
    return X_ERROR_FILE_NOT_FOUND;
  }

  // Mount the STFS package as a virtual filesystem device
  auto device = std::make_unique<rex::filesystem::StfsContainerDevice>("", package_path);
  if (!device->Initialize()) {
    return X_ERROR_ACCESS_DENIED;
  }

  // Derive install destination:
  // root_path_/0000000000000000/{title_id}/00000002/{filename}/
  auto file_name = rex::path_to_utf8(package_path.filename());

  XCONTENT_AGGREGATE_DATA content_data;
  content_data.device_id = static_cast<uint32_t>(DummyDeviceId::HDD);
  content_data.content_type = XContentType::kMarketplaceContent;
  content_data.title_id = kernel_state_->title_id();
  content_data.xuid = 0;
  content_data.set_file_name(file_name);

  // Read display name from STFS metadata
  auto display_name = device->header().metadata.display_name(rex::system::XLanguage::kEnglish);
  if (!display_name.empty()) {
    content_data.set_display_name(display_name);
  } else {
    content_data.set_display_name(rex::path_to_utf16(package_path.filename()));
  }

  auto install_path = ResolvePackagePath(0, content_data);

  // Create destination directory
  std::error_code ec;
  std::filesystem::create_directories(install_path, ec);
  if (ec) {
    return X_ERROR_ACCESS_DENIED;
  }

  // Extract all files breadth-first
  auto* root = device->ResolvePath("");
  if (!root) {
    return X_ERROR_ACCESS_DENIED;
  }

  std::queue<rex::filesystem::Entry*> queue;
  queue.push(root);

  while (!queue.empty()) {
    auto* entry = queue.front();
    queue.pop();

    for (auto& child : entry->children()) {
      queue.push(child.get());
    }

    auto result = ExtractEntry(entry, install_path);
    if (result != X_ERROR_SUCCESS) {
      return result;
    }
  }

  // Compute license mask from STFS header licenses
  uint32_t license_mask = 0;
  for (size_t i = 0; i < 0x10; i++) {
    if (device->header().header.licenses[i].license_flags) {
      license_mask |= device->header().header.licenses[i].license_bits;
    }
  }

  // Write .header file
  return WriteContentHeaderFile(0, content_data, license_mask);
}

}  // namespace xam
}  // namespace system
}  // namespace rex
