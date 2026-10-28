// mk9-nx 1.0.1: the game files on the SD card against the release's list (mk9_file_check.h).
//
// The list is app/mk9_game_files.txt (tools/game_file_list.py: "<size> <sha256> <path>" a line), compiled in as
// kMk9GameFiles (mk9_builtin_config.h). Two checks, both before the intro holds the display:
//   sizes   every start: each game folder of the list is read with fsDirRead, which gives every entry's size (64 a
//           call; MK9 has 6 folders, so a few milliseconds instead of a stat per file). Catches a file missing or cut
//           short, the usual result of an interrupted copy or extraction.
//   hashes  the start after a crash in the game's code: the SDK's guest crash hook writes the game files opened last
//           (cache/file_check_after_crash.txt) and this start hashes them (libnx's SHA-256) before anything else.
//           Catches a file of the right size with damaged bytes (a failing or fake SD card).
// A bad file: the system's error dialog names it (rex::ShowSimpleMessageBox) and the process ends the way
// windowed_app_main_switch.cpp's ExitProcess does. mk9_file_check = false turns both off (modified game files).

#include "mk9_file_check.h"

#include <rex/cvar.h>

REXCVAR_DEFINE_BOOL(mk9_file_check, true, "MK9",
                    "Switch: check the game files on the SD card against the release's list (their sizes at every "
                    "start, the SHA-256 of the files open at a crash at the next start) and name a bad one on screen")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(mk9_file_check_test, 0, "MK9",
                     "Tests only: 1 shows an error dialog 2 s into the intro and exits (the intro stops for it), 2 "
                     "writes the crash marker at the 40th game file opened, without crashing (the next start hashes "
                     "those files)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

#if defined(__SWITCH__) && defined(MK9_BUILTIN_CONFIG)

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <switch.h>

#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/system.h>

#include "mk9_builtin_config.h"

extern "C" {
extern u32 __nx_applet_exit_mode;
NX_NORETURN void __libnx_exit(int rc);
}

namespace mk9::file_check {
namespace {

struct GameFile {
  std::string path;  // relative to game_root, '/' separated
  uint64_t size = 0;
  std::array<uint8_t, 32> sha256{};
};

std::string Lower(std::string_view text) {
  std::string out(text);
  std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  return out;
}

std::vector<GameFile> ParseList() {
  std::vector<GameFile> files;
  std::string_view text = kMk9GameFiles;
  while (!text.empty()) {
    const size_t end = text.find('\n');
    std::string_view line = text.substr(0, end);
    text = end == std::string_view::npos ? std::string_view() : text.substr(end + 1);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t a = line.find(' ');
    const size_t b = a == std::string_view::npos ? a : line.find(' ', a + 1);
    if (b == std::string_view::npos || b - a - 1 != 64) {
      continue;
    }
    GameFile f;
    f.size = std::strtoull(std::string(line.substr(0, a)).c_str(), nullptr, 10);
    for (size_t i = 0; i < 32; ++i) {
      f.sha256[i] = uint8_t(std::stoul(std::string(line.substr(a + 1 + 2 * i, 2)), nullptr, 16));
    }
    f.path = std::string(line.substr(b + 1));
    files.push_back(std::move(f));
  }
  return files;
}

// "sdmc:/switch/mk9-nx/game_root" or "/switch/mk9-nx/game_root": the path on the SD's file system.
std::string OnCard(const std::filesystem::path& path) {
  std::string s = path.generic_string();
  if (s.rfind("sdmc:", 0) == 0) {
    s = s.substr(5);
  }
  while (s.size() > 1 && s.back() == '/') {
    s.pop_back();
  }
  return s;
}

std::string Bytes(uint64_t n) {
  std::string digits = std::to_string(n);
  for (int i = int(digits.size()) - 3; i > 0; i -= 3) {
    digits.insert(size_t(i), ",");
  }
  return digits;
}

[[noreturn]] void Exit() {
  REXLOG_INFO("Switch: exiting process (game files)");
  rex::FlushLogging();
  std::fflush(nullptr);
  __nx_applet_exit_mode = 1;
  __libnx_exit(1);
}

[[noreturn]] void ShowAndExit(const std::string& game_root, const std::vector<std::string>& problems) {
  for (const std::string& p : problems) {
    REXLOG_ERROR("[files] {}", p);
  }
  std::string text = "Mortal Kombat cannot start: " + std::to_string(problems.size()) +
                     (problems.size() == 1 ? " game file is" : " game files are") + " missing or damaged in sdmc:" +
                     game_root + ":\n\n";
  for (size_t i = 0; i < problems.size() && i < 8; ++i) {
    text += "- " + problems[i] + "\n";
  }
  if (problems.size() > 8) {
    text += "- and " + std::to_string(problems.size() - 8) + " more (logs/mk9_*.log has them all)\n";
  }
  text += "\nCopy the game_root folder from mk9-nx.zip to the SD card again (on a PC, with a card reader). If files "
          "are damaged again, test the SD card (for example with H2testw).";
  rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error, text);
  Exit();
}

// name (lower case) -> size of the files in one folder; false when the folder cannot be opened.
bool ListFolder(FsFileSystem* fs, const std::string& folder, std::map<std::string, uint64_t>& out) {
  FsDir dir;
  if (R_FAILED(fsFsOpenDirectory(fs, folder.c_str(), FsDirOpenMode_ReadFiles, &dir))) {
    return false;
  }
  std::vector<FsDirectoryEntry> entries(64);
  for (;;) {
    s64 count = 0;
    if (R_FAILED(fsDirRead(&dir, &count, entries.size(), entries.data())) || count <= 0) {
      break;
    }
    for (s64 i = 0; i < count; ++i) {
      out[Lower(entries[size_t(i)].name)] = uint64_t(entries[size_t(i)].file_size);
    }
  }
  fsDirClose(&dir);
  return true;
}

void CheckSizes(const std::string& root, const std::vector<GameFile>& files, std::vector<std::string>& problems) {
  FsFileSystem* fs = fsdevGetDeviceFileSystem("sdmc");
  if (!fs) {
    REXLOG_WARN("[files] no SD file system: sizes not checked");
    return;
  }
  std::map<std::string, std::map<std::string, uint64_t>> folders;  // folder (as listed) -> its files
  std::map<std::string, bool> opened;
  size_t checked = 0;
  for (const GameFile& f : files) {
    const size_t slash = f.path.rfind('/');
    const std::string folder = slash == std::string::npos ? std::string() : f.path.substr(0, slash);
    const std::string name = Lower(slash == std::string::npos ? f.path : f.path.substr(slash + 1));
    if (!opened.count(folder)) {
      opened[folder] = ListFolder(fs, folder.empty() ? root : root + "/" + folder, folders[folder]);
    }
    const auto& listed = folders[folder];
    const auto it = listed.find(name);
    if (it == listed.end()) {
      problems.push_back(f.path + ": missing");
    } else if (it->second != f.size) {
      problems.push_back(f.path + ": " + Bytes(it->second) + " bytes, should be " + Bytes(f.size));
    }
    ++checked;
  }
  REXLOG_INFO("[files] {} game files: sizes checked in {} folders, {} wrong", checked, folders.size(), problems.size());
}

// The crash hook's marker: the game files opened last, newest first, one relative path a line.
char g_marker[512] = {};
constexpr size_t kSlots = 32;
constexpr size_t kSlotLength = 200;
char g_opened[kSlots][kSlotLength] = {};
std::atomic<uint32_t> g_next{0};

void NoteCrash();

// D:\Asset\Core.xxx (or game:\...) -> Asset/Core.xxx; anything else (saves, UPDATE:, cache) is not a game file.
void NoteOpened(std::string_view guest_path) {
  const size_t colon = guest_path.find(":\\");
  if (colon == std::string_view::npos) {
    return;
  }
  const std::string device = Lower(guest_path.substr(0, colon));
  if (device != "d" && device != "game") {
    return;
  }
  std::string_view rel = guest_path.substr(colon + 2);
  if (rel.empty() || rel.size() >= kSlotLength) {
    return;
  }
  const uint32_t last = g_next.load(std::memory_order_relaxed);
  if (last && std::strncmp(g_opened[(last - 1) % kSlots], rel.data(), rel.size()) == 0 &&
      g_opened[(last - 1) % kSlots][rel.size()] == '\0') {
    return;  // the same file again
  }
  char* slot = g_opened[g_next.fetch_add(1, std::memory_order_relaxed) % kSlots];
  for (size_t i = 0; i < rel.size(); ++i) {
    slot[i] = rel[i] == '\\' ? '/' : rel[i];
  }
  slot[rel.size()] = '\0';
  // Tests (mk9_file_check_test = 2): the crash marker at the 40th game file opened, without a crash. (A thread of its
  // own sleeping 60 s left the whole log empty on the console.)
  if (REXCVAR_GET(mk9_file_check_test) == 2 && g_next.load(std::memory_order_relaxed) == 40) {
    REXLOG_INFO("[files] test: the crash marker written at the 40th game file opened");
    NoteCrash();
  }
}

// From the faulting thread: no allocation, no locks of the game's.
void NoteCrash() {
  if (!g_marker[0]) {
    return;
  }
  std::FILE* f = std::fopen(g_marker, "w");
  if (!f) {
    return;
  }
  const uint32_t next = g_next.load(std::memory_order_acquire);
  const uint32_t count = next < kSlots ? next : uint32_t(kSlots);
  for (uint32_t i = 1; i <= count; ++i) {
    std::fputs(g_opened[(next - i) % kSlots], f);
    std::fputc('\n', f);
  }
  std::fclose(f);
}

constexpr size_t kHashedAtMost = 16;

void CheckHashesAfterCrash(const std::string& root, const std::vector<GameFile>& files,
                           std::vector<std::string>& problems) {
  std::FILE* marker = std::fopen(g_marker, "r");
  if (!marker) {
    return;
  }
  std::vector<std::string> names;
  char line[kSlotLength + 2];
  while (std::fgets(line, sizeof(line), marker)) {
    std::string name(line);
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) {
      name.pop_back();
    }
    if (!name.empty() && std::find(names.begin(), names.end(), Lower(name)) == names.end()) {
      names.push_back(Lower(name));
    }
  }
  std::fclose(marker);
  std::remove(g_marker);  // once: a crash while hashing must not repeat this at every start

  std::map<std::string, const GameFile*> by_name;
  for (const GameFile& f : files) {
    by_name[Lower(f.path)] = &f;
  }
  const u64 started = armGetSystemTick();
  std::vector<uint8_t> buffer(1 << 20);
  size_t hashed = 0;
  uint64_t bytes = 0;
  for (const std::string& name : names) {
    const auto it = by_name.find(name);
    if (it == by_name.end() || hashed >= kHashedAtMost) {
      continue;
    }
    const GameFile& f = *it->second;
    ++hashed;
    std::FILE* in = std::fopen((root + "/" + f.path).c_str(), "rb");
    if (!in) {
      problems.push_back(f.path + ": cannot be read");
      continue;
    }
    Sha256Context ctx;
    sha256ContextCreate(&ctx);
    size_t got;
    while ((got = std::fread(buffer.data(), 1, buffer.size(), in)) > 0) {
      sha256ContextUpdate(&ctx, buffer.data(), got);
      bytes += got;
    }
    const bool read_error = std::ferror(in);
    std::fclose(in);
    std::array<uint8_t, 32> digest{};
    sha256ContextGetHash(&ctx, digest.data());
    if (read_error) {
      problems.push_back(f.path + ": cannot be read to the end");
    } else if (digest != f.sha256) {
      problems.push_back(f.path + ": damaged (its bytes differ from the game's)");
    }
  }
  REXLOG_INFO("[files] after a crash: {} game files opened last hashed ({:.1f} MB) in {:.1f} s, {} damaged", hashed,
              double(bytes) / (1024.0 * 1024.0), double(armTicksToNs(armGetSystemTick() - started)) / 1e9,
              problems.size());
}

}  // namespace

void AtStart(const std::filesystem::path& game_root) {
  if (!REXCVAR_GET(mk9_file_check) || game_root.empty()) {
    return;
  }
  const std::vector<GameFile> files = ParseList();
  if (files.empty()) {
    return;
  }
  const std::string root = OnCard(game_root);
  // The marker beside the VFS index: <folder of the NRO>/cache/.
  const std::string cache = OnCard(rex::filesystem::GetExecutableFolder() / "cache");
  std::error_code ec;
  std::filesystem::create_directories(cache, ec);
  std::snprintf(g_marker, sizeof(g_marker), "%s/file_check_after_crash.txt", cache.c_str());

  std::vector<std::string> problems;
  CheckSizes(root, files, problems);
  if (problems.empty()) {
    CheckHashesAfterCrash(root, files, problems);
  }
  if (!problems.empty()) {
    ShowAndExit(root, problems);
  }
  rex::SetFileOpenedHook(&NoteOpened);
  rex::SetGuestCrashHook(&NoteCrash);
}

void TestAfterIntroStarted() {
  switch (REXCVAR_GET(mk9_file_check_test)) {
    case 1:
      svcSleepThread(2'000'000'000);
      REXLOG_INFO("[files] test: an error dialog during the intro");
      rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error,
                                "mk9_file_check_test = 1: a startup error shown while the intro plays.");
      Exit();
    case 2:
      break;  // NoteOpened writes the marker at the 40th game file

    default:
      break;
  }
}

}  // namespace mk9::file_check

#else

namespace mk9::file_check {
void AtStart(const std::filesystem::path&) {}
void TestAfterIntroStarted() {}
}  // namespace mk9::file_check

#endif
