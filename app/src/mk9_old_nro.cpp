// mk9-nx 1.0.4: an older mk9.nro beside mk9-nx.nro (mk9_old_nro.h).

#include "mk9_old_nro.h"

#include <rex/cvar.h>

REXCVAR_DEFINE_BOOL(mk9_replace_old_nro, true, "MK9",
                    "Switch: started as mk9-nx.nro, replace another mk9.nro in the folder by this game (forwarders "
                    "made for mk9.nro then start it); started as mk9.nro, take a newer mk9-nx.nro beside it and ask to "
                    "start again")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

#if defined(__SWITCH__)

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include <switch.h>

#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/system.h>

extern "C" {
extern u32 __nx_applet_exit_mode;
NX_NORETURN void __libnx_exit(int rc);
}

namespace mk9::old_nro {
namespace {

constexpr long kSample = 1 << 20;

long FileSize(FILE* f) {
  if (std::fseek(f, 0, SEEK_END) != 0) {
    return -1;
  }
  return std::ftell(f);
}

// The NACP's display version ("1.0.4"): the NRO header's size at 0x18 is where the asset header
// ("ASET", version, icon, NACP, RomFS as offset/size pairs from it) starts; the version is at 0x3060 of the NACP.
std::string Version(const std::string& path) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    return {};
  }
  std::string version;
  uint32_t nro_size = 0;
  char aset[4] = {};
  uint64_t nacp_offset = 0, nacp_size = 0;
  char text[0x10] = {};
  if (std::fseek(f, 0x18, SEEK_SET) == 0 && std::fread(&nro_size, 4, 1, f) == 1 &&
      std::fseek(f, long(nro_size), SEEK_SET) == 0 && std::fread(aset, 1, 4, f) == 4 &&
      std::memcmp(aset, "ASET", 4) == 0 && std::fseek(f, long(nro_size) + 0x18, SEEK_SET) == 0 &&
      std::fread(&nacp_offset, 8, 1, f) == 1 && std::fread(&nacp_size, 8, 1, f) == 1 && nacp_size >= 0x3070 &&
      std::fseek(f, long(nro_size + nacp_offset + 0x3060), SEEK_SET) == 0 && std::fread(text, 1, 0x10, f) == 0x10) {
    version.assign(text, strnlen(text, sizeof(text)));
  }
  std::fclose(f);
  return version;
}

// "1.0.4" -> {1, 0, 4}
std::array<int, 3> Numbers(const std::string& version) {
  std::array<int, 3> n{};
  std::sscanf(version.c_str(), "%d.%d.%d", &n[0], &n[1], &n[2]);
  return n;
}

// The same file: the same size and the same first, middle and last megabyte (reading both whole would take seconds at
// every start).
bool Same(const std::string& a, const std::string& b) {
  FILE* fa = std::fopen(a.c_str(), "rb");
  FILE* fb = std::fopen(b.c_str(), "rb");
  bool same = fa && fb;
  const long size = same ? FileSize(fa) : -1;
  same = same && size > 0 && FileSize(fb) == size;
  std::vector<char> ba(kSample), bb(kSample);
  for (const long at : {0L, size / 2, size - kSample}) {
    if (!same) {
      break;
    }
    const long from = at < 0 ? 0 : at;
    const size_t n = size_t(std::min(kSample, size - from));
    same = std::fseek(fa, from, SEEK_SET) == 0 && std::fseek(fb, from, SEEK_SET) == 0 &&
           std::fread(ba.data(), 1, n, fa) == n && std::fread(bb.data(), 1, n, fb) == n &&
           std::memcmp(ba.data(), bb.data(), n) == 0;
  }
  if (fa) std::fclose(fa);
  if (fb) std::fclose(fb);
  return same;
}

// `from` over `old`: written beside it first, then put in its place, so an interrupted copy never leaves a broken
// mk9.nro (the next start makes it again).
bool Replace(const std::string& from, const std::string& old) {
  const std::string tmp = old + ".new";
  FILE* in = std::fopen(from.c_str(), "rb");
  FILE* out = in ? std::fopen(tmp.c_str(), "wb") : nullptr;
  bool ok = in && out;
  std::vector<char> buffer(4 << 20);
  while (ok) {
    const size_t n = std::fread(buffer.data(), 1, buffer.size(), in);
    if (n == 0) {
      ok = !std::ferror(in);
      break;
    }
    ok = std::fwrite(buffer.data(), 1, n, out) == n;
  }
  if (in) std::fclose(in);
  if (out && std::fclose(out) != 0) ok = false;
  std::error_code ec;
  if (ok) {
    std::filesystem::remove(old, ec);
    std::filesystem::rename(tmp, old, ec);
    ok = !ec;
  }
  if (!ok) {
    std::filesystem::remove(tmp, ec);
    REXLOG_WARN("[nro] {} could not be replaced by {}", old, from);
    return false;
  }
  REXLOG_INFO("[nro] {} replaced by {} (forwarders made for it start this version now)", old, from);
  return true;
}

// The copy's thread: libnx's (std::thread::detach throws on this platform: the first build ended there, 2026-11-05),
// created at a priority the kernel takes (0x3E: 0xE001, invalid priority) and lowered below the game's threads once it
// runs, as the uam thread is. It is never waited for: a game closed during the copy leaves mk9.nro.new, which the next
// start writes again.
Thread g_thread;
std::string g_self, g_old;

void ReplaceMain(void*) {
  svcSetThreadPriority(CUR_THREAD_HANDLE, 0x3B);
  Replace(g_self, g_old);
}

}  // namespace

void AtStart() {
  if (!REXCVAR_GET(mk9_replace_old_nro)) {
    return;
  }
  const std::filesystem::path exe = rex::filesystem::GetExecutablePath();
  const std::string stem = exe.stem().string();
  const std::string self = exe.string();
  if (stem == "mk9-nx") {
    const std::string old = (exe.parent_path() / "mk9.nro").string();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(old, ec) || Same(self, old)) {
      return;
    }
    REXLOG_INFO("[nro] {} is another build ({}; this is {}): replacing it", old, Version(old), Version(self));
    g_self = self;
    g_old = old;
    Result rc = threadCreate(&g_thread, ReplaceMain, nullptr, nullptr, 0x10000, 0x2C, 2);
    if (R_SUCCEEDED(rc)) {
      rc = threadStart(&g_thread);
    }
    if (R_FAILED(rc)) {
      REXLOG_WARN("[nro] no thread to replace {}: 0x{:08X}", old, rc);
    }
  } else if (stem == "mk9") {
    const std::string next = (exe.parent_path() / "mk9-nx.nro").string();
    std::error_code ec;
    if (!std::filesystem::is_regular_file(next, ec)) {
      return;
    }
    const std::string mine = Version(self), theirs = Version(next);
    if (mine.empty() || theirs.empty() || !(Numbers(theirs) > Numbers(mine))) {
      return;
    }
    // Not envSetNextLoad: the game's threads are already running here, and when the loader takes the process back and
    // unloads this NRO they run into unmapped code (an instruction abort, on the console, 2026-11-05). This file takes
    // the new game, and the next start is it.
    REXLOG_INFO("[nro] {} is newer ({} > {}): taking it", next, theirs, mine);
    if (!Replace(next, self)) {
      return;
    }
    // Error: the only type the Switch shows (system_posix.cpp), the system's dialog
    rex::ShowSimpleMessageBox(rex::SimpleMessageBoxType::Error,
                              "Mortal Kombat was updated to version " + theirs + ". Please start the game again.");
    rex::FlushLogging();
    std::fflush(nullptr);
    __nx_applet_exit_mode = 1;
    __libnx_exit(0);
  }
}

}  // namespace mk9::old_nro

#else

namespace mk9::old_nro {
void AtStart() {}
}  // namespace mk9::old_nro

#endif
