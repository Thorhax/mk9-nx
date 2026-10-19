/*
 * Docked clocks when Reverse-NX says docked.
 *
 * Reverse-NX only lies to the game. Nobody lies to the sysmodule that sets the clocks (sys-clk and its
 * forks, including Horizon OC's sys-clk-plus): to choose between the handheld and docked columns of the
 * profile it asks the hardware (`Board::GetProfile()` -> `apmExtGetPerformanceMode()`), so with the
 * console in hand it always takes the handheld column, whether the profile is the game's or the global
 * one.
 *
 * sys-clk has a command to learn about Reverse-NX (`SysClkIpcCmd_SetReverseNXRTMode`), but in Horizon OC
 * it is empty: `IpcService::SetReverseNXRTMode(mode) { return 0; }`, and the call to
 * `rnxSync->ToggleSync(...)` is commented out in its clock_manager. That is why nothing happens however
 * much the mode is changed.
 *
 * What does work in Horizon OC is the manual override, which is what its own overlay uses when the
 * sliders are moved, and which takes precedence over everything in its loop:
 *
 *     targetHz = overrideFreqs[module];                                       // 1st: what sys:clk is told
 *     if (!targetHz) targetHz = GetAutoClockHz(applicationId, ..., profile);  // 2nd: this game's profile
 *     if (!targetHz) targetHz = GetAutoClockHz(GLOBAL_PROFILE_ID, ..., ...);  // 3rd: the global one
 *
 * So here the sysmodule itself is asked for the numbers configured for the docked column (command 5,
 * GetProfiles, first the game's and otherwise the global profile's, the same way its loop does) and
 * they are sent with command 8 (SetOverride) while Reverse-NX says docked. On returning to handheld
 * they are released.
 *
 * No frequency is invented: if nothing is set in that column, nothing is touched.
 *
 * Careful: the override does not expire. It is released on returning to handheld, when the switch is
 * turned off and when the game exits cleanly (destructor), but if the game closes abnormally it stays
 * set until it is touched in the Horizon OC overlay or the console is restarted.
 *
 * The expensive lesson: `smGetService` for a service that is not registered does not return an error,
 * it leaves the thread waiting forever for someone to register it. That hung the profiler thread as
 * soon as docked mode was selected (Horizon OC does not register "sys:clk"), and with that thread the
 * Reverse-NX heartbeat stopped too (hence the mode never returned to Handheld) and, behind that sm
 * request, the audio. Now sm command 65100 (AtmosphereHasService) is asked first, which answers yes or
 * no and never blocks.
 *
 * References: Horizon-OC/sys-clk-plus (common/include/sysclk/ipc.h, sysmodule/src/clock_manager.cpp and
 * config.cpp); Atmosphere-NX/Atmosphere (sm_user_interface.hpp) for command 65100.
 */

#include "rex/ui/switch_sysclk.h"

#if REX_PLATFORM_SWITCH

#include <switch.h>

#include <atomic>
#include <cstdio>
#include <cstring>

/* This object is compiled separately (rexglue_switch_startup) and has no fmt: warnings go to stderr. */

namespace rex::ui::switch_sysclk {
namespace {

constexpr unsigned kModules = 3;   // CPU, GPU, MEM
constexpr unsigned kProfiles = 5;  // handheld, +charging, +USB charging, +official charger, docked
constexpr unsigned kDockedProfile = 4;
constexpr unsigned kHandheldProfile = 0;

/*
 * Stock handheld clocks, used when nothing is set in that column. They are the factory ones, without a
 * drop of overclock: CPU the same in both modes (not touched), GPU 307.2 and memory 1331.2. With the
 * console in the dock the sysmodule takes the docked column and stays at 768/1600, so for Reverse-NX's
 * "Fake Handheld" to mean anything they have to be forced.
 */
constexpr uint32_t kHandheldOfSeriesHz[kModules] = {0u, 307200000u, 1331200000u};
constexpr uint64_t kGlobalProfile = 0xA111111111111111ull;  // sys-clk GLOBAL_PROFILE_ID

constexpr uint32_t kOrderProfiles = 5;   // SysClkIpcCmd_GetProfiles
constexpr uint32_t kOrderOverride = 8;   // SysClkIpcCmd_SetOverride

/* SysClkTitleProfileList: the configured MHz, per profile and module. */
struct ProfilesList {
  uint32_t mhz[kProfiles][kModules];
};

/* SysClkIpc_SetOverride_Args. */
struct ArgsOverride {
  uint32_t shader_module;
  uint32_t hz;
};

void Warning(const char* text) { std::fprintf(stderr, "%s\n", text); }
void WarningNum(const char* text, long long number) {
  std::fprintf(stderr, "%s 0x%llX (%lld)\n", text, (unsigned long long)number, number);
}

Service g_service{};
bool g_open = false;
int g_attempts = 0;
bool g_applied = false;             // we are forcing the clocks of a mode that is not the real one
bool g_applied_docked = false;   // and in which direction (true docked, false handheld)
bool g_set[kModules] = {false, false, false};  // which modules WE have changed
std::atomic<int> g_enabled{0};  // off by default: this raises the clocks, which must be requested explicitly

Result RequestProfiles(uint64_t tid, ProfilesList* output) {
  return serviceDispatchIn(&g_service, kOrderProfiles, tid,
                           .buffer_attrs = {SfBufferAttr_HipcMapAlias | SfBufferAttr_Out},
                           .buffers = {{output, sizeof(*output)}});
}

/*
 * sys-clk command 0: GetApiVersion. It checks that on the other side there really is a
 * sysmodule with this interface and not something else with a similar name.
 */
Result RequestVersionApi(uint32_t* output) {
  return serviceDispatchOut(&g_service, 0, *output);
}

/*
 * sys-clk command 1: GetVersionString. Read-only. It identifies the exact fork and version, which
 * is the only way to know which commands its API really has without sending it anything blindly.
 */
Result RequestVersionText(char* output, size_t bytes) {
  return serviceDispatch(&g_service, 1,
                         .buffer_attrs = {SfBufferAttr_HipcMapAlias | SfBufferAttr_Out},
                         .buffers = {{output, bytes}});
}

/*
 * sys-clk command 2: GetCurrentContext. Read-only.
 *
 * It returns a structure as raw data, and its size changes between forks. The size for hoc:clk is
 * unknown, so several are tried: if the size does not match, the dispatch fails and that is all
 * (nothing is written anywhere). For the one that works, the bytes are dumped in hexadecimal.
 * Comparing that dump in Fake Handheld and in Fake Docked shows which field is the profile, without
 * having to guess the structure.
 */
/*
 * The structure size comes from the type, hence one template per size. If it does not match what
 * the sysmodule returns, the dispatch fails and nothing else happens.
 */
template <unsigned N>
Result RequestContextFor(unsigned char* output) {
  struct Block {
    unsigned char b[N];
  };
  Block tmp{};
  const Result rc = serviceDispatchOut(&g_service, 2, tmp);
  if (R_SUCCEEDED(rc)) {
    std::memcpy(output, tmp.b, N);
  }
  return rc;
}

void Dump(const unsigned char* buf, unsigned size) {
  char line[340];
  int n = std::snprintf(line, sizeof(line), "[clocks] context (%u bytes):", size);
  for (unsigned i = 0; i < size && n > 0 && unsigned(n) + 4 < sizeof(line); ++i) {
    n += std::snprintf(line + n, sizeof(line) - unsigned(n), " %02X", buf[i]);
  }
  Warning(line);
}

void DumpContext() {
  unsigned char buf[80];
  std::memset(buf, 0, sizeof(buf));
  if (R_SUCCEEDED(RequestContextFor<28>(buf))) { Dump(buf, 28); return; }
  if (R_SUCCEEDED(RequestContextFor<32>(buf))) { Dump(buf, 32); return; }
  if (R_SUCCEEDED(RequestContextFor<36>(buf))) { Dump(buf, 36); return; }
  if (R_SUCCEEDED(RequestContextFor<40>(buf))) { Dump(buf, 40); return; }
  if (R_SUCCEEDED(RequestContextFor<44>(buf))) { Dump(buf, 44); return; }
  if (R_SUCCEEDED(RequestContextFor<48>(buf))) { Dump(buf, 48); return; }
  if (R_SUCCEEDED(RequestContextFor<52>(buf))) { Dump(buf, 52); return; }
  if (R_SUCCEEDED(RequestContextFor<56>(buf))) { Dump(buf, 56); return; }
  if (R_SUCCEEDED(RequestContextFor<64>(buf))) { Dump(buf, 64); return; }
  if (R_SUCCEEDED(RequestContextFor<72>(buf))) { Dump(buf, 72); return; }
  Warning("[clocks] GetCurrentContext is not accepted with any of the sizes tried");
}

Result PutOverride(uint32_t shader_module, uint32_t hz) {
  const ArgsOverride args{shader_module, hz};
  return serviceDispatchIn(&g_service, kOrderOverride, args);
}

/*
 * Is that service registered? sm command 65100 (AtmosphereHasService). It answers yes or no; it never
 * waits. Without this, requesting a service that does not exist hangs the thread forever. On 12.0.0+
 * sm speaks TIPC, and CMIF on earlier versions.
 */
bool HasService(const char* name) {
  const SmServiceName encoded = smEncodeName(name);
  bool has = false;
  if (hosversionAtLeast(12, 0, 0)) {
    TipcService* sm = smGetServiceSessionTipc();
    if (!sm || R_FAILED(tipcDispatchInOut(sm, 65100, encoded, has))) {
      return false;
    }
  } else {
    Service* sm = smGetServiceSession();
    if (!sm || R_FAILED(serviceDispatchInOut(sm, 65100, encoded, has))) {
      return false;
    }
  }
  return has;
}

bool Open() {
  if (g_open) {
    return true;
  }
  if (g_attempts >= 5) {
    return false;  // not installed; do not keep insisting forever
  }
  ++g_attempts;
  /*
   * Not only "sys:clk".
   *
   * Horizon OC 2.4.2 does not register that name, and giving up there with a warning meant the clocks
   * never followed Reverse-NX on such a console. sys-clk-plus does use "sys:clk" with the same
   * commands (5 GetProfiles, 8 SetOverride), so the interface is the usual one; what changes between
   * forks is the name. The known ones are tried and the log says which exist, which is the only way to
   * find out the name on a console that is not at hand.
   *
   * HasService answers yes or no and never waits, so asking about several is free.
   */
  static const char* const kNames[] = {"sys:clk",  "hoc:clk", "hocclk",
                                         "sysclk",   "clk:sys", "sys:oc"};
  const char* chosen = nullptr;
  char found[160];
  found[0] = '\0';
  for (const char* name : kNames) {
    if (!HasService(name)) {
      continue;
    }
    if (found[0]) {
      std::strncat(found, ", ", sizeof(found) - std::strlen(found) - 1);
    }
    std::strncat(found, name, sizeof(found) - std::strlen(found) - 1);
    if (!chosen) {
      chosen = name;
    }
  }
  if (!chosen) {
    if (g_attempts == 1) {
      Warning("[clocks] no known clock sysmodule found (tried: sys:clk, hoc:clk, "
            "hocclk, sysclk, clk:sys, sys:oc): there is nobody to ask");
    }
    g_attempts = 5;  // do not ask again
    return false;
  }
  const Result rc = smGetService(&g_service, chosen);
  if (R_FAILED(rc)) {
    if (g_attempts == 1) {
      std::fprintf(stderr, "[clocks] '%s' exists but could not be opened. Error 0x%X\n", chosen,
                   (unsigned)rc);
    }
    return false;
  }
  // Make sure it really speaks our interface before sending it clock commands.
  uint32_t version = 0;
  if (R_FAILED(RequestVersionApi(&version))) {
    std::fprintf(stderr,
                 "[clocks] '%s' opened but does not answer GetApiVersion: it is not the sys-clk "
                 "interface, nothing is touched (services found: %s)\n",
                 chosen, found);
    serviceClose(&g_service);
    g_attempts = 5;
    return false;
  }
  char version_text[64];
  std::memset(version_text, 0, sizeof(version_text));
  if (R_SUCCEEDED(RequestVersionText(version_text, sizeof(version_text)))) {
    version_text[sizeof(version_text) - 1] = 0;
    char line[160];
    std::snprintf(line, sizeof(line), "[clocks] sysmodule version: %s", version_text);
    Warning(line);
  } else {
    Warning("[clocks] the sysmodule does not answer GetVersionString (command 1)");
  }
  DumpContext();
  std::fprintf(stderr, "[clocks] sysmodule '%s', API %u (services found: %s)\n", chosen,
               (unsigned)version, found);
  g_open = true;
  Warning("[clocks] ready: the clocks can follow Reverse-NX");
  return true;
}

/*
 * The MHz of the docked column: first this game's and, if there are none, the global profile's. It is
 * the same order the sysmodule follows.
 */
void ReadProfile(unsigned profile, uint32_t output[kModules]) {
  std::memset(output, 0, sizeof(uint32_t) * kModules);

  uint64_t tid = 0;
  svcGetInfo(&tid, InfoType_ProgramId, CUR_PROCESS_HANDLE, 0);

  ProfilesList list{};
  if (tid && R_SUCCEEDED(RequestProfiles(tid, &list))) {
    for (unsigned m = 0; m < kModules; ++m) {
      output[m] = list.mhz[profile][m];
    }
  }
  bool missing = false;
  for (unsigned m = 0; m < kModules; ++m) {
    if (!output[m]) {
      missing = true;
    }
  }
  if (!missing) {
    return;
  }
  ProfilesList global{};
  if (R_SUCCEEDED(RequestProfiles(kGlobalProfile, &global))) {
    for (unsigned m = 0; m < kModules; ++m) {
      if (!output[m]) {
        output[m] = global.mhz[profile][m];
      }
    }
  }
}

void Apply(bool a_docked) {
  // The state is marked whatever happens: if nothing is configured, there is no need to retry every
  // second or to repeat the warning.
  g_applied = true;
  uint32_t mhz[kModules] = {0, 0, 0};
  ReadProfile(a_docked ? kDockedProfile : kHandheldProfile, mhz);
  uint32_t hz[kModules] = {0, 0, 0};
  for (unsigned m = 0; m < kModules; ++m) {
    hz[m] = mhz[m] * 1000000u;
  }
  if (!a_docked) {
    // Going down to handheld has to do something: if that column is empty the stock clocks are used,
    // which is what "Fake Handheld" is expected to mean.
    for (unsigned m = 0; m < kModules; ++m) {
      if (!hz[m]) {
        hz[m] = kHandheldOfSeriesHz[m];
      }
    }
  }
  if (!hz[0] && !hz[1] && !hz[2]) {
    Warning("[clocks] nothing is set in that column: nothing is touched");
    return;
  }
  for (unsigned m = 0; m < kModules; ++m) {
    if (hz[m] && R_SUCCEEDED(PutOverride(m, hz[m]))) {
      g_set[m] = true;
    }
  }
  std::fprintf(stderr,
               "[clocks] %s by Reverse-NX: CPU %u kHz, GPU %u kHz, memory %u kHz (0 = as it was)\n",
               a_docked ? "docked" : "handheld", hz[0] / 1000u, hz[1] / 1000u, hz[2] / 1000u);
}

void Drop() {
  g_applied = false;
  bool any = false;
  for (unsigned m = 0; m < kModules; ++m) {
    // Only what we set is released: a clock forced by hand from the overlay is not cleared.
    if (g_set[m]) {
      PutOverride(m, 0);
      g_set[m] = false;
      any = true;
    }
  }
  if (any) {
    Warning("[clocks] released: the sysmodule sets them back");
  }
}

/* In case the game exits cleanly: never leave the docked clocks set in handheld mode. */
__attribute__((destructor)) void OnExit() {
  if (g_open) {
    Drop();
  }
}

}  // namespace

void FollowMode(bool effective_docked, bool docked_real) {
  // Both directions. Faking docked mode with the console in hand is not enough: the opposite case
  // (console in the dock and Reverse-NX on "Fake Handheld") left everything untouched and the
  // sysmodule kept the docked column, that is, GPU 768 MHz and memory 1600. It acts whenever the
  // effective mode and the real one differ.
  const bool want =
      g_enabled.load(std::memory_order_relaxed) != 0 && effective_docked != docked_real;
  if (want == g_applied && (!want || effective_docked == g_applied_docked)) {
    return;
  }
  if (!Open()) {
    return;
  }
  if (want) {
    if (g_applied && effective_docked != g_applied_docked) {
      Drop();  // direction change: remove the old one, then apply the new one
    }
    g_applied_docked = effective_docked;
    Apply(effective_docked);
  } else {
    Drop();
  }
}

void Enable(bool enabled) {
  g_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed);
  if (!enabled && g_applied) {
    Drop();
  }
}

}  // namespace rex::ui::switch_sysclk

extern "C" void RexSwitchClocksReverseEnable(int enabled) {
  rex::ui::switch_sysclk::Enable(enabled != 0);
}

#endif  // REX_PLATFORM_SWITCH
