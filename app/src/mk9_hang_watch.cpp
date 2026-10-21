// Hang reports: when the game stops presenting (no Swap, g_mk9_game_frames) for mk9_hang_dump_s seconds, the stacks of
// every registered thread (guest threads, Renut's render and background threads, the deko3d compile worker) are
// appended to logs/rex/rex_crash.log (RexSwitchDumpThreads, sdk/src/core/exception_handler_switch.cpp). Added
// 2026-10-21 for the main-menu freeze in Ryujinx: every guest thread stopped after a START press with no fault and no
// log line saying where. Symbolize the image+0x offsets with the ELF: aarch64-none-elf-addr2line -f -C -e mk9.elf.
// Switch only; elsewhere a debugger does this better.

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <thread>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/logging/api.h>
#include <rex/platform.h>

#include "mk9_hang_watch.h"

REXCVAR_DEFINE_INT32(mk9_hang_dump_s, 20, "MK9",
                     "Switch: when no frame is presented for this many seconds, write every thread's stack to "
                     "logs/rex/rex_crash.log (again at three times as long in the same stall; at most 4 dumps). "
                     "0 = off")
    .range(0, 3600);

extern std::atomic<uint64_t> g_mk9_game_frames;  // src/mk9_frame_counter.cpp

#if REX_PLATFORM_SWITCH
extern "C" void RexSwitchDumpThreads(const char* reason);
#endif

namespace mk9::hang_watch {

void Start() {
#if REX_PLATFORM_SWITCH
  const int32_t limit_s = REXCVAR_GET(mk9_hang_dump_s);
  if (limit_s <= 0) return;
  // Never joined: it lives as long as the process (a joinable std::thread destroyed at exit would terminate).
  new std::thread([limit_s] {
    using clock = std::chrono::steady_clock;
    uint64_t last_frames = g_mk9_game_frames.load(std::memory_order_relaxed);
    auto last_change = clock::now();
    int dumps = 0, dumps_this_stall = 0;
    while (dumps < 4) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
      const uint64_t frames = g_mk9_game_frames.load(std::memory_order_relaxed);
      const auto now = clock::now();
      if (frames != last_frames || frames == 0) {  // nothing before the first frame: startup is not a hang
        last_frames = frames;
        last_change = now;
        dumps_this_stall = 0;
        continue;
      }
      const auto stalled_s = std::chrono::duration_cast<std::chrono::seconds>(now - last_change).count();
      const int64_t due_s = dumps_this_stall == 0 ? limit_s : int64_t(limit_s) * 3;
      if (dumps_this_stall >= 2 || stalled_s < due_s) continue;
      char reason[96];
      std::snprintf(reason, sizeof(reason), "no frame presented for %lld s (frame %llu)",
                    static_cast<long long>(stalled_s), static_cast<unsigned long long>(frames));
      RexSwitchDumpThreads(reason);
      ++dumps;
      ++dumps_this_stall;
      REXLOG_WARN("[hang] {}: every registered thread's stack is in logs/rex/rex_crash.log (mk9_hang_dump_s)",
                  reason);
      rex::FlushLogging();
    }
  });
#endif
}

}  // namespace mk9::hang_watch
