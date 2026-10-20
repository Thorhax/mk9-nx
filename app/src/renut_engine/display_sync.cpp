// banjo-nx (2026-10-19, latency batch 3): see display_sync.h. After futari-nx's futari_vblank_display_sync and
// futari_vblank_phase_auto (the user's own port, docs/futari-notes.md "Input lag"), for a game paced by the frame
// limiter rather than by its vertical blank, whose frames take more than one refresh from release to image.
#include "renut_engine/display_sync.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/frame_latency.h>
#include <rex/logging.h>
#include <rex/platform.h>

#include "me/me_platform_compat.h"  // RexSwitchPinCurrentThreadToCore and the priority helper
#include "renut_engine/gpu/backend/video.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"  // StartLongLivedThread

#if REX_PLATFORM_SWITCH
#include <switch.h>  // viGetDisplayVsyncEvent
#endif

// sa2-nx (2026-10-20): on by default in SA2 (the user's call: Banjo's latency set); SA2's frame pacing
// (gpu/sa2_renut_pacing.cpp) takes its releases from here.
REXCVAR_DEFINE_BOOL(banjo_display_sync, true, "Banjo/Enhancements",
                    "OFF by default (2026-10-19): in Ryujinx every run with the thread active stalled or hung (runs 65, "
                    "67, 68), the runs without it did not; to be tried on the console with configs/"
                    "banjo_display_sync_on.toml. Latency batch 3 (2026-10-19, the Switch): a thread follows the display's refresh (its vsync "
                    "event); the frame limiter (frame_cap) releases each frame at a refresh plus a phase instead of on "
                    "its own timer, whose phase to the display slid (an image finished just after a refresh waited "
                    "almost a whole refresh more; now and then one was shown three refreshes and the next one). A frame "
                    "that ends late is released at once, as before. The [display] line every 10 s gives present -> "
                    "display and input -> display (renut_engine/display_sync.cpp)");
REXCVAR_DEFINE_BOOL(banjo_display_phase_auto, true, "Banjo/Enhancements",
                    "banjo_display_sync: the phase found by measurement: as late after the refresh as the frames' "
                    "images still reach the display at the refresh they reached before (0.25 ms later every 90 frames "
                    "in time, 2 ms earlier at one that is not, and no later than that for a minute)");
REXCVAR_DEFINE_INT32(banjo_display_phase_max_us, 12000, "Banjo/Enhancements",
                     "banjo_display_phase_auto: the latest it puts the release after the refresh (microseconds)")
    .range(0, 16000);
REXCVAR_DEFINE_INT32(banjo_display_phase_us, 0, "Banjo/Enhancements",
                     "banjo_display_sync with banjo_display_phase_auto off: the release this long after the refresh "
                     "(microseconds)")
    .range(0, 16000);

#if REX_PLATFORM_SWITCH
// banjo-nx (2026-10-20): libnx's default window (default_window.o: __nx_win_init, __nx_win_exit, nwindowGetDefault)
// done here, the same calls in the same order (from its disassembly), so its display gives the refresh thread below
// its vsync event. A second viOpenDefaultDisplay in the process failed on the console (package 36i's logs: "cannot
// open the display"), the display being libnx's already. Defining all three keeps default_window.o out of the link;
// libnx's init.o calls the two through its weak references as before.
namespace {
ViDisplay g_win_display;
ViLayer g_win_layer;
NWindow g_win;
std::atomic<bool> g_win_ready{false};
}  // namespace

extern "C" {
NWindow *nwindowGetDefault(void) { return &g_win; }

void __nx_win_init(void) {
  Result rc = viInitialize(ViServiceType_Default);
  if (R_SUCCEEDED(rc)) {
    rc = viOpenDisplay("Default", &g_win_display);
    if (R_SUCCEEDED(rc)) {
      rc = viCreateLayer(&g_win_display, &g_win_layer);
      if (R_SUCCEEDED(rc)) {
        rc = viSetLayerScalingMode(&g_win_layer, ViScalingMode_FitToLayer);
        if (R_SUCCEEDED(rc))
          rc = nwindowCreateFromLayer(&g_win, &g_win_layer);
        if (R_SUCCEEDED(rc)) {
          nwindowSetDimensions(&g_win, 1280, 720);
          g_win_ready.store(true, std::memory_order_release);
          return;
        }
        viCloseLayer(&g_win_layer);
      }
      viCloseDisplay(&g_win_display);
    }
    viExit();
  }
  diagAbortWithResult(0x2559);  // MAKERESULT(Module_Libnx, LibnxError_BadGfxInit), as libnx
}

void __nx_win_exit(void) {
  g_win_ready.store(false, std::memory_order_release);
  nwindowClose(&g_win);
  viCloseLayer(&g_win_layer);
  viCloseDisplay(&g_win_display);
  viExit();
}
}  // extern "C"
#endif

namespace banjo::display_sync {
namespace {

uint64_t NowNs() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

constexpr int kVsyncThreadPriority = 0x2C;  // the render thread's: it wakes at each refresh and does little
constexpr int kVsyncThreadCore = 2;         // the background core (the game and render threads run on 0 and 1)
constexpr uint64_t kGridStaleNs = 100'000'000ull;  // no refresh for 100 ms: the limiter's own timer
constexpr uint32_t kGoodFrames = 90;               // 3 s at 30 fps in time: the phase 0.25 ms later
constexpr uint64_t kPhaseStepNs = 250'000ull;
constexpr uint64_t kPhaseBackNs = 2'000'000ull;
constexpr uint64_t kHoldNs = 60'000'000'000ull;    // after a miss, no later than the phase it left, for a minute
constexpr uint32_t kFloorFrames = 32;              // the refresh count a frame takes: the least of the last 32

std::atomic<bool> g_running{false};
std::atomic<uint64_t> g_last_vsync{0};  // the latest refresh (steady_clock ns)
std::atomic<uint64_t> g_period{0};      // the refresh period (ns), averaged
std::atomic<uint64_t> g_phase{0};       // the release after the refresh (ns)

// The limiter's releases, by the Swap of the frame each released.
struct Release {
  uint64_t swap = 0;
  uint64_t at = 0;
  uint64_t anchor = 0;  // the refresh the release is a phase after (0: not anchored, a late frame)
};
constexpr uint32_t kReleases = 64;
Release g_releases[kReleases];

// Presented frames not yet seen on the display (the display thread's, but NotePresented adds).
struct Shown {
  uint64_t swap = 0;
  uint64_t present_index = 0;
  uint64_t present_ns = 0;
  uint64_t input_ns = 0;
  uint64_t anchor = 0;
};
constexpr uint32_t kShown = 8;
std::mutex g_mutex;  // g_shown*, g_releases, the sums: a few times a frame
Shown g_shown[kShown];
uint32_t g_shown_head = 0, g_shown_count = 0;

struct Sum {
  uint64_t n = 0;
  double total = 0, max = 0;
  void Add(double ms) {
    ++n;
    total += ms;
    max = std::max(max, ms);
  }
  double Mean() const { return n ? total / double(n) : 0.0; }
};
Sum g_present_display, g_input_display, g_release_display;
uint64_t g_refreshes = 0, g_superseded = 0, g_misses = 0, g_misses_total = 0, g_phase_ups = 0;
uint64_t g_refreshes_with_image = 0;
// Diagnostics (2026-10-19, emulator runs 65 and 67 stalled with the display's refresh followed): the slowest fence test
// and the longest release wait asked for since the last report.
std::atomic<uint64_t> g_fence_test_max_ns{0}, g_release_wait_max_ns{0}, g_release_clamped{0};

// The phase (the display thread's).
uint32_t g_floor[kFloorFrames];
uint32_t g_floor_next = 0, g_floor_count = 0;
uint32_t g_good = 0;
uint64_t g_ceiling = 0, g_hold_until = 0;

double Ms(uint64_t from, uint64_t to) { return to > from ? double(to - from) / 1e6 : 0.0; }

// A displayed frame's refresh count from its anchor: the phase's measurement. Requires g_mutex.
void PhaseFrame(uint32_t refreshes, uint64_t now) {
  if (!REXCVAR_GET(banjo_display_phase_auto))
    return;
  const uint64_t max = uint64_t(REXCVAR_GET(banjo_display_phase_max_us)) * 1000;
  if (g_hold_until && now > g_hold_until) {
    g_hold_until = 0;
    g_ceiling = max;
  }
  if (!g_hold_until)
    g_ceiling = max;
  uint32_t floor = refreshes;
  for (uint32_t i = 0; i < g_floor_count; ++i)
    floor = std::min(floor, g_floor[i]);
  g_floor[g_floor_next] = refreshes;
  g_floor_next = (g_floor_next + 1) % kFloorFrames;
  g_floor_count = std::min(g_floor_count + 1, kFloorFrames);
  const uint64_t phase = g_phase.load(std::memory_order_relaxed);
  if (refreshes > floor && g_floor_count > 1) {
    // Its image reached the display a refresh later than the frames before: back off, and stay there a while. The
    // frames from here make the floor again (a scene that needs a refresh more settles at it).
    ++g_misses;
    ++g_misses_total;
    const uint64_t back = phase > kPhaseBackNs ? phase - kPhaseBackNs : 0;
    g_phase.store(back, std::memory_order_relaxed);
    g_ceiling = back;
    g_hold_until = now + kHoldNs;
    g_good = 0;
    g_floor_count = 0;
    g_floor_next = 0;
    return;
  }
  if (++g_good >= kGoodFrames) {
    g_good = 0;
    const uint64_t up = std::min(phase + kPhaseStepNs, g_ceiling);
    if (up > phase) {
      g_phase.store(up, std::memory_order_relaxed);
      ++g_phase_ups;
    }
  }
}

// At a refresh: the newest presented frame whose GPU work is done is the one the display shows from it; older ones
// not shown yet never are. The fences are tested without g_mutex (NotePresented takes it on the render thread at every
// present): the frames are copied out first and found again by their present index.
void AtRefresh(uint64_t now, uint64_t period) {
  uint64_t indices[kShown];
  uint32_t count = 0;
  {
    std::lock_guard lock(g_mutex);
    ++g_refreshes;
    for (uint32_t k = 0; k < g_shown_count; ++k)
      indices[count++] = g_shown[(g_shown_head + k) % kShown].present_index;
  }
  uint64_t done_index = 0;
  for (uint32_t k = 0; k < count; ++k) {
    // -1 (unknown): too old to matter, or no fences (not the deko3d back end): taken as done.
    const uint64_t t0 = NowNs();
    const int done = renut::gpu::Video::PresentDone(indices[k]);
    const uint64_t took = NowNs() - t0;
    if (took > g_fence_test_max_ns.load(std::memory_order_relaxed))
      g_fence_test_max_ns.store(took, std::memory_order_relaxed);
    if (done == 0)
      break;  // in order on one queue: the later ones are not done either
    done_index = indices[k];
  }
  if (!done_index)
    return;
  std::lock_guard lock(g_mutex);
  int newest = -1;
  for (uint32_t k = 0; k < g_shown_count; ++k) {
    if (g_shown[(g_shown_head + k) % kShown].present_index == done_index) {
      newest = int(k);
      break;
    }
  }
  if (newest < 0)
    return;  // dropped meanwhile (eight presents without a refresh)
  ++g_refreshes_with_image;
  g_superseded += uint64_t(newest);
  const Shown s = g_shown[(g_shown_head + uint32_t(newest)) % kShown];
  g_shown_head = (g_shown_head + uint32_t(newest) + 1) % kShown;
  g_shown_count -= uint32_t(newest) + 1;
  g_present_display.Add(Ms(s.present_ns, now));
  if (s.input_ns)
    g_input_display.Add(Ms(s.input_ns, now));
  if (s.anchor && period) {
    g_release_display.Add(Ms(s.anchor, now));
    const double refreshes = double(now - s.anchor) / double(period);
    PhaseFrame(uint32_t(std::lround(refreshes)), now);
  }
}

void Run() {
#if REX_PLATFORM_SWITCH
  const bool priority_ok = RexSwitchSetCurrentThreadPriorityOk(kVsyncThreadPriority);
  const bool core_ok = RexSwitchPinCurrentThreadToCore(kVsyncThreadCore);
  // 2026-10-20: the default window's display (opened by __nx_win_init above); a display of its own only without it.
  ViDisplay own{};
  ViDisplay *display = g_win_ready.load(std::memory_order_acquire) ? &g_win_display : nullptr;
  Event vsync{};
  if (!display) {
    const Result rc = viOpenDefaultDisplay(&own);
    if (R_FAILED(rc)) {
      REXLOG_WARN("[display] cannot open the display (0x{:X}): the frame limiter keeps its own timer "
                  "(banjo_display_sync)", rc);
      return;
    }
    display = &own;
  }
  if (const Result rc = viGetDisplayVsyncEvent(display, &vsync); R_FAILED(rc)) {
    REXLOG_WARN("[display] no vsync event from the display (0x{:X}): the frame limiter keeps its own timer "
                "(banjo_display_sync)", rc);
    if (display == &own)
      viCloseDisplay(&own);
    return;
  }
  g_phase.store(REXCVAR_GET(banjo_display_phase_auto) ? 0 : uint64_t(REXCVAR_GET(banjo_display_phase_us)) * 1000,
                std::memory_order_relaxed);
  REXLOG_INFO("[display] the frame limiter follows the display's refresh (banjo_display_sync), the phase {} "
              "(thread at priority 0x{:X}{}, core {}{})",
              REXCVAR_GET(banjo_display_phase_auto)
                  ? fmt::format("found by measurement, at most {} us (banjo_display_phase_auto)",
                                REXCVAR_GET(banjo_display_phase_max_us))
                  : fmt::format("{} us (banjo_display_phase_us)", REXCVAR_GET(banjo_display_phase_us)),
              kVsyncThreadPriority, priority_ok ? "" : " NOT accepted", kVsyncThreadCore, core_ok ? "" : " NOT accepted");
  g_running.store(true, std::memory_order_release);
  uint64_t last = 0;
  double period = 1e9 / 60.0;
  for (;;) {
    // 100 ms: should the display stop refreshing (sleep mode), the grid goes stale and the limiter uses its timer.
    if (R_FAILED(eventWait(&vsync, 100'000'000ull)))
      continue;
    eventClear(&vsync);
    const uint64_t now = NowNs();
    if (last) {
      const uint64_t gap = now - last;
      if (gap < 8'000'000ull)
        continue;  // a guard: an event that did not clear
      // One refresh apart (not two after a missed wake): the period, averaged over about a second.
      const double refreshes = std::max(1.0, std::round(double(gap) / period));
      if (refreshes <= 4.0)
        period += (double(gap) / refreshes - period) / 64.0;
    }
    last = now;
    g_period.store(uint64_t(period), std::memory_order_relaxed);
    g_last_vsync.store(now, std::memory_order_release);
    AtRefresh(now, uint64_t(period));
  }
#endif
}

}  // namespace

void Start() {
#if REX_PLATFORM_SWITCH
  static std::once_flag once;
  std::call_once(once, [] {
    if (REXCVAR_GET(banjo_display_sync))
      banjo::renut_gpu::StartLongLivedThread(&Run, 128 * 1024);
  });
#endif
}

uint64_t NextRelease(uint64_t now, uint64_t previous, uint64_t frame_ns, bool *anchored) {
  *anchored = false;
  if (!g_running.load(std::memory_order_acquire))
    return 0;
  const uint64_t vsync = g_last_vsync.load(std::memory_order_acquire);
  const uint64_t period = g_period.load(std::memory_order_relaxed);
  if (!vsync || !period || now > vsync + kGridStaleNs)
    return 0;
  const uint64_t refreshes = std::max<uint64_t>(1, uint64_t(std::llround(double(frame_ns) / double(period))));
  const uint64_t frame = refreshes * period;
  // A frame that ended after its time: at once, as the timer did (no waiting for a later refresh: no 20 fps).
  if (previous && now >= previous + frame)
    return now;
  // The first refresh plus the phase at or after both now and half a refresh before the frame's time (the previous
  // release was a refresh plus the phase too, or a late one: the grid is found again here).
  const uint64_t phase = g_phase.load(std::memory_order_relaxed);
  const uint64_t earliest = std::max(now, previous ? previous + frame - period / 2 : now);
  const int64_t from = int64_t(earliest) - int64_t(vsync + phase);
  const int64_t n = from <= 0 ? -((-from) / int64_t(period)) : (from + int64_t(period) - 1) / int64_t(period);
  const int64_t at = int64_t(vsync + phase) + n * int64_t(period);
  const uint64_t release = uint64_t(std::max<int64_t>(at, int64_t(now)));
  const uint64_t wait = release - now;
  if (wait > g_release_wait_max_ns.load(std::memory_order_relaxed))
    g_release_wait_max_ns.store(wait, std::memory_order_relaxed);
  if (wait > frame + period) {
    // Never more than a frame and a refresh: a grid that came out wrong must not hold the game (a safety clamp).
    g_release_clamped.fetch_add(1, std::memory_order_relaxed);
    return now;
  }
  *anchored = true;
  return release;
}

void NoteRelease(uint64_t at, bool anchored) {
  if (!g_running.load(std::memory_order_acquire))
    return;
  const uint64_t swap = rex::latency::SwapCount() + 1;
  std::lock_guard lock(g_mutex);
  Release &r = g_releases[swap % kReleases];
  r.swap = swap;
  r.at = at;
  r.anchor = anchored ? at - std::min(at, g_phase.load(std::memory_order_relaxed)) : 0;
}

void NotePresented(uint64_t swap) {
  if (!g_running.load(std::memory_order_acquire))
    return;
  const uint64_t now = NowNs();
  const uint64_t index = renut::gpu::Video::PresentCount();
  const uint64_t input = rex::latency::InputTimeOf(swap);
  std::lock_guard lock(g_mutex);
  const Release &r = g_releases[swap % kReleases];
  Shown s;
  s.swap = swap;
  s.present_index = index;
  s.present_ns = now;
  s.input_ns = input;
  s.anchor = r.swap == swap ? r.anchor : 0;
  if (g_shown_count == kShown) {  // the display has not taken one for 8 presents: drop the oldest
    g_shown_head = (g_shown_head + 1) % kShown;
    --g_shown_count;
    ++g_superseded;
  }
  g_shown[(g_shown_head + g_shown_count) % kShown] = s;
  ++g_shown_count;
}

std::string Report() {
  if (!g_running.load(std::memory_order_acquire))
    return {};
  std::lock_guard lock(g_mutex);
  const uint64_t period = g_period.load(std::memory_order_relaxed);
  std::string line = fmt::format(
      "refresh {:.3f} ms, {} refreshes ({} with a new image); present -> display {:.1f} ms (max {:.1f}), input -> "
      "display {:.1f} ms (max {:.1f}), release -> display {:.1f} ms; {} images never shown (a newer one was ready "
      "first); phase {:.2f} ms after the refresh{}, {} misses ({} in all), {} steps later; slowest fence test {:.2f} "
      "ms, longest release wait {:.1f} ms, {} clamped",
      double(period) / 1e6, g_refreshes, g_refreshes_with_image, g_present_display.Mean(), g_present_display.max,
      g_input_display.Mean(), g_input_display.max, g_release_display.Mean(), g_superseded,
      double(g_phase.load(std::memory_order_relaxed)) / 1e6,
      g_hold_until ? fmt::format(" (held at most {:.2f} ms)", double(g_ceiling) / 1e6) : std::string(), g_misses,
      g_misses_total, g_phase_ups, double(g_fence_test_max_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
      double(g_release_wait_max_ns.exchange(0, std::memory_order_relaxed)) / 1e6,
      g_release_clamped.exchange(0, std::memory_order_relaxed));
  g_present_display = g_input_display = g_release_display = Sum{};
  g_refreshes = g_refreshes_with_image = g_superseded = g_misses = g_phase_ups = 0;
  return line;
}

}  // namespace banjo::display_sync
