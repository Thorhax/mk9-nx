// sa2-nx (2026-10-20): the game's frame pacing with Renut's renderer (sa2_renderer = "renut").
//
// On the console SA2's frames are paced by the vertical blank: after each Swap the GPU waits for the flip (a flag the
// game's vblank interrupt callback clears), and the game thread blocks on its fences once it is a frame or two ahead.
// SA2's game speed follows its frame rate, so that wait is what keeps the game at its speed. Renut's renderer has no
// GPU ring and no vblank interrupt: without this the game ran as fast as the host could go (300-1,900 frames a second
// on the PC test with the window minimized).
//
// Here each Swap is released at the presentation interval the game asked for (D3DPRESENT_PARAMETERS::
// PresentationInterval, which SA2's Direct3D keeps in the device: InitializePresentationParameters copies the parameters
// to +13948, so the interval is at +14000; SynchronizeToPresentationInterval reads it there): ONE (or DEFAULT) one
// refresh of 60 Hz, TWO two, THREE three, IMMEDIATE none. A frame that is late:
//   - sa2_vsync_relaxed on (the player's default): goes at once, and the next one is due an interval after it (the
//     native renderer's relaxed vsync: 50-59 fps instead of a fall to 30);
//   - off: waits for the next refresh of the 60 Hz grid, as fixed vsync does.
#include "renut_engine/gpu/sa2_renut_pacing.h"

#include <algorithm>
#include <chrono>
#include <string>
#include <thread>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef CREATE_WAITABLE_TIMER_HIGH_RESOLUTION
#define CREATE_WAITABLE_TIMER_HIGH_RESOLUTION 0x00000002
#endif
#endif

#include <rex/cvar.h>
#include <rex/logging.h>

#include "cpu/sa2_enhancements.h"
#include "renut_engine/gpu/backend/queued.h"
#include "renut_engine/display_sync.h"  // sa2-nx (2026-10-20): banjo_display_sync
#include "renut_engine/gpu/guest/guest_memory.h"

REXCVAR_DEFINE_BOOL(sa2_renut_vblank_pacing, true, "SA2",
                    "Renut's renderer (sa2_renderer = renut): each Swap is released at the game's presentation interval "
                    "(60 Hz refreshes), as the console's vertical blank paces the game; SA2's game speed follows its "
                    "frame rate. Off: the game runs as fast as the host renders (tests only)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// The native renderer's relaxed vsync key (src/me/me_native_system.cpp), the player's setting.
extern ::sa2::enh::Enhancement g_vsync_relaxed;

namespace sa2::renut {

namespace {

using Clock = std::chrono::steady_clock;
constexpr uint32_t kDeviceOffsetPresentationInterval = 13872;  // mk9-nx: sub_82BF6928 reads it [SA2 14000];  // the device's copy of D3DPRESENT_PARAMETERS + 52
constexpr Clock::duration kRefresh = std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / 60.0));

// Refreshes per frame for D3DPRESENT_INTERVAL_*: as SA2's SynchronizeToPresentationInterval (sub_825589D8) maps them.
uint32_t RefreshesPerFrame(uint32_t interval) {
  switch (interval) {
    case 0:  // DEFAULT
    case 1:  // ONE
      return 1;
    case 2:  // TWO
      return 2;
    case 4:  // THREE
      return 3;
    default:  // IMMEDIATE (0x80000000) and anything else
      return 0;
  }
}

Clock::time_point g_grid;  // the first Swap
Clock::time_point g_due;   // when the next Swap may go (on the 60 Hz grid while the game keeps up)
bool g_started = false;

// The 10 s report (2026-10-20, for the console test): the game thread's time from one Swap's release to the next
// Swap (its frame: the game's work, its Direct3D calls, the renderer's hooks, and any wait on the render thread), and
// the pacing's sleeps. Game thread only.
struct Stats {
  Clock::time_point since{};
  Clock::time_point released{};  // the last Swap's release
  uint64_t frames = 0, late = 0;
  double work_ms = 0.0, work_max_ms = 0.0, sleep_ms = 0.0;
};
Stats g_stats;

double Ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }

// At a Swap's release ('now' the Swap's arrival).
void NoteRelease(Clock::time_point arrived, Clock::time_point released, bool late) {
  Stats &s = g_stats;
  if (s.released != Clock::time_point{}) {
    const double work = Ms(arrived - s.released);
    s.work_ms += work;
    s.work_max_ms = std::max(s.work_max_ms, work);
    s.sleep_ms += Ms(released - arrived);
    ++s.frames;
    s.late += late ? 1 : 0;
  }
  s.released = released;
  if (s.since == Clock::time_point{}) s.since = released;
  if (released - s.since < std::chrono::seconds(10)) return;
  if (s.frames)
    REXLOG_INFO("[renut] game thread, {:.1f} s: {} frames, {:.2f} ms a frame from Swap to Swap (most {:.2f} ms), "
                "{:.2f} ms a frame paced; {} frames late (sa2_renut_vblank_pacing)",
                Ms(released - s.since) / 1000.0, s.frames, s.work_ms / double(s.frames), s.work_max_ms,
                s.sleep_ms / double(s.frames), s.late);
  // The game thread's other measure: what its draws keep of the device (backend/queued.cpp).
  if (const std::string deltas = ::renut::gpu::SnapshotDeltaReport(); !deltas.empty())
    REXLOG_INFO("[renut] {}", deltas);
  s = Stats{};
  s.since = s.released = released;
}

// Until 't': a high-resolution waitable timer on Windows (a plain sleep there overshoots by up to a couple of
// milliseconds, and yielding under load gives the time slice away for a whole quantum), the kernel's sleep elsewhere
// (the Switch's is precise); the last 200 us spinning.
void SleepUntil(Clock::time_point t) {
  constexpr auto kSpin = std::chrono::microseconds(200);
#if defined(_WIN32)
  static HANDLE timer = [] {
    HANDLE h = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
    return h ? h : CreateWaitableTimerExW(nullptr, nullptr, 0, TIMER_ALL_ACCESS);
  }();
#endif
  for (;;) {
    const auto now = Clock::now();
    if (now >= t) return;
    const auto left = t - now;
    if (left <= kSpin) continue;
    const auto wait = left - kSpin;
#if defined(_WIN32)
    LARGE_INTEGER due;
    due.QuadPart = -LONGLONG(std::chrono::duration_cast<std::chrono::nanoseconds>(wait).count() / 100);
    if (timer && SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE)) {
      WaitForSingleObject(timer, INFINITE);
      continue;
    }
#endif
    std::this_thread::sleep_for(wait);
  }
}

}  // namespace

// sa2-nx (2026-10-20, banjo_display_sync): the last release taken from the display's refresh (steady_clock ns).
uint64_t g_display_previous_ns = 0;

void PaceSwap(uint32_t device_va) {
  static const bool on = REXCVAR_GET(sa2_renut_vblank_pacing);
  if (!on || !device_va) return;
  const auto *interval_word = ::renut::gpu::GuestPtr<const be_u32>(device_va + kDeviceOffsetPresentationInterval);
  const uint32_t refreshes = RefreshesPerFrame(interval_word ? uint32_t(*interval_word) : 1);
  const auto now = Clock::now();
  // sa2-nx (2026-10-20, banjo_display_sync, Banjo's latency set): on the Switch the release follows the display's own
  // refresh (its vsync event) plus a measured phase, as late as the frame's image still makes the refresh, instead of
  // this clock's 60 Hz grid, which drifts against the display's. A late frame goes at once (as relaxed vsync); fixed
  // vsync (sa2_vsync_relaxed off) and an unknown refresh (the PC, the first frames) keep the grid below.
  ::banjo::display_sync::Start();
  if (g_started && refreshes && g_vsync_relaxed.active) {
    const auto period = kRefresh * refreshes;
    const uint64_t now_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count());
    const uint64_t period_ns = uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(period).count());
    bool anchored = false;
    const uint64_t release = ::banjo::display_sync::NextRelease(now_ns, g_display_previous_ns, period_ns, &anchored);
    if (release) {
      const auto at = Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::nanoseconds(release)));
      if (release > now_ns) SleepUntil(at);
      g_display_previous_ns = release;
      g_due = at + period;  // the grid follows, for when the refresh is not known for a while
      ::banjo::display_sync::NoteRelease(release, anchored);
      NoteRelease(now, Clock::now(), !anchored);
      return;
    }
  }
  if (!g_started || !refreshes) {
    g_started = true;
    g_grid = now;
    g_due = now + kRefresh * (refreshes ? refreshes : 1);
    NoteRelease(now, now, false);
    return;
  }
  const auto period = kRefresh * refreshes;
  if (now <= g_due) {
    // On time: released at the due refresh; the next is due a period after it (a fixed grid, so the sleep's own
    // lateness does not add up).
    SleepUntil(g_due);
    g_due += period;
    NoteRelease(now, Clock::now(), false);
    return;
  }
  if (g_vsync_relaxed.active) {
    // Late: at once. A little late (under half a refresh: the host's scheduling), the grid is kept; more, the next
    // frame is due a period after this one.
    g_due = now - g_due < kRefresh / 2 ? g_due + period : now + period;
    NoteRelease(now, now, true);
    return;
  }
  // Late, fixed vsync: the next refresh of the grid (g_due lies on it).
  const auto refreshes_late = (now - g_due + kRefresh - Clock::duration(1)) / kRefresh;
  const auto release = g_due + kRefresh * refreshes_late;
  SleepUntil(release);
  g_due = release + period;
  NoteRelease(now, Clock::now(), true);
}

}  // namespace sa2::renut
