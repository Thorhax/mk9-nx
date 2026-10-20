// banjo-nx: the frame statistics Renut's renderer reports into (renut_engine/perf/frame_stats.h, copied from Renut by
// tools/renut_gpu_import.py). Renut's own frame_stats.cpp also hooks the game's main loop (appMainTickPreDraw /
// appMainDraw), which banjo-nx's FPS code hooks already, so this keeps the numbers without those hooks: a game frame is
// one D3DDevice_Swap, its length the time between two.
#include "renut_engine/perf/frame_stats.h"

#include <algorithm>
#include <chrono>
#include <mutex>

// Counter 0 of the console profiler: presented frames, which the Status Monitor overlay's FPS (SaltyNX) and the
// profiler's "game N fps" are made from (sdk/src/ui/switch_perf.cpp; a stub on the PC, me_pc_stubs.cpp).
extern "C" void RexSwitchPerfCount(unsigned id);

#include "profiler/profiler.h"                      // the on-screen profiler's GPU, present and render thread rows
#include "renut_engine/gpu/banjo_renut_tuning.h"  // the [renut] report and its in-run A/B

namespace renut::frame_stats {

namespace {

using Clock = std::chrono::steady_clock;

struct State {
  std::mutex mutex;
  Snapshot snap;
  Clock::time_point last_frame{};
  Clock::time_point window_start{};
  int window_frames = 0;
  double window_worst = 0.0;
  double gpu_sum = 0.0, render_sum = 0.0, wait_sum = 0.0;
  int gpu_n = 0, render_n = 0;
};

State& S() {
  static State s;
  return s;
}

}  // namespace

void OnGameFrame() {
  banjo::renut_gpu::StatsGameFrame();
  State& s = S();
  std::lock_guard lock(s.mutex);
  const auto now = Clock::now();
  if (s.last_frame != Clock::time_point{}) {
    const double ms = std::chrono::duration<double, std::milli>(now - s.last_frame).count();
    Snapshot& snap = s.snap;
    if (snap.history_count < Snapshot::kHistory) {
      snap.history[snap.history_count++] = float(ms);
    } else {
      std::copy(snap.history + 1, snap.history + Snapshot::kHistory, snap.history);
      snap.history[Snapshot::kHistory - 1] = float(ms);
    }
    s.window_worst = std::max(s.window_worst, ms);
    ++s.window_frames;
  } else {
    s.window_start = now;
  }
  s.last_frame = now;
  const double window = std::chrono::duration<double>(now - s.window_start).count();
  if (window >= 1.0 && s.window_frames) {
    Snapshot& snap = s.snap;
    snap.fps = s.window_frames / window;
    snap.frame_ms = window * 1000.0 / s.window_frames;
    snap.worst_ms = s.window_worst;
    snap.cpu_ms = std::max(0.0, snap.frame_ms - s.wait_sum / s.window_frames);
    if (s.gpu_n) {
      snap.gpu_ms = s.gpu_sum / s.gpu_n;
      snap.gpu_known = true;
    }
    if (s.render_n) {
      snap.render_thread_ms = s.render_sum / s.render_n;
      snap.render_thread_known = true;
    }
    s.window_start = now;
    s.window_frames = 0;
    s.window_worst = 0.0;
    s.gpu_sum = s.render_sum = s.wait_sum = 0.0;
    s.gpu_n = s.render_n = 0;
  }
}

void OnGpuFrame(double ms) {
  banjo::renut_gpu::StatsGpuFrame(ms);
  banjo::profiler::SetGpuFrameMs(ms);  // the profiler window's "GPU Frame" (package 27: nobody fed it before)
  State& s = S();
  std::lock_guard lock(s.mutex);
  s.gpu_sum += ms;
  ++s.gpu_n;
}

void OnRenderThreadFrame(double busy_ms) {
  banjo::renut_gpu::StatsRenderThreadFrame(busy_ms);
  // The profiler window (package 27): the render thread's frame ends with its present, so this is also the time
  // between two host presents ("Present"); one thread, as RecordHostPresent needs.
  banjo::profiler::SetRenderThreadMs(busy_ms);
  banjo::profiler::RecordHostPresent();
  // Package 27c: the presented frame for the console's FPS. Only banjo-nx's own renderer (and the SDK's) counted
  // them, so with Renut's renderer the Status Monitor kept an old value (19.7) and the profile said "game 0.0 fps".
  RexSwitchPerfCount(0);
  State& s = S();
  std::lock_guard lock(s.mutex);
  s.render_sum += busy_ms;
  ++s.render_n;
}

void OnGameThreadWait(double ms) {
  State& s = S();
  std::lock_guard lock(s.mutex);
  s.wait_sum += ms;
}

Snapshot Get() {
  State& s = S();
  std::lock_guard lock(s.mutex);
  return s.snap;
}

}  // namespace renut::frame_stats
