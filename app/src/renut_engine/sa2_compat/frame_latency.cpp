// Frame latency probe (rex/frame_latency.h). banjo-nx 2026-10-19: from futari-nx (the user's port), without the
// display vsync and trace parts, with the late input hook.
#include <rex/frame_latency.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include <fmt/format.h>

namespace rex::latency {
namespace {

uint64_t Now() {
  return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                      std::chrono::steady_clock::now().time_since_epoch())
                      .count());
}

// The frames in flight, by the ordinal of the Swap that ends them (1-based). Far more than can be in flight.
constexpr uint32_t kFrames = 64;
struct Frame {
  std::atomic<uint64_t> ordinal{0};
  std::atomic<uint64_t> input{0};
  std::atomic<uint64_t> swap{0};
};
Frame g_frames[kFrames];
std::atomic<uint64_t> g_swaps{0};        // the game's Swaps so far
std::atomic<uint64_t> g_input_frame{0};  // the frame whose first controller read is recorded

struct Sum {
  uint64_t n = 0;
  double total = 0;
  double max = 0;
  void Add(double ms) {
    ++n;
    total += ms;
    if (ms > max) max = ms;
  }
  double Mean() const { return n ? total / double(n) : 0.0; }
};

// Once per frame at the present: a mutex is cheap enough there.
std::mutex g_mutex;
Sum g_input_swap, g_swap_present, g_input_present;
Sum g_input_gpu;  // banjo-nx (2026-10-20): input -> the GPU finished the frame (NoteGpuDone)

double Ms(uint64_t from, uint64_t to) { return to > from ? double(to - from) / 1e6 : 0.0; }

std::atomic<FrameReadDelayFunction> g_delay_fn{nullptr};
std::atomic<uint64_t> g_delayed_frame{0};  // the frame whose first read was delayed (or looked at)

}  // namespace

void NoteInputRead() {
  const uint64_t frame = g_swaps.load(std::memory_order_relaxed) + 1;
  if (g_input_frame.load(std::memory_order_relaxed) == frame) return;  // not the frame's first read
  g_input_frame.store(frame, std::memory_order_relaxed);
  Frame& f = g_frames[frame % kFrames];
  f.ordinal.store(0, std::memory_order_relaxed);
  f.swap.store(0, std::memory_order_relaxed);
  f.input.store(Now(), std::memory_order_relaxed);
  f.ordinal.store(frame, std::memory_order_release);
}

void NoteGuestSwap() {
  const uint64_t frame = g_swaps.fetch_add(1, std::memory_order_relaxed) + 1;
  Frame& f = g_frames[frame % kFrames];
  if (f.ordinal.load(std::memory_order_acquire) == frame) f.swap.store(Now(), std::memory_order_release);
}

uint64_t SwapCount() { return g_swaps.load(std::memory_order_acquire); }

uint64_t InputTimeOf(uint64_t swap) {
  const Frame& f = g_frames[swap % kFrames];
  if (f.ordinal.load(std::memory_order_acquire) != swap) return 0;
  return f.input.load(std::memory_order_relaxed);
}

void NotePresented(uint64_t swap) {
  const uint64_t now = Now();
  Frame& f = g_frames[swap % kFrames];
  if (f.ordinal.load(std::memory_order_acquire) != swap) return;
  const uint64_t input = f.input.load(std::memory_order_relaxed);
  const uint64_t at_swap = f.swap.load(std::memory_order_acquire);
  if (!input || !at_swap) return;
  std::lock_guard lock(g_mutex);
  g_input_swap.Add(Ms(input, at_swap));
  g_swap_present.Add(Ms(at_swap, now));
  g_input_present.Add(Ms(input, now));
}

void NoteGpuDone(uint64_t swap) {
  const uint64_t now = Now();
  const Frame& f = g_frames[swap % kFrames];
  if (!swap || f.ordinal.load(std::memory_order_acquire) != swap) return;
  const uint64_t input = f.input.load(std::memory_order_relaxed);
  if (!input) return;
  std::lock_guard lock(g_mutex);
  g_input_gpu.Add(Ms(input, now));
}

std::string Report() {
  std::lock_guard lock(g_mutex);
  if (!g_input_present.n) return {};
  std::string line = fmt::format(
      "input -> swap {:.1f} ms (max {:.1f}), swap -> present {:.1f} (max {:.1f}), input -> present {:.1f} (max "
      "{:.1f}); {} frames",
      g_input_swap.Mean(), g_input_swap.max, g_swap_present.Mean(), g_swap_present.max, g_input_present.Mean(),
      g_input_present.max, g_input_present.n);
  // banjo-nx (2026-10-20): what the display can show first: the frame's GPU work done (seen within a few ms).
  if (g_input_gpu.n)
    line += fmt::format("; input -> GPU done {:.1f} ms (max {:.1f}, {} frames)", g_input_gpu.Mean(), g_input_gpu.max,
                        g_input_gpu.n);
  g_input_swap = g_swap_present = g_input_present = g_input_gpu = Sum{};
  return line;
}

void SetFrameReadDelayFunction(FrameReadDelayFunction fn) { g_delay_fn.store(fn, std::memory_order_release); }

void BeforeInputRead() {
  const FrameReadDelayFunction fn = g_delay_fn.load(std::memory_order_acquire);
  if (!fn) return;
  const uint64_t frame = g_swaps.load(std::memory_order_relaxed) + 1;
  uint64_t seen = g_delayed_frame.load(std::memory_order_relaxed);
  if (seen == frame || !g_delayed_frame.compare_exchange_strong(seen, frame, std::memory_order_relaxed)) return;
  if (const uint32_t us = fn()) std::this_thread::sleep_for(std::chrono::microseconds(us));
}

}  // namespace rex::latency
