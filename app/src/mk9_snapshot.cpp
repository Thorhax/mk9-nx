// Still snapshots of the game image (see mk9_snapshot.h): at each time of mk9_snapshot_s (seconds after start,
// comma-separated; fractions allowed; "a-b@step" = every step seconds from a to b, for effects that last a frame or
// two), the presenter's guest output is written to captures/snap_<s>s.ppm next to the executable (binary PPM, RGB;
// <s> with milliseconds when the time has a fraction). Convert or view with any image tool.

#include "mk9_snapshot.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <filesystem>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/ui/presenter.h>
#include <fmt/format.h>

REXCVAR_DEFINE_STRING(mk9_snapshot_s, "", "MK9",
                      "Seconds after start at which to write a snapshot of the game image to captures/ next to the "
                      "executable, comma-separated (e.g. \"10,20,30\" or \"150-152@0.05\"; empty = off)");

namespace mk9::snapshot {
namespace {

std::thread g_thread;
std::mutex g_mutex;
std::condition_variable g_cv;
bool g_stop = false;

bool WritePpm(const std::filesystem::path& path, const rex::ui::RawImage& image) {
  std::FILE* f = std::fopen(path.string().c_str(), "wb");
  if (!f) return false;
  std::fprintf(f, "P6\n%u %u\n255\n", image.width, image.height);
  std::vector<uint8_t> row(size_t(image.width) * 3);
  for (uint32_t y = 0; y < image.height; ++y) {
    const uint8_t* in = image.data.data() + size_t(y) * image.stride;
    for (uint32_t x = 0; x < image.width; ++x) {
      row[x * 3 + 0] = in[x * 4 + 0];
      row[x * 3 + 1] = in[x * 4 + 1];
      row[x * 3 + 2] = in[x * 4 + 2];
    }
    std::fwrite(row.data(), 1, row.size(), f);
  }
  std::fclose(f);
  return true;
}

void Run(std::function<bool(rex::ui::RawImage&)> grab, std::vector<double> times) {
  const auto start = std::chrono::steady_clock::now();
  const auto folder = rex::filesystem::GetExecutableFolder() / "captures";
  std::error_code ec;
  std::filesystem::create_directories(folder, ec);
  for (double s : times) {
    {
      std::unique_lock lock(g_mutex);
      const auto when = start + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                                    std::chrono::duration<double>(s));
      if (g_cv.wait_until(lock, when, [] { return g_stop; })) return;
    }
    rex::ui::RawImage image;
    if (!grab || !grab(image) || !image.width || !image.height) {
      REXLOG_WARN("[snapshot] {} s: no guest output yet", s);
      continue;
    }
    const long ms = std::lround(s * 1000.0);
    const std::string label = ms % 1000 ? fmt::format("{}.{:03}", ms / 1000, ms % 1000) : std::to_string(ms / 1000);
    const auto path = folder / ("snap_" + label + "s.ppm");
    if (WritePpm(path, image)) {
      REXLOG_INFO("[snapshot] {} s: {}x{} written to {}", s, image.width, image.height, path.string());
    }
  }
}

}  // namespace

void Start(std::function<rex::ui::Presenter*()> get_presenter) {
  StartWithSource([get_presenter = std::move(get_presenter)](rex::ui::RawImage& image) {
    rex::ui::Presenter* presenter = get_presenter ? get_presenter() : nullptr;
    return presenter && presenter->CaptureGuestOutput(image);
  });
}

void StartWithSource(std::function<bool(rex::ui::RawImage&)> grab) {
  const std::string spec = REXCVAR_GET(mk9_snapshot_s);
  if (spec.empty()) return;
  std::vector<double> times;
  std::stringstream ss(spec);
  for (std::string item; std::getline(ss, item, ',');) {
    try {
      const auto dash = item.find('-'), at = item.find('@');
      if (dash != std::string::npos && at != std::string::npos && dash < at) {
        const double a = std::stod(item.substr(0, dash)), b = std::stod(item.substr(dash + 1, at - dash - 1));
        const double step = std::stod(item.substr(at + 1));
        if (step > 0 && b >= a && (b - a) / step <= 2000)
          for (double t = a; t <= b + 1e-9; t += step) times.push_back(t);
      } else {
        times.push_back(std::stod(item));
      }
    } catch (...) {
    }
  }
  std::sort(times.begin(), times.end());
  if (times.empty()) return;
  g_stop = false;
  g_thread = std::thread(Run, std::move(grab), std::move(times));
}

void Stop() {
  {
    std::lock_guard lock(g_mutex);
    g_stop = true;
  }
  g_cv.notify_all();
  if (g_thread.joinable()) g_thread.join();
}

}  // namespace mk9::snapshot
