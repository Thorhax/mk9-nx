// U1 host test: a package loaded through its index (LibraryShaders::LoadIndexed, SPIR-V read on first
// use) is identical to the full load.
//   test_shader_index <package.mesp> <package.idx> [--all] [--other <index of another package>]
//     every entry: container bytes, fingerprint, stage, order, Search result, stored SPIR-V facts (words,
//     XXH3, kills) against the full load; SPIR-V bytes of 2000 random entries (all
//     with --all), half read on demand and half by Preload on another thread racing Spirv() calls;
//     a damaged index and (with --other) another package's index are rejected.
//   test_shader_index --time-full <package.mesp> | --time-index <package.mesp> <package.idx>
//     load time and peak RSS of one load mode alone.
#include "../../app/src/me/native/sa2_shader_library.h"

#include <sys/resource.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <thread>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace fs = std::filesystem;
using sa2::native::LibraryShaders;
using sa2::native::Shader;
using Clock = std::chrono::steady_clock;

// True once the lazily read SPIR-V of an indexed entry has been published.
static bool SpirvResident(const Shader& s) { return s.ready.v.load(std::memory_order_acquire); }

static double Ms(Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); }
static double PeakMb() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
#ifdef __APPLE__
  return double(u.ru_maxrss) / (1024.0 * 1024.0);
#else
  return double(u.ru_maxrss) / 1024.0;
#endif
}
static void Check(bool ok, const char* what, size_t i = SIZE_MAX) {
  if (ok) return;
  if (i != SIZE_MAX) std::fprintf(stderr, "FAIL %s (entry %zu)\n", what, i);
  else std::fprintf(stderr, "FAIL %s\n", what);
  std::exit(1);
}

int main(int argc, char** argv) try {
  if (argc == 3 && std::string_view(argv[1]) == "--time-full") {
    const auto t = Clock::now();
    LibraryShaders full;
    full.Load(fs::path(argv[2]));
    std::printf("full load: %zu entries, %.0f ms, peak RSS %.0f MB\n", full.shaders().size(), Ms(t), PeakMb());
    return 0;
  }
  if (argc == 4 && std::string_view(argv[1]) == "--time-index") {
    const auto t = Clock::now();
    LibraryShaders lazy;
    lazy.LoadIndexed(argv[2], argv[3]);
    const double ms = Ms(t);
    const double rss = PeakMb();
    const auto t2 = Clock::now();
    size_t bytes = 0;
    for (size_t i = 0; i < lazy.shaders().size(); i += 25) bytes += lazy.shaders()[i].Spirv().size() * 4;
    std::printf("index load: %zu entries, %.0f ms, peak RSS %.0f MB; then 1 in 25 SPIR-V on demand (%zu, %.1f MB) "
                "%.0f ms, peak RSS %.0f MB\n",
                lazy.shaders().size(), ms, rss, (lazy.shaders().size() + 24) / 25, bytes / 1048576.0, Ms(t2), PeakMb());
    return 0;
  }
  if (argc < 3) throw std::runtime_error("usage: see the comment at the top");
  const fs::path package = argv[1], index = argv[2];
  bool all = false;
  fs::path other;
  for (int i = 3; i < argc; ++i) {
    if (std::string_view(argv[i]) == "--all") all = true;
    else if (std::string_view(argv[i]) == "--other" && i + 1 < argc) other = argv[++i];
  }

  auto t = Clock::now();
  LibraryShaders full;
  full.Load(package);
  std::printf("full load %.0f ms\n", Ms(t));
  t = Clock::now();
  LibraryShaders lazy;
  lazy.LoadIndexed(package, index);
  std::printf("index load %.0f ms\n", Ms(t));

  const auto& a = full.shaders();
  const auto& b = lazy.shaders();
  Check(a.size() == b.size(), "entry count");
  Check(lazy.is_indexed() && !full.is_indexed(), "load modes");
  for (size_t i = 0; i < a.size(); ++i) {
    Check(b[i].indexed() && !a[i].indexed(), "entry mode", i);
    Check(a[i].original == b[i].original, "container bytes", i);
    Check(a[i].fingerprint == b[i].fingerprint && a[i].vertices == b[i].vertices, "fingerprint/stage", i);
    Check(b[i].words == a[i].spirv.size(), "SPIR-V words", i);
    Check(b[i].spirv_fingerprint == XXH3_64bits(a[i].spirv.data(), a[i].spirv.size() * 4), "SPIR-V hash", i);
    Check(b[i].kills == sa2::native::CountKillsSpirv(a[i].spirv), "kills", i);
    // What the start-up code and IdentifyContainer look up: every container finds the same entry.
    Check(full.Search(a[i].original) == &a[i], "full Search", i);
    Check(lazy.Search(a[i].original) == &b[i], "index Search", i);
    Check(!SpirvResident(b[i]), "nothing resident after the index load", i);
  }
  std::printf("all %zu entries: containers, fingerprints, order, Search and stored SPIR-V facts identical\n",
              a.size());

  // SPIR-V bytes: a random sample (or all), half through Preload on another thread while this thread
  // calls Spirv() on the same entries in another order.
  std::vector<size_t> sample(a.size());
  for (size_t i = 0; i < sample.size(); ++i) sample[i] = i;
  std::mt19937_64 rng(0x5EED0001);
  std::shuffle(sample.begin(), sample.end(), rng);
  if (!all) sample.resize(std::min<size_t>(2000, sample.size()));
  std::vector<const Shader*> prefetch;
  for (size_t k = 0; k < sample.size(); k += 2) prefetch.push_back(&b[sample[k]]);
  sa2::native::StatsPreload stats;
  t = Clock::now();
  std::thread thread([&] { stats = lazy.Preload(prefetch, nullptr); });
  size_t bytes = 0;
  for (size_t k = sample.size(); k-- > 0;) {
    const auto& spirv = b[sample[k]].Spirv();
    Check(spirv == a[sample[k]].spirv, "SPIR-V bytes", sample[k]);
    bytes += spirv.size() * 4;
  }
  thread.join();
  Check(!stats.error, "Preload error");
  Check(stats.inputs + stats.already_resident == prefetch.size(), "Preload count");
  for (size_t k : sample) {
    Check(SpirvResident(b[k]) && b[k].Spirv() == a[k].spirv, "SPIR-V after Preload", k);
    Check(b[k].Spirv().data() == b[k].Spirv().data(), "stable address", k);
  }
  size_t resident = 0;
  for (const auto& s : b) resident += SpirvResident(s);
  Check(resident == sample.size(), "only the sampled entries are resident");
  std::printf("SPIR-V of %zu entries identical (%.1f MB, %.0f ms); Preload read %u, %u already resident\n",
              sample.size(), bytes / 1048576.0, Ms(t), stats.inputs, stats.already_resident);

  // A damaged index is rejected and leaves the library as it was.
  {
    std::ifstream f(index, std::ios::binary);
    std::vector<char> d((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    d[d.size() / 2] ^= 1;
    const fs::path damaged = fs::temp_directory_path() / "test_shader_index_damaged.idx";
    std::ofstream(damaged, std::ios::binary).write(d.data(), std::streamsize(d.size()));
    bool rejected = false;
    try {
      lazy.LoadIndexed(package, damaged);
    } catch (const std::exception& e) {
      rejected = true;
      std::printf("damaged index rejected: %s\n", e.what());
    }
    fs::remove(damaged);
    Check(rejected && lazy.shaders().size() == a.size() && lazy.is_indexed(), "damaged index");
  }
  if (!other.empty()) {
    bool rejected = false;
    try {
      LibraryShaders mismatch;
      mismatch.LoadIndexed(package, other);
    } catch (const std::exception& e) {
      rejected = true;
      std::printf("other package's index rejected: %s\n", e.what());
    }
    Check(rejected, "other package's index");
  }
  std::printf("OK\n");
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
