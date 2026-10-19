// Sonic Adventure 2: the switchable enhancements (see sa2_enhancements.h).
#include "sa2_enhancements.h"

#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <fmt/format.h>

#include <rex/logging.h>

#ifndef SA2_BUILD_FLAVOR
#define SA2_BUILD_FLAVOR "classic codegen"
#endif

REXCVAR_DEFINE_BOOL(sa2_cpu_enhancements, true, "SA2 enhancements",
                    "Master switch of every CPU enhancement (each also has its own key)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_noncpu_enhancements, true, "SA2 enhancements",
                    "Master switch of every non-CPU enhancement (each also has its own key)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// Read by the SDK's Switch profiler (sdk/src/ui/switch_perf.cpp) besides logs/rex/stacks_profile.flag: the CPU
// sampler (1 ms, every thread; logs/rex/rex_profile.log). On in the A/B and profile configs, off in the default.
REXCVAR_DEFINE_BOOL(switch_profile_stacks, false, "SA2 enhancements",
                    "Switch: CPU stack sampling every 1 ms (rex_profile.log), as logs/rex/stacks_profile.flag does")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// 2026-10-08: the profiler's 10-second report (logs/rex/rex_profile.log) at all. Off in the release's built-in
// settings (no logs/ folder); the test configs turn it on. The overlay's FPS does not depend on it.
REXCVAR_DEFINE_BOOL(switch_profile_report, true, "SA2 enhancements",
                    "Switch: the profiler writes its 10-second report to logs/rex/rex_profile.log (false = no file)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_enhancements_report_s, 10, "SA2 enhancements",
                     "Seconds between two '[enh] uses' log lines (0 = no periodic line)")
    .range(0, 3600)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace sa2::enh {
namespace {

std::vector<Enhancement*>& List() {
  static std::vector<Enhancement*> list;
  return list;
}

std::atomic<uint64_t> g_verify_checks{0};
std::atomic<uint64_t> g_verify_differences{0};

// Enhancements defined elsewhere as plain cvars: the renderer's ring-thread fast paths (masseffect-nx's measured
// set, each self-checked against the old path), the renderer's GPU optimisations (docs/gpu-optimisation-candidates.md)
// and the nfsmw NVK driver paths. Per row: key, group, the value that turns it on, the value that selects the previous
// path. With the group master on, a key the configuration does not set (still at its compiled default: an old sa2.toml
// on the card lacks the newer keys) is turned on; a key the toml, the environment or the command line sets is respected.
// With the master off every row is set to its previous-path value. on == nullptr: explicit only (never turned on here).
struct External {
  const char* key;
  Group group;
  const char* on;
  const char* off;
};
constexpr External kExternal[] = {
    // CPU: the ring thread (PC lockstep capture with all of these on: every gameplay frame identical).
    {"sa2_native_flat_identity", kCpu, "true", "false"},
    {"sa2_native_fast_pair", kCpu, "true", "false"},
    {"sa2_native_raw_microcode", kCpu, "true", "false"},
    {"sa2_native_pm4_fast", kCpu, "true", "false"},
    {"sa2_native_compare_fast", kCpu, "true", "false"},
    {"sa2_native_fast_copy", kCpu, "true", "false"},
    // Round 6: 2 = the hardware CRC32 pair on the Switch (+crc); a build without it falls back to 1 by itself.
    {"sa2_native_fingerprint_mode", kCpu, "2", "0"},
    {"sa2_native_dedupe_max_fingerprint", kCpu, "16384", "0"},
    {"sa2_native_indices_cache", kCpu, "true", "false"},
    {"sa2_native_pipelines_prewarm", kCpu, "true", "false"},
    // Round 3: the upload buffer (every draw's vertices and indices) in cached memory, published with
    // vkFlushMappedMemoryRanges before each submit. The console measured 5,251 MB/s of CPU writes there against
    // 2,506 MB/s in the uncached type the SDK picks; the shared constants already use the cached type this way.
    {"sa2_native_upload_memory", kCpu, "1", "0"},
    // Round 5: deferred Vulkan recording and its two refinements, on (PC lockstep captures pixel-identical; the
    // 100+ ms PC frames it once gave were Windows' timer period, fixed). configs/sa2_test_deferred_off.toml: the A/B.
    {"sa2_native_deferred_recording", kCpu, "true", "false"},
    {"sa2_deferred_native_fast", kCpu, "true", "false"},
    {"sa2_deferred_native_update", kCpu, "true", "false"},
    // Non-CPU: the renderer's GPU optimisations.
    {"sa2_native_edram4_clear_color0", kNonCpu, "true", "false"},
    {"sa2_native_constants_dedupe", kNonCpu, "true", "false"},
    {"sa2_native_pass_narrow_end", kNonCpu, "true", "false"},
    {"sa2_native_zcull_1x_depth", kNonCpu, nullptr, "false"},  // needs a long play test (GPU loss in Unleashed)
    // Non-CPU: the nfsmw NVK driver's paths (constants by differences, per-draw emission).
    {"sa2_native_set4_differences", kNonCpu, "true", "false"},
    {"sa2_native_nvk_emission", kNonCpu, "true", "false"},
    {"sa2_native_nvk_cbufs", kNonCpu, "true", "false"},
    {"sa2_native_nvk_dynamic", kNonCpu, "true", "false"},
    {"sa2_native_nvk_preload", kNonCpu, "true", "false"},
};

bool g_cpu_master = false;
bool g_noncpu_master = false;
bool g_resolved = false;

}  // namespace

bool Register(Enhancement* e) {
  List().push_back(e);
  return true;
}

void VerifyCheck(uint64_t n) { g_verify_checks.fetch_add(n, std::memory_order_relaxed); }
void VerifyDifference() { g_verify_differences.fetch_add(1, std::memory_order_relaxed); }
// Read by name once Resolve has run (a by-name lookup is a string map search: hot paths may call this).
bool g_verify_mode = false;
bool VerifyMode() { return g_resolved ? g_verify_mode : rex::cvar::GetFlagByName("sa2_cpu_verify") == "true"; }

bool CpuMaster() { return g_cpu_master; }
bool NonCpuMaster() { return g_noncpu_master; }

void Resolve() {
  if (g_resolved) return;
  g_resolved = true;
  g_cpu_master = REXCVAR_GET(sa2_cpu_enhancements);
  g_noncpu_master = REXCVAR_GET(sa2_noncpu_enhancements);
  g_verify_mode = rex::cvar::GetFlagByName("sa2_cpu_verify") == "true";
  if (VerifyMode()) {
    // The renderer's own self-checks (old and new path compared, DIFFERENCE/MISMATCH logged) check every use.
    rex::cvar::SetFlagByName("sa2_native_verify_n", "2147483647");
    rex::cvar::SetFlagByName("sa2_native_fast_copy_verify", "2147483647");
    REXLOG_INFO("[verify] sa2_cpu_verify: every native replacement call and every renderer fast path is checked "
                "against the original path; '[verify]' lines report the checks and differences (must stay 0)");
  }
  std::string line;
  for (Enhancement* e : List()) {
    const bool own = e->own();
    const bool master = e->group == kCpu ? g_cpu_master : g_noncpu_master;
    e->active = own && master;
    line += fmt::format(" {}={}", e->key, e->active ? "on" : (own ? "on-but-master-off" : "off"));
  }
  int defaulted_on = 0;
  for (const External& x : kExternal) {
    const bool master = x.group == kCpu ? g_cpu_master : g_noncpu_master;
    if (master && x.on && rex::cvar::GetFlagSource(x.key) == rex::cvar::Source::kDefault &&
        rex::cvar::GetFlagByName(x.key) != x.on) {
      rex::cvar::SetFlagByName(x.key, x.on);  // not in the configuration: on
      ++defaulted_on;
    }
    const bool own = rex::cvar::GetFlagByName(x.key) != x.off;
    if (own && !master) rex::cvar::SetFlagByName(x.key, x.off);
    line += fmt::format(" {}={}", x.key, own && master ? "on" : (own ? "on-but-master-off" : "off"));
  }
  REXLOG_INFO("[enh] sa2-nx build {} {} ({}); sa2_cpu_enhancements={} sa2_noncpu_enhancements={} ({} not in the "
              "configuration, turned on);{}",
              __DATE__, __TIME__, SA2_BUILD_FLAVOR, g_cpu_master ? "on" : "off", g_noncpu_master ? "on" : "off",
              defaulted_on, line);
}

bool GuestCodeMatches(const uint8_t* base, uint32_t address, uint32_t bytes, uint64_t fnv, const char* what) {
  uint64_t h = 0xCBF29CE484222325ull;
  const uint8_t* p = base + address;
  for (uint32_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001B3ull;
  }
  if (h != fnv) {
    REXLOG_WARN("[enh] {}: the game code at {:08X} is not the code it was written from (hash {:016X}, expected "
                "{:016X}); it stays off",
                what, address, h, fnv);
    return false;
  }
  return true;
}

void StartReport() {
  const int seconds = REXCVAR_GET(sa2_enhancements_report_s);
  if (seconds <= 0) return;
  // Never joined and never destroyed (libnx cannot detach a thread; see the SDK's threading_posix.cpp).
  new std::thread([seconds] {
    std::vector<uint64_t> last_uses(List().size(), 0), last_amount(List().size(), 0);
    for (;;) {
      std::this_thread::sleep_for(std::chrono::seconds(seconds));
      std::string line;
      for (size_t i = 0; i < List().size(); ++i) {
        Enhancement* e = List()[i];
        if (!e->active) continue;
        const uint64_t uses = e->uses.load(std::memory_order_relaxed);
        const uint64_t amount = e->amount.load(std::memory_order_relaxed);
        const uint64_t du = uses - last_uses[i], da = amount - last_amount[i];
        last_uses[i] = uses;
        last_amount[i] = amount;
        if (!du && !da) continue;
        line += fmt::format(" {} {}", e->key, du);
        if (e->amount_unit) line += fmt::format(" ({} {})", da, e->amount_unit);
      }
      if (!line.empty()) REXLOG_INFO("[enh] uses in the last {} s:{}", seconds, line);
      const uint64_t checks = g_verify_checks.load(std::memory_order_relaxed);
      const uint64_t differences = g_verify_differences.load(std::memory_order_relaxed);
      if (checks || differences)
        REXLOG_INFO("[verify] checks done: native replacements {} | differences (native replacements + renderer "
                    "self-checks): {}",
                    checks, differences);
    }
  });
}

}  // namespace sa2::enh
