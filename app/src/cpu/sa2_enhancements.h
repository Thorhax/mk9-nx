// Sonic Adventure 2: the switchable enhancements (performance changes that keep the image and the game's behaviour
// the same). Every enhancement has
//
//   - its own key in the toml (a cvar, default on unless its file says otherwise);
//   - a group: CPU (sa2_cpu_enhancements) or non-CPU (sa2_noncpu_enhancements). An enhancement runs only when its own
//     key AND its group's master key are on;
//   - one entry in the startup line "[enh] ..." (which keys are on, which are active), and a use counter printed every
//     10 seconds while it is used, so a console log shows that it really ran;
//   - three lines in docs/enhancements.md: why it is exact, what could break it, its key.
//
// Build-time changes (codegen, compiler flags) cannot be switched at run time: they are a build option plus an A/B
// NRO pair, listed in docs/enhancements.md too, and the build names them in the first log line.
//
// Use:
//   SA2_ENHANCEMENT(g_gpu_wait, sa2_cpu_gpu_wait_sleep, sa2::enh::kCpu, true, "description");
//   if (g_gpu_wait.active) { g_gpu_wait.Use(); ... }
//
// `active` is resolved once, after the configuration is read and before any guest thread exists (Resolve, called by
// the app's OnPostInitLogging), and is a plain bool afterwards: a hot path pays one byte load.
#pragma once

#include <atomic>
#include <cstdint>

#include <rex/cvar.h>

namespace sa2::enh {

enum Group : uint8_t { kCpu = 0, kNonCpu = 1 };

struct Enhancement {
  const char* key;
  Group group;
  const char* what;
  bool (*own)();  // reads the enhancement's own cvar
  bool active = false;
  std::atomic<uint64_t> uses{0};
  // Extra number for the periodic line (e.g. microseconds slept), with its unit; 0 = none.
  std::atomic<uint64_t> amount{0};
  const char* amount_unit = nullptr;

  void Use(uint64_t n = 1) { uses.fetch_add(n, std::memory_order_relaxed); }
  // For per-draw / per-call hot paths: a plain relaxed load and store (no read-modify-write; racing threads may lose
  // a count, it is only a report), and one shared add per 256 uses.
  std::atomic<uint32_t> sampled{0};
  void UseSampled() {
    const uint32_t n = sampled.load(std::memory_order_relaxed) + 1;
    sampled.store(n, std::memory_order_relaxed);
    if ((n & 255) == 0) Use(256);
  }
  void Add(uint64_t n) { amount.fetch_add(n, std::memory_order_relaxed); }
};

// Static registration (the macro below); safe at static-initialisation time.
bool Register(Enhancement* e);

// Reads the masters and every key, sets `active`, logs the summary line. Called once by the app.
void Resolve();

// Group master switches (resolved by Resolve).
bool CpuMaster();
bool NonCpuMaster();

// Starts the 10-second use report (a small background thread). Called once by the app after setup.
void StartReport();

// Verify counters (sa2_cpu_verify; '[verify]' line of the 10 s report): every comparison of an enhancement with the
// original path counts a check, every mismatch a difference (which must stay 0).
void VerifyCheck(uint64_t n = 1);
void VerifyDifference();
bool VerifyMode();

// Non-CPU: the handheld GPU profile (sa2_gpu_boost; sa2_gpu_boost.cpp). Called once by the app before setup.
void ApplyGpuBoost();

// True when the loaded guest code at `address` (`bytes` long) has the FNV-1a 64 hash a replacement was written from
// (tools/guest_code_hash.py). Logs a warning naming `what` when it does not.
bool GuestCodeMatches(const uint8_t* base, uint32_t address, uint32_t bytes, uint64_t fnv, const char* what);

}  // namespace sa2::enh

#define SA2_ENHANCEMENT(VAR, KEY, GROUP, DEFAULT, DESC)                                                     \
  REXCVAR_DEFINE_BOOL(KEY, DEFAULT, "SA2 enhancements", DESC).lifecycle(rex::cvar::Lifecycle::kInitOnly); \
  ::sa2::enh::Enhancement VAR{#KEY, GROUP, DESC, [] { return REXCVAR_GET(KEY); }};                       \
  [[maybe_unused]] static const bool VAR##_registered_ = ::sa2::enh::Register(&VAR)
