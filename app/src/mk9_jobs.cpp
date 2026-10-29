// mk9-nx: the game's job system, a deadlock fixed (jobs.toml).
//
// A parent job runs sub_82819C28 for its group of child jobs: it pops them from the global pending queue
// ([0x8341FC5C], sub_82779990) one by one from the end of the group's array and runs them, and when a pop fails (a
// worker took that job) it spins on the group's counter (+4, decremented as each child finishes) until it is 0. It
// never pops again. The game has two workers; a pop that came before the children were queued left both workers in
// that loop with the children pending and nobody to run them (the Krypt, 2026-10-29: 753 jobs pending, the idle
// list empty, both workers spinning, the groups at 9 and 6). Mk9JobHelperRetry, at the loop's head: while the counter
// is not 0, sleep a moment and pop again, from the next job of the group. A job a worker took is not in the queue, so
// popping it fails and the next one is tried; the counter still decides when the loop ends.

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>

#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ppc/context.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>

REXCVAR_DEFINE_BOOL(mk9_job_helper_retry, true, "MK9",
                    "A parent job waiting for its group of child jobs keeps popping them from the queue instead of "
                    "only spinning (the Krypt hang)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

uint32_t Load32(uint32_t address) {
  uint32_t v;
  std::memcpy(&v, rex::system::kernel_state()->memory()->TranslateVirtual(address), 4);
  return __builtin_bswap32(v);
}

std::atomic<uint64_t> g_retries{0};

}  // namespace

// r29: 4 x the index after the job tried last (the game pops jobs[r29 / 4 - 1]); r30: the group.
bool Mk9JobHelperRetry(PPCRegister& r29, PPCRegister& r30) {
  static const bool on = REXCVAR_GET(mk9_job_helper_retry);
  if (!on || Load32(r30.u32 + 4) == 0) {
    return false;  // the game's loop: it sees 0 and ends
  }
  // the group's job count, as the game read it at the start ([group + 0x38] + 0x44)
  const uint32_t jobs = Load32(Load32(r30.u32 + 0x38) + 0x44);
  uint32_t next = r29.u32 >= 8 ? r29.u32 - 4 : jobs * 4;
  if (next == 0 || next > jobs * 4) {
    next = jobs * 4;
  }
  r29.u64 = next;
  const uint64_t n = g_retries.fetch_add(1, std::memory_order_relaxed);
  if ((n & (n - 1)) == 0) {  // 1, 2, 4, 8, ...: rare in the log
    REXLOG_INFO("[jobs] helper of group {:08X} retries its pops ({} so far, {} children left)", r30.u32, n + 1,
                Load32(r30.u32 + 4));
  }
  std::this_thread::sleep_for(std::chrono::microseconds(50));
  return true;
}
