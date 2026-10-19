#pragma once
/**
 * @file        sys_counters.h
 * @brief       Passive counters of the host runtime's own overhead (wake-ups, lock stalls) shown in the
 *              Switch profiler report line "system". Relaxed atomic adds; compiled out off the Switch.
 *
 * The ids index the profiler's counter array (src/ui/switch_perf.cpp, kCounterCount); the report prints
 * the per-second delta of each. Keep the enum in sync with the report line there.
 */

#include <cstdint>

namespace rex::syscount {

enum : unsigned {
  kAlertSlice = 40,       // alertable wait woke from its poll slice without anything to do
  kAlertWake = 41,        // alertable wait interrupted by a queued user callback / APC hint
  kMultiBlock = 42,       // WaitMultiple parked on its wake point (event-driven)
  kMultiWake = 43,        // ... woken by a signalled handle
  kMultiSafety = 44,      // ... woke by the safety re-poll (a handle changed without telling us)
  kMultiPoll = 45,        // legacy WaitMultiple 1 ms sleep polls (thread_wait_blocking = false)
  kTimestampTick = 46,    // KeTimeStampBundle timer callbacks
  kClockCall = 47,        // UpdateGuestClock calls
  kClockBlocked = 48,     // ... that found the clock mutex taken
  kObjLookup = 49,        // ObjectTable::LookupObject calls
  kObjBlocked = 50,       // ... that found the global critical region taken
  kVolumeMask = 51,       // XAudioGetVoiceCategoryVolumeChangeMask calls
  kVblankTick = 52,       // native vblank thread wake-ups (app)
  kAudioNative = 53,      // native audio DSP replacements run (app)
  kAudioMismatch = 54,    // ... validate mode mismatches (app)
  kCritSecWait = 55,      // RtlEnterCriticalSection fell into a full host wait
  kCount = 56,
};

}  // namespace rex::syscount

#if defined(__SWITCH__)
extern "C" void RexSwitchPerfAdd(unsigned id, uint64_t value);
#define REX_SYS_COUNT(id, n) RexSwitchPerfAdd((id), static_cast<uint64_t>(n))
#else
#define REX_SYS_COUNT(id, n) ((void)0)
#endif
