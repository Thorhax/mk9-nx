/* Horizon console performance profile (apm). See switch_apm.cpp. */
#pragma once

#include "rex/platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * GPU MHz to request in handheld mode: 0 = touch nothing, 384 or 460. Called before
 * RexSwitchApmApply.
 */
void RexSwitchApmRequestGpuMhz(int mhz);

/*
 * CPU boost mode (appletSetCpuBoostMode): 0 = normal, 1 = FastLoad (CPU 1785 MHz; the system also throttles the
 * GPU to its minimum while it is on), 2 = Type2. Applied immediately; the result is printed with the real
 * clocks one second later by RexSwitchApmWatch. Diagnostic: the GPU throttle makes mode 1 unusable in game
 * unless this firmware lets the GPU request win.
 */
void RexSwitchApmCpuBoost(int mode);

/*
 * Accept a configuration even if it raises the RAM to 1600. Off by default: if this firmware does
 * not have the "high GPU + unchanged RAM" pair, keeping the GPU as it was is preferred. Called
 * before RexSwitchApmApply.
 */
void RexSwitchApmAllowRam1600(int allow);

/*
 * Requests the profile. Does nothing if 0 was requested, if the console is docked, or if it was
 * already called. If the configuration does not exist in this firmware, apm returns an error and
 * changes nothing.
 */
void RexSwitchApmApply(void);

/*
 * One tick per second from the profile thread. If the memory clock went up on its own after the
 * configuration was applied (the clock change takes time, so an immediate check read the old
 * value), it is set back with clkrst to what the console had originally. It does nothing, and does
 * not even open clkrst, if nothing was changed.
 */
void RexSwitchApmWatch(void);

#ifdef __cplusplus
}
#endif
