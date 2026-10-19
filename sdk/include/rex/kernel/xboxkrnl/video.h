/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <atomic>
#include <cstdint>

#include <rex/system/xtypes.h>
#include <rex/system/xvideo.h>

namespace rex::runtime {
class ExportResolver;
}

namespace rex::system {
class KernelState;
}

namespace rex::kernel::xboxkrnl {

void VdQueryVideoMode(system::X_VIDEO_MODE* video_mode);

// Frames the guest handed to VdSwap so far.
uint64_t VdSwapCount();
// cvar vblank_per_swap: the guest vblank timers (the emulated GPU's and a native renderer's) give one vblank per frame
// the guest swaps instead of one per 1/60 s, so game logic that ticks per vblank advances the same number of ticks per
// frame whatever the host speed: two runs (e.g. two renderers) show the same frames and can be compared pixel by pixel.
bool VblankPerSwap();
// For VblankPerSwap(): waits until the guest swapped since `last_swap` (updated) or the timeout passed (a guest that
// waits for vblanks without swapping, e.g. while loading); false if `running` turned false.
bool WaitSwapForVblank(uint64_t& last_swap, const std::atomic<bool>& running);

// Register video variable exports (VdGlobalDevice, VdHSIOCalibrationLock, etc.)
// Must be called during kernel initialization before XEX modules are loaded.
void RegisterVideoExports(rex::runtime::ExportResolver* export_resolver,
                          system::KernelState* kernel_state);

}  // namespace rex::kernel::xboxkrnl
