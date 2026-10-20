// banjo-nx (2026-10-19, latency batch 3): the frame limiter locked to the display's refresh, and the input -> display
// measurement (renut_engine/display_sync.cpp).
//
// On the Switch the compositor shows a new image only at its 60 Hz refresh (there is no tearing mode; swap interval 0
// only stops the present from waiting). The frame limiter (frame_cap) released the game's frames on its own 33.3 ms
// timer, at a phase to the refresh that slid as the two clocks drifted: an image finished just after a refresh waited
// almost 16.7 ms more, and now and then one was shown for three refreshes and the next for one. Here a thread follows
// the display's vsync event (viGetDisplayVsyncEvent): the limiter releases each frame at a refresh plus a phase, and
// the phase is found by measurement (banjo_display_phase_auto): as late as the frame's image still makes the refresh it
// made before. A frame that ends late is released at once as before (no snapping to 20 fps).
#pragma once

#include <cstdint>
#include <string>

namespace banjo::display_sync {

// Starts the thread on the first call (the Switch, banjo_display_sync); a no-op elsewhere and after the first call.
void Start();

// The next frame's release for the frame limiter, in std::chrono::steady_clock nanoseconds: the frame now ending at
// 'now', the previous one released at 'previous' (0: none), 'frame_ns' the limiter's frame time. 0 when the display's
// refresh is not known (the limiter keeps its own timer then). '*anchored': the release is a refresh plus the phase.
uint64_t NextRelease(uint64_t now, uint64_t previous, uint64_t frame_ns, bool *anchored);

// The limiter released the game's next frame (its Swap is rex::latency::SwapCount() + 1) at 'at'.
void NoteRelease(uint64_t at, bool anchored);

// The renderer presented the image of the guest's 'swap'-th Swap (after Video::Present).
void NotePresented(uint64_t swap);

// One line for the 10 s report; empty when the thread does not run.
std::string Report();

}  // namespace banjo::display_sync
