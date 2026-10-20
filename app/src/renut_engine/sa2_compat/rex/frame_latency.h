/**
 * Frame latency probe (banjo-nx 2026-10-19, from futari-nx's sdk/include/rex/frame_latency.h, the user's own port):
 * how long a frame takes from the moment the game reads its controller to the moment it is presented. Three moments
 * per frame, all on one steady clock:
 *
 *   input     the game's first controller read after its previous Swap (InputSystem::GetState)
 *   swap      the game's Swap that ends the frame
 *   present   the renderer handed that Swap's image to the display
 *
 * Every function is cheap (a few relaxed atomics) and safe from any thread. Report() gives one line of means and
 * maxima since the last report, for the renderer's 10 s report.
 *
 * Late input (banjo-nx): SetFrameReadDelayFunction sets a function the input system calls at the first controller
 * read after each Swap; it returns microseconds to sleep before reading (0: none).
 */
#pragma once

#include <cstdint>
#include <string>

namespace rex::latency {

// The game read a controller. Only the first read after a Swap counts (the frame's input).
void NoteInputRead();
// The game's Swap: the frame that began with the last NoteInputRead is complete on the CPU side.
void NoteGuestSwap();
// The renderer presented the image of the guest's `swap`-th Swap (1-based, counted the same way as the Swaps).
void NotePresented(uint64_t swap);
// banjo-nx (2026-10-20): the GPU has finished the image of the guest's `swap`-th Swap (seen now): the report's "input
// -> GPU done", the earliest the display can show the frame (the present above is when it was handed over).
void NoteGpuDone(uint64_t swap);
// The Swaps so far.
uint64_t SwapCount();
// The time (std::chrono::steady_clock nanoseconds) of the first controller read of the frame the 'swap'-th Swap ends;
// 0 when not known (banjo-nx 2026-10-19, latency batch 3: the input -> display measurement).
uint64_t InputTimeOf(uint64_t swap);
// One line: "input -> swap a ms, swap -> present b ms, input -> present c ms; n frames". Empty when no frame
// completed.
std::string Report();

// Late input: the delay before the frame's first controller read (see above).
using FrameReadDelayFunction = uint32_t (*)();
void SetFrameReadDelayFunction(FrameReadDelayFunction fn);
// Called by the input system before every read: sleeps the delay at the first read after a Swap.
void BeforeInputRead();

}  // namespace rex::latency
