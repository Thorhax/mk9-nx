// Short video recordings of the game image for testing, taken from the presenter's guest output (what the game
// presents), so they work with the window minimized and nothing can cover them. Same design and file format as the
// banjo-nx recorder: tools/make_video.sh turns the raw file into an MP4 and a contact sheet.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

namespace rex::ui {
class Presenter;
struct RawImage;
}

// Frames the game has flipped (sub_82557FC0, once per presented frame); the recorder waits on it.
extern std::atomic<uint64_t> g_sa2_game_frames;

namespace sa2::capture {

// Starts the recorder thread when sa2_video_s > 0.
void Start(std::function<rex::ui::Presenter*()> get_presenter);

// The same with another source of the presented frame (sa2-nx 2026-10-20: Renut's renderer, which has no SDK
// presenter: renut::gpu::CaptureFrame). Its Swap counts g_sa2_game_frames. No lockstep mode (that is the PM4 ring's).
void StartWithSource(std::function<bool(rex::ui::RawImage&)> grab);

// Lockstep with Renut's renderer (sa2-nx 2026-10-20; sa2_video_lockstep, sa2_video_every = 1): its Swap hook asks for
// the capture of the frame it is queueing (RenutLockstepOn), waits for that frame's image and hands it over here with
// the game frame it ends; this returns once the recorder has taken it, so no frame is skipped or taken late and two
// runs label the same frames the same, however loaded the PC is.
bool RenutLockstepOn();
void RenutLockstepFrame(uint64_t game_frame, const rex::ui::RawImage* image);

// Stops it (writes what it has, if anything).
void Stop();

// Called by the flip hook before the game flips: with sa2_video_lockstep and a recording in progress, waits until the
// recorder has taken the previous frame.
void BeforeFlip();

}  // namespace sa2::capture
