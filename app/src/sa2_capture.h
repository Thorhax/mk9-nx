// Short video recordings of the game image for testing, taken from the presenter's guest output (what the game
// presents), so they work with the window minimized and nothing can cover them. Same design and file format as the
// banjo-nx recorder: tools/make_video.sh turns the raw file into an MP4 and a contact sheet.

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>

namespace rex::ui {
class Presenter;
}

// Frames the game has flipped (sub_82557FC0, once per presented frame); the recorder waits on it.
extern std::atomic<uint64_t> g_sa2_game_frames;

namespace sa2::capture {

// Starts the recorder thread when sa2_video_s > 0.
void Start(std::function<rex::ui::Presenter*()> get_presenter);

// Stops it (writes what it has, if anything).
void Stop();

// Called by the flip hook before the game flips: with sa2_video_lockstep and a recording in progress, waits until the
// recorder has taken the previous frame.
void BeforeFlip();

}  // namespace sa2::capture
