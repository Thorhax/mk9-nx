/**
 * @file    gpu/backend/gpu_profile.h
 * @brief   Where the GPU's time goes, by kind of work (renut_gpu_profile).
 *
 * With renut_gpu_profile above 0, each frame's command lists write a
 * timestamp whenever the work changes kind (draws into another target, a
 * resolve, a depth resolve, host mip levels, present); once the frame's fence
 * has passed, the spans between them are added to their kind, and every
 * renut_gpu_profile seconds the log lists the kinds by GPU milliseconds per
 * frame. Off, it records nothing. All functions run on the render thread with
 * State().mutex held.
 */
#pragma once

#include <cstdint>
#include <string_view>

namespace renut::gpu {
struct VideoState;
}

namespace renut::gpu::gpu_profile {

// The id of a kind of work, by name.
uint32_t Label(std::string_view name);

// Work recorded from here on in the open command list is 'label''s.
void Mark(VideoState &s, uint32_t label);
// banjo-nx (package 28b): a label's name, and a draw counted for the label it is drawn under (the report's draws a
// frame; banjo_renut_diag_gpu_shaders).
std::string_view Name(uint32_t label);
void CountDraw(VideoState &s, uint32_t label);

// A command list of the frame begins or ends (BeginListTiming,
// EndListTiming).
void BeginList(VideoState &s);
void EndList(VideoState &s, bool last_in_frame);

// The slot's frame has finished on the GPU (its fence was waited).
void Collect(VideoState &s, uint32_t slot);

// banjo-nx (2026-10-19, render-thread proposal 7): whether the frame being recorded is timed (set by BeginList): labels
// are worked out only for timed frames (with the per-shader breakdown, one frame in four).
extern bool g_recording_timed;
inline bool Timed() { return g_recording_timed; }

} // namespace renut::gpu::gpu_profile
