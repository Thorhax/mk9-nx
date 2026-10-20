/**
 * sa2-nx (2026-10-20): a function the input system calls at every controller state read (InputSystem::GetState),
 * before the read: Renut's late input delays the frame's first read there and its latency probe notes the read
 * (rex::latency::BeforeInputRead / NoteInputRead, app/src/renut_engine/sa2_compat/frame_latency.cpp; banjo-nx calls
 * them from its SDK directly). None by default.
 */
#pragma once

namespace rex::input {

using ReadHook = void (*)();
void SetReadHook(ReadHook hook);

}  // namespace rex::input
