// Scripted gamepad for automated tests: a synthetic player-1 device that presses buttons at given times after start
// (cvars mk9_input_script / mk9_input_script_file). The SDK puts synthetic devices on player 1 and merges them with
// any real pad, so a script and a controller can be used together.

#pragma once

#include <memory>

#include <rex/system/interfaces/input.h>

namespace mk9::script_input {

// The SDK's default input system (SDL pads, keyboard, the NOP stand-in) plus the scripted pad when a script is set.
std::unique_ptr<rex::system::IInputSystem> CreateInputSystem(bool tool_mode);

}  // namespace mk9::script_input
