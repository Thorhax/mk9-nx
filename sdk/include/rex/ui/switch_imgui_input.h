/**
 * @file        rex/ui/switch_imgui_input.h
 * @brief       Switch gamepad and touch screen for the SDK's ImGui menus
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <cfloat>
#include <cstdint>

#include <imgui.h>

/*
 * The SDK menus, with the gamepad and the touch screen
 *
 * On PC the menus (settings, console, achievements) are driven with mouse and keyboard. The Switch
 * has neither and nothing fed input to ImGui: the menu showed up but could not be navigated or
 * touched. This translates the gamepad and the touch panel into ImGui events.
 *
 *  - window_switch.cpp reads HID (ReadUiEntry). It is the only file that includes switch.h.
 *  - ImGuiDrawer::Draw calls ReadUiEntry and ApplyEntry on the UI thread, right before
 *    ImGui::NewFrame. While a menu is open Draw runs continuously, so the gamepad responds at the
 *    UI rate, not the game's.
 *  - ReXApp does not pass the gamepad to the game while a menu has navigation focus or a finger is
 *    on a window (GameReceivesGamepad).
 *  - The L+R+D-pad and L+R+ZL shortcuts (switch_input_driver.cpp) do not reach the menu: those
 *    buttons do not count until they are released.
 *  - L+R and the right stick move the debug overlay (F3) if it is open (debug_overlay.cpp);
 *    meanwhile the game does not see L, R or that stick.
 *
 * Buttons, as in the game's menus (input_xbox_layout, in switch_input_driver.cpp):
 *   A               accept: check, open, choose
 *   B               back: close a dropdown, leave a field
 *   X (top)         activate a text field (without a keyboard it does not type; B leaves)
 *   Y (left)        nothing (see ApplyEntry)
 *   D-pad           move
 *   left stick      scroll
 * With input_xbox_layout = true they go by position, as in 1.0.0: the bottom one (B)
 * accepts and the right one (A) goes back.
 * Touching the screen is a click.
 */

namespace rex::ui::nx {

// HidNpadButton bits from libnx. Copied to avoid including switch.h, which clashes
// with SDK types; window_switch.cpp checks with static_assert that they match.
inline constexpr uint64_t kButtonA = uint64_t(1) << 0;
inline constexpr uint64_t kButtonB = uint64_t(1) << 1;
inline constexpr uint64_t kButtonX = uint64_t(1) << 2;
inline constexpr uint64_t kButtonY = uint64_t(1) << 3;
inline constexpr uint64_t kButtonStickL = uint64_t(1) << 4;
inline constexpr uint64_t kButtonStickR = uint64_t(1) << 5;
inline constexpr uint64_t kButtonL = uint64_t(1) << 6;
inline constexpr uint64_t kButtonR = uint64_t(1) << 7;
inline constexpr uint64_t kButtonZL = uint64_t(1) << 8;
inline constexpr uint64_t kButtonZR = uint64_t(1) << 9;
inline constexpr uint64_t kButtonPlus = uint64_t(1) << 10;
inline constexpr uint64_t kButtonMinus = uint64_t(1) << 11;
inline constexpr uint64_t kButtonLeft = uint64_t(1) << 12;
inline constexpr uint64_t kButtonTop = uint64_t(1) << 13;
inline constexpr uint64_t kButtonRight = uint64_t(1) << 14;
inline constexpr uint64_t kButtonBottom = uint64_t(1) << 15;

// What was read from HID in one UI frame.
struct UiEntry {
  bool connected_gamepad = false;
  uint64_t buttons = 0;
  // As HID reports them: -32767..32767, up and right positive.
  int32_t stick_left_x = 0;
  int32_t stick_left_y = 0;
  int32_t stick_right_x = 0;
  int32_t stick_right_y = 0;
  bool touching = false;
  // Touched point, already in ImGui logical coordinates.
  float touch_x = 0.0f;
  float touch_y = 0.0f;
  // input_xbox_layout: face buttons by position instead of by letter.
  bool per_position = false;
};

// What must be remembered from one frame to the next.
struct StateUiEntry {
  bool touched = false;
  // Buttons of an L+R+... shortcut that are still held.
  uint64_t of_shortcut = 0;
};

// In window_switch.cpp; UI thread only. The factors convert from the touch panel
// (1280x720) to ImGui logical coordinates.
void ReadUiEntry(UiEntry& output, float touch_to_logical_x, float touch_to_logical_y);

inline void ConfigureNavigation(ImGuiIO& io) {
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
  // A accepts, as in the game; ApplyEntry sets it every frame from input_xbox_layout.
  io.ConfigNavSwapGamepadButtons = true;
}

namespace detail {

// One direction of a stick axis, 0..1, with a dead zone.
inline float Sense(int32_t value, bool positive) {
  constexpr float kMax = 32767.0f;
  constexpr float kDeadZone = 0.25f;
  float v = float(value) / kMax;
  if (!positive) {
    v = -v;
  }
  if (v <= kDeadZone) {
    return 0.0f;
  }
  v = (v - kDeadZone) / (1.0f - kDeadZone);
  return v > 1.0f ? 1.0f : v;
}

inline void Stick(ImGuiIO& io, ImGuiKey top, ImGuiKey bottom, ImGuiKey left,
                  ImGuiKey right, int32_t x, int32_t y) {
  const float a = Sense(y, true);
  const float b = Sense(y, false);
  const float i = Sense(x, false);
  const float d = Sense(x, true);
  io.AddKeyAnalogEvent(top, a > 0.0f, a);
  io.AddKeyAnalogEvent(bottom, b > 0.0f, b);
  io.AddKeyAnalogEvent(left, i > 0.0f, i);
  io.AddKeyAnalogEvent(right, d > 0.0f, d);
}

}  // namespace detail

// Translates a HID reading into ImGui events. ImGui discards events that repeat
// the previous state, so it can be called every frame.
inline void ApplyEntry(ImGuiIO& io, const UiEntry& e, StateUiEntry& state) {
  if (e.connected_gamepad) {
    io.BackendFlags |= ImGuiBackendFlags_HasGamepad;
  } else {
    io.BackendFlags &= ~ImGuiBackendFlags_HasGamepad;
  }
  // ImGui reads the buttons by position: it accepts with the bottom one (B on the Switch) and goes
  // back with the right one (A). Swapped, A accepts and B goes back, as in the game with the mapping
  // by letter.
  io.ConfigNavSwapGamepadButtons = !e.per_position;

  const uint64_t read = e.connected_gamepad ? e.buttons : 0;
  // With L and R held, the D-pad and ZL are input driver shortcuts: they open and close menus or
  // start the A/B tests. The driver only sees them when the game polls the gamepad, and the menu
  // changes later, on the UI thread; meanwhile the held D-pad would repeat here and move the focus.
  // They stay excluded even if L or R is released before them.
  constexpr uint64_t kModifier = kButtonL | kButtonR;
  constexpr uint64_t kOfShortcut =
      kButtonTop | kButtonBottom | kButtonLeft | kButtonRight | kButtonZL;
  if ((read & kModifier) == kModifier) {
    state.of_shortcut |= read & kOfShortcut;
  }
  state.of_shortcut &= read;
  const uint64_t b = read & ~state.of_shortcut;

  // Y is not passed to ImGui. The left face button is ImGui's menu button: held, it opens the window
  // selector, which on the console moved the F3 overlay with L/R and the stick and took focus away
  // from Settings.
  struct Button {
    uint64_t bit;
    ImGuiKey keystroke;
  };
  static constexpr Button kButtons[] = {
      {kButtonB, ImGuiKey_GamepadFaceDown},         {kButtonA, ImGuiKey_GamepadFaceRight},
      {kButtonX, ImGuiKey_GamepadFaceUp},           {kButtonTop, ImGuiKey_GamepadDpadUp},
      {kButtonBottom, ImGuiKey_GamepadDpadDown},     {kButtonLeft, ImGuiKey_GamepadDpadLeft},
      {kButtonRight, ImGuiKey_GamepadDpadRight},  {kButtonL, ImGuiKey_GamepadL1},
      {kButtonR, ImGuiKey_GamepadR1},               {kButtonStickL, ImGuiKey_GamepadL3},
      {kButtonStickR, ImGuiKey_GamepadR3},          {kButtonPlus, ImGuiKey_GamepadStart},
      {kButtonMinus, ImGuiKey_GamepadBack},
  };
  for (const Button& button : kButtons) {
    io.AddKeyEvent(button.keystroke, (b & button.bit) != 0);
  }
  // ZL and ZR are digital on Switch.
  const bool zl = (b & kButtonZL) != 0;
  const bool zr = (b & kButtonZR) != 0;
  io.AddKeyAnalogEvent(ImGuiKey_GamepadL2, zl, zl ? 1.0f : 0.0f);
  io.AddKeyAnalogEvent(ImGuiKey_GamepadR2, zr, zr ? 1.0f : 0.0f);
  detail::Stick(io, ImGuiKey_GamepadLStickUp, ImGuiKey_GamepadLStickDown,
                 ImGuiKey_GamepadLStickLeft, ImGuiKey_GamepadLStickRight,
                 e.connected_gamepad ? e.stick_left_x : 0, e.connected_gamepad ? e.stick_left_y : 0);
  detail::Stick(io, ImGuiKey_GamepadRStickUp, ImGuiKey_GamepadRStickDown,
                 ImGuiKey_GamepadRStickLeft, ImGuiKey_GamepadRStickRight,
                 e.connected_gamepad ? e.stick_right_x : 0, e.connected_gamepad ? e.stick_right_y : 0);

  // The touch panel as a mouse: touching presses and lifting releases at the same point. ImGui
  // defers to another frame any movement that arrives after the release, so the click lands where
  // the finger was before the pointer is removed, and nothing stays highlighted without a finger.
  if (e.touching) {
    io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
    io.AddMousePosEvent(e.touch_x, e.touch_y);
    if (!state.touched) {
      io.AddMouseButtonEvent(ImGuiMouseButton_Left, true);
    }
  } else if (state.touched) {
    io.AddMouseSourceEvent(ImGuiMouseSource_TouchScreen);
    io.AddMouseButtonEvent(ImGuiMouseButton_Left, false);
    io.AddMousePosEvent(-FLT_MAX, -FLT_MAX);
  }
  state.touched = e.touching;
}

// The gamepad belongs to the menu, not the game, while a window has navigation focus or a
// finger is on it. Windows that only display data (the debug overlay) carry NoNavInputs and
// do not take it.
inline bool GameReceivesGamepad(const ImGuiIO& io) {
  return !io.WantCaptureMouse && !io.NavActive;
}

}  // namespace rex::ui::nx
