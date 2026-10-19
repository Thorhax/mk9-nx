/**
 * @file        input/switch/switch_input_driver.cpp
 * @brief       Input driver for Nintendo Switch pads through libnx HID
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <rex/input/switch/switch_input_driver.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <tuple>

#include <rex/chrono/clock.h>
#include <rex/cvar.h>
#include <rex/logging.h>
#include <rex/ui/keybinds.h>
#include <rex/ui/ui_event.h>
#include <rex/ui/virtual_key.h>

#include <switch.h>

// Face buttons. By position (the default since 2026-10-07, the user's choice): as on the Xbox 360 pad the game was
// made for, the bottom button (B on Switch controllers) acts as A, the right one (A) as B, the left one (Y) as X and
// the top one (X) as Y. false = by letter (the Switch's A acts as A), as in other Switch games. It applies at once;
// the settings menu reads it too (window_switch.cpp). input_p1_* / input_p2_* below override single buttons.
REXCVAR_DEFINE_BOOL(input_xbox_layout, true, "Input",
                    "Face buttons by position, as on an Xbox 360 pad (the bottom button, B, acts as A). "
                    "false = by letter: the Switch's A acts as A");

// Remapping, per player: which Switch buttons press each Xbox 360 button, read at start (sa2.toml sections
// [input_p1] and [input_p2]: a = "B, ZR" there is input_p1_a). Empty = the default (the layout above for A/B/X/Y; for
// the rest, the button in the same place). A comma-separated list of: A B X Y L R ZL ZR PLUS MINUS LSTICK RSTICK UP
// DOWN LEFT RIGHT SL SR, or NONE. Player 1's map also serves players 3 and 4. Single Joy-Cons held sideways name their
// four face buttons by the place they take there (the right one is A, the bottom one B, the top one X, the left one Y),
// as two-handed controllers do.
#define REX_INPUT_MAP_CVARS(P, WHO)                                                                                 \
  REXCVAR_DEFINE_STRING(input_##P##_a, "", "Input", WHO ": Switch buttons for the Xbox A button (empty = default)"); \
  REXCVAR_DEFINE_STRING(input_##P##_b, "", "Input", WHO ": Switch buttons for the Xbox B button");                  \
  REXCVAR_DEFINE_STRING(input_##P##_x, "", "Input", WHO ": Switch buttons for the Xbox X button");                  \
  REXCVAR_DEFINE_STRING(input_##P##_y, "", "Input", WHO ": Switch buttons for the Xbox Y button");                  \
  REXCVAR_DEFINE_STRING(input_##P##_lb, "", "Input", WHO ": Switch buttons for the left bumper (default L)");       \
  REXCVAR_DEFINE_STRING(input_##P##_rb, "", "Input", WHO ": Switch buttons for the right bumper (default R)");      \
  REXCVAR_DEFINE_STRING(input_##P##_lt, "", "Input", WHO ": Switch buttons for the left trigger (default ZL)");     \
  REXCVAR_DEFINE_STRING(input_##P##_rt, "", "Input", WHO ": Switch buttons for the right trigger (default ZR)");    \
  REXCVAR_DEFINE_STRING(input_##P##_start, "", "Input", WHO ": Switch buttons for Start (default PLUS)");           \
  REXCVAR_DEFINE_STRING(input_##P##_back, "", "Input", WHO ": Switch buttons for Back (default MINUS)");            \
  REXCVAR_DEFINE_STRING(input_##P##_ls, "", "Input", WHO ": Switch buttons for the left stick click (LSTICK)");     \
  REXCVAR_DEFINE_STRING(input_##P##_rs, "", "Input", WHO ": Switch buttons for the right stick click (RSTICK)");    \
  REXCVAR_DEFINE_STRING(input_##P##_up, "", "Input", WHO ": Switch buttons for D-pad up (default UP)");             \
  REXCVAR_DEFINE_STRING(input_##P##_down, "", "Input", WHO ": Switch buttons for D-pad down (default DOWN)");       \
  REXCVAR_DEFINE_STRING(input_##P##_left, "", "Input", WHO ": Switch buttons for D-pad left (default LEFT)");       \
  REXCVAR_DEFINE_STRING(input_##P##_right, "", "Input", WHO ": Switch buttons for D-pad right (default RIGHT)")
REX_INPUT_MAP_CVARS(p1, "Player 1");
REX_INPUT_MAP_CVARS(p2, "Player 2");
#undef REX_INPUT_MAP_CVARS

// Single Joy-Cons (one per player, for multiplayer) are held sideways: the system is told so (it then turns the
// stick for that grip), and the four buttons under the player's right thumb become the face buttons. SL and SR are
// the triggers (ZL / ZR: the camera in Sonic Adventure 2), the minus or plus button is Start, the stick is the left
// stick. false = held upright (the system's default; no remapping).
REXCVAR_DEFINE_BOOL(input_joycon_sideways, true, "Input",
                    "Single Joy-Cons are held sideways, one per player (face buttons, stick and SL/SR turned "
                    "accordingly)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// Route recording for unattended test runs: the game's own pad input, written and played back here.
// MINUS on pad 1 marks the start (and, while recording, the end), so the load time before the route does not
// matter; the game does not see MINUS while either is set. Lines: "ms<TAB>npad buttons (hex)<TAB>lx ly rx ry".
REXCVAR_DEFINE_STRING(input_record, "", "Input",
                      "Record pad 1 to this file (e.g. sdmc:/switch/masseffect-nx/route.tsv) between two MINUS presses");
REXCVAR_DEFINE_STRING(input_play, "", "Input",
                      "Play this recorded file (input_record) back as pad 1, starting at the next MINUS press");

namespace rex::ui {
// rex/ui/overlay/debug_overlay.h (not included here because it pulls in ImGui).
bool DebugOverlayOpen();
}  // namespace rex::ui

namespace rex::input::nx {

namespace {

constexpr int16_t kThumbThreshold = 0x4E00;
constexpr uint8_t kTriggerThreshold = 0x1F;
constexpr uint32_t kRepeatDelayMs = 400;
constexpr uint32_t kRepeatRateMs = 100;

// Low byte of a DeviceId is the slot, the upper half a per-run connection
// counter, so a pad that reconnects never gets an id it had before.
constexpr uint64_t kDeviceIdTag = 0x4E5800;  // "NX"

// Which Switch buttons press each Xbox 360 button: per player (1, and 2; players 3-4 use player 1's), one table per
// face-button layout (input_xbox_layout), that player's overrides (input_p1_* / input_p2_*) applied to both. Built
// once at Setup.
struct ButtonTable {
  std::array<uint64_t, 16> buttons{};  // per XInput button bit (X_INPUT_GAMEPAD_*: bit i)
  uint64_t left_trigger = 0, right_trigger = 0;
};
std::array<std::array<ButtonTable, 2>, 2> g_tables;  // [player 1 / 2][0 by letter, 1 by position]

constexpr int BitIndex(uint16_t bit) { return std::countr_zero(bit); }

// One name of input_p1_* / input_p2_*: the Switch buttons it stands for (0 for NONE), false when unknown.
bool SwitchButtonByName(std::string name, uint64_t& out) {
  for (char& c : name) c = char(std::toupper(static_cast<unsigned char>(c)));
  static const std::pair<const char*, uint64_t> kNames[] = {
      {"A", HidNpadButton_A},          {"B", HidNpadButton_B},          {"X", HidNpadButton_X},
      {"Y", HidNpadButton_Y},          {"L", HidNpadButton_L},          {"R", HidNpadButton_R},
      {"ZL", HidNpadButton_ZL},        {"ZR", HidNpadButton_ZR},        {"PLUS", HidNpadButton_Plus},
      {"+", HidNpadButton_Plus},       {"MINUS", HidNpadButton_Minus},  {"-", HidNpadButton_Minus},
      {"LSTICK", HidNpadButton_StickL}, {"L3", HidNpadButton_StickL},  {"RSTICK", HidNpadButton_StickR},
      {"R3", HidNpadButton_StickR},    {"UP", HidNpadButton_Up},        {"DOWN", HidNpadButton_Down},
      {"LEFT", HidNpadButton_Left},    {"RIGHT", HidNpadButton_Right},  {"SL", HidNpadButton_AnySL},
      {"SR", HidNpadButton_AnySR},     {"NONE", 0},
  };
  for (const auto& [n, mask] : kNames)
    if (name == n) {
      out = mask;
      return true;
    }
  return false;
}

// "A, ZR" -> their buttons; false (and logged) when a name is unknown, then the default stays.
bool ParseButtonList(const char* cvar, const std::string& list, uint64_t& out) {
  uint64_t mask = 0;
  std::string token;
  auto flush = [&]() {
    if (token.empty()) return true;
    uint64_t m = 0;
    if (!SwitchButtonByName(token, m)) {
      REXLOG_WARN("{}: unknown Switch button \"{}\" (A B X Y L R ZL ZR PLUS MINUS LSTICK RSTICK UP DOWN LEFT RIGHT "
                  "SL SR NONE); the default is kept",
                  cvar, token);
      return false;
    }
    mask |= m;
    token.clear();
    return true;
  };
  for (char c : list) {
    if (c == ',' || c == ';' || c == '|' || c == ' ' || c == '\t') {
      if (!flush()) return false;
    } else {
      token += c;
    }
  }
  if (!flush()) return false;
  out = mask;
  return true;
}

void BuildButtonTables() {
  for (auto& player : g_tables) {
    for (int layout = 0; layout < 2; ++layout) {
      ButtonTable& t = player[layout];
      t = {};
      const bool by_position = layout == 1;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_A)] = by_position ? HidNpadButton_B : HidNpadButton_A;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_B)] = by_position ? HidNpadButton_A : HidNpadButton_B;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_X)] = by_position ? HidNpadButton_Y : HidNpadButton_X;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_Y)] = by_position ? HidNpadButton_X : HidNpadButton_Y;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_DPAD_UP)] = HidNpadButton_Up;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_DPAD_DOWN)] = HidNpadButton_Down;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_DPAD_LEFT)] = HidNpadButton_Left;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_DPAD_RIGHT)] = HidNpadButton_Right;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_START)] = HidNpadButton_Plus;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_BACK)] = HidNpadButton_Minus;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_LEFT_SHOULDER)] = HidNpadButton_L;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_RIGHT_SHOULDER)] = HidNpadButton_R;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_LEFT_THUMB)] = HidNpadButton_StickL;
      t.buttons[BitIndex(X_INPUT_GAMEPAD_RIGHT_THUMB)] = HidNpadButton_StickR;
      t.left_trigger = HidNpadButton_ZL;
      t.right_trigger = HidNpadButton_ZR;
    }
  }
  // The overrides: the same in both layouts, since they name Switch buttons.
  struct Override {
    const char* name;
    std::string value;
    int bit;  // XInput bit index; -1 / -2 = left / right trigger
  };
#define REX_INPUT_OVERRIDES(P)                                                                      \
  {                                                                                                 \
    {"input_" #P "_a", REXCVAR_GET(input_##P##_a), BitIndex(X_INPUT_GAMEPAD_A)},                  \
    {"input_" #P "_b", REXCVAR_GET(input_##P##_b), BitIndex(X_INPUT_GAMEPAD_B)},                  \
    {"input_" #P "_x", REXCVAR_GET(input_##P##_x), BitIndex(X_INPUT_GAMEPAD_X)},                  \
    {"input_" #P "_y", REXCVAR_GET(input_##P##_y), BitIndex(X_INPUT_GAMEPAD_Y)},                  \
    {"input_" #P "_lb", REXCVAR_GET(input_##P##_lb), BitIndex(X_INPUT_GAMEPAD_LEFT_SHOULDER)},    \
    {"input_" #P "_rb", REXCVAR_GET(input_##P##_rb), BitIndex(X_INPUT_GAMEPAD_RIGHT_SHOULDER)},   \
    {"input_" #P "_lt", REXCVAR_GET(input_##P##_lt), -1},                                         \
    {"input_" #P "_rt", REXCVAR_GET(input_##P##_rt), -2},                                         \
    {"input_" #P "_start", REXCVAR_GET(input_##P##_start), BitIndex(X_INPUT_GAMEPAD_START)},      \
    {"input_" #P "_back", REXCVAR_GET(input_##P##_back), BitIndex(X_INPUT_GAMEPAD_BACK)},         \
    {"input_" #P "_ls", REXCVAR_GET(input_##P##_ls), BitIndex(X_INPUT_GAMEPAD_LEFT_THUMB)},       \
    {"input_" #P "_rs", REXCVAR_GET(input_##P##_rs), BitIndex(X_INPUT_GAMEPAD_RIGHT_THUMB)},      \
    {"input_" #P "_up", REXCVAR_GET(input_##P##_up), BitIndex(X_INPUT_GAMEPAD_DPAD_UP)},          \
    {"input_" #P "_down", REXCVAR_GET(input_##P##_down), BitIndex(X_INPUT_GAMEPAD_DPAD_DOWN)},    \
    {"input_" #P "_left", REXCVAR_GET(input_##P##_left), BitIndex(X_INPUT_GAMEPAD_DPAD_LEFT)},    \
    {"input_" #P "_right", REXCVAR_GET(input_##P##_right), BitIndex(X_INPUT_GAMEPAD_DPAD_RIGHT)}, \
  }
  const Override overrides[2][16] = {REX_INPUT_OVERRIDES(p1), REX_INPUT_OVERRIDES(p2)};
#undef REX_INPUT_OVERRIDES
  for (size_t player = 0; player < g_tables.size(); ++player) {
    for (const Override& o : overrides[player]) {
      if (o.value.empty()) continue;
      uint64_t mask = 0;
      if (!ParseButtonList(o.name, o.value, mask)) continue;
      for (ButtonTable& t : g_tables[player]) {
        if (o.bit == -1) t.left_trigger = mask;
        else if (o.bit == -2) t.right_trigger = mask;
        else t.buttons[o.bit] = mask;
      }
      REXLOG_INFO("{} = {}", o.name, o.value);
    }
  }
}

// The presses of a single Joy-Con held sideways, as a two-handed controller reports the same places (the right face
// button is A, the bottom one B, the top one X, the left one Y). The system names a single Joy-Con's buttons as they
// are printed; held sideways the left Joy-Con's arrow buttons and the right Joy-Con's letters turn a quarter: same
// table as devkitPro's SDL (pad_mapping_left_joy / pad_mapping_right_joy). SL / SR are the triggers (the camera in
// Sonic Adventure 2) and stay readable as SL / SR for input_p1_* / input_p2_*; minus or plus is Start; the stick
// click is LSTICK.
uint64_t SidewaysLeftJoyCon(uint64_t p) {
  uint64_t v = 0;
  if (p & HidNpadButton_Down) v |= HidNpadButton_A;
  if (p & HidNpadButton_Left) v |= HidNpadButton_B;
  if (p & HidNpadButton_Right) v |= HidNpadButton_X;
  if (p & HidNpadButton_Up) v |= HidNpadButton_Y;
  if (p & HidNpadButton_LeftSL) v |= HidNpadButton_ZL | HidNpadButton_LeftSL;
  if (p & HidNpadButton_LeftSR) v |= HidNpadButton_ZR | HidNpadButton_LeftSR;
  if (p & HidNpadButton_Minus) v |= HidNpadButton_Plus;
  if (p & HidNpadButton_StickL) v |= HidNpadButton_StickL;
  return v;
}
uint64_t SidewaysRightJoyCon(uint64_t p) {
  uint64_t v = 0;
  if (p & HidNpadButton_X) v |= HidNpadButton_A;
  if (p & HidNpadButton_A) v |= HidNpadButton_B;
  if (p & HidNpadButton_Y) v |= HidNpadButton_X;
  if (p & HidNpadButton_B) v |= HidNpadButton_Y;
  if (p & HidNpadButton_RightSL) v |= HidNpadButton_ZL | HidNpadButton_RightSL;
  if (p & HidNpadButton_RightSR) v |= HidNpadButton_ZR | HidNpadButton_RightSR;
  if (p & HidNpadButton_Plus) v |= HidNpadButton_Plus;
  if (p & HidNpadButton_StickR) v |= HidNpadButton_StickL;
  return v;
}

// The order of this list is also the order in which events are sent if
// multiple buttons change at once. Same table as the SDL driver.
constexpr std::array<rex::ui::VirtualKey, 34> kVkLookup = {
    // 00 - True buttons from xinput button field
    rex::ui::VirtualKey::kXInputPadDpadUp,
    rex::ui::VirtualKey::kXInputPadDpadDown,
    rex::ui::VirtualKey::kXInputPadDpadLeft,
    rex::ui::VirtualKey::kXInputPadDpadRight,
    rex::ui::VirtualKey::kXInputPadStart,
    rex::ui::VirtualKey::kXInputPadBack,
    rex::ui::VirtualKey::kXInputPadLThumbPress,
    rex::ui::VirtualKey::kXInputPadRThumbPress,
    rex::ui::VirtualKey::kXInputPadLShoulder,
    rex::ui::VirtualKey::kXInputPadRShoulder,
    rex::ui::VirtualKey::kNone, /* Guide has no VK */
    rex::ui::VirtualKey::kNone, /* Unknown */
    rex::ui::VirtualKey::kXInputPadA,
    rex::ui::VirtualKey::kXInputPadB,
    rex::ui::VirtualKey::kXInputPadX,
    rex::ui::VirtualKey::kXInputPadY,
    // 16 - Fake buttons generated from analog inputs
    rex::ui::VirtualKey::kXInputPadLTrigger,
    rex::ui::VirtualKey::kXInputPadRTrigger,
    // 18
    rex::ui::VirtualKey::kXInputPadLThumbUp,
    rex::ui::VirtualKey::kXInputPadLThumbDown,
    rex::ui::VirtualKey::kXInputPadLThumbRight,
    rex::ui::VirtualKey::kXInputPadLThumbLeft,
    rex::ui::VirtualKey::kXInputPadLThumbUpLeft,
    rex::ui::VirtualKey::kXInputPadLThumbUpRight,
    rex::ui::VirtualKey::kXInputPadLThumbDownRight,
    rex::ui::VirtualKey::kXInputPadLThumbDownLeft,
    // 26
    rex::ui::VirtualKey::kXInputPadRThumbUp,
    rex::ui::VirtualKey::kXInputPadRThumbDown,
    rex::ui::VirtualKey::kXInputPadRThumbRight,
    rex::ui::VirtualKey::kXInputPadRThumbLeft,
    rex::ui::VirtualKey::kXInputPadRThumbUpLeft,
    rex::ui::VirtualKey::kXInputPadRThumbUpRight,
    rex::ui::VirtualKey::kXInputPadRThumbDownRight,
    rex::ui::VirtualKey::kXInputPadRThumbDownLeft,
};

enum class RepeatState {
  kIdle,       // no buttons pressed or repeating has ended
  kWaiting,    // a button is held and the delay is awaited
  kRepeating,  // actively repeating at a rate
};

int16_t ClampStickAxis(s32 value) {
  return static_cast<int16_t>(std::clamp<s32>(value, INT16_MIN, INT16_MAX));
}

// Check if the analog inputs exceed their thresholds to become a button press
// and build the bitfield.
uint64_t AnalogToKeyfield(const X_INPUT_GAMEPAD& gamepad) {
  uint64_t f = 0;

  f |= static_cast<uint64_t>(gamepad.left_trigger > kTriggerThreshold) << 16;
  f |= static_cast<uint64_t>(gamepad.right_trigger > kTriggerThreshold) << 17;

  auto thumb_x = static_cast<int16_t>(gamepad.thumb_lx);
  auto thumb_y = static_cast<int16_t>(gamepad.thumb_ly);
  for (size_t i = 0; i <= 8; i = i + 8) {
    uint64_t u = thumb_y > kThumbThreshold;
    uint64_t d = thumb_y < ~kThumbThreshold;
    uint64_t r = thumb_x > kThumbThreshold;
    uint64_t l = thumb_x < ~kThumbThreshold;
    if (u && l) {
      u = l = 0;
      f |= uint64_t(1) << (22 + i);
    }
    if (u && r) {
      u = r = 0;
      f |= uint64_t(1) << (23 + i);
    }
    if (d && r) {
      d = r = 0;
      f |= uint64_t(1) << (24 + i);
    }
    if (d && l) {
      d = l = 0;
      f |= uint64_t(1) << (25 + i);
    }
    f |= u << (18 + i);
    f |= d << (19 + i);
    f |= r << (20 + i);
    f |= l << (21 + i);

    thumb_x = static_cast<int16_t>(gamepad.thumb_rx);
    thumb_y = static_cast<int16_t>(gamepad.thumb_ry);
  }
  return f;
}

}  // namespace

struct SwitchInputDriver::Slot {
  HidNpadIdType npad_id = HidNpadIdType_No1;
  PadState pad{};
  bool connected = false;
  DeviceId id = DeviceId::kInvalid;
  // The guest player this controller is (0 = player 1), as the input system assigned it (OnDevicesAssigned): picks
  // the button table (player 2's map for player 2, player 1's for the others).
  uint32_t ordinal = 0;

  X_INPUT_STATE state{};
  bool state_changed = false;
  bool is_active = true;

  /*
   * One bit per menu shortcut, so it fires only on press and not while held.
   * See DispatchMenuShortcuts.
   */
  uint64_t menu_shortcuts = 0;

  // Keystroke synthesis, per pad rather than per guest user.
  uint64_t key_buttons = 0;
  RepeatState repeat_state = RepeatState::kIdle;
  uint8_t repeat_index = 0;
  uint32_t repeat_time = 0;

  // Vibration devices are initialized for one controller layout; redone only
  // when the attached layout changes.
  HidNpadIdType vibration_npad_id = HidNpadIdType_No1;
  uint32_t vibration_style = 0;
  int vibration_count = 0;
  std::array<HidVibrationDeviceHandle, 2> vibration_handles{};
  // The last value sent to the controller. Stack sampling put 5.8 % of the thread that prepares
  // each frame inside SetState: the game calls XInputSetState very often, and every call ended
  // in hidSendVibrationValues, an IPC request that blocks the thread. With vibration off
  // (input_rumble = false) it kept sending "zero" over and over. If the requested value is
  // the same as the controller already has, nothing is sent: the controller is already in that
  // state.
  bool vibration_sent = false;
  uint16_t vibration_left = 0;
  uint16_t vibration_right = 0;
};

SwitchInputDriver::SwitchInputDriver(rex::ui::Window* window, size_t window_z_order)
    : InputDriver(window, window_z_order) {}

SwitchInputDriver::~SwitchInputDriver() {
  if (!initialized_) {
    return;
  }
  // XInput vibration holds until changed, so a pad left rumbling would keep
  // rumbling after the process exits.
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& slot : slots_) {
    if (slot && slot->vibration_count) {
      std::array<HidVibrationValue, 2> stop{};
      for (auto& value : stop) {
        value.freq_low = 160.0f;
        value.freq_high = 320.0f;
      }
      hidSendVibrationValues(slot->vibration_handles.data(), stop.data(), slot->vibration_count);
    }
  }
}

X_STATUS SwitchInputDriver::Setup() {
  // Reference counted in libnx. MarathonRecomp-NX found that without it the
  // first padUpdate can abort with 2345-0008 when HID was not brought up by
  // __appInit.
  Result rc = hidInitialize();
  if (R_FAILED(rc)) {
    REXLOG_ERROR("SwitchInputDriver: hidInitialize failed: 0x{:08X}", rc);
    return X_STATUS_UNSUCCESSFUL;
  }
  padConfigureInput(kPlayerCount, HidNpadStyleSet_NpadStandard);
  // Single Joy-Cons are held sideways (input_joycon_sideways): the system then turns their stick for that grip.
  if (REXCVAR_GET(input_joycon_sideways)) {
    hidSetNpadJoyHoldType(HidNpadJoyHoldType_Horizontal);
  }

  // Slot 0 the handheld, slots 1-4 the system's players No1-No4 (see the header).
  static constexpr std::array<HidNpadIdType, kSlotCount> kNpadIds = {
      HidNpadIdType_Handheld, HidNpadIdType_No1, HidNpadIdType_No2, HidNpadIdType_No3, HidNpadIdType_No4};
  for (size_t i = 0; i < kSlotCount; ++i) {
    auto slot = std::make_unique<Slot>();
    slot->npad_id = kNpadIds[i];
    padInitialize(&slot->pad, kNpadIds[i]);
    slots_[i] = std::move(slot);
  }
  BuildButtonTables();
  hidPermitVibration(true);
  initialized_ = true;
  return X_STATUS_SUCCESS;
}

namespace {
struct RouteSample {
  uint32_t ms;
  uint64_t buttons;
  int32_t lx, ly, rx, ry;
};
struct RouteState {
  bool configured = false;
  std::string record_path, play_path;
  FILE* record = nullptr;
  bool recording = false, playing = false, minus_before = false;
  uint64_t start_tick = 0, last_flush_tick = 0;
  RouteSample last{};
  std::vector<RouteSample> samples;
  size_t next = 0;
  RouteSample current{};
};
RouteState route_;
uint32_t RouteMs(uint64_t start) { return uint32_t(armTicksToNs(armGetSystemTick() - start) / 1000000); }
}  // namespace

// input_record / input_play (see the cvars above). Pad 1 only, on the polling thread.
static void RouteRecording(uint64_t& held, HidAnalogStickState& left, HidAnalogStickState& right) {
  RouteState& r = route_;
  if (!r.configured) {
    r.configured = true;
    r.record_path = REXCVAR_GET(input_record);
    r.play_path = REXCVAR_GET(input_play);
    if (!r.play_path.empty()) {
      if (FILE* f = std::fopen(r.play_path.c_str(), "r")) {
        RouteSample m{};
        unsigned long long buttons = 0;
        while (std::fscanf(f, "%u\t%llx\t%d %d %d %d", &m.ms, &buttons, &m.lx, &m.ly, &m.rx, &m.ry) == 6) {
          m.buttons = buttons;
          r.samples.push_back(m);
        }
        std::fclose(f);
      }
      REXLOG_INFO("input_play: {} samples from {}", r.samples.size(), r.play_path);
    }
  }
  if (r.record_path.empty() && r.play_path.empty()) {
    return;
  }
  const bool minus = (held & HidNpadButton_Minus) != 0;
  const bool minus_pressed = minus && !r.minus_before;
  r.minus_before = minus;
  held &= ~uint64_t(HidNpadButton_Minus);  // the marker is not game input
  if (!r.record_path.empty()) {
    if (minus_pressed && !r.recording) {
      r.record = std::fopen(r.record_path.c_str(), "w");
      r.recording = r.record != nullptr;
      r.start_tick = r.last_flush_tick = armGetSystemTick();
      r.last = RouteSample{~0u, ~0ull, 0, 0, 0, 0};
      REXLOG_INFO("input_record: recording to {} ({})", r.record_path, r.recording ? "open" : "FAILED");
    } else if (minus_pressed && r.recording) {
      std::fclose(r.record);
      r.record = nullptr;
      r.recording = false;
      REXLOG_INFO("input_record: recording closed");
    }
    if (r.recording) {
      const RouteSample m{RouteMs(r.start_tick), held, left.x, left.y, right.x, right.y};
      if (m.buttons != r.last.buttons || m.lx != r.last.lx || m.ly != r.last.ly || m.rx != r.last.rx ||
          m.ry != r.last.ry) {
        std::fprintf(r.record, "%u\t%llx\t%d %d %d %d\n", m.ms, static_cast<unsigned long long>(m.buttons), m.lx,
                     m.ly, m.rx, m.ry);
        r.last = m;
      }
      if (armTicksToNs(armGetSystemTick() - r.last_flush_tick) > 1000000000ull) {
        std::fflush(r.record);
        r.last_flush_tick = armGetSystemTick();
      }
    }
  }
  if (!r.play_path.empty() && !r.samples.empty()) {
    if (minus_pressed && !r.playing) {
      r.playing = true;
      r.next = 0;
      r.current = RouteSample{};
      r.start_tick = armGetSystemTick();
      REXLOG_INFO("input_play: playback started");
    }
    if (r.playing) {
      const uint32_t now = RouteMs(r.start_tick);
      while (r.next < r.samples.size() && r.samples[r.next].ms <= now) {
        r.current = r.samples[r.next++];
      }
      if (r.next >= r.samples.size() && now > r.samples.back().ms + 500) {
        r.playing = false;
        r.current = RouteSample{};
        REXLOG_INFO("input_play: playback finished");
      }
      // The recording replaces the pad completely (sys-botbase's own presses included).
      held = r.current.buttons;
      left.x = r.current.lx;
      left.y = r.current.ly;
      right.x = r.current.rx;
      right.y = r.current.ry;
    }
  }
}

void SwitchInputDriver::Poll(size_t index) {
  Slot& slot = *slots_[index];
  padUpdate(&slot.pad);

  const bool connected = padIsConnected(&slot.pad);
  if (connected != slot.connected) {
    slot.connected = connected;
    if (connected) {
      slot.id = DeviceId((uint64_t(++connection_counter_) << 32) | (kDeviceIdTag << 8) | index);
      slot.state = {};
      slot.state_changed = true;
      slot.key_buttons = 0;
      slot.repeat_state = RepeatState::kIdle;
      slot.vibration_count = 0;
    } else {
      slot.id = DeviceId::kInvalid;
    }
  }
  if (!connected) {
    return;
  }

  uint64_t held = padGetButtons(&slot.pad);
  HidAnalogStickState left = padGetStickPos(&slot.pad, 0);
  HidAnalogStickState right = padGetStickPos(&slot.pad, 1);
  // A single Joy-Con held sideways (input_joycon_sideways): its presses as a two-handed controller's, and its one
  // stick as the left stick (the right Joy-Con reports its stick as stick 1). The system already turned the stick.
  if (index != 0 && REXCVAR_GET(input_joycon_sideways)) {
    const uint32_t style = padGetStyleSet(&slot.pad);
    const uint32_t whole = HidNpadStyleTag_NpadJoyDual | HidNpadStyleTag_NpadFullKey | HidNpadStyleTag_NpadHandheld;
    if (!(style & whole) && (style & HidNpadStyleTag_NpadJoyLeft)) {
      held = SidewaysLeftJoyCon(held);
      right = HidAnalogStickState{};
    } else if (!(style & whole) && (style & HidNpadStyleTag_NpadJoyRight)) {
      held = SidewaysRightJoyCon(held);
      left = right;
      right = HidAnalogStickState{};
    }
  }
  // Route recording (a test tool) follows player 1: the handheld, or No1 when the handheld is not in use.
  if (index == 0 || (index == 1 && !slots_[0]->connected)) {
    RouteRecording(held, left, right);
  }
  DispatchMenuShortcuts(slot, held);
  X_INPUT_GAMEPAD gamepad{};
  const ButtonTable& table = g_tables[slot.ordinal == 1 ? 1 : 0][REXCVAR_GET(input_xbox_layout) ? 1 : 0];
  uint16_t buttons = 0;
  for (int i = 0; i < 16; ++i) {
    if (held & table.buttons[i]) {
      buttons |= uint16_t(1u << i);
    }
  }
  gamepad.buttons = buttons;
  // ZL and ZR are digital.
  gamepad.left_trigger = (held & table.left_trigger) ? 0xFF : 0;
  gamepad.right_trigger = (held & table.right_trigger) ? 0xFF : 0;
  // HID reports up as positive Y, like XInput, so no inversion is needed.
  gamepad.thumb_lx = ClampStickAxis(left.x);
  gamepad.thumb_ly = ClampStickAxis(left.y);
  gamepad.thumb_rx = ClampStickAxis(right.x);
  gamepad.thumb_ry = ClampStickAxis(right.y);
  /*
   * With the debug overlay open, L+R and the right stick move it (debug_overlay.cpp,
   * on the UI thread). While L and R are held, the game sees neither those two
   * buttons nor the right stick.
   */
  constexpr uint64_t kMoveOverlay = HidNpadButton_L | HidNpadButton_R;
  if ((held & kMoveOverlay) == kMoveOverlay && rex::ui::DebugOverlayOpen()) {
    // The XInput state fields are stored big-endian: they are rewritten whole.
    gamepad.buttons = static_cast<uint16_t>(
        buttons & ~(X_INPUT_GAMEPAD_LEFT_SHOULDER | X_INPUT_GAMEPAD_RIGHT_SHOULDER));
    gamepad.thumb_rx = static_cast<int16_t>(0);
    gamepad.thumb_ry = static_cast<int16_t>(0);
  }

  if (std::memcmp(&gamepad, &slot.state.gamepad, sizeof(gamepad)) != 0) {
    slot.state.gamepad = gamepad;
    slot.state_changed = true;
  }
}


/* From the profiler (switch_perf.cpp): turns the GPU A/B test lap on and off. */
extern "C" void RexSwitchPerfToggleAb(void);
/*
 * The SDK menus, on the controller
 *
 * On the PC they are function keys (see RegisterBind in rex_app.cpp): F3 debug,
 * F4 settings, F7 achievements and the tilde key for the log console. The Switch
 * has no keyboard, so they go to combinations with L+R, which no game uses
 * together with the D-pad:
 *
 *   L + R + Up    -> debug overlay (counters, statistics)
 *   L + R + Right -> settings
 *   L + R + Down  -> log console
 *   L + R + Left  -> achievements
 *   L + R + ZL    -> GPU A/B test lap (for measuring, not for playing)
 *   L + R + right stick -> moves the debug overlay if it is open
 *                          (its position is saved; see Poll and debug_overlay.cpp)
 *
 * It fires once per press, not while held: each shortcut stores its bit in
 * slot.menu_shortcuts with the state from the previous poll.
 */
void SwitchInputDriver::DispatchMenuShortcuts(Slot& slot, uint64_t held) {
  struct Shortcut {
    uint64_t button;
    rex::ui::VirtualKey keystroke;
  };
  constexpr uint64_t kModifier = HidNpadButton_L | HidNpadButton_R;
  static const Shortcut kShortcuts[] = {
      {HidNpadButton_Up, rex::ui::VirtualKey::kF3},
      {HidNpadButton_Right, rex::ui::VirtualKey::kF4},
      {HidNpadButton_Down, rex::ui::VirtualKey::kOem3},
      {HidNpadButton_Left, rex::ui::VirtualKey::kF7},
  };

  const bool modifier = (held & kModifier) == kModifier;
  for (size_t i = 0; i < std::size(kShortcuts); ++i) {
    const uint64_t bit = uint64_t(1) << i;
    const bool now = modifier && (held & kShortcuts[i].button) != 0;
    const bool before = (slot.menu_shortcuts & bit) != 0;
    if (now && !before) {
      /*
       * Each shortcut creates or destroys an ImGui menu, and the UI thread
       * draws them continuously while any is open. From this game thread a
       * menu could be deleted in the middle of its Draw, so it is done on the
       * UI thread.
       */
      const rex::ui::VirtualKey vk = kShortcuts[i].keystroke;
      auto press = [vk] {
        rex::ui::KeyEvent keystroke(nullptr, vk, 1, false, false, false, false, false);
        rex::ui::ProcessKeyEvent(keystroke);
      };
      rex::ui::Window* const window = window_ui_.load(std::memory_order_acquire);
      if (!window || !window->app_context().CallInUIThread(press)) {
        press();
      }
      REXLOG_INFO("menu shortcut: {}", rex::ui::VirtualKeyToString(vk));
    }
    slot.menu_shortcuts = now ? (slot.menu_shortcuts | bit) : (slot.menu_shortcuts & ~bit);
  }

  /*
   * L+R+ZL starts the A/B test lap (switch_perf.cpp). It sends no key to the
   * game: it talks directly to the profiler. It is kept apart from the loop
   * above because that one translates buttons into keys and this is not a key.
   */
  const uint64_t kBitAb = uint64_t(1) << 8;
  const bool ab_now = modifier && (held & HidNpadButton_ZL) != 0;
  const bool ab_before = (slot.menu_shortcuts & kBitAb) != 0;
  if (ab_now && !ab_before) {
    RexSwitchPerfToggleAb();
    REXLOG_INFO("menu shortcut: GPU A/B tests");
  }
  slot.menu_shortcuts = ab_now ? (slot.menu_shortcuts | kBitAb)
                                 : (slot.menu_shortcuts & ~kBitAb);

  /*
   * L+R+ZR opens the system's controller applet: connect a second controller, change the player order, or split a
   * Joy-Con pair into two sideways Joy-Cons for two players. On the UI thread, like the menus (it blocks until the
   * player closes it).
   */
  const uint64_t kBitApplet = uint64_t(1) << 9;
  const bool applet_now = modifier && (held & HidNpadButton_ZR) != 0;
  const bool applet_before = (slot.menu_shortcuts & kBitApplet) != 0;
  if (applet_now && !applet_before) {
    REXLOG_INFO("menu shortcut: controllers");
    auto show = [this] { ShowControllerApplet(); };
    rex::ui::Window* const window = window_ui_.load(std::memory_order_acquire);
    if (!window || !window->app_context().CallInUIThread(show)) {
      show();
    }
  }
  slot.menu_shortcuts = applet_now ? (slot.menu_shortcuts | kBitApplet) : (slot.menu_shortcuts & ~kBitApplet);
}

void SwitchInputDriver::ShowControllerApplet() {
  HidLaControllerSupportArg arg;
  hidLaCreateControllerSupportArg(&arg);
  arg.hdr.player_count_min = 0;
  arg.hdr.player_count_max = static_cast<s8>(kPlayerCount);
  arg.hdr.enable_take_over_connection = 1;  // the controllers already connected stay
  arg.hdr.enable_left_justify = 1;          // players numbered without gaps
  arg.hdr.enable_permit_joy_dual = 1;       // a Joy-Con pair is one player
  arg.hdr.enable_single_mode = 0;           // several players (single Joy-Cons sideways allowed)
  HidLaControllerSupportResultInfo info{};
  const Result rc = hidLaShowControllerSupport(&info, &arg);
  if (R_FAILED(rc)) {
    REXLOG_WARN("controller applet: failed (0x{:08X})", rc);
  } else {
    REXLOG_INFO("controller applet: {} player(s), selected npad {}", int(info.player_count), int(info.selected_id));
  }
}

SwitchInputDriver::Slot* SwitchInputDriver::FindSlot(DeviceId id) {
  const size_t index = static_cast<uint64_t>(id) & 0xFF;
  if (id == DeviceId::kInvalid || index >= kSlotCount) {
    return nullptr;
  }
  Poll(index);
  Slot* slot = slots_[index].get();
  return slot->connected && slot->id == id ? slot : nullptr;
}

void SwitchInputDriver::EnumerateDevices(std::vector<DeviceInfo>& out) {
  if (!initialized_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  for (size_t i = 0; i < kSlotCount; ++i) {
    Poll(i);
    const Slot& slot = *slots_[i];
    if (!slot.connected) {
      continue;
    }
    DeviceInfo info;
    info.id = slot.id;
    const uint32_t style = padGetStyleSet(&slot.pad);
    if (i == 0) {
      info.name = "Nintendo Switch (handheld)";
    } else if (style & HidNpadStyleTag_NpadFullKey) {
      info.name = "Pro Controller";
    } else if (style & HidNpadStyleTag_NpadJoyDual) {
      info.name = "Joy-Con (L/R)";
    } else if (style & HidNpadStyleTag_NpadJoyLeft) {
      info.name = "Joy-Con (L)";
    } else if (style & HidNpadStyleTag_NpadJoyRight) {
      info.name = "Joy-Con (R)";
    } else {
      info.name = "Nintendo Switch controller";
    }
    info.guid = i == 0 ? std::string("switch-handheld") : "switch-npad-" + std::to_string(i);
    info.synthetic = false;
    out.push_back(std::move(info));
  }
}

void SwitchInputDriver::OnDevicesAssigned(const std::vector<DeviceInfo>& devices) {
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto& slot : slots_) {
    if (!slot) continue;
    uint32_t ordinal = 0;
    for (const DeviceInfo& device : devices)
      if (device.id == slot->id) ordinal = device.ordinal;
    slot->ordinal = ordinal;
  }
}

X_RESULT SwitchInputDriver::GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) {
  std::lock_guard<std::mutex> lock(mutex_);
  Slot* slot = FindSlot(id);
  if (!slot) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  // Make sure packet_number is only incremented by 1, even if there have been
  // multiple updates between GetState calls. Also track `is_active` to
  // increment the packet number if it changed.
  const bool is_active = this->is_active();
  if (is_active != slot->is_active || (is_active && slot->state_changed)) {
    slot->state.packet_number = uint32_t(slot->state.packet_number) + 1;
    slot->is_active = is_active;
    slot->state_changed = false;
  }
  std::memcpy(out_state, &slot->state, sizeof(*out_state));
  if (!is_active) {
    // Simulate an "untouched" controller.
    std::memset(&out_state->gamepad, 0, sizeof(out_state->gamepad));
  }
  return X_ERROR_SUCCESS;
}

X_RESULT SwitchInputDriver::GetDeviceCapabilities(DeviceId id, uint32_t flags,
                                                  X_INPUT_CAPABILITIES* out_caps) {
  (void)flags;
  if (!out_caps) {
    return X_ERROR_BAD_ARGUMENTS;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Slot* slot = FindSlot(id);
  if (!slot) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  std::memset(out_caps, 0, sizeof(*out_caps));
  out_caps->type = 0x01;      // XINPUT_DEVTYPE_GAMEPAD
  out_caps->sub_type = 0x01;  // XINPUT_DEVSUBTYPE_GAMEPAD
  const bool handheld = slot->npad_id == HidNpadIdType_Handheld;
  out_caps->flags = handheld ? 0 : X_INPUT_CAPS_WIRELESS;
  out_caps->gamepad.buttons = 0xF3FF;
  out_caps->gamepad.left_trigger = 0xFF;
  out_caps->gamepad.right_trigger = 0xFF;
  out_caps->gamepad.thumb_lx = static_cast<int16_t>(0xFFFFu);
  out_caps->gamepad.thumb_ly = static_cast<int16_t>(0xFFFFu);
  out_caps->gamepad.thumb_rx = static_cast<int16_t>(0xFFFFu);
  out_caps->gamepad.thumb_ry = static_cast<int16_t>(0xFFFFu);
  out_caps->vibration.left_motor_speed = 0xFFFFu;
  out_caps->vibration.right_motor_speed = 0xFFFFu;
  return X_ERROR_SUCCESS;
}

X_RESULT SwitchInputDriver::SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration) {
  std::lock_guard<std::mutex> lock(mutex_);
  Slot* slot = FindSlot(id);
  if (!slot) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  HidNpadIdType npad_id = slot->npad_id;
  uint32_t style = 0;
  int count = 0;
  if (slot->npad_id == HidNpadIdType_Handheld) {
    style = HidNpadStyleTag_NpadHandheld;
    count = 2;
  } else {
    const uint32_t attached = hidGetNpadStyleSet(npad_id);
    if (attached & HidNpadStyleTag_NpadFullKey) {
      style = HidNpadStyleTag_NpadFullKey;
      count = 2;
    } else if (attached & HidNpadStyleTag_NpadJoyDual) {
      style = HidNpadStyleTag_NpadJoyDual;
      count = 2;
    } else if (attached & HidNpadStyleTag_NpadJoyLeft) {
      style = HidNpadStyleTag_NpadJoyLeft;
      count = 1;
    } else if (attached & HidNpadStyleTag_NpadJoyRight) {
      style = HidNpadStyleTag_NpadJoyRight;
      count = 1;
    } else {
      // Nothing that can rumble.
      return X_ERROR_SUCCESS;
    }
  }

  if (!slot->vibration_count || slot->vibration_npad_id != npad_id ||
      slot->vibration_style != style) {
    Result rc = hidInitializeVibrationDevices(slot->vibration_handles.data(), count, npad_id,
                                              static_cast<HidNpadStyleTag>(style));
    if (R_FAILED(rc)) {
      slot->vibration_count = 0;
      return X_ERROR_FUNCTION_FAILED;
    }
    slot->vibration_npad_id = npad_id;
    slot->vibration_style = style;
    slot->vibration_count = count;
    slot->vibration_sent = false;  // new or reassigned controller: it needs the state sent
  }

  // See the vibration_sent comment in Slot: same value as the last one sent, nothing to do.
  const uint16_t left = uint16_t(vibration->left_motor_speed);
  const uint16_t right = uint16_t(vibration->right_motor_speed);
  if (slot->vibration_sent && slot->vibration_left == left &&
      slot->vibration_right == right) {
    return X_ERROR_SUCCESS;
  }

  // XInput's left motor is the heavy low-frequency one, the right motor the
  // light high-frequency one.
  HidVibrationValue value{};
  value.amp_low = float(uint16_t(vibration->left_motor_speed)) / 65535.0f;
  value.freq_low = 160.0f;
  value.amp_high = float(uint16_t(vibration->right_motor_speed)) / 65535.0f;
  value.freq_high = 320.0f;
  std::array<HidVibrationValue, 2> values = {value, value};
  if (R_FAILED(hidSendVibrationValues(slot->vibration_handles.data(), values.data(),
                                      slot->vibration_count))) {
    slot->vibration_sent = false;  // retried on the next call
    return X_ERROR_FUNCTION_FAILED;
  }
  slot->vibration_sent = true;
  slot->vibration_left = left;
  slot->vibration_right = right;
  return X_ERROR_SUCCESS;
}

X_RESULT SwitchInputDriver::GetDeviceKeystroke(DeviceId id, uint32_t flags,
                                               X_INPUT_KEYSTROKE* out_keystroke) {
  (void)flags;
  if (!out_keystroke) {
    return X_ERROR_BAD_ARGUMENTS;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  Slot* slot = FindSlot(id);
  if (!slot) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  // If input is not active (e.g. due to a dialog overlay), force buttons to
  // "unpressed". The algorithm will automatically send UP events when
  // 'is_active()' goes low and DOWN events when it goes high again.
  const bool is_active = this->is_active();
  const uint64_t curr_butts =
      is_active ? (static_cast<uint64_t>(static_cast<uint16_t>(slot->state.gamepad.buttons)) |
                   AnalogToKeyfield(slot->state.gamepad))
                : uint64_t(0);

  // Handle repeating
  const auto guest_now = static_cast<uint32_t>(rex::chrono::Clock::QueryGuestUptimeMillis());
  if (slot->repeat_state == RepeatState::kWaiting &&
      (slot->repeat_time + kRepeatDelayMs < guest_now)) {
    slot->repeat_state = RepeatState::kRepeating;
  }
  if (slot->repeat_state == RepeatState::kRepeating &&
      (slot->repeat_time + kRepeatRateMs < guest_now)) {
    slot->repeat_time = guest_now;
    rex::ui::VirtualKey vk = kVkLookup.at(slot->repeat_index);
    out_keystroke->virtual_key = uint16_t(vk);
    out_keystroke->unicode = 0;
    // InputSystem stamps the guest user this device is assigned to.
    out_keystroke->user_index = 0;
    out_keystroke->hid_code = 0;
    out_keystroke->flags = X_INPUT_KEYSTROKE_KEYDOWN | X_INPUT_KEYSTROKE_REPEAT;
    return X_ERROR_SUCCESS;
  }

  const uint64_t butts_changed = curr_butts ^ slot->key_buttons;
  if (!butts_changed) {
    return X_ERROR_EMPTY;
  }

  // First try to clear buttons with up events. This is to match xinput
  // behavior when transitioning thumb sticks, e.g. so that THUMB_UPLEFT is
  // up before THUMB_LEFT is down.
  for (auto [clear_pass, pass] = std::tuple{true, 0}; pass < 2; clear_pass = false, pass++) {
    for (uint8_t i = 0; i < uint8_t(std::size(kVkLookup)); i++) {
      const uint64_t fbutton = uint64_t(1) << i;
      if (!(butts_changed & fbutton)) {
        continue;
      }
      rex::ui::VirtualKey vk = kVkLookup.at(i);
      if (vk == rex::ui::VirtualKey::kNone) {
        continue;
      }

      out_keystroke->virtual_key = uint16_t(vk);
      out_keystroke->unicode = 0;
      out_keystroke->user_index = 0;
      out_keystroke->hid_code = 0;

      const bool is_pressed = curr_butts & fbutton;
      if (clear_pass && !is_pressed) {
        out_keystroke->flags = X_INPUT_KEYSTROKE_KEYUP;
        slot->key_buttons &= ~fbutton;
        slot->repeat_state = RepeatState::kIdle;
        return X_ERROR_SUCCESS;
      }
      if (!clear_pass && is_pressed) {
        out_keystroke->flags = X_INPUT_KEYSTROKE_KEYDOWN;
        slot->key_buttons |= fbutton;
        slot->repeat_state = RepeatState::kWaiting;
        slot->repeat_index = i;
        slot->repeat_time = guest_now;
        return X_ERROR_SUCCESS;
      }
    }
  }
  return X_ERROR_EMPTY;
}

}  // namespace rex::input::nx
