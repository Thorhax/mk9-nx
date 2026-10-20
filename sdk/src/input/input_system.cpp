/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include <rex/dbg.h>
#include <rex/input/device_assignment.h>
#include <rex/input/flags.h>
#include <rex/input/input_driver.h>
#include <rex/input/input_system.h>
#include <rex/input/read_hook.h>  // sa2-nx (2026-10-20)
#include <rex/input/mnk/mnk_input_driver.h>
#include <rex/input/nop/nop_input_driver.h>
#if REX_PLATFORM_SWITCH
#include <rex/input/switch/switch_input_driver.h>
#else
#include <rex/input/sdl/sdl_input_driver.h>
#endif
#include <rex/input/state_merge.h>
#include <rex/input/xinput/xinput_input_driver.h>
#include <rex/logging.h>

#if REX_PLATFORM_SWITCH
REXCVAR_DEFINE_STRING(input_backend, "switch", "Input", "Input backend: switch")
    .allowed({"switch"});
#else
REXCVAR_DEFINE_STRING(input_backend, "sdl", "Input", "Input backend: sdl, xinput")
    .allowed({"sdl", "xinput"});
#endif

REXCVAR_DEFINE_BOOL(guide_button, false, "Input", "Enable guide button pass-through");
// sa2-nx (2026-10-20, from banjo-nx's input_script, with the stick and hold times of SA2's PC SA2_AUTO_INPUT): scripted
// pad input on any platform (the Switch's too: Ryujinx test runs), read from sa2.toml at boot.
REXCVAR_DEFINE_STRING(input_script, "", "Input",
                      "Test automation: pad input for user 0 as 'seconds:button[:hold seconds],...' counted from the "
                      "first input poll (wall time). Buttons A B X Y START BACK UP DOWN LEFT RIGHT LB RB LT RT, sticks "
                      "LX+ LX- LY+ LY- RX+ RX- RY+ RY-; held 0.2 s unless a hold is given. While set, user 0 always "
                      "has a pad connected (works minimized, with no controller) and reads only the script")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(input_rumble, false, "Input",
                    "Controller vibration (off by default: controllers only get "
                    "zero vibration). true = the game's vibration");
namespace rex::input {

namespace {

// Synthetic devices are parked past every physical ordinal so they cannot push
// a real pad off guest user 0. SlotAssignment routes them by their synthetic
// flag and never reads this value.
constexpr uint32_t kSyntheticOrdinal = UINT32_MAX;

}  // namespace

InputSystem::InputSystem(rex::ui::Window* window) : window_(window) {}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() {
  return X_STATUS_SUCCESS;
}

void InputSystem::Shutdown() {
  // device_owners_ holds raw driver pointers.
  devices_.clear();
  device_owners_.clear();
  drivers_.clear();
}

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::AttachWindow(rex::ui::Window* window) {
  window_ = window;
  for (auto& driver : drivers_) {
    driver->OnWindowAvailable(window);
  }
}

void InputSystem::SetActiveCallback(std::function<bool()> callback) {
  for (auto& driver : drivers_) {
    driver->set_is_active_callback(callback);
  }
}

void InputSystem::SetDeviceAssignment(std::unique_ptr<DeviceAssignment> assignment) {
  assignment_ = std::move(assignment);
  if (assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

void InputSystem::RefreshDevices() {
  /*
   * At most one enumeration every 50 ms.
   *
   * Stack sampling found the thread that prepares each game frame spending 5.8 % of its time
   * inside SetState (vibration), and every game call to the controller went through here:
   * three new vectors and a query to each driver, several times per frame. A controller is not
   * connected or disconnected 30 times per second; with 50 ms a new controller shows up before
   * anyone notices. The first call always enumerates.
   */
  {
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              std::chrono::steady_clock::now().time_since_epoch())
                              .count();
    const int64_t last = last_refresh_ns_.load(std::memory_order_relaxed);
    if (last != 0 && !devices_.empty() && now - last < 50000000) {
      return;
    }
    last_refresh_ns_.store(now, std::memory_order_relaxed);
  }
  std::vector<DeviceInfo> seen;
  std::vector<InputDriver*> owners;
  std::vector<DeviceInfo> enumerated;
  for (auto& driver : drivers_) {
    enumerated.clear();
    driver->EnumerateDevices(enumerated);
    for (auto& info : enumerated) {
      seen.push_back(info);
      owners.push_back(driver.get());
    }
  }

  // Carry forward ordinals already handed out, so a device keeps its guest user
  // when another pad is unplugged.
  bool changed = seen.size() != devices_.size();
  std::vector<bool> fresh(seen.size(), false);
  for (size_t i = 0; i < seen.size(); i++) {
    auto existing = std::find_if(devices_.begin(), devices_.end(),
                                 [&](const DeviceInfo& d) { return d.id == seen[i].id; });
    if (existing != devices_.end()) {
      seen[i].ordinal = existing->ordinal;
      continue;
    }
    fresh[i] = true;
    changed = true;
  }

  // Runs after the carry-forward pass so a new device cannot take an ordinal a
  // live one is still holding. Lowest free rather than a growing counter,
  // because a reconnected pad arrives as a new device and would otherwise walk
  // off the end of the guest users.
  for (size_t i = 0; i < seen.size(); i++) {
    if (!fresh[i]) {
      continue;
    }
    // Only physical devices consume an ordinal, so the first pad to connect is
    // guest user 0 however many synthetic devices enumerated ahead of it.
    if (seen[i].synthetic) {
      seen[i].ordinal = kSyntheticOrdinal;
      fresh[i] = false;
      continue;
    }
    auto taken = [&](uint32_t candidate) {
      for (size_t j = 0; j < seen.size(); j++) {
        if (!fresh[j] && !seen[j].synthetic && seen[j].ordinal == candidate) {
          return true;
        }
      }
      return false;
    };
    uint32_t ordinal = 0;
    while (taken(ordinal)) {
      ordinal++;
    }
    seen[i].ordinal = ordinal;
    fresh[i] = false;
  }

  for (const auto& old : devices_) {
    if (std::none_of(seen.begin(), seen.end(),
                     [&](const DeviceInfo& d) { return d.id == old.id; })) {
      active_devices_.Forget(old.id);
      changed = true;
    }
  }

  std::vector<size_t> order(seen.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  // Stable: synthetic devices share one ordinal, and their relative order
  // decides which answers GetCapabilities when no pad is attached.
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return seen[a].ordinal < seen[b].ordinal; });

  devices_.clear();
  device_owners_.clear();
  for (size_t i : order) {
    devices_.push_back(seen[i]);
    device_owners_.push_back(owners[i]);
  }

  if (changed && assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
  // sa2-nx: drivers with per-player settings (the Switch's per-player button maps) learn each device's player.
  if (changed) {
    for (auto& driver : drivers_) {
      driver->OnDevicesAssigned(devices_);
    }
  }
}

InputDriver* InputSystem::DriverForDevice(DeviceId id) {
  for (size_t i = 0; i < devices_.size(); i++) {
    if (devices_[i].id == id) {
      return device_owners_[i];
    }
  }
  return nullptr;
}

namespace {
// input_script: the presses, parsed once.
struct ScriptedPress {
  double at = 0.0, hold = 0.2;
  uint16_t buttons = 0;
  uint8_t lt = 0, rt = 0;
  int16_t lx = 0, ly = 0, rx = 0, ry = 0;
};
const std::vector<ScriptedPress>& InputScript() {
  static const std::vector<ScriptedPress> presses = [] {
    std::vector<ScriptedPress> out;
    const std::string text = REXCVAR_GET(input_script);
    size_t at = 0;
    while (at < text.size()) {
      size_t end = text.find_first_of(",;", at);
      if (end == std::string::npos) end = text.size();
      const std::string item = text.substr(at, end - at);
      at = end + 1;
      const size_t c1 = item.find(':');
      if (c1 == std::string::npos) continue;
      const size_t c2 = item.find(':', c1 + 1);
      ScriptedPress p;
      p.at = std::atof(item.substr(0, c1).c_str());
      const std::string name = item.substr(c1 + 1, c2 == std::string::npos ? std::string::npos : c2 - c1 - 1);
      if (c2 != std::string::npos) p.hold = std::atof(item.substr(c2 + 1).c_str());
      static const std::pair<const char*, uint16_t> kButtons[] = {
          {"UP", 0x0001},   {"DOWN", 0x0002}, {"LEFT", 0x0004}, {"RIGHT", 0x0008}, {"START", 0x0010},
          {"BACK", 0x0020}, {"LB", 0x0100},   {"RB", 0x0200},   {"A", 0x1000},     {"B", 0x2000},
          {"X", 0x4000},    {"Y", 0x8000}};
      bool known = false;
      for (const auto& [button, bits] : kButtons)
        if (name == button) p.buttons = bits, known = true;
      if (name == "LT") p.lt = 0xFF, known = true;
      if (name == "RT") p.rt = 0xFF, known = true;
      if (name == "LX+") p.lx = INT16_MAX, known = true;
      if (name == "LX-") p.lx = -INT16_MAX, known = true;
      if (name == "LY+") p.ly = INT16_MAX, known = true;
      if (name == "LY-") p.ly = -INT16_MAX, known = true;
      if (name == "RX+") p.rx = INT16_MAX, known = true;
      if (name == "RX-") p.rx = -INT16_MAX, known = true;
      if (name == "RY+") p.ry = INT16_MAX, known = true;
      if (name == "RY-") p.ry = -INT16_MAX, known = true;
      if (known) out.push_back(p);
    }
    if (!out.empty()) REXLOG_INFO("input_script: {} presses scripted", out.size());
    return out;
  }();
  return presses;
}
// What the script holds now for user 0 (the clock starts at the first poll).
ScriptedPress ScriptedState() {
  static const auto start = std::chrono::steady_clock::now();
  const double now = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  ScriptedPress s;
  for (const ScriptedPress& p : InputScript()) {
    if (now < p.at || now >= p.at + p.hold) continue;
    s.buttons |= p.buttons;
    s.lt = std::max(s.lt, p.lt);
    s.rt = std::max(s.rt, p.rt);
    if (p.lx) s.lx = p.lx;
    if (p.ly) s.ly = p.ly;
    if (p.rx) s.rx = p.rx;
    if (p.ry) s.ry = p.ry;
    static double logged = -1.0;
    if (logged != p.at) {
      logged = p.at;
      REXLOG_INFO("input_script: input at {:.1f} s (scripted for {:.1f} s)", now, p.at);
    }
  }
  return s;
}
}  // namespace

const DeviceInfo* InputSystem::DeviceInfoFor(DeviceId id) const {
  for (const auto& device : devices_) {
    if (device.id == id) {
      return &device;
    }
  }
  return nullptr;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");
  if (out_caps && user_index == 0 && !InputScript().empty()) {  // input_script: a standard pad
    std::memset(out_caps, 0, sizeof(*out_caps));
    out_caps->type = 0x01;      // XINPUT_DEVTYPE_GAMEPAD
    out_caps->sub_type = 0x01;  // XINPUT_DEVSUBTYPE_GAMEPAD
    out_caps->gamepad.buttons = 0xF3FF;
    out_caps->gamepad.left_trigger = 0xFF;
    out_caps->gamepad.right_trigger = 0xFF;
    return X_ERROR_SUCCESS;
  }
  if (!out_caps || !assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);
  if (ids.empty()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  // Prefer the pad in hand, so button glyphs follow it rather than whichever
  // device enumerated first.
  DeviceId chosen = active_devices_.Active(user_index);
  if (std::find(ids.begin(), ids.end(), chosen) == ids.end()) {
    chosen = ids.front();
  }

  auto* driver = DriverForDevice(chosen);
  if (!driver) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  return driver->GetDeviceCapabilities(chosen, flags, out_caps);
}

// sa2-nx (2026-10-20, rex/input/read_hook.h): the app's function called before every controller state read.
namespace {
std::atomic<ReadHook> g_read_hook{nullptr};
}  // namespace

void SetReadHook(ReadHook hook) { g_read_hook.store(hook, std::memory_order_release); }

X_RESULT InputSystem::GetState(uint32_t user_index, X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");
  if (const ReadHook hook = g_read_hook.load(std::memory_order_acquire))
    hook();  // sa2-nx (2026-10-20): Renut's late input and latency probe (banjo-nx's input latency set)
  if (user_index == 0 && !InputScript().empty()) {  // input_script: only the scripted input
    const ScriptedPress s = ScriptedState();
    static uint32_t packet = 1;
    static uint64_t last = 0;
    const uint64_t now = uint64_t(s.buttons) | uint64_t(s.lt) << 16 | uint64_t(s.rt) << 24 |
                         uint64_t(uint16_t(s.lx)) << 32 | uint64_t(uint16_t(s.ly)) << 48;
    if (now != last) {
      last = now;
      ++packet;
    }
    if (out_state) {
      std::memset(out_state, 0, sizeof(*out_state));
      out_state->packet_number = packet;
      out_state->gamepad.buttons = s.buttons;
      out_state->gamepad.left_trigger = s.lt;
      out_state->gamepad.right_trigger = s.rt;
      out_state->gamepad.thumb_lx = s.lx;
      out_state->gamepad.thumb_ly = s.ly;
      out_state->gamepad.thumb_rx = s.rx;
      out_state->gamepad.thumb_ry = s.ry;
    }
    return X_ERROR_SUCCESS;
  }
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  X_INPUT_STATE merged = {};
  bool any = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_INPUT_STATE state = {};
    if (driver->GetDeviceState(id, &state) != X_ERROR_SUCCESS) {
      continue;
    }
    active_devices_.Observe(user_index, id, state.gamepad);
    if (!any) {
      merged = state;
      any = true;
    } else {
      MergeInto(merged, state);
    }
  }

  if (!any) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  if (out_state) {
    *out_state = merged;
  }
  return X_ERROR_SUCCESS;
}

X_RESULT InputSystem::SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  // Every pad on this user belongs to the same player, so all of them buzz.
  // Only pads decide the result: synthetic devices accept any vibration and
  // would otherwise report success for a pad that never rumbled.
  bool any_pad = false;
  bool any_synthetic = false;
  bool pad_rumbled = false;
  X_RESULT pad_error = X_ERROR_DEVICE_NOT_CONNECTED;
  // input_rumble = false: the game still sees its call serviced, but the controllers only
  // receive zero vibration.
  X_INPUT_VIBRATION no_rumble{};
  if (!REXCVAR_GET(input_rumble)) {
    vibration = &no_rumble;
  }
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    const DeviceInfo* info = DeviceInfoFor(id);
    if (!driver || !info) {
      continue;
    }
    X_RESULT result = driver->SetDeviceVibration(id, vibration);
    if (info->synthetic) {
      any_synthetic = true;
      continue;
    }
    any_pad = true;
    if (result == X_ERROR_SUCCESS) {
      pad_rumbled = true;
    } else {
      pad_error = result;
    }
  }
  if (any_pad) {
    return pad_rumbled ? X_ERROR_SUCCESS : pad_error;
  }
  return any_synthetic ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");
  if ((user_index == 0 || user_index == 0xFF) && !InputScript().empty()) {
    // input_script: menus read key events; one down / up event per change of the scripted buttons.
    static uint16_t reported = 0;
    const uint16_t buttons = ScriptedState().buttons;
    const uint16_t changed = uint16_t(buttons ^ reported);
    if (!changed) return X_ERROR_EMPTY;
    static const std::pair<uint16_t, uint16_t> kKeys[] = {
        {0x1000, 0x5800}, {0x2000, 0x5801}, {0x4000, 0x5802}, {0x8000, 0x5803}, {0x0200, 0x5804},
        {0x0100, 0x5805}, {0x0001, 0x5810}, {0x0002, 0x5811}, {0x0004, 0x5812}, {0x0008, 0x5813},
        {0x0010, 0x5814}, {0x0020, 0x5815}};
    for (const auto& [bit, virtual_key] : kKeys) {
      if (!(changed & bit)) continue;
      reported = uint16_t(reported ^ bit);
      if (out_keystroke) {
        std::memset(out_keystroke, 0, sizeof(*out_keystroke));
        out_keystroke->virtual_key = virtual_key;
        out_keystroke->flags = (buttons & bit) ? 0x0001 : 0x0002;  // XINPUT_KEYSTROKE_KEYDOWN / KEYUP
        out_keystroke->user_index = 0;
      }
      return X_ERROR_SUCCESS;
    }
    reported = buttons;
    return X_ERROR_EMPTY;
  }
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  bool any_connected = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_RESULT result = driver->GetDeviceKeystroke(id, flags, out_keystroke);
    if (result == X_ERROR_SUCCESS) {
      out_keystroke->user_index = static_cast<uint8_t>(user_index);
      return result;
    }
    if (result != X_ERROR_DEVICE_NOT_CONNECTED) {
      any_connected = true;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

std::unique_ptr<InputSystem> CreateDefaultInputSystem(bool tool_mode) {
  auto input = std::make_unique<InputSystem>(nullptr);

  if (!tool_mode) {
#if REX_PLATFORM_WIN32
    if (REXCVAR_GET(input_backend) == "xinput") {
      auto xinput_driver = std::make_unique<xinput::XinputInputDriver>(nullptr, 0);
      if (xinput_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(xinput_driver));
      }
    }
#endif

#if REX_PLATFORM_SWITCH
    if (REXCVAR_GET(input_backend) == "switch") {
      auto switch_driver = std::make_unique<nx::SwitchInputDriver>(nullptr, 0);
      if (switch_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(switch_driver));
      }
    }
#else
    if (REXCVAR_GET(input_backend) == "sdl") {
      auto sdl_driver = std::make_unique<sdl::SDLInputDriver>(nullptr, 0);
      if (sdl_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(sdl_driver));
      }
    }
#endif

    // MnK driver (keyboard/mouse -> controller emulation)
    auto mnk_driver = std::make_unique<mnk::MnkInputDriver>(nullptr, 0);
    if (mnk_driver->Setup() == X_STATUS_SUCCESS) {
      input->AddDriver(std::move(mnk_driver));
    }
  }

  // NOP driver (primary in tool mode, fallback otherwise)
  uint8_t nop_index = tool_mode ? 0 : 1;
  input->AddDriver(std::make_unique<nop::NopInputDriver>(nullptr, nop_index));
  input->SetDeviceAssignment(std::make_unique<SlotAssignment>());
  return input;
}

}  // namespace rex::input
