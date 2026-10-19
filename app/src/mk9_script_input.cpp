// Scripted gamepad (see mk9_script_input.h).
//
// A script is a list of events, one per line in mk9_input_script_file or separated by ';' in mk9_input_script:
//
//     <seconds after start> <BUTTON[+BUTTON...]> [hold seconds, default 0.15]
//
//   20 START          press Start at 20 s for 0.15 s
//   25.5 DOWN         d-pad down
//   30 A+B 0.5        A and B together for half a second
//   40 LS_RIGHT 2     left stick fully right for 2 s
//
// Buttons: A B X Y START BACK LB RB LT RT UP DOWN LEFT RIGHT LS RS (stick clicks), LS_UP LS_DOWN LS_LEFT LS_RIGHT,
// RS_UP RS_DOWN RS_LEFT RS_RIGHT. '#' starts a comment. Times count from the creation of the input system, which is
// within a second of the process start (the snapshot times of mk9_snapshot_s count from about the same point).
// Every press and release is logged ("[script] ..."), so the log lines up with snapshots.

#include "mk9_script_input.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <rex/cvar.h>
#include <rex/input/input_driver.h>
#include <rex/input/input_system.h>
#include <rex/logging.h>

extern std::atomic<uint64_t> g_mk9_game_frames;  // src/mk9_frame_counter.cpp

REXCVAR_DEFINE_STRING(mk9_input_script, "", "MK9",
                      "Scripted gamepad events for tests, ';'-separated: \"<seconds> <BUTTON[+BUTTON]> [hold]\" "
                      "(e.g. \"20 START;25 A\"); empty = off");
REXCVAR_DEFINE_BOOL(mk9_input_script_frames, false, "MK9",
                    "Scripted gamepad times count game frames / 60 (presented frames, src/mk9_frame_counter.cpp) "
                    "instead of wall-clock seconds: with vblank_per_swap two runs press at the same frames");
REXCVAR_DEFINE_STRING(mk9_input_script_file, "", "MK9",
                      "File with scripted gamepad events, one per line (same format as mk9_input_script)");

namespace mk9::script_input {
namespace {

using namespace rex;  // X_STATUS, X_RESULT and their values

using rex::input::DeviceId;
using rex::input::DeviceInfo;

constexpr DeviceId kScriptDevice = static_cast<DeviceId>(0x55534634);  // "MK9"

struct Event {
  double start_s = 0;
  double hold_s = 0.15;
  uint16_t buttons = 0;
  uint8_t left_trigger = 0, right_trigger = 0;
  int16_t lx = 0, ly = 0, rx = 0, ry = 0;
  std::string text;
};

bool ApplyName(const std::string& name, Event& e) {
  struct Button {
    const char* name;
    uint16_t bit;
  };
  static const Button kButtons[] = {
      {"UP", rex::input::X_INPUT_GAMEPAD_DPAD_UP},       {"DOWN", rex::input::X_INPUT_GAMEPAD_DPAD_DOWN},
      {"LEFT", rex::input::X_INPUT_GAMEPAD_DPAD_LEFT},   {"RIGHT", rex::input::X_INPUT_GAMEPAD_DPAD_RIGHT},
      {"START", rex::input::X_INPUT_GAMEPAD_START},      {"BACK", rex::input::X_INPUT_GAMEPAD_BACK},
      {"LS", rex::input::X_INPUT_GAMEPAD_LEFT_THUMB},    {"RS", rex::input::X_INPUT_GAMEPAD_RIGHT_THUMB},
      {"LB", rex::input::X_INPUT_GAMEPAD_LEFT_SHOULDER}, {"RB", rex::input::X_INPUT_GAMEPAD_RIGHT_SHOULDER},
      {"A", rex::input::X_INPUT_GAMEPAD_A},              {"B", rex::input::X_INPUT_GAMEPAD_B},
      {"X", rex::input::X_INPUT_GAMEPAD_X},              {"Y", rex::input::X_INPUT_GAMEPAD_Y},
  };
  for (const auto& b : kButtons) {
    if (name == b.name) {
      e.buttons |= b.bit;
      return true;
    }
  }
  constexpr int16_t kFull = 32767, kFullNeg = -32768;
  if (name == "LT") e.left_trigger = 255;
  else if (name == "RT") e.right_trigger = 255;
  else if (name == "LS_UP") e.ly = kFull;
  else if (name == "LS_DOWN") e.ly = kFullNeg;
  else if (name == "LS_LEFT") e.lx = kFullNeg;
  else if (name == "LS_RIGHT") e.lx = kFull;
  else if (name == "RS_UP") e.ry = kFull;
  else if (name == "RS_DOWN") e.ry = kFullNeg;
  else if (name == "RS_LEFT") e.rx = kFullNeg;
  else if (name == "RS_RIGHT") e.rx = kFull;
  else return false;
  return true;
}

void ParseLine(std::string line, std::vector<Event>& out) {
  if (auto hash = line.find('#'); hash != std::string::npos) line.resize(hash);
  std::istringstream in(line);
  Event e;
  std::string names;
  if (!(in >> e.start_s >> names)) return;
  in >> e.hold_s;
  std::istringstream parts(names);
  for (std::string name; std::getline(parts, name, '+');) {
    std::transform(name.begin(), name.end(), name.begin(), ::toupper);
    if (!ApplyName(name, e)) {
      REXLOG_WARN("[script] unknown button '{}' in \"{}\"", name, line);
      return;
    }
  }
  e.text = names;
  out.push_back(e);
}

class ScriptInputDriver final : public rex::input::InputDriver {
 public:
  explicit ScriptInputDriver(std::vector<Event> events)
      : InputDriver(nullptr, 0), events_(std::move(events)), start_(std::chrono::steady_clock::now()),
        logged_(events_.size(), 0) {}

  X_STATUS Setup() override { return X_STATUS_SUCCESS; }

  void EnumerateDevices(std::vector<DeviceInfo>& out) override {
    DeviceInfo info;
    info.id = kScriptDevice;
    info.name = "Script";
    info.synthetic = true;
    out.push_back(info);
  }

  X_RESULT GetDeviceState(DeviceId id, rex::input::X_INPUT_STATE* out_state) override {
    if (id != kScriptDevice) return X_ERROR_DEVICE_NOT_CONNECTED;
    if (!out_state) return X_ERROR_SUCCESS;
    const double now = REXCVAR_GET(mk9_input_script_frames)
        ? double(g_mk9_game_frames.load(std::memory_order_relaxed)) / 60.0
        : std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    uint16_t buttons = 0;
    uint8_t lt = 0, rt = 0;
    int16_t lx = 0, ly = 0, rx = 0, ry = 0;
    std::lock_guard lock(mutex_);
    for (size_t i = 0; i < events_.size(); ++i) {
      const auto& e = events_[i];
      const bool active = now >= e.start_s && now < e.start_s + e.hold_s;
      if (active) {
        buttons |= e.buttons;
        lt = std::max(lt, e.left_trigger);
        rt = std::max(rt, e.right_trigger);
        if (e.lx) lx = e.lx;
        if (e.ly) ly = e.ly;
        if (e.rx) rx = e.rx;
        if (e.ry) ry = e.ry;
      }
      // 0 = not yet, 1 = pressed, 2 = released: log each transition once.
      if (active && logged_[i] == 0) {
        logged_[i] = 1;
        REXLOG_INFO("[script] {:.2f} s: press {} (for {:.2f} s)", now, e.text, e.hold_s);
      } else if (!active && now >= e.start_s + e.hold_s && logged_[i] < 2) {
        logged_[i] = 2;
        REXLOG_INFO("[script] {:.2f} s: release {}", now, e.text);
      }
    }
    const uint64_t key = uint64_t(buttons) | uint64_t(lt) << 16 | uint64_t(rt) << 24 |
                         uint64_t(uint16_t(lx)) << 32 | uint64_t(uint16_t(ly)) << 48;
    if (key != last_key_ || rx != last_rx_ || ry != last_ry_) {
      ++packet_;
      last_key_ = key;
      last_rx_ = rx;
      last_ry_ = ry;
    }
    std::memset(out_state, 0, sizeof(*out_state));
    out_state->packet_number = packet_;
    out_state->gamepad.buttons = buttons;
    out_state->gamepad.left_trigger = lt;
    out_state->gamepad.right_trigger = rt;
    out_state->gamepad.thumb_lx = lx;
    out_state->gamepad.thumb_ly = ly;
    out_state->gamepad.thumb_rx = rx;
    out_state->gamepad.thumb_ry = ry;
    return X_ERROR_SUCCESS;
  }

  X_RESULT GetDeviceCapabilities(DeviceId id, uint32_t, rex::input::X_INPUT_CAPABILITIES* out_caps) override {
    if (id != kScriptDevice) return X_ERROR_DEVICE_NOT_CONNECTED;
    if (out_caps) {
      std::memset(out_caps, 0, sizeof(*out_caps));
      out_caps->type = 0x01;      // XINPUT_DEVTYPE_GAMEPAD
      out_caps->sub_type = 0x01;  // XINPUT_DEVSUBTYPE_GAMEPAD
      out_caps->gamepad.buttons = 0xFFFF;
      out_caps->gamepad.left_trigger = 0xFF;
      out_caps->gamepad.right_trigger = 0xFF;
      out_caps->gamepad.thumb_lx = static_cast<int16_t>(0x7FFF);
      out_caps->gamepad.thumb_ly = static_cast<int16_t>(0x7FFF);
      out_caps->gamepad.thumb_rx = static_cast<int16_t>(0x7FFF);
      out_caps->gamepad.thumb_ry = static_cast<int16_t>(0x7FFF);
    }
    return X_ERROR_SUCCESS;
  }

  X_RESULT SetDeviceVibration(DeviceId id, rex::input::X_INPUT_VIBRATION*) override {
    return id == kScriptDevice ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
  }

  X_RESULT GetDeviceKeystroke(DeviceId id, uint32_t, rex::input::X_INPUT_KEYSTROKE*) override {
    return id == kScriptDevice ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
  }

 private:
  std::vector<Event> events_;
  std::chrono::steady_clock::time_point start_;
  std::mutex mutex_;
  std::vector<uint8_t> logged_;
  uint32_t packet_ = 1;
  uint64_t last_key_ = 0;
  int16_t last_rx_ = 0, last_ry_ = 0;
};

std::vector<Event> LoadScript() {
  std::vector<Event> events;
  std::istringstream inline_script{std::string(REXCVAR_GET(mk9_input_script))};
  for (std::string line; std::getline(inline_script, line, ';');) ParseLine(line, events);
  const std::string path = REXCVAR_GET(mk9_input_script_file);
  if (!path.empty()) {
    std::ifstream file(path);
    if (!file) REXLOG_WARN("[script] cannot open {}", path);
    for (std::string line; std::getline(file, line);) ParseLine(line, events);
  }
  return events;
}

}  // namespace

std::unique_ptr<rex::system::IInputSystem> CreateInputSystem(bool tool_mode) {
  auto input = rex::input::CreateDefaultInputSystem(tool_mode);
  if (!tool_mode) {
    auto events = LoadScript();
    if (!events.empty()) {
      REXLOG_INFO("[script] scripted gamepad on player 1: {} events, last at {:.1f} s", events.size(),
                  events.back().start_s);
      input->AddDriver(std::make_unique<ScriptInputDriver>(std::move(events)));
    }
  }
  return input;
}

}  // namespace mk9::script_input
