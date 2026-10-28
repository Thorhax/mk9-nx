/**
 * @file        core/system_posix.cpp
 * @brief       POSIX implementations of rex/system.h platform helpers.
 *
 * @copyright   Copyright (c) 2026 Tom Clay
 * @license     BSD 3-Clause License
 */

#include <cstdio>
#include <string>

#include <rex/system.h>

#if defined(__SWITCH__)
#include <switch.h>
extern "C" void RexSwitchStopOutputHolder(void);  // rex/ui/windowed_app_context_switch.h (rexui)
#endif

namespace rex {

// TODO(tomc): add linux support for showing a native message box
void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message) {
  const char* level = "INFO";
  switch (type) {
    case SimpleMessageBoxType::Help:
      level = "INFO";
      break;
    case SimpleMessageBoxType::Warning:
      level = "WARNING";
      break;
    case SimpleMessageBoxType::Error:
      level = "ERROR";
      break;
  }
  std::fprintf(stderr, "[%s] %.*s\n", level, static_cast<int>(message.size()), message.data());
  std::fflush(stderr);
#if defined(__SWITCH__)
  // mk9-nx 1.0.1: an error is shown on the console too (before, only the log had it and the player saw a crash).
  // The system's error dialog; its text holds at most 0x800 bytes. The intro lets go of the display first.
  if (type != SimpleMessageBoxType::Error) {
    return;
  }
  RexSwitchStopOutputHolder();
  std::string text(message.substr(0, 0x7F0));
  ErrorApplicationConfig config;
  if (R_SUCCEEDED(errorApplicationCreate(&config, text.c_str(), nullptr))) {
    errorApplicationShow(&config);
  }
#endif
}

}  // namespace rex
