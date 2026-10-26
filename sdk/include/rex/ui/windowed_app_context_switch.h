/**
 * @file        rex/ui/windowed_app_context_switch.h
 * @brief       UI loop for Nintendo Switch on top of the libnx applet
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#pragma once

#include <memory>
#include <vector>

#include <rex/ui/windowed_app_context.h>

namespace rex::ui {

class WindowSwitch;

// The UI thread is the main thread. It sleeps until there is work: functions
// posted from other threads, a paint request, an applet message (focus, HOME,
// dock, exit request) or a change of the display resolution.
// Output hold (any thread). While something else uses the display and the audio device (a startup video drawn with a
// libnx framebuffer and audout), the windows make no surface and the audio output plays nothing: the game runs as usual
// but is neither shown nor heard, and its controllers read as untouched. RexSwitchReleaseOutput, called once the
// holder has closed both, gives the windows their surfaces back on the UI thread and lets the audio output open.
extern "C" void RexSwitchHoldOutput(void);
extern "C" void RexSwitchReleaseOutput(void);
extern "C" bool RexSwitchOutputHeld(void);

class SwitchWindowedAppContext final : public WindowedAppContext {
 public:
  SwitchWindowedAppContext();
  ~SwitchWindowedAppContext() override;

  // Hooks the applet messages. Must be called, and succeed, before any other
  // use.
  bool Initialize();

  void NotifyUILoopOfPendingFunctions() override;
  void PlatformQuitFromUIThread() override;

  int RunMainMessageLoop();

  // Any thread.
  void WakeUILoop() { NotifyUILoopOfPendingFunctions(); }

  // UI thread only.
  void RegisterWindow(WindowSwitch* window);
  void UnregisterWindow(WindowSwitch* window);

  // Called by the applet hook, on the UI thread, from inside appletMainLoop.
  void OnAppletHook(int hook_type);

  // UI thread: gives every window its surface again (RexSwitchReleaseOutput).
  void ReconnectWindowSurfaces();

 private:
  // Holds libnx types, which stay out of this header.
  struct Impl;

  bool IsRegistered(const WindowSwitch* window) const;
  void ProcessQuitRequest();

  std::unique_ptr<Impl> impl_;
  std::vector<WindowSwitch*> windows_;
  bool display_changed_ = false;
  bool focus_changed_ = false;
  bool quit_requested_ = false;
};

}  // namespace rex::ui
