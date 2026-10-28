/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <filesystem>
#include <string_view>

#include <rex/platform.h>
#include <rex/string.h>

namespace rex {

#if REX_PLATFORM_ANDROID
bool InitializeAndroidSystemForApplicationContext();
void ShutdownAndroidSystem();
#endif

// The URL must include the protocol.
void LaunchWebBrowser(const std::string_view url);
void LaunchFileExplorer(const std::filesystem::path& path);

enum class SimpleMessageBoxType {
  Help,
  Warning,
  Error,
};

// This is expected to block the caller until the message box is closed.
// Switch (mk9-nx 1.0.1): a system error dialog (libnx's error applet), after the output's holder (the intro) let go.
void ShowSimpleMessageBox(SimpleMessageBoxType type, std::string_view message);

// mk9-nx 1.0.1: hooks an app may set (system_hooks.cpp). FileOpened: every guest file opened successfully (its guest
// path, e.g. D:\Asset\Core.xxx), from the opening thread. GuestCrash: an unhandled guest access violation, from the
// faulting thread just before the process ends; it must not take locks the game may hold.
using FileOpenedHook = void (*)(std::string_view guest_path);
void SetFileOpenedHook(FileOpenedHook hook);
void NotifyFileOpened(std::string_view guest_path);
using GuestCrashHook = void (*)();
void SetGuestCrashHook(GuestCrashHook hook);
void NotifyGuestCrash();

}  // namespace rex
