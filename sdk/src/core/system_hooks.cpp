/**
 * @file        core/system_hooks.cpp
 * @brief       The app's hooks of rex/system.h (mk9-nx 1.0.1: the game file check, src/mk9_file_check.cpp there).
 */

#include <atomic>

#include <rex/system.h>

namespace rex {

namespace {
std::atomic<FileOpenedHook> g_file_opened{nullptr};
std::atomic<GuestCrashHook> g_guest_crash{nullptr};
}  // namespace

void SetFileOpenedHook(FileOpenedHook hook) { g_file_opened.store(hook, std::memory_order_release); }

void NotifyFileOpened(std::string_view guest_path) {
  if (FileOpenedHook hook = g_file_opened.load(std::memory_order_acquire)) {
    hook(guest_path);
  }
}

void SetGuestCrashHook(GuestCrashHook hook) { g_guest_crash.store(hook, std::memory_order_release); }

void NotifyGuestCrash() {
  // Once: the report can come from more than one faulting thread.
  if (GuestCrashHook hook = g_guest_crash.exchange(nullptr, std::memory_order_acq_rel)) {
    hook();
  }
}

}  // namespace rex
