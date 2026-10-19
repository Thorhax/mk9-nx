/**
 * @file        rex/system/static_modules.h
 * @brief       Recompiled guest DLL modules linked into the executable (no dynamic loading)
 *
 * A guest DLL (an .xex the title loads with XexLoadImage) is recompiled into a host shared library that
 * KernelState::LoadUserModule opens by name and asks for its two entry points, ReXModule_Register and
 * ReXModule_GetImageInfo. A platform without dynamic loading (an NRO on the Switch) links the module statically
 * instead and registers those two entry points here, under the shared library name the generated module registry
 * uses ("<project>_<module>"); LoadUserModule looks here first.
 */

#pragma once

#include <rex/image_info.h>
#include <rex/system/function_dispatcher.h>

namespace rex::system {

struct StaticModuleEntry {
  runtime::FunctionDispatcher::RegisterFn register_fn = nullptr;
  const PPCImageInfo* (*get_image_info)() = nullptr;
};

// Registers a statically linked module (call before the title loads it, e.g. from a static initializer).
void RegisterStaticModule(const char* shared_lib_name, runtime::FunctionDispatcher::RegisterFn register_fn,
                          const PPCImageInfo* (*get_image_info)());
// The entry points of a statically linked module, or nullptr.
const StaticModuleEntry* FindStaticModule(const char* shared_lib_name);

}  // namespace rex::system
