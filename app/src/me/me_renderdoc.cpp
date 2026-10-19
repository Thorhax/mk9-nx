// banjo-nx - RenderDoc frame capture on demand (PC diagnostic, banjo_renderdoc_capture_s).
//
// The SDK looks for RenderDoc before vkCreateInstance, when a layer enabled through VK_INSTANCE_LAYERS has not loaded
// renderdoc.dll yet; by the time a capture is wanted the layer is in the process, so its in-application API is taken
// from the loaded module here.

#include <cstdint>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <renderdoc_app.h>
#endif

namespace banjo::renderdoc {

#if defined(_WIN32)
static RENDERDOC_API_1_0_0* Api() {
  HMODULE module = GetModuleHandleA("renderdoc.dll");
  if (!module) return nullptr;
  auto get_api = reinterpret_cast<pRENDERDOC_GetAPI>(GetProcAddress(module, "RENDERDOC_GetAPI"));
  RENDERDOC_API_1_0_0* api = nullptr;
  if (!get_api || !get_api(eRENDERDOC_API_Version_1_0_0, reinterpret_cast<void**>(&api))) return nullptr;
  return api;
}
#endif

// Starts capturing everything submitted from now on (any device, any window): the native renderer ends it at the
// next guest Swap, so the capture holds exactly one game frame even with the window minimized (no present needed).
bool Start(const char* path_template) {
#if defined(_WIN32)
  RENDERDOC_API_1_0_0* api = Api();
  if (!api) return false;
  api->SetLogFilePathTemplate(path_template);
  api->StartFrameCapture(nullptr, nullptr);
  return true;
#else
  (void)path_template;
  return false;
#endif
}

bool End() {
#if defined(_WIN32)
  RENDERDOC_API_1_0_0* api = Api();
  return api && api->EndFrameCapture(nullptr, nullptr);
#else
  return false;
#endif
}

}  // namespace banjo::renderdoc
