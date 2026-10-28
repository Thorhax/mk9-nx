// Mortal Kombat Komplete Edition: the ReXGlue application object (paths, GPU plugin choice, analysis aids).

#pragma once

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <optional>
#include <system_error>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>
#include <rex/rex_app.h>
#include <rex/runtime.h>

#include "cpu/sa2_enhancements.h"
#include "me/me_native_system.h"
#include "sa2_capture.h"
#include "mk9_script_input.h"
#include "mk9_snapshot.h"
#include "mk9_hang_watch.h"
#include "mk9_file_check.h"
#include "thorhax_intro.h"  // app/thorhax_intro (from usf4-nx)

// mk9-nx (2026-10-20): Renut's Direct3D-level renderer (src/renut_engine/gpu, from reNut by masterspike52 through
// banjo-nx and sa2-nx), chosen at boot with sa2_renderer = "renut" (docs/mk9-renut-census.md).
#if SA2_RENUT_RENDERER
#include "renut_engine/gpu/banjo_renut_select.h"
#include "renut_engine/gpu/overlay/app_integration.h"
#include "renut_engine/gpu/banjo_renut_capture.h"
#endif

#if defined(MK9_BUILTIN_CONFIG)
#include "mk9_builtin_config.h"  // generated from app/mk9_switch_defaults.toml (CMake)
#endif

class Mk9App : public rex::ReXApp {
 public:
  using rex::ReXApp::ReXApp;

  static std::unique_ptr<rex::ui::WindowedApp> Create(rex::ui::WindowedAppContext& ctx) {
    return std::unique_ptr<Mk9App>(new Mk9App(ctx, "mk9", PPCImageConfig));
  }

  // The Switch build's fixed settings (app/mk9_switch_defaults.toml, compiled in): applied before mk9.toml, which
  // holds only the player's options (and may still override any of these).
  std::string_view BuiltInConfig() const override {
#if defined(MK9_BUILTIN_CONFIG)
    return kMk9BuiltInConfig;
#else
    return {};
#endif
  }
  // Without a mk9.toml next to the NRO, the player's default one is written there (app/mk9_switch.toml).
  std::string_view DefaultConfigFile() const override {
#if defined(MK9_BUILTIN_CONFIG)
    return kMk9PlayerConfig;
#else
    return {};
#endif
  }

  void OnPostInitLogging() override {
    // The runtime defaults to no GPU plugin: select ReXGlue's Xenos implementation (stage 3, emulated GPU), keeping
    // an explicit command-line or config-file choice.
    if (!rex::cvar::HasNonDefaultValue("gpu_plugin")) {
      rex::cvar::SetFlagByName("gpu_plugin", "xenos");
    }
    // Which renderer enhancements run (sa2-nx's switchable set), in one line; before any guest thread exists.
    sa2::enh::Resolve();
  }

  // Without --game_data_root, use a game_root/ folder next to the executable, and on a PC build tree the
  // repository's assets/game_root. The disc names the executable Default.xex. An explicit path always wins.
  void OnConfigurePaths(rex::PathConfig& paths) override {
    if (!paths.game_data_root.empty()) {
      return;
    }
    std::error_code ec;
    const auto folder = rex::filesystem::GetExecutableFolder();
    if (folder.empty()) {
      return;
    }
    for (auto candidate = folder; !candidate.empty(); candidate = candidate.parent_path()) {
      for (const auto& sub : {std::filesystem::path("game_root"),
                              std::filesystem::path("assets") / "game_root"}) {
        auto pack = candidate / sub;
        pack += ".nxpack";
        if (std::filesystem::is_regular_file(candidate / sub / "Default.xex", ec) ||
            std::filesystem::is_regular_file(candidate / sub / "default.xex", ec) ||
            std::filesystem::is_regular_file(pack, ec)) {
          paths.game_data_root = candidate / sub;
          return;
        }
      }
      if (candidate == candidate.parent_path()) {
        break;
      }
    }
  }

  // Input: the SDK's default devices plus, for tests, the scripted gamepad (mk9_input_script[_file]).
  // Stage 4: with sa2_renderer_native = true the graphics system is the native Vulkan renderer (src/me), and the
  // emulated Xenos GPU plugin is not loaded (ReXApp only loads it when config.graphics is empty).
  void OnPreSetup(rex::RuntimeConfig& config) override {
    // Switch (1.0.1): the game files against the release's list; a bad one is named on screen and the game ends here.
    mk9::file_check::AtStart(game_data_root());
    // The Thorhax intro video, while the game boots behind it (nothing has the window or the audio device yet).
    thorhax::intro::Start();
    mk9::file_check::TestAfterIntroStarted();
    config.input_factory = mk9::script_input::CreateInputSystem;
    sa2::enh::ApplyGpuBoost();
#if SA2_RENUT_RENDERER
    if (banjo::renut_gpu::Active()) {
      // No SDK graphics system and no GPU plugin: the renderer is the app's (its own window swapchain and thread).
      renut::gpu::ConfigureRuntime(config);
      REXLOG_INFO("graphics: Renut's Direct3D-level renderer (sa2_renderer = renut: no PM4 ring, no EDRAM)");
      return;
    }
#endif
    if (me::native::Enabled()) {
      config.graphics = me::native::CreateGraphicsSystem();
      REXLOG_INFO("graphics: native Vulkan renderer (PM4 ring)");
    }
  }

#if SA2_RENUT_RENDERER
  // Renut's renderer draws the SDK's menus through its own plume drawer, resizes its swapchain itself, and starts once
  // the window exists (here, before the game: the paths are final). As sa2-nx 1.0.3's Sa2App.
  std::unique_ptr<rex::ui::ImmediateDrawer> OnCreateImmediateDrawer() override {
    if (banjo::renut_gpu::Active()) return renut::gpu::CreateOverlayDrawer();
    return rex::ReXApp::OnCreateImmediateDrawer();
  }
  void OnWindowPixelSizeChanged(uint32_t pixel_width, uint32_t pixel_height) override {
    rex::ReXApp::OnWindowPixelSizeChanged(pixel_width, pixel_height);
    if (banjo::renut_gpu::Active()) renut::gpu::OnWindowResized();
  }
  std::optional<rex::PathConfig> OnFinalizePaths(const rex::PathConfig& defaults,
                                                 std::function<void(rex::PathConfig)> resume) override {
    if (banjo::renut_gpu::Active()) renut::gpu::StartRenderer(app_context(), [this] { return imgui_drawer(); });
    return rex::ReXApp::OnFinalizePaths(defaults, std::move(resume));
  }
#endif

  // Test snapshots of the game image (mk9_snapshot_s), read from the presenter's guest output.
  void OnPostSetup() override {
    mk9::hang_watch::Start();  // Switch: thread stacks in logs/rex/rex_crash.log when no frame comes (mk9_hang_dump_s)
    // Every 10 s: how often each active enhancement was used (a log shows they really ran).
    sa2::enh::StartReport();
    // Short video recordings of the game image (sa2_video_s; sa2-nx's recorder, lockstep with vblank_per_swap): raw
    // frames + a per-frame XXH3 of the full-resolution image, to compare two runs frame for frame.
#if SA2_RENUT_RENDERER
    if (banjo::renut_gpu::Active()) {
      auto grab = [](rex::ui::RawImage& image) { return renut::gpu::CaptureFrame(image); };
      sa2::capture::StartWithSource(grab);
      mk9::snapshot::StartWithSource(grab);
      return;
    }
#endif
    sa2::capture::Start([this]() -> rex::ui::Presenter* {
      const auto* rt = runtime();
      const auto* graphics = rt ? rt->graphics_system() : nullptr;
      return graphics ? graphics->presenter() : nullptr;
    });
    mk9::snapshot::Start([this]() -> rex::ui::Presenter* {
      const auto* rt = runtime();
      const auto* graphics = rt ? rt->graphics_system() : nullptr;
      return graphics ? graphics->presenter() : nullptr;
    });
  }

  void OnShutdown() override {
#if SA2_RENUT_RENDERER
    if (banjo::renut_gpu::Active()) renut::gpu::ShutdownRenderer();
#endif
    sa2::capture::Stop();
    mk9::snapshot::Stop();
  }

  // Analysis aid: with MK9_DUMP_IMAGE=<file>, write the loaded guest image (image_base..image_base+image_size) so
  // code pointers stored in data can be scanned for (tools/pointer_scan.py). Does nothing unless the variable is set.
  void OnPostLoadXexImage() override {
    if (!runtime() || !runtime()->memory()) {
      return;
    }
    const char* path = std::getenv("MK9_DUMP_IMAGE");
    if (!path) {
      return;
    }
    const uint8_t* image =
        runtime()->memory()->TranslateVirtual<const uint8_t*>(PPCImageConfig.image_base);
    if (std::FILE* f = std::fopen(path, "wb")) {
      std::fwrite(image, 1, PPCImageConfig.image_size, f);
      std::fclose(f);
      REXLOG_INFO("guest image written to {}", path);
    }
  }
};
