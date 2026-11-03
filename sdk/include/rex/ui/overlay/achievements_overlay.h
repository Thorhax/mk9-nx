/**
 * @file        rex/ui/overlay/achievements_overlay.h
 * @brief       Default ImGui achievements overlay dialog.
 *
 * @copyright   Copyright (c) 2026 Rien Gupta <rgupta9@scu.edu>
 *              All rights reserved.
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */
#pragma once

#include <functional>

#include <rex/system/achievement_manager.h>
#include <rex/ui/imgui_dialog.h>
#include <rex/ui/overlay/achievement_icon_cache.h>

namespace rex {
class Runtime;
}  // namespace rex

namespace rex::ui {

class ImmediateDrawer;
class ImmediateTexture;

class AchievementsOverlayDialog : public ImGuiDialog {
 public:
  AchievementsOverlayDialog(ImGuiDrawer* imgui_drawer, ImmediateDrawer* immediate_drawer,
                            rex::Runtime* runtime, rex::system::AchievementManager* achievements);
  ~AchievementsOverlayDialog() override;

  // mk9-nx (2026-11-03): called once when + or the menus' back button is released over the list (its owner closes
  // it; on the Switch it was only L+R+Left again, which players did not know).
  void SetCloseRequest(std::function<void()> close_request) { close_request_ = std::move(close_request); }

 protected:
  void OnDraw(ImGuiIO& io) override;

 private:
  // Lazily loads icon_path, or icons/<image_id>.png when icon_path is empty.
  ImmediateTexture* GetIcon(const rex::system::AchievementInfo& achievement);

  rex::system::AchievementManager* achievements_ = nullptr;
  AchievementIconCache icon_cache_;
  std::function<void()> close_request_;
  bool close_requested_ = false;
};

}  // namespace rex::ui
