// SA2 - native renderer: render targets, copies (resolve and clear) and presentation of the game's
// image.
//
// Each Swap presents the destination of the last copy (RB_COPY_DEST_BASE), and during the videos and the
// title screen there are 2 copies per frame, both with a clear. This part does that without EDRAM: a render
// target is one Vulkan image per (base, format, pitch), and a resolved texture is another image per
// destination address, which is where the Swap looks it up through its fetch constant.

#pragma once

#include "sa2_native_draws.h"

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace rex::memory {
class Memory;
}
namespace rex::ui {
class Presenter;
}
namespace rex::ui::vulkan {
class VulkanDevice;
}

namespace sa2::native {

// Registers a copy needs, taken from the ring sink's register mirror.
struct RegistersCopy {
  uint32_t rb_surface_info = 0;
  uint32_t rb_color_info[4] = {};
  uint32_t rb_depth_info = 0;
  uint32_t rb_copy_control = 0;
  uint32_t rb_copy_dest_base = 0;
  uint32_t rb_copy_dest_pitch = 0;
  uint32_t rb_copy_dest_info = 0;
  uint32_t rb_color_clear = 0;
  uint32_t rb_color_clear_lo = 0;
  uint32_t rb_depth_clear = 0;
  uint32_t pa_sc_window_offset = 0;
  uint32_t pa_sc_window_scissor_tl = 0;
  uint32_t pa_sc_window_scissor_br = 0;
  uint32_t pa_su_sc_mode_cntl = 0;
  uint32_t pa_su_vtx_cntl = 0;
  // Fetch constant 0 as vertices: the copy rectangle (D3D9 puts it there).
  uint32_t fetch_vertices[2] = {};
};

// Fetch constant 0 as a texture at the time of the Swap: VdSwap writes it.
struct TextureSwap {
  uint32_t dword[6] = {};
};

class TargetsNative {
 public:
  // nullptr if the device is missing or presentation could not be set up.
  static std::unique_ptr<TargetsNative> Create(const rex::ui::vulkan::VulkanDevice* vulkan_device,
                                                rex::memory::Memory* memory);
  virtual ~TargetsNative() = default;

  // Resolve and/or clear. false if it could not be done (each cause is logged once).
  virtual bool Copy(const RegistersCopy& registers) = 0;

  // Submits pending work and draws the Swap's texture into the presenter's output. false if that Swap has
  // no resolved texture.
  virtual bool Present(rex::ui::Presenter* presenter, const TextureSwap& swap, uint32_t width,
                         uint32_t height) = 0;

  // 256-entry gamma ramp loaded by the game, 10 bits per channel (red, green, blue). Subsequent outputs
  // apply it the way the Xbox 360 display does. Called from the same thread as Present.
  virtual void RampGamma(const std::array<std::array<uint16_t, 3>, 256>& ramp) = 0;

  // Records a ring draw with its shaders. false if it could not.
  virtual bool Draw(const SubmissionDraw& submission) = 0;
  virtual StatsDraws StatsOfDraws() const = 0;
  // Deferred vertex copies finished: before returning the read pointer to the game.
  virtual void WaitUploads() = 0;
  // Vertex copies queued and not done yet (only to measure the fences).
  virtual size_t PendingCopies() const { return 0; }
  // Copies, clears, presented Swaps and rejected operations, accumulated.
  virtual void Stats(uint64_t& copies, uint64_t& cleared, uint64_t& presented,
                            uint64_t& rejections) const = 0;
  // GPU nanoseconds accumulated per category (kGpuOther..., sa2_native_draws.h).
  virtual void TimeGpuPerCategory(
      std::array<uint64_t, kGpuCategories>& nanoseconds) const = 0;
  // Shaded fragments, vertex invocations and clipped primitives, accumulated per pass category (only
  // with sa2_native_stats_pipeline).
  virtual void StatsPipeline(std::array<uint64_t, kGpuCategories>& fragments,
                                    std::array<uint64_t, kGpuCategories>& vertices,
                                    std::array<uint64_t, kGpuCategories>& primitives) const = 0;
};

}  // namespace sa2::native
