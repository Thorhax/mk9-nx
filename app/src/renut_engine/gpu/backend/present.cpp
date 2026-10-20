/**
 * @file    gpu/backend/present.cpp
 * @brief   End of frame: swapchain acquire, guest image scan-out, overlay,
 *          submit and present.
 */
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include <plume_render_interface.h>

#include <rex/cvar.h>

#include "renut_engine/gpu/backend/backend.h"
#include "renut_engine/gpu/backend/gpu_profile.h"
#include "renut_engine/gpu/render/draw.h"
#include "renut_engine/gpu/render/textures.h"
#include "renut_engine/gpu/log.h"
#include "renut_engine/gpu/overlay/overlay_ui.h"
#include "renut_engine/gpu/overlay/overlay_drawer.h"  // banjo-nx (2026-10-19): HasLatest
#include "renut_engine/gpu/banjo_renut_tuning.h"
#include "renut_engine/gpu/backend/render_thread.h"
#include "renut_engine/gpu/backend/video.h"
#include "renut_engine/gpu/backend/video_internal.h"
#include "renut_engine/gpu/banjo_renut_capture.h"
#if defined(PLUME_DEKO3D)
#include <plume_deko3d.h>  // banjo-nx: banjo_renut_capture_raw's format check
#endif

// banjo-nx (2026-10-17, deko3d renderer tests): the captures copy the game's frame texture as it is (4-byte formats only,
// its channels as BGRA), without the scan-out pass: which of the two a wrong capture comes from.
REXCVAR_DEFINE_BOOL(banjo_renut_capture_raw, false, "Banjo/Renut",
                    "Test: frame captures copy the game's frame texture itself instead of the scanned-out picture "
                    "(deko3d builds, 4-byte formats)");

// banjo-nx (2026-10-18, the deko3d depth of field): one of the game's textures on screen and in the captures in place of
// the frame, to compare a pass's inputs between the back ends (the [trace] resolve lines name the headers).
REXCVAR_DEFINE_STRING(banjo_renut_diag_show_texture, "", "Banjo/Renut",
                      "Diagnostic: show the host texture of this guest texture header (hex address, as the [trace] "
                      "resolve lines name them) instead of the frame, on screen and in captures");
REXCVAR_DEFINE_DOUBLE(banjo_renut_diag_show_gamma, 1.0, "Banjo/Renut",
                      "Diagnostic: banjo_renut_diag_show_texture's exponent (depth near 1 shows with 64 or so)");

// sa2-nx (2026-10-20): SA2's display gamma ramp at scan-out. Its Direct3D sets the ramp once at boot through the
// display's curve (VdGetCurrentDisplayGamma: [64] = 193, [128] = 462 of 1023, not the identity), and the Xbox 360's
// display controller applies it to every pixel it sends; the emulated GPU and sa2-nx's native renderer do the same
// (DC_LUT registers). Renut presented without it: every frame came out lighter than both (12-16 levels in the
// midtones, renut-renderer.md). Same lookup as theirs: each channel's 8-bit value picks its entry.
REXCVAR_DEFINE_BOOL(sa2_renut_gamma_ramp, true, "SA2/Renderer",
                    "Renut's renderer: apply the game's display gamma ramp at scan-out, as the Xbox 360's display "
                    "controller, the emulated GPU and the native renderer do (off: the frame as rendered, lighter)");

REXCVAR_DEFINE_BOOL(renut_vsync, true, "Nuts&Bolts/Graphics",
                    "Show each frame at the display's refresh: no tearing, and the game runs at "
                    "the refresh rate (frame_cap then only limits below it). Off shows frames at "
                    "once, with tearing.");

namespace renut::gpu {

namespace {

// renut_vsync onto the swapchain (both backends start with it on). Vulkan
// makes the change by recreating the swapchain, which needsResize asks for.
void ApplyVsync(VideoState &s) {
  static bool applied = true;
  const bool vsync = REXCVAR_GET(renut_vsync);
  if (vsync == applied)
    return;
  s.swap_chain->setVsyncEnabled(vsync);
  applied = vsync;
  RGPU_INFO("Vsync {}", vsync ? "on" : "off");
}

void RebuildSwapChain(VideoState &s) {
  // Submit rather than abandon an open list: plume tracks layouts on record.
  if (s.list_open)
    detail::SubmitAndAdvance(s, nullptr, nullptr);
  detail::WaitAllSubmitted(s);
  s.swap_framebuffers.clear();
  if (!s.swap_chain->resize() || !detail::BuildSwapChainTargets(s)) {
    if (s.swap_chain->getWidth() && s.swap_chain->getHeight())
      RGPU_ERROR("Swapchain resize failed");  // 0x0 while minimized is expected
  }
}

// The most of the game's frame trimmed off either axis to fill the output.
constexpr float kMaxTrim = 0.15f;

struct FrameFit {
  plume::RenderViewport viewport;
  float visible[2] = {1.0f, 1.0f};  // the share of the frame shown on each axis
};

// The frame fills the output (a window that is not the frame's shape, say a
// maximized one) without bars: it covers it, keeping its proportions, and the
// part that overflows is trimmed evenly from both edges. Past kMaxTrim, the
// rest is fitted with bars.
FrameFit FitFrame(uint32_t out_w, uint32_t out_h, uint32_t src_w, uint32_t src_h) {
  FrameFit fit;
  fit.viewport = plume::RenderViewport(0.0f, 0.0f, float(out_w), float(out_h));
  if (src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0)
    return fit;
  const float out_aspect = float(out_w) / float(out_h);
  float src_aspect = float(src_w) / float(src_h);
  if (src_aspect > out_aspect) {
    fit.visible[0] = std::max(out_aspect / src_aspect, 1.0f - kMaxTrim);
    src_aspect *= fit.visible[0];
  } else {
    fit.visible[1] = std::max(src_aspect / out_aspect, 1.0f - kMaxTrim);
    src_aspect /= fit.visible[1];
  }
  float w = float(out_w), h = float(out_h);
  if (src_aspect > out_aspect)
    h = w / src_aspect;
  else
    w = h * src_aspect;
  fit.viewport = plume::RenderViewport((float(out_w) - w) * 0.5f, (float(out_h) - h) * 0.5f, w, h);
  return fit;
}

// sa2-nx (2026-10-20, sa2_renut_gamma_ramp): the game's display gamma ramp (Video::SetGammaRamp), a 256 x 1 float
// texture in a bindless slot that the present shader reads.
struct GammaRamp {
  // The game thread's last copy (SetGammaRamp's compare, that thread only).
  uint8_t seen[1536] = {};
  bool seen_any = false;
  // Handed to the render thread.
  std::mutex mutex;
  uint8_t words[1536] = {};
  std::atomic<uint64_t> version{0};
  // Render thread, under State().mutex.
  uint64_t uploaded = 0;  // the version the texels below hold
  float texels[256 * 4] = {};
  std::unique_ptr<plume::RenderTexture> texture;
  std::unique_ptr<plume::RenderTextureView> view;
  uint32_t slot = kInvalidDescriptor;
};
GammaRamp g_gamma;

constexpr plume::RenderFormat kGammaRampFormat = plume::RenderFormat::R32G32B32A32_FLOAT;

// Before the frame's scan-out: the ramp goes to the texture. Every frame (4 KB), not only when the game changes it
// (SA2: once, at boot): banjo-nx found a frame's command list can be dropped at start-up, and its one upload with it
// (its run 78: black output until the ramp changed again).
void UpdateGammaRampLocked(VideoState &s) {
  GammaRamp &g = g_gamma;
  const uint64_t version = g.version.load(std::memory_order_acquire);
  if (!version || !REXCVAR_GET(sa2_renut_gamma_ramp))
    return;
  if (!g.texture) {
    g.texture = s.device->createTexture(plume::RenderTextureDesc::Texture2D(256, 1, 1, kGammaRampFormat));
    if (!g.texture)
      return;
    g.texture->setName("sa2-nx gamma ramp");
    g.view = g.texture->createTextureView(plume::RenderTextureViewDesc::Texture2D(kGammaRampFormat));
    g.slot = Video::AllocateTextureSlotLocked();
    if (!g.view || g.slot == kInvalidDescriptor) {
      g.view.reset();
      g.texture.reset();
      g.slot = kInvalidDescriptor;
      return;
    }
    Video::SetTextureSlotLocked(g.slot, g.texture.get(), g.view.get());
  }
  if (!Video::EnsureCommandListLocked())
    return;
  UploadAllocation staging = s.uploads[s.frame].Allocate(256 * 16, 512);
  if (!staging)
    return;
  const bool changed = version != g.uploaded;
  if (changed) {
    uint8_t words[1536];
    {
      std::lock_guard lock(g.mutex);
      std::memcpy(words, g.words, sizeof(words));
    }
    // The native renderer's table (sa2_native_targets.cpp RecomputeRamp): the 10-bit value / 1023, alpha 1.
    for (uint32_t i = 0; i < 256; ++i) {
      for (uint32_t c = 0; c < 3; ++c) {
        const uint8_t *w = words + (c * 256 + i) * 2;
        const uint32_t value = ((uint32_t(w[0]) << 8 | w[1]) >> 6) & 0x3FF;
        g.texels[i * 4 + c] = float(value) / 1023.0f;
      }
      g.texels[i * 4 + 3] = 1.0f;
    }
  }
  std::memcpy(staging.cpu, g.texels, sizeof(g.texels));
  const float *texels = g.texels;
  plume::RenderCommandList *cmd = s.command_list;
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderTextureBarrier(g.texture.get(), plume::RenderTextureLayout::COPY_DEST));
  cmd->copyTextureRegion(plume::RenderTextureCopyLocation::Subresource(g.texture.get(), 0),
                         plume::RenderTextureCopyLocation::PlacedFootprint(staging.ref.ref, kGammaRampFormat, 256, 1,
                                                                           1, 256, staging.ref.offset));
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(g.texture.get(), plume::RenderTextureLayout::SHADER_READ));
  if (changed)
    RGPU_INFO("sa2-nx: the game's display gamma ramp at scan-out (sa2_renut_gamma_ramp): [0] {:.0f}, [64] {:.0f}, "
              "[128] {:.0f}, [255] {:.0f} of 1023 (red)",
              texels[0] * 1023.0f, texels[64 * 4] * 1023.0f, texels[128 * 4] * 1023.0f, texels[255 * 4] * 1023.0f);
  g.uploaded = version;
}

void RecordScanOut(VideoState &s, const PresentImage &image, uint32_t out_w, uint32_t out_h) {
  plume::RenderCommandList *cmd = s.command_list;
  if (image.layout && *image.layout != plume::RenderTextureLayout::SHADER_READ) {
    cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                  plume::RenderTextureBarrier(image.texture, plume::RenderTextureLayout::SHADER_READ));
    *image.layout = plume::RenderTextureLayout::SHADER_READ;
  }

  const uint32_t src_w = image.source_width ? image.source_width : image.width;
  const uint32_t src_h = image.source_height ? image.source_height : image.height;
  const FrameFit fit = FitFrame(out_w, out_h, src_w, src_h);
  const plume::RenderViewport &vp = fit.viewport;
  cmd->setViewports(vp);
  cmd->setScissors(plume::RenderRect(int32_t(vp.x), int32_t(vp.y), int32_t(vp.x + vp.width),
                                     int32_t(vp.y + vp.height)));

  cmd->setGraphicsPipelineLayout(s.utility_layout.get());
  cmd->setPipeline(s.present_pipeline.get());
  cmd->setGraphicsDescriptorSet(s.texture_set.get(), 0);
  cmd->setGraphicsDescriptorSet(s.sampler_set.get(), 1);

  UtilityPush push{};
  // The frame's share of the texture, less what the fit trims from its edges.
  const float frame_u = image.width ? float(src_w) / float(image.width) : 1.0f;
  const float frame_v = image.height ? float(src_h) / float(image.height) : 1.0f;
  push.scale[0] = frame_u * fit.visible[0];
  push.scale[1] = frame_v * fit.visible[1];
  push.translate[0] = frame_u * (1.0f - fit.visible[0]) * 0.5f;
  push.translate[1] = frame_v * (1.0f - fit.visible[1]) * 0.5f;
  push.texture_index = image.descriptor;
  push.sampler_index = kLinearClampSamplerSlot;
  push.param0 = image.gamma > 0.0f ? image.gamma : s.output_gamma;
  // sa2-nx: the game's gamma ramp (not on banjo_renut_diag_show_texture's texture): its slot as param1's bits, 0 =
  // none (slots 0-2 are the null textures, never the ramp's).
  const uint32_t ramp_slot = image.gamma <= 0.0f && g_gamma.uploaded && g_gamma.slot != kInvalidDescriptor &&
                                     REXCVAR_GET(sa2_renut_gamma_ramp)
                                 ? g_gamma.slot
                                 : 0u;
  push.param1 = std::bit_cast<float>(ramp_slot);
  cmd->setGraphicsPushConstants(0, &push, 0, sizeof(push));
  cmd->drawInstanced(3, 1, 0, 0);
}

// banjo-nx: the test captures (banjo_renut_capture.h). CaptureFrame asks; the next Present, minimized or not, scans
// the guest image out once more into an offscreen target of the frame's size and reads it back.
struct FrameCapture {
  std::mutex mutex;
  std::condition_variable cv;
  std::atomic<uint64_t> requested{0};
  uint64_t served = 0;  // under mutex
  bool ok = false;
  rex::ui::RawImage image;
  // Render thread only. Kept between captures of one size (the recording takes one per frame); never in flight
  // between them, as each capture waits for the GPU.
  std::unique_ptr<plume::RenderTexture> target;
  std::unique_ptr<plume::RenderFramebuffer> framebuffer;
  std::unique_ptr<plume::RenderBuffer> readback;
  uint32_t width = 0, height = 0, row_texels = 0;
  uint64_t served_render = 0;
};
FrameCapture g_capture;

// The present pipeline's format (kSwapChainFormat in video.cpp).
constexpr plume::RenderFormat kCaptureFormat = plume::RenderFormat::B8G8R8A8_UNORM;

bool CaptureFrameLocked(VideoState &s, const PresentImage &image, rex::ui::RawImage &out) {
  const uint32_t w = image.source_width ? image.source_width : image.width;
  const uint32_t h = image.source_height ? image.source_height : image.height;
  if (!w || !h || !Video::EnsureCommandListLocked())
    return false;
  FrameCapture &c = g_capture;
  if (!c.target || c.width != w || c.height != h) {
    c.framebuffer.reset();
    c.target.reset();
    c.readback.reset();
    plume::RenderTextureDesc desc = plume::RenderTextureDesc::Texture2D(w, h, 1, kCaptureFormat);
    desc.flags |= plume::RenderTextureFlag::RENDER_TARGET;
    c.target = s.device->createTexture(desc);
    if (!c.target)
      return false;
    const plume::RenderTexture *attachment = c.target.get();
    c.framebuffer = s.device->createFramebuffer(plume::RenderFramebufferDesc(&attachment, 1));
    c.row_texels = (w + 63) & ~63u;  // D3D12 copies rows of 256-byte multiples
    c.readback = s.device->createBuffer(plume::RenderBufferDesc::ReadbackBuffer(uint64_t(c.row_texels) * h * 4));
    if (!c.framebuffer || !c.readback) {
      c.framebuffer.reset();
      c.target.reset();
      c.readback.reset();
      return false;
    }
    c.width = w;
    c.height = h;
  }
  plume::RenderCommandList *cmd = s.command_list;
  bool raw = false;
  bool raw_rgba = false;  // the frame texture's bytes are R G B A (else B G R A, as the scan-out's readback)
#if defined(PLUME_DEKO3D)
  if (REXCVAR_GET(banjo_renut_capture_raw) && IsDeko3D()) {
    const auto *t = static_cast<const plume::Deko3DTexture *>(image.texture);
    if (t && t->valid) {
      switch (t->desc.format) {
      case plume::RenderFormat::R8G8B8A8_TYPELESS:
      case plume::RenderFormat::R8G8B8A8_UNORM:
        raw = raw_rgba = true;
        break;
      case plume::RenderFormat::B8G8R8A8_UNORM:
        raw = true;
        break;
      default:
        break;
      }
    }
    static int logged = 0;
    if (t && logged++ < 4)
      RGPU_INFO("banjo_renut_capture_raw: frame texture {}x{} format {}{}", t->desc.width, t->desc.height,
                int(t->desc.format), raw ? "" : ": not 8-bit RGBA or BGRA, the scan-out is captured");
  }
#endif
  if (raw) {
    cmd->barriers(plume::RenderBarrierStage::COPY,
                  plume::RenderTextureBarrier(image.texture, plume::RenderTextureLayout::COPY_SOURCE));
    if (image.layout)
      *image.layout = plume::RenderTextureLayout::COPY_SOURCE;
    const plume::RenderBox box(0, 0, int32_t(w), int32_t(h));
    cmd->copyTextureRegion(
        plume::RenderTextureCopyLocation::PlacedFootprint(c.readback.get(), kCaptureFormat, w, h, 1, c.row_texels),
        plume::RenderTextureCopyLocation::Subresource(image.texture), 0, 0, 0, &box);
  } else {
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(c.target.get(), plume::RenderTextureLayout::COLOR_WRITE));
  cmd->setFramebuffer(c.framebuffer.get());
  RecordScanOut(s, image, w, h);
  cmd->setFramebuffer(nullptr);
  cmd->barriers(plume::RenderBarrierStage::COPY,
                plume::RenderTextureBarrier(c.target.get(), plume::RenderTextureLayout::COPY_SOURCE));
  cmd->copyTextureRegion(
      plume::RenderTextureCopyLocation::PlacedFootprint(c.readback.get(), kCaptureFormat, w, h, 1, c.row_texels),
      plume::RenderTextureCopyLocation::Subresource(c.target.get()));
  }
  // Submits the frame's work with the capture and waits; the next list starts with nothing bound.
  detail::SubmitAndWaitLocked(s);
  const auto *data = static_cast<const uint8_t *>(c.readback->map());
  if (!data)
    return false;
  out.width = w;
  out.height = h;
  out.stride = size_t(w) * 4;
  out.data.resize(size_t(w) * h * 4);
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t *row = data + size_t(y) * c.row_texels * 4;
    uint8_t *dest = out.data.data() + size_t(y) * w * 4;
    // B G R A -> R G B X; a raw RGBA frame texture is in order already (before 2026-10-17 it was swapped too: every
    // raw capture of the Ryujinx runs had red and blue exchanged, the screen itself did not).
    const uint32_t r = raw_rgba ? 0 : 2, b = raw_rgba ? 2 : 0;
    for (uint32_t x = 0; x < w; ++x) {
      dest[x * 4 + 0] = row[x * 4 + r];
      dest[x * 4 + 1] = row[x * 4 + 1];
      dest[x * 4 + 2] = row[x * 4 + b];
      dest[x * 4 + 3] = 0xFF;
    }
  }
  c.readback->unmap();
  return true;
}

// From PresentLocked: one relaxed load when nobody asked.
void ServeCaptureLocked(VideoState &s, const PresentImage *image) {
  FrameCapture &c = g_capture;
  const uint64_t requested = c.requested.load(std::memory_order_acquire);
  if (requested == c.served_render || !image || !image->texture || image->descriptor == kInvalidDescriptor)
    return;
  rex::ui::RawImage out;
  const bool ok = CaptureFrameLocked(s, *image, out);
  c.served_render = requested;
  {
    std::lock_guard lock(c.mutex);
    c.image = std::move(out);
    c.ok = ok;
    c.served = requested;
  }
  c.cv.notify_all();
}

// banjo_renut_diag_show_texture: the texture to show instead of the frame, when it has a 2D host texture and slot.
bool DiagTextureImageLocked(PresentImage &out) {
  static const u32 va = [] {
    const std::string v = REXCVAR_GET(banjo_renut_diag_show_texture);
    return v.empty() ? 0u : u32(std::strtoul(v.c_str(), nullptr, 16));
  }();
  if (!va)
    return false;
  HostTexture *t = FindHostTextureLocked(va);
  if (!t || !t->texture || t->slot == UINT32_MAX || t->slot == kInvalidDescriptor ||
      t->dimension != TextureDimension::Tex2D || t->stacked)
    return false;
  out.texture = t->texture.get();
  out.layout = &t->layout;
  out.descriptor = t->slot;
  out.width = t->alloc_width;
  out.height = t->alloc_height;
  out.gamma = float(REXCVAR_GET(banjo_renut_diag_show_gamma));
  return true;
}

// Records the back-buffer pass and presents. Caller holds s.mutex.
void PresentLocked(VideoState &s, const PresentImage *image) {
  // Logs, once, why the GPU stopped if an earlier submission faulted.
  ReportDeviceLost(s.device.get());
  PresentImage diag_image;
  if (image && DiagTextureImageLocked(diag_image))
    image = &diag_image;
  ApplyVsync(s);
  if (s.swap_chain->needsResize() || s.resize_requested.exchange(false, std::memory_order_acq_rel))
    RebuildSwapChain(s);
  UpdateGammaRampLocked(s);  // sa2-nx (2026-10-20): before the capture and the scan-out, which both apply it
  ServeCaptureLocked(s, image);  // banjo-nx: test captures (banjo_renut_capture.h)
  if (s.swap_framebuffers.empty()) {
    // Minimized or failed resize: keep the frame ring moving without a present.
    if (s.list_open)
      detail::SubmitAndAdvance(s, nullptr, nullptr);
    return;
  }

  plume::RenderCommandSemaphore *acquire = s.acquire_semaphores[s.frame].get();
  uint32_t image_index = 0;
  auto acquire_image = [&] {
    render_thread::WaitScope wait;
    return s.swap_chain->acquireTexture(acquire, &image_index);
  };
  if (!acquire_image()) {
    RebuildSwapChain(s);
    acquire = s.acquire_semaphores[s.frame].get();
    if (s.swap_framebuffers.empty() || !acquire_image()) {
      if (s.list_open)
        detail::SubmitAndAdvance(s, nullptr, nullptr);
      return;
    }
  }

  if (!Video::EnsureCommandListLocked())
    return;
  static const uint32_t kProfileLabel = gpu_profile::Label("present and overlays");
  gpu_profile::Mark(s, kProfileLabel);
  plume::RenderCommandList *cmd = s.command_list;
  plume::RenderTexture *back = s.swap_chain->getTexture(image_index);
  plume::RenderFramebuffer *back_fb = s.swap_framebuffers[image_index].get();
  const uint32_t out_w = s.swap_chain->getWidth();
  const uint32_t out_h = s.swap_chain->getHeight();

  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(back, plume::RenderTextureLayout::COLOR_WRITE));
  cmd->setFramebuffer(back_fb);
  const bool scan_out = image && image->texture && image->descriptor != kInvalidDescriptor;
  // Letterbox bars (and the whole output when there is no guest image). banjo-nx (2026-10-19, GPU): not when the
  // frame covers the whole output, where the scan-out writes every pixel the clear would.
  bool covers = false;
  if (scan_out) {
    const FrameFit fit = FitFrame(out_w, out_h, image->source_width ? image->source_width : image->width,
                                  image->source_height ? image->source_height : image->height);
    covers = fit.viewport.x <= 0.0f && fit.viewport.y <= 0.0f && fit.viewport.x + fit.viewport.width >= float(out_w) &&
             fit.viewport.y + fit.viewport.height >= float(out_h);
  }
  if (!covers)
    cmd->clearColor(0, plume::RenderColor(0.0f, 0.0f, 0.0f, 1.0f));

  if (scan_out)
    RecordScanOut(s, *image, out_w, out_h);

  cmd->setViewports(plume::RenderViewport(0.0f, 0.0f, float(out_w), float(out_h)));
  cmd->setScissors(plume::RenderRect(0, 0, int32_t(out_w), int32_t(out_h)));
  // The SDK's own menus, then reNut's overlay over them.
  if (s.overlay_hook)
    s.overlay_hook(cmd, back_fb, out_w, out_h);
  overlay::DrawLocked(cmd, out_w, out_h);

  cmd->setFramebuffer(nullptr);
  cmd->barriers(plume::RenderBarrierStage::GRAPHICS,
                plume::RenderTextureBarrier(back, plume::RenderTextureLayout::PRESENT));

  plume::RenderCommandSemaphore *present_semaphore = s.present_semaphores[image_index].get();
  detail::SubmitAndAdvance(s, acquire, present_semaphore);
  plume::RenderCommandSemaphore *waits[] = {present_semaphore};
  bool presented;
  {
    // With vsync the present can hold until the display takes a frame.
    render_thread::WaitScope wait;
    presented = s.swap_chain->present(image_index, waits, 1);
  }
  if (!presented)
    s.resize_requested.store(true, std::memory_order_release);
}

} // namespace

namespace {

// A guest that has not presented for this long is stalled (loading, movie
// decode); the UI pump then presents the overlay on its own.
constexpr int64_t kGuestStallMs = 100;

int64_t NowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

} // namespace

// banjo-nx (2026-10-19, latency batch 3): the fences of the last guest presents' frames, as copies (the slot's fence is
// re-armed two frames later), for the display thread (renut_engine/display_sync.cpp) to test without waiting. Its own
// mutex: the render thread holds State().mutex for whole frames.
namespace {
constexpr uint32_t kPresentFences = 8;
struct PresentFence {
  uint64_t index = 0;
#if defined(PLUME_DEKO3D)
  plume::Deko3DFenceCopy fence;
#endif
};
std::mutex g_present_fence_mutex;
PresentFence g_present_fences[kPresentFences];
std::atomic<uint64_t> g_present_count{0};
} // namespace

void Video::SetGammaRamp(const uint8_t *big_endian_words) {
  GammaRamp &g = g_gamma;
  if (!big_endian_words || (g.seen_any && std::memcmp(g.seen, big_endian_words, sizeof(g.seen)) == 0))
    return;
  // A device that has no ramp yet (all zero) would black the picture out: the emulated display starts linear.
  bool any = false;
  for (uint32_t i = 0; i < sizeof(g.seen) && !any; ++i)
    any = big_endian_words[i] != 0;
  if (!any)
    return;
  std::memcpy(g.seen, big_endian_words, sizeof(g.seen));
  g.seen_any = true;
  {
    std::lock_guard lock(g.mutex);
    std::memcpy(g.words, big_endian_words, sizeof(g.words));
  }
  g.version.fetch_add(1, std::memory_order_acq_rel);
}

uint64_t Video::PresentCount() {
  return g_present_count.load(std::memory_order_acquire);
}

int Video::PresentDone(uint64_t index) {
#if defined(PLUME_DEKO3D)
  if (!index || index > g_present_count.load(std::memory_order_acquire))
    return -1;
  plume::Deko3DFenceCopy fence;
  {
    // Copied out: the test (an ioctl) runs without the mutex the render thread takes at every present.
    std::lock_guard lock(g_present_fence_mutex);
    const PresentFence &p = g_present_fences[index % kPresentFences];
    if (p.index != index)
      return 1;  // older than the ring: long finished (two frames in flight)
    fence = p.fence;
  }
  return plume::Deko3DFenceCopyDone(fence) ? 1 : 0;
#else
  (void)index;
  return -1;
#endif
}

void Video::Present(const PresentImage *image) {
  auto &s = State();
  if (s.shutting_down.load(std::memory_order_acquire))
    return;
  std::lock_guard lock(s.mutex);
  if (!s.host_ready || s.shutting_down.load(std::memory_order_acquire))
    return;
  PresentLocked(s, image);
  s.last_guest_present_ms.store(NowMs(), std::memory_order_release);
  detail::LatencyTrimLocked(s);  // banjo-nx (2026-10-20, input latency, banjo_deko3d_latency_trim)
  {
    // The frame just submitted: the slot before the one SubmitAndAdvance moved to.
    std::lock_guard fence_lock(g_present_fence_mutex);
    const uint64_t index = g_present_count.load(std::memory_order_relaxed) + 1;
    PresentFence &p = g_present_fences[index % kPresentFences];
    p.index = index;
#if defined(PLUME_DEKO3D)
    p.fence = plume::Deko3DCopyFence(s.fences[(s.frame + kFramesInFlight - 1) % kFramesInFlight].get());
#endif
    g_present_count.store(index, std::memory_order_release);
  }
}

void Video::PresentOverlayOnly() {
  auto &s = State();
  if (s.shutting_down.load(std::memory_order_acquire))
    return;
  const int64_t now = NowMs();
  if (s.guest_ready &&
      now - s.last_guest_present_ms.load(std::memory_order_acquire) < kGuestStallMs)
    return;
  // The render thread holds the mutex while it waits on the UI thread for the
  // overlay, so never block on it here.
  std::unique_lock lock(s.mutex, std::try_to_lock);
  if (!lock.owns_lock() || !s.host_ready || s.shutting_down.load(std::memory_order_acquire))
    return;
  // banjo-nx (2026-10-19, the colour flash): while the guest stalls (loading) with nothing of the overlay to show,
  // nothing is presented: the display keeps the last frame, as the console's does. The re-present of the last front
  // buffer showed whatever that texture held by then, and the game may free or remake it while it loads: on deko3d a
  // new image's memory is not cleared (Vulkan's came zeroed), a full-screen frame of other data. With the overlay up
  // the last frame is presented under it, but only from a texture a resolve wrote (else the overlay on black).
  // banjo_renut_stall_present = true: as before.
  if (s.guest_ready && !banjo::renut_gpu::StallPresentOn() && !OverlayDrawer::HasLatest())
    return;
  PresentImage image =
      s.guest_ready && s.last_front_buffer ? FrontBufferImageLocked(s.last_front_buffer) : PresentImage{};
  if (image.texture && !banjo::renut_gpu::StallPresentOn() && !FrontBufferWrittenLocked(s.last_front_buffer))
    image = PresentImage{};
  PresentLocked(s, image.texture ? &image : nullptr);
}

bool CaptureFrame(rex::ui::RawImage &out, int timeout_ms) {
  FrameCapture &c = g_capture;
  std::unique_lock lock(c.mutex);
  const uint64_t ticket = c.requested.fetch_add(1, std::memory_order_acq_rel) + 1;
  if (!c.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return c.served >= ticket; }) || !c.ok)
    return false;
  out = c.image;
  return true;
}

// sa2-nx (2026-10-20): the two halves (banjo_renut_capture.h).
uint64_t RequestFrameCapture() {
  return g_capture.requested.fetch_add(1, std::memory_order_acq_rel) + 1;
}

bool WaitFrameCapture(uint64_t ticket, rex::ui::RawImage &out, int timeout_ms) {
  FrameCapture &c = g_capture;
  std::unique_lock lock(c.mutex);
  if (!c.cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return c.served >= ticket; }) || !c.ok ||
      c.served != ticket)
    return false;
  out = c.image;
  return true;
}

} // namespace renut::gpu
