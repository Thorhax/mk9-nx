// SA2 - native renderer: drawing with the native shader library onto the render targets of
// sa2_native_targets.cpp.
//
// Report tags: the log lines of the renderer are prefixed "C2" (render targets, copies and presentation,
// sa2_native_targets.cpp), "C3" (textures) and "C4"/"C6" (sampler and draw statistics); the comments
// use the same tags to name the report line they talk about.
//
// Started as the version that validated the whole chain on PC with the videos and the title screen: 2D
// textures (base level), guest vertices and indices, constants, pipelines per state and render passes.
// What is not covered is rejected with the cause logged once (details in the .cpp).

#pragma once

#include <rex/ui/vulkan/device.h>
#include "me_sample_count_contract.h"
#include "me_stencil_clear_ownership.h"

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rex::memory {
class Memory;
}

namespace sa2::native {

static_assert(VK_SAMPLE_COUNT_1_BIT == 1 && VK_SAMPLE_COUNT_2_BIT == 2 &&
              VK_SAMPLE_COUNT_4_BIT == 4);

struct ShaderEntry;

// Image of a render target or a texture, in a host format.
struct ImageNative {
  VkImage image = VK_NULL_HANDLE;
  // Only if the image has its own dedicated allocation. Images from the texture pool leave this NULL on
  // purpose: their memory is a chunk of a shared slab and cannot be freed on its own.
  VkDeviceMemory memory = VK_NULL_HANDLE;
  // Texture pool slab, or UINT32_MAX if the image does not come from the pool. The literal is used here
  // instead of kBlockPoolInvalid to keep that header out of this one.
  uint32_t pool_block = 0xFFFFFFFFu;
  VkImageView view = VK_NULL_HANDLE;
  uint32_t width = 0;
  uint32_t height = 0;
  // sa2_native_shadows_scale: the shadow map is drawn smaller than the guest requests and upscaled when
  // resolved. This holds the size the guest thinks it has; 0 = the same.
  uint32_t guest_width = 0;
  uint32_t guest_height = 0;
  // sa2-nx render resolution (sa2_render_height_handheld / _docked, me_resolution_scale.h): width / height above are
  // the GUEST size; these are the VkImage's extent, the guest size scaled by scale_q / 4 (0 = the same as width /
  // height: images that are never scaled, such as the guest textures). Only Vulkan commands use them.
  uint32_t host_width = 0;
  uint32_t host_height = 0;
  uint8_t scale_q = 4;
  // The VkImageUsageFlags it was created with (render targets and resolved textures), to recreate it at another scale.
  uint32_t vk_usage = 0;
  // ZCULL: the scene's depth images are created without TRANSFER_DST so the driver can assign them a ZCULL
  // plane. Without that usage they cannot be cleared with vkCmdClearDepthStencilImage or receive copies:
  // they have to be cleared by opening a pass with loadOp = CLEAR, and they cannot be swapped with their
  // resolved texture.
  bool accepts_target_of_copy = true;
  VkFormat format = VK_FORMAT_UNDEFINED;
  // Actual VkImage allocation samples, NOT the guest MSAA enum or a raster-grid
  // expansion. Legacy images are 1x. Real multisample allocation remains gated
  // until the material shader, alias/import, resolve and cache contracts exist.
  VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT;
  // Color render targets only: the guest EDRAM representation. Host RGBA16F is used for several
  // incompatible Xenos views (10:10:10:2 UNORM, 7e3 and true 64-bpp FP16), so VkFormat alone cannot
  // describe how physical 32-bit EDRAM words map to pixels.
  uint8_t edram_format = 0;
  // Experimental common depth/color physical ownership (alias diagnostic mode 4).
  bool edram_depth = false;
  // MODE4 float24 attachment: host_z = guest_z * 0.5 (D32S8 only).
  bool depth_float24_half = false;
  // R32 sampled resolve already converted back to guest depth, point-sampled RRRR.
  bool resolved_depth_guestspace = false;
  // Guest texture format of the most recent resolve, for fetch contract diagnostics.
  uint32_t resolved_guest_format = 0xFFFFFFFFu;
  uint16_t edram_base = 0;
  bool edram_64bpp = false;
  // Log2 of an opt-in horizontal raster expansion. Unlike shadow scaling, these
  // texels retain distinct physical X samples; edram_msaa_x is the REMAINING
  // collapsed physical-to-host shift. This is not full/native MSAA support.
  uint32_t raster_grid_x = 0;
  // Remaining collapsed physical sample grid (Y is still single-sample).
  uint8_t edram_msaa_x = 0;
  uint8_t edram_msaa_y = 0;
  bool prepared = false;  // already in GENERAL and initialized
  // Resolved texture with RB_COPY_DEST_INFO.copy_dest_swap: the guest sees it with red and blue swapped
  // relative to the render target it comes from.
  bool swap_rb = false;
  // sa2_native_resolver_no_copy: when a whole render target is resolved, its image is swapped with the
  // resolved texture's instead of copied. The target keeps that texture's old content: if the game draws
  // on it again without clearing first, the content has to be brought back.
  bool invalid_content = false;
  uint32_t resolved_base = 0;  // where its content is while invalid_content
  // Mass Effect EDRAM mode 4, O(1) bookkeeping fast paths: while the global ownership epoch is still
  // edram4_epoch, every physical tile of edram4_own (raster area) is owned by this image; edram4_exported
  // = another view cached one of its tile versions since its tiles were last re-versioned.
  uint64_t edram4_epoch = 0;
  VkRect2D edram4_own{};
  bool edram4_exported = true;
  // Mass Effect mode 4: a draw with stencil enabled has used this depth view (lazy stencil import).
  bool edram4_usa_stencil = false;
  // Mass Effect mode 4: stencil bits that can be non-zero anywhere in this depth image (draw REPLACE
  // references / INCR-DECR-INVERT write masks, clear values, bits imported from other images; color
  // sources count as all 8). Depth imports skip the bit passes of bits outside the source's mask.
  uint8_t edram4_bits_stencil = 0;
};

// sa2-nx render resolution: the VkImage extent of an image (its guest size when it is not scaled).
inline uint32_t HostWidth(const ImageNative& image) { return image.host_width ? image.host_width : image.width; }
inline uint32_t HostHeight(const ImageNative& image) { return image.host_height ? image.host_height : image.height; }

// GPU time categories (ContextTargets::MarkGpu and the GPU-per-Swap report).
inline constexpr uint32_t kGpuOther = 0;
inline constexpr uint32_t kGpuShadows = 1;   // depth-only targets of 1600 or more
inline constexpr uint32_t kGpuScene = 2;    // 1280
inline constexpr uint32_t kGpuMirror = 3;   // 640: half-resolution intermediate targets
inline constexpr uint32_t kGpu320 = 4;       // 320: small intermediate targets
inline constexpr uint32_t kGpuSmaller = 5;   // under 320
inline constexpr uint32_t kGpuCopies = 6;    // copies to resolved textures
inline constexpr uint32_t kGpuCleared = 7;  // color and depth clears
// 1280 targets without depth: full-screen post-processing and HUD ("scene" is kept for the ones with
// depth, the geometry).
inline constexpr uint32_t kGpuSceneNoDepth = 8;
// Not a pass type but the GPU gap between the end of one submission and the start of the next.
inline constexpr uint32_t kGpuGapBetweenJobs = 9;
// EDRAM mode-4 transfers by kind (previously all in "other").
inline constexpr uint32_t kGpuEdramImport = 10;  // depth import (color/depth -> depth)
inline constexpr uint32_t kGpuEdramExport = 11;  // depth export (depth -> color)
inline constexpr uint32_t kGpuEdramAlias = 12;   // color alias and the other conversions
inline constexpr uint32_t kGpuEdramImport9 = 13;  // depth imports that still need the 8 stencil passes
inline constexpr uint32_t kGpuCategories = 14;

// Stages of the C6 report. The last four split the pass change, which on the console is the most
// expensive and most variable stage.
inline constexpr uint32_t kStagesDraw = 12;

// What the render target code (sa2_native_targets.cpp) provides to the draws. All on the ring thread.
class ContextTargets {
 public:
  virtual ~ContextTargets() = default;
  // Command buffer of the frame's work, recording. nullptr on failure.
  virtual VkCommandBuffer CommandsWork() = 0;
  // sa2_msaa: the host sample count of the game's multisampled single-sample targets (1 = none).
  virtual uint32_t HostMsaaSamples() const { return 1; }
  // Upload command buffer, recording: submitted right before the work one.
  virtual VkCommandBuffer CommandsUpload() = 0;
  // Changes every time the work buffer starts recording again.
  virtual uint64_t GenerationCommands() const = 0;
  // Render targets already in GENERAL. May record commands: call outside a pass.
  virtual ImageNative* TargetColor(uint32_t base, uint32_t format, uint32_t pitch) = 0;
  virtual ImageNative* TargetDepth(uint32_t base, uint32_t format, uint32_t pitch) = 0;
  // Texture resolved by the render target code at that physical address, or nullptr.
  virtual const ImageNative* ResolvedTexture(uint32_t address) = 0;
  // A known resolve may have a pitch-sized backing, unlike the fetch's logical size.
  // Called outside the current draw's pass, after ResolvedTexture accounted/materialized the read.
  // nullptr means an unsupported/failed resolved fetch, NOT permission to read stale guest RAM.
  virtual const ImageNative* ResolvedTextureForFetch(uint32_t address,
                                                     const uint32_t fetch[6],
                                                     const ImageNative& source) {
    (void)address;
    (void)fetch;
    return &source;
  }
  // Submits what was recorded and continues in the other slot, with its upload buffer empty.
  virtual bool SendAndWait() = 0;
  // Waits for the GPU to finish everything pending and starts recording again. It is expensive (a
  // one-frame stutter), so it is only used as a last resort when memory runs out: with the GPU idle,
  // textures can be released regardless of when they were last used, because none is in use.
  virtual bool WaitGpuOfTheAll() = 0;
  // GPU timestamp: whatever is recorded from here counts toward that category.
  virtual void MarkGpu(uint32_t category) = 0;
  // Diagnostic: names the GPU interval the last mark opened ((VS number << 16) | PS number of its first draw).
  virtual void LabelMarkGpu(uint32_t label) { (void)label; }
  // Diagnostic: the pass the last mark opened (width, height, color 0 / depth host formats) and its draws.
  virtual void DescribeMarkGpu(uint64_t description) { (void)description; }
  virtual void CountDrawMarkGpu() {}

  // Pipeline statistics of a whole pass (shaded fragments, vertex invocations and primitives reaching
  // clipping), accumulated per category. UINT32_MAX if not measured.
  virtual uint32_t BeginStats(uint32_t category) = 0;
  virtual void FinishStats(uint32_t index) = 0;

  // The same per draw, tagged with the pixel shader number, in an occasional diagnostic frame. Returns
  // UINT32_MAX if not measured or if there is no room left.
  virtual uint32_t BeginStatsDraw(uint32_t label, uint32_t category,
                                             uint32_t vs_more_one = 0) = 0;  // ME: VS number + 1
  virtual void FinishStatsDraw(uint32_t index) = 0;
};

struct SubmissionDraw {
  const uint32_t* registers = nullptr;       // register mirror of the ring sink
  const ShaderEntry* vs = nullptr;
  const ShaderEntry* ps = nullptr;
  std::span<const uint32_t> vs_microcode;  // patched by D3D (last IM_LOAD)
  std::span<const uint32_t> ps_microcode;  // authoritative HOST-order last PS IM_LOAD
  // Optional proven guest-pixel raster bottom for bounded unclipped utility
  // rectangles. Absence preserves the conservative scissor ownership range.
  std::optional<uint32_t> edram_used_height_estimate;
  // Default-ineligible, opt-in full physical-tile stencil replacement proof.
  me::native::StencilClearOwnershipProof edram_stencil_clear;
  // proven full overwrite (guest pixels x0,y0,x1,y1) of the views in edram_overwrite_slots
  // (bit 0 depth, bits 1-4 color 0-3). Tiles fully inside need no ownership transfer before the draw.
  std::optional<std::array<int32_t, 4>> edram_overwrite_rect;
  uint32_t edram_overwrite_slots = 0;
  // The constant post-viewport depth the proven rectangle writes (no bias, no depth export), if any.
  std::optional<float> edram_overwrite_depth;
  // sa2-nx: the constant colour the proven rectangle writes to color 0 (Direct3D's Clear: the pixel shader is
  // oC0 = r0 and the three vertices export the same interpolator 0, bit for bit), if any.
  std::optional<std::array<float, 4>> edram_overwrite_color0;
  // Proven rectangle (guest pixels) in which the draw replaces every stencil bit (REPLACE/ALWAYS, mask 0xFF,
  // both faces), whatever it does with depth: a depth import before it needs no stencil passes there.
  std::optional<std::array<int32_t, 4>> edram_stencil_replace_rect;
  // Bounding rectangle (guest pixels) of everything a multi-rectangle draw covers (rect lists of 2..8 rects);
  // mode 4 limits its sync/publish area to it. Single rectangles use edram_overwrite_rect.
  std::optional<std::array<int32_t, 4>> edram_bounds_rect;
  // All proven rectangles of a multi-rectangle draw (edram_overwrite_rect is the largest of them).
  std::array<std::array<int32_t, 4>, 8> edram_overwrite_rects{};
  uint32_t edram_overwrite_rect_count = 0;
  uint64_t generation_vs = 0;                // changes with every VS IM_LOAD
  // sa2-nx: XXH3_64bits of vs_microcode (computed when the ring identified the load); valid when vs_program_hash_set.
  uint64_t vs_program_hash = 0;
  bool vs_program_hash_set = false;
  uint64_t generation_constants_vs = 0;     // changes when 0x4000-0x43FF are written
  uint64_t generation_constants_ps = 0;     // changes when 0x4400-0x47FF are written
  /*
   * The two most expensive stages of recording a draw on the console are "textures" (4.1 ms per frame) and
   * the viewport/scissor that the C6 report includes in "pipeline". Both are pure functions of registers
   * that almost never change between consecutive draws, so the ring sink tracks when they really change
   * (it compares the value before writing it, as with the constants) and the count arrives here.
   *
   * A consumer can keep the last value seen next to its result (the 48+16 texture slots and the 32 1/size
   * words of the shared block; the VkViewport, the VkRect2D and ndc[4]) and skip the whole recomputation
   * while it does not change; the framing cache does this with generation_framing. A generation that goes
   * up too often only causes extra work, never a wrong draw.
   */
  uint64_t generation_fetch = 0;             // changes when 0x4800-0x48BF are written (fetch constants)
  uint64_t generation_framing = 0;          // viewport, scissor, clip, rasterization mode
};

struct StatsDraws {
  uint64_t drawn = 0;
  uint64_t rejected = 0;
  uint64_t pipelines = 0;
  uint64_t textures = 0;
  uint64_t uploads_texture = 0;
  uint64_t uploaded_megabytes = 0;  // vertices, indices, constants and textures
  uint64_t megabytes_textures = 0;  // texture images created (base level, no eviction)
  uint64_t ms_pipelines = 0;  // creating pipelines, accumulated
  // Accumulated for "C6 counters" (the system prints the differences per report).
  uint64_t passes = 0;             // render passes started
  uint64_t full_sends = 0;     // submissions due to a full upload buffer
  uint64_t ns_full_sends = 0;  // inside those submissions, including the wait for the GPU
  uint64_t bytes_vertices = 0;
  // Deduplication of vertex uploads within the frame.
  uint64_t dedupe_hits = 0;
  uint64_t dedupe_bytes = 0;
  uint64_t dedupe_collisions = 0;
  uint64_t bytes_indices = 0;
  uint64_t samplers = 0;          // samplers prepared
  uint64_t samplers_cache = 0;    // of those, resolved by the register cache
  uint64_t ns_passes = 0;          // inside FinishPass + BeginPass on a pass change
  uint64_t ns_vertices = 0;       // copying vertices with byte swap
  uint64_t computed_inputs = 0;   // ComputeEntry due to another VS generation
  uint64_t ns_inputs = 0;           // inside ComputeEntry
  uint64_t reused_inputs = 0; // EntryFor with the same generation and VS
  uint64_t passes_per_generation = 0;  // pass change due to a new command buffer
  uint64_t passes_per_target = 0;     // pass change due to other render targets
  uint64_t resumed_passes = 0;      // the same pass, closed earlier by a copy or a clear
  uint64_t ns_render_pass = 0;        // in vkCmdBeginRenderPass + vkCmdEndRenderPass
  uint64_t texels_passes = 0;          // area opened, summed over every pass opening
  // The same area, split by render target type (indices kGpuShadows..kGpuSmaller).
  std::array<uint64_t, kGpuCategories> texels_per_category{};
  // Pass openings per render target type, to know the average area of each.
  std::array<uint64_t, kGpuCategories> passes_per_category{};
  // Draws and triangles per render target type. With each one's GPU time (C2 report) it is possible to
  // tell whether an expensive pass is geometry-bound or pixel-bound: the counts do not depend on the
  // machine, so what is measured on PC holds for the console.
  std::array<uint64_t, kGpuCategories> draws_per_category{};
  // Draws with no color to write, by whether their pixel shader is needed.
  uint64_t draws_ps_useless = 0;
  uint64_t draws_ps_required = 0;
  // Shadow map draws by their constant c1 (g_bShadowMapAlphaEnabled).
  uint64_t shadows_active_alpha = 0;
  uint64_t shadows_off_alpha = 0;
  std::array<uint64_t, kGpuCategories> triangles_per_category{};
  // Submissions (upload slot changes) with constants through UBOs and in total, for the report.
  uint64_t sends_ubo = 0;
  uint64_t sends = 0;
  // How many draws really change the 488 bytes of shared constants. It decides whether the comparison or
  // the double memcpy is what needs to get cheaper.
  uint64_t shared_looked = 0;
  uint64_t shared_changed = 0;
  uint64_t bytes_repeated_frame = 0;  // sa2_native_diag_vertices_repeated
  uint64_t bytes_equal_previous = 0;
  uint64_t ns_hash_vertices = 0;
  // Rejections by cause (Reject in the .cpp), most frequent first.
  std::vector<std::pair<uint32_t, uint64_t>> causes;
  // Accumulated nanoseconds per stage of the draws that get recorded: state, indices, textures, pass,
  // uploads, pipeline and recording.
  std::array<uint64_t, kStagesDraw> stages_ns{};
  // Scene draws that allow or prevent early depth rejection.
  uint64_t scene_with_discard = 0;
  uint64_t scene_no_discard = 0;
  // Recorded draws with the stopwatch (1 in kStopwatchEach): the divisor of stages_ns and ns_vertices.
  uint64_t drawn_timed = 0;
  // Shadow map vegetation dropped right on entry, before paying for indices, textures, upload and pass. It
  // was counted but never printed anywhere, so the gain could not be verified at first. Now it is printed.
  uint64_t vegetation_soon = 0;
};

class DrawsVulkan {
 public:
  // nullptr if the device lacks capabilities (the reason is logged).
  static std::unique_ptr<DrawsVulkan> Create(const rex::ui::vulkan::VulkanDevice* vulkan_device,
                                              rex::memory::Memory* memory,
                                              ContextTargets* context);
  virtual ~DrawsVulkan() = default;

  virtual bool Draw(const SubmissionDraw& submission) = 0;
  // Before copies, clears and any command outside a pass.
  virtual void FinishPass() = 0;
  // External graphics commands may replace state retained across render passes.
  virtual void NotifyGraphicsExternalState() = 0;
  // sa2-nx render resolution: every render target was recreated at another scale (quarters of the guest size, from
  // q_old to q_new): host-pixel data kept per target (the passes' used heights) is converted.
  virtual void NotifyRescale(uint32_t q_old, uint32_t q_new) { (void)q_old; (void)q_new; }
  // ZCULL: clears a depth image by opening a pass with loadOp = CLEAR, instead of with
  // vkCmdClearDepthStencilImage. Needed for images created without TRANSFER_DST, the only ones the driver
  // can give a ZCULL plane. Returns false if it could not.
  virtual bool ClearDepthInPass(VkCommandBuffer commands, const ImageNative& image,
                                       float depth, uint32_t stencil,
                                       const VkRect2D* area = nullptr) = 0;
  // Aspect-only clear: LOAD/STORE depth unchanged, clear just stencil in area.
  // The caller closes the previous pass before obtaining the command buffer.
  virtual bool ClearStencilInPass(VkCommandBuffer commands, const ImageNative& image,
                                   uint32_t stencil, const VkRect2D& area) = 0;
  // the counterpart for depth: LOAD both aspects, clear just depth in area (stencil kept).
  virtual bool ClearOnlyDepthInPass(VkCommandBuffer commands, const ImageNative& image,
                                           float depth, const VkRect2D& area) {
    (void)commands; (void)image; (void)depth; (void)area;
    return false;
  }
  // sa2_native_clear_area_util. Clears only the `area` rectangle of a color image by opening a pass
  // with loadOp = CLEAR. Returns false if it could not.
  virtual bool ClearColorInPass(VkCommandBuffer commands, const ImageNative& image, const VkClearColorValue& color,
                                 const VkRect2D& area) {
    (void)commands;
    (void)image;
    (void)color;
    (void)area;
    return false;
  }
  // Right before submitting: closes the pass and publishes the upload buffer.
  virtual void BeforeSend() = 0;
  // Starts work in that work slot. Its upload buffer starts over: the GPU has finished the last work
  // submitted with it.
  virtual void UseSlot(uint32_t slot) = 0;
  // A render target image is about to be destroyed: it is removed from the descriptors.
  virtual void ForgetImage(VkImage image) = 0;
  // A render target view is about to be destroyed (Destroy, in sa2_native_targets.cpp, with the GPU idle for that
  // image): the FramebufferFor cache framebuffers that use it are destroyed. Without this, if Vulkan gave
  // the same handle to another view, the cache returned a framebuffer created on the dead view.
  virtual void ForgetView(VkImageView view) { (void)view; }
  // The GPU is out of memory. Stops the GPU, releases half the texture cache and reports whether anything
  // was freed. Called by both texture and render target allocations: whichever runs out of memory first
  // calls this before giving up.
  virtual bool DropTexturesPerMissingOfMemory() = 0;
  // Before every render target copy: a resolved texture may change image or channels.
  virtual void InvalidateTextures() = 0;
  // usf4-nx: only the texture cache entries whose fetch constant has this base address (the crops of a resolved
  // texture are sampled at its own base; ResolvedWritten in sa2_native_targets.cpp).
  virtual void InvalidateTexturesAt(uint32_t base) { (void)base; InvalidateTextures(); }

  // Invalidates in the texture caches only what points to these two images. Used by the image swap of
  // sa2_native_resolver_no_copy: dropping the whole cache twice per frame costs more than the copy it
  // saves (measured on PC).
  virtual void InvalidateImages(VkImage a, VkImage b) = 0;
  virtual StatsDraws Stats() const = 0;
  // Draws recorded since start-up; the render target code counts those that fall between two copies.
  virtual uint64_t Drawn() const = 0;
  // Waits for the copy thread to finish the pending vertex copies (sa2_native_uploads_thread).
  virtual void WaitUploads() = 0;
  // Vertex copies queued and not done yet (only to measure the fences).
  virtual size_t PendingCopies() const { return 0; }
};

/*
 * The guest synchronization epoch: upload deduplication (the vertex dedupe, the index cache) forgets what it
 * recorded when it changes. masseffect-nx meant it to advance at every guest wait for the GPU, but nothing ever
 * advanced it. sa2-nx round 7 (sa2_native_dedupe_sync_epochs, me_native_system.cpp): the ring advances it, in
 * stream order, at every packet that writes something the game can observe (fences, memory writes, interrupts):
 * only after one of those may the game rewrite memory an earlier draw read.
 */
extern std::atomic<uint32_t> g_synchronizations_ring;

/*
 * Reports of the ring thread go through DeferredReport: the line is formatted on the calling thread (it
 * reads the ring's state, which is only consistent on its own thread) and handed to it. Defined in
 * sa2_native_glue.cpp, which logs it directly.
 */
void DeferredReport(std::string line);

// Like REXLOG_INFO, but the report thread does the SD write (DeferredReport).
#define SA2_REPORT_RING(...) ::sa2::native::DeferredReport(fmt::format(__VA_ARGS__))

}  // namespace sa2::native
