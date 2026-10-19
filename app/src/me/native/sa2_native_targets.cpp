// SA2 - native renderer: render targets, copies (resolve and clear) and presentation of the game's image
// (see sa2_native_targets.h).
//
// WHAT IT COVERS
//   - k_8_8_8_8 and k_8_8_8_8_GAMMA color render targets without MSAA, as
//     R8G8B8A8 images of pitch x max(720, pitch) (capped at 2048).
//   - Copy of a rectangle of the render target to the resolved texture at
//     RB_COPY_DEST_BASE (destination format k_8_8_8_8), with the rectangle and
//     the base computed like the SDK's GetResolveInfo.
//   - Render targets with MSAA: used with 1 sample (MSAA hangs NVK on the Switch).
//   - Color clear with RB_COLOR_CLEAR, converted like the SDK's render target cache does.
//   - Presentation: the resolved texture requested by fetch constant 0 of the
//     Swap is drawn into the presenter output with the SDK's own shaders. By
//     default, with the gamma ramp the game loads, like the Xbox 360 display
//     (the ramp pixel shader is src/native/native/shaders/sa2_output_ramp_gamma_ps.h).
//
// DRAWS (sa2_native_draws.cpp)
//   This class lends them the render targets (color and depth), the frame's
//   command buffer and an upload command buffer that is submitted right before it.
//
// Everything is used only by the native system's PM4 ring thread: no locks.

#include "cpu/sa2_enhancements.h"
#include "sa2_deferred_recording.h"
#include "sa2_native_targets.h"
#include "sa2_waits_hitch.h"
#include "sa2_native_shaders.h"

#include "me_formats_color.h"  // HDR render targets
#include "me_resolve.h"
#include "me_edram_layout.h"
#include "me_msaa_raster_grid.h"
#include "me_edram_ownership.h"
#include "me_depth.h"
#include "me_resolved_allocation.h"
#include "me_resolved_sampling.h"
#include "me_msaa_depth_resolve.h"
#include "me_resolution_scale.h"
#include "../me_shader_identity.h"
#include "../sa2_render_mode.h"

#include <rex/cvar.h>
#include <rex/platform.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/xenos.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/logging.h>
#include <rex/system/xmemory.h>
#include <rex/ui/presenter.h>
#include <rex/ui/vulkan/device.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/util.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <deque>

extern "C" uint32_t g_banjo_last_reject_cause;  // banjo-nx: sa2_native_draws.cpp, for the frame trace

// Counter 0 of the console profiler (game FPS).
extern "C" void RexSwitchPerfCount(unsigned id);
// Interval of a long frame for stack sampling (the "during the hitches" section of the profiler report).
extern "C" void RexSwitchPerfHitch(uint64_t start, uint64_t end);
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace gr = rex::graphics;

namespace sa2::native {
bool CompareActiveFast();  // sa2_native_draws.cpp: sa2_native_compare_fast
}
// banjo-nx: Nuts & Bolts resolves several sizes and formats to one scratch address every frame (1EA9E000:
// 768x768 depth, 640x360 and 320x180 colour). Each change used to destroy the previous image behind a full GPU
// drain (three per frame, the renderer's main stall on this game). Parked images are kept per (address, size,
// format) and put back when that shape returns, with no wait: same path as a resolve into an existing image.
REXCVAR_DEFINE_BOOL(banjo_native_park_resolved, true, "Nuts&Bolts/Graphics",
                    "Native renderer: keep resolved textures whose address is reused with another size/format and "
                    "reuse them, instead of destroying them behind a full GPU wait")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
#if REX_PLATFORM_SWITCH
#define SA2_EDRAM4_PAIR_REPORT_DEFAULT false
#else
#define SA2_EDRAM4_PAIR_REPORT_DEFAULT true
#endif
REXCVAR_DEFINE_BOOL(sa2_native_edram4_pair_report, SA2_EDRAM4_PAIR_REPORT_DEFAULT, "SA2",
                    "EDRAM mode 4 report: transfers broken down by view pair and by draw (string keys per transfer "
                    "run; off by default on the Switch, where it costs ring-thread time)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// sa2-nx CPU enhancement: the EDRAM mode 4 transfer-trace text is formatted when a trace line is logged instead of for
// every draw slot until 60 traces have been logged (on the PC profile ~4 % of the ring thread). Exact: same log text.
SA2_ENHANCEMENT(g_lazy_trace_text, sa2_cpu_lazy_trace_text, sa2::enh::kCpu, true,
                "Ring thread: format the EDRAM transfer-trace text only when a trace line is logged");
// sa2-nx CPU enhancement (round 2): RedirectClearDepthEDRAM4 is asked for every proven rectangle, and the usual answer
// ("depth not proven constant") formatted a string key and looked it up in a string map each time. Counted by
// (literal, depth info) and formatted when the 10 s report is written. Exact: the same report line.
// sa2-nx CPU enhancement (round 2): GetTarget / GetDepth are asked for every slot of every draw (console profile:
// 2.2 % + 1.2 % of the ring thread): a 4-slot memo of their inputs in front of the work and the map lookup.
SA2_ENHANCEMENT(g_target_memo, sa2_cpu_ring_target_memo, sa2::enh::kCpu, true,
                "Ring thread: render target and depth target lookups answered from a small memo of their inputs (the "
                "maps never drop an entry, so the same inputs always give the same image)");
SA2_ENHANCEMENT(g_redirect_key_int, sa2_cpu_ring_redirect_key, sa2::enh::kCpu, true,
                "Ring thread: the depth-clear redirect's decline reasons are counted by number and formatted only for "
                "the report, not once per proven rectangle");
// sa2-nx CPU enhancement (round 5, the direct renderer's first back-end step): PrepareDrawEDRAM4 runs for every draw
// (console: 9 % of the ring thread self, plus Synchronize/Publish/lookups, ~3 us a draw at 1020 MHz) although most
// draws keep the render targets of the draw before them. A draw whose render-target inputs equal the previous draw's
// reuses its result when every check the full path would pass on its fast paths still holds (EdramMemoStillFast).
// GPU round (2026-10-07): a same-format RGBA8 color resolve as one fragment pass on the 3D engine instead of
// vkCmdCopyImage, which NVK records on the copy engine (NV90B5): each of SA2's ~15 resolves a frame switched engines,
// with a wait for idle each way. Same bytes (UNORM8 round-trips exactly). Non-CPU group (a GPU change).
SA2_ENHANCEMENT(g_resolve_draw, sa2_gpu_resolve_draw, sa2::enh::kNonCpu, true,
                "GPU: same-format RGBA8 color resolves are drawn on the 3D engine (one fragment pass that copies the "
                "texels) instead of vkCmdCopyImage on NVK's copy engine (no engine switch and wait for idle per "
                "resolve)");
// GPU round (2026-10-07): SA2 resolves each 512x512 offscreen render twice in a row (its texture, then 1E8A0000) and
// its frame twice (1EFB0000, then 1F6F8000), with nothing drawn, cleared or imported in between, and never reads
// 1E8A0000 or 1F6F8000. The second copy of such a pair is not made: that texture becomes an alias of the first one,
// which holds the same bytes. They are copied from it only when something reads the alias or before the first
// texture changes, and when the first texture is about to be overwritten whole, the two images are exchanged instead.
SA2_ENHANCEMENT(g_resolve_alias, sa2_gpu_resolve_alias, sa2::enh::kNonCpu, true,
                "GPU: a color resolve that repeats the previous one (same render target and rectangle, nothing in "
                "between) into another texture is not copied; that texture takes the bytes of the first one when "
                "something reads it or before the first one changes (a copy, or an exchange of the two images when "
                "the first one is overwritten whole). SA2's 1E8A0000 and 1F6F8000 are never read");
SA2_ENHANCEMENT(g_edram_memo, sa2_cpu_ring_edram_memo, sa2::enh::kCpu, true,
                "Ring thread: a draw with the same render-target state as the previous draw reuses its EDRAM preparation "
                "(only when the full path would do no work: same images, tiles owned in the same epoch, nothing pending; "
                "self-checked against the full path)");
REXCVAR_DEFINE_BOOL(sa2_native_zcull_1x_depth, false, "SA2",
                    "Native renderer: 1x depth views keep their ZCULL plane (no TRANSFER_DST for copy-engine stencil "
                    "imports, which then use masked stencil passes). Off until measured on the console")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_color0, true, "SA2",
                    "EDRAM mode 4: a depth + color 0 clear drawn through a 4x alias (Direct3D's Clear) is cleared in "
                    "the 1x owners of both instead of drawn and transferred 4x -> 1x (same bytes; false = as before)");
REXCVAR_DEFINE_BOOL(banjo_native_edram4_skip_unsupported, true, "Nuts&Bolts/Graphics",
                    "Native renderer: when the previous content of EDRAM tiles cannot be converted for a draw's target "
                    "(e.g. 2x MSAA <-> 1x), draw anyway over the target's own content instead of dropping the draw");
REXCVAR_DEFINE_BOOL(sa2_native_reuse_alloc_resolved, true, "SA2 renderer",
                   "Reuse fence-safe resolved image allocations in experimental EDRAM mode 4; "
                   "discard all old contents before reuse. false = original fenced recreation");
REXCVAR_DEFINE_BOOL(sa2_native_logical_resolved_size, true, "SA2 renderer",
                   "Experimental MODE4: sample resolved textures at the proven logical fetch extent, "
                   "not their padded row pitch. false = existing direct resolved image");
REXCVAR_DEFINE_INT32(sa2_native_edram_alias_mode, 4, "SA2 renderer",
                     "EDRAM diagnostic: 0 no transfers, 1 exact base, 2 physical tiles, 3 scissor tiles, "
                     "4 experimental shared depth/color ownership (not a complete bit-exact emulation)")
    .range(0, 4);

REXCVAR_DEFINE_INT32(sa2_diag_resolves_from, 0, "SA2",
                     "Diagnostic: logs the raw registers of resolves number N to N+79 (0 = off)");
// Diagnostic of sa2_gpu_resolve_alias: every resolve and alias event of 40 presented frames from this one (0 = off).
REXCVAR_DEFINE_INT32(sa2_diag_alias_frames, 0, "SA2",
                     "Diagnostic: logs every resolve and every sa2_gpu_resolve_alias event of the 40 presented frames "
                     "from this one (0 = off)");
// Test of sa2_gpu_resolve_alias's copy on read: SA2 never reads its aliases, so this makes the Swap read one.
REXCVAR_DEFINE_BOOL(sa2_diag_present_alias, false, "SA2",
                    "Diagnostic: the Swap paints an alias of the front buffer (sa2_gpu_resolve_alias: SA2's 1F6F8000, "
                    "the same bytes) instead of the front buffer, so the alias is copied on every Swap. The image must "
                    "not change");
REXCVAR_DEFINE_INT32(sa2_native_shadows_scale, 100, "SA2 renderer",
                     "Native renderer: draws the game's two 1600x1600 shadow maps at this percentage "
                     "and scales them up when they are resolved. 100 = like the Xbox 360 "
                     "(1600, which is 2000 of its 2048 EDRAM tiles); 64 = like the PC version of this "
                     "game (1024). The value is read when the first map is created and does not change at run time")
    .range(50, 100);
REXCVAR_DEFINE_BOOL(sa2_native_resolver_no_copy, false, "SA2 renderer",
                    "Native renderer (off for correctness): when a whole render target is resolved to a "
                    "texture of the same size and format, swap the images instead of copying the pixels. "
                    "On MoltenVK the resolved image's old bindless descriptor stays in the argument buffer "
                    "after the swap and ends up referring to the active attachment, which creates read/write "
                    "feedback, horizontal stripes and content from earlier frames. Enable it only with "
                    "per-work-slot descriptor snapshots or on a backend that is proven not to touch "
                    "unselected bindless resources");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_cache_sync, true, "SA2 renderer",
                    "EDRAM mode 4: do not transfer a physical tile again to a view that already received its "
                    "current version (read-only rebinds). false = transfer on every bind, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_area_rect, true, "SA2 renderer",
                    "EDRAM mode 4: a draw with a proven rectangle synchronizes and publishes only the tiles of "
                    "that rectangle, not of its whole scissor. false = whole scissor, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_consumer, true, "SA2 renderer",
                    "EDRAM mode 4: a redirected clear goes straight into the view that pulled the same clear's tiles "
                    "last time (and becomes their owner), so that view's next bind transfers nothing");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_zero, true, "SA2 renderer",
                    "EDRAM mode 4: a redirected depth clear to 0 is applied to a depth owner of the other depth "
                    "encoding (D24S8 / D24FS8) too: the EDRAM word is the same");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_partial_clear, true, "SA2 renderer",
                    "EDRAM mode 4: redirected clears also take partial edge tiles (exact pixel bounds only): the "
                    "covered part is cleared in the tile's owner instead of drawing the clear after a transfer");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_color, true, "SA2 renderer",
                    "EDRAM mode 4: a redirected depth+stencil clear whose tiles a color view owns clears that view "
                    "with the same EDRAM word decoded as its format (as the depth->color export would), instead of "
                    "drawing it and exporting all tiles back");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_stencil, true, "SA2 renderer",
                    "EDRAM mode 4: a proven full-mask constant stencil clear rectangle (depth untouched) drawn "
                    "through another view (the 4x alias D3D Clear uses) is applied to the stencil of each tile's "
                    "current depth owner instead; avoids the full depth round trip owner -> alias -> owner");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_redirected_clear, true, "SA2 renderer",
                    "EDRAM mode 4: a proven constant-depth rectangle (depth ALWAYS, no stencil, no color) over "
                    "tiles owned by another depth view is applied as a depth clear in that owner's view, "
                    "without transferring the tiles to the drawn view and back. false = draw, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_import_area_tiles, true, "SA2 renderer",
                    "EDRAM mode 4: the depth import pass's render area is the transferred tiles' rectangle "
                    "instead of the whole destination. false = whole destination, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clear_alias, true, "SA2 renderer",
                    "EDRAM mode 4: a redirected depth clear over tiles the 4x view owns itself goes to that view's "
                    "known 1x alias (same base and tile pitch), which becomes the owner. false = draw, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_fast, true, "SA2 renderer",
                    "EDRAM mode 4: O(1) sync/publish when a view already owns every tile of the draw area (ownership "
                    "epoch). false = always scan the tiles, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_stencil_lazy, true, "SA2 renderer",
                    "EDRAM mode 4: depth imports into a view that never used stencil skip the 8 stencil passes and "
                    "remember where the stencil stayed; importing back there keeps it. false = always 9 passes");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_stencil_light, true, "SA2 renderer",
                    "EDRAM mode 4: the 8 stencil-bit passes of a depth import use a stencil-only fragment shader "
                    "(no depth fetch, no gl_FragDepth). false = the full import shader, as before");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_bits_stencil, true, "SA2 renderer",
                    "EDRAM mode 4: depth imports skip the stencil-bit passes of bits that can never be set in the "
                    "source image (tracked from draws, clears and imports). false = all 8 passes");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_stencil_copy, true, "SA2 renderer",
                    "EDRAM mode 4: depth imports into 1x views write the stencil with compute + "
                    "vkCmdCopyBufferToImage(STENCIL) instead of 8 masked draw passes (1x depth views get "
                    "TRANSFER_DST, which costs them their ZCULL plane). false = 8 passes, as before");
REXCVAR_DECLARE(bool, sa2_native_edram4_stencil_inert);
REXCVAR_DECLARE(std::string, sa2_diag_trace_tile);
REXCVAR_DEFINE_INT32(sa2_diag_trace_from_s, 95, "SA2 renderer",
                     "Diagnostic: seconds after the renderer starts before sa2_diag_trace_tile is traced");
REXCVAR_DEFINE_INT32(sa2_diag_dump_marks_s, 0, "SA2 renderer",
                     "Diagnostic: N > 0 seconds after start, log the GPU mark sequence (category, first draw's "
                     "shaders, ms) of 6 consecutive work slots");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_batch, true, "SA2 renderer",
                    "EDRAM mode 4: the transfers one sync needs (one per tile-row run) share one barrier pair "
                    "(compute) and one render pass (depth imports) instead of one each; false = one by one");
// usf4-nx: Direct3D clears USF4's shadow maps (512x512 in a 560-pixel view, 1024x1024 in a 1040 one) in 32/64-pixel
// strips. The last strip leaves the right edge of the last tile column uncovered, so every tile row of that column
// is imported on its own (one tile, one import draw of up to 9 passes): ~75 per frame, ~2 ms of GPU on the Switch.
// The runs of one sync that form a column (same source, same tile offset, same x range, rows touching) become one
// import draw over the column's rectangle; its bookkeeping still walks the original runs only.
REXCVAR_DEFINE_BOOL(usf4_edram4_import_columns, true, "USF4",
                    "EDRAM mode 4: depth imports of one sync that form a vertical column of tile runs (same source "
                    "and offset) are drawn as one import over the column's rectangle instead of one per tile row");
REXCVAR_DEFINE_BOOL(sa2_native_conversion_frag, false, "SA2 renderer",
                    "EDRAM mode 4: conversions between 32-bit color views of one EDRAM region (RGBA8 / UNORM10 / 7e3: "
                    "the HDR scene color round trips of the post-processing chain) as a fragment pass over the run's "
                    "tile rectangle instead of a compute dispatch (no storage-image traffic, no 3D<->compute engine "
                    "switch). Same words bit for bit. false = compute shaders")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// usf4-nx 0.6.9: a stencil-replace clear (one rectangle, stencil ALWAYS/REPLACE, write mask 0xFF) over a 512x512, 1024x1024
// or 320x180 target leaves the right or bottom tile of each row partly outside the rectangle (the pitch is padded to whole
// 80x16 tiles: 560, 1040, ...). Those tiles used to get their old stencil (8 bit passes, 3.4 ms of GPU per frame in fights
// on the Switch at 768 MHz) for the padding pixels the target never draws. Now the edge tiles count as replaced too: the
// padding keeps the view's own stale stencil instead of the previous owner's (not bit-exact there; the game clears the
// scene's stencil before using it, and no difference was seen on the PC or the console). Switch 0.6.9 test: 0.3 ms left
// in that category, Status Monitor GPU 74-76 % in fights (0.6.8 profiling build: 81-95 %).
REXCVAR_DEFINE_BOOL(usf4_resolved_crops_invalidate_narrow, true, "USF4",
    "A resolve into a texture with crops invalidates only the texture cache entries at that base address (the crops "
    "are sampled there) instead of the whole cache (~2 times per frame, ~80 texture lookups redone each). false = "
    "whole cache, as before")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(usf4_edram4_stencil_edges_skip, true, "USF4",
                    "EDRAM mode 4: tiles only partly inside a stencil-replace clear rectangle skip their old stencil "
                    "(the padding pixels outside the rectangle keep stale bits instead of 8 stencil passes). "
                    "false = exact, as before")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_native_edram4_stencil_copy_min_tiles, 32, "SA2 renderer",
                     "EDRAM mode 4: smallest import (tiles) whose stencil goes through the copy engine; smaller "
                     "ones use masked stencil draws in the import pass (no engine switches). 0 = always copy")
    .range(0, 2048);
REXCVAR_DEFINE_BOOL(usf4_copy_tiles_merged, true, "SA2 renderer",
                    "EDRAM mode 4 tile copies (vkCmdCopyImage): neighbouring tiles as one rectangle instead of one "
                    "region per tile (same pixels; NVK records each colour region as its own render pass)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_depth_samples_x, true, "SA2 renderer",
                    "Experimental MODE4: preserve horizontal samples in depth-only 4x MSAA draws. "
                    "Y samples remain collapsed; expanded-depth direct resolves and FragCoord shaders "
                    "are explicitly rejected. Not full MSAA conformance.");
/*
 * Swap also when the resolve command does not clear the render target (with sa2_native_resolver_no_copy
 * on). Requiring the clear was only a precaution: if the game draws into the render target again without
 * clearing it, RestoreContent brings the image back from the resolved texture, so the worst case is
 * paying the same copy later. the "resolves without copy" and "restores" lines of the report show whether
 * it helps.
 */
REXCVAR_DEFINE_BOOL(sa2_native_swap_no_clear, true, "SA2 renderer",
                    "Native renderer: swap the image also when the resolve command does not clear the "
                    "target. Removes the 1600x1600 copy of the second shadow map (2.4 ms real). If the game "
                    "draws on top without clearing, the content is restored: worst case, the same as "
                    "without the swap. Watch 'restores' in the log: it should stay at 0");
/*
 * Before resolving a depth render target whose content went out in an image swap (invalid_content), the
 * content is copied back from the texture that holds it. A swapped resolve leaves the render target with the
 * texture's old image; if the game resolves again without drawing in between, the second resolve would read
 * that stale image (a shadow map one frame old: flickering shadows). The Xbox 360 reads the same EDRAM in
 * both resolves, so copying back is the exact behaviour. Report line: "restores per frame",
 * the "to resolve" figure. false = previous behaviour.
 */
REXCVAR_DEFINE_BOOL(sa2_native_resolver_valid_content, true, "SA2 renderer",
                    "Native renderer: before resolving a depth target whose content left in an image "
                    "swap, bring it back. Fixes the flickering menu shadows (the scene was sampling "
                    "the previous frame's map). false = previous behavior");
/*
 * Restores after a swap that did not clear were measured at one restore per swap, so a swap without a clear
 * saves nothing by itself: the copy is paid later with two operations instead of one. Restores are recorded
 * in the copy inventory (NoteCopy in RestoreContent) so that the report adds up, and the setting
 * below trims their cost.
 */
REXCVAR_DEFINE_BOOL(sa2_native_restore_area_util, true, "SA2 renderer",
                    "Native renderer: when bringing back the content of a swapped target, copy only the "
                    "rows the game really resolves instead of the whole image. Targets are created with "
                    "height = max(720, pitch) (the scene is 1280x1280 to draw 1280x720), and what lies "
                    "below the useful area is never drawn nor read");
/*
 * The front buffer is drawn from its render target, without copying it (sa2_native_lazy_front).
 *
 * At the end of each frame the game resolves its output render target to a front buffer that only the Swap
 * reads (PaintOutput). If the address was a front buffer in the last kFrontFrames and the copy is 1
 * to 1, of the whole texture and from the corner of the render target, it is deferred:
 *   - in the Swap it is drawn from the image that holds the content (texel (x, y) of that image is the same
 *     byte the copy would have left); with FXAA, without the ramp or with another output the copy is
 *     recorded before submitting the work and the texture is drawn as usual;
 *   - if the game clears the whole render target, it gets a spare image and the one with the content is
 *     kept for the front buffer (at most kFrontImagesMax spares): nothing is copied;
 *   - if the render target is written some other way first (a pass, a restore, a swap), a draw samples the
 *     front buffer, or another resolve arrives that does not cover it entirely, the copy is recorded first
 *     (exact);
 *   - if another resolve arrives that covers it entirely, the deferred copy is dropped.
 * GUARD: if something requests a front buffer whose copy can no longer be made, DIFFERENCE in the log and
 * it turns off for the session. Report line: "lazy front". false = always copy.
 */
REXCVAR_DEFINE_BOOL(sa2_native_lazy_front, false, "SA2 renderer",
                    "Native renderer: the Swap paints the front buffer from its render target (or a "
                    "retained image) instead of copying it to the texture first: 0.59 Mpixels less per frame. Same "
                    "image; with FXAA it is copied as usual. It checks itself. false = always copy, as before");
/*
 * Repeated clears: a clear is skipped when (a) the last clear of that render target used the same value,
 * (b) the global draw counter has not gone up since then and (c) the render target has not changed image
 * through a swap nor received a restore. Then the clear cannot change a single bit.
 */
REXCVAR_DEFINE_BOOL(sa2_native_skip_cleared_repeated, true, "SA2 renderer",
                    "Native renderer: skip a clear when the target is already cleared "
                    "with that same value and nothing has been drawn since. Does not change a single pixel; "
                    "'clears skipped' in the report says how many are saved");
// Defined in sa2_native_draws.cpp; here it is only read so as not to open two queries of the same type
// at once.
REXCVAR_DECLARE(int32_t, sa2_native_stats_per_draw_s);
REXCVAR_DECLARE(bool, sa2_native_pass_narrow_dependency);
REXCVAR_DEFINE_BOOL(sa2_native_stats_pipeline, false, "SA2 renderer",
                    "Native renderer: counts shaded fragments, vertex invocations and clipped "
                    "primitives per pass type (target report). The image does not change, "
                    "but the queries cost GPU time: for measuring only");
REXCVAR_DEFINE_INT32(sa2_native_resolved_read_texels, 4096, "SA2 renderer",
                     "Native renderer: resolved textures of up to this many texels are also copied "
                     "to guest memory, which the game reads for its exposure (0 = "
                     "none; 4096 = 64x64, the ones exposure uses; 57600 = 320x180)");
REXCVAR_DEFINE_BOOL(sa2_native_invalidate_textures_each_copy, false, "SA2 renderer",
                    "Native renderer: drop the texture caches on every copy (the old behavior). "
                    "Otherwise they are dropped only when a resolved texture is created, rebuilt, prepared or "
                    "changes its channel order");
/*
 * Work slots in use (there is room for 4; each costs a 64 MB upload buffer). With 2 the CPU can only be one
 * frame ahead of the GPU and stalls inside the recording call waiting for the previous work; with 3 it can be two
 * ahead. A fourth slot breaks the image: the output slots (kSlotsOutput) are a separate fixed set of 3,
 * and letting the CPU run three frames ahead reuses an output slot whose image is still on display. To raise
 * this number, kSlotsOutput must be raised as well.
 */
REXCVAR_DEFINE_INT32(sa2_native_slots_work, 3, "SA2 renderer",
                     "Native renderer: work slots (1 to 4; 1 for diagnosis). With more, the CPU runs more frames "
                     "ahead of the GPU and stalls less; each costs 64 MB. 2 = the old two-slot behavior")
    .range(1, 4)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// ZCULL (hierarchical depth culling): enabled by removing TRANSFER_DST from the game's depth buffers, which
// makes them eligible for a ZCULL plane in the driver.
REXCVAR_DEFINE_BOOL(sa2_native_zcull, true, "SA2 renderer",
                    "Hierarchical depth culling (ZCULL): removes TRANSFER_DST from the game's "
                    "depth buffers so the driver gives them a ZCULL plane");

// Each Swap's output is painted through a rotating set of 3 output slots (the SDK hands out a mailbox of three
// output images), so the ring does not wait for the previous presentation to finish.
REXCVAR_DEFINE_BOOL(sa2_native_output_no_wait, true, "SA2 renderer",
                    "Native renderer: paint each Swap's output through 3 rotating slots instead of one, so "
                    "the ring does not wait for the GPU to finish the previous output (on the console the ring "
                    "waited 17 ms there per Swap, almost a vsync). Does not change the image; false goes back "
                    "to the old behavior");
// In NVK, TOP_OF_PIPE is PIPELINE_LOCATION_NONE: the timestamp is released when the GPU reads the command,
// not when the previous work finishes, and the per-category breakdown is approximate (a copy gets charged
// with the draw of the previous pass). BOTTOM_OF_PIPE is PIPELINE_LOCATION_ALL: it is released when all
// previous work finishes.

REXCVAR_DEFINE_BOOL(sa2_native_precise_marks, true, "SA2 renderer",
                    "Native renderer (measurement): GPU timestamps between categories use BOTTOM_OF_PIPE "
                    "(released when the previous work finishes) instead of TOP_OF_PIPE (released when the command "
                    "is read): exact per-category breakdown. Does not change the image");
/*
 * GPU gap work (heavy views): every category change writes a GPU timestamp (~60 per frame: passes, EDRAM
 * exports/imports/aliases, copies). In NVK each one is FLUSH_PENDING_WRITES + a report semaphore released
 * after all preceding writes, on top of the wait-for-idle that the pass / compute switches already pay.
 * false = only the first and the last timestamp of each submission are written: the report keeps
 * "total" and "gap" (GPU busy time and idle gap per Swap) but the per-category columns collapse into
 * "other". Does not change the image. Meant to be A/B-tested on the console: compare fps and "total" minus
 * "gap" of the same route with true and false.
 */
// sa2-nx: off by default on the Switch (each mark is a GPU flush there; the report keeps "total" and "gap"), on for
// PC measurements. Set it in the toml to profile per category on the console.
#if REX_PLATFORM_SWITCH
#define SA2_GPU_MARKS_CATEGORIES_DEFAULT false
#else
#define SA2_GPU_MARKS_CATEGORIES_DEFAULT true
#endif
// usf4-nx: Direct3D's Clear through a 4x alias writes its color word into EDRAM tiles that a depth view owns (USF4:
// color base 000/4F8 over the depth views 0E0/060). Without this the color half is declined and the whole clear is
// drawn in the alias, and the depth comes back through 9-pass imports (83 % of them in a fight).
REXCVAR_DEFINE_BOOL(usf4_edram4_clear_into_depth, false, "USF4",
                    "EDRAM mode 4: a redirected color-0 clear whose tiles a D24S8/D24FS8 depth view owns clears that "
                    "view with the same 32-bit word read as depth/stencil (depth = word >> 8, stencil = word & 0xFF)");
// usf4-nx: with the above, the color half went to the depth owners and came back to color through depth exports
// (Switch 0.3.0: category 11 from 210 to ~715 ms per 10 s). The color-0 clear's tiles are stamped like the depth
// redirect's, and the view that pulls them next (seen twice in a row) gets the clear and the tiles directly.
REXCVAR_DEFINE_BOOL(usf4_edram4_clear_color_consumer, false, "USF4",
                    "EDRAM mode 4: a redirected color-0 clear goes straight into the view that read its tiles next "
                    "the last two times (a 1x/2x RGBA8 color view, or a depth view with usf4_edram4_clear_into_depth), "
                    "which takes ownership of them, instead of the tiles' current owners");
REXCVAR_DEFINE_BOOL(sa2_gpu_marks_categories, SA2_GPU_MARKS_CATEGORIES_DEFAULT, "SA2 renderer",
                    "Write a GPU timestamp at every category change (per-category GPU report). false: only the "
                    "first and last timestamp per submission (no per-category breakdown, fewer GPU flushes)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_native_reads_each, 1, "SA2 renderer",
                     "Native renderer: for each target, one of every N small copies is read from and "
                     "written to guest memory (1 = all); the game uses them for its exposure");
// The Xbox 360 passes the image through the gamma ramp the game loads, and MASSEFFECT does not load the identity
// (measured: [64] = 273 and [128] = 539 in 10 bits, instead of 256 and 513).
REXCVAR_DEFINE_BOOL(sa2_native_ramp_gamma, true, "SA2 renderer",
                    "Native renderer: apply the gamma ramp the game loads at the output, like the Xbox 360 "
                    "display (without it, midtones and shadows come out darker). false: the image as is")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace sa2::native {
namespace shaders {
// The same SPIR-V the SDK presenter uses to draw the game image
// (the SDK's vulkan presenter).
#include "vulkan_spirv/guest_output_bilinear_ps.h"
#include "vulkan_spirv/guest_output_triangle_strip_rect_vs.h"
// The same sampling as guest_output_bilinear_ps with the game's gamma ramp
// (precompiled SPIR-V of the ramp pixel shader).
#include "shaders/sa2_output_ramp_gamma_ps.h"
const uint32_t sa2_edram_7e3_to_rgba8_cs[] = {
#include "sa2_edram_7e3_to_rgba8.inc"
};
const uint32_t sa2_edram_rgba8_to_7e3_cs[] = {
#include "sa2_edram_rgba8_to_7e3.inc"
};
const uint32_t sa2_edram_16f_to_16f_cs[] = {
#include "sa2_edram_16f_to_16f.inc"
};
const uint32_t me_resolve_exp_bias_cs[] = {
#include "me_resolve_exp_bias.inc"
};
const uint32_t me_edram_depth_to_rgba8_cs[] = {
#include "me_edram_depth_to_rgba8.inc"
};
const uint32_t me_edram_depth_to_16f_cs[] = {
#include "me_edram_depth_to_16f.inc"
};
const uint32_t me_edram_import_vs[] = {
#include "me_edram_import.inc"
};
const uint32_t me_edram_color_to_depth_fs[] = {
#include "me_edram_color_to_depth.inc"
};
const uint32_t me_edram_color_to_color_fs[] = {
#include "me_edram_color_to_color.inc"
};
const uint32_t me_edram_r64_to_r64_fs[] = {
#include "me_edram_r64_to_r64.inc"
};
const uint32_t sa2_resolve_copy_fs[] = {
#include "sa2_resolve_copy.inc"
};
const uint32_t sa2_depth_resolve_ms_fs[] = {  // sa2_msaa
#include "sa2_depth_resolve_ms.inc"
};
const uint32_t me_edram_raw64_to_rgba8_cs[] = {
#include "me_edram_raw64_to_rgba8.inc"
};
const uint32_t me_edram_rgba8_to_raw64_cs[] = {
#include "me_edram_rgba8_to_raw64.inc"
};
const uint32_t me_edram_raw64_to_16f_cs[] = {
#include "me_edram_raw64_to_16f.inc"
};
const uint32_t me_edram_16f_to_raw64_cs[] = {
#include "me_edram_16f_to_raw64.inc"
};
const uint32_t me_edram_raw64_to_raw64_cs[] = {
#include "me_edram_raw64_to_raw64.inc"
};
const uint32_t me_edram_depth_to_raw64_cs[] = {
#include "me_edram_depth_to_raw64.inc"
};
const uint32_t me_edram_raw64_to_depth_fs[] = {
#include "me_edram_raw64_to_depth.inc"
};
const uint32_t me_depth_resolve_guestspace_cs[] = {
#include "me_depth_resolve_guestspace.inc"
};
const uint32_t me_depth_resolve_guestspace_msaa2_cs[] = {
#include <me_depth_resolve_guestspace_msaa2_generated.inc>
};
const uint32_t me_edram_depth_to_depth_fs[] = {
#include "me_edram_depth_to_depth.inc"
};
const uint32_t me_edram_depth_to_stencil_fs[] = {
#include "me_edram_depth_to_stencil.inc"
};
const uint32_t me_edram_color_to_stencil_fs[] = {
#include "me_edram_color_to_stencil.inc"
};
const uint32_t me_edram_depth_to_depth_msaa2_fs[] = {
// Use the CMake-generated initializer; the frozen embedded copy is a test oracle.
#include <me_edram_depth_to_depth_msaa2_generated.inc>
};
const uint32_t me_edram_depth_msaa2_to_depth_1x_fs[] = {
#include <me_edram_depth_msaa2_to_depth_1x_generated.inc>
};
const uint32_t me_edram_stencil_to_buffer_cs[] = {
#include <me_edram_stencil_to_buffer_generated.inc>
};
const uint32_t me_edram_stencil_msaa2_to_buffer_cs[] = {
#include <me_edram_stencil_msaa2_to_buffer_generated.inc>
};
const uint32_t me_edram_color_stencil_to_buffer_cs[] = {
#include <me_edram_color_stencil_to_buffer_generated.inc>
};
const uint32_t me_edram_depth_to_stencil_msaa2_fs[] = {
#include <me_edram_depth_to_stencil_msaa2_generated.inc>
};
const uint32_t me_edram_depth_msaa2_to_stencil_1x_fs[] = {
#include <me_edram_depth_msaa2_to_stencil_1x_generated.inc>
};
}  // namespace shaders

namespace {

// usf4 0.6.7: a working list kept between calls instead of a std::vector allocated per call. operator new and
// malloc's lock in SynchronizeEDRAM4, CopyTilesEDRAM4 and RedirectClearDepthEDRAM4 showed in the console ring
// profile (0.6.6). One kept vector per nesting level and element type (these functions can run inside each other,
// e.g. a stencil import synchronizing its source); deeper than kLevels, a plain local vector. Ring thread only.
template <typename T>
class KeptList {
 public:
  KeptList() : level_(Level()++) {
    list_ = level_ < kLevels ? &Pool()[level_] : &own_;
    list_->clear();
  }
  ~KeptList() {
    list_->clear();
    --Level();
  }
  KeptList(const KeptList&) = delete;
  KeptList& operator=(const KeptList&) = delete;
  std::vector<T>& get() { return *list_; }

 private:
  static constexpr int kLevels = 4;
  static int& Level() {
    static thread_local int level = 0;
    return level;
  }
  static std::array<std::vector<T>, kLevels>& Pool() {
    static thread_local std::array<std::vector<T>, kLevels> pool;
    return pool;
  }
  int level_;
  std::vector<T>* list_;
  std::vector<T> own_;
};

namespace xenos = rex::graphics::xenos;
using rex::ui::vulkan::VulkanDevice;
using rex::ui::vulkan::VulkanPresenter;

constexpr VkFormat kColorFormat = VK_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t kMaxTargetHeight = 2048;
// sa2-nx: 2048 (was 256). SA2's split-screen gameplay records ~440 EDRAM conversions per frame, so 256 ran out about
// once a frame and each rollover is a fenced SendWork (a full CPU <-> GPU drain plus an extra submission). The sets
// are tiny (one or two image descriptors); the commands and their order are unchanged.
constexpr uint32_t kConversionsEDRAMPerSlot = 2048;
// usf4-nx: the mode-4 EDRAM model has 4096 tiles, twice the Xbox 360's 2048. A surface is still at most 2048 tiles;
// the room past 2048 only matters when a surface would run past the end of the real EDRAM and wrap onto tile 0. The
// one-tile scene (usf4_single_tile) does that: three 1280x720 surfaces (color 0, RT1, depth at 1472) need 2192 tiles,
// so the depth's last 144 tiles wrapped onto color 0's first rows, and the two took those tiles from each other on
// every draw (each change of owner ends the publish fast path for everyone: 30 M tiles walked per second on the
// Switch, the ring thread's top cost). Without wrapping each keeps its own tiles; a game that fits in 2048 tiles
// (USF4's three strips: 0..1918) sees no difference.
constexpr uint32_t kTilesEDRAM4 = me::native::kOwnershipEdramTiles;
constexpr uint32_t kMaskEDRAM4 = kTilesEDRAM4 - 1;
constexpr VkImageSubresourceRange kRangeColor = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
constexpr VkImageSubresourceRange kRangeDepth = {
    VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT, 0, 1, 0, 1};
using FnCopyImage = void(VKAPI_PTR*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage,
                                        VkImageLayout, uint32_t, const VkImageCopy*);
using FnClearDepth = void(VKAPI_PTR*)(VkCommandBuffer, VkImage, VkImageLayout,
                                             const VkClearDepthStencilValue*, uint32_t,
                                             const VkImageSubresourceRange*);
using FnBlit = void(VKAPI_PTR*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout,
                                uint32_t, const VkImageBlit*, VkFilter);

uint8_t ClassEdramFORMAT(uint32_t format) {
  using F = xenos::ColorRenderTargetFormat;
  switch (F(format)) {
    case F::k_8_8_8_8:
    case F::k_8_8_8_8_GAMMA: return 0;
    case F::k_2_10_10_10:
    case F::k_2_10_10_10_AS_10_10_10_10: return 2;
    case F::k_2_10_10_10_FLOAT:
    case F::k_2_10_10_10_FLOAT_AS_16_16_16_16: return 3;
    case F::k_16_16_16_16_FLOAT: return 7;
    default: return uint8_t(format);
  }
}

// Direct3D 11 16.8 fixed point with rounding, like ui::FloatToD3D11Fixed16p8.
int32_t Fixed16p8(float value) {
  if (!(std::abs(value) >= 1.0f / 512.0f)) {
    return 0;
  }
  const double scaled = std::clamp(double(value) * 256.0, -2147483392.0, 2147483392.0);
  return int32_t(std::lround(scaled));
}

// Address of a texel in a 32x32-tiled texture (same layout as the SDK's texture util).
int32_t TileDisplacement2D(int32_t x, int32_t y, uint32_t pitch, uint32_t log2_bytes) {
  pitch = (pitch + 31) & ~uint32_t(31);
  const int32_t macro = ((x >> 5) + (y >> 5) * int32_t(pitch >> 5)) << (log2_bytes + 7);
  const int32_t micro = ((x & 7) + ((y & 0xE) << 2)) << log2_bytes;
  const int32_t offset = macro + ((micro & ~0xF) << 1) + (micro & 0xF) + ((y & 1) << 4);
  return ((offset & ~0x1FF) << 3) + ((y & 16) << 7) + ((offset & 0x1C0) << 2) +
         (((((y & 8) >> 2) + (x >> 3)) & 3) << 6) + (offset & 0x3F);
}

// With 4-byte texels, TileDisplacement2D(x, y, pitch, 2) is
// (((x >> 5) + (y >> 5) * (pitch aligned to 32 >> 5)) << 12) + table[(y & 31) * 32 + (x & 31)]:
// the part inside the 32x32 tile does not depend on the pitch (WriteReads).
const std::array<uint16_t, 1024>& TableTile2DTexel4() {
  static const std::array<uint16_t, 1024> table = [] {
    std::array<uint16_t, 1024> t{};
    for (int32_t y = 0; y < 32; ++y) {
      for (int32_t x = 0; x < 32; ++x) {
        t[size_t(y) * 32 + size_t(x)] = uint16_t(TileDisplacement2D(x, y, 32, 2));
      }
    }
    return t;
  }();
  return table;
}

int32_t ExtenderSign15(uint32_t value) {
  return int32_t(value << 17) >> 17;
}

using Image = ImageNative;  // prepared = already in GENERAL and cleared

struct Resolved {
  Image image;
  uint32_t guest_format = 0;
  bool swap_rb = false;
  // Times the draws have requested this address since the last report. It lives here and not in the report
  // map because ResolvedTexture is on the hot path (one call per texture and draw, ~8,000 per frame) and the
  // lookup of the resolved texture is done anyway.
  uint64_t reads = 0;
  uint64_t revision = 0;
  // sa2_gpu_resolve_alias: its bytes are still in another texture (ContextTargets::aliases_).
  bool alias_pending = false;
};

struct ResolvedClip {
  uint32_t address = 0;
  Image image;
  VkImage source = VK_NULL_HANDLE;
  uint64_t revision = 0;
  VkDeviceSize bytes = 0;
};

// Allocation cache only, never an alternative authoritative texture at this address.
struct ResolvedSleeping {
  uint32_t base = 0;
  Image image;
  VkDeviceSize bytes = 0;
  uint64_t order_retirement = 0;
};

// report of readbacks per render target (base and size) and cadence of sa2_native_reads_each.
struct TargetRead {
  uint32_t base = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t copies = 0;    // small copies seen in the interval
  uint64_t skipped = 0;  // of those, the ones not read because of sa2_native_reads_each
};

// Copy inventory (NoteCopy): every resolved texture, with its size and how many times each one is read. That answers whether a
// copy is needed: a resolved texture that is copied every frame and never requested is wasted bandwidth.
struct CopyTarget {
  uint32_t width = 0;
  uint32_t height = 0;
  uint64_t copies = 0;
  uint64_t with_draws = 0;  // copies with some draw since the previous copy
  uint64_t draws = 0;
  uint64_t pixels = 0;
};

// What is known about a render target between game commands. It serves two purposes:
//  - skipping a clear that changes nothing (sa2_native_skip_cleared_repeated);
//  - restoring only the rows the game really uses (sa2_native_restore_area_util).
// It lives in a separate map and not in ImageNative because that structure belongs to another file.
struct TargetState {
  bool clean_clear = false;    // the content is exactly the last clear, with nothing on top
  uint64_t clear_value = 0;     // packed color, or depth+stencil
  uint64_t draws_on_clear = 0; // global draw counter at that moment
  uint32_t used_height = 0;        // the largest y1 the game has resolved from this render target
};

// Readback of a small resolved texture (sa2_native_resolved_read_texels):
// host-visible buffer it is copied to and what is needed to write it to the guest.
struct Read {
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  uint8_t* data = nullptr;
  VkDeviceSize bytes = 0;
  bool coherent = true;
};
struct PendingRead {
  Read* read;
  uint32_t base;          // RB_COPY_DEST_BASE
  int32_t x0;             // texel of the texture where the rectangle starts
  int32_t y0;
  uint32_t width;
  uint32_t height;
  uint32_t pitch;         // RB_COPY_DEST_PITCH
  uint32_t target_height;
  uint32_t info;          // RB_COPY_DEST_INFO
};

// Output slots with sa2_native_output_no_wait (without it, only slot 0 is used).
constexpr uint32_t kSlotsOutput = 3;
// GPU timestamps per slot (the last one is kept for the final timestamp).
// ME: mode-4 EDRAM operations mark their own GPU time. Eden Prime records a whole frame in one submission
// with several hundred category changes: at 1024 the marks ran out and the rest of the frame was billed to
// the last category (a saturated, meaningless "edram_import 207 ms").
constexpr uint32_t kMarksPerSlot = 8192;
constexpr uint8_t kGpuEnd = 0xFF;
// Passes measured per work unit (~16 per frame including the resumed ones).
constexpr uint32_t kStatsPerSlot = 64;
constexpr uint32_t kCountersStat = 3;  // vertices, clipped primitives and fragments
// Draws measured in a diagnostic frame (the scene has ~1200).
constexpr uint32_t kStatsDrawPerSlot = 8192;  // ME frames exceed 2048 draws
constexpr uint32_t kLabelsShader = 512;
// Buckets of the copy breakdown by size (pixels of the copy).

// One of the work slots: while a frame is being recorded, the previous ones can still be
// on the GPU with the other one.
struct SlotWork {
  VkCommandPool pool_work = VK_NULL_HANDLE;
  VkCommandPool pool_upload = VK_NULL_HANDLE;
  VkCommandBuffer work = VK_NULL_HANDLE;
  VkCommandBuffer upload = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  // usf4_deferred_async_submit: the worker queue position of this slot's last submission (0 = submitted directly).
  uint64_t ticket = 0;
  // Descriptor sets for typeless EDRAM conversions are reset only after this
  // slot's fence, so every dispatch keeps the views recorded with it.
  VkDescriptorPool pool_conversion_edram = VK_NULL_HANDLE;
  VkDescriptorPool pool_conversion_depth_edram = VK_NULL_HANDLE;
  VkDescriptorPool pool_import_depth_edram = VK_NULL_HANDLE;
  VkDescriptorPool pool_conv_color_frag = VK_NULL_HANDLE;  // sa2_native_conversion_frag
  VkDescriptorPool pool_depth_resolve_ms = VK_NULL_HANDLE;  // sa2_msaa: depth resolves of multisampled targets
  // Copy-engine stencil import (Mass Effect mode 4): bytes computed into this buffer, then copied.
  VkDescriptorPool pool_stencil_copy = VK_NULL_HANDLE;
  VkBuffer buffer_stencil_copy = VK_NULL_HANDLE;
  VkDeviceMemory memory_stencil_copy = VK_NULL_HANDLE;
  uint32_t conversions_edram = 0;
  bool pending = false;
  uint64_t order = 0;                        // submission number
  // Wall-clock time of the work, from vkQueueSubmit to the signaled fence. The GPU timestamps give
  // 30.5 ms per frame while the hardware's own counter says 99.7 % load on a 55 ms frame: either there is
  // work we do not mark or the timestamp scale is wrong. This bounds the truth from above (the CPU sees
  // the fence a little late) and the timestamps from below.
  std::chrono::steady_clock::time_point sent{};
  std::vector<PendingRead> reads;  // readbacks submitted with the work
  std::vector<uint8_t> categories;  // GPU category of each timestamp written
  // sa2-nx: timestamp queries of this slot written since its last reset (all of them before the first recording):
  // BeginRecording resets only those instead of all kMarksPerSlot (a copy-engine fill of 8192 queries per
  // submission on NVK).
  uint32_t marks_written = kMarksPerSlot;
  std::vector<uint32_t> labels;  // first draw (VS << 16 | PS) of each mark's interval, 0 = none
  std::vector<uint64_t> descriptions;  // the pass of each mark (DescribeMarkGpu)
  std::vector<uint32_t> draws_mark;  // draws recorded in each mark's interval
  bool precise_marks = false;      // intermediate marks with BOTTOM_OF_PIPE
  // Statistics queries recorded, with the category of each one's pass.
  std::vector<std::pair<uint32_t, uint8_t>> stats;
  // Per-draw queries, with the pixel shader number of each one.
  std::vector<std::pair<uint32_t, uint16_t>> stats_draw;
  // the full pixel shader number + 1 and category (<< 24) of each per-draw query (the
  // uint16 label above folds ME's ~30000 pixel shaders into 512).
  std::vector<uint32_t> draw_ps;
  std::vector<uint32_t> draw_vs;
};

class TargetsVulkan final : public TargetsNative, public ContextTargets {
 public:
  TargetsVulkan(const VulkanDevice* vulkan_device, rex::memory::Memory* memory)
      : vulkan_device_(vulkan_device),
        dfn_(deferred::Table(vulkan_device->functions())),
        device_(vulkan_device->device()),
        memory_(memory),
        family_(vulkan_device->queue_family_graphics_compute()) {
    // Mass Effect (Switch): any EDRAM mode. Outside mode 4 every resolve whose size or format changed at
    // the same address took a full GPU drain (site 4 of WaitGpu: up to ~16 per second on the console,
    // ~4 s of fence waits per 10 s) instead of retiring the old image behind its submission fence.
    resolved_pool_startup_eligible_ =
        !REXCVAR_GET(sa2_native_resolver_no_copy);
  }

  ~TargetsVulkan() override {
    ++waits_gpu_reason_[1]; WaitGpu();
    draws_.reset();  // their framebuffers and views point to these images
    for (const auto& [image, framebuffer] : framebuffers_import_depth_edram_)
      if (framebuffer) dfn_.vkDestroyFramebuffer(device_, framebuffer, nullptr);
    framebuffers_import_depth_edram_.clear();
    for (const auto& [key, framebuffer] : framebuffers_conv_color_frag_)
      if (framebuffer) dfn_.vkDestroyFramebuffer(device_, framebuffer, nullptr);
    framebuffers_conv_color_frag_.clear();
    for (const auto& [image, view] : views_raw64_rt_edram_) dfn_.vkDestroyImageView(device_, view, nullptr);
    views_raw64_rt_edram_.clear();
    if (fs_conv_r64_frag_) dfn_.vkDestroyShaderModule(device_, fs_conv_r64_frag_, nullptr);
    if (fs_resolve_copy_) dfn_.vkDestroyShaderModule(device_, fs_resolve_copy_, nullptr);
    for (const auto& [image, framebuffer] : framebuffers_depth_resolve_ms_)  // sa2_msaa
      if (framebuffer) dfn_.vkDestroyFramebuffer(device_, framebuffer, nullptr);
    for (const auto& [format, pass] : passes_depth_resolve_ms_) {
      if (pass.pipeline) dfn_.vkDestroyPipeline(device_, pass.pipeline, nullptr);
      if (pass.render_pass) dfn_.vkDestroyRenderPass(device_, pass.render_pass, nullptr);
    }
    if (fs_depth_resolve_ms_) dfn_.vkDestroyShaderModule(device_, fs_depth_resolve_ms_, nullptr);
    if (vs_depth_resolve_ms_) dfn_.vkDestroyShaderModule(device_, vs_depth_resolve_ms_, nullptr);
    if (layout_pipeline_depth_resolve_ms_)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_depth_resolve_ms_, nullptr);
    if (layout_depth_resolve_ms_) dfn_.vkDestroyDescriptorSetLayout(device_, layout_depth_resolve_ms_, nullptr);
    for (const auto& [format, pass] : passes_conv_color_frag_) {
      if (pass.pipeline) dfn_.vkDestroyPipeline(device_, pass.pipeline, nullptr);
      if (pass.render_pass) dfn_.vkDestroyRenderPass(device_, pass.render_pass, nullptr);
    }
    if (vs_conv_color_frag_) dfn_.vkDestroyShaderModule(device_, vs_conv_color_frag_, nullptr);
    if (fs_conv_color_frag_) dfn_.vkDestroyShaderModule(device_, fs_conv_color_frag_, nullptr);
    if (layout_pipeline_conv_color_frag_)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_conv_color_frag_, nullptr);
    if (layout_conv_color_frag_)
      dfn_.vkDestroyDescriptorSetLayout(device_, layout_conv_color_frag_, nullptr);
    for (const auto& [format, pass] : passes_import_depth_edram_) {
      for (VkPipeline p : pass.pipelines) if (p) dfn_.vkDestroyPipeline(device_, p, nullptr);
      if (pass.render_pass) dfn_.vkDestroyRenderPass(device_, pass.render_pass, nullptr);
    }
    if (vs_import_depth_edram_) dfn_.vkDestroyShaderModule(device_, vs_import_depth_edram_, nullptr);
    if (fs_import_depth_edram_) dfn_.vkDestroyShaderModule(device_, fs_import_depth_edram_, nullptr);
    if (fs_depth_to_depth_edram_) dfn_.vkDestroyShaderModule(device_, fs_depth_to_depth_edram_, nullptr);
    if (fs_depth_to_stencil_edram_) dfn_.vkDestroyShaderModule(device_, fs_depth_to_stencil_edram_, nullptr);
    if (fs_color_to_stencil_edram_) dfn_.vkDestroyShaderModule(device_, fs_color_to_stencil_edram_, nullptr);
    if (fs_stencil_msaa2_edram_) dfn_.vkDestroyShaderModule(device_, fs_stencil_msaa2_edram_, nullptr);
    if (fs_stencil_msaa2_to_1x_edram_) dfn_.vkDestroyShaderModule(device_, fs_stencil_msaa2_to_1x_edram_, nullptr);
    if (fs_depth_to_depth_msaa2_edram_)
      dfn_.vkDestroyShaderModule(device_, fs_depth_to_depth_msaa2_edram_, nullptr);
    if (fs_depth_msaa2_to_depth_1x_edram_)
      dfn_.vkDestroyShaderModule(device_, fs_depth_msaa2_to_depth_1x_edram_, nullptr);
    if (pipeline_resolve_depth_msaa2_edram_)
      dfn_.vkDestroyPipeline(device_, pipeline_resolve_depth_msaa2_edram_, nullptr);
    if (shader_resolve_depth_msaa2_edram_)
      dfn_.vkDestroyShaderModule(device_, shader_resolve_depth_msaa2_edram_, nullptr);
    if (fs_raw64_to_depth_edram_) dfn_.vkDestroyShaderModule(device_, fs_raw64_to_depth_edram_, nullptr);
    for (const auto& [image, view] : views_raw64_edram_) dfn_.vkDestroyImageView(device_, view, nullptr);
    views_raw64_edram_.clear();
    if (layout_pipeline_import_depth_edram_)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_import_depth_edram_, nullptr);
    if (layout_import_depth_edram_)
      dfn_.vkDestroyDescriptorSetLayout(device_, layout_import_depth_edram_, nullptr);
    for (const auto& [image, views] : views_depth_edram_) {
      if (views.depth) dfn_.vkDestroyImageView(device_, views.depth, nullptr);
      if (views.stencil) dfn_.vkDestroyImageView(device_, views.stencil, nullptr);
    }
    views_depth_edram_.clear();
    for (auto& [key, image] : depths_) {
      Destroy(image);
    }
    for (auto& [key, image] : targets_) {
      Destroy(image);
    }
    for (auto& [key, resolved] : resolved_) {
      Destroy(resolved.image);
    }
    for (auto& [key, resolved] : parked_resolved_) Destroy(resolved.image);  // banjo-nx
    for (auto& sleeping : resolved_sleeping_) Destroy(sleeping.image);
    for (auto& clip : resolved_clips_) Destroy(clip.image);
    for (auto& [key, staging] : staging_reads_) Destroy(staging);  // sa2-nx render resolution
    for (ImageFront& replenished : fronts_images_) {  // sa2_native_lazy_front
      Destroy(replenished.image);
    }
    pending_reads_.clear();
    for (SlotWork& slot : slots_) {
      slot.reads.clear();
    }
    for (auto& [key, read] : reads_) {
      DestroyRead(read);
    }
    for (auto& f : framebuffers_) {
      if (f.framebuffer != VK_NULL_HANDLE) {
        dfn_.vkDestroyFramebuffer(device_, f.framebuffer, nullptr);
      }
    }
    if (pipeline_ != VK_NULL_HANDLE) dfn_.vkDestroyPipeline(device_, pipeline_, nullptr);
    for (VkPipeline p : pipelines_conversion_edram_) {
      if (p != VK_NULL_HANDLE) dfn_.vkDestroyPipeline(device_, p, nullptr);
    }
    for (VkShaderModule m : shaders_conversion_edram_) {
      if (m != VK_NULL_HANDLE) dfn_.vkDestroyShaderModule(device_, m, nullptr);
    }
    if (layout_pipeline_conversion_edram_ != VK_NULL_HANDLE)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_conversion_edram_, nullptr);
    if (layout_conversion_edram_ != VK_NULL_HANDLE)
      dfn_.vkDestroyDescriptorSetLayout(device_, layout_conversion_edram_, nullptr);
    for (VkPipeline p : pipelines_conversion_depth_edram_)
      if (p) dfn_.vkDestroyPipeline(device_, p, nullptr);
    for (VkShaderModule m : shaders_conversion_depth_edram_)
      if (m) dfn_.vkDestroyShaderModule(device_, m, nullptr);
    if (layout_pipeline_conversion_depth_edram_)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_conversion_depth_edram_, nullptr);
    if (layout_conversion_depth_edram_)
      dfn_.vkDestroyDescriptorSetLayout(device_, layout_conversion_depth_edram_, nullptr);
    if (sampler_depth_edram_) dfn_.vkDestroySampler(device_, sampler_depth_edram_, nullptr);
    for (VkPipeline p : pipelines_ramp_) {
      if (p != VK_NULL_HANDLE) dfn_.vkDestroyPipeline(device_, p, nullptr);
    }
    if (fs_ramp_ != VK_NULL_HANDLE) dfn_.vkDestroyShaderModule(device_, fs_ramp_, nullptr);
    for (RampOutput& ramp : ramps_output_) {
      if (ramp.buffer != VK_NULL_HANDLE) dfn_.vkDestroyBuffer(device_, ramp.buffer, nullptr);
      if (ramp.memory != VK_NULL_HANDLE) dfn_.vkFreeMemory(device_, ramp.memory, nullptr);
    }
    if (layout_pipeline_ != VK_NULL_HANDLE)
      dfn_.vkDestroyPipelineLayout(device_, layout_pipeline_, nullptr);
    if (pool_descriptors_ != VK_NULL_HANDLE)
      dfn_.vkDestroyDescriptorPool(device_, pool_descriptors_, nullptr);
    if (layout_descriptors_ != VK_NULL_HANDLE)
      dfn_.vkDestroyDescriptorSetLayout(device_, layout_descriptors_, nullptr);
    if (sampler_ != VK_NULL_HANDLE) dfn_.vkDestroySampler(device_, sampler_, nullptr);
    if (vs_ != VK_NULL_HANDLE) dfn_.vkDestroyShaderModule(device_, vs_, nullptr);
    if (fs_ != VK_NULL_HANDLE) dfn_.vkDestroyShaderModule(device_, fs_, nullptr);
    if (render_pass_output_ != VK_NULL_HANDLE)
      dfn_.vkDestroyRenderPass(device_, render_pass_output_, nullptr);
    for (SlotWork& slot : slots_) {
      if (slot.pool_conversion_edram != VK_NULL_HANDLE)
        dfn_.vkDestroyDescriptorPool(device_, slot.pool_conversion_edram, nullptr);
      if (slot.pool_conversion_depth_edram)
        dfn_.vkDestroyDescriptorPool(device_, slot.pool_conversion_depth_edram, nullptr);
      if (slot.pool_import_depth_edram)
        dfn_.vkDestroyDescriptorPool(device_, slot.pool_import_depth_edram, nullptr);
      if (slot.pool_conv_color_frag)
        dfn_.vkDestroyDescriptorPool(device_, slot.pool_conv_color_frag, nullptr);
      if (slot.pool_depth_resolve_ms)
        dfn_.vkDestroyDescriptorPool(device_, slot.pool_depth_resolve_ms, nullptr);
      if (slot.pool_stencil_copy) dfn_.vkDestroyDescriptorPool(device_, slot.pool_stencil_copy, nullptr);
      if (slot.buffer_stencil_copy) dfn_.vkDestroyBuffer(device_, slot.buffer_stencil_copy, nullptr);
      if (slot.memory_stencil_copy) dfn_.vkFreeMemory(device_, slot.memory_stencil_copy, nullptr);
      if (slot.fence != VK_NULL_HANDLE) dfn_.vkDestroyFence(device_, slot.fence, nullptr);
      if (slot.pool_work != VK_NULL_HANDLE)
        dfn_.vkDestroyCommandPool(device_, slot.pool_work, nullptr);
      if (slot.pool_upload != VK_NULL_HANDLE)
        dfn_.vkDestroyCommandPool(device_, slot.pool_upload, nullptr);
    }
    if (queries_ != VK_NULL_HANDLE) dfn_.vkDestroyQueryPool(device_, queries_, nullptr);
    if (marks_draw_ != VK_NULL_HANDLE) dfn_.vkDestroyQueryPool(device_, marks_draw_, nullptr);
    for (uint32_t i = 0; i < kSlotsOutput; ++i) {
      if (fences_output_[i] != VK_NULL_HANDLE) dfn_.vkDestroyFence(device_, fences_output_[i], nullptr);
      if (pools_output_[i] != VK_NULL_HANDLE) dfn_.vkDestroyCommandPool(device_, pools_output_[i], nullptr);
    }
  }

  bool Initialize() {
    // sa2-nx render resolution: the starting scale (handheld / docked height), before any image exists.
    scale_q_ = sa2::render::WantedScaleQuarters();
    REXLOG_INFO("[res] native renderer: render targets and resolved textures at {}/4 of the guest size ({}x{} for the "
                "game's 1280x720)", scale_q_, ScaleSize(1280, scale_q_), ScaleSize(720, scale_q_));
    // sa2_msaa: host multisampling of the game's single-sample render targets, when the device has it for color,
    // depth/stencil attachments and sampling (the depth resolve reads samples), plus vkCmdResolveImage.
    msaa_samples_ = sa2::render::MsaaSamples();
    if (msaa_samples_ > 1) {
      const auto& properties = vulkan_device_->properties();
      const VkSampleCountFlags bit = VkSampleCountFlags(msaa_samples_);
      const bool supported = (properties.framebufferColorSampleCounts & bit) &&
                             (properties.framebufferDepthSampleCounts & bit) &&
                             (properties.framebufferStencilSampleCounts & bit) &&
                             (properties.sampledImageDepthSampleCounts & bit);
      if (supported)
        resolve_image_ = reinterpret_cast<FnResolve>(deferred::Proc(
            vulkan_device_->vulkan_instance()->functions().vkGetDeviceProcAddr, device_, "vkCmdResolveImage"));
      if (!supported || !resolve_image_) {
        REXLOG_WARN("[res] MSAA {}x is not available on this device: single sample", msaa_samples_);
        msaa_samples_ = 1;
      } else {
        REXLOG_INFO("[res] MSAA: the game's single-sample render targets with {} samples per pixel, resolved at the "
                    "game's copies (sa2_msaa)", msaa_samples_);
      }
    }
    // vkCmdCopyImage is not in the SDK's function table: it is requested from the driver.
    copy_image_ = reinterpret_cast<FnCopyImage>(
        deferred::Proc(vulkan_device_->vulkan_instance()->functions().vkGetDeviceProcAddr, device_, "vkCmdCopyImage"));
    if (!copy_image_) {
      REXLOG_ERROR("[native] targets: the driver does not provide vkCmdCopyImage");
      return false;
    }
    clear_depth_ = reinterpret_cast<FnClearDepth>(
        deferred::Proc(vulkan_device_->vulkan_instance()->functions().vkGetDeviceProcAddr, device_, "vkCmdClearDepthStencilImage"));
    blit_ = reinterpret_cast<FnBlit>(
        deferred::Proc(vulkan_device_->vulkan_instance()->functions().vkGetDeviceProcAddr, device_, "vkCmdBlitImage"));
    // Attachment, copy and clear source and destination, and sampling of the resolved ones (shadows).
    const VkFormatFeatureFlags kUsesDepth =
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        VK_FORMAT_FEATURE_TRANSFER_SRC_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    // To draw the shadow map smaller it has to be scaled up when resolving it, and that is a vkCmdBlitImage on
    // depth. If the driver does not provide it, nothing is scaled.
    const VkFormatFeatureFlags kScaledUses =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    VkFormatProperties raw64_properties{};
    vulkan_device_->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
        vulkan_device_->physical_device(), VK_FORMAT_R16G16B16A16_UINT, &raw64_properties);
    const VkFormatFeatureFlags raw64_needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT;
    raw64_edram_supported_ = (raw64_properties.optimalTilingFeatures & raw64_needed) == raw64_needed;
    VkFormatFeatureFlags chosen_uses = 0;
    for (const VkFormat candidate : {VK_FORMAT_D24_UNORM_S8_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT}) {
      VkFormatProperties properties{};
      vulkan_device_->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
          vulkan_device_->physical_device(), candidate, &properties);
      const VkFormatFeatureFlags uses = properties.optimalTilingFeatures & kUsesDepth;
      if (!(uses & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT)) {
        continue;
      }
      if (format_depth_ == VK_FORMAT_UNDEFINED || uses == kUsesDepth) {
        format_depth_ = candidate;
        chosen_uses = uses;
        scalable_depth_ =
            (properties.optimalTilingFeatures & kScaledUses) == kScaledUses;
      }
      if (uses == kUsesDepth) {
        break;
      }
    }
    if (format_depth_ != VK_FORMAT_UNDEFINED) {
      const VkFormatFeatureFlags missing = kUsesDepth & ~chosen_uses;
      REXLOG_INFO("[native] targets: depth format {}{}{}{}",
                  format_depth_ == VK_FORMAT_D24_UNORM_S8_UINT ? "D24S8" : "D32S8",
                  (missing & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) ? ", no sampling" : "",
                  (missing & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT) ? ", no copy source" : "",
                  (missing & VK_FORMAT_FEATURE_TRANSFER_DST_BIT) ? ", no copy destination" : "");
      if (!scalable_depth_) {
        REXLOG_INFO("[native] targets: the driver cannot scale depth: the shadow map stays at its size");
      }
    }
    // One pool per command buffer: the SDK table does not include
    // vkResetCommandBuffer either, so the whole pool is reset.
    VkCommandPoolCreateInfo info_pool{};
    info_pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info_pool.queueFamilyIndex = family_;
    VkCommandBufferAllocateInfo reserve{};
    reserve.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    reserve.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    reserve.commandBufferCount = 1;
    VkFenceCreateInfo info_fence{};
    info_fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    for (SlotWork& slot : slots_) {
      if (dfn_.vkCreateCommandPool(device_, &info_pool, nullptr, &slot.pool_work) !=
              VK_SUCCESS ||
          dfn_.vkCreateCommandPool(device_, &info_pool, nullptr, &slot.pool_upload) !=
              VK_SUCCESS ||
          dfn_.vkCreateFence(device_, &info_fence, nullptr, &slot.fence) != VK_SUCCESS) {
        return false;
      }
      reserve.commandPool = slot.pool_work;
      if (dfn_.vkAllocateCommandBuffers(device_, &reserve, &slot.work) != VK_SUCCESS) {
        return false;
      }
      reserve.commandPool = slot.pool_upload;
      if (dfn_.vkAllocateCommandBuffers(device_, &reserve, &slot.upload) != VK_SUCCESS) {
        return false;
      }
    }
    // Typeless EDRAM conversion. Work is expressed in physical 32-bit EDRAM words; true 64-bpp
    // render targets consume two adjacent words per pixel.
    VkDescriptorSetLayoutBinding bindings_conversion[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      bindings_conversion[i].binding = i;
      bindings_conversion[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      bindings_conversion[i].descriptorCount = 1;
      bindings_conversion[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo info_layout_conversion{};
    info_layout_conversion.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info_layout_conversion.bindingCount = 2;
    info_layout_conversion.pBindings = bindings_conversion;
    if (dfn_.vkCreateDescriptorSetLayout(device_, &info_layout_conversion, nullptr,
                                         &layout_conversion_edram_) != VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange range_conversion{};
    range_conversion.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range_conversion.size = sizeof(uint32_t) * 19;  // sa2-nx: + the scaled tile size (tile_width, tile_height)
    VkPipelineLayoutCreateInfo info_layout_pipeline_conversion{};
    info_layout_pipeline_conversion.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info_layout_pipeline_conversion.setLayoutCount = 1;
    info_layout_pipeline_conversion.pSetLayouts = &layout_conversion_edram_;
    info_layout_pipeline_conversion.pushConstantRangeCount = 1;
    info_layout_pipeline_conversion.pPushConstantRanges = &range_conversion;
    if (dfn_.vkCreatePipelineLayout(device_, &info_layout_pipeline_conversion, nullptr,
                                    &layout_pipeline_conversion_edram_) != VK_SUCCESS) {
      return false;
    }
const std::array<std::pair<const uint32_t*, size_t>, 9> codes_conversion{{
        {shaders::sa2_edram_7e3_to_rgba8_cs, sizeof(shaders::sa2_edram_7e3_to_rgba8_cs)},
        {shaders::sa2_edram_rgba8_to_7e3_cs, sizeof(shaders::sa2_edram_rgba8_to_7e3_cs)},
        {shaders::sa2_edram_16f_to_16f_cs, sizeof(shaders::sa2_edram_16f_to_16f_cs)},
        {shaders::me_resolve_exp_bias_cs, sizeof(shaders::me_resolve_exp_bias_cs)},
        {shaders::me_edram_raw64_to_rgba8_cs, sizeof(shaders::me_edram_raw64_to_rgba8_cs)},
        {shaders::me_edram_rgba8_to_raw64_cs, sizeof(shaders::me_edram_rgba8_to_raw64_cs)},
        {shaders::me_edram_raw64_to_16f_cs, sizeof(shaders::me_edram_raw64_to_16f_cs)},
        {shaders::me_edram_16f_to_raw64_cs, sizeof(shaders::me_edram_16f_to_raw64_cs)},
        {shaders::me_edram_raw64_to_raw64_cs, sizeof(shaders::me_edram_raw64_to_raw64_cs)},
    }};
    for (uint32_t i = 0; i < codes_conversion.size(); ++i) {
      VkShaderModuleCreateInfo info_module_conversion{};
      info_module_conversion.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
      info_module_conversion.codeSize = codes_conversion[i].second;
      info_module_conversion.pCode = codes_conversion[i].first;
      if (dfn_.vkCreateShaderModule(device_, &info_module_conversion, nullptr,
                                    &shaders_conversion_edram_[i]) != VK_SUCCESS) {
        return false;
      }
      VkComputePipelineCreateInfo info_pipeline_conversion{};
      info_pipeline_conversion.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
      info_pipeline_conversion.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      info_pipeline_conversion.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      info_pipeline_conversion.stage.module = shaders_conversion_edram_[i];
      info_pipeline_conversion.stage.pName = "main";
      info_pipeline_conversion.layout = layout_pipeline_conversion_edram_;
      if (dfn_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &info_pipeline_conversion,
                                        nullptr, &pipelines_conversion_edram_[i]) != VK_SUCCESS) {
        return false;
      }
    }
    VkDescriptorPoolSize pool_size_conversion{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                 kConversionsEDRAMPerSlot * 2};
    VkDescriptorPoolCreateInfo info_pool_conversion{};
    info_pool_conversion.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info_pool_conversion.maxSets = kConversionsEDRAMPerSlot;
    info_pool_conversion.poolSizeCount = 1;
    info_pool_conversion.pPoolSizes = &pool_size_conversion;
    for (SlotWork& slot : slots_) {
      if (dfn_.vkCreateDescriptorPool(device_, &info_pool_conversion, nullptr,
                                      &slot.pool_conversion_edram) != VK_SUCCESS) {
        return false;
      }
    }
    // Independent layout: sampled depth/stencil aspects cannot use storage-image descriptors.
    VkDescriptorSetLayoutBinding bindings_depth[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
      bindings_depth[i].binding = i;
      bindings_depth[i].descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                              : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      bindings_depth[i].descriptorCount = 1;
      bindings_depth[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    info_layout_conversion.bindingCount = 3;
    info_layout_conversion.pBindings = bindings_depth;
    if (dfn_.vkCreateDescriptorSetLayout(device_, &info_layout_conversion, nullptr,
                                         &layout_conversion_depth_edram_) != VK_SUCCESS) return false;
    info_layout_pipeline_conversion.pSetLayouts = &layout_conversion_depth_edram_;
    if (dfn_.vkCreatePipelineLayout(device_, &info_layout_pipeline_conversion, nullptr,
                                     &layout_pipeline_conversion_depth_edram_) != VK_SUCCESS) return false;
    const std::array<std::pair<const uint32_t*, size_t>, 4> codes_depth{{
        {shaders::me_edram_depth_to_rgba8_cs, sizeof(shaders::me_edram_depth_to_rgba8_cs)},
        {shaders::me_edram_depth_to_16f_cs, sizeof(shaders::me_edram_depth_to_16f_cs)},
        {shaders::me_edram_depth_to_raw64_cs, sizeof(shaders::me_edram_depth_to_raw64_cs)},
        {shaders::me_depth_resolve_guestspace_cs, sizeof(shaders::me_depth_resolve_guestspace_cs)},
    }};
    for (uint32_t i = 0; i < codes_depth.size(); ++i) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = codes_depth[i].second;
      module.pCode = codes_depth[i].first;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr,
                                     &shaders_conversion_depth_edram_[i]) != VK_SUCCESS) return false;
      VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
      pipeline.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
      pipeline.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
      pipeline.stage.module = shaders_conversion_depth_edram_[i];
      pipeline.stage.pName = "main";
      pipeline.layout = layout_pipeline_conversion_depth_edram_;
      if (dfn_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr,
                                          &pipelines_conversion_depth_edram_[i]) != VK_SUCCESS) return false;
    }
    VkDescriptorPoolSize depth_sizes[2] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kConversionsEDRAMPerSlot * 2},
        {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, kConversionsEDRAMPerSlot}};
    info_pool_conversion.poolSizeCount = 2;
    info_pool_conversion.pPoolSizes = depth_sizes;
    for (SlotWork& slot : slots_)
      if (dfn_.vkCreateDescriptorPool(device_, &info_pool_conversion, nullptr,
                                       &slot.pool_conversion_depth_edram) != VK_SUCCESS) return false;
    VkSamplerCreateInfo depth_sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    depth_sampler.magFilter = depth_sampler.minFilter = VK_FILTER_NEAREST;
    depth_sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    depth_sampler.addressModeU = depth_sampler.addressModeV = depth_sampler.addressModeW =
        VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (dfn_.vkCreateSampler(device_, &depth_sampler, nullptr, &sampler_depth_edram_) != VK_SUCCESS)
      return false;
    for (uint32_t i = 0; i < kSlotsOutput; ++i) {
      if (dfn_.vkCreateCommandPool(device_, &info_pool, nullptr, &pools_output_[i]) != VK_SUCCESS) {
        return false;
      }
      reserve.commandPool = pools_output_[i];
      if (dfn_.vkAllocateCommandBuffers(device_, &reserve, &commands_output_[i]) != VK_SUCCESS) {
        return false;
      }
      if (dfn_.vkCreateFence(device_, &info_fence, nullptr, &fences_output_[i]) != VK_SUCCESS) {
        return false;
      }
    }
    used_slots_ = uint32_t(std::clamp(REXCVAR_GET(sa2_native_slots_work), 1,
                                          int32_t(slots_.size())));
    REXLOG_INFO("[native] targets: work slots (sa2_native_slots_work) = {} of {}", used_slots_,
                slots_.size());
    output_no_wait_ = REXCVAR_GET(sa2_native_output_no_wait);
    invalidate_each_copy_ = REXCVAR_GET(sa2_native_invalidate_textures_each_copy);
    REXLOG_INFO("[native] targets: texture caches dropped on every copy (sa2_native_invalidate_textures_each_copy) = {}",
                invalidate_each_copy_ ? "YES" : "no");
    REXLOG_INFO("[native] targets: output does not wait for the previous one (sa2_native_output_no_wait) = {}",
                output_no_wait_ ? "YES" : "no");
    precise_marks_ = REXCVAR_GET(sa2_native_precise_marks);
    marks_categories_ = REXCVAR_GET(sa2_gpu_marks_categories);
    REXLOG_INFO("[native] targets: GPU timestamps per category (sa2_gpu_marks_categories) = {}",
                marks_categories_ ? "YES" : "no");
    REXLOG_INFO("[native] targets: precise GPU timestamps (sa2_native_precise_marks) = {}",
                precise_marks_ ? "YES" : "no");
    // GPU time per work unit: two timestamps per slot. Without timestamp bits on the queue, it is not measured.
    {
      const auto& ifn = vulkan_device_->vulkan_instance()->functions();
      uint32_t families = 0;
      ifn.vkGetPhysicalDeviceQueueFamilyProperties(vulkan_device_->physical_device(), &families,
                                                   nullptr);
      std::vector<VkQueueFamilyProperties> queues(families);
      ifn.vkGetPhysicalDeviceQueueFamilyProperties(vulkan_device_->physical_device(), &families,
                                                   queues.data());
      VkPhysicalDeviceProperties physical{};
      ifn.vkGetPhysicalDeviceProperties(vulkan_device_->physical_device(), &physical);
      write_mark_ = reinterpret_cast<FnWriteMark>(
          deferred::Proc(ifn.vkGetDeviceProcAddr, device_, "vkCmdWriteTimestamp"));
      // usf4-nx: not through the deferred proxy (it drained the worker on every read). Every read is in CompleteA,
      // after the slot's fence: the submission, and so every command that wrote the queries, has been recorded.
      read_queries_ = reinterpret_cast<FnReadQueries>(ifn.vkGetDeviceProcAddr(device_, "vkGetQueryPoolResults"));
      if (family_ < families && queues[family_].timestampValidBits && write_mark_ &&
          read_queries_ && physical.limits.timestampPeriod > 0.0f) {
        VkQueryPoolCreateInfo info_queries{};
        info_queries.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info_queries.queryType = VK_QUERY_TYPE_TIMESTAMP;
        info_queries.queryCount = uint32_t(slots_.size() * kMarksPerSlot);
        if (dfn_.vkCreateQueryPool(device_, &info_queries, nullptr, &queries_) != VK_SUCCESS) {
          queries_ = VK_NULL_HANDLE;
        }
        period_mark_ns_ = physical.limits.timestampPeriod;
      }
      REXLOG_INFO("[native] targets: GPU time per Swap {}",
                  queries_ != VK_NULL_HANDLE ? "measured with timestamps"
                                               : "not available (the queue has no timestamps)");
      // Pipeline statistics per pass. The order of the counters is the order of the bits,
      // not the order of this list: vertices, clipped primitives and fragments.
      if (vulkan_device_->properties().pipelineStatisticsQuery && read_queries_) {
        VkQueryPoolCreateInfo info_stats{};
        info_stats.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info_stats.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        info_stats.queryCount = uint32_t(slots_.size() * kStatsPerSlot);
        info_stats.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        if (dfn_.vkCreateQueryPool(device_, &info_stats, nullptr, &stats_) != VK_SUCCESS) {
          stats_ = VK_NULL_HANDLE;
        }
      }
      if (stats_ != VK_NULL_HANDLE) {
        VkQueryPoolCreateInfo info_draw{};
        info_draw.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        info_draw.queryType = VK_QUERY_TYPE_PIPELINE_STATISTICS;
        info_draw.queryCount = uint32_t(slots_.size() * kStatsDrawPerSlot);
        info_draw.pipelineStatistics =
            VK_QUERY_PIPELINE_STATISTIC_VERTEX_SHADER_INVOCATIONS_BIT |
            VK_QUERY_PIPELINE_STATISTIC_CLIPPING_PRIMITIVES_BIT |
            VK_QUERY_PIPELINE_STATISTIC_FRAGMENT_SHADER_INVOCATIONS_BIT;
        if (dfn_.vkCreateQueryPool(device_, &info_draw, nullptr, &stats_draw_) != VK_SUCCESS) {
          stats_draw_ = VK_NULL_HANDLE;
        }
        if (stats_draw_ != VK_NULL_HANDLE && queries_ != VK_NULL_HANDLE) {
          VkQueryPoolCreateInfo info_marks_draw{};
          info_marks_draw.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
          info_marks_draw.queryType = VK_QUERY_TYPE_TIMESTAMP;
          info_marks_draw.queryCount = uint32_t(slots_.size() * kStatsDrawPerSlot * 2);
          if (dfn_.vkCreateQueryPool(device_, &info_marks_draw, nullptr, &marks_draw_) != VK_SUCCESS) {
            marks_draw_ = VK_NULL_HANDLE;
          }
        }
      }
      REXLOG_INFO("[native] targets: pipeline statistics per pass {}",
                  stats_ != VK_NULL_HANDLE ? "available (sa2_native_stats_pipeline)"
                                                  : "not available");
    }

    VkSamplerCreateInfo info_sampler{};
    info_sampler.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    info_sampler.magFilter = VK_FILTER_LINEAR;
    info_sampler.minFilter = VK_FILTER_LINEAR;
    info_sampler.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info_sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info_sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info_sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    if (dfn_.vkCreateSampler(device_, &info_sampler, nullptr, &sampler_) != VK_SUCCESS) {
      return false;
    }

    // Same layout as the presenter: image at 0 and sampler at 1, and the gamma ramp at 2 (the pipeline without
    // the ramp does not use it).
    VkDescriptorSetLayoutBinding bindings[3]{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    bindings[1].pImmutableSamplers = &sampler_;
    bindings[2].binding = 2;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    VkDescriptorSetLayoutCreateInfo info_layout{};
    info_layout.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    info_layout.bindingCount = 3;
    info_layout.pBindings = bindings;
    if (dfn_.vkCreateDescriptorSetLayout(device_, &info_layout, nullptr, &layout_descriptors_) !=
        VK_SUCCESS) {
      return false;
    }
    VkPushConstantRange ranges[2]{};
    ranges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ranges[0].offset = 0;
    ranges[0].size = 16;  // GuestOutputPaintRectangleConstants
    ranges[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    ranges[1].offset = 16;
    ranges[1].size = 16;  // Presenter::BilinearConstants
    VkPipelineLayoutCreateInfo info_pipeline_layout{};
    info_pipeline_layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    info_pipeline_layout.setLayoutCount = 1;
    info_pipeline_layout.pSetLayouts = &layout_descriptors_;
    info_pipeline_layout.pushConstantRangeCount = 2;
    info_pipeline_layout.pPushConstantRanges = ranges;
    if (dfn_.vkCreatePipelineLayout(device_, &info_pipeline_layout, nullptr, &layout_pipeline_) !=
        VK_SUCCESS) {
      return false;
    }
    VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, kSlotsOutput},
                                       {VK_DESCRIPTOR_TYPE_SAMPLER, kSlotsOutput},
                                       {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, kSlotsOutput}};
    VkDescriptorPoolCreateInfo info_pool_desc{};
    info_pool_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info_pool_desc.maxSets = kSlotsOutput;
    info_pool_desc.poolSizeCount = 3;
    info_pool_desc.pPoolSizes = sizes;
    if (dfn_.vkCreateDescriptorPool(device_, &info_pool_desc, nullptr, &pool_descriptors_) !=
        VK_SUCCESS) {
      return false;
    }
    VkDescriptorSetAllocateInfo reserve_desc{};
    reserve_desc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    reserve_desc.descriptorPool = pool_descriptors_;
    reserve_desc.descriptorSetCount = 1;
    reserve_desc.pSetLayouts = &layout_descriptors_;
    for (uint32_t i = 0; i < kSlotsOutput; ++i) {
      if (dfn_.vkAllocateDescriptorSets(device_, &reserve_desc, &descriptors_output_[i]) != VK_SUCCESS) {
        return false;
      }
    }
    // One ramp buffer per output slot, so it is only changed in a slot the GPU is no longer using. Without
    // them, the output has no ramp (as before).
    ramp_gamma_ = REXCVAR_GET(sa2_native_ramp_gamma);
    // MASSEFFECT loads a gamma ramp before its first visible frame, but Mass Effect may not. The table is
    // zero-initialized, so leaving it untouched makes the output shader turn every pixel black until the
    // guest eventually writes a ramp. Start from the identity ramp declared below.
    if (ramp_gamma_) RecomputeRamp();
    for (uint32_t i = 0; i < kSlotsOutput && ramp_gamma_; ++i) {
      RampOutput& ramp = ramps_output_[i];
      uint32_t type = 0;
      void* mapped = nullptr;
      if (!rex::ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device_, sizeof(ramp_values_), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
              rex::ui::vulkan::util::MemoryPurpose::kUpload, ramp.buffer, ramp.memory, &type) ||
          dfn_.vkMapMemory(device_, ramp.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        REXLOG_WARN("[native] targets: could not create the gamma ramp buffer: the output goes without it");
        ramp_gamma_ = false;
        break;
      }
      ramp.data = static_cast<uint8_t*>(mapped);
      ramp.type = type;
      VkDescriptorBufferInfo info_buffer{ramp.buffer, 0, sizeof(ramp_values_)};
      VkWriteDescriptorSet write{};
      write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      write.dstSet = descriptors_output_[i];
      write.dstBinding = 2;
      write.descriptorCount = 1;
      write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      write.pBufferInfo = &info_buffer;
      dfn_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    }
    if (!ramp_gamma_) {
      // Without a ramp, binding 2 stays unwritten: the pipeline without the ramp does not read it.
      for (RampOutput& ramp : ramps_output_) {
        if (ramp.buffer != VK_NULL_HANDLE) dfn_.vkDestroyBuffer(device_, ramp.buffer, nullptr);
        if (ramp.memory != VK_NULL_HANDLE) dfn_.vkFreeMemory(device_, ramp.memory, nullptr);
        ramp = RampOutput{};
      }
    }

    VkShaderModuleCreateInfo info_module{};
    info_module.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info_module.codeSize = sizeof(shaders::guest_output_triangle_strip_rect_vs);
    info_module.pCode = shaders::guest_output_triangle_strip_rect_vs;
    if (dfn_.vkCreateShaderModule(device_, &info_module, nullptr, &vs_) != VK_SUCCESS) {
      return false;
    }
    info_module.codeSize = sizeof(shaders::guest_output_bilinear_ps);
    info_module.pCode = shaders::guest_output_bilinear_ps;
    if (dfn_.vkCreateShaderModule(device_, &info_module, nullptr, &fs_) != VK_SUCCESS) {
      return false;
    }
    if (ramp_gamma_) {
      info_module.codeSize = sizeof(shaders::sa2_output_ramp_gamma_ps);
      info_module.pCode = shaders::sa2_output_ramp_gamma_ps;
      if (dfn_.vkCreateShaderModule(device_, &info_module, nullptr, &fs_ramp_) != VK_SUCCESS) {
        return false;
      }
    }

    VkAttachmentDescription attached{};
    attached.format = VulkanPresenter::kGuestOutputFormat;
    attached.samples = VK_SAMPLE_COUNT_1_BIT;
    attached.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attached.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attached.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attached.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attached.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attached.finalLayout = VulkanPresenter::kGuestOutputInternalLayout;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkRenderPassCreateInfo info_rp{};
    info_rp.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    info_rp.attachmentCount = 1;
    info_rp.pAttachments = &attached;
    info_rp.subpassCount = 1;
    info_rp.pSubpasses = &subpass;
    if (dfn_.vkCreateRenderPass(device_, &info_rp, nullptr, &render_pass_output_) != VK_SUCCESS) {
      return false;
    }

    pipeline_ = CreatePipelineOutput(fs_, nullptr);
    if (pipeline_ == VK_NULL_HANDLE) {
      return false;
    }
    // The ramp variants (bilinear and exact texel) are created now (PipelineRamp).
    if (ramp_gamma_ && (PipelineRamp(0) == VK_NULL_HANDLE || PipelineRamp(1) == VK_NULL_HANDLE)) {
      return false;
    }
    REXLOG_INFO("[native] targets: the game's gamma ramp on the output (sa2_native_ramp_gamma) = {}",
                ramp_gamma_ ? "YES" : "no");
    // Draw code: without the required capabilities only copies and presentation remain.
    draws_ = DrawsVulkan::Create(vulkan_device_, memory_, this);
    return true;
  }

  // Pipeline of the output pass with the fragment shader fs (and its specialization constants, if any).
  VkPipeline CreatePipelineOutput(VkShaderModule fs, const VkSpecializationInfo* special) {
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";
    stages[1].pSpecializationInfo = special;
    VkPipelineVertexInputStateCreateInfo entry{};
    entry.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    VkPipelineInputAssemblyStateCreateInfo assembly{};
    assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;
    VkPipelineViewportStateCreateInfo view{};
    view.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    view.viewportCount = 1;
    view.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterization.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo sampling{};
    sampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    sampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState attached_blend{};
    attached_blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                    VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1;
    blend.pAttachments = &attached_blend;
    const VkDynamicState dynamic_offsets[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = dynamic_offsets;
    VkGraphicsPipelineCreateInfo info_pipeline{};
    info_pipeline.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    info_pipeline.stageCount = 2;
    info_pipeline.pStages = stages;
    info_pipeline.pVertexInputState = &entry;
    info_pipeline.pInputAssemblyState = &assembly;
    info_pipeline.pViewportState = &view;
    info_pipeline.pRasterizationState = &rasterization;
    info_pipeline.pMultisampleState = &sampling;
    info_pipeline.pColorBlendState = &blend;
    info_pipeline.pDynamicState = &dynamic;
    info_pipeline.layout = layout_pipeline_;
    info_pipeline.renderPass = render_pass_output_;
    info_pipeline.basePipelineIndex = -1;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (dfn_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &info_pipeline, nullptr, &pipeline) !=
        VK_SUCCESS) {
      return VK_NULL_HANDLE;
    }
    return pipeline;
  }

  // Variant of the output pass with the ramp, by index: 0 bilinear, 1 exact texel (specialization constant 0
  // of the shader; the grading and FXAA constants 1 and 2 stay off). VK_NULL_HANDLE if it fails (and it is
  // logged).
  VkPipeline PipelineRamp(uint32_t index) {
    VkPipeline& pipeline = pipelines_ramp_[index];
    if (pipeline == VK_NULL_HANDLE && !pipelines_failed_ramp_[index]) {
      const VkSpecializationMapEntry inputs[3] = {{0, 0, sizeof(VkBool32)},
                                                    {1, sizeof(VkBool32), sizeof(VkBool32)},
                                                    {2, 2 * sizeof(VkBool32), sizeof(VkBool32)}};
      const VkBool32 values[3] = {(index & 1) ? VK_TRUE : VK_FALSE, VK_FALSE, VK_FALSE};
      const VkSpecializationInfo special{3, inputs, sizeof(values), values};
      const auto before = std::chrono::steady_clock::now();
      pipeline = CreatePipelineOutput(fs_ramp_, &special);
      if (pipeline == VK_NULL_HANDLE) {
        pipelines_failed_ramp_[index] = true;
        REXLOG_ERROR("[native] targets: could not create variant {} of the output pass", index);
      } else {
        REXLOG_INFO("[native] targets: variant {} of the output pass (exact {}) created in {:.1f} ms",
                    index, (index & 1) ? "yes" : "no",
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before).count());
      }
    }
    return pipeline;
  }

  bool Copy(const RegistersCopy& reg) override { return CopyInternal(reg); }

  bool CopyInternal(const RegistersCopy& reg) {
    edram4_operation_failed_ = false;
    // sa2_gpu_resolve_alias: the resolve right before this one is the candidate source of an alias. Any other
    // resolve, clear or failure in between ends it (only a color resolve recorded for real sets it again).
    const LastResolve previous_resolve = last_resolve_;
    last_resolve_.valid = false;
    if (DiagAlias()) {
      int32_t rx0 = 0, ry0 = 0, rx1 = 0, ry1 = 0;
      Rectangle(reg, reg.rb_surface_info & 0x3FFF, rx0, ry0, rx1, ry1);
      REXLOG_INFO("[diag alias] frame {} resolve: drawn {} control {:08X} color_info {:08X} depth_info {:08X} surface "
                  "{:08X} dest {:08X} pitch {:08X} info {:08X} rect {},{}-{},{} | previous valid {} {:08X}",
                  presented_, draws_ ? draws_->Drawn() : 0, reg.rb_copy_control,
                  (reg.rb_copy_control & 7) < 4 ? reg.rb_color_info[reg.rb_copy_control & 7] : 0, reg.rb_depth_info,
                  reg.rb_surface_info, reg.rb_copy_dest_base, reg.rb_copy_dest_pitch, reg.rb_copy_dest_info, rx0, ry0,
                  rx1, ry1, previous_resolve.valid, previous_resolve.address);
    }
    // Preflight SDK clear ranges and backing padding before resolve/clear mutations.
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4 &&
        !ValidateSimultaneousClearEDRAM4(reg)) return false;
    // Draws since the previous copy: those of the render target being resolved now (copy inventory).
    const uint64_t drawn = draws_ ? draws_->Drawn() : 0;
    const uint64_t draws_before = drawn - drawn_last_copy_;
    drawn_last_copy_ = drawn;
    if (draws_) {
      draws_->FinishPass();  // copying and clearing are not allowed inside a pass
      // No longer on every copy; see GetResolved and Prepare.
      if (invalidate_each_copy_) {
        draws_->InvalidateTextures();
      }
    }
    if (Record()) {
      MarkGpu(kGpuCopies);  // GPU time of copies and clears (report)
    }
    const uint32_t control = reg.rb_copy_control;
    const uint32_t source = control & 0x7;
    const bool clear_color = (control >> 8) & 0x1;
    const uint32_t command = (control >> 20) & 0x3;
    // sa2_diag_resolves_from (diagnostic): the raw registers of a window of resolves (copy number N .. N+79).
    {
      static const int32_t diag_from = REXCVAR_GET(sa2_diag_resolves_from);
      static uint64_t diag_n = 0;
      const uint64_t n = diag_n++;
      if (diag_from > 0 && n >= uint64_t(diag_from) && n < uint64_t(diag_from) + 80) {
        int32_t rx0 = 0, ry0 = 0, rx1 = 0, ry1 = 0;
        Rectangle(reg, reg.rb_surface_info & 0x3FFF, rx0, ry0, rx1, ry1);
        REXLOG_INFO("[native] resolve #{} rect {},{}-{},{}", n, rx0, ry0, rx1, ry1);
        REXLOG_INFO("[native] resolve #{}: draws before {} | control {:08X} (source {}, clear color {}, clear depth {}, "
                    "command {}) surface {:08X} color_info {:08X} depth_info {:08X} | dest base {:08X} pitch {:08X} "
                    "info {:08X} | window offset {:08X} scissor {:08X}-{:08X} | rect vertices fetch {:08X} {:08X}",
                    n, draws_before, control, source, clear_color ? 1 : 0, (control >> 9) & 1, command,
                    reg.rb_surface_info, source < 4 ? reg.rb_color_info[source] : 0, reg.rb_depth_info,
                    reg.rb_copy_dest_base, reg.rb_copy_dest_pitch, reg.rb_copy_dest_info, reg.pa_sc_window_offset,
                    reg.pa_sc_window_scissor_tl, reg.pa_sc_window_scissor_br, reg.fetch_vertices[0],
                    reg.fetch_vertices[1]);
      }
    }
    const bool do_copy = command == uint32_t(xenos::CopyCommand::kRaw) ||
                        command == uint32_t(xenos::CopyCommand::kConvert);
    const uint32_t pitch = reg.rb_surface_info & 0x3FFF;
    const uint32_t msaa = (reg.rb_surface_info >> 16) & 0x3;
    msaa_edram_actual_ = msaa;
    // Do not infer draw eligibility here. ClearDepth explicitly selects
    // an expanded clear backing so untouched partial-tile borders retain X.
    depth_raster_grid_eligible_ = false;
    if (msaa != uint32_t(xenos::MsaaSamples::k1X) && warned_.insert(1).second) {
      REXLOG_INFO("[native] targets: MSAA target: used with 1 sample");
    }
    if (source >= xenos::kMaxColorRenderTargets) {
      // From depth: the copy goes to a resolved texture with the host depth
      // format, which the draws sample as k_24_8. With a depth source no color
      // is cleared (the SDK's IsClearingColor).
      const bool will_clear = ((control >> 9) & 0x1) && clear_depth_;
      if (do_copy) {
        CopyDepth(reg, pitch, will_clear);
      }
      if (will_clear) {
        ClearDepth(reg, pitch);
      }
      return !edram4_operation_failed_;
    }
    const uint32_t info_color = reg.rb_color_info[source];
    const uint32_t color_format = (info_color >> 16) & 0xF;
    if (HostFormatTargetColor(color_format) == VK_FORMAT_UNDEFINED) {
      return Reject(100 + color_format, "render target format not supported yet");
    }
    int32_t x0, y0, x1, y1;
    if (!Rectangle(reg, pitch, x0, y0, x1, y1)) {
      return false;
    }

    // First the objects (creating or recreating can submit work), then recording.
    Image* target_render = GetTarget(info_color & 0xFFF, color_format, pitch);
    if (!target_render) {
      return false;
    }
    // The game has just said which area of this render target matters to it. It is the data RestoreContent
    // uses to stop copying the bottom rows that nobody draws or reads (the scene render target is created
    // 1280x1280 to draw 1280x720).
    NoteAreaUtil(*target_render, y1);
    uint32_t base_resolved = 0;
    Resolved* resolved = nullptr;
    uint32_t format_resolved_target = 0;
    uint32_t dx = 0, dy = 0;
    if (do_copy) {
      const uint32_t info_target = reg.rb_copy_dest_info;
      const uint32_t target_format = (info_target >> 7) & 0x3F;
      format_resolved_target = target_format;
      if ((info_target >> 3) & 0x1) {
        Reject(3, "copy to a 3D texture or array: not supported yet");
      } else if (HostFormatResolved(target_format) == VK_FORMAT_UNDEFINED) {
        // 300 + format, apart from the draws' 200 + render target format.
        Reject(300 + target_format, "copy format not supported yet");
      } else {
        const VkFormat host_format_target = HostFormatResolved(target_format);
        const uint32_t log2_texel = host_format_target == VK_FORMAT_R16G16B16A16_SFLOAT ||
                                            host_format_target == VK_FORMAT_R32G32_SFLOAT
                                        ? 3
                                        : 2;
        const uint32_t pitch_target = reg.rb_copy_dest_pitch & 0x3FFF;
        const uint32_t target_height = (reg.rb_copy_dest_pitch >> 16) & 0x3FFF;
        // 4 bytes per texel: base in multiples of 32 texels (GetResolveInfo).
        const uint32_t base_x = uint32_t(x0) & ~uint32_t(31);
        const uint32_t base_y = uint32_t(y0) & ~uint32_t(31);
        const uint32_t base =
            reg.rb_copy_dest_base +
            uint32_t(TileDisplacement2D(int32_t(base_x), int32_t(base_y), pitch_target, log2_texel));
        base_resolved = base & 0x1FFFFFFF;
        dx = uint32_t(x0) - base_x;
        dy = uint32_t(y0) - base_y;
        // banjo-nx: the rows really copied, not the destination height: Nuts & Bolts resolves its second predicated
        // tile (rows 384..719) with the base advanced by 384 rows and the full 1280x720 destination size, which
        // overran the 1280x720 texture of the first tile and became a texture of its own (bottom half missing).
        resolved = SearchResolvedPartial(base_resolved, pitch_target,
                                         std::min(target_height, uint32_t(std::max(y1 - y0, 0))),
                                         target_format, (info_target >> 24) & 0x1,
                                         host_format_target, log2_texel, base_resolved, dx, dy);
        if (!resolved) {
          resolved = GetResolved(base_resolved, pitch_target, target_height, target_format,
                                     (info_target >> 24) & 0x1, host_format_target);
        }
        if (resolved) {
          NoteCopy(base_resolved, resolved->image.width, resolved->image.height, draws_before);
        }
      }
    }

    if (!Record()) {
      return false;
    }
    Prepare(*target_render);
    area_edram_ = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4 && do_copy &&
        !(edram4_reason_ = 2, SynchronizeEDRAM4(*target_render, area_edram_))) return false;
    ActivateTargetColor(info_color & 0xFFF, pitch, *target_render);
    if (resolved) {
      Prepare(resolved->image);
      // What the game asks for and what fits in the render target.
      const uint32_t request_width =
          std::min(uint32_t(x1 - x0), target_render->width - uint32_t(x0));
      const uint32_t request_height = std::min(uint32_t(y1 - y0), target_render->height - uint32_t(y0));
      const uint32_t fits_width = resolved->image.width > dx ? resolved->image.width - dx : 0;
      const uint32_t fits_height = resolved->image.height > dy ? resolved->image.height - dy : 0;
      const uint32_t width = std::min(request_width, fits_width);
      const uint32_t height = std::min(request_height, fits_height);
      // With the scene at a higher resolution than the render target (sa2_internal_resolution = 1920x1080
      // and a 1280x720 front buffer), copying 1 to 1 takes only a piece: the image comes out cropped, with part
      // of the scene and the HUD out of frame. When it does not fit, it is shrunk with a linear blit, which is exactly
      // the scaling wanted. If it fits, it is copied as always, bit for bit.
      // It only shrinks when the scene really has to be reduced (by a factor of 1.25 or more) and both images
      // are color: vkCmdBlitImage with a linear filter on a depth target is not valid and brings the process
      // down. Mismatches of a few pixels (320x184 -> 320x180) are still cropped as always.
      const bool multisampled = !me::native::IsSingleSample(uint32_t(target_render->sample_count));  // sa2_msaa
      const bool shrink = blit_ != nullptr && fits_width && fits_height && !multisampled &&
                           target_render->format == kColorFormat &&
                           resolved->image.format == kColorFormat &&
                           (request_width * 4 >= fits_width * 5 || request_height * 4 >= fits_height * 5);
      if (shrink && uint32_t(x0) < target_render->width && uint32_t(y0) < target_render->height) {
        BeforeResolvedWrite(base_resolved);  // sa2_gpu_resolve_alias
        // sa2_native_lazy_front. If this texture had a deferred copy, it is dropped if this resolve
        // covers it entirely, and recorded first otherwise.
        ResolverPreviousFront(base_resolved, dx == 0 && dy == 0 && fits_width == resolved->image.width &&
                                                   fits_height == resolved->image.height);
        VkImageBlit reduction{};
        reduction.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reduction.srcOffsets[0] = {HostEdge(*target_render, x0), HostEdge(*target_render, y0), 0};
        reduction.srcOffsets[1] = {HostEdge(*target_render, x0 + int32_t(request_width)),
                                   HostEdge(*target_render, y0 + int32_t(request_height)), 1};
        reduction.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        reduction.dstOffsets[0] = {HostEdge(resolved->image, int32_t(dx)), HostEdge(resolved->image, int32_t(dy)), 0};
        reduction.dstOffsets[1] = {HostEdge(resolved->image, int32_t(dx + fits_width)),
                                   HostEdge(resolved->image, int32_t(dy + fits_height)), 1};
        BarrierBeforeCopyResolve();
        blit_(commands_work_, target_render->image, VK_IMAGE_LAYOUT_GENERAL,
              resolved->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &reduction, VK_FILTER_LINEAR);
        BarrierAfterCopyResolve();
        ++copies_;
        ++reductions_;
        if (reductions_ <= 4) {
          REXLOG_INFO("[resolution] the {}x{} scene is downscaled to {}x{} when resolved (supersampling)",
                      request_width, request_height, fits_width, fits_height);
        }
        // A linear-filter blit reads the large rectangle and writes the small one, so it costs more than a copy
        // of the output size. The pixels read are recorded, since they are what dominates.
        ResolvedWritten(base_resolved, uint64_t(request_width) * request_height);
        ResolvedRead(reg, *resolved, x0, y0, dx, dy, fits_width, fits_height);
      } else if (width && height && uint32_t(x0) < target_render->width &&
                 uint32_t(y0) < target_render->height && dx < resolved->image.width &&
                 dy < resolved->image.height) {
        const int32_t exp_bias =
            me::native::ColorResolveExponentBias(command, reg.rb_copy_dest_info);
        if (exp_bias && multisampled) {
          return Reject(14, "resolve exponent bias from a multisampled target (sa2_msaa) not supported");
        } else if (exp_bias) {
          // The HDR loop requests -3 (1/8) on each convert resolve. A numeric
          // vkCmdCopyImage / blit ignores this and amplifies repeated feedback.
          BeforeResolvedWrite(base_resolved);  // sa2_gpu_resolve_alias
          ResolverPreviousFront(base_resolved, false);
          if (!ResolverWithBias(*target_render, resolved->image, x0, y0, dx, dy,
                                width, height, exp_bias)) {
            return Reject(14, "resolve exponent bias / format not supported");
          }
          ++copies_;
          ++converted_copies_;
          ResolvedWritten(base_resolved, uint64_t(width) * height);
        } else if (target_render->format != resolved->image.format) {
          // resolve with a format conversion (HDR render target to another texture format),
          // a 1 to 1 blit.
          const uint64_t conversion_key =
              uint64_t(color_format) | (uint64_t(format_resolved_target) << 8) |
              (uint64_t(target_render->format) << 16) |
              (uint64_t(resolved->image.format) << 48);
          if (recorded_conversions_.insert(conversion_key).second) {
            REXLOG_INFO(
                "[native] targets: resolve conversion: guest RT {} host {} base {:03X}/{} -> guest texture {} host {} "
                "{:08X} {}x{} (rect {}x{} at {},{} -> {},{})",
                color_format, uint32_t(target_render->format), info_color & 0xFFF, pitch,
                format_resolved_target, uint32_t(resolved->image.format), base_resolved,
                resolved->image.width, resolved->image.height, width, height, x0, y0, dx, dy);
          }
          if (!blit_ || multisampled) {
            Reject(4, multisampled ? "copy with format conversion from a multisampled target (sa2_msaa)"
                                   : "copy with format conversion without vkCmdBlitImage");
          } else {
            VkImageBlit conversion{};
            conversion.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            conversion.srcOffsets[0] = {HostEdge(*target_render, x0), HostEdge(*target_render, y0), 0};
            conversion.srcOffsets[1] = {HostEdge(*target_render, x0 + int32_t(width)),
                                        HostEdge(*target_render, y0 + int32_t(height)), 1};
            conversion.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            conversion.dstOffsets[0] = {HostEdge(resolved->image, int32_t(dx)), HostEdge(resolved->image, int32_t(dy)),
                                        0};
            conversion.dstOffsets[1] = {HostEdge(resolved->image, int32_t(dx + width)),
                                        HostEdge(resolved->image, int32_t(dy + height)), 1};
            BeforeResolvedWrite(base_resolved);  // sa2_gpu_resolve_alias
            ResolverPreviousFront(base_resolved, dx == 0 && dy == 0 && width == resolved->image.width &&
                                                       height == resolved->image.height);
            BarrierBeforeCopyResolve();
            blit_(commands_work_, target_render->image, VK_IMAGE_LAYOUT_GENERAL, resolved->image.image,
                  VK_IMAGE_LAYOUT_GENERAL, 1, &conversion, VK_FILTER_NEAREST);
            BarrierAfterCopyResolve();
            ++copies_;
            ++converted_copies_;
            ResolvedWritten(base_resolved, uint64_t(width) * height);
          }
        } else {
        VkImageCopy copy{};
        copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.srcOffset = {x0, y0, 0};
        copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        copy.dstOffset = {int32_t(dx), int32_t(dy), 0};
        copy.extent = {width, height, 1};
        // sa2_gpu_resolve_alias: the same bytes as the previous resolve's texture, which still holds them.
        if (MakeAlias(previous_resolve, *target_render, *resolved, base_resolved, x0, y0, dx, dy, width, height)) {
          ++copies_;  // a resolve like any other for the reports (the alias report counts how many were not copied)
        } else {
        BeforeResolvedWrite(base_resolved, dx, dy, width, height);  // sa2_gpu_resolve_alias (covers the rectangle)
        // sa2_native_lazy_front. The previous copy to this texture, if still deferred, is dropped (this
        // one covers it entirely) or recorded first; and if this texture is a front buffer that only the Swap
        // reads, it is deferred.
        ResolverPreviousFront(base_resolved,
                                dx == 0 && dy == 0 && width == resolved->image.width && height == resolved->image.height);
        if (multisampled || !PostponeCopyFront(base_resolved, *target_render, *resolved, copy)) {
          // sa2_gpu_resolve_draw: on the 3D engine when it applies.
          if (multisampled) {
            // sa2_msaa: the samples averaged into the single-sample texture (the anti-aliasing).
            const VkImageCopy host_copy = HostCopy(*target_render, resolved->image, copy);
            VkImageResolve resolve{};
            resolve.srcSubresource = host_copy.srcSubresource;
            resolve.srcOffset = host_copy.srcOffset;
            resolve.dstSubresource = host_copy.dstSubresource;
            resolve.dstOffset = host_copy.dstOffset;
            resolve.extent = host_copy.extent;
            BarrierBeforeCopyResolve();
            resolve_image_(commands_work_, target_render->image, VK_IMAGE_LAYOUT_GENERAL, resolved->image.image,
                           VK_IMAGE_LAYOUT_GENERAL, 1, &resolve);
            BarrierAfterCopyResolve();
            ++msaa_resolves_;
            sa2::render::NoteMsaaUse();
          } else if (!ResolveColorFrag(*target_render, resolved->image, x0, y0, dx, dy, width, height)) {
            const VkImageCopy host_copy = HostCopy(*target_render, resolved->image, copy);
            BarrierBeforeCopyResolve();
            copy_image_(commands_work_, target_render->image, VK_IMAGE_LAYOUT_GENERAL,
                           resolved->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &host_copy);
            BarrierAfterCopyResolve();
          }
          ++copies_;
          ResolvedWritten(base_resolved, uint64_t(width) * height);
          ResolvedRead(reg, *resolved, x0, y0, dx, dy, width, height);
          // sa2_gpu_resolve_alias: the candidate source of the next resolve's alias (not when this command also
          // clears: the render target changes right after the copy).
          if (g_resolve_alias.active && !clear_color && !((control >> 9) & 0x1)) {
            last_resolve_.valid = true;
            last_resolve_.target = target_render->image;
            last_resolve_.x0 = x0;
            last_resolve_.y0 = y0;
            last_resolve_.width = width;
            last_resolve_.height = height;
            last_resolve_.address = base_resolved;
            last_resolve_.image = resolved->image.image;
            last_resolve_.revision = resolved->revision;
            last_resolve_.dx = dx;
            last_resolve_.dy = dy;
            last_resolve_.drawn = draws_ ? draws_->Drawn() : 0;
            last_resolve_.transfers = edram4_transfers_all_;
            last_resolve_.restored = restored_pixels_;
            last_resolve_.edram_version = edram4_version_;
          }
        }
        }
        }
      }
    }
    if (clear_color) {
      const auto clear_rect = me::native::ClipClearRectangle(
          x0, y0, x1, y1, target_render->width, target_render->height);
      const VkRect2D clear_area{{clear_rect.x, clear_rect.y},
                                {clear_rect.width, clear_rect.height}};
      const bool clear_whole = clear_rect.whole(target_render->width, target_render->height);
      if (!clear_rect.width || !clear_rect.height) return Reject(15, "empty color clear rectangle");
      const uint64_t raw_value =
          uint64_t(reg.rb_color_clear) | (uint64_t(reg.rb_color_clear_lo) << 32);
      if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4) {
        me::native::EdramClearPlan plan;
        VkRect2D sdk_area{};
        me::native::EdramClearRectangles rectangles;
        if (!GetPlanClearedEDRAM4(reg, plan, sdk_area) ||
            !PrepareRangeClearEDRAM4(*target_render, plan.color, sdk_area, rectangles)) return false;
        if (rectangles.count && !draws_)
          return FailureEDRAM4("SDK color clear needs rectangle render passes", *target_render);
        if (plan.color.length) {
          ForgetClear(*target_render);
          const VkClearColorValue color = ClearValueColor(color_format, raw_value);
          for (uint32_t i = 0; i < rectangles.count; ++i) {
            const auto& rectangle = rectangles.rectangles[i];
            const VkRect2D region{{int32_t(rectangle.x), int32_t(rectangle.y)},
                                  {rectangle.width, rectangle.height}};
            BarrierClearEDRAM4();
            if (!draws_->ClearColorInPass(commands_work_, *target_render, color, region))
              return FailureEDRAM4("SDK color clear render pass failed", *target_render);
            BarrierClearEDRAM4();
          }
          if (rectangles.count) {
            MarkGpu(kGpuCleared);
            ++cleared_;
          }
          PublishRangeClearEDRAM4(*target_render, plan.color);
        }
      } else {
      /*
       * Two things that were missing here.
       *  1. A WHOLE-image clear leaves the target with a known color, so discarded contents need not
       *     be restored. A partial clear must preserve every pixel outside the requested rectangle
       *     (color did not).
       *  2. If the render target is already cleared to that same color and nothing has been drawn since
       *     then, the vkCmdClearColorImage does not change a single bit and is skipped.
       */
      // A partial clear does not restore discarded pixels outside its rectangle.
      if (clear_whole) target_render->invalid_content = false;
      else if (!RestoreContent(*target_render) &&
               REXCVAR_GET(sa2_native_edram_alias_mode) == 4) return false;
      if (clear_whole && REXCVAR_GET(sa2_native_edram_alias_mode) != 4 &&
          RedundantClear(*target_render, raw_value)) {
        ++cleared_skipped_;
      } else {
        // sa2_native_lazy_front. If this render target is the source of a deferred front buffer, the
        // clear goes to a spare image and the one with the content is kept for the front buffer (no copy).
        if (clear_whole) RotateFrontBeforeClear(*target_render);
        else ForgetClear(*target_render);
        MarkGpu(kGpuCleared);  // clears, separate from copies (report)
        const VkClearColorValue color = ClearValueColor(color_format, raw_value);  // per format
        if (!clear_whole) {
          BarrierBeforeCopyResolve();
          if (!draws_ || !draws_->ClearColorInPass(commands_work_, *target_render,
                                                       color, clear_area)) {
            return Reject(16, "partial color clear render pass failed");
          }
          BarrierAfterCopyResolve();
        } else {
          BarrierBeforeCopyResolve();
          dfn_.vkCmdClearColorImage(commands_work_, target_render->image,
                                    VK_IMAGE_LAYOUT_GENERAL, &color, 1, &kRangeColor);
          BarrierAfterCopyResolve();
        }
        ++cleared_;
      }
      }
    }
    if (((control >> 9) & 0x1) && clear_depth_) {
      ClearDepth(reg, pitch);
    }
    return !edram4_operation_failed_;
  }

  void ClearDepth(const RegistersCopy& reg, uint32_t pitch) {
    const uint32_t info = reg.rb_depth_info;
    // A partial clear preserves the rest of each published physical tile. It
    // must not import retained samples into a collapsed image before clearing.
    const bool previous_eligibility = depth_raster_grid_eligible_;
    depth_raster_grid_eligible_ = msaa_edram_actual_ == 2;
    Image* depth = GetDepth(info & 0xFFF, (info >> 16) & 0x1, pitch);
    depth_raster_grid_eligible_ = previous_eligibility;
    if (!depth || !Record()) {
      return;
    }
    Prepare(*depth);
    int32_t x0, y0, x1, y1;
    if (!Rectangle(reg, pitch, x0, y0, x1, y1)) return;
    x0 *= 1u << depth->raster_grid_x;
    x1 *= 1u << depth->raster_grid_x;
    if (depth->guest_width && depth->guest_height) {
      x0 = int32_t(int64_t(x0) * depth->width / depth->guest_width);
      y0 = int32_t(int64_t(y0) * depth->height / depth->guest_height);
      x1 = int32_t((int64_t(x1) * depth->width + depth->guest_width - 1) /
                    depth->guest_width);
      y1 = int32_t((int64_t(y1) * depth->height + depth->guest_height - 1) /
                    depth->guest_height);
    }
    const auto clear_rect = me::native::ClipClearRectangle(
        x0, y0, x1, y1, depth->width, depth->height);
    if (!clear_rect.width || !clear_rect.height) return;
    const VkRect2D clear_area{{clear_rect.x, clear_rect.y},
                              {clear_rect.width, clear_rect.height}};
    const bool clear_whole = clear_rect.whole(depth->width, depth->height);
    const bool mode4 = REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    if (mode4) {
      me::native::EdramClearPlan plan;
      VkRect2D sdk_area{};
      me::native::EdramClearRectangles rectangles;
      if (!GetPlanClearedEDRAM4(reg, plan, sdk_area) ||
          !PrepareRangeClearEDRAM4(*depth, plan.depth, sdk_area, rectangles)) return;
      if (rectangles.count && !draws_) {
        FailureEDRAM4("SDK depth clear needs rectangle render passes", *depth);
        return;
      }
      if (!plan.depth.length) return;
      ForgetClear(*depth);
      const VkClearDepthStencilValue value{me::native::NativeDepthClearValue(info, reg.rb_depth_clear, depth->depth_float24_half),
                                           reg.rb_depth_clear & 255u};
      for (uint32_t i = 0; i < rectangles.count; ++i) {
        const auto& rectangle = rectangles.rectangles[i];
        const VkRect2D region{{int32_t(rectangle.x), int32_t(rectangle.y)},
                              {rectangle.width, rectangle.height}};
        BarrierClearEDRAM4();
        depth->edram4_bits_stencil |= uint8_t(value.stencil);
        if (!draws_->ClearDepthInPass(commands_work_, *depth,
                                                value.depth, value.stencil, &region)) {
          FailureEDRAM4("SDK depth clear render pass failed", *depth);
          return;
        }
        BarrierClearEDRAM4();
      }
      if (rectangles.count) {
        MarkGpu(kGpuCleared);
      }
      PublishRangeClearEDRAM4(*depth, plan.depth);
      return;
    }
    if (!clear_whole && !draws_) {
      Reject(17, "partial depth clear requires a render pass");
      return;
    }
    if (clear_whole) depth->invalid_content = false;
    else {
      if (!RestoreContent(*depth) && mode4) return;
      ForgetClear(*depth);
    }
    const VkClearDepthStencilValue val{me::native::NativeDepthClearValue(info, reg.rb_depth_clear, depth->depth_float24_half),
                                         reg.rb_depth_clear & 0xFF};
    // If it is already cleared with this same value and nobody has drawn anything since then, the clear does
    // not change a single bit. Mind the path below: the per-pass clear also resets the ZCULL plane, so that
    // one is never skipped (only the vkCmdClearDepthStencilImage one).
    if (!mode4 && clear_whole && depth->accepts_target_of_copy &&
        RedundantClear(*depth, uint64_t(reg.rb_depth_clear))) {
      ++cleared_skipped_depth_;
      return;
    }
    MarkGpu(kGpuCleared);  // clears, separate from copies (report)
    // ZCULL: see Prepare. Without TRANSFER_DST the clear has to be done by opening a pass.
    if ((!clear_whole || !depth->accepts_target_of_copy) && draws_) {
      // This bool cannot be dropped. If the clear pass cannot be opened, the depth keeps the previous frame's
      // content and the ZCULL hi-Z is not reset either (only a loadOp = CLEAR resets it), so the whole frame
      // culls against stale data. Without TRANSFER_DST there is no alternative path, so at least it is
      // counted and reported.
      depth->edram4_bits_stencil |= uint8_t(val.stencil);
      const bool cleared = draws_->ClearDepthInPass(commands_work_, *depth, val.depth,
                                                              val.stencil, clear_whole ? nullptr : &clear_area);
      if (!cleared) {
        if (mode4) FailureEDRAM4("depth clear render pass failed", *depth);
        if (++cleared_in_failed_pass_ <= 8) {
          REXLOG_WARN("[native] targets: could NOT clear the {}x{} depth by opening a pass; it keeps "
                      "its previous content (failure {})",
                      depth->width, depth->height, cleared_in_failed_pass_);
        }
      }
      if (mode4 && cleared) PublishEDRAM4(*depth, clear_area);
      return;
    }
    depth->edram4_bits_stencil |= uint8_t(val.stencil);
    BarrierBeforeCopyResolve();
    clear_depth_(commands_work_, depth->image, VK_IMAGE_LAYOUT_GENERAL, &val, 1,
                        &kRangeDepth);
    BarrierAfterCopyResolve();
    if (mode4) PublishEDRAM4(*depth, clear_area);
  }

  // Depth copy: the rectangle of the depth render target to a resolved texture
  // of the same host format, at the base GetResolveInfo computes
  // (4 bytes per texel, like k_24_8).
  // will_clear: this same game command clears the render target right after resolving it. Only then is it worth
  // swapping the images: if the game kept drawing on top, the content would have to be brought back and the
  // copy would be paid anyway (measured: one restore per frame on the shadow map).
  void CopyDepth(const RegistersCopy& reg, uint32_t pitch, bool will_clear) {
    const bool mode4 = REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    const auto failure = [&](const char* reason, const Image* owner = nullptr) {
      if (!mode4) return;
      Image diagnostic;
      diagnostic.edram_depth = true;
      diagnostic.edram_base = uint16_t(reg.rb_depth_info & 0xFFFu);
      diagnostic.edram_format = uint8_t((reg.rb_depth_info >> 16) & 1u);
      FailureEDRAM4(reason, owner ? *owner : diagnostic);
    };
    const uint32_t info_target = reg.rb_copy_dest_info;
    if ((info_target >> 3) & 0x1) {
      Reject(3, "copy to a 3D texture or array: not supported yet");
      failure("depth resolve array/3D destination unsupported");
      return;
    }
    if (!copy_image_ || !pitch) {
      failure("depth resolve missing transfer function or pitch");
      return;
    }
    int32_t x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    const bool vertex_contract =
        (reg.fetch_vertices[0] & 3u) == uint32_t(xenos::FetchConstantType::kVertex) &&
        ((reg.fetch_vertices[1] >> 2) & 0xFFFFFFu) == 6u;
    if (!Rectangle(reg, pitch, x0, y0, x1, y1)) {
      if (!vertex_contract) failure("depth resolve unsupported rectangle vertex contract");
      return;  // An empty scissor/rectangle is a successful no-op.
    }
    if (x0 < 0 || y0 < 0 || x1 < x0 || y1 < y0) {
      failure("depth resolve invalid signed rectangle");
      return;
    }
    const uint32_t info = reg.rb_depth_info;
    if (mode4 && REXCVAR_GET(sa2_native_depth_samples_x) && msaa_edram_actual_ == 2) {
      // A direct resolve would collapse the retained samples again, or select
      // the collapsed allocation and hide that loss. No such shortcut in this
      // bounded experiment: sample-select/average resolve needs its own path.
      failure("4x depth resolve unsupported with horizontal sample preservation");
      return;
    }
    Image* depth = GetDepth(info & 0xFFF, (info >> 16) & 0x1, pitch);
    if (!depth) {
      failure("depth resolve source allocation/format failure");
      return;
    }
    if (mode4) {
      if (!Record()) {
        failure("depth resolve cannot begin recording", depth);
        return;
      }
      edram4_reason_ = 3;
      if (!SynchronizeEDRAM4(*depth,
          VkRect2D{{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}})) {
        failure("depth resolve source synchronization failed", depth);
        return;
      }
    }
    const uint32_t pitch_target = reg.rb_copy_dest_pitch & 0x3FFF;
    const uint32_t target_height = (reg.rb_copy_dest_pitch >> 16) & 0x3FFF;
    if (mode4 && (!pitch_target || !target_height)) {
      failure("depth resolve zero destination dimensions", depth);
      return;
    }
    const uint32_t base_x = uint32_t(x0) & ~uint32_t(31);
    const uint32_t base_y = uint32_t(y0) & ~uint32_t(31);
    const uint32_t base =
        reg.rb_copy_dest_base +
        uint32_t(TileDisplacement2D(int32_t(base_x), int32_t(base_y), pitch_target, 2));
    uint32_t dx = uint32_t(x0) - base_x;
    uint32_t dy = uint32_t(y0) - base_y;
    const uint32_t texture_format = ((info >> 16) & 0x1) ? 23 : 22;  // k_24_8_FLOAT : k_24_8
    const VkFormat resolved_host_format = depth->depth_float24_half
        ? VK_FORMAT_R32_SFLOAT : depth->format;
    if (depth->depth_float24_half) {
      VkFormatProperties properties{};
      vulkan_device_->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
          vulkan_device_->physical_device(), resolved_host_format, &properties);
      const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
          VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
          VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
      if ((properties.optimalTilingFeatures & needed) != needed) {
        FailureEDRAM4("guestspace depth resolve requires sampled/storage R32 format", *depth);
        return;
      }
    }
    if (depth->depth_float24_half && (depth->guest_width || depth->guest_height)) {
      FailureEDRAM4("scaled half-range depth resolve unsupported", *depth);
      return;
    }
    // With the scaled shadow map, the resolved texture stays at that same size.
    // The scene picks it by address, without looking at the size, and samples it with normalized coordinates
    // and point sampling, so 1024 read with the UVs of 1600 gives the same texel as 1024 upscaled to 1600
    // with NEAREST: the image does not change. What is saved is the upscaling blit (1.78 ms on the console)
    // and the larger half of the copy (10.24 MB -> 4.2 MB, twice per frame).
    uint32_t resolved_width = pitch_target, resolved_height = target_height;
    if (depth->guest_width && depth->guest_width != depth->width) {
      resolved_width = uint32_t(uint64_t(pitch_target) * depth->width / depth->guest_width);
      resolved_height = uint32_t(uint64_t(target_height) * depth->height / depth->guest_height);
    }
    uint32_t base_resolved = base & 0x1FFFFFFF;
    // banjo-nx: the rows really copied (see the color resolve).
    Resolved* resolved = SearchResolvedPartial(base_resolved, resolved_width,
                                               std::min(resolved_height, uint32_t(std::max(y1 - y0, 0))),
                                               texture_format, false, resolved_host_format, 2,
                                               base_resolved, dx, dy);
    if (!resolved) {
      resolved = GetResolved(base_resolved, resolved_width, resolved_height,
                                 texture_format, false, resolved_host_format);
    }
    if (!resolved || !Record()) {
      failure("depth resolve destination allocation/recording failure", depth);
      return;
    }
    BeforeResolvedWrite(base_resolved);  // sa2_gpu_resolve_alias
    resolved->image.resolved_depth_guestspace = depth->depth_float24_half;
    // Depth is part of the per-render-target inventory too, so the shadow map (the most expensive copy in the
    // frame) shows up in it.
    NoteCopy(base_resolved, resolved->image.width, resolved->image.height, 0);
    Prepare(*depth);
    Prepare(resolved->image);
    if (mode4 && (!depth->prepared || !resolved->image.prepared)) {
      failure("depth resolve source/destination initialization failure", depth);
      return;
    }
    NoteAreaUtil(*depth, y1);  // the area the game really resolves
    // With the scaled shadow map, the rectangle the guest sends is in 1600-pixel units even though the image
    // is smaller. Everything below reasons in guest pixels.
    const uint32_t guest_width = depth->guest_width ? depth->guest_width : depth->width;
    const uint32_t guest_height = depth->guest_height ? depth->guest_height : depth->height;
    const bool scaled = guest_width != depth->width || guest_height != depth->height;
    if (mode4 && (uint32_t(x0) >= guest_width || uint32_t(y0) >= guest_height ||
                  dx >= resolved->image.width || dy >= resolved->image.height)) {
      failure("depth resolve source/destination offset outside backing", depth);
      return;
    }
    const uint32_t width = std::min({uint32_t(x1 - x0), guest_width - uint32_t(x0),
                                     resolved->image.width - dx});
    const uint32_t height = std::min({uint32_t(y1 - y0), guest_height - uint32_t(y0),
                                    resolved->image.height - dy});
    if (!width || !height || uint32_t(x0) >= guest_width ||
        uint32_t(y0) >= guest_height || dx >= resolved->image.width ||
        dy >= resolved->image.height) {
      return;
    }
    // sa2_native_resolver_valid_content. If the content of this render target went away in an earlier
    // swap and nobody brought it back, it is in another texture: it is brought back before reading it (flickering
    // shadow maps).
    if (depth->invalid_content) {
      if (REXCVAR_GET(sa2_native_resolver_valid_content)) {
        if (!RestoreContent(*depth, true) &&
            REXCVAR_GET(sa2_native_edram_alias_mode) == 4) return;
      } else {
        ++resolver_old_content_;  // previous behavior: the old image is resolved
      }
    }
    if (depth->depth_float24_half) {
      // Eager MODE4 compute resolve: never routed through deferred VkImageCopy/swaps.
      if (!ResolverDepthGuestspaceEDRAM4(*depth, resolved->image,
          uint32_t(x0), uint32_t(y0), dx, dy, width, height,
          (reg.rb_copy_control >> 4) & 7u)) return;
      ++copies_;
      ResolvedWritten(base_resolved, uint64_t(width) * height);
      return;
    }
    if (scaled) {
      // Source and destination have the same reduced size, so it is a normal 1 to 1 copy
      // with the rectangle converted to image pixels. No blit, no scaling.
      const auto aX = [&](int32_t v) {
        return int32_t(int64_t(v) * depth->width / guest_width);
      };
      const auto aY = [&](int32_t v) {
        return int32_t(int64_t(v) * depth->height / guest_height);
      };
      const uint32_t img_width = std::min(
          {uint32_t(std::max(1, aX(x0 + int32_t(width)) - aX(x0))),
           depth->width - uint32_t(aX(x0)), resolved->image.width - uint32_t(aX(int32_t(dx)))});
      const uint32_t img_height = std::min(
          {uint32_t(std::max(1, aY(y0 + int32_t(height)) - aY(y0))),
           depth->height - uint32_t(aY(y0)), resolved->image.height - uint32_t(aY(int32_t(dy)))});
      if (!img_width || !img_height) {
        return;
      }
      VkImageCopy copy{};
      copy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
      copy.srcOffset = {aX(x0), aY(y0), 0};
      copy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
      copy.dstOffset = {aX(int32_t(dx)), aY(int32_t(dy)), 0};
      copy.extent = {img_width, img_height, 1};
      copy = HostCopy(*depth, resolved->image, copy);  // sa2-nx render resolution
      BarrierBeforeCopyResolve();
      copy_image_(commands_work_, depth->image, VK_IMAGE_LAYOUT_GENERAL,
                     resolved->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
      BarrierAfterCopyResolve();
      ++copies_;
      ResolvedWritten(base_resolved, uint64_t(img_width) * img_height);
      return;
    }
    // If the whole render target is resolved to a texture of the same size, the images are swapped and
    // nothing is copied (the shadow map is 1600x1600: 10 MB per copy).
    if (ResolverNoCopy()) {  // and if it is not swapped, why not
      const bool whole = x0 == 0 && y0 == 0 && dx == 0 && dy == 0 &&
                          width == depth->width && height == depth->height;
      // The clear is no longer a requirement (see sa2_native_swap_no_clear).
      const bool can = whole && (will_clear || REXCVAR_GET(sa2_native_swap_no_clear));
      if (can) {
        if (SwapWithResolved(*depth, *resolved, base_resolved)) {
          if (!will_clear) {
            ++swaps_no_clear_;
          }
          return;
        }
        ++no_swap_[2];  // the swap itself could not be done
      } else {
        /*
         * The label was backwards ever since sa2_native_swap_no_clear existed. With that setting
         * on, `can` = whole, so everything that lands here is "not resolved whole" and the log counted it as
         * "the command does not clear the render target". It is now split by the real cause: if the resolve
         * is whole, the cause is the clear; otherwise, it is the size.
         */
        ++no_swap_[whole ? 0 : 1];  // 0: the command does not clear the render target; 1: not the whole render target
        if (warnings_no_swap_ < 8) {
          ++warnings_no_swap_;
          REXLOG_INFO("[native] targets: no swap: {}x{} of {}x{} at ({},{})->({},{}), clears {} (base {:03X})",
                      width, height, depth->width, depth->height, x0, y0, dx, dy, will_clear,
                      base & 0xFFF);
        }
      }
    }
    VkImageCopy copy{};
    copy.srcSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    copy.srcOffset = {x0, y0, 0};
    copy.dstSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, 1};
    copy.dstOffset = {int32_t(dx), int32_t(dy), 0};
    copy.extent = {width, height, 1};
    if (!me::native::IsSingleSample(uint32_t(depth->sample_count))) {
      // sa2_msaa: a multisampled depth target resolves through a fragment pass (sample 0).
      if (!ResolveDepthMs(*depth, resolved->image, copy)) {
        Reject(18, "depth resolve from a multisampled target (sa2_msaa) failed");
        failure("multisampled depth resolve failed", depth);
        return;
      }
      ++copies_;
      ResolvedWritten(base_resolved, uint64_t(width) * height);
      return;
    }
    copy = HostCopy(*depth, resolved->image, copy);  // sa2-nx render resolution
    BarrierBeforeCopyResolve();
    copy_image_(commands_work_, depth->image, VK_IMAGE_LAYOUT_GENERAL,
                   resolved->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    BarrierAfterCopyResolve();
    ++copies_;
    ResolvedWritten(base_resolved, uint64_t(width) * height);
  }

  void RampGamma(const std::array<std::array<uint16_t, 3>, 256>& ramp) override {
    ramp_game_ = ramp;
    RecomputeRamp();
  }

  // The output table = the game's ramp in 0-1, followed by the neutral mix (saturation 1, vibrance 0,
  // 1 / gamma 1) and effects (no vignette, no scanlines) that the output shader still reads.
  void RecomputeRamp() {
    for (uint32_t i = 0; i < 256; ++i) {
      for (uint32_t c = 0; c < 3; ++c) {
        ramp_values_[i * 4 + c] = float(ramp_game_[i][c] & 0x3FF) / 1023.0f;
      }
      ramp_values_[i * 4 + 3] = 1.0f;
    }
    const size_t extra = 256 * 4;
    ramp_values_[extra] = 1.0f;      // saturation
    ramp_values_[extra + 1] = 0.0f;  // vibrance
    ramp_values_[extra + 2] = 1.0f;  // 1 / gamma
    ramp_values_[extra + 3] = 0.0f;
    ramp_values_[extra + 4] = 0.0f;  // vignette
    ramp_values_[extra + 5] = 0.0f;  // scanlines
    ramp_values_[extra + 6] = 0.0f;
    ramp_values_[extra + 7] = 0.0f;
    ++version_ramp_;
  }

  // ---- sa2-nx render resolution: live rescale (docked <-> handheld) -------------------------------------------------
  //
  // Called at the end of each Present, after the Swap's image was painted: when the wanted scale (the console's mode and
  // sa2_render_height_*) differs from the one the images were made at, every render target, depth target and resolved
  // texture gets a new VkImage at the new scale with its content scaled into it (color linear, depth nearest), and
  // keeps everything else: its ImageNative (so every pointer, EDRAM tile owner and memo stays valid), its guest size,
  // its address. Deferred work that names image handles (lazy front copies, resolve aliases) is recorded first. The
  // GPU is drained before and after (a one-frame hitch, only when the mode changes).
  // The device memory of the images the render scale applies to (render targets, depth targets, resolved and front
  // textures): for the [res] lines. Only at a rescale and in the 10 s report, when the scale or MSAA is in use.
  uint64_t BytesScaledImages() {
    uint64_t bytes = 0;
    const auto add = [&](const Image& image) {
      if (image.image == VK_NULL_HANDLE) return;
      VkMemoryRequirements requirements{};
      dfn_.vkGetImageMemoryRequirements(device_, image.image, &requirements);
      bytes += requirements.size;
    };
    for (const auto& [key, image] : targets_) add(image);
    for (const auto& [key, image] : depths_) add(image);
    for (const auto& [key, resolved] : resolved_) add(resolved.image);
    for (const auto& [key, resolved] : parked_resolved_) add(resolved.image);
    for (const ImageFront& front : fronts_images_) add(front.image);
    return bytes;
  }

  void RescaleIfWanted() {
    const uint32_t wanted = sa2::render::WantedScaleQuarters();
    if (wanted != scale_q_) RescaleAll(wanted);
  }

  // The Vulkan objects of an image that is being replaced (not its bookkeeping: the ImageNative lives on).
  void ReleaseHandles(const Image& old) {
    if (draws_ && old.view != VK_NULL_HANDLE) draws_->ForgetView(old.view);
    if (draws_ && old.image != VK_NULL_HANDLE) draws_->ForgetImage(old.image);
    if (auto it = views_raw64_edram_.find(old.image); it != views_raw64_edram_.end()) {
      dfn_.vkDestroyImageView(device_, it->second, nullptr);
      views_raw64_edram_.erase(it);
    }
    if (auto it = views_depth_edram_.find(old.image); it != views_depth_edram_.end()) {
      if (it->second.depth) dfn_.vkDestroyImageView(device_, it->second.depth, nullptr);
      if (it->second.stencil) dfn_.vkDestroyImageView(device_, it->second.stencil, nullptr);
      views_depth_edram_.erase(it);
    }
    for (uint32_t variant = 0; variant < 3; ++variant)
      if (auto it = framebuffers_conv_color_frag_.find({old.image, variant}); it != framebuffers_conv_color_frag_.end()) {
        if (it->second) dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);
        framebuffers_conv_color_frag_.erase(it);
      }
    if (auto it = views_raw64_rt_edram_.find(old.image); it != views_raw64_rt_edram_.end()) {
      dfn_.vkDestroyImageView(device_, it->second, nullptr);
      views_raw64_rt_edram_.erase(it);
    }
    if (auto it = framebuffers_import_depth_edram_.find(old.image); it != framebuffers_import_depth_edram_.end()) {
      dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);
      framebuffers_import_depth_edram_.erase(it);
    }
    if (auto it = framebuffers_depth_resolve_ms_.find(old.image); it != framebuffers_depth_resolve_ms_.end()) {
      if (it->second) dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);
      framebuffers_depth_resolve_ms_.erase(it);
    }
    if (old.view != VK_NULL_HANDLE) dfn_.vkDestroyImageView(device_, old.view, nullptr);
    if (old.image != VK_NULL_HANDLE) dfn_.vkDestroyImage(device_, old.image, nullptr);
    if (old.memory != VK_NULL_HANDLE) dfn_.vkFreeMemory(device_, old.memory, nullptr);
  }

  void RescaleAll(uint32_t q) {
    const auto started = std::chrono::steady_clock::now();
    const uint32_t q_old = scale_q_;
    const uint64_t bytes_old = BytesScaledImages();
    if (draws_) draws_->FinishPass();
    // Deferred copies name image handles: recorded now, at the old scale, so none refers to a replaced image.
    while (!pending_fronts_.empty()) RecordCopyFront(pending_fronts_.begin()->first);
    while (!aliases_.empty()) {
      const uint32_t address = aliases_.begin()->first;
      if (!MaterializeAlias(address)) DropAlias(address);  // (the guard has already turned the aliases off)
    }
    last_resolve_.valid = false;
    edram4_memo_.valid = false;
    // Everything recorded so far runs on the old images; then nothing uses them.
    ++waits_gpu_reason_[9];  // site9: render scale change (sa2-nx)
    WaitGpu();
    if (resolved_pool_device_error_ || !Record()) return;
    // Idle pool allocations and logical crops are caches: dropped, made again on demand at the new scale.
    while (!resolved_sleeping_.empty()) EvictResolvedSleeping(resolved_sleeping_.size() - 1);
    for (auto& clip : resolved_clips_) {
      if (draws_) draws_->ForgetImage(clip.image.image);
      Destroy(clip.image);
    }
    resolved_clips_.clear();
    clips_bytes_ = 0;
    scale_q_ = q;
    std::vector<Image> olds;
    uint32_t images = 0, blitted = 0, cleared = 0, failed = 0;
    const auto rescale = [&](Image& image) {
      if (image.image == VK_NULL_HANDLE || image.scale_q == q) return;
      Image fresh;
      if (!CreateScaled(fresh, image.width, image.height, image.vk_usage, image.format, image.sample_count, q)) {
        ++failed;
        return;  // keeps its old scale: transfers with it decline (scale check), the next rescale tries again
      }
      ++images;
      const bool depth = IsDepthFormat(image.format);
      VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
      barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = fresh.image;
      barrier.subresourceRange = depth ? kRangeDepth : kRangeColor;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                                0, nullptr, 0, nullptr, 1, &barrier);
      // The content, scaled: color with a linear filter, depth / stencil and float formats with the nearest texel
      // (depth needs blit support and a TRANSFER_DST destination: without them it starts cleared to the far plane,
      // which every SA2 frame clears anyway). Multisampled images cannot be blitted: they start cleared.
      const bool blit_ok = image.prepared && blit_ && me::native::IsSingleSample(uint32_t(image.sample_count)) &&
                           (image.vk_usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) &&
                           (image.vk_usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) && (!depth || scalable_depth_);
      if (blit_ok) {
        VkImageBlit region{};
        region.srcSubresource = {VkImageAspectFlags(depth ? VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT
                                                          : VK_IMAGE_ASPECT_COLOR_BIT), 0, 0, 1};
        region.dstSubresource = region.srcSubresource;
        region.srcOffsets[1] = {int32_t(HostWidth(image)), int32_t(HostHeight(image)), 1};
        region.dstOffsets[1] = {int32_t(HostWidth(fresh)), int32_t(HostHeight(fresh)), 1};
        const bool linear = image.format == VK_FORMAT_R8G8B8A8_UNORM ||
                            image.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
        BarrierBeforeCopyResolve();
        blit_(commands_work_, image.image, VK_IMAGE_LAYOUT_GENERAL, fresh.image, VK_IMAGE_LAYOUT_GENERAL, 1, &region,
              linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST);
        BarrierAfterCopyResolve();
        fresh.prepared = true;
        ++blitted;
      } else if (image.prepared) {
        fresh.prepared = false;  // Prepare clears it (color 0, depth far) before its next use
        ++cleared;
      }
      olds.push_back(Image{});
      Image& old = olds.back();
      old.image = image.image;
      old.memory = image.memory;
      old.view = image.view;
      image.image = fresh.image;
      image.memory = fresh.memory;
      image.view = fresh.view;
      image.host_width = fresh.host_width;
      image.host_height = fresh.host_height;
      image.scale_q = fresh.scale_q;
      image.prepared = fresh.prepared;
    };
    for (auto& [key, image] : targets_) rescale(image);
    for (auto& [key, image] : depths_) rescale(image);
    for (auto& [key, resolved] : resolved_) rescale(resolved.image);
    for (auto& [key, resolved] : parked_resolved_) rescale(resolved.image);
    for (ImageFront& front : fronts_images_) rescale(front.image);
    for (auto& [key, resolved] : resolved_) ++resolved.revision;  // new images: crops and caches resample them
    // The blits run, then the old images go.
    ++waits_gpu_reason_[9];
    WaitGpu();
    for (const Image& old : olds) ReleaseHandles(old);
    ++edram4_epoch_;
    if (draws_) {
      draws_->NotifyRescale(q_old, q);
      draws_->InvalidateTextures();
    }
    Record();
    ++rescales_;
    sa2::render::NoteRescale();
    REXLOG_INFO("[res] render scale {}/4 -> {}/4 ({}x{} -> {}x{} for 1280x720): {} images recreated ({} with their "
                "content scaled, {} cleared, {} failed) in {:.1f} ms; {:.1f} -> {:.1f} MB of images",
                q_old, q, ScaleSize(1280, q_old), ScaleSize(720, q_old), ScaleSize(1280, q), ScaleSize(720, q), images,
                blitted, cleared, failed,
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count(),
                double(bytes_old) / 1048576.0, double(BytesScaledImages()) / 1048576.0);
  }

  bool Present(rex::ui::Presenter* presenter, const TextureSwap& swap, uint32_t width,
                 uint32_t height) override {
    edram4_cleared_in_frame_.clear();  // redirected clear ordinals restart every frame
    // The per-draw diagnostic window opens here, on the PM4 ring thread, which is the one that records the
    // draws. Inside the paint call it would be another thread and the window would catch an arbitrary piece
    // of the frame (33 of 500 shadow draws were measured).
    OpenWindowDiagnostic();
    RexSwitchPerfCount(0);  // the profiler's "game N fps"
    // Interval since the previous Swap, for the stutter report below.
    {
      const auto now = std::chrono::steady_clock::now();
      if (last_swap_ != std::chrono::steady_clock::time_point{}) {
        const double ms = std::chrono::duration<double, std::milli>(now - last_swap_).count();
        // Frames over 30 ms go to stack sampling (usf4 0.7.3: was 45; the console's fight hitches are 43-52 ms). The window starts one normal frame (33 ms) earlier, because
        // the game thread prepares the frame ahead of the ring. Ticks at 19.2 MHz.
        if (ms > 30.0) {
          uint64_t end = 0;
#if defined(__aarch64__)
          asm volatile("mrs %0, cntpct_el0" : "=r"(end));  // libnx's armGetSystemTick
#endif
          const uint64_t long_value = uint64_t((ms + 33.3) * 19200.0);
          RexSwitchPerfHitch(end > long_value ? end - long_value : 0, end);
        }
        /*
         * What happens in a frame that runs long.
         *
         * Measured on the console: 7.3 % of frames exceed 50 ms and there are peaks of 2.1 seconds. Pipelines
         * are stable (116, 11 ms) and the texture cache no longer evicts, so the cause is something else.
         *
         * When a frame runs long, what that frame did (not the running total) is dumped. If the culprit is
         * loading textures, or creating pipelines, or a big copy, or simply that the game sent three times as
         * many draws, it shows here. It only writes when there is a stutter, so it costs nothing in the normal
         * case.
         */
        /*
         * The previous dump was not enough, and this is why.
         *
         * The stutters are 60-70 ms frames (median 67, against an average of 39.34) with an exactly normal
         * render load: 14 copies, 4 clears, 2 swaps, 1 restore, the same as any other frame. And they are not
         * quantized to the vblank (34 of them fall between 60.0 and 62.5 ms, which is not a multiple of
         * 16.67). So there are ~26 extra ms that are not in the drawing and the old dump did not see them.
         *
         * So now the time breakdown is dumped, not only the work count:
         *   - GPU work of that frame, with the timestamps we already measure. If it goes up to 60 ms it is
         *     scene load; if it stays at 33 the time went to the CPU or to waiting.
         *   - how long the thread took to record and how long to present.
         *   - new textures and pipelines, the usual suspects of an isolated peak.
         * With that, the log says where the 26 ms come from without guessing.
         */
        // The cap is 1000 warnings so that a whole session fits; it only writes when there is a stutter, so it
        // costs nothing in the normal case.
        if (ms > 60.0 && warnings_hitch_ < 1000) {
          ++warnings_hitch_;
          /* gpu_ns_ are raw GPU timestamps: multiply by 1.627 to get real milliseconds (NVK timestamp scale). */
          const double gpu_ms = double(gpu_ns_ - hitch_gpu_ns_) / 1e6 * 1.627;
          // The [hitch] lines go to the report thread (SA2_REPORT_RING).
          SA2_REPORT_RING(
              "[hitch] frame of {:.1f} ms (GPU {:.1f} real, record {:.1f}): {} copies, "
              "{} clears, {} swaps, {} restores, {} GPU waits",
              ms, gpu_ms, double(ns_record_ - hitch_ns_record_) / 1e6, copies_ - hitch_copies_,
              cleared_ - cleared_hitch_, swaps_ - hitch_resolves_,
              restores_ - hitch_restores_, waits_gpu_ - hitch_waits_);
          namespace e = sa2::waits;
          // Waits of this frame, in ms (see sa2_waits_hitch.h): the GPU fence and the output slot,
          // and what the ring re-checked of the textures.
          SA2_REPORT_RING(
              "[hitch] waits (ms): GPU fence {:.1f}, output {:.1f} | textures checked {:.1f} MB, "
              "postponed {}",
              double(ns_waits_gpu_ - hitch_ns_waits_gpu_) / 1e6,
              double(ns_wait_output_ - hitch_ns_wait_output_) / 1e6,
              double(e::g_bytes_fingerprint.load(std::memory_order_relaxed) - hitch_bytes_fingerprint_) / 1048576.0,
              e::g_postponed_fingerprints.load(std::memory_order_relaxed) - hitch_postponed_fingerprints_);
          // What the ring spent this frame on.
          const auto dif = [](const std::atomic<uint64_t>& a, uint64_t before) {
            return a.load(std::memory_order_relaxed) - before;
          };
          // And how long the ring waited for the vertex copy thread (WaitUploads). The wait before each
          // submit is inside "working" and the one before returning the read pointer is outside: it is read
          // separately, not added.
          // And what the ring spent creating textures (image, memory and view; "textures" starts afterwards),
          // how many the binding thread bound and how long the ring waited for it
          // (sa2_native_textures_binding_thread).
          SA2_REPORT_RING("[hitch] ring: textures {:.1f} ms ({} uploads, {:.1f} MB; "
                      "{} created; fingerprints {:.1f} + {:.1f} ms); waiting for the vertex copy thread {:.1f} ms "
                      "({} times), helping it {:.1f} ms ({} copies); "
                      "creating textures {:.1f} ms on the ring, {} bound on the thread, waiting for it {:.1f} ms",
                      double(dif(e::g_ns_textures, hitch_ns_textures_)) / 1e6,
                      dif(e::g_textures_uploads, hitch_textures_uploads_),
                      double(dif(e::g_bytes_uploaded, hitch_bytes_uploaded_)) / 1048576.0,
                      dif(e::g_created_textures, hitch_created_textures_),
                      double(dif(e::g_ns_raw_fingerprint, hitch_ns_raw_fingerprint_)) / 1e6,
                      double(dif(e::g_ns_data_fingerprint, hitch_ns_data_fingerprint_)) / 1e6,
                      double(dif(e::g_ns_waiting_copies, hitch_ns_waiting_copies_)) / 1e6,
                      dif(e::g_waits_copies, hitch_waits_copies_),
                      double(dif(e::g_ns_helping_copies, hitch_ns_helping_copies_)) / 1e6,
                      dif(e::g_helped_copies, hitch_helped_copies_),
                      double(dif(e::g_ns_create_textures, hitch_ns_create_textures_)) / 1e6,
                      dif(e::g_bound_textures_thread, hitch_bound_textures_thread_),
                      double(dif(e::g_ns_waiting_bindings, hitch_ns_waiting_bindings_)) / 1e6);
        }
        hitch_ns_waits_gpu_ = ns_waits_gpu_;
        hitch_ns_textures_ = sa2::waits::g_ns_textures.load(std::memory_order_relaxed);
        hitch_textures_uploads_ = sa2::waits::g_textures_uploads.load(std::memory_order_relaxed);
        hitch_bytes_uploaded_ = sa2::waits::g_bytes_uploaded.load(std::memory_order_relaxed);
        hitch_created_textures_ = sa2::waits::g_created_textures.load(std::memory_order_relaxed);
        hitch_ns_raw_fingerprint_ = sa2::waits::g_ns_raw_fingerprint.load(std::memory_order_relaxed);
        hitch_ns_data_fingerprint_ = sa2::waits::g_ns_data_fingerprint.load(std::memory_order_relaxed);
        hitch_bytes_fingerprint_ = sa2::waits::g_bytes_fingerprint.load(std::memory_order_relaxed);
        hitch_postponed_fingerprints_ = sa2::waits::g_postponed_fingerprints.load(std::memory_order_relaxed);
        hitch_ns_waiting_copies_ = sa2::waits::g_ns_waiting_copies.load(std::memory_order_relaxed);  // 170
        hitch_waits_copies_ = sa2::waits::g_waits_copies.load(std::memory_order_relaxed);
        hitch_ns_helping_copies_ = sa2::waits::g_ns_helping_copies.load(std::memory_order_relaxed);  // 185
        hitch_helped_copies_ = sa2::waits::g_helped_copies.load(std::memory_order_relaxed);
        // Creating textures and the binding thread (sa2_native_textures_binding_thread).
        hitch_ns_create_textures_ = sa2::waits::g_ns_create_textures.load(std::memory_order_relaxed);
        hitch_bound_textures_thread_ = sa2::waits::g_bound_textures_thread.load(std::memory_order_relaxed);
        hitch_ns_waiting_bindings_ = sa2::waits::g_ns_waiting_bindings.load(std::memory_order_relaxed);
        hitch_ns_wait_output_ = ns_wait_output_;
        hitch_gpu_ns_ = gpu_ns_;
        hitch_ns_record_ = ns_record_;
        hitch_copies_ = copies_;
        hitch_restores_ = restores_;
        hitch_waits_ = waits_gpu_;
        cleared_hitch_ = cleared_;
        hitch_resolves_ = swaps_;
      }
      last_swap_ = now;
    }
    uint32_t base = (swap.dword[1] & 0xFFFFF000) & 0x1FFFFFFF;
    static const bool diag_present_alias = REXCVAR_GET(sa2_diag_present_alias);
    if (diag_present_alias) {  // sa2_diag_present_alias
      for (const auto& [alias_address, alias] : aliases_) {
        if (alias.source == base && alias.sx == 0 && alias.sy == 0 && alias.dx == 0 && alias.dy == 0) {
          base = alias_address;
          break;
        }
      }
    }
    ++trace_frame_;
    // sa2_native_lazy_front. Before submitting the work: either it is drawn from the image that
    // holds the content (no copy) or the deferred copy is recorded now, ahead of the output.
    if (DiagAlias()) {
      const auto pr = resolved_.find(base);
      REXLOG_INFO("[diag alias] frame {} PRESENT {:08X} alias {} img {}", presented_, base, aliases_.count(base),
                  pr != resolved_.end() ? (void*)pr->second.image.image : nullptr);
    }
    if (!aliases_.empty() && aliases_.count(base)) MaterializeAlias(base);  // sa2_gpu_resolve_alias: it is read
    Image* const source_front = FrontOnPresent(base, width, height);
    // Without waiting for the GPU: the output goes after the work on the same queue.
    if (!presenter || !SendWork(false)) {
      return false;
    }
    auto it = resolved_.find(base);
    if (it == resolved_.end() || !it->second.image.prepared) {
      Reject(4, "Swap without a resolved texture: the test color is painted");
      return false;
    }
    Resolved& resolved = it->second;
    const uint32_t w = std::min(width ? width : resolved.image.width, resolved.image.width);
    const uint32_t h = std::min(height ? height : resolved.image.height, resolved.image.height);
    // sa2-nx render resolution: the guest output is the front buffer's host size (1920x1080 docked by default); the
    // presenter scales it to the display with the game's 16:9 aspect.
    const Image& painted_image = source_front ? *source_front : resolved.image;
    const uint32_t host_w = std::min<uint32_t>(ScaleSize(w, painted_image.scale_q), HostWidth(painted_image));
    const uint32_t host_h = std::min<uint32_t>(ScaleSize(h, painted_image.scale_q), HostHeight(painted_image));
    bool painted = false;
    if (deferred::AsyncSubmit()) {
      // usf4_deferred_async_submit: the output is painted and handed to the presenter on the worker, after the
      // frame's commands and its submission (queue order), so the presenter never sees an output whose work has not
      // been submitted yet. The job keeps its own copy of the source (PaintOutput reads its view and size); destroying
      // that image drains the worker first, so the view outlives the job.
      Image source_copy = painted_image;
      const bool from_target = source_front != nullptr;
      deferred::RunOnWorker(new std::function<void()>([this, presenter, source_copy, host_w, host_h,
                                                       from_target]() mutable {
        presenter->RefreshGuestOutput(
            host_w, host_h, 1280, 720, [&](rex::ui::Presenter::GuestOutputRefreshContext& base_context) {
              auto& context = static_cast<VulkanPresenter::VulkanGuestOutputRefreshContext&>(base_context);
              return PaintOutput(context, source_copy, host_w, host_h, from_target);
            });
      }));
      ++presented_;
      RescaleIfWanted();
      return true;
    }
    // How much of RefreshGuestOutput belongs to the SDK (before and after the callback) and how much to
    // PaintOutput.
    presenter->RefreshGuestOutput(
        host_w, host_h, 1280, 720, [&](rex::ui::Presenter::GuestOutputRefreshContext& base_context) {
          auto& context =
              static_cast<VulkanPresenter::VulkanGuestOutputRefreshContext&>(base_context);
          painted = PaintOutput(context, source_front ? *source_front : resolved.image, host_w, host_h,
                                 source_front != nullptr);  // sa2_native_lazy_front
          return painted;
        });
    if (painted) {
      ++presented_;
    }
    RescaleIfWanted();  // sa2-nx render resolution: docked <-> handheld, at this frame boundary
    return painted;
  }

  // Per-draw diagnostic window, one whole frame every N seconds. It is decided
  // in the Swap because the upload buffer rotates several times per frame.
  void OpenWindowDiagnostic() {
    const int32_t each = REXCVAR_GET(sa2_native_stats_per_draw_s);
    if (each <= 0) {
      window_diagnostic_ = false;
      return;
    }
    const auto now = std::chrono::steady_clock::now();
    if (window_diagnostic_) {
      window_diagnostic_ = false;  // that frame was already measured
      return;
    }
    if (std::chrono::duration_cast<std::chrono::seconds>(now - last_window_).count() >= each) {
      ReportCostPerShader();  // the previous window, read by now
      last_window_ = now;
      window_diagnostic_ = true;
      read_window_ = false;
    }
  }

  void Stats(uint64_t& copies, uint64_t& cleared, uint64_t& presented,
                    uint64_t& rejections) const override {
    copies = copies_;
    cleared = cleared_;
    presented = presented_;
    rejections = rejections_;
  }

  // report: waits of the ring for the GPU in CompleteA.
  uint64_t waits_gpu_ = 0;
  uint64_t ns_waits_gpu_ = 0;
  // report: cost of Record.
  uint64_t ns_record_ = 0;

  void TimeGpuPerCategory(
      std::array<uint64_t, kGpuCategories>& nanoseconds) const override {
    nanoseconds = gpu_categories_ns_;
  }

  void StatsPipeline(std::array<uint64_t, kGpuCategories>& fragments,
                            std::array<uint64_t, kGpuCategories>& vertices,
                            std::array<uint64_t, kGpuCategories>& primitives) const override {
    fragments = fragments_category_;
    vertices = vertices_category_;
    primitives = primitives_category_;
  }

  bool Draw(const SubmissionDraw& submission) override {
    depth_raster_grid_eligible_ = false;
    if (submission.registers) {
      msaa_edram_actual_ = (submission.registers[gr::XE_GPU_REG_RB_SURFACE_INFO] >> 16) & 3;
    }
    const bool mode4 = REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    // A redirected clear is a draw the memo does not see: the next draw takes the full path.
    if (mode4 && RedirectClearDepthEDRAM4(submission)) { edram4_memo_.valid = false; return true; }
    if (mode4 && RedirectClearStencilEDRAM4(submission)) { edram4_memo_.valid = false; return true; }
    if (mode4 && !PrepareDrawEDRAM4Memo(submission)) return false;
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 3 && submission.registers && submission.ps) {
      const uint32_t* r = submission.registers;
      const uint32_t tl = r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
      const uint32_t br = r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
      const uint32_t window = r[gr::XE_GPU_REG_PA_SC_WINDOW_OFFSET];
      const int32_t wx = (tl >> 31) ? 0 : int32_t(window << 17) >> 17;
      const int32_t wy = (tl >> 31) ? 0 : int32_t((window >> 16) << 17) >> 17;
      const int32_t x0 = std::max(0, int32_t(tl & 0x3FFF) + wx);
      const int32_t y0 = std::max(0, int32_t((tl >> 16) & 0x3FFF) + wy);
      const int32_t x1 = std::max(x0, int32_t(br & 0x3FFF) + wx);
      const int32_t y1 = std::max(y0, int32_t((br >> 16) & 0x3FFF) + wy);
      area_edram_ = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
      const uint32_t pitch = r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF;
      const uint32_t mask = r[gr::XE_GPU_REG_RB_COLOR_MASK];
      // A scissor can grow without opening a new render pass. Refresh ownership
      // on each color-writing draw, not only when the target is first bound.
      if ((r[gr::XE_GPU_REG_RB_MODECONTROL] & 7) == uint32_t(xenos::EdramMode::kColorDepth)) {
        for (uint32_t i = 0; i < 4; ++i) {
          if (!((mask >> (i * 4)) & 0xF) || !(submission.ps->outputs & (1u << i))) continue;
          const uint32_t info = r[kDiagRegsColorInfo[i]];
          Image* image = GetTarget(info & 0xFFF, (info >> 16) & 0xF, pitch);
          if (image && Record()) {
            Prepare(*image);
            ActivateTargetColor(info & 0xFFF, pitch, *image);
          }
        }
      }
    }
    const uint64_t draw_count_before = draws_ ? draws_->Drawn() : 0;
    const bool result = draws_ && draws_->Draw(submission);
    if (mode4 && draws_ && draws_->Drawn() == draw_count_before &&
        DrawTouchesTraceTileEDRAM4() && TraceEventEDRAM4(TraceTileEDRAM4()))
      REXLOG_INFO("[native] EDRAM TILE TRACE draw-result frame={} physical={} before={} after={} success={}",
                  trace_frame_, TraceTileEDRAM4(), draw_count_before,
                  draws_ ? draws_->Drawn() : 0, result);
    // A successful no-op (empty/skipped primitive, scissor or disabled pass)
    // must not claim pixels the current draw never wrote. Accepted deferred
    // sky draws also increment this counter and are flushed by FinishPass.
    if (mode4 && result && draws_->Drawn() > draw_count_before) {
      for (uint32_t slot = 0; slot < 5; ++slot) {
        if (slot == 0 && edram4_draw_images_[0] && edram4_draw_writes_[0] &&
            submission.edram_stencil_clear.eligible) {
          if (!PublishStencilClearEDRAM4(*edram4_draw_images_[0],
                                          submission.edram_stencil_clear,
                                          edram4_draw_plan_.length_tiles[0])) return false;
        } else if (edram4_draw_images_[slot] && edram4_draw_writes_[slot]) {
          edram4_published_no_stencil_ = slot == 0 && !edram4_draw_writes_stencil_;
          PublishEDRAM4(*edram4_draw_images_[slot],
                         RasterArea(*edram4_draw_images_[slot], edram4_draw_area_),
                         edram4_draw_plan_.length_tiles[slot]);
          edram4_published_no_stencil_ = false;
        }
      }
    }
    return result;
  }

  StatsDraws StatsOfDraws() const override {
    return draws_ ? draws_->Stats() : StatsDraws{};
  }

  void WaitUploads() override {
    if (draws_) {
      draws_->WaitUploads();
    }
  }

  size_t PendingCopies() const override {  // fence measurement only
    return draws_ ? draws_->PendingCopies() : 0;
  }

  // --- Interface lent to the draw code ----------------------------------------

  VkCommandBuffer CommandsWork() override {
    return Record() ? commands_work_ : VK_NULL_HANDLE;
  }

  uint32_t HostMsaaSamples() const override { return msaa_samples_; }

  // sa2_msaa: a single-sample image, or one of the host multisampled game targets.
  bool SamplesSupported(const Image& image) const {
    return me::native::IsSingleSample(uint32_t(image.sample_count)) ||
           (msaa_samples_ > 1 && uint32_t(image.sample_count) == msaa_samples_);
  }

  VkCommandBuffer CommandsUpload() override {
    if (!Record()) {
      return VK_NULL_HANDLE;
    }
    if (!recording_upload_) {
      VkCommandBufferBeginInfo start{};
      start.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
      start.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
      // usf4_deferred_async_submit: directly (its pool was reset in BeginRecording, nothing of it is queued).
      if ((deferred::AsyncSubmit() ? deferred::Real() : dfn_).vkBeginCommandBuffer(commands_upload_, &start) !=
          VK_SUCCESS) {
        return VK_NULL_HANDLE;
      }
      // Cached images may still be sampled by the preceding submission. Queue
      // submission order alone does not order those reads against new uploads.
      VkMemoryBarrier before_uploads{};
      before_uploads.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      before_uploads.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      before_uploads.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      dfn_.vkCmdPipelineBarrier(commands_upload_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before_uploads,
                                0, nullptr, 0, nullptr);
      recording_upload_ = true;
    }
    return commands_upload_;
  }

  uint64_t GenerationCommands() const override { return generation_commands_; }

  ImageNative* TargetColor(uint32_t base, uint32_t format, uint32_t pitch) override {
    Image* image = GetTarget(base, format, pitch);
    if (!image || !Record()) {
      return nullptr;
    }
    Prepare(*image);
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4) {
      BeforeWriteColor(*image);
      if (!RestoreContent(*image)) return nullptr;
      return image; // Acquired before DrawsVulkan can start a render pass.
    }
    ActivateTargetColor(base, pitch, *image);
    BeforeWriteColor(*image);  // sa2_native_lazy_front
    // Mandatory since color is also swapped. Here a pass starts that draws on top; if the content went away
    // in a swap and nobody has cleared it, it has to be brought back. Without this, the first frame that
    // draws without clearing shows the previous frame.
    RestoreContent(*image);
    return image;
  }

  ImageNative* TargetDepth(uint32_t base, uint32_t format, uint32_t pitch) override {
    Image* image = GetDepth(base, format, pitch);
    if (!image || !Record()) {
      return nullptr;
    }
    Prepare(*image);
    // A pass that draws on top starts here, so if the content went away in a swap and nobody has cleared it
    // since then, it has to be brought back.
    if (!RestoreContent(*image) && REXCVAR_GET(sa2_native_edram_alias_mode) == 4)
      return nullptr;
    return image;
  }

  const ImageNative* ResolvedTexture(uint32_t address) override {
    const auto it = resolved_.find(address);
    if (it == resolved_.end()) {
      return nullptr;
    }
    // And how many times each address is requested, for the per-render-target copy report. A resolved
    // texture that is copied every frame and never requested is a copy that is not needed. It is a ++ on the
    // entry that has already been looked up: it costs nothing even though this is called thousands of times
    // per frame.
    ++it->second.reads;
    if (!presented_fronts_.empty() && presented_fronts_.count(address)) {
      NoteReadFront(address);  // sa2_native_lazy_front
    }
    // sa2_gpu_resolve_alias: a draw reads it, so it gets its bytes now (before the draw). MaterializeAlias neither
    // adds nor removes entries of resolved_: `it` stays valid.
    if (it->second.alias_pending) [[unlikely]] MaterializeAlias(address);
    return it->second.image.prepared ? &it->second.image : nullptr;
  }

  const ImageNative* ResolvedTextureForFetch(uint32_t address, const uint32_t fetch[6],
                                              const ImageNative& source) override {
    const uint32_t width = (fetch[2] & 0x1FFFu) + 1;
    const uint32_t height = ((fetch[2] >> 13) & 0x1FFFu) + 1;
    const auto failure = [&](const char* reason) -> const ImageNative* {
      if (++clips_failures_ <= 32)
        REXLOG_ERROR("[native] resolved logical crop rejected {:08X} {}x{}: {}", address,
                     width, height, reason);
      return nullptr;  // The caller distinguishes this from no resolved image; no stale RAM fallback.
    };
    // This guard also covers the exact-size and disabled-crop fast paths. A
    // known multisample backing must never fall through to stale guest RAM or
    // become an ordinary sampler2D / buffer-image crop.
    if (!me::native::IsSingleSample(uint32_t(source.sample_count)))
      return failure("multisample backing requires a proven single-sample resolve");
    if (!REXCVAR_GET(sa2_native_logical_resolved_size) ||
        REXCVAR_GET(sa2_native_edram_alias_mode) != 4 ||
        !resolved_pool_startup_eligible_ || ResolverNoCopy() ||
        (width == source.width && height == source.height)) return &source;
    if (resolved_pool_device_error_) return failure("GPU fence/submission error");
    // ResolvedTexture was already called exactly once. Materialize any remaining logical write
    // before taking its revision, without incrementing read counters a second time.
    if (pending_fronts_.count(address)) RecordCopyFront(address);
    if (!aliases_.empty() && aliases_.count(address)) MaterializeAlias(address);  // sa2_gpu_resolve_alias
    auto r = resolved_.find(address);
    if (r == resolved_.end() || !r->second.image.prepared ||
        r->second.image.invalid_content ||
        expired_fronts_.count(address) ||
        pending_fronts_.count(address)) return failure("unmaterialized or stale source");
    const Image canonical = r->second.image;
    if (!me::native::IsSingleSample(uint32_t(canonical.sample_count)))
      return failure("materialized crop source is multisampled");
    const uint64_t source_revision = r->second.revision;
    uint32_t texel_bytes = 0;
    switch (canonical.format) {
      case VK_FORMAT_R8G8B8A8_UNORM:
      case VK_FORMAT_A2B10G10R10_UNORM_PACK32:
      case VK_FORMAT_R16G16_SFLOAT:
      case VK_FORMAT_R32_SFLOAT: texel_bytes = 4; break;
      case VK_FORMAT_R16G16B16A16_SFLOAT:
      case VK_FORMAT_R32G32_SFLOAT: texel_bytes = 8; break;
      default: break;  // Native depth/stencil crops need a separate aspect contract.
    }
    xenos::xe_gpu_texture_fetch_t texture_fetch{};
    std::memcpy(&texture_fetch, fetch, sizeof(texture_fetch));
    uint32_t base_page = 0, mip_page = 0, mip_min = 0, mip_max = 0;
    gr::texture_util::GetSubresourcesFromFetchConstant(texture_fetch, nullptr, nullptr, nullptr,
                                                     &base_page, &mip_page, &mip_min, &mip_max);
    me::native::ResolvedSamplingContract contract{};
    contract.logical_width = width;
    contract.logical_height = height;
    contract.guest_pitch = ((fetch[0] >> 22) & 0x1FFu) << 5;
    contract.backing_width = canonical.width;
    contract.backing_height = canonical.height;
    contract.bytes_per_texel = texel_bytes;
    contract.base_formats_compatible = canonical.resolved_guest_format < 64 &&
        gr::GetBaseFormat(xenos::TextureFormat(canonical.resolved_guest_format)) ==
        gr::GetBaseFormat(texture_fetch.format);
    contract.source_origin_zero = texture_fetch.tiled && base_page && (base_page << 12) == address &&
        (!texture_fetch.packed_mips || gr::texture_util::GetPackedMipLevel(width, height) != 0);
    contract.single_level_2d = texture_fetch.type == xenos::FetchConstantType::kTexture &&
        texture_fetch.dimension == xenos::DataDimension::k2DOrStacked && !texture_fetch.stacked &&
        mip_min == 0 && mip_max == 0 && mip_page == 0 && texture_fetch.mip_address == 0;
    contract.scale_1x = canonical.guest_width == 0 && canonical.guest_height == 0;
    constexpr VkDeviceSize kMaxBytes = 128ull << 20;
    const auto plan = me::native::AnalyzeResolvedSampling(contract, kMaxBytes);
    if (plan.mode == me::native::ResolvedSamplingMode::Unsupported) {
      if (clips_failures_ < 32)
        REXLOG_ERROR("[native] resolved crop contract failure {}: pitch {} backing {}x{} "
                     "guest formats {}/{} host {}", int(plan.failure), contract.guest_pitch,
                     canonical.width, canonical.height, canonical.resolved_guest_format,
                     uint32_t(texture_fetch.format), int(canonical.format));
      return failure("unproven logical fetch layout");
    }
    if (plan.mode == me::native::ResolvedSamplingMode::Direct) return &r->second.image;
    if (!copy_image_) return failure("image-copy entry point unavailable");
    ResolvedClip* crop = nullptr;
    for (auto& entry : resolved_clips_)
      if (entry.address == address && entry.image.width == width &&
          entry.image.height == height && entry.image.format == canonical.format) {
        crop = &entry;
        break;
      }
    if (crop && crop->source == canonical.image && crop->revision == r->second.revision) {
      ++clips_cache_hits_;
      return &crop->image;
    }
    if (!crop) {
      if (!me::native::ResolvedAllocationFits(resolved_clips_.size(), clips_bytes_,
                                             plan.copy_bytes)) return failure("bounded crop cache capacity");
      ResolvedClip new_value;
      new_value.address = address;
      if (!Create(new_value.image, width, height, VK_IMAGE_USAGE_SAMPLED_BIT |
                  VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  canonical.format)) return failure("crop allocation failed");
      VkMemoryRequirements requirements{};
      dfn_.vkGetImageMemoryRequirements(device_, new_value.image.image, &requirements);
      if (!me::native::ResolvedAllocationFits(resolved_clips_.size(), clips_bytes_,
                                             requirements.size)) {
        Destroy(new_value.image);  // Fresh, unrecorded, no descriptors or consumers.
        return failure("actual Vulkan allocation exceeds bounded crop cache");
      }
      new_value.bytes = requirements.size;
      clips_bytes_ += new_value.bytes;
      resolved_clips_.push_back(new_value);
      crop = &resolved_clips_.back();
      ++clips_allocations_;
    }
    if (draws_) draws_->FinishPass();
    if (!Record()) return failure("crop work recording failed");
    r = resolved_.find(address);  // Closing a shadow pass may record resolves or submit / rotate.
    if (r == resolved_.end() || r->second.image.image != canonical.image ||
        r->second.revision != source_revision) return failure("source changed while closing previous pass");
    Prepare(crop->image);  // Fresh only; reused crops stay GENERAL and are never cleared in UPLOAD.
    if (!crop->image.prepared) return failure("crop initialization failed");
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    VkImageCopy copy{};
    copy.srcSubresource = copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.extent = {width, height, 1};
    copy = HostCopy(canonical, crop->image, copy);  // sa2-nx render resolution
    copy_image_(commands_work_, canonical.image, VK_IMAGE_LAYOUT_GENERAL,
                    crop->image.image, VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
    before.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    crop->source = canonical.image;
    crop->revision = source_revision;
    crop->image.swap_rb = canonical.swap_rb;
    crop->image.resolved_guest_format = canonical.resolved_guest_format;
    crop->image.resolved_depth_guestspace = canonical.resolved_depth_guestspace;
    ++clips_copies_;
    const auto now = std::chrono::steady_clock::now();
    if (clips_copies_ <= 8 || now - clips_report_ >= std::chrono::seconds(10)) {
      clips_report_ = now;
      REXLOG_INFO("[native] resolved logical crop {:08X}: {}x{} -> {}x{}, revision {}; "
                  "allocations={}, copies={}, cache hits={}, failures={}, bytes={}", address,
                  canonical.width, canonical.height, width, height, crop->revision,
                  clips_allocations_, clips_copies_, clips_cache_hits_, clips_failures_, clips_bytes_);
    }
    return &crop->image;
  }

  // sa2_native_lazy_front. Deferred copy to a front buffer, and the render target's spare images.
  struct PendingFront {
    Image* target = nullptr;           // source render target (targets_ never erases: the pointer stays valid)
    VkImage source_vk = VK_NULL_HANDLE;  // image holding the content: the target's own or a retained one
    int32_t retained = -1;               // index in fronts_images_ if it is no longer in the target
    VkImage texture_vk = VK_NULL_HANDLE;
    VkImageCopy copy{};
  };
  struct ImageFront {
    Image image;              // same size, format and usage as the target: swapped with it
    uint32_t retained_per = 0;  // address of the front buffer that retains it; 0 = free
  };

  // Is this address a front buffer read only by the Swap? A Swap presented it recently and no draw has
  // sampled it in that time.
  bool IsFrontOnlyOfTheSwap(uint32_t address) const {
    const auto presented = presented_fronts_.find(address);
    if (presented == presented_fronts_.end() || presented_ - presented->second > kFrontFrames) {
      return false;
    }
    const auto read = read_fronts_.find(address);
    return read == read_fronts_.end() || presented_ - read->second > kFrontFrames;
  }

  // The retained image of a deferred copy becomes free again.
  void ReleaseRetained(const PendingFront& pending) {
    if (pending.retained >= 0 && size_t(pending.retained) < fronts_images_.size()) {
      fronts_images_[size_t(pending.retained)].retained_per = 0;
    }
  }

  // The image holding the deferred copy's content, if it is still the same one; otherwise nullptr.
  Image* SourceFront(const PendingFront& pending) {
    Image* source = nullptr;
    if (pending.retained >= 0) {
      if (size_t(pending.retained) < fronts_images_.size()) {
        source = &fronts_images_[size_t(pending.retained)].image;
      }
    } else {
      source = pending.target;
    }
    return source && source->image == pending.source_vk && source->prepared ? source : nullptr;
  }

  // Records the deferred copy to that front buffer (outside a pass) and removes it from the list.
  void RecordCopyFront(uint32_t address) {
    const auto p = pending_fronts_.find(address);
    if (p == pending_fronts_.end()) {
      return;
    }
    const PendingFront pending = p->second;
    pending_fronts_.erase(p);
    const auto r = resolved_.find(address);
    Image* const source = SourceFront(pending);
    if (r == resolved_.end() || r->second.image.image != pending.texture_vk || !source || !copy_image_ ||
        !Record()) {
      ReleaseRetained(pending);
      expired_fronts_.insert(address);  // should not happen: every image change goes through the hooks first
      return;
    }
    if (draws_) {
      draws_->FinishPass();  // may come from a draw (ResolvedTexture): the copy goes outside the pass
    }
    BeforeResolvedWrite(address);  // sa2_gpu_resolve_alias
    MarkGpu(kGpuCopies);
    const VkImageCopy host_copy = HostCopy(*source, r->second.image, pending.copy);  // sa2-nx render resolution
    BarrierBeforeCopyResolve();
    copy_image_(commands_work_, pending.source_vk, VK_IMAGE_LAYOUT_GENERAL, pending.texture_vk,
                   VK_IMAGE_LAYOUT_GENERAL, 1, &host_copy);
    BarrierAfterCopyResolve();
    ++copies_;
    ResolvedWritten(address, uint64_t(pending.copy.extent.width) * pending.copy.extent.height);
    expired_fronts_.erase(address);
    ReleaseRetained(pending);
  }

  // Another resolve (or an image swap) reaches that texture. If it covers the whole texture, the deferred
  // copy is unnecessary and is dropped: nobody will see its content. Otherwise it is recorded first.
  void ResolverPreviousFront(uint32_t address, bool whole) {
    if (whole && !expired_fronts_.empty()) {
      expired_fronts_.erase(address);  // fully covered: it is no longer missing any copy
    }
    if (pending_fronts_.empty()) {
      return;
    }
    const auto p = pending_fronts_.find(address);
    if (p == pending_fronts_.end()) {
      return;
    }
    if (!whole) {
      RecordCopyFront(address);
      ++copied_front_write_;
      return;
    }
    front_saved_pixels_ += uint64_t(p->second.copy.extent.width) * p->second.copy.extent.height;
    ReleaseRetained(p->second);
    pending_fronts_.erase(p);
    ++replaced_front_;
  }

  // Defers the copy if it is 1:1, covers the whole texture from the target's corner, and that address is
  // read only by the Swap.
  bool PostponeCopyFront(uint32_t address, Image& target, const Resolved& resolved, const VkImageCopy& copy) {
    if (off_front_ || !REXCVAR_GET(sa2_native_lazy_front)) {
      return false;
    }
    if (copy.srcOffset.x != 0 || copy.srcOffset.y != 0 || copy.dstOffset.x != 0 || copy.dstOffset.y != 0 ||
        copy.extent.width != resolved.image.width || copy.extent.height != resolved.image.height ||
        target.format != kColorFormat || resolved.image.format != kColorFormat) {
      return false;
    }
    // Small ones are read back for the guest (ResolvedRead): those are never deferred.
    const int32_t texels_read = REXCVAR_GET(sa2_native_resolved_read_texels);
    if (uint64_t(copy.extent.width) * copy.extent.height <= uint64_t(std::max<int32_t>(texels_read, 0))) {
      return false;
    }
    if (!IsFrontOnlyOfTheSwap(address)) {
      return false;
    }
    BeforeResolvedWrite(address);  // sa2_gpu_resolve_alias
    PendingFront& pending = pending_fronts_[address];
    pending.target = &target;
    pending.source_vk = target.image;
    pending.retained = -1;
    pending.texture_vk = resolved.image.image;
    pending.copy = copy;
    ResolvedWritten(address);  // new contents (deferred)
    expired_fronts_.erase(address);
    ++postponed_front_;
    return true;
  }

  // Before writing to a color target in any way other than a full clear (a pass, a restore, an image
  // swap): deferred copies whose source is its image are recorded now (exact).
  void BeforeWriteColor(const Image& target) {
    if (pending_fronts_.empty() || target.image == VK_NULL_HANDLE) {
      return;
    }
    for (auto it = pending_fronts_.begin(); it != pending_fronts_.end();) {
      if (it->second.retained >= 0 || it->second.source_vk != target.image) {
        ++it;
        continue;
      }
      const uint32_t address = it->first;
      ++it;  // RecordCopyFront erases that entry: the iterator is already on the next one
      RecordCopyFront(address);
      ++copied_front_write_;
    }
  }

  // Before a full clear of a color target. If its image is the source of a deferred front buffer, the
  // target takes a free spare image of the same size and the one holding the content is retained for the
  // front buffer: the clear does not need the old content, so nothing is copied. Without a spare (or with
  // two front buffers from the same image), the copy is recorded first, as usual.
  void RotateFrontBeforeClear(Image& target) {
    if (pending_fronts_.empty() || target.image == VK_NULL_HANDLE) {
      return;
    }
    uint32_t base = 0;
    uint32_t how_many = 0;
    for (const auto& [address, pending] : pending_fronts_) {
      if (pending.retained < 0 && pending.source_vk == target.image) {
        base = address;
        ++how_many;
      }
    }
    if (how_many == 0) {
      return;
    }
    int32_t free = -1;
    if (how_many == 1) {
      for (size_t i = 0; i < fronts_images_.size(); ++i) {
        const Image& candidate = fronts_images_[i].image;
        if (fronts_images_[i].retained_per == 0 && candidate.image != VK_NULL_HANDLE &&
            candidate.width == target.width && candidate.height == target.height &&
            candidate.format == target.format) {
          free = int32_t(i);
          break;
        }
      }
      if (free < 0 && fronts_images_.size() < kFrontImagesMax) {
        ImageFront new_entry;
        // Same usage flags as GetTarget: the image becomes the render target.
        if (Create(new_entry.image, target.width, target.height,
                  VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                  target.format)) {
          Prepare(new_entry.image);
          fronts_images_.push_back(new_entry);
          free = int32_t(fronts_images_.size()) - 1;
          REXLOG_INFO("[native] targets: lazy front: spare image {} of {}x{} for the front buffer target",
                      fronts_images_.size(), target.width, target.height);
        }
      }
    }
    if (free < 0) {
      ++front_no_replenished_;
      BeforeWriteColor(target);  // the copy is recorded before the clear, as usual
      return;
    }
    Image& replenished = fronts_images_[size_t(free)].image;
    std::swap(target.image, replenished.image);
    std::swap(target.memory, replenished.memory);
    std::swap(target.view, replenished.view);
    std::swap(target.prepared, replenished.prepared);
    fronts_images_[size_t(free)].retained_per = base;
    pending_fronts_[base].retained = free;  // source_vk is still the content image, now retained
    if (draws_) {
      draws_->InvalidateImages(target.image, replenished.image);
    }
    ++front_rotations_;
  }

  // A draw samples a front buffer. If its copy is deferred, it is recorded now (before the draw).
  void NoteReadFront(uint32_t address) {
    read_fronts_[address] = presented_;
    if (!pending_fronts_.empty() && pending_fronts_.count(address)) {
      RecordCopyFront(address);
      ++copied_front_read_;
    }
    if (!expired_fronts_.empty() && expired_fronts_.erase(address)) {
      ++front_late_reads_;
      if (!off_front_) {
        off_front_ = true;
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] targets: lazy front: DIFFERENCE, a draw samples front buffer {:08X} without its last "
                     "copy. Turned off for the rest of the session: always copy",
                     address);
      }
    }
  }

  // The Swap of that front buffer, before the submission. Returns the image that can be presented without
  // a copy (the target's or the retained one, with the front buffer in its corner) or nullptr: in that
  // case any deferred copy is recorded here (in the same submission, ahead of the output) and the output
  // reads the texture as usual.
  Image* FrontOnPresent(uint32_t address, uint32_t width, uint32_t height) {
    presented_fronts_[address] = presented_;
    if (!expired_fronts_.empty() && expired_fronts_.erase(address)) {
      ++front_late_reads_;
      if (!off_front_) {
        off_front_ = true;
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] targets: lazy front: DIFFERENCE, the Swap paints front buffer {:08X} without its last copy. "
                     "Turned off for the rest of the session: always copy",
                     address);
      }
    }
    if (pending_fronts_.empty()) {
      return nullptr;
    }
    const auto p = pending_fronts_.find(address);
    if (p == pending_fronts_.end()) {
      return nullptr;
    }
    const auto r = resolved_.find(address);
    Image* const source = SourceFront(p->second);
    bool can = source && r != resolved_.end() && r->second.image.image == p->second.texture_vk &&
                 r->second.image.prepared && ramp_gamma_ &&
                 !off_front_;
    if (can) {
      // Same computation as Present: the output has the texture's size (exact variant) and the copy
      // covers it.
      const Image& texture = r->second.image;
      const uint32_t w = std::min(width ? width : texture.width, texture.width);
      const uint32_t h = std::min(height ? height : texture.height, texture.height);
      can = w == texture.width && h == texture.height && p->second.copy.extent.width == w &&
              p->second.copy.extent.height == h;
    }
    if (!can) {
      RecordCopyFront(address);
      ++copied_front_swap_;
      return nullptr;
    }
    if (p->second.retained >= 0) {
      ++painted_retained_front_;
    } else {
      ++painted_front_target_;
    }
    return source;
  }

  // Upload buffer full: submit what has been recorded and continue in the other slot, with its empty
  // upload buffer (Record only waits if that slot's last submission is still on the GPU).
  bool SendAndWait() override { return SendWork(false) && Record(); }

  // The function above submits and continues; this one waits. Same order as every other place that must
  // destroy something the GPU might still be reading (see the resolved texture that changes size).
  bool WaitGpuOfTheAll() override {
    ++waits_gpu_reason_[2]; WaitGpu();
    return Record();
  }

  void MarkGpu(uint32_t category) override {
    if (queries_ == VK_NULL_HANDLE || !recording_) {
      return;
    }
    SlotWork& slot = slots_[slot_];
    if (slot.categories.size() + 1 >= kMarksPerSlot) {
      ++exhausted_marks_;
      return;
    }
    if (
(!slot.categories.empty() && slot.categories.back() == category)) {
      return;  // same category continues
    }
    if (!marks_categories_ && !slot.categories.empty()) {
      return;  // sa2_gpu_marks_categories = false: first mark of the submission only
    }
    write_mark_(commands_work_,
                    slot.precise_marks ? VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    queries_, slot_ * kMarksPerSlot + uint32_t(slot.categories.size()));
    slot.categories.push_back(uint8_t(category));
    slot.marks_written = std::max(slot.marks_written, uint32_t(slot.categories.size()));
    slot.labels.push_back(0);
    slot.descriptions.push_back(0);
    slot.draws_mark.push_back(0);
  }

  void DescribeMarkGpu(uint64_t description) override {
    SlotWork& slot = slots_[slot_];
    if (!slot.descriptions.empty()) slot.descriptions.back() = description;
  }

  void CountDrawMarkGpu() override {
    SlotWork& slot = slots_[slot_];
    if (!slot.draws_mark.empty()) ++slot.draws_mark.back();
  }

  void LabelMarkGpu(uint32_t label) override {
    SlotWork& slot = slots_[slot_];
    if (!slot.labels.empty() && !slot.labels.back()) slot.labels.back() = label;
  }

  // One query per pass, from before vkCmdBeginRenderPass to after EndRenderPass.
  uint32_t BeginStats(uint32_t category) override {
    // With the per-draw diagnostic enabled this one is not opened: two queries of the same type cannot be
    // active at once, and the per-pass query would enclose the per-draw ones.
    if (stats_ == VK_NULL_HANDLE || !recording_ ||
        !REXCVAR_GET(sa2_native_stats_pipeline) ||
        REXCVAR_GET(sa2_native_stats_per_draw_s) > 0) {
      return UINT32_MAX;
    }
    SlotWork& slot = slots_[slot_];
    if (slot.stats.size() >= kStatsPerSlot) {
      ++stats_no_site_;
      if (stats_no_site_ % 200 == 1) {
        REXLOG_WARN("[native] targets: no room for pass statistics ({} times): the work has more than {} "
                    "passes", stats_no_site_, kStatsPerSlot);
      }
      return UINT32_MAX;
    }
    const uint32_t index =
        slot_ * kStatsPerSlot + uint32_t(slot.stats.size());
    dfn_.vkCmdBeginQuery(commands_work_, stats_, index, 0);
    slot.stats.emplace_back(index, uint8_t(category));
    return index;
  }

  void FinishStats(uint32_t index) override {
    if (stats_ != VK_NULL_HANDLE && index != UINT32_MAX && recording_) {
      dfn_.vkCmdEndQuery(commands_work_, stats_, index);
    }
  }

  // One query per draw, tagged with its pixel shader.
  uint32_t BeginStatsDraw(uint32_t label, uint32_t category, uint32_t vs_more_one) override {
    if (stats_draw_ == VK_NULL_HANDLE || !recording_ || !window_diagnostic_) {
      return UINT32_MAX;
    }
    SlotWork& slot = slots_[slot_];
    if (slot.stats_draw.size() >= kStatsDrawPerSlot) {
      ++stats_draw_no_site_;
      if (stats_draw_no_site_ % 2000 == 1) {
        REXLOG_WARN("[native] targets: no room for per-draw statistics ({} times)",
                    stats_draw_no_site_);
      }
      return UINT32_MAX;
    }
    const uint32_t index = slot_ * kStatsDrawPerSlot +
                            uint32_t(slot.stats_draw.size());
    // the draw's GPU time. ALL_COMMANDS waits for the previous work, so the pair brackets
    // this draw alone (serialized: only for ranking pixel shaders, not for the frame's total).
    if (marks_draw_ != VK_NULL_HANDLE) {
      write_mark_(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, marks_draw_, index * 2);
    }
    dfn_.vkCmdBeginQuery(commands_work_, stats_draw_, index, 0);
    slot.stats_draw.emplace_back(
        index, uint16_t((category % kGpuCategories) * kLabelsShader + label % kLabelsShader));
    slot.draw_ps.push_back(((category % kGpuCategories) << 24) | (label & 0xFFFFFF));
    slot.draw_vs.push_back(vs_more_one);
    return index;
  }

  void FinishStatsDraw(uint32_t index) override {
    if (stats_draw_ != VK_NULL_HANDLE && index != UINT32_MAX && recording_) {
      dfn_.vkCmdEndQuery(commands_work_, stats_draw_, index);
      if (marks_draw_ != VK_NULL_HANDLE) {
        write_mark_(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, marks_draw_, index * 2 + 1);
      }
    }
  }

 private:
  // sa2-nx render resolution (me_resolution_scale.h): guest coordinates of an image in its host pixels. Every Vulkan
  // command on a render target or resolved texture goes through these; with scale_q 4 they return their input.
  static int32_t HostEdge(const Image& image, int32_t guest) { return ScaleEdge(guest, image.scale_q); }
  static VkRect2D HostRect(const Image& image, const VkRect2D& guest) { return ScaleRect(guest, image.scale_q); }
  // A guest-unit copy between two images (each side at its own scale; the extent is what both have).
  static VkImageCopy HostCopy(const Image& source, const Image& target, VkImageCopy copy) {
    if (source.scale_q == kScaleQuartersNative && target.scale_q == kScaleQuartersNative) return copy;
    const VkRect2D s = ScaleRect(VkRect2D{{copy.srcOffset.x, copy.srcOffset.y}, {copy.extent.width, copy.extent.height}},
                                 source.scale_q);
    const VkRect2D d = ScaleRect(VkRect2D{{copy.dstOffset.x, copy.dstOffset.y}, {copy.extent.width, copy.extent.height}},
                                 target.scale_q);
    copy.srcOffset = {s.offset.x, s.offset.y, copy.srcOffset.z};
    copy.dstOffset = {d.offset.x, d.offset.y, copy.dstOffset.z};
    copy.extent.width = std::min(s.extent.width, d.extent.width);
    copy.extent.height = std::min(s.extent.height, d.extent.height);
    return copy;
  }
  static void HostCopies(const Image& source, const Image& target, VkImageCopy* copies, uint32_t count) {
    for (uint32_t i = 0; i < count; ++i) copies[i] = HostCopy(source, target, copies[i]);
  }

  bool Reject(uint32_t cause, const char* text) {
    ++rejections_;
    if (warned_.insert(cause).second) {
      REXLOG_WARN("[native] targets: {} (cause {})", text, cause);
    }
    return false;
  }

  // Rectangle covered by the copy, as in the SDK's GetResolveInfo.
  bool Rectangle(const RegistersCopy& reg, uint32_t pitch, int32_t& x0, int32_t& y0,
                  int32_t& x1, int32_t& y1) {
    const uint32_t type = reg.fetch_vertices[0] & 0x3;
    const uint32_t address = reg.fetch_vertices[0] >> 2;
    const auto order = static_cast<xenos::Endian>(reg.fetch_vertices[1] & 0x3);
    const uint32_t size = (reg.fetch_vertices[1] >> 2) & 0xFFFFFF;
    if (type != uint32_t(xenos::FetchConstantType::kVertex) || size != 3 * 2) {
      return Reject(5, "copy vertices in an unsupported format");
    }
    const uint8_t* vertices = memory_->TranslatePhysical(address * 4);
    const float mid_pixel =
        (reg.pa_su_vtx_cntl & 0x1) == uint32_t(xenos::PixelCenter::kD3DZero) ? 0.5f : 0.0f;
    int32_t fixed[6];
    for (int i = 0; i < 6; ++i) {
      float value;
      std::memcpy(&value, vertices + i * 4, sizeof(value));
      fixed[i] = Fixed16p8(xenos::GpuSwap(value, order) + mid_pixel);
    }
    x0 = (std::min({fixed[0], fixed[2], fixed[4]}) + 127) >> 8;
    y0 = (std::min({fixed[1], fixed[3], fixed[5]}) + 127) >> 8;
    x1 = (std::max({fixed[0], fixed[2], fixed[4]}) + 127) >> 8;
    y1 = (std::max({fixed[1], fixed[3], fixed[5]}) + 127) >> 8;

    const int32_t x_displacement = ExtenderSign15(reg.pa_sc_window_offset & 0x7FFF);
    const int32_t y_displacement = ExtenderSign15((reg.pa_sc_window_offset >> 16) & 0x7FFF);
    if ((reg.pa_su_sc_mode_cntl >> 16) & 0x1) {  // vtx_window_offset_enable
      x0 += x_displacement;
      y0 += y_displacement;
      x1 += x_displacement;
      y1 += y_displacement;
    }
    // Window scissor (GetScissor without clamping to the pitch).
    int32_t left = int32_t(reg.pa_sc_window_scissor_tl & 0x3FFF);
    int32_t top = int32_t((reg.pa_sc_window_scissor_tl >> 16) & 0x3FFF);
    int32_t right = int32_t(reg.pa_sc_window_scissor_br & 0x3FFF);
    int32_t bottom = int32_t((reg.pa_sc_window_scissor_br >> 16) & 0x3FFF);
    if (!((reg.pa_sc_window_scissor_tl >> 31) & 0x1)) {  // window_offset_disable
      left += x_displacement;
      top += y_displacement;
      right += x_displacement;
      bottom += y_displacement;
    }
    left = std::max(left, 0);
    top = std::max(top, 0);
    right = std::max(right, left);
    bottom = std::max(bottom, top);
    x0 = std::clamp(x0, left, right);
    y0 = std::clamp(y0, top, bottom);
    x1 = std::clamp(x1, left, right);
    y1 = std::clamp(y1, top, bottom);
    // D3D9 aligns to 8 (kResolveAlignmentPixels).
    x0 &= ~int32_t(7);
    y0 &= ~int32_t(7);
    x1 = (x1 + 7) & ~int32_t(7);
    y1 = (y1 + 7) & ~int32_t(7);
    const int32_t pitch_aligned = int32_t(pitch & ~uint32_t(7));
    x0 = std::min(x0, pitch_aligned);
    x1 = std::min(x1, pitch_aligned);
    if (x0 >= x1 || y0 >= y1) {
      return Reject(6, "empty copy rectangle");
    }
    return true;
  }

  // sa2_cpu_ring_target_memo: GetTarget / GetDepth answered from a 4-slot memo of their inputs. Both maps only ever
  // gain entries (no erase, no clear while the targets live) and an unordered_map node never moves, so the same
  // inputs always lead to the same Image. Only found or created images are memoized; rejections run the full path.
  struct TargetMemo {
    uint64_t in = ~0ull;
    Image* image = nullptr;
  };
  std::array<TargetMemo, 4> target_memo_{}, depth_memo_{};

  Image* GetTarget(uint32_t base, uint32_t format, uint32_t pitch) {
    const uint64_t memo_in = (uint64_t(msaa_edram_actual_) << 40) | (uint64_t(base & 0xFFFFF) << 20) |
                             (uint64_t(format & 0xF) << 16) | (pitch & 0xFFFF);
    TargetMemo& memo = target_memo_[(base ^ (pitch >> 4)) & 3];
    if (g_target_memo.active) {
      if (memo.in == memo_in) {
        g_target_memo.UseSampled();
        return memo.image;
      }
    }
    Image* const found = GetTargetLookup(base, format, pitch);
    if (found && g_target_memo.active && base <= 0xFFFFF && format <= 0xF && pitch <= 0xFFFF)
      memo = {memo_in, found};
    return found;
  }

  Image* GetTargetLookup(uint32_t base, uint32_t format, uint32_t pitch) {
    if (!pitch) {
      Reject(7, "render target with pitch 0");
      return nullptr;
    }
    const VkFormat host_format = HostFormatTargetColor(format);  // per format
    if (host_format == VK_FORMAT_UNDEFINED) {
      Reject(100 + format, "render target format not supported yet");
      return nullptr;
    }
    // EDRAM is typeless storage. UE3 deliberately alternates, at the same tile base, the normal and
    // "AS" variants used for blending (3 <-> 12 and 2 <-> 10). They have the same host representation;
    // keeping separate VkImages loses everything drawn through the other guest view. Key by the guest
    // storage class as well as the host format: UNORM10, 7e3 and true FP16 may all use host RGBA16F,
    // but they must be reinterpreted through physical words rather than sharing numeric texels.
    const uint8_t class_edram = ClassEdramFORMAT(format);
    const bool edram_64bpp = xenos::IsColorRenderTargetFormat64bpp(
        xenos::ColorRenderTargetFormat(format));
    const uint64_t key = (uint64_t(msaa_edram_actual_) << 44) |
                           (uint64_t(base) << 32) | (uint64_t(class_edram) << 24) |
                           (uint64_t(uint32_t(host_format) & 0xFF) << 16) | pitch;
    auto it = targets_.find(key);
    if (it != targets_.end()) {
      return &it->second;
    }
    /*
     * The height comes from the pitch, so the scene's color target measures 1280x1280 to draw 1280x720
     * and can never be swapped with its resolved texture. With a pitch of 1280 or less the game never
     * draws below 720 (measured viewport and scissor), so 720 rows would be enough and the sizes would
     * match. Above 1280 (the internal 1080p mode) the larger height is needed.
     */
    /*
     * Creating the color target 720 rows high (instead of deriving the height from the pitch), so that it
     * matches the resolved texture and the images can be swapped instead of copied, was tried: it breaks
     * the scene (the screen fills with a yellow smear). Reverted.
     */
    const uint32_t height = std::min(kMaxTargetHeight, std::max<uint32_t>(720, (pitch + 15) & ~15u));
    Image image;
    image.edram_base = uint16_t(base);
    image.edram_format = class_edram;
    image.edram_64bpp = edram_64bpp;
    image.edram_msaa_x = msaa_edram_actual_ >= 2;
    image.edram_msaa_y = msaa_edram_actual_ >= 1;
    // sa2_msaa: the game's single-sample RGBA8 targets get host samples (no storage use: the compute conversions
    // decline multisampled views and the fragment ones take their place).
    const bool multisampled = msaa_samples_ > 1 && msaa_edram_actual_ == 0 && host_format == kColorFormat;
    if (!Create(image, pitch, height,
               VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                   VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                   (multisampled ? 0u : VK_IMAGE_USAGE_STORAGE_BIT),
               host_format,
               multisampled ? VkSampleCountFlagBits(msaa_samples_) : VK_SAMPLE_COUNT_1_BIT)) {
      Reject(8, "could not create a render target");
      return nullptr;
    }
    REXLOG_INFO("[native] targets: render target base {:03X}, format {}, {}x{}", base, format, pitch, height);
    return &targets_.emplace(key, image).first->second;
  }

  struct OwnerTileEDRAM {
    Image* image = nullptr;
    uint16_t tile_local = 0;
    // content version of this physical tile (bumped whenever its owner writes it).
    uint32_t version = 0;
  };

  // Mass Effect mode-4 sync cache. On the Switch ~150 depth imports per frame (9-pass stencil path)
  // re-copied tiles a read-only view had already received: ownership only moves on a write, so every
  // read-only rebind imported the same unchanged tiles again. Each view remembers the version of each
  // physical tile it last received; an unchanged tile is not transferred again.
  uint32_t edram4_version_ = 0;
  // why each mode-4 transfer happens (report). Caller: 0 draw, 1 clear, 2 color resolve,
  // 3 depth resolve. Kind: 0 depth->depth X-grid/collapsed switch, 1 depth->depth other, 2 color->depth,
  // 3 depth->color, 4 color->color. Full = the run covers the whole target.
  uint32_t edram4_reason_ = 0;
  uint64_t edram4_overwrites_ = 0;  // draw binds that used a proven full overwrite
  uint64_t edram4_redirected_ = 0, edram4_redirected_tiles_ = 0, edram4_redirected_no_ = 0;
  std::map<std::string, uint64_t> edram4_stencil_no_reasons_, edram4_depth_no_reasons_;
  // sa2_cpu_ring_redirect_key: RedirectClearDepthEDRAM4's no() counts by (reason literal, depth info).
  std::map<std::pair<const char*, uint32_t>, uint64_t> edram4_depth_no_fast_;
  uint64_t edram4_redirected_color_tiles_ = 0;
  // Tiles a redirected clear wrote (by the view the game drew it on) and the view that next pulled them.
  // A redirected clear is identified by its view and its ordinal among that view's clears in the frame (the
  // same view is cleared several times per frame, each time read next by a different view).
  struct RedirectedStamp { uint64_t site = 0; uint64_t version = 0; };
  std::unordered_map<const Image*, uint32_t> edram4_cleared_in_frame_;
  static uint64_t SiteClear(const Image* drawn, uint32_t ordinal) {
    return (uint64_t(reinterpret_cast<uintptr_t>(drawn)) << 8) ^ ordinal;
  }
  std::array<RedirectedStamp, kTilesEDRAM4> edram4_stamps_{};
  struct Consumer { Image* view = nullptr; uint32_t confidence = 0; };
  std::unordered_map<uint64_t, Consumer> edram4_consumer_;
  uint64_t edram4_consumer_uses_ = 0;
  std::array<uint64_t, kGpuCategories> cat_intervals_{}, cat_max_ns_{}, cat_long_{};
  std::unordered_map<uint64_t, uint64_t> gpu_labels_ns_;
  int32_t dump_remaining_marks_ = 0;
  bool dump_done_marks_ = false;
  std::chrono::steady_clock::time_point dump_start_ = std::chrono::steady_clock::now();  // (category, first VS/PS) -> GPU ns
  uint64_t edram4_stencil_redirected_ = 0, edram4_stencil_redirected_tiles_ = 0, edram4_stencil_redirected_no_ = 0;
  uint64_t exhausted_marks_ = 0;
  int8_t d32s8_supported_ = -1;  // cached format support of the half-range depth host format
  uint64_t mark_max_ns_ = 0;      // longest single mark-to-mark interval since the last report
  std::string mark_max_desc_;
  uint64_t edram4_epoch_ = 1;  // bumped whenever any physical tile changes owner (fast paths above)
  uint64_t edram4_sync_fast_ = 0, edram4_pub_fast_ = 0;
  static bool ContainsRect(const VkRect2D& a, const VkRect2D& b) {  // a contains b
    return b.extent.width && b.extent.height && b.offset.x >= a.offset.x && b.offset.y >= a.offset.y &&
           int64_t(b.offset.x) + b.extent.width <= int64_t(a.offset.x) + a.extent.width &&
           int64_t(b.offset.y) + b.extent.height <= int64_t(a.offset.y) + a.extent.height;
  }
  // CPU cost of the mode-4 bookkeeping (report): calls and tiles visited by sync and publish.
  uint64_t edram4_sync_calls_ = 0, edram4_sync_visited_ = 0, edram4_pub_calls_ = 0, edram4_pub_visited_ = 0;  // GPU category marks dropped because a slot ran out (report)
  std::string edram4_draw_desc_;      // the draw being synchronized (transfer trace)
  uint32_t edram4_traces_ = 0;
  uint32_t banjo_plan_failures_ = 0;  // banjo-nx: invalid EDRAM4 plans logged
  uint64_t banjo_transfers_skipped_ = 0;  // banjo-nx: unsupported EDRAM4 transfers skipped
  std::array<std::array<uint64_t, 5>, 4> edram4_transfer_ops_{}, edram4_transfer_tiles_{};
  std::array<uint64_t, 5> edram4_transfer_full_{};
  struct ParEDRAM4 {
    uint64_t ops = 0, tiles = 0;
    std::string example;
    std::map<std::string, std::pair<uint64_t, uint64_t>> draws;  // short draw key -> ops, tiles
  };
  std::map<std::string, ParEDRAM4> edram4_transfer_pairs_;  // "src->dst" -> ops, tiles, first draw
  const SubmissionDraw* edram4_submission_ = nullptr;  // the draw being synchronized (nullptr: not a draw)
  uint32_t edram4_slot_ = 0;
  std::string DescribeDrawEDRAM4() const {
    if (!edram4_submission_ || !edram4_submission_->registers) return "-";
    const auto& p = *edram4_submission_;
    const uint32_t* r = p.registers;
    return fmt::format(
        "slot{} VS n{} PS n{} prim {} count {} dc {:08X} stencil {:08X} mode {} mask {:08X} blend0 {:08X} "
        "scissor {},{}+{}x{} surf {:08X} depth {:08X} color0 {:08X}",
        edram4_slot_, p.vs ? int(p.vs->number) : -1, p.ps ? int(p.ps->number) : -1,
        r[gr::XE_GPU_REG_VGT_DRAW_INITIATOR] & 63, r[gr::XE_GPU_REG_VGT_DRAW_INITIATOR] >> 16,
        r[gr::XE_GPU_REG_RB_DEPTHCONTROL], r[gr::XE_GPU_REG_RB_STENCILREFMASK],
        r[gr::XE_GPU_REG_RB_MODECONTROL] & 7, r[gr::XE_GPU_REG_RB_COLOR_MASK],
        r[gr::XE_GPU_REG_RB_BLENDCONTROL0], edram4_draw_area_.offset.x, edram4_draw_area_.offset.y,
        edram4_draw_area_.extent.width, edram4_draw_area_.extent.height,
        r[gr::XE_GPU_REG_RB_SURFACE_INFO], r[gr::XE_GPU_REG_RB_DEPTH_INFO], r[kDiagRegsColorInfo[0]]);
  }
  // The draw being synchronized, for the transfer trace (from edram4_submission_ / edram4_slot_ / edram4_draw_area_).
  std::string TraceDrawEDRAM4() const {
    if (!edram4_submission_ || !edram4_submission_->registers) return "-";
    const auto& p = *edram4_submission_;
    const uint32_t* r = p.registers;
    return fmt::format(
        "slot{} VS n{} PS n{} prim {} count {} dc {:08X} clip {:08X} vte {:08X} mask {:08X} blend0 {:08X} "
        "scissor {},{}+{}x{} overwrite {:X} est {}",
        edram4_slot_, p.vs ? int(p.vs->number) : -1, p.ps ? int(p.ps->number) : -1,
        r[gr::XE_GPU_REG_VGT_DRAW_INITIATOR] & 63, r[gr::XE_GPU_REG_VGT_DRAW_INITIATOR] >> 16,
        r[gr::XE_GPU_REG_RB_DEPTHCONTROL], r[gr::XE_GPU_REG_PA_CL_CLIP_CNTL], r[gr::XE_GPU_REG_PA_CL_VTE_CNTL],
        r[gr::XE_GPU_REG_RB_COLOR_MASK], r[gr::XE_GPU_REG_RB_BLENDCONTROL0], edram4_draw_area_.offset.x,
        edram4_draw_area_.offset.y, edram4_draw_area_.extent.width, edram4_draw_area_.extent.height,
        p.edram_overwrite_slots, p.edram_used_height_estimate ? int(*p.edram_used_height_estimate) : -1);
  }
  std::unordered_map<const Image*, std::vector<uint32_t>> edram4_views_;
  uint64_t edram4_sync_skipped_ = 0, edram4_sync_copied_ = 0;
  bool SynchronizedEDRAM4(const Image& view, uint32_t physical, uint32_t version) const {
    if (!version || !REXCVAR_GET(sa2_native_edram4_cache_sync)) return false;
    const auto it = edram4_views_.find(&view);
    return it != edram4_views_.end() && it->second[physical & kMaskEDRAM4] == version;
  }
  void NoteSynchronizedEDRAM4(const Image& view, uint32_t physical, uint32_t version) {
    auto& versions = edram4_views_[&view];
    if (versions.empty()) versions.assign(kTilesEDRAM4, 0);
    versions[physical & kMaskEDRAM4] = version;
  }
  void ForgetSynchronizedEDRAM4(const Image& view) {
    edram4_views_.erase(&view);
    edram4_alias_1x_.erase(&view);
    for (auto it = edram4_consumer_.begin(); it != edram4_consumer_.end();)
      it = it->second.view == &view ? edram4_consumer_.erase(it) : std::next(it);
    edram4_stencil_source_.erase(&view);
    for (auto& [v, sources] : edram4_stencil_source_)
      for (auto& f : sources) if (f.image == &view) f = {};
    for (auto it = edram4_alias_1x_.begin(); it != edram4_alias_1x_.end();)
      it = it->second == &view ? edram4_alias_1x_.erase(it) : std::next(it);
  }
  // A 4x MSAA depth view -> the 1x depth view with the same EDRAM base and tile pitch that last took its
  // tiles (the same bytes seen per sample: Eden Prime's 440x720 4x clear alias of the 880x880 shadow map).
  std::unordered_map<const Image*, Image*> edram4_alias_1x_;
  // One sync's transfers recorded as a batch (sa2_native_edram4_batch): their destination tiles are
  // disjoint and sources are only read, so they need one barrier pair / one import pass, not one each.
  struct SpanEDRAM4 {
    Image* source;
    uint32_t tile_source, tile, count;
    bool native_msaa_import;
    // usf4_edram4_import_columns: a merged column. [tile, +count) is then the linear range from the first run to
    // the end of the last one, rect the column (only its tiles are drawn) and runs the original (tile, count) runs.
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    VkRect2D rect{};
  };
  uint64_t edram4_columns_merged_ = 0, edram4_columns_runs_ = 0, edram4_columns_drawn_ = 0, edram4_columns_failed_ = 0;
  struct CopyStencilPending {
    Image* source;
    Image* target;
    uint32_t source_start, target_start, count;
    VkRect2D scissor;
  };
  struct BatchEDRAM4 {
    bool active = false;
    bool compute_pending = false;  // an "after" barrier of compute transfers is owed
    bool import_open = false;
    VkRenderPass import_pass = VK_NULL_HANDLE;
    VkFramebuffer import_fb = VK_NULL_HANDLE;
    bool has_area = false;
    VkRect2D area_import{};
    std::vector<CopyStencilPending> copies_stencil;
  } edram4_batch_;
  uint64_t edram4_batch_spans_ = 0, edram4_batch_saved_passes_ = 0, edram4_batch_saved_barriers_ = 0;
  // Lazy stencil import (no VK_EXT_shader_stencil_export on the Switch: stencil costs 8 extra passes). A depth
  // view that never had a stencil-enabled draw receives depth only; per tile it records where the real
  // stencil stayed (image, tile). Importing back into that image/tile keeps the destination's own stencil
  // (exactly what the Xbox does: a depth-only pass leaves the stencil bits of the EDRAM words untouched).
  struct SourceStencil { const Image* image = nullptr; uint16_t tile = 0; };
  std::unordered_map<const Image*, std::vector<SourceStencil>> edram4_stencil_source_;
  // Every write of a deferred stencil source goes through here, so sa2_cpu_ring_edram_memo sees new pending work.
  uint64_t edram4_stencil_sources_serial_ = 0;
  std::vector<SourceStencil>& StencilSourcesWrite(const Image* view) {
    ++edram4_stencil_sources_serial_;
    return edram4_stencil_source_[view];
  }

  // sa2_cpu_ring_edram_memo: the previous draw's PrepareDrawEDRAM4 inputs and outputs.
  struct EdramDrawMemo {
    bool valid = false;
    std::array<uint32_t, 14> key{};    // the registers PrepareDrawEDRAM4 reads (EdramMemoKey)
    bool skipped_mode = false;         // RB_MODECONTROL not 4/5: the full path returns before any EDRAM work
    bool ps_null = true;
    uint32_t ps_outputs = 0;           // p.ps->outputs & 15: which color slots the pixel shader writes
    int64_t used_height = -1;          // p.edram_used_height_estimate, -1 = none
    uint64_t stencil_serial = 0;       // edram4_stencil_sources_serial_ after the full path ran
    std::array<Image*, 5> images{};
    std::array<bool, 5> writes{};
    bool writes_stencil = false;
    bool depth_raster_grid_eligible = false;
    VkRect2D area{};
    me::native::EdramBoundPlan plan{};
    uint32_t last_slot = 0;            // edram4_slot_ after the full path (the last active slot)
  };
  EdramDrawMemo edram4_memo_;
  bool edram4_memo_on_ = true;         // false for the session after a DIFFERENCE
  uint32_t edram4_memo_checks_left_ = 4096;  // the first hits also run the full path and compare
  uint64_t edram4_memo_hits_ = 0, edram4_memo_misses_ = 0, edram4_memo_ineligible_ = 0, edram4_memo_checked_ = 0;
  bool edram4_published_no_stencil_ = false;
  std::vector<VkRect2D> edram4_cuts_extra_;  // more proven-overwrite rects of the current draw (raster)
  bool edram4_import_stencil_only_ = false;  // ImportColorDepthEDRAM4: fetch only the deferred stencil
  uint64_t edram4_stencil_fetched_ = 0, edram4_stencil_fetched_tiles_ = 0;
  uint64_t edram4_bits_skipped_ = 0;
  std::optional<VkRect2D> edram4_stencil_replacement_;  // raster rect whose stencil the current draw replaces
  uint64_t edram4_stencil_replaced_ = 0;
  std::map<std::string, std::pair<uint64_t, uint64_t>> edram4_import9_pairs_;  // 9-pass imports: ops, tiles
  bool edram4_draw_writes_stencil_ = false;  // the draw being prepared writes stencil (slot 0)  // the depth publish in progress leaves the stencil bits alone
  uint64_t edram4_stencil_inherited_ = 0;
  uint64_t edram4_stencil_skipped_ = 0, edram4_stencil_returned_ = 0, edram4_stencil_inexact_ = 0,
           edram4_stencil_late_ = 0, edram4_stencil_exported_ = 0;
  bool StencilDeferredEDRAM4(const Image& view, uint32_t start, uint32_t count) const {
    const auto it = edram4_stencil_source_.find(&view);
    if (it == edram4_stencil_source_.end()) return false;
    for (uint32_t i = 0; i < count; ++i) if (it->second[(start + i) & kMaskEDRAM4].image) return true;
    return false;
  }
  void StencilRealEDRAM4(const Image& view, uint32_t start, uint32_t count) {
    const auto it = edram4_stencil_source_.find(&view);
    if (it == edram4_stencil_source_.end()) return;
    for (uint32_t i = 0; i < count; ++i) it->second[(start + i) & kMaskEDRAM4] = {};
  }

  struct ViewsDepthEDRAM {
    VkImageView depth = VK_NULL_HANDLE;
    VkImageView stencil = VK_NULL_HANDLE;
  };

  struct PassImportDepthEDRAM {
    VkRenderPass render_pass = VK_NULL_HANDLE;
    std::array<VkPipeline, 6> pipelines{}; // numeric color / depth / raw64 sources, each initial / stencil-bit
  };

  // sa2_native_conversion_frag: render pass + pipeline per destination host format.
  struct PassConversionColorFrag {
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };

  bool FailureEDRAM4(const char* reason, const Image& target, const Image* source = nullptr) {
    edram4_operation_failed_ = true;
    const auto class_value = [](const Image& image) {
      return uint32_t(image.edram_format) | (uint32_t(image.edram_depth) << 8) |
             (uint32_t(image.edram_64bpp) << 9) | (uint32_t(image.edram_msaa_x) << 10) |
             (uint32_t(image.edram_msaa_y) << 11);
    };
    const std::string key = std::string(reason) + "/" + std::to_string(class_value(target)) + "/" +
                            std::to_string(source ? class_value(*source) : UINT32_MAX);
    auto found = edram4_rejections_per_class_.find(key);
    if (found == edram4_rejections_per_class_.end()) {
      if (edram4_rejections_per_class_.size() >= 128) {
        if (!edram4_log_cap_) {
          edram4_log_cap_ = true;
          REXLOG_WARN("[native] EDRAM mode4 reject log capped at 128 distinct contracts; total failures still counted");
        }
        return false;
      }
      found = edram4_rejections_per_class_.emplace(key, 0).first;
    }
    if (++found->second <= 3) {
      REXLOG_WARN("[native] EDRAM mode4 REJECT {}: dest {:03X}/{} depth {} source {:03X}/{} depth {} "
                  "(diagnostic transfers, NOT complete emulation)",
                  reason, target.edram_base, target.edram_format, target.edram_depth,
                  source ? source->edram_base : 0u, source ? source->edram_format : 0u,
                  source ? source->edram_depth : false);
    }
    return false;
  }

  bool TileUsedEDRAM4(const Image& image, uint32_t tile, const VkRect2D& area,
                       bool* covered_whole = nullptr) const {
    const uint32_t pitch_tiles = PitchTilesEDRAM(image);
    if (!pitch_tiles || !area.extent.width || !area.extent.height) return false;
    const uint32_t tile_width = 80u >> (uint32_t(image.edram_64bpp) + image.edram_msaa_x);
    const uint32_t tile_height = 16u >> image.edram_msaa_y;
    const uint32_t x = (tile % pitch_tiles) * tile_width;
    const uint32_t y = (tile / pitch_tiles) * tile_height;
    const int64_t x0 = std::max<int64_t>(0, area.offset.x);
    const int64_t y0 = std::max<int64_t>(0, area.offset.y);
    const int64_t x1 = int64_t(area.offset.x) + area.extent.width;
    const int64_t y1 = int64_t(area.offset.y) + area.extent.height;
    if (covered_whole)
      *covered_whole = x >= x0 && y >= y0 && x + tile_width <= x1 && y + tile_height <= y1 &&
                          x + tile_width <= image.width && y + tile_height <= image.height;
    return x < image.width && y < image.height && x < x1 && y < y1 &&
           int64_t(x + tile_width) > x0 && int64_t(y + tile_height) > y0;
  }

  // The tile rectangle TileUsedEDRAM4 accepts for `area` (columns tx0..tx1, rows ty0..ty1, exclusive):
  // the mode-4 loops visit only these instead of testing every tile of the view (CPU cost per draw on
  // the Switch's A57: the full scan was most of the ring thread).
  struct RangeTiles4 { uint32_t pitch = 0, tx0 = 0, tx1 = 0, ty0 = 0, ty1 = 0; };
  static bool RangeTilesEDRAM4(const Image& image, const VkRect2D& area, RangeTiles4& r) {
    r.pitch = PitchTilesEDRAM(image);
    if (!r.pitch || !area.extent.width || !area.extent.height) return false;
    const uint32_t tile_width = 80u >> (uint32_t(image.edram_64bpp) + image.edram_msaa_x);
    const uint32_t tile_height = 16u >> image.edram_msaa_y;
    const int64_t x0 = std::max<int64_t>(0, area.offset.x), y0 = std::max<int64_t>(0, area.offset.y);
    const int64_t x1 = std::min<int64_t>(int64_t(area.offset.x) + area.extent.width, image.width);
    const int64_t y1 = std::min<int64_t>(int64_t(area.offset.y) + area.extent.height, image.height);
    if (x1 <= x0 || y1 <= y0) return false;
    r.tx0 = uint32_t(x0 / tile_width);
    r.tx1 = std::min<uint32_t>(r.pitch, uint32_t((x1 + tile_width - 1) / tile_width));
    r.ty0 = uint32_t(y0 / tile_height);
    r.ty1 = uint32_t((y1 + tile_height - 1) / tile_height);
    return r.tx1 > r.tx0;
  }

  bool ValidateSimultaneousClearEDRAM4(const RegistersCopy& reg) {
    me::native::EdramClearPlan plan;
    VkRect2D area{};
    if (!GetPlanClearedEDRAM4(reg, plan, area)) return false;
    if (!plan.depth.length && !plan.color.length) return true;
    const uint32_t control = reg.rb_copy_control;
    const uint32_t source = control & 7u;
    const uint32_t pitch = reg.rb_surface_info & 0x3FFF;
    const uint32_t msaa = (reg.rb_surface_info >> 16) & 3u;
    Image color{}, depth{};
    const uint32_t color_info = source < xenos::kMaxColorRenderTargets ? reg.rb_color_info[source] : 0;
    color.edram_base = color_info & 0xFFF;
    color.edram_format = (color_info >> 16) & 15u;
    color.edram_64bpp = xenos::IsColorRenderTargetFormat64bpp(
        xenos::ColorRenderTargetFormat(color.edram_format));
    depth.edram_base = reg.rb_depth_info & 0xFFF;
    depth.edram_format = (reg.rb_depth_info >> 16) & 1u;
    depth.edram_depth = true;
    // Match the actual unscaled native allocations, not an assumed 720px frame.
    color.width = depth.width = pitch;
    color.height = depth.height = std::min(kMaxTargetHeight,
        std::max<uint32_t>(720, (pitch + 15) & ~15u));
    color.edram_msaa_x = depth.edram_msaa_x = msaa >= 2;
    color.edram_msaa_y = depth.edram_msaa_y = msaa >= 1;
    const me::native::EdramPixelRectangle cutout{uint32_t(area.offset.x), uint32_t(area.offset.y),
        area.extent.width, area.extent.height};
    if (!me::native::PlanEdramClearRectangles(plan.color, PitchTilesEDRAM(color), msaa,
            color.edram_64bpp, color.width, color.height, cutout).valid ||
        !me::native::PlanEdramClearRectangles(plan.depth, PitchTilesEDRAM(depth), msaa,
            false, depth.width, depth.height, cutout).valid)
      return FailureEDRAM4("SDK clear span lacks complete physical backing", color, &depth);
    return true;
  }

  bool GetPlanClearedEDRAM4(const RegistersCopy& reg, me::native::EdramClearPlan& plan,
                                 VkRect2D& area) {
    const uint32_t control = reg.rb_copy_control, source = control & 7u;
    const bool color = source < xenos::kMaxColorRenderTargets && (control & (1u << 8));
    const bool depth = (control & (1u << 9)) && clear_depth_;
    if (!color && !depth) { plan.valid = true; return true; }
    const uint32_t pitch = reg.rb_surface_info & 0x3FFF;
    const uint32_t msaa = (reg.rb_surface_info >> 16) & 3u;
    Image diagnostic{};
    if (!pitch || pitch > 8192 || msaa > 2)
      return FailureEDRAM4("invalid clear physical layout", diagnostic);
    int32_t x0, y0, x1, y1;
    if (!Rectangle(reg, pitch, x0, y0, x1, y1))
      return FailureEDRAM4("clear rectangle unavailable", diagnostic);
    // Rectangle returns the original-view rectangle, including window offsets
    // and 8px resolve alignment. Unlike ResolveInfo's offset base, this code
    // never rebases the EDRAM source, so no base-offset rows must be added again.
    const uint32_t height = std::min(kMaxTargetHeight,
        std::max<uint32_t>(720, (pitch + 15) & ~15u));
    const auto rectangle = me::native::ClipClearRectangle(x0, y0, x1, y1, pitch, height);
    area = {{rectangle.x, rectangle.y}, {rectangle.width, rectangle.height}};
    me::native::EdramClearRequest request;
    request.depth_base = reg.rb_depth_info & 0xFFF;
    request.color_base = color ? reg.rb_color_info[source] & 0xFFF : 0;
    request.pitch_tiles_at_32bpp = me::native::EdramLayout(msaa, false).pitch_tiles(pitch);
    request.msaa = msaa;
    request.clear_depth = depth;
    request.clear_color = color;
    request.color_wide = color && xenos::IsColorRenderTargetFormat64bpp(
        xenos::ColorRenderTargetFormat((reg.rb_color_info[source] >> 16) & 15));
    request.x_pixels = uint32_t(rectangle.x);
    request.y_pixels = uint32_t(rectangle.y);
    request.width_pixels = rectangle.width;
    request.height_pixels = rectangle.height;
    plan = me::native::PlanEdramClearRanges(request);
    if (!plan.valid) return FailureEDRAM4("invalid SDK clear range", diagnostic);
    return true;
  }

  bool PrepareRangeClearEDRAM4(Image& image, me::native::EdramRelativeSpan span,
      const VkRect2D& cutout, me::native::EdramClearRectangles& rectangles) {
    if (image.guest_width || image.guest_height)
      return FailureEDRAM4("scaled clear has no physical backing", image);
    const VkRect2D host_cutout = RasterArea(image, cutout);
    rectangles = me::native::PlanEdramClearRectangles(span, PitchTilesEDRAM(image),
        image.edram_msaa_x ? 2u : image.edram_msaa_y, image.edram_64bpp, image.width, image.height,
        {uint32_t(host_cutout.offset.x), uint32_t(host_cutout.offset.y), host_cutout.extent.width, host_cutout.extent.height});
    if (!rectangles.valid) return FailureEDRAM4("clear span has missing tile padding", image);
    if (!span.length) return true;
    const VkRect2D backing{{0, 0}, {image.width, image.height}};
    // Import every contiguous SDK tile, including row gaps and partial-tile
    // borders. Fully overwritten tiles alone can skip their former contents.
    edram4_reason_ = 1;
    return SynchronizeEDRAM4(image, backing, span.end(), true, span.start, &host_cutout);
  }

  void PublishRangeClearEDRAM4(Image& image, me::native::EdramRelativeSpan span) {
    for (uint32_t tile = span.start; tile < span.end(); ++tile) {
      const uint32_t physical = (image.edram_base + tile) & kMaskEDRAM4;
      if (physical == TraceTileEDRAM4() && TraceEventEDRAM4(physical)) {
        const auto old = owners_tiles_edram4_[physical];
        REXLOG_INFO("[native] EDRAM TILE TRACE clear-publish frame={} draw={} physical={} "
                    "before=[{}] after=[{}] range={}:{}",
                    trace_frame_, draws_ ? draws_->Drawn() : 0,
                    physical, TraceImageEDRAM4(old.image, old.tile_local),
                    TraceImageEDRAM4(&image, tile), span.start, span.end());
      }
      owners_tiles_edram4_[(image.edram_base + tile) & kMaskEDRAM4] = {&image, uint16_t(tile), ++edram4_version_};
    }
    ++edram4_epoch_;
    // An EDRAM clear writes the whole 32-bit word: the stencil is real (known) again.
    if (image.edram_depth) StencilRealEDRAM4(image, span.start, span.end() - span.start);
  }

  void BarrierClearEDRAM4() {
    // A rectangle loadOp=CLEAR writes attachment stages, not TRANSFER. Include
    // both sides' reads as clear destinations may have been sampled earlier.
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = barrier.dstAccessMask =
        VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  // Called several times per draw. With no tile selected (the normal case) every path below returns UINT32_MAX, so once
  // the first call has found that out, the rest return without the function-static guard, the cvar string or the clock.
  uint32_t TraceTileEDRAM4() const {
    if (trace_tile_none_ && trace_tile_fast_) return UINT32_MAX;
    return TraceTileEDRAM4Slow();
  }
  const bool trace_tile_fast_ = CompareActiveFast();
  uint64_t trace_frame_ = 0;  // Swap interval ordinal, for the EDRAM tile trace
  mutable bool trace_tile_none_ = false;

  uint32_t TraceTileEDRAM4Slow() const {
    static const uint32_t selected = [] {
      static const std::string from_cvar = [] {
        std::string v = REXCVAR_GET(sa2_diag_trace_tile);
        std::erase(v, '"');
        REXLOG_INFO("[native] EDRAM tile trace cvar: '{}'", v);
        return v;
      }();
      const char* value = from_cvar.c_str();
      if (!value || !*value) return UINT32_MAX;
      uint32_t tile = 0;
      for (const char* p = value; *p; ++p) {
        if (*p < '0' || *p > '9' || tile > kMaskEDRAM4 / 10u) return UINT32_MAX;
        tile = tile * 10u + uint32_t(*p - '0');
        if (tile >= kTilesEDRAM4) return UINT32_MAX;
      }
      return tile;
    }();
    if (selected == UINT32_MAX) {
      trace_tile_none_ = true;
      if (trace_tile_fast_) return UINT32_MAX;
    }
    // Tile chosen by sa2_diag_trace_tile: traced from sa2_diag_trace_from_s after start.
    if (selected == UINT32_MAX || REXCVAR_GET(sa2_diag_trace_tile).empty()) return UINT32_MAX;
    static const auto start = std::chrono::steady_clock::now();
    if (std::chrono::steady_clock::now() - start <
        std::chrono::seconds(REXCVAR_GET(sa2_diag_trace_from_s))) return UINT32_MAX;
    static const bool warned = [&] {
      REXLOG_INFO("[native] EDRAM tile trace active: physical tile {}", selected);
      return true;
    }();
    (void)warned;
    return selected;  // TraceEventEDRAM4 caps the events (1024)
  }

  bool TraceEventEDRAM4(uint32_t tile) const {
    if (tile == UINT32_MAX) return false;
    // Diagnostic-only, ring-thread state: at most 1024 events per Swap interval.
    static uint64_t frame = UINT64_MAX;
    static uint32_t events = 0;
    if (frame != trace_frame_) {
      frame = trace_frame_;
      events = 0;
    }
    if (events++ < 1024) return true;
    if (events == 1025)
      REXLOG_WARN("[native] EDRAM TILE TRACE truncated frame={} physical={} cap=1024",
                  frame, tile);
    return false;
  }

  std::string TraceImageEDRAM4(const Image* image, uint32_t local) const {
    if (!image) return "none";
    return fmt::format("token={:X} vkimage={:X} base={:03X} guestfmt={} hostfmt={} depth={} pitchtiles={} "
                       "extent={}x{} msaaXY={},{} gridX={} half={} wide={} local={}",
                       reinterpret_cast<uintptr_t>(image), (uint64_t)image->image, image->edram_base,
                       uint32_t(image->edram_format), uint32_t(image->format),
                       image->edram_depth, PitchTilesEDRAM(*image),
                       image->width, image->height, image->edram_msaa_x, image->edram_msaa_y,
                       image->raster_grid_x, image->depth_float24_half, image->edram_64bpp, local);
  }

  bool DrawTouchesTraceTileEDRAM4() const {
    const uint32_t physical = TraceTileEDRAM4();
    if (physical == UINT32_MAX) return false;
    for (uint32_t slot = 0; slot < 5; ++slot) {
      const Image* image = edram4_draw_images_[slot];
      if (!image) continue;
      const uint32_t local = (physical - image->edram_base) & kMaskEDRAM4;
      if (local < edram4_draw_plan_.length_tiles[slot] &&
          TileUsedEDRAM4(*image, local, RasterArea(*image, edram4_draw_area_))) return true;
    }
    return false;
  }

  // keep_version: the view only received the owner's current contents (a load on bind, no write).
  // The tile keeps its version, and the previous owner is recorded as still holding it, so binding the
  // previous owner again does not import the same unchanged tiles back (Mass Effect: depth exported to a
  // k_16_16 color view for reading, then a full 9-pass import back every frame).
  void PublishEDRAM4(Image& image, const VkRect2D& area, uint32_t limit = 2048,
                      bool keep_version = false) {
    const uint32_t rows = (image.height + (16u >> image.edram_msaa_y) - 1) /
                           (16u >> image.edram_msaa_y);
    const uint32_t tiles = std::min({PitchTilesEDRAM(image) * rows, limit, 2048u});
    const uint32_t trace_physical = TraceTileEDRAM4();
    RangeTiles4 range;
    ++edram4_pub_calls_;
    // Fast path: all tiles of the area already ours and nobody cached their versions: nothing changes.
    if (REXCVAR_GET(sa2_native_edram4_fast) && !keep_version &&
        image.edram4_epoch == edram4_epoch_ && !image.edram4_exported &&
        ContainsRect(image.edram4_own, area)) {
      ++edram4_pub_fast_;
      return;
    }
    if (!RangeTilesEDRAM4(image, area, range)) return;
    bool change_owner = false;
    edram4_pub_visited_ += uint64_t(range.tx1 - range.tx0) * (range.ty1 - range.ty0);
    for (uint32_t ty = range.ty0; ty < range.ty1 && ty * range.pitch < tiles; ++ty)
    for (uint32_t tile = ty * range.pitch + range.tx0; tile < std::min(ty * range.pitch + range.tx1, tiles); ++tile) {
      {
        const uint32_t physical = (image.edram_base + tile) & kMaskEDRAM4;
        const auto old = owners_tiles_edram4_[physical];
        if (physical == trace_physical &&
            (old.image != &image || old.tile_local != tile) && TraceEventEDRAM4(physical)) {
          REXLOG_INFO("[native] EDRAM TILE TRACE publish frame={} draw={} physical={} "
                      "before=[{}] after=[{}] area={},{}+{}x{} limit={}",
                      trace_frame_, draws_ ? draws_->Drawn() : 0,
                      physical, TraceImageEDRAM4(old.image, old.tile_local),
                      TraceImageEDRAM4(&image, tile), area.offset.x, area.offset.y,
                      area.extent.width, area.extent.height, limit);
        }
        if (old.image != &image || old.tile_local != tile) {
          change_owner = true;
          // Lazy stencil: a depth draw that does not write stencil leaves the EDRAM stencil bits as they were,
          // so this view's stencil for the tile is wherever the previous owner's was.
          if (image.edram_depth && REXCVAR_GET(sa2_native_edram4_stencil_lazy)) {
            if (edram4_published_no_stencil_ && old.image &&
                (old.image->edram_depth ||
                 (!old.image->edram_64bpp && (old.image->edram_format == 0 || old.image->edram_format == 2 ||
                                               old.image->edram_format == 3)))) {
              const auto previous = edram4_stencil_source_.find(old.image);
              const SourceStencil f = previous != edram4_stencil_source_.end() && previous->second[old.tile_local & kMaskEDRAM4].image
                  ? previous->second[old.tile_local & kMaskEDRAM4] : SourceStencil{old.image, uint16_t(old.tile_local & kMaskEDRAM4)};
              auto& sources = StencilSourcesWrite(&image);
              if (sources.empty()) sources.resize(kTilesEDRAM4);
              sources[tile & kMaskEDRAM4] = f.image == &image && f.tile == (tile & kMaskEDRAM4) ? SourceStencil{} : f;
              ++edram4_stencil_inherited_;
            } else {
              StencilRealEDRAM4(image, tile, 1);
            }
          }
        }
        if (keep_version && old.version && REXCVAR_GET(sa2_native_edram4_cache_sync)) {
          if (old.image && old.image != &image) {
            NoteSynchronizedEDRAM4(*old.image, physical, old.version);
            image.edram4_exported = true;  // the old owner holds this version of our tile
          }
          NoteSynchronizedEDRAM4(image, physical, old.version);
          owners_tiles_edram4_[physical] = {&image, uint16_t(tile), old.version};
        } else {
          owners_tiles_edram4_[physical] = {&image, uint16_t(tile), ++edram4_version_};
        }
      }
    }
    if (change_owner) ++edram4_epoch_;
    // Every tile of the area is ours now (when the tile limit did not cut it).
    if (range.ty1 * range.pitch <= tiles) {
      const VkRect2D own{{std::max(0, area.offset.x), std::max(0, area.offset.y)},
                            {uint32_t(std::min<int64_t>(int64_t(area.offset.x) + area.extent.width, image.width) -
                                      std::max(0, area.offset.x)),
                             uint32_t(std::min<int64_t>(int64_t(area.offset.y) + area.extent.height, image.height) -
                                      std::max(0, area.offset.y))}};
      const bool previous_covers = image.edram4_epoch != edram4_epoch_ || ContainsRect(own, image.edram4_own);
      if (!keep_version && previous_covers) image.edram4_exported = false;  // all our cached tiles re-versioned
      if (image.edram4_epoch != edram4_epoch_ || ContainsRect(own, image.edram4_own)) {
        image.edram4_own = own;
        image.edram4_epoch = edram4_epoch_;
      }
    }
  }

  // A fully proven constant stencil clear must not make its collapsed depth
  // copy authoritative. Clear just S8 in each existing canonical depth owner,
  // preserving every D32 bit and its physical ownership. This deliberately
  // does NOT handle conditional/partial stencil draws or color-word owners.
  bool PublishStencilClearEDRAM4(Image& drawn,
      const me::native::StencilClearOwnershipProof& proof, uint32_t limit) {
    if (!proof.eligible || !drawn.edram_depth || drawn.guest_width || drawn.guest_height ||
        proof.length_tiles != limit || !limit || limit > 2048 ||
        PitchTilesEDRAM(drawn) != proof.columns_tiles)
      return FailureEDRAM4("canonical stencil clear invalid physical proof", drawn);
    struct Region { Image* owner; uint32_t local, first, count; VkRect2D area; };
    std::vector<Region> regions;
    // Validate ALL old owners and complete tile backing before recording any
    // canonical write. New/unowned tiles may be published from the drawn view.
    for (uint32_t tile = 0; tile < limit;) {
      const auto owner = owners_tiles_edram4_[(drawn.edram_base + tile) & kMaskEDRAM4];
      if (!owner.image || owner.image == &drawn) { ++tile; continue; }
      Image& canonical = *owner.image;
      if (!canonical.edram_depth || canonical.edram_64bpp ||
          canonical.guest_width || canonical.guest_height || canonical.image == drawn.image ||
          !canonical.prepared)
        return FailureEDRAM4("canonical stencil clear needs an independent depth owner", drawn, &canonical);
      const uint32_t pitch = PitchTilesEDRAM(canonical);
      const uint32_t width = 80u >> canonical.edram_msaa_x;
      const uint32_t height = 16u >> canonical.edram_msaa_y;
      if (!pitch || canonical.width % width)
        return FailureEDRAM4("canonical stencil clear lacks physical padding", drawn, &canonical);
      const uint32_t x = (owner.tile_local % pitch) * width;
      const uint32_t y = (owner.tile_local / pitch) * height;
      if (x >= canonical.width || y >= canonical.height ||
          width > canonical.width - x || height > canonical.height - y)
        return FailureEDRAM4("canonical stencil clear exceeds depth backing", drawn, &canonical);
      uint32_t count = 1;
      while (tile + count < limit && (owner.tile_local % pitch) + count < pitch) {
        const auto next = owners_tiles_edram4_[(drawn.edram_base + tile + count) & kMaskEDRAM4];
        if (next.image != &canonical || next.tile_local != owner.tile_local + count) break;
        ++count;
      }
      if (uint64_t(width) * count > canonical.width - x)
        return FailureEDRAM4("canonical stencil clear row exceeds depth backing", drawn, &canonical);
      regions.push_back({&canonical, owner.tile_local, tile, count,
                        {{int32_t(x), int32_t(y)}, {width * count, height}}});
      tile += count;
    }
    if (!regions.empty()) {
      if (!draws_) return FailureEDRAM4("canonical stencil clear needs attachment commands", drawn);
      draws_->FinishPass();
      if (!Record()) return false;
      for (const auto& region : regions) {
        Image& canonical = *region.owner;
        Prepare(canonical);
        if (!canonical.prepared || !RestoreContent(canonical)) return false;
        ForgetClear(canonical);
        BarrierClearEDRAM4();
        canonical.edram4_bits_stencil |= uint8_t(proof.stencil_reference);
        StencilRealEDRAM4(canonical, region.local, region.count);
        if (!draws_->ClearStencilInPass(commands_work_, canonical,
                                           proof.stencil_reference, region.area))
          return FailureEDRAM4("canonical stencil-only attachment clear failed", drawn, &canonical);
        BarrierClearEDRAM4();
        const uint32_t physical = TraceTileEDRAM4();
        const uint32_t local = (physical - drawn.edram_base) & kMaskEDRAM4;
        if (physical != UINT32_MAX && local >= region.first &&
            local - region.first < region.count && TraceEventEDRAM4(physical))
          REXLOG_INFO("[native] EDRAM TILE TRACE stencil-only-preserve frame={} draw={} physical={} "
                      "drawn=[{}] canonical=[{}] stencil={} area={},{}+{}x{}",
                      trace_frame_, draws_->Drawn(), physical,
                      TraceImageEDRAM4(&drawn, local),
                      TraceImageEDRAM4(&canonical, region.local + local - region.first),
                      proof.stencil_reference, region.area.offset.x, region.area.offset.y,
                      region.area.extent.width, region.area.extent.height);
      }
    }
    StencilRealEDRAM4(drawn, 0, limit);
    for (uint32_t tile = 0; tile < limit; ++tile) {
      auto& owner = owners_tiles_edram4_[(drawn.edram_base + tile) & kMaskEDRAM4];
      if (!owner.image) { owner = {&drawn, uint16_t(tile)}; ++edram4_epoch_; }
      owner.version = ++edram4_version_;  // the owner's stencil changed
    }
    if (++edram4_stencil_preserves_ <= 16)
      REXLOG_INFO("[native] canonical stencil-only clear preserved depth: draw={} base={:03X} "
                  "tiles={} canonical row regions={} reference={}",
                  draws_ ? draws_->Drawn() : 0, drawn.edram_base, limit,
                  regions.size(), proof.stencil_reference);
    return true;
  }

  bool SynchronizeEDRAM4(Image& target, const VkRect2D& area, uint32_t limit = 2048,
                         bool clear_overwrite = false, uint32_t start = 0,
                         const VkRect2D* clear_cutout = nullptr) {
    if (target.guest_width || target.guest_height)
      return FailureEDRAM4("scaled depth has no bit-preserving physical view", target);
    if (!SamplesSupported(target) && !IsDepthNative2xEDRAM4(target))  // sa2_msaa: the host multisampled targets
      return FailureEDRAM4("unsupported hardware multisample physical destination", target);
    // Never restore a swapped image AFTER physical transfers: it would overwrite imported tiles.
    Prepare(target);
    if (!target.prepared) return FailureEDRAM4("destination was not prepared", target);
    if (!RestoreContent(target)) return false;
    const uint32_t rows = (target.height + (16u >> target.edram_msaa_y) - 1) /
                           (16u >> target.edram_msaa_y);
    const uint32_t tiles = std::min({PitchTilesEDRAM(target) * rows, limit, 2048u});
    const uint32_t trace_physical = TraceTileEDRAM4();
    const uint32_t trace_local = (trace_physical - target.edram_base) & kMaskEDRAM4;
    if (trace_physical != UINT32_MAX && trace_local >= start && trace_local < tiles &&
        owners_tiles_edram4_[trace_physical].image != &target &&
        TileUsedEDRAM4(target, trace_local, area) && TraceEventEDRAM4(trace_physical)) {
      const auto owner = owners_tiles_edram4_[trace_physical];
      bool entire = false;
      TileUsedEDRAM4(target, trace_local, clear_cutout ? *clear_cutout : area, &entire);
      if (owner.image != &target)
        REXLOG_INFO("[native] EDRAM TILE TRACE sync-bind frame={} draw={} physical={} "
                  "target=[{}] owner=[{}] same={} clear_overwrite={} entire={} "
                  "area={},{}+{}x{} range={}:{}",
                  trace_frame_, draws_ ? draws_->Drawn() : 0,
                  trace_physical, TraceImageEDRAM4(&target, trace_local),
                  TraceImageEDRAM4(owner.image, owner.tile_local), owner.image == &target,
                  clear_overwrite, entire, area.offset.x, area.offset.y,
                  area.extent.width, area.extent.height, start, tiles);
    }
    if (REXCVAR_GET(sa2_native_edram4_fast) && target.edram4_epoch == edram4_epoch_ &&
        ContainsRect(target.edram4_own, area)) {
      ++edram4_sync_fast_;
      return true;  // every tile of the area is ours: nothing to transfer (the slow loop would skip them all)
    }
    RangeTiles4 range;
    const bool has_range = RangeTilesEDRAM4(target, area, range);
    ++edram4_sync_calls_;
    if (has_range) edram4_sync_visited_ += uint64_t(range.tx1 - range.tx0) * (range.ty1 - range.ty0);
    const bool cache_sync = REXCVAR_GET(sa2_native_edram4_cache_sync);
    const std::vector<uint32_t>* versions_target = nullptr;
    const auto synchronized = [&](uint32_t physical, uint32_t version) {
      if (!version || !cache_sync) return false;
      if (!versions_target) {
        const auto it = edram4_views_.find(&target);
        if (it == edram4_views_.end()) return false;
        versions_target = &it->second;  // node-based map: stays valid while entries are added
      }
      return (*versions_target)[physical & kMaskEDRAM4] == version;
    };
    // Phase 1 collects the runs (tile-row pieces) to transfer; phase 2 records them.
    KeptList<SpanEDRAM4> spans_kept;  // usf4 0.6.7, see KeptList
    std::vector<SpanEDRAM4>& spans = spans_kept.get();
    // Full-width areas are one contiguous span (a transfer run may cross rows, as before).
    const bool complete_width = has_range && range.tx0 == 0 && range.tx1 == range.pitch;
    const uint32_t ty_end = complete_width ? range.ty0 + 1 : range.ty1;
    for (uint32_t ty = has_range ? range.ty0 : 0; has_range && ty < ty_end && ty * range.pitch < tiles; ++ty) {
    const uint32_t row_end = std::min(complete_width ? range.ty1 * range.pitch : ty * range.pitch + range.tx1,
                                       tiles);
    for (uint32_t tile = std::max(ty * range.pitch + range.tx0, start); tile < row_end;) {
      const auto owner_state = owners_tiles_edram4_[(target.edram_base + tile) & kMaskEDRAM4];
      if (!owner_state.image || owner_state.image == &target) { ++tile; continue; }
      bool whole = false;
      if (!TileUsedEDRAM4(target, tile, area, &whole)) { ++tile; continue; }
      if (clear_cutout) {
        whole = false;
        TileUsedEDRAM4(target, tile, *clear_cutout, &whole);
        for (size_t c = 0; !whole && clear_overwrite && c < edram4_cuts_extra_.size(); ++c)
          TileUsedEDRAM4(target, tile, edram4_cuts_extra_[c], &whole);
      }
      if (clear_overwrite && whole) {
        ++tile;
        continue;
      }
      if (synchronized(target.edram_base + tile, owner_state.version)) {
        ++edram4_sync_skipped_;
        ++tile;
        continue;
      }
      Image& source = *owner_state.image;
      const bool native_msaa_import = IsDepthNative2xEDRAM4(source) &&
          (IsDepthNative2xEDRAM4(target) || IsDepthNative1xEDRAM4(target));
      // sa2_msaa: host multisampled targets go on to the transfers, which take a single-sample source into them
      // (fragment passes) and decline the rest (skipped like any unsupported transfer).
      if ((!SamplesSupported(source) || !SamplesSupported(target)) && !native_msaa_import)
        return FailureEDRAM4("mixed hardware samples / color multisample physical alias unsupported",
                           target, &source);
      if (source.guest_width || source.guest_height)
        return FailureEDRAM4("scaled source has no physical transfer", target, &source);
      const uint32_t source_tile_width = 80u >>
          (uint32_t(source.edram_64bpp) + source.edram_msaa_x);
      // The current native image is only pitch pixels wide, not tile-rounded like the SDK.
      // Missing padding samples cannot silently become zero in a cross-view transfer.
      if (source.width % source_tile_width)
        return FailureEDRAM4("source lacks tile-rounded physical padding", target, &source);
      Prepare(source);
      if (!source.prepared) return FailureEDRAM4("source was not prepared", target, &source);
      // Restored in phase 2, right before its own run (as before the split): a restore forgets lazy-stencil
      // records pointing at this source, which an earlier run may still fetch.
      uint32_t count = 1;
      while (tile + count < row_end) {
        bool whole_next = false;
        const auto next = owners_tiles_edram4_[(target.edram_base + tile + count) & kMaskEDRAM4];
        const bool used = TileUsedEDRAM4(target, tile + count, area, &whole_next);
        if (clear_cutout) {
          whole_next = false;
          TileUsedEDRAM4(target, tile + count, *clear_cutout, &whole_next);
          for (size_t c = 0; !whole_next && clear_overwrite && c < edram4_cuts_extra_.size(); ++c)
            TileUsedEDRAM4(target, tile + count, edram4_cuts_extra_[c], &whole_next);
        }
        if (!used ||
            (clear_overwrite && whole_next) || next.image != &source ||
            next.tile_local != owner_state.tile_local + count ||
            synchronized(target.edram_base + tile + count, next.version)) break;
        ++count;
      }
      const uint32_t source_tile_height = 16u >> source.edram_msaa_y;
      if ((uint32_t(owner_state.tile_local) + count - 1) / PitchTilesEDRAM(source) *
              source_tile_height + source_tile_height > source.height)
        return FailureEDRAM4("source tile range exceeds available physical rows", target, &source);
      if (draws_) draws_->FinishPass();
      ForgetClear(target);
      if (trace_physical != UINT32_MAX && trace_local >= tile &&
          trace_local - tile < count && TraceEventEDRAM4(trace_physical)) {
        const uint32_t source_local = owner_state.tile_local + trace_local - tile;
        const uint32_t dest_width = 80u >> (uint32_t(target.edram_64bpp) + target.edram_msaa_x);
        const uint32_t dest_height = 16u >> target.edram_msaa_y;
        REXLOG_INFO("[native] EDRAM TILE TRACE import-attempt frame={} draw={} physical={} "
                    "source=[{}] target=[{}] groupedSource={} groupedDest={} count={} "
                    "sourceTileOrigin={},{} targetTileOrigin={},{} sourceTileExtent={}x{} "
                    "targetTileExtent={}x{} physicalSampleToHostShiftSrc={},{} dst={},{}",
                    trace_frame_, draws_ ? draws_->Drawn() : 0,
                    trace_physical, TraceImageEDRAM4(&source, source_local),
                    TraceImageEDRAM4(&target, trace_local), owner_state.tile_local, tile, count,
                    (source_local % PitchTilesEDRAM(source)) * source_tile_width,
                    (source_local / PitchTilesEDRAM(source)) * source_tile_height,
                    (trace_local % PitchTilesEDRAM(target)) * dest_width,
                    (trace_local / PitchTilesEDRAM(target)) * dest_height,
                    source_tile_width, source_tile_height, dest_width, dest_height,
                    source.edram_msaa_x, source.edram_msaa_y,
                    target.edram_msaa_x, target.edram_msaa_y);
      }
      {
        uint32_t class_value = 4;
        if (target.edram_depth && source.edram_depth)
          class_value = (source.edram_msaa_x != target.edram_msaa_x ||
                   source.raster_grid_x != target.raster_grid_x) ? 0 : 1;
        else if (target.edram_depth) class_value = 2;
        else if (source.edram_depth) class_value = 3;
        const uint32_t reason = std::min<uint32_t>(edram4_reason_, 3);
        ++edram4_transfer_ops_[reason][class_value];
        ++edram4_transfers_all_;  // sa2_gpu_resolve_alias
        edram4_transfer_tiles_[reason][class_value] += count;
        if (count + 8 >= tiles) ++edram4_transfer_full_[class_value];
        if (edram4_reason_ == 0 && count >= 64 && edram4_traces_ < 60) {
          ++edram4_traces_;
          REXLOG_INFO("[native] EDRAM mode4 transfer trace {}: {} tiles {:03X} {}x{} my{} -> {}x{} my{} by {}",
                      edram4_traces_, count, target.edram_base, source.width, source.height, source.edram_msaa_y,
                      target.width, target.height, target.edram_msaa_y,
                      g_lazy_trace_text.active ? TraceDrawEDRAM4() : edram4_draw_desc_);
        }
        const auto describe = [](const Image& im) {
          return fmt::format("{}{:03X}/f{}:{}x{}:mx{}my{}g{}{}", im.edram_depth ? "D" : "C", im.edram_base,
                             im.edram_format, im.width, im.height, im.edram_msaa_x, im.edram_msaa_y,
                             im.raster_grid_x, im.depth_float24_half ? "h" : "");
        };
        // sa2-nx: the per-pair / per-draw breakdown formats two to four strings and looks them up in string maps
        // for every transfer run (hundreds per frame on the ring thread): diagnostics only.
        static const bool pair_report = REXCVAR_GET(sa2_native_edram4_pair_report);
        if (pair_report) {
        auto& par = edram4_transfer_pairs_[describe(source) + "->" + describe(target)];
        if (!par.ops) par.example = edram4_reason_ == 0 ? DescribeDrawEDRAM4() : "-";
        if (edram4_reason_ == 0 && edram4_submission_ && edram4_submission_->registers) {
          const auto& q = *edram4_submission_;
          const bool ow = q.edram_overwrite_rect && (q.edram_overwrite_slots & (1u << edram4_slot_));
          auto& d = par.draws[fmt::format("VS{}/PS{}/p{}/m{}/dc{:X}/ow{}", q.vs ? int(q.vs->number) : -1,
                                            q.ps ? int(q.ps->number) : -1,
                                            q.registers[gr::XE_GPU_REG_VGT_DRAW_INITIATOR] & 63,
                                            q.registers[gr::XE_GPU_REG_RB_MODECONTROL] & 7,
                                            q.registers[gr::XE_GPU_REG_RB_DEPTHCONTROL],
                                            ow ? fmt::format("{},{}-{},{}", (*q.edram_overwrite_rect)[0],
                                                             (*q.edram_overwrite_rect)[1],
                                                             (*q.edram_overwrite_rect)[2],
                                                             (*q.edram_overwrite_rect)[3])
                                               : std::string("-"))];
          ++d.first;
          d.second += count;
        }
        ++par.ops;
        par.tiles += count;
        }
      }
      spans.push_back({&source, owner_state.tile_local, tile, count, native_msaa_import});
      {
        const RedirectedStamp& stamp = edram4_stamps_[(target.edram_base + tile) & kMaskEDRAM4];
        if (stamp.site && stamp.version == owner_state.version) {
          Consumer& c = edram4_consumer_[stamp.site];
          c = c.view == &target ? Consumer{&target, std::min(c.confidence + 1, 8u)} : Consumer{&target, 1};
        }
      }
      tile += count;
    }
    }
    // Phase 2: the runs, in order. Batched, consecutive runs of one kind share one barrier pair / one pass.
    if (spans.empty()) {
      if (!target.edram_depth && !clear_overwrite) PublishEDRAM4(target, area, limit, true);
      return true;
    }
    // usf4_edram4_import_columns: consecutive one-row runs that go through ImportColorDepthEDRAM4 from the same
    // source, at the same tile offset (one linear source mapping), over the same x range in touching tile rows,
    // become one span: the import draws the column's rectangle once (its shader maps tiles linearly and only the
    // rectangle is rasterized), and the bookkeeping walks the original runs.
    if (target.edram_depth && spans.size() > 1 && !edram4_import_stencil_only_ &&
        REXCVAR_GET(usf4_edram4_import_columns)) {
      const auto imports = [&](const SpanEDRAM4& t) {
        const Image& src = *t.source;
        return t.native_msaa_import || !src.edram_depth || src.depth_float24_half != target.depth_float24_half ||
               src.edram_format != target.edram_format || src.edram_msaa_x != target.edram_msaa_x ||
               src.edram_msaa_y != target.edram_msaa_y || !target.accepts_target_of_copy;
      };
      KeptList<SpanEDRAM4> merged_kept;
      std::vector<SpanEDRAM4>& merged = merged_kept.get();
      for (size_t i = 0; i < spans.size();) {
        VkRect2D first{};
        size_t j = i + 1;
        if (imports(spans[i]) && !spans[i].source->invalid_content &&
            RectImportEDRAM4(target, spans[i].tile, spans[i].count, first)) {
          VkRect2D last = first;
          const uint32_t offset = spans[i].tile_source - spans[i].tile;
          for (; j < spans.size(); ++j) {
            const SpanEDRAM4& n = spans[j];
            VkRect2D r{};
            if (n.source != spans[i].source || n.native_msaa_import != spans[i].native_msaa_import ||
                n.tile_source - n.tile != offset || !RectImportEDRAM4(target, n.tile, n.count, r) ||
                r.offset.x != first.offset.x || r.extent.width != first.extent.width ||
                r.offset.y != last.offset.y + int32_t(last.extent.height))
              break;
            // One row per run (the rectangle is then exactly the run's tiles).
            if (n.tile / PitchTilesEDRAM(target) != (n.tile + n.count - 1) / PitchTilesEDRAM(target)) break;
            last = r;
          }
          if (spans[i].tile / PitchTilesEDRAM(target) != (spans[i].tile + spans[i].count - 1) /
                                                            PitchTilesEDRAM(target))
            j = i + 1;
        }
        if (j - i < 2) {
          merged.push_back(std::move(spans[i]));
          ++i;
          continue;
        }
        SpanEDRAM4 column = spans[i];
        const SpanEDRAM4& end = spans[j - 1];
        column.count = end.tile + end.count - column.tile;
        column.rect = first;
        VkRect2D end_rect{};
        RectImportEDRAM4(target, end.tile, end.count, end_rect);
        column.rect.extent.height = uint32_t(end_rect.offset.y + int32_t(end_rect.extent.height) - first.offset.y);
        for (size_t k = i; k < j; ++k) column.runs.push_back({spans[k].tile, spans[k].count});
        ++edram4_columns_merged_;
        edram4_columns_runs_ += j - i;
        merged.push_back(std::move(column));
        i = j;
      }
      spans.swap(merged);
    }
    const bool batch = REXCVAR_GET(sa2_native_edram4_batch) && spans.size() > 1 && !edram4_batch_.active &&
        slots_[slot_].conversions_edram + 2 * spans.size() + 16 < kConversionsEDRAMPerSlot;
    if (batch) {
      edram4_batch_ = {};
      edram4_batch_.active = true;
      edram4_batch_spans_ += spans.size();
      if (target.edram_depth)
        for (const auto& t : spans) {
          VkRect2D r;
          if (!t.runs.empty()) r = t.rect;
          else if (!RectImportEDRAM4(target, t.tile, t.count, r)) continue;
          if (!edram4_batch_.has_area) {
            edram4_batch_.area_import = r;
            edram4_batch_.has_area = true;
            continue;
          }
          VkRect2D& u = edram4_batch_.area_import;
          const int32_t x0 = std::min(u.offset.x, r.offset.x), y0 = std::min(u.offset.y, r.offset.y);
          const int32_t x1 = std::max<int32_t>(u.offset.x + u.extent.width, r.offset.x + r.extent.width);
          const int32_t y1 = std::max<int32_t>(u.offset.y + u.extent.height, r.offset.y + r.extent.height);
          u = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
        }
    }
    const auto finish_batch = [&](bool ok) {
      if (!batch) return ok;
      const bool closed = CloseBatchEDRAM4();
      edram4_batch_.active = false;
      // A deferred stencil copy that failed leaves tiles already recorded as synced: forget them all.
      if (!closed) edram4_views_.erase(&target);
      return ok && closed;
    };
    for (const SpanEDRAM4& t : spans) {
      Image& source = *t.source;
      const uint32_t tile = t.tile, count = t.count;
      const bool native_msaa_import = t.native_msaa_import;
      struct { uint32_t tile_local; } owner_state{t.tile_source};
      if (source.invalid_content) {
        if (batch && !CloseBatchEDRAM4()) return finish_batch(false);
        if (!RestoreContent(source)) return finish_batch(false);
      }
      if (target.edram_depth && source.edram_depth && source.edram_msaa_x && source.edram_msaa_y &&
          !target.edram_msaa_x && !target.edram_msaa_y && source.edram_base == target.edram_base &&
          PitchTilesEDRAM(source) == PitchTilesEDRAM(target) && source.edram_format == target.edram_format &&
          source.depth_float24_half == target.depth_float24_half)
        edram4_alias_1x_[&source] = &target;
      bool success = false;
      if (target.edram_depth) {
        if (native_msaa_import || !source.edram_depth) {
          success = ImportColorDepthEDRAM4(source, target, owner_state.tile_local, tile, count, &t);
        } else if (source.depth_float24_half != target.depth_float24_half ||
            source.edram_format != target.edram_format ||
            source.edram_msaa_x != target.edram_msaa_x ||
            source.edram_msaa_y != target.edram_msaa_y || !target.accepts_target_of_copy)
          success = ImportColorDepthEDRAM4(source, target, owner_state.tile_local, tile, count, &t);
        else {
          if (batch && !CloseBatchEDRAM4()) return finish_batch(false);
          success = CopyTilesEDRAM4(source, target, owner_state.tile_local, tile, count);
          target.edram4_bits_stencil |= source.edram4_bits_stencil;
          // A plain copy carries the stencil plane as it is: carry its lazy-stencil record with it.
          if (success && REXCVAR_GET(sa2_native_edram4_stencil_lazy)) {
            if (StencilDeferredEDRAM4(source, owner_state.tile_local, count)) {
              if (target.edram4_usa_stencil) ++edram4_stencil_inexact_;
              auto& sources = StencilSourcesWrite(&target);
              if (sources.empty()) sources.resize(kTilesEDRAM4);
              const auto& de = edram4_stencil_source_[&source];
              for (uint32_t i = 0; i < count; ++i)
                sources[(tile + i) & kMaskEDRAM4] = de[(owner_state.tile_local + i) & kMaskEDRAM4];
            } else {
              StencilRealEDRAM4(target, tile, count);
            }
          }
        }
      } else if (source.edram_depth) {
        if (StencilDeferredEDRAM4(source, owner_state.tile_local, count)) {
          ++edram4_stencil_exported_;  // the export packs the stencil byte: bring the real one first
          // The late fetch imports into the source view: outside this batch.
          if (batch && !CloseBatchEDRAM4()) return finish_batch(false);
          edram4_batch_.active = false;
          const bool fetched = FetchStencilDeferredEDRAM4(source, owner_state.tile_local, count);
          edram4_batch_.active = batch;
          if (!fetched) return finish_batch(false);
        }
        success = ConvertDepthEDRAM4(source, target, owner_state.tile_local, tile, count);
      } else if (source.format == target.format && source.edram_format == target.edram_format &&
                 source.edram_64bpp == target.edram_64bpp &&
                 source.edram_msaa_x == target.edram_msaa_x && source.edram_msaa_y == target.edram_msaa_y) {
        if (batch && !CloseBatchEDRAM4()) return finish_batch(false);
        success = CopyTilesEDRAM4(source, target, owner_state.tile_local, tile, count);
      } else {
        success = ConvertAliasEDRAM(source, target, owner_state.tile_local, tile, count);
      }
      if (!t.runs.empty()) ++(success ? edram4_columns_drawn_ : edram4_columns_failed_);
      if (!success && REXCVAR_GET(banjo_native_edram4_skip_unsupported)) {
        // banjo-nx: a conversion the renderer does not have (2x MSAA <-> 1x color alias, depth -> color) used to
        // drop the whole draw: Nuts & Bolts' 2x MSAA scene target (base 768) overlaps the 1x one (base 720), so
        // ~1800 scene draws a frame were lost. Each scene pass clears those tiles first, so the draw goes on with
        // the target's own content there; the tiles count as synchronized so the transfer is not retried.
        if (banjo_transfers_skipped_++ < 8)
          REXLOG_INFO("[native] EDRAM4 transfer skipped (unsupported): dest base {} msaa {}x{} depth {} <- source "
                      "base {} msaa {}x{} depth {}, {} tiles", target.edram_base, target.edram_msaa_x,
                      target.edram_msaa_y, target.edram_depth, source.edram_base, source.edram_msaa_x,
                      source.edram_msaa_y, source.edram_depth, count);
        const std::vector<std::pair<uint32_t, uint32_t>> skipped_one{{tile, count}};
        for (const auto& run : t.runs.empty() ? skipped_one : t.runs)
          for (uint32_t i = 0; i < run.second; ++i) {
            const uint32_t physical = (target.edram_base + run.first + i) & kMaskEDRAM4;
            NoteSynchronizedEDRAM4(target, physical, owners_tiles_edram4_[physical].version);
          }
        continue;
      }
      if (!success) return finish_batch(FailureEDRAM4("physical transfer unsupported or failed", target, &source));
      const std::vector<std::pair<uint32_t, uint32_t>> one{{tile, count}};
      for (const auto& run : t.runs.empty() ? one : t.runs) {
        for (uint32_t i = 0; i < run.second; ++i) {
          const uint32_t physical = (target.edram_base + run.first + i) & kMaskEDRAM4;
          NoteSynchronizedEDRAM4(target, physical, owners_tiles_edram4_[physical].version);
          if (owners_tiles_edram4_[physical].image) owners_tiles_edram4_[physical].image->edram4_exported = true;
        }
        edram4_sync_copied_ += run.second;
      }
    }
    if (!finish_batch(true)) return false;
    // Loading into a color view is now authoritative even when a later draw is skipped. Depth remains
    // owned by its old view until an actual writing draw/clear succeeds (no claims on read-only binds).
    if (!target.edram_depth && !clear_overwrite) PublishEDRAM4(target, area, limit, true);
    return true;
  }

  // Destination rectangle of a depth import of tiles [target_start, +count).
  bool RectImportEDRAM4(const Image& target, uint32_t target_start, uint32_t count, VkRect2D& rect) const {
    const uint32_t destination_pitch = PitchTilesEDRAM(target);
    if (!destination_pitch || !count) return false;
    const uint32_t destination_tile_width = 80u >> target.edram_msaa_x;
    const uint32_t destination_tile_height = 16u >> target.edram_msaa_y;
    const uint32_t first_row = target_start / destination_pitch;
    const uint32_t last_row = (target_start + count - 1) / destination_pitch;
    const uint32_t sx = first_row == last_row ? (target_start % destination_pitch) * destination_tile_width : 0;
    const uint32_t sy = first_row * destination_tile_height;
    const uint32_t right = first_row == last_row
        ? std::min(target.width, ((target_start + count - 1) % destination_pitch + 1) * destination_tile_width)
        : target.width;
    const uint32_t bottom = std::min(target.height, (last_row + 1) * destination_tile_height);
    if (sx >= target.width || sy >= target.height || right <= sx || bottom <= sy) return false;
    rect = {{int32_t(sx), int32_t(sy)}, {right - sx, bottom - sy}};
    return true;
  }

  // Batched compute transfers: the "after" barrier of the last one (a superset of both kinds' masks).
  void CloseComputeBatchEDRAM4() {
    if (!edram4_batch_.compute_pending) return;
    edram4_batch_.compute_pending = false;
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
  }

  // Batched depth imports: end the shared pass, then the copy-engine stencil copies of its runs.
  bool CloseImportBatchEDRAM4() {
    if (edram4_batch_.import_open) {
      edram4_batch_.import_open = false;
      dfn_.vkCmdEndRenderPass(commands_work_);
      if (draws_) draws_->NotifyGraphicsExternalState();
    }
    bool ok = true;
    std::vector<CopyStencilPending> copies;
    copies.swap(edram4_batch_.copies_stencil);
    for (const auto& c : copies)
      if (!CopyStencilEDRAM4(*c.source, *c.target, c.source_start, c.target_start, c.count, c.scissor))
        ok = FailureEDRAM4("copy-engine stencil import failed", *c.target, c.source);
    return ok;
  }

  bool CloseBatchEDRAM4() {
    CloseComputeBatchEDRAM4();
    return CloseImportBatchEDRAM4();
  }

  bool CopyTilesEDRAM4(Image& source, Image& target, uint32_t source_start,
                         uint32_t target_start, uint32_t count) {
    // The native2x depth utility is routed separately. No multisample image
    // may silently enter the legacy physical-copy mapping.
    // sa2_msaa: vkCmdCopyImage between two images of the same sample count copies every sample.
    if (source.sample_count != target.sample_count || !SamplesSupported(source)) return false;
    if (!copy_image_ || source.scale_q != target.scale_q) return false;
    const uint32_t tile_width = 80u >> (uint32_t(target.edram_64bpp) + target.edram_msaa_x);
    const uint32_t tile_height = 16u >> target.edram_msaa_y;
    // usf4 0.6.7: working lists kept between calls (KeptList).
    KeptList<VkImageCopy> regions_kept;
    std::vector<VkImageCopy>& regions = regions_kept.get();
    // usf4-nx (usf4_copy_tiles_merged): tiles that are neighbours in both images go in one rectangle (a row of tiles,
    // then rows of the same span): the same pixels with far fewer regions. NVK copies colour images region by
    // region through vk_meta (pipeline, two temporary views, a descriptor set, a render pass and a draw each): a
    // fight frame had ~16,000 one-tile regions.
    static const bool merged = REXCVAR_GET(usf4_copy_tiles_merged);
    struct Rect { uint32_t sx, sy, dx, dy, w, h; };
    KeptList<Rect> rects_kept, stacked_kept;
    std::vector<Rect>& rects = rects_kept.get();
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t so = source_start + i, sd = target_start + i;
      const uint32_t sx = (so % PitchTilesEDRAM(source)) * tile_width;
      const uint32_t sy = (so / PitchTilesEDRAM(source)) * tile_height;
      const uint32_t dx = (sd % PitchTilesEDRAM(target)) * tile_width;
      const uint32_t dy = (sd / PitchTilesEDRAM(target)) * tile_height;
      if (sx >= source.width || sy >= source.height || dx >= target.width || dy >= target.height)
        return false;
      if (source.width - sx < std::min(tile_width, target.width - dx) ||
          source.height - sy < std::min(tile_height, target.height - dy)) return false;
      const VkExtent3D extent{std::min({tile_width, source.width - sx, target.width - dx}),
                              std::min({tile_height, source.height - sy, target.height - dy}), 1};
      Rect t{sx, sy, dx, dy, extent.width, extent.height};
      if (merged && !rects.empty()) {
        Rect& last = rects.back();
        if (last.sy == t.sy && last.dy == t.dy && last.h == t.h && last.sx + last.w == t.sx &&
            last.dx + last.w == t.dx) {
          last.w += t.w;  // the next tile of the same row in both images
          continue;
        }
      }
      rects.push_back(t);
    }
    if (merged && rects.size() > 1) {
      // Rows of the same span, one below the other in both images.
      std::vector<Rect>& stacked = stacked_kept.get();
      for (const Rect& r : rects) {
        bool joined = false;
        for (auto it = stacked.rbegin(); it != stacked.rend() && it->sy + it->h >= r.sy; ++it) {
          if (it->sx == r.sx && it->dx == r.dx && it->w == r.w && it->sy + it->h == r.sy &&
              it->dy + it->h == r.dy) {
            it->h += r.h;
            joined = true;
            break;
          }
        }
        if (!joined) stacked.push_back(r);
      }
      rects.swap(stacked);
    }
    for (const Rect& t : rects) {
      const auto add = [&](VkImageAspectFlags aspect) {
        VkImageCopy r{};
        r.srcSubresource = r.dstSubresource = {aspect, 0, 0, 1};
        r.srcOffset = {int32_t(t.sx), int32_t(t.sy), 0};
        r.dstOffset = {int32_t(t.dx), int32_t(t.dy), 0};
        r.extent = {t.w, t.h, 1};
        regions.push_back(r);
      };
      if (target.edram_depth) {
        add(VK_IMAGE_ASPECT_DEPTH_BIT);
        add(VK_IMAGE_ASPECT_STENCIL_BIT);
      } else add(VK_IMAGE_ASPECT_COLOR_BIT);
    }
    HostCopies(source, target, regions.data(), uint32_t(regions.size()));  // sa2-nx render resolution
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before,
                                0, nullptr, 0, nullptr);
    copy_image_(commands_work_, source.image, VK_IMAGE_LAYOUT_GENERAL, target.image,
                    VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after,
                                0, nullptr, 0, nullptr);
    return true;
  }

  // Mass Effect (Eden Prime point-light shadows): the game clears an 880x880 1x depth shadow map with a
  // D3D Clear rectangle through a 440-pixel 4x MSAA view of the same EDRAM (a quarter of the pixels to
  // shade). Drawn literally, every clear transfers the tiles 1x -> 4x and the next shadow draw 4x -> 1x
  // (9 passes each on Switch without stencil export). A proven, tile-aligned, constant-depth rectangle
  // that touches only depth (ALWAYS, stencil off) is instead applied as a depth-only clear in each tile's
  // current owner; ownership stays with the owner (new version), the drawn view is not touched.
  // ReXGlue xenos Float32To20e4 without rounding (me_edram_depth_a_*.comp A20e4).
  static uint32_t WordFloat24EDRAM4(float z) {
    if (!(z > 0.0f)) return 0u;
    uint32_t bits = std::bit_cast<uint32_t>(z);
    if (bits >= 0x3FFFFFF8u) return 0xFFFFFFu;
    if (bits < 0x38800000u) bits = ((bits & 0x7FFFFFu) | 0x800000u) >> std::min(113u - (bits >> 23), 24u);
    else bits += 0xC8000000u;
    return (bits >> 3) & 0xFFFFFFu;
  }
  // me_edram_depth_to_16f.comp From7e3.
  static float From7e3EDRAM4(uint32_t f10) {
    f10 &= 0x3FFu;
    if (!f10) return 0.0f;
    uint32_t mantissa = f10 & 0x7Fu, exponent = f10 >> 7;
    if (!exponent) {
      const uint32_t displacement = 7u - uint32_t(std::bit_width(mantissa) - 1);
      exponent = 1u - displacement;
      mantissa = (mantissa << displacement) & 0x7Fu;
    }
    return std::bit_cast<float>(((exponent + 124u) << 23) | (mantissa << 16));
  }

  // Partial tile of a redirected clear: [px0,px1) x [py0,py1) of tile `local` of `drawn` (its pixels), mapped
  // into the tile's current owner. Same-geometry depth owners (and the drawn view itself) take a sub-rect;
  // a color owner takes the words the depth layout puts there (the 40-word half swap of the export shaders),
  // only when they cover whole owner pixels.
  template <typename ColorFor, typename Region>
  bool PartialEDRAM4(Image& drawn, uint32_t local, uint32_t px0, uint32_t px1, uint32_t py0, uint32_t py1,
                     const ColorFor& color_for, Region& region) {
    const auto owner = owners_tiles_edram4_[(drawn.edram_base + local) & kMaskEDRAM4];
    Image* w = owner.image ? owner.image : nullptr;
    if (!w || w->guest_width || w->guest_height || w->raster_grid_x || !w->prepared || w->invalid_content ||
        w->edram_64bpp || !SamplesSupported(*w)) return false;
    const uint32_t wp = PitchTilesEDRAM(*w);
    if (!wp) return false;
    const uint32_t ox = (owner.tile_local % wp) * (80u >> w->edram_msaa_x);
    const uint32_t oy = (owner.tile_local / wp) * (16u >> w->edram_msaa_y);
    // Samples of the tile the clear covers.
    const uint32_t sx0 = px0 << drawn.edram_msaa_x, sx1 = px1 << drawn.edram_msaa_x;
    const uint32_t sy0 = py0 << drawn.edram_msaa_y, sy1 = py1 << drawn.edram_msaa_y;
    const uint32_t mx = (1u << w->edram_msaa_x) - 1u, my = (1u << w->edram_msaa_y) - 1u;
    if ((sy0 & my) || (sy1 & my)) return false;
    const auto rect = [&](uint32_t a, uint32_t b) -> VkRect2D {
      return {{int32_t(ox + (a >> w->edram_msaa_x)), int32_t(oy + (sy0 >> w->edram_msaa_y))},
              {(b - a) >> w->edram_msaa_x, (sy1 - sy0) >> w->edram_msaa_y}};
    };
    region.owner = w;
    region.first_local = local;
    region.owner_local = owner.tile_local;
    region.count = 1;
    region.reassign = false;
    region.area2 = {};
    region.partial = true;
    if (w->edram_depth) {
      if (w != &drawn && (w->edram_format != drawn.edram_format ||
                          w->depth_float24_half != drawn.depth_float24_half)) return false;
      if ((sx0 & mx) || (sx1 & mx)) return false;
      region.area = rect(sx0, sx1);
    } else {
      if (!color_for(*w)) return false;
      // Depth word d sits where color word (d < 40 ? d + 40 : d - 40) is.
      const uint32_t a0 = sx0, a1 = std::min(sx1, 40u), b0 = std::max(sx0, 40u), b1 = sx1;
      bool first = true;
      if (a1 > a0) {
        if (((a0 + 40) & mx) || ((a1 + 40) & mx)) return false;
        region.area = rect(a0 + 40, a1 + 40);
        first = false;
      }
      if (b1 > b0) {
        if (((b0 - 40) & mx) || ((b1 - 40) & mx)) return false;
        (first ? region.area : region.area2) = rect(b0 - 40, b1 - 40);
      }
    }
    const VkRect2D& r = region.area;
    return r.extent.width && r.extent.height && uint32_t(r.offset.x) + r.extent.width <= w->width &&
           uint32_t(r.offset.y) + r.extent.height <= w->height &&
           (!region.area2.extent.width ||
            uint32_t(region.area2.offset.x) + region.area2.extent.width <= w->width);
  }


  bool RedirectClearDepthEDRAM4(const SubmissionDraw& p) {
    if (!REXCVAR_GET(sa2_native_edram4_redirected_clear) || !p.registers || !p.edram_overwrite_rect ||
        !(p.edram_overwrite_slots & 1) || !draws_) return false;
    const auto no = [&](const char* reason) {
      if (g_redirect_key_int.active) {  // folded into the string map when the report is written: same report
        g_redirect_key_int.UseSampled();
        ++edram4_depth_no_fast_[{reason, p.registers[gr::XE_GPU_REG_RB_DEPTH_INFO]}];
        return false;
      }
      ++edram4_depth_no_reasons_[fmt::format("{} (depth info {:08X})", reason,
                                             p.registers[gr::XE_GPU_REG_RB_DEPTH_INFO])];
      return false;
    };
    if (!p.edram_overwrite_depth) return no("depth not proven constant");
    // sa2-nx: depth + color 0 (Direct3D's Clear through the 4x alias). The color half is planned first, with no
    // side effect, and committed after the depth half: both or neither.
    ColorClearEDRAM4 color_clear;
    if (p.edram_overwrite_slots == 3) {
      if (!PlanClearColor0EDRAM4(p, color_clear)) {
        // usf4-nx: the reason in the report (static strings, so the key stays a pointer-free literal set).
        static std::unordered_map<std::string, std::string> keys;
        auto& k = keys[color0_plan_reason_];
        if (k.empty()) k = std::string("also writes color (color 0: ") + color0_plan_reason_ + ")";
        return no(k.c_str());
      }
    } else if (p.edram_overwrite_slots != 1) {
      return no("also writes color");
    }
    const uint32_t* r = p.registers;
    const uint32_t mode = r[gr::XE_GPU_REG_RB_MODECONTROL] & 7;
    const uint32_t dc = r[gr::XE_GPU_REG_RB_DEPTHCONTROL];
    if (mode != 5 && !(mode == uint32_t(xenos::EdramMode::kColorDepth) &&
                       (!r[gr::XE_GPU_REG_RB_COLOR_MASK] || !color_clear.runs.empty())))
      return no("color mask on");
    if ((dc & 0x76) != 0x76) return no("depth not ALWAYS+write");
    // Stencil on: only a full-mask constant stencil write over the same rectangle (a depth+stencil D3D Clear);
    // both aspects are then cleared in the owner.
    int32_t stencil_value = -1;
    if (dc & 1) {
      if (!REXCVAR_GET(sa2_native_edram4_clear_stencil) || !p.edram_stencil_replace_rect ||
          *p.edram_stencil_replace_rect != *p.edram_overwrite_rect) return no("stencil not replaced over the rect");
      const auto value_for = [](uint32_t zpass, uint32_t refmask) -> int32_t {
        return zpass == 2 ? int32_t(refmask & 0xFFu) : zpass == 1 ? 0 : -1;
      };
      stencil_value = value_for((dc >> 14) & 7, r[gr::XE_GPU_REG_RB_STENCILREFMASK]);
      if (stencil_value < 0 || ((dc & 0x80) &&
          value_for((dc >> 26) & 7, r[gr::XE_GPU_REG_RB_STENCILREFMASK_BF]) != stencil_value))
        return no("stencil value not constant");
    }
    // D24FS8: draws round depth to float24; only 0 and 1 (exact in float24) can be cleared without it.
    if (((r[gr::XE_GPU_REG_RB_DEPTH_INFO] >> 16) & 1) &&
        !(*p.edram_overwrite_depth == 0.0f || *p.edram_overwrite_depth == 1.0f)) return no("D24FS8 depth not 0/1");
    const uint32_t info = r[gr::XE_GPU_REG_RB_DEPTH_INFO];
    const uint32_t pitch = r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF;
    if (!Record()) return false;
    Image* drawn = GetDepth(info & 0xFFF, (info >> 16) & 1, pitch);
    if (!drawn || drawn->guest_width || drawn->guest_height || drawn->raster_grid_x) return no("drawn view unsupported");
    const auto& o = *p.edram_overwrite_rect;
    const uint32_t tw = 80u >> drawn->edram_msaa_x, th = 16u >> drawn->edram_msaa_y;
    // A color owner gets the EDRAM word the clear writes, decoded as its own format (the export shaders'
    // decoding); needs the whole word, so the stencil byte too.
    std::optional<VkClearColorValue> color_rgba8, color_16f;
    if (stencil_value >= 0 && REXCVAR_GET(sa2_native_edram4_clear_color)) {
      const float z = std::clamp(*p.edram_overwrite_depth, 0.0f, 1.0f);
      const uint32_t depth = drawn->edram_format == 0 ? uint32_t(std::nearbyint(double(z) * 16777215.0))
                                                             : WordFloat24EDRAM4(z);
      const uint32_t word = (depth << 8) | uint32_t(stencil_value);
      VkClearColorValue a{}, b{};
      for (uint32_t i = 0; i < 4; ++i) a.float32[i] = float((word >> (8 * i)) & 255u) * (1.0f / 255.0f);
      color_rgba8 = a;
      for (uint32_t i = 0; i < 3; ++i) b.float32[i] = From7e3EDRAM4(word >> (10 * i));
      b.float32[3] = float(word >> 30) * (1.0f / 3.0f);
      color_16f = b;  // 7e3 (format 3); UNORM10 (format 2) is decided per owner below
    }
    const bool word_zero = color_rgba8 && stencil_value == 0 && *p.edram_overwrite_depth == 0.0f;
    const auto color_for = [&](const Image& w) -> std::optional<VkClearColorValue> {
      if (w.edram_depth || !color_rgba8) return std::nullopt;
      if (word_zero) return VkClearColorValue{};  // all-zero words are 0 in every color format
      if (w.edram_64bpp) return std::nullopt;
      if (w.format == VK_FORMAT_R8G8B8A8_UNORM && (w.edram_format == 0 || w.edram_format == 1)) return color_rgba8;
      if (w.format == VK_FORMAT_R16G16B16A16_SFLOAT && w.edram_format == 3) return color_16f;
      if (w.format == VK_FORMAT_R16G16B16A16_SFLOAT && w.edram_format == 2) {
        const uint32_t word = [&] {
          VkClearColorValue a = *color_rgba8;
          uint32_t v = 0;
          for (uint32_t i = 0; i < 4; ++i) v |= uint32_t(std::lround(a.float32[i] * 255.0f)) << (8 * i);
          return v;
        }();
        VkClearColorValue c{};
        for (uint32_t i = 0; i < 3; ++i) c.float32[i] = float((word >> (10 * i)) & 0x3FFu) / 1023.0f;
        c.float32[3] = float(word >> 30) / 3.0f;
        return c;
      }
      return std::nullopt;
    };
    // Partial edge tiles are cleared in their owner too when the bounds are exact (the draw touches no pixel
    // outside the proven rect): sa2_native_edram4_partial_clear.
    const bool aligned = !(o[0] % tw || o[1] % th || o[2] % tw || o[3] % th);
    const bool partial_ok = REXCVAR_GET(sa2_native_edram4_partial_clear) && p.edram_bounds_rect &&
        *p.edram_bounds_rect == o && o[2] > o[0] && o[3] > o[1];
    if (o[0] < 0 || o[1] < 0 || (!aligned && !partial_ok) ||
        uint32_t(o[2]) > drawn->width || uint32_t(o[3]) > drawn->height) {
      ++edram4_redirected_no_;
      ++edram4_depth_no_reasons_[fmt::format("rect {},{}-{},{} not tile-aligned in {:03X}:{}x{}:mx{}my{}", o[0], o[1],
                                             o[2], o[3], drawn->edram_base, drawn->width, drawn->height,
                                             drawn->edram_msaa_x, drawn->edram_msaa_y)];
      return false;
    }
    const uint32_t pitch_tiles = PitchTilesEDRAM(*drawn);
    // Depth 0 is the all-zero 24-bit field in both D24S8 and D24FS8 (and 0 in every host encoding): another
    // depth encoding's owner can take the clear as is.
    const bool zero_depth = *p.edram_overwrite_depth == 0.0f && REXCVAR_GET(sa2_native_edram4_clear_zero);
    struct Region {
      Image* owner;
      uint32_t first_local, owner_local, count;
      VkRect2D area;
      bool reassign;
      VkRect2D area2{};  // a partial tile of a color owner can map to two rects (depth/color half-tile swap)
      bool partial = false;
    };
    KeptList<Region> regions_kept;  // KeptList: see its comment
    std::vector<Region>& regions = regions_kept.get();
    // Target of a tile: its current owner if that is another depth view; if the drawn view itself owns it
    // (or nobody does), the drawn view's known 1x alias, which then becomes the owner.
    const auto alias_it = edram4_alias_1x_.find(drawn);
    Image* const alias = alias_it != edram4_alias_1x_.end() && REXCVAR_GET(sa2_native_edram4_clear_alias)
                              ? alias_it->second : nullptr;
    struct Goal { Image* w; uint32_t local; bool reassign; };
    const auto goal = [&](uint32_t local) -> Goal {
      const auto owner = owners_tiles_edram4_[(drawn->edram_base + local) & kMaskEDRAM4];
      // A depth owner with the same encoding is cleared in place. Any other owner (a color view, a different
      // encoding) hands the tile to the 1x alias: the clear writes all of its depth, and the lazy-stencil
      // record keeps pointing at the old owner's bits (a depth-only clear leaves them in EDRAM).
      const bool owner_utilizable = owner.image && owner.image != drawn && owner.image->edram_depth &&
          ((owner.image->edram_format == drawn->edram_format &&
            owner.image->depth_float24_half == drawn->depth_float24_half) || zero_depth);
      // A color owner can only be a later stencil-fetch source in the classes the import reads.
      const bool color_supported = owner.image && !owner.image->edram_depth && !owner.image->edram_64bpp &&
          (owner.image->edram_format == 0 || owner.image->edram_format == 2 || owner.image->edram_format == 3);
      if (owner_utilizable || (owner.image && owner.image != drawn && (!alias || (!owner.image->edram_depth &&
                                                                                    !color_supported))))
        return {owner.image, owner.tile_local, false};
      return {alias, local, true};
    };
    const uint32_t ty_end = (uint32_t(o[3]) + th - 1) / th, tx_end = (uint32_t(o[2]) + tw - 1) / tw;
    // Learned consumer (the view that last pulled this clear's tiles): clear straight into it and hand it the
    // tiles, so its next bind transfers nothing. Ownership stays exact whatever view is chosen.
    bool direct = false;
    const uint64_t site = SiteClear(drawn, edram4_cleared_in_frame_[drawn]++);
    if (aligned && REXCVAR_GET(sa2_native_edram4_clear_consumer)) {
      const auto it = edram4_consumer_.find(site);
      // Only a consumer seen twice in a row (one that alternates would just move the transfer).
      Image* c = it != edram4_consumer_.end() && it->second.confidence >= 2 ? it->second.view : nullptr;
      const bool c_ok = c && c != drawn && c->image != drawn->image && c->prepared && !c->invalid_content &&
          !c->guest_width && !c->guest_height && !c->raster_grid_x &&
          SamplesSupported(*c) &&
          (c->edram_depth ? !c->edram_64bpp && ((c->edram_format == drawn->edram_format &&
                                                      c->depth_float24_half == drawn->depth_float24_half) ||
                                                     zero_depth)
                                : bool(color_for(*c)));
      const uint32_t cp = c_ok ? PitchTilesEDRAM(*c) : 0;
      const uint32_t cw = c_ok ? 80u >> (c->edram_msaa_x + uint32_t(c->edram_64bpp)) : 0;
      const uint32_t ch = c_ok ? 16u >> c->edram_msaa_y : 0;
      if (cp && !(c->width % cw)) {
        direct = true;
        for (uint32_t ty = uint32_t(o[1]) / th; direct && ty < ty_end; ++ty) {
          for (uint32_t tx = uint32_t(o[0]) / tw; tx < tx_end;) {
            const uint32_t local = ty * pitch_tiles + tx;
            const uint32_t lc = (drawn->edram_base + local - c->edram_base) & kMaskEDRAM4;
            uint32_t count = 1;
            while (tx + count < tx_end && (lc % cp) + count < cp) ++count;
            const uint32_t x = (lc % cp) * cw, y = (lc / cp) * ch, width = cw * count;
            if (x >= c->width || y >= c->height || width > c->width - x || ch > c->height - y) {
              direct = false;
              break;
            }
            regions.push_back({c, local, lc, count, {{int32_t(x), int32_t(y)}, {width, ch}}, true});
            tx += count;
          }
        }
        if (direct) ++edram4_consumer_uses_;
        else regions.clear();
      }
    }
    const auto tile_whole = [&](uint32_t tx, uint32_t ty) {
      return int32_t(tx * tw) >= o[0] && int32_t((tx + 1) * tw) <= o[2] &&
             int32_t(ty * th) >= o[1] && int32_t((ty + 1) * th) <= o[3];
    };
    for (uint32_t ty = uint32_t(o[1]) / th; !direct && ty < ty_end; ++ty) {
      for (uint32_t tx = uint32_t(o[0]) / tw; tx < tx_end;) {
        const uint32_t local = ty * pitch_tiles + tx;
        if (!tile_whole(tx, ty)) {
          // The covered part of the tile, in the drawn view's pixels relative to the tile.
          const uint32_t px0 = uint32_t(std::max<int32_t>(o[0], int32_t(tx * tw))) - tx * tw;
          const uint32_t px1 = uint32_t(std::min<int32_t>(o[2], int32_t((tx + 1) * tw))) - tx * tw;
          const uint32_t py0 = uint32_t(std::max<int32_t>(o[1], int32_t(ty * th))) - ty * th;
          const uint32_t py1 = uint32_t(std::min<int32_t>(o[3], int32_t((ty + 1) * th))) - ty * th;
          Region region{};
          if (!PartialEDRAM4(*drawn, local, px0, px1, py0, py1, color_for, region)) {
            ++edram4_redirected_no_;
            ++edram4_depth_no_reasons_["partial tile: owner unsupported"];
            return false;
          }
          regions.push_back(region);
          ++tx;
          continue;
        }
        const Goal t = goal(local);
        Image* w = t.w;
        const bool color_ok = w && w != drawn && !t.reassign && color_for(*w) && !w->guest_width && !w->guest_height &&
            w->prepared && !w->invalid_content && !w->raster_grid_x &&
            SamplesSupported(*w);
        if (!color_ok && (!w || w == drawn || !w->edram_depth || w->edram_64bpp || w->guest_width ||
            w->guest_height || w->image == drawn->image || !w->prepared ||
            ((w->edram_format != drawn->edram_format || w->depth_float24_half != drawn->depth_float24_half) &&
             !zero_depth) ||
            !SamplesSupported(*w))) {
          ++edram4_redirected_no_;
          ++edram4_depth_no_reasons_[!w ? std::string("no target (no 1x alias)") : w == drawn
              ? std::string("drawn view") : fmt::format("target {}{:03X}/f{}:{}x{}:mx{}my{} samples{} (clear of "
                  "{:03X}/f{} mx{}my{} depth {} stencil {})",
                  w->edram_depth ? "D" : "C", w->edram_base, w->edram_format, w->width, w->height,
                  w->edram_msaa_x, w->edram_msaa_y, uint32_t(w->sample_count), drawn->edram_base,
                  drawn->edram_format, drawn->edram_msaa_x, drawn->edram_msaa_y, *p.edram_overwrite_depth,
                  stencil_value)];
          return false;
        }
        const uint32_t wp = PitchTilesEDRAM(*w);
        const uint32_t ww = 80u >> w->edram_msaa_x, wh = 16u >> w->edram_msaa_y;
        if (!wp || w->width % ww || (t.reassign && wp != pitch_tiles)) { ++edram4_redirected_no_; return false; }
        uint32_t count = 1;
        while (tx + count < tx_end && tile_whole(tx + count, ty) && (t.local % wp) + count < wp) {
          const Goal n = goal(local + count);
          if (n.w != w || n.local != t.local + count || n.reassign != t.reassign) break;
          ++count;
        }
        const uint32_t x = (t.local % wp) * ww << w->raster_grid_x;
        const uint32_t y = (t.local / wp) * wh;
        const uint32_t width = (ww * count) << w->raster_grid_x;
        if (x >= w->width || y >= w->height || width > w->width - x || wh > w->height - y) {
          ++edram4_redirected_no_;
          return false;
        }
        regions.push_back({w, local, t.local, count, {{int32_t(x), int32_t(y)}, {width, wh}}, t.reassign});
        tx += count;
      }
    }
    if (regions.empty()) return false;
    draws_->FinishPass();
    if (!Record()) return false;
    // sa2-nx: every region is prepared first, then all the clears go between ONE barrier pair, with the regions of
    // one owner that line up (consecutive tile rows of a full-width clear) merged into one rectangle: a full-screen
    // redirected clear was 45 render passes and 90 ALL -> ALL barriers (each a GPU wait-for-idle on NVK). The
    // regions are distinct tiles, so the clears have no hazard between them; the bookkeeping below is unchanged.
    struct MergedClear {
      Image* w;
      VkRect2D rect;
    };
    KeptList<MergedClear> merged_kept;
    std::vector<MergedClear>& merged = merged_kept.get();
    const auto add_clear = [&](Image* w, const VkRect2D& rect) {
      for (auto& m : merged) {
        if (m.w == w && m.rect.offset.x == rect.offset.x && m.rect.extent.width == rect.extent.width &&
            m.rect.offset.y + int32_t(m.rect.extent.height) == rect.offset.y) {
          m.rect.extent.height += rect.extent.height;
          return;
        }
      }
      merged.push_back({w, rect});
    };
    for (const auto& region : regions) {
      Image& w = *region.owner;
      Prepare(w);
      if (!w.prepared || !RestoreContent(w)) return false;
      ForgetClear(w);
      if (!w.edram_depth) {
        if (!color_for(w)) return FailureEDRAM4("redirected depth clear into a color owner failed", *drawn, &w);
      } else if (region.partial && stencil_value >= 0 && StencilDeferredEDRAM4(w, region.owner_local, 1) &&
                 !FetchStencilDeferredEDRAM4(w, region.owner_local, 1)) {
        // A partial stencil clear keeps the rest of the tile's stencil: make it real here first.
        return FailureEDRAM4("partial redirected clear: stencil fetch failed", *drawn, &w);
      }
      add_clear(&w, region.area);
      if (region.area2.extent.width) add_clear(&w, region.area2);
    }
    BarrierClearEDRAM4();
    for (const auto& m : merged) {
      Image& w = *m.w;
      if (!w.edram_depth) {
        if (!draws_->ClearColorInPass(commands_work_, w, *color_for(w), m.rect))
          return FailureEDRAM4("redirected depth clear into a color owner failed", *drawn, &w);
        continue;
      }
      const float native = me::native::GuestDepthToNative(*p.edram_overwrite_depth, w.depth_float24_half);
      if (stencil_value >= 0 ? !draws_->ClearDepthInPass(commands_work_, w, native,
                                                                  uint32_t(stencil_value), &m.rect)
                             : !draws_->ClearOnlyDepthInPass(commands_work_, w, native, m.rect))
        return FailureEDRAM4("redirected depth clear failed", *drawn, &w);
    }
    BarrierClearEDRAM4();
    edram4_redirected_merged_ += merged.size();
    for (const auto& region : regions) {
      Image& w = *region.owner;
      if (!w.edram_depth) {
        for (uint32_t i = 0; i < region.count; ++i) {
          const uint32_t physical = (drawn->edram_base + region.first_local + i) & kMaskEDRAM4;
          auto& d = owners_tiles_edram4_[physical];
          if (region.reassign) {
            d = {&w, uint16_t((region.owner_local + i) & kMaskEDRAM4), 0};
            ++edram4_epoch_;
          }
          d.version = ++edram4_version_;
          edram4_stamps_[physical] = {site, d.version};
        }
        edram4_redirected_color_tiles_ += region.count;
        continue;
      }
      if (stencil_value >= 0) w.edram4_bits_stencil |= uint8_t(stencil_value);
      for (uint32_t i = 0; i < region.count; ++i) {
        auto& d = owners_tiles_edram4_[(drawn->edram_base + region.first_local + i) & kMaskEDRAM4];
        if (region.reassign) {
          const uint32_t local = (region.owner_local + i) & kMaskEDRAM4;  // the new owner's own tile index
          if (stencil_value < 0 && REXCVAR_GET(sa2_native_edram4_stencil_lazy) && d.image &&
              d.image != &w) {
            const auto previous = edram4_stencil_source_.find(d.image);
            const SourceStencil f = previous != edram4_stencil_source_.end() && previous->second[d.tile_local & kMaskEDRAM4].image
                ? previous->second[d.tile_local & kMaskEDRAM4] : SourceStencil{d.image, uint16_t(d.tile_local & kMaskEDRAM4)};
            auto& sources = StencilSourcesWrite(&w);
            if (sources.empty()) sources.resize(kTilesEDRAM4);
            sources[local] = f.image == &w && f.tile == local ? SourceStencil{} : f;
          }
          d = {&w, uint16_t(local), 0};
          ++edram4_epoch_;
        }
        d.version = ++edram4_version_;
        edram4_stamps_[(drawn->edram_base + region.first_local + i) & kMaskEDRAM4] = {site, d.version};
      }
      if (stencil_value >= 0) StencilRealEDRAM4(w, region.owner_local, region.count);  // both aspects written
      edram4_redirected_tiles_ += region.count;
    }
    if (!color_clear.runs.empty() && !CommitClearColor0EDRAM4(color_clear))
      return FailureEDRAM4("redirected color 0 clear failed", *drawn, color_clear.drawn);
    ++edram4_redirected_;
    return true;
  }

  // sa2-nx: the color half of a depth + color 0 clear drawn through a 4x alias (Direct3D's Clear: SA2 clears its 1x
  // 1280x720 back buffers, and the 560-wide split-screen ones, through 4x aliases of the same tiles every frame).
  // Drawn literally, the alias gets the clear and every tile then moves 4x -> 1x (ConvertAliasColorFragEDRAM4).
  // Instead each tile's current owner, a 1x RGBA8 view, gets the clear word directly: a tile is 80x16 samples
  // whatever the view, the clear writes the same word to all of them, and the transfer copies words verbatim, so
  // the owner ends up with exactly the bytes it would have received. Owners other than RGBA8 color views (or the
  // drawn alias itself, on the first frame) decline: the draw then goes the normal way.
  struct ColorClearEDRAM4 {
    struct Run {
      Image* owner;
      uint32_t first_local, count;
      VkRect2D area;
      bool depth_owner = false;  // usf4-nx: the owner is a depth view (usf4_edram4_clear_into_depth)
      uint32_t owner_local = 0;  // the owner's own index of the run's first tile
      bool reassign = false;  // usf4-nx: the owner is the learned consumer and takes the tiles
    };
    std::vector<Run> runs;
    VkClearColorValue value{};
    Image* drawn = nullptr;
    uint32_t word = 0;  // usf4-nx: the 32-bit EDRAM word the clear writes (byte c = channel c)
    uint64_t site = 0;  // usf4-nx: SiteClear of the drawn color view (usf4_edram4_clear_color_consumer)
  };
  // usf4-nx: ReXGlue xenos Float20e4To32 (the inverse of WordFloat24EDRAM4).
  static float FromFloat24EDRAM4(uint32_t f24) {
    f24 &= 0xFFFFFFu;
    if (!f24) return 0.0f;
    uint32_t mantissa = f24 & 0xFFFFFu, exponent = f24 >> 20;
    if (!exponent) {
      const uint32_t shift = 20u - uint32_t(std::bit_width(mantissa) - 1);
      exponent = 1u - shift;
      mantissa = (mantissa << shift) & 0xFFFFFu;
    }
    return std::bit_cast<float>(((exponent + 112u) << 23) | (mantissa << 3));
  }
  bool PlanClearColor0EDRAM4(const SubmissionDraw& p, ColorClearEDRAM4& plan) {
    // usf4-nx: why the color half was declined, for the depth clear redirect's report.
    const auto plan_no = [&](const char* reason) { color0_plan_reason_ = reason; return false; };
    color0_plan_reason_ = "";
    if (!REXCVAR_GET(sa2_native_edram4_clear_color0) || !p.edram_overwrite_color0 || !p.edram_overwrite_rect)
      return plan_no("off or no constant color/rect");
    const uint32_t* r = p.registers;
    const uint32_t info = r[gr::XE_GPU_REG_RB_COLOR_INFO];
    if (((info >> 16) & 0xF) != 0) return plan_no("not k_8_8_8_8");  // k_8_8_8_8 only (no gamma, no float formats)
    // The bytes the ROP stores: round to nearest of the clamped value. A value within float rounding of a tie
    // could round either way on the GPU: declined.
    for (uint32_t c = 0; c < 4; ++c) {
      const float v = (*p.edram_overwrite_color0)[c];
      if (std::isnan(v)) return plan_no("color NaN");
      const double f = double(std::clamp(v, 0.0f, 1.0f)) * 255.0;
      if (std::abs(f - std::floor(f) - 0.5) < 1.0 / 1024.0) return plan_no("color on a rounding tie");
      plan.value.float32[c] = float(std::lround(f)) / 255.0f;
      plan.word |= uint32_t(std::lround(f)) << (8 * c);
    }
    if (!Record()) return plan_no("no record");
    Image* drawn = GetTarget(info & 0xFFF, 0, r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF);
    if (!drawn || drawn->edram_depth || drawn->edram_64bpp || drawn->guest_width || drawn->guest_height ||
        drawn->raster_grid_x || drawn->format != VK_FORMAT_R8G8B8A8_UNORM) return plan_no("drawn view unsupported");
    const auto& o = *p.edram_overwrite_rect;
    const uint32_t tw = 80u >> drawn->edram_msaa_x, th = 16u >> drawn->edram_msaa_y;
    if (o[0] < 0 || o[1] < 0 || o[0] % tw || o[1] % th || o[2] % tw || o[3] % th || o[2] <= o[0] ||
        o[3] <= o[1] || uint32_t(o[2]) > drawn->width || uint32_t(o[3]) > drawn->height) {
      static int logged = 0;  // usf4-nx diagnostic
      if (logged < 12 && ++logged)
        REXLOG_INFO("[usf4diag] color0 rect {},{}-{},{} tile {}x{} drawn {}x{} base {:03X} msaa {}/{} pitch {}", o[0], o[1],
                    o[2], o[3], tw, th, drawn->width, drawn->height, drawn->edram_base, drawn->edram_msaa_x,
                    drawn->edram_msaa_y, r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF);
      return plan_no("rect not tile-aligned or outside");
    }
    const uint32_t pitch_tiles = PitchTilesEDRAM(*drawn);
    if (!pitch_tiles) return plan_no("no pitch");
    plan.site = SiteClear(drawn, edram4_cleared_in_frame_[drawn]++);
    if (REXCVAR_GET(usf4_edram4_clear_color_consumer) && PlanColor0ConsumerEDRAM4(plan, *drawn, o, tw, th, pitch_tiles)) {
      plan.drawn = drawn;
      return true;
    }
    for (uint32_t ty = uint32_t(o[1]) / th; ty < uint32_t(o[3]) / th; ++ty) {
      for (uint32_t tx = uint32_t(o[0]) / tw; tx < uint32_t(o[2]) / tw;) {
        const uint32_t local = ty * pitch_tiles + tx;
        const auto owner = owners_tiles_edram4_[(drawn->edram_base + local) & kMaskEDRAM4];
        Image* w = owner.image;
        const bool depth_owner = w && w != drawn && w->image != drawn->image && w->edram_depth &&
            REXCVAR_GET(usf4_edram4_clear_into_depth) && !w->edram_64bpp &&
            (w->edram_format == 0 || w->edram_format == 1) && w->prepared && !w->invalid_content &&
            !w->guest_width && !w->guest_height && !w->raster_grid_x && SamplesSupported(*w);
        if (!depth_owner && (!w || w == drawn || w->image == drawn->image || w->edram_depth || w->edram_64bpp ||
            w->format != VK_FORMAT_R8G8B8A8_UNORM || w->edram_format != 0 || !w->prepared || w->invalid_content ||
            w->guest_width || w->guest_height || w->raster_grid_x ||
            !SamplesSupported(*w))) {
          static int logged = 0;  // usf4-nx diagnostic
          if (logged < 12 && ++logged)
            REXLOG_INFO("[usf4diag] color0 owner: none {} self {} depth {} 64bpp {} fmt {} edramfmt {} prepared {} "
                        "invalid {} guest {}x{} grid {} samples_ok {} | owner base {:03X} {}x{} msaa {}/{} | drawn base "
                        "{:03X} {}x{} msaa {}/{}", !w, w == drawn, w ? w->edram_depth : 0, w ? w->edram_64bpp : 0,
                        w ? int(w->format) : -1, w ? w->edram_format : 0, w ? w->prepared : 0,
                        w ? w->invalid_content : 0, w ? w->guest_width : 0, w ? w->guest_height : 0,
                        w ? w->raster_grid_x : 0, w ? SamplesSupported(*w) : 0, w ? w->edram_base : 0,
                        w ? w->width : 0, w ? w->height : 0, w ? w->edram_msaa_x : 0, w ? w->edram_msaa_y : 0,
                        drawn->edram_base, drawn->width, drawn->height, drawn->edram_msaa_x, drawn->edram_msaa_y);
          return plan_no("owner unsupported");
        }
        const uint32_t wp = PitchTilesEDRAM(*w);
        const uint32_t ww = 80u >> w->edram_msaa_x, wh = 16u >> w->edram_msaa_y;
        if (!wp || w->width % ww) return plan_no("owner width not tile-aligned");
        uint32_t count = 1;
        while (tx + count < uint32_t(o[2]) / tw && (owner.tile_local % wp) + count < wp) {
          const auto next = owners_tiles_edram4_[(drawn->edram_base + local + count) & kMaskEDRAM4];
          if (next.image != w || next.tile_local != owner.tile_local + count) break;
          ++count;
        }
        const uint32_t x = (owner.tile_local % wp) * ww, y = (owner.tile_local / wp) * wh, width = ww * count;
        if (x >= w->width || y >= w->height || width > w->width - x || wh > w->height - y) return plan_no("owner run outside the owner");
        plan.runs.push_back({w, local, count, {{int32_t(x), int32_t(y)}, {width, wh}}, depth_owner,
                             uint32_t(owner.tile_local)});
        tx += count;
      }
    }
    plan.drawn = drawn;
    if (plan.runs.empty()) color0_plan_reason_ = "no runs";
    return !plan.runs.empty();
  }
  const char* color0_plan_reason_ = "";
  // usf4-nx: the color-0 clear's runs all in the learned consumer of its site (the view that pulled these tiles
  // after the last two such clears), mapped tile by tile as the depth redirect's direct path does. false = the
  // usual per-owner plan.
  bool PlanColor0ConsumerEDRAM4(ColorClearEDRAM4& plan, const Image& drawn, const std::array<int32_t, 4>& o,
                                uint32_t tw, uint32_t th, uint32_t pitch_tiles) {
    const auto it = edram4_consumer_.find(plan.site);
    Image* c = it != edram4_consumer_.end() && it->second.confidence >= 2 ? it->second.view : nullptr;
    if (!c || c == &drawn || c->image == drawn.image || !c->prepared || c->invalid_content || c->guest_width ||
        c->guest_height || c->raster_grid_x || c->edram_64bpp || !SamplesSupported(*c)) {
      ++usf4_color0_consumer_no_[c ? "consumer view unsupported" : "no consumer seen twice"];
      return false;
    }
    if (c->edram_depth ? !REXCVAR_GET(usf4_edram4_clear_into_depth) || (c->edram_format != 0 && c->edram_format != 1)
                       : c->format != VK_FORMAT_R8G8B8A8_UNORM || c->edram_format != 0) {
      ++usf4_color0_consumer_no_["consumer format"];
      return false;
    }
    const uint32_t cp = PitchTilesEDRAM(*c);
    const uint32_t cw = 80u >> c->edram_msaa_x, ch = 16u >> c->edram_msaa_y;
    if (!cp || c->width % cw) {
      ++usf4_color0_consumer_no_["consumer pitch"];
      return false;
    }
    std::vector<ColorClearEDRAM4::Run> runs;
    const uint32_t tx_end = uint32_t(o[2]) / tw;
    for (uint32_t ty = uint32_t(o[1]) / th; ty < uint32_t(o[3]) / th; ++ty) {
      for (uint32_t tx = uint32_t(o[0]) / tw; tx < tx_end;) {
        const uint32_t local = ty * pitch_tiles + tx;
        const uint32_t lc = (drawn.edram_base + local - c->edram_base) & kMaskEDRAM4;
        uint32_t count = 1;
        while (tx + count < tx_end && (lc % cp) + count < cp) ++count;
        const uint32_t x = (lc % cp) * cw, y = (lc / cp) * ch, width = cw * count;
        if (x >= c->width || y >= c->height || width > c->width - x || ch > c->height - y) {
          ++usf4_color0_consumer_no_["tiles outside the consumer"];
          return false;
        }
        runs.push_back({c, local, count, {{int32_t(x), int32_t(y)}, {width, ch}}, bool(c->edram_depth), lc, true});
        tx += count;
      }
    }
    plan.runs = std::move(runs);
    ++(c->edram_depth ? usf4_color0_consumer_depth_ : usf4_color0_consumer_color_);
    return true;
  }
  uint64_t usf4_color0_consumer_color_ = 0, usf4_color0_consumer_depth_ = 0;
  std::map<std::string, uint64_t> usf4_color0_consumer_no_;
  bool CommitClearColor0EDRAM4(const ColorClearEDRAM4& plan) {
    // Runs of consecutive tile rows of one owner that line up form one rectangle: a full-screen clear is one clear.
    struct Area {
      Image* w;
      VkRect2D rect;
    };
    std::vector<Area> areas;
    for (const auto& run : plan.runs) {
      // (A depth owner's runs merge the same way: one owner, one value.)
      if (!areas.empty() && areas.back().w == run.owner && areas.back().rect.offset.x == run.area.offset.x &&
          areas.back().rect.extent.width == run.area.extent.width &&
          areas.back().rect.offset.y + int32_t(areas.back().rect.extent.height) == run.area.offset.y) {
        areas.back().rect.extent.height += run.area.extent.height;
      } else {
        areas.push_back({run.owner, run.area});
      }
    }
    draws_->FinishPass();
    if (!Record()) return false;
    for (const auto& a : areas) {
      Prepare(*a.w);
      if (!a.w->prepared || !RestoreContent(*a.w)) return false;
      ForgetClear(*a.w);
    }
    BarrierClearEDRAM4();
    const uint32_t stencil = plan.word & 0xFFu;
    for (const auto& a : areas) {
      if (a.w->edram_depth) {
        // usf4-nx: the word the clear writes, read as the owner's depth/stencil (D24S8: unorm24; D24FS8: 20e4).
        const uint32_t d24 = plan.word >> 8;
        const float guest = a.w->edram_format == 0 ? float(double(d24) / 16777215.0) : FromFloat24EDRAM4(d24);
        const float native = me::native::GuestDepthToNative(guest, a.w->depth_float24_half);
        if (!draws_->ClearDepthInPass(commands_work_, *a.w, native, stencil, &a.rect)) return false;
        continue;
      }
      if (!draws_->ClearColorInPass(commands_work_, *a.w, plan.value, a.rect)) return false;
    }
    BarrierClearEDRAM4();
    for (const auto& run : plan.runs) {
      for (uint32_t i = 0; i < run.count; ++i) {
        const uint32_t physical = (plan.drawn->edram_base + run.first_local + i) & kMaskEDRAM4;
        auto& d = owners_tiles_edram4_[physical];
        if (run.reassign) {  // usf4-nx: the consumer takes the tile (the clear wrote all of its word)
          d = {run.owner, uint16_t((run.owner_local + i) & kMaskEDRAM4), 0};
          ++edram4_epoch_;
        }
        d.version = ++edram4_version_;
        edram4_stamps_[physical] = {plan.site, d.version};  // usf4-nx: who pulls them next (consumer learning)
      }
      if (run.reassign && !run.owner->edram_depth) StencilRealEDRAM4(*run.owner, run.owner_local, run.count);
      if (run.depth_owner) {  // both aspects written, as the depth redirect's in-place clears
        run.owner->edram4_bits_stencil |= uint8_t(stencil);
        StencilRealEDRAM4(*run.owner, run.owner_local, run.count);
        usf4_color0_into_depth_tiles_ += run.count;
      }
      edram4_redirected_color0_tiles_ += run.count;
    }
    edram4_redirected_color0_areas_ += areas.size();
    return true;
  }
  uint64_t edram4_redirected_color0_tiles_ = 0, edram4_redirected_color0_areas_ = 0;
  uint64_t usf4_color0_into_depth_tiles_ = 0;  // usf4-nx: of those, tiles cleared into depth owners
  uint64_t edram4_redirected_merged_ = 0;  // clears the redirected depth clears issued after merging their regions

  // Mass Effect (every frame, main scene depth D000): a full-screen stencil-only D3D Clear (stencil ALWAYS /
  // REPLACE or ZERO, write mask 0xFF, depth test off) drawn through the 4x alias of the 1x depth buffer. Drawn
  // literally it moves all depth 1x -> 4x and back (two full transfers, one with 9 stencil passes). The
  // stencil of each tile is instead set in its current depth owner; depth stays, ownership stays.
  bool RedirectClearStencilEDRAM4(const SubmissionDraw& p) {
    if (!REXCVAR_GET(sa2_native_edram4_clear_stencil) || !p.registers || !p.edram_stencil_replace_rect ||
        !draws_) return false;
    const uint32_t* r = p.registers;
    const uint32_t mode = r[gr::XE_GPU_REG_RB_MODECONTROL] & 7;
    const uint32_t dc = r[gr::XE_GPU_REG_RB_DEPTHCONTROL];
    if (mode != 5 && !(mode == uint32_t(xenos::EdramMode::kColorDepth) && !r[gr::XE_GPU_REG_RB_COLOR_MASK]))
      return false;
    if (!(dc & 1) || (dc & 6)) return false;  // stencil on, depth test and write off: depth untouched
    const auto value_for = [](uint32_t zpass, uint32_t refmask) -> int32_t {
      return zpass == 2 ? int32_t(refmask & 0xFFu) : zpass == 1 ? 0 : -1;
    };
    const int32_t value = value_for((dc >> 14) & 7, r[gr::XE_GPU_REG_RB_STENCILREFMASK]);
    if (value < 0) return false;
    if ((dc & 0x80) && value_for((dc >> 26) & 7, r[gr::XE_GPU_REG_RB_STENCILREFMASK_BF]) != value) return false;
    const uint32_t info = r[gr::XE_GPU_REG_RB_DEPTH_INFO];
    const uint32_t pitch = r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF;
    if (!Record()) return false;
    Image* drawn = GetDepth(info & 0xFFF, (info >> 16) & 1, pitch);
    if (!drawn || drawn->guest_width || drawn->guest_height || drawn->raster_grid_x) return false;
    const auto& o = *p.edram_stencil_replace_rect;
    const uint32_t tw = 80u >> drawn->edram_msaa_x, th = 16u >> drawn->edram_msaa_y;
    if (o[0] < 0 || o[1] < 0 || o[2] <= o[0] || o[3] <= o[1] || o[0] % tw || o[1] % th || o[2] % tw ||
        o[3] % th || uint32_t(o[2]) > drawn->width || uint32_t(o[3]) > drawn->height) {
      ++edram4_stencil_redirected_no_;
      return false;
    }
    const uint32_t pitch_tiles = PitchTilesEDRAM(*drawn);
    struct Region { Image* owner; uint32_t first_local, owner_local, count; VkRect2D area; };
    std::vector<Region> regions;
    for (uint32_t ty = uint32_t(o[1]) / th; ty < uint32_t(o[3]) / th; ++ty) {
      for (uint32_t tx = uint32_t(o[0]) / tw; tx < uint32_t(o[2]) / tw;) {
        const uint32_t local = ty * pitch_tiles + tx;
        const auto owner = owners_tiles_edram4_[(drawn->edram_base + local) & kMaskEDRAM4];
        Image* w = owner.image;
        // Only another single-sample depth view can take the clear (the drawn view itself, a color owner or
        // no owner: draw it literally).
        if (!w || w == drawn || !w->edram_depth || w->edram_64bpp || w->guest_width || w->guest_height ||
            w->image == drawn->image || !w->prepared || w->invalid_content ||
            !SamplesSupported(*w)) {
          ++edram4_stencil_redirected_no_;
          ++edram4_stencil_no_reasons_[!w ? std::string("no owner") : w == drawn ? std::string("drawn view owns")
              : fmt::format("owner {}{:03X}/f{}:{}x{}:mx{}my{} samples{}", w->edram_depth ? "D" : "C",
                            w->edram_base, w->edram_format, w->width, w->height, w->edram_msaa_x, w->edram_msaa_y,
                            uint32_t(w->sample_count))];
          return false;
        }
        const uint32_t wp = PitchTilesEDRAM(*w);
        const uint32_t ww = 80u >> w->edram_msaa_x, wh = 16u >> w->edram_msaa_y;
        if (!wp || w->width % ww) { ++edram4_stencil_redirected_no_; return false; }
        uint32_t count = 1;
        while (tx + count < uint32_t(o[2]) / tw && (owner.tile_local % wp) + count < wp) {
          const auto n = owners_tiles_edram4_[(drawn->edram_base + local + count) & kMaskEDRAM4];
          if (n.image != w || n.tile_local != owner.tile_local + count) break;
          ++count;
        }
        const uint32_t x = (owner.tile_local % wp) * ww << w->raster_grid_x;
        const uint32_t y = (owner.tile_local / wp) * wh;
        const uint32_t width = (ww * count) << w->raster_grid_x;
        if (x >= w->width || y >= w->height || width > w->width - x || wh > w->height - y) {
          ++edram4_stencil_redirected_no_;
          return false;
        }
        regions.push_back({w, local, uint32_t(owner.tile_local), count, {{int32_t(x), int32_t(y)}, {width, wh}}});
        tx += count;
      }
    }
    if (regions.empty()) return false;
    draws_->FinishPass();
    if (!Record()) return false;
    for (const auto& region : regions) {
      Image& w = *region.owner;
      ForgetClear(w);
      BarrierClearEDRAM4();
      if (!draws_->ClearStencilInPass(commands_work_, w, uint32_t(value), region.area))
        return FailureEDRAM4("redirected stencil clear failed", *drawn, &w);
      BarrierClearEDRAM4();
      // The owner's stencil is now real (and known) for these tiles: no lazy record may bring older bits.
      StencilRealEDRAM4(w, region.owner_local, region.count);
      w.edram4_bits_stencil |= uint8_t(value);
      for (uint32_t i = 0; i < region.count; ++i)
        owners_tiles_edram4_[(drawn->edram_base + region.first_local + i) & kMaskEDRAM4].version = ++edram4_version_;
      edram4_stencil_redirected_tiles_ += region.count;
    }
    ++edram4_stencil_redirected_;
    return true;
  }

  // Lazy stencil, second half: before a draw (or an export) needs the real stencil of `view`'s tiles in
  // [start, start + count), copy it from where it stayed (8 bit passes after a stencil-only clear;
  // depth untouched). Runs are grouped by source image and contiguous source tiles.
  bool FetchStencilDeferredEDRAM4(Image& view, uint32_t start, uint32_t count) {
    const auto it = edram4_stencil_source_.find(&view);
    if (it == edram4_stencil_source_.end()) return true;
    std::vector<SpanEDRAM4> fetches;  // the runs to fetch (tile = view tile, tile_source = source tile)
    for (uint32_t i = 0; i < count;) {
      const SourceStencil f = it->second[(start + i) & kMaskEDRAM4];
      if (!f.image) { ++i; continue; }
      uint32_t n = 1;
      while (i + n < count) {
        const SourceStencil g = it->second[(start + i + n) & kMaskEDRAM4];
        if (g.image != f.image || g.tile != ((f.tile + n) & kMaskEDRAM4)) break;
        ++n;
      }
      fetches.push_back({const_cast<Image*>(f.image), uint32_t(f.tile), (start + i) & kMaskEDRAM4, n, false, {}, {}});
      i += n;
    }
    // usf4_edram4_import_columns: one-row runs from the same source at the same tile offset, over the same x range
    // in touching tile rows (the strips of a clear leave one partial tile per row), are fetched as one import.
    for (size_t a = 0; a < fetches.size();) {
      size_t b = a + 1;
      VkRect2D first{}, last{};
      const uint32_t pitch = PitchTilesEDRAM(view);
      const auto one_row = [&](const SpanEDRAM4& t) {
        return pitch && t.tile / pitch == (t.tile + t.count - 1) / pitch && t.tile + t.count <= kTilesEDRAM4;
      };
      if (REXCVAR_GET(usf4_edram4_import_columns) && one_row(fetches[a]) &&
          RectImportEDRAM4(view, fetches[a].tile, fetches[a].count, first)) {
        last = first;
        for (; b < fetches.size(); ++b) {
          const SpanEDRAM4& n = fetches[b];
          VkRect2D r{};
          if (n.source != fetches[a].source ||
              n.tile_source - n.tile != fetches[a].tile_source - fetches[a].tile || n.tile < fetches[a].tile ||
              !one_row(n) || !RectImportEDRAM4(view, n.tile, n.count, r) || r.offset.x != first.offset.x ||
              r.extent.width != first.extent.width || r.offset.y != last.offset.y + int32_t(last.extent.height))
            break;
          last = r;
        }
      }
      Image& src_data = *fetches[a].source;
      SpanEDRAM4 column = fetches[a];
      if (b - a > 1) {
        column.count = fetches[b - 1].tile + fetches[b - 1].count - column.tile;
        column.rect = first;
        column.rect.extent.height = uint32_t(last.offset.y + int32_t(last.extent.height) - first.offset.y);
        for (size_t k = a; k < b; ++k) column.runs.push_back({fetches[k].tile, fetches[k].count});
        ++edram4_columns_merged_;
        edram4_columns_runs_ += b - a;
      }
      edram4_import_stencil_only_ = true;
      const bool ok = src_data.prepared &&
          ImportColorDepthEDRAM4(src_data, view, column.tile_source, column.tile, column.count,
                                 column.runs.empty() ? nullptr : &column);
      edram4_import_stencil_only_ = false;
      if (!ok) return FailureEDRAM4("deferred stencil fetch failed", view, &src_data);
      for (size_t k = a; k < b; ++k) {
        StencilRealEDRAM4(view, fetches[k].tile, fetches[k].count);
        edram4_stencil_fetched_tiles_ += fetches[k].count;
      }
      ++edram4_stencil_fetched_;
      a = b;
    }
    return true;
  }

  // sa2_cpu_ring_edram_memo. Eligible draws: no per-draw proven rectangle, stencil replace rect or stencil clear proof
  // (they make the area or the stencil work specific to one draw). The key is every register PrepareDrawEDRAM4 reads.
  bool EdramMemoKey(const SubmissionDraw& p, std::array<uint32_t, 14>& key) const {
    if (!p.registers || !p.vs || p.edram_bounds_rect || p.edram_overwrite_rect || p.edram_overwrite_rect_count ||
        p.edram_stencil_replace_rect || p.edram_stencil_clear.eligible)
      return false;
    const uint32_t* r = p.registers;
    key = {r[gr::XE_GPU_REG_RB_MODECONTROL], r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL],
           r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR], r[gr::XE_GPU_REG_PA_SC_WINDOW_OFFSET],
           r[gr::XE_GPU_REG_RB_SURFACE_INFO], r[gr::XE_GPU_REG_RB_DEPTHCONTROL], r[gr::XE_GPU_REG_RB_DEPTH_INFO],
           r[gr::XE_GPU_REG_RB_COLOR_MASK], r[kDiagRegsColorInfo[0]], r[kDiagRegsColorInfo[1]],
           r[kDiagRegsColorInfo[2]], r[kDiagRegsColorInfo[3]], r[gr::XE_GPU_REG_RB_STENCILREFMASK],
           r[gr::XE_GPU_REG_RB_STENCILREFMASK_BF]};
    return true;
  }

  // What the full path checks before each of its fast paths, for every slot the memo holds: the image prepared with
  // valid content (Prepare / RestoreContent return at once), the draw's raster area owned in the current epoch
  // (SynchronizeEDRAM4 returns at once), stencil already marked used. No deferred stencil source was written since
  // (stencil_serial, checked by the caller), so FetchStencilDeferredEDRAM4 has nothing to fetch. GetTarget/GetDepth
  // give the same images for the same key: their maps never drop an entry.
  bool EdramMemoStillFast(const EdramDrawMemo& m) const {
    if (!recording_) return false;
    const uint32_t dc = m.key[5];
    for (uint32_t slot = 0; slot < 5; ++slot) {
      if (!(m.plan.active_mask & (1u << slot))) continue;
      const Image* image = m.images[slot];
      if (!image || !image->prepared || image->invalid_content) return false;
      if (image->edram4_epoch != edram4_epoch_) return false;
      if (!ContainsRect(image->edram4_own, RasterArea(*image, m.area))) return false;
      if (slot == 0 && (dc & 1) && !image->edram4_usa_stencil) return false;
    }
    return true;
  }

  // The state PrepareDrawEDRAM4 leaves, as the full path would leave it for this draw.
  void EdramMemoApply(const EdramDrawMemo& m) {
    edram4_cuts_extra_.clear();
    edram4_stencil_replacement_.reset();
    edram4_submission_ = nullptr;
    edram4_draw_images_ = m.images;
    edram4_draw_writes_ = m.writes;
    if (m.skipped_mode) return;  // the full path returns before touching the rest
    edram4_draw_writes_stencil_ = m.writes_stencil;
    edram4_draw_area_ = m.area;
    edram4_draw_plan_ = m.plan;
    depth_raster_grid_eligible_ = m.depth_raster_grid_eligible;
    edram4_reason_ = 0;
    edram4_slot_ = m.last_slot;
  }

  static bool SameRect(const VkRect2D& a, const VkRect2D& b) {
    return a.offset.x == b.offset.x && a.offset.y == b.offset.y && a.extent.width == b.extent.width &&
           a.extent.height == b.extent.height;
  }
  static bool SamePlan(const me::native::EdramBoundPlan& a, const me::native::EdramBoundPlan& b) {
    return a.valid == b.valid && a.requested_mask == b.requested_mask && a.active_mask == b.active_mask &&
           a.suppressed_mask == b.suppressed_mask && a.pitch_tiles_at_32bpp == b.pitch_tiles_at_32bpp &&
           a.length_tiles == b.length_tiles && a.sorted_slots == b.sorted_slots && a.count == b.count;
  }

  // A memo hit checked against the full path: run it, then require the same outputs and no work done by it (no
  // synchronization past its fast path, no deferred stencil fetched, no ownership change).
  bool EdramMemoCheck(const SubmissionDraw& p, const EdramDrawMemo& m) {
    const uint64_t sync_before = edram4_sync_calls_, fetched_before = edram4_stencil_fetched_;
    const uint64_t late_before = edram4_stencil_late_, epoch_before = edram4_epoch_;
    const uint64_t serial_before = edram4_stencil_sources_serial_;
    const bool ok = PrepareDrawEDRAM4(p);
    ++edram4_memo_checked_;
    sa2::enh::VerifyCheck();
    const char* diff = nullptr;
    if (!ok) diff = "the full path failed";
    else if (edram4_sync_calls_ != sync_before) diff = "a synchronization left its fast path";
    else if (edram4_stencil_fetched_ != fetched_before || edram4_stencil_late_ != late_before) diff = "stencil work";
    else if (edram4_epoch_ != epoch_before || edram4_stencil_sources_serial_ != serial_before) diff = "ownership changed";
    else if (edram4_draw_images_ != m.images || edram4_draw_writes_ != m.writes) diff = "images or writes";
    else if (!m.skipped_mode &&
             (edram4_draw_writes_stencil_ != m.writes_stencil || !SameRect(edram4_draw_area_, m.area) ||
              !SamePlan(edram4_draw_plan_, m.plan) || depth_raster_grid_eligible_ != m.depth_raster_grid_eligible ||
              edram4_slot_ != m.last_slot))
      diff = "area, plan or stencil";
    if (diff) {
      sa2::enh::VerifyDifference();
      REXLOG_ERROR("[native] DIFFERENCE: EDRAM draw memo ({}) at draw {}; sa2_cpu_ring_edram_memo off for the session",
                   diff, draws_ ? draws_->Drawn() : 0);
      edram4_memo_on_ = false;
      edram4_memo_.valid = false;
    }
    return ok;
  }

  bool PrepareDrawEDRAM4Memo(const SubmissionDraw& p) {
    std::array<uint32_t, 14> key;
    EdramDrawMemo& m = edram4_memo_;
    if (!g_edram_memo.active || !edram4_memo_on_ || !REXCVAR_GET(sa2_native_edram4_fast) ||
        TraceTileEDRAM4() != UINT32_MAX || !EdramMemoKey(p, key)) {
      m.valid = false;
      ++edram4_memo_ineligible_;
      return PrepareDrawEDRAM4(p);
    }
    const bool ps_null = !p.ps;
    const uint32_t ps_outputs = p.ps ? (p.ps->outputs & 15u) : 0u;
    const int64_t used = p.edram_used_height_estimate ? int64_t(*p.edram_used_height_estimate) : -1;
    if (m.valid && m.key == key && m.ps_null == ps_null && m.ps_outputs == ps_outputs && m.used_height == used &&
        m.stencil_serial == edram4_stencil_sources_serial_ && (m.skipped_mode || EdramMemoStillFast(m))) {
      ++edram4_memo_hits_;
      g_edram_memo.UseSampled();
      if (edram4_memo_checks_left_ > 0 || sa2::enh::VerifyMode()) {
        if (edram4_memo_checks_left_ > 0) --edram4_memo_checks_left_;
        return EdramMemoCheck(p, m);
      }
      EdramMemoApply(m);
      return true;
    }
    ++edram4_memo_misses_;
    const bool ok = PrepareDrawEDRAM4(p);
    if (!ok) {
      m.valid = false;
      return false;
    }
    const uint32_t mode = key[0] & 7;
    m.valid = true;
    m.key = key;
    m.ps_null = ps_null;
    m.ps_outputs = ps_outputs;
    m.used_height = used;
    m.stencil_serial = edram4_stencil_sources_serial_;
    m.skipped_mode = mode != uint32_t(xenos::EdramMode::kColorDepth) && mode != 5;
    m.images = edram4_draw_images_;
    m.writes = edram4_draw_writes_;
    m.writes_stencil = edram4_draw_writes_stencil_;
    m.depth_raster_grid_eligible = depth_raster_grid_eligible_;
    m.area = edram4_draw_area_;
    m.plan = edram4_draw_plan_;
    m.last_slot = edram4_slot_;
    return true;
  }

  bool PrepareDrawEDRAM4(const SubmissionDraw& p) {
    // Per-draw state must not leak past this call (early returns included): stale cutouts would make a
    // later clear skip imports, a stale replace rect would zero stencil, a stale request would dangle.
    edram4_cuts_extra_.clear();
    edram4_stencil_replacement_.reset();
    struct Clean {
      TargetsVulkan* d;
      ~Clean() { d->edram4_cuts_extra_.clear(); d->edram4_stencil_replacement_.reset(); d->edram4_submission_ = nullptr; }
    } clean{this};
    edram4_draw_images_.fill(nullptr);
    edram4_draw_writes_.fill(false);
    if (!p.registers || !p.vs) return false;
    const uint32_t* r = p.registers;
    const uint32_t mode = r[gr::XE_GPU_REG_RB_MODECONTROL] & 7;
    if (mode != uint32_t(xenos::EdramMode::kColorDepth) && mode != 5) return true;
    const uint32_t tl = r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL];
    const uint32_t br = r[gr::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR];
    const uint32_t window = r[gr::XE_GPU_REG_PA_SC_WINDOW_OFFSET];
    const int32_t wx = (tl >> 31) ? 0 : int32_t(window << 17) >> 17;
    const int32_t wy = (tl >> 31) ? 0 : int32_t((window >> 16) << 17) >> 17;
    const int32_t x0 = std::max(0, int32_t(tl & 0x3FFF) + wx);
    const int32_t y0 = std::max(0, int32_t((tl >> 16) & 0x3FFF) + wy);
    const int32_t x1 = std::max(x0, int32_t(br & 0x3FFF) + wx);
    const int32_t y1 = std::max(y0, int32_t((br >> 16) & 0x3FFF) + wy);
    edram4_draw_area_ = {{x0, y0}, {uint32_t(x1 - x0), uint32_t(y1 - y0)}};
    // A proven rectangle (CPU-executed VS) bounds every covered pixel: the draw touches nothing outside
    // it, so tiles outside need no transfer even when the scissor is the whole window (D3D Clear rects).
    if ((p.edram_bounds_rect || p.edram_overwrite_rect) && REXCVAR_GET(sa2_native_edram4_area_rect)) {
      const auto& o = p.edram_bounds_rect ? *p.edram_bounds_rect : *p.edram_overwrite_rect;
      const int32_t ax0 = std::max(x0, o[0]), ay0 = std::max(y0, o[1]);
      const int32_t ax1 = std::max(ax0, std::min(x1, o[2])), ay1 = std::max(ay0, std::min(y1, o[3]));
      edram4_draw_area_ = {{ax0, ay0}, {uint32_t(ax1 - ax0), uint32_t(ay1 - ay0)}};
    }
    me::native::EdramBoundRequest request;
    request.pitch_pixels = r[gr::XE_GPU_REG_RB_SURFACE_INFO] & 0x3FFF;
    request.msaa = msaa_edram_actual_;
    request.used_height_pixels = std::min(uint32_t(y1), kMaxTargetHeight);
    if (p.edram_used_height_estimate)
      request.used_height_pixels = std::min(request.used_height_pixels, *p.edram_used_height_estimate);
    const uint32_t dc = r[gr::XE_GPU_REG_RB_DEPTHCONTROL];
    request.depth_used = (dc & 3) != 0;
    request.views[0].base = r[gr::XE_GPU_REG_RB_DEPTH_INFO] & 0xFFF;
    const uint32_t mask = r[gr::XE_GPU_REG_RB_COLOR_MASK];
    for (uint32_t i = 0; i < 4; ++i) {
      const uint32_t info = r[kDiagRegsColorInfo[i]];
      request.views[i + 1] = {info & 0xFFF,
          xenos::IsColorRenderTargetFormat64bpp(xenos::ColorRenderTargetFormat((info >> 16) & 15))};
      if (mode == uint32_t(xenos::EdramMode::kColorDepth) && p.ps && (p.ps->outputs & (1u << i)))
        request.normalized_color_mask |= uint16_t(mask & (15u << (i * 4)));
    }
    depth_raster_grid_eligible_ = request.depth_used && !request.normalized_color_mask;
    edram4_draw_plan_ = me::native::PlanEdramBoundRanges(request);
    if (!edram4_draw_plan_.valid) {
      ::g_banjo_last_reject_cause = 9002;  // banjo-nx: frame trace
      if (banjo_plan_failures_++ < 12)
        REXLOG_INFO("[native] EDRAM4 plan invalid: pitch {} msaa {} used height {} (scissor {},{}-{},{} window {:08X}"
                    " est {}) depth base {} used {} color base {} mask {:04X}", request.pitch_pixels, request.msaa,
                    request.used_height_pixels, x0, y0, x1, y1, window,
                    p.edram_used_height_estimate ? int(*p.edram_used_height_estimate) : -1, request.views[0].base,
                    request.depth_used, request.views[1].base, request.normalized_color_mask);
      return false;
    }
    if (!Record()) return false;
    for (uint32_t slot = 0; slot < 5; ++slot) {
      if (!(edram4_draw_plan_.active_mask & (1u << slot))) continue;
      const uint32_t info = slot ? r[kDiagRegsColorInfo[slot - 1]] : r[gr::XE_GPU_REG_RB_DEPTH_INFO];
      Image* image = slot ? GetTarget(info & 0xFFF, (info >> 16) & 15, request.pitch_pixels)
                             : GetDepth(info & 0xFFF, (info >> 16) & 1, request.pitch_pixels);
      edram4_reason_ = 0;
      edram4_submission_ = &p;
      edram4_slot_ = slot;
      if (!image) { ::g_banjo_last_reject_cause = 9004; return false; }
      if (slot == 0 && (dc & 1) && !image->edram4_usa_stencil) {
        image->edram4_usa_stencil = true;  // from now on its imports carry stencil
        if (StencilDeferredEDRAM4(*image, 0, 2048)) ++edram4_stencil_late_;
      }
      // sa2-nx (sa2_cpu_lazy_trace_text): the text is made when a trace line is logged, not for every draw slot.
      if (edram4_traces_ < 60 && !g_lazy_trace_text.active) edram4_draw_desc_ = TraceDrawEDRAM4();
      edram4_stencil_replacement_.reset();
      if (slot == 0 && p.edram_stencil_replace_rect && REXCVAR_GET(sa2_native_edram4_stencil_lazy)) {
        const auto& sr = *p.edram_stencil_replace_rect;
        edram4_stencil_replacement_ = RasterArea(*image, VkRect2D{{sr[0], sr[1]},
            {uint32_t(sr[2] - sr[0]), uint32_t(sr[3] - sr[1])}});
      }
      edram4_cuts_extra_.clear();
      for (uint32_t k = 0; k < p.edram_overwrite_rect_count && k < p.edram_overwrite_rects.size(); ++k) {
        const auto& q = p.edram_overwrite_rects[k];
        edram4_cuts_extra_.push_back(RasterArea(*image, VkRect2D{{q[0], q[1]},
            {uint32_t(q[2] - q[0]), uint32_t(q[3] - q[1])}}));
      }
      if (p.edram_overwrite_rect && (p.edram_overwrite_slots & (1u << slot))) {
        const auto& o = *p.edram_overwrite_rect;
        const VkRect2D overwritten = RasterArea(*image, VkRect2D{{o[0], o[1]},
            {uint32_t(o[2] - o[0]), uint32_t(o[3] - o[1])}});
        if (!SynchronizeEDRAM4(*image, RasterArea(*image, edram4_draw_area_),
                               edram4_draw_plan_.length_tiles[slot], true, 0, &overwritten))
          return false;
        ++edram4_overwrites_;
      } else if (!SynchronizeEDRAM4(*image, RasterArea(*image, edram4_draw_area_),
                                    edram4_draw_plan_.length_tiles[slot])) {
        ::g_banjo_last_reject_cause = 9005;
        return false;
      }
      edram4_cuts_extra_.clear();
      edram4_stencil_replacement_.reset();
      if (slot == 0 && (dc & 1)) {
        // Tiles entirely inside a proven stencil-replace rect get a known stencil from this draw; every other
        // tile the draw touches needs its real stencil first (a partial replace keeps the rest).
        if (p.edram_stencil_replace_rect) {
          const auto& sr = *p.edram_stencil_replace_rect;
          const VkRect2D replacement = RasterArea(*image, VkRect2D{{sr[0], sr[1]},
              {uint32_t(sr[2] - sr[0]), uint32_t(sr[3] - sr[1])}});
          const bool edges_skip = REXCVAR_GET(usf4_edram4_stencil_edges_skip);
          for (uint32_t t = 0; t < edram4_draw_plan_.length_tiles[0] && t < 2048; ++t) {
            bool whole = false;
            if (TileUsedEDRAM4(*image, t, replacement, &whole) && (whole || edges_skip)) StencilRealEDRAM4(*image, t, 1);
          }
        }
        if (!FetchStencilDeferredEDRAM4(*image, 0, edram4_draw_plan_.length_tiles[0])) return false;
      }
      edram4_draw_images_[slot] = image;
      const uint32_t front = r[gr::XE_GPU_REG_RB_STENCILREFMASK];
      const uint32_t back = (dc & 0x80) ? r[gr::XE_GPU_REG_RB_STENCILREFMASK_BF] : front;
      const bool front_writes = ((front >> 16) & 255u) &&
          (((dc >> 11) & 7) || ((dc >> 14) & 7) || ((dc >> 17) & 7));
      const bool back_writes = ((back >> 16) & 255u) && ((dc & 0x80)
          ? (((dc >> 23) & 7) || ((dc >> 26) & 7) || ((dc >> 29) & 7)) : front_writes);
      edram4_draw_writes_[slot] = slot != 0 || ((dc & 6) == 6) ||
                                    ((dc & 1) && (front_writes || back_writes));
      if (slot == 0) edram4_draw_writes_stencil_ = (dc & 1) && (front_writes || back_writes);
      if (slot == 0 && (dc & 1)) {
        // Bits this draw can set: REPLACE writes ref & wmask; INCR/DECR/INVERT (3..7) any bit of wmask.
        const auto bits_face = [](uint32_t refmask, uint32_t ops) -> uint8_t {
          const uint32_t ref = refmask & 255u, wmask = (refmask >> 16) & 255u;
          uint32_t b = 0;
          for (uint32_t k = 0; k < 3; ++k) {
            const uint32_t op = (ops >> (k * 3)) & 7u;
            if (op == 2) b |= ref & wmask;
            else if (op >= 3) b |= wmask;
          }
          return uint8_t(b);
        };
        image->edram4_bits_stencil |= bits_face(front, (dc >> 11) & 0x1FFu);
        if (dc & 0x80) image->edram4_bits_stencil |= bits_face(back, (dc >> 23) & 0x1FFu);
      }
    }
    const std::array<uint32_t, 18> trace_state{
        r[gr::XE_GPU_REG_RB_SURFACE_INFO], r[gr::XE_GPU_REG_RB_DEPTH_INFO], dc, mask,
        r[gr::XE_GPU_REG_PA_CL_CLIP_CNTL], r[gr::XE_GPU_REG_PA_CL_VTE_CNTL],
        r[gr::XE_GPU_REG_PA_CL_VPORT_YSCALE], r[gr::XE_GPU_REG_PA_CL_VPORT_YOFFSET],
        tl, br, window, r[gr::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL],
        r[gr::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR], request.used_height_pixels,
        r[gr::XE_GPU_REG_PA_SU_SC_MODE_CNTL], r[gr::XE_GPU_REG_PA_SU_VTX_CNTL],
        r[gr::XE_GPU_REG_RB_COLOR_INFO], mode};
    static uint64_t trace_state_frame = UINT64_MAX;
    static std::array<uint32_t, 18> last_trace_state{};
    if (DrawTouchesTraceTileEDRAM4() &&
        (trace_state_frame != trace_frame_ || last_trace_state != trace_state) &&
        TraceEventEDRAM4(TraceTileEDRAM4())) {
      trace_state_frame = trace_frame_;
      last_trace_state = trace_state;
      REXLOG_INFO("[native] EDRAM TILE TRACE draw-request frame={} nextDraw={} physical={} "
                  "vs={} ps={} surface={:08X} depthinfo={:08X} depthcontrol={:08X} colorMask={:08X} "
                  "clip={:08X} vte={:08X} yscale={:08X} yoffset={:08X} windowTL={:08X} "
                  "windowBR={:08X} windowOffset={:08X} screenTL={:08X} screenBR={:08X} usedHeight={} "
                  "suMode={:08X} vtxCntl={:08X}",
                  trace_frame_, draws_ ? draws_->Drawn() + 1 : 0,
                  TraceTileEDRAM4(), p.vs->number, p.ps ? int32_t(p.ps->number) : -1,
                  r[gr::XE_GPU_REG_RB_SURFACE_INFO], r[gr::XE_GPU_REG_RB_DEPTH_INFO], dc, mask,
                  r[gr::XE_GPU_REG_PA_CL_CLIP_CNTL], r[gr::XE_GPU_REG_PA_CL_VTE_CNTL],
                  r[gr::XE_GPU_REG_PA_CL_VPORT_YSCALE], r[gr::XE_GPU_REG_PA_CL_VPORT_YOFFSET],
                  tl, br, window, r[gr::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL],
                  r[gr::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR], request.used_height_pixels,
                  r[gr::XE_GPU_REG_PA_SU_SC_MODE_CNTL], r[gr::XE_GPU_REG_PA_SU_VTX_CNTL]);
    }
    return true;
  }

  // Draw/scissor requests are guest pixel rectangles. Ownership and transfer
  // shaders operate on actual host raster texels, including retained X samples.
  static VkRect2D RasterArea(const ImageNative& image, VkRect2D area) {
    area.offset.x = int32_t(int64_t(area.offset.x) * (1u << image.raster_grid_x));
    area.extent.width *= 1u << image.raster_grid_x;
    return area;
  }

  static uint32_t PitchTilesEDRAM(const Image& image) {
    return me::native::EdramLayout(image.edram_msaa_x ? 2 : image.edram_msaa_y,
                                   image.edram_64bpp).pitch_tiles(image.width);
  }

  // Synchronize aliases by physical EDRAM tile, not merely by equal base. Mass Effect has overlapping
  // targets at different bases (notably 0x2D0 and 0x400), so exact-base aliasing alone leaves stale bands.
  // Merges VkImageCopy regions that tile a larger rectangle in both images: first runs along a row
  // (adjacent in source and destination, same rows), then stacks of equal runs. Same pixels copied.
  static void MergeRegionsCopy(std::vector<VkImageCopy>& regions) {
    if (regions.size() < 2) return;
    const auto per_row = [](const VkImageCopy& a, const VkImageCopy& b) {
      return std::tie(a.srcOffset.y, a.dstOffset.y, a.srcOffset.x) <
             std::tie(b.srcOffset.y, b.dstOffset.y, b.srcOffset.x);
    };
    std::sort(regions.begin(), regions.end(), per_row);
    std::vector<VkImageCopy> rows;
    rows.reserve(regions.size());
    for (const VkImageCopy& r : regions) {
      if (!rows.empty()) {
        VkImageCopy& u = rows.back();
        if (u.srcOffset.y == r.srcOffset.y && u.dstOffset.y == r.dstOffset.y &&
            u.extent.height == r.extent.height &&
            u.srcOffset.x + int32_t(u.extent.width) == r.srcOffset.x &&
            u.dstOffset.x + int32_t(u.extent.width) == r.dstOffset.x) {
          u.extent.width += r.extent.width;
          continue;
        }
      }
      rows.push_back(r);
    }
    std::sort(rows.begin(), rows.end(), [](const VkImageCopy& a, const VkImageCopy& b) {
      return std::tie(a.srcOffset.x, a.dstOffset.x, a.extent.width, a.srcOffset.y) <
             std::tie(b.srcOffset.x, b.dstOffset.x, b.extent.width, b.srcOffset.y);
    });
    regions.clear();
    for (const VkImageCopy& r : rows) {
      if (!regions.empty()) {
        VkImageCopy& u = regions.back();
        if (u.srcOffset.x == r.srcOffset.x && u.dstOffset.x == r.dstOffset.x &&
            u.extent.width == r.extent.width &&
            u.srcOffset.y + int32_t(u.extent.height) == r.srcOffset.y &&
            u.dstOffset.y + int32_t(u.extent.height) == r.dstOffset.y) {
          u.extent.height += r.extent.height;
          continue;
        }
      }
      regions.push_back(r);
    }
  }

  // Mass Effect report: mode 2/3 EDRAM overlap work since the last report.
  void ReportOverlapsEDRAM() {
    if (!overlaps_copies_ && !overlaps_conversions_) return;
    REXLOG_INFO("[native] EDRAM overlap sync: {} copy batches, {} tiles in {} merged regions; "
                "{} format conversions over {} tiles",
                overlaps_copies_, overlaps_tiles_, overlaps_regions_, overlaps_conversions_,
                overlaps_tiles_converted_);
    overlaps_copies_ = overlaps_tiles_ = overlaps_regions_ = 0;
    overlaps_conversions_ = overlaps_tiles_converted_ = 0;
  }
  uint64_t overlaps_copies_ = 0, overlaps_tiles_ = 0, overlaps_regions_ = 0;
  uint64_t overlaps_conversions_ = 0, overlaps_tiles_converted_ = 0;

  void SynchronizeOverlapsEDRAM(uint32_t base, Image& target, const VkRect2D* area = nullptr) {
    if (!copy_image_ || !target.width || !target.height) return;
    constexpr uint32_t kTileWidth = xenos::kEdramTileWidthSamples;
    constexpr uint32_t kTileHeight = xenos::kEdramTileHeightSamples;
    constexpr uint32_t kTilesMask = xenos::kEdramTileCount - 1;
    const uint32_t pitch_target = PitchTilesEDRAM(target);
    const uint32_t tile_height_target = kTileHeight >> target.edram_msaa_y;
    const uint32_t rows_target = (target.height + tile_height_target - 1) / tile_height_target;
    const uint32_t tiles_target =
        std::min(pitch_target * rows_target, xenos::kEdramTileCount);
    const auto tile_used = [&](uint32_t tile) {
      if (!area) return true;
      if (!area->extent.width || !area->extent.height) return false;
      const uint32_t tile_width = kTileWidth >> (uint32_t(target.edram_64bpp) + target.edram_msaa_x);
      const uint32_t x = (tile % pitch_target) * tile_width;
      const uint32_t y = (tile / pitch_target) * tile_height_target;
      return x < target.width && y < target.height &&
          x < uint32_t(area->offset.x) + area->extent.width &&
          x + tile_width > uint32_t(area->offset.x) &&
          y < uint32_t(area->offset.y) + area->extent.height &&
          y + tile_height_target > uint32_t(area->offset.y);
    };
    std::unordered_map<Image*, std::vector<VkImageCopy>> per_source;
    struct ConversionRange {
      Image* source;
      uint32_t tile_source;
      uint32_t tile_target;
      uint32_t count;
    };
    std::vector<ConversionRange> conversions;
    uint32_t distinct_formats = 0;
    for (uint32_t tile_target = 0; tile_target < tiles_target; ++tile_target) {
      if (!tile_used(tile_target)) continue;
      OwnerTileEDRAM& owner = owners_tiles_edram_[(base + tile_target) & kTilesMask];
      if (!owner.image || owner.image == &target) continue;
      Image& source = *owner.image;
      if (source.format != target.format ||
          source.edram_format != target.edram_format ||
          source.edram_64bpp != target.edram_64bpp ||
          source.edram_msaa_x != target.edram_msaa_x ||
          source.edram_msaa_y != target.edram_msaa_y || !source.width || !source.height) {
        const uint32_t source_start = owner.tile_local;
        uint32_t count = 1;
        // One compute dispatch can convert a physically contiguous run even when the
        // two image views use different pitches.
        while (tile_target + count < tiles_target && tile_used(tile_target + count)) {
          OwnerTileEDRAM& next =
              owners_tiles_edram_[(base + tile_target + count) & kTilesMask];
          if (next.image != &source ||
              uint32_t(next.tile_local) != source_start + count) {
            break;
          }
          ++count;
        }
        conversions.push_back({&source, source_start, tile_target, count});
        distinct_formats += count;
        tile_target += count - 1;
        continue;
      }
      const uint32_t pitch_source = PitchTilesEDRAM(source);
      const uint32_t tile_width_source = kTileWidth >> (uint32_t(source.edram_64bpp) + source.edram_msaa_x);
      const uint32_t tile_width_target = kTileWidth >> (uint32_t(target.edram_64bpp) + target.edram_msaa_x);
      const uint32_t sx = (uint32_t(owner.tile_local) % pitch_source) * tile_width_source;
      const uint32_t sy = (uint32_t(owner.tile_local) / pitch_source) * (kTileHeight >> source.edram_msaa_y);
      const uint32_t dx = (tile_target % pitch_target) * tile_width_target;
      const uint32_t dy = (tile_target / pitch_target) * tile_height_target;
      if (sx >= source.width || sy >= source.height || dx >= target.width || dy >= target.height) continue;
      VkImageCopy region{};
      region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.srcOffset = {int32_t(sx), int32_t(sy), 0};
      region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.dstOffset = {int32_t(dx), int32_t(dy), 0};
      region.extent = {std::min(tile_width_target, std::min(source.width - sx, target.width - dx)),
                       std::min(tile_height_target, std::min(source.height - sy, target.height - dy)), 1};
      per_source[&source].push_back(region);
    }
    if (!per_source.empty()) {
      if (draws_) draws_->FinishPass();
      ForgetClear(target);
      BarrierBeforeCopyResolve();
      uint32_t regions_total = 0;
      for (auto& [source, regions] : per_source) {
        regions_total += uint32_t(regions.size());
        overlaps_tiles_ += regions.size();
        // The same pixels in fewer regions: NVK pays a fixed cost per VkImageCopy region, and a whole
        // 1280x720 rebind is 720 tiles of 80x16 (measured on the Switch: ~110 ms per frame in mode 2).
        MergeRegionsCopy(regions);
        overlaps_regions_ += regions.size();
        HostCopies(*source, target, regions.data(), uint32_t(regions.size()));  // sa2-nx render resolution
        copy_image_(commands_work_, source->image, VK_IMAGE_LAYOUT_GENERAL, target.image,
                       VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
      }
      ++overlaps_copies_;
      BarrierAfterCopyResolve();
      const uint64_t label = (uint64_t(base) << 32) | uint32_t(target.format);
      if (overlaps_edram_recorded_.insert(label).second) {
        REXLOG_INFO("[native] targets: physical EDRAM base {:03X}: {} tiles from {} images",
                    base, regions_total, per_source.size());
      }
    }
    uint32_t tiles_converted = 0;
    bool conversion_failed = false;
    if (!conversions.empty()) {
      if (draws_) draws_->FinishPass();
      ForgetClear(target);
    }
    for (const ConversionRange& conversion : conversions) {
      ++overlaps_conversions_;
      overlaps_tiles_converted_ += conversion.count;
      if (ConvertAliasEDRAM(*conversion.source, target, conversion.tile_source,
                              conversion.tile_target, conversion.count)) {
        tiles_converted += conversion.count;
      } else {
        conversion_failed = true;
      }
    }
    if (distinct_formats) {
      const uint64_t label = (uint64_t(base) << 32) | 0xFFFFFFFFu;
      if (overlaps_edram_recorded_.insert(label).second) {
        REXLOG_INFO("[native] targets: physical EDRAM base {:03X}: {}/{} tiles converted between formats in {} ranges",
                    base, tiles_converted, distinct_formats, conversions.size());
      }
    }
    if (conversion_failed) {
      REXLOG_WARN("[native] targets: physical EDRAM base {:03X}: a conversion failed; the new owner is not published",
                  base);
      return;
    }
    // This image now contains the newest known snapshot for the full physical range. Marking on bind is
    // conservative but correct: unchanged tiles were copied above, and later draws write into this image.
    for (uint32_t tile_target = 0; tile_target < tiles_target; ++tile_target) {
      if (!tile_used(tile_target)) continue;
      owners_tiles_edram_[(base + tile_target) & kTilesMask] =
          {&target, uint16_t(tile_target)};
    }
  }

  // Xenos EDRAM has no image format or surface pitch attached to its tiles. Mass Effect rebinds base 2D0
  // as 1280-wide 7e3, then 640-wide RGBA8, then 1280-wide RGBA8. The old key included the pitch, leaving
  // three unrelated host images and producing horizontal strips made from stale frames. Track the newest
  // representation per base and move the physical 80x16 tiles when either format or pitch changes.
  void ActivateTargetColor(uint32_t base, uint32_t pitch, Image& target) {
    // Physical ownership is authoritative. An additional exact-base copy can
    // overwrite tiles that have since been written through another base.
    const int32_t mode = REXCVAR_GET(sa2_native_edram_alias_mode);
    if (!mode) return;
    if (mode == 4) return; // Strict common ownership is prepared outside the native draw pass.
    if (mode == 2 || mode == 3) {
      SynchronizeOverlapsEDRAM(base, target, mode == 3 ? &area_edram_ : nullptr);
      return;
    }
    const uint64_t key = base;
    auto [it, new_entry] = targets_color_active_.try_emplace(key, &target);
    Image* previous = it->second;
    if (new_entry || previous == &target) {
      return;
    }
    if (draws_) {
      draws_->FinishPass();
    }
    Prepare(*previous);
    Prepare(target);
    bool copied = false;
    // If format and pitch change together, first convert the complete old representation into the
    // destination format at the old pitch. It preserves the tiles outside the narrower view, which are
    // needed when the 1280-wide view is rebound after the 640-wide pass.
    const bool same_representation = previous->format == target.format &&
        previous->edram_format == target.edram_format &&
        previous->edram_64bpp == target.edram_64bpp;
    if (!same_representation && previous->width != target.width) {
      const uint64_t intermediate_key = (uint64_t(base) << 32) |
                                        (uint64_t(target.edram_format) << 24) |
                                        (uint64_t(uint32_t(target.format) & 0xFF) << 16) |
                                        previous->width;
      auto intermediate_it = targets_.find(intermediate_key);
      if (intermediate_it != targets_.end()) {
        Image& intermediate = intermediate_it->second;
        Prepare(intermediate);
        if (ConvertAliasEDRAM(*previous, intermediate)) {
          copied = CopyAliasEDRAMPitch(intermediate, target);
        }
      }
    }
    if (!copied && same_representation && previous->width != target.width) {
      copied = CopyAliasEDRAMPitch(*previous, target);
    }
    if (!copied && ConvertAliasEDRAM(*previous, target)) {
      copied = true;
    }
    if (copied) {
      ForgetClear(target);
      const uint64_t pair = (uint64_t(uint32_t(previous->format)) << 32) |
                              uint32_t(target.format);
      if (conversions_edram_recorded_.insert(pair).second) {
        REXLOG_INFO("[native] targets: exact EDRAM alias base {:03X}: {}/host {} -> {}/host {}", base,
                    previous->width, uint32_t(previous->format), target.width,
                    uint32_t(target.format));
      }
    } else if (blit_) {
      VkImageBlit region{};
      region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.srcOffsets[1] = {HostEdge(*previous, int32_t(std::min(previous->width, target.width))),
                              HostEdge(*previous, int32_t(std::min(previous->height, target.height))), 1};
      region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.dstOffsets[1] = {HostEdge(target, int32_t(std::min(previous->width, target.width))),
                              HostEdge(target, int32_t(std::min(previous->height, target.height))), 1};
      // Both aliases stay in GENERAL, so ending the render pass is not by itself a memory
      // dependency for this transfer. Publish the last attachment/transfer write before reading
      // the old representation, and publish the blit before the new representation is used as an
      // attachment or sampled texture. Without these barriers, Metal may observe an older EDRAM
      // alias: Mass Effect then alternates correctly shaded characters with flat-colour geometry.
      BarrierBeforeCopyResolve();
      blit_(commands_work_, previous->image, VK_IMAGE_LAYOUT_GENERAL, target.image,
            VK_IMAGE_LAYOUT_GENERAL, 1, &region, VK_FILTER_NEAREST);
      BarrierAfterCopyResolve();
      ForgetClear(target);
      const uint64_t pair = (uint64_t(uint32_t(previous->format)) << 32) |
                              uint32_t(target.format);
      if (conversions_edram_recorded_.insert(pair).second) {
        REXLOG_INFO("[native] targets: alias EDRAM base {:03X}/{}: host {} -> {}", base, pitch,
                    uint32_t(previous->format), uint32_t(target.format));
      }
    } else {
      Reject(4, "EDRAM alias between formats without vkCmdBlitImage");
    }
    it->second = &target;
  }

  bool CopyAliasEDRAMPitch(Image& source, Image& target) {
    if (!copy_image_ || source.format != target.format ||
        source.edram_format != target.edram_format ||
        source.edram_64bpp != target.edram_64bpp || !source.width || !target.width) {
      return false;
    }
    const uint32_t kTileWidth = xenos::kEdramTileWidthSamples >> uint32_t(source.edram_64bpp);
    constexpr uint32_t kTileHeight = xenos::kEdramTileHeightSamples;
    const uint32_t pitch_source = (source.width + kTileWidth - 1) / kTileWidth;
    const uint32_t pitch_target = (target.width + kTileWidth - 1) / kTileWidth;
    const uint32_t rows_source = (source.height + kTileHeight - 1) / kTileHeight;
    const uint32_t rows_target = (target.height + kTileHeight - 1) / kTileHeight;
    const uint32_t tiles = std::min(pitch_source * rows_source, pitch_target * rows_target);
    std::vector<VkImageCopy> regions;
    regions.reserve(std::min<uint32_t>(tiles, 256));
    for (uint32_t tile = 0; tile < tiles;) {
      const uint32_t sx_tile = tile % pitch_source;
      const uint32_t sy_tile = tile / pitch_source;
      const uint32_t dx_tile = tile % pitch_target;
      const uint32_t dy_tile = tile / pitch_target;
      const uint32_t run = std::min({tiles - tile, pitch_source - sx_tile,
                                         pitch_target - dx_tile});
      const uint32_t sx = sx_tile * kTileWidth;
      const uint32_t sy = sy_tile * kTileHeight;
      const uint32_t dx = dx_tile * kTileWidth;
      const uint32_t dy = dy_tile * kTileHeight;
      VkImageCopy region{};
      region.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.srcOffset = {int32_t(sx), int32_t(sy), 0};
      region.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      region.dstOffset = {int32_t(dx), int32_t(dy), 0};
      region.extent = {std::min({run * kTileWidth, source.width - sx,
                                 target.width - dx}),
                       std::min({kTileHeight, source.height - sy, target.height - dy}), 1};
      if (region.extent.width && region.extent.height) {
        regions.push_back(region);
      }
      tile += run;
    }
    if (regions.empty()) {
      return false;
    }
    HostCopies(source, target, regions.data(), uint32_t(regions.size()));  // sa2-nx render resolution
    BarrierBeforeCopyResolve();
    copy_image_(commands_work_, source.image, VK_IMAGE_LAYOUT_GENERAL, target.image,
                   VK_IMAGE_LAYOUT_GENERAL, uint32_t(regions.size()), regions.data());
    BarrierAfterCopyResolve();
    return true;
  }

  bool ResolverWithBias(Image& source, Image& target, int32_t sx, int32_t sy,
                        uint32_t dx, uint32_t dy, uint32_t width, uint32_t height,
                        int32_t exp_bias) {
    if (source.format != VK_FORMAT_R16G16B16A16_SFLOAT ||
        target.format != VK_FORMAT_R16G16B16A16_SFLOAT ||
        source.image == target.image || !width || !height) return false;
    if (!pipelines_conversion_edram_[3] ||
        !EnsureCapacityConversionEDRAM4("resolve bias")) return false;
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo reserve{};
    reserve.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    reserve.descriptorPool = slot.pool_conversion_edram;
    reserve.descriptorSetCount = 1;
    reserve.pSetLayouts = &layout_conversion_edram_;
    if (dfn_.vkAllocateDescriptorSets(device_, &reserve, &set) != VK_SUCCESS) return false;
    ++slot.conversions_edram;
    const VkDescriptorImageInfo images[2] = {
        {VK_NULL_HANDLE, source.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[i].pImageInfo = &images[i];
    }
    dfn_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    // Order previous sampling as well as attachment/resolve writes against this
    // replacement. Keep the dependency outside render passes.
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before,
                              0, nullptr, 0, nullptr);
    // sa2-nx render resolution: the rectangles in host pixels (each image at its own scale).
    const VkRect2D host_source = HostRect(source, VkRect2D{{sx, sy}, {width, height}});
    const VkRect2D host_target = HostRect(target, VkRect2D{{int32_t(dx), int32_t(dy)}, {width, height}});
    const uint32_t host_width = std::min(host_source.extent.width, host_target.extent.width);
    const uint32_t host_height = std::min(host_source.extent.height, host_target.extent.height);
    const struct {
      int32_t source_offset[2];
      int32_t destination_offset[2];
      uint32_t extent[2];
      int32_t exponent_bias;
    } constants = {{host_source.offset.x, host_source.offset.y}, {host_target.offset.x, host_target.offset.y},
                   {host_width, host_height}, exp_bias};
    static_assert(sizeof(constants) == 28);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelines_conversion_edram_[3]);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 layout_pipeline_conversion_edram_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conversion_edram_,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    dfn_.vkCmdDispatch(commands_work_, (host_width + 7) / 8, (host_height + 7) / 8, 1);
    VkMemoryBarrier after{};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after,
                              0, nullptr, 0, nullptr);
    return true;
  }

  VkImageView GetViewRaw64EDRAM4(Image& image) {
    if (!raw64_edram_supported_ || image.format != VK_FORMAT_R16G16B16A16_SFLOAT) return VK_NULL_HANDLE;
    auto [it, fresh] = views_raw64_edram_.try_emplace(image.image, VK_NULL_HANDLE);
    if (fresh) {
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = image.image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = VK_FORMAT_R16G16B16A16_UINT;
      view.subresourceRange = kRangeColor;
      VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
      usage.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
      view.pNext = &usage;
      if (dfn_.vkCreateImageView(device_, &view, nullptr, &it->second) != VK_SUCCESS) {
        views_raw64_edram_.erase(it);
        return VK_NULL_HANDLE;
      }
    }
    return it->second;
  }

  ViewsDepthEDRAM* GetViewsDepthEDRAM4(Image& source) {
    auto [it, new_value] = views_depth_edram_.try_emplace(source.image);
    if (new_value) {
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = source.image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = source.format;
      view.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
      if (dfn_.vkCreateImageView(device_, &view, nullptr, &it->second.depth) != VK_SUCCESS) {
        views_depth_edram_.erase(it);
        return nullptr;
      }
      view.subresourceRange.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      if (dfn_.vkCreateImageView(device_, &view, nullptr, &it->second.stencil) != VK_SUCCESS) {
        dfn_.vkDestroyImageView(device_, it->second.depth, nullptr);
        views_depth_edram_.erase(it);
        return nullptr;
      }
    }
    return &it->second;
  }

  static bool IsDepthNative2xEDRAM4(const Image& image) {
    return image.sample_count == VK_SAMPLE_COUNT_2_BIT && image.edram_depth &&
        image.format == VK_FORMAT_D32_SFLOAT_S8_UINT && !image.edram_64bpp &&
        !image.guest_width && !image.guest_height && image.raster_grid_x == 0 &&
        image.edram_msaa_x == 0 && image.edram_msaa_y == 1 && image.edram_format <= 1 &&
        (!image.depth_float24_half || image.edram_format == 1);
  }

  static bool IsDepthNative1xEDRAM4(const Image& image) {
    // Actual 1x AND guest physical 1x, not a legacy collapsed 2x/4x image.
    return image.sample_count == VK_SAMPLE_COUNT_1_BIT && image.edram_depth &&
        image.format == VK_FORMAT_D32_SFLOAT_S8_UINT && !image.edram_64bpp &&
        !image.guest_width && !image.guest_height && image.raster_grid_x == 0 &&
        image.edram_msaa_x == 0 && image.edram_msaa_y == 0 && image.edram_format <= 1 &&
        (!image.depth_float24_half || image.edram_format == 1);
  }

  bool AcceptsImportDepthNative2xEDRAM4(VkFormat format) {
    if (format != VK_FORMAT_D32_SFLOAT_S8_UINT) return false;
    if (native2x_supported_ >= 0) return native2x_supported_ != 0;  // device capabilities do not change
    native2x_supported_ = AcceptsImportDepthNative2xNoCacheEDRAM4(format) ? 1 : 0;
    return native2x_supported_ != 0;
  }
  int8_t native2x_supported_ = -1;
  bool AcceptsImportDepthNative2xNoCacheEDRAM4(VkFormat format) {
    const auto& properties = vulkan_device_->properties();
    if (!properties.sampleRateShading || !properties.standardSampleLocations) return false;
    for (VkSampleCountFlags mask : {properties.framebufferDepthSampleCounts,
        properties.framebufferStencilSampleCounts, properties.sampledImageDepthSampleCounts,
        properties.sampledImageStencilSampleCounts})
      if (!me::native::IsHardwareSampleCountSupportedByMask(VK_SAMPLE_COUNT_2_BIT, mask)) return false;
    const auto& ifn = vulkan_device_->vulkan_instance()->functions();
    VkFormatProperties format_properties{};
    ifn.vkGetPhysicalDeviceFormatProperties(vulkan_device_->physical_device(), format, &format_properties);
    constexpr VkFormatFeatureFlags required = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                              VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if ((format_properties.optimalTilingFeatures & required) != required) return false;
    VkImageFormatProperties image_properties{};
    const auto image_format_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(
        ifn.vkGetInstanceProcAddr(vulkan_device_->vulkan_instance()->instance(),
                                 "vkGetPhysicalDeviceImageFormatProperties"));
    return image_format_properties && image_format_properties(vulkan_device_->physical_device(), format,
        VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, 0,
        &image_properties) == VK_SUCCESS &&
        me::native::IsHardwareSampleCountSupportedByMask(VK_SAMPLE_COUNT_2_BIT, image_properties.sampleCounts);
  }

  PassImportDepthEDRAM* EnsureImportDepthEDRAM4(VkFormat format,
      VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT,
      VkSampleCountFlagBits source_sample_count = VK_SAMPLE_COUNT_1_BIT) {
    // sa2_msaa: a single-sample source imported into a host multisampled game target (every sample of a pixel gets
    // the pixel's value, as the target's single guest sample).
    const bool host_msaa = msaa_samples_ > 1 && uint32_t(sample_count) == msaa_samples_ &&
                           me::native::IsSingleSample(uint32_t(source_sample_count));
    const bool native2x = sample_count == VK_SAMPLE_COUNT_2_BIT && !host_msaa;
    const bool source_msaa2 = source_sample_count == VK_SAMPLE_COUNT_2_BIT;
    if ((!me::native::IsSingleSample(uint32_t(sample_count)) && !native2x && !host_msaa) ||
        (!me::native::IsSingleSample(uint32_t(source_sample_count)) && !source_msaa2) ||
        (native2x && !source_msaa2) ||
        (source_msaa2 && !AcceptsImportDepthNative2xEDRAM4(format))) return nullptr;
    if (import_depth_edram_failed_) return nullptr;
    const auto fail = [&]() -> PassImportDepthEDRAM* {
      import_depth_edram_failed_ = true;
      return nullptr;
    };
    if (!layout_import_depth_edram_) {
      VkDescriptorSetLayoutBinding binding[2]{};
      for (uint32_t i = 0; i < 2; ++i) {
        binding[i].binding = i;
        binding[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        binding[i].descriptorCount = 1;
        binding[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
      }
      VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      layout.bindingCount = 2;
      layout.pBindings = binding;
      if (dfn_.vkCreateDescriptorSetLayout(device_, &layout, nullptr, &layout_import_depth_edram_) != VK_SUCCESS)
        return fail();
      VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 80};  // sa2-nx: + the scaled tile size
      VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipeline_layout.setLayoutCount = 1;
      pipeline_layout.pSetLayouts = &layout_import_depth_edram_;
      pipeline_layout.pushConstantRangeCount = 1;
      pipeline_layout.pPushConstantRanges = &range;
      if (dfn_.vkCreatePipelineLayout(device_, &pipeline_layout, nullptr,
                                       &layout_pipeline_import_depth_edram_) != VK_SUCCESS) return fail();
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_import_vs);
      module.pCode = shaders::me_edram_import_vs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &vs_import_depth_edram_) != VK_SUCCESS)
        return fail();
      module.codeSize = sizeof(shaders::me_edram_color_to_depth_fs);
      module.pCode = shaders::me_edram_color_to_depth_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_import_depth_edram_) != VK_SUCCESS)
        return fail();
      module.codeSize = sizeof(shaders::me_edram_raw64_to_depth_fs);
      module.pCode = shaders::me_edram_raw64_to_depth_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_raw64_to_depth_edram_) != VK_SUCCESS)
        return fail();
      module.codeSize = sizeof(shaders::me_edram_depth_to_depth_fs);
      module.pCode = shaders::me_edram_depth_to_depth_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_depth_to_depth_edram_) != VK_SUCCESS)
        return fail();
      module.codeSize = sizeof(shaders::me_edram_depth_to_stencil_fs);
      module.pCode = shaders::me_edram_depth_to_stencil_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_depth_to_stencil_edram_) != VK_SUCCESS)
        return fail();
      module.codeSize = sizeof(shaders::me_edram_color_to_stencil_fs);
      module.pCode = shaders::me_edram_color_to_stencil_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_color_to_stencil_edram_) != VK_SUCCESS)
        return fail();
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kConversionsEDRAMPerSlot * 2};
      VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool.maxSets = kConversionsEDRAMPerSlot;
      pool.poolSizeCount = 1;
      pool.pPoolSizes = &size;
      for (auto& slot : slots_)
        if (dfn_.vkCreateDescriptorPool(device_, &pool, nullptr, &slot.pool_import_depth_edram) != VK_SUCCESS)
          return fail();
    }
    if (native2x && !fs_stencil_msaa2_edram_) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_depth_to_stencil_msaa2_fs);
      module.pCode = shaders::me_edram_depth_to_stencil_msaa2_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_stencil_msaa2_edram_) != VK_SUCCESS)
        return fail();
    }
    if (source_msaa2 && !native2x && !fs_stencil_msaa2_to_1x_edram_) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_depth_msaa2_to_stencil_1x_fs);
      module.pCode = shaders::me_edram_depth_msaa2_to_stencil_1x_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_stencil_msaa2_to_1x_edram_) != VK_SUCCESS)
        return fail();
    }
    if (native2x && !fs_depth_to_depth_msaa2_edram_) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_depth_to_depth_msaa2_fs);
      module.pCode = shaders::me_edram_depth_to_depth_msaa2_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_depth_to_depth_msaa2_edram_) != VK_SUCCESS)
        return fail();
    }
    if (source_msaa2 && !native2x && !fs_depth_msaa2_to_depth_1x_edram_) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_depth_msaa2_to_depth_1x_fs);
      module.pCode = shaders::me_edram_depth_msaa2_to_depth_1x_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_depth_msaa2_to_depth_1x_edram_) != VK_SUCCESS)
        return fail();
    }
    // Source type is part of the PSO identity: 1x->1x and 2x->1x both have a
    // 1x attachment but use incompatible sampler2D / sampler2DMS shaders.
    const uint64_t pass_key = uint64_t(uint32_t(format)) | (uint64_t(sample_count) << 32) |
                              (uint64_t(source_sample_count) << 40);
    auto [it, new_value] = passes_import_depth_edram_.try_emplace(pass_key);
    PassImportDepthEDRAM& pass = it->second;
    if (!new_value) return pass.render_pass && pass.pipelines[2] && pass.pipelines[3] &&
        (source_msaa2 || (pass.pipelines[0] && pass.pipelines[1] && pass.pipelines[4] && pass.pipelines[5]))
        ? &pass : nullptr;
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = sample_count;
    attachment.loadOp = attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    VkSubpassDependency dependencies[2]{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    // Same narrowing as the draw passes (sa2_native_pass_narrow_dependency): compute/copy writes
    // into the source end with their own barriers; only attachment and transfer writes need this one.
    dependencies[0].srcAccessMask = REXCVAR_GET(sa2_native_pass_narrow_dependency)
        ? VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT)
        : VkAccessFlags(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    dependencies[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkRenderPassCreateInfo render_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass.attachmentCount = 1;
    render_pass.pAttachments = &attachment;
    render_pass.subpassCount = 1;
    render_pass.pSubpasses = &subpass;
    render_pass.dependencyCount = 2;
    render_pass.pDependencies = dependencies;
    if (dfn_.vkCreateRenderPass(device_, &render_pass, nullptr, &pass.render_pass) != VK_SUCCESS)
      return fail();
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_import_depth_edram_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_import_depth_edram_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = sample_count;
    multisample.sampleShadingEnable = native2x ? VK_TRUE : VK_FALSE;
    multisample.minSampleShading = native2x ? 1.0f : 0.0f;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    depth.stencilTestEnable = VK_TRUE;
    depth.front.failOp = depth.front.depthFailOp = VK_STENCIL_OP_KEEP;
    depth.front.passOp = VK_STENCIL_OP_REPLACE;
    depth.front.compareOp = VK_COMPARE_OP_ALWAYS;
    depth.front.compareMask = depth.front.writeMask = 255;
    depth.back = depth.front;
    VkPipelineColorBlendStateCreateInfo color{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = uint32_t(std::size(dynamic_states));
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout_pipeline_import_depth_edram_;
    pipeline.renderPass = pass.render_pass;
    for (uint32_t i = 0; i < 6; ++i) {
      if (source_msaa2 && (i < 2 || i >= 4)) continue;  // Only same-format depth sources.
      depth.depthTestEnable = depth.depthWriteEnable = (i & 1) == 0 ? VK_TRUE : VK_FALSE;
      stages[1].module = source_msaa2 ? (native2x ? fs_depth_to_depth_msaa2_edram_ :
          fs_depth_msaa2_to_depth_1x_edram_) : i < 2 ? fs_import_depth_edram_ :
          i < 4 ? fs_depth_to_depth_edram_ : fs_raw64_to_depth_edram_;
      // Stencil-bit pipelines (odd): a shader that only fetches the stencil byte and discards, without
      // gl_FragDepth (which forced late tests and the full depth path on every one of the 8 passes).
      if ((i & 1) && REXCVAR_GET(sa2_native_edram4_stencil_light))
        stages[1].module = source_msaa2 ? (native2x ? fs_stencil_msaa2_edram_ : fs_stencil_msaa2_to_1x_edram_)
            : i == 1 ? fs_color_to_stencil_edram_ : i == 3 ? fs_depth_to_stencil_edram_ : stages[1].module;
      if (dfn_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr,
                                           &pass.pipelines[i]) != VK_SUCCESS) return fail();
    }
    return &pass;
  }

  // ---- Copy-engine stencil import (1x destinations) ---------------------------------------------------
  static constexpr VkDeviceSize kBufferStencilCopy = 2u << 20;  // 1280x1280 bytes fit (1.6 MB)
  VkDescriptorSetLayout layout_stencil_copy_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_stencil_copy_ = VK_NULL_HANDLE;
  VkShaderModule shaders_stencil_copy_[3] = {};
  VkPipeline pipelines_stencil_copy_[3] = {};  // 0: 1x depth source, 1: 2x MSAA depth source, 2: color source
  bool stencil_failed_copy_ = false;
  uint64_t edram4_stencil_copies_ = 0, edram4_stencil_copies_tiles_ = 0;

  bool EnsureStencilCopyEDRAM4() {
    if (stencil_failed_copy_) return false;
    SlotWork& slot = slots_[slot_];
    if (pipelines_stencil_copy_[0] && slot.buffer_stencil_copy) return true;
    const auto fail = [&]() { stencil_failed_copy_ = true; REXLOG_WARN("[native] copy-engine stencil import unavailable"); return false; };
    if (!layout_stencil_copy_) {
      VkDescriptorSetLayoutBinding b[2]{};
      b[0] = {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      b[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
      VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      li.bindingCount = 2;
      li.pBindings = b;
      if (dfn_.vkCreateDescriptorSetLayout(device_, &li, nullptr, &layout_stencil_copy_) != VK_SUCCESS) return fail();
      VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, 64};  // sa2-nx: + the scaled tile size
      VkPipelineLayoutCreateInfo pl{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pl.setLayoutCount = 1;
      pl.pSetLayouts = &layout_stencil_copy_;
      pl.pushConstantRangeCount = 1;
      pl.pPushConstantRanges = &range;
      if (dfn_.vkCreatePipelineLayout(device_, &pl, nullptr, &layout_pipeline_stencil_copy_) != VK_SUCCESS) return fail();
      const std::pair<const uint32_t*, size_t> codes[3] = {
          {shaders::me_edram_stencil_to_buffer_cs, sizeof(shaders::me_edram_stencil_to_buffer_cs)},
          {shaders::me_edram_stencil_msaa2_to_buffer_cs, sizeof(shaders::me_edram_stencil_msaa2_to_buffer_cs)},
          {shaders::me_edram_color_stencil_to_buffer_cs, sizeof(shaders::me_edram_color_stencil_to_buffer_cs)}};
      for (uint32_t i = 0; i < 3; ++i) {
        VkShaderModuleCreateInfo m{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        m.codeSize = codes[i].second;
        m.pCode = codes[i].first;
        if (dfn_.vkCreateShaderModule(device_, &m, nullptr, &shaders_stencil_copy_[i]) != VK_SUCCESS) return fail();
        VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cp.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cp.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module = shaders_stencil_copy_[i];
        cp.stage.pName = "main";
        cp.layout = layout_pipeline_stencil_copy_;
        if (dfn_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cp, nullptr, &pipelines_stencil_copy_[i]) != VK_SUCCESS)
          return fail();
      }
    }
    if (!slot.buffer_stencil_copy) {
      VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
      bi.size = kBufferStencilCopy;
      bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
      bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
      if (dfn_.vkCreateBuffer(device_, &bi, nullptr, &slot.buffer_stencil_copy) != VK_SUCCESS) return fail();
      VkMemoryRequirements req;
      dfn_.vkGetBufferMemoryRequirements(device_, slot.buffer_stencil_copy, &req);
      const uint32_t type = rex::ui::vulkan::util::ChooseMemoryType(vulkan_device_->memory_types(), req.memoryTypeBits,
                                                                     rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal);
      if (type == UINT32_MAX) return fail();
      VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
      ai.allocationSize = req.size;
      ai.memoryTypeIndex = type;
      if (dfn_.vkAllocateMemory(device_, &ai, nullptr, &slot.memory_stencil_copy) != VK_SUCCESS) return fail();
      if (dfn_.vkBindBufferMemory(device_, slot.buffer_stencil_copy, slot.memory_stencil_copy, 0) != VK_SUCCESS)
        return fail();
      VkDescriptorPoolSize sizes[2] = {{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kConversionsEDRAMPerSlot},
                                       {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kConversionsEDRAMPerSlot}};
      VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pi.maxSets = kConversionsEDRAMPerSlot;
      pi.poolSizeCount = 2;
      pi.pPoolSizes = sizes;
      if (dfn_.vkCreateDescriptorPool(device_, &pi, nullptr, &slot.pool_stencil_copy) != VK_SUCCESS) return fail();
    }
    return true;
  }

  // Writes the stencil of destination tiles [target_start, +count) (1x, TRANSFER_DST) from the source's
  // stencil through the copy engine: compute -> buffer -> vkCmdCopyBufferToImage(STENCIL), one region per
  // tile-row run so tiles outside the range keep their stencil. Depth is untouched.
  bool CopyStencilEDRAM4(Image& source, Image& target, uint32_t source_start, uint32_t target_start,
                           uint32_t count, const VkRect2D& guest_rect) {
    // sa2-nx render resolution: the import rectangle and the tiles in host pixels (virtual EDRAM, 20q x 4q tiles).
    const VkRect2D rect = HostRect(target, guest_rect);
    const uint32_t tile_width = ScaledTileWidth(target.scale_q), tile_height = ScaledTileHeight(target.scale_q);
    if (source.scale_q != target.scale_q) return false;
    const bool msaa2 = source.sample_count == VK_SAMPLE_COUNT_2_BIT;
    const bool color = !source.edram_depth;
    ViewsDepthEDRAM* views = color ? nullptr : GetViewsDepthEDRAM4(source);
    const VkImageView view_source = color ? source.view : views ? views->stencil : VK_NULL_HANDLE;
    if (!view_source || (color && msaa2) || rect.extent.width % 4 ||
        VkDeviceSize(rect.extent.width) * rect.extent.height > kBufferStencilCopy) return false;
    SlotWork& slot = slots_[slot_];
    MarkGpu(kGpuEdramImport9);  // diagnostic split: the copy-engine stencil part of an import
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    alloc.descriptorPool = slot.pool_stencil_copy;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts = &layout_stencil_copy_;
    if (dfn_.vkAllocateDescriptorSets(device_, &alloc, &set) != VK_SUCCESS) return false;
    const VkDescriptorImageInfo img{sampler_depth_edram_, view_source, VK_IMAGE_LAYOUT_GENERAL};
    const VkDescriptorBufferInfo buf{slot.buffer_stencil_copy, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      w[i].dstSet = set;
      w[i].dstBinding = i;
      w[i].descriptorCount = 1;
    }
    w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w[0].pImageInfo = &img;
    w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w[1].pBufferInfo = &buf;
    dfn_.vkUpdateDescriptorSets(device_, 2, w, 0, nullptr);
    struct {
      uint32_t source_size[2], pitch_tiles_source, pitch_tiles_target, tile_source_start, tile_target_start,
          tiles_count, source_msaa_x, source_msaa_y, rect_x, rect_y, rect_w, rect_h, source_format;
      uint32_t tile_width, tile_height;  // sa2-nx render resolution
    } k{{HostWidth(source), HostHeight(source)}, PitchTilesEDRAM(source), PitchTilesEDRAM(target), source_start,
        target_start, count, source.edram_msaa_x, source.edram_msaa_y, uint32_t(rect.offset.x),
        uint32_t(rect.offset.y), rect.extent.width, rect.extent.height, uint32_t(source.edram_format),
        tile_width, tile_height};
    static_assert(sizeof(k) == 64);
    // Called right after the import pass, whose end dependency already orders everything before it.
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelines_stencil_copy_[color ? 2 : msaa2 ? 1 : 0]);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE, layout_pipeline_stencil_copy_,
                                 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_stencil_copy_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            sizeof(k), &k);
    dfn_.vkCmdDispatch(commands_work_, (rect.extent.width / 4 + 7) / 8, (rect.extent.height + 7) / 8, 1);
    VkMemoryBarrier mid{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mid.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    mid.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_,
                              VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &mid, 0, nullptr, 0, nullptr);
    std::vector<VkBufferImageCopy> regions;
    const uint32_t pitch = PitchTilesEDRAM(target);
    for (uint32_t t = 0; t < count;) {
      const uint32_t tile = target_start + t, row = tile / pitch, col = tile % pitch;
      const uint32_t n = std::min(count - t, pitch - col);
      const uint32_t x = col * tile_width, y = row * tile_height;
      if (x >= HostWidth(target) || y >= HostHeight(target)) { t += n; continue; }
      VkBufferImageCopy r{};
      r.bufferOffset = VkDeviceSize(y - uint32_t(rect.offset.y)) * rect.extent.width + (x - uint32_t(rect.offset.x));
      r.bufferRowLength = rect.extent.width;
      r.bufferImageHeight = rect.extent.height;
      r.imageSubresource = {VK_IMAGE_ASPECT_STENCIL_BIT, 0, 0, 1};
      r.imageOffset = {int32_t(x), int32_t(y), 0};
      r.imageExtent = {std::min(n * tile_width, HostWidth(target) - x), std::min(tile_height, HostHeight(target) - y),
                       1};
      regions.push_back(r);
      t += n;
    }
    dfn_.vkCmdCopyBufferToImage(commands_work_, slot.buffer_stencil_copy, target.image, VK_IMAGE_LAYOUT_GENERAL,
                                uint32_t(regions.size()), regions.data());
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                              0, 1, &after, 0, nullptr, 0, nullptr);
    ++edram4_stencil_copies_;
    edram4_stencil_copies_tiles_ += count;
    return true;
  }

  bool ImportColorDepthEDRAM4(Image& source, Image& target,
                                 uint32_t source_start, uint32_t target_start, uint32_t count,
                                 const SpanEDRAM4* column = nullptr) {
    // usf4_edram4_import_columns: the tiles really imported, as (offset from target_start, count) runs.
    std::vector<std::pair<uint32_t, uint32_t>> runs;
    if (column && !column->runs.empty())
      for (const auto& r : column->runs) runs.push_back({r.first - target_start, r.second});
    else
      runs.push_back({0u, count});
    const auto deferred_runs = [&](const Image& view, uint32_t start) {
      for (const auto& r : runs) if (StencilDeferredEDRAM4(view, start + r.first, r.second)) return true;
      return false;
    };
    const auto real_runs = [&]() { for (const auto& r : runs) StencilRealEDRAM4(target, target_start + r.first, r.second); };
    const bool source_depth = source.edram_depth;
    const bool source_raw64 = source.edram_64bpp;
    const bool native_msaa_import = IsDepthNative2xEDRAM4(source) &&
        (IsDepthNative2xEDRAM4(target) || IsDepthNative1xEDRAM4(target));
    // sa2_msaa: a single-sample source into a host multisampled target is an ordinary import (N-sample pass). A host
    // multisampled source has no import here (it would need a sample-reading shader): declined quietly, and the
    // caller skips the transfer like any unsupported one (banjo_native_edram4_skip_unsupported).
    const bool host_msaa_target = me::native::IsSingleSample(uint32_t(source.sample_count)) && msaa_samples_ > 1 &&
                                  uint32_t(target.sample_count) == msaa_samples_;
    if (msaa_samples_ > 1 && uint32_t(source.sample_count) == msaa_samples_ && !IsDepthNative2xEDRAM4(source))
      return false;
    if ((!me::native::IsSingleSample(uint32_t(source.sample_count)) ||
         !me::native::IsSingleSample(uint32_t(target.sample_count))) && !host_msaa_target) {
      if (!native_msaa_import || source.edram_format != target.edram_format ||
          source.depth_float24_half != target.depth_float24_half || source.image == target.image)
        return FailureEDRAM4("native2x import requires distinct same-format depth views", target, &source);
      // The shader also discards invalid contracts, but a discard cannot be
      // reported as a successful physical transfer / ownership publication.
      if (!count || source_start >= 2048 || target_start >= 2048 ||
          count > 2048 - source_start || count > 2048 - target_start)
        return FailureEDRAM4("native2x import tile range exceeds physical EDRAM", target, &source);
    }
    const VkImageView raw_source = source_raw64 ? GetViewRaw64EDRAM4(source) : VK_NULL_HANDLE;
    if (source_raw64 && (!raw_source || source.edram_format != 7))
      return FailureEDRAM4("raw FP16 depth import UINT view unavailable", target, &source);
    if (!target.edram_depth ||
        source.guest_width || target.guest_width || !count || target.edram_format > 1 ||
        (source_depth ? source.edram_format > 1 :
          (!source_raw64 && source.edram_format != 0 && source.edram_format != 2 && source.edram_format != 3)))
      return FailureEDRAM4("depth import source class / scaled view unsupported", target, &source);
    const uint32_t source_pitch = PitchTilesEDRAM(source);
    if (native_msaa_import && (source_pitch > 2048 || PitchTilesEDRAM(target) > 2048))
      return FailureEDRAM4("native2x import pitch exceeds physical EDRAM", target, &source);
    const uint32_t source_tile_width = 80u >> (source.edram_msaa_x + uint32_t(source_raw64));
    const uint32_t source_tile_height = 16u >> source.edram_msaa_y;
    if (!source_pitch || source.width % source_tile_width ||
        (source_start + count - 1) / source_pitch * source_tile_height + source_tile_height > source.height)
      return FailureEDRAM4("color->depth missing source sample/padding", target, &source);
    const uint32_t destination_pitch = PitchTilesEDRAM(target);
    if (!destination_pitch || !count)
      return FailureEDRAM4("depth import empty destination tile range", target, &source);
    const uint32_t destination_tile_width = 80u >> target.edram_msaa_x;
    const uint32_t destination_tile_height = 16u >> target.edram_msaa_y;
    const uint32_t first_row = target_start / destination_pitch;
    const uint32_t last_row = (target_start + count - 1) / destination_pitch;
    const uint32_t sx = first_row == last_row ? (target_start % destination_pitch) * destination_tile_width : 0;
    const uint32_t sy = first_row * destination_tile_height;
    const uint32_t right = first_row == last_row
        ? std::min(target.width, ((target_start + count - 1) % destination_pitch + 1) * destination_tile_width)
        : target.width;
    const uint32_t bottom = std::min(target.height, (last_row + 1) * destination_tile_height);
    if (sx >= target.width || sy >= target.height || right <= sx || bottom <= sy)
      return FailureEDRAM4("depth import destination tile range outside attachment", target, &source);
    const VkRect2D scissor = column && !column->runs.empty() ? column->rect
                                                              : VkRect2D{{int32_t(sx), int32_t(sy)}, {right - sx, bottom - sy}};
    // sa2-nx render resolution: the pass works in host pixels (both views at the same scale).
    const VkRect2D host_scissor = HostRect(target, scissor);
    if (source.scale_q != target.scale_q)
      return FailureEDRAM4("depth import between images of different render scales", target, &source);
    PassImportDepthEDRAM* pass = EnsureImportDepthEDRAM4(target.format, target.sample_count,
                                                        source.sample_count);
    if (!pass || !sampler_depth_edram_)
      return FailureEDRAM4("color->depth graphics resources unavailable", target, &source);
    auto [framebuffer, new_value] = framebuffers_import_depth_edram_.try_emplace(target.image, VK_NULL_HANDLE);
    if (new_value) {
      VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      info.renderPass = pass->render_pass;
      info.attachmentCount = 1;
      info.pAttachments = &target.view;
      info.width = HostWidth(target);
      info.height = HostHeight(target);
      info.layers = 1;
      if (dfn_.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer->second) != VK_SUCCESS) {
        framebuffers_import_depth_edram_.erase(framebuffer);
        return false;
      }
    }
    // Closing a shadow pass in the capacity helper can recursively insert
    // another framebuffer and rehash this map. Keep the immutable handle, not
    // an unordered_map iterator, across command-slot rollover / reentrancy.
    const VkFramebuffer import_framebuffer = framebuffer->second;
    if (!EnsureCapacityConversionEDRAM4("depth import")) return false;
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = slot.pool_import_depth_edram;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout_import_depth_edram_;
    if (dfn_.vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return false;
    ++slot.conversions_edram;
    ViewsDepthEDRAM* depth_views = source_depth ? GetViewsDepthEDRAM4(source) : nullptr;
    if (source_depth && !depth_views) return false;
    VkDescriptorImageInfo sampled[2] = {
        {sampler_depth_edram_, source_depth ? depth_views->depth : source_raw64 ? raw_source : source.view, VK_IMAGE_LAYOUT_GENERAL},
        {sampler_depth_edram_, source_depth ? depth_views->stencil : VK_NULL_HANDLE, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet descriptor[2]{};
    for (uint32_t i = 0; i < (source_depth ? 2u : 1u); ++i) {
      descriptor[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor[i].dstSet = set;
      descriptor[i].dstBinding = i;
      descriptor[i].descriptorCount = 1;
      descriptor[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      descriptor[i].pImageInfo = &sampled[i];
    }
    dfn_.vkUpdateDescriptorSets(device_, source_depth ? 2u : 1u, descriptor, 0, nullptr);
    struct Constants {
      uint32_t source_size[2], target_size[2];
      uint32_t pitch_source, pitch_target, source_start, target_start, count;
      uint32_t source_format, target_format, source_64bpp, target_64bpp;
      uint32_t source_msaa_x, source_msaa_y, target_msaa_x, target_msaa_y;
      uint32_t stencil_mask;
      uint32_t tile_width, tile_height;  // sa2-nx render resolution: the tile in host samples
    } constants{{HostWidth(source), HostHeight(source)}, {HostWidth(target), HostHeight(target)},
                PitchTilesEDRAM(source), PitchTilesEDRAM(target), source_start, target_start, count,
                uint32_t(source.edram_format) | (source.depth_float24_half ? 256u : 0u),
                uint32_t(target.edram_format) | (target.depth_float24_half ? 256u : 0u),
                uint32_t(source_raw64), 0, source.edram_msaa_x, source.edram_msaa_y,
                target.edram_msaa_x, target.edram_msaa_y, 0, ScaledTileWidth(target.scale_q),
                ScaledTileHeight(target.scale_q)};
    static_assert(sizeof(Constants) == 80);
    if (draws_) draws_->FinishPass();
    const bool in_batch = edram4_batch_.active;
    if (in_batch) CloseComputeBatchEDRAM4();
    const bool continue_pass = in_batch && edram4_batch_.import_open &&
        edram4_batch_.import_pass == pass->render_pass && edram4_batch_.import_fb == import_framebuffer;
    if (in_batch && !continue_pass && !CloseImportBatchEDRAM4()) return false;
    // The render pass's external dependencies (EnsureImportDepthEDRAM4) already make the needed barriers;
    // on NVK every explicit barrier is a GPU idle wait plus cache flushes, ~100 small imports per frame.
    if (continue_pass) {
      ++edram4_batch_saved_passes_;
    } else {
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass->render_pass;
    begin.framebuffer = import_framebuffer;
    // Only the tiles' rectangle: a whole-image render area made every (often 1-tile) import cost the
    // same as a full-target pass on the Switch (Maxwell/NVK). The scissor bounds all fragments anyway.
    begin.renderArea = REXCVAR_GET(sa2_native_edram4_import_area_tiles)
        ? host_scissor : VkRect2D{{0, 0}, {HostWidth(target), HostHeight(target)}};
    // A batch's pass covers every import run of the sync (their union, computed before the first one).
    if (in_batch && edram4_batch_.has_area && REXCVAR_GET(sa2_native_edram4_import_area_tiles) &&
        ContainsRect(edram4_batch_.area_import, scissor))
      begin.renderArea = HostRect(target, edram4_batch_.area_import);
    dfn_.vkCmdBeginRenderPass(commands_work_, &begin, VK_SUBPASS_CONTENTS_INLINE);
    if (in_batch) {
      edram4_batch_.import_open = true;
      edram4_batch_.import_pass = pass->render_pass;
      edram4_batch_.import_fb = import_framebuffer;
    }
    }
    // The procedural triangle covers the viewport: make it the tiles' rectangle, so each of the up to 9
    // passes rasterizes only those tiles (the shaders use gl_FragCoord, not the viewport). With a
    // whole-target viewport every pass of a 1-tile import cost ~0.4 ms on the Switch.
    const VkViewport viewport = REXCVAR_GET(sa2_native_edram4_import_area_tiles)
        ? VkViewport{float(host_scissor.offset.x), float(host_scissor.offset.y), float(host_scissor.extent.width),
                     float(host_scissor.extent.height), 0, 1}
        : VkViewport{0, 0, float(HostWidth(target)), float(HostHeight(target)), 0, 1};
    dfn_.vkCmdSetViewport(commands_work_, 0, 1, &viewport);
    dfn_.vkCmdSetScissor(commands_work_, 0, 1, &host_scissor);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                   layout_pipeline_import_depth_edram_, 0, 1, &set, 0, nullptr);
    uint32_t steps = 9;  // depth + 8 stencil bits
    bool keep_stencil = false;
    const auto& sr = edram4_stencil_replacement_;
    const bool stencil_replaced = (sr && REXCVAR_GET(usf4_edram4_stencil_edges_skip) &&
        int64_t(scissor.offset.x) < int64_t(sr->offset.x) + sr->extent.width &&
        int64_t(sr->offset.x) < int64_t(scissor.offset.x) + scissor.extent.width &&
        int64_t(scissor.offset.y) < int64_t(sr->offset.y) + sr->extent.height &&
        int64_t(sr->offset.y) < int64_t(scissor.offset.y) + scissor.extent.height) ||
        (edram4_stencil_replacement_ &&
        scissor.offset.x >= edram4_stencil_replacement_->offset.x &&
        scissor.offset.y >= edram4_stencil_replacement_->offset.y &&
        int64_t(scissor.offset.x) + scissor.extent.width <=
            int64_t(edram4_stencil_replacement_->offset.x) + edram4_stencil_replacement_->extent.width &&
        int64_t(scissor.offset.y) + scissor.extent.height <=
            int64_t(edram4_stencil_replacement_->offset.y) + edram4_stencil_replacement_->extent.height);
    if (stencil_replaced) {
      steps = 1;  // the draw about to run replaces all of this stencil: only depth must arrive
      real_runs();
      ++edram4_stencil_replaced_;
    } else if (REXCVAR_GET(sa2_native_edram4_stencil_lazy)) {
      const auto source_source = edram4_stencil_source_.find(&source);
      const auto source_for = [&](uint32_t i) -> SourceStencil {
        const uint32_t local = (source_start + i) & kMaskEDRAM4;
        if (source_source != edram4_stencil_source_.end() && source_source->second[local].image)
          return source_source->second[local];
        return {&source, uint16_t(local)};
      };
      // Per draw: a draw that does not test or write stencil gets depth only; the stencil is fetched later,
      // for the tiles a stencil-using draw actually binds (FetchStencilDeferredEDRAM4). Clears and resolves
      // keep the per-view rule.
      // An inert stencil (test ALWAYS on both faces, no write) is no stencil use either.
      const uint32_t dc_draw = edram4_submission_ && edram4_submission_->registers
          ? edram4_submission_->registers[gr::XE_GPU_REG_RB_DEPTHCONTROL] : 1u;
      const bool stencil_inert = REXCVAR_GET(sa2_native_edram4_stencil_inert) &&
          ((dc_draw >> 8) & 7) == 7 && (!(dc_draw & 0x80) || ((dc_draw >> 20) & 7) == 7) &&
          !edram4_draw_writes_stencil_;
      const bool draw_no_stencil = edram4_reason_ == 0 && edram4_submission_ && edram4_submission_->registers &&
          edram4_slot_ == 0 && (!(dc_draw & 1) || stencil_inert);
      if (edram4_import_stencil_only_) {
        // explicit late stencil fetch: no decision here
      } else if (!target.edram4_usa_stencil || draw_no_stencil) {
        auto& sources = StencilSourcesWrite(&target);
        if (sources.empty()) sources.resize(kTilesEDRAM4);
        for (const auto& r : runs)
          for (uint32_t i = r.first; i < r.first + r.second; ++i)
            sources[(target_start + i) & kMaskEDRAM4] = source_for(i);
        steps = 1;
        ++edram4_stencil_skipped_;
      } else if (deferred_runs(source, source_start)) {
        bool own = true;
        for (const auto& r : runs)
          for (uint32_t i = r.first; i < r.first + r.second && own; ++i) {
            const SourceStencil f = source_for(i);
            own = f.image == &target && f.tile == ((target_start + i) & kMaskEDRAM4);
          }
        if (own) {
          steps = 1;
          keep_stencil = true;  // the real stencil never left the destination
          ++edram4_stencil_returned_;
        } else {
          ++edram4_stencil_inexact_;  // stencil lived elsewhere: the 8 passes copy the view's stale bits
        }
        real_runs();
      } else {
        real_runs();
      }
    }
    uint32_t first_step = 0;
    // 1x TRANSFER_DST destination with a depth source: the stencil goes through the copy engine after the
    // pass (CopyStencilEDRAM4) instead of 8 masked draws; the pass keeps only the depth draw.
    // Color sources too (RGBA8 / UNORM10 / 7e3 classes, 1x): the stencil is the color word's low byte.
    const bool copy_stencil = REXCVAR_GET(sa2_native_edram4_stencil_copy) && runs.size() == 1 &&
        (source_depth || (source.edram_format == 0 || source.edram_format == 2 || source.edram_format == 3)) &&
        !source_raw64 && (steps > 1 || edram4_import_stencil_only_) &&
        me::native::IsSingleSample(uint32_t(target.sample_count)) && target.accepts_target_of_copy &&
        !target.edram_msaa_x && !target.edram_msaa_y && !target.raster_grid_x &&
        (me::native::IsSingleSample(uint32_t(source.sample_count)) ||
         (source.sample_count == VK_SAMPLE_COUNT_2_BIT && source.edram_msaa_y == 1 && !source.edram_msaa_x)) &&
        (source_depth || me::native::IsSingleSample(uint32_t(source.sample_count))) &&
        host_scissor.extent.width % 4 == 0 &&
        VkDeviceSize(host_scissor.extent.width) * host_scissor.extent.height <= kBufferStencilCopy &&
        // Small imports: the copy path adds 3D->compute->copy->3D switches (each a GPU wait-for-idle on NVK),
        // which cost more than a few masked stencil draws inside the same 3D pass.
        count >= uint32_t(REXCVAR_GET(sa2_native_edram4_stencil_copy_min_tiles)) &&
        EnsureStencilCopyEDRAM4();
    if (copy_stencil) {
      steps = edram4_import_stencil_only_ ? 0 : 1;
    } else if (edram4_import_stencil_only_) {
      steps = 9;
      first_step = 1;  // depth stays: zero just the stencil of the tiles, then OR the 8 bits in
      VkClearAttachment clear{};
      clear.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      // One rect per tile-row run: a multi-row range has a full-width scissor that also covers tiles
      // before/after the run, whose stencil must survive.
      std::vector<VkClearRect> rects;
      const uint32_t dp = PitchTilesEDRAM(target);
      const uint32_t tw = 80u >> target.edram_msaa_x, th = 16u >> target.edram_msaa_y;
      for (const auto& run : runs)
      for (uint32_t t = run.first; t < run.first + run.second && dp;) {
        const uint32_t tile = target_start + t, col = tile % dp;
        const uint32_t n = std::min(run.first + run.second - t, dp - col);
        const uint32_t x = (col * tw) << target.raster_grid_x, y = (tile / dp) * th;
        if (x < target.width && y < target.height)
          rects.push_back({HostRect(target, VkRect2D{{int32_t(x), int32_t(y)},
                                                     {std::min((n * tw) << target.raster_grid_x, target.width - x),
                                                      std::min(th, target.height - y)}}),
                           0, 1});
        t += n;
      }
      if (!rects.empty()) dfn_.vkCmdClearAttachments(commands_work_, 1, &clear, uint32_t(rects.size()), rects.data());
    }
    if (steps > 1) {
      MarkGpu(kGpuEdramImport9);
      auto& n = edram4_import9_pairs_[fmt::format("{}{:03X}:{}x{}:mx{}my{}->{:03X}:{}x{}:mx{}my{}{}",
          source.edram_depth ? "D" : "C", source.edram_base, source.width, source.height, source.edram_msaa_x,
          source.edram_msaa_y, target.edram_base, target.width, target.height, target.edram_msaa_x,
          target.edram_msaa_y, deferred_runs(source, source_start) ? " deferred" : "")];
      ++n.first;
      for (const auto& r : runs) n.second += r.second;
    }
    const uint8_t bits_source = source_depth && !source_raw64 ? source.edram4_bits_stencil : uint8_t(255);
    target.edram4_bits_stencil |= bits_source;
    const bool skip_bits = REXCVAR_GET(sa2_native_edram4_bits_stencil);
    for (uint32_t step = first_step; step < steps; ++step) {
      if (step && skip_bits && !(bits_source & (1u << (step - 1)))) {
        ++edram4_bits_skipped_;
        continue;  // this bit is 0 everywhere in the source: pass 0 / the clear already wrote it
      }
      constants.stencil_mask = step ? (1u << (step - 1)) : 0u;
      const uint32_t mask = step ? constants.stencil_mask : (keep_stencil ? 0u : 255u);
      dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                               pass->pipelines[(source_depth ? 2u : source_raw64 ? 4u : 0u) + (step != 0)]);
      dfn_.vkCmdSetStencilReference(commands_work_, VK_STENCIL_FACE_FRONT_AND_BACK, constants.stencil_mask);
      dfn_.vkCmdSetStencilWriteMask(commands_work_, VK_STENCIL_FACE_FRONT_AND_BACK, mask);
      dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_import_depth_edram_,
                                VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(constants), &constants);
      dfn_.vkCmdDraw(commands_work_, 3, 1, 0, 0);
    }
    if (in_batch) {
      // The pass stays open for the next run; the stencil copy follows when the batch closes it.
      if (copy_stencil)
        edram4_batch_.copies_stencil.push_back({&source, &target, source_start, target_start, count, scissor});
    } else {
    dfn_.vkCmdEndRenderPass(commands_work_);
    if (copy_stencil && !CopyStencilEDRAM4(source, target, source_start, target_start, count, scissor))
      return FailureEDRAM4("copy-engine stencil import failed", target, &source);
    if (draws_) draws_->NotifyGraphicsExternalState();
    }
    if (++edram4_imports_ <= 32)
      REXLOG_INFO("[native] EDRAM mode4 {}->depth {:03X}/{} to {:03X}/{} tiles {}:{} count {} "
                  "(9-pass stencil fallback; hardware samples {}->{}, nativeMSimport={})",
                  source_depth ? "depth" : "color", source.edram_base, source.edram_format,
                  target.edram_base, target.edram_format,
                  source_start, target_start, count,
                  uint32_t(source.sample_count), uint32_t(target.sample_count), native_msaa_import);
    if (target.edram_format == 1 && !target.depth_float24_half && !edram4_float_import_warned_) {
      edram4_float_import_warned_ = true;
      REXLOG_WARN("[native] EDRAM mode4 D24FS8 import clamps guest depth >1: native unhalved host range "
                  "cannot preserve these raw float24 encodings; NOT conformance");
    }
    return true;
  }

  bool EnsureResolveDepthNative2xEDRAM4() {
    if (pipeline_resolve_depth_msaa2_edram_) return true;
    if (!AcceptsImportDepthNative2xEDRAM4(VK_FORMAT_D32_SFLOAT_S8_UINT) ||
        !layout_pipeline_conversion_depth_edram_) return false;
    if (!shader_resolve_depth_msaa2_edram_) {
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_depth_resolve_guestspace_msaa2_cs);
      module.pCode = shaders::me_depth_resolve_guestspace_msaa2_cs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr,
                                     &shader_resolve_depth_msaa2_edram_) != VK_SUCCESS) return false;
    }
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipeline.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0,
                      VK_SHADER_STAGE_COMPUTE_BIT, shader_resolve_depth_msaa2_edram_, "main", nullptr};
    pipeline.layout = layout_pipeline_conversion_depth_edram_;
    return dfn_.vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr,
                                          &pipeline_resolve_depth_msaa2_edram_) == VK_SUCCESS;
  }

  bool ResolverDepthGuestspaceEDRAM4(Image& source, Image& target,
      uint32_t sx, uint32_t sy, uint32_t dx, uint32_t dy, uint32_t width, uint32_t height,
      uint32_t raw_sample_select = 0) {
    if (!source.depth_float24_half || target.format != VK_FORMAT_R32_SFLOAT ||
        !target.resolved_depth_guestspace || !width || !height ||
        sx >= source.width || sy >= source.height || dx >= target.width || dy >= target.height ||
        width > source.width - sx || height > source.height - sy ||
        width > target.width - dx || height > target.height - dy)
      return FailureEDRAM4("guestspace depth resolve invalid contract/rectangle", target, &source);
    const bool native2x = IsDepthNative2xEDRAM4(source);
    if (!me::native::IsSingleSample(uint32_t(target.sample_count)) ||
        (!me::native::IsSingleSample(uint32_t(source.sample_count)) && !native2x))
      return FailureEDRAM4("guestspace depth resolve hardware sample contract unsupported", target, &source);
    const auto selected = native2x ? me::native::SanitizeNative2xDepthCopySampleSelect(raw_sample_select)
                                   : std::optional<uint32_t>(0);
    if (!selected || (native2x && !EnsureResolveDepthNative2xEDRAM4()))
      return FailureEDRAM4("native2x guestspace resolve selector/capability/resources unsupported", target, &source);
    const VkPipeline pipeline = native2x ? pipeline_resolve_depth_msaa2_edram_
                                        : pipelines_conversion_depth_edram_[3];
    if (!EnsureCapacityConversionEDRAM4("guestspace depth resolve"))
      return FailureEDRAM4("guestspace resolve cannot acquire recording slot", target, &source);
    ViewsDepthEDRAM* views = GetViewsDepthEDRAM4(source);
    if (!views || !pipeline)
      return FailureEDRAM4("guestspace depth resolve resources unavailable", target, &source);
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = slot.pool_conversion_depth_edram;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout_conversion_depth_edram_;
    if (dfn_.vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS)
      return FailureEDRAM4("guestspace resolve descriptor allocation failed", target, &source);
    ++slot.conversions_edram;
    VkDescriptorImageInfo images[2] = {
        {sampler_depth_edram_, views->depth, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, target.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = i ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &images[i];
    }
    dfn_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);
    struct Constants {
      uint32_t source_size[2], destination_size[2], source_offset[2], destination_offset[2], extent[2];
      float guest_factor;
      uint32_t sample_select;
    } constants{{HostWidth(source), HostHeight(source)}, {HostWidth(target), HostHeight(target)},
                {sx, sy}, {dx, dy}, {width, height}, 2.0f, *selected};
    // sa2-nx render resolution: the rectangles in host pixels.
    {
      const VkRect2D hs = HostRect(source, VkRect2D{{int32_t(sx), int32_t(sy)}, {width, height}});
      const VkRect2D hd = HostRect(target, VkRect2D{{int32_t(dx), int32_t(dy)}, {width, height}});
      constants.source_offset[0] = uint32_t(hs.offset.x);
      constants.source_offset[1] = uint32_t(hs.offset.y);
      constants.destination_offset[0] = uint32_t(hd.offset.x);
      constants.destination_offset[1] = uint32_t(hd.offset.y);
      constants.extent[0] = std::min(hs.extent.width, hd.extent.width);
      constants.extent[1] = std::min(hs.extent.height, hd.extent.height);
    }
    static_assert(offsetof(Constants, sample_select) == 44 && sizeof(Constants) == 48);
    // Keep the original 1x shader's 44-byte ABI byte-for-byte. The dormant MS
    // variant adds ONLY the already sanitized guest selector at offset44.
    const uint32_t constant_bytes = native2x ? 48u : 44u;
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
        layout_pipeline_conversion_depth_edram_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conversion_depth_edram_,
        VK_SHADER_STAGE_COMPUTE_BIT, 0, constant_bytes, &constants);
    dfn_.vkCmdDispatch(commands_work_, (constants.extent[0] + 7) / 8, (constants.extent[1] + 7) / 8, 1);
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after, 0, nullptr, 0, nullptr);
    return true;
  }

  bool ConvertDepthEDRAM4(Image& source, Image& target,
                             uint32_t source_start, uint32_t target_start, uint32_t count) {
    if (!source.edram_depth || target.edram_depth)
      return FailureEDRAM4("depth export source/destination class unsupported", target, &source);
    if (!me::native::IsSingleSample(uint32_t(source.sample_count)) ||
        !me::native::IsSingleSample(uint32_t(target.sample_count)))
      return false;  // sa2_msaa: the compute export reads and writes single-sample views (skipped as unsupported)
    const VkImageView raw_destination = target.edram_64bpp ? GetViewRaw64EDRAM4(target) : VK_NULL_HANDLE;
    if (target.edram_64bpp && (!raw_destination || target.edram_format != 7))
      return FailureEDRAM4("raw FP16 UINT sampled/storage view unavailable", target, &source);
    const uint32_t pipeline = target.edram_64bpp ? 2u : target.format == VK_FORMAT_R8G8B8A8_UNORM ? 0u :
        (target.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
         (target.edram_format == 2 || target.edram_format == 3)) ? 1u : UINT32_MAX;
    if (pipeline == UINT32_MAX || source.edram_format > 1 || !count) return false;
    ViewsDepthEDRAM* views = GetViewsDepthEDRAM4(source);
    if (!views) return false;
    if (!pipelines_conversion_depth_edram_[pipeline] ||
        !EnsureCapacityConversionEDRAM4("depth export")) return false;
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = slot.pool_conversion_depth_edram;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout_conversion_depth_edram_;
    if (dfn_.vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return false;
    ++slot.conversions_edram;
    VkDescriptorImageInfo images[3] = {
        {sampler_depth_edram_, views->depth, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, raw_destination ? raw_destination : target.view, VK_IMAGE_LAYOUT_GENERAL},
        {sampler_depth_edram_, views->stencil, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = i == 1 ? VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                                       : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      writes[i].pImageInfo = &images[i];
    }
    dfn_.vkUpdateDescriptorSets(device_, 3, writes, 0, nullptr);
    if (source.scale_q != target.scale_q)
      return FailureEDRAM4("depth export between images of different render scales", target, &source);
    struct Constants {
      uint32_t source_size[2], target_size[2];
      uint32_t pitch_source, pitch_target, source_start, target_start, count;
      uint32_t source_format, target_format, source_64bpp, target_64bpp;
      uint32_t source_msaa_x, source_msaa_y, target_msaa_x, target_msaa_y;
      uint32_t tile_width, tile_height;  // sa2-nx render resolution: the tile in host samples
    } constants{{HostWidth(source), HostHeight(source)}, {HostWidth(target), HostHeight(target)},
                PitchTilesEDRAM(source), PitchTilesEDRAM(target), source_start, target_start,
                count, uint32_t(source.edram_format) | (source.depth_float24_half ? 256u : 0u),
                target.edram_format, 0, uint32_t(target.edram_64bpp),
                source.edram_msaa_x, source.edram_msaa_y, target.edram_msaa_x, target.edram_msaa_y,
                ScaledTileWidth(target.scale_q), ScaledTileHeight(target.scale_q)};
    static_assert(sizeof(Constants) == 76);
    if (draws_) draws_->FinishPass();
    if (edram4_batch_.active && !CloseImportBatchEDRAM4()) return false;
    if (edram4_batch_.compute_pending) {
      ++edram4_batch_saved_barriers_;  // the previous compute transfer of this batch is still "inside"
    } else {
    VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before,
                                0, nullptr, 0, nullptr);
    }
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                             pipelines_conversion_depth_edram_[pipeline]);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                   layout_pipeline_conversion_depth_edram_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conversion_depth_edram_,
                              VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    dfn_.vkCmdDispatch(commands_work_, (count * constants.tile_width + 7) / 8, (constants.tile_height + 7) / 8, 1);
    if (edram4_batch_.active) {
      edram4_batch_.compute_pending = true;
    } else {
    VkMemoryBarrier after{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    after.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after,
                                0, nullptr, 0, nullptr);
    }
    if (++edram4_exports_ <= 32)
      REXLOG_INFO("[native] EDRAM mode4 depth->color {:03X}/{} to {:03X}/{} tiles {}:{} count {} "
                  "guest MSAA source {}{} destination {}{} (single-sample approximation)",
                  source.edram_base, source.edram_format, target.edram_base, target.edram_format,
                  source_start, target_start, count, source.edram_msaa_x, source.edram_msaa_y,
                  target.edram_msaa_x, target.edram_msaa_y);
    return true;
  }

  // sa2_native_conversion_frag: the raw UINT view of a 64-bit color image as a color attachment.
  VkImageView GetViewRaw64RTEDRAM4(Image& image) {
    if (!raw64_edram_supported_ || image.format != VK_FORMAT_R16G16B16A16_SFLOAT) return VK_NULL_HANDLE;
    auto [it, fresh] = views_raw64_rt_edram_.try_emplace(image.image, VK_NULL_HANDLE);
    if (fresh) {
      VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
      view.image = image.image;
      view.viewType = VK_IMAGE_VIEW_TYPE_2D;
      view.format = VK_FORMAT_R16G16B16A16_UINT;
      view.subresourceRange = kRangeColor;
      VkImageViewUsageCreateInfo usage{VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO};
      usage.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
      view.pNext = &usage;
      if (dfn_.vkCreateImageView(device_, &view, nullptr, &it->second) != VK_SUCCESS) {
        views_raw64_rt_edram_.erase(it);
        return VK_NULL_HANDLE;
      }
    }
    return it->second;
  }

  // sa2_native_conversion_frag. Lazily creates what the fragment version of the color <-> color
  // transfers needs: layouts, shaders and, per destination host format, a LOAD/STORE render pass and its
  // pipeline. nullptr (and the compute path from then on) if anything cannot be created.
  // samples: the destination's host sample count (sa2_msaa: a host multisampled game target; 1 otherwise).
  PassConversionColorFrag* EnsureConversionColorFrag(VkFormat format, uint32_t variant, uint32_t samples = 1) {
    if (conv_color_frag_failed_ || !sampler_depth_edram_) return nullptr;
    const auto fail = [&]() -> PassConversionColorFrag* {
      conv_color_frag_failed_ = true;
      REXLOG_WARN("[native] EDRAM mode4 fragment color conversions unavailable: compute path");
      return nullptr;
    };
    if (!layout_conv_color_frag_) {
      VkDescriptorSetLayoutBinding binding{};
      binding.binding = 0;
      binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      binding.descriptorCount = 1;
      binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
      VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      layout.bindingCount = 1;
      layout.pBindings = &binding;
      if (dfn_.vkCreateDescriptorSetLayout(device_, &layout, nullptr, &layout_conv_color_frag_) != VK_SUCCESS)
        return fail();
      VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 76};  // sa2-nx: + the scaled tile size
      VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipeline_layout.setLayoutCount = 1;
      pipeline_layout.pSetLayouts = &layout_conv_color_frag_;
      pipeline_layout.pushConstantRangeCount = 1;
      pipeline_layout.pPushConstantRanges = &range;
      if (dfn_.vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &layout_pipeline_conv_color_frag_) !=
          VK_SUCCESS)
        return fail();
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_import_vs);
      module.pCode = shaders::me_edram_import_vs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &vs_conv_color_frag_) != VK_SUCCESS) return fail();
      module.codeSize = sizeof(shaders::me_edram_color_to_color_fs);
      module.pCode = shaders::me_edram_color_to_color_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_conv_color_frag_) != VK_SUCCESS) return fail();
      module.codeSize = sizeof(shaders::me_edram_r64_to_r64_fs);
      module.pCode = shaders::me_edram_r64_to_r64_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_conv_r64_frag_) != VK_SUCCESS) return fail();
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kConversionsEDRAMPerSlot};
      VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool.maxSets = kConversionsEDRAMPerSlot;
      pool.poolSizeCount = 1;
      pool.pPoolSizes = &size;
      for (auto& slot : slots_)
        if (dfn_.vkCreateDescriptorPool(device_, &pool, nullptr, &slot.pool_conv_color_frag) != VK_SUCCESS)
          return fail();
    }
    if (variant == 2 && !fs_resolve_copy_) {  // sa2_gpu_resolve_draw
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::sa2_resolve_copy_fs);
      module.pCode = shaders::sa2_resolve_copy_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_resolve_copy_) != VK_SUCCESS) {
        fs_resolve_copy_ = VK_NULL_HANDLE;
        return nullptr;
      }
    }
    auto [it, new_value] = passes_conv_color_frag_.try_emplace(uint64_t(uint32_t(format)) | (uint64_t(variant) << 32) |
                                                               (uint64_t(samples > 1 ? samples : 0) << 40));
    PassConversionColorFrag& pass = it->second;
    if (!new_value) return pass.render_pass && pass.pipeline ? &pass : nullptr;
    // A failure of the resolve copy's own pass (variant 2) only declines that path.
    const auto fail_pass = [&]() -> PassConversionColorFrag* {
      if (variant != 2) return fail();
      REXLOG_WARN("[native] targets: the resolve copy pass (sa2_gpu_resolve_draw) could not be created: copies as before");
      return nullptr;
    };
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = samples > 1 ? VkSampleCountFlagBits(samples) : VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &reference;
    VkSubpassDependency dependencies[2]{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    // Same narrowing as the depth imports (sa2_native_pass_narrow_dependency): the compute and copy
    // transfers end with their own barriers; only attachment and transfer writes need this one.
    dependencies[0].srcAccessMask = REXCVAR_GET(sa2_native_pass_narrow_dependency)
        ? VkAccessFlags(VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_TRANSFER_WRITE_BIT)
        : VkAccessFlags(VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    dependencies[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                    VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    if (variant == 2) {
      // sa2_gpu_resolve_draw: exact masks instead of MEMORY_READ | MEMORY_WRITE, which NVK turns into invalidations of
      // every cache (macro engine data, constants, QMDs, shader data, textures) at each pass end. Before: whatever wrote
      // the render target or the texture (draws, copies, compute conversions); after: what reads the texture (draws,
      // the output pass, copies) and what writes either image next (WAW/WAR).
      dependencies[0].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                                      VK_ACCESS_SHADER_WRITE_BIT;
      dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
      dependencies[1].dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
                                     VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                     VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                     VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
      dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                                      VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                                      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    }
    VkRenderPassCreateInfo render_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass.attachmentCount = 1;
    render_pass.pAttachments = &attachment;
    render_pass.subpassCount = 1;
    render_pass.pSubpasses = &subpass;
    render_pass.dependencyCount = 2;
    render_pass.pDependencies = dependencies;
    if (dfn_.vkCreateRenderPass(device_, &render_pass, nullptr, &pass.render_pass) != VK_SUCCESS) return fail_pass();
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_conv_color_frag_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = variant == 2 ? fs_resolve_copy_ : variant ? fs_conv_r64_frag_ : fs_conv_color_frag_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = samples > 1 ? VkSampleCountFlagBits(samples) : VK_SAMPLE_COUNT_1_BIT;
    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = VK_FALSE;
    blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo color{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    color.attachmentCount = 1;
    color.pAttachments = &blend_attachment;
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = uint32_t(std::size(dynamic_states));
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pColorBlendState = &color;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout_pipeline_conv_color_frag_;
    pipeline.renderPass = pass.render_pass;
    if (dfn_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pass.pipeline) != VK_SUCCESS)
      return fail_pass();
    return &pass;
  }

  // sa2_gpu_resolve_draw: a same-format RGBA8 color resolve as one fragment pass on the 3D engine (variant 2 of the
  // color fragment passes; shaders/sa2_resolve_copy.frag) instead of vkCmdCopyImage, which NVK records on the copy
  // engine (NV90B5) with a wait for idle at each engine switch. The pass reads source texel (x, y) + delta and writes
  // destination texel (x, y); UNORM8 values round-trip exactly, as in the EDRAM color conversions. The render pass's
  // own dependencies order it like BarrierBeforeCopyResolve / BarrierAfterCopyResolve did. Returns false WITHOUT
  // recording anything when it does not apply: the caller records the copy as before.
  bool ResolveColorFrag(Image& source, Image& target, int32_t sx, int32_t sy, uint32_t dx, uint32_t dy,
                        uint32_t width, uint32_t height) {
    if (!g_resolve_draw.active || conv_color_frag_failed_ || !sampler_depth_edram_) return false;
    if (source.format != VK_FORMAT_R8G8B8A8_UNORM || target.format != VK_FORMAT_R8G8B8A8_UNORM ||
        source.image == target.image || !source.view || !target.view || !width || !height ||
        source.guest_width || source.guest_height || target.guest_width || target.guest_height ||
        source.raster_grid_x || target.raster_grid_x || source.scale_q != target.scale_q ||
        !me::native::IsSingleSample(uint32_t(source.sample_count)) ||
        !me::native::IsSingleSample(uint32_t(target.sample_count)) || sx < 0 || sy < 0 ||
        uint64_t(sx) + width > source.width || uint64_t(sy) + height > source.height ||
        uint64_t(dx) + width > target.width || uint64_t(dy) + height > target.height ||
        slots_[slot_].conversions_edram >= kConversionsEDRAMPerSlot) {
      ++resolve_draws_declined_;
      return false;
    }
    PassConversionColorFrag* pass = EnsureConversionColorFrag(VK_FORMAT_R8G8B8A8_UNORM, 2);
    if (!pass) {
      ++resolve_draws_declined_;
      return false;
    }
    auto [framebuffer, new_value] = framebuffers_conv_color_frag_.try_emplace({target.image, 2u}, VK_NULL_HANDLE);
    if (new_value) {
      VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      info.renderPass = pass->render_pass;
      info.attachmentCount = 1;
      info.pAttachments = &target.view;
      info.width = HostWidth(target);
      info.height = HostHeight(target);
      info.layers = 1;
      if (dfn_.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer->second) != VK_SUCCESS) {
        framebuffers_conv_color_frag_.erase(framebuffer);
        ++resolve_draws_declined_;
        return false;
      }
    }
    const VkFramebuffer framebuffer_copy = framebuffer->second;
    if (draws_) draws_->FinishPass();
    if (edram4_batch_.active) {
      CloseComputeBatchEDRAM4();
      if (!CloseImportBatchEDRAM4()) {
        ++resolve_draws_declined_;
        return false;
      }
    }
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo reserve{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    reserve.descriptorPool = slot.pool_conv_color_frag;
    reserve.descriptorSetCount = 1;
    reserve.pSetLayouts = &layout_conv_color_frag_;
    if (dfn_.vkAllocateDescriptorSets(device_, &reserve, &set) != VK_SUCCESS) {
      ++resolve_draws_declined_;
      return false;
    }
    ++slot.conversions_edram;
    VkDescriptorImageInfo sampled{sampler_depth_edram_, source.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &sampled;
    dfn_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    // sa2-nx render resolution: in host pixels (resolve rectangles are 8-aligned: exact at every scale).
    const VkRect2D source_area = HostRect(source, VkRect2D{{sx, sy}, {width, height}});
    VkRect2D area = HostRect(target, VkRect2D{{int32_t(dx), int32_t(dy)}, {width, height}});
    area.extent.width = std::min(area.extent.width, source_area.extent.width);
    area.extent.height = std::min(area.extent.height, source_area.extent.height);
    const int32_t delta[2] = {source_area.offset.x - area.offset.x, source_area.offset.y - area.offset.y};
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass->render_pass;
    begin.framebuffer = framebuffer_copy;
    begin.renderArea = area;
    dfn_.vkCmdBeginRenderPass(commands_work_, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{float(area.offset.x), float(area.offset.y), float(area.extent.width),
                              float(area.extent.height), 0, 1};
    dfn_.vkCmdSetViewport(commands_work_, 0, 1, &viewport);
    dfn_.vkCmdSetScissor(commands_work_, 0, 1, &area);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->pipeline);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 layout_pipeline_conv_color_frag_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conv_color_frag_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(delta), delta);
    dfn_.vkCmdDraw(commands_work_, 3, 1, 0, 0);
    dfn_.vkCmdEndRenderPass(commands_work_);
    if (draws_) draws_->NotifyGraphicsExternalState();
    if (++resolve_draws_ == 1)
      REXLOG_INFO("[native] targets: resolves drawn on the 3D engine (sa2_gpu_resolve_draw): first one {}x{} at "
                  "{},{} -> {},{}", width, height, sx, sy, dx, dy);
    g_resolve_draw.UseSampled();
    return true;
  }

  // ---- sa2_msaa: depth resolve of a multisampled depth target ---------------------------------------------------------
  // vkCmdResolveImage has no depth: one fragment pass writes sample 0 of each source pixel (the guest's only sample) as
  // the depth of the single-sample resolved texture (shaders/sa2_depth_resolve_ms.frag). Its stencil stays: the game
  // samples resolved depth as k_24_8 depth only.
  struct PassDepthResolveMs {
    VkRenderPass render_pass = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
  };
  std::unordered_map<uint32_t, PassDepthResolveMs> passes_depth_resolve_ms_;  // by host depth format
  std::unordered_map<VkImage, VkFramebuffer> framebuffers_depth_resolve_ms_;  // by destination image
  VkDescriptorSetLayout layout_depth_resolve_ms_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_depth_resolve_ms_ = VK_NULL_HANDLE;
  VkShaderModule vs_depth_resolve_ms_ = VK_NULL_HANDLE, fs_depth_resolve_ms_ = VK_NULL_HANDLE;
  bool depth_resolve_ms_failed_ = false;

  PassDepthResolveMs* EnsureDepthResolveMs(VkFormat format) {
    if (depth_resolve_ms_failed_ || !sampler_depth_edram_) return nullptr;
    const auto fail = [&]() -> PassDepthResolveMs* {
      depth_resolve_ms_failed_ = true;
      REXLOG_WARN("[res] MSAA depth resolve pass could not be created: depth resolves of multisampled targets fail");
      return nullptr;
    };
    if (!layout_depth_resolve_ms_) {
      VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT,
                                           nullptr};
      VkDescriptorSetLayoutCreateInfo layout{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
      layout.bindingCount = 1;
      layout.pBindings = &binding;
      if (dfn_.vkCreateDescriptorSetLayout(device_, &layout, nullptr, &layout_depth_resolve_ms_) != VK_SUCCESS)
        return fail();
      VkPushConstantRange range{VK_SHADER_STAGE_FRAGMENT_BIT, 0, 8};
      VkPipelineLayoutCreateInfo pipeline_layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
      pipeline_layout.setLayoutCount = 1;
      pipeline_layout.pSetLayouts = &layout_depth_resolve_ms_;
      pipeline_layout.pushConstantRangeCount = 1;
      pipeline_layout.pPushConstantRanges = &range;
      if (dfn_.vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &layout_pipeline_depth_resolve_ms_) !=
          VK_SUCCESS)
        return fail();
      VkShaderModuleCreateInfo module{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      module.codeSize = sizeof(shaders::me_edram_import_vs);
      module.pCode = shaders::me_edram_import_vs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &vs_depth_resolve_ms_) != VK_SUCCESS) return fail();
      module.codeSize = sizeof(shaders::sa2_depth_resolve_ms_fs);
      module.pCode = shaders::sa2_depth_resolve_ms_fs;
      if (dfn_.vkCreateShaderModule(device_, &module, nullptr, &fs_depth_resolve_ms_) != VK_SUCCESS) return fail();
      VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, kConversionsEDRAMPerSlot};
      VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
      pool.maxSets = kConversionsEDRAMPerSlot;
      pool.poolSizeCount = 1;
      pool.pPoolSizes = &size;
      for (auto& slot : slots_)
        if (dfn_.vkCreateDescriptorPool(device_, &pool, nullptr, &slot.pool_depth_resolve_ms) != VK_SUCCESS)
          return fail();
    }
    auto [it, fresh] = passes_depth_resolve_ms_.try_emplace(uint32_t(format));
    PassDepthResolveMs& pass = it->second;
    if (!fresh) return pass.render_pass && pass.pipeline ? &pass : nullptr;
    VkAttachmentDescription attachment{};
    attachment.format = format;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    attachment.storeOp = attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.initialLayout = attachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkAttachmentReference reference{0, VK_IMAGE_LAYOUT_GENERAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.pDepthStencilAttachment = &reference;
    VkSubpassDependency dependencies[2]{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_SHADER_READ_BIT |
        VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    VkRenderPassCreateInfo render_pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    render_pass.attachmentCount = 1;
    render_pass.pAttachments = &attachment;
    render_pass.subpassCount = 1;
    render_pass.pSubpasses = &subpass;
    render_pass.dependencyCount = 2;
    render_pass.pDependencies = dependencies;
    if (dfn_.vkCreateRenderPass(device_, &render_pass, nullptr, &pass.render_pass) != VK_SUCCESS) return fail();
    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs_depth_resolve_ms_;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs_depth_resolve_ms_;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo assembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1.0f;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depth.depthTestEnable = depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp = VK_COMPARE_OP_ALWAYS;
    VkPipelineColorBlendStateCreateInfo color{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    const VkDynamicState dynamic_states[] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = uint32_t(std::size(dynamic_states));
    dynamic.pDynamicStates = dynamic_states;
    VkGraphicsPipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipeline.stageCount = 2;
    pipeline.pStages = stages;
    pipeline.pVertexInputState = &vertex;
    pipeline.pInputAssemblyState = &assembly;
    pipeline.pViewportState = &viewport;
    pipeline.pRasterizationState = &raster;
    pipeline.pMultisampleState = &multisample;
    pipeline.pDepthStencilState = &depth;
    pipeline.pColorBlendState = &color;
    pipeline.pDynamicState = &dynamic;
    pipeline.layout = layout_pipeline_depth_resolve_ms_;
    pipeline.renderPass = pass.render_pass;
    if (dfn_.vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &pipeline, nullptr, &pass.pipeline) != VK_SUCCESS)
      return fail();
    return &pass;
  }

  // Records the resolve of the guest rectangle `copy` (guest units, as for vkCmdCopyImage) of the multisampled depth
  // target `source` into the single-sample texture `target`. false = not recorded.
  bool ResolveDepthMs(Image& source, Image& target, const VkImageCopy& copy) {
    if (!IsDepthFormat(target.format) || !me::native::IsSingleSample(uint32_t(target.sample_count)) ||
        !(target.vk_usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT))
      return false;
    PassDepthResolveMs* pass = EnsureDepthResolveMs(target.format);
    ViewsDepthEDRAM* views = GetViewsDepthEDRAM4(source);
    if (!pass || !views || !EnsureCapacityConversionEDRAM4("msaa depth resolve")) return false;
    auto [framebuffer, fresh] = framebuffers_depth_resolve_ms_.try_emplace(target.image, VK_NULL_HANDLE);
    if (fresh) {
      VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      info.renderPass = pass->render_pass;
      info.attachmentCount = 1;
      info.pAttachments = &target.view;
      info.width = HostWidth(target);
      info.height = HostHeight(target);
      info.layers = 1;
      if (dfn_.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer->second) != VK_SUCCESS) {
        framebuffers_depth_resolve_ms_.erase(framebuffer);
        return false;
      }
    }
    const VkFramebuffer framebuffer_resolve = framebuffer->second;
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = slot.pool_depth_resolve_ms;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &layout_depth_resolve_ms_;
    if (dfn_.vkAllocateDescriptorSets(device_, &allocate, &set) != VK_SUCCESS) return false;
    ++slot.conversions_edram;
    const VkDescriptorImageInfo sampled{sampler_depth_edram_, views->depth, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &sampled;
    dfn_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    const VkImageCopy host = HostCopy(source, target, copy);
    const VkRect2D area{{host.dstOffset.x, host.dstOffset.y}, {host.extent.width, host.extent.height}};
    const int32_t delta[2] = {host.srcOffset.x - host.dstOffset.x, host.srcOffset.y - host.dstOffset.y};
    if (draws_) draws_->FinishPass();
    if (edram4_batch_.active) {
      CloseComputeBatchEDRAM4();
      if (!CloseImportBatchEDRAM4()) return false;
    }
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass->render_pass;
    begin.framebuffer = framebuffer_resolve;
    begin.renderArea = area;
    dfn_.vkCmdBeginRenderPass(commands_work_, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{float(area.offset.x), float(area.offset.y), float(area.extent.width),
                              float(area.extent.height), 0, 1};
    dfn_.vkCmdSetViewport(commands_work_, 0, 1, &viewport);
    dfn_.vkCmdSetScissor(commands_work_, 0, 1, &area);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->pipeline);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS, layout_pipeline_depth_resolve_ms_, 0,
                                 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_depth_resolve_ms_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(delta), delta);
    dfn_.vkCmdDraw(commands_work_, 3, 1, 0, 0);
    dfn_.vkCmdEndRenderPass(commands_work_);
    if (draws_) draws_->NotifyGraphicsExternalState();
    ++msaa_depth_resolves_;
    sa2::render::NoteMsaaUse();
    return true;
  }

  /*
   * sa2_gpu_resolve_alias. A color resolve that repeats the previous one into another texture is not copied: that
   * texture (the alias) takes the bytes of the first one (the source), which holds them, later:
   *  - when something reads the alias (a draw, the Swap, a restore): copied then, from the source (MaterializeAlias);
   *  - before the source changes in any way (a resolve or a restore into it, another image at its address): copied
   *    first, or, when the alias and the source are whole textures of the same shape and the coming resolve covers the
   *    whole source, the two images are exchanged (ExchangeAlias): the alias gets the image with the bytes and the
   *    source the other one, which that resolve overwrites whole;
   *  - never, when a resolve into the alias covers it first: nobody could see those bytes.
   * Why it is exact: the previous resolve (MakeAlias's `p`) copied the same rectangle of the same render target image,
   * the command right before this one, with no draw, clear (its own command's included), EDRAM import or restore in
   * between, so the render target still holds the bytes it copied; and its texture still has the revision it left,
   * so it holds them too. Every place that writes a resolved texture calls BeforeResolvedWrite first and every place
   * that reads one materializes it first (ResolvedTexture, ResolvedTextureForFetch, Present, RestoreContent).
   * GUARD: an alias whose source is not what it was (image or revision) cannot get its bytes: DIFFERENCE in the log
   * and the enhancement turns off for the session. Checked at every materialization and every drop.
   */
  struct AliasResolve {
    uint32_t source = 0;  // the texture that holds the bytes
    VkImage source_image = VK_NULL_HANDLE;
    uint64_t source_revision = 0;
    VkImage image = VK_NULL_HANDLE;  // the alias's image when it was made
    int32_t sx = 0, sy = 0;          // the rectangle in the source
    uint32_t dx = 0, dy = 0, width = 0, height = 0;  // and in the alias
  };
  struct LastResolve {  // the color resolve recorded for real by the previous command
    bool valid = false;
    VkImage target = VK_NULL_HANDLE;  // the render target image it read
    int32_t x0 = 0, y0 = 0;
    uint32_t width = 0, height = 0;
    uint32_t address = 0;  // and the texture it wrote, at (dx, dy)
    VkImage image = VK_NULL_HANDLE;
    uint64_t revision = 0;
    uint32_t dx = 0, dy = 0;
    uint64_t drawn = 0, transfers = 0, restored = 0;  // draws, EDRAM imports and restores so far
    // EDRAM mode 4's tile ownership version (edram4_version_): every write of a render target publishes its tiles,
    // the draws and also what Drawn() does not count, Direct3D's Clear draws redirected to EDRAM clears (SA2 clears
    // its frame between the two resolves of a fade: an alias made then showed the frame from before the clear).
    uint64_t edram_version = 0;
  };

  bool MakeAlias(const LastResolve& p, const Image& target, Resolved& resolved, uint32_t address, int32_t x0,
                 int32_t y0, uint32_t dx, uint32_t dy, uint32_t width, uint32_t height) {
    if (!g_resolve_alias.active || off_alias_ || !p.valid || REXCVAR_GET(sa2_native_lazy_front) ||
        !pending_fronts_.empty())
      return false;
    // Only in EDRAM mode 4, where every write of a render target moves edram4_version_ (the other modes have writes
    // no counter here sees).
    if (REXCVAR_GET(sa2_native_edram_alias_mode) != 4) return false;
    if (p.target != target.image || target.invalid_content || p.x0 != x0 || p.y0 != y0 || p.width != width ||
        p.height != height || p.address == address || p.drawn != (draws_ ? draws_->Drawn() : 0) ||
        p.transfers != edram4_transfers_all_ || p.restored != restored_pixels_ || p.edram_version != edram4_version_)
      return false;
    // Small resolves are read back for the guest right after their copy (ResolvedRead): never aliased.
    const int32_t texels_read = REXCVAR_GET(sa2_native_resolved_read_texels);
    if (texels_read > 0 && uint64_t(width) * height <= uint64_t(texels_read)) return false;
    const auto s = resolved_.find(p.address);
    if (s == resolved_.end() || s->second.image.image != p.image || s->second.revision != p.revision ||
        s->second.alias_pending || !s->second.image.prepared || !resolved.image.prepared ||
        s->second.image.format != resolved.image.format || resolved.image.image == p.image ||
        s->second.image.scale_q != resolved.image.scale_q)  // sa2-nx render resolution
      return false;
    // This texture's bytes change: the aliases reading from it get theirs first, and its own alias is covered by
    // this one or copied first.
    BeforeResolvedWrite(address, dx, dy, width, height);
    if (off_alias_ || aliases_.count(address)) return false;
    AliasResolve& alias = aliases_[address];
    alias.source = p.address;
    alias.source_image = p.image;
    alias.source_revision = p.revision;
    alias.image = resolved.image.image;
    alias.sx = int32_t(p.dx);
    alias.sy = int32_t(p.dy);
    alias.dx = dx;
    alias.dy = dy;
    alias.width = width;
    alias.height = height;
    resolved.alias_pending = true;
    ResolvedWritten(address);  // new bytes for its readers (crops)
    // The draws keep the slot of a texture for the rest of the frame without asking for it again
    // (PrepareTexture's caches): those that use this image ask again, so a later read materializes it. Nothing
    // to do for a texture nobody samples.
    if (draws_) draws_->InvalidateImages(resolved.image.image, VK_NULL_HANDLE);
    ++aliases_made_;
    if (DiagAlias())
      REXLOG_INFO("[diag alias] frame {} MAKE {:08X} ({},{} {}x{}) = {:08X} ({},{}) rev {} img {}", presented_, address,
                  dx, dy, width, height, p.address, p.dx, p.dy, p.revision, (void*)resolved.image.image);
    g_resolve_alias.UseSampled();
    return true;
  }

  bool DiagAlias() const {
    static const int32_t from = REXCVAR_GET(sa2_diag_alias_frames);
    return from > 0 && presented_ >= uint64_t(from) && presented_ < uint64_t(from) + 40;
  }

  // The source of `alias` still holds the bytes the alias was made with (GUARD).
  bool AliasSourceIntact(const AliasResolve& alias) {
    sa2::enh::VerifyCheck();
    const auto s = resolved_.find(alias.source);
    return s != resolved_.end() && s->second.image.image == alias.source_image &&
           s->second.revision == alias.source_revision && s->second.image.prepared;
  }

  void AliasFailure(uint32_t address, const char* why) {
    if (off_alias_) return;
    off_alias_ = true;
    sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
    REXLOG_ERROR("[native] targets: DIFFERENCE: resolve alias {:08X} ({}). sa2_gpu_resolve_alias off for the rest of "
                 "the session: every resolve is copied", address, why);
  }

  void DropAlias(uint32_t address) {
    aliases_.erase(address);
    if (const auto r = resolved_.find(address); r != resolved_.end()) r->second.alias_pending = false;
  }

  // The alias at `address` gets its bytes now: copied from its source (on the 3D engine when it applies), outside
  // any pass. false if the source no longer holds them (GUARD).
  bool MaterializeAlias(uint32_t address) {
    const auto a = aliases_.find(address);
    if (a == aliases_.end()) return true;
    const AliasResolve alias = a->second;
    DropAlias(address);
    const auto r = resolved_.find(address);
    const auto s = resolved_.find(alias.source);
    if (r == resolved_.end() || r->second.image.image != alias.image || !r->second.image.prepared) {
      AliasFailure(address, "its image changed without its bytes");
      return false;
    }
    if (!AliasSourceIntact(alias)) {
      AliasFailure(address, "its source changed before it was read");
      return false;
    }
    if (!Record()) {
      AliasFailure(address, "the copy could not be recorded");
      return false;
    }
    if (draws_) draws_->FinishPass();  // may come from a draw (ResolvedTexture): the copy goes outside the pass
    MarkGpu(kGpuCopies);
    if (!ResolveColorFrag(s->second.image, r->second.image, alias.sx, alias.sy, alias.dx, alias.dy, alias.width,
                          alias.height)) {
      VkImageCopy copy{};
      copy.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copy.srcOffset = {alias.sx, alias.sy, 0};
      copy.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      copy.dstOffset = {int32_t(alias.dx), int32_t(alias.dy), 0};
      copy.extent = {alias.width, alias.height, 1};
      copy = HostCopy(s->second.image, r->second.image, copy);  // sa2-nx render resolution
      BarrierBeforeCopyResolve();
      copy_image_(commands_work_, s->second.image.image, VK_IMAGE_LAYOUT_GENERAL, r->second.image.image,
                  VK_IMAGE_LAYOUT_GENERAL, 1, &copy);
      BarrierAfterCopyResolve();
    }
    // Not ResolvedWritten: its revision changed when the alias was made, and these are the bytes it promised.
    ResolvedPixelsCopied(address, uint64_t(alias.width) * alias.height);
    ++aliases_copied_;
    if (DiagAlias())
      REXLOG_INFO("[diag alias] frame {} COPY {:08X} ({},{} {}x{}) <- {:08X} ({},{}) img {} <- {}", presented_, address,
                  alias.dx, alias.dy, alias.width, alias.height, alias.source, alias.sx, alias.sy,
                  (void*)r->second.image.image, (void*)s->second.image.image);
    return true;
  }

  // The source of the alias at `alias_address` is about to be overwritten whole (rectangle dx, dy, width, height):
  // the alias takes the source's image, which holds its bytes, and the source the alias's image, which that write
  // covers. Only whole textures of the same shape, nothing else holding either image. false: not done (copy it).
  bool ExchangeAlias(uint32_t alias_address, uint32_t dx, uint32_t dy, uint32_t width, uint32_t height) {
    const auto a = aliases_.find(alias_address);
    if (a == aliases_.end()) return false;
    const AliasResolve alias = a->second;
    const auto r = resolved_.find(alias_address);
    const auto s = resolved_.find(alias.source);
    if (r == resolved_.end() || s == resolved_.end() || r == s) return false;
    Image& mine = r->second.image;
    Image& source = s->second.image;
    if (!width || dx || dy || width != source.width || height != source.height || alias.sx || alias.sy ||
        alias.dx || alias.dy || alias.width != mine.width || alias.height != mine.height ||
        mine.width != source.width || mine.height != source.height || mine.format != source.format ||
        mine.scale_q != source.scale_q ||  // sa2-nx render resolution
        mine.sample_count != source.sample_count || mine.accepts_target_of_copy != source.accepts_target_of_copy ||
        mine.guest_width || mine.guest_height || source.guest_width || source.guest_height ||
        mine.image != alias.image || !mine.prepared || !AliasSourceIntact(alias))
      return false;
    // A render target whose content went out to one of them in a resolve without copy reads it back by address.
    for (const auto* images : {&targets_, &depths_})
      for (const auto& [key, image] : *images)
        if (image.invalid_content && (image.resolved_base == alias_address || image.resolved_base == alias.source))
          return false;
    std::swap(mine.image, source.image);
    std::swap(mine.memory, source.memory);
    std::swap(mine.pool_block, source.pool_block);
    std::swap(mine.view, source.view);
    std::swap(mine.prepared, source.prepared);
    DropAlias(alias_address);
    ResolvedWritten(alias_address);  // another image at both addresses
    ResolvedWritten(alias.source);
    // Only what points to these two images (as SwapWithResolved): the views go with their images.
    if (draws_) draws_->InvalidateImages(mine.image, source.image);
    ++aliases_exchanged_;
    return true;
  }

  // sa2_gpu_resolve_alias: the resolved texture at `address` is about to receive new bytes, in the rectangle
  // (dx, dy, width, height) when the caller is sure to write it (width 0: unknown). The aliases reading from it get
  // their bytes first (an exchange when the write covers it whole, otherwise a copy). Its own alias is dropped when
  // the write covers it (nobody could see those bytes), otherwise copied first.
  void BeforeResolvedWrite(uint32_t address, uint32_t dx = 0, uint32_t dy = 0, uint32_t width = 0,
                           uint32_t height = 0) {
    if (DiagAlias()) {
      REXLOG_INFO("[diag alias] frame {} WRITE {:08X} ({},{} {}x{}), {} aliases; transfers {} restored {}", presented_,
                  address, dx, dy, width, height, aliases_.size(), edram4_transfers_all_, restored_pixels_);
      for (const uint32_t t : {0u, 8u, 16u, 24u, 0x100u, 0x108u, 0x2C0u, 0x2C8u}) {
        const auto& o = owners_tiles_edram4_[t];
        if (o.image)
          REXLOG_INFO("[diag alias]   tile {:03X}: owner {} base {:03X} {}x{} depth {} local {} version {}", t,
                      (void*)o.image->image, o.image->edram_base, o.image->width, o.image->height,
                      o.image->edram_depth, o.tile_local, o.version);
        else
          REXLOG_INFO("[diag alias]   tile {:03X}: no owner", t);
      }
    }
    if (aliases_.empty()) return;
    std::array<uint32_t, 8> readers{};
    size_t n = 0;
    bool more = false;
    for (const auto& [alias_address, alias] : aliases_) {
      if (alias.source != address) continue;
      if (n < readers.size()) readers[n++] = alias_address;
      else more = true;
    }
    // An exchange gives the source's image to one alias: with several, the others would lose their bytes, so then
    // all are copied.
    for (size_t i = 0; i < n; ++i)
      if (n > 1 || more || !ExchangeAlias(readers[i], dx, dy, width, height)) MaterializeAlias(readers[i]);
    if (more) {  // (more than 8 aliases of one texture: not expected)
      for (auto it = aliases_.begin(); it != aliases_.end();) {
        const uint32_t alias_address = it->first;
        const bool reader = it->second.source == address;
        ++it;
        if (reader) MaterializeAlias(alias_address);
      }
    }
    const auto a = aliases_.find(address);
    if (a == aliases_.end()) return;
    const AliasResolve& alias = a->second;
    if (width && dx <= alias.dx && dy <= alias.dy && uint64_t(dx) + width >= uint64_t(alias.dx) + alias.width &&
        uint64_t(dy) + height >= uint64_t(alias.dy) + alias.height) {
      if (!AliasSourceIntact(alias)) {  // GUARD: a write to its source went around BeforeResolvedWrite
        AliasFailure(address, "its source changed without giving it its bytes");
      }
      DropAlias(address);
      ++aliases_dropped_;
      if (DiagAlias())
        REXLOG_INFO("[diag alias] frame {} DROP {:08X} (covered by {},{} {}x{})", presented_, address, dx, dy, width,
                    height);
    } else {
      MaterializeAlias(address);
    }
  }

  // The image at `address` is about to be parked or destroyed: its readers and its own alias get their bytes first.
  void AliasesBeforeImageGoes(uint32_t address) {
    if (aliases_.empty()) return;
    BeforeResolvedWrite(address);
    MaterializeAlias(address);
  }

  // The copy inventory (NoteCopy) counts the pixels each texture really receives.
  void ResolvedPixelsCopied(uint32_t base, uint64_t pixels) {
    const auto c = copies_per_target_.find(base);
    if (c != copies_per_target_.end()) c->second.pixels += pixels;
  }

  // sa2_native_conversion_frag: one run of tiles of a 32-bit color view into another one as a fragment
  // pass (see src/native/native/shaders/me_edram_color_to_color.frag). Returns false WITHOUT recording anything when it does not
  // apply (the caller then records the compute shader).
  bool ConvertAliasColorFragEDRAM4(Image& source, Image& target, uint32_t source_start,
                                     uint32_t target_start, uint32_t count) {
    const auto NoFrag = [&](uint32_t reason) {
      ++conv_color_frag_reasons_[std::min<uint32_t>(reason, 15)];
      return false;
    };
    if (conv_color_frag_failed_ || !count || source.edram_64bpp != target.edram_64bpp ||
        source.scale_q != target.scale_q ||  // sa2-nx render resolution: one scale per transfer
        source.edram_depth || target.edram_depth || source.guest_width || target.guest_width ||
        !me::native::IsSingleSample(uint32_t(source.sample_count)) ||
        !SamplesSupported(target))  // sa2_msaa: into a host multisampled target (every sample gets the pixel)
      return NoFrag(0);
    // Variant 0: the three 32-bit combinations of the compute shaders (16F -> 8888, 8888 -> 16F, 16F -> 16F).
    // Variant 1: 64-bit FP16 views both sides (raw UINT texels, the compute raw64 -> raw64 shader).
    const bool source_8888 = source.format == VK_FORMAT_R8G8B8A8_UNORM;
    const bool target_8888 = target.format == VK_FORMAT_R8G8B8A8_UNORM;
    const uint32_t variant = source.edram_64bpp ? 1u : 0u;
    if (variant == 0) {
      // RGBA8 -> RGBA8 only between different sample layouts (ConvertAliasEDRAM's MSAA alias case): the bytes
      // pass through (bit 16 of both formats); an identical layout is a plain tile copy.
      const bool same_layout = source.edram_msaa_x == target.edram_msaa_x &&
                               source.edram_msaa_y == target.edram_msaa_y;
      if ((source_8888 && target_8888 && same_layout) ||
          (!source_8888 && source.format != VK_FORMAT_R16G16B16A16_SFLOAT) ||
          (!target_8888 && target.format != VK_FORMAT_R16G16B16A16_SFLOAT))
        return NoFrag(1);
    } else if (!target.edram_64bpp || source.format != VK_FORMAT_R16G16B16A16_SFLOAT ||
               target.format != VK_FORMAT_R16G16B16A16_SFLOAT) {
      return NoFrag(2);
    }
    if (!PitchTilesEDRAM(source) || !PitchTilesEDRAM(target)) return NoFrag(3);
    // Destination rectangle of the run (a 64-bit tile is 40 pixels wide, a 32-bit one 80, before MSAA).
    VkRect2D scissor;
    {
      const uint32_t pitch = PitchTilesEDRAM(target);
      const uint32_t tile_w = 80u >> (uint32_t(target.edram_64bpp) + target.edram_msaa_x);
      const uint32_t tile_h = 16u >> target.edram_msaa_y;
      const uint32_t first = target_start / pitch, last = (target_start + count - 1) / pitch;
      const uint32_t sx = first == last ? (target_start % pitch) * tile_w : 0;
      const uint32_t sy = first * tile_h;
      const uint32_t right = first == last
          ? std::min(target.width, ((target_start + count - 1) % pitch + 1) * tile_w) : target.width;
      const uint32_t bottom = std::min(target.height, (last + 1) * tile_h);
      if (!tile_w || !tile_h || sx >= target.width || sy >= target.height || right <= sx || bottom <= sy)
        return NoFrag(4);
      scissor = {{int32_t(sx), int32_t(sy)}, {right - sx, bottom - sy}};
    }
    VkImageView view_source = source.view, view_target = target.view;
    if (variant) {
      view_source = GetViewRaw64EDRAM4(source);
      view_target = GetViewRaw64RTEDRAM4(target);
      if (!view_source || !view_target) return NoFrag(5);
    }
    PassConversionColorFrag* pass = EnsureConversionColorFrag(
        variant ? VK_FORMAT_R16G16B16A16_UINT : target.format, variant, uint32_t(target.sample_count));
    if (!pass) return NoFrag(6);
    auto [framebuffer, new_value] = framebuffers_conv_color_frag_.try_emplace({target.image, variant}, VK_NULL_HANDLE);
    if (new_value) {
      VkFramebufferCreateInfo info{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
      info.renderPass = pass->render_pass;
      info.attachmentCount = 1;
      info.pAttachments = &view_target;
      info.width = HostWidth(target);
      info.height = HostHeight(target);
      info.layers = 1;
      if (dfn_.vkCreateFramebuffer(device_, &info, nullptr, &framebuffer->second) != VK_SUCCESS) {
        framebuffers_conv_color_frag_.erase(framebuffer);
        return NoFrag(7);
      }
    }
    // Closing a pass in the capacity helper can rehash the map: keep the handle, not the iterator.
    const VkFramebuffer framebuffer_conversion = framebuffer->second;
    if (!EnsureCapacityConversionEDRAM4("color alias")) return false;
    // Diagnostic label of the GPU interval (report "GPU time by pass": VS = 16 + class, PS = tiles).
    LabelMarkGpu(((16u + (variant ? 4u : source_8888 ? 2u : target_8888 ? 1u : 3u)) << 16) |
                      std::min(count, 0xFFFFu));
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo reserve{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    reserve.descriptorPool = slot.pool_conv_color_frag;
    reserve.descriptorSetCount = 1;
    reserve.pSetLayouts = &layout_conv_color_frag_;
    if (dfn_.vkAllocateDescriptorSets(device_, &reserve, &set) != VK_SUCCESS) return false;
    ++slot.conversions_edram;
    VkDescriptorImageInfo sampled{sampler_depth_edram_, view_source, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.pImageInfo = &sampled;
    dfn_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);
    struct Constants {
      uint32_t source_size[2], target_size[2];
      uint32_t pitch_source, pitch_target, source_start, target_start, count;
      uint32_t source_format, target_format, source_64bpp, target_64bpp;
      uint32_t source_msaa_x, source_msaa_y, target_msaa_x, target_msaa_y;
      uint32_t tile_width, tile_height;  // sa2-nx render resolution: the tile in host samples
    } constants{{HostWidth(source), HostHeight(source)}, {HostWidth(target), HostHeight(target)},
                 PitchTilesEDRAM(source), PitchTilesEDRAM(target), source_start, target_start, count,
                 uint32_t(source.edram_format) | (source_8888 ? 0x10000u : 0u),
                 uint32_t(target.edram_format) | (target_8888 ? 0x10000u : 0u), variant, variant,
                 source.edram_msaa_x, source.edram_msaa_y, target.edram_msaa_x, target.edram_msaa_y,
                 ScaledTileWidth(target.scale_q), ScaledTileHeight(target.scale_q)};
    static_assert(sizeof(Constants) == 76);
    if (draws_) draws_->FinishPass();
    if (edram4_batch_.active) {
      CloseComputeBatchEDRAM4();
      if (!CloseImportBatchEDRAM4()) return false;
    }
    // sa2-nx render resolution: the run's rectangle in host pixels.
    const VkRect2D host_scissor = HostRect(target, scissor);
    VkRenderPassBeginInfo begin{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    begin.renderPass = pass->render_pass;
    begin.framebuffer = framebuffer_conversion;
    begin.renderArea = host_scissor;
    dfn_.vkCmdBeginRenderPass(commands_work_, &begin, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{float(host_scissor.offset.x), float(host_scissor.offset.y),
                              float(host_scissor.extent.width), float(host_scissor.extent.height), 0, 1};
    dfn_.vkCmdSetViewport(commands_work_, 0, 1, &viewport);
    dfn_.vkCmdSetScissor(commands_work_, 0, 1, &host_scissor);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS, pass->pipeline);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 layout_pipeline_conv_color_frag_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conv_color_frag_, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                            sizeof(constants), &constants);
    dfn_.vkCmdDraw(commands_work_, 3, 1, 0, 0);
    dfn_.vkCmdEndRenderPass(commands_work_);
    if (draws_) draws_->NotifyGraphicsExternalState();
    if (++conv_color_frag_uses_ == 1)
      REXLOG_INFO("[native] EDRAM mode4 color conversions as fragment passes (sa2_native_conversion_frag): "
                  "first one {:03X}/f{} -> {:03X}/f{}, {} tiles, rect {},{}+{}x{}",
                  source.edram_base, source.edram_format, target.edram_base, target.edram_format, count,
                  scissor.offset.x, scissor.offset.y, scissor.extent.width, scissor.extent.height);
    return true;
  }

  bool ConvertAliasEDRAM(Image& source, Image& target,
                           uint32_t tile_source_start = 0,
                           uint32_t tile_target_start = 0,
                           uint32_t tiles_count = 0) {
    if (source.scale_q != target.scale_q) return false;  // sa2-nx render resolution: one scale per transfer
    // sa2_msaa: the compute conversions read and write single-sample storage views; the fragment one (RGBA8 below)
    // takes a single-sample source into a multisampled target.
    if (!me::native::IsSingleSample(uint32_t(source.sample_count)) ||
        (!me::native::IsSingleSample(uint32_t(target.sample_count)) &&
         !(source.format == VK_FORMAT_R8G8B8A8_UNORM && target.format == VK_FORMAT_R8G8B8A8_UNORM)))
      return false;
    // sa2-nx: Direct3D's Clear of SA2's 1x 1280x720 back buffer draws through a 4x MSAA alias of the same EDRAM
    // tiles (640x360 pixels of 4 samples each, pitch 640): both views are RGBA8, only the sample layout differs.
    // No compute shader covers that pair (they convert formats), so it always takes the fragment pass, which maps
    // physical samples between the two layouts. It used to be skipped as unsupported
    // (banjo_native_edram4_skip_unsupported): the back buffer was never cleared and every boot screen stayed
    // behind the next ones as a ghost.
    if (source.format == VK_FORMAT_R8G8B8A8_UNORM && target.format == VK_FORMAT_R8G8B8A8_UNORM) {
      if (!tiles_count || source.edram_64bpp || target.edram_64bpp) return false;
      return ConvertAliasColorFragEDRAM4(source, target, tile_source_start, tile_target_start, tiles_count);
    }
    uint32_t pipeline = UINT32_MAX;
    if (source.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
        target.format == VK_FORMAT_R8G8B8A8_UNORM) {
      pipeline = 0;
    } else if (source.format == VK_FORMAT_R8G8B8A8_UNORM &&
               target.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
      pipeline = 1;
    } else if (source.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
               target.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
      pipeline = 2;
    } else {
      return false;
    }
    const bool raw_alias = REXCVAR_GET(sa2_native_edram_alias_mode) == 4 &&
                           (source.edram_64bpp || target.edram_64bpp);
    VkImageView raw_source = VK_NULL_HANDLE, raw_destination = VK_NULL_HANDLE;
    if (raw_alias) {
      if (source.edram_64bpp) raw_source = GetViewRaw64EDRAM4(source);
      if (target.edram_64bpp) raw_destination = GetViewRaw64EDRAM4(target);
      if ((source.edram_64bpp && !raw_source) || (target.edram_64bpp && !raw_destination))
        return FailureEDRAM4("raw FP16 color alias UINT views unavailable", target, &source);
      pipeline = source.edram_64bpp
          ? (target.edram_64bpp ? 8u : target.format == VK_FORMAT_R8G8B8A8_UNORM ? 4u : 6u)
          : source.format == VK_FORMAT_R8G8B8A8_UNORM ? 5u : 7u;
    }
    // sa2_native_conversion_frag: the 32-bit pipelines (0-2) and raw64 -> raw64 (8) as a fragment pass.
    if ((pipeline <= 2 || pipeline == 8) && tiles_count && REXCVAR_GET(sa2_native_edram_alias_mode) == 4 &&
        REXCVAR_GET(sa2_native_conversion_frag) &&
        ConvertAliasColorFragEDRAM4(source, target, tile_source_start, tile_target_start, tiles_count))
      return true;
    else if ((pipeline <= 2 || pipeline == 8) && tiles_count && REXCVAR_GET(sa2_native_conversion_frag))
      ++conv_color_frag_rejections_;
    if (pipeline >= pipelines_conversion_edram_.size() ||
        pipelines_conversion_edram_[pipeline] == VK_NULL_HANDLE ||
        !EnsureCapacityConversionEDRAM4("color alias")) {
      return false;
    }
    LabelMarkGpu(((pipeline + 1u) << 16) | std::min(tiles_count, 0xFFFFu));  // report label (compute)
    SlotWork& slot = slots_[slot_];
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo reserve{};
    reserve.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    reserve.descriptorPool = slot.pool_conversion_edram;
    reserve.descriptorSetCount = 1;
    reserve.pSetLayouts = &layout_conversion_edram_;
    if (dfn_.vkAllocateDescriptorSets(device_, &reserve, &set) != VK_SUCCESS) {
      return false;
    }
    ++slot.conversions_edram;
    const VkDescriptorImageInfo images[2] = {
        {VK_NULL_HANDLE, raw_source ? raw_source : source.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, raw_destination ? raw_destination : target.view, VK_IMAGE_LAYOUT_GENERAL},
    };
    VkWriteDescriptorSet writes[2]{};
    for (uint32_t i = 0; i < 2; ++i) {
      writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      writes[i].dstSet = set;
      writes[i].dstBinding = i;
      writes[i].descriptorCount = 1;
      writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
      writes[i].pImageInfo = &images[i];
    }
    dfn_.vkUpdateDescriptorSets(device_, 2, writes, 0, nullptr);

    struct ConstantsConversion {
      uint32_t source_size[2];
      uint32_t target_size[2];
      uint32_t pitch_tiles_source;
      uint32_t pitch_tiles_target;
      uint32_t tile_source_start;
      uint32_t tile_target_start;
      uint32_t tiles_count;
      uint32_t source_format;
      uint32_t target_format;
      uint32_t source_64bpp;
      uint32_t target_64bpp;
      uint32_t source_msaa_x, source_msaa_y;
      uint32_t target_msaa_x, target_msaa_y;
      uint32_t tile_width, tile_height;  // sa2-nx render resolution: the tile in host samples
    } constants = {{HostWidth(source), HostHeight(source)},
                     {HostWidth(target), HostHeight(target)},
                     PitchTilesEDRAM(source),
                     PitchTilesEDRAM(target),
                     tile_source_start,
                     tile_target_start,
                     tiles_count,
                     source.edram_format,
                     target.edram_format,
                     uint32_t(source.edram_64bpp),
                     uint32_t(target.edram_64bpp),
                     source.edram_msaa_x, source.edram_msaa_y,
                     target.edram_msaa_x, target.edram_msaa_y,
                     ScaledTileWidth(target.scale_q), ScaledTileHeight(target.scale_q)};
    static_assert(sizeof(ConstantsConversion) == 76);
    VkMemoryBarrier before{};
    before.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    // Destination aliases may still be sampled by preceding draws. Order reads as
    // well as writes; COLOR_ATTACHMENT_OUTPUT alone does not cover fragment reads.
    before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    before.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    if (edram4_batch_.active && !CloseImportBatchEDRAM4()) return false;
    if (edram4_batch_.compute_pending)
      ++edram4_batch_saved_barriers_;
    else
    dfn_.vkCmdPipelineBarrier(
        commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    dfn_.vkCmdBindPipeline(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                           pipelines_conversion_edram_[pipeline]);
    dfn_.vkCmdBindDescriptorSets(commands_work_, VK_PIPELINE_BIND_POINT_COMPUTE,
                                 layout_pipeline_conversion_edram_, 0, 1, &set, 0, nullptr);
    dfn_.vkCmdPushConstants(commands_work_, layout_pipeline_conversion_edram_,
                            VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants), &constants);
    const uint32_t dispatch_width = tiles_count
                                        ? tiles_count * constants.tile_width
                                        : constants.pitch_tiles_target * constants.tile_width;
    const uint32_t dispatch_height = tiles_count ? constants.tile_height
                                                   : HostHeight(target) << target.edram_msaa_y;
    dfn_.vkCmdDispatch(commands_work_, (dispatch_width + 7) / 8,
                       (dispatch_height + 7) / 8, 1);
    if (edram4_batch_.active) {
      edram4_batch_.compute_pending = true;
      return true;
    }
    VkMemoryBarrier after{};
    after.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    after.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    after.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
                             VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                             VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(
        commands_work_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT |
            VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        0, 1, &after, 0, nullptr, 0, nullptr);
    return true;
  }

  Image* GetDepth(uint32_t base, uint32_t format, uint32_t pitch) {
    // The inputs of the path below: the arguments, the EDRAM MSAA mode, whether the draw may use the depth raster
    // grid, and the two cvars it reads (format_depth_ and the D32S8 answer are fixed once known).
    const int32_t alias_mode = REXCVAR_GET(sa2_native_edram_alias_mode);
    const bool samples_x = REXCVAR_GET(sa2_native_depth_samples_x);
    const uint64_t memo_in = (uint64_t(uint32_t(alias_mode) & 0xFF) << 48) | (uint64_t(samples_x) << 47) |
                             (uint64_t(depth_raster_grid_eligible_) << 46) |
                             (uint64_t(msaa_edram_actual_ & 0xF) << 40) | (uint64_t(base & 0xFFFFF) << 20) |
                             (uint64_t(format & 0xF) << 16) | (pitch & 0xFFFF);
    TargetMemo& memo = depth_memo_[(base ^ (pitch >> 4)) & 3];
    if (g_target_memo.active && memo.in == memo_in) {
      g_target_memo.UseSampled();
      return memo.image;
    }
    Image* const found = GetDepthLookup(base, format, pitch);
    if (found && g_target_memo.active && base <= 0xFFFFF && format <= 0xF && pitch <= 0xFFFF &&
        msaa_edram_actual_ <= 0xF)
      memo = {memo_in, found};
    return found;
  }

  Image* GetDepthLookup(uint32_t base, uint32_t format, uint32_t pitch) {
    if (!pitch || format_depth_ == VK_FORMAT_UNDEFINED) {
      Reject(11, "no depth target (pitch 0 or format unavailable)");
      return nullptr;
    }
    const bool half = format == 1 && REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    const VkFormat host_format = half ? VK_FORMAT_D32_SFLOAT_S8_UINT : format_depth_;
    if (half) {
      // Asked once: this ran on every draw (2 % of the Switch ring thread inside NVK).
      if (d32s8_supported_ < 0) {
        VkFormatProperties props{};
        vulkan_device_->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
            vulkan_device_->physical_device(), host_format, &props);
        const VkFormatFeatureFlags needed = VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT |
            VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_SRC_BIT |
            VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
        d32s8_supported_ = (props.optimalTilingFeatures & needed) == needed ? 1 : 0;
      }
      if (!d32s8_supported_) {
        Reject(12, "MODE4 half-range requires sampled D32S8 attachment and transfers");
        return nullptr;
      }
    }
    const auto raster_grid = me::native::GetMsaaRasterGrid(
        REXCVAR_GET(sa2_native_depth_samples_x) &&
          REXCVAR_GET(sa2_native_edram_alias_mode) == 4,
        msaa_edram_actual_, depth_raster_grid_eligible_);
    if (!raster_grid.valid) {
      Reject(12, "invalid depth MSAA raster grid");
      return nullptr;
    }
    const uint32_t grid_x_log2 = raster_grid.grid_x == 2 ? 1u : 0u;
    const uint64_t key = (uint64_t(grid_x_log2) << 49) |
                           (uint64_t(half) << 48) | (uint64_t(msaa_edram_actual_) << 32) |
                           (uint64_t(base) << 20) | (uint64_t(format) << 16) | pitch;
    auto it = depths_.find(key);
    if (it != depths_.end()) {
      return &it->second;
    }
    const uint32_t height = std::min(kMaxTargetHeight, std::max<uint32_t>(720, (pitch + 15) & ~15u));
    // Smaller shadow map.
    //
    // The game requests 1600x1600 shadow maps, which is 2000 of the 2048 80x16 tiles of the Xbox 360 EDRAM. The
    // map can be drawn at sa2_native_shadows_scale percent and upscaled when resolved, so the texture
    // the scene samples is the usual one and only the detail goes down.
    //
    // It is recognized by its 1600x1600 size, which no other render target uses. The scale is read only
    // once (when the first map is created) so that changing it at runtime does not leave old images unused.
    uint32_t image_width = pitch, image_height = height;
    uint32_t guest_width = 0, guest_height = 0;
    if (pitch == kSideShadows && height == kSideShadows) {
      if (scale_shadows_ == 0) {
        scale_shadows_ = uint32_t(std::clamp(REXCVAR_GET(sa2_native_shadows_scale), 50, 100));
        if (!blit_ || !scalable_depth_) {
          scale_shadows_ = 100;
        }
      }
      if (scale_shadows_ < 100) {
        image_width = std::max<uint32_t>(64, ((pitch * scale_shadows_ / 100) + 15) & ~15u);
        image_height = std::max<uint32_t>(64, ((height * scale_shadows_ / 100) + 15) & ~15u);
        guest_width = pitch;
        guest_height = height;
        REXLOG_INFO("[native] targets: shadow map at {} %: drawn and resolved at {}x{} (the game asks for {}x{})",
                    scale_shadows_, image_width, image_height, pitch, height);
      }
    }
    if (grid_x_log2 && (guest_width || guest_height || image_width > 4096)) {
      Reject(12, "expanded MSAA depth cannot combine with scaled shadows or exceed 8192 width");
      return nullptr;
    }
    image_width <<= grid_x_log2;
    // The usage flag that decides whether there is hierarchical depth culling.
    //
    // The driver only assigns a ZCULL plane if the usage flags fit in
    //   DEPTH_STENCIL_ATTACHMENT | TRANSFER_SRC | SAMPLED | INPUT_ATTACHMENT
    // (nvk_image.c). TRANSFER_DST disqualifies the image, and we only request it for
    // vkCmdClearDepthStencilImage, which the specification requires with that usage.
    //
    // TRANSFER_DST is dropped from the scene depth and kept for the shadow map. The scene resolves a
    // sub-rectangle of a larger image, so it is never "whole" and never swaps with its resolved texture; the
    // shadow map does swap (SwapWithResolved requires accepts_target_of_copy) and a restore is a copy
    // into this image (RestoreContent -> copy_image_), which requires TRANSFER_DST.
    //
    // In exchange, the scene depth can no longer be cleared with vkCmdClearDepthStencilImage: it is cleared
    // by opening a pass with loadOp = CLEAR (DrawsVulkan::ClearDepthInPass).
    // With sa2_native_zcull off, every image gets TRANSFER_DST: none receives a ZCULL plane and clears
    // go back to vkCmdClearDepthStencilImage, the path used before ZCULL.
    const bool zcull = REXCVAR_GET(sa2_native_zcull);
    const bool is_map_of_shadows = (pitch == kSideShadows && height == kSideShadows);
    // 1x depth views receive imported stencil through the copy engine (TRANSFER_DST).
    // sa2-nx (sa2_native_zcull_1x_depth): SA2's 1x depth views never import stencil once the aliased Clears are
    // redirected (0 EDRAM conversions per frame), so the copy-engine stencil path is not worth their ZCULL plane;
    // an import that does happen takes the masked stencil passes (copy_stencil requires accepts_target_of_copy).
    const bool with_transfer_dst = is_map_of_shadows || !zcull ||
        (REXCVAR_GET(sa2_native_edram4_stencil_copy) && msaa_edram_actual_ == 0 &&
         !REXCVAR_GET(sa2_native_zcull_1x_depth));
    VkImageUsageFlags usage_depth = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT;
    if (with_transfer_dst) {
      usage_depth |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
    Image image;
    image.edram_depth = true;
    image.depth_float24_half = half;
    image.edram_base = uint16_t(base);
    image.edram_format = uint8_t(format);
    image.raster_grid_x = grid_x_log2;
    image.edram_msaa_x = uint32_t(msaa_edram_actual_ >= 2) - grid_x_log2;
    image.edram_msaa_y = msaa_edram_actual_ >= 1;
    // Source of the depth copies (CopyDepth): TRANSFER_SRC is required.
    // SAMPLED: with sa2_native_resolver_no_copy this image can end up being the resolved texture.
    // sa2_msaa: the game's single-sample depth targets get host samples (not the expanded or scaled-shadow ones).
    const bool multisampled = msaa_samples_ > 1 && msaa_edram_actual_ == 0 && !grid_x_log2 && !half &&
                              !guest_width && !guest_height;
    if (!Create(image, image_width, image_height, usage_depth, host_format,
                multisampled ? VkSampleCountFlagBits(msaa_samples_) : VK_SAMPLE_COUNT_1_BIT)) {
      Reject(12, "could not create a depth target");
      return nullptr;
    }
    image.guest_width = guest_width;
    image.guest_height = guest_height;
    image.accepts_target_of_copy = with_transfer_dst;
    REXLOG_INFO("[native] targets: depth target base {:03X}, format {}, {}x{}", base,
                format, pitch, height);
    if (grid_x_log2)
      REXLOG_INFO("[native] depth raster X samples retained: base {:03X}/{} guest {}x{} -> host {}x{}; "
                  "Y still collapsed, not native MSAA", base, format, pitch, height, image_width, image_height);
    return &depths_.emplace(key, image).first->second;
  }

  // Predicated tiling may resolve one logical texture as multiple rectangles. For the later rectangles,
  // D3D9 advances RB_COPY_DEST_BASE by the tiled byte offset and sends local (0,0)-based vertices again.
  // Such a base is not a new texture: locate the already-created parent image and turn the physical base
  // delta back into a destination x/y. This is format/geometry based and applies equally to color and depth.
  Resolved* SearchResolvedPartial(uint32_t base_segment, uint32_t segment_width,
                                  uint32_t segment_height, uint32_t guest_format,
                                  bool swap_rb, VkFormat host_format, uint32_t log2_texel,
                                  uint32_t& base_resolved, uint32_t& dx, uint32_t& dy) {
    if (resolved_.count(base_segment)) {
      return nullptr;  // An exact texture at this address takes precedence.
    }
    for (auto& [base_candidate, candidate] : resolved_) {
      if (base_candidate & 0x80000000u || candidate.guest_format != guest_format ||
          candidate.swap_rb != swap_rb || candidate.image.format != host_format ||
          candidate.image.width != segment_width) {
        continue;
      }
      for (uint32_t y = 0; y < candidate.image.height; y += 32) {
        for (uint32_t x = 0; x < candidate.image.width; x += 32) {
          const uint32_t address =
              (base_candidate + uint32_t(TileDisplacement2D(int32_t(x), int32_t(y),
                                                                 segment_width, log2_texel))) &
              0x1FFFFFFF;
          if (address != base_segment || x + dx >= candidate.image.width ||
              y + dy >= candidate.image.height ||
              y + dy + segment_height > candidate.image.height) {
            continue;
          }
          base_resolved = base_candidate;
          dx += x;
          dy += y;
          const uint64_t key = (uint64_t(base_candidate) << 32) | base_segment;
          if (partial_recorded_.insert(key).second) {
            REXLOG_INFO("[native] targets: partial resolve: advanced base {:08X} belongs to {:08X}; "
                        "local destination {},{} inside {}x{} (segment {}x{})",
                        base_segment, base_candidate, dx, dy, candidate.image.width,
                        candidate.image.height, segment_width, segment_height);
          }
          return &candidate;
        }
      }
    }
    return nullptr;
  }

  void ResolvedReportPool(bool detail = false) {
    if (REXCVAR_GET(sa2_native_edram_alias_mode) != 4) return;
    const auto now = std::chrono::steady_clock::now();
    if (!detail && now - resolved_pool_report_ < std::chrono::seconds(10)) return;
    resolved_pool_report_ = now;
    REXLOG_INFO("[native] resolved allocation pool enabled={}: allocations={}, reuse={}, "
                "retirements={}, evictions={}, eviction waits={}, ineligible={}, busy={}, "
                "dormant={} / {} bytes", REXCVAR_GET(sa2_native_reuse_alloc_resolved),
                resolved_pool_allocations_, resolved_pool_hits_, resolved_pool_retirements_,
                resolved_pool_evictions_, resolved_pool_waits_, resolved_pool_ineligible_,
                resolved_pool_busy_, resolved_sleeping_.size(), resolved_sleeping_bytes_);
  }

  bool CanSleepResolved(const Image& image) {
    // Swapped attachments and deferred jobs carry logical identity beyond a resolve. Leave their
    // original fenced replacement path unchanged, including all legacy modes.
    // banjo-nx: which condition refuses (diagnostic, first occurrences only).
    static uint32_t why_logged = 0;
    auto refuse = [&](const char* why) {
      if (why_logged < 24) {
        ++why_logged;
        REXLOG_INFO("[native] banjo: resolved image {}x{} fmt {} not poolable: {}", image.width, image.height,
                    uint32_t(image.format), why);
      }
      return false;
    };
    if (!resolved_pool_startup_eligible_) return refuse("startup");
    if (resolved_pool_device_error_) return refuse("device error");
    if (!me::native::IsSingleSample(uint32_t(image.sample_count))) return refuse("msaa");
    if (!REXCVAR_GET(sa2_native_reuse_alloc_resolved)) return refuse("cvar");
    if (ResolverNoCopy()) return refuse("resolve without copy");
    if (IsDepthFormat(image.format)) return refuse("depth format");
    if (!image.prepared) return refuse("not prepared");
    if (image.invalid_content) return refuse("invalid content");
    if (!pending_fronts_.empty()) return refuse("pending fronts");
    for (const auto& owner : owners_tiles_edram4_)
      if (owner.image == &image) return refuse("edram4 owner");
    for (const auto& [key, target] : targets_)
      if (target.invalid_content) return refuse("a color target has invalid content");
    for (const auto& [key, target] : depths_)
      if (target.invalid_content) return refuse("a depth target has invalid content");
    return true;
  }

  bool ResolvedSleepingFinished(const ResolvedSleeping& sleeping) {
    // Both native WORK and OUTPUT are submitted to family_, queue 0. A *future* WORK fence
    // therefore covers output already queued when this image was logically retired. An older
    // work fence by itself would not prove this. Never reuse an unsubmitted retirement marker.
    if (resolved_pool_device_error_ || sends_ < sleeping.order_retirement) return false;
    for (const auto& slot : slots_) {
      if (!slot.pending || slot.order <= order_completed_work_) continue;
      deferred::WaitTicket(slot.ticket);  // usf4_deferred_async_submit: the fence must have been submitted
      const VkResult status = dfn_.vkGetFenceStatus(device_, slot.fence);
      if (status == VK_SUCCESS) order_completed_work_ =
          me::native::ObserveResolvedAllocationCompletion(order_completed_work_, slot.order, true);
      else if (status != VK_NOT_READY) {
        resolved_pool_device_error_ = true;
        REXLOG_ERROR("[native] resolved allocation pool disabled: fence status {}", int(status));
        return false;
      }
    }
    return me::native::ResolvedAllocationReady(sleeping.order_retirement, order_completed_work_);
  }

  void EvictResolvedSleeping(size_t index) {
    auto& sleeping = resolved_sleeping_[index];
    if (draws_) draws_->ForgetImage(sleeping.image.image);
    Destroy(sleeping.image);
    resolved_sleeping_bytes_ -= sleeping.bytes;
    resolved_sleeping_.erase(resolved_sleeping_.begin() + index);
    ++resolved_pool_evictions_;
  }

  bool SleepResolved(uint32_t base, Image& image) {
    if (!CanSleepResolved(image)) return false;
    VkMemoryRequirements requirements{};
    dfn_.vkGetImageMemoryRequirements(device_, image.image, &requirements);
    constexpr VkDeviceSize kMaxBytes = 128ull * 1024 * 1024;
    if (requirements.size > kMaxBytes) return false;
    while (!me::native::ResolvedAllocationFits(resolved_sleeping_.size(),
                                              resolved_sleeping_bytes_, requirements.size)) {
      size_t safe = resolved_sleeping_.size();
      for (size_t i = 0; i < resolved_sleeping_.size(); ++i)
        if (ResolvedSleepingFinished(resolved_sleeping_[i])) { safe = i; break; }
      if (safe == resolved_sleeping_.size()) {
        // Rare bounded-cache eviction only: cover both work and presenter consumers before
        // ForgetImage mutates UPDATE_AFTER_BIND descriptors or destroys their views.
        ++resolved_pool_waits_;
        ++waits_gpu_reason_[3]; WaitGpu();
        if (resolved_pool_device_error_) return false;
        if (!ResolvedSleepingFinished(resolved_sleeping_[0])) {
          resolved_pool_device_error_ = true;
          REXLOG_ERROR("[native] resolved allocation eviction lacks successful retirement fence");
          return false;
        }
        safe = 0;
      }
      EvictResolvedSleeping(safe);
    }
    if (draws_) draws_->FinishPass();
    if (!Record() || !CanSleepResolved(image)) return false;
    resolved_sleeping_.push_back({base, image, requirements.size, sends_ + 1});
    resolved_sleeping_bytes_ += requirements.size;
    image = {};
    ++resolved_pool_retirements_;
    return true;
  }

  bool WakeResolved(uint32_t base, uint32_t width, uint32_t height, VkFormat format,
                        Image& image) {
    if (resolved_pool_device_error_ || !REXCVAR_GET(sa2_native_reuse_alloc_resolved) ||
        ResolverNoCopy() ||
        !pending_fronts_.empty()) return false;
    for (size_t i = 0; i < resolved_sleeping_.size(); ++i) {
      const auto& sleeping = resolved_sleeping_[i];
      if (sleeping.base != base || sleeping.image.width != width ||
          sleeping.image.height != height || sleeping.image.format != format ||
          sleeping.image.scale_q != scale_q_) continue;
      if (!CanSleepResolved(sleeping.image)) continue;
      if (!ResolvedSleepingFinished(sleeping)) { ++resolved_pool_busy_; continue; }
      if (draws_) draws_->FinishPass();
      if (!Record()) return false;
      // Do not use Prepare / UPLOAD here: upload executes before older recorded WORK reads.
      // Keep GENERAL and order a whole-image discard inside WORK, exactly like a new zero image.
      const Image backing = resolved_sleeping_[i].image;
      image = {};
      image.image = backing.image;
      image.memory = backing.memory;
      image.view = backing.view;
      image.width = backing.width;
      image.height = backing.height;
      image.host_width = backing.host_width;  // sa2-nx render resolution
      image.host_height = backing.host_height;
      image.scale_q = backing.scale_q;
      image.vk_usage = backing.vk_usage;
      image.format = backing.format;
      image.prepared = true;
      resolved_sleeping_bytes_ -= resolved_sleeping_[i].bytes;
      resolved_sleeping_.erase(resolved_sleeping_.begin() + i);
      VkImageMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.oldLayout = barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
      barrier.srcQueueFamilyIndex = barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
      barrier.image = image.image;
      barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
      const VkCommandBuffer cmd = slots_[slot_].work;
      dfn_.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      const VkClearColorValue zero{};
      dfn_.vkCmdClearColorImage(cmd, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero, 1,
                               &barrier.subresourceRange);
      barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      dfn_.vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
      image.swap_rb = false;
      image.depth_float24_half = false;
      image.resolved_depth_guestspace = false;
      image.resolved_guest_format = UINT32_MAX;
      ++resolved_pool_hits_;
      ResolvedReportPool(resolved_pool_hits_ <= 8);
      return true;
    }
    return false;
  }

  Resolved* GetResolved(uint32_t base, uint32_t width, uint32_t height, uint32_t format,
                            bool swap_rb, VkFormat host_format = kColorFormat) {
    if (!width || !height) {
      Reject(9, "copy to a texture of size 0");
      return nullptr;
    }
    uint64_t reads = 0;
    auto it = resolved_.find(base);
    if (it != resolved_.end()) {
      if (it->second.image.width == width && it->second.image.height == height &&
          it->second.image.format == host_format &&
          me::native::IsSingleSample(uint32_t(it->second.image.sample_count))) {
        if (it->second.guest_format != format || it->second.swap_rb != swap_rb) {
          BeforeResolvedWrite(base);  // sa2_gpu_resolve_alias: before the revision moves (a resolve follows)
          it->second.revision = ++revision_resolved_;
          if (draws_) draws_->InvalidateTextures();  // changes the swizzle it is sampled with
        }
        it->second.guest_format = format;
        it->second.swap_rb = swap_rb;
        it->second.image.swap_rb = swap_rb;
        it->second.image.resolved_guest_format = format;
        ResolvedReportPool();
        return &it->second;
      }
      // Another size or format at the same address: the old image may still be in use.
      // Diagnostic (cutscene dark frame): the new image starts empty, the 360 would keep the bytes.
      if (changes_resolved_recorded_format_ < 400) {
        ++changes_resolved_recorded_format_;
        REXLOG_INFO("[native] resolved texture changes at {:08X}: {}x{} guest format {} host {} -> {}x{} guest {} host {} "
                    "(swap {})",
                    base, it->second.image.width, it->second.image.height, it->second.guest_format,
                    uint32_t(it->second.image.format), width, height, format, uint32_t(host_format), presented_);
      }
      // sa2_gpu_resolve_alias: the image goes (parked with its bytes, or destroyed): the aliases reading from it, and
      // its own alias, get their bytes while it is still here. `it` stays valid (no entry is added or removed).
      AliasesBeforeImageGoes(base);
      reads = it->second.reads;
      // banjo-nx: park the old shape and reuse a parked one of the requested shape (banjo_native_park_resolved).
      // Not while an EDRAM tile owner points at the image (the owner keeps an Image*), nor with resolves
      // swapped into targets (ResolverNoCopy), and at most 64 parked images.
      bool owned = false;
      for (const auto& owner : owners_tiles_edram4_)
        if (owner.image == &it->second.image) { owned = true; break; }
      if (REXCVAR_GET(banjo_native_park_resolved) && !ResolverNoCopy() && !owned && parked_resolved_.size() < 64) {
        const uint64_t old_key = ParkedKey(base, it->second.image.width, it->second.image.height,
                                           uint32_t(it->second.image.format));
        parked_resolved_[old_key] = std::move(it->second);
        resolved_.erase(it);
        ++parked_resolved_stores_;
        auto p = parked_resolved_.find(ParkedKey(base, width, height, uint32_t(host_format)));
        if (p != parked_resolved_.end()) {
          Resolved back = std::move(p->second);
          parked_resolved_.erase(p);
          back.revision = ++revision_resolved_;
          back.guest_format = format;
          back.swap_rb = swap_rb;
          back.image.swap_rb = swap_rb;
          back.image.resolved_guest_format = format;
          back.reads = reads;
          ++parked_resolved_reuses_;
          if (draws_) draws_->InvalidateTextures();  // that address is now sampled from this image
          return &resolved_.emplace(base, std::move(back)).first->second;
        }
        it = resolved_.end();
      }
      if (it != resolved_.end() && !SleepResolved(base, it->second.image)) {
        ++resolved_pool_ineligible_;
        if (!SendWork(true)) {
          resolved_pool_device_error_ = true;
          return nullptr;
        }
        ++waits_gpu_reason_[4]; WaitGpu();
        if (resolved_pool_device_error_) return nullptr;
        if (draws_) draws_->ForgetImage(it->second.image.image);
        Destroy(it->second.image);
      }
      if (it != resolved_.end()) resolved_.erase(it);
    }
    Resolved resolved;
    resolved.reads = reads;
    resolved.revision = ++revision_resolved_;
    // The attachment usage (color or depth) is needed for sa2_native_resolver_no_copy: this image can
    // end up being the render target it is swapped with.
    const VkImageUsageFlags usage_target = host_format == VK_FORMAT_R32_SFLOAT ? 0u :
        IsDepthFormat(host_format) ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                         : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    const bool reused = WakeResolved(base, width, height, host_format, resolved.image);
    if (resolved_pool_device_error_) return nullptr;
    if (!reused && !Create(resolved.image, width, height,
               VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                   VK_IMAGE_USAGE_TRANSFER_DST_BIT | usage_target |
                   ((host_format == VK_FORMAT_R16G16B16A16_SFLOAT || host_format == VK_FORMAT_R32_SFLOAT)
                        ? VkImageUsageFlags(VK_IMAGE_USAGE_STORAGE_BIT) : VkImageUsageFlags(0)),
               host_format)) {
      Reject(10, "could not create a resolved texture");
      return nullptr;
    }
    resolved.guest_format = format;
    resolved.swap_rb = swap_rb;
    resolved.image.swap_rb = swap_rb;
    resolved.image.resolved_guest_format = format;
    if (!reused) {
      ++resolved_pool_allocations_;
      REXLOG_INFO("[native] targets: resolved texture at {:08X}, {}x{}, format {}", base, width, height,
                  format);
    }
    if (draws_) {
      draws_->InvalidateTextures();  // that address is now sampled from the resolved texture
    }
    ResolvedReportPool();
    return &resolved_.emplace(base, resolved).first->second;
  }

  bool ResolverNoCopy() { return REXCVAR_GET(sa2_native_resolver_no_copy); }

  static bool IsDepthFormat(VkFormat format) {
    return format == VK_FORMAT_D24_UNORM_S8_UINT || format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D16_UNORM || format == VK_FORMAT_D32_SFLOAT ||
           format == VK_FORMAT_X8_D24_UNORM_PACK32;
  }

  // Resolve without copying. The whole render target becomes the resolved texture and that texture's
  // old image stays as the render target. What the game sees is the same; what is saved is moving the
  // pixels (the two 1600x1600 shadow map copies are 41 MB per frame). false if it is not possible.
  bool SwapWithResolved(Image& target, Resolved& resolved, uint32_t base) {
    // Resolved textures are sampled with an ordinary sampler2D. Do not swap a
    // future multisample allocation into this 1x identity (or vice versa).
    if (!me::native::IsSingleSample(uint32_t(target.sample_count)) ||
        !me::native::IsSingleSample(uint32_t(resolved.image.sample_count))) return false;
    // Old depth swap restoration copies only the depth aspect. Physical aliasing also needs the
    // original stencil byte; keep both aspects in their original owner in diagnostic mode 4.
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4 && target.edram_depth) return false;
    if (target.width != resolved.image.width || target.height != resolved.image.height ||
        target.format != resolved.image.format || target.scale_q != resolved.image.scale_q) {
      return false;
    }
    // ZCULL safety net. If the render target was created without TRANSFER_DST (so it can have a ZCULL
    // plane), swapping it would put it in the resolved texture's place, which does receive copies: an
    // illegal write. And the other way round, the resolved texture (with TRANSFER_DST and no plane) would
    // become the depth target and ZCULL would switch off on alternate frames. Fall back to the usual copy
    // path instead.
    if (!target.accepts_target_of_copy) {
      return false;
    }
    // Careful: ForgetImage must not be called here. It frees the descriptor slot and reuses it in the
    // same frame, so draws already recorded with it end up reading another image (flicker). Bumping the
    // generation is enough: later draws resolve their view again, and existing views stay valid because
    // they go with their image.
    // sa2_native_lazy_front. Same for front buffers: deferred copies whose source is this target
    // are recorded first, and the texture receives the whole target (its deferred copy is unnecessary).
    if (!pending_fronts_.empty()) {
      BeforeWriteColor(target);
      ResolverPreviousFront(base, true);
    }
    BeforeResolvedWrite(base);  // sa2_gpu_resolve_alias
    std::swap(target.image, resolved.image.image);
    std::swap(target.memory, resolved.image.memory);
    std::swap(target.view, resolved.image.view);
    std::swap(target.prepared, resolved.image.prepared);
    resolved.image.swap_rb = resolved.swap_rb;  // belongs to the content, not the image
    resolved.image.invalid_content = false;
    target.invalid_content = true;  // its content is now in the resolved texture
    target.resolved_base = base;
    ++swaps_;
    ResolvedWritten(base);   // if it was lent, it has valid content again
    ForgetClear(target);  // and the target keeps another image: its clear no longer holds
    if (draws_) {
      // Only what points to these two images: invalidating the whole cache here costs more than the copy
      // it saves (+0.18 ms of scene per frame when measured).
      draws_->InvalidateImages(target.image, resolved.image.image);
    }
    return true;
  }

  // If the render target lost its content in an image swap and the game is about to draw on top, the
  // content is brought back from the resolved texture.
  //
  // It happens once for every swap without a clear. Restores are counted and trimmed to the useful area.
  //
  // for_resolver = requested by a resolve (sa2_native_resolver_valid_content), not by a pass that
  // will draw on top. Then it is only copied.
  bool RestoreContent(Image& target, bool for_resolver = false) {
    if (!target.invalid_content) {
      return true;
    }
    ForgetSynchronizedEDRAM4(target);  // its content is replaced
    const bool mode4 = REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    // Preserve the legacy behavior outside the experimental strict path.
    if (!mode4) target.invalid_content = false;
    // sa2_native_lazy_front. Same for front buffers: if the texture about to be read has a
    // deferred copy, it is recorded first; and so are deferred copies whose source is this render target.
    if (!pending_fronts_.empty()) {
      if (pending_fronts_.count(target.resolved_base)) {
        RecordCopyFront(target.resolved_base);
        ++copied_front_read_;
      }
      BeforeWriteColor(target);
    }
    // sa2_gpu_resolve_alias: that texture is read.
    if (!aliases_.empty() && aliases_.count(target.resolved_base)) MaterializeAlias(target.resolved_base);
    auto it = resolved_.find(target.resolved_base);
    if (mode4 && target.depth_float24_half)
      return FailureEDRAM4("half-range owner cannot restore stencil from guestspace R32 resolve", target);
    if (it == resolved_.end() || it->second.image.width != target.width ||
        it->second.image.height != target.height || it->second.image.format != target.format ||
        it->second.image.scale_q != target.scale_q || !copy_image_ || !Record()) {
      if (for_resolver) {
        ++resolver_no_source_;  // unknown where to bring it from; resolve whatever is there
      }
      if (mode4) FailureEDRAM4("swapped owner restoration has no valid source", target);
      return false;
    }
    ++it->second.reads;  // usf4-nx: a restore reads the texture too (copy report: not "nobody reads it")
    if (for_resolver) {
      // sa2_native_resolver_valid_content. Only the copy, with its barriers.
      if (!CopyOfLapForResolver(target, it->second.image)) return false;
      target.invalid_content = false;
      return true;
    }
    Prepare(it->second.image);
    Prepare(target);
    if (mode4 && (!target.prepared || !it->second.image.prepared ||
                  !target.accepts_target_of_copy)) {
      FailureEDRAM4("swapped owner restore images cannot receive the copy", target);
      return false;
    }
    const VkImageAspectFlags aspect =
        IsDepthFormat(target.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    // Render targets are created with height = max(720, pitch): the scene measures 1280x1280 to draw
    // 1280x720 and the blur targets 320x720 to draw 320x180. What lies below the useful area is never
    // drawn or resolved, so copying it wastes bandwidth. The useful area is the largest y1 the game has
    // asked to resolve from this target; until there is data, the whole target is copied.
    uint32_t height = target.height;
    if (!mode4 && REXCVAR_GET(sa2_native_restore_area_util)) {
      const auto e = target_state_.find(&target);
      if (e != target_state_.end() && e->second.used_height &&
          e->second.used_height < target.height) {
        height = e->second.used_height;
        ++clipped_restores_;
        pixels_restore_saved_ += uint64_t(target.width) * (target.height - height);
      }
    }
    VkImageCopy copy{};
    copy.srcSubresource = {aspect, 0, 0, 1};
    copy.dstSubresource = {aspect, 0, 0, 1};
    copy.extent = {target.width, height, 1};
    MarkGpu(kGpuCopies);
    std::array<VkImageCopy, 2> restore_regions{copy, copy};
    uint32_t restore_count = 1;
    if (mode4 && aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      // A whole D32S8 image swap moved BOTH aspects into the resolved backing.
      restore_regions[1].srcSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      restore_regions[1].dstSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      restore_count = 2;
    }
    if (mode4) {
      if (draws_) draws_->FinishPass();
      VkMemoryBarrier before{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      before.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
      before.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
          VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &before, 0, nullptr, 0, nullptr);
    }
    HostCopies(it->second.image, target, restore_regions.data(), restore_count);  // sa2-nx render resolution
    copy_image_(commands_work_, it->second.image.image, VK_IMAGE_LAYOUT_GENERAL,
                   target.image, VK_IMAGE_LAYOUT_GENERAL, restore_count, restore_regions.data());
    if (mode4) BarrierAfterCopyResolve();
    target.invalid_content = false;
    ++restores_;
    restored_pixels_ += uint64_t(target.width) * height;
    ForgetClear(target);
    return true;
  }

  // Every ordinary resolve uses GENERAL for both images, so there is no layout transition to provide a
  // memory dependency. Queue order alone is not a cache flush: publish attachment writes before the copy,
  // then publish the resolved texture before a shader samples it (or another transfer touches it).
  void BarrierBeforeCopyResolve() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_,
                              VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  void BarrierAfterCopyResolve() {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                            VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                  VK_PIPELINE_STAGE_TRANSFER_BIT |
                                  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                                  VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);
  }

  /*
   * sa2_native_resolver_valid_content. The render target gets its content back from the resolved
   * texture where the image swap left it, just before a resolve reads it.
   *
   * The copy goes between two barriers, and here they are needed: the source is the image that was just
   * drawn as the depth target (the previous resolve moved it into the texture by swapping), and what is
   * copied will be sampled as soon as the current resolve swaps it. NVK only waits for the GPU and
   * flushes the texture cache at a barrier (the normal restore still goes without them). Each one costs a
   * pipeline drain. The whole target is copied: that is what the EDRAM held and what the resolve reads.
   */
  bool CopyOfLapForResolver(Image& target, Image& source) {
    Prepare(source);
    Prepare(target);
    const bool mode4 = REXCVAR_GET(sa2_native_edram_alias_mode) == 4;
    if (mode4 && (!target.prepared || !source.prepared ||
                  !target.accepts_target_of_copy))
      return FailureEDRAM4("resolve owner restore images cannot receive the copy", target, &source);
    const VkImageAspectFlags aspect =
        IsDepthFormat(target.format) ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    constexpr VkPipelineStageFlags kWrites =
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT |
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    if (mode4) {
      if (draws_) draws_->FinishPass();
      barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    }
    dfn_.vkCmdPipelineBarrier(commands_work_, mode4 ? VK_PIPELINE_STAGE_ALL_COMMANDS_BIT : kWrites,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0,
                              nullptr, 0, nullptr);
    VkImageCopy copy{};
    copy.srcSubresource = {aspect, 0, 0, 1};
    copy.dstSubresource = {aspect, 0, 0, 1};
    copy.extent = {target.width, target.height, 1};
    MarkGpu(kGpuCopies);
    std::array<VkImageCopy, 2> restore_regions{copy, copy};
    uint32_t restore_count = 1;
    if (mode4 && aspect == VK_IMAGE_ASPECT_DEPTH_BIT) {
      restore_regions[1].srcSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      restore_regions[1].dstSubresource.aspectMask = VK_IMAGE_ASPECT_STENCIL_BIT;
      restore_count = 2;
    }
    HostCopies(source, target, restore_regions.data(), restore_count);  // sa2-nx render resolution
    copy_image_(commands_work_, source.image, VK_IMAGE_LAYOUT_GENERAL, target.image, VK_IMAGE_LAYOUT_GENERAL,
                   restore_count, restore_regions.data());
    // And the copied data made visible to what follows: the current resolve (copy or swap), the draws that
    // sample the texture that receives it, and the pass that draws to the render target again.
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands_work_, VK_PIPELINE_STAGE_TRANSFER_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                                  VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                              0, 1, &barrier, 0, nullptr, 0, nullptr);
    ++restores_for_resolver_;
    restored_pixels_ += uint64_t(target.width) * target.height;
    ForgetClear(target);
    return true;
  }

  // The resolved texture has just received new content. `pixels` = the pixels actually copied (0 if the resolve swapped
  // images), so that the per-target inventory shows where the traffic goes.
  void ResolvedWritten(uint32_t base, uint64_t pixels = 0) {
    if (auto it = resolved_.find(base); it != resolved_.end()) {
      it->second.revision = ++revision_resolved_;
      // The direct image updates in place, but derived ROIs must go through PrepareTexture again,
      // including consecutive draws within one frame. Partial writes invalidate the complete ROI.
      for (const auto& crop : resolved_clips_) {
        if (crop.address == base) {
          if (draws_) {
            if (REXCVAR_GET(usf4_resolved_crops_invalidate_narrow)) draws_->InvalidateTexturesAt(base);
            else draws_->InvalidateTextures();
          }
          break;
        }
      }
    }
    if (pixels) {
      const auto c = copies_per_target_.find(base);
      if (c != copies_per_target_.end()) {
        c->second.pixels += pixels;
      }
    }
  }

  // The render target's content is no longer that of its last clear (something was copied over it or
  // it changed image). The game's next clear cannot be skipped.
  //
  // It also answers the other question about clears: if the target's content changes without anything
  // drawn since it was cleared, that clear was wiped out by a swap or a restore and served no purpose.
  // It is the equivalent of a loadOp = CLEAR that gets thrown away: these are DONT_CARE candidates.
  void ForgetClear(const Image& target) {
    const auto e = target_state_.find(&target);
    if (e == target_state_.end()) {
      return;
    }
    if (e->second.clean_clear &&
        e->second.draws_on_clear == (draws_ ? draws_->Drawn() : 0)) {
      ++cleared_useless_;
      cleared_useless_pixels_ += uint64_t(target.width) * target.height;
    }
    e->second.clean_clear = false;
  }

  // The largest rectangle the game resolves from this render target: the area it actually uses.
  void NoteAreaUtil(const Image& target, int32_t y1) {
    if (y1 > 0) {
      auto& e = target_state_[&target];
      e.used_height = std::max(e.used_height, std::min(uint32_t(y1), target.height));
    }
  }

  // A clear that does not change a single bit. It is skipped if the render target is already cleared
  // to that same value and nothing has been drawn, anywhere, since then. Returns true if it must be
  // skipped.
  bool RedundantClear(const Image& target, uint64_t value) {
    auto& e = target_state_[&target];
    const uint64_t drawn = draws_ ? draws_->Drawn() : 0;
    if (REXCVAR_GET(sa2_native_skip_cleared_repeated) && e.clean_clear &&
        e.clear_value == value && e.draws_on_clear == drawn) {
      cleared_skipped_pixels_ += uint64_t(target.width) * target.height;
      return true;
    }
    e.clean_clear = true;
    e.clear_value = value;
    e.draws_on_clear = drawn;
    return false;
  }

  // width / height are the GUEST size: the VkImage is that size scaled by scale_q_ / 4 (sa2-nx render resolution,
  // me_resolution_scale.h); image.width / height keep the guest size, image.host_width / host_height the extent.
  bool Create(Image& image, uint32_t width, uint32_t height, VkImageUsageFlags usage,
             VkFormat format = kColorFormat,
             VkSampleCountFlagBits sample_count = VK_SAMPLE_COUNT_1_BIT) {
    return CreateScaled(image, width, height, usage, format, sample_count, scale_q_);
  }

  bool CreateScaled(Image& image, uint32_t width, uint32_t height, VkImageUsageFlags usage, VkFormat format,
                    VkSampleCountFlagBits sample_count, uint32_t q) {
    ForgetSynchronizedEDRAM4(image);  // new contents
    // Deliberate interim allocation gate: no real 2x/4x image may reach the
    // unconverted utility shaders, pipeline caches or guest resolve paths.
    // Lifting it requires format/device capability AND all those contracts,
    // not merely changing VkImageCreateInfo.samples.
    // sa2_msaa: the host multisampling of the game's single-sample targets is the one exception.
    if (!me::native::IsSingleSample(uint32_t(sample_count)) &&
        !(msaa_samples_ > 1 && uint32_t(sample_count) == msaa_samples_))
      return Reject(9, "real multisample allocation is not enabled in this backend");
    VkImageCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    info.flags = format == VK_FORMAT_R16G16B16A16_SFLOAT ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0;
    info.imageType = VK_IMAGE_TYPE_2D;
    info.format = format;
    info.extent = {ScaleSize(width, q), ScaleSize(height, q), 1};
    info.mipLevels = 1;
    info.arrayLayers = 1;
    info.samples = sample_count;
    info.tiling = VK_IMAGE_TILING_OPTIMAL;
    info.usage = usage;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (!rex::ui::vulkan::util::CreateDedicatedAllocationImage(
            vulkan_device_, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, image.image,
            image.memory)) {
      // The GPU can run out of memory here too. Render targets are few and large, so one that does not fit
      // shows up at once: the texture cache is asked to release memory and the allocation is retried once.
      if (!draws_ || !draws_->DropTexturesPerMissingOfMemory() ||
          !rex::ui::vulkan::util::CreateDedicatedAllocationImage(
              vulkan_device_, info, rex::ui::vulkan::util::MemoryPurpose::kDeviceLocal, image.image,
              image.memory)) {
        REXLOG_ERROR("[native] targets: out of GPU memory for a {}x{} target (format {}) and dropping the cache "
                     "was not enough",
                     info.extent.width, info.extent.height, uint32_t(format));
        return false;
      }
    }
    VkImageViewCreateInfo info_view{};
    info_view.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    info_view.image = image.image;
    info_view.viewType = VK_IMAGE_VIEW_TYPE_2D;
    info_view.format = format;
    info_view.subresourceRange = IsDepthFormat(format) ? kRangeDepth : kRangeColor;
    if (dfn_.vkCreateImageView(device_, &info_view, nullptr, &image.view) != VK_SUCCESS) {
      Destroy(image);
      return false;
    }
    image.width = width;
    image.height = height;
    image.host_width = info.extent.width;
    image.host_height = info.extent.height;
    image.scale_q = uint8_t(q);
    image.vk_usage = usage;
    image.format = format;
    image.sample_count = sample_count;
    image.prepared = false;
    return true;
  }

  void Destroy(Image& image) {
    ForgetSynchronizedEDRAM4(image);
    // sa2_native_lazy_front. A deferred copy cannot keep a destroyed image. If the texture is
    // what gets destroyed (GetResolved recreates it with another size or format), its content is lost
    // just as before: the copy is simply dropped. If it is the source (only at shutdown: render targets
    // are not destroyed, and whatever leaves a target through a swap is recorded first), the texture lacks
    // that copy: it becomes stale and, if something requests it before another full resolve, the guard
    // trips.
    if (!pending_fronts_.empty() && image.image != VK_NULL_HANDLE) {
      for (auto it = pending_fronts_.begin(); it != pending_fronts_.end();) {
        if (it->second.texture_vk == image.image) {
          ReleaseRetained(it->second);
          it = pending_fronts_.erase(it);
        } else if (it->second.source_vk == image.image) {
          expired_fronts_.insert(it->first);
          ReleaseRetained(it->second);
          it = pending_fronts_.erase(it);
        } else {
          ++it;
        }
      }
    }
    // The DrawsVulkan framebuffers that use this view are destroyed before it: Vulkan may give the same
    // handle to a new view and FramebufferFor would return a stale one (see
    // sa2_native_framebuffers_forget_views). At shutdown, draws_ is already gone and so are its
    // framebuffers.
    if (draws_ && image.view != VK_NULL_HANDLE) {
      draws_->ForgetView(image.view);
    }
    if (auto it = views_raw64_edram_.find(image.image); it != views_raw64_edram_.end()) {
      dfn_.vkDestroyImageView(device_, it->second, nullptr);
      views_raw64_edram_.erase(it);
    }
    if (auto it = views_depth_edram_.find(image.image); it != views_depth_edram_.end()) {
      if (it->second.depth) dfn_.vkDestroyImageView(device_, it->second.depth, nullptr);
      if (it->second.stencil) dfn_.vkDestroyImageView(device_, it->second.stencil, nullptr);
      views_depth_edram_.erase(it);
    }
    for (uint32_t variant = 0; variant < 3; ++variant)  // 2: sa2_gpu_resolve_draw
      if (auto it = framebuffers_conv_color_frag_.find({image.image, variant});
          it != framebuffers_conv_color_frag_.end()) {
        if (it->second) dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);
        framebuffers_conv_color_frag_.erase(it);
      }
    if (auto it = views_raw64_rt_edram_.find(image.image); it != views_raw64_rt_edram_.end()) {
      dfn_.vkDestroyImageView(device_, it->second, nullptr);
      views_raw64_rt_edram_.erase(it);
    }
    if (auto it = framebuffers_import_depth_edram_.find(image.image);
        it != framebuffers_import_depth_edram_.end()) {
      dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);
      framebuffers_import_depth_edram_.erase(it);
    }
    if (auto it = framebuffers_depth_resolve_ms_.find(image.image); it != framebuffers_depth_resolve_ms_.end()) {
      if (it->second) dfn_.vkDestroyFramebuffer(device_, it->second, nullptr);  // sa2_msaa
      framebuffers_depth_resolve_ms_.erase(it);
    }
    for (auto& owner : owners_tiles_edram4_) if (owner.image == &image) owner = {};
    ++edram4_epoch_;
    if (image.view != VK_NULL_HANDLE) dfn_.vkDestroyImageView(device_, image.view, nullptr);
    if (image.image != VK_NULL_HANDLE) dfn_.vkDestroyImage(device_, image.image, nullptr);
    if (image.memory != VK_NULL_HANDLE) dfn_.vkFreeMemory(device_, image.memory, nullptr);
    image = Image{};
  }

  // First time: to GENERAL (valid for copy, clear and sampling) and cleared to zero.
  void Prepare(Image& image) {
    if (image.prepared) {
      return;
    }
    // In the upload command buffer, which runs before the work one (copies of resolved textures are recorded
    // there and, on their first frame, read them unprepared).
    const VkCommandBuffer commands = CommandsUpload();
    if (commands == VK_NULL_HANDLE) {
      return;
    }
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    const bool depth = image.format == VK_FORMAT_D24_UNORM_S8_UINT ||
                             image.format == VK_FORMAT_D32_SFLOAT_S8_UINT;
    barrier.image = image.image;
    barrier.subresourceRange = depth ? kRangeDepth : kRangeColor;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dfn_.vkCmdPipelineBarrier(commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                              &barrier);
    if (depth) {
      // ZCULL: images created without TRANSFER_DST (the ones eligible for a ZCULL plane) cannot be
      // cleared with vkCmdClearDepthStencilImage. They are cleared by opening a pass with
      // loadOp = CLEAR. The barrier above (UNDEFINED -> GENERAL) stays as it is: it is what makes the
      // driver zero the ZCULL plane, and without it the hardware kills the context.
      if (!image.accepts_target_of_copy && draws_) {
        if (!draws_->ClearDepthInPass(commands, image, 1.0f, 0) &&
            ++cleared_in_failed_pass_ <= 8) {
          REXLOG_WARN("[native] targets: could NOT prepare the {}x{} depth by opening a clear "
                      "pass (failure {})",
                      image.width, image.height, cleared_in_failed_pass_);
        }
      } else if (clear_depth_) {
        const VkClearDepthStencilValue far_value{1.0f, 0};  // banjo-nx: 'far' is a Windows macro
        clear_depth_(commands, image.image, VK_IMAGE_LAYOUT_GENERAL, &far_value, 1,
                            &kRangeDepth);
      }
    } else {
      const VkClearColorValue zero{};
      dfn_.vkCmdClearColorImage(commands, image.image, VK_IMAGE_LAYOUT_GENERAL, &zero,
                                1, &kRangeColor);
    }
    image.prepared = true;
    if (draws_) {
      draws_->InvalidateTextures();  // ResolvedTexture only returns prepared ones
    }
  }

  // All three descriptor pools share this budget. Never reset a live pool or
  // retain a slot reference / descriptor allocation across submission rotation.
  std::map<std::string, uint64_t, std::less<>> edram4_operations_;  // ME: mode-4 operations per kind since the last report
  bool EnsureCapacityConversionEDRAM4(const char* operation) {
    if (REXCVAR_GET(sa2_native_edram_alias_mode) != 4)
      return slots_[slot_].conversions_edram < kConversionsEDRAMPerSlot;
    // Close BEFORE obtaining descriptors: closing a shadow pass may itself
    // submit and rotate the slot. Also avoids recursive submit in BeforeSend.
    if (draws_) draws_->FinishPass();
    if (!Record()) return false;
    // sa2-nx: heterogeneous lookup, no std::string built per operation once the kind is known.
    {
      auto it = edram4_operations_.find(std::string_view(operation));
      if (it == edram4_operations_.end()) it = edram4_operations_.emplace(operation, 0).first;
      ++it->second;
    }
    // ME: the conversion's GPU time, by kind (until the next pass marks its own category).
    MarkGpu(std::strcmp(operation, "depth import") == 0   ? kGpuEdramImport
              : std::strcmp(operation, "depth export") == 0 ? kGpuEdramExport
                                                            : kGpuEdramAlias);
    if (slots_[slot_].conversions_edram < kConversionsEDRAMPerSlot) return true;
    const uint32_t slot_previous = slot_;
    const uint32_t used = slots_[slot_].conversions_edram;
    ++edram4_capacity_rollovers_;
    if (edram4_capacity_rollovers_ <= 8)
      REXLOG_WARN("[native] EDRAM mode4 conversion capacity rollover {}: slot {} count {}/{} operation {}",
                  edram4_capacity_rollovers_, slot_previous, used,
                  kConversionsEDRAMPerSlot, operation);
    // Deliberately fenced, including the one-slot configuration. BeginRecording
    // alone is the only place resetting descriptor pools after Complete.
    if (!SendWork(true) || !Record()) {
      ++edram4_capacity_rollover_failures_;
      REXLOG_ERROR("[native] EDRAM mode4 conversion capacity rollover failed: slot {} operation {}",
                   slot_previous, operation);
      return false;
    }
    return slots_[slot_].conversions_edram < kConversionsEDRAMPerSlot;
  }

  bool Record() {
    if (recording_) {
      return true;
    }
    const auto before_record = std::chrono::steady_clock::now();
    const bool recording = BeginRecording();
    ns_record_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - before_record).count());
    return recording;
  }

  bool BeginRecording() {
    // Slots rotate: while one frame is recorded, the previous ones can still be on the GPU. It only waits
    // if this slot's last submission has not finished yet, so the more slots there are, the further ahead
    // the CPU can get before it has to stop.
    slot_ = (slot_ + 1) % used_slots_;
    SlotWork& slot = slots_[slot_];
    Complete(slot);
    if (resolved_pool_device_error_) return false;
    // usf4_deferred_async_submit: Complete waited for this slot's ticket and fence, so the worker has recorded and
    // submitted everything that used these pools, and the GPU has finished it. The resets and the begin go to the
    // driver directly (through the proxy each would drain the worker, which may still be recording the previous
    // frame into another slot's command buffers); the commands that follow are queued after them.
    const auto& fns = deferred::AsyncSubmit() ? deferred::Real() : dfn_;
    fns.vkResetCommandPool(device_, slot.pool_work, 0);
    fns.vkResetCommandPool(device_, slot.pool_upload, 0);
    fns.vkResetDescriptorPool(device_, slot.pool_conversion_edram, 0);
    fns.vkResetDescriptorPool(device_, slot.pool_conversion_depth_edram, 0);
    if (slot.pool_import_depth_edram)
      fns.vkResetDescriptorPool(device_, slot.pool_import_depth_edram, 0);
    if (slot.pool_conv_color_frag) fns.vkResetDescriptorPool(device_, slot.pool_conv_color_frag, 0);
    if (slot.pool_depth_resolve_ms) fns.vkResetDescriptorPool(device_, slot.pool_depth_resolve_ms, 0);
    if (slot.pool_stencil_copy) fns.vkResetDescriptorPool(device_, slot.pool_stencil_copy, 0);
    slot.conversions_edram = 0;
    commands_work_ = slot.work;
    commands_upload_ = slot.upload;
    recording_upload_ = false;
    VkCommandBufferBeginInfo start{};
    start.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    start.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (fns.vkBeginCommandBuffer(commands_work_, &start) != VK_SUCCESS) {
      return false;
    }
    slot.categories.clear();
    slot.labels.clear();
    slot.descriptions.clear();
    slot.draws_mark.clear();
    if (queries_ != VK_NULL_HANDLE && slot.marks_written) {
      dfn_.vkCmdResetQueryPool(commands_work_, queries_, slot_ * kMarksPerSlot,
                               std::min(slot.marks_written, kMarksPerSlot));
      slot.marks_written = 0;
    }
    /*
     * Do not reset pools that this frame will not use.
     *
     * This used to happen always, with any toml: 64 statistics queries + 2048 per-draw statistics
     * queries = 2,144 resets per frame. And in our NVK on Tegra the bulk path is excluded for the layout
     * of these pools (nvk_query_pool.c:351: everything that is not a timestamp falls into
     * ALIGNED_INTERLEAVED), so each reset is a 5-dword SET_REPORT_SEMAPHORE with
     * RELEASE_AFTER_ALL_PRECEEDING_WRITES_COMPLETE and PIPELINE_LOCATION_ALL: 2,144 chained writes that
     * wait on each other, 42.9 KB of command stream, at the start of every frame.
     *
     * And they are almost never used: the per-pass statistics are effectively disabled (see the `> 0`
     * check in BeginStats: with stats_per_draw_s set, the per-pass query is never
     * opened), and the per-draw ones are only used one frame every 20 seconds.
     *
     * The reset comes before the first MarkGpu, so its cost does not show up in any category of the
     * breakdown: it is swallowed by the "gap between work items" (gap between submissions). Estimated at
     * 0.3-1.1 real ms per frame; the measured upper bound (the minimum gap over a whole session) is
     * 2.81 ms.
     *
     * With both off, 32 resets remain instead of 2,144: -98.5 %. The CPU vectors are cleared anyway,
     * which costs nothing.
     */
    slot.stats.clear();  // those never submitted are not read
    slot.stats_draw.clear();
    slot.draw_ps.clear();
    slot.draw_vs.clear();
    if (marks_draw_ != VK_NULL_HANDLE && window_diagnostic_) {
      dfn_.vkCmdResetQueryPool(commands_work_, marks_draw_, slot_ * kStatsDrawPerSlot * 2,
                               kStatsDrawPerSlot * 2);
    }
    if (stats_ != VK_NULL_HANDLE && REXCVAR_GET(sa2_native_stats_pipeline) &&
        REXCVAR_GET(sa2_native_stats_per_draw_s) <= 0) {
      dfn_.vkCmdResetQueryPool(commands_work_, stats_,
                               slot_ * kStatsPerSlot, kStatsPerSlot);
    }
    if (stats_draw_ != VK_NULL_HANDLE && window_diagnostic_) {
      dfn_.vkCmdResetQueryPool(commands_work_, stats_draw_,
                               slot_ * kStatsDrawPerSlot,
                               kStatsDrawPerSlot);
    }
    slot.precise_marks = precise_marks_;
    recording_ = true;
    MarkGpu(kGpuOther);
    ++generation_commands_;
    if (draws_) {
      draws_->UseSlot(slot_);
    }
    return true;
  }

  // Waits for that slot's submission. Earlier submissions that have already finished are collected
  // first, so read-backs are written in frame order.
  /*
   * This was what prevented CPU and GPU from overlapping.
   *
   * It used to wait as well for every earlier slot still pending, so each frame drained the whole GPU
   * before recording continued. That is why the report always said "work overlapped on the GPU 0",
   * why adding a third slot changed nothing (it waited for all of them, however many there were) and
   * why the wait moved elsewhere instead of disappearing.
   *
   * Now it only blocks on the slot about to be reused, the only one whose resources are needed. Earlier
   * ones are collected only if they have already finished (polling without waiting), so their
   * timestamps are read in order when possible; one still running is collected when its turn comes.
   */
  void Complete(SlotWork& slot) {
    for (SlotWork& other : slots_) {
      if (&other != &slot && other.pending && other.order < slot.order &&
          (deferred::WaitTicket(other.ticket), dfn_.vkGetFenceStatus(device_, other.fence) == VK_SUCCESS)) {
        CompleteA(other);
      }
    }
    CompleteA(slot);
  }

  void CompleteA(SlotWork& slot) {
    if (!slot.pending) {
      return;
    }
    deferred::WaitTicket(slot.ticket);  // usf4_deferred_async_submit: the fence must have been submitted
    const auto before_wait = std::chrono::steady_clock::now();
    const VkResult result_wait = dfn_.vkWaitForFences(device_, 1, &slot.fence, VK_TRUE, UINT64_MAX);
    if (result_wait != VK_SUCCESS) {
      resolved_pool_device_error_ = true;
      REXLOG_ERROR("[native] work fence wait failed {}; resolved allocation cache disabled", int(result_wait));
      return;
    }
    order_completed_work_ = me::native::ObserveResolvedAllocationCompletion(
        order_completed_work_, slot.order, true);
    ns_waits_gpu_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now() - before_wait)
                                    .count());
    ++waits_gpu_;
    dfn_.vkResetFences(device_, 1, &slot.fence);
    slot.pending = false;
    const uint32_t marked = uint32_t(slot.categories.size());
    if (queries_ != VK_NULL_HANDLE && marked >= 2) {
      static std::vector<uint64_t> marks_buffer(kMarksPerSlot);  // 64 KB: not on the ring thread's stack
      uint64_t* const marks = marks_buffer.data();
      const uint32_t index = uint32_t(&slot - slots_.data());
      if (read_queries_(device_, queries_, index * kMarksPerSlot, marked,
                          sizeof(uint64_t) * marked, marks, sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT) == VK_SUCCESS &&
          marks[marked - 1] >= marks[0]) {
        // Diagnostic: the whole mark sequence of a few work slots, with labels and durations.
        if (dump_remaining_marks_ > 0 || (REXCVAR_GET(sa2_diag_dump_marks_s) > 0 && !dump_done_marks_ &&
            std::chrono::steady_clock::now() - dump_start_ >
                std::chrono::seconds(REXCVAR_GET(sa2_diag_dump_marks_s)))) {
          if (!dump_done_marks_) { dump_done_marks_ = true; dump_remaining_marks_ = 6; }
          --dump_remaining_marks_;
          std::string list;
          for (uint32_t i = 0; i + 1 < marked; ++i) {
            const double ms = marks[i + 1] >= marks[i] ? double(marks[i + 1] - marks[i]) * period_mark_ns_ / 1e6 : -1;
            if (ms < 0.05) continue;
            const uint32_t e = i < slot.labels.size() ? slot.labels[i] : 0;
            const uint64_t dsc = i < slot.descriptions.size() ? slot.descriptions[i] : 0;
            const uint32_t nd = i < slot.draws_mark.size() ? slot.draws_mark[i] : 0;
            list += fmt::format(" [{}:c{} VS{} PS{} {}x{} f{}/d{} {} draws {:.2f}]", i, slot.categories[i], e >> 16,
                                 e & 0xFFFF, dsc >> 48, (dsc >> 32) & 0xFFFF, (dsc >> 16) & 0xFFFF, dsc & 0xFFFF, nd, ms);
          }
          REXLOG_INFO("[native] GPU marks of one slot ({} marks):{}", marked, list);
        }
        for (uint32_t i = 0; i + 1 < marked; ++i) {
          if (marks[i + 1] >= marks[i] && slot.categories[i] < kGpuCategories) {
            const uint64_t ns = uint64_t(double(marks[i + 1] - marks[i]) * period_mark_ns_);
            gpu_categories_ns_[slot.categories[i]] += ns;
            if (i < slot.labels.size())
              gpu_labels_ns_[(uint64_t(slot.categories[i]) << 32) | slot.labels[i]] += ns;
            ++cat_intervals_[slot.categories[i]];
            cat_max_ns_[slot.categories[i]] = std::max(cat_max_ns_[slot.categories[i]], ns);
            if (ns >= 1000000) ++cat_long_[slot.categories[i]];  // >= 1 ms
            // Diagnostic: the longest single interval between two marks (a stall shows up as one block).
            if (ns > mark_max_ns_) {
              mark_max_ns_ = ns;
              mark_max_desc_ = fmt::format("{:.2f} ms in category {} at mark {}/{} (prev cat {}, next cat {})",
                  double(ns) / 1e6, slot.categories[i], i, marked,
                  i ? int(slot.categories[i - 1]) : -1,
                  i + 1 < marked - 1 ? int(slot.categories[i + 1]) : -1);
            }
          }
        }
        // GPU gap since the end of the previous submission (report). Submissions are read in the order
        // they were sent; if one could not be read, mark_previous_end_ is 0 and that gap is not counted.
        if (mark_previous_end_ != 0) {
          if (marks[0] >= mark_previous_end_) {
            gpu_categories_ns_[kGpuGapBetweenJobs] +=
                uint64_t(double(marks[0] - mark_previous_end_) * period_mark_ns_);
          }
        }
        mark_previous_end_ = marks[marked - 1];
        gpu_ns_ += uint64_t(double(marks[marked - 1] - marks[0]) * period_mark_ns_);
      } else {
        mark_previous_end_ = 0;  // without this submission's timestamps, the next gap is not counted
      }
    } else {
      mark_previous_end_ = 0;
    }
    slot.categories.clear();
    slot.labels.clear();
    slot.descriptions.clear();
    slot.draws_mark.clear();
    ReadStats(slot);
    ReadStatsDraw(slot);
    WriteReads(slot.reads);
  }

  // Per-draw statistics of a finished submission, added to its pixel shader.
  void ReadStatsDraw(SlotWork& slot) {
    if (slot.stats_draw.empty()) {
      return;
    }
    const uint32_t n = uint32_t(slot.stats_draw.size());
    std::vector<uint64_t> values(size_t(n) * kCountersStat, 0);
    if (stats_draw_ != VK_NULL_HANDLE && read_queries_ &&
        read_queries_(device_, stats_draw_, slot.stats_draw.front().first, n,
                        sizeof(uint64_t) * values.size(), values.data(),
                        sizeof(uint64_t) * kCountersStat,
                        VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      for (uint32_t i = 0; i < n; ++i) {
      }
      // GPU time per (category, full pixel shader number).
      std::vector<uint64_t> marks(size_t(n) * 2, 0);
      const bool with_marks =
          marks_draw_ != VK_NULL_HANDLE && slot.draw_ps.size() == n &&
          read_queries_(device_, marks_draw_, slot.stats_draw.front().first * 2, n * 2,
                          sizeof(uint64_t) * marks.size(), marks.data(), sizeof(uint64_t),
                          VK_QUERY_RESULT_64_BIT) == VK_SUCCESS;
      if (slot.draw_ps.size() == n) {
        for (uint32_t i = 0; i < n; ++i) {
          CostShader& cost = cost_per_shader_[slot.draw_ps[i]];
          if (with_marks && marks[i * 2 + 1] > marks[i * 2]) {
            const uint64_t ns = uint64_t(double(marks[i * 2 + 1] - marks[i * 2]) * period_mark_ns_);
            cost.ns += ns;
            if (ns >= cost.draw_more_expensive_ns && i < slot.draw_vs.size()) {
              cost.draw_more_expensive_ns = uint32_t(std::min<uint64_t>(ns, UINT32_MAX));
              cost.vs = slot.draw_vs[i];
            }
          }
          cost.vertices += values[i * kCountersStat + 0];
          cost.primitives += values[i * kCountersStat + 1];
          cost.fragments += values[i * kCountersStat + 2];
          ++cost.draws;
        }
      }
      // A diagnostic frame may span several submissions: the first one of each window counts.
      if (!read_window_) {
        read_window_ = true;
        ++cost_frames_;
      }
    }
    slot.stats_draw.clear();
    slot.draw_ps.clear();
    slot.draw_vs.clear();
  }

  // the pixel shaders with the most GPU time in the diagnostic frames since the last report
  // (sa2_native_stats_per_draw_s). Times are serialized per draw and x1.627 on Switch (NVK
  // timestamps).
  void ReportCostPerShader() {
    if (cost_per_shader_.empty() || !cost_frames_) {
      return;
    }
    constexpr double kScale = 1.627;
    std::vector<std::pair<uint32_t, CostShader>> order(cost_per_shader_.begin(), cost_per_shader_.end());
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    std::array<double, kGpuCategories> per_category{};
    double total = 0.0;
    uint64_t draws = 0, fragments = 0;
    for (const auto& [key, cost] : order) {
      const double ms = double(cost.ns) * kScale / 1e6 / double(cost_frames_);
      per_category[(key >> 24) % kGpuCategories] += ms;
      total += ms;
      draws += cost.draws;
      fragments += cost.fragments;
    }
    std::string categories;
    for (uint32_t c = 0; c < kGpuCategories; ++c) {
      if (per_category[c] >= 0.05) categories += fmt::format(" c{} {:.1f}", c, per_category[c]);
    }
    REXLOG_INFO("[native] per-draw GPU time ({} diagnostic frames, serialized): {:.1f} ms/frame, {} draws, "
                "{:.2f} Mfrag; by category (ms):{}",
                cost_frames_, total, draws / cost_frames_,
                double(fragments) / 1e6 / double(cost_frames_), categories);
    const size_t n = std::min<size_t>(order.size(), 25);
    for (size_t i = 0; i < n; ++i) {
      const auto& [key, cost] = order[i];
      const double ms = double(cost.ns) * kScale / 1e6 / double(cost_frames_);
      const uint32_t ps = key & 0xFFFFFF;
      REXLOG_INFO("[native]   #{} PS n{} cat {}: {:.2f} ms/frame, {} draws, {:.3f} Mfrag, {:.1f} ns/frag, "
                  "{} vertices, {} clip prims, costliest draw {:.2f} ms with VS n{}",
                  i + 1, ps ? int(ps) - 1 : -1, key >> 24, ms, cost.draws / cost_frames_,
                  double(cost.fragments) / 1e6 / double(cost_frames_),
                  cost.fragments ? double(cost.ns) * kScale / double(cost.fragments) : 0.0,
                  cost.vertices / cost_frames_, cost.primitives / cost_frames_,
                  double(cost.draw_more_expensive_ns) * kScale / 1e6, cost.vs ? int(cost.vs) - 1 : -1);
    }
    cost_per_shader_.clear();
    cost_frames_ = 0;
  }

  // Pass statistics of a finished submission, added to their category.
  void ReadStats(SlotWork& slot) {
    if (slot.stats.empty()) {
      return;
    }
    const uint32_t n = uint32_t(slot.stats.size());
    std::array<uint64_t, kStatsPerSlot * kCountersStat> values{};
    if (stats_ != VK_NULL_HANDLE && read_queries_ &&
        read_queries_(device_, stats_, slot.stats.front().first, n,
                        sizeof(uint64_t) * kCountersStat * n, values.data(),
                        sizeof(uint64_t) * kCountersStat,
                        VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
      for (uint32_t i = 0; i < n; ++i) {
        const uint8_t c = slot.stats[i].second;
        if (c < kGpuCategories) {
          vertices_category_[c] += values[i * kCountersStat + 0];
          primitives_category_[c] += values[i * kCountersStat + 1];
          fragments_category_[c] += values[i * kCountersStat + 2];
        }
      }
    }
    slot.stats.clear();
  }

  bool SendWork(bool wait) {
    if (recording_) {
      ++report_sends_;
      if (draws_) {
        draws_->BeforeSend();  // closes the pass and publishes the upload buffer
      }
      recording_ = false;
      std::array<VkCommandBuffer, 2> buffers{};
      uint32_t n = 0;
      if (recording_upload_) {
        recording_upload_ = false;
        // Make uploaded texture texels and attachment initialization visible to
        // the work command buffer in this same submission.
        VkMemoryBarrier after_uploads{};
        after_uploads.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        after_uploads.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
        after_uploads.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
        dfn_.vkCmdPipelineBarrier(commands_upload_, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                  VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &after_uploads,
                                  0, nullptr, 0, nullptr);
        // usf4_deferred_async_submit: ended on the worker with the work buffer (through the proxy it would drain).
        if (!deferred::AsyncSubmit() && dfn_.vkEndCommandBuffer(commands_upload_) != VK_SUCCESS) {
          return false;
        }
        buffers[n++] = commands_upload_;
      }
      if (queries_ != VK_NULL_HANDLE) {
        // Final mark: closes the last span (MarkGpu leaves room for it).
        SlotWork& marked = slots_[slot_];
        write_mark_(commands_work_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_,
                        slot_ * kMarksPerSlot + uint32_t(marked.categories.size()));
        marked.categories.push_back(kGpuEnd);
        marked.marks_written = std::max(marked.marks_written, uint32_t(marked.categories.size()));
      }
      if (deferred::AsyncSubmit()) {
        // usf4_deferred_async_submit: ending the buffers and submitting them run on the worker after the commands
        // already queued; the ring thread goes on. Whoever needs this slot's fence waits for its ticket first.
        buffers[n++] = commands_work_;
        const VkFence fence = slots_[slot_].fence;
        slots_[slot_].ticket = deferred::RunOnWorker(new std::function<void()>([this, buffers, n, fence] {
          for (uint32_t i = 0; i < n; ++i) {
            if (deferred::Real().vkEndCommandBuffer(buffers[i]) != VK_SUCCESS) {
              REXLOG_ERROR("[native] async submission: vkEndCommandBuffer failed");
              return;
            }
          }
          VkSubmitInfo send{};
          send.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
          send.commandBufferCount = n;
          send.pCommandBuffers = buffers.data();
          const auto queue = vulkan_device_->AcquireQueue(family_, 0);
          if (deferred::Real().vkQueueSubmit(queue.queue(), 1, &send, fence) != VK_SUCCESS)
            REXLOG_ERROR("[native] async submission: vkQueueSubmit failed");
        }));
      } else {
      if (dfn_.vkEndCommandBuffer(commands_work_) != VK_SUCCESS) {
        return false;
      }
      buffers[n++] = commands_work_;
      VkSubmitInfo send{};
      send.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      send.commandBufferCount = n;
      send.pCommandBuffers = buffers.data();
      {
        const auto queue = vulkan_device_->AcquireQueue(family_, 0);
        if (dfn_.vkQueueSubmit(queue.queue(), 1, &send, slots_[slot_].fence) != VK_SUCCESS) {
          return false;
        }
      }
      slots_[slot_].ticket = 0;
      }
      SlotWork& slot = slots_[slot_];
      slot.pending = true;
      slot.order = ++sends_;
      slot.sent = std::chrono::steady_clock::now();
      // Read-backs recorded in this submission are written when it finishes.
      slot.reads.insert(slot.reads.end(), pending_reads_.begin(),
                             pending_reads_.end());
      pending_reads_.clear();
    }
    if (wait) {
      Complete(slots_[slot_]);
    }
    return true;
  }

  // banjo-nx: parked resolved images (banjo_native_park_resolved), by address, size and host format.
  static uint64_t ParkedKey(uint32_t base, uint32_t width, uint32_t height, uint32_t format) {
    return (uint64_t(base) << 32) ^ (uint64_t(width & 0xFFF) << 20) ^ (uint64_t(height & 0xFFF) << 8) ^
           (uint64_t(format) * 0x9E3779B1ull);
  }
  std::unordered_map<uint64_t, Resolved> parked_resolved_;
  uint64_t parked_resolved_stores_ = 0, parked_resolved_reuses_ = 0;

  // full GPU drains per call site since the last report (see ReportWaitsGpu).
  std::array<uint64_t, 16> waits_gpu_reason_{};
  void ReportWaitsGpu() {
    std::string list;
    if (parked_resolved_stores_ || parked_resolved_reuses_)
      list += fmt::format(" (banjo parked resolves: {} stored, {} reused, {} parked)", parked_resolved_stores_,
                          parked_resolved_reuses_, parked_resolved_.size());
    parked_resolved_stores_ = parked_resolved_reuses_ = 0;
    for (size_t i = 1; i < waits_gpu_reason_.size(); ++i) {
      if (waits_gpu_reason_[i]) list += fmt::format(" site{}={}", i, waits_gpu_reason_[i]);
      waits_gpu_reason_[i] = 0;
    }
    // sa2-nx: printed whether or not any EDRAM transfer happened (with the aliased clears redirected there are none).
    if (edram4_redirected_color0_tiles_ || edram4_redirected_)
      REXLOG_INFO("[native] EDRAM mode4 redirected clears: {} depth clears issued as {} merged clears; color 0 alias "
                  "clears into their 1x owners: {} tiles in {} clears", edram4_redirected_, edram4_redirected_merged_,
                  edram4_redirected_color0_tiles_, edram4_redirected_color0_areas_);
    if (usf4_color0_into_depth_tiles_)
      REXLOG_INFO("[native] EDRAM mode4 color 0 clears into depth owners (usf4_edram4_clear_into_depth): {} tiles",
                  usf4_color0_into_depth_tiles_);
    usf4_color0_into_depth_tiles_ = 0;
    if (usf4_color0_consumer_color_ || usf4_color0_consumer_depth_ || !usf4_color0_consumer_no_.empty()) {
      std::string no;
      for (const auto& [reason, n] : usf4_color0_consumer_no_) no += fmt::format(" [{}: {}]", reason, n);
      REXLOG_INFO("[native] EDRAM mode4 color 0 clears into their consumer (usf4_edram4_clear_color_consumer): {} "
                  "into a color view, {} into a depth view; declined:{}", usf4_color0_consumer_color_,
                  usf4_color0_consumer_depth_, no.empty() ? " none" : no);
      usf4_color0_consumer_color_ = usf4_color0_consumer_depth_ = 0;
      usf4_color0_consumer_no_.clear();
    }
    edram4_redirected_color0_tiles_ = edram4_redirected_color0_areas_ = edram4_redirected_merged_ = 0;
    if (const uint64_t memo_total = edram4_memo_hits_ + edram4_memo_misses_ + edram4_memo_ineligible_) {
      REXLOG_INFO("[native] EDRAM draw memo (sa2_cpu_ring_edram_memo {}): {} draws prepared from the memo, {} by the "
                  "full path, {} not eligible ({:.1f} % from the memo); {} checked against the full path",
                  g_edram_memo.active && edram4_memo_on_ ? "on" : "off", edram4_memo_hits_, edram4_memo_misses_,
                  edram4_memo_ineligible_, 100.0 * double(edram4_memo_hits_) / double(memo_total),
                  edram4_memo_checked_);
      edram4_memo_hits_ = edram4_memo_misses_ = edram4_memo_ineligible_ = edram4_memo_checked_ = 0;
    }
    if (!edram4_operations_.empty()) {
      std::string ops;
      for (const auto& [name, n] : edram4_operations_) ops += fmt::format(" {}={}", name, n);
      edram4_operations_.clear();
      REXLOG_INFO("[native] EDRAM mode4 operations since last report:{}; sync cache: {} tiles skipped, "
                  "{} transferred; batched: {} runs, {} import passes and {} compute barrier pairs saved",
                  ops, edram4_sync_skipped_, edram4_sync_copied_, edram4_batch_spans_,
                  edram4_batch_saved_passes_, edram4_batch_saved_barriers_);
      edram4_sync_skipped_ = edram4_sync_copied_ = 0;
      edram4_batch_spans_ = edram4_batch_saved_passes_ = edram4_batch_saved_barriers_ = 0;
      static const char* kCaller[4] = {"draw", "clear", "color-resolve", "depth-resolve"};
      static const char* kKind[5] = {"depth-repr-switch", "depth-depth", "color->depth", "depth->color",
                                     "color->color"};
      std::string causes;
      for (uint32_t m = 0; m < 4; ++m)
        for (uint32_t k = 0; k < 5; ++k)
          if (edram4_transfer_ops_[m][k])
            causes += fmt::format(" {}/{}={}ops/{}tiles", kCaller[m], kKind[k], edram4_transfer_ops_[m][k],
                                  edram4_transfer_tiles_[m][k]);
      std::string full;
      for (uint32_t k = 0; k < 5; ++k)
        if (edram4_transfer_full_[k]) full += fmt::format(" {}={}", kKind[k], edram4_transfer_full_[k]);
      REXLOG_INFO("[native] EDRAM mode4 transfer causes:{}; whole-target runs:{}", causes,
                  full.empty() ? " none" : full);
      if (REXCVAR_GET(sa2_native_conversion_frag)) {
        REXLOG_INFO("[native] EDRAM mode4 color conversions as fragment passes: {} done, {} declined (compute) since "
                    "the last report", conv_color_frag_uses_ - conv_color_frag_uses_report_,
                    conv_color_frag_rejections_ - conv_color_frag_rejections_report_);
        {
          std::string m;
          for (uint32_t k = 0; k < conv_color_frag_reasons_.size(); ++k)
            if (conv_color_frag_reasons_[k]) m += fmt::format(" [{}: {}]", k, conv_color_frag_reasons_[k]);
          if (!m.empty()) REXLOG_INFO("[native] EDRAM mode4 fragment conversions declined by (return #):{}", m);
          conv_color_frag_reasons_ = {};
        }
        conv_color_frag_uses_report_ = conv_color_frag_uses_;
        conv_color_frag_rejections_report_ = conv_color_frag_rejections_;
      }
      std::vector<std::pair<std::string, ParEDRAM4>> pairs(edram4_transfer_pairs_.begin(),
                                                           edram4_transfer_pairs_.end());
      std::sort(pairs.begin(), pairs.end(),
                [](const auto& a, const auto& b) { return a.second.ops > b.second.ops; });
      std::string top;
      for (size_t i = 0; i < std::min<size_t>(pairs.size(), 10); ++i)
        top += fmt::format(" [{} {}ops/{}tiles]", pairs[i].first, pairs[i].second.ops, pairs[i].second.tiles);
      for (size_t i = 0; i < std::min<size_t>(pairs.size(), 8); ++i)
      {
        std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> d(pairs[i].second.draws.begin(),
                                                                             pairs[i].second.draws.end());
        std::sort(d.begin(), d.end(), [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
        std::string list;
        for (size_t j = 0; j < std::min<size_t>(d.size(), 5); ++j)
          list += fmt::format(" {}={}ops/{}t", d[j].first, d[j].second.first, d[j].second.second);
        REXLOG_INFO("[native] EDRAM mode4 pair {} draws:{} | first: {}", pairs[i].first, list,
                    pairs[i].second.example);
      }
      REXLOG_INFO("[native] EDRAM mode4 top transfer pairs:{}; proven-overwrite binds {}; redirected depth "
                  "clears {} ({} tiles), declined {}; redirected stencil clears {} ({} tiles), declined {}; "
                  "GPU marks dropped {}", top, edram4_overwrites_,
                  edram4_redirected_, edram4_redirected_tiles_, edram4_redirected_no_,
                  edram4_stencil_redirected_, edram4_stencil_redirected_tiles_, edram4_stencil_redirected_no_,
                  exhausted_marks_);
      exhausted_marks_ = 0;
      if (!edram4_stencil_no_reasons_.empty()) {
        std::string m;
        for (const auto& [k, v] : edram4_stencil_no_reasons_) m += fmt::format(" [{}: {}]", k, v);
        REXLOG_INFO("[native] EDRAM mode4 stencil clear redirect declined by:{}", m);
        edram4_stencil_no_reasons_.clear();
      }
      for (const auto& [k, v] : edram4_depth_no_fast_)
        edram4_depth_no_reasons_[fmt::format("{} (depth info {:08X})", k.first, k.second)] += v;
      edram4_depth_no_fast_.clear();
      if (!edram4_depth_no_reasons_.empty()) {
        std::string m;
        for (const auto& [k, v] : edram4_depth_no_reasons_) m += fmt::format(" [{}: {}]", k, v);
        REXLOG_INFO("[native] EDRAM mode4 depth clear redirect declined by:{}", m);
        edram4_depth_no_reasons_.clear();
      }
      if (edram4_redirected_color_tiles_)
        REXLOG_INFO("[native] EDRAM mode4 redirected clears into color owners: {} tiles", edram4_redirected_color_tiles_);
      edram4_redirected_color_tiles_ = 0;
      if (edram4_consumer_uses_)
        REXLOG_INFO("[native] EDRAM mode4 redirected clears straight into their consumer: {}", edram4_consumer_uses_);
      edram4_consumer_uses_ = 0;
      edram4_stencil_redirected_ = edram4_stencil_redirected_tiles_ = edram4_stencil_redirected_no_ = 0;
      REXLOG_INFO("[native] GPU longest single interval since last report: {}", mark_max_desc_.empty() ? "-" : mark_max_desc_);
      {
        std::string cats;
        for (uint32_t c = 0; c < kGpuCategories; ++c)
          if (cat_intervals_[c])
            cats += fmt::format(" [cat {}: {} intervals, {} >= 1 ms, max {:.2f} ms]", c, cat_intervals_[c],
                                cat_long_[c], double(cat_max_ns_[c]) / 1e6);
        REXLOG_INFO("[native] GPU intervals by category:{}", cats);
        std::vector<std::pair<uint64_t, uint64_t>> top(gpu_labels_ns_.begin(), gpu_labels_ns_.end());
        std::sort(top.begin(), top.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string list;
        for (size_t k = 0; k < std::min<size_t>(top.size(), 16); ++k)
          list += fmt::format(" [cat {} VS{} PS{}: {:.1f} ms]", top[k].first >> 32, (top[k].first >> 16) & 0xFFFF,
                               top[k].first & 0xFFFF, double(top[k].second) / 1e6);
        REXLOG_INFO("[native] GPU time by pass (category, first draw's shaders), 10 s:{}", list);
        gpu_labels_ns_.clear();
        cat_intervals_ = {};
        cat_max_ns_ = {};
        cat_long_ = {};
      }
      mark_max_ns_ = 0;
      mark_max_desc_.clear();
      REXLOG_INFO("[native] EDRAM mode4 CPU bookkeeping since last report: sync {} calls / {} tiles, publish {} calls "
                  "/ {} tiles; O(1) fast paths: sync {}, publish {}", edram4_sync_calls_, edram4_sync_visited_,
                  edram4_pub_calls_, edram4_pub_visited_, edram4_sync_fast_, edram4_pub_fast_);
      edram4_sync_fast_ = edram4_pub_fast_ = 0;
      REXLOG_INFO("[native] EDRAM mode4 lazy stencil: {} depth-only imports, {} returned home (stencil kept), "
                  "{} inexact, {} late stencil users, {} exports of deferred stencil, {} tiles inherited by depth-only "
                  "draws, {} before a full stencil replace, {} late fetches ({} tiles)", edram4_stencil_skipped_,
                  edram4_stencil_returned_, edram4_stencil_inexact_, edram4_stencil_late_,
                  edram4_stencil_exported_, edram4_stencil_inherited_, edram4_stencil_replaced_,
                  edram4_stencil_fetched_, edram4_stencil_fetched_tiles_);
      edram4_stencil_inherited_ = edram4_stencil_replaced_ = edram4_stencil_fetched_ =
          edram4_stencil_fetched_tiles_ = 0;
      std::string nine;
      for (const auto& [k, v] : edram4_import9_pairs_) nine += fmt::format(" [{} {}ops/{}t]", k, v.first, v.second);
      REXLOG_INFO("[native] EDRAM mode4 9-pass imports:{}; stencil bit passes skipped (bit never set): {}; "
                  "copy-engine stencil imports {} ({} tiles); column imports (usf4_edram4_import_columns) {} "
                  "replacing {} runs, drawn {} failed {}",
                  nine.empty() ? " none" : nine, edram4_bits_skipped_, edram4_stencil_copies_,
                  edram4_stencil_copies_tiles_, edram4_columns_merged_, edram4_columns_runs_,
                  edram4_columns_drawn_, edram4_columns_failed_);
      edram4_bits_skipped_ = edram4_stencil_copies_ = edram4_stencil_copies_tiles_ = 0;
      edram4_columns_merged_ = edram4_columns_runs_ = edram4_columns_drawn_ = edram4_columns_failed_ = 0;
      edram4_import9_pairs_.clear();
      edram4_stencil_skipped_ = edram4_stencil_returned_ = edram4_stencil_inexact_ = edram4_stencil_late_ =
          edram4_stencil_exported_ = 0;
      edram4_sync_calls_ = edram4_sync_visited_ = edram4_pub_calls_ = edram4_pub_visited_ = 0;
      edram4_overwrites_ = edram4_redirected_ = edram4_redirected_tiles_ = edram4_redirected_no_ = 0;
      edram4_transfer_pairs_.clear();
      edram4_transfer_ops_ = {}; edram4_transfer_tiles_ = {}; edram4_transfer_full_ = {};
    }
    uint64_t full_count = 0, ns_full = 0;
    if (draws_) {
      const auto e = draws_->Stats();
      full_count = e.full_sends - report_full_sends_;
      ns_full = e.ns_full_sends - report_ns_full_sends_;
      report_full_sends_ = e.full_sends;
      report_ns_full_sends_ = e.ns_full_sends;
    }
    REXLOG_INFO("[native] submissions since last report: {} ({} for a full upload buffer, {:.1f} ms inside "
                "them); slot fence waits {} ({:.1f} ms); full GPU drains:{}",
                report_sends_, full_count, double(ns_full) / 1e6, waits_gpu_ - report_waits_gpu_,
                double(ns_waits_gpu_ - report_ns_waits_gpu_) / 1e6, list.empty() ? " none" : list);
    report_sends_ = 0;
    report_waits_gpu_ = waits_gpu_;
    report_ns_waits_gpu_ = ns_waits_gpu_;
  }
  uint64_t report_sends_ = 0, report_full_sends_ = 0, report_ns_full_sends_ = 0;
  uint64_t report_waits_gpu_ = 0, report_ns_waits_gpu_ = 0;

  void WaitGpu() {
    if (recording_) {
      if (!SendWork(false)) {
        resolved_pool_device_error_ = true;
        REXLOG_ERROR("[native] GPU submission failed; resolved allocation cache disabled");
        return;
      }
    }
    for (SlotWork& slot : slots_) {
      Complete(slot);
    }
    WaitOutputs();
  }

  // All pending outputs, before destroying or reusing what they use.
  void WaitOutputs() {
    // usf4_deferred_async_submit: the outputs are painted on the worker; on the ring thread, let it finish first
    // (on the worker itself this is a no-op).
    deferred::Drain();
    for (uint32_t i = 0; i < kSlotsOutput; ++i) {
      if (pending_outputs_[i]) {
        const VkResult waited = dfn_.vkWaitForFences(device_, 1, &fences_output_[i], VK_TRUE, UINT64_MAX);
        if (waited != VK_SUCCESS) {
          resolved_pool_device_error_ = true;
          REXLOG_ERROR("[native] output fence wait failed {}; resolved allocation cache disabled", int(waited));
          return;
        }
        dfn_.vkResetFences(device_, 1, &fences_output_[i]);
        pending_outputs_[i] = false;
      }
    }
  }

  // Copy inventory for the report: for each resolved texture, how many copies there are and how many had
  // draws since the previous copy (a resolve without draws copies again what the render target already
  // had). It has to be complete (address, size, Mpixels and reads) because it is the only thing that tells
  // whether a copy is needed; ~20 addresses fit easily in one line.
  void NoteCopy(uint32_t base, uint32_t width, uint32_t height, uint64_t draws) {
    CopyTarget& d = copies_per_target_[base];
    d.width = width;
    d.height = height;
    ++d.copies;
    d.draws += draws;
    d.with_draws += draws != 0 ? 1 : 0;
    const auto now = std::chrono::steady_clock::now();
    if (now - report_copies_ < std::chrono::seconds(10)) {
      return;
    }
    report_copies_ = now;
    if (resolve_draws_ != resolve_draws_report_ || resolve_draws_declined_ != resolve_draws_declined_report_) {
      REXLOG_INFO("[native] targets: resolves drawn on the 3D engine (sa2_gpu_resolve_draw) since the previous "
                  "report: {}, copied as before: {}", resolve_draws_ - resolve_draws_report_,
                  resolve_draws_declined_ - resolve_draws_declined_report_);
      resolve_draws_report_ = resolve_draws_;
      resolve_draws_declined_report_ = resolve_draws_declined_;
    }
    if (aliases_made_ != aliases_made_report_ || aliases_copied_ != aliases_copied_report_ ||
        aliases_exchanged_ != aliases_exchanged_report_) {
      REXLOG_INFO("[native] targets: repeated resolves not copied (sa2_gpu_resolve_alias) since the previous report: "
                  "{}; of the earlier ones, {} overwritten before anything read them, {} given their image by an "
                  "exchange, {} copied (read, or their source changed); pending {}",
                  aliases_made_ - aliases_made_report_, aliases_dropped_ - aliases_dropped_report_,
                  aliases_exchanged_ - aliases_exchanged_report_, aliases_copied_ - aliases_copied_report_,
                  aliases_.size());
      aliases_made_report_ = aliases_made_;
      aliases_dropped_report_ = aliases_dropped_;
      aliases_copied_report_ = aliases_copied_;
      aliases_exchanged_report_ = aliases_exchanged_;
    }
    if (REXCVAR_GET(sa2_native_edram_alias_mode) == 4)
      REXLOG_INFO("[native] EDRAM mode4 conversion capacity totals: {} rollovers, {} failed, current slot {} count {}/{}",
                  edram4_capacity_rollovers_, edram4_capacity_rollover_failures_, slot_,
                  slots_[slot_].conversions_edram, kConversionsEDRAMPerSlot);
    std::vector<std::pair<uint32_t, CopyTarget>> order(copies_per_target_.begin(),
                                                         copies_per_target_.end());
    std::sort(order.begin(), order.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    std::string list;
    uint32_t written = 0;
    for (const auto& [address, c] : order) {
      if (++written > 48) {  // defensive cap: ~20 addresses per frame, but a load can touch many more
        list += fmt::format(" (and {} more addresses)", order.size() - 48);
        break;
      }
      const auto r = resolved_.find(address);
      const uint64_t reads = r != resolved_.end() ? r->second.reads : 0;
      if (r != resolved_.end()) {
        r->second.reads = 0;
      }
      list += fmt::format(" {:08X} {}x{}: {} copies, {} with draws ({:.1f} draws per copy), "
                           "{:.2f} Mpixels, {} reads{};",
                           address, c.width, c.height, c.copies, c.with_draws,
                           c.copies ? double(c.draws) / double(c.copies) : 0.0,
                           double(c.pixels) / 1e6, reads,
                           !c.copies || reads ? ""
                           // usf4-nx: the Swap reads the front buffer without going through ResolvedTexture
                           : presented_fronts_.count(address) ? " (presented by the Swap)"
                                                              : " *** NOBODY READS IT ***");
    }
    copies_per_target_.clear();
    SA2_REPORT_RING("[native] targets: faces resolved since the previous report:{}", list);
    ReportOverlapsEDRAM();
    ReportWaitsGpu();
    // sa2-nx render resolution / MSAA: what the scaled images take (only once either has been used).
    if (scale_q_ != kScaleQuartersNative || msaa_samples_ > 1 || rescales_) {
      SA2_REPORT_RING("[res] images at {}/4 of the game's size{}: {:.1f} MB (render targets, depth targets, "
                             "resolved textures); {} rescales so far", scale_q_,
                             msaa_samples_ > 1 ? fmt::format(", {}x MSAA", msaa_samples_) : std::string(),
                             double(BytesScaledImages()) / 1048576.0, rescales_);
    }
    if (swaps_ || restores_) {
      // Restores are the verdict. If they go up, the swap without a clear saves nothing: the same copy is
      // paid, just later.
      SA2_REPORT_RING("[native] targets: resolves without copy: {} swaps ({} without clear) and {} "
                  "restores since startup -> {}",
                  swaps_, swaps_no_clear_, restores_,
                  restores_ == 0 ? "not a single restore: the saving is clean"
                                       : "*** there are restores: the saving is NOT clean ***");
      // sa2_native_resolver_valid_content. One per frame when shadows are resolved twice in a row; 0 otherwise. The
      // "no source" count is resolves of a render target whose content is no longer anywhere (must be
      // 0); the other count is the resolves that, with the setting off, would read the stale image.
      if (restores_for_resolver_ || resolver_no_source_ || resolver_old_content_) {
        SA2_REPORT_RING("[native] targets: resolve with valid content: {} times the content was brought back "
                             "before resolving (the resolve used to read the previous frame's image: the "
                             "menu flicker); {} without source; {} read as before (setting off)",
                             restores_for_resolver_, resolver_no_source_, resolver_old_content_);
      }
      // And what they cost per frame, which is the only thing that decides. 0.78 real ms per Mpixel
      // copied and 0.075 per Mpixel cleared, measured here.
      const double frames = double(presented_ - presented_report_copies_);
      const double perFot = frames > 0.0 ? 1.0 / frames : 0.0;
      const double mp_rest = double(restored_pixels_ - restored_previous_pixels_) / 1e6;
      const double mp_saving =
          double(pixels_restore_saved_ - pixels_restore_saved_previous_) / 1e6;
      SA2_REPORT_RING("[native] targets: restores per frame: {:.2f} Mpixels copied ({:.2f} ms "
                  "real) and {:.2f} Mpixels saved ({:.2f} ms); since startup: {} clipped "
                  "to the useful area",
                  mp_rest * perFot, mp_rest * perFot * 0.78, mp_saving * perFot,
                  mp_saving * perFot * 0.78, clipped_restores_);
      restored_previous_pixels_ = restored_pixels_;
      pixels_restore_saved_previous_ = pixels_restore_saved_;
    }
    if (cleared_skipped_ || cleared_skipped_depth_) {
      // Clears that did not change a single bit (sa2_native_skip_cleared_repeated). A clear costs
      // 0.075 real ms per Mpixel: ten times less than a copy, but it adds up.
      const double frames = double(presented_ - presented_report_copies_);
      const double perFot = frames > 0.0 ? 1.0 / frames : 0.0;
      const double mp = double(cleared_skipped_pixels_ - cleared_skipped_previous_pixels_) / 1e6;
      SA2_REPORT_RING("[native] targets: clears skipped: {} color and {} depth since startup; "
                  "{:.2f} Mpixels per frame not cleared ({:.2f} ms real)",
                  cleared_skipped_, cleared_skipped_depth_, mp * perFot,
                  mp * perFot * 0.075);
      cleared_skipped_previous_pixels_ = cleared_skipped_pixels_;
    }
    if (cleared_useless_) {
      // Clears wiped out by a swap or a restore before anything was drawn. These are the real candidates
      // for removal (the equivalent of a loadOp = DONT_CARE).
      const double frames = double(presented_ - presented_report_copies_);
      const double perFot = frames > 0.0 ? 1.0 / frames : 0.0;
      const double mp = double(cleared_useless_pixels_ - cleared_useless_previous_pixels_) / 1e6;
      SA2_REPORT_RING("[native] targets: clears that served no purpose (the target changed content without "
                  "anything being drawn): {} since startup; {:.2f} Mpixels per frame "
                  "({:.2f} ms real)",
                  cleared_useless_, mp * perFot, mp * perFot * 0.075);
      cleared_useless_previous_pixels_ = cleared_useless_pixels_;
    }
    if (postponed_front_ || front_late_reads_) {  // sa2_native_lazy_front
      const double frames = double(presented_ - presented_report_copies_);
      const double perFot = frames > 0.0 ? 1.0 / frames : 0.0;
      const double mp = double(front_saved_pixels_ - front_saved_previous_pixels_) / 1e6;
      SA2_REPORT_RING("[native] targets: lazy front (deferred): {} copies to front buffers postponed since startup; "
                           "Swaps painted from the target {} and from a retained image {}; {} clears on a "
                           "spare image ({} without a spare); recorded: {} when sampled, {} before writing the "
                           "target again or another resolve, {} in the Swap (FXAA, no ramp or another output); {} dropped "
                           "without copying (another resolve covers them entirely): {:.2f} Mpixels per frame not copied (~{:.2f} "
                           "ms real); {} spare images; late reads {}{}",
                           postponed_front_, painted_front_target_, painted_retained_front_,
                           front_rotations_, front_no_replenished_, copied_front_read_,
                           copied_front_write_, copied_front_swap_, replaced_front_, mp * perFot,
                           mp * perFot * 0.60, fronts_images_.size(), front_late_reads_,
                           off_front_ ? " *** TURNED OFF BY THE GUARD ***" : " (0 = the image is the same)");
      front_saved_previous_pixels_ = front_saved_pixels_;
    }
    presented_report_copies_ = presented_;
    if (no_swap_[0] || no_swap_[1] || no_swap_[2]) {
      SA2_REPORT_RING("[native] targets: depth copies without swap since the previous report: {} because the "
                  "command does not clear the target, {} because it is not resolved whole, {} because it was not possible",
                  no_swap_[0], no_swap_[1], no_swap_[2]);
      no_swap_ = {};
    }
  }

  // Read-back of a small resolved texture: the copied rectangle goes to a host-visible buffer in the
  // same submission; WriteReads moves it into guest memory.
  void ResolvedRead(const RegistersCopy& reg, const Resolved& resolved, int32_t x0, int32_t y0,
                    uint32_t dx, uint32_t dy, uint32_t width, uint32_t height) {
    const int32_t max = REXCVAR_GET(sa2_native_resolved_read_texels);
    if (max <= 0 || uint64_t(width) * height > uint64_t(max) ||
        resolved.image.format != kColorFormat) {
      return;
    }
    const uint64_t target_key =
        (uint64_t(reg.rb_copy_dest_base) << 28) ^ (uint64_t(width) << 14) ^ uint64_t(height);
    TargetRead& target = reads_per_target_[target_key];
    target.base = reg.rb_copy_dest_base;
    target.width = width;
    target.height = height;
    const uint64_t each = uint64_t(std::max(REXCVAR_GET(sa2_native_reads_each), 1));
    if (target.copies++ % each != 0) {
      ++target.skipped;  // exposure changes slowly: the guest keeps the previous one
      return;
    }
    const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
    const uint64_t key = (uint64_t(reg.rb_copy_dest_base) << 32) ^ (uint64_t(uint32_t(x0)) << 16) ^
                           uint64_t(uint32_t(y0));
    // One buffer per slot: the previous submission may still be copying into the other slot's.
    Read* pointer = &reads_[key ^ (uint64_t(slot_) << 63)];
    if (pointer->bytes < bytes && pointer->buffer != VK_NULL_HANDLE) {
      // Larger: the old buffer may have recorded or submitted copies.
      SendWork(true);
      ++waits_gpu_reason_[5]; WaitGpu();
      DestroyRead(*pointer);
      if (!Record()) {
        return;
      }
      pointer = &reads_[key ^ (uint64_t(slot_) << 63)];
      if (pointer->bytes < bytes && pointer->buffer != VK_NULL_HANDLE) {
        DestroyRead(*pointer);  // the GPU has nothing pending any more
      }
    }
    Read& read = *pointer;
    if (read.bytes < bytes) {
      uint32_t type = 0;
      if (!rex::ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device_, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
              rex::ui::vulkan::util::MemoryPurpose::kReadback, read.buffer, read.memory, &type)) {
        read = Read{};
        Reject(16, "could not create a read-back buffer for a resolved texture");
        return;
      }
      void* mapped = nullptr;
      if (dfn_.vkMapMemory(device_, read.memory, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS) {
        DestroyRead(read);
        Reject(16, "could not create a read-back buffer for a resolved texture");
        return;
      }
      read.data = static_cast<uint8_t*>(mapped);
      read.bytes = bytes;
      read.coherent = (vulkan_device_->memory_types().host_coherent >> type) & 0x1;
    }
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageOffset = {int32_t(dx), int32_t(dy), 0};
    copy.imageExtent = {width, height, 1};
    VkImage read_image = resolved.image.image;
    if (resolved.image.scale_q != kScaleQuartersNative) {
      // sa2-nx render resolution: the guest reads guest-size texels. The scaled rectangle is filtered down into a
      // guest-size staging image first (a box average at 2x, two taps at 1.5x), and that one is read back.
      Image* staging = StagingRead(width, height);
      if (!staging || !blit_) {
        Reject(16, "could not create a read-back staging image for a scaled resolved texture");
        return;
      }
      const VkRect2D host = HostRect(resolved.image, VkRect2D{{int32_t(dx), int32_t(dy)}, {width, height}});
      VkImageBlit down{};
      down.srcSubresource = down.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
      down.srcOffsets[0] = {host.offset.x, host.offset.y, 0};
      down.srcOffsets[1] = {host.offset.x + int32_t(host.extent.width), host.offset.y + int32_t(host.extent.height), 1};
      down.dstOffsets[1] = {int32_t(width), int32_t(height), 1};
      BarrierBeforeCopyResolve();
      blit_(commands_work_, resolved.image.image, VK_IMAGE_LAYOUT_GENERAL, staging->image, VK_IMAGE_LAYOUT_GENERAL, 1,
            &down, VK_FILTER_LINEAR);
      BarrierAfterCopyResolve();
      read_image = staging->image;
      copy.imageOffset = {0, 0, 0};
    }
    dfn_.vkCmdCopyImageToBuffer(commands_work_, read_image, VK_IMAGE_LAYOUT_GENERAL,
                                read.buffer, 1, &copy);
    pending_reads_.push_back({&read, reg.rb_copy_dest_base, x0, y0, width, height,
                                    reg.rb_copy_dest_pitch & 0x3FFF,
                                    (reg.rb_copy_dest_pitch >> 16) & 0x3FFF, reg.rb_copy_dest_info});
    if (done_reads_++ == 0) {
      REXLOG_INFO("[native] targets: read-back of resolved textures of up to {} texels (the "
                  "first: {:08X}, {}x{})",
                  max, reg.rb_copy_dest_base, width, height);
    }
  }

  // The submission finished: the submitted read-backs go to guest memory. Loaded with its fetch
  // constant (GpuSwap per word and R8G8B8A8), each texel must give the same channels as the resolved
  // texture on the GPU: (B, G, R, A) with copy_dest_swap and (R, G, B, A) without it.
  void WriteReads(std::vector<PendingRead>& reads) {
    const auto before_reads = std::chrono::steady_clock::now();
    if (before_reads - report_reads_ >= std::chrono::seconds(10)) {
      std::string list;
      for (auto it = reads_per_target_.begin(); it != reads_per_target_.end();) {
        TargetRead& d = it->second;
        if (!d.copies) {
          it = reads_per_target_.erase(it);  // has not appeared again
          continue;
        }
        list += fmt::format(" {:08X} {}x{} {}/{};", d.base, d.width, d.height, d.copies - d.skipped,
                             d.copies);
        d.copies = 0;
        d.skipped = 0;
        ++it;
      }
      if (!list.empty()) {
        SA2_REPORT_RING("[native] targets: read-backs per target (done/copies):{}", list);
      }
      report_reads_ = before_reads;
    }
    for (const PendingRead& p : reads) {
      const Read& read = *p.read;
      if (!read.data) {
        continue;
      }
      if (!read.coherent) {
        VkMappedMemoryRange range{};
        range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
        range.memory = read.memory;
        range.size = VK_WHOLE_SIZE;
        dfn_.vkInvalidateMappedMemoryRanges(device_, 1, &range);
      }
      const bool swap = (p.info >> 24) & 0x1;
      const uint32_t order_copy = p.info & 0x7;  // Endian128: 0-3 as xenos::Endian
      const auto order =
          order_copy <= 3 ? static_cast<xenos::Endian>(order_copy) : xenos::Endian::k8in32;
      // Each texel is a permutation of its 4 bytes (R/B swap plus GpuSwap): it is computed once with
      // marked bytes. Addresses come from the tile and the tiling table, and the guest's physical memory
      // is contiguous.
      const uint32_t permutation =
          xenos::GpuSwap(swap ? uint32_t(0x03000102) : uint32_t(0x03020100), order);
      const uint8_t b0 = uint8_t(permutation), b1 = uint8_t(permutation >> 8),
                    b2 = uint8_t(permutation >> 16), b3 = uint8_t(permutation >> 24);
      uint8_t* const physical = memory_->TranslatePhysical(0);
      const uint64_t base = uint64_t(p.base & 0x1FFFFFFF);
      const auto& table = TableTile2DTexel4();
      const uint64_t tiles_per_row = ((p.pitch + 31) & ~uint32_t(31)) >> 5;
      for (uint32_t j = 0; j < p.height; ++j) {
        const uint32_t ty = uint32_t(p.y0) + j;
        if (p.target_height && ty >= p.target_height) {
          break;
        }
        const uint64_t row = base + ((uint64_t(ty >> 5) * tiles_per_row) << 12);
        const uint16_t* const local = table.data() + size_t(ty & 31) * 32;
        const uint8_t* s = read.data + size_t(j) * p.width * 4;
        for (uint32_t i = 0; i < p.width; ++i, s += 4) {
          const uint32_t tx = uint32_t(p.x0) + i;
          if (tx >= p.pitch) {
            break;
          }
          const uint64_t address = row + (uint64_t(tx >> 5) << 12) + local[tx & 31];
          if (address + 4 > 0x20000000) {
            continue;
          }
          const uint8_t t0 = s[b0], t1 = s[b1], t2 = s[b2], t3 = s[b3];
          uint8_t* const d = physical + address;
          d[0] = t0;
          d[1] = t1;
          d[2] = t2;
          d[3] = t3;
        }
      }
    }
    reads.clear();
  }

  // sa2-nx render resolution: guest-size (unscaled) staging images for read-backs of scaled resolved textures,
  // one per size; recorded and read in the same slot order as the read-back buffers.
  std::unordered_map<uint64_t, Image> staging_reads_;
  Image* StagingRead(uint32_t width, uint32_t height) {
    auto [it, fresh] = staging_reads_.try_emplace((uint64_t(width) << 32) | height);
    if (fresh && !CreateScaled(it->second, width, height,
                               VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT, kColorFormat,
                               VK_SAMPLE_COUNT_1_BIT, kScaleQuartersNative)) {
      staging_reads_.erase(it);
      return nullptr;
    }
    Prepare(it->second);
    return it->second.prepared ? &it->second : nullptr;
  }

  void DestroyRead(Read& read) {
    if (read.memory != VK_NULL_HANDLE) {
      if (read.data) {
        dfn_.vkUnmapMemory(device_, read.memory);
      }
      dfn_.vkFreeMemory(device_, read.memory, nullptr);
    }
    if (read.buffer != VK_NULL_HANDLE) {
      dfn_.vkDestroyBuffer(device_, read.buffer, nullptr);
    }
    read = Read{};
  }

  // from_target = the source is a render target's image (or a retained one) with the front buffer in its
  // corner and larger than the output (sa2_native_lazy_front). Only requested with the exact variant.
  bool PaintOutput(VulkanPresenter::VulkanGuestOutputRefreshContext& context, Image& source,
                    uint32_t width, uint32_t height, bool from_target = false) {
    // With sa2_native_output_no_wait, the next of 3 slots, and it only waits if that one is still pending.
    // Without it, always slot 0: it waits for the previous output.
    const uint32_t s = output_no_wait_ ? (output_current_ + 1) % kSlotsOutput : 0;
    if (pending_outputs_[s]) {
      const auto before_wait = std::chrono::steady_clock::now();  // output-wait report
      dfn_.vkWaitForFences(device_, 1, &fences_output_[s], VK_TRUE, UINT64_MAX);
      ns_wait_output_ += Ns(before_wait, std::chrono::steady_clock::now());
      dfn_.vkResetFences(device_, 1, &fences_output_[s]);
      pending_outputs_[s] = false;
    }
    output_current_ = s;
    // The gamma ramp, if it changed since this slot's last output (the GPU is no longer reading it).
    if (ramp_gamma_ && ramps_output_[s].version != version_ramp_) {
      RampOutput& ramp = ramps_output_[s];
      std::memcpy(ramp.data, ramp_values_.data(), sizeof(ramp_values_));
      rex::ui::vulkan::util::FlushMappedMemoryRange(vulkan_device_, ramp.memory, ramp.type);
      ramp.version = version_ramp_;
    }
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
    for (const auto& f : framebuffers_) {
      if (f.framebuffer != VK_NULL_HANDLE && f.version == context.image_version()) {
        framebuffer = f.framebuffer;
      }
    }
    if (framebuffer == VK_NULL_HANDLE) {
      auto& f = framebuffers_[next_framebuffer_];
      next_framebuffer_ = (next_framebuffer_ + 1) % framebuffers_.size();
      if (f.framebuffer != VK_NULL_HANDLE) {
        WaitOutputs();  // a pending output could still use this framebuffer
        dfn_.vkDestroyFramebuffer(device_, f.framebuffer, nullptr);
        f.framebuffer = VK_NULL_HANDLE;
      }
      VkImageView view = context.image_view();
      VkFramebufferCreateInfo info{};
      info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
      info.renderPass = render_pass_output_;
      info.attachmentCount = 1;
      info.pAttachments = &view;
      info.width = width;
      info.height = height;
      info.layers = 1;
      if (dfn_.vkCreateFramebuffer(device_, &info, nullptr, &f.framebuffer) != VK_SUCCESS) {
        f.framebuffer = VK_NULL_HANDLE;
        return false;
      }
      f.version = context.image_version();
      framebuffer = f.framebuffer;
    }

    VkDescriptorImageInfo info_image{};
    info_image.imageView = source.view;
    info_image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptors_output_[s];
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &info_image;
    dfn_.vkUpdateDescriptorSets(device_, 1, &write, 0, nullptr);

    dfn_.vkResetCommandPool(device_, pools_output_[s], 0);
    VkCommandBufferBeginInfo start{};
    start.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    start.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (dfn_.vkBeginCommandBuffer(commands_output_[s], &start) != VK_SUCCESS) {
      return false;
    }
    VkRenderPassBeginInfo pass{};
    pass.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pass.renderPass = render_pass_output_;
    pass.framebuffer = framebuffer;
    pass.renderArea.extent = {width, height};
    if (from_target) {
      // That image was just written as a render target (or copy destination) in the frame's work, which goes in
      // another submission: make the output read it fully written.
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      dfn_.vkCmdPipelineBarrier(commands_output_[s],
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    dfn_.vkCmdBeginRenderPass(commands_output_[s], &pass, VK_SUBPASS_CONTENTS_INLINE);
    const VkViewport viewport{0.0f, 0.0f, float(width), float(height), 0.0f, 1.0f};
    dfn_.vkCmdSetViewport(commands_output_[s], 0, 1, &viewport);
    const VkRect2D scissor{{0, 0}, {width, height}};
    dfn_.vkCmdSetScissor(commands_output_[s], 0, 1, &scissor);
    VkPipeline pipeline = pipeline_;
    if (ramp_gamma_) {
      // With an output the size of the source (the normal case), the exact texel, unfiltered.
      // From a render target, the output has the front buffer's size, not the image's
      // (FrontOnPresent only requests it that way): texelFetch of the pixel, the same texel the texture
      // would hold.
      const bool exact = from_target || (width == HostWidth(source) && height == HostHeight(source));
      pipeline = PipelineRamp(exact ? 1 : 0);
      if (pipeline == VK_NULL_HANDLE) {
        pipeline = pipelines_ramp_[exact ? 1 : 0];  // requested variant missing: the usual one
      }
      if (!exact && !warned_bilinear_ramp_) {
        warned_bilinear_ramp_ = true;
        REXLOG_INFO("[native] targets: {}x{} output from a {}x{} source: ramp with bilinear sampling", width,
                    height, HostWidth(source), HostHeight(source));
      }
    }
    dfn_.vkCmdBindPipeline(commands_output_[s], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    dfn_.vkCmdBindDescriptorSets(commands_output_[s], VK_PIPELINE_BIND_POINT_GRAPHICS,
                                 layout_pipeline_, 0, 1, &descriptors_output_[s], 0, nullptr);
    const float rectangle[4] = {-1.0f, -1.0f, 2.0f, 2.0f};  // x, y, width, height in NDC
    dfn_.vkCmdPushConstants(commands_output_[s], layout_pipeline_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                            sizeof(rectangle), rectangle);
    struct {
      int32_t displacement[2];
      float inverse[2];
    } bilinear = {{0, 0}, {1.0f / float(width), 1.0f / float(height)}};
    dfn_.vkCmdPushConstants(commands_output_[s], layout_pipeline_, VK_SHADER_STAGE_FRAGMENT_BIT, 16,
                            sizeof(bilinear), &bilinear);
    dfn_.vkCmdDraw(commands_output_[s], 4, 1, 0, 0);
    dfn_.vkCmdEndRenderPass(commands_output_[s]);
    if (from_target) {
      // And nothing submitted later (the next frame draws and clears that image again) writes it before the
      // output has read it.
      dfn_.vkCmdPipelineBarrier(commands_output_[s], VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                nullptr, 0, nullptr, 0, nullptr);
    }
    if (dfn_.vkEndCommandBuffer(commands_output_[s]) != VK_SUCCESS) {
      return false;
    }
    VkSubmitInfo send{};
    send.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    send.commandBufferCount = 1;
    send.pCommandBuffers = &commands_output_[s];
    {
      const auto queue = vulkan_device_->AcquireQueue(family_, 0);
      if (dfn_.vkQueueSubmit(queue.queue(), 1, &send, fences_output_[s]) != VK_SUCCESS) {
        return false;
      }
    }
    pending_outputs_[s] = true;
    context.SetIs8bpc(true);
    return true;
  }

  struct Framebuffer {
    uint64_t version = UINT64_MAX;
    VkFramebuffer framebuffer = VK_NULL_HANDLE;
  };

  const VulkanDevice* vulkan_device_;
  const VulkanDevice::Functions& dfn_;
  VkDevice device_;
  rex::memory::Memory* memory_;
  uint32_t family_;

  std::array<VkCommandPool, kSlotsOutput> pools_output_{};  // output slots
  FnCopyImage copy_image_ = nullptr;
  uint32_t reductions_ = 0;  // resolves that had to shrink
  // Those of the slot being recorded (Record switches them).
  VkCommandBuffer commands_work_ = VK_NULL_HANDLE;
  VkCommandBuffer commands_upload_ = VK_NULL_HANDLE;
  // Three slots by default, up to four. With two, the CPU is only one frame ahead and waits for the GPU inside
  // Record() half the time. How many are actually used is set by sa2_native_slots_work.
  std::array<SlotWork, 4> slots_{};
  uint32_t used_slots_ = 3;  // set by the cvar (sa2_native_slots_work)
  uint32_t slot_ = 1;  // Record starts with slot 0
  uint64_t sends_ = 0;
  std::array<VkCommandBuffer, kSlotsOutput> commands_output_{};
  std::array<VkFence, kSlotsOutput> fences_output_{};
  bool recording_ = false;
  std::array<bool, kSlotsOutput> pending_outputs_{};
  uint32_t output_current_ = 0;
  bool output_no_wait_ = false;
  bool precise_marks_ = false;
  bool marks_categories_ = true;
  bool invalidate_each_copy_ = false;
  bool recording_upload_ = false;
  uint64_t generation_commands_ = 0;
  FnClearDepth clear_depth_ = nullptr;
  // GPU time of the submissions (timestamps, if the queue supports them).
  using FnWriteMark = void(VKAPI_PTR*)(VkCommandBuffer, VkPipelineStageFlags, VkQueryPool,
                                          uint32_t);
  using FnReadQueries = VkResult(VKAPI_PTR*)(VkDevice, VkQueryPool, uint32_t, uint32_t, size_t,
                                              void*, VkDeviceSize, VkQueryResultFlags);
  FnWriteMark write_mark_ = nullptr;
  FnReadQueries read_queries_ = nullptr;
  VkQueryPool queries_ = VK_NULL_HANDLE;
  // Pipeline statistics per pass category.
  VkQueryPool stats_ = VK_NULL_HANDLE;
  std::array<uint64_t, kGpuCategories> fragments_category_{};
  std::array<uint64_t, kGpuCategories> vertices_category_{};
  std::array<uint64_t, kGpuCategories> primitives_category_{};
  uint64_t stats_no_site_ = 0;
  VkQueryPool stats_draw_ = VK_NULL_HANDLE;
  // two timestamps per diagnostic draw (before / after), and GPU time, fragments and draws
  // per (category, pixel shader) of the diagnostic frames since the last report.
  VkQueryPool marks_draw_ = VK_NULL_HANDLE;
  struct CostShader {
    uint64_t ns = 0, fragments = 0, draws = 0, vertices = 0, primitives = 0;
    uint32_t vs = 0, draw_more_expensive_ns = 0;  // VS of the costliest draw
  };
  std::unordered_map<uint32_t, CostShader> cost_per_shader_;
  uint64_t cost_frames_ = 0;
  uint64_t stats_draw_no_site_ = 0;
  std::array<uint64_t, 3> no_swap_{};          // no clear, not whole, not possible
  uint32_t warnings_no_swap_ = 0;
  bool window_diagnostic_ = false;
  bool read_window_ = false;
  std::chrono::steady_clock::time_point last_window_ = std::chrono::steady_clock::now();
  double period_mark_ns_ = 0.0;
  uint64_t gpu_ns_ = 0;
  std::array<uint64_t, kGpuCategories> gpu_categories_ns_{};
  uint64_t mark_previous_end_ = 0;  // end timestamp of the last submission read (gap between submissions)
  // Present: how long it waits for the previous output.
  uint64_t ns_wait_output_ = 0;
  static uint64_t Ns(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point until) {
    return until > from ? uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(until - from).count()) : 0;
  }
  std::chrono::steady_clock::time_point last_swap_{};
  FnBlit blit_ = nullptr;
  // The game's shadow map, and the scale it is being drawn at (0 = not decided yet).
  static constexpr uint32_t kSideShadows = 1600;
  bool scalable_depth_ = false;
  uint32_t scale_shadows_ = 0;
  // Read-back of small resolved textures (sa2_native_resolved_read_texels).
  std::unordered_map<uint64_t, Read> reads_;
  std::unordered_map<uint64_t, TargetRead> reads_per_target_;  // target report and cadence
  std::chrono::steady_clock::time_point report_reads_{};
  std::vector<PendingRead> pending_reads_;  // recorded in the current submission
  uint64_t done_reads_ = 0;
  // Copy inventory (NoteCopy).
  uint64_t drawn_last_copy_ = 0;
  std::unordered_map<uint32_t, CopyTarget> copies_per_target_;
  std::chrono::steady_clock::time_point report_copies_{};
  VkFormat format_depth_ = VK_FORMAT_UNDEFINED;
  // Draw code: draws with the library shaders (nullptr without the required capabilities).
  std::unique_ptr<DrawsVulkan> draws_;

  std::unordered_map<uint64_t, Image> targets_;
  std::unordered_map<uint64_t, Image*> targets_color_active_;
  std::array<OwnerTileEDRAM, xenos::kEdramTileCount> owners_tiles_edram_{};
  std::array<OwnerTileEDRAM, kTilesEDRAM4> owners_tiles_edram4_{};
  std::unordered_map<VkImage, ViewsDepthEDRAM> views_depth_edram_;
  std::unordered_map<uint64_t, PassImportDepthEDRAM> passes_import_depth_edram_;
  std::unordered_map<VkImage, VkFramebuffer> framebuffers_import_depth_edram_;
  // sa2_native_conversion_frag: color <-> color conversions as one fragment pass per run.
  std::unordered_map<uint64_t, PassConversionColorFrag> passes_conv_color_frag_;  // format | variant << 32
  std::map<std::pair<VkImage, uint32_t>, VkFramebuffer> framebuffers_conv_color_frag_;  // (image, variant)
  std::unordered_map<VkImage, VkImageView> views_raw64_rt_edram_;  // UINT views as color attachments
  VkShaderModule fs_conv_r64_frag_ = VK_NULL_HANDLE;
  VkShaderModule fs_resolve_copy_ = VK_NULL_HANDLE;  // sa2_gpu_resolve_draw
  uint64_t resolve_draws_ = 0, resolve_draws_declined_ = 0, resolve_draws_report_ = 0, resolve_draws_declined_report_ = 0;
  // sa2_gpu_resolve_alias (the types are declared with MakeAlias).
  std::unordered_map<uint32_t, AliasResolve> aliases_;  // by the alias's address
  LastResolve last_resolve_;
  bool off_alias_ = false;
  uint64_t edram4_transfers_all_ = 0;
  uint64_t aliases_made_ = 0, aliases_dropped_ = 0, aliases_copied_ = 0, aliases_exchanged_ = 0;
  uint64_t aliases_made_report_ = 0, aliases_dropped_report_ = 0, aliases_copied_report_ = 0,
           aliases_exchanged_report_ = 0;
  VkDescriptorSetLayout layout_conv_color_frag_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_conv_color_frag_ = VK_NULL_HANDLE;
  VkShaderModule vs_conv_color_frag_ = VK_NULL_HANDLE;
  VkShaderModule fs_conv_color_frag_ = VK_NULL_HANDLE;
  bool conv_color_frag_failed_ = false;
  uint64_t conv_color_frag_uses_ = 0;
  std::array<uint64_t, 16> conv_color_frag_reasons_{};  // why the fragment path declined (return order in the function)
  uint64_t conv_color_frag_rejections_ = 0, conv_color_frag_uses_report_ = 0, conv_color_frag_rejections_report_ = 0;
  std::array<Image*, 5> edram4_draw_images_{};
  std::array<bool, 5> edram4_draw_writes_{};
  me::native::EdramBoundPlan edram4_draw_plan_{};
  VkRect2D edram4_draw_area_{};
  uint64_t edram4_exports_ = 0;
  uint64_t edram4_imports_ = 0;
  uint64_t edram4_stencil_preserves_ = 0;
  uint64_t edram4_capacity_rollovers_ = 0, edram4_capacity_rollover_failures_ = 0;
  std::unordered_map<std::string, uint32_t> edram4_rejections_per_class_;
  bool edram4_log_cap_ = false;
  bool edram4_operation_failed_ = false;
  bool import_depth_edram_failed_ = false, edram4_float_import_warned_ = false;
  std::unordered_set<uint64_t> overlaps_edram_recorded_;
  std::unordered_set<uint64_t> conversions_edram_recorded_;
  std::unordered_map<uint32_t, Resolved> resolved_;
  uint64_t revision_resolved_ = 0;
  std::deque<ResolvedClip> resolved_clips_;
  VkDeviceSize clips_bytes_ = 0;
  uint64_t clips_allocations_ = 0, clips_copies_ = 0, clips_cache_hits_ = 0;
  uint64_t clips_failures_ = 0;
  std::chrono::steady_clock::time_point clips_report_{};
  std::vector<ResolvedSleeping> resolved_sleeping_;
  VkDeviceSize resolved_sleeping_bytes_ = 0;
  uint64_t order_completed_work_ = 0;
  bool resolved_pool_device_error_ = false;
  bool resolved_pool_startup_eligible_ = false;
  uint64_t resolved_pool_hits_ = 0, resolved_pool_retirements_ = 0;
  uint64_t resolved_pool_evictions_ = 0, resolved_pool_waits_ = 0;
  uint64_t resolved_pool_allocations_ = 0, resolved_pool_ineligible_ = 0, resolved_pool_busy_ = 0;
  std::chrono::steady_clock::time_point resolved_pool_report_{};
  std::unordered_set<uint64_t> partial_recorded_;
  VkRect2D area_edram_{};
  uint32_t msaa_edram_actual_ = 0;
  // sa2-nx render resolution: the scale (quarters of the guest size, me_resolution_scale.h) new images are created at;
  // RescaleAll moves every existing image to a new one at a frame boundary (docked <-> handheld).
  uint32_t scale_q_ = kScaleQuartersNative;
  uint64_t rescales_ = 0;
  // sa2_msaa: the host sample count of the game's single-sample render targets (1 = none, the default), and the
  // resolve that copies one into a single-sample texture.
  uint32_t msaa_samples_ = 1;
  using FnResolve = void(VKAPI_PTR*)(VkCommandBuffer, VkImage, VkImageLayout, VkImage, VkImageLayout, uint32_t,
                                     const VkImageResolve*);
  FnResolve resolve_image_ = nullptr;
  uint64_t msaa_resolves_ = 0, msaa_depth_resolves_ = 0;
  bool depth_raster_grid_eligible_ = false;
  static constexpr uint32_t kDiagRegsColorInfo[4] = {0x2001, 0x2003, 0x2004, 0x2005};  // RB_COLOR_INFO, RB_COLORn_INFO
  std::unordered_map<uint64_t, Image> depths_;

  VkSampler sampler_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout_conversion_edram_ = VK_NULL_HANDLE;
  VkDescriptorSetLayout layout_conversion_depth_edram_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_conversion_depth_edram_ = VK_NULL_HANDLE;
  VkSampler sampler_depth_edram_ = VK_NULL_HANDLE;
  std::array<VkShaderModule, 4> shaders_conversion_depth_edram_{};
  std::array<VkPipeline, 4> pipelines_conversion_depth_edram_{};
  VkDescriptorSetLayout layout_import_depth_edram_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_import_depth_edram_ = VK_NULL_HANDLE;
  VkShaderModule vs_import_depth_edram_ = VK_NULL_HANDLE, fs_import_depth_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_depth_to_depth_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_depth_to_stencil_edram_ = VK_NULL_HANDLE, fs_color_to_stencil_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_stencil_msaa2_edram_ = VK_NULL_HANDLE, fs_stencil_msaa2_to_1x_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_depth_to_depth_msaa2_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_depth_msaa2_to_depth_1x_edram_ = VK_NULL_HANDLE;
  VkShaderModule shader_resolve_depth_msaa2_edram_ = VK_NULL_HANDLE;
  VkPipeline pipeline_resolve_depth_msaa2_edram_ = VK_NULL_HANDLE;
  VkShaderModule fs_raw64_to_depth_edram_ = VK_NULL_HANDLE;
  bool raw64_edram_supported_ = false;
  std::unordered_map<VkImage, VkImageView> views_raw64_edram_;
  VkPipelineLayout layout_pipeline_conversion_edram_ = VK_NULL_HANDLE;
  std::array<VkShaderModule, 9> shaders_conversion_edram_{};
  std::array<VkPipeline, 9> pipelines_conversion_edram_{};
  VkDescriptorSetLayout layout_descriptors_ = VK_NULL_HANDLE;
  VkPipelineLayout layout_pipeline_ = VK_NULL_HANDLE;
  VkDescriptorPool pool_descriptors_ = VK_NULL_HANDLE;
  std::array<VkDescriptorSet, kSlotsOutput> descriptors_output_{};
  VkShaderModule vs_ = VK_NULL_HANDLE;
  VkShaderModule fs_ = VK_NULL_HANDLE;
  VkRenderPass render_pass_output_ = VK_NULL_HANDLE;
  VkPipeline pipeline_ = VK_NULL_HANDLE;
  // Output with the game's gamma ramp (sa2_native_ramp_gamma).
  struct RampOutput {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* data = nullptr;
    uint32_t type = 0;
    uint64_t version = UINT64_MAX;  // the ramp_values_ version the buffer holds
  };
  bool ramp_gamma_ = false;
  std::array<RampOutput, kSlotsOutput> ramps_output_{};
  VkShaderModule fs_ramp_ = VK_NULL_HANDLE;
  // Index 0 bilinear (like guest_output_bilinear_ps), 1 exact texel (output the same size as the source).
  std::array<VkPipeline, 2> pipelines_ramp_{};
  std::array<bool, 2> pipelines_failed_ramp_{};
  bool warned_bilinear_ramp_ = false;
  // The game's ramp in 10 bits (red, green, blue). Until the game loads its own, the identity the SDK starts
  // with (i * 1023 / 255).
  std::array<std::array<uint16_t, 3>, 256> ramp_game_ = [] {
    std::array<std::array<uint16_t, 3>, 256> t{};
    for (uint32_t i = 0; i < 256; ++i) {
      t[i].fill(uint16_t(i * 0x3FF / 0xFF));
    }
    return t;
  }();
  // The output UBO in std140: one vec4 per entry (red, green and blue in 0-1, with the single-channel
  // post-processing applied), followed by the mix (saturation, vibrance, 1 / gamma) and effects (vignette,
  // scanlines). Filled by RecomputeRamp.
  static constexpr size_t kFloatRamp = 256 * 4 + 8;
  std::array<float, kFloatRamp> ramp_values_{};
  uint64_t version_ramp_ = 0;
  std::array<Framebuffer, VulkanPresenter::kMaxActiveGuestOutputImageVersions> framebuffers_{};
  size_t next_framebuffer_ = 0;

  uint64_t swaps_ = 0;    // resolves without a copy
  uint64_t restores_ = 0;
  uint64_t restores_for_resolver_ = 0;  // sa2_native_resolver_valid_content
  uint64_t resolver_no_source_ = 0;           // the content was in no texture
  uint64_t resolver_old_content_ = 0;      // with the setting off, resolves of the stale image
  uint64_t swaps_no_clear_ = 0;  // the ones that used to copy 1600x1600
  // Copies and clears removed, and what they cost.
  std::unordered_map<const Image*, TargetState> target_state_;
  uint64_t cleared_skipped_ = 0;             // colour
  uint64_t cleared_skipped_depth_ = 0;
  uint64_t cleared_skipped_pixels_ = 0;
  uint64_t cleared_useless_ = 0;             // clears wiped out by a swap with no draw in between
  uint64_t cleared_useless_pixels_ = 0;
  uint64_t clipped_restores_ = 0;     // restores that did not copy the whole image
  uint64_t restored_pixels_ = 0;           // the ones actually copied (so the report adds up)
  uint64_t pixels_restore_saved_ = 0;   // the ones not copied, due to useful area or swap
  // sa2_native_lazy_front (see PendingFront).
  std::unordered_map<uint32_t, PendingFront> pending_fronts_;  // per texture address
  std::vector<ImageFront> fronts_images_;                         // spares and retained
  std::unordered_map<uint32_t, uint64_t> presented_fronts_;  // address -> presented_ at its last Swap
  std::unordered_map<uint32_t, uint64_t> read_fronts_;       // address -> last sampling by a draw
  std::unordered_set<uint32_t> expired_fronts_;              // missing a copy that can no longer be made
  bool off_front_ = false;
  uint64_t postponed_front_ = 0;
  uint64_t painted_front_target_ = 0;
  uint64_t painted_retained_front_ = 0;
  uint64_t front_rotations_ = 0;
  uint64_t front_no_replenished_ = 0;
  uint64_t copied_front_read_ = 0;
  uint64_t copied_front_write_ = 0;
  uint64_t copied_front_swap_ = 0;
  uint64_t replaced_front_ = 0;
  uint64_t front_late_reads_ = 0;
  uint64_t front_saved_pixels_ = 0;
  uint64_t front_saved_previous_pixels_ = 0;
  static constexpr uint64_t kFrontFrames = 8;    // window for "the Swap presents it" and "nobody samples it"
  static constexpr size_t kFrontImagesMax = 3;     // spares of the front buffer's target (~4 MB each at 1040)
  // The values above as they were at the previous report: the new lines are per frame.
  uint64_t restored_previous_pixels_ = 0;
  uint64_t pixels_restore_saved_previous_ = 0;
  uint64_t cleared_skipped_previous_pixels_ = 0;
  uint64_t cleared_useless_previous_pixels_ = 0;
  uint64_t presented_report_copies_ = 0;
  // State of the previous frame, for the stutter dump.
  uint64_t hitch_copies_ = 0, cleared_hitch_ = 0, hitch_resolves_ = 0;
  uint64_t hitch_restores_ = 0, hitch_waits_ = 0;
  uint64_t hitch_ns_waits_gpu_ = 0, hitch_ns_wait_output_ = 0;
  uint64_t hitch_bytes_fingerprint_ = 0, hitch_postponed_fingerprints_ = 0;
  uint64_t hitch_ns_textures_ = 0;
  uint64_t hitch_textures_uploads_ = 0, hitch_bytes_uploaded_ = 0, hitch_created_textures_ = 0;
  uint64_t hitch_ns_raw_fingerprint_ = 0, hitch_ns_data_fingerprint_ = 0;
  uint64_t hitch_ns_waiting_copies_ = 0, hitch_waits_copies_ = 0;
  uint64_t hitch_ns_helping_copies_ = 0, hitch_helped_copies_ = 0;
  uint64_t hitch_ns_create_textures_ = 0, hitch_bound_textures_thread_ = 0;
  uint64_t hitch_ns_waiting_bindings_ = 0;
  uint64_t hitch_gpu_ns_ = 0;      // GPU timestamps when the previous frame closed
  uint64_t hitch_ns_record_ = 0;   // ns recording when the previous frame closed
  uint32_t warnings_hitch_ = 0;
  uint64_t copies_ = 0;
  uint64_t converted_copies_ = 0;  // resolves with a format conversion (blit)
  std::unordered_set<uint64_t> recorded_conversions_;
  uint64_t cleared_in_failed_pass_ = 0;  // clears through a pass that could not be opened
  uint64_t cleared_ = 0;
  uint64_t presented_ = 0;
  uint32_t changes_resolved_recorded_format_ = 0;  // ME: log of format/size changes at a resolve address
  uint64_t rejections_ = 0;
  std::unordered_set<uint32_t> warned_;
};

}  // namespace

std::unique_ptr<TargetsNative> TargetsNative::Create(const VulkanDevice* vulkan_device,
                                                        rex::memory::Memory* memory) {
  if (!vulkan_device || !memory) {
    return nullptr;
  }
  auto targets = std::make_unique<TargetsVulkan>(vulkan_device, memory);
  if (!targets->Initialize()) {
    REXLOG_ERROR("[native] targets: could not prepare the target presentation");
    return nullptr;
  }
  return targets;
}

}  // namespace sa2::native
