// banjo-nx (2026-10-17): see banjo_renut_tuning.h.
#include "renut_engine/gpu/backend/render_thread.h"
#include "renut_engine/gpu/backend/video.h"
#include <rex/frame_latency.h>
#include "renut_engine/display_sync.h"
#include "renut_engine/gpu/banjo_renut_tuning.h"
#include "renut_engine/gpu/banjo_renut_record.h"
#if defined(PLUME_DEKO3D)
#include "renut_engine/gpu/banjo_renut_deko3d.h"
#endif

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>

#include <rex/platform.h>

#if REX_PLATFORM_SWITCH
#include <pthread.h>
#endif

#include <plume_render_interface.h>
#if REX_PLATFORM_SWITCH
#include <plume_vulkan.h>  // the VkPipeline of a plume pipeline (BeforePipelineBind)
#endif

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/logging.h>

#include "cpu/banjo_enhancements.h"
#include "me/masseffect/banjo_pmull_fingerprint.h"
#include "me/me_platform_compat.h"  // RexSwitchPinCurrentThreadToCore and the priority helper (no-ops off the Switch)
#include "renut_engine/gpu/backend/backend.h"
#include "shader_cache.h"

// banjo-nx's own renderer defines these (me_native_system.cpp, masseffect_native_draws.cpp); they mean the same job
// here: the renderer's main thread and its background pipeline compiles.
REXCVAR_DECLARE(int32_t, banjo_ring_core);
REXCVAR_DECLARE(int32_t, banjo_background_compile_core);
REXCVAR_DECLARE(int32_t, banjo_prewarm_threads);
// banjo-nx's renderer's NVK keys (masseffect_native_draws.cpp), applied to Renut's renderer too (ControlNvkLocked).
REXCVAR_DECLARE(bool, banjo_gpu_nvk_draw_contract);
REXCVAR_DECLARE(bool, masseffect_native_nvk_emission);
REXCVAR_DECLARE(bool, masseffect_native_nvk_cbufs);
REXCVAR_DECLARE(bool, masseffect_native_nvk_dynamic);
REXCVAR_DECLARE(bool, masseffect_native_nvk_preload);
REXCVAR_DECLARE(int32_t, masseffect_native_nvk_measure);
REXCVAR_DECLARE(bool, masseffect_native_set4_differences);
// banjo-nx's renderer's robustness key (masseffect_native_draws.cpp), for Renut's pipelines too (package 27).
REXCVAR_DECLARE(bool, banjo_gpu_robustness_off);
REXCVAR_DECLARE(bool, banjo_gpu_bool_spec);  // the same key for Renut's pipelines (package 27)
REXCVAR_DECLARE(bool, banjo_gpu_blend_skip);  // likewise
REXCVAR_DECLARE(bool, banjo_gpu_tess_grid);   // likewise

// banjo-nx (2026-10-17, package 30): banjo_gpu_blend_skip under Renut's renderer, off. With it on, matched frames of
// the intro cutscene on the PC (Vulkan) blurred what the depth of field keeps sharp (Kazooie, the radio) where the
// emulated renderer does not; off, or the five big material shaders left out of it, or banjo_gpu_bool_spec off, they
// matched. On deko3d (the console, Ryujinx) the river water was missing with it on and drawn with it off. Why its
// discard is not the blend's no-op there is not found yet; the native renderer keeps banjo_gpu_blend_skip.
REXCVAR_DEFINE_BOOL(banjo_renut_blend_skip, false, "Banjo/Enhancements",
                    "Renut's renderer: banjo_gpu_blend_skip (off since package 30: it broke the depth of field with "
                    "banjo_gpu_bool_spec, and the deko3d renderer's river water)");

REXCVAR_DEFINE_BOOL(banjo_renut_half_pixel_offset, true, "Banjo/Enhancements",
                    "Renut's renderer: D3D9 pixel centres (PA_SU_VTX_CNTL.pix_center 0) move the viewport half a pixel, "
                    "as the GPU and Xenia do (before 2026-10-18 everything drew half a pixel up-left: the depth of "
                    "field's blur a whole 320x180 texel, the halo above Kazooie's crest)");

REXCVAR_DEFINE_BOOL(banjo_renut_point_minmax_radius, true, "Banjo/Enhancements",
                    "Renut's renderer: the point size limits (PA_SU_POINT_MINMAX) are radii, as the hardware and Xenia "
                    "read them (before 2026-10-18 they were taken as diameters: the depth of field's largest bokeh sprites "
                    "drew at half size)");

REXCVAR_DEFINE_BOOL(banjo_renut_depth_words_stencil, true, "Banjo/Enhancements",
                    "Renut's renderer: a depth resolve's guest words carry the stencil (they read 0 before 2026-10-18), "
                    "which the depth of field tests to keep the characters sharp");

REXCVAR_DEFINE_BOOL(banjo_renut_alpha_to_mask_test, true, "Banjo/Enhancements",
                    "Renut's renderer: alpha to coverage on a single-sample target is an alpha test at 0.5, as the "
                    "native renderer's banjo_native_alpha_to_mask_test (without it NVIDIA GPUs dither the grass, fur and "
                    "foliage edges in a checker pattern: the 'non-square pixels', 2026-10-18)");

REXCVAR_DEFINE_BOOL(banjo_gpu_depth_only, true, "Banjo/Enhancements",
                    "Renut's renderer (package 28; Unleashed / Marathon's depth-only pipelines): a draw that writes no "
                    "colour into a depth target, without the alpha test or alpha to coverage, whose pixel shader "
                    "neither kills pixels, writes depth nor exports to memory, is drawn without the shader (the "
                    "rasterizer writes the same depth and stencil); its constants and textures are not set up. Shadow "
                    "maps and depth passes: about half the draws. Same image");
REXCVAR_DEFINE_BOOL(banjo_gpu_output_trim, true, "Banjo/Enhancements",
                    "Renut's renderer (package 28, Vulkan; Unleashed / Marathon's trimmed pixel outputs): the colour "
                    "channels a pipeline never writes (write mask, the target's format, no target) are 0 from the "
                    "pixel shader's exits on, after the alpha test, through a specialization constant from the "
                    "pipeline key, so NAK drops what computed only them (the colour of alpha-tested shadow draws, the "
                    "channels of masked passes). The alpha stays when a written channel blends with it or alpha to "
                    "coverage is on. No new pipelines (the key has the mask already). Same image");
REXCVAR_DEFINE_BOOL(banjo_gpu_depth_words_view, true, "Banjo/Enhancements",
                    "Renut's renderer (package 28; fewer resolve copies): a texture over a depth resolve's memory (the "
                    "guest words, an 8_8_8_8 fetch; the game reads nearly every depth resolve so) samples a view of "
                    "the resolve's words texture instead of a copy of the whole texture made at its first bind after "
                    "each resolve (a 4-byte read and write a texel). The same bytes: every draw binds its textures "
                    "again, so the draws after a resolve read its words either way. Off in packages 28b-32 (the depth of field "
                    "was broken by other causes then: the blend skip, the stencil, the half pixel); on again since "
                    "2026-10-18 with the depth of field matching the emulated renderer");
REXCVAR_DEFINE_BOOL(banjo_deko3d_draw_resolves, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): single-sampled colour resolves are drawn (a texel copy into the "
                    "destination as a render target) instead of copied on the copy engine, which needs the whole GPU "
                    "idle before and after it. Same bits");
REXCVAR_DEFINE_BOOL(banjo_deko3d_depth_compression, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): depth/stencil targets get the GPU's hardware compression, as the "
                    "colour targets have (the main pass reads and writes far less depth memory). Same pixels");
// sa2-nx (2026-10-20): off by default. With it, SA2's draws in Ryujinx sampled every texture as if at its smallest mip
// (each surface one flat colour: Sonic's quills beige, the roads grey) while the generic variant (the function read
// at run time) is right; bisected over tools/deko3d_glsl_opt.py's passes: the alpha pass alone gives it, and only
// with this key on. uam's intermediate code for both variants reads the same (the texture fetch first), so the
// difference is in the Maxwell code or in how the emulator runs it: to be checked on the console before it is on.
REXCVAR_DEFINE_BOOL(banjo_deko3d_alpha_spec, false, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): the alpha test's function is part of the pipeline (specialization "
                    "constant 903 of the deko3d shaders): a draw without the alpha test has no compare and no discard "
                    "in its pixel shader, and the GPU tests depth before shading. Same pixels. sa2-nx: off by default "
                    "(SA2's textures sampled wrong with it in Ryujinx; docs/renut-renderer.md)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_tess_spec, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): a pipeline that is not a tessellated draw's gives its vertex shader "
                    "the patch kind 0 as specialization constant 904: the tessellation code goes. No new pipelines. "
                    "Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_vs_outputs, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): a pipeline's vertex shader writes only the outputs its pixel shader "
                    "reads (none for depth-only draws: the shadow maps), specialization constant 905: the others and "
                    "what computes them go. Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_uam_prmt, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): uam turns the shaders' byte swaps (the big-endian words the vertex "
                    "fetch reads from memory: bones, skinned vertices) into Maxwell's PRMT, one instruction each. Off: "
                    "the shifts and masks as before, with a DKSH cache of their own (deko3d_shaders_f446.bin, "
                    "compiled by the console as it goes). Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_uam_gvn, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): uam's value numbering over the dominator tree: what a shader computes "
                    "again later (the texture fetch of the early alpha test, repeated by the main body; equivalent "
                    "if / else branches) is reused, where it does not cost the pixel shader a step of 8 registers. "
                    "Off: a DKSH cache of its own (deko3d_shaders_f<flags>.bin, compiled by the console as it goes). "
                    "Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_uam_uniform_branches, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): uam leaves out the reconvergence point (SSY / SYNC) of an if / else "
                    "whose condition comes from constant buffers only (the same for every thread: it cannot diverge). "
                    "Off: a DKSH cache of its own (deko3d_shaders_f<flags>.bin, compiled by the console as it goes). "
                    "Same pixels");
REXCVAR_DEFINE_BOOL(banjo_renut_stall_present, false, "Banjo/Enhancements",
                    "The colour flash (2026-10-19): while the game stalls (loading) the overlay pump re-presents the "
                    "last front buffer every 16 ms (true, as before). Off: nothing is presented then unless a menu or "
                    "notification is up (the display keeps the last frame, as the console does), and a front buffer "
                    "no resolve wrote is never shown");
REXCVAR_DEFINE_BOOL(banjo_renut_device_swizzle, true, "Banjo/Enhancements",
                    "The colour flash (2026-10-20): a texture whose header in guest memory describes the same data as "
                    "the device's fetch constant but differs in the fields SetTexture copies unchanged (signs, pitch, "
                    "tiling, swizzle, number format, exponent adjust, mip address) is sampled through the device's "
                    "fetch constant (what the GPU reads): the header is a temporary the game has rewritten (RenderDoc "
                    "capture of Ryujinx: a bloom level read through the swizzle 1,1,0,0 turned the whole frame "
                    "yellow). The same pixels whenever the header is current. Off: the header's, as before");
REXCVAR_DEFINE_INT32(banjo_renut_portrait_fur_shells, 1, "Banjo/Enhancements",
                     "NOT exact, the user's call (2026-10-20): banjo_renut_fur_shells for the fur drawn into targets of "
                     "128 x 128 or less (the dialogue portrait, 96 x 96, where 21 shells cost ~2.5 ms of the heaviest "
                     "frames for a few thousand pixels). 1 every shell; 3 draws 7 of 21")
    .range(1, 7);
REXCVAR_DEFINE_INT32(banjo_renut_fur_shells, 1, "Banjo/Enhancements",
                     "NOT exact, the user's call (2026-10-20): 1 draws every fur shell (as the game). N > 1: of each fur "
                     "part's draws in a frame (the game draws every part of the body once a shell; a part = the same "
                     "shaders and index range; the fur family of tools/nullps_shaders.txt), every Nth is drawn, the "
                     "first always: 2 draws 11 of Banjo's 21 shells (thinner fur, the same length). The fur is about "
                     "18 ms of the heaviest frame's GPU at the stock clock")
    .range(1, 4);
REXCVAR_DEFINE_BOOL(banjo_deko3d_uam_wide_rows, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-20, GPU): uam reads each bone row of tools/deko3d_glsl_opt.py's aw4 view "
                    "(a uvec4 of storage buffer 15) with one 128-bit load instead of four 32-bit ones (banjo flag 1024): "
                    "the fur vertex shader 48 -> 12 loads, 596 -> 560 instructions; the console's profile shows those "
                    "vertex shaders memory-bound. Same words, same pixels. Aligned by construction: the fetch data's "
                    "storage binding starts on 256 bytes (render/draw.cpp) and a row is element n of 16 bytes. Off: the "
                    "32-bit loads, with a DKSH cache of its own (deko3d_shaders_f447.bin)");
REXCVAR_DEFINE_BOOL(banjo_renut_diag_null_ps, false, "Banjo/Diagnostics",
                    "Diagnostic (2026-10-20): the fur family's pixel shaders (tools/nullps_shaders.txt) discard at once, so the GPU profile shows their draws' vertex and rasterisation cost alone. The fur vanishes: a measurement, not a picture (configs/banjo_renut_diag_fur_vertex_only.toml)");
// sa2-nx (2026-10-20): on by default in SA2 (the user's call: Banjo's latency set).
REXCVAR_DEFINE_BOOL(banjo_deko3d_eager_kicks, true, "Banjo/Enhancements",
                    "Input latency A/B (2026-10-20): the frame goes to the GPU in more, earlier parts: a command list ends at a game kick once it holds 100 draws (not 200, at most 6 a frame, not 3) and goes to the GPU at that kick (its upload memory cleaned first), not at the next one. At the frame cap the GPU then works while the render thread is still replaying the frame instead of mostly after the present: the frame is done sooner after its controller read (the [latency] line's \"input -> GPU done\"). The same commands in the same order (configs/banjo_deko3d_eager_kicks_ab.toml; in-run A/B target 18)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_skin_once, false, "Banjo/Enhancements",
                    "GPU A/B (2026-10-20): the fur's skinned meshes (tools/skinonce_shaders.txt) skinned once a frame: a part's first draw of the frame writes each vertex's three skinned matrix rows to a buffer (constant 913, shared word v[84].z = 1), its other shells read them there (= 2) instead of loading 12 bone rows and blending them (the vertex shader about half as long). Only where every input of the skinning is the same draw for draw (shaders, index and vertex buffers, bone data and its constants); the same rows, bit for bit, so the same pixels (configs/banjo_deko3d_skin_once_on.toml; in-run A/B target 17)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_light_branch, false, "Banjo/Enhancements",
                    "GPU A/B (2026-10-20): in the heaviest lit pixel shaders (tools/lightbranch_shaders.txt) each point light's work runs in a branch on its attenuation test, so warps whose pixels the light does not reach skip it (a light is about 35 instructions); where every light reaches, the branches cost a few instructions. The same pixels; whether it is faster is the console's to say (configs/banjo_deko3d_light_branch_on.toml)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_gpr32, false, "Banjo/Enhancements",
                    "GPU A/B (2026-10-19): the pixel shaders of 33-40 GPRs (the fur family, the top of the GPU profile) "
                    "compiled with 32 at most, so the GPU keeps every warp slot (64 instead of about 48) to hide texture "
                    "latency, at the price of a few spills (the fur shader: 15 loads, 5 stores). The same pixels; "
                    "whether it is faster is the console's to say (configs/banjo_deko3d_gpr32_on.toml)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_early_z, true, "Banjo/Enhancements",
                    "GPU (2026-10-19): a pixel shader that discards (alpha test) in a draw that writes neither depth "
                    "nor stencil (the blended fur shells, particles) gets its depth test before it "
                    "(layout(early_fragment_tests), constant 909): the pixels the depth test rejects no longer run it. "
                    "The same pixels: with no depth or stencil write the discard has nothing to undo");
REXCVAR_DEFINE_BOOL(banjo_deko3d_txq_constants, true, "Banjo/Enhancements",
                    "GPU (2026-10-19): a pixel shader's texture size queries of level 0 read the size from a buffer "
                    "plume deko3d fills from the views it binds, instead of a texture instruction per pixel (A5E2, the "
                    "full-screen pass second in the GPU profile, ran 4 fetches and 4 queries a pixel; the fur shells' "
                    "stacked textures 1). Specialization constant 908 (tools/deko3d_glsl_opt.py, pass txq)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_txq_verify, false, "Banjo/Enhancements",
                    "Test of banjo_deko3d_txq_constants: the shaders run the query as before and paint the pixel "
                    "magenta where the buffer's size differs (constant 908 = 2; the variants compile at run time)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_early_hoist, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): in a pixel shader with Renut's early alpha test, the main body's first "
                    "texture region (the fur's texture-array or 2D fetch) goes ahead of the test, so uam drops the "
                    "test's copy of it (specialization constant 907; the fur shader 4 texture fetches fewer, 2 more "
                    "registers). Same pixels");
REXCVAR_DEFINE_BOOL(banjo_renut_depth_words_lazy, true, "Banjo/Enhancements",
                    "Renut's renderer (2026-10-19): a depth resolve writes its words target (the 8_8_8_8 view of the "
                    "resolved memory, depth and stencil as bytes) only for destinations something reads that way (on "
                    "the emulator route: the depth of field's, 1 of 5 resolves a frame; the shadow maps' never), and "
                    "in the first 300 resolves of each. A read of words not written is served by writing them then "
                    "from the unchanged source; the 10 s depth resolve line counts both. Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_depth_handover, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, GPU): a depth resolve of a whole depth target into a texture of its "
                    "size that needs no words is not drawn: the texture is sampled from the target's image until the "
                    "target is bound again (then the resolve is drawn after all), and at the target's next whole clear "
                    "the texture takes that image and the target another. The shadow maps, resolved and cleared once a "
                    "frame, are never copied. The float a resolve writes is the depth it loads: same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_depth24, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, GPU): a D24S8 depth target (the shadow maps, small render-to-texture "
                    "targets) as 24-bit unorm depth with stencil, the Xenos's own format, instead of 32-bit float: half "
                    "the bytes written by its draws and read by its resolves and samples. Its depth is then the 24-bit "
                    "value the Xbox 360 keeps (the float target held more bits than the console). The 20e4 float depth "
                    "(D24FS8) stays 32-bit float");
REXCVAR_DEFINE_BOOL(banjo_deko3d_push_constants, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, render-thread proposals 2 and 5): a draw's float constants and "
                    "SharedConstants go into three fixed uniform buffers as pipelined updates (dkCmdBufPushConstants) "
                    "of the 64-byte groups that differ from what the GPU holds, instead of a new ring allocation of "
                    "the whole set per draw (4,352 and 1,344 bytes), cleaned at the submit and bound again. Same values "
                    "for every register a shader reads");
REXCVAR_DEFINE_BOOL(banjo_deko3d_push_constants_verify, false, "Banjo/Diagnostics",
                    "banjo_deko3d_push_constants: after every draw's updates, checks that the model of the uniform "
                    "buffers holds the guest's value of every register the shaders read (and their literals), and "
                    "logs a draw where it does not (a test: slower)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_kick_submit, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, input latency): at the game's kicks (several a frame) the command "
                    "list recorded so far is ended and handed to the GPU at the next kick, instead of the whole frame "
                    "at the present: the GPU starts a frame's work while the render thread still records the rest, so "
                    "the frame (and the controller input in it) reaches the screen sooner. At most 3 more lists a "
                    "frame, each of at least 200 draws. Same commands in the same order");
REXCVAR_DEFINE_BOOL(banjo_deko3d_latency_trim, false, "Banjo/Enhancements",
                    "Input latency (2026-10-20, deko3d, no vsync): where the GPU is the limit a recorded frame waited in "
                    "the GPU's queue behind the one running (the GPU's frame time less the recording time: 30 ms and "
                    "more in the heaviest scenes) and the game had read its controller that much earlier than needed. "
                    "On: after each present the render thread waits until the GPU has finished all but the frame's last "
                    "part (a fence at each kick submission), so the next frame is recorded, and the game thread "
                    "released, just in time; if the GPU then runs dry it stops for 60 frames. The [latency] line's "
                    "\"input -> GPU done\" and the \"latency trim\" line say what it does. Same pixels");
REXCVAR_DEFINE_BOOL(banjo_late_input, true, "Banjo/Enhancements",
                    "Input latency (2026-10-19): when the render thread is the slower one, the game's frame waits at "
                    "its Swap for the frame before; its controller read is delayed by the least such wait of the last "
                    "16 frames, less 2 ms, so the frame starts later with newer input and still reaches the render "
                    "thread in time (backend/render_thread.cpp). No wait (the game thread or the frame cap the limit): "
                    "no delay. The [latency] line every 10 s gives input -> swap -> present");
REXCVAR_DEFINE_BOOL(banjo_deko3d_ring_cleaner, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, render-thread proposal 23): the upload ring's cache clean (what the "
                    "CPU wrote, written out for the GPU before each submit) done at each kick by a thread on the "
                    "background core instead of by the render thread at the submit; a range the thread has not "
                    "started when the submit needs it is cleaned by the render thread. Same memory for the GPU");
REXCVAR_DEFINE_BOOL(banjo_renut_store_cards, true, "Banjo/Enhancements",
                    "Renut's renderer (2026-10-19): cached vertex and index copies proven by the store cards (every "
                    "guest store outside the stack marks its 4 KB page, tools/store_cards.py) instead of hashing them "
                    "at their first use after every kick; a page written since is hashed as before. Only with "
                    "banjo_renut_exact_streams, and only when the generated code marks the cards. The same pixels");
REXCVAR_DEFINE_BOOL(banjo_renut_store_cards_verify, false, "Banjo/Diagnostics",
                    "PC test of banjo_renut_store_cards: every copy the cards prove is hashed as well, and a copy whose "
                    "bytes changed under clean cards is logged (a writer of guest memory that marks none) and hashed");
REXCVAR_DEFINE_BOOL(banjo_renut_cpu_memos, true, "Banjo/Enhancements",
                    "Renut's renderer (2026-10-19): the render thread's per-draw lookups memoised (render targets, "
                    "texture bindings, vertex streams, memexport checks), each keyed by everything its lookup reads and "
                    "an epoch bumped by every change of what it reads. The same results");
REXCVAR_DEFINE_BOOL(banjo_renut_pipeline_memo, true, "Banjo/Enhancements",
                    "Renut's renderer (2026-10-19): a 64-entry memo of the pipeline each key gave (built, or the generic "
                    "one a specialisation draws with while pending or after failing), before the key's XXH3 and the "
                    "map. The same pipelines");
REXCVAR_DEFINE_BOOL(banjo_renut_resolve_skip, true, "Banjo/Enhancements",
                    "Renut's renderer (2026-10-19): a resolve the same as the last one into its texture (same source, "
                    "rectangle and place) whose source nothing has written since (no draw that can write it, no clear) "
                    "is not done again: the texture already holds it. The count is in the 10 s depth resolve line. Same "
                    "pixels");
REXCVAR_DEFINE_INT32(banjo_deko3d_compile_priority, 0x3E, "Banjo/Enhancements",
                     "deko3d renderer (2026-10-18): the uam shader compile thread's priority (Horizon: higher is lower; "
                     "the game's threads are 0x3B, the render thread 0x2C), on banjo_background_compile_core; a thread "
                     "waiting for a compile lends its priority. 0: as created (any core)");
REXCVAR_DEFINE_BOOL(banjo_deko3d_barrier_merge, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, GPU): barriers recorded one after another with no GPU work between "
                    "them are emitted as one (the strongest wait, every invalidation) right before the next command "
                    "that does work: one pipeline drain instead of several. Same pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_barrier_lean, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-19, GPU): a transition into a layout that is only written next keeps "
                    "the texture cache (only a read after a write needs it invalidated, and that read's transition "
                    "does it), and a texture going from one read layout to another needs no barrier at all. Same "
                    "pixels");
REXCVAR_DEFINE_BOOL(banjo_deko3d_zcull, true, "Banjo/Enhancements",
                    "deko3d renderer (2026-10-18): hierarchical depth culling (Zcull) on the queue: tiles a draw cannot "
                    "pass the depth test in are skipped before shading. Same pixels");
REXCVAR_DEFINE_BOOL(banjo_renut_cache_save_quiet, true, "Banjo/Enhancements",
                    "Renut's renderer (package 28; as Unleashed saves its pipeline cache at loading screens): the "
                    "pipeline cache is saved once no pipeline was built for 15 s (and at most once a minute; after "
                    "5 minutes of steady compiling regardless), not every minute with new pipelines: the save holds "
                    "the driver's cache lock, which a compile on the render thread waits for. Same pipelines");

#if REX_PLATFORM_SWITCH
// The two contracts with the linked NVK (mesa-switch-main-nfsmw-source: nvk_cmd_buffer.h, struct nvk_switch_dibujo and
// struct nvk_switch_set4), field for field as banjo-nx's renderer declares them (masseffect_native_draws.cpp; the same
// definitions, so one type each across the program). Weak: an NVK without them leaves the addresses null.
extern "C" {
struct NvkSwitchSet4Counts {
  uint64_t bindings_difference;
  uint64_t complete_bindings;
  uint64_t writes_root;
  uint64_t dwords_root;
  uint64_t cbufs_dirty;
  uint64_t cbufs_saved;
  uint64_t draws;
  uint64_t checks;
  uint64_t differences;
};
struct NvkSwitchSet4 {
  int32_t version;
  int32_t request;
  int32_t environment;
  int32_t off;
  NvkSwitchSet4Counts total;
};
extern NvkSwitchSet4 nvk_switch_set4 __attribute__((weak));
struct NvkSwPart {
  uint64_t times;
  uint64_t ticks;  // CNTPCT_EL0, a ticks_per_second
};
struct NvkSwImprovement {
  int32_t request;  // written by the app: 1 yes, 0 no
  int32_t off;  // 1: its guard saw a MISMATCH
  uint64_t uses;
  uint64_t validated;
  uint64_t differences;
  uint64_t no_check;
};
struct NvkSwitchDraw {
  int32_t version;
  int32_t measure;
  int32_t measure_failures;
  int32_t environment;
  uint64_t ticks_per_second;
  NvkSwPart parts[16];   // enum nvk_sw_part
  uint64_t counts[13];    // enum nvk_sw_count
  NvkSwImprovement improvements[5];  // emission, cbufs, dynamic, set4_fast (not applied), prefetch
};
extern NvkSwitchDraw nvk_switch_draw __attribute__((weak));
extern NvkSwitchDraw nvk_switch_dibujo __attribute__((weak));
void vk_switch_preload_pipeline(VkPipeline pipeline) __attribute__((weak));
}
static_assert(sizeof(NvkSwitchSet4) == 16 + 9 * 8, "NvkSwitchSet4 must have the same size as in NVK");
static_assert(sizeof(NvkSwitchDraw) == 24 + 16 * 16 + 13 * 8 + 5 * 40, "NvkSwitchDraw: same size as in NVK");
#endif

REXCVAR_DEFINE_BOOL(banjo_renut_pipeline_cache, true, "Banjo/Enhancements",
                    "Renut's renderer (Vulkan): the driver's compiled pipelines persist in cache/renut_pipelines.bin "
                    "(plume's pipeline cache), so the next start builds the listed pipelines from the file. Saved "
                    "when new pipelines were compiled, at most once a minute, and at exit. Same pipelines");
REXCVAR_DEFINE_BOOL(banjo_renut_pmull_hash, true, "Banjo/Enhancements",
                    "Renut's renderer, ARM only: its per-frame content hashes of guest data (vertex streams, index "
                    "ranges, texture samples) use banjo-nx's PMULL-folded 64-bit CRC instead of XXH3, half the "
                    "instructions per byte on the A57. Checked against the bit-by-bit reference at start");
REXCVAR_DEFINE_BOOL(banjo_renut_upload_cached, true, "Banjo/Enhancements",
                    "Renut's renderer, the Switch only: its upload ring (every draw's constants, converted vertices "
                    "and indices, texture uploads) in CPU-cached memory, flushed before each submission, instead of "
                    "write-combined memory: the A57 wrote 5,251 MB/s there against 2,506 (sa2-nx, banjo-nx's "
                    "masseffect_native_upload_memory = 1). On the PC's discrete GPU cached memory is slower to read, "
                    "so the PC keeps write-combined. Same bytes");
REXCVAR_DEFINE_INT32(banjo_renut_ab_target, 0, "Banjo",
                     "Renut's renderer, in-run A/B: 0 none; 1 banjo_renut_precise_barriers; 2 the render thread's pin "
                     "to banjo_ring_core (the Switch); 3 banjo_renut_shared_dedupe; 4 banjo_renut_push_point_only; "
                     "5 banjo_renut_capture_pool; 6 banjo_renut_exact_streams (what the exact rule costs); 7 "
                     "banjo_gpu_bool_spec (the generic pipelines in the off half); 8 banjo_gpu_blend_skip; 9 "
                     "banjo_gpu_tess_grid; 10 none (was banjo_renut_check_worker, removed in package 28b); 11 banjo_gpu_depth_only; 12 banjo_gpu_output_trim; 13 "
                     "banjo_gpu_depth_words_view; 14 banjo_renut_record_worker; 15 banjo_deko3d_light_branch, 16 "
                     "banjo_deko3d_gpr32 and 17 banjo_deko3d_skin_once (2026-10-20: the key on; the off half's "
                     "pipelines are built without its constant); 18 banjo_deko3d_eager_kicks (19, banjo_late_input_gpu, "
                     "removed). The part is "
                     "switched off and on every banjo_renut_ab_s seconds; the \"[renut A/B]\" line every 10 s gives "
                     "both halves' frame rate, render thread and GPU time")
    .range(0, 18);
REXCVAR_DEFINE_INT32(banjo_renut_ab_s, 4, "Banjo", "Renut's renderer: the half-period of banjo_renut_ab_target, in seconds")
    .range(1, 60);
REXCVAR_DEFINE_BOOL(banjo_renut_push_point_only, true, "Banjo/Enhancements",
                    "Renut's renderer (Vulkan): with the constants in uniform buffers, the push constant (the three "
                    "constant addresses) is read only by the point sprites' geometry shader, so only point-list draws "
                    "push it, not every draw (NVK: a root-table upload a draw less). Same output");
REXCVAR_DEFINE_BOOL(banjo_renut_capture_pool, true, "Banjo/Enhancements",
                    "Renut's renderer: a queued draw's capture (its 12.7 KB device snapshot and the closure the render "
                    "thread runs) comes from a pool and goes back to it, instead of three heap allocations per draw on "
                    "the game thread freed on the render thread (the Switch's malloc has one lock). Same data");
REXCVAR_DEFINE_BOOL(banjo_renut_neon_swap, true, "Banjo/Enhancements",
                    "Renut's renderer, ARM only: the vertex conversion's byte swaps (guest big-endian words to the "
                    "host's) 16 bytes per NEON byte reverse instead of a word at a time. Same bytes");
REXCVAR_DEFINE_BOOL(banjo_renut_shared_dedupe, true, "Banjo/Enhancements",
                    "Renut's renderer: a draw whose SharedConstants (texture and sampler slots, booleans, render "
                    "state; 1,344 bytes) equal the previous draw's on the same command list reuses that upload: no "
                    "copy into upload memory and no descriptor set bind (NVK: a root-table update). Exact compare");
REXCVAR_DEFINE_BOOL(banjo_renut_queue_notify_idle, true, "Banjo/Enhancements",
                    "Renut's renderer: work queued for the render thread (every draw) wakes it only while it waits, "
                    "instead of a condition-variable notify per item (a kernel call on the Switch, about 1,500 a frame "
                    "from the game threads). The render thread checks the queue under the same lock before it sleeps");
REXCVAR_DEFINE_BOOL(banjo_renut_linked_stages, true, "Banjo/Enhancements",
                    "Renut's renderer, the Switch: NVK compiles each pipeline's stages together (its default), not "
                    "with the per-stage keys of masseffect_nvk_cache_per_stage (disable_lto), so the vertex outputs the "
                    "pixel shader does not read are dropped: Renut's vertex shaders write every output, zeros for the "
                    "unused ones. More compiling, kept by banjo_renut_pipeline_cache. Same image");
REXCVAR_DEFINE_BOOL(banjo_renut_exact_streams, true, "Banjo",
                    "Renut's renderer: a cached copy of a vertex stream or index range is hashed whole on its first use "
                    "after each kick (where the game may learn the GPU is done with a buffer and rewrite it) and "
                    "converted again if its guest bytes changed, instead of Renut's check on a background thread once "
                    "every 4 frames, after which a rewritten range could draw from its old copy for about 5 frames. "
                    "The exact rule (banjo-nx's renderer proves its persistent vertex blocks the same way); not under "
                    "a master switch");
#if REX_PLATFORM_SWITCH
constexpr bool kMemoryPolicyDefault = true;
#else
constexpr bool kMemoryPolicyDefault = false;
#endif
REXCVAR_DEFINE_BOOL(banjo_renut_memory_policy, kMemoryPolicyDefault, "Banjo",
                    "Renut's renderer: its memory limits sized for the console (on by default on the Switch, where NVK "
                    "gives about 1,280 MB of GPU memory budget; Renut sized them for a PC): idle host textures released "
                    "below banjo_renut_textures_reserve_mb of budget left, the stream cache capped at "
                    "banjo_renut_stream_cache_mb and emptied of copies idle banjo_renut_stream_cache_idle_s, upload ring "
                    "chunks of a heavy frame released after banjo_renut_upload_trim_frames. The same picture: a released "
                    "texture or copy is made again from guest memory. Not under a master switch");
REXCVAR_DEFINE_INT32(banjo_renut_textures_reserve_mb, 192, "Banjo",
                     "banjo_renut_memory_policy: with less than this much of the GPU memory budget left, or the "
                     "bindless table three quarters full (checked every 32 frames), the host textures not bound for "
                     "120 frames are released (made again from guest memory when bound). Never a resolve destination "
                     "(banjo_renut_keep_gpu_written). 0: Renut's rule only (85 % of the budget, 600 frames idle, "
                     "checked every 120)")
    .range(0, 16384);
REXCVAR_DEFINE_BOOL(banjo_renut_keep_gpu_written, true, "Banjo",
                    "Renut's renderer: a host texture the GPU wrote (a resolve destination; Renut writes nothing back "
                    "to guest memory) is never released by an eviction rule, as it could not be made again from guest "
                    "memory (Renut's backstop released them after 3,600 idle frames). Replaced as before when the game "
                    "frees or reuses their memory. The exact rule: not under a master switch");
REXCVAR_DEFINE_BOOL(banjo_renut_diag_gpu_shaders, false, "Banjo/Diagnostics",
                    "Renut's renderer, a measurement (package 28b): renut_gpu_profile's passes split by pixel shader "
                    "and blending (a timestamp where the shader changes), with draws a frame, 60 lines: the main "
                    "1280x720 pass broken down in ms. The timestamps cost some GPU and render thread time");
REXCVAR_DEFINE_BOOL(banjo_renut_diag_trim_poison, false, "Banjo/Diagnostics",
                    "Renut's renderer, a test of banjo_gpu_output_trim (package 28): the channels it trims are 60000 "
                    "instead of 0, so a channel trimmed wrongly shows; the image must be the same as without the test");
REXCVAR_DEFINE_BOOL(banjo_renut_diag_resolve_trace, false, "Banjo/Diagnostics",
                    "Renut's renderer, a measurement (package 28): one frame every 30 s, every resolve (source and "
                    "destination), clear, change of render targets (with its draws) and first sample of a resolved "
                    "texture, as \"[trace]\" lines");
REXCVAR_DEFINE_INT32(banjo_renut_diag_build_delay_ms, 0, "Banjo/Diagnostics",
                     "Renut's renderer, a test (package 27c): every pipeline build waits this long first, on whichever "
                     "thread builds it, so the PC compiles as slowly as the console (the one-frame flashes seen there "
                     "around compiles)")
    .range(0, 2000);
REXCVAR_DEFINE_BOOL(banjo_renut_diag_poison_slots, false, "Banjo/Diagnostics",
                    "Renut's renderer, a test: descriptor slots given back point at a magenta texture instead of the "
                    "null one, so a draw still sampling a released texture shows magenta (sa2-nx's poison test). With "
                    "a large banjo_renut_textures_reserve_mb it checks the eviction rules");
REXCVAR_DEFINE_INT32(banjo_renut_stream_cache_mb, 128, "Banjo",
                     "banjo_renut_memory_policy: the converted-stream cache (vertex streams and index ranges kept "
                     "across frames) takes no more copies past this size (Renut: 768 MB)")
    .range(16, 768);
REXCVAR_DEFINE_INT32(banjo_renut_stream_cache_idle_s, 20, "Banjo",
                     "banjo_renut_memory_policy: a cached copy no draw read for this many seconds (at 30 frames a "
                     "second) goes (Renut: 120 s)")
    .range(2, 120);
REXCVAR_DEFINE_INT32(banjo_renut_upload_trim_frames, 900, "Banjo",
                     "banjo_renut_memory_policy: upload ring chunks beyond what the frames since the last trim used "
                     "(a loading frame's texture uploads grow the ring by 32 MB chunks) are released after this many "
                     "frames (Renut: 600). banjo-nx (2026-10-19, render-thread proposal 18): 900 (was 120): a chunk "
                     "released after 60 frames without a heavy frame was made again by the next one on the render "
                     "thread, mid-draw (a 32 MB heap allocation and its GPU mapping)")
    .range(8, 3600);
REXCVAR_DEFINE_BOOL(banjo_renut_precise_barriers, true, "Banjo/Enhancements",
                    "Renut's renderer (Vulkan): plume's barriers take their access masks from the layouts and the "
                    "declared accesses instead of MEMORY_READ | MEMORY_WRITE on both sides, which NVK turns into "
                    "cache flushes and invalidations at every barrier. Same ordering for every access that exists");

namespace banjo::renut_gpu {

// banjo_renut_shared_dedupe's counts (render thread), read by the [renut] report (game thread).
std::atomic<uint64_t> g_shared_reused{0}, g_shared_uploaded{0};
// banjo_renut_exact_streams' counts and banjo_renut_memory_policy's evictions, for the same report.
std::atomic<uint64_t> g_exact_checks{0}, g_exact_bytes{0}, g_exact_stale{0}, g_late_stale{0}, g_reserve_evicted{0};
// Package 27c: how the checks spread over a frame (the design of cheaper ones): checks of a copy already checked in the
// same frame, the kicks that start the intervals, and the game's fence queries.
std::atomic<uint64_t> g_exact_repeat{0}, g_exact_repeat_bytes{0}, g_kicks{0}, g_fence_queries{0};
std::atomic<uint64_t> g_exact_memo{0}, g_exact_memo_bytes{0};
// banjo-nx (2026-10-19): banjo_renut_store_cards' checks.
std::atomic<uint64_t> g_cards_clean{0}, g_cards_clean_bytes{0}, g_cards_dirty{0}, g_cards_missed{0};
std::atomic<uint64_t> g_tex_cards[4], g_tex_cards_bytes{0};  // proven, hashed, re-uploads saved, changed (verify)
std::atomic<uint64_t> g_kind_depth_only{0}, g_kind_depth_plain{0}, g_kind_nothing{0}, g_kind_partial{0};
std::atomic<uint64_t> g_depth_only_draws{0};  // banjo_gpu_depth_only (package 28)
std::atomic<uint64_t> g_resolves_drawn{0};  // banjo_deko3d_draw_resolves (2026-10-18)
std::atomic<uint64_t> g_resolves_skipped[2];  // banjo_renut_resolve_skip (2026-10-19): colour, depth
std::atomic<uint64_t> g_words_lazy[3];  // banjo_renut_depth_words_lazy: not written, written late, read stale
std::atomic<uint64_t> g_handover[6];    // banjo_deko3d_depth_handover (CountHandOver)
std::atomic<uint64_t> g_fetch_misaligned[2];  // 2026-10-20: draws checked, draws off 16 bytes (CountFetchAlignment)
std::atomic<uint64_t> g_device_swizzle[2];  // 2026-10-20: binds through the device's swizzle, frames with one
std::atomic<uint32_t> g_device_swizzle_frame{0};
std::atomic<uint64_t> g_fur_shells[2];  // 2026-10-20 banjo_renut_fur_shells: drawn, skipped
std::atomic<uint64_t> g_skin_once[8];  // 2026-10-20 banjo_deko3d_skin_once: written, read, plain, no room; + 4 not fur
std::atomic<uint64_t> g_exp_adjust[3];  // 2026-10-20: binds with an exponent adjust: all, negative, positive
std::atomic<uint64_t> g_skin_miss[6];  // near misses by field: streams, fetch window, index, first vertex, shared, any
std::atomic<uint64_t> g_header_differ[7];  // 2026-10-20: binds whose header (same data) differs from the device, per dword
std::atomic<uint64_t> g_ring_published{0};  // render-thread proposal 23
std::atomic<uint64_t> g_kick_submit[4];      // lists submitted at a kick, ranges cleaned by the thread / by the render thread
std::atomic<uint64_t> g_pushed[4];      // banjo_deko3d_push_constants: float bytes, shared bytes, draws, verify misses
std::atomic<uint64_t> g_depth_float_sampled{0};  // 2026-10-18 census
std::atomic<uint64_t> g_words_viewed{0}, g_words_copied{0};  // banjo_gpu_depth_words_view (package 28)
// banjo_gpu_bool_spec's: draws with specialised bools, of them drawn with the generic pipeline, specialised built.
std::atomic<uint64_t> g_bool_draws{0}, g_bool_generic{0}, g_bool_built{0};
// banjo_gpu_tess_grid's: the corners the per-corner form would shade, and the grid's vertices.
std::atomic<uint64_t> g_tess_corners{0}, g_tess_vertices{0};
// banjo_renut_cache_save_quiet (package 28): pipelines built (any thread) and when the last one was, steady-clock ms.
std::atomic<uint64_t> g_pipelines_built{0};
std::atomic<int64_t> g_last_build_ms{0};

int64_t SteadyMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void NotePipelineBuilt() {
  g_pipelines_built.fetch_add(1, std::memory_order_relaxed);
  g_last_build_ms.store(SteadyMs(), std::memory_order_relaxed);
}

// The A/B word (banjo_renut_ab); the header reads it inline.
std::atomic<int> g_ab_off{0};

namespace {

constexpr int kRenderThreadPriority = 0x2C;   // above the game's threads (0x3B), as banjo-nx's ring decoder
// The lowest an application thread takes is 0x3B (package 27c: 0x3F was refused on the console, "NOT accepted",
// and the workers ran at the default; UnleashedRecomp-NX's pipeline compiler takes 0x3B).
constexpr int kBackgroundPriority = 0x3B;

bool PipelineCacheOn() {
  return ::renut::gpu::IsVulkan() && EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_renut_pipeline_cache));
}

// The saver: plume's save asks the driver for its whole cache under the driver's lock, which pipeline creation on the
// render thread also takes, so it runs on its own low-priority thread, only when something new was compiled.
struct Saver {
  std::mutex mutex;
  std::condition_variable cv;
  bool stop = false;
  std::thread thread;
  plume::RenderDevice *device = nullptr;
};
Saver g_saver;

void SaveNow(const char *why) {
  if (g_saver.device && g_saver.device->savePipelineCache())
    REXLOG_INFO("[renut] pipeline cache saved ({})", why);
}

// The PMULL fingerprint against its bit-by-bit reference over lengths 0-300 (every tail and fold path) of
// pseudo-random bytes, with and without a seed. Any difference keeps XXH3 for the session.
bool PmullSelfTest() {
#if BANJO_PMULL_FINGERPRINT
  uint8_t data[300];
  uint64_t x = 0x9E3779B97F4A7C15ull;
  for (uint8_t &b : data) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    b = uint8_t(x >> 24);
  }
  for (size_t n = 0; n <= sizeof(data); ++n) {
    for (const uint64_t seed : {uint64_t(0), uint64_t(0xC0FFEE1234567890ull)}) {
      if (pmull_fingerprint::Fingerprint(data, n, seed) != pmull_fingerprint::Reference(data, n, seed))
        return false;
    }
  }
  return true;
#else
  return false;
#endif
}

// The 10 s report and the A/B halves. Game frames come from the game thread, the rest from the render thread.
struct Stats {
  std::mutex mutex;
  std::chrono::steady_clock::time_point start{}, last_report{}, last_frame{};
  struct Half {
    uint64_t game_frames = 0, render_frames = 0, gpu_frames = 0;
    double render_ms = 0.0, gpu_ms = 0.0, seconds = 0.0;
  };
  Half total, halves[2];   // halves[1]: the part on, [0]: off
  int half = 1;            // the half now running (the render thread applies it)
  int applied = -1;        // the half the render thread applied last
  std::chrono::steady_clock::time_point half_start{};
};
Stats g_stats;

const char *AbName(int target) {
  switch (target) {
  case 1: return "banjo_renut_precise_barriers";
  case 2: return "render thread pin (banjo_ring_core)";
  case 3: return "banjo_renut_shared_dedupe";
  case 4: return "banjo_renut_push_point_only";
  case 5: return "banjo_renut_capture_pool";
  case 6: return "banjo_renut_exact_streams";
  case 7: return "banjo_gpu_bool_spec";
  case 8: return "banjo_gpu_blend_skip";
  case 9: return "banjo_gpu_tess_grid";
  case 11: return "banjo_gpu_depth_only";
  case 12: return "banjo_gpu_output_trim";
  case 13: return "banjo_gpu_depth_words_view";
  case 14: return "banjo_renut_record_worker";
  case 15: return "banjo_deko3d_light_branch";
  case 16: return "banjo_deko3d_gpr32";
  case 17: return "banjo_deko3d_skin_once";
  case 18: return "banjo_deko3d_eager_kicks";
  default: return "none";
  }
}

// The A/B target whose part is in its "off" half now (0: none). The per-draw keys (3-5) read it on every use.

bool g_precise_barriers_on = false;  // banjo_renut_precise_barriers as configured (BeforeDeviceCreated)

}  // namespace

namespace {

// Render thread: the part as the half says.
void AbApply(int target, bool on) {
  g_ab_off.store(on ? 0 : target, std::memory_order_relaxed);
  if (target == 1) {
    plume::SetVulkanPreciseBarriers(on && g_precise_barriers_on);
  } else if (target == 2) {
#if REX_PLATFORM_SWITCH
    const int32_t core = REXCVAR_GET(banjo_ring_core);
    if (core >= 0)
      (void)(on ? RexSwitchPinCurrentThreadToCore(int(core)) : RexSwitchUnpinCurrentThread(-1));
#endif
  }
}

}  // namespace

void StatsGameFrame() {
  const auto now = std::chrono::steady_clock::now();
  std::lock_guard lock(g_stats.mutex);
  if (g_stats.start == std::chrono::steady_clock::time_point{}) {
    g_stats.start = g_stats.last_report = g_stats.half_start = g_stats.last_frame = now;
    return;
  }
  ++g_stats.total.game_frames;
  ++g_stats.halves[g_stats.half].game_frames;
  // Each frame's time goes to the half it ran in, so a half still running at a report counts what it has so far.
  g_stats.halves[g_stats.half].seconds += std::chrono::duration<double>(now - g_stats.last_frame).count();
  g_stats.last_frame = now;
  const int target = REXCVAR_GET(banjo_renut_ab_target);
  if (target && std::chrono::duration<double>(now - g_stats.half_start).count() >= double(REXCVAR_GET(banjo_renut_ab_s))) {
    g_stats.half ^= 1;
    g_stats.half_start = now;
  }
  const double since = std::chrono::duration<double>(now - g_stats.last_report).count();
  if (since < 10.0)
    return;
  const Stats::Half &t = g_stats.total;
  const uint64_t reused = g_shared_reused.exchange(0, std::memory_order_relaxed);
  const uint64_t uploaded = g_shared_uploaded.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_checks = g_exact_checks.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_bytes = g_exact_bytes.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_stale = g_exact_stale.exchange(0, std::memory_order_relaxed);
  const uint64_t late_stale = g_late_stale.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_repeat = g_exact_repeat.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_repeat_bytes = g_exact_repeat_bytes.exchange(0, std::memory_order_relaxed);
  const uint64_t kicks = g_kicks.exchange(0, std::memory_order_relaxed);
  const uint64_t fence_queries = g_fence_queries.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_memo = g_exact_memo.exchange(0, std::memory_order_relaxed);
  const uint64_t exact_memo_bytes = g_exact_memo_bytes.exchange(0, std::memory_order_relaxed);
  const uint64_t evicted = g_reserve_evicted.exchange(0, std::memory_order_relaxed);
  const uint64_t bool_draws = g_bool_draws.exchange(0, std::memory_order_relaxed);
  const uint64_t bool_generic = g_bool_generic.exchange(0, std::memory_order_relaxed);
  const uint64_t bool_built = g_bool_built.exchange(0, std::memory_order_relaxed);
  const uint64_t tess_corners = g_tess_corners.exchange(0, std::memory_order_relaxed);
  const uint64_t tess_vertices = g_tess_vertices.exchange(0, std::memory_order_relaxed);
  REXLOG_INFO("[renut] {:.1f} s: {} game frames ({:.2f} fps), render thread busy {:.2f} ms a frame ({} frames), "
              "GPU {:.2f} ms a frame ({} frames); {} draws, SharedConstants reused by {} (banjo_renut_shared_dedupe); "
              "cached streams: {} checks of {:.1f} MB ({} of them, {:.1f} MB, of a copy already checked in the frame; {} "
              "answered by the interval memo, {:.1f} MB not read again; {} kicks, {} fence queries), {} stale caught (banjo_renut_exact_streams), {} found stale late; {} textures released under the reserve (banjo_renut_memory_policy); specialised bools: {} "
              "draws, {} of them on the generic pipeline while theirs was built, {} pipelines built "
              "(banjo_gpu_bool_spec); tessellation: {} grid vertices for {} corners (banjo_gpu_tess_grid)",
              since, t.game_frames, double(t.game_frames) / since,
              t.render_frames ? t.render_ms / double(t.render_frames) : 0.0, t.render_frames,
              t.gpu_frames ? t.gpu_ms / double(t.gpu_frames) : 0.0, t.gpu_frames, reused + uploaded, reused,
              exact_checks, double(exact_bytes) / double(1u << 20), exact_repeat,
              double(exact_repeat_bytes) / double(1u << 20), exact_memo, double(exact_memo_bytes) / double(1u << 20),
              kicks, fence_queries, exact_stale, late_stale, evicted,
              bool_draws,
              bool_generic, bool_built, tess_vertices, tess_corners);
  if (g_latched_ready.load(std::memory_order_acquire) && g_latched.store_cards) {
    const uint64_t clean = g_cards_clean.exchange(0, std::memory_order_relaxed);
    const uint64_t clean_bytes = g_cards_clean_bytes.exchange(0, std::memory_order_relaxed);
    const uint64_t dirty = g_cards_dirty.exchange(0, std::memory_order_relaxed);
    const uint64_t missed = g_cards_missed.exchange(0, std::memory_order_relaxed);
    REXLOG_INFO("[renut] store cards: {} copies proven by their cards ({:.1f} MB not hashed), {} with a page written "
                "(hashed){} (banjo_renut_store_cards); textures: {} checks proven, {} hashed, {} dynamic re-uploads "
                "not made ({:.1f} MB not read){}",
                clean, double(clean_bytes) / double(1u << 20), dirty,
                g_latched.store_cards_verify ? fmt::format("; verify: {} changed under clean cards", missed)
                                             : std::string(),
                g_tex_cards[0].exchange(0, std::memory_order_relaxed),
                g_tex_cards[1].exchange(0, std::memory_order_relaxed),
                g_tex_cards[2].exchange(0, std::memory_order_relaxed),
                double(g_tex_cards_bytes.exchange(0, std::memory_order_relaxed)) / double(1u << 20),
                g_latched.store_cards_verify
                    ? fmt::format("; verify: {} textures changed under clean cards",
                                  g_tex_cards[3].exchange(0, std::memory_order_relaxed))
                    : std::string());
  }
  if (target) {
    const Stats::Half &on = g_stats.halves[1], &off = g_stats.halves[0];
    auto line = [](const Stats::Half &h) {
      return fmt::format("{:.2f} fps, render thread {:.2f} ms a frame, GPU {:.2f} ms a frame ({:.0f} s)",
                         h.seconds > 0.0 ? double(h.game_frames) / h.seconds : 0.0,
                         h.render_frames ? h.render_ms / double(h.render_frames) : 0.0,
                         h.gpu_frames ? h.gpu_ms / double(h.gpu_frames) : 0.0, h.seconds);
    };
    REXLOG_INFO("[renut A/B] {} (target {}): on {}; off {}", AbName(target), target, line(on), line(off));
    g_stats.halves[0] = g_stats.halves[1] = {};
  }
  ReportNvk();
  record::Report(static_cast<int>(since + 0.5));  // banjo_renut_record_worker (package 28b)
#if defined(PLUME_DEKO3D)
  if (::renut::gpu::IsDeko3D())  // banjo-nx (2026-10-17): the deko3d back end
    REXLOG_INFO("[renut] {}", deko3d::Report());
#endif
  REXLOG_INFO("[renut] draw kinds: {} depth-only ({} of them without alpha test or alpha to coverage, with a pixel "
              "shader; {} drawn without it, banjo_gpu_depth_only), {} writing nothing, {} writing part of a target's "
              "channels",
              g_kind_depth_only.exchange(0, std::memory_order_relaxed),
              g_kind_depth_plain.exchange(0, std::memory_order_relaxed),
              g_depth_only_draws.exchange(0, std::memory_order_relaxed),
              g_kind_nothing.exchange(0, std::memory_order_relaxed), g_kind_partial.exchange(0, std::memory_order_relaxed));
  REXLOG_INFO("[renut] depth resolve words: {} binds through the resolve's own texture (banjo_gpu_depth_words_view), "
              "{} with the copy (made only when resolved since); {} binds of a depth resolve's float target; {} colour "
              "resolves drawn (banjo_deko3d_draw_resolves); resolves of an unchanged source not done again "
              "(banjo_renut_resolve_skip): {} colour, {} depth; depth resolves without their words "
              "(banjo_renut_depth_words_lazy): {}, words written late {}, read stale {}",
              g_words_viewed.exchange(0, std::memory_order_relaxed), g_words_copied.exchange(0, std::memory_order_relaxed),
              g_depth_float_sampled.exchange(0, std::memory_order_relaxed),
              g_resolves_drawn.exchange(0, std::memory_order_relaxed),
              g_resolves_skipped[0].exchange(0, std::memory_order_relaxed),
              g_resolves_skipped[1].exchange(0, std::memory_order_relaxed),
              g_words_lazy[0].exchange(0, std::memory_order_relaxed),
              g_words_lazy[1].exchange(0, std::memory_order_relaxed),
              g_words_lazy[2].exchange(0, std::memory_order_relaxed));
  if (PushConstantsOn()) {
    REXLOG_INFO("[renut] pushed constants (banjo_deko3d_push_constants): {} draws, {:.1f} KB of float constants and "
                "{:.1f} KB of shared constants pushed{}",
                g_pushed[2].exchange(0, std::memory_order_relaxed),
                double(g_pushed[0].exchange(0, std::memory_order_relaxed)) / 1024.0,
                double(g_pushed[1].exchange(0, std::memory_order_relaxed)) / 1024.0,
                PushConstantsVerifyOn() ? fmt::format("; verify: {} draws with a register the model got wrong",
                                                      g_pushed[3].exchange(0, std::memory_order_relaxed))
                                        : std::string());
  }
  {
    // banjo-nx (2026-10-19, input latency): the probe and the late input delay.
    const std::string probe = rex::latency::Report();
    REXLOG_INFO("[latency] {}; {}", probe.empty() ? std::string("no frame presented") : probe,
                ::renut::gpu::render_thread::LateInputReport());
    if (const std::string trim = ::renut::gpu::Video::LatencyTrimReport(); !trim.empty())
      REXLOG_INFO("[latency] {}", trim);  // banjo-nx (2026-10-20, banjo_deko3d_latency_trim)
    // banjo-nx (2026-10-19, latency batch 3): the display's refresh, the phase and input -> display.
    if (const std::string display = banjo::display_sync::Report(); !display.empty())
      REXLOG_INFO("[display] {}", display);
  }
  REXLOG_INFO("[renut] upload ring: {:.1f} MB cleaned (render-thread proposal 23); ranges cleaned by the cleaner "
              "thread {}, by the render thread {}; command lists submitted at a kick {} (banjo_deko3d_kick_submit)",
              double(g_ring_published.exchange(0, std::memory_order_relaxed)) / double(1u << 20),
              g_kick_submit[1].exchange(0, std::memory_order_relaxed),
              g_kick_submit[2].exchange(0, std::memory_order_relaxed),
              g_kick_submit[0].exchange(0, std::memory_order_relaxed));
  // banjo-nx (2026-10-20): tools/deko3d_glsl_opt.py's aw4 pass reads bone rows as 16-byte loads; the fetch data's
  // offset, slots and size must stay multiples of 16 (render/draw.cpp UploadVertexFetchData). Expected: 0 off.
  if (const uint64_t checked = g_fetch_misaligned[0].exchange(0, std::memory_order_relaxed)) {
    const uint64_t off = g_fetch_misaligned[1].exchange(0, std::memory_order_relaxed);
    if (off)
      REXLOG_WARN("[renut] fetch data: {} of {} draws with an offset, slot or size off 16 bytes (the aw4 loads read "
                  "the wrong words there)", off, checked);
  }
  if (const uint64_t n = g_exp_adjust[0].exchange(0, std::memory_order_relaxed)) {
    const uint64_t neg = g_exp_adjust[1].exchange(0, std::memory_order_relaxed);
    const uint64_t pos = g_exp_adjust[2].exchange(0, std::memory_order_relaxed);
    REXLOG_WARN("[renut] exponent adjust: {} texture binds with a fetch exponent adjust ({} negative, {} positive), "
                "which Renut does not apply (the sampled values are 2^adjust too large or small)", n, neg, pos);
  }
  if (REXCVAR_GET(banjo_deko3d_skin_once)) {
    uint64_t k[8];
    for (int i = 0; i < 8; ++i)
      k[i] = g_skin_once[i].exchange(0, std::memory_order_relaxed);
    uint64_t m[6];
    for (int i = 0; i < 6; ++i)
      m[i] = g_skin_miss[i].exchange(0, std::memory_order_relaxed);
    REXLOG_INFO("[renut] skin once (banjo_deko3d_skin_once): {} fur draws skinned and wrote their rows, {} read them, "
                "{} skinned as before (not the part's first draw with the same inputs: none to read), {} without room; "
                "writes that missed a same-size entry {}: vertex data {}, bone data {}, indices {}, first vertex {}, "
                "shared constants {}; other skinned draws (shadow maps, depth only): {} wrote, {} read, {} as before, {} "
                "without room",
                k[0], k[1], k[2], k[3], m[5], m[0], m[1], m[2], m[3], m[4], k[4], k[5], k[6], k[7]);
  }
  if (REXCVAR_GET(banjo_renut_fur_shells) > 1 || REXCVAR_GET(banjo_renut_portrait_fur_shells) > 1) {
    const uint64_t drawn = g_fur_shells[0].exchange(0, std::memory_order_relaxed);
    const uint64_t skipped = g_fur_shells[1].exchange(0, std::memory_order_relaxed);
    REXLOG_INFO("[renut] fur shells (banjo_renut_fur_shells {}, portrait {}, not exact): {} drawn, {} not drawn",
                REXCVAR_GET(banjo_renut_fur_shells), REXCVAR_GET(banjo_renut_portrait_fur_shells), drawn, skipped);
  }
  if (const uint64_t binds = g_header_differ[6].exchange(0, std::memory_order_relaxed)) {
    uint64_t d[6];
    for (int i = 0; i < 6; ++i)
      d[i] = g_header_differ[i].exchange(0, std::memory_order_relaxed);
    REXLOG_INFO("[flash] header vs device (same data): {} binds with other fields; by dword: 0 (signs, pitch, tiling) "
                "{}, 3 (swizzle, number format, exponent) {}, 4 (mip levels) {}, 5 (mips) {}, 1/2 {}", binds, d[0], d[3],
                d[4], d[5], d[1] + d[2]);
  }
  if (const uint64_t binds = g_device_swizzle[0].exchange(0, std::memory_order_relaxed)) {
    const uint64_t frames = g_device_swizzle[1].exchange(0, std::memory_order_relaxed);
    REXLOG_INFO("[flash] device swizzle: {} binds in {} frames whose texture header's swizzle differed from the "
                "device's; sampled through the {}'s (banjo_renut_device_swizzle)", binds, frames,
                DeviceSwizzleOn() ? "device" : "header");
  }
  if (DepthHandOverOn()) {
    REXLOG_INFO("[renut] depth hand-over (banjo_deko3d_depth_handover): {} resolves not drawn, {} binds of a target's "
                "image, {} images taken at a whole clear; {} drawn after all ({} from a taken image); {} lost",
                g_handover[0].exchange(0, std::memory_order_relaxed), g_handover[1].exchange(0, std::memory_order_relaxed),
                g_handover[2].exchange(0, std::memory_order_relaxed), g_handover[3].exchange(0, std::memory_order_relaxed),
                g_handover[4].exchange(0, std::memory_order_relaxed), g_handover[5].exchange(0, std::memory_order_relaxed));
  }
  // The generated code's self-checks (their "[verify]" line and any DIFFERENCE): banjo-nx's renderer reports them
  // in its own 10 s report; with Renut's renderer nobody did (package 27).
  banjo::CodegenChecksReport(static_cast<int>(since + 0.5));
  g_stats.total = {};
  g_stats.last_report = now;
}

void StatsRenderThreadFrame(double busy_ms) {
  ControlNvkLocked();  // banjo_gpu_nvk_draw_contract (the Switch): requests and guards, once a frame
  int half, target;
  bool apply;
  {
    std::lock_guard lock(g_stats.mutex);
    ++g_stats.total.render_frames;
    g_stats.total.render_ms += busy_ms;
    ++g_stats.halves[g_stats.half].render_frames;
    g_stats.halves[g_stats.half].render_ms += busy_ms;
    target = REXCVAR_GET(banjo_renut_ab_target);
    half = g_stats.half;
    apply = target && g_stats.applied != half;
    g_stats.applied = half;
  }
  if (apply)
    AbApply(target, half == 1);
}

void StatsGpuFrame(double ms) {
  std::lock_guard lock(g_stats.mutex);
  ++g_stats.total.gpu_frames;
  g_stats.total.gpu_ms += ms;
  ++g_stats.halves[g_stats.half].gpu_frames;
  g_stats.halves[g_stats.half].gpu_ms += ms;
}

bool g_pmull_hash = false;



// banjo-nx (2026-10-19, CPU proposal 5): a counter only the render thread adds to: a load and a store, not an exclusive
// read-modify-write loop (the A57 has no LSE atomics). The 10 s report's exchange may rarely lose one count.
inline void Bump(std::atomic<uint64_t> &c, uint64_t v) {
  c.store(c.load(std::memory_order_relaxed) + v, std::memory_order_relaxed);
}

void CountTessGrid(uint64_t corners, uint64_t vertices) {
  Bump(g_tess_corners, corners);
  Bump(g_tess_vertices, vertices);
}



bool RecordWorkerAbOn() {
  return g_ab_off.load(std::memory_order_relaxed) != 14;
}


bool Deko3DDepthCompressionOn() { return REXCVAR_GET(banjo_deko3d_depth_compression); }
bool Deko3DZcullOn() { return REXCVAR_GET(banjo_deko3d_zcull); }
bool Deko3DBarrierMergeOn() { return REXCVAR_GET(banjo_deko3d_barrier_merge); }
bool Deko3DBarrierLeanOn() { return REXCVAR_GET(banjo_deko3d_barrier_lean); }
bool Deko3DVsOutputsOn() { return REXCVAR_GET(banjo_deko3d_vs_outputs); }
bool Deko3DUamPrmtOn() { return REXCVAR_GET(banjo_deko3d_uam_prmt); }
bool EarlyHoistOn() { return REXCVAR_GET(banjo_deko3d_early_hoist); }
bool StallPresentOn() { return REXCVAR_GET(banjo_renut_stall_present); }
bool EarlyZOn() { return REXCVAR_GET(banjo_deko3d_early_z); }
bool Gpr32On() { return REXCVAR_GET(banjo_deko3d_gpr32); }
bool LightBranchOn() { return REXCVAR_GET(banjo_deko3d_light_branch); }
bool SkinOnceOn() { return REXCVAR_GET(banjo_deko3d_skin_once); }
bool EagerKicksOn() {
  return REXCVAR_GET(banjo_deko3d_eager_kicks) && g_ab_off.load(std::memory_order_relaxed) != 18;
}
void CountExpAdjust(int adjust) {
  g_exp_adjust[0].fetch_add(1, std::memory_order_relaxed);
  g_exp_adjust[adjust < 0 ? 1 : 2].fetch_add(1, std::memory_order_relaxed);
}
void CountSkinOnce(uint32_t kind) { g_skin_once[kind & 7].fetch_add(1, std::memory_order_relaxed); }
void CountSkinMiss(uint32_t fields) {
  for (int i = 0; i < 5; ++i)
    if (fields & (1u << i))
      g_skin_miss[i].fetch_add(1, std::memory_order_relaxed);
  g_skin_miss[5].fetch_add(1, std::memory_order_relaxed);
}
bool NullPsOn() { return REXCVAR_GET(banjo_renut_diag_null_ps); }
bool DeviceSwizzleOn() { return REXCVAR_GET(banjo_renut_device_swizzle); }
bool LatencyTrimOn() { return REXCVAR_GET(banjo_deko3d_latency_trim); }
uint32_t FurShellEvery() { return uint32_t(REXCVAR_GET(banjo_renut_fur_shells)); }
uint32_t PortraitFurShellEvery() { return uint32_t(REXCVAR_GET(banjo_renut_portrait_fur_shells)); }
void CountFurShell(bool drawn) { g_fur_shells[drawn ? 0 : 1].fetch_add(1, std::memory_order_relaxed); }
uint32_t TxqMode() {
  return REXCVAR_GET(banjo_deko3d_txq_verify) ? 2u : REXCVAR_GET(banjo_deko3d_txq_constants) ? 1u : 0u;
}
unsigned Deko3DUamFlags() {
  // PRMT (1), offsets in the load (2), the shift distribution (4), the input interpolation cache (8), the choice
  // per pixel shader (16), value numbering (32) keeping the interpolations (128), no reconvergence around uniform
  // branches (256), bone rows as 128-bit loads (1024, 2026-10-20): docs/deko3d-shader-optimisations.md
  unsigned flags = 1471u;
  if (!REXCVAR_GET(banjo_deko3d_uam_wide_rows))
    flags &= ~1024u;
  if (!REXCVAR_GET(banjo_deko3d_uam_prmt))
    flags &= ~1u;
  if (!REXCVAR_GET(banjo_deko3d_uam_gvn))
    flags &= ~32u;
  if (!REXCVAR_GET(banjo_deko3d_uam_uniform_branches))
    flags &= ~256u;
  return flags;
}
int Deko3DCompileCore() { return REXCVAR_GET(banjo_background_compile_core); }
int Deko3DCompilePriority() { return REXCVAR_GET(banjo_deko3d_compile_priority); }


void CountDepthFloatSampled() {
  Bump(g_depth_float_sampled, 1);
}

void CountDrawnResolve() {
  g_resolves_drawn.fetch_add(1, std::memory_order_relaxed);
}



void CountKickSubmit(int what) {
  g_kick_submit[what].store(g_kick_submit[what].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void CountRingPublished(uint64_t bytes) {
  g_ring_published.store(g_ring_published.load(std::memory_order_relaxed) + bytes, std::memory_order_relaxed);
}

void CountPushedConstants(uint64_t bytes, int what) {
  g_pushed[what].store(g_pushed[what].load(std::memory_order_relaxed) + bytes, std::memory_order_relaxed);
}

void CountFetchAlignment(bool aligned) {
  g_fetch_misaligned[0].fetch_add(1, std::memory_order_relaxed);
  if (!aligned)
    g_fetch_misaligned[1].fetch_add(1, std::memory_order_relaxed);
}

void CountHeaderDescDiffer(uint32_t dwords) {
  g_header_differ[6].fetch_add(1, std::memory_order_relaxed);
  for (int i = 0; i < 6; ++i)
    if (dwords & (1u << i))
      g_header_differ[i].fetch_add(1, std::memory_order_relaxed);
}

void CountDeviceSwizzle(uint32_t frame) {
  g_device_swizzle[0].fetch_add(1, std::memory_order_relaxed);
  if (g_device_swizzle_frame.exchange(frame, std::memory_order_relaxed) != frame)
    g_device_swizzle[1].fetch_add(1, std::memory_order_relaxed);
}

void CountHandOver(int what) {
  g_handover[what].store(g_handover[what].load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void CountLazyWords(int what) {
  g_words_lazy[what].fetch_add(1, std::memory_order_relaxed);
}

void CountSkippedResolve(bool depth) {
  g_resolves_skipped[depth ? 1 : 0].fetch_add(1, std::memory_order_relaxed);
}

void CountDepthWords(bool viewed) {
  (viewed ? g_words_viewed : g_words_copied).fetch_add(1, std::memory_order_relaxed);
}



void CountDepthOnly() {
  g_depth_only_draws.fetch_add(1, std::memory_order_relaxed);
}






void CountBoolSpec(BoolSpecEvent event) {
  Bump(event == BoolSpecEvent::kDraw      ? g_bool_draws
       : event == BoolSpecEvent::kGeneric ? g_bool_generic
                                          : g_bool_built,
       1);
}


void CountExactCheck(uint64_t bytes, bool stale, bool repeat_in_frame) {
  Bump(g_exact_checks, 1);
  Bump(g_exact_bytes, bytes);
  if (stale)
    Bump(g_exact_stale, 1);
  if (repeat_in_frame) {
    Bump(g_exact_repeat, 1);
    Bump(g_exact_repeat_bytes, bytes);
  }
}

void CountCards(uint64_t bytes, bool clean) {
  if (clean) {
    Bump(g_cards_clean, 1);
    Bump(g_cards_clean_bytes, bytes);
  } else {
    Bump(g_cards_dirty, 1);
  }
}

void CountCardsMissed(uint64_t) { Bump(g_cards_missed, 1); }

void CountTextureCards(uint64_t bytes, int what) {
  Bump(g_tex_cards[what], 1);
  if (what == 0 || what == 2)
    Bump(g_tex_cards_bytes, bytes);
}

void CountExactMemoHit(uint64_t bytes) {
  Bump(g_exact_memo, 1);
  Bump(g_exact_memo_bytes, bytes);
}

void CountDrawKind(bool depth_only, bool depth_only_plain, bool writes_nothing, bool partial_mask) {
  if (depth_only)
    Bump(g_kind_depth_only, 1);
  if (depth_only_plain)
    Bump(g_kind_depth_plain, 1);
  if (writes_nothing)
    Bump(g_kind_nothing, 1);
  if (partial_mask)
    Bump(g_kind_partial, 1);
}

void CountKick() {
  g_kicks.fetch_add(1, std::memory_order_relaxed);
}

void CountFenceQuery() {
  g_fence_queries.fetch_add(1, std::memory_order_relaxed);
}

void CountLateStale(uint64_t copies) {
  g_late_stale.fetch_add(copies, std::memory_order_relaxed);
}

uint64_t TextureReserveBytes() {
  return REXCVAR_GET(banjo_renut_memory_policy) ? uint64_t(REXCVAR_GET(banjo_renut_textures_reserve_mb)) << 20 : 0;
}

uint64_t StreamCacheMaxBytes(uint64_t renut_max) {
  return REXCVAR_GET(banjo_renut_memory_policy) ? uint64_t(REXCVAR_GET(banjo_renut_stream_cache_mb)) << 20 : renut_max;
}

uint64_t StreamCacheIdleFrames(uint64_t renut_frames) {
  return REXCVAR_GET(banjo_renut_memory_policy) ? uint64_t(REXCVAR_GET(banjo_renut_stream_cache_idle_s)) * 30
                                                : renut_frames;
}

uint32_t UploadTrimResets(uint32_t renut_resets, uint32_t frames_in_flight) {
  if (!REXCVAR_GET(banjo_renut_memory_policy))
    return renut_resets;
  return std::max<uint32_t>(1, uint32_t(REXCVAR_GET(banjo_renut_upload_trim_frames)) / std::max<uint32_t>(frames_in_flight, 1));
}

void CountReserveEviction(uint64_t textures) {
  g_reserve_evicted.fetch_add(textures, std::memory_order_relaxed);
}

#if REX_PLATFORM_SWITCH
namespace {

// The per-draw contract under either name (banjo_gpu_nvk_draw_contract: the linked Mesa exports nfsmw-nx's).
NvkSwitchDraw *NvkDrawContract() {
  if (&nvk_switch_draw != nullptr)
    return &nvk_switch_draw;
  static const bool also_dibujo =
      EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_nvk_draw_contract));
  return also_dibujo ? &nvk_switch_dibujo : nullptr;
}

struct NvkState {
  bool logged = false, preload = false;
  bool off_warned[5] = {}, set4_off_warned = false;
  uint64_t parts[16][2] = {}, counts[13] = {}, improvements[5][4] = {}, set4[4] = {};
  bool reported = false;
};
NvkState g_nvk;

}  // namespace
#endif

void ControlNvkLocked() {
#if REX_PLATFORM_SWITCH
  NvkSwitchDraw *const nvk = NvkDrawContract();
  NvkSwitchSet4 *const set4 = &nvk_switch_set4;
  const bool draw_ok = nvk != nullptr && nvk->version == 1;
  const bool set4_ok = set4 != nullptr && set4->version == 1;
  const bool set4_request = REXCVAR_GET(masseffect_native_set4_differences);
  const int32_t requests[5] = {REXCVAR_GET(masseffect_native_nvk_emission) ? 1 : 0,
                               REXCVAR_GET(masseffect_native_nvk_cbufs) ? 1 : 0,
                               REXCVAR_GET(masseffect_native_nvk_dynamic) ? 1 : 0,
                               0,  // the fast set 4 path is not applied (as banjo-nx's renderer)
                               REXCVAR_GET(masseffect_native_nvk_preload) ? 1 : 0};
  if (draw_ok) {
    int32_t measure = std::max<int32_t>(0, REXCVAR_GET(masseffect_native_nvk_measure));
    if (measure > 0)
      measure = int32_t(std::bit_floor(uint32_t(measure)));  // NVK wants a power of 2
    __atomic_store_n(&nvk->measure, measure, __ATOMIC_RELAXED);
    for (size_t m = 0; m < 5; ++m)
      __atomic_store_n(&nvk->improvements[m].request, requests[m], __ATOMIC_RELAXED);
    static constexpr const char *kNames[5] = {"emission", "cbufs", "dynamic", "fast set 4", "prefetch"};
    for (size_t m = 0; m < 5; ++m) {
      if (!g_nvk.off_warned[m] && __atomic_load_n(&nvk->improvements[m].off, __ATOMIC_RELAXED) != 0) {
        g_nvk.off_warned[m] = true;
        REXLOG_ERROR("[renut] NVK per draw: MISMATCH seen by the guard of '{}' ({} differing checks; details in "
                     "rex_stderr.log). Switched off by NVK for the rest of the session: done as usual",
                     kNames[m], __atomic_load_n(&nvk->improvements[m].differences, __ATOMIC_RELAXED));
      }
    }
  }
  g_nvk.preload = draw_ok && requests[4] != 0;
  if (set4_ok) {
    __atomic_store_n(&set4->request, set4_request ? 1 : 0, __ATOMIC_RELAXED);
    if (!g_nvk.set4_off_warned && __atomic_load_n(&set4->off, __ATOMIC_RELAXED) != 0) {
      g_nvk.set4_off_warned = true;
      REXLOG_ERROR("[renut] NVK dynamic descriptors by differences: MISMATCH seen by its guard ({} cbufs wrongly "
                   "bound, each fixed in its own draw). Switched off for the rest of the session",
                   __atomic_load_n(&set4->total.differences, __ATOMIC_RELAXED));
    }
  }
  if (!g_nvk.logged) {
    g_nvk.logged = true;
    if (draw_ok) {
      REXLOG_INFO("[renut] NVK per draw (banjo_gpu_nvk_draw_contract): improvements requested (emission {}, cbufs {}, "
                  "dynamic {}, prefetch {}), measuring 1 in {} calls; dynamic descriptors by differences {}",
                  requests[0], requests[1], requests[2], requests[4], REXCVAR_GET(masseffect_native_nvk_measure),
                  set4_ok ? (set4_request ? "requested" : "not requested") : "not in this NVK");
    } else {
      REXLOG_INFO("[renut] NVK per draw: this NVK lacks the contract (or banjo_gpu_nvk_draw_contract is off): no "
                  "measurement and no improvements; dynamic descriptors by differences {}",
                  set4_ok ? (set4_request ? "requested" : "not requested") : "not in this NVK");
    }
  }
#endif
}

void BeforePipelineBind(plume::RenderPipeline *pipeline) {
#if REX_PLATFORM_SWITCH
  // PRFM only: it reads and writes nothing and cannot fail (banjo-nx's renderer does the same before its binds).
  if (g_nvk.preload && pipeline && vk_switch_preload_pipeline && ::renut::gpu::IsVulkan())
    vk_switch_preload_pipeline(static_cast<plume::VulkanGraphicsPipeline *>(pipeline)->vk);
#else
  (void)pipeline;
#endif
}

void ReportNvk() {
#if REX_PLATFORM_SWITCH
  NvkSwitchDraw *const nvk = NvkDrawContract();
  if (nvk == nullptr || nvk->version != 1)
    return;
  uint64_t times[16], ticks[16], counts[13], improvements[5][4];
  for (size_t i = 0; i < 16; ++i) {
    const uint64_t v = __atomic_load_n(&nvk->parts[i].times, __ATOMIC_RELAXED);
    const uint64_t t = __atomic_load_n(&nvk->parts[i].ticks, __ATOMIC_RELAXED);
    times[i] = v - g_nvk.parts[i][0];
    ticks[i] = t - g_nvk.parts[i][1];
    g_nvk.parts[i][0] = v;
    g_nvk.parts[i][1] = t;
  }
  for (size_t i = 0; i < 13; ++i) {
    const uint64_t c = __atomic_load_n(&nvk->counts[i], __ATOMIC_RELAXED);
    counts[i] = c - g_nvk.counts[i];
    g_nvk.counts[i] = c;
  }
  for (size_t m = 0; m < 5; ++m) {
    const uint64_t c[4] = {__atomic_load_n(&nvk->improvements[m].uses, __ATOMIC_RELAXED),
                           __atomic_load_n(&nvk->improvements[m].validated, __ATOMIC_RELAXED),
                           __atomic_load_n(&nvk->improvements[m].differences, __ATOMIC_RELAXED),
                           __atomic_load_n(&nvk->improvements[m].no_check, __ATOMIC_RELAXED)};
    for (size_t k = 0; k < 4; ++k) {
      improvements[m][k] = c[k] - g_nvk.improvements[m][k];
      g_nvk.improvements[m][k] = c[k];
    }
  }
  if (!g_nvk.reported) {  // the first call takes the starting point
    g_nvk.reported = true;
    return;
  }
  const uint64_t tps = __atomic_load_n(&nvk->ticks_per_second, __ATOMIC_RELAXED);
  const double us_tick = tps ? 1e6 / double(tps) : 0.0;
  const uint64_t draws = counts[0];
  auto part = [&](size_t i) {
    return fmt::format("{:.2f} [{:.2f}]", times[i] ? double(ticks[i]) * us_tick / double(times[i]) : 0.0,
                       draws ? double(ticks[i]) * us_tick / double(draws) : 0.0);
  };
  if (draws) {
    REXLOG_INFO("[renut] NVK parts (us per measured call [us per measured draw]; {} measured draws): whole draw {} | "
                "push desc {} | dynamic {} | shaders {} | cbufs {} | emit {} | BindPipeline: shaders {} state copy {} "
                "| sets {} (root {} dirty {}) | root table {} | new chunk {} | BindVertexBuffers {}",
                draws, part(0), part(1), part(2), part(4), part(5), part(6), part(7), part(9), part(10), part(11),
                part(12), part(13), part(14), part(15));
  }
  static constexpr size_t kShown[4] = {0, 1, 2, 4};
  static constexpr const char *kNames[5] = {"emission", "cbufs", "dynamic", "fast set 4", "prefetch"};
  std::string text;
  for (const size_t m : kShown) {
    const char *state = __atomic_load_n(&nvk->improvements[m].off, __ATOMIC_RELAXED) != 0 ? "SWITCHED OFF by the guard"
                        : __atomic_load_n(&nvk->improvements[m].request, __ATOMIC_RELAXED) == 0 ? "not requested"
                                                                                           : "on";
    text += fmt::format(" | {} ({}): {} uses, {} checked equal, {} differing, {} unchecked", kNames[m], state,
                        improvements[m][0], improvements[m][1], improvements[m][2], improvements[m][3]);
  }
  if (NvkSwitchSet4 *const set4 = &nvk_switch_set4; set4 != nullptr && set4->version == 1) {
    const uint64_t c[4] = {__atomic_load_n(&set4->total.bindings_difference, __ATOMIC_RELAXED),
                           __atomic_load_n(&set4->total.cbufs_saved, __ATOMIC_RELAXED),
                           __atomic_load_n(&set4->total.checks, __ATOMIC_RELAXED),
                           __atomic_load_n(&set4->total.differences, __ATOMIC_RELAXED)};
    text += fmt::format(" | dynamic descriptors by differences ({}): {} binds, {} cbuf rebinds saved, {} checked, {} "
                        "differing",
                        __atomic_load_n(&set4->off, __ATOMIC_RELAXED) ? "SWITCHED OFF by the guard"
                        : __atomic_load_n(&set4->request, __ATOMIC_RELAXED) > 0 ? "on" : "not requested",
                        c[0] - g_nvk.set4[0], c[1] - g_nvk.set4[1], c[2] - g_nvk.set4[2], c[3] - g_nvk.set4[3]);
    for (size_t k = 0; k < 4; ++k)
      g_nvk.set4[k] = c[k];
  }
  REXLOG_INFO("[renut] NVK improvements (last 10 s){}", text);
#endif
}


std::atomic<bool> g_trace_frame{false};  // banjo-nx (2026-10-19): read inline by ResolveTraceOn
namespace {
std::chrono::steady_clock::time_point g_trace_last{};
}  // namespace

void ResolveTraceFrameStart() {
  if (!REXCVAR_GET(banjo_renut_diag_resolve_trace)) {
    g_trace_frame.store(false, std::memory_order_relaxed);
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  if (g_trace_frame.load(std::memory_order_relaxed)) {
    g_trace_frame.store(false, std::memory_order_relaxed);
    REXLOG_INFO("[trace] frame end");
  } else if (g_trace_last == std::chrono::steady_clock::time_point{} ||
             now - g_trace_last >= std::chrono::seconds(30)) {
    g_trace_last = now;
    g_trace_frame.store(true, std::memory_order_relaxed);
    REXLOG_INFO("[trace] frame start");
  }
}


int DiagBuildDelayMs() {
  return REXCVAR_GET(banjo_renut_diag_build_delay_ms);
}


uint64_t PmullHash(const void *data, size_t size, uint64_t seed) {
#if BANJO_PMULL_FINGERPRINT
  return pmull_fingerprint::Fingerprint(static_cast<const uint8_t *>(data), size, seed);
#else
  return XXH3_64bits_withSeed(data, size, seed);
#endif
}

void OnRenderThreadStart() {
#if REX_PLATFORM_SWITCH
  const bool priority_ok = RexSwitchSetCurrentThreadPriorityOk(kRenderThreadPriority);
  const int32_t core = REXCVAR_GET(banjo_ring_core);
  const bool core_ok = core < 0 || RexSwitchPinCurrentThreadToCore(int(core));
  REXLOG_INFO("[renut] render thread at priority 0x{:X}{}, {} (banjo_ring_core), now on core {}",
              kRenderThreadPriority, priority_ok ? "" : " (NOT accepted)",
              core < 0 ? std::string("not pinned")
                       : fmt::format("pinned to core {}{}", core, core_ok ? "" : " (NOT accepted)"),
              RexSwitchCurrentCore());
#endif
}

void OnBackgroundThreadStart(const char *what) {
#if REX_PLATFORM_SWITCH
  const bool priority_ok = RexSwitchSetCurrentThreadPriorityOk(kBackgroundPriority);
  const int32_t core = REXCVAR_GET(banjo_background_compile_core);
  const bool core_ok = core < 0 || RexSwitchPinCurrentThreadToCore(int(core));
  REXLOG_INFO("[renut] {} at priority 0x{:X}{}, {} (banjo_background_compile_core)", what, kBackgroundPriority,
              priority_ok ? "" : " (NOT accepted)",
              core < 0 ? std::string("not pinned")
                       : fmt::format("pinned to core {}{}", core, core_ok ? "" : " (NOT accepted)"));
#else
  (void)what;
#endif
}

void OnPrebuildWorkerStart() {
  OnBackgroundThreadStart("pipeline prebuild worker");
}

uint32_t PrebuildWorkers(uint32_t renut_count) {
#if REX_PLATFORM_SWITCH
  (void)renut_count;
  return uint32_t(std::clamp<int32_t>(REXCVAR_GET(banjo_prewarm_threads), 1, 3));
#else
  return renut_count;
#endif
}

void StartLongLivedThread(void (*fn)(), size_t stack_bytes) {
#if REX_PLATFORM_SWITCH
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setstacksize(&attr, stack_bytes);
  pthread_t thread;
  const int error = pthread_create(
      &thread, &attr, [](void *arg) -> void * {
        reinterpret_cast<void (*)()>(arg)();
        return nullptr;
      },
      reinterpret_cast<void *>(fn));
  pthread_attr_destroy(&attr);
  if (error) {
    REXLOG_ERROR("[renut] thread creation failed ({}), {} KB stack: running it as a std::thread", error,
                 stack_bytes >> 10);
    new std::thread(fn);
  }
#else
  (void)stack_bytes;
  new std::thread(fn);  // never joined or destroyed (a joinable std::thread destroyed at exit would terminate)
#endif
}

std::filesystem::path NextToExecutable(const std::string &path) {
  const std::filesystem::path p(path);
  return p.is_relative() ? rex::filesystem::GetExecutableFolder() / p : p;
}


void CountSharedConstants(bool reused) {
  Bump(reused ? g_shared_reused : g_shared_uploaded, 1);
}





bool UploadCached() {
#if REX_PLATFORM_SWITCH
  static const bool on = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_upload_cached));
  return on;
#else
  return false;
#endif
}

void BeforeDeviceCreated() {
  // The two rules outside the enhancement table (no master switch), stated once.
  REXLOG_INFO("[renut] banjo_renut_exact_streams {}; banjo_renut_keep_gpu_written {}{}; banjo_renut_memory_policy {}{}",
              REXCVAR_GET(banjo_renut_exact_streams), REXCVAR_GET(banjo_renut_keep_gpu_written),
              REXCVAR_GET(banjo_renut_diag_poison_slots) ? " (TEST: banjo_renut_diag_poison_slots)" : "",
              REXCVAR_GET(banjo_renut_memory_policy),
              REXCVAR_GET(banjo_renut_memory_policy)
                  ? fmt::format(" (textures reserve {} MB, stream cache {} MB and {} s idle, upload trim {} frames)",
                                REXCVAR_GET(banjo_renut_textures_reserve_mb), REXCVAR_GET(banjo_renut_stream_cache_mb),
                                REXCVAR_GET(banjo_renut_stream_cache_idle_s), REXCVAR_GET(banjo_renut_upload_trim_frames))
                  : std::string());
  if (pmull_fingerprint::kAvailable && EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_pmull_hash))) {
    g_pmull_hash = PmullSelfTest();
    REXLOG_INFO("[renut] content hashes: {}", g_pmull_hash ? "PMULL CRC-64 (banjo_renut_pmull_hash)"
                                                           : "XXH3 (the PMULL self-test FAILED)");
  }
  if (!::renut::gpu::IsVulkan())
    return;
  g_precise_barriers_on = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_renut_precise_barriers));
  plume::SetVulkanPreciseBarriers(g_precise_barriers_on);
  // banjo_gpu_robustness_off (package 27, from banjo-nx's renderer): Renut's pipelines without buffer and image
  // robustness, so NVK drops its two descriptor index clamps from every texture handle. Renut's indices are in range by
  // construction: bindless slots below the 8,192 of the table (unused ones point at the null texture), constant
  // registers inside the 4,096 bytes of their uniform buffer (Maxwell's constant banks return 0 past a bank's size in
  // any case), the rest raw buffer-address loads, which robustness does not cover.
  plume::SetVulkanPipelineRobustnessOff(EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_robustness_off)));
  if (!PipelineCacheOn())
    return;
  std::error_code ec;
  const std::filesystem::path folder = rex::filesystem::GetExecutableFolder() / "cache";
  std::filesystem::create_directories(folder, ec);
  const std::filesystem::path path = folder / "renut_pipelines.bin";
  // The shader package in the tag: a new one starts the file over, so entries of shaders no build has any more do not
  // pile up (banjo-nx's own cache once reached 258 MB that way). The driver checks its own version and cache UUID.
  const std::string tag = fmt::format("renut|layout {}|spirv {} {}|entries {}", g_shaderCacheLayout,
                                      g_spirvCacheCompressedSize, g_spirvCacheDecompressedSize,
                                      g_shaderCacheEntryCount);
  plume::ConfigureVulkanPipelineCache(path.string(), tag);
#if defined(__SWITCH__)
  // A save writes nothing, without asking the driver for its data, when every creation since the last one hit.
  plume::SetSwitchPipelineCacheSaveOnMiss(true);
#endif
  REXLOG_INFO("[renut] pipeline cache {} (banjo_renut_pipeline_cache), key \"{}\"", path.string(), tag);
}

void OnDeviceCreated(plume::RenderDevice *device) {
  if (device && ::renut::gpu::IsVulkan()) {
    const bool wanted = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_robustness_off));
    REXLOG_INFO("[renut] pipeline robustness (banjo_gpu_robustness_off): {}",
                !wanted ? "as the device has it (key off)"
                : plume::VulkanPipelineRobustnessAvailable(device) ? "off for the game's pipelines (VK_EXT_pipeline_robustness)"
                                                                   : "as the device has it (no VK_EXT_pipeline_robustness)");
  }
  if (!device || !PipelineCacheOn())
    return;
  g_saver.device = device;
  g_saver.stop = false;
  g_saver.thread = std::thread([] {
#if REX_PLATFORM_SWITCH
    (void)RexSwitchSetCurrentThreadPriorityOk(kBackgroundPriority);
    const int32_t core = REXCVAR_GET(banjo_background_compile_core);
    if (core >= 0)
      (void)RexSwitchPinCurrentThreadToCore(int(core));
#endif
    std::unique_lock lock(g_saver.mutex);
    // banjo_renut_cache_save_quiet (package 28): a save once the compiling has stopped for 15 s (a level loaded, a
    // scene shown), at most once a minute, or after 5 minutes of compiling without such a pause.
    const bool quiet = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_renut_cache_save_quiet));
    uint64_t saved_builds = g_pipelines_built.load(std::memory_order_relaxed);
    int64_t last_save = SteadyMs();
    for (;;) {
      if (g_saver.cv.wait_for(lock, std::chrono::seconds(quiet ? 5 : 60), [] { return g_saver.stop; }))
        return;
      lock.unlock();
      if (!quiet) {
        SaveNow("new pipelines in the last minute");
      } else {
        const uint64_t builds = g_pipelines_built.load(std::memory_order_relaxed);
        const int64_t now = SteadyMs();
        const int64_t since_build = now - g_last_build_ms.load(std::memory_order_relaxed);
        if (builds != saved_builds && now - last_save >= 60000 && (since_build >= 15000 || now - last_save >= 300000)) {
          SaveNow(since_build >= 15000 ? "no pipeline built for 15 s" : "5 minutes of compiling");
          saved_builds = builds;
          last_save = now;
        }
      }
      lock.lock();
    }
  });
}

void BeforeDeviceDestroyed() {
  record::Shutdown();  // banjo-nx (package 28b): the record worker has played everything and stopped
  if (!g_saver.thread.joinable())
    return;
  {
    std::lock_guard lock(g_saver.mutex);
    g_saver.stop = true;
  }
  g_saver.cv.notify_all();
  g_saver.thread.join();
  SaveNow("exit");
}

// banjo-nx (2026-10-19, CPU proposal 5): the per-draw switches, latched once (the cvars are final by the first draw,
// as the function-local statics they replace assumed); the A/B word stays a live relaxed load (a plain load on
// ARM).
Latched g_latched;
std::atomic<bool> g_latched_ready{false};

void LatchTuning() {
  g_latched.bool_spec = ::renut::gpu::UsesVulkanBindings() && EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_bool_spec));
  g_latched.tess_grid = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_tess_grid));
  g_latched.depth_only = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_depth_only));
  g_latched.output_trim = ::renut::gpu::UsesVulkanBindings() && EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_output_trim));
  g_latched.depth_words_view = EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_depth_words_view));
  g_latched.tess_spec = REXCVAR_GET(banjo_deko3d_tess_spec);
  g_latched.alpha_spec = REXCVAR_GET(banjo_deko3d_alpha_spec);
  g_latched.draw_resolves = REXCVAR_GET(banjo_deko3d_draw_resolves);
  g_latched.resolve_skip = REXCVAR_GET(banjo_renut_resolve_skip);
  g_latched.depth_words_lazy = REXCVAR_GET(banjo_renut_depth_words_lazy);
  g_latched.depth_handover = REXCVAR_GET(banjo_deko3d_depth_handover);
  g_latched.depth24 = REXCVAR_GET(banjo_deko3d_depth24);
  g_latched.push_constants = REXCVAR_GET(banjo_deko3d_push_constants);
  g_latched.kick_submit = REXCVAR_GET(banjo_deko3d_kick_submit);
  g_latched.late_input = REXCVAR_GET(banjo_late_input);
  g_latched.ring_cleaner = REXCVAR_GET(banjo_deko3d_ring_cleaner);
  g_latched.push_constants_verify = REXCVAR_GET(banjo_deko3d_push_constants_verify);
  g_latched.diag_gpu_shaders = REXCVAR_GET(banjo_renut_diag_gpu_shaders);
  g_latched.diag_trim_poison = REXCVAR_GET(banjo_renut_diag_trim_poison);
  g_latched.half_pixel_offset = REXCVAR_GET(banjo_renut_half_pixel_offset);
  g_latched.point_minmax_radius = REXCVAR_GET(banjo_renut_point_minmax_radius);
  g_latched.depth_words_stencil = REXCVAR_GET(banjo_renut_depth_words_stencil);
  g_latched.alpha_to_mask_test = REXCVAR_GET(banjo_renut_alpha_to_mask_test);
  g_latched.blend_skip = ::renut::gpu::UsesVulkanBindings() && REXCVAR_GET(banjo_renut_blend_skip) && EnhancementActive(EnhancementGroup::kNonCpu, REXCVAR_GET(banjo_gpu_blend_skip));
  g_latched.exact_streams = REXCVAR_GET(banjo_renut_exact_streams);
  g_latched.keep_gpu_written = REXCVAR_GET(banjo_renut_keep_gpu_written);
  g_latched.poison_slots = REXCVAR_GET(banjo_renut_diag_poison_slots);
  g_latched.shared_dedupe = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_shared_dedupe));
  g_latched.queue_notify_idle = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_queue_notify_idle));
  g_latched.neon_swap = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_neon_swap));
  g_latched.capture_pool = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_capture_pool));
  g_latched.push_point_only = EnhancementActive(EnhancementGroup::kCpu, REXCVAR_GET(banjo_renut_push_point_only));
  g_latched.pipeline_memo = REXCVAR_GET(banjo_renut_pipeline_memo);
  g_latched.cpu_memos = REXCVAR_GET(banjo_renut_cpu_memos);
  g_latched.store_cards = REXCVAR_GET(banjo_renut_store_cards) && g_banjo_store_cards;
  g_latched.store_cards_verify = g_latched.store_cards && REXCVAR_GET(banjo_renut_store_cards_verify);
  g_latched_ready.store(true, std::memory_order_release);
}

}  // namespace banjo::renut_gpu
