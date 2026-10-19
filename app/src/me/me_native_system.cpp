// The native graphics system: an IGraphicsSystem that the app installs in config.graphics, so the Xenos emulation
// plugin is not loaded and no guest memory is watched. The game keeps its own Direct3D; a ring thread reads the PM4
// command list it writes, keeps a copy of the GPU registers, returns the read pointer, delivers interrupts, writes
// SCRATCH_REG/MEM_WRITE/EVENT_WRITE values the game waits for, and counts draws, copies and Swaps. Draws, resolves
// and Swaps are handed to the modules in src/me/native (TargetsNative, DrawsVulkan, ShadersNative)
// with the native shader package (sa2_shaders.mesp next to the executable).
//
// The app uses it when sa2_renderer_native is true (see sa2_app.h).

#include "cpu/sa2_enhancements.h"
#include "me_native_system.h"
#include "sa2_api.h"
#include "sa2_render_mode.h"
#include <rex/platform.h>
#include "me_shader_identity.h"
#include "me_shader_candidate_policy.h"
#include "me_native_draw_extent_estimator.h"
#include "me_pm4_runs.h"
#include "me_object_table.h"
#include "me_record_table.h"
#include "me_native_ps_no_kill.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <thread>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <deque>
#include <map>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <rex/graphics/xenos.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/logging.h>
#include <rex/memory/utils.h>
#include <rex/ppc.h>
#include <rex/system/function_dispatcher.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>
#include <rex/system/xthread.h>
#include <rex/system/xvideo.h>
#include <rex/thread.h>
#include <rex/ui/presenter.h>
#include <rex/ui/vulkan/provider.h>
#include <rex/ui/windowed_app_context.h>
#include <rex/cvar.h>
#if defined(__ARM_NEON)
#include "me_platform_compat.h"  // banjo-nx: NEON on aarch64, SIMDe elsewhere
#endif
#include <rex/sys_counters.h>
#define XXH_INLINE_ALL
#include <xxhash.h>
#include <rex/filesystem.h>
#include <rex/graphics/registers.h>

#include "native/sa2_deferred_recording.h"
#include "native/sa2_native_targets.h"
#include "native/sa2_native_shaders.h"
#include "native/sa2_shader_library.h"

namespace sa2::native {
extern const ShadersNative* g_active_library;  // sa2_native_glue.cpp
}
namespace banjo::renderdoc {
bool Start(const char* path_template);  // banjo_renderdoc.cpp
bool End();
}
extern "C" uint32_t g_banjo_last_reject_cause;  // sa2_native_draws.cpp (C linkage)

// banjo-nx: defined here (masseffect-nx defined it in me_audio_hooks.cpp, which hooks Mass Effect's audio).
REXCVAR_DEFINE_INT32(sa2_diag_ring_swap_delay_us, 0, "SA2",
                     "Test only: the ring thread spins this long at each swap (a slower GPU frame on a fast PC: the 60 -> "
                     "30 fall of fixed vsync, and sa2_vsync_relaxed, can be measured without the console)")
    .range(0, 1000000);
REXCVAR_DEFINE_INT32(sa2_diag_wait_log, 0, "SA2",
                     "Diagnostic: log this many ring WAIT_REG_MEM waits longer than 2 ms (what they poll, the value "
                     "before and after, the vblank and swap counts)")
    .range(0, 100000);
REXCVAR_DEFINE_BOOL(sa2_vblank_sleep_exact, true, "SA2 renderer",
                    "Native renderer: the vblank thread sleeps until the exact next vblank instead of 1 ms steps")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(banjo_renderdoc_capture_s, 0, "Nuts&Bolts/Graphics",
                     "Diagnostic: this many seconds after the renderer starts, ask RenderDoc (when its layer is loaded) "
                     "to capture one frame into capturas/rdc next to the exe (0 = off)");
REXCVAR_DEFINE_INT32(sa2_api_discover_draws, 0, "SA2",
                     "API-level renderer, phase 1 (diagnostic): save this many draws' Direct3D device struct (as the "
                     "game called the draw) beside the register file the ring rebuilt for them, to logs/"
                     "api_discover.bin for tools/d3d_shadow_discover.py. 0 = off")
    .range(0, 100000)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_api_discover_every, 16, "SA2",
                     "API-level renderer discovery: sample one draw in N (spreads the samples over more scenes)")
    .range(1, 1 << 20)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(banjo_trace_frame_s, 0, "Nuts&Bolts/Graphics",
                     "Diagnostic: this many seconds after the renderer starts, log one whole frame: every resolve, "
                     "every render-target change and every draw that samples a resolve destination (0 = off)");
// usf4-nx: keep tracing every frame until this many seconds after the renderer starts (0 = one frame only), to catch
// effects that last a few frames at a time that cannot be predicted to the frame (hit effects).
REXCVAR_DEFINE_INT32(usf4_trace_until_s, 0, "USF4",
                     "Diagnostic: with banjo_trace_frame_s, trace every frame until this many seconds after the "
                     "renderer starts (0 = one frame)");

// Read by Sa2App::PatchCoalescedHashForLocalTesting (sa2_app.h). The Switch has no environment
// variables: debug runs set it in sa2.toml.
REXCVAR_DEFINE_STRING(sa2_coalesced_sha1, "", "SA2 renderer",
                      "Local testing: SHA-1 (40 hex digits) of a modified Coalesced.ini (tools/coalesced.py, e.g. "
                      "no-startup-movies) accepted instead of the disc's; the same as SA2_COALESCED_SHA1")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_shaders_index, true, "SA2 renderer",
                    "If sa2_shaders.mesp.idx matches the package, load only that index (entry table and "
                    "original containers) and read each shader's SPIR-V from the package on first use; otherwise, or "
                    "with false, load the whole package as before")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_repeated_identity, true, "SA2 renderer",
                    "Native renderer: a shader IM_LOAD identical (same address and microcode) to the stage's current "
                    "program keeps its identification instead of hashing and matching again. false = always")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_edram4_several_rect, true, "SA2 renderer",
                    "Native renderer: auto-index rect lists of 2..8 rectangles (D3D Clear of several regions) are "
                    "proven per rectangle, so mode 4 skips the transfers they overwrite. false = only single rects")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
// sa2-nx: on by default on the Switch (the emulated GPU runs at a few frames per second there); on PC the emulated
// GPU stays the default reference (sa2_renderer_native = true in sa2.toml or REX_SA2_RENDERER_NATIVE).
#if REX_PLATFORM_SWITCH
#define SA2_RENDERER_NATIVE_DEFAULT true
#else
#define SA2_RENDERER_NATIVE_DEFAULT false
#endif
// sa2-nx (after futari-nx): the Switch has no environment variables, so the shader dumps that extend the library
// (SA2_SHADER_DISCOVERY, SA2_SHADER_MISSING, SA2_VERTEX_VARIANTS on the PC) also go under one
// folder set here: <folder>/discovery, <folder>/missing and <folder>/variants. The environment variables win.
REXCVAR_DEFINE_STRING(sa2_shader_misses, "", "SA2",
                      "Folder (relative = next to the executable) where the native renderer writes every shader its "
                      "library lacks, once each: discovery/ (raw microcode, input of tools/wrap_discovered.py), "
                      "missing/ (rebuilt containers) and variants/ (vertex shaders as Direct3D patched them, input "
                      "of repair_vertex_variant_declarations). Empty = off")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_renderer_native, SA2_RENDERER_NATIVE_DEFAULT, "SA2 renderer",
                    "Switch: use the native renderer (true) or ReXGlue's Xenos emulation (false, needs "
                    "gpu_plugin = \"xenos\")")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_edram4_stencil_inert, true, "SA2 renderer",
                    "EDRAM mode 4: a draw whose stencil test is ALWAYS and writes nothing counts as stencil-off "
                    "(full depth overwrite proof; lazy stencil kept where it is)");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_center_ogl, true, "SA2 renderer",
                    "EDRAM mode 4 proofs: honor OpenGL pixel centers (PA_SU_VTX_CNTL.pix_center) instead of always "
                    "assuming D3D centers (which lost the first column/row of full-screen passes)");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_ps_no_kill, true, "SA2 renderer",
                    "EDRAM mode 4 proofs: a pixel shader with no KILL anywhere (fetches and branches allowed) cannot "
                    "discard, not only straight-line ALU shaders");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_vs_alu, true, "SA2 renderer",
                    "EDRAM mode 4 proofs: the rectangle estimator admits vertex shaders that compute the position "
                    "with ALU math and (non-relative) constants, run on the live constant bank");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_quad_triangles, true, "SA2 renderer",
                    "EDRAM mode 4 proofs: a full-screen quad drawn as two triangles (6-vertex list / 4-vertex strip) "
                    "that are exactly an axis-aligned rectangle split on a diagonal is proven like a rectangle");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_clip_inside, true, "SA2 renderer",
                    "EDRAM mode 4 proofs: a rectangle drawn with clipping on (no user planes) and every vertex inside "
                    "the clip volume, with an XY viewport transform, is proven like an unclipped screen-space one");
REXCVAR_DEFINE_BOOL(sa2_native_registers_table, true, "SA2 renderer",
                    "Draw records (game thread -> ring) in a fixed bucket table instead of a std::map: no "
                    "allocation per draw under the shared lock")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_fast_registers, true, "SA2 renderer",
                    "Ring: runs of register writes into the constant/fetch range skip the per-word special-register "
                    "checks and bump their generation once per run (exact)");
REXCVAR_DEFINE_BOOL(sa2_native_identity_memo, true, "SA2 renderer",
                    "Shader identification memo by microcode content (type, XXH3, size): a program loaded again "
                    "after others skips the library matching; the draw-time identity guard is unchanged");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_exact_edges, true, "SA2 renderer",
                    "EDRAM mode 4: a proven rectangle whose edges lie exactly on pixel boundaries (D3D pixel "
                    "centers, no window offset, w = 1) bounds the draw by itself, without the 1-pixel pad; the pad "
                    "made syncs transfer whole tile rows/columns the draw never touches");
REXCVAR_DEFINE_BOOL(sa2_native_edram4_overwrite, true, "SA2 renderer",
                    "EDRAM mode 4: skip the ownership transfer of tiles that a proven full-screen rectangle "
                    "overwrites completely (depth ALWAYS without stencil, or opaque full-mask color)");
// Ring-thread CPU switches: every one keeps the old path and checks itself against it for the first
// sa2_native_verify_n uses (a DIFFERENCE line in the log and the old path for the rest of the session).
REXCVAR_DEFINE_BOOL(sa2_native_flat_identity, false, "SA2 renderer",
                    "Ring: shader identity checks (VS fetch-patch match, PS microcode match) memoized in small "
                    "direct-mapped tables keyed by (library entry, load generation) instead of an unordered_map and a "
                    "word-by-word compare per use; exact (pure function of the loaded microcode and the entry)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_fast_pair, false, "SA2 renderer",
                    "Ring: draw/record pairing without the cold cache lines and locks: record start addresses in a "
                    "compact array (the best record of a packet is found in two 32-byte rows), object -> library entry "
                    "through a lock-free table, no draw-queue mutex per unpaired draw, prefetch while draining")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_raw_microcode, false, "SA2 renderer",
                    "Ring: IM_LOAD of the same address and size as the last load of that stage compares the guest "
                    "words against a raw copy of the last load (one memcmp) instead of byte-swapping them first; "
                    "exact (the swap is a bijection)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_BOOL(sa2_native_pm4_fast, false, "SA2 renderer",
                    "Ring: runs of constant/fetch register writes (type 0, SET_CONSTANT2/SET_SHADER_CONSTANTS, "
                    "LOAD_ALU_CONSTANT) are byte-swapped, compared and stored four words at a time with NEON "
                    "straight from the guest words; same register contents and generation bumps")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(sa2_native_verify_n, 2048, "SA2 renderer",
                     "Self-check of the switches above: the old and the new path are both evaluated for the first N "
                     "uses and compared; any difference logs DIFFERENCE and turns that switch off. 0 = no check")
    .range(0, 2147483647)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(sa2_native_vs_identity_tolerance, 0, "SA2 renderer",
                     "Experimental: a vertex shader the ring loaded may differ from its library candidate (same size, "
                     "fetches still checked) in up to N ALU/CF words and still be drawn with that candidate. 0 = "
                     "exact identity (the draw is dropped when no candidate matches, as before). Try 4-16 when the "
                     "log shows 'unresolved draw shader pairs' (e.g. VS CD057930742AFE84, 120 words, one draw per "
                     "frame in the pause menu); see the '[native] VS word diff' lines")
    .range(0, 64)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

// sa2-nx CPU enhancements, round 2 (console profile 2026-10-06: the ring thread ~20 us per draw at stock clocks, the
// main thread ~15 ms per frame). Each is exact: same draws, same identification, same log lines (docs/enhancements.md).
// Round 7 (Banjo's and SASRT's finding, 2026-10-07): the vertex dedupe and the index cache forget what they recorded
// when sa2::native::g_synchronizations_ring changes, the guest synchronization their design relies on, but
// nothing ever advanced it (masseffect-nx's frontend never published its wait generation): ranges above
// sa2_native_dedupe_max_fingerprint were matched by address alone for a whole upload buffer. The ring now
// advances it in stream order at every packet that writes something the game can observe (EVENT_WRITE_SHD fences,
// MEM_WRITE, REG_TO_MEM, COND_WRITE, INTERRUPT): only after one of those may the game rewrite memory an earlier draw
// read. More copies, never a stale one. Key: sa2_native_dedupe_sync_epochs (non-CPU group: a correctness fix).
SA2_ENHANCEMENT(g_sync_epochs, sa2_native_dedupe_sync_epochs, sa2::enh::kNonCpu, true,
                "Ring thread: the vertex dedupe and the index cache forget their recorded ranges at every GPU-to-game "
                "signal in the command stream (fence and memory writes, interrupts), the guest synchronization they "
                "were designed around (its counter was never advanced)");
namespace sa2::native {
extern std::atomic<uint32_t> g_synchronizations_ring;
}  // namespace sa2::native
namespace {
inline void AdvanceSyncEpoch() {
  if (g_sync_epochs.active) {
    sa2::native::g_synchronizations_ring.fetch_add(1, std::memory_order_relaxed);
    g_sync_epochs.UseSampled();
  }
}
}  // namespace
// Round 6: a draw whose shader-identity inputs equal the previous draw's reuses its final VS/PS choice.
SA2_ENHANCEMENT(g_identity_memo, sa2_cpu_ring_identity_memo, sa2::enh::kCpu, true,
                "Ring thread: a directly paired draw with the same shader objects, object-table version and loaded "
                "VS/PS generations as the previous draw reuses its pairing and final VS/PS choice instead of running "
                "the identity checks again (both paths compared in verify mode)");
// Round 6: both shader objects of a draw already finished (NoteShaderObject's lock-free table): both calls skipped.
SA2_ENHANCEMENT(g_note_shader_pair, sa2_cpu_note_shader_pair, sa2::enh::kCpu, true,
                "Game thread: the draw hooks skip NoteShaderObject for a VS/PS pair whose objects are both finished "
                "(two table loads instead of two calls, four times per draw counting the shader-load hook)");
// sa2-nx (2026-10-07, the user's request): relaxed vsync. The emulated display gave the game a vblank every 1/60 s
// whatever it did, and the game waits for a vblank after each swap: a frame that took 17-33 ms waited for the next
// one, so a frame rate just under 60 fell straight to 30, and the game's speed with it (it runs per frame). Now the
// vblank comes when the GPU waits for it, like a variable-refresh display. After each swap the ring waits (WAIT_REG_MEM)
// for a flag in memory to drop to 0, and the game's vblank interrupt callback clears it: the flip. That wait asks for
// the vblank, which comes at once, but never sooner than 1/60 s after the previous one (the 60 cap stays): a 20 ms frame
// is shown after 20 ms (50 fps, the game at 83 % speed) instead of 33 ms (30 fps, 50 %). Which wait is the flip is
// learned at start (the memory a wait polls that keeps ending right after a vblank; PC: 1FCA3004, == 0); until then,
// and whenever nobody asks for kVsyncRelaxedGrace (loading, or a frame slower than 20 fps), the regular ticks. Earlier
// versions failed the PC test with a slow ring (18 ms of ring work per frame; fixed vsync: 30 fps): a vblank at the
// game's VdSwap call, at the ring's XE_SWAP, or for ANY unmet wait (a wait on the game thread mid-frame took the
// vblank before the flag was set, and the flip waited a full refresh more: 13-29 fps).
SA2_ENHANCEMENT(g_vsync_relaxed, sa2_vsync_relaxed, sa2::enh::kNonCpu, true,
                "Relaxed vsync: the guest's vblank follows its swaps (one per swap, at most one per 1/60 s; the regular "
                "1/60 s ticks while it does not swap), so a frame that misses 60 fps is shown when it is ready instead "
                "of at the next vblank: no fall from 60 straight to 30");
// Round 11: the native Direct3D layer's ring of records readied ahead of its head (ApiRecordPublish).
SA2_ENHANCEMENT(g_api_zero_ahead, sa2_api_ring_zero_ahead, sa2::enh::kCpu, true,
                "Native Direct3D layer: free ring space ahead of the record head is allocated as zeroed cache lines "
                "(DC ZVA: no DRAM read before the record's stores) instead of write-prefetched; every zeroed block is "
                "checked to be free space in verify mode");
SA2_ENHANCEMENT(g_note_shader_fast, sa2_cpu_note_shader_fast, sa2::enh::kCpu, true,
                "Main thread, NoteShaderObject (twice per draw): the startup diagnostic stops taking its mutex once its "
                "32 lines are written, and finished objects are remembered in one table shared by the game threads "
                "instead of a thread_local one (no TLS lookup per call)");
SA2_ENHANCEMENT(g_extent_key_int, sa2_cpu_ring_extent_key, sa2::enh::kCpu, true,
                "Ring thread: the bounded-draw extent diagnostic dedupes on its five numbers instead of formatting a "
                "string key on every bounded draw (SA2 never fills its 32 slots, so it formatted forever)");
SA2_ENHANCEMENT(g_estimator_reason, sa2_cpu_ring_estimator_reason, sa2::enh::kCpu, true,
                "Ring thread: the draw-extent estimator keeps its reject reason as a pointer to the literal instead of "
                "a std::string (a heap allocation per call for reasons over 15 characters)");
SA2_ENHANCEMENT(g_rect_list_in_place, sa2_cpu_ring_rect_list_in_place, sa2::enh::kCpu, true,
                "Ring thread: a rect list of several rectangles is proven on the register file itself (the index offset "
                "set per rectangle and restored) instead of on an 80 KB copy of it");
// From nfsmw-nx (nfsmw_relevo_prioridad): on Horizon only priority 0x3B is time-sliced, in 10 ms slices, so the game
// thread that draws can wait a whole slice behind another runnable 0x3B game thread on its core. One step up
// (0x3A) it preempts them instead. Scheduling only: same game, same frames. RISK: recompiled code spin-waits; if
// that thread spins on something a 0x3B thread on the same core must do, that thread waits for a core change. Off
// by default until a console A/B (configs/sa2_test_main_priority_on.toml) says it helps and nothing hangs.
SA2_ENHANCEMENT(g_main_priority, sa2_cpu_main_thread_priority, sa2::enh::kCpu, false,
                "Main thread: the thread that draws runs one priority step above the other game threads (no 10 ms "
                "time-slice waits behind them)");
REXCVAR_DEFINE_INT32(sa2_main_thread_priority, 0x3A, "SA2 enhancements",
                     "Horizon priority for sa2_cpu_main_thread_priority (0x1C-0x3B; the game threads run at 0x3B)")
    .range(0x1C, 0x3B)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
SA2_ENHANCEMENT(g_identity_accept_fast, sa2_cpu_ring_identity_accept_fast, sa2::enh::kCpu, true,
                "Ring thread: a VS candidate whose flat identity memo already says 'matches the loaded program' skips "
                "the mismatch memo's hash-set lookup (it can only hold pairs that did not match)");
// sa2-nx, the direct renderer step 1 (round 5): Direct3D's per-draw state packets reach the ring as a host record (one
// NOP packet in the stream instead of ~14 packets to parse). See namespace direct below.
SA2_ENHANCEMENT(g_direct_state, sa2_cpu_direct_state, sa2::enh::kCpu, true,
                "Direct renderer: Direct3D's per-draw state packets become a host record the ring applies with the same "
                "register functions at the same place in the stream (one NOP packet instead of ~14 to parse; "
                "self-checked)");
// Round 6: the marker's three words are put back once the ring has applied the record (sasrt-nx's write-back).
SA2_ENHANCEMENT(g_direct_restore, sa2_cpu_direct_state_restore, sa2::enh::kCpu, true,
                "Direct renderer: the ring writes the three words the marker NOP replaced back into the command buffer "
                "after applying the record, so the packets are as Direct3D wrote them (a range run twice, or a read "
                "back, sees the original PM4)");
// Round 6: direct-state records by reference (the game thread writes the packet skeleton, not the values).
SA2_ENHANCEMENT(g_direct_refs, sa2_cpu_direct_state_refs, sa2::enh::kCpu, true,
                "Direct renderer: the state record lists each packet's register and where its values are in the "
                "command buffer instead of copying and byte-swapping them on the game thread; the ring reads them in "
                "place (the packets the marker overwrites are still copied)");
// Round 6: the draw's pairing record (the bound shader objects, NoteDrawCall) travels inside its direct-state record
// instead of through the record queue and the ring's address table.
REXCVAR_DEFINE_BOOL(sa2_direct_state_census, false, "SA2",
                    "Diagnostic: which register groups each direct-state record writes (the 10 most common combinations "
                    "every 10 s), to size what a draw that changes only constants could skip");
// sa2-nx, the direct renderer step 2 (2026-10-07): Direct3D's state flush writes the draw's record itself.
SA2_ENHANCEMENT(g_direct_flush, sa2_cpu_direct_flush, sa2::enh::kCpu, true,
                "Direct renderer: inside a converted draw, Direct3D's flush of the shader constants, fetch constants and "
                "register groups writes the draw's record directly instead of PM4 packets the hook decodes again (one "
                "3-word placeholder NOP in the command buffer; self-checked against the original function)");
SA2_ENHANCEMENT(g_direct_pair, sa2_cpu_direct_pair, sa2::enh::kCpu, true,
                "Direct renderer: a converted draw's Draw* record rides in its state record (no queue push on the game "
                "thread, no table lookup on the ring; a submission during the call pushes it as before; checked against "
                "the table in verify mode)");

namespace me::native {
namespace {

namespace xenos = rex::graphics::xenos;
using rex::X_STATUS;
using Clock = std::chrono::steady_clock;

// GPU MMIO window and register file.
constexpr uint32_t kMmioBase = 0x7FC80000;
constexpr uint32_t kMmioMask = 0xFFFF0000;
constexpr uint32_t kMmioSize = 0x0000FFFF;
constexpr uint32_t kRegisterCount = 0x5003;
constexpr uint32_t kRegCpRbWptr = 0x01C5;
constexpr uint32_t kRegScratchUmsk = 0x01DC;
constexpr uint32_t kRegScratchAddr = 0x01DD;
constexpr uint32_t kRegScratch0 = 0x0578;
constexpr uint32_t kRegScratch7 = 0x057F;
constexpr uint32_t kRegCoherStatusHost = 0x0A31;
constexpr uint32_t kRegRbEdramTiming = 0x0F00;
constexpr uint32_t kRegRbBcControl = 0x0F01;
// Display gamma ramp registers. Native presentation owns the Xenos output path, so it must mirror
// the table written by Direct3D just like the SDK command processor does.
constexpr uint32_t kRegGammaFirst = 0x1921;       // DC_LUT_RW_MODE
constexpr uint32_t kRegGammaIndex = 0x1922;       // DC_LUT_RW_INDEX
constexpr uint32_t kRegGammaSequential = 0x1923;  // DC_LUT_SEQ_COLOR
constexpr uint32_t kRegGamma30 = 0x1925;          // DC_LUT_30_COLOR
constexpr uint32_t kRegGammaMask = 0x1927;        // DC_LUT_WRITE_EN_MASK
constexpr uint32_t kRegGammaLast = 0x1927;
constexpr uint32_t kRegD1ModeVCounter = 0x194C;
constexpr uint32_t kRegD1ModeVblankVlineStatus = 0x1951;
constexpr uint32_t kRegD1ModeViewportSize = 0x1961;
constexpr uint32_t kRegVgtEventInitiator = 0x21F9;
constexpr uint32_t kRegVgtDmaBase = 0x21FA;
constexpr uint32_t kRegVgtDmaSize = 0x21FB;
constexpr uint32_t kRegVgtDrawInitiator = 0x21FC;
constexpr uint32_t kRegRbModeControl = 0x2208;
constexpr uint32_t kRegRbSampleCountAddr = 0x2325;
constexpr int kMaxIndirectDepth = 4;
constexpr auto kWaitRegMemMax = std::chrono::milliseconds(200);
constexpr uint16_t kExtentMax = 2048 >> 3;
// Registers whose change invalidates viewport/raster state.
constexpr uint32_t kRegViewportFirst = 0x2080;
constexpr uint32_t kRegViewportLast = 0x2302;
constexpr uint32_t kRegFetchFirst = 0x4800;
constexpr uint32_t kRegFetchLast = 0x48BF;
inline bool IsViewportRegister(uint32_t i) {
  if (i - kRegViewportFirst > kRegViewportLast - kRegViewportFirst) return false;
  return i <= 0x2082 || (i >= 0x210F && i <= 0x2114) || (i >= 0x2204 && i <= 0x2206) || i == 0x2302;
}

// Big-endian words of a command buffer; the ring wraps (power-of-two size), an indirect buffer
// does not.
struct Reader {
  const uint8_t* base = nullptr;
  uint32_t physical = 0;  // physical address of word 0 (to key packets by address)
  uint32_t mask = 0;
  uint32_t pos = 0;
  uint32_t end = 0;
  uint32_t Pending() const { return mask ? ((end - pos) & mask) : (end - pos); }
  uint32_t Peek() const { return rex::memory::load_and_swap<uint32_t>(base + size_t(pos) * 4); }
  uint32_t Read() {
    const uint32_t v = Peek();
    Advance(1);
    return v;
  }
  void Advance(uint32_t words) { pos = mask ? ((pos + words) & mask) : (pos + words); }
};

bool Compare(uint32_t info, uint32_t value, uint32_t ref) {
  switch (info & 0x7) {
    case 0x0: return false;
    case 0x1: return value < ref;
    case 0x2: return value <= ref;
    case 0x3: return value == ref;
    case 0x4: return value != ref;
    case 0x5: return value >= ref;
    case 0x6: return value > ref;
    default: return true;
  }
}

// Physical microcode address -> library entry, filled on the game thread by NoteShaderObject and read
// by the ring thread on IM_LOAD.
std::mutex g_by_address_mutex;
std::unordered_map<uint32_t, const sa2::native::ShaderEntry*> g_by_address;
std::unordered_set<uint32_t> g_objects_seen;
// Immediate loads (IM_LOAD_IMMEDIATE) carry no address: they are found by the XXH3 of the microcode
// (host order) of the objects the game binds, the VS one taken at draw time (already patched). Hashes
// that two different entries share are counted and dropped.
std::unordered_map<uint64_t, const sa2::native::ShaderEntry*> g_by_code;
std::unordered_set<uint32_t> g_code_mapped;  // objects whose (current) microcode is in g_by_code
std::unordered_map<uint32_t, const sa2::native::ShaderEntry*> g_entry_of_object;
uint64_t g_code_conflicts = 0;

// sa2_native_fast_pair: g_entry_of_object (written by the game thread under g_by_address_mutex, never erased)
// mirrored in a lock-free table the ring reads without the mutex (me_object_table.h).
ObjectEntryTable<const sa2::native::ShaderEntry*> g_objects;
// Bumped after every Publish (sa2_cpu_ring_identity_memo: an unchanged version means every object maps to the same entry).
std::atomic<uint64_t> g_objects_version{0};
inline void PublishObject(uint32_t object, const sa2::native::ShaderEntry* entry) {
  g_objects.Publish(object, entry);
  g_objects_version.fetch_add(1, std::memory_order_release);
}
inline bool LookupObject(uint32_t object, const sa2::native::ShaderEntry*& entry) { return g_objects.Lookup(object, entry); }

// Draw records from the game thread (NoteDrawCall).
struct DrawRecord {
  uint32_t type, r5, r6, r7, vs, ps;
  uint32_t start = 0;  // physical address of Direct3D's command buffer write pointer before the call
  int32_t snapshot = -1;  // sa2_api_discover_draws: slot of the device copy taken at the call, -1 = none
  uint32_t device = 0;    // the device the call was made on (r3)
};

// sa2_cpu_direct_state (the direct renderer, step 1: Direct3D's per-draw state without PM4 on the ring).
//
// Direct3D's draw functions flush the dirty state as PM4 packets (about 14 per draw: register runs, constants, fetch
// constants, the occasional shader load, and ~11 alignment fillers the ring stepped over one by one) and then write the
// draw packet. Right after the original draw function returns, the hook (EndDrawDirect) decodes the packets it wrote
// before the draw packet into a host record: register runs already in host byte order, and the few other packets
// (shader loads, waits) as they were. The packets are then covered by one NOP packet whose payload names the record; the
// draw packet stays where it was, and so does every address Direct3D keeps. The ring applies the record when it reaches
// the NOP, with the same register functions the packets would have gone through, at the same place in the stream: the
// same register file at every draw.
//
// Left as it was (and processed as PM4, as before): a draw during which anything was submitted or the command buffer
// changed segment (the ring may already be reading those words), packets the converter does not know, predicated
// packets, a NOP with payload in the first three words (inline data another packet may point at).
//
// Checked: the first 4096 records (all of them in verify mode) are re-encoded and compared with the packets Direct3D
// wrote, word for word (SameAsWritten; DIFFERENCE: no more records), and the ring finds every record at its place in
// the order (a marker without its record: an error, and no more records). Not handled: a command buffer range the GPU
// runs twice (predicated tiling replays; SA2 has none, the report counts them): its second pass would find no record.
namespace direct {
constexpr uint32_t kMagic = 0x53413244u;  // 'SA2D', first payload word of the marker NOP
constexpr uint32_t kRingWords = 1u << 21;  // 8 MB of records (the ring runs a few frames behind)
// kPair (sa2_cpu_direct_pair): the draw's Draw* record (NoteDrawCall) rides in its state record:
// [tag][draw packet's physical address][type r5 r6 r7 vs ps start snapshot device].
// sa2_cpu_direct_state_refs: a packet whose payload the marker does not overwrite is recorded by reference, its values
// read by the ring from the command buffer (intact under the marker NOP until the ring has passed it):
// kRunRef / kRunSingleRef [tag][register][payload word offset], kType1Ref [tag][header][payload word offset],
// kType3Ref [tag][header][header word offset]; offsets in words from the region's first word (the marker's).
// kRestore (sa2_cpu_direct_state_restore): [tag][the region's three original words, guest order]: the ring writes them
// back once the record is applied, so the command buffer is never left changed (a replayed range parses the packets).
enum Op : uint32_t {
  kRun = 1, kRunSingle = 2, kType1 = 3, kType3 = 4, kPair = 5,
  kRunRef = 6, kRunSingleRef = 7, kType1Ref = 8, kType3Ref = 9, kRestore = 10
};
constexpr uint32_t kPairWords = 10;
std::atomic<uint64_t> g_wptr_writes{0};   // guest writes of CP_RB_WPTR (WriteMmio)
// Set by the draw thread while it checks g_wptr_writes and writes a marker; a submission (WriteMmio) waits for it to
// clear before it publishes the write pointer. Either the draw thread sees the submission (and leaves its packets) or
// the submission sees the marker being written (and publishes it whole): seq_cst on both sides.
std::atomic<bool> g_marking{false};
std::unique_ptr<uint32_t[]> g_ring;       // records, word-indexed modulo kRingWords
std::atomic<uint64_t> g_head{0};          // producer: next free word (monotonic)
std::atomic<uint64_t> g_tail{0};          // consumer: first word not yet released
uint64_t g_seq = 0;                       // producer: records written
std::atomic<int32_t> g_checks_left{4096};  // records still carrying the original packets for the ring's check
std::atomic<bool> g_failed{false};        // a check found a difference: no more records
// Statistics (report): converted draws and why the others were left.
std::atomic<uint64_t> g_converted{0}, g_words_covered{0}, g_skip_submitted{0}, g_skip_unknown{0}, g_skip_small{0},
    g_skip_nop_head{0}, g_skip_no_draw{0}, g_skip_segment{0}, g_skip_full{0}, g_checked{0}, g_kick_waits{0},
    g_record_words{0};
std::atomic<uint32_t> g_unknown_opcode{0};  // the last type-3 opcode that stopped a conversion (report)
constexpr uint32_t kPad = 0xFFFFFFFFu;     // the rest of the ring's last words is unused: continue at word 0
// The draw thread's counters: one writer, a plain load and store (no read-modify-write per draw); only a report.
inline void Bump(std::atomic<uint64_t>& c, uint64_t n) { c.store(c.load(std::memory_order_relaxed) + n, std::memory_order_relaxed); }

inline uint8_t* Guest(uint8_t* base, uint32_t address) {
#if REX_PLATFORM_WIN32 || (REX_PLATFORM_MAC && REX_ARCH_ARM64)
  return base + address + (address >= 0xE0000000u ? 0x1000u : 0u);  // the generated code's REX_PHYS_HOST_OFFSET
#else
  return base + address;
#endif
}

// The type-3 packets a record may carry before the draw packet: no effect but their own on the register file and the
// shader state, in stream order (the ring runs them through Type3 with their original words).
inline bool Carried(uint32_t opcode) {
  switch (opcode) {
    case xenos::PM4_NOP:
    case xenos::PM4_IM_LOAD:
    case xenos::PM4_IM_LOAD_IMMEDIATE:
    case xenos::PM4_SET_CONSTANT:
    case xenos::PM4_LOAD_ALU_CONSTANT:
    case xenos::PM4_INVALIDATE_STATE:
    case xenos::PM4_WAIT_REG_MEM:
    case xenos::PM4_REG_RMW:
    case xenos::PM4_EVENT_WRITE:  // a register write (VGT_EVENT_INITIATOR)
      return true;
    default:
      return false;
  }
}
inline bool IsDraw(uint32_t opcode) {
  return opcode == xenos::PM4_DRAW_INDX || opcode == xenos::PM4_DRAW_INDX_2 || opcode == xenos::PM4_DRAW_INDX_BIN ||
         opcode == xenos::PM4_DRAW_INDX_2_BIN;
}

// sa2_cpu_direct_flush (the direct renderer, step 2). Inside a converted draw call, Direct3D's flush functions (the
// hooks, FlushRunsDirect) leave a placeholder NOP [kFlushHeader][kFlushMagic][spot] in the command buffer and keep their
// ops on the drawing thread (FlushState). EndDrawDirect takes them into the call's record at the placeholder's place;
// whatever it does not take (the draw keeps its packets, or this thread submits in the middle of the call) becomes a
// record of its own, named by the placeholder rewritten into a marker (PublishSpots), before the call returns and
// before that submission is published (WriteMmio). From its first placeholder to the end of the call the drawing thread
// holds g_marking, so a submission from another thread waits until every placeholder is published. One drawing thread
// only (g_marking is one flag): a second one turns the native flush off.
constexpr uint32_t kFlushMagic = 0x53413246u;   // "SA2F"
constexpr uint32_t kFlushHeader = 0xC0011000u;  // type-3 NOP, two payload words (three words in all, as a marker)
constexpr uint32_t kFlushWords = 1u << 14;      // the ops of one draw call (64 KB)
constexpr uint32_t kFlushSpots = 32;
struct FlushSpot {
  uint32_t address;     // the placeholder's header (guest)
  uint32_t begin, end;  // its ops in FlushState::ops
  bool taken;           // published (in the call's record, or alone)
};
struct FlushState {
  bool active = false;   // inside a converted draw call
  bool marking = false;  // holds g_marking since the call's first placeholder
  uint32_t device = 0;
  uint64_t submissions = 0;  // g_wptr_writes at BeginDrawDirect
  uint32_t size = 0;         // words of ops
  uint32_t spots = 0;
  uint8_t* base = nullptr;
  FlushSpot spot[kFlushSpots];
  uint32_t ops[kFlushWords];
};
thread_local FlushState* t_flush = nullptr;   // allocated by the drawing thread at its first draw
std::atomic<bool> g_flush_failed{false};      // a self-check found a difference: no more native flushes
std::atomic<uint64_t> g_flush_thread{0};      // the drawing thread (key: the address of its t_flush)
std::atomic<bool> g_flush_multi{false};       // a second drawing thread: no more native flushes
std::atomic<int32_t> g_flush_checks_left{4096};
std::atomic<uint64_t> g_flush_calls[3]{}, g_flush_words{0}, g_flush_slow{0}, g_flush_room{0}, g_flush_original{0},
    g_flush_taken{0}, g_flush_alone{0}, g_flush_own_kicks{0}, g_flush_checked{0};
inline bool FlushHoldsMarking() { return t_flush && t_flush->marking; }

// Every placeholder of this thread's call that is not published yet becomes a record of its own, named by the
// placeholder rewritten into a marker [header][kMagic][seq] (the same three words as EndDrawDirect's marker): the ring
// applies it where the packets would have been. Drawing thread only. The room was checked by FlushRunsDirect (the ring
// only frees words meanwhile).
void PublishSpots(FlushState& f) {
  constexpr uint64_t mask = kRingWords - 1;
  if (!g_ring) [[unlikely]] g_ring = std::make_unique<uint32_t[]>(kRingWords);
  for (uint32_t i = 0; i < f.spots; ++i) {
    FlushSpot& s = f.spot[i];
    if (s.taken) continue;
    const uint32_t need = 2 + (s.end - s.begin);
    const uint64_t pos = g_head.load(std::memory_order_relaxed);
    uint64_t start = pos;
    if ((pos & mask) + need > kRingWords) start = (pos | mask) + 1;  // wrap
    if (start + need - g_tail.load(std::memory_order_acquire) > kRingWords) [[unlikely]] {
      // Cannot happen (the room was checked); the placeholder stays and the ring reports it.
      REXLOG_ERROR("[native] sa2_cpu_direct_flush: no room to publish a placeholder's record; no more records");
      g_failed.store(true, std::memory_order_relaxed);
      g_flush_failed.store(true, std::memory_order_relaxed);
      continue;
    }
    uint32_t* const rec = &g_ring[start & mask];
    std::memcpy(rec + 2, f.ops + s.begin, size_t(s.end - s.begin) * 4);
    if (start != pos) g_ring[pos & mask] = kPad;
    uint32_t seq = uint32_t(++g_seq);
    if (seq == kPad) seq = uint32_t(++g_seq);
    rec[0] = seq;
    rec[1] = need;
    g_head.store(start + need, std::memory_order_release);
    uint8_t* const marker = Guest(f.base, s.address);
    rex::memory::store_and_swap<uint32_t>(marker + 4, kMagic);
    rex::memory::store_and_swap<uint32_t>(marker + 8, seq);
    s.taken = true;
    Bump(g_flush_alone, 1);
  }
}

// WriteMmio, before a submission is published: true when this thread is in the middle of a call that holds g_marking
// (its own submission, made inside Direct3D's draw function): its placeholders are published first, and it must not
// wait for the marking it holds itself.
inline bool FlushOwnKick() {
  FlushState* const f = t_flush;
  if (!f || !f->marking) return false;
  PublishSpots(*f);
  Bump(g_flush_own_kicks, 1);
  return true;
}
}  // namespace direct

// Phase 1 of the API-level renderer (docs/api-renderer-assessment.md): where does Direct3D keep each GPU register?
// For sa2_api_discover_draws draws (one in sa2_api_discover_every), NoteDrawCall copies the device struct as the game
// calls the draw (before Direct3D flushes it into PM4), and the ring, when it pairs that draw, writes the copy next to
// the register file it rebuilt from the PM4 for the same draw: logs/api_discover.bin. tools/d3d_shadow_discover.py
// then finds, for every GPU register, the device offset that holds its value in every sample. Diagnostic only.
namespace discover {
constexpr uint32_t kDeviceBytes = 0x8000;  // past the 2008 XDK's 0x5E00-byte device (UnleashedRecomp-NX's GuestDevice)
constexpr int kSlots = 256;
std::mutex g_mutex;
std::vector<std::vector<uint8_t>> g_slots;  // allocated on the first sample
std::vector<int> g_free;
int g_taken = 0;    // samples copied (game thread, under g_mutex)
int g_written = 0;  // samples written (ring thread)
FILE* g_file = nullptr;

// Game thread (NoteDrawCall): the slot holding this draw's device copy, or -1.
int Take(const uint8_t* base, uint32_t device, uint64_t n) {
  const int32_t want = REXCVAR_GET(sa2_api_discover_draws);
  if (n % uint64_t(REXCVAR_GET(sa2_api_discover_every))) return -1;
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_taken >= want) return -1;
  if (g_slots.empty()) {
    g_slots.assign(kSlots, std::vector<uint8_t>(kDeviceBytes));
    for (int i = kSlots - 1; i >= 0; --i) g_free.push_back(i);
  }
  if (g_free.empty()) return -1;  // the ring is far behind, or records were dropped unpaired: skip this draw
  const int slot = g_free.back();
  g_free.pop_back();
  std::memcpy(g_slots[slot].data(), base + device, kDeviceBytes);
  ++g_taken;
  return slot;
}

// Ring thread (PairByAddress): one sample = a 56-byte header, the register file (host order), the device copy (guest
// order, big-endian). Header words: 'SA2D', version 1, sample, type, r5, r6, r7, vs, ps, record start, packet
// address, device, register count, device bytes.
void Write(const DrawRecord& r, const uint32_t* registers, uint32_t count, uint32_t packet) {
  std::lock_guard<std::mutex> lock(g_mutex);
  const int32_t want = REXCVAR_GET(sa2_api_discover_draws);
  if (!g_file && g_written < want) {
    const auto folder = rex::filesystem::GetExecutableFolder() / "logs";
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const auto path = folder / "api_discover.bin";
    g_file = std::fopen(path.string().c_str(), "wb");
    if (!g_file) REXLOG_ERROR("[discover] cannot write {}", path.string());
  }
  if (g_file) {
    const uint32_t header[14] = {0x53413244u, 1u, uint32_t(g_written), r.type, r.r5, r.r6, r.r7, r.vs, r.ps,
                                 r.start, packet, r.device, count, kDeviceBytes};
    std::fwrite(header, sizeof(header), 1, g_file);
    std::fwrite(registers, sizeof(uint32_t), count, g_file);
    std::fwrite(g_slots[size_t(r.snapshot)].data(), 1, kDeviceBytes, g_file);
    ++g_written;
    if (g_written % 500 == 0 || g_written >= want) {
      REXLOG_INFO("[discover] {} of {} samples written (logs/api_discover.bin)", g_written, want);
    }
    if (g_written >= want) {
      std::fclose(g_file);
      g_file = nullptr;
    }
  }
  g_free.push_back(r.snapshot);
}
}  // namespace discover
// The ring runs several frames behind the game thread (~800 records measured), so records are never aged
// out; the window search drops the ones skipped when a later one matches.

// Index/vertex count the ring carries for `prims` primitives of a Xenos primitive type
// (sub_82228568 takes a primitive count and multiplies it, mullw at +175).
uint32_t CountForPrimitives(uint32_t type, uint32_t prims) {
  switch (type) {
    case 1: return prims;             // point list
    case 2: return prims * 2;         // line list
    case 3: return prims + 1;         // line strip
    case 4: return prims * 3;         // triangle list
    case 5: case 6: return prims + 2; // triangle fan, strip
    case 8: return prims * 3;         // rectangle list
    case 13: return prims * 4;        // quad list
    default: return prims;
  }
}
std::mutex g_draw_mutex;
std::deque<DrawRecord> g_draw_queue;
// The same records keyed by where Direct3D was about to write (device+48, the command buffer write
// pointer): the ring finds the record of a DRAW_INDX by its packet address, whatever the order.
std::map<uint32_t, DrawRecord> g_draw_by_address;
// The same index without allocations (sa2_native_registers_table): 8 KB address buckets of 8 records;
// a lookup checks the packet's bucket and the previous one (records are at most 0x2000 below the packet).
// (me_record_table.h: the buckets, with a dense array of start addresses beside the slots for sa2_native_fast_pair.)
DrawRecordTable<DrawRecord> g_records;
using DrawRecordSlot = DrawRecordTable<DrawRecord>::Slot;
bool RecordTable() {
  static const bool table = REXCVAR_GET(sa2_native_registers_table);
  return table;
}
// Draw records from the game thread to the ring thread without the shared mutex (the record table is then
// owned by the ring thread alone): a bounded multi-producer queue (Vyukov) of (record, start). The ring
// drains it into g_record_table before every lookup. The ring runs a few frames behind (~800 records),
// so 16384 slots never fill in practice; a full queue makes the producer wait.
struct QueuedRecord {
  std::atomic<uint64_t> sequence{0};
  DrawRecord record;
  uint32_t start = 0;
};
constexpr uint64_t kRecordQueueSize = 16384;
std::vector<QueuedRecord> g_record_queue(kRecordQueueSize);
std::atomic<uint64_t> g_record_queue_tail{0};  // next slot to fill (producers)
uint64_t g_record_queue_head = 0;               // next slot to drain (ring thread only)
bool g_record_queue_ready = [] {
  for (uint64_t i = 0; i < kRecordQueueSize; ++i) g_record_queue[i].sequence.store(i, std::memory_order_relaxed);
  return true;
}();

bool g_record_fast = false;  // sa2_native_fast_pair (set when the graphics system starts)

void InsertRecordInTable(DrawRecord&& record, uint32_t start) {
  g_records.Insert(std::move(record), start, g_record_fast);
}

void PushRecord(DrawRecord&& record, uint32_t start) {
  uint64_t pos = g_record_queue_tail.load(std::memory_order_relaxed);
  for (;;) {
    QueuedRecord& q = g_record_queue[pos & (kRecordQueueSize - 1)];
    const uint64_t seq = q.sequence.load(std::memory_order_acquire);
    if (seq == pos) {
      if (g_record_queue_tail.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
        q.record = std::move(record);
        q.start = start;
        q.sequence.store(pos + 1, std::memory_order_release);
        return;
      }
    } else if (seq < pos) {
      std::this_thread::yield();  // full: the ring thread has not drained yet
      pos = g_record_queue_tail.load(std::memory_order_relaxed);
    } else {
      pos = g_record_queue_tail.load(std::memory_order_relaxed);
    }
  }
}

// Ring thread: moves every queued record into the table, in order.
void DrainRecords() {
  for (;;) {
    QueuedRecord& q = g_record_queue[g_record_queue_head & (kRecordQueueSize - 1)];
    if (q.sequence.load(std::memory_order_acquire) != g_record_queue_head + 1) return;
    if (g_record_fast) {  // the next records were written by the game thread's core: start pulling their lines
      __builtin_prefetch(&g_record_queue[(g_record_queue_head + 1) & (kRecordQueueSize - 1)]);
      __builtin_prefetch(&g_record_queue[(g_record_queue_head + 2) & (kRecordQueueSize - 1)]);
    }
    InsertRecordInTable(std::move(q.record), q.start);
    q.sequence.store(g_record_queue_head + kRecordQueueSize, std::memory_order_release);
    ++g_record_queue_head;
  }
}

// sa2_cpu_direct_pair: the Draw* record of a call in progress, held back from the record queue (one slot per drawing
// thread). It ends exactly once: claimed by EndDrawDirect for the call's direct record (kTaken), or pushed to the queue,
// by EndDrawDirect when the draw keeps its packets or by a submission made during the call (WriteMmio pushes every
// pending record before it publishes the write pointer, so the ring never sees a draw before its record).
namespace direct {
struct PendingPair {
  static constexpr uint32_t kNone = 0, kPending = 1, kTaken = 2, kPushed = 3;
  std::atomic<uint32_t> state{kNone};
  DrawRecord record{};
  uint32_t start = 0;
};
constexpr int kPendingSlots = 8;
std::atomic<PendingPair*> g_pending_slots[kPendingSlots];
std::atomic<int> g_pending_count{0};
std::atomic<uint64_t> g_pairs_carried{0}, g_pairs_pushed_late{0}, g_pairs_pushed_at_submission{0};
thread_local PendingPair* t_pending = nullptr;
thread_local bool t_pending_full = false;
// A submission from a thread that never drew while records were held back: Direct3D's device is used by one thread,
// so this should never happen; if it does, the held-back records stop (sa2_cpu_direct_pair off from then on).
std::atomic<bool> g_pair_off{false};
// This thread's slot (registered on first use; nullptr when 8 threads already draw: their records go through the queue).
inline PendingPair* ThreadPending() {
  if (t_pending || t_pending_full) [[likely]] return t_pending;
  const int i = g_pending_count.fetch_add(1, std::memory_order_acq_rel);
  if (i >= kPendingSlots) {
    t_pending_full = true;
    return nullptr;
  }
  t_pending = new PendingPair();  // lives as long as the process (the registry keeps it)
  g_pending_slots[i].store(t_pending, std::memory_order_release);
  return t_pending;
}
// A submission (WriteMmio, before it publishes the write pointer): every pending record goes to the queue now.
inline void FlushPendingPairs() {
  const int n = std::min(g_pending_count.load(std::memory_order_acquire), kPendingSlots);
  if (!t_pending && !t_pending_full && !g_pair_off.exchange(true, std::memory_order_relaxed))
    REXLOG_WARN("[native] sa2_cpu_direct_pair: a submission came from a thread that does not draw; draw records go "
                "through the queue from now on");
  for (int i = 0; i < n; ++i) {
    PendingPair* p = g_pending_slots[i].load(std::memory_order_acquire);
    uint32_t expected = PendingPair::kPending;
    if (p && p->state.load(std::memory_order_acquire) == PendingPair::kPending &&
        p->state.compare_exchange_strong(expected, PendingPair::kPushed, std::memory_order_acq_rel)) {
      PushRecord(std::move(p->record), p->start);
      g_pairs_pushed_at_submission.fetch_add(1, std::memory_order_relaxed);
    }
  }
}
}  // namespace direct
std::atomic<bool> g_native_active{false};
// The ring's read position after its last batch (RingIdle) and the write pointer the game last published.
std::atomic<uint32_t> g_ring_processed{0}, g_ring_published{0};
// XE_SWAP packets the ring has presented (PresentedSwaps: lockstep video captures), and the app's hook called right
// after each one (SetPresentHook).
std::atomic<uint64_t> g_presented_swaps{0};
std::atomic<void (*)(uint64_t)> g_present_hook{nullptr};
std::atomic<uint64_t> g_draw_records{0};

static const char* ShaderDumpFolder(const char* environment, const char* sub);  // sa2-nx, below

uint64_t CodeHash(const uint8_t* big_endian, uint32_t words) {
  std::vector<uint32_t> host(words);
  for (uint32_t i = 0; i < words; ++i) host[i] = rex::memory::load_and_swap<uint32_t>(big_endian + i * 4);
  return XXH3_64bits(host.data(), host.size() * sizeof(uint32_t));
}

// Whole-library index by unmasked microcode (host order), unique hashes only: for immediate loads of
// shaders the game never binds through the device (Direct3D's own clear/copy shaders).
std::unordered_map<uint64_t, const sa2::native::ShaderEntry*> g_library_by_code;

bool OriginalMicrocode(const sa2::native::ShaderEntry& entry, uint32_t& virtual_size,
                       uint32_t& offset, uint32_t& bytes, std::vector<uint32_t>* words = nullptr) {
  if (!entry.shader || entry.shader->original.size() < 28) return false;
  const auto& o = entry.shader->original;
  virtual_size = rex::memory::load_and_swap<uint32_t>(o.data() + 4);
  const uint32_t shader_header = rex::memory::load_and_swap<uint32_t>(o.data() + 24);
  if (virtual_size > o.size() || shader_header + 8 > virtual_size) return false;
  offset = rex::memory::load_and_swap<uint32_t>(o.data() + shader_header);
  bytes = rex::memory::load_and_swap<uint32_t>(o.data() + shader_header + 4);
  if (!bytes || (bytes & 3) || size_t(virtual_size) + offset + bytes > o.size()) return false;
  if (words) {
    words->resize(bytes / 4);
    for (size_t i = 0; i < words->size(); ++i) {
      (*words)[i] = rex::memory::load_and_swap<uint32_t>(o.data() + virtual_size + offset + i * 4);
    }
  }
  return true;
}

// Direct3D's clear/copy/video shaders are loaded with IM_LOAD_IMMEDIATE after link-time rewriting.
// Their original containers are in the library, but the exact ring microcode is not. Materialize a
// stable container from the closest register-linked template so the normal offline translator can add
// the variant on the next run. Package shaders (with a constant table) are deliberately not candidates.
void DumpMissingImmediate(const sa2::native::ShadersNative& library, bool vertex,
                          const std::vector<uint32_t>& microcode) {
  const char* folder = ShaderDumpFolder("SA2_SHADER_MISSING", "missing");
  if (!folder || !*folder || microcode.empty()) return;
  const uint64_t code_hash = XXH3_64bits(microcode.data(), microcode.size() * sizeof(uint32_t)) ^
                             (vertex ? 1ull : 0ull);
  static std::mutex mutex;
  static std::unordered_set<uint64_t> dumped;
  std::lock_guard<std::mutex> lock(mutex);
  if (!dumped.insert(code_hash).second) return;

  const sa2::native::ShaderEntry* best = nullptr;
  uint64_t best_score = UINT64_MAX;
  uint32_t best_vsize = 0, best_offset = 0, best_bytes = 0, best_different = 0;
  for (uint32_t n = 0;; ++n) {
    const auto* candidate = library.PerNumber(n);
    if (!candidate) break;
    if (candidate->vertices != vertex || !candidate->binding_per_register) continue;
    uint32_t vsize = 0, offset = 0, bytes = 0;
    std::vector<uint32_t> original;
    if (!OriginalMicrocode(*candidate, vsize, offset, bytes, &original) ||
        original.size() != microcode.size()) {
      continue;
    }
    uint32_t different = 0;
    uint64_t bits = 0;
    for (size_t i = 0; i < original.size(); ++i) {
      const uint32_t delta = original[i] ^ microcode[i];
      different += delta != 0;
      bits += std::popcount(delta);
    }
    const uint64_t score = uint64_t(different) * 1000 + bits;
    if (score < best_score) {
      best = candidate;
      best_score = score;
      best_vsize = vsize;
      best_offset = offset;
      best_bytes = bytes;
      best_different = different;
    }
  }
  if (!best || !best->shader) {
    REXLOG_WARN("[native] no internal {} template for an immediate shader of {} words",
                vertex ? "VS" : "PS", microcode.size());
    return;
  }
  std::vector<uint8_t> container = best->shader->original;
  for (size_t i = 0; i < microcode.size(); ++i) {
    rex::memory::store_and_swap<uint32_t>(container.data() + best_vsize + best_offset + i * 4,
                                          microcode[i]);
  }
  std::error_code ec;
  std::filesystem::create_directories(folder, ec);
  const uint64_t h = XXH3_64bits(container.data(), container.size());
  const auto path = std::filesystem::path(folder) /
                    fmt::format("{}_{:016x}.bin", vertex ? "vs" : "ps", h);
  if (std::FILE* f = std::fopen(path.string().c_str(), "wb")) {
    std::fwrite(container.data(), 1, container.size(), f);
    std::fclose(f);
    REXLOG_INFO("[native] immediate {} variant kept: {} (template n{}, {} of {} words changed)",
                vertex ? "VS" : "PS", path.string(), best->number, best_different,
                best_bytes / 4);
  }
}

// sa2-nx: the folder of one shader dump: the environment variable on the PC, else <sa2_shader_misses>/<sub>
// (relative to the executable). Empty = that dump is off. The returned string lives for the whole run.
static const char* ShaderDumpFolder(const char* environment, const char* sub) {
  static std::mutex mutex;
  static std::unordered_map<std::string, std::string> folders;
  std::lock_guard<std::mutex> lock(mutex);
  auto [it, inserted] = folders.try_emplace(environment);
  if (inserted) {
    if (const char* env = std::getenv(environment); env && *env) {
      it->second = env;
    } else if (const std::string base = REXCVAR_GET(sa2_shader_misses); !base.empty()) {
      std::filesystem::path path(base);
      if (path.is_relative()) path = rex::filesystem::GetExecutableFolder() / path;
      it->second = (path / sub).string();
    }
  }
  return it->second.c_str();
}

// Lossless discovery artifact for every shader which the current library cannot identify. Unlike
// DumpMissingImmediate this deliberately does not borrow a container or guess an identity: the file
// is the exact big-endian microcode from the ring plus a small human-readable sidecar. It is input
// for instruction-aware classification of Direct3D's internal shaders.
void DumpRawMicrocode(bool vertex, uint32_t address, const std::vector<uint32_t>& microcode) {
  static const char* const folder = ShaderDumpFolder("SA2_SHADER_DISCOVERY", "discovery");
  if (!folder || !*folder || microcode.empty()) return;
  const uint64_t hash = XXH3_64bits(microcode.data(), microcode.size() * sizeof(uint32_t));
  const uint64_t key = hash ^ (vertex ? 1ull : 0ull);
  static std::mutex mutex;
  static std::unordered_set<uint64_t> dumped;
  std::lock_guard<std::mutex> lock(mutex);
  if (!dumped.insert(key).second) return;

  std::error_code ec;
  std::filesystem::create_directories(folder, ec);
  const std::string stem = fmt::format("{}_{:016x}", vertex ? "vs" : "ps", hash);
  const auto raw_path = std::filesystem::path(folder) / (stem + ".ucode");
  if (std::FILE* f = std::fopen(raw_path.string().c_str(), "wb")) {
    for (uint32_t word : microcode) {
      uint8_t bytes[4] = {uint8_t(word >> 24), uint8_t(word >> 16), uint8_t(word >> 8),
                          uint8_t(word)};
      std::fwrite(bytes, 1, sizeof(bytes), f);
    }
    std::fclose(f);
  }
  const auto text_path = std::filesystem::path(folder) / (stem + ".txt");
  if (std::FILE* f = std::fopen(text_path.string().c_str(), "w")) {
    std::fprintf(f, "stage=%s\naddress=%08X\nwords=%zu\nhash=%016llX\n", vertex ? "vs" : "ps",
                 address, microcode.size(), static_cast<unsigned long long>(hash));
    for (size_t i = 0; i < microcode.size(); ++i) std::fprintf(f, "%04zu %08X\n", i, microcode[i]);
    std::fclose(f);
  }
  REXLOG_INFO("[native] raw unidentified {} kept: {} ({} words)", vertex ? "VS" : "PS",
              raw_path.string(), microcode.size());
}

void BuildLibraryCodeIndex(const sa2::native::ShadersNative& library) {
  std::unordered_set<uint64_t> shared;
  auto register_link_score = [](const sa2::native::ShaderEntry* e) {
    if (!e || !e->binding_per_register || !e->shader) return uint32_t(0);
    // Current raw-PS wrappers carry all 16 register-linked interpolator mappings after the
    // 32-byte shader header. Older discovery wrappers stopped at the header, so XenosRecomp
    // initialized the missing inputs to zero (in particular UI alpha). Prefer the complete
    // wrapper when several containers contain byte-for-byte identical microcode.
    if (!e->vertices && e->shader->original.size() >= 28) {
      const auto& o = e->shader->original;
      const uint32_t vsize = rex::memory::load_and_swap<uint32_t>(o.data() + 4);
      const uint32_t sh = rex::memory::load_and_swap<uint32_t>(o.data() + 24);
      if (sh <= vsize && vsize - sh >= 32 + 16 * 4) return uint32_t(3);
    }
    return uint32_t(2);
  };
  for (uint32_t n = 0;; ++n) {
    const sa2::native::ShaderEntry* e = library.PerNumber(n);
    if (!e) break;
    if (!e->shader) continue;
    const auto& o = e->shader->original;
    // sa2-nx: which container each entry number is (XXH3 of the container = the vs_/ps_<hash>.bin name of the
    // dumped ones), to read the n<N> of the logs.
    REXLOG_INFO("[native] library n{}: {}{} container {:016x} ({} bytes)", n, e->vertices ? "VS" : "PS",
                e->binding_per_register ? " register-linked" : "", XXH3_64bits(o.data(), o.size()), o.size());
    if (o.size() < 28) continue;
    const uint32_t vsize = rex::memory::load_and_swap<uint32_t>(o.data() + 4);
    const uint32_t sh = rex::memory::load_and_swap<uint32_t>(o.data() + 24);
    if (sh + 8 > vsize || vsize > o.size()) continue;
    const uint32_t off = rex::memory::load_and_swap<uint32_t>(o.data() + sh);
    const uint32_t bytes = rex::memory::load_and_swap<uint32_t>(o.data() + sh + 4);
    if (size_t(vsize) + off + bytes > o.size() || !bytes) continue;
    const uint64_t h = CodeHash(o.data() + vsize + off, bytes / 4) ^ (e->vertices ? 1 : 0);
    auto [slot, inserted] = g_library_by_code.emplace(h, e);
    if (!inserted) {
      shared.insert(h);
      if (register_link_score(e) > register_link_score(slot->second)) slot->second = e;
    }
  }
  // Hashes several containers share are kept (first entry): identical microcode reads the same
  // constant and fetch registers, only the constant table names differ.
  REXLOG_INFO("[native] library microcode index: {} entries, {} shared by more than one container",
              g_library_by_code.size(), shared.size());
}

const sa2::native::ShaderEntry* ByCode(const std::vector<uint32_t>& microcode, bool vertex,
    const sa2::native::ShaderEntry** observed = nullptr) {
  const uint64_t h = XXH3_64bits(microcode.data(), microcode.size() * sizeof(uint32_t));
  const sa2::native::ShaderEntry* entry = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_by_address_mutex);
    auto it = g_by_code.find(h);
    if (it != g_by_code.end()) entry = it->second;
  }
  if (!entry) {
    auto it = g_library_by_code.find(h ^ (vertex ? 1 : 0));
    if (it != g_library_by_code.end()) entry = it->second;
  }
  if (observed) *observed = entry;  // Preserve rejected candidates for bounded diagnostics.
  if (!entry) return nullptr;
  // VS fetch-patch compatibility remains checked by IdentifyShader, unchanged.
  if (vertex) return entry->vertices ? entry : nullptr;
  if (!PixelShaderIdentityMatches(
      {ShaderIdentityStage::Pixel, microcode},
      {entry->vertices ? ShaderIdentityStage::Vertex : ShaderIdentityStage::Pixel,
       entry->microcode})) return nullptr;
  return entry;
}
std::atomic<const sa2::native::ShadersNative*> g_library{nullptr};
// Vertex shaders seen only after Direct3D patched their microcode (sub_8222F1C8) are found by the
// virtual part of their container (header, constant table, vertex elements); hashes shared by more
// than one library entry are left out.
std::unordered_map<uint64_t, const sa2::native::ShaderEntry*> g_vs_by_virtual;
bool g_vs_by_virtual_built = false;

void BuildVirtualIndex(const sa2::native::ShadersNative* library) {
  std::unordered_set<uint64_t> shared;
  for (uint32_t n = 0;; ++n) {
    const sa2::native::ShaderEntry* e = library->PerNumber(n);
    if (!e) break;
    if (!e->vertices || !e->shader || e->shader->original.size() < 8) continue;
    const auto& o = e->shader->original;
    const uint32_t vsize = rex::memory::load_and_swap<uint32_t>(o.data() + 4);
    if (vsize > o.size()) continue;
    const uint64_t h = XXH3_64bits(o.data(), vsize);
    if (!g_vs_by_virtual.emplace(h, e).second) shared.insert(h);
  }
  for (uint64_t h : shared) g_vs_by_virtual.erase(h);
  g_vs_by_virtual_built = true;
}

const sa2::native::ShaderEntry* ByAddress(uint32_t physical) {
  std::lock_guard<std::mutex> lock(g_by_address_mutex);
  auto it = g_by_address.find(physical);
  return it == g_by_address.end() ? nullptr : it->second;
}

class NativeGraphicsSystem final : public rex::system::IGraphicsSystem {
 public:
  NativeGraphicsSystem() : registers_(kRegisterCount, 0) {}
  ~NativeGraphicsSystem() override { Shutdown(); }

  X_STATUS SetupPresentation(rex::ui::WindowedAppContext* app_context) override {
    if (presenter_) return X_STATUS_SUCCESS;
    app_context_ = app_context;
    if (!provider_) {
      // The XenosRecomp SPIR-V needs device features that must be requested at creation.
      rex::cvar::SetFlagByName("vulkan_native_shader_features", "true");
      provider_ = rex::ui::vulkan::VulkanProvider::Create(true, true);
      if (!provider_) {
        REXLOG_ERROR("[native] could not create the Vulkan device");
        return X_STATUS_UNSUCCESSFUL;
      }
    }
    auto create = [this]() { presenter_ = provider_->CreatePresenter(); };
    if (app_context_) {
      app_context_->CallInUIThreadSynchronous(create);
    } else {
      create();
    }
    if (!presenter_) {
      REXLOG_ERROR("[native] could not create the presenter");
      return X_STATUS_UNSUCCESSFUL;
    }
    REXLOG_INFO("[native] native graphics system: no Xenos emulation");
    return X_STATUS_SUCCESS;
  }

  X_STATUS SetupGuestGpu(rex::runtime::FunctionDispatcher* dispatcher,
                         rex::system::KernelState* kernel_state) override {
    dispatcher_ = dispatcher;
    kernel_state_ = kernel_state;
    memory_ = dispatcher->memory();
    if (!mmio_registered_) {
      mmio_registered_ = memory_->AddVirtualMappedRange(kMmioBase, kMmioMask, kMmioSize, this,
                                                        &MmioRead, &MmioWrite);
      if (!mmio_registered_) {
        REXLOG_ERROR("[native] could not register the GPU MMIO range");
        return X_STATUS_UNSUCCESSFUL;
      }
    }
    if (active_.exchange(true)) return X_STATUS_SUCCESS;
    g_vs_identity_alu_tolerance.store(uint32_t(REXCVAR_GET(sa2_native_vs_identity_tolerance)),
                                      std::memory_order_relaxed);
    identity_flat_ = REXCVAR_GET(sa2_native_flat_identity);
    identity_accept_fast_ = g_identity_accept_fast.active;
    g_record_fast = pairing_fast_ = REXCVAR_GET(sa2_native_fast_pair);
    raw_microcode_ = REXCVAR_GET(sa2_native_raw_microcode);
    pm4_fast_ = REXCVAR_GET(sa2_native_pm4_fast);
    check_identity_ = check_pairing_ = check_raw_ = check_pm4_ = check_objects_ =
        uint32_t(std::max<int32_t>(0, REXCVAR_GET(sa2_native_verify_n)));
    if (identity_flat_ || pairing_fast_ || raw_microcode_ || pm4_fast_)
      REXLOG_INFO("[native] ring CPU switches: flat identity {}, fast pairing {}, raw microcode compare {}, fast PM4 "
                  "runs {}; self-check of the first {} uses of each", identity_flat_, pairing_fast_, raw_microcode_,
                  pm4_fast_, check_identity_);
    g_native_active.store(true, std::memory_order_release);
    const char* lib = std::getenv("SA2_SHADER_LIBRARY");
    const auto lib_path = lib && *lib ? std::filesystem::path(lib)
                                      : rex::filesystem::GetExecutableFolder() / "sa2_shaders.mesp";  // sa2-nx
    if (shaders_.Load(lib_path, REXCVAR_GET(sa2_shaders_index))) {
      sa2::native::g_active_library = &shaders_;
      BuildLibraryCodeIndex(shaders_);
      g_library.store(&shaders_, std::memory_order_release);
      REXLOG_INFO("[native] shader library {}", lib_path.string());
    } else {
      REXLOG_WARN("[native] no shader library at {}: nothing will be drawn", lib_path.string());
    }
    start_ = last_report_ = Clock::now();
    vblank_thread_ = rex::system::object_ref<rex::system::XHostThread>(new rex::system::XHostThread(
        kernel_state_, 128 * 1024, 0, [this]() { return VblankLoop(); }));
    vblank_thread_->set_name("GPU VSync native");
    vblank_thread_->Create();
    ring_thread_ = rex::system::object_ref<rex::system::XHostThread>(new rex::system::XHostThread(
        kernel_state_, 128 * 1024, 0, [this]() { return RingLoop(); }));
    ring_thread_->set_name("GPU ring native");
    ring_thread_->Create();
    return X_STATUS_SUCCESS;
  }

  bool has_presentation() const override { return presenter_ != nullptr; }
  rex::ui::GraphicsProvider* provider() const override { return provider_.get(); }
  rex::ui::Presenter* presenter() const override { return presenter_.get(); }

  void SetInterruptCallback(uint32_t callback, uint32_t user_data) override {
    callback_data_.store(user_data, std::memory_order_release);
    callback_.store(callback, std::memory_order_release);
    REXLOG_INFO("[native] interrupt callback {:08X} ({:08X})", callback, user_data);
  }

  void InitializeRingBuffer(uint32_t ptr, uint32_t size_log2) override {
    ring_words_.store(uint32_t(1) << (size_log2 + 1), std::memory_order_release);
    ring_base_.store(ptr, std::memory_order_release);
    ring_generation_.fetch_add(1, std::memory_order_acq_rel);
    REXLOG_INFO("[native] ring at {:08X}, {} words", ptr, uint32_t(1) << (size_log2 + 1));
  }

  void EnableReadPointerWriteBack(uint32_t ptr, uint32_t block_size_log2) override {
    (void)block_size_log2;
    read_writeback_.store(ptr, std::memory_order_release);
  }

  void Shutdown() override {
    g_native_active.store(false, std::memory_order_release);
    if (active_.exchange(false)) {
      { std::lock_guard<std::mutex> lock(ring_mutex_); }
      ring_cv_.notify_all();
      if (ring_thread_) {
        ring_thread_->Wait(0, 0, 0, nullptr);
        ring_thread_.reset();
      }
      if (vblank_thread_) {
        vblank_thread_->Wait(0, 0, 0, nullptr);
        vblank_thread_.reset();
      }
      Report(true);
    }
    draw_extent_estimator_.reset();
    extent_diagnostics_.clear();
    extent_keys_.clear();
    targets_.reset();
    if (presenter_) {
      if (app_context_) app_context_->CallInUIThreadSynchronous([this]() { presenter_.reset(); });
      presenter_.reset();
    }
    provider_.reset();
  }

 private:
  // --- MMIO ---
  static uint32_t MmioRead(void*, void* ctx, uint32_t address) {
    return static_cast<NativeGraphicsSystem*>(ctx)->ReadMmio(address);
  }
  static void MmioWrite(void*, void* ctx, uint32_t address, uint32_t value) {
    static_cast<NativeGraphicsSystem*>(ctx)->WriteMmio(address, value);
  }

  uint32_t ReadMmio(uint32_t address) {
    const uint32_t r = (address & 0xFFFF) / 4;
    switch (r) {
      case kRegRbEdramTiming: return 0x08100748;
      case kRegRbBcControl: return 0x0000200E;
      case kRegD1ModeVCounter: {
        rex::system::X_VIDEO_MODE mode;
        rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
        return std::min(uint32_t(mode.display_height), uint32_t(0x0FFF));
      }
      case kRegD1ModeVblankVlineStatus: return 1;
      case kRegD1ModeViewportSize: {
        rex::system::X_VIDEO_MODE mode;
        rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
        return (std::min(uint32_t(mode.display_width), uint32_t(0x0FFF)) << 16) |
               std::min(uint32_t(mode.display_height), uint32_t(0x0FFF));
      }
      default: break;
    }
    return r < kRegisterCount ? registers_[r] : 0;
  }

  void WriteMmio(uint32_t address, uint32_t value) {
    const uint32_t r = (address & 0xFFFF) / 4;
    if (r == kRegCpRbWptr) {
      // sa2_cpu_direct_state: every submission, whoever makes it. A draw hook only rewrites its own packets when
      // none happened during the draw call (the ring may already be reading a submitted segment); one being written
      // right now is published whole (g_marking).
      direct::g_wptr_writes.fetch_add(1, std::memory_order_seq_cst);
      // sa2_cpu_direct_flush: this thread's own submission inside a draw call: its placeholders become records now.
      const bool own_marking = direct::FlushOwnKick();
      if (!own_marking && direct::g_marking.load(std::memory_order_seq_cst)) [[unlikely]] {
        direct::g_kick_waits.fetch_add(1, std::memory_order_relaxed);
        for (uint32_t spin = 0; direct::g_marking.load(std::memory_order_seq_cst); ++spin) {
          if (spin > 64) rex::thread::MaybeYield();
        }
      }
      // sa2_cpu_direct_pair: a Draw* record still held back goes to the queue before the ring can see its draw.
      if (direct::g_pending_count.load(std::memory_order_relaxed)) direct::FlushPendingPairs();
      // Wake the ring thread only when it waits (an empty lock + notify on every kick cost the render
      // thread ~2 %). seq_cst on both sides (here and ring_waiting_ in RingLoop): either the ring thread's
      // predicate sees the new pointer or this load sees it waiting; its 5 ms timeout bounds any miss.
      write_pointer_.store(value, std::memory_order_seq_cst);
      g_ring_published.store(value, std::memory_order_release);
      if (ring_waiting_.load(std::memory_order_seq_cst)) {
        { std::lock_guard<std::mutex> lock(ring_mutex_); }
        ring_cv_.notify_one();
      }
    }
    if (r < kRegisterCount) registers_[r] = value;  // no side effects through MMIO
  }

  void NoteGammaRampWrite(uint32_t index, uint32_t value) {
    ++gamma_writes_;
    if (index == kRegGammaIndex) {
      gamma_component_ = 0;
    } else if (index == kRegGammaMask) {
      gamma_mask_ = value & 0b111;
    } else if (index == kRegGammaSequential) {
      const uint32_t i = registers_[kRegGammaIndex] & 0xFF;
      if (gamma_mask_ & (UINT32_C(1) << (2 - gamma_component_))) {
        gamma_ramp_[i][gamma_component_] = uint16_t((value >> 6) & 0x3FF);
        ++gamma_version_;
      }
      if (++gamma_component_ >= 3) {
        gamma_component_ = 0;
        registers_[kRegGammaIndex] =
            (registers_[kRegGammaIndex] & ~UINT32_C(0xFF)) | ((i + 1) & 0xFF);
      }
    } else if (index == kRegGamma30) {
      const uint32_t i = registers_[kRegGammaIndex] & 0xFF;
      if (gamma_mask_ & 0b001) gamma_ramp_[i][2] = uint16_t(value & 0x3FF);
      if (gamma_mask_ & 0b010) gamma_ramp_[i][1] = uint16_t((value >> 10) & 0x3FF);
      if (gamma_mask_ & 0b100) gamma_ramp_[i][0] = uint16_t((value >> 20) & 0x3FF);
      if (gamma_mask_) ++gamma_version_;
      gamma_component_ = 0;
      registers_[kRegGammaIndex] =
          (registers_[kRegGammaIndex] & ~UINT32_C(0xFF)) | ((i + 1) & 0xFF);
    }
  }

  // --- Registers written by the ring ---
  void WriteRegister(uint32_t index, uint32_t value) {
    if (index >= kRegisterCount) return;
    if (index == kRegCoherStatusHost) value |= 0x80000000u;
    if (registers_[index] != value) {  // generations: what a draw must re-upload
      if (index >= 0x4000 && index < 0x4800) {
        ++(index < 0x4400 ? gen_constants_vs_ : gen_constants_ps_);
      } else if (index >= kRegFetchFirst && index <= kRegFetchLast) {
        ++gen_fetch_;
      } else if (IsViewportRegister(index)) {
        ++gen_viewport_;
      }
    }
    if (index >= kRegGammaFirst && index <= kRegGammaLast) {
      NoteGammaRampWrite(index, value);
    }
    registers_[index] = value;
    if (index >= kRegScratch0 && index <= kRegScratch7) {
      const uint32_t n = index - kRegScratch0;
      if ((1u << n) & registers_[kRegScratchUmsk]) {
        rex::memory::store_and_swap<uint32_t>(
            memory_->TranslatePhysical(registers_[kRegScratchAddr] + n * 4), value);
      }
    }
  }

  // A run of register words into the constant/fetch range (0x4000-0x48BF), the bulk of every draw's writes:
  // no per-word special-register checks, and each generation bumped once if any word of its part changed.
  template <typename Next>
  void WriteConstantRun(uint32_t index, uint32_t count, Next next) {
    if (!REXCVAR_GET(sa2_native_fast_registers) || index < 0x4000 ||
        uint64_t(index) + count > uint64_t(kRegFetchLast) + 1) {
      for (uint32_t i = 0; i < count; ++i) WriteRegister(index + i, next());
      return;
    }
    bool vs = false, ps = false, fetch = false;
    uint32_t* regs = registers_.data();
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t r = index + i, v = next();
      if (regs[r] != v) {
        regs[r] = v;
        if (r < 0x4400) vs = true;
        else if (r < 0x4800) ps = true;
        else fetch = true;
      }
    }
    if (vs) ++gen_constants_vs_;
    if (ps) ++gen_constants_ps_;
    if (fetch) ++gen_fetch_;
  }

  // WriteConstantRun over host-order words (sa2_cpu_direct_state's records): each part of the constant/fetch range
  // (VS constants, PS constants, fetch constants) compared and copied whole, its generation bumped once if it changed,
  // as WriteConstantRun does word by word. Outside that range (or with the fast path off) every word goes through
  // WriteRegister, as there.
  void WriteConstantRunHost(uint32_t index, uint32_t count, const uint32_t* v) {
    if (!REXCVAR_GET(sa2_native_fast_registers) || index < 0x4000 ||
        uint64_t(index) + count > uint64_t(kRegFetchLast) + 1) {
      for (uint32_t i = 0; i < count; ++i) WriteRegister(index + i, v[i]);
      return;
    }
    uint32_t* regs = registers_.data();
    while (count) {
      const uint32_t part_end = index < 0x4400 ? 0x4400u : index < 0x4800 ? 0x4800u : kRegFetchLast + 1;
      const uint32_t n = std::min(count, part_end - index);
      if (std::memcmp(regs + index, v, size_t(n) * 4) != 0) {
        std::memcpy(regs + index, v, size_t(n) * 4);
        ++(index < 0x4400 ? gen_constants_vs_ : index < 0x4800 ? gen_constants_ps_ : gen_fetch_);
      }
      index += n;
      v += n;
      count -= n;
    }
  }

  // The count guest words (big-endian, as the Reader sees them) at the reader position, if they are contiguous in
  // memory (indirect buffers never wrap; the ring only at its end).
  static const uint32_t* ContiguousWords(const Reader& data, uint32_t count) {
    if (data.mask && uint64_t(data.pos) + count > uint64_t(data.mask) + 1) return nullptr;
    return reinterpret_cast<const uint32_t*>(data.base + size_t(data.pos) * 4);
  }

  // WriteConstantRun over guest words (big-endian) in memory: the constant/fetch range (0x4000-0x48BF) is compared and
  // stored four words at a time; each generation is bumped once if any word of its part changed. Outside that range
  // (or with the fast path off) every word goes through WriteRegister as before.
  void WriteConstantRunRaw(uint32_t index, uint32_t count, const uint32_t* guest) {
    if (!REXCVAR_GET(sa2_native_fast_registers) || index < 0x4000 ||
        uint64_t(index) + count > uint64_t(kRegFetchLast) + 1) {
      for (uint32_t i = 0; i < count; ++i) WriteRegister(index + i, __builtin_bswap32(guest[i]));
      return;
    }
    uint32_t* regs = registers_.data();
    std::vector<uint32_t> before;
    const bool check = check_pm4_ > 0;
    if (check) {
      --check_pm4_;
      before.assign(regs + index, regs + index + count);
    }
    __builtin_prefetch(guest + count);
    __builtin_prefetch(guest + count + 16);
    const RunChange change = ApplyRunRaw(regs, index, count, guest);
    if (change.vs) ++gen_constants_vs_;
    if (change.ps) ++gen_constants_ps_;
    if (change.fetch) ++gen_fetch_;
    if (check) {  // the original loop on the saved words
      std::vector<uint32_t> expected(before);
      uint32_t i = 0;
      const RunChange want = ApplyRunReference(expected.data(), index, count,
                                               [&] { return __builtin_bswap32(guest[i++]); });
      if (!(want == change) || !std::equal(expected.begin(), expected.end(), regs + index)) {
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] DIFFERENCE: fast PM4 register run (index {:04X} count {}): generations vs {}/{} ps {}/{} "
                     "fetch {}/{} or values differ; the old path from now on", index, count, change.vs, want.vs, change.ps,
                     want.ps, change.fetch, want.fetch);
        pm4_fast_ = false;
        for (uint32_t k = 0; k < count; ++k) regs[index + k] = __builtin_bswap32(guest[k]);
      }
    }
  }

  uint32_t ReadMemory(uint32_t address) const {
    uint32_t v;
    std::memcpy(&v, memory_->TranslatePhysical(address & ~3u), 4);
    return xenos::GpuSwap(v, static_cast<xenos::Endian>(address & 0x3));
  }

  void WriteMemory(uint32_t address, uint32_t value) {
    const uint32_t v = xenos::GpuSwap(value, static_cast<xenos::Endian>(address & 0x3));
    std::memcpy(memory_->TranslatePhysical(address & ~3u), &v, 4);
  }

  // --- Threads ---
  void Interrupt(uint32_t source, uint32_t cpu) {
    const uint32_t callback = callback_.load(std::memory_order_acquire);
    if (!callback || !dispatcher_) return;
    auto* thread = rex::system::XThread::GetCurrentThread();
    if (!thread) return;
    thread->SetActiveCpu(uint8_t(cpu));
    uint64_t args[] = {source, callback_data_.load(std::memory_order_acquire)};
    dispatcher_->ExecuteInterrupt(thread->thread_state(), callback, args, 2);
    interrupts_.fetch_add(1, std::memory_order_relaxed);
    // sa2-nx: what the game's interrupt callback wrote (swap and vblank counters) can end a D3D GPU wait too.
    NotifyRingProgress();
  }

  // sa2_vsync_relaxed: no request for this long after a vblank and the regular ticks resume (loading screens).
  static constexpr auto kVsyncRelaxedGrace = std::chrono::milliseconds(50);

  // Every vblank interrupt (not the swaps, which bump counter_ too): their count and the time of the last one, for
  // FlipWaitEnded.
  // sa2_diag_wait_log: the game's flip queue around a vblank (Direct3D's device: vblank count +16952, queue read/write
  // indices +17108/+17112, 16 entries of {surface, due vblank} at +16980; the flag the ring's flip wait polls is
  // released by an entry with surface 0).
  std::string DiagFlipQueue() {
    const uint32_t d = callback_data_.load(std::memory_order_acquire);
    if (!d) return "no device";
    const auto rd = [&](uint32_t off) {
      uint32_t v;
      std::memcpy(&v, memory_->TranslateVirtual(d + off), 4);
      return __builtin_bswap32(v);
    };
    const uint32_t r = rd(17108), w = rd(17112);
    std::string s = fmt::format("vblank# {} queue {}..{}", rd(16952), r, w);
    for (uint32_t i = r; i != w && i - r < 4; ++i) {
      const uint32_t e = 16980 + ((i << 3) & 0x78);
      s += fmt::format(" [{:08X} due {}]", rd(e), rd(e + 4));
    }
    return s;
  }

  void NoteVblank() {
    last_vblank_ns_.store(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count(),
                          std::memory_order_relaxed);
    vblanks_.fetch_add(1, std::memory_order_release);
  }

  // sa2_vsync_relaxed, ring thread: a memory wait that waited ended. A vote when it ended right after a vblank that came
  // during it (within the ring's poll: 4 ms, the PC's sleeps are coarse). The flip wait (the game's vblank callback
  // clears its flag) votes every time; a wait the game thread releases does so only by chance (about 4 ms in 16.7), so
  // a memory is learned at 8 votes and at least 90 % of its ended waits (four candidates kept).
  void FlipWaitEnded(uint32_t poll, uint64_t vblanks_before) {
    int slot = -1;
    for (int i = 0; i < 4; ++i)
      if (flip_votes_[i].poll == poll) slot = i;
    if (slot < 0) {  // replace the candidate with the fewest votes
      slot = 0;
      for (int i = 1; i < 4; ++i)
        if (flip_votes_[i].votes < flip_votes_[slot].votes) slot = i;
      flip_votes_[slot] = {poll, 0, 0};
    }
    FlipVotes& c = flip_votes_[slot];
    ++c.ended;
    if (vblanks_.load(std::memory_order_acquire) == vblanks_before) return;
    const int64_t now = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    if (now - last_vblank_ns_.load(std::memory_order_relaxed) > 4000000) return;
    ++c.votes;
    if (c.votes >= 8 && c.votes * 10 >= c.ended * 9 && flip_poll_ != poll) {
      flip_poll_ = poll;
      REXLOG_INFO("[native] sa2_vsync_relaxed: the flip wait polls {:08X} (the game's vblank callback releases it; {} "
                  "of {} waits ended right after a vblank)", poll, c.votes, c.ended);
    }
  }

  // sa2_vsync_relaxed, ring thread: the flip wait is not met; the vblank thread gives the next vblank as soon as allowed.
  void RequestVblank() {
    if (vblank_wanted_.load(std::memory_order_relaxed)) return;
    vblank_asked_.store(Clock::now().time_since_epoch().count(), std::memory_order_relaxed);
    vblank_wanted_.store(true, std::memory_order_release);
    { std::lock_guard<std::mutex> lock(relaxed_mutex_); }
    relaxed_cv_.notify_one();
  }

  int VblankLoop() {
    rex::system::X_VIDEO_MODE mode;
    rex::kernel::xboxkrnl::VdQueryVideoMode(&mode);
    const double hz = std::max(1.0, double(float(mode.refresh_rate)));
    const auto interval =
        std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(1.0 / hz));
    auto next = Clock::now() + interval;
    uint64_t last_swap = rex::kernel::xboxkrnl::VdSwapCount();
    auto last_vblank = Clock::now();
    bool idle = false;  // sa2_vsync_relaxed: nobody asked within the grace period (regular ticks)
    while (active_.load(std::memory_order_acquire)) {
      if (rex::kernel::xboxkrnl::VblankPerSwap()) {  // deterministic runs (cvar vblank_per_swap, see the SDK)
        if (rex::kernel::xboxkrnl::WaitSwapForVblank(last_swap, active_)) {
          counter_.fetch_add(1, std::memory_order_relaxed);
          Interrupt(0, 2);
        }
        next = Clock::now() + interval;
        continue;
      }
      if (g_vsync_relaxed.active) {
        // Asked for (the ring waits): the vblank at once, or at the earliest 1/60 s after the previous one. Nobody
        // asking: the regular ticks once the grace period ran out. The times are the vblanks' slots (when each was
        // due or asked for), not when this thread woke up: on time, the vblanks stay on the 1/60 s grid (PC: timing
        // from the wake-ups added their latency to every frame, 57.5 fps instead of 60).
        const auto earliest = last_vblank + interval;
        const auto deadline = last_vblank + (idle ? interval : kVsyncRelaxedGrace);
        bool wanted = false;
        {
          std::unique_lock<std::mutex> lock(relaxed_mutex_);
          while (active_.load(std::memory_order_acquire)) {
            const auto now = Clock::now();
            wanted = vblank_wanted_.load(std::memory_order_acquire);
            if ((wanted && now >= earliest) || now >= deadline) break;
            const auto until = wanted ? std::min(earliest, deadline) : deadline;
            relaxed_cv_.wait_for(lock, std::min<Clock::duration>(until - now, std::chrono::milliseconds(50)));
          }
        }
        if (!active_.load(std::memory_order_acquire)) break;
        const auto now = Clock::now();
        if (wanted && now >= earliest) {
          const auto asked = Clock::time_point(Clock::duration(vblank_asked_.load(std::memory_order_acquire)));
          if (asked > earliest) {  // a late frame, shown when ready instead of at the next 1/60 s
            last_vblank = std::min(asked, now);
            if (asked - earliest > std::chrono::microseconds(500)) g_vsync_relaxed.Use();
          } else {
            last_vblank = earliest;
          }
          idle = false;
        } else {
          last_vblank = now - deadline > std::chrono::milliseconds(250) ? now : deadline;
          idle = true;
        }
        vblank_wanted_.store(false, std::memory_order_release);
        next = now + interval;  // the fixed timer below, if the key is switched off, starts from here
        counter_.fetch_add(1, std::memory_order_relaxed);
        std::string before;
        if (diag_vblank_logged_ < REXCVAR_GET(sa2_diag_wait_log) && flip_poll_) [[unlikely]] before = DiagFlipQueue();
        NoteVblank();  // before the callback: the ring may see the flip released while it still runs
        Interrupt(0, 2);
        if (!before.empty()) {
          ++diag_vblank_logged_;
          REXLOG_INFO("[diag] relaxed vblank ({}): {} -> {}", wanted ? "asked" : "tick", before, DiagFlipQueue());
        }
        REX_SYS_COUNT(rex::syscount::kVblankTick, 1);
        continue;
      }
      last_vblank = Clock::now();
      const auto now = Clock::now();
      if (now - next > std::chrono::milliseconds(250)) next = now;
      while (now >= next) {
        counter_.fetch_add(1, std::memory_order_relaxed);
        NoteVblank();  // before the callback: the ring may see the flip released while it still runs
        Interrupt(0, 2);
        next += interval;
      }
      if (REXCVAR_GET(sa2_vblank_sleep_exact)) {
        // Sleep to the next vblank instead of waking up 1000 times a second (60 of them useful).
        const auto wait = std::chrono::duration_cast<std::chrono::microseconds>(next - Clock::now());
        rex::thread::Sleep(std::clamp(wait, std::chrono::microseconds(200), std::chrono::microseconds(50000)));
      } else {
        rex::thread::Sleep(std::chrono::milliseconds(1));
      }
      REX_SYS_COUNT(rex::syscount::kVblankTick, 1);
    }
    return 0;
  }

  int RingLoop() {
    uint32_t read = 0;
    uint32_t generation = ring_generation_.load(std::memory_order_acquire);
    while (active_.load(std::memory_order_acquire)) {
      {
        std::unique_lock<std::mutex> lock(ring_mutex_);
        ring_waiting_.store(true, std::memory_order_seq_cst);
        ring_cv_.wait_for(lock, std::chrono::milliseconds(5), [&] {
          const uint32_t words = ring_words_.load(std::memory_order_acquire);
          return !active_.load(std::memory_order_acquire) ||
                 (words && (write_pointer_.load(std::memory_order_seq_cst) & (words - 1)) != read);
        });
        ring_waiting_.store(false, std::memory_order_relaxed);
      }
      if (!active_.load(std::memory_order_acquire)) break;
      Report(false);
      const uint32_t gen = ring_generation_.load(std::memory_order_acquire);
      if (gen != generation) {
        generation = gen;
        read = 0;
      }
      const uint32_t base = ring_base_.load(std::memory_order_acquire);
      const uint32_t words = ring_words_.load(std::memory_order_acquire);
      if (!base || !words) continue;
      const uint32_t written = write_pointer_.load(std::memory_order_acquire) & (words - 1);
      if (written == read) continue;
      Reader reader;
      reader.base = memory_->TranslatePhysical(base);
      reader.physical = base;
      reader.mask = words - 1;
      reader.pos = read;
      reader.end = written;
      const auto work_start = Clock::now();
      while (reader.Pending()) {
        if (!Packet(reader, 0)) break;  // partial packet: the rest comes with the next WPTR
      }
      ring_busy_ns_ += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - work_start).count());
      read = reader.pos;
      g_ring_processed.store(read, std::memory_order_release);  // RingIdle (lockstep captures)
      const uint32_t writeback = read_writeback_.load(std::memory_order_acquire);
      if (writeback) {
        rex::memory::store_and_swap<uint32_t>(memory_->TranslatePhysical(writeback), read);
      }
      NotifyRingProgress();
      sa2::native::deferred::Flush();  // Batched mode: the worker must not wait for the next WPTR
    }
    return 0;
  }

  // --- PM4 ---
  bool Packet(Reader& reader, int depth) {
    const uint32_t packet = reader.Peek();
    uint32_t count = 0;
    switch (packet >> 30) {
      case 0: count = packet ? ((packet >> 16) & 0x3FFF) + 1 : 0; break;
      case 1: count = 2; break;
      case 2: count = 0; break;
      default: count = ((packet >> 16) & 0x3FFF) + 1; break;
    }
    if (reader.Pending() < count + 1) return false;
    if (pm4_fast_) {  // the stream was written by the game thread's core: pull the lines ahead of the parse
      const uint32_t ahead = reader.pos + count + 1 + 48;
      __builtin_prefetch(reader.base + size_t(reader.mask ? (ahead & reader.mask) : ahead) * 4);
    }
    packet_address_ = reader.physical + reader.pos * 4;
    reader.Advance(1);
    Reader data = reader;
    reader.Advance(count);
    ++packets_;
    if (!packet) return true;
    switch (packet >> 30) {
      case 0: {
        const uint32_t index = packet & 0x7FFF;
        const bool single = (packet >> 15) & 0x1;
        ++type0_packets_;
        type0_words_ += count;
        type0_single_ += single;
        if (single) {
          for (uint32_t i = 0; i < count; ++i) WriteRegister(index, data.Read());
        } else if (const uint32_t* words = pm4_fast_ ? ContiguousWords(data, count) : nullptr) {
          WriteConstantRunRaw(index, count, words);
        } else {
          WriteConstantRun(index, count, [&] { return data.Read(); });
        }
        break;
      }
      case 1: {
        const uint32_t v1 = data.Read(), v2 = data.Read();
        WriteRegister(packet & 0x7FF, v1);
        WriteRegister((packet >> 11) & 0x7FF, v2);
        break;
      }
      case 2: break;
      default: Type3(data, packet, count, depth); break;
    }
    return true;
  }

  static uint32_t ConstantBase(uint32_t type) {
    switch (type) {
      case 0: return 0x4000;  // ALU
      case 1: return 0x4800;  // fetch
      case 2: return 0x4900;  // bool
      case 3: return 0x4908;  // loop
      case 4: return 0x2000;  // registers
      default: return UINT32_MAX;
    }
  }

  void Type3(Reader& data, uint32_t packet, uint32_t words, int depth) {
    const uint32_t opcode = (packet >> 8) & 0x7F;
    ++opcodes_[opcode];
    opcode_words_[opcode] += words;
    if ((packet & 0x1) && (!(bin_select_ & bin_mask_) || opcode == xenos::PM4_XE_SWAP)) {
      // A draw the predicate drops still had its Draw* record: consume it so the pairing stays in step.
      if ((opcode == xenos::PM4_DRAW_INDX || opcode == xenos::PM4_DRAW_INDX_2) &&
          words >= (opcode == xenos::PM4_DRAW_INDX ? 2u : 1u)) {
        if (opcode == xenos::PM4_DRAW_INDX) data.Read();
        const uint32_t initiator = data.Read();
        const uint32_t saved = registers_[kRegVgtDrawInitiator];
        registers_[kRegVgtDrawInitiator] = initiator;
        PairDraw();
        registers_[kRegVgtDrawInitiator] = saved;
        ++draws_predicated_;
      }
      return;
    }
    switch (opcode) {
      case xenos::PM4_INTERRUPT: {
        if (words < 1) break;
        AdvanceSyncEpoch();
        const uint32_t cpus = data.Read();
        for (uint32_t cpu = 0; cpu < 6; ++cpu) {
          if (cpus & (1u << cpu)) Interrupt(1, cpu);
        }
        break;
      }
      case xenos::PM4_XE_SWAP:
        if (const int32_t delay = REXCVAR_GET(sa2_diag_ring_swap_delay_us); delay > 0) [[unlikely]] {
          const auto until = Clock::now() + std::chrono::microseconds(delay);  // busy, like ring work
          while (Clock::now() < until) {
          }
        }
        if (words >= 4) {
          data.Read();
          data.Read();  // frontbuffer
          swap_width_ = data.Read();
          swap_height_ = data.Read();
        }
        ++swaps_;
        counter_.fetch_add(1, std::memory_order_relaxed);
        TraceSwap();
        Present();
        {
          const uint64_t presented = g_presented_swaps.fetch_add(1, std::memory_order_release) + 1;
          if (auto* hook = g_present_hook.load(std::memory_order_acquire)) [[unlikely]] hook(presented);
          sa2::render::OnPresented(presented);  // PC: the docked/handheld test toggles (sa2_render_test_toggle_at)
        }
        break;
      case xenos::PM4_INDIRECT_BUFFER:
      case xenos::PM4_INDIRECT_BUFFER_PFD: {
        if (words < 2) break;
        const uint32_t address = data.Read() & 0x1FFFFFFF;
        const uint32_t length = data.Read() & 0xFFFFF;
        if (depth < kMaxIndirectDepth) {
          Reader r;
          r.base = memory_->TranslatePhysical(address);
          r.physical = address;
          r.end = length;
          while (r.Pending()) {
            if (!Packet(r, depth + 1)) break;
          }
        }
        break;
      }
      case xenos::PM4_WAIT_REG_MEM: {
        if (words < 5) break;
        const uint32_t info = data.Read(), poll = data.Read(), ref = data.Read(), mask = data.Read();
        const auto wait_start = Clock::now();
        const auto limit = wait_start + kWaitRegMemMax;
        // The time spent waiting for the game (not ring work): counted from the first unmet check.
        struct WaitTime {
          uint64_t& total;
          std::chrono::steady_clock::time_point start;
          bool waited = false;
          ~WaitTime() {
            if (waited)
              total += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count());
          }
        } wait_time{ring_wait_ns_, wait_start};
        const uint32_t diag_first = (info & 0x10) ? ReadMemory(poll) : Register(poll);
        const uint64_t diag_vblanks = counter_.load(std::memory_order_relaxed);
        const uint64_t vblanks_before = vblanks_.load(std::memory_order_acquire);
        const bool flip_wait = (info & 0x10) && poll == flip_poll_ && poll != 0;
        bool asked = false;              // sa2_vsync_relaxed: this wait asked for its vblank
        uint64_t asked_vblanks = 0;
        for (;;) {
          if (!(info & 0x10) && poll == kRegCoherStatusHost && (registers_[poll] & 0x80000000u)) {
            registers_[poll] = 0;  // MakeCoherent: no shared memory to synchronize
          }
          const uint32_t value = (info & 0x10) ? ReadMemory(poll) : Register(poll);
          if (Compare(info, value & mask, ref)) {
            if (wait_time.waited && (info & 0x10)) FlipWaitEnded(poll, vblanks_before);
            if (wait_time.waited && diag_wait_logged_ < REXCVAR_GET(sa2_diag_wait_log)) [[unlikely]] {
              const double ms = std::chrono::duration<double, std::milli>(Clock::now() - wait_start).count();
              if (ms > 2.0) {
                ++diag_wait_logged_;
                const int64_t now_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
                REXLOG_INFO("[diag] WAIT_REG_MEM {:.2f} ms: {} {:08X} info {:08X} ref {:08X} mask {:08X}: {:08X} -> {:08X}; "
                            "vblank counter {} -> {}, ring swaps {}; vblanks {} -> {}, ended {} us after the last one",
                            ms, (info & 0x10) ? "memory" : "register", poll, info, ref, mask, diag_first, value,
                            diag_vblanks, counter_.load(std::memory_order_relaxed),
                            g_presented_swaps.load(std::memory_order_relaxed), vblanks_before,
                            vblanks_.load(std::memory_order_relaxed),
                            (now_ns - last_vblank_ns_.load(std::memory_order_relaxed)) / 1000);
              }
            }
            break;
          }
          wait_time.waited = true;
          // The flip: a vblank as soon as one is allowed. Once per wait: a second request while the first one's
          // callback is still releasing the flag would spend the next 1/60 s window for nothing (PC log: the next
          // flip then waited a full refresh). Again only if a vblank came 1 ms ago or more and the flag is still set.
          if (flip_wait && g_vsync_relaxed.active) {
            const uint64_t vb = vblanks_.load(std::memory_order_acquire);
            if (!asked || (vb != asked_vblanks &&
                           std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
                                       .count() - last_vblank_ns_.load(std::memory_order_relaxed) > 1000000)) {
              asked = true;
              asked_vblanks = vb;
              RequestVblank();
            }
          }
          if (Clock::now() >= limit || !active_.load(std::memory_order_acquire)) {
            if (waits_timed_out_++ == 0) {
              REXLOG_WARN("[native] WAIT_REG_MEM not met ({} {:08X} ref {:08X} mask {:08X})",
                          (info & 0x10) ? "memory" : "register", poll, ref, mask);
            }
            break;
          }
          rex::thread::Sleep(std::chrono::microseconds(50));
        }
        break;
      }
      case xenos::PM4_REG_RMW: {
        if (words < 3) break;
        const uint32_t info = data.Read(), and_v = data.Read(), or_v = data.Read();
        uint32_t v = Register(info & 0x1FFF);
        v &= ((info >> 31) & 0x1) ? Register(and_v & 0x1FFF) : and_v;
        v |= ((info >> 30) & 0x1) ? Register(or_v & 0x1FFF) : or_v;
        WriteRegister(info & 0x1FFF, v);
        break;
      }
      case xenos::PM4_REG_TO_MEM: {
        if (words < 2) break;
        AdvanceSyncEpoch();
        const uint32_t reg = data.Read(), address = data.Read();
        WriteMemory(address, Register(reg));
        break;
      }
      case xenos::PM4_MEM_WRITE: {
        if (words < 1) break;
        AdvanceSyncEpoch();
        uint32_t address = data.Read();
        for (uint32_t i = 1; i < words; ++i, address += 4) WriteMemory(address, data.Read());
        break;
      }
      case xenos::PM4_COND_WRITE: {
        if (words < 6) break;
        const uint32_t info = data.Read(), poll = data.Read(), ref = data.Read(), mask = data.Read();
        const uint32_t dest = data.Read(), value = data.Read();
        const uint32_t v = (info & 0x10) ? ReadMemory(poll) : Register(poll);
        if (Compare(info, v & mask, ref)) {
          if (info & 0x100) {
            AdvanceSyncEpoch();
            WriteMemory(dest, value);
          } else {
            WriteRegister(dest, value);
          }
        }
        break;
      }
      case xenos::PM4_EVENT_WRITE:
        if (words < 1) break;
        WriteRegister(kRegVgtEventInitiator, data.Read() & 0x3F);
        break;
      case xenos::PM4_EVENT_WRITE_SHD: {
        if (words < 3) break;
        const uint32_t initiator = data.Read(), address = data.Read(), value = data.Read();
        AdvanceSyncEpoch();
        WriteRegister(kRegVgtEventInitiator, initiator & 0x3F);
        WriteMemory(address, ((initiator >> 31) & 0x1) ? counter_.load(std::memory_order_relaxed)
                                                       : value);
        // The D3D fence word the game's wait loop compares (sub_8222C768 / sub_8222FA98): wake it now, not
        // at the end of the ring segment.
        NotifyRingProgress();
        break;
      }
      case xenos::PM4_EVENT_WRITE_EXT: {
        if (words < 2) break;
        const uint32_t initiator = data.Read(), address = data.Read();
        WriteRegister(kRegVgtEventInitiator, initiator & 0x3F);
        const uint16_t extent[] = {0, kExtentMax, 0, kExtentMax, 0, 1};
        uint8_t* dest = memory_->TranslatePhysical(address & ~3u);
        for (size_t i = 0; i < std::size(extent); ++i) {
          rex::memory::store_and_swap<uint16_t>(dest + i * 2, extent[i]);
        }
        break;
      }
      case xenos::PM4_EVENT_WRITE_ZPD: {
        if (words < 1) break;
        WriteRegister(kRegVgtEventInitiator, data.Read() & 0x3F);
        // D3D9 occlusion queries on Xbox 360:
        // Issue(D3DISSUE_BEGIN): RB_SAMPLE_COUNT_ADDR points to query->begin (+0x20).
        // Issue(D3DISSUE_END):   RB_SAMPLE_COUNT_ADDR points to query->end   (+0x00).
        // D3D places the byte-swapped sentinel 0xFFFFFEED in the end structure before EVENT_WRITE_ZPD.
        const uint32_t address = Register(kRegRbSampleCountAddr);
        if (address) {
          auto* counts =
              memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(address);
          const uint32_t finished = rex::byte_swap(uint32_t(0xFFFFFEED));
          const bool end = counts->ZPass_A == finished || counts->ZPass_B == finished ||
                           counts->ZFail_A == finished || counts->ZFail_B == finished;
          std::memset(counts, 0, sizeof(*counts));
          // SDK fallback: begin = 0, end = 1000 samples.
          if (end) counts->Total_A = counts->ZPass_A = 1000;
          static uint32_t query_reports = 0;
          if (query_reports++ < 30) {
            REXLOG_INFO("[native] ZPD {} at {:08X}: {} samples", end ? "end" : "begin", address,
                        uint32_t(counts->ZPass_A));
          }
        }
        break;
      }
      case xenos::PM4_SET_CONSTANT: {
        if (words < 1) break;
        const uint32_t type_index = data.Read();
        const uint32_t base = ConstantBase((type_index >> 16) & 0xFF);
        if (base != UINT32_MAX) {
          const uint32_t index = base + (type_index & 0x7FF);
          for (uint32_t i = 1; i < words; ++i) WriteRegister(index + i - 1, data.Read());
        }
        break;
      }
      case xenos::PM4_SET_CONSTANT2:
      case xenos::PM4_SET_SHADER_CONSTANTS: {
        if (words < 1) break;
        const uint32_t index = data.Read() & 0xFFFF;
        if (const uint32_t* raw = pm4_fast_ ? ContiguousWords(data, words - 1) : nullptr) {
          WriteConstantRunRaw(index, words - 1, raw);
        } else {
          WriteConstantRun(index, words - 1, [&] { return data.Read(); });
        }
        break;
      }
      case xenos::PM4_LOAD_ALU_CONSTANT: {
        if (words < 3) break;
        const uint32_t address = data.Read() & 0x3FFFFFFF;
        const uint32_t type_index = data.Read();
        const uint32_t n = data.Read() & 0xFFF;
        const uint32_t base = ConstantBase((type_index >> 16) & 0xFF);
        if (base != UINT32_MAX) {
          const uint8_t* src = memory_->TranslatePhysical(address);
          const uint32_t index = base + (type_index & 0x7FF);
          uint32_t i = 0;
          if (pm4_fast_) {
            WriteConstantRunRaw(index, n, reinterpret_cast<const uint32_t*>(src));
          } else {
            WriteConstantRun(index, n, [&] { return rex::memory::load_and_swap<uint32_t>(src + size_t(i++) * 4); });
          }
        }
        break;
      }
      case xenos::PM4_SET_BIN_MASK_LO:
        if (words >= 1) bin_mask_ = (bin_mask_ & 0xFFFFFFFF00000000ull) | data.Read();
        break;
      case xenos::PM4_SET_BIN_MASK_HI:
        if (words >= 1) bin_mask_ = (bin_mask_ & 0xFFFFFFFFull) | (uint64_t(data.Read()) << 32);
        break;
      case xenos::PM4_SET_BIN_SELECT_LO:
        if (words >= 1) bin_select_ = (bin_select_ & 0xFFFFFFFF00000000ull) | data.Read();
        break;
      case xenos::PM4_SET_BIN_SELECT_HI:
        if (words >= 1) bin_select_ = (bin_select_ & 0xFFFFFFFFull) | (uint64_t(data.Read()) << 32);
        break;
      case xenos::PM4_SET_BIN_MASK:
        if (words >= 2) {
          const uint64_t hi = data.Read(), lo = data.Read();
          bin_mask_ = (hi << 32) | lo;
        }
        break;
      case xenos::PM4_SET_BIN_SELECT:
        if (words >= 2) {
          const uint64_t hi = data.Read(), lo = data.Read();
          bin_select_ = (hi << 32) | lo;
        }
        break;
      case xenos::PM4_DRAW_INDX:
      case xenos::PM4_DRAW_INDX_2: {
        const uint32_t skip = opcode == xenos::PM4_DRAW_INDX ? 1 : 0;
        if (words < skip + 1) break;
        if (skip) data.Read();
        const uint32_t initiator = data.Read();
        WriteRegister(kRegVgtDrawInitiator, initiator);
        if (((initiator >> 6) & 0x3) == uint32_t(xenos::SourceSelect::kDMA) && words >= skip + 3) {
          WriteRegister(kRegVgtDmaBase, data.Read());
          WriteRegister(kRegVgtDmaSize, data.Read());
        }
        if ((Register(kRegRbModeControl) & 0x7) == uint32_t(xenos::EdramMode::kCopy)) {
          ++copies_;
          resolve_destinations_.insert(Register(rex::graphics::XE_GPU_REG_RB_COPY_DEST_BASE) & 0x1FFFF000u);
          if (trace_phase_ == 1) TraceCopy();
          Copy();
        } else {
          ++draws_;
          if (depth > 0) ++draws_indirect_;
          PairDraw();
          trace_result_ = -1;
          if (draw_vs_ && (draw_ps_ || (Register(kRegRbModeControl) & 0x7) == 5)) {
            Draw();
          } else {
            ++draws_unidentified_;
            ++missing_draw_pairs_[{draw_vs_ ? 0 : current_vs_hash_, draw_ps_ ? 0 : current_ps_hash_}];
          }
          if (trace_phase_ == 1) TraceDraw();
        }
        break;
      }
      case xenos::PM4_IM_LOAD: {
        if (words < 2) break;
        const uint32_t address_type = data.Read();
        const uint32_t size = data.Read() & 0xFFFF;
        im_load_address_ = address_type & 0x1FFFFFFCu;
        const uint8_t* src = memory_->TranslatePhysical(address_type & ~3u);
        const uint32_t load_type = address_type & 0x3;
        if (raw_microcode_ && load_type <= 1 && ReloadsSameMicrocode(load_type, src, size)) break;
        microcode_.resize(size);
        for (uint32_t i = 0; i < size; ++i) {
          microcode_[i] = rex::memory::load_and_swap<uint32_t>(src + size_t(i) * 4);
        }
        IdentifyShader(load_type);
        if (raw_microcode_ && load_type <= 1) {
          // Every path of IdentifyShader leaves the stage's microcode equal to this load: keep its raw words.
          auto& raw = load_type == 1 ? raw_ps_ : raw_vs_;
          raw.assign(reinterpret_cast<const uint32_t*>(src), reinterpret_cast<const uint32_t*>(src) + size);
          (load_type == 1 ? raw_ps_valid_ : raw_vs_valid_) = true;
        }
        break;
      }
      case xenos::PM4_IM_LOAD_IMMEDIATE: {
        if (words < 2) break;
        const uint32_t type = data.Read();
        const uint32_t size = data.Read() & 0xFFFF;
        if (size > words - 2) break;
        im_load_address_ = 0;  // immediate: the microcode is in the ring
        microcode_.resize(size);
        for (uint32_t i = 0; i < size; ++i) microcode_[i] = data.Read();
        IdentifyShader(type & 0x3);
        if ((type & 0x3) == 0) raw_vs_valid_ = false;  // the raw copy no longer matches the stage's microcode
        else if ((type & 0x3) == 1) raw_ps_valid_ = false;
        break;
      }
      case xenos::PM4_NOP:
        // sa2_cpu_direct_state: the NOP that covers a draw's state packets names the host record that replaces them.
        if (words >= 2 && data.Peek() == direct::kMagic) {
          data.Advance(1);
          ApplyDirectRecord(data.Read(), depth);
        } else if (words >= 2 && data.Peek() == direct::kFlushMagic) [[unlikely]] {
          // sa2_cpu_direct_flush: a placeholder the drawing thread never published: its draw state is missing.
          if (direct_flush_lost_++ < 8)
            REXLOG_ERROR("[native] DIFFERENCE: sa2_cpu_direct_flush placeholder at {:08X} reached the ring "
                         "unpublished: its draw state is missing; native flushes off", packet_address_);
          sa2::enh::VerifyDifference();
          direct::g_flush_failed.store(true, std::memory_order_relaxed);
        }
        break;
      default: break;  // IM_LOAD, IM_LOAD_IMMEDIATE, NOP, ...: nothing to do without drawing
    }
  }

  // sa2_cpu_direct_state, ring thread: the record a marker NOP names, applied where its packets were. Register runs go
  // through the same register writes as in Packet(); type-3 packets through Type3 with their original words and
  // address. Records come in the order of their markers (one producer, one stream); the records of markers the ring
  // never reached (none so far) are released on the way.
  void ApplyDirectRecord(uint32_t seq, int depth) {
    using namespace direct;
    // The region starts at the marker NOP (references count words from it).
    const uint32_t marker = packet_address_;
    constexpr uint64_t mask = kRingWords - 1;
    const uint64_t first = g_tail.load(std::memory_order_relaxed);
    const uint64_t head = g_head.load(std::memory_order_acquire);
    const uint32_t* ring = g_ring.get();
    uint64_t tail = first;
    bool found = false;
    while (ring && tail < head) {
      const uint32_t s = ring[tail & mask];
      if (s == kPad) {
        tail = (tail | mask) + 1;
        continue;
      }
      if (s == seq) {
        found = true;
        break;
      }
      if (int32_t(s - seq) > 0) break;  // every record left is newer: this marker's record was applied already
      tail += ring[(tail & mask) + 1];
      ++direct_skipped_;
    }
    if (!found) {
      // A marker the ring sees a second time (a replayed range) or a lost record: the packets under it are gone.
      if (direct_lost_++ < 8)
        REXLOG_ERROR("[native] direct-state record {} not found (oldest kept: {}): its draw state is missing; no more "
                     "records", seq, ring && first < head ? ring[first & mask] : 0);
      g_failed.store(true, std::memory_order_relaxed);
      return;
    }
    const uint32_t* rec = ring + (tail & mask);
    const uint32_t total = rec[1];
    const uint32_t* op = rec + 2;
    const uint32_t* const end = rec + total;
    const uint8_t* const region = memory_->TranslatePhysical(marker);
    const uint32_t* restore = nullptr;
    if (census_) [[unlikely]] Census(op, end, region);
    while (op < end) {
      const uint32_t tag = *op++;
      const uint32_t count = tag >> 8;
      switch (tag & 0xFF) {
        case kRun: {
          const uint32_t index = *op++;
          WriteConstantRunHost(index, count, op);
          op += count;
          break;
        }
        case kRunSingle: {
          const uint32_t index = *op++;
          for (uint32_t i = 0; i < count; ++i) WriteRegister(index, op[i]);
          op += count;
          break;
        }
        case kType1: {
          const uint32_t h = op[0];
          WriteRegister(h & 0x7FF, op[1]);
          WriteRegister((h >> 11) & 0x7FF, op[2]);
          op += 3;
          break;
        }
        case kRunRef:
        case kRunSingleRef: {  // as Packet() handles a type-0 packet, on the words in the command buffer
          const uint32_t index = op[0];
          const uint32_t* v = reinterpret_cast<const uint32_t*>(region + size_t(op[1]) * 4);
          if ((tag & 0xFF) == kRunSingleRef) {
            for (uint32_t i = 0; i < count; ++i) WriteRegister(index, __builtin_bswap32(v[i]));
          } else if (pm4_fast_) {
            WriteConstantRunRaw(index, count, v);
          } else {
            uint32_t i = 0;
            WriteConstantRun(index, count, [&] { return __builtin_bswap32(v[i++]); });
          }
          op += 2;
          break;
        }
        case kType1Ref: {
          const uint32_t h = op[0];
          const uint32_t* v = reinterpret_cast<const uint32_t*>(region + size_t(op[1]) * 4);
          WriteRegister(h & 0x7FF, __builtin_bswap32(v[0]));
          WriteRegister((h >> 11) & 0x7FF, __builtin_bswap32(v[1]));
          op += 2;
          break;
        }
        case kType3Ref: {
          const uint32_t h = op[0], off = op[1];
          packet_address_ = marker + off * 4;
          Reader r;
          r.base = region + size_t(off + 1) * 4;
          r.physical = marker + (off + 1) * 4;
          r.mask = 0;
          r.pos = 0;
          r.end = count;
          Type3(r, h, count, depth);
          op += 2;
          break;
        }
        case kRestore:  // sa2_cpu_direct_state_restore: written back once every other op of the record is applied
          restore = op;
          op += 3;
          break;
        case kPair: {  // sa2_cpu_direct_pair: the Draw* record of the draw packet right after this marker
          direct_pair_draw_ = op[0];
          direct_pair_.type = op[1];
          direct_pair_.r5 = op[2];
          direct_pair_.r6 = op[3];
          direct_pair_.r7 = op[4];
          direct_pair_.vs = op[5];
          direct_pair_.ps = op[6];
          direct_pair_.start = op[7];
          direct_pair_.snapshot = int32_t(op[8]);
          direct_pair_.device = op[9];
          direct_pair_valid_ = true;
          op += kPairWords;
          break;
        }
        case kType3: {
          const uint32_t h = op[0];
          packet_address_ = op[1];
          Reader r;
          r.base = reinterpret_cast<const uint8_t*>(op + 2);
          r.physical = op[1] + 4;
          r.mask = 0;
          r.pos = 0;
          r.end = count;
          Type3(r, h, count, depth);
          op += 2 + count;
          break;
        }
        default:
          op = end;  // cannot happen: written by EndDrawDirect
          break;
      }
    }
    if (restore) {
      // The Reader is already past the region (the NOP covered it), so this changes nothing for this pass.
      std::memcpy(const_cast<uint8_t*>(region), restore, 12);
      ++direct_restored_;
    }
    ++direct_applied_;
    g_tail.store(tail + total, std::memory_order_release);
  }
  uint64_t direct_applied_ = 0, direct_lost_ = 0, direct_skipped_ = 0, direct_restored_ = 0;
  uint64_t direct_flush_lost_ = 0;  // sa2_cpu_direct_flush placeholders that reached the ring unpublished
  // sa2_cpu_ring_identity_memo (PairDraw / Draw).
  struct IdentityMemoKey {
    uint32_t object_vs = 0, object_ps = 0;
    bool accepted = false, pairing_fast = false, identity_flat = false;
    uint64_t objects_version = 0, vs_generation = 0, ps_generation = 0;
    const sa2::native::ShaderEntry* vs = nullptr;
    const sa2::native::ShaderEntry* ps = nullptr;
    bool operator==(const IdentityMemoKey&) const = default;
  };
  IdentityMemoKey identity_memo_key_, identity_memo_last_key_;
  const sa2::native::ShaderEntry* identity_memo_pair_vs_ = nullptr;
  const sa2::native::ShaderEntry* identity_memo_pair_ps_ = nullptr;
  const sa2::native::ShaderEntry* identity_memo_final_vs_ = nullptr;
  bool identity_memo_final_proven_ = false, identity_memo_valid_ = false, identity_memo_hit_ = false,
       identity_memo_pending_ = false;
  uint64_t identity_memo_checked_ = 0;
  // sa2_direct_state_census: the register groups a record writes, as a bit mask (CensusGroup), counted per mask.
  const bool census_ = REXCVAR_GET(sa2_direct_state_census);
  std::unordered_map<uint32_t, uint64_t> census_masks_;
  static uint32_t CensusGroup(uint32_t r) {
    if (r < 0x2000) return 1u << 0;                       // control
    if (r < 0x2080) return 1u << 1;                       // 0x2000 group
    if (r < 0x2100) return 1u << 2;                       // viewport/window 0x2080
    if (r < 0x2180) return 1u << 3;                       // 0x2100 (VGT/PA)
    if (r < 0x2200) return 1u << 4;                       // 0x2180 (SQ/program)
    if (r < 0x2280) return 1u << 5;                       // 0x2200 (RB/depth/blend)
    if (r < 0x2300) return 1u << 6;                       // 0x2280 (PA_SU/SC)
    if (r < 0x2380) return 1u << 7;                       // 0x2300 (RB misc)
    if (r < 0x2400) return 1u << 8;                       // 0x2380 bank
    if (r < 0x4000) return 1u << 9;                       // other
    if (r < 0x4400) return 1u << 10;                      // VS constants
    if (r < 0x4800) return 1u << 11;                      // PS constants
    if (r < 0x4860) return 1u << 12;                      // fetch slots 0-15 (textures)
    if (r < 0x48C0) return 1u << 13;                      // fetch slots 16-31 (vertices)
    if (r < 0x4908) return 1u << 14;                      // bool constants
    if (r < 0x4928) return 1u << 15;                      // loop constants
    return 1u << 16;                                      // 0x5000 and up
  }
  // Census: `mask` = the groups the record writes, `changed` = the groups whose values it really changes (compared with
  // the register file before the record is applied); per register of the 0x2100-0x23FF render-state range, how often
  // its value changes.
  std::unordered_map<uint32_t, uint64_t> census_changed_;
  std::unordered_map<uint32_t, uint64_t> census_register_changes_;
  void Census(const uint32_t* op, const uint32_t* end, const uint8_t* region) {
    uint32_t mask = 0, changed = 0;
    while (op < end) {
      const uint32_t tag = *op++, count = tag >> 8;
      switch (tag & 0xFF) {
        case direct::kRunRef:
        case direct::kRunSingleRef: {  // the same accounting as an inline run, on the command buffer's words
          const uint32_t index = op[0];
          const uint32_t* v = reinterpret_cast<const uint32_t*>(region + size_t(op[1]) * 4);
          const bool run = (tag & 0xFF) == direct::kRunRef;
          const uint32_t last = run ? index + count - 1 : index;
          for (uint32_t g = CensusGroup(index); ; ) {
            mask |= g;
            if (g == CensusGroup(last)) break;
            g <<= 1;
          }
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t r = run ? index + i : index;
            if (r < kRegisterCount && registers_[r] != __builtin_bswap32(v[i])) {
              changed |= CensusGroup(r);
              if (r >= 0x2000 && r < 0x2400) ++census_register_changes_[r];
            }
          }
          op += 2;
          break;
        }
        case direct::kType1Ref: {
          const uint32_t* v = reinterpret_cast<const uint32_t*>(region + size_t(op[1]) * 4);
          mask |= CensusGroup(op[0] & 0x7FF) | CensusGroup((op[0] >> 11) & 0x7FF);
          if (registers_[op[0] & 0x7FF] != __builtin_bswap32(v[0])) changed |= CensusGroup(op[0] & 0x7FF);
          if (registers_[(op[0] >> 11) & 0x7FF] != __builtin_bswap32(v[1])) changed |= CensusGroup((op[0] >> 11) & 0x7FF);
          op += 2;
          break;
        }
        case direct::kType3Ref: {
          const uint32_t opcode = (op[0] >> 8) & 0x7F;
          mask |= (opcode == xenos::PM4_IM_LOAD || opcode == xenos::PM4_IM_LOAD_IMMEDIATE) ? 1u << 20 : 1u << 21;
          op += 2;
          break;
        }
        case direct::kRun:
        case direct::kRunSingle: {
          const uint32_t index = *op;
          const bool run = (tag & 0xFF) == direct::kRun;
          const uint32_t last = run ? index + count - 1 : index;
          for (uint32_t g = CensusGroup(index); ; ) {  // every group the run spans
            mask |= g;
            if (g == CensusGroup(last)) break;
            g <<= 1;
          }
          for (uint32_t i = 0; i < count; ++i) {
            const uint32_t r = run ? index + i : index;
            if (r < kRegisterCount && registers_[r] != op[1 + i]) {
              changed |= CensusGroup(r);
              if (r >= 0x2000 && r < 0x2400) ++census_register_changes_[r];
            }
          }
          op += 1 + count;
          break;
        }
        case direct::kType1: {
          mask |= CensusGroup(op[0] & 0x7FF) | CensusGroup((op[0] >> 11) & 0x7FF);
          if (registers_[op[0] & 0x7FF] != op[1]) changed |= CensusGroup(op[0] & 0x7FF);
          if (registers_[(op[0] >> 11) & 0x7FF] != op[2]) changed |= CensusGroup((op[0] >> 11) & 0x7FF);
          op += 3;
          break;
        }
        case direct::kType3: {
          const uint32_t opcode = (op[0] >> 8) & 0x7F;
          mask |= (opcode == xenos::PM4_IM_LOAD || opcode == xenos::PM4_IM_LOAD_IMMEDIATE) ? 1u << 20 : 1u << 21;
          op += 2 + count;
          break;
        }
        case direct::kPair: op += direct::kPairWords; break;
        case direct::kRestore: op += 3; break;
        default: op = end; break;
      }
    }
    ++census_masks_[mask];
    ++census_changed_[changed | (mask & (3u << 20))];  // shader loads and other type-3 packets count as written
  }
  // sa2_cpu_direct_pair: the Draw* record a direct record carried, for the draw packet at direct_pair_draw_.
  DrawRecord direct_pair_{};
  uint32_t direct_pair_draw_ = 0;
  bool direct_pair_valid_ = false;
  uint64_t direct_pairs_used_ = 0, direct_pairs_checked_ = 0, direct_pairs_table_missed_ = 0;
  // The ring thread's time in its parse loop and, of that, in WAIT_REG_MEM waits for the game (Report).
  uint64_t ring_busy_ns_ = 0, ring_wait_ns_ = 0;
  uint64_t report_ring_busy_ns_ = 0, report_ring_wait_ns_ = 0, report_ring_swaps_ = 0, report_ring_draws_ = 0;

  uint32_t Register(uint32_t index) const { return index < kRegisterCount ? registers_[index] : 0; }

  // --- Native drawing ---
  bool EnsureTargets() {
    if (!targets_ && !targets_failed_) {
      targets_ = sa2::native::TargetsNative::Create(provider_ ? provider_->vulkan_device() : nullptr,
                                                        memory_);
      if (!targets_) {
        targets_failed_ = true;
        REXLOG_ERROR("[native] could not create the native render targets");
      }
    }
    return targets_ != nullptr;
  }

  bool ComputeVertexIdentity(const sa2::native::ShaderEntry* candidate) const {
    if (VertexShaderIdentityMatches(
            {ShaderIdentityStage::Vertex, vs_microcode_},
            {candidate->vertices ? ShaderIdentityStage::Vertex : ShaderIdentityStage::Pixel, candidate->microcode},
            candidate->elements, [](const auto& element) { return element.instruction; }))
      return true;
    // sa2_gpu_vs_fetch_reorder: only the entry IdentifyShader chose for this very load, and only while the proof holds
    // on its words (sa2::native::FetchReordered).
    return candidate == vs_reorder_entry_ && sa2::native::FetchReordered(*candidate, vs_microcode_);
  }
  bool ComputePixelIdentity(const sa2::native::ShaderEntry* candidate) const {
    return PixelShaderIdentityMatches(
        {ShaderIdentityStage::Pixel, ps_microcode_},
        {candidate->vertices ? ShaderIdentityStage::Vertex : ShaderIdentityStage::Pixel, candidate->microcode});
  }
  // Direct-mapped memo cell of a library entry for one stage's current load generation.
  struct IdentityCell {
    const sa2::native::ShaderEntry* candidate = nullptr;
    uint64_t generation = 0;
    bool result = false;
  };
  static constexpr size_t kIdentityCells = 1024;
  static size_t IdentityCellOf(const void* entry) {
    uintptr_t x = uintptr_t(entry) >> 4;
    x ^= x >> 10;
    return size_t(x) & (kIdentityCells - 1);
  }
  // sa2::native::FetchCoherent(*entry, vs_microcode_), memoized per VS load generation (a pure function of both).
  bool FetchCoherent(const sa2::native::ShaderEntry* entry) {
    if (!identity_flat_) return sa2::native::FetchCoherent(*entry, vs_microcode_);
    IdentityCell& cell = fc_cells_[IdentityCellOf(entry)];
    if (cell.candidate != entry || cell.generation != vs_identity_generation_) {
      cell = {entry, vs_identity_generation_, sa2::native::FetchCoherent(*entry, vs_microcode_)};
    } else if (check_identity_ > 0) {
      --check_identity_;
      if (cell.result != sa2::native::FetchCoherent(*entry, vs_microcode_)) {
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] DIFFERENCE: flat fetch-coherence memo disagrees with the direct check (entry n{}); the old "
                     "path from now on", entry->number);
        identity_flat_ = false;
        return sa2::native::FetchCoherent(*entry, vs_microcode_);
      }
    }
    return cell.result;
  }
  bool MatchesPixelShaderIdentity(const sa2::native::ShaderEntry* candidate) {
    if (!candidate) return false;
    if (!identity_flat_) return ComputePixelIdentity(candidate);
    IdentityCell& cell = ps_cells_[IdentityCellOf(candidate)];
    if (cell.candidate != candidate || cell.generation != ps_identity_generation_) {
      cell = {candidate, ps_identity_generation_, ComputePixelIdentity(candidate)};
    } else if (check_identity_ > 0) {
      --check_identity_;
      if (cell.result != ComputePixelIdentity(candidate)) {
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] DIFFERENCE: flat PS identity memo disagrees with the direct compare (entry n{}); the "
                     "old path from now on", candidate->number);
        identity_flat_ = false;
        return ComputePixelIdentity(candidate);
      }
    }
    return cell.result;
  }

  bool MatchesVertexShaderIdentity(const sa2::native::ShaderEntry* candidate) {
    if (!candidate) return false;
    if (identity_flat_) {
      IdentityCell& cell = vs_cells_[IdentityCellOf(candidate)];
      if (cell.candidate != candidate || cell.generation != vs_identity_generation_) {
        cell = {candidate, vs_identity_generation_, ComputeVertexIdentity(candidate)};
      } else if (check_identity_ > 0) {
        --check_identity_;
        if (cell.result != ComputeVertexIdentity(candidate)) {
          sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
          REXLOG_ERROR("[native] DIFFERENCE: flat VS identity memo disagrees with the direct compare (entry n{}); the "
                       "old path from now on", candidate->number);
          identity_flat_ = false;
          return ComputeVertexIdentity(candidate);
        }
      }
      return cell.result;
    }
    auto& cached = vs_identity_cache_[candidate];
    if (cached.first != vs_identity_generation_) {
      cached.first = vs_identity_generation_;
      cached.second = ComputeVertexIdentity(candidate);
    }
    return cached.second;
  }

  // Once per (loaded program, candidate): every differing word between the loaded VS and the library candidate
  // (same size only), so an unresolved shader can be understood or added to the library without a debugger.
  void LogVertexWordDiff(const sa2::native::ShaderEntry& candidate) {
    if (!candidate.vertices || vs_microcode_.empty() || vs_microcode_.size() != candidate.microcode.size() ||
        vs_diff_logged_.size() >= 40) return;
    const uint64_t key = current_vs_hash_ ^ (uint64_t(candidate.number) * 0x9E3779B97F4A7C15ull);
    if (!vs_diff_logged_.insert(key).second) return;
    std::string diffs;
    size_t count = 0;
    for (size_t w = 0; w < vs_microcode_.size(); ++w) {
      if (vs_microcode_[w] == candidate.microcode[w]) continue;
      if (++count <= 24)
        diffs += fmt::format(" [w{} i{}.{} {:08X}>{:08X}]", w, w / 3, w % 3, vs_microcode_[w], candidate.microcode[w]);
    }
    std::string declared;
    for (const auto& element : candidate.elements) declared += fmt::format(" {}", element.instruction);
    REXLOG_WARN("[native] VS word diff loaded_hash={:016X} words={} candidate n{} differing_words={} "
                "(loaded>library, first 24) declared_fetch_instr:{} diffs:{}",
                current_vs_hash_, vs_microcode_.size(), candidate.number, count, declared, diffs);
  }
  std::unordered_set<uint64_t> vs_diff_logged_;

  bool AcceptVertexShaderIdentity(const sa2::native::ShaderEntry* candidate,
                                  const char* provenance) {
    if (!candidate) return false;
    // sa2_cpu_ring_identity_accept_fast: the flat memo cell of this candidate already holds "matches" for the current
    // load generation, i.e. the comparison below was made for this very program and passed. The mismatch memo only
    // receives pairs whose comparison failed, so it cannot hold this one: answer without its hash-set lookup. While
    // the self-check counter runs (and in verify mode) the lookup is still made and must agree.
    if (identity_accept_fast_ && identity_flat_) {
      const IdentityCell& cell = vs_cells_[IdentityCellOf(candidate)];
      if (cell.candidate == candidate && cell.generation == vs_identity_generation_ && cell.result) {
        if (check_identity_ == 0) {
          g_identity_accept_fast.UseSampled();
          return true;
        }
        sa2::enh::VerifyCheck();
        if (vs_mismatch_memo_.count(VsMismatchKey{candidate, current_vs_hash_, vs_microcode_.size()})) {
          sa2::enh::VerifyDifference();
          REXLOG_ERROR("[native] DIFFERENCE: VS identity fast accept (entry n{}) but the mismatch memo holds the pair; "
                       "the old path from now on", candidate->number);
          identity_accept_fast_ = false;
        }
      }
    }
    // sa2-nx: a (library candidate, loaded program) pair that failed once fails every time: both are immutable (the
    // loaded words are named by their XXH3 and size). SA2 re-checked the same few D3D-rewritten pairs ~10,000 times
    // a second (word comparison, a dump attempt and, every 256th, a hash and a warning). Positive results are not
    // memoized: the comparison stays the only way to accept.
    const VsMismatchKey memo_key{candidate, current_vs_hash_, vs_microcode_.size()};
    if (vs_mismatch_memo_.count(memo_key)) {
      ++vs_mismatch_memo_hits_;
      return false;
    }
    if (MatchesVertexShaderIdentity(candidate)) return true;
    vs_mismatch_memo_.insert(memo_key);
    DumpLoadedVertexVariant(*candidate);
    LogVertexWordDiff(*candidate);
    const uint64_t mismatch = ++vs_identity_mismatches_;
    if (mismatch <= 32) {
      // Diagnostic only: mirror the predicate's rejection order without relaxing
      // any field. Loaded words are raw; the library words are already masked.
      constexpr uint32_t masks[3]{0x0007FFFFu, 0x80000FFFu, 0x80000000u};
      const char* reason = "unknown";
      size_t failed_word = SIZE_MAX;
      uint32_t failed_mask = UINT32_MAX;
      if (!candidate->vertices) reason = "wrong-stage";
      else if (vs_microcode_.empty()) reason = "empty-loaded";
      else if (vs_microcode_.size() != candidate->microcode.size()) reason = "word-count";
      else {
        bool failed = false;
        for (const auto& element : candidate->elements) {
          const size_t start = size_t(element.instruction) * 3;
          if (start + 3 > candidate->microcode.size()) {
            reason = "declared-fetch-out-of-bounds";
            failed_word = start;
            failed = true;
            break;
          }
          if ((candidate->microcode[start] & 0x1Fu) != 0u) {
            reason = "declared-fetch-opcode";
            failed_word = start;
            failed = true;
            break;
          }
          for (size_t lane = 0; lane < 3; ++lane) {
            if ((candidate->microcode[start + lane] & masks[lane]) !=
                candidate->microcode[start + lane]) {
              reason = "candidate-not-normalized";
              failed_word = start + lane;
              failed_mask = masks[lane];
              failed = true;
              break;
            }
          }
          if (failed) break;
          if (!VertexFetchSwizzleRepresentable(candidate->microcode[start + 1] & 0xFFFu,
                                               vs_microcode_[start + 1] & 0xFFFu)) {
            reason = "unrepresentable-fetch-swizzle";
            failed_word = start + 1;
            failed_mask = 0xFFFu;
            failed = true;
            break;
          }
        }
        if (!failed) {
          for (size_t word = 0; word < vs_microcode_.size(); ++word) {
            uint32_t mask = UINT32_MAX;
            for (const auto& element : candidate->elements) {
              const size_t start = size_t(element.instruction) * 3;
              if (word >= start && word - start < 3) {
                mask = word - start == 1 ? 0x80000000u : masks[word - start];
                break;
              }
            }
            if ((vs_microcode_[word] & mask) != (candidate->microcode[word] & mask)) {
              reason = mask == UINT32_MAX ? "non-fetch-word" : "masked-fetch-word";
              failed_word = word;
              failed_mask = mask;
              break;
            }
          }
        }
      }
      const uint32_t raw = failed_word < vs_microcode_.size() ? vs_microcode_[failed_word] : 0;
      const uint32_t selected = failed_word < candidate->microcode.size()
                                    ? candidate->microcode[failed_word] : 0;
      std::string declared;
      for (const auto& element : candidate->elements)
        declared += fmt::format(" {}", element.instruction);
      REXLOG_WARN("[native] VS identity detail mismatch={} reason={} word={} instruction={} lane={} "
                  "loaded={:08X} selected={:08X} mask={:08X} masked_loaded={:08X} declared_triples={}",
                  mismatch, reason, failed_word, failed_word / 3, failed_word % 3,
                  raw, selected, failed_mask, raw & failed_mask, declared);
    }
    if (mismatch <= 32 || (mismatch & 255) == 0) {
      const uint64_t selected_hash = XXH3_64bits(candidate->microcode.data(),
                                                candidate->microcode.size() * sizeof(uint32_t));
      REXLOG_WARN("[native] VS identity mismatch {} draw={} packet={:08X} "
                  "provenance={} loaded_host_hash={:016X} loaded_words={} selected_n={} "
                  "selected_stage={} selected_normalized_host_hash={:016X} selected_words={} "
                  "selected_container={:016X}",
                  mismatch, draws_, packet_address_, provenance,
                  current_vs_hash_, vs_microcode_.size(), candidate->number,
                  candidate->vertices ? "VS" : "PS", selected_hash, candidate->microcode.size(),
                  candidate->shader ? candidate->shader->fingerprint : 0);
    }
    return false;
  }

  const sa2::native::ShaderEntry* PreferVertexShaderCandidate(
      const sa2::native::ShaderEntry* current,
      const sa2::native::ShaderEntry* candidate, const char* provenance) {
    // Keep diagnostics/variant discovery independent of the rollout policy.
    if (!candidate || !AcceptVertexShaderIdentity(candidate, provenance)) return current;
    return ShouldReplaceVertexCandidate(current != nullptr, MatchesVertexShaderIdentity(current),
                                        true, MatchesVertexShaderIdentity(candidate),
                                        false)
               ? candidate : current;
  }

  bool AcceptPixelShaderIdentity(const sa2::native::ShaderEntry* candidate,
                                  const char* provenance) {
    if (!candidate) return false;
    const bool matches = MatchesPixelShaderIdentity(candidate);
    if (matches) return true;
    const uint64_t mismatch = ++ps_identity_mismatches_;
    if (mismatch <= 32 || (mismatch & 255) == 0) {
      const uint64_t selected_hash = XXH3_64bits(candidate->microcode.data(),
                                                candidate->microcode.size() * sizeof(uint32_t));
      REXLOG_WARN("[native] PS identity mismatch {} draw={} packet={:08X} "
                  "provenance={} loaded_host_hash={:016X} loaded_words={} selected_n={} "
                  "selected_stage={} selected_host_hash={:016X} selected_words={} "
                  "selected_container={:016X}",
                  mismatch, draws_, packet_address_, provenance,
                  current_ps_hash_, ps_microcode_.size(), candidate->number,
                  candidate->vertices ? "VS" : "PS", selected_hash, candidate->microcode.size(),
                  candidate->shader ? candidate->shader->fingerprint : 0);
    }
    return false;
  }

  // sa2_native_raw_microcode: IdentifyShader's "same program reloaded from the same address" test on the raw
  // guest words. True = nothing to do (the stage keeps its entry, hashes and generations).
  bool ReloadsSameMicrocode(uint32_t type, const uint8_t* src, uint32_t size) {
    if (!REXCVAR_GET(sa2_native_repeated_identity)) return false;
    const auto& raw = type == 1 ? raw_ps_ : raw_vs_;
    const auto* previous_entry = type == 1 ? ps_ : vs_;
    if (!previous_entry || !(type == 1 ? raw_ps_valid_ : raw_vs_valid_) || raw.size() != size ||
        (type == 1 ? last_ps_load_address_ : last_vs_load_address_) != im_load_address_ ||
        std::memcmp(raw.data(), src, size_t(size) * 4) != 0) {
      return false;
    }
    if (check_raw_ > 0) {
      --check_raw_;
      const auto& swapped = type == 1 ? ps_microcode_ : vs_microcode_;
      bool equal = swapped.size() == size;
      for (uint32_t i = 0; equal && i < size; ++i)
        equal = swapped[i] == rex::memory::load_and_swap<uint32_t>(src + size_t(i) * 4);
      if (!equal) {
        sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
        REXLOG_ERROR("[native] DIFFERENCE: raw microcode compare said 'same program' ({} words, type {}, address "
                     "{:08X}) but the swapped words differ; the old path from now on", size, type, im_load_address_);
        raw_microcode_ = false;
        return false;
      }
    }
    ++shader_loads_reused_;
    return true;
  }

  void IdentifyShader(uint32_t type) {
    if (type > 1) return;
    // The same program reloaded from the same address (D3D re-sends it for most draws): keep the identified
    // entry, the hashes and the generations, instead of copying, hashing twice and matching again.
    if (REXCVAR_GET(sa2_native_repeated_identity)) {
      const auto& previous = type == 1 ? ps_microcode_ : vs_microcode_;
      const auto* previous_entry = type == 1 ? ps_ : vs_;
      if (previous_entry && previous.size() == microcode_.size() &&
          (type == 1 ? last_ps_load_address_ : last_vs_load_address_) == im_load_address_ &&
          std::memcmp(previous.data(), microcode_.data(), microcode_.size() * sizeof(uint32_t)) == 0) {
        ++shader_loads_reused_;
        return;
      }
    }
    (type == 1 ? last_ps_load_address_ : last_vs_load_address_) = im_load_address_;
    // Memo by content (sa2_native_identity_memo): the same program loaded again after others, from any
    // address, gets the entry identified before; the draw-time guard still checks it against this microcode.
    const uint64_t memo_hash = XXH3_64bits(microcode_.data(), microcode_.size() * sizeof(uint32_t));
    const uint64_t memo_key = memo_hash ^ (uint64_t(type) << 63) ^ (uint64_t(microcode_.size()) << 40);
    if (REXCVAR_GET(sa2_native_identity_memo)) {
      const auto it = identity_memo_.find(memo_key);
      if (it != identity_memo_.end()) {
        if (type == 1) {
          ps_microcode_ = microcode_;
          ++ps_identity_generation_;
          current_ps_hash_ = memo_hash;
          ps_ = it->second;
        } else {
          vs_microcode_ = microcode_;
          current_vs_hash_ = memo_hash;
          ++vs_identity_generation_;
          vs_ = it->second;
          vs_reorder_entry_ =
              !vs_reorder_memo_keys_.empty() && vs_reorder_memo_keys_.count(memo_key) ? it->second : nullptr;
          gen_vs_ = ++gen_microcode_;
        }
        ++shaders_identified_;
        ++identity_memo_hits_;
        return;
      }
    }
    if (type == 1) {
      ps_microcode_ = microcode_;
      ++ps_identity_generation_;
      current_ps_hash_ = memo_hash;
    } else {
      // Identity checks below must see THIS IM_LOAD, not the previous VS load.
      vs_microcode_ = microcode_;
      current_vs_hash_ = memo_hash;
      ++vs_identity_generation_;
      vs_reorder_entry_ = nullptr;
    }
    // A VS address identifies the object, not necessarily the variant currently in the ring: D3D
    // rewrites vertex fetches for each declaration. Prefer the actual loaded microcode (exact, then
    // the library's masked-fetch matcher), and accept the address entry only if it is compatible.
    const sa2::native::ShaderEntry* by_address =
        im_load_address_ ? ByAddress(im_load_address_) : nullptr;
    const sa2::native::ShaderEntry* entry = nullptr;
    if (type == 0) {
      const sa2::native::ShaderEntry* observed = nullptr;
      entry = ByCode(microcode_, true, &observed);
      if (observed && !AcceptVertexShaderIdentity(observed, "identify-code")) entry = nullptr;
      if (entry && (entry->microcode.size() != microcode_.size() ||
                    !sa2::native::FetchCoherent(*entry, microcode_))) {
        entry = nullptr;
      }
      // Guard OFF is permission for a last-resort legacy fallback, not for a
      // stale fast/address candidate to hide a proven current program.
      if (!MatchesVertexShaderIdentity(entry) && shaders_.loaded()) {
        const auto* matched = shaders_.Identify(true, microcode_);
        entry = PreferVertexShaderCandidate(entry, matched, "identify-library");
      }
      if (by_address && (!entry || (!MatchesVertexShaderIdentity(entry) &&
                                   MatchesVertexShaderIdentity(by_address))) &&
          by_address->microcode.size() == microcode_.size() &&
          sa2::native::FetchCoherent(*by_address, microcode_)) {
        entry = PreferVertexShaderCandidate(entry, by_address, "identify-address");
      }
      if (entry == by_address && entry) ++shaders_by_address_;
      else if (entry) ++shaders_by_code_;
      // sa2-nx (sa2_gpu_vs_fetch_reorder): no entry is this program in its own fetch order. Direct3D's patcher may have
      // moved the declared fetches to each other's slots for the vertex declaration (me_vertex_fetch_reorder.h): the
      // library shader the proof holds for becomes this load's entry, and the one candidate the identity checks accept
      // for it. Its memo cells and a mismatch memoized just above were made without it: a new generation, and that
      // pair forgotten.
      if (!entry && shaders_.loaded()) {
        if (const auto* reordered = shaders_.IdentifyReordered(microcode_)) {
          vs_reorder_entry_ = reordered;
          ++vs_identity_generation_;
          vs_mismatch_memo_.erase(VsMismatchKey{reordered, current_vs_hash_, vs_microcode_.size()});
          if (MatchesVertexShaderIdentity(reordered)) {
            entry = reordered;
            ++shaders_reordered_;
            if (REXCVAR_GET(sa2_native_identity_memo)) vs_reorder_memo_keys_.insert(memo_key);
          } else {
            vs_reorder_entry_ = nullptr;
          }
        }
      }
    } else {
      if (by_address &&
          AcceptPixelShaderIdentity(by_address, "identify-address")) {
        entry = by_address;
        ++shaders_by_address_;
      }
      if (!entry) {
        const sa2::native::ShaderEntry* observed = nullptr;
        const auto* code_entry = ByCode(microcode_, false, &observed);
        const bool accepted = observed && AcceptPixelShaderIdentity(observed, "identify-code");
        if (code_entry && accepted) {
          entry = code_entry;
          ++shaders_by_code_;
        }
      }
      if (!entry && shaders_.loaded()) {
        const auto* matched = shaders_.Identify(false, microcode_);
        if (matched && AcceptPixelShaderIdentity(matched, "identify-library")) entry = matched;
      }
    }
    (type == 0 ? vs_ : ps_) = entry;
    (type == 0 ? current_vs_hash_ : current_ps_hash_) = memo_hash;
    if (entry && REXCVAR_GET(sa2_native_identity_memo)) {
      if (identity_memo_.size() > 8192) {
        identity_memo_.clear();
        vs_reorder_memo_keys_.clear();
        if (type == 0 && entry == vs_reorder_entry_) vs_reorder_memo_keys_.insert(memo_key);
      }
      identity_memo_.emplace(memo_key, entry);
    }
    if (type == 0) {
      gen_vs_ = ++gen_microcode_;
    }
    ++(entry ? shaders_identified_ : shaders_unidentified_);
    if (!entry) {
      ++(im_load_address_ ? (type == 0 ? miss_vs_addr_ : miss_ps_addr_) : miss_immediate_);
      DumpRawMicrocode(type == 0, im_load_address_, microcode_);
      if (!im_load_address_ && shaders_.loaded()) {
        DumpMissingImmediate(shaders_, type == 0, microcode_);
      }
    }
    if (!entry && logged_misses_ < 12) {  // diagnostic: what the ring loads that the library lacks
      ++logged_misses_;
      std::string keys;
      {
        std::lock_guard<std::mutex> lock(g_by_address_mutex);
        for (const auto& [k, v] : g_by_address) {
          if ((k >> 12) == (im_load_address_ >> 12)) keys += fmt::format(" {:08X}", k);
        }
      }
      REXLOG_INFO("[native] unidentified {} IM_LOAD at {:08X} (keys in page:{}): {} words, {:08X} {:08X} {:08X} {:08X}",
                  type == 0 ? "VS" : "PS", im_load_address_, keys, microcode_.size(), microcode_.size() > 0 ? microcode_[0] : 0,
                  microcode_.size() > 1 ? microcode_[1] : 0, microcode_.size() > 2 ? microcode_[2] : 0,
                  microcode_.size() > 3 ? microcode_[3] : 0);
    }
  }

  // Opt-in discovery only, not permission to select a mismatching program. Keep
  // the SAME-size template's complete physical payload, including DEF literals;
  // changing only its instruction bytes makes offline translation auditable.
  void DumpLoadedVertexVariant(const sa2::native::ShaderEntry& candidate) {
    static const char* const folder = ShaderDumpFolder("SA2_VERTEX_VARIANTS", "variants");
    if (!folder || !*folder || !candidate.vertices || !candidate.shader || vs_microcode_.empty() ||
        vs_microcode_.size() != candidate.microcode.size() || dumped_vertex_variants_.size() >= 64) return;
    const auto& original = candidate.shader->original;
    if (original.size() < 28 || original.size() > 8u * 1024 * 1024) return;
    const uint32_t flags = rex::memory::load_and_swap<uint32_t>(original.data());
    const uint32_t virtual_bytes = rex::memory::load_and_swap<uint32_t>(original.data() + 4);
    const uint32_t physical_bytes = rex::memory::load_and_swap<uint32_t>(original.data() + 8);
    const uint32_t header = rex::memory::load_and_swap<uint32_t>(original.data() + 24);
    if ((flags & 0xFFFFFF00u) != 0x102A1100u || !(flags & 1u) || virtual_bytes > original.size() ||
        physical_bytes > original.size() - virtual_bytes || header > virtual_bytes ||
        virtual_bytes - header < 8) return;
    const uint32_t code_offset = rex::memory::load_and_swap<uint32_t>(original.data() + header);
    const uint32_t code_bytes = rex::memory::load_and_swap<uint32_t>(original.data() + header + 4);
    if (code_bytes != vs_microcode_.size() * 4 || code_offset > physical_bytes ||
        code_bytes > physical_bytes - code_offset) return;
    const uint64_t loaded_hash = XXH3_64bits(vs_microcode_.data(), vs_microcode_.size() * 4);
    const uint64_t key = XXH3_64bits_withSeed(original.data(), original.size(), loaded_hash);
    if (!dumped_vertex_variants_.insert(key).second) return;
    std::vector<uint8_t> container(original);
    for (size_t i = 0; i < vs_microcode_.size(); ++i)
      rex::memory::store_and_swap<uint32_t>(container.data() + virtual_bytes + code_offset + i * 4,
                                          vs_microcode_[i]);
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    if (ec) { REXLOG_WARN("[native] vertex variant discovery directory failed: {}", ec.message()); return; }
    const auto path = std::filesystem::path(folder) / fmt::format("vs_from_n{}_{:016x}.bin", candidate.number, key);
    const auto pending_path = std::filesystem::path(path.string() + ".pending");
    bool good = false;
    if (auto* file = std::fopen(pending_path.string().c_str(), "wbx")) {
      good = std::fwrite(container.data(), 1, container.size(), file) == container.size();
      if (std::fclose(file) != 0) good = false;
    }
    if (good) {
      std::filesystem::create_hard_link(pending_path, path, ec);
      good = !ec;
    }
    REXLOG_INFO("[native] vertex variant discovery written={} template_n={} loaded_host_hash={:016X} "
                "words={} (complete template retained, NOT verified compatible): {}",
                good, candidate.number, loaded_hash, vs_microcode_.size(), path.string());
  }

  // With SA2_SHADER_MISSING=<folder>: the container of a Direct3D vertex shader as the ring loaded
  // it (rewritten for the vertex declaration), made of the object's container header and the ring's
  // microcode, once per microcode. shaders/tools/build_shader_spirv.sh adds it to the library.
  void DumpRewrittenVertexShader(const sa2::native::ShaderEntry& object_entry) {
    static const char* const folder = ShaderDumpFolder("SA2_SHADER_MISSING", "missing");
    if (!folder || !*folder || !object_entry.shader || vs_microcode_.empty()) return;
    const uint64_t code = XXH3_64bits(vs_microcode_.data(), vs_microcode_.size() * sizeof(uint32_t));
    if (!dumped_rewritten_.insert(code).second) return;
    const auto& o = object_entry.shader->original;
    if (o.size() < 28) return;
    const uint32_t vsize = rex::memory::load_and_swap<uint32_t>(o.data() + 4);
    const uint32_t sh = rex::memory::load_and_swap<uint32_t>(o.data() + 24);
    if (vsize > o.size() || sh + 8 > vsize) return;
    const uint32_t bytes = uint32_t(vs_microcode_.size() * 4);
    std::vector<uint8_t> container(o.begin(), o.begin() + vsize);
    rex::memory::store_and_swap<uint32_t>(container.data() + 8, bytes);   // physical size
    rex::memory::store_and_swap<uint32_t>(container.data() + sh, 0);      // microcode offset
    rex::memory::store_and_swap<uint32_t>(container.data() + sh + 4, bytes);
    container.resize(size_t(vsize) + bytes);
    for (size_t i = 0; i < vs_microcode_.size(); ++i) {
      rex::memory::store_and_swap<uint32_t>(container.data() + vsize + i * 4, vs_microcode_[i]);
    }
    std::error_code ec;
    std::filesystem::create_directories(folder, ec);
    const auto path = std::filesystem::path(folder) /
                      fmt::format("vs_{:016x}.bin", XXH3_64bits(container.data(), container.size()));
    if (std::FILE* f = std::fopen(path.string().c_str(), "wb")) {
      std::fwrite(container.data(), 1, container.size(), f);
      std::fclose(f);
      REXLOG_INFO("[native] rewritten Direct3D vertex shader kept: {}", path.string());
    }
  }

  // The library entries of the VS and PS objects a record names (nullptr = unknown object).
  void LookupObjectEntries(uint32_t object_vs, uint32_t object_ps, const sa2::native::ShaderEntry*& vs,
                           const sa2::native::ShaderEntry*& ps) {
    vs = ps = nullptr;
    if (pairing_fast_ && LookupObject(object_vs, vs) && LookupObject(object_ps, ps)) {
      if (check_objects_ > 0) {
        --check_objects_;
        const sa2::native::ShaderEntry* want_vs = nullptr;
        const sa2::native::ShaderEntry* want_ps = nullptr;
        {
          std::lock_guard<std::mutex> lock(g_by_address_mutex);
          auto a = g_entry_of_object.find(object_vs);
          auto b = g_entry_of_object.find(object_ps);
          if (a != g_entry_of_object.end()) want_vs = a->second;
          if (b != g_entry_of_object.end()) want_ps = b->second;
        }
        if (want_vs != vs || want_ps != ps) {  // the game thread may have just added or replaced an entry: look again
          const sa2::native::ShaderEntry* again_vs = nullptr;
          const sa2::native::ShaderEntry* again_ps = nullptr;
          LookupObject(object_vs, again_vs);
          LookupObject(object_ps, again_ps);
          if (again_vs != want_vs || again_ps != want_ps) {
            sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
            REXLOG_ERROR("[native] DIFFERENCE: lock-free object table: VS {:08X} {} / {}, PS {:08X} {} / {} (table / "
                         "map); the old path from now on", object_vs, static_cast<const void*>(again_vs),
                         static_cast<const void*>(want_vs), object_ps, static_cast<const void*>(again_ps),
                         static_cast<const void*>(want_ps));
            pairing_fast_ = false;
          }
          vs = want_vs;
          ps = want_ps;
        }
      }
      return;
    }
    vs = ps = nullptr;
    std::lock_guard<std::mutex> lock(g_by_address_mutex);
    auto a = g_entry_of_object.find(object_vs);
    auto b = g_entry_of_object.find(object_ps);
    if (a != g_entry_of_object.end()) vs = a->second;
    if (b != g_entry_of_object.end()) ps = b->second;
  }

  // By packet address: the record whose command-buffer start is the closest one below this DRAW_INDX.
  bool PairByAddress(uint32_t type) {
    DrawRecord r;
    if (direct_pair_valid_ && direct_pair_draw_ == packet_address_) {
      // sa2_cpu_direct_pair: this draw's own record, carried by its direct record. The table's rule all the same: at
      // most 0x2000 below the packet and of the same type.
      direct_pair_valid_ = false;
      ++direct_pairs_used_;
      if (sa2::enh::VerifyMode() && RecordTable()) [[unlikely]] {
        // Verify mode pushed the record to the queue too: the table must hold the same record for this packet.
        DrainRecords();
        DrawRecordSlot* best = g_records.FindScan(packet_address_);
        if (best && best->start == direct_pair_.start) {
          ++direct_pairs_checked_;
          sa2::enh::VerifyCheck();
          const DrawRecord& t = best->record;
          if (t.type != direct_pair_.type || t.r5 != direct_pair_.r5 || t.r6 != direct_pair_.r6 ||
              t.r7 != direct_pair_.r7 || t.vs != direct_pair_.vs || t.ps != direct_pair_.ps ||
              t.device != direct_pair_.device) {
            sa2::enh::VerifyDifference();
            REXLOG_ERROR("[native] DIFFERENCE: direct pair for packet {:08X} (VS {:08X} PS {:08X} type {}) differs from "
                         "the table's record (VS {:08X} PS {:08X} type {})", packet_address_, direct_pair_.vs,
                         direct_pair_.ps, direct_pair_.type, t.vs, t.ps, t.type);
          }
          g_records.Consume(best);
        } else {
          ++direct_pairs_table_missed_;  // evicted from its bucket (the table holds 8 records per 8 KB)
        }
      }
      if (packet_address_ - direct_pair_.start > 0x2000 || direct_pair_.type != type) return false;
      r = direct_pair_;
    } else if (RecordTable()) {
      DrainRecords();  // the table belongs to the ring thread: no lock
      DrawRecordSlot* best = nullptr;
      if (pairing_fast_) {
        best = g_records.FindDense(packet_address_);
        if (check_pairing_ > 0) {
          --check_pairing_;
          DrawRecordSlot* old = g_records.FindScan(packet_address_);
          if (old != best) {
            sa2::enh::VerifyDifference();  // sa2_cpu_verify: counted in the [verify] line
            REXLOG_ERROR("[native] DIFFERENCE: fast record lookup for packet {:08X} chose slot {} but the scan chose {}; "
                         "the old path from now on", packet_address_, best ? int64_t(g_records.IndexOf(best)) : -1,
                         old ? int64_t(g_records.IndexOf(old)) : -1);
            pairing_fast_ = false;
            g_record_fast = false;
            best = old;
          }
        }
      } else {
        best = g_records.FindScan(packet_address_);
      }
      // The map's rule: the nearest record at or below the packet, within 0x2000 and of the same type.
      if (!best || packet_address_ - best->start > 0x2000 || best->record.type != type) return false;
      r = std::move(best->record);
      g_records.Consume(best);
    } else {
      std::lock_guard<std::mutex> lock(g_draw_mutex);
      auto it = g_draw_by_address.upper_bound(packet_address_);
      if (it == g_draw_by_address.begin()) return false;
      --it;
      if (packet_address_ - it->first > 0x2000 || it->second.type != type) return false;
      r = it->second;
      g_draw_by_address.erase(it);
    }
    // sa2_api_discover_draws: the register file as the PM4 left it for this draw, beside the device copy.
    if (r.snapshot >= 0) [[unlikely]] discover::Write(r, registers_.data(), kRegisterCount, packet_address_);
    const sa2::native::ShaderEntry* vs = nullptr;
    const sa2::native::ShaderEntry* ps = nullptr;
    LookupObjectEntries(r.vs, r.ps, vs, ps);
    if (vs && !AcceptVertexShaderIdentity(vs, "pair-address-object")) vs = nullptr;
    if (vs && vs->binding_per_register) {
      // Direct3D rewrote this vertex shader for the vertex declaration: the variant the ring loaded is
      // its own library entry, found by microcode (vs_). Without it, keep a copy to extend the library.
      if (vs_ && vs_->binding_per_register) {
        vs = vs_;
      } else {
        DumpRewrittenVertexShader(*vs);
      }
    }
    // Audit the object even when the old register-linked override would hide a
    // stale entry. The guard protects every path, not just shader loading.
    if (ps && !AcceptPixelShaderIdentity(ps, "pair-address-object")) ps = nullptr;
    if (ps && ps->binding_per_register) {
      // Direct3D reuses its small internal PS objects and overwrites the same ring allocation with
      // different microcode. The object entry therefore describes only the first variant observed;
      // the IM_LOAD at this draw is authoritative, exactly as for rewritten register-linked VS.
      ps = ps_ && ps_->binding_per_register ? ps_ : nullptr;
    }
    // The IM_LOAD is the authoritative patched variant. A Draw* record may point at the
    // unpatched object, or at a different variant queued ahead of this indirect buffer.
    // Do not combine such an entry with the current ring microcode: ComputeEntry would
    // reject it with cause 20 (or derive the wrong vertex layout).
    if (vs && AcceptVertexShaderIdentity(vs, "pair-address-selected") &&
        vs->microcode.size() == vs_microcode_.size() &&
        FetchCoherent(vs)) {
      draw_vs_ = PreferVertexShaderCandidate(draw_vs_, vs, "pair-address-preference");
    }
    if (ps && AcceptPixelShaderIdentity(ps, "pair-address-selected")) draw_ps_ = ps;
    ++draws_paired_by_address_;
    if (!vs) {
      ++paired_no_vs_;
      if (missing_objects_.size() < 32 || missing_objects_.count(r.vs)) ++missing_objects_[r.vs];
    }
    if (!ps) {
      ++paired_no_ps_;
      if (missing_objects_.size() < 32 || missing_objects_.count(r.ps)) ++missing_objects_[r.ps];
    }
    return true;
  }

  // The first of up to 8 pending records with the same primitive type and
  // a matching count gives the VS and PS objects; otherwise the ring's own identification.
  void PairDraw() {
    identity_memo_hit_ = false;
    identity_memo_pending_ = false;
    if (g_identity_memo.active && direct_pair_valid_ && direct_pair_draw_ == packet_address_ &&
        direct_pair_.snapshot < 0) {
      // sa2_cpu_ring_identity_memo: everything the pairing and the draw-time identity checks read.
      IdentityMemoKey key;
      key.object_vs = direct_pair_.vs;
      key.object_ps = direct_pair_.ps;
      key.accepted = packet_address_ - direct_pair_.start <= 0x2000 &&
                     direct_pair_.type == (Register(kRegVgtDrawInitiator) & 0x3F);
      key.objects_version = g_objects_version.load(std::memory_order_acquire);
      key.vs = vs_;
      key.ps = ps_;
      key.vs_generation = vs_identity_generation_;
      key.ps_generation = ps_identity_generation_;
      key.pairing_fast = pairing_fast_;
      key.identity_flat = identity_flat_;
      // Only accepted pairs: a refused one goes on to the replay and queue paths, whose state is not in the key.
      if (key.accepted && identity_memo_valid_ && key == identity_memo_last_key_ && !sa2::enh::VerifyMode()) {
        direct_pair_valid_ = false;
        ++direct_pairs_used_;
        if (key.accepted) ++draws_paired_by_address_;
        draw_vs_ = identity_memo_pair_vs_;
        draw_ps_ = identity_memo_pair_ps_;
        identity_memo_hit_ = true;
        g_identity_memo.UseSampled();
        return;
      }
      identity_memo_key_ = key;
      identity_memo_pending_ = key.accepted;  // Draw() stores the outcome under this key
    }
    draw_vs_ = vs_ && AcceptVertexShaderIdentity(vs_, "pair-initial-ring") ? vs_ : nullptr;
    draw_ps_ = ps_;
    if (PairByAddress(Register(kRegVgtDrawInitiator) & 0x3F)) {
      // usf4-nx: remember the pairing for the predicated-tiling replays of this packet (below), as the queue path
      // does. PairByAddress consumes the draw's record, so without this tiles 2-3 found neither a record nor a
      // remembered pairing and drew with the ring's own pick (USF4's hit effects: VS n78 instead of n434).
      NotePairedPacket({swaps_, draw_vs_, draw_ps_});
      return;
    }
    // Predicated tiling replays the same command buffer once per EDRAM tile: a DRAW_INDX packet at an
    // address already paired in this frame reuses that pairing and consumes no record.
    if (auto it = FindPairedPacket(); it) {
      // usf4-nx: the replay is the same packet with the same state, so it takes the first tile's vertex shader
      // whenever that one still matches the loaded microcode. PreferVertexShaderCandidate kept the ring's current
      // entry when both matched: two library shaders that differ only in their vertex fetches (masked by the
      // identity check) then drew tile 1 with the bound object's shader and tiles 2-3 with the other, whose vertex
      // layout gave USF4's hit effects their black bars (PS n538 with VS n434 in tile 1, n78 in tiles 2-3; frame
      // trace, 2026-10-11). SA2 has no predicated tiling, so the path was never taken there.
      if (it->vs && AcceptVertexShaderIdentity(it->vs, "pair-replay"))
        draw_vs_ = it->vs;
      else
        draw_vs_ = PreferVertexShaderCandidate(draw_vs_, it->vs, "pair-replay");
      if (it->ps && AcceptPixelShaderIdentity(it->ps, "pair-replay"))
        draw_ps_ = it->ps;
      ++draws_replayed_;
      return;
    }
    if (!pairing_fast_) {  // nothing ever fills g_draw_queue (records go through the lock-free queue)
      std::lock_guard<std::mutex> lock(g_draw_mutex);
      while (pending_.size() < 4096 && !g_draw_queue.empty()) {
        pending_.push_back(g_draw_queue.front());
        g_draw_queue.pop_front();
      }
    }
    const uint32_t initiator = Register(kRegVgtDrawInitiator);
    const uint32_t type = initiator & 0x3F, count = initiator >> 16;
    const size_t n = std::min<size_t>(pending_.size(), 64);
    for (size_t i = 0; i < n; ++i) {
      const DrawRecord& r = pending_[i];
      if (r.type != type) continue;
      if (r.r5 != count && r.r6 != count && r.r7 != count && CountForPrimitives(type, r.r7) != count &&
          CountForPrimitives(type, r.r6) != count) {
        continue;
      }
      const sa2::native::ShaderEntry* vs = nullptr;
      const sa2::native::ShaderEntry* ps = nullptr;
      LookupObjectEntries(r.vs, r.ps, vs, ps);
      pending_.erase(pending_.begin(), pending_.begin() + std::ptrdiff_t(i + 1));
      ++draws_paired_;
      draw_vs_ = PreferVertexShaderCandidate(draw_vs_, vs, "pair-queue");
      if (ps && AcceptPixelShaderIdentity(ps, "pair-queue")) draw_ps_ = ps;
      NotePairedPacket({swaps_, draw_vs_, ps});
      return;
    }
    ++draws_unpaired_;
    if (logged_unpaired_ < 30 && draws_ > 60000 && type != 1) {
      ++logged_unpaired_;
      std::string p;
      for (size_t i = 0; i < pending_.size() && i < 4; ++i) {
        p += fmt::format(" [t{} {} {} {}]", pending_[i].type, pending_[i].r5, pending_[i].r6, pending_[i].r7);
      }
      REXLOG_INFO("[native] unpaired ring draw: type {} count {} source {}; pending {}:{}", type, count,
                  (initiator >> 6) & 3, pending_.size(), p);
    }
  }

  // A proven rectangle (CPU-executed VS) that overwrites every covered pixel of its views:
  // no discard, full sample mask, depth written with ALWAYS and stencil off, or color fully masked
  // without blending. The EDRAM tiles fully inside need no transfer from their previous owner
  // (xenia's "spurious ownership transfer round trips"; here the shadow maps and the scene share EDRAM).
  void ProveRectangleList(sa2::native::SubmissionDraw& request, uint32_t rects) {
    if (!request.ps) return;
    // sa2_cpu_ring_rect_list_in_place: the estimator reads the register file itself, with the two words it needs
    // changed set right before each estimate and restored right after (nothing else runs in between), instead of an
    // 80 KB copy per call. It sees the same words either way.
    const bool in_place = g_rect_list_in_place.active;
    std::vector<uint32_t> regs_copy;
    if (!in_place) regs_copy = registers_;
    else g_rect_list_in_place.UseSampled();
    std::vector<uint32_t>& regs = in_place ? registers_ : regs_copy;
    const uint32_t initiator = regs[kRegVgtDrawInitiator];
    const uint32_t offset = regs[rex::graphics::XE_GPU_REG_VGT_INDX_OFFSET];
    std::optional<std::array<int32_t, 4>> best, bounds;
    int64_t best_area = -1;
    uint32_t slots = 0;
    uint32_t max_y = 0;
    for (uint32_t k = 0; k < rects; ++k) {
      regs[kRegVgtDrawInitiator] = (initiator & 0xFFFFu) | (3u << 16);
      regs[rex::graphics::XE_GPU_REG_VGT_INDX_OFFSET] = offset + 3u * k;
      NativeDrawExtentEstimator::Diagnostics diagnostic;
      const auto estimate = draw_extent_estimator_->Estimate(regs, vs_microcode_, &diagnostic);
      if (in_place) {
        regs[kRegVgtDrawInitiator] = initiator;
        regs[rex::graphics::XE_GPU_REG_VGT_INDX_OFFSET] = offset;
      }
      if (!estimate) { ++multi_rect_rejects_[0]; return; }  // every rectangle must be proven
      max_y = std::max(max_y, *estimate);
      StencilClearOwnershipInput proof;
      proof.depth_control = Register(rex::graphics::XE_GPU_REG_RB_DEPTHCONTROL);
      proof.stencil_front = Register(rex::graphics::XE_GPU_REG_RB_STENCILREFMASK);
      proof.stencil_back = Register(rex::graphics::XE_GPU_REG_RB_STENCILREFMASK_BF);
      proof.color_control = Register(rex::graphics::XE_GPU_REG_RB_COLORCONTROL);
      proof.su_sc_mode = Register(rex::graphics::XE_GPU_REG_PA_SU_SC_MODE_CNTL);
      proof.vte_cntl = Register(rex::graphics::XE_GPU_REG_PA_CL_VTE_CNTL);
      const uint32_t tl = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL);
      const uint32_t br = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR);
      const uint32_t window_offset = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_OFFSET);
      const auto sign15 = [](uint32_t v) { return int32_t(v & 0x7FFF) - int32_t((v & 0x4000) << 1); };
      const int32_t wx = (tl >> 31) ? 0 : sign15(window_offset);
      const int32_t wy = (tl >> 31) ? 0 : sign15(window_offset >> 16);
      const uint32_t screen_tl = Register(rex::graphics::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL);
      const uint32_t screen_br = Register(rex::graphics::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR);
      proof.scissor_x0 = uint32_t(std::max({0, int32_t(tl & 0x3FFF) + wx, sign15(screen_tl)}));
      proof.scissor_y0 = uint32_t(std::max({0, int32_t((tl >> 16) & 0x3FFF) + wy, sign15(screen_tl >> 16)}));
      proof.scissor_x1 = uint32_t(std::max(0, std::min(int32_t(br & 0x3FFF) + wx, sign15(screen_br))));
      proof.scissor_y1 = uint32_t(std::max(0, std::min(int32_t((br >> 16) & 0x3FFF) + wy, sign15(screen_br >> 16))));
      proof.post_vs_position = diagnostic.position;
      proof.xy_raster = diagnostic.xy_raster;
      proof.ps_no_discard = ProveStraightLinePixelNoKill(ps_microcode_) ||
          (REXCVAR_GET(sa2_native_edram4_ps_no_kill) && ProvePixelNoKillAnywhere(ps_microcode_));
      proof.ps_no_depth_export = !(request.ps->outputs & 0x10);
      proof.sample_coverage_proven_full = (Register(rex::graphics::XE_GPU_REG_PA_SC_AA_MASK) & 0xFFFFu) == 0xFFFFu;
      sa2::native::SubmissionDraw one = request;
      one.edram_overwrite_rect.reset();
      ProveFullOverwrite(one, proof);
      if (!one.edram_overwrite_rect) { ++multi_rect_rejects_[1]; return; }
      if (k && one.edram_overwrite_slots != slots) { ++multi_rect_rejects_[2]; return; }
      request.edram_overwrite_rects[k] = *one.edram_overwrite_rect;
      slots = one.edram_overwrite_slots;
      const auto& r = *one.edram_overwrite_rect;
      const int64_t area = int64_t(r[2] - r[0]) * (r[3] - r[1]);
      if (area > best_area) { best_area = area; best = r; }
      const auto& b = one.edram_bounds_rect ? *one.edram_bounds_rect : r;
      if (!bounds) bounds = b;
      else bounds = std::array<int32_t, 4>{std::min((*bounds)[0], b[0]), std::min((*bounds)[1], b[1]),
                                           std::max((*bounds)[2], b[2]), std::max((*bounds)[3], b[3])};
    }
    // Pixels the coverage rectangles exclude (partial pixels at the edges) are not overwritten, but they are
    // covered by the bounds only if covered at all: the proven rects are exactly the covered pixel sets.
    request.edram_used_height_estimate = max_y;
    request.edram_overwrite_rect = best;
    request.edram_overwrite_slots = slots;
    request.edram_bounds_rect = bounds;
    request.edram_overwrite_rect_count = rects;
    ++multi_rect_proven_;
  }
  uint64_t multi_rect_proven_ = 0;
  std::unordered_map<uint64_t, const sa2::native::ShaderEntry*> identity_memo_;
  uint64_t identity_memo_hits_ = 0;
  uint64_t exact_edges_ = 0;
  uint32_t edges_traces_ = 0, alias_traces_ = 0, traces_depth_ = 0;
  std::array<uint64_t, 3> multi_rect_rejects_{};  // estimator, overwrite proof, slot mismatch

  void ProveFullOverwrite(sa2::native::SubmissionDraw& request, const StencilClearOwnershipInput& proof) {
    if (!REXCVAR_GET(sa2_native_edram4_overwrite) || !request.ps) return;
    if (!proof.ps_no_discard) { ++overwrite_rejects_[0]; return; }
    if (!proof.sample_coverage_proven_full) { ++overwrite_rejects_[1]; return; }
    const auto& pos = proof.post_vs_position;
    float x0 = pos[0][0], x1 = pos[0][0], y0 = pos[0][1], y1 = pos[0][1];
    for (const auto& v : pos) {
      x0 = std::min(x0, v[0]); x1 = std::max(x1, v[0]);
      y0 = std::min(y0, v[1]); y1 = std::max(y1, v[1]);
    }
    // OpenGL pixel centers (PA_SU_VTX_CNTL.pix_center = 1, at i + 0.5): the same coverage as D3D centers
    // with every position half a pixel lower. The rest of the proof uses the D3D convention.
    if (REXCVAR_GET(sa2_native_edram4_center_ogl) &&
        (Register(rex::graphics::XE_GPU_REG_PA_SU_VTX_CNTL) & 1)) {
      x0 -= 0.5f; x1 -= 0.5f; y0 -= 0.5f; y1 -= 0.5f;
    }
    // D3D9 pixel space: pixel i spans [i - 0.5, i + 0.5), so [-0.5, W - 0.5] covers pixels 0..W-1.
    // A small epsilon rejects rectangles that stop short of a pixel by rounding.
    constexpr float kEps = 1.0f / 64.0f;
    std::array<int32_t, 4> rect{int32_t(std::ceil(x0 + 0.5f - kEps)), int32_t(std::ceil(y0 + 0.5f - kEps)),
                                int32_t(std::floor(x1 + 0.5f + kEps)), int32_t(std::floor(y1 + 0.5f + kEps))};
    rect[0] = std::max<int32_t>(rect[0], int32_t(proof.scissor_x0));
    rect[1] = std::max<int32_t>(rect[1], int32_t(proof.scissor_y0));
    rect[2] = std::min<int32_t>(rect[2], int32_t(proof.scissor_x1));
    rect[3] = std::min<int32_t>(rect[3], int32_t(proof.scissor_y1));
    if (rect[2] <= rect[0] || rect[3] <= rect[1]) { ++overwrite_rejects_[2]; return; }
    uint32_t slots = 0;
    const uint32_t dc = proof.depth_control;
    const bool z_enable = dc & 2, z_write = dc & 4, stencil = dc & 1;
    // Stencil is fully replaced when every sample passes (func ALWAYS; depth func ALWAYS so zfail never
    // happens), the pass op is REPLACE and the write mask is 0xFF, on both faces (D3D Clear rectangles).
    const auto stencil_replaced = [&](uint32_t func, uint32_t zpass, uint32_t refmask) {
      // REPLACE (2) writes the reference, ZERO (1) writes 0: either way every bit gets a known value.
      return func == 7 && (zpass == 2 || zpass == 1) && ((refmask >> 16) & 0xFFu) == 0xFFu;
    };
    // An inert stencil (test ALWAYS, every op KEEP or write mask 0, both faces) neither kills nor writes: the
    // draw is a depth-only draw for the overwrite proof (the depth restore pass, Z ALWAYS + stencil ALWAYS/KEEP).
    const auto inert = [&](uint32_t func, uint32_t ops, uint32_t refmask) {
      return func == 7 && (ops == 0 || ((refmask >> 16) & 0xFFu) == 0);
    };
    const bool stencil_inert = stencil && REXCVAR_GET(sa2_native_edram4_stencil_inert) &&
        inert((dc >> 8) & 7, (dc >> 11) & 0x1FF, proof.stencil_front) &&
        (!(dc & 0x80) || inert((dc >> 20) & 7, (dc >> 23) & 0x1FF, proof.stencil_back));
    bool stencil_ok = !stencil || stencil_inert;
    if (stencil && !stencil_inert) {
      stencil_ok = stencil_replaced((dc >> 8) & 7, (dc >> 14) & 7, proof.stencil_front) &&
                   (!(dc & 0x80) || stencil_replaced((dc >> 20) & 7, (dc >> 26) & 7, proof.stencil_back));
    }
    // Alpha test or alpha-to-coverage can kill samples: then neither depth nor stencil is fully written.
    const bool killable = ((proof.color_control >> 3) & 1) || ((proof.color_control >> 4) & 1);
    if (!killable && stencil && stencil_ok && !stencil_inert && (!z_enable || ((dc >> 4) & 7) == 7))
      request.edram_stencil_replace_rect = rect;  // every sample reaches the stencil pass op
    if (!killable && z_enable && z_write && ((dc >> 4) & 7) == 7 && stencil_ok) slots |= 1;
    else if ((dc & 6) == 6) ++overwrite_rejects_[3];
    const uint32_t edram_mode = Register(rex::graphics::XE_GPU_REG_RB_MODECONTROL) & 7;
    const bool alpha_test = (proof.color_control >> 3) & 1;
    if (edram_mode == uint32_t(xenos::EdramMode::kColorDepth) && !alpha_test) {
      const uint32_t mask = Register(rex::graphics::XE_GPU_REG_RB_COLOR_MASK);
      static constexpr uint32_t kBlend[4] = {rex::graphics::XE_GPU_REG_RB_BLENDCONTROL0,
                                             rex::graphics::XE_GPU_REG_RB_BLENDCONTROL1,
                                             rex::graphics::XE_GPU_REG_RB_BLENDCONTROL2,
                                             rex::graphics::XE_GPU_REG_RB_BLENDCONTROL3};
      for (uint32_t i = 0; i < 4; ++i) {
        if (!(request.ps->outputs & (1u << i))) continue;
        if (((mask >> (i * 4)) & 15u) == 15u && (Register(kBlend[i]) & 0x1FFF1FFFu) == 0x00010001u)
          slots |= 2u << i;
      }
    }
    {
      // Diagnostic: a clear whose depth and color 0 share the EDRAM base (D3D Clear of both through the 4x alias).
      const uint32_t di = Register(rex::graphics::XE_GPU_REG_RB_DEPTH_INFO), ci = Register(rex::graphics::XE_GPU_REG_RB_COLOR_INFO);
      if ((di & 0xFFF) == (ci & 0xFFF) && alias_traces_ < 12) {
        ++alias_traces_;
        REXLOG_INFO("[native] aliased depth/color clear proof: slots {:X} mode {} dc {:08X} colorctl {:08X} mask {:08X} "
                    "blend0 {:08X} ps outputs {:X} depthinfo {:08X} colorinfo {:08X} rect {},{}-{},{} killable {}",
                    slots, edram_mode, dc, proof.color_control, Register(rex::graphics::XE_GPU_REG_RB_COLOR_MASK),
                    Register(rex::graphics::XE_GPU_REG_RB_BLENDCONTROL0), request.ps->outputs, di, ci,
                    rect[0], rect[1], rect[2], rect[3], killable);
      }
    }
    if (!slots) { ++overwrite_rejects_[4]; return; }
    ++overwrite_rejects_[5];
    // Area bound for mode 4: rounded outward and padded by a pixel, so it can only over-cover (the
    // overwrite rect above is rounded inward on purpose).
    // Edges exactly on pixel boundaries (k - 0.5 with D3D pixel centers) touch no sample of the neighbouring
    // pixel at 1x or 4x (samples at least 0.25 inside), so the inward rect is also the exact bound.
    const auto in_edge = [](float v) {
      const float b = v + 0.5f;
      return std::abs(b - std::round(b)) < 1.0f / 1024.0f;
    };
    const uint32_t vte_xy = proof.vte_cntl;
    const bool w_one = proof.xy_raster || (vte_xy & 0x100) ||
                       (pos[0][3] == 1.0f && pos[1][3] == 1.0f && pos[2][3] == 1.0f);
    const bool exact = REXCVAR_GET(sa2_native_edram4_exact_edges) && w_one &&
        (!(Register(rex::graphics::XE_GPU_REG_PA_SU_VTX_CNTL) & 1) || REXCVAR_GET(sa2_native_edram4_center_ogl)) &&
        (!(proof.su_sc_mode & (1u << 16)) || !Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_OFFSET)) &&
        in_edge(x0) && in_edge(x1) && in_edge(y0) && in_edge(y1);
    if (exact) ++exact_edges_;
    else if (REXCVAR_GET(sa2_native_edram4_exact_edges) && edges_traces_ < 24) {
      ++edges_traces_;
      REXLOG_INFO("[native] edges not exact: x {}..{} y {}..{} w {} {} {} vte {:08X} vtx_cntl {:08X} su_sc {:08X} "
                  "rect {},{}-{},{} scissor {},{}-{},{}", x0, x1, y0, y1, pos[0][3], pos[1][3], pos[2][3], vte_xy,
                  Register(rex::graphics::XE_GPU_REG_PA_SU_VTX_CNTL), proof.su_sc_mode, rect[0], rect[1], rect[2],
                  rect[3], proof.scissor_x0, proof.scissor_y0, proof.scissor_x1, proof.scissor_y1);
    }
    request.edram_bounds_rect = exact ? rect : std::array<int32_t, 4>{
        std::max<int32_t>(int32_t(std::floor(x0 + 0.5f)) - 1, int32_t(proof.scissor_x0)),
        std::max<int32_t>(int32_t(std::floor(y0 + 0.5f)) - 1, int32_t(proof.scissor_y0)),
        std::min<int32_t>(int32_t(std::ceil(x1 + 0.5f)) + 1, int32_t(proof.scissor_x1)),
        std::min<int32_t>(int32_t(std::ceil(y1 + 0.5f)) + 1, int32_t(proof.scissor_y1))};
    request.edram_overwrite_rect = rect;
    request.edram_overwrite_slots = slots;
    // sa2-nx: constant colour 0 (Direct3D's Clear): the pixel shader passes interpolator 0 through and the three
    // vertices export the same value bit for bit, so every covered pixel gets that colour (to float rounding of the
    // barycentric sum; the colour clear declines values near an 8-bit rounding tie).
    if ((slots & 2) && request.ps->outputs == 1 && proof.ps_color0_is_r0 && proof.interpolator0_valid &&
        std::memcmp(&proof.post_vs_interpolator0[0], &proof.post_vs_interpolator0[1], sizeof(float) * 4) == 0 &&
        std::memcmp(&proof.post_vs_interpolator0[0], &proof.post_vs_interpolator0[2], sizeof(float) * 4) == 0)
      request.edram_overwrite_color0 = proof.post_vs_interpolator0[0];
    // Constant depth: the same post-viewport Z at all three corners, no polygon offset, no PS depth
    // export. Lets mode 4 apply the rectangle as a depth clear in the current owner's view.
    if ((slots & 1) && proof.ps_no_depth_export && !(proof.su_sc_mode & 0x3800u)) {
      const uint32_t vte = proof.vte_cntl;
      const float zs = (vte & 0x10) ? std::bit_cast<float>(Register(rex::graphics::XE_GPU_REG_PA_CL_VPORT_ZSCALE))
                                    : 1.0f;
      const float zo = (vte & 0x20) ? std::bit_cast<float>(Register(rex::graphics::XE_GPU_REG_PA_CL_VPORT_ZOFFSET))
                                    : 0.0f;
      std::optional<float> z;
      bool constant = true;
      for (const auto& v : pos) {
        float zf = v[2];
        if (!(vte & 0x200)) zf = (vte & 0x400) ? zf / v[3] : zf * v[3];
        zf = zf * zs + zo;
        // A hair below 0 (-3.7e-9 in ME's clears) stores the all-zero depth field in both encodings (unorm24
        // clamps, float24 maps z <= 0 to 0).
        if (zf < 0.0f && zf > -1.0e-6f) zf = 0.0f;
        if (!std::isfinite(zf) || zf < 0.0f || zf > 1.0f || (z && *z != zf)) { constant = false; break; }
        z = zf;
      }
      if (constant && z) request.edram_overwrite_depth = z;
    }
    if ((slots & 1) && !request.edram_overwrite_depth && traces_depth_ < 16) {
      ++traces_depth_;
      REXLOG_INFO("[native] constant depth not proven: depthinfo {:08X} ps_no_depth_export {} su_sc {:08X} vte {:08X} "
                  "z {} {} {} w {} {} {} zscale {} zoffset {}", Register(rex::graphics::XE_GPU_REG_RB_DEPTH_INFO),
                  proof.ps_no_depth_export, proof.su_sc_mode, proof.vte_cntl, pos[0][2], pos[1][2], pos[2][2],
                  pos[0][3], pos[1][3], pos[2][3],
                  std::bit_cast<float>(Register(rex::graphics::XE_GPU_REG_PA_CL_VPORT_ZSCALE)),
                  std::bit_cast<float>(Register(rex::graphics::XE_GPU_REG_PA_CL_VPORT_ZOFFSET)));
    }
  }

  std::array<uint64_t, 6> overwrite_rejects_{};  // discard, sample mask, empty rect, depth/stencil, no slot, ok
  void TraceSwap() {
#if !REX_PLATFORM_SWITCH && !REX_PLATFORM_MAC
    const int32_t capture_at = REXCVAR_GET(banjo_renderdoc_capture_s);
    if (renderdoc_capturing_) {
      renderdoc_capturing_ = false;
      REXLOG_INFO("[native] RenderDoc capture of swap {} ended: {}", swaps_, banjo::renderdoc::End() ? "ok" : "FAILED");
    }
    if (capture_at > 0 && !renderdoc_triggered_ && Clock::now() - start_ >= std::chrono::seconds(capture_at)) {
      renderdoc_triggered_ = true;
      const auto folder = rex::filesystem::GetExecutableFolder() / "capturas" / "rdc";
      std::error_code ec;
      std::filesystem::create_directories(folder, ec);
      if (banjo::renderdoc::Start((folder / "banjo").string().c_str())) {
        renderdoc_capturing_ = true;
        REXLOG_INFO("[native] RenderDoc capture started at swap {} ({})", swaps_, folder.string());
      } else {
        REXLOG_WARN("[native] RenderDoc capture requested but RenderDoc is not loaded");
      }
    }
#endif
    const int32_t at = REXCVAR_GET(banjo_trace_frame_s);
    if (trace_phase_ == 0 && at > 0 && Clock::now() - start_ >= std::chrono::seconds(at)) {
      trace_phase_ = 1;
      REXLOG_INFO("[trace] ---- frame after swap {} ----", swaps_);
    } else if (trace_phase_ == 1 && REXCVAR_GET(usf4_trace_until_s) > at &&
               Clock::now() - start_ < std::chrono::seconds(REXCVAR_GET(usf4_trace_until_s))) {
      REXLOG_INFO("[trace] ---- frame after swap {} ({} draws in the previous) ----", swaps_, trace_draws_);
      trace_draws_ = 0;
      trace_quiet_draws_ = 0;
    } else if (trace_phase_ == 1) {
      trace_phase_ = 2;
      REXLOG_INFO("[trace] ---- end at swap {}: {} draws ({} without resolve sampling not listed) ----", swaps_,
                  trace_draws_, trace_quiet_draws_);
    }
  }

  void TraceCopy() {
    namespace g = rex::graphics;
    const uint32_t surface = Register(g::XE_GPU_REG_RB_SURFACE_INFO), color = Register(g::XE_GPU_REG_RB_COLOR_INFO);
    const uint32_t depth = Register(g::XE_GPU_REG_RB_DEPTH_INFO), control = Register(g::XE_GPU_REG_RB_COPY_CONTROL);
    const uint32_t pitch = Register(g::XE_GPU_REG_RB_COPY_DEST_PITCH), info = Register(g::XE_GPU_REG_RB_COPY_DEST_INFO);
    const uint32_t tl = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL), br = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR);
    REXLOG_INFO("[trace] #{} RESOLVE src {} (color base {} fmt {} / depth base {}) pitch {} msaa {} -> {:08X} {}x{} "
                "fmt {} | control {:08X} (clear c{} d{}) clear {:08X} | window {},{}-{},{} offset {:08X}",
                draws_, (control & 7) == 4 ? "depth" : "color", color & 0xFFF, (color >> 16) & 0xF, depth & 0xFFF,
                surface & 0x3FFF, (surface >> 16) & 3, Register(g::XE_GPU_REG_RB_COPY_DEST_BASE), pitch & 0x3FFF,
                (pitch >> 16) & 0x3FFF, (info >> 7) & 0x3F, control, (control >> 8) & 1, (control >> 9) & 1,
                Register(g::XE_GPU_REG_RB_COLOR_CLEAR), tl & 0x3FFF, (tl >> 16) & 0x3FFF, br & 0x3FFF,
                (br >> 16) & 0x3FFF, Register(g::XE_GPU_REG_PA_SC_WINDOW_OFFSET));
  }

  void TraceDraw() {
    namespace g = rex::graphics;
    ++trace_draws_;
    const uint32_t target[4] = {Register(g::XE_GPU_REG_RB_SURFACE_INFO), Register(g::XE_GPU_REG_RB_COLOR_INFO),
                                Register(g::XE_GPU_REG_RB_DEPTH_INFO), Register(kRegRbModeControl)};
    if (std::memcmp(target, trace_target_, sizeof(target))) {
      std::memcpy(trace_target_, target, sizeof(target));
      REXLOG_INFO("[trace] #{} TARGET pitch {} msaa {} color base {} fmt {} depth base {} fmt {} mode {}", draws_,
                  target[0] & 0x3FFF, (target[0] >> 16) & 3, target[1] & 0xFFF, (target[1] >> 16) & 0xF,
                  target[2] & 0xFFF, (target[2] >> 16) & 1, target[3] & 7);
    }
    std::string textures;
    uint32_t used = 0;  // texture slots the pixel shader samples (all slots when it is unknown)
    if (draw_ps_)
      for (const auto& sampler : draw_ps_->samplers) used |= 1u << (sampler.register_value & 31);
    else
      used = ~0u;
    bool any_resolved = false;
    for (uint32_t slot = 0; slot < 32; ++slot) {
      if (!(used >> slot & 1)) continue;
      const uint32_t* f = &registers_[g::XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + slot * 6];
      if ((f[0] & 3) != 2) continue;
      const uint32_t base = f[1] & 0x1FFFF000u;
      any_resolved |= resolve_destinations_.count(base) != 0;
      textures += fmt::format(" t{}={:08X} {}x{} fmt {} pitch {} tiled {}", slot, base, (f[2] & 0x1FFF) + 1,
                              ((f[2] >> 13) & 0x1FFF) + 1, f[1] & 0x3F, ((f[0] >> 22) & 0x1FF) << 5,
                              (f[0] >> 31) & 1);
    }
    // Every draw into the final 1280-wide targets is listed; others only when they sample a resolve destination.
    // sa2-nx: SA2_TRACE_ALL_DRAWS=1 lists every draw of the traced frame (the clears that D3D draws with its
    // own shaders go into other targets and were never listed).
    static const bool trace_all = [] {
      const char* v = std::getenv("SA2_TRACE_ALL_DRAWS");
      return v && *v && *v != '0';
    }();
    if (!trace_all && !any_resolved && (target[0] & 0x3FFF) != 1280) {
      ++trace_quiet_draws_;
      return;
    }
    if ((Register(kRegVgtDrawInitiator) & 63) == 5) {  // usf4-nx: screen-space quads: their first VS constants
      std::string c;
      for (uint32_t i = 0; i < 16; ++i)
        c += fmt::format("{}{:g}", i % 4 ? "," : " | c" + std::to_string(i / 4) + " ",
                         std::bit_cast<float>(Register(g::XE_GPU_REG_SHADER_CONSTANT_000_X + i)));
      REXLOG_INFO("[trace] #{} QUAD VS constants{}", draws_, c);
    }
    const uint32_t tl = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL), br = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR);
    REXLOG_INFO("[trace] #{} DRAW {} VS n{} PS n{} prim {} count {} scissor {},{}-{},{} blend0 {:08X} mask {:08X} "
                "depth {:08X} rt1 {:08X} blend1 {:08X} c0 {:08X} {:08X} {:08X} {:08X} vp x {:.1f}+{:.1f} y {:.1f}+{:.1f} "
                "woff {:08X}:{}", draws_,
                trace_result_ == 1 ? "ok" : trace_result_ == 0 ? fmt::format("REJ{}", g_banjo_last_reject_cause)
                                                                : std::string("SKIP"),
                draw_vs_ ? int64_t(draw_vs_->number) : -1, draw_ps_ ? int64_t(draw_ps_->number) : -1,
                Register(kRegVgtDrawInitiator) & 63, Register(kRegVgtDrawInitiator) >> 16, tl & 0x3FFF,
                (tl >> 16) & 0x3FFF, br & 0x3FFF, (br >> 16) & 0x3FFF, Register(g::XE_GPU_REG_RB_BLENDCONTROL0),
                Register(g::XE_GPU_REG_RB_COLOR_MASK), Register(g::XE_GPU_REG_RB_DEPTHCONTROL),
                Register(g::XE_GPU_REG_RB_COLOR1_INFO), Register(g::XE_GPU_REG_RB_BLENDCONTROL1),
                Register(g::XE_GPU_REG_SHADER_CONSTANT_256_X), Register(g::XE_GPU_REG_SHADER_CONSTANT_256_Y),
                Register(g::XE_GPU_REG_SHADER_CONSTANT_256_Z), Register(g::XE_GPU_REG_SHADER_CONSTANT_256_W),
                std::bit_cast<float>(Register(g::XE_GPU_REG_PA_CL_VPORT_XOFFSET)),
                std::bit_cast<float>(Register(g::XE_GPU_REG_PA_CL_VPORT_XSCALE)),
                std::bit_cast<float>(Register(g::XE_GPU_REG_PA_CL_VPORT_YOFFSET)),
                std::bit_cast<float>(Register(g::XE_GPU_REG_PA_CL_VPORT_YSCALE)), Register(g::XE_GPU_REG_PA_SC_WINDOW_OFFSET),
                textures);
  }

  void Draw() {
    const sa2::native::ShaderEntry* const pair_vs = draw_vs_;
    const sa2::native::ShaderEntry* const pair_ps = draw_ps_;
    bool final_proven = false;
    if (identity_memo_hit_) {
      // sa2_cpu_ring_identity_memo: the same inputs as the draw that stored the memo, so the same final choice.
      identity_memo_hit_ = false;
      draw_vs_ = identity_memo_final_vs_;
      if (!EnsureTargets()) return;
      ++(identity_memo_final_proven_ ? vs_final_proven_ : vs_final_unproven_);
      final_proven = identity_memo_final_proven_;
    } else {
    if (!MatchesVertexShaderIdentity(draw_vs_) && MatchesVertexShaderIdentity(vs_))
      draw_vs_ = PreferVertexShaderCandidate(draw_vs_, vs_, "draw-final-ring-preference");
    if (draw_vs_ && !AcceptVertexShaderIdentity(draw_vs_, "draw-final")) {
      // No unchecked fallback: the last loaded variant is the only authority.
      draw_vs_ = vs_ && AcceptVertexShaderIdentity(vs_, "draw-final-ring") ? vs_ : nullptr;
      if (!draw_vs_) { ++draws_unidentified_; identity_memo_pending_ = false; return; }
    }
    if (!EnsureTargets()) { identity_memo_pending_ = false; return; }
    // Candidate warnings include discarded object/queue alternatives. Count
    // the actual final selection separately, without changing rollout policy.
    if (!draw_vs_) {
      ++vs_final_missing_;
    } else if (MatchesVertexShaderIdentity(draw_vs_)) {
      ++vs_final_proven_;
      final_proven = true;
    } else {
      const uint64_t unproven = ++vs_final_unproven_;
      if (unproven <= 8 || (unproven & 255) == 0) {
        REXLOG_WARN("[native] FINAL VS unproven {} draw={} loaded_host_hash={:016X} "
                    "loaded_words={} selected_n={} selected_container={:016X}",
                    unproven, draws_, current_vs_hash_, vs_microcode_.size(),
                    draw_vs_->number, draw_vs_->shader ? draw_vs_->shader->fingerprint : 0);
      }
    }
    // sa2_cpu_ring_identity_memo: this draw's inputs and outcome, for the next draw. In verify mode the memo is never
    // used, and a draw with the stored key must have reached the stored outcome through the full checks.
    if (identity_memo_pending_ && draw_vs_) {
      if (sa2::enh::VerifyMode() && identity_memo_valid_ && identity_memo_key_ == identity_memo_last_key_) {
        sa2::enh::VerifyCheck();
        ++identity_memo_checked_;
        if (identity_memo_pair_vs_ != pair_vs || identity_memo_pair_ps_ != pair_ps ||
            identity_memo_final_vs_ != draw_vs_ || identity_memo_final_proven_ != final_proven) {
          sa2::enh::VerifyDifference();
          REXLOG_ERROR("[native] DIFFERENCE: identity memo at draw {}: memo VS n{} PS n{} final n{}, checks VS n{} PS "
                       "n{} final n{}", draws_, identity_memo_pair_vs_ ? int(identity_memo_pair_vs_->number) : -1,
                       identity_memo_pair_ps_ ? int(identity_memo_pair_ps_->number) : -1,
                       identity_memo_final_vs_ ? int(identity_memo_final_vs_->number) : -1,
                       pair_vs ? int(pair_vs->number) : -1, pair_ps ? int(pair_ps->number) : -1,
                       int(draw_vs_->number));
        }
      }
      identity_memo_pair_vs_ = pair_vs;
      identity_memo_pair_ps_ = pair_ps;
      identity_memo_final_vs_ = draw_vs_;
      identity_memo_final_proven_ = final_proven;
      identity_memo_last_key_ = identity_memo_key_;
      identity_memo_valid_ = true;
    }
    identity_memo_pending_ = false;
    }  // the identity checks (not a memo hit)
    sa2::native::SubmissionDraw request;
    request.registers = registers_.data();
    request.vs = draw_vs_;
    request.ps = draw_ps_;
    request.vs_microcode = vs_microcode_;
    request.ps_microcode = ps_microcode_;
    request.generation_vs = gen_vs_;
    request.generation_constants_vs = gen_constants_vs_;
    request.generation_constants_ps = gen_constants_ps_;
    request.generation_fetch = gen_fetch_;
    request.generation_framing = gen_viewport_;
    // The ring hashed this very program when it identified the load (IdentifyShader sets both together).
    request.vs_program_hash = current_vs_hash_;
    request.vs_program_hash_set = !vs_microcode_.empty();
    if (memory_ &&
        ((Register(rex::graphics::XE_GPU_REG_PA_CL_CLIP_CNTL) & 0x10000u) ||
         REXCVAR_GET(sa2_native_edram4_clip_inside)) &&
        MatchesVertexShaderIdentity(draw_vs_)) {
      const uint32_t initiator = Register(kRegVgtDrawInitiator);
      // The adapter performs the full admission proof. This outer filter avoids
      // copying/analyzing unrelated normal draws when the opt-in is enabled.
      const uint32_t rect_vertices = initiator >> 16;
      if ((initiator & 63u) == uint32_t(xenos::PrimitiveType::kRectangleList) &&
          ((initiator >> 6) & 3u) == uint32_t(xenos::SourceSelect::kAutoIndex) &&
          rect_vertices > 3u && rect_vertices % 3u == 0 && rect_vertices <= 24u &&
          REXCVAR_GET(sa2_native_edram4_several_rect)) {
        // Rect lists of several rectangles (D3D Clear of two regions): prove each one with the same
        // single-rectangle estimator on a register copy whose index offset selects that rectangle.
        if (!draw_extent_estimator_) { draw_extent_estimator_ = std::make_unique<NativeDrawExtentEstimator>(*memory_); draw_extent_estimator_->AllowClipInside(REXCVAR_GET(sa2_native_edram4_clip_inside)); draw_extent_estimator_->AllowQuadTriangles(REXCVAR_GET(sa2_native_edram4_quad_triangles)); draw_extent_estimator_->AllowAlu(REXCVAR_GET(sa2_native_edram4_vs_alu)); draw_extent_estimator_->ReasonAsPointer(g_estimator_reason.active); }
        ProveRectangleList(request, rect_vertices / 3u);
      } else if ((((initiator >> 6) & 3u) == uint32_t(xenos::SourceSelect::kAutoIndex) ||
                  (((initiator >> 6) & 3u) == uint32_t(xenos::SourceSelect::kDMA) &&
                   (initiator & 63u) != uint32_t(xenos::PrimitiveType::kRectangleList))) &&
          (((initiator & 63u) == uint32_t(xenos::PrimitiveType::kRectangleList) && (initiator >> 16) == 3u) ||
           (REXCVAR_GET(sa2_native_edram4_quad_triangles) &&
            (((initiator & 63u) == uint32_t(xenos::PrimitiveType::kTriangleList) && (initiator >> 16) == 6u) ||
             ((initiator & 63u) == uint32_t(xenos::PrimitiveType::kTriangleStrip) && (initiator >> 16) == 4u))))) {
        if (!draw_extent_estimator_) { draw_extent_estimator_ = std::make_unique<NativeDrawExtentEstimator>(*memory_); draw_extent_estimator_->AllowClipInside(REXCVAR_GET(sa2_native_edram4_clip_inside)); draw_extent_estimator_->AllowQuadTriangles(REXCVAR_GET(sa2_native_edram4_quad_triangles)); draw_extent_estimator_->AllowAlu(REXCVAR_GET(sa2_native_edram4_vs_alu)); draw_extent_estimator_->ReasonAsPointer(g_estimator_reason.active); }
        NativeDrawExtentEstimator::Diagnostics diagnostic;
        request.edram_used_height_estimate = draw_extent_estimator_->Estimate(registers_, vs_microcode_, &diagnostic);
        if (g_estimator_reason.active) g_estimator_reason.UseSampled();
        if (request.edram_used_height_estimate && request.ps) {
          StencilClearOwnershipInput proof;
          proof.depth_control = Register(rex::graphics::XE_GPU_REG_RB_DEPTHCONTROL);
          proof.stencil_front = Register(rex::graphics::XE_GPU_REG_RB_STENCILREFMASK);
          proof.stencil_back = Register(rex::graphics::XE_GPU_REG_RB_STENCILREFMASK_BF);
          proof.color_control = Register(rex::graphics::XE_GPU_REG_RB_COLORCONTROL);
          proof.su_sc_mode = Register(rex::graphics::XE_GPU_REG_PA_SU_SC_MODE_CNTL);
          proof.vtx_cntl = Register(rex::graphics::XE_GPU_REG_PA_SU_VTX_CNTL);
          proof.vte_cntl = Register(rex::graphics::XE_GPU_REG_PA_CL_VTE_CNTL);
          proof.window_offset = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_OFFSET);
          proof.clip_cntl = Register(rex::graphics::XE_GPU_REG_PA_CL_CLIP_CNTL);
          const uint32_t surface = Register(rex::graphics::XE_GPU_REG_RB_SURFACE_INFO);
          proof.msaa_enum = (surface >> 16) & 3;
          proof.physical_pitch_tiles = (((surface & 0x3FFF) << uint32_t(proof.msaa_enum == 2)) + 79) / 80;
          const uint32_t tl = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL);
          const uint32_t br = Register(rex::graphics::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR);
          const auto sign15 = [](uint32_t v) { return int32_t(v & 0x7FFF) - int32_t((v & 0x4000) << 1); };
          const int32_t wx = (tl >> 31) ? 0 : sign15(proof.window_offset);
          const int32_t wy = (tl >> 31) ? 0 : sign15(proof.window_offset >> 16);
          const uint32_t screen_tl = Register(rex::graphics::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_TL);
          const uint32_t screen_br = Register(rex::graphics::XE_GPU_REG_PA_SC_SCREEN_SCISSOR_BR);
          proof.scissor_x0 = uint32_t(std::max({0, int32_t(tl & 0x3FFF) + wx, sign15(screen_tl)}));
          proof.scissor_y0 = uint32_t(std::max({0, int32_t((tl >> 16) & 0x3FFF) + wy, sign15(screen_tl >> 16)}));
          proof.scissor_x1 = uint32_t(std::max(0, std::min(int32_t(br & 0x3FFF) + wx, sign15(screen_br))));
          proof.scissor_y1 = uint32_t(std::max(0, std::min(int32_t((br >> 16) & 0x3FFF) + wy, sign15(screen_br >> 16))));
          proof.post_vs_position = diagnostic.position;
      proof.xy_raster = diagnostic.xy_raster;
          proof.post_vs_interpolator0 = diagnostic.interpolator0;
          proof.interpolator0_valid = diagnostic.interpolator0_valid;
          proof.ps_color0_is_r0 = ProvePixelColor0IsR0(ps_microcode_);
          proof.rectangle_proven = true;  // Successful bounded adapter proof, not bbox-only.
          proof.ps_identity_proven = MatchesPixelShaderIdentity(request.ps);
          // Counting SPIR-V OpKill is not proof: an optimizer may merge guest
          // and alpha-test kills. Inspect every admitted loaded ALU instead.
          proof.ps_no_discard = ProveStraightLinePixelNoKill(ps_microcode_) ||
          (REXCVAR_GET(sa2_native_edram4_ps_no_kill) && ProvePixelNoKillAnywhere(ps_microcode_));
          proof.ps_no_depth_export = !(request.ps->outputs & 0x10);
          // Admit only the all-enabled 16-bit AA mask, never an unknown mask.
          // The native pipeline has no additional programmable sample mask.
          proof.sample_coverage_proven_full = (Register(rex::graphics::XE_GPU_REG_PA_SC_AA_MASK) & 0xFFFFu) == 0xFFFFu;
          request.edram_stencil_clear = ProveStencilClearOwnership(proof);
          ProveFullOverwrite(request, proof);
          if ((proof.depth_control & 7) == 1 && stencil_clear_diagnostics_++ < 16)
            REXLOG_INFO("[native] canonical stencil clear candidate: VS={} PS={} eligible={} reason={} "
                        "aaMask={:08X} physical={}x{} tiles={} ref={}",
                        request.vs->number, request.ps->number, request.edram_stencil_clear.eligible,
                        request.edram_stencil_clear.reason, Register(rex::graphics::XE_GPU_REG_PA_SC_AA_MASK),
                        request.edram_stencil_clear.physical_width, request.edram_stencil_clear.physical_height,
                        request.edram_stencil_clear.length_tiles, request.edram_stencil_clear.stencil_reference);
        }
        // sa2_cpu_ring_extent_key: the same five values deduped as numbers (and the reason's bytes) instead of a
        // formatted string; SA2 never fills the 32 slots, so the old key was formatted on every bounded draw.
        bool extent_new = false;
        if (g_extent_key_int.active) {
          if (extent_keys_.size() < 32) {
            g_extent_key_int.UseSampled();
            const std::string_view why = diagnostic.Reason();
            const uint64_t numbers[3] = {current_vs_hash_,
                uint64_t(Register(rex::graphics::XE_GPU_REG_RB_DEPTH_INFO)) << 32 |
                    Register(rex::graphics::XE_GPU_REG_RB_SURFACE_INFO),
                request.edram_used_height_estimate.value_or(UINT32_MAX)};
            extent_new = extent_keys_.insert(XXH3_64bits_withSeed(why.data(), why.size(),
                                                                  XXH3_64bits(numbers, sizeof(numbers)))).second;
          }
        } else if (extent_diagnostics_.size() < 32) {
          const std::string key = fmt::format("{:X}/{:X}/{}/{}/{}", current_vs_hash_,
              Register(rex::graphics::XE_GPU_REG_RB_DEPTH_INFO),
              Register(rex::graphics::XE_GPU_REG_RB_SURFACE_INFO),
              request.edram_used_height_estimate.value_or(UINT32_MAX), diagnostic.Reason());
          extent_new = extent_diagnostics_.insert(key).second;
        }
        {
          if (extent_new)
            REXLOG_INFO("[native] bounded draw extent: code={:016X} VS={} depth={:08X} surface={:08X} "
                        "accepted={} maxY={} reason={} postVS=[{},{},{},{}; {},{},{},{}; {},{},{},{}]",
                        current_vs_hash_, request.vs->number,
                        Register(rex::graphics::XE_GPU_REG_RB_DEPTH_INFO),
                        Register(rex::graphics::XE_GPU_REG_RB_SURFACE_INFO),
                        bool(request.edram_used_height_estimate),
                        request.edram_used_height_estimate.value_or(UINT32_MAX), diagnostic.Reason(),
                        diagnostic.position[0][0], diagnostic.position[0][1], diagnostic.position[0][2], diagnostic.position[0][3],
                        diagnostic.position[1][0], diagnostic.position[1][1], diagnostic.position[1][2], diagnostic.position[1][3],
                        diagnostic.position[2][0], diagnostic.position[2][1], diagnostic.position[2][2], diagnostic.position[2][3]);
        }
      }
    }
    ++native_draw_attempts_;
    g_banjo_last_reject_cause = 0;
    const bool drawn = targets_->Draw(request);
    trace_result_ = drawn ? 1 : 0;
    if (!drawn) {
      if (++native_draw_failures_ <= 8)
        REXLOG_WARN("[native] draw rejected before completion; visual conformance is not established");
    }
  }

  void Copy() {
    if (!EnsureTargets()) return;
    namespace g = rex::graphics;
    sa2::native::RegistersCopy r;
    r.rb_surface_info = Register(g::XE_GPU_REG_RB_SURFACE_INFO);
    r.rb_color_info[0] = Register(g::XE_GPU_REG_RB_COLOR_INFO);
    r.rb_color_info[1] = Register(g::XE_GPU_REG_RB_COLOR1_INFO);
    r.rb_color_info[2] = Register(g::XE_GPU_REG_RB_COLOR2_INFO);
    r.rb_color_info[3] = Register(g::XE_GPU_REG_RB_COLOR3_INFO);
    r.rb_depth_info = Register(g::XE_GPU_REG_RB_DEPTH_INFO);
    r.rb_copy_control = Register(g::XE_GPU_REG_RB_COPY_CONTROL);
    r.rb_copy_dest_base = Register(g::XE_GPU_REG_RB_COPY_DEST_BASE);
    r.rb_copy_dest_pitch = Register(g::XE_GPU_REG_RB_COPY_DEST_PITCH);
    r.rb_copy_dest_info = Register(g::XE_GPU_REG_RB_COPY_DEST_INFO);
    r.rb_color_clear = Register(g::XE_GPU_REG_RB_COLOR_CLEAR);
    r.rb_color_clear_lo = Register(g::XE_GPU_REG_RB_COLOR_CLEAR_LO);
    r.rb_depth_clear = Register(g::XE_GPU_REG_RB_DEPTH_CLEAR);
    r.pa_sc_window_offset = Register(g::XE_GPU_REG_PA_SC_WINDOW_OFFSET);
    r.pa_sc_window_scissor_tl = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_TL);
    r.pa_sc_window_scissor_br = Register(g::XE_GPU_REG_PA_SC_WINDOW_SCISSOR_BR);
    r.pa_su_sc_mode_cntl = Register(g::XE_GPU_REG_PA_SU_SC_MODE_CNTL);
    r.pa_su_vtx_cntl = Register(g::XE_GPU_REG_PA_SU_VTX_CNTL);
    r.fetch_vertices[0] = Register(g::XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0);
    r.fetch_vertices[1] = Register(g::XE_GPU_REG_SHADER_CONSTANT_FETCH_00_1);
    ++native_copy_attempts_;
    if (!targets_->Copy(r)) {
      if (++native_copy_failures_ <= 8)
        REXLOG_WARN("[native] copy/clear rejected before completion; visual conformance is not established");
    }
  }

  void Present() {
    if (!presenter_ || !EnsureTargets()) return;
    if (gamma_sent_version_ != gamma_version_) {
      gamma_sent_version_ = gamma_version_;
      targets_->RampGamma(gamma_ramp_);
      if (!gamma_logged_) {
        gamma_logged_ = true;
        REXLOG_INFO("[native] game gamma ramp: {} writes, [0]={}/{}/{}, [64]={}/{}/{}, "
                    "[128]={}/{}/{}, [255]={}/{}/{}",
                    gamma_writes_, gamma_ramp_[0][0], gamma_ramp_[0][1], gamma_ramp_[0][2],
                    gamma_ramp_[64][0], gamma_ramp_[64][1], gamma_ramp_[64][2],
                    gamma_ramp_[128][0], gamma_ramp_[128][1], gamma_ramp_[128][2],
                    gamma_ramp_[255][0], gamma_ramp_[255][1], gamma_ramp_[255][2]);
      }
    }
    sa2::native::TextureSwap texture;
    for (uint32_t i = 0; i < 6; ++i) {
      texture.dword[i] = Register(rex::graphics::XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 + i);
    }
    if (targets_->Present(presenter_.get(), texture, swap_width_, swap_height_)) {
      ++presented_;
    }
  }

  void Report(bool force) {
    const auto now = Clock::now();
    if (!force && now - last_report_ < std::chrono::seconds(10)) return;
    const double secs = std::chrono::duration<double>(now - last_report_).count();
    last_report_ = now;
    // GPU time per category, real milliseconds per Swap of this interval: raw NVK timestamps x 1.627 on the Switch.
    if (targets_) {
      std::array<uint64_t, sa2::native::kGpuCategories> gpu{}, frag{}, vert{}, prim{};
      targets_->TimeGpuPerCategory(gpu);
      targets_->StatsPipeline(frag, vert, prim);
      constexpr double kScale = 1.627;
      const uint64_t swaps = swaps_ - report_gpu_swaps_;
      if (swaps) {
        static constexpr const char* kNames[sa2::native::kGpuCategories] = {
            "other",  "shadows",    "scene",        "640",         "320",         "smaller",     "copies",
            "clears", "scene_no_z", "gap",         "edram_import", "edram_export", "edram_alias", "edram_import9"};
        std::string t;
        double total = 0;
        for (uint32_t i = 0; i < sa2::native::kGpuCategories; ++i) {
          const double ms = double(gpu[i] - report_gpu_ns_[i]) * kScale / 1e6 / double(swaps);
          total += ms;
          t += fmt::format(" {} {:.1f} ms ({:.2f} Mfrag)", kNames[i], ms,
                           double(frag[i] - report_gpu_frag_[i]) / 1e6 / double(swaps));
        }
        REXLOG_INFO("[native] GPU per Swap ({} Swaps): total {:.1f} ms;{}", swaps, total, t);
        REXLOG_INFO("[native] overwrite proof: discard {} samplemask {} emptyrect {} depthstencil {} noslot {} ok {}",
                    overwrite_rejects_[0], overwrite_rejects_[1], overwrite_rejects_[2], overwrite_rejects_[3],
                    overwrite_rejects_[4], overwrite_rejects_[5]);
        overwrite_rejects_ = {};
      }
      report_gpu_ns_ = gpu;
      report_gpu_frag_ = frag;
      report_gpu_swaps_ = swaps_;
    }
    {  // PM4 mix of this interval (what the ring parses per frame): type 0 and the busiest type 3 opcodes
      std::array<std::pair<uint64_t, uint32_t>, 128> order;
      for (uint32_t i = 0; i < 128; ++i) order[i] = {opcodes_[i], i};
      std::partial_sort(order.begin(), order.begin() + 8, order.end(), std::greater<>());
      std::string mix;
      for (uint32_t i = 0; i < 8 && order[i].first; ++i)
        mix += fmt::format(" {:02X}:{}pk/{}w", order[i].second, order[i].first, opcode_words_[order[i].second]);
      REXLOG_INFO("[native] PM4 mix (cumulative): {} packets, type 0: {} ({} words, {} single-register); type 3 opcodes "
                  "(packets/words):{}", packets_, type0_packets_, type0_words_, type0_single_, mix);
    }
    {  // The ring thread's own time of this interval: its parse loop, less the WAIT_REG_MEM waits for the game.
      const uint64_t swaps = swaps_ - report_ring_swaps_, draws = draws_ - report_ring_draws_;
      const double busy = double(ring_busy_ns_ - report_ring_busy_ns_) / 1e6;
      const double wait = double(ring_wait_ns_ - report_ring_wait_ns_) / 1e6;
      if (swaps)
        REXLOG_INFO("[native] ring thread: {:.2f} ms of work per Swap ({:.2f} us per draw, {} draws per Swap), "
                    "{:.2f} ms per Swap waiting for the game (WAIT_REG_MEM)",
                    (busy - wait) / double(swaps), draws ? (busy - wait) * 1000.0 / double(draws) : 0.0,
                    draws / swaps, wait / double(swaps));
      report_ring_swaps_ = swaps_;
      report_ring_draws_ = draws_;
      report_ring_busy_ns_ = ring_busy_ns_;
      report_ring_wait_ns_ = ring_wait_ns_;
    }
    sa2::api::Report();  // the native Direct3D layer (sa2_api_*)
    if (g_direct_state.active) {
      using namespace direct;
      REXLOG_INFO("[native] direct state (cumulative): {} draws converted ({} words of PM4 not parsed, {:.1f} record words "
                  "per draw), {} records "
                  "applied ({} written back), {} released unreached, {} lost{}; left as PM4: submitted {}, segment {}, "
                  "unknown packet {} "
                  "(last opcode {:02X}), NOP head {}, small {}, no draw {}, ring full {}; {} records checked; "
                  "submissions that waited for a marker {}",
                  g_converted.load(), g_words_covered.load(),
                  g_converted.load() ? double(g_record_words.load()) / double(g_converted.load()) : 0.0,
                  direct_applied_, direct_restored_, direct_skipped_, direct_lost_,
                  g_failed.load() ? " (STOPPED)" : "", g_skip_submitted.load(), g_skip_segment.load(),
                  g_skip_unknown.load(), g_unknown_opcode.load(), g_skip_nop_head.load(), g_skip_small.load(),
                  g_skip_no_draw.load(), g_skip_full.load(), g_checked.load(), g_kick_waits.load());
      if (g_direct_flush.active)
        REXLOG_INFO("[native] direct flush (sa2_cpu_direct_flush, cumulative): {} constant, {} fetch, {} register group "
                    "flushes written into records ({} words, {} placeholders taken into their draw's record, {} "
                    "published alone, {} own submissions in a call); by the original: {} slow path, {} no room, {} "
                    "submitted first; {} checked against the original{}{}",
                    g_flush_calls[0].load(), g_flush_calls[1].load(), g_flush_calls[2].load(), g_flush_words.load(),
                    g_flush_taken.load(), g_flush_alone.load(), g_flush_own_kicks.load(), g_flush_slow.load(),
                    g_flush_room.load(), g_flush_original.load(), g_flush_checked.load(),
                    g_flush_failed.load() ? " (STOPPED: a difference)" : "",
                    g_flush_multi.load() ? " (off: a second drawing thread)" : "");
      if (census_ && !census_masks_.empty()) {
        std::vector<std::pair<uint64_t, uint32_t>> v;
        uint64_t all = 0;
        for (const auto& [m, n] : census_masks_) { v.push_back({n, m}); all += n; }
        std::sort(v.begin(), v.end(), std::greater<>());
        std::string t;
        for (size_t i = 0; i < v.size() && i < 12; ++i)
          t += fmt::format(" {:06X}:{:.1f}%", v[i].second, 100.0 * double(v[i].first) / double(all));
        REXLOG_INFO("[native] direct-state census (cumulative, {} records; group bits: 0 control, 1 0x2000, 2 viewport "
                    "0x2080, 3 0x2100, 4 0x2180, 5 0x2200, 6 0x2280, 7 0x2300, 8 0x2380, 9 other, 10 VS const, 11 PS "
                    "const, 12 tex fetch, 13 vtx fetch, 14 bool, 15 loop, 16 0x5000+, 20 shader load, 21 other "
                    "type-3):{}", all, t);
        std::vector<std::pair<uint64_t, uint32_t>> c;
        for (const auto& [m, n] : census_changed_) c.push_back({n, m});
        std::sort(c.begin(), c.end(), std::greater<>());
        std::string u;
        for (size_t i = 0; i < c.size() && i < 12; ++i)
          u += fmt::format(" {:06X}:{:.1f}%", c[i].second, 100.0 * double(c[i].first) / double(all));
        std::vector<std::pair<uint64_t, uint32_t>> g;
        for (const auto& [r, n] : census_register_changes_) g.push_back({n, r});
        std::sort(g.begin(), g.end(), std::greater<>());
        std::string w;
        for (size_t i = 0; i < g.size() && i < 16; ++i)
          w += fmt::format(" {:04X}:{:.1f}%", g[i].second, 100.0 * double(g[i].first) / double(all));
        REXLOG_INFO("[native] direct-state census, groups really CHANGED:{} | render-state registers changed (per "
                    "record):{}", u, w);
      }
      if (g_direct_pair.active)
        REXLOG_INFO("[native] direct pair (cumulative): {} Draw* records carried in direct records, {} used by the ring "
                    "({} checked against the table in verify mode, {} not in the table); pushed to the queue: {} by "
                    "draws that kept their packets, {} by submissions during the call{}",
                    g_pairs_carried.load(), direct_pairs_used_, direct_pairs_checked_, direct_pairs_table_missed_,
                    g_pairs_pushed_late.load(), g_pairs_pushed_at_submission.load(),
                    g_pair_off.load() ? " (OFF: a submission from a thread that does not draw)" : "");
    }
    REXLOG_INFO("[native] shader loads reused (identical reload): {}; multi-rect draws proven: {} (rejected: "
                "estimator {}, overwrite proof {}, slot mismatch {}); pixel-exact bounds: {}; identity memo hits {}",
                shader_loads_reused_,
                multi_rect_proven_, multi_rect_rejects_[0], multi_rect_rejects_[1], multi_rect_rejects_[2],
                exact_edges_, identity_memo_hits_);
    identity_memo_hits_ = 0;
    multi_rect_proven_ = 0;
    exact_edges_ = 0;
    multi_rect_rejects_ = {};
    shader_loads_reused_ = 0;
    REXLOG_INFO("[native] PS identity audit: {} mismatching candidate checks "
                "(all selection paths, cumulative; not distinct draws)", ps_identity_mismatches_);
    REXLOG_INFO("[native] VS identity audit: {} mismatching candidate checks "
                "(declared fetch patches only; all ALU/CF words compared; cumulative), {} known mismatches "
                "skipped ({} pairs memoized)", vs_identity_mismatches_, vs_mismatch_memo_hits_,
                vs_mismatch_memo_.size());
    REXLOG_INFO("[native] FINAL VS selection audit: {} proven, {} unproven, {} missing "
                "(cumulative attempted draws; not GPU/DEF/ABI equivalence proof)",
                vs_final_proven_, vs_final_unproven_, vs_final_missing_);
    if (shaders_reordered_)
      REXLOG_INFO("[native] VS loads identified with their fetches reordered by Direct3D (sa2_gpu_vs_fetch_reorder): "
                  "{} (cumulative)", shaders_reordered_);
    uint64_t copies = 0, clears = 0, presented = 0, rejected = 0;
    if (targets_) targets_->Stats(copies, clears, presented, rejected);
    const auto st = targets_ ? targets_->StatsOfDraws() : sa2::native::StatsDraws{};
    REXLOG_INFO("[native] completion coverage: draw failures {}/{}; copy/clear failures {}/{} "
                "(includes target preparation; successful returns alone do not certify rendering)",
                native_draw_failures_, native_draw_attempts_, native_copy_failures_, native_copy_attempts_);
    REXLOG_INFO("[native] {:.1f} s: {} Swaps, {} draws ({} without shaders), {} copies, {} packets, "
                "{} interrupts, {} WAIT_REG_MEM timeouts; shaders {} identified ({} by address) / {} not; native: "
                "{} drawn, {} rejected, {} pipelines, {} textures, {} presented",
                secs, swaps_ - reported_swaps_, draws_ - reported_draws_, draws_unidentified_,
                copies_ - reported_copies_, packets_ - reported_packets_,
                interrupts_.load() - reported_interrupts_, waits_timed_out_, shaders_identified_,
                shaders_by_address_, shaders_unidentified_, st.drawn, st.rejected, st.pipelines, st.textures,
                presented);
    std::string causes;
    for (size_t i = 0; i < st.causes.size() && i < 6; ++i) {
      causes += fmt::format(" {}x{}", st.causes[i].first, st.causes[i].second);
    }
    if (!causes.empty()) REXLOG_INFO("[native] rejection causes (code x count):{}", causes);
    if (!missing_draw_pairs_.empty()) {
      std::vector<std::pair<std::pair<uint64_t, uint64_t>, uint64_t>> ranked(
          missing_draw_pairs_.begin(), missing_draw_pairs_.end());
      std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.second > b.second;
      });
      std::string missing;
      for (size_t i = 0; i < ranked.size() && i < 12; ++i) {
        missing += fmt::format(" {}x(VS={:016X},PS={:016X})", ranked[i].second,
                               ranked[i].first.first, ranked[i].first.second);
      }
      REXLOG_INFO("[native] unresolved draw shader pairs (cumulative, 0 means identified):{}", missing);
    }
    {
      std::lock_guard<std::mutex> lock(g_by_address_mutex);
      REXLOG_INFO("[native] tile replays reusing a pairing {}, draws dropped by predicate {}",
                  draws_replayed_, draws_predicated_);
      std::string miss;
      for (auto& [o, n] : missing_objects_) miss += fmt::format(" {:08X}x{}", o, n);
      REXLOG_INFO("[native] draws paired by command-buffer address {} (no VS entry {}, no PS entry {});"
                  " objects without an entry:{}", draws_paired_by_address_, paired_no_vs_, paired_no_ps_, miss);
      REXLOG_INFO("[native] draws paired with a Draw* record {}, unpaired {}; "
                  "draws inside indirect buffers {} of {}; Draw* records {}",
                  draws_paired_, draws_unpaired_, draws_indirect_, draws_,
                  g_draw_records.load());
      REXLOG_INFO("[native] identified by code {}; unidentified loads: VS by address {}, PS by address "
                  "{}, immediate {}; {} objects seen, {} addresses and {} codes mapped, {} code conflicts",
                  shaders_by_code_, miss_vs_addr_, miss_ps_addr_, miss_immediate_, g_objects_seen.size(),
                  g_by_address.size(), g_by_code.size(), g_code_conflicts);
    }
    reported_swaps_ = swaps_;
    reported_draws_ = draws_;
    reported_copies_ = copies_;
    reported_packets_ = packets_;
    reported_interrupts_ = interrupts_.load();
  }

  sa2::native::ShadersNative shaders_;
  std::unique_ptr<sa2::native::TargetsNative> targets_;
  bool targets_failed_ = false;
  const sa2::native::ShaderEntry* vs_ = nullptr;
  const sa2::native::ShaderEntry* ps_ = nullptr;
  std::vector<uint32_t> microcode_, vs_microcode_, ps_microcode_;
  uint64_t gen_vs_ = 0, gen_microcode_ = 0, gen_constants_vs_ = 0, gen_constants_ps_ = 0,
           gen_fetch_ = 0, gen_viewport_ = 0;
  uint32_t swap_width_ = 1280, swap_height_ = 720;
  uint32_t logged_misses_ = 0;
  uint32_t im_load_address_ = 0;
  uint64_t shaders_by_code_ = 0, draws_paired_ = 0, draws_unpaired_ = 0;
  uint64_t native_draw_attempts_ = 0, native_draw_failures_ = 0;
  std::unique_ptr<NativeDrawExtentEstimator> draw_extent_estimator_;
  std::unordered_set<std::string> extent_diagnostics_;
  std::unordered_set<uint64_t> extent_keys_;  // sa2_cpu_ring_extent_key
  uint32_t stencil_clear_diagnostics_ = 0;
  uint64_t native_copy_attempts_ = 0, native_copy_failures_ = 0;
  uint32_t logged_unpaired_ = 0;
  uint64_t draws_paired_by_address_ = 0, paired_no_vs_ = 0, paired_no_ps_ = 0;
  std::unordered_map<uint32_t, uint64_t> missing_objects_;
  std::unordered_set<uint64_t> dumped_rewritten_;
  std::array<uint64_t, sa2::native::kGpuCategories> report_gpu_ns_{}, report_gpu_frag_{};
  uint64_t report_gpu_swaps_ = 0;
  std::unordered_set<uint64_t> dumped_vertex_variants_;
  uint64_t draws_indirect_ = 0, draws_replayed_ = 0, draws_predicated_ = 0;
  uint32_t packet_address_ = 0;
  struct PairedPacket {
    uint64_t frame;
    const sa2::native::ShaderEntry* vs;
    const sa2::native::ShaderEntry* ps;
  };
  // usf4 0.6.7: the packets paired in this frame, by command-buffer address, in a fixed table (linear probing)
  // whose entries of earlier frames count as empty. It was an unordered_map never emptied: a node allocation for
  // every new address (operator new under PairDraw in the console ring profile) and unbounded growth. Only
  // entries of the current frame were ever used. A frame with more paired packets than fit loses the pairing of
  // the extra ones for predicated-tiling replays, which then pair as a first draw does.
  struct PairedSlot {
    uint32_t address = 0;
    bool used = false;
    PairedPacket packet{};
  };
  static constexpr uint32_t kPairedSlots = 8192;  // power of 2
  std::vector<PairedSlot> paired_packets_ = std::vector<PairedSlot>(kPairedSlots);
  static uint32_t PairedIndex(uint32_t address) { return (address >> 2) * 0x9E3779B1u >> (32 - 13); }
  void NotePairedPacket(const PairedPacket& packet) {
    uint32_t i = PairedIndex(packet_address_);
    for (uint32_t probe = 0; probe < 64; ++probe, i = (i + 1) & (kPairedSlots - 1)) {
      PairedSlot& slot = paired_packets_[i];
      if (!slot.used || slot.packet.frame != swaps_ || slot.address == packet_address_) {
        slot = {packet_address_, true, packet};
        return;
      }
    }
  }
  const PairedPacket* FindPairedPacket() const {
    uint32_t i = PairedIndex(packet_address_);
    for (uint32_t probe = 0; probe < 64; ++probe, i = (i + 1) & (kPairedSlots - 1)) {
      const PairedSlot& slot = paired_packets_[i];
      if (!slot.used || slot.packet.frame != swaps_) return nullptr;  // empty for this frame: not there
      if (slot.address == packet_address_) return &slot.packet;
    }
    return nullptr;
  }
  std::deque<DrawRecord> pending_;
  const sa2::native::ShaderEntry* draw_vs_ = nullptr;
  const sa2::native::ShaderEntry* draw_ps_ = nullptr;
  uint64_t shaders_by_address_ = 0, miss_vs_addr_ = 0, miss_ps_addr_ = 0, miss_immediate_ = 0;
  uint64_t draws_unidentified_ = 0, shaders_identified_ = 0, shaders_unidentified_ = 0, presented_ = 0;
  uint64_t current_vs_hash_ = 0, current_ps_hash_ = 0;
  struct VsMismatchKey {
    const sa2::native::ShaderEntry* candidate;
    uint64_t loaded_hash;
    size_t loaded_words;
    bool operator==(const VsMismatchKey&) const = default;
  };
  struct VsMismatchKeyHash {
    size_t operator()(const VsMismatchKey& k) const {
      return size_t(k.loaded_hash ^ (uint64_t(reinterpret_cast<uintptr_t>(k.candidate)) * 0x9E3779B97F4A7C15ull) ^
                    (uint64_t(k.loaded_words) << 48));
    }
  };
  std::unordered_set<VsMismatchKey, VsMismatchKeyHash> vs_mismatch_memo_;
  uint64_t vs_mismatch_memo_hits_ = 0;
  uint32_t last_vs_load_address_ = UINT32_MAX, last_ps_load_address_ = UINT32_MAX;
  uint64_t shader_loads_reused_ = 0;
  uint64_t ps_identity_mismatches_ = 0;
  uint64_t vs_identity_mismatches_ = 0;
  uint64_t vs_final_proven_ = 0, vs_final_unproven_ = 0, vs_final_missing_ = 0;
  uint64_t vs_identity_generation_ = 1;
  uint64_t ps_identity_generation_ = 1;
  // sa2_gpu_vs_fetch_reorder: the entry the current VS load was identified as with its declared fetches reordered by
  // Direct3D (IdentifyShader; nullptr for every other load), the one candidate ComputeVertexIdentity accepts beyond
  // the ordered predicate, after proving it again on the loaded words; and the identity memo keys of such loads, so
  // that a memo hit restores it.
  const sa2::native::ShaderEntry* vs_reorder_entry_ = nullptr;
  std::unordered_set<uint64_t> vs_reorder_memo_keys_;
  uint64_t shaders_reordered_ = 0;
  std::unordered_map<const sa2::native::ShaderEntry*, std::pair<uint64_t, bool>> vs_identity_cache_;
  std::array<IdentityCell, kIdentityCells> vs_cells_{}, ps_cells_{}, fc_cells_{};
  // Ring CPU switches (sa2_native_flat_identity, _fast_pair, _raw_microcode, _pm4_fast) and the
  // remaining self-check uses of each.
  bool identity_flat_ = false, pairing_fast_ = false, raw_microcode_ = false, pm4_fast_ = false;
  bool identity_accept_fast_ = false;  // sa2_cpu_ring_identity_accept_fast (off for the session after a DIFFERENCE)
  uint32_t check_identity_ = 0, check_pairing_ = 0, check_raw_ = 0, check_pm4_ = 0, check_objects_ = 0;
  std::vector<uint32_t> raw_vs_, raw_ps_;  // the guest words of the last IM_LOAD of each stage (raw_microcode_)
  bool raw_vs_valid_ = false, raw_ps_valid_ = false;
  std::map<std::pair<uint64_t, uint64_t>, uint64_t> missing_draw_pairs_;
  rex::ui::WindowedAppContext* app_context_ = nullptr;
  std::unique_ptr<rex::ui::vulkan::VulkanProvider> provider_;
  std::unique_ptr<rex::ui::Presenter> presenter_;
  rex::runtime::FunctionDispatcher* dispatcher_ = nullptr;
  rex::system::KernelState* kernel_state_ = nullptr;
  rex::memory::Memory* memory_ = nullptr;
  bool mmio_registered_ = false;
  std::atomic<bool> active_{false};
  std::vector<uint32_t> registers_;
  std::array<std::array<uint16_t, 3>, 256> gamma_ramp_ = [] {
    std::array<std::array<uint16_t, 3>, 256> ramp{};
    for (uint32_t i = 0; i < 256; ++i) ramp[i].fill(uint16_t(i * 0x3FF / 0xFF));
    return ramp;
  }();
  uint32_t gamma_component_ = 0;
  uint32_t gamma_mask_ = 0b111;
  uint64_t gamma_version_ = 0, gamma_sent_version_ = 0, gamma_writes_ = 0;
  bool gamma_logged_ = false;
  std::atomic<uint32_t> callback_{0}, callback_data_{0};
  std::atomic<uint32_t> ring_base_{0}, ring_words_{0}, ring_generation_{0};
  std::atomic<uint32_t> write_pointer_{0}, read_writeback_{0};
  std::atomic<uint32_t> counter_{0};
  std::mutex ring_mutex_;
  std::atomic<bool> ring_waiting_{false};
  std::condition_variable ring_cv_;
  rex::system::object_ref<rex::system::XHostThread> ring_thread_, vblank_thread_;
  std::mutex relaxed_mutex_;               // sa2_vsync_relaxed: the ring's waits wake the vblank thread
  std::condition_variable relaxed_cv_;
  std::atomic<bool> vblank_wanted_{false};
  std::atomic<Clock::rep> vblank_asked_{0};  // when the flip wait asked (Clock ticks)
  std::atomic<uint64_t> vblanks_{0};        // vblank interrupts given (NoteVblank)
  std::atomic<int64_t> last_vblank_ns_{0};  // ... and the time of the last one
  uint32_t flip_poll_ = 0;                  // ring only: the learned flip wait's memory (FlipWaitEnded)
  struct FlipVotes {
    uint32_t poll, votes, ended;
  } flip_votes_[4] = {};
  int32_t diag_wait_logged_ = 0;  // sa2_diag_wait_log
  int32_t diag_vblank_logged_ = 0;
  uint64_t bin_mask_ = ~0ull, bin_select_ = ~0ull;
  // Ring-thread counters.
  uint64_t packets_ = 0, draws_ = 0, copies_ = 0, swaps_ = 0, waits_timed_out_ = 0;
  uint64_t opcodes_[128] = {};
  uint64_t opcode_words_[128] = {}, type0_packets_ = 0, type0_words_ = 0, type0_single_ = 0;  // PM4 mix report
  std::atomic<uint64_t> interrupts_{0};
  uint64_t reported_swaps_ = 0, reported_draws_ = 0, reported_copies_ = 0, reported_packets_ = 0,
           reported_interrupts_ = 0;
  Clock::time_point start_, last_report_;
  // banjo_trace_frame_s: 0 waiting, 1 tracing the current frame, 2 done.
  int trace_phase_ = 0;
  bool renderdoc_triggered_ = false;
  bool renderdoc_capturing_ = false;
  int trace_result_ = -1;  // last draw: -1 not attempted, 0 rejected, 1 drawn
  uint64_t trace_draws_ = 0, trace_quiet_draws_ = 0;
  uint32_t trace_target_[4] = {~0u, ~0u, ~0u, ~0u};
  std::unordered_set<uint32_t> resolve_destinations_;  // 4 KB pages of every resolve destination seen
};

}  // namespace

bool HasEntry(uint32_t object) {
  std::lock_guard<std::mutex> lock(g_by_address_mutex);
  return g_entry_of_object.count(object) != 0;
}

// sa2_cpu_note_shader_fast: shader objects whose NoteShaderObject work is finished (seen and code-mapped, or seen and
// never identified; nothing is ever erased from those sets), by (object >> 4) & 511. A hit returns exactly where the
// locked path would. ShaderObjectsFinished reads it for a whole draw.
static std::array<std::atomic<uint32_t>, 512> g_finished_shared{};

// banjo-nx: host pointer of a guest virtual address in a physical-memory view (the shader microcode, 0xA0000000 and
// up). On the Switch and on macOS the views map 1:1 onto `base`; on Windows the SDK shifts the 0xE0000000 view by
// 0x1000 (its 64 KB allocation granularity), so `base + address` reads the wrong bytes there. TranslateVirtual
// applies each heap's offset.
static const uint8_t* GuestPhysicalView(const uint8_t* base, uint32_t address) {
#if defined(_WIN32)
  return rex::system::kernel_state()->memory()->TranslateVirtual(address);
#else
  return base + address;
#endif
}

bool ShaderObjectsFinished(uint32_t vs, uint32_t ps) {
  if (!g_note_shader_pair.active || !g_note_shader_fast.active) return false;
  if (g_finished_shared[(vs >> 4) & 511].load(std::memory_order_relaxed) != vs ||
      g_finished_shared[(ps >> 4) & 511].load(std::memory_order_relaxed) != ps)
    return false;
  g_note_shader_pair.UseSampled();
  return true;
}

void NoteShaderObject(const uint8_t* base, uint32_t object, bool vertex) {
  const sa2::native::ShadersNative* library = g_library.load(std::memory_order_acquire);
  auto load32 = [base](uint32_t a) { return rex::memory::load_and_swap<uint32_t>(base + a); };
  bool trace_this = false;  // sa2-nx diagnostic: log the path taken for the first objects
  // sa2_cpu_note_shader_fast: once the 32 traced objects are logged the set cannot change, so the mutex is skipped.
  // Round 3: SA2 binds fewer than 32 distinct objects, so the set never filled and every call still took the mutex
  // for an insert that finds the object (console: ~2.4 % of the main thread in that insert plus ~3 % in the lock). A
  // key already in the set is remembered in a small lock-free table and skips the lock: it can never be logged again.
  static std::atomic<bool> traced_full{false};
  static std::array<std::atomic<uint64_t>, 64> traced_known{};
  const uint64_t trace_key = uint64_t(object) << 1 | (vertex ? 1 : 0);
  std::atomic<uint64_t>& trace_slot = traced_known[(trace_key >> 3 ^ trace_key >> 11) & 63];
  const bool trace_known = g_note_shader_fast.active && trace_slot.load(std::memory_order_relaxed) == trace_key + 1;
  if (!trace_known && (!g_note_shader_fast.active || !traced_full.load(std::memory_order_relaxed))) {
    // sa2-nx diagnostic: the first distinct objects this receives
    static std::mutex trace_mutex;
    static std::unordered_set<uint64_t> traced;
    std::lock_guard<std::mutex> lock(trace_mutex);
    if (traced.size() < 32 && traced.insert(trace_key).second) {
      trace_this = true;
      REXLOG_INFO("[native] NoteShaderObject({}, {:08X}): library {}, word0 {:08X}", vertex ? "VS" : "PS", object,
                  library ? "loaded" : "none",
                  object >= 0x40000000 && object < 0x7F000000 ? load32(object) : 0u);
    }
    if (traced.size() >= 32) traced_full.store(true, std::memory_order_relaxed);
    // In the set now (inserted or already there), or the set is full: either way never logged again.
    if (g_note_shader_fast.active && (traced.count(trace_key) || traced_full.load(std::memory_order_relaxed)))
      trace_slot.store(trace_key + 1, std::memory_order_relaxed);
  }
  if (!library || object < 0x40000000 || object >= 0x7F000000) { if (trace_this) REXLOG_INFO("[native]   {:08X} exit L15", object); return; }
  if ((load32(object) & 0xFF) != (vertex ? 6u : 7u)) {
    // sa2-nx diagnostic: what is bound where a shader object was expected (once per object, first 16).
    static std::mutex logged_mutex;
    static std::unordered_set<uint32_t> logged;
    std::lock_guard<std::mutex> lock(logged_mutex);
    if (logged.size() < 16 && logged.insert(object).second) {
      std::string words;
      for (uint32_t i = 0; i < 0x30; i += 4) words += fmt::format(" {:08X}", load32(object + i));
      REXLOG_WARN("[native] {} slot holds {:08X}, not a shader object (type byte {:02X}):{}", vertex ? "VS" : "PS",
                  object, load32(object) & 0xFF, words);
    }
    if (trace_this) REXLOG_INFO("[native]   {:08X} exit L27", object);
    return;
  }
  // Objects whose work is finished (seen and code-mapped: the locked path below only returns for them) are
  // remembered per game thread, so the per-draw call skips the shared mutex (contended with the ring thread).
  // sa2_cpu_note_shader_fast: "finished" is a property of the object, not of the thread (seen and code-mapped, or seen
  // and never identified; nothing is ever erased from those sets), so one table serves every game thread without a
  // TLS lookup. A hit returns exactly where the locked path below would; a racing reader at worst misses.
  std::array<std::atomic<uint32_t>, 512>& finished_shared = g_finished_shared;
  std::atomic<uint32_t>* finished_slot = nullptr;
  uint32_t* finished_local = nullptr;
  if (g_note_shader_fast.active) {
    finished_slot = &finished_shared[(object >> 4) & 511];
    if (finished_slot->load(std::memory_order_relaxed) == object) {
      g_note_shader_fast.UseSampled();
      if (trace_this) REXLOG_INFO("[native]   {:08X} exit L33", object);
      return;
    }
  } else {
    thread_local std::array<uint32_t, 512> finished_cache{};
    finished_local = &finished_cache[(object >> 4) & 511];
    if (*finished_local == object) { if (trace_this) REXLOG_INFO("[native]   {:08X} exit L33", object); return; }
  }
  const auto mark_finished = [&] {
    if (finished_slot) finished_slot->store(object, std::memory_order_relaxed);
    else *finished_local = object;
  };
  {
    std::lock_guard<std::mutex> lock(g_by_address_mutex);
    if (!g_objects_seen.insert(object).second) {
      // Already identified: map its current microcode once (for VS, after D3D patched it).
      auto it = g_entry_of_object.find(object);
      if (it != g_entry_of_object.end() && g_code_mapped.count(object)) mark_finished();
      // Seen but never identified: only the first sighting identifies an object (below), so this stays
      // unidentified; remember it too instead of taking the mutex on every draw (~0.5 % of the render thread).
      if (it == g_entry_of_object.end()) mark_finished();
      if (it == g_entry_of_object.end() || g_code_mapped.count(object)) { if (trace_this) REXLOG_INFO("[native]   {:08X} exit L43", object); return; }
      const uint32_t mc = load32(object + (vertex ? 0x20 : 0x18));
      const uint32_t header = object + (vertex ? 0x368 : 0x28);
      const uint32_t sh = header + load32(header + 24);
      const uint32_t mc_offset = load32(sh), mc_bytes = load32(sh + 4);
      if (mc < 0xA0000000u || !mc_bytes || mc_bytes > 0x40000) { if (trace_this) REXLOG_INFO("[native]   {:08X} exit L48", object); return; }
      const uint64_t h = CodeHash(GuestPhysicalView(base, mc) + mc_offset, mc_bytes / 4);
      auto [slot, inserted] = g_by_code.emplace(h, it->second);
      if (!inserted && slot->second != it->second) ++g_code_conflicts;
      g_code_mapped.insert(object);
      mark_finished();
      if (trace_this) REXLOG_INFO("[native]   {:08X} exit L54", object);
      return;
    }
  }
  // Same layout as src/me_shader_dump.cpp: container copy in the object, microcode pointer.
  const uint32_t header = object + (vertex ? 0x368 : 0x28);
  const uint32_t virtual_size = load32(header + 4);
  const uint32_t physical_size = load32(header + 8);
  const uint32_t microcode = load32(object + (vertex ? 0x20 : 0x18));
  if ((load32(header) & 0xFFFFFF00u) != 0x102A1100u || virtual_size < 28 || virtual_size > 0x40000 ||
      !physical_size || physical_size > 0x40000 || microcode < 0xA0000000u) {
    // sa2-nx: the object does not have the container layout this expects (a newer XDK, or one of Direct3D's own
    // shader objects). Say what is there, once per object, so the layout can be checked.
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1, std::memory_order_relaxed) < 16) {
      std::string words;
      for (uint32_t i = 0; i < 0x40; i += 4) words += fmt::format(" {:08X}", load32(object + i));
      REXLOG_WARN("[native] {} object {:08X} not identified: container word {:08X} sizes {}/{}, microcode {:08X}; "
                  "object:{}", vertex ? "VS" : "PS", object, load32(header), virtual_size, physical_size, microcode,
                  words);
    }
    if (trace_this) REXLOG_INFO("[native]   {:08X} exit L74", object);
    return;
  }
  std::vector<uint8_t> container(virtual_size + physical_size);
  std::memcpy(container.data(), base + header, virtual_size);
  std::memcpy(container.data() + virtual_size, GuestPhysicalView(base, microcode), physical_size);
  const sa2::native::ShaderEntry* entry = library->IdentifyContainer(container);
  if (!entry && vertex) {
    std::lock_guard<std::mutex> lock(g_by_address_mutex);
    if (!g_vs_by_virtual_built) BuildVirtualIndex(library);
    auto it = g_vs_by_virtual.find(XXH3_64bits(container.data(), virtual_size));
    if (it != g_vs_by_virtual.end()) entry = it->second;
  }
  // Direct3D's internal containers may not be in the package byte-for-byte, because discovery
  // replaces their incomplete virtual metadata with a conservative register-linked wrapper.
  // Their physical microcode is still authoritative. The library code index deliberately prefers
  // the complete wrapper when the same code occurs in multiple containers.
  if (!entry) {
    const uint32_t shader_header = virtual_size >= 28 ? load32(header + 24) : 0;
    const uint32_t physical_offset =
        shader_header + 8 <= virtual_size ? load32(header + shader_header) : 0;
    const uint32_t shader_bytes =
        shader_header + 8 <= virtual_size ? load32(header + shader_header + 4) : 0;
    if (shader_bytes && !(shader_bytes & 3) &&
        uint64_t(physical_offset) + shader_bytes <= physical_size) {
      const uint64_t code = CodeHash(container.data() + virtual_size + physical_offset,
                                     shader_bytes / 4) ^ (vertex ? 1ull : 0ull);
      if (auto it = g_library_by_code.find(code); it != g_library_by_code.end()) {
        entry = it->second;
        REXLOG_INFO("[native] shader object {:08X} matched register-linked {} n{} by microcode",
                    object, vertex ? "VS" : "PS", entry->number);
      }
    }
  }
  if (!entry) {
    const uint32_t shader_header = virtual_size >= 28 ? load32(header + 24) : 0;
    const uint32_t physical_offset = shader_header + 8 <= virtual_size ? load32(header + shader_header) : 0;
    const uint32_t shader_bytes = shader_header + 8 <= virtual_size ? load32(header + shader_header + 4) : 0;
    REXLOG_INFO("[native] shader object {:08X} not in the library: {}_{:016x}; mc {:08X}, "
                "virtual {} physical {}, shader header +{:X}, code +{:X}/{} -> ring {:08X}",
                object, vertex ? "vs" : "ps", XXH3_64bits(container.data(), container.size()),
                microcode, virtual_size, physical_size, shader_header, physical_offset, shader_bytes,
                (microcode & 0x1FFFFFFFu) + physical_offset);
    // Not in the library (containers Direct3D or the engine build at startup): keep a copy with
    // SA2_SHADER_MISSING=<folder> so the library can be extended (shaders/tools/build_shader_spirv.sh).
    if (const char* folder = ShaderDumpFolder("SA2_SHADER_MISSING", "missing"); folder && *folder) {
      std::error_code ec;
      std::filesystem::create_directories(folder, ec);
      const uint64_t h = XXH3_64bits(container.data(), container.size());
      const auto path = std::filesystem::path(folder) /
                        fmt::format("{}_{:016x}.bin", vertex ? "vs" : "ps", h);
      if (std::FILE* f = std::fopen(path.string().c_str(), "wb")) {
        std::fwrite(container.data(), 1, container.size(), f);
        std::fclose(f);
      }
    }
    if (trace_this) REXLOG_INFO("[native]   {:08X} exit L129", object);
    return;
  }
  const uint32_t shader_header = rex::memory::load_and_swap<uint32_t>(container.data() + 24);
  const uint32_t physical_offset =
      shader_header + 4 <= virtual_size
          ? rex::memory::load_and_swap<uint32_t>(container.data() + shader_header)
          : 0;
  const uint32_t mc_bytes = shader_header + 8 <= virtual_size
                                ? rex::memory::load_and_swap<uint32_t>(container.data() + shader_header + 4)
                                : 0;
  std::lock_guard<std::mutex> lock(g_by_address_mutex);
  g_by_address[(microcode & 0x1FFFFFFFu) + physical_offset] = entry;
  g_entry_of_object[object] = entry;
  PublishObject(object, entry);
  if (trace_this) REXLOG_INFO("[native]   {:08X} identified as n{}", object, entry->number);
  if (!vertex) {  // pixel shaders are never patched: map their microcode now
    if (mc_bytes && virtual_size + physical_offset + mc_bytes <= container.size()) {
      const uint64_t h = CodeHash(container.data() + virtual_size + physical_offset, mc_bytes / 4);
      auto [slot, inserted] = g_by_code.emplace(h, entry);
      if (!inserted && slot->second != entry) ++g_code_conflicts;
      g_code_mapped.insert(object);
    }
  }
}

namespace {
// The bound shader objects in the game's D3D device (SetDeviceShaderOffsets); Mass Effect's by default.
std::atomic<uint32_t> g_device_vs_offset{12416};
std::atomic<uint32_t> g_device_ps_offset{12412};
}  // namespace

uint64_t PresentedSwaps() { return g_presented_swaps.load(std::memory_order_acquire); }

bool NativeActive() { return g_native_active.load(std::memory_order_acquire); }

namespace {
// The native Direct3D layer's records (the drawing thread is their only producer). The consumer's g_tail is read again
// only when this older value says there is no room: a value read earlier only ever understates the room, and reading
// the line the ring thread keeps writing cost a cache miss per record.
uint64_t g_api_tail_seen = 0;
uint64_t g_api_ready = 0;                // ring words (monotonic) up to which free space was made ready (zeroed)
uint64_t g_api_reserved = ~0ull;         // ReserveApiRecord: the record's first word (monotonic)
uint64_t g_api_reserved_head = ~0ull;    // ... and g_head when it was reserved

// Zero-ahead for the ring of records. The ring (8 MB) is far larger than the Switch's 2 MB L2, so every record line
// was a write miss: the A57 reads the line from DRAM before the stores can complete, and the publish's store-release
// waited for all of them (round 10 console: most of the publish's samples). DC ZVA allocates a block of zeros in the
// cache without reading memory, so free ring space is readied this way a little ahead of the head. The instruction
// is allowed in user mode when DCZID_EL0.DZP is clear (newlib's memset checks the same); its block size is
// DCZID_EL0.BS (log2 of 4-byte words). Elsewhere a memset of the same blocks runs the same logic, so the PC's
// verify and pixel tests cover which words are zeroed.
constexpr uint32_t kReadyAheadWords = 1024;
inline uint32_t ZeroBlockWords() {
#if defined(__aarch64__)
  static const uint32_t words = [] {
    uint64_t id;
    asm volatile("mrs %0, dczid_el0" : "=r"(id));
    return (id & 16) ? 0u : (1u << (id & 15));
  }();
  return words;
#else
  return 16;
#endif
}
inline void ZeroBlock(uint32_t* p, uint32_t words) {
#if defined(__aarch64__)
  (void)words;
  asm volatile("dc zva, %0" : : "r"(p) : "memory");
#else
  std::memset(p, 0, size_t(words) * 4);
#endif
}

// The first word of a record of `need` words at the head (the rest of the lap skipped when it does not fit), or ~0 when
// the ring is behind (no room).
uint64_t ApiRecordStart(uint32_t need) {
  using namespace direct;
  constexpr uint64_t mask = kRingWords - 1;
  const uint64_t pos = g_head.load(std::memory_order_relaxed);
  uint64_t start = pos;
  if ((pos & mask) + need > kRingWords) start = (pos | mask) + 1;  // wrap
  if (start + need - g_api_tail_seen > kRingWords) {
    g_api_tail_seen = g_tail.load(std::memory_order_acquire);
    if (start + need - g_api_tail_seen > kRingWords) return ~0ull;
  }
  return start;
}

// Publishes the record of `need` words at `start` (its ops written), the head having been at `pos`: its header, the
// skip mark at `pos` when it wrapped, the head (store-release), and the prefetch of the next records' lines.
uint32_t ApiRecordPublish(uint64_t pos, uint64_t start, uint32_t need) {
  using namespace direct;
  constexpr uint64_t mask = kRingWords - 1;
  uint32_t* const rec = &g_ring[start & mask];
  if (start != pos) g_ring[pos & mask] = kPad;
  uint32_t seq = uint32_t(++g_seq);
  if (seq == kPad) seq = uint32_t(++g_seq);
  rec[0] = seq;
  rec[1] = need;
  const uint64_t head = start + need;
  g_head.store(head, std::memory_order_release);
  Bump(g_record_words, need);
  // The next records' space, readied ahead (zero-ahead above; a write prefetch where DC ZVA is not allowed): whole
  // blocks only, each once, from the first block boundary at or past the new head (the block holding the head may
  // hold this record's last words), and only free space: below the consumer's tail plus the ring's size, judged by the
  // cached older tail (which only understates the room).
  if (g_api_tail_seen + kRingWords < head + kReadyAheadWords) [[unlikely]]  // once a lap: the cached tail is stale
    g_api_tail_seen = g_tail.load(std::memory_order_acquire);
  const uint64_t limit = std::min(head + kReadyAheadWords, g_api_tail_seen + kRingWords);
  if (const uint32_t block = g_api_zero_ahead.active ? ZeroBlockWords() : 0) [[likely]] {
    uint64_t from = std::max(g_api_ready, (head + block - 1) & ~uint64_t(block - 1));
    if (sa2::enh::VerifyMode()) [[unlikely]] {
      // Self-check: every block about to be zeroed lies past the published head and inside the free space.
      sa2::enh::VerifyCheck();
      const uint64_t tail = g_tail.load(std::memory_order_acquire);
      if (from < head || (from < limit && limit > tail + kRingWords)) {
        sa2::enh::VerifyDifference();
        REXLOG_ERROR("[native] DIFFERENCE: sa2_api_ring_zero_ahead would zero words {}..{} (head {}, tail {})", from,
                     limit, head, tail);
      }
    }
    const uint64_t first = from;
    for (; from + block <= limit; from += block) ZeroBlock(&g_ring[from & mask], block);
    g_api_ready = std::max(g_api_ready, from);
    if (from != first) g_api_zero_ahead.UseSampled();
  } else {
    uint64_t from = std::max(g_api_ready, head);
    for (; from < limit; from += 16) {
      if ((from & mask) + 16 > kRingWords) continue;  // the lap's last partial line: the next lap starts at word 0
      __builtin_prefetch(&g_ring[from & mask], 1, 3);
    }
    g_api_ready = std::max(g_api_ready, limit);
  }
  return seq;
}
}  // namespace

uint32_t PublishApiRecord(const uint32_t* ops, uint32_t n) {
  using namespace direct;
  constexpr uint64_t mask = kRingWords - 1;
  if (g_failed.load(std::memory_order_relaxed)) return 0;
  if (!g_ring) [[unlikely]] g_ring = std::make_unique<uint32_t[]>(kRingWords);
  const uint32_t need = 2 + n;
  if (need > kRingWords / 4) return 0;
  const uint64_t pos = g_head.load(std::memory_order_relaxed);
  const uint64_t start = ApiRecordStart(need);
  if (start == ~0ull) return 0;  // the ring is behind: no room
  std::memcpy(&g_ring[(start & mask) + 2], ops, size_t(n) * 4);
  g_api_reserved = ~0ull;  // a reservation does not survive another record
  return ApiRecordPublish(pos, start, need);
}

uint32_t* ReserveApiRecord(uint32_t max_words) {
  using namespace direct;
  constexpr uint64_t mask = kRingWords - 1;
  if (g_failed.load(std::memory_order_relaxed)) return nullptr;
  if (!g_ring) [[unlikely]] g_ring = std::make_unique<uint32_t[]>(kRingWords);
  const uint32_t need = 2 + max_words;
  if (need > kRingWords / 4) return nullptr;
  const uint64_t start = ApiRecordStart(need);
  if (start == ~0ull) return nullptr;
  g_api_reserved = start;
  g_api_reserved_head = g_head.load(std::memory_order_relaxed);
  return &g_ring[(start & mask) + 2];
}

uint32_t CommitApiRecord(const uint32_t* ops, uint32_t n) {
  using namespace direct;
  constexpr uint64_t mask = kRingWords - 1;
  if (g_failed.load(std::memory_order_relaxed)) return 0;
  const uint64_t pos = g_head.load(std::memory_order_relaxed);
  // One producer: nothing else publishes between the reservation and its commit. If something did, the reserved words
  // may have been written over: refused (logged by the caller), never published half.
  if (g_api_reserved == ~0ull || pos != g_api_reserved_head || ops != &g_ring[(g_api_reserved & mask) + 2])
    [[unlikely]] return 0;
  const uint64_t start = g_api_reserved;
  g_api_reserved = ~0ull;
  return ApiRecordPublish(pos, start, 2 + n);
}

void SetPresentHook(void (*hook)(uint64_t presented)) { g_present_hook.store(hook, std::memory_order_release); }

bool RingIdle() {
  // The ring's position modulo its size equals the published write pointer's (both are word indices in the ring).
  if (!g_native_active.load(std::memory_order_acquire)) return true;
  return g_ring_processed.load(std::memory_order_acquire) == g_ring_published.load(std::memory_order_acquire);
}

void SetDeviceShaderOffsets(uint32_t vertex_shader_offset, uint32_t pixel_shader_offset) {
  g_device_vs_offset.store(vertex_shader_offset, std::memory_order_relaxed);
  g_device_ps_offset.store(pixel_shader_offset, std::memory_order_relaxed);
}

void NoteDrawCall(const uint8_t* base, uint32_t device, uint32_t type, uint32_t r5, uint32_t r6, uint32_t r7,
                  DrawDirect* direct) {
  if (!g_native_active.load(std::memory_order_acquire)) return;
  // sa2_cpu_main_thread_priority (off by default; configs/sa2_test_main_priority_on.toml): the thread that draws is
  // raised once, on its first draw, one step above the other game threads (from nfsmw-nx's frame-handoff threads).
  static std::atomic<bool> priority_done{false};
  if (g_main_priority.active && !priority_done.load(std::memory_order_relaxed)) [[unlikely]] {
    priority_done.store(true, std::memory_order_relaxed);
    const int32_t priority = REXCVAR_GET(sa2_main_thread_priority);
#if REX_PLATFORM_SWITCH
    const bool ok = RexSwitchSetCurrentThreadPriorityOk(int(priority));
#else
    const bool ok = true;  // Horizon priorities only
#endif
    g_main_priority.Use();
    REXLOG_INFO("[enh] sa2_cpu_main_thread_priority: the drawing game thread now runs at priority {:#x} ({}); the "
                "other game threads stay at 0x3B", uint32_t(priority), ok ? "accepted" : "REJECTED by the kernel");
  }
  // The device the call was made on (r3). This used to be Mass Effect's device address (0x4004B000) and offsets,
  // which in another game reads unrelated memory: every record then carried a wrong command-buffer position and
  // no ring draw could be paired by address.
  if (device < 0x40000000 || device >= 0x7F000000) return;
  const uint32_t vs =
      rex::memory::load_and_swap<uint32_t>(base + device + g_device_vs_offset.load(std::memory_order_relaxed));
  const uint32_t ps =
      rex::memory::load_and_swap<uint32_t>(base + device + g_device_ps_offset.load(std::memory_order_relaxed));
  if (!ShaderObjectsFinished(vs, ps)) {
    NoteShaderObject(base, vs, true);
    NoteShaderObject(base, ps, false);
  }
  const uint64_t n = g_draw_records.fetch_add(1, std::memory_order_relaxed);
  const uint32_t start = rex::memory::load_and_swap<uint32_t>(base + device + 48) & 0x1FFFFFFFu;
  DrawRecord record{type, r5, r6, r7, vs, ps, start};
  record.device = device;
  if (REXCVAR_GET(sa2_api_discover_draws) > 0) [[unlikely]] record.snapshot = discover::Take(base, device, n);
  // sa2_cpu_direct_pair: held back for the call's direct record (EndDrawDirect); a submission during the call or a draw
  // that keeps its packets pushes it as below. Verify mode pushes it too, for the ring's comparison with the table.
  if (direct && direct->device && g_direct_pair.active && RecordTable() &&
      !direct::g_pair_off.load(std::memory_order_relaxed)) {
    if (direct::PendingPair* p = direct::ThreadPending()) {
      p->record = record;
      p->start = start;
      p->state.store(direct::PendingPair::kPending, std::memory_order_release);
      direct->pair = true;
      if (!sa2::enh::VerifyMode()) return;
    }
  }
  if (RecordTable()) {
    PushRecord(std::move(record), start);  // lock-free; the ring inserts it into its table
    return;
  }
  std::lock_guard<std::mutex> lock(g_draw_mutex);
  g_draw_by_address[start] = std::move(record);
  if (g_draw_by_address.size() > 65536) g_draw_by_address.erase(g_draw_by_address.begin());
}

// sa2_cpu_direct_state, game thread: what the draw hook needs from before the original draw function runs.
DrawDirect BeginDrawDirect(const uint8_t* base, uint32_t device) {
  DrawDirect d;
  if (!g_direct_state.active || direct::g_failed.load(std::memory_order_relaxed) ||
      !g_native_active.load(std::memory_order_acquire) || device < 0x40000000 || device >= 0x7F000000)
    return d;
  d.device = device;
  d.start = rex::memory::load_and_swap<uint32_t>(base + device + 48);  // the last word written so far
  d.limit = rex::memory::load_and_swap<uint32_t>(base + device + 52);
  d.soft_limit = rex::memory::load_and_swap<uint32_t>(base + device + 56);
  d.submissions = direct::g_wptr_writes.load(std::memory_order_relaxed);
  // sa2_cpu_direct_flush: this call's flush functions may write its record (FlushRunsDirect). g_marking is one flag, so
  // only one thread may draw this way: a second one turns the native flush off for good.
  if (g_direct_flush.active && !direct::g_flush_failed.load(std::memory_order_relaxed) &&
      !direct::g_flush_multi.load(std::memory_order_relaxed)) {
    using namespace direct;
    const uint64_t me = uint64_t(reinterpret_cast<uintptr_t>(&t_flush));  // one per thread
    uint64_t owner = g_flush_thread.load(std::memory_order_relaxed);
    if (owner == 0 && g_flush_thread.compare_exchange_strong(owner, me, std::memory_order_relaxed)) owner = me;
    if (owner != me) {
      if (!g_flush_multi.exchange(true, std::memory_order_relaxed))
        REXLOG_WARN("[native] sa2_cpu_direct_flush: a second thread draws; Direct3D's flushes write PM4 again");
    } else {
      FlushState* f = t_flush ? t_flush : (t_flush = new FlushState());
      f->active = true;
      f->marking = false;
      f->device = device;
      f->submissions = d.submissions;
      f->size = 0;
      f->spots = 0;
    }
  }
  return d;
}

namespace {
// n big-endian words to host order (vectorised by the compiler: one rev32 per four words).
inline void SwapWords(uint32_t* dst, const uint8_t* src, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t w;
    std::memcpy(&w, src + size_t(i) * 4, 4);
    dst[i] = __builtin_bswap32(w);
  }
}

// sa2_cpu_direct_flush: the spot of this call whose placeholder NOP is at guest `p` (header `h`), or nullptr.
const direct::FlushSpot* PlaceholderAt(uint8_t* base, uint32_t p, uint32_t h, const direct::FlushState* f) {
  if (h != direct::kFlushHeader || !f) return nullptr;
  if (rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p + 4)) != direct::kFlushMagic) return nullptr;
  const uint32_t i = rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p + 8));
  return i < f->spots && f->spot[i].address == p ? &f->spot[i] : nullptr;
}

// Re-encodes the record's operations as packets and compares them with what Direct3D wrote (zero words, type-2
// fillers and NOPs aside): the decoder's own check, for the first records and every record in verify mode. A
// placeholder of sa2_cpu_direct_flush stands for its spot's ops (FlushCheck compared those with the original).
bool SameAsWritten(uint8_t* base, uint32_t first, uint32_t draw_at, const uint32_t* ops, size_t size,
                   const direct::FlushState* f) {
  size_t o = 0;
  for (uint32_t p = first; p < draw_at;) {
    const uint32_t h = rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p));
    const uint32_t type = h >> 30;
    if ((type == 0 && !h) || type == 2) { p += 4; continue; }
    if (type == 3 && ((h >> 8) & 0x7F) == xenos::PM4_NOP) {
      if (const direct::FlushSpot* s = PlaceholderAt(base, p, h, f)) {
        const size_t n = s->end - s->begin;
        if (o + n > size || std::memcmp(ops + o, f->ops + s->begin, n * 4) != 0) return false;
        o += n;
      }
      p += 4 * (2 + ((h >> 16) & 0x3FFF));
      continue;
    }
    if (o >= size) return false;
    const uint32_t tag = ops[o], kind = tag & 0xFF, count = tag >> 8;
    const uint32_t off = (p - first) / 4;
    if (kind == direct::kRunRef || kind == direct::kRunSingleRef || kind == direct::kType1Ref ||
        kind == direct::kType3Ref) {
      // A reference: the same packet, by register and offset (its values are the command buffer's own).
      const uint32_t n = type == 1 ? 2 : ((h >> 16) & 0x3FFF) + 1;
      bool same = false;
      if (type == 0)
        same = kind == (((h >> 15) & 1) ? direct::kRunSingleRef : direct::kRunRef) && count == n &&
               ops[o + 1] == (h & 0x7FFF) && ops[o + 2] == off + 1;
      else if (type == 1)
        same = kind == direct::kType1Ref && ops[o + 1] == h && ops[o + 2] == off + 1;
      else
        same = kind == direct::kType3Ref && count == n && ops[o + 1] == h && ops[o + 2] == off;
      if (!same || off < 2) return false;
      o += 3;
      p += 4 * (1 + n);
      continue;
    }
    if (type == 0) {
      const uint32_t n = ((h >> 16) & 0x3FFF) + 1;
      if (kind != (((h >> 15) & 1) ? direct::kRunSingle : direct::kRun) || count != n || ops[o + 1] != (h & 0x7FFF))
        return false;
      for (uint32_t i = 0; i < n; ++i)
        if (ops[o + 2 + i] != rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p + 4 + 4 * i))) return false;
      o += 2 + n;
      p += 4 * (1 + n);
    } else if (type == 1) {
      if (kind != direct::kType1 || ops[o + 1] != h ||
          ops[o + 2] != rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p + 4)) ||
          ops[o + 3] != rex::memory::load_and_swap<uint32_t>(direct::Guest(base, p + 8)))
        return false;
      o += 4;
      p += 12;
    } else {
      const uint32_t n = ((h >> 16) & 0x3FFF) + 1;
      if (kind != direct::kType3 || count != n || ops[o + 1] != h || ops[o + 2] != (p & 0x1FFFFFFFu) ||
          std::memcmp(&ops[o + 3], direct::Guest(base, p + 4), size_t(n) * 4) != 0)
        return false;
      o += 3 + n;
      p += 4 * (1 + n);
    }
  }
  return o == size;
}

// sa2_cpu_direct_flush's self-check: the original function, run on a copy of the device whose command buffer is a
// scratch buffer (r3 and the shadow argument rebased into the copy), must write exactly these register runs (fillers
// aside) and leave the copy's dirty masks (+0..47, +60..63) as they were. The real device is never touched (a
// submission from another thread could read its write pointer at any moment), and the PPC context is put back.
// false: a difference (logged).
bool FlushCheck(PPCContext& ctx, uint8_t* base, uint32_t device, uint32_t shadow, const uint32_t* ops, uint32_t n,
                void (*original)(PPCContext&, uint8_t*), FlushKind kind) {
  using namespace direct;
  constexpr uint32_t kCopy = 0x8000;           // the device (2008 XDK: 0x5E00 bytes; SA2's shadows end below 0x2C00)
  constexpr uint32_t kBuffer = 256 * 1024;     // the command buffer
  constexpr uint32_t kMasks = 64;              // the dirty masks and the command buffer pointers (48-59 differ)
  static uint32_t scratch = 0;
  if (!scratch) scratch = rex::system::kernel_state()->memory()->SystemHeapAlloc(kCopy + kBuffer, 0x1000);
  if (!scratch || shadow - device >= kCopy) return true;  // nothing to check against
  const uint32_t copy = scratch, buffer = scratch + kCopy;
  std::memcpy(base + copy, base + device, kCopy);
  rex::memory::store_and_swap<uint32_t>(base + copy + 48, buffer);
  rex::memory::store_and_swap<uint32_t>(base + copy + 52, buffer + kBuffer - 4096);
  rex::memory::store_and_swap<uint32_t>(base + copy + 56, buffer + kBuffer - 4096);
  uint8_t before[kMasks];
  std::memcpy(before, base + copy, kMasks);
  const PPCContext saved = ctx;
  ctx.r3.u64 = copy;
  if (kind != FlushKind::kFetch) ctx.r6.u64 = copy + (shadow - device);  // the fetch flush takes its shadow from r3
  original(ctx, base);
  const uint32_t last = rex::memory::load_and_swap<uint32_t>(base + copy + 48);
  std::memcpy(before + 48, base + copy + 48, 4);  // the write pointer moves: not compared
  ctx = saved;
  const char* diff = nullptr;
  uint32_t diff_offset = 0, diff_before = 0, diff_after = 0;
  for (uint32_t i = 0; i < kMasks; i += 4) {
    uint32_t x, y;
    std::memcpy(&x, before + i, 4);
    std::memcpy(&y, base + copy + i, 4);
    if (x != y) {
      diff = "the original changed a dirty mask or a limit";
      diff_offset = i;
      diff_before = __builtin_bswap32(x);
      diff_after = __builtin_bswap32(y);
      break;
    }
  }
  uint32_t o = 0;
  for (uint32_t p = buffer + 4; !diff && p <= last;) {
    const uint32_t h = rex::memory::load_and_swap<uint32_t>(Guest(base, p));
    if (h == 0x80000000u || h == 0) {  // type-2 filler, zero word
      p += 4;
      continue;
    }
    const uint32_t count = ((h >> 16) & 0x3FFF) + 1;
    if ((h >> 30) != 0 || ((h >> 15) & 1)) {
      diff = "a packet other than a register run";
    } else if (o + 2 + count > n || ops[o] != (kRun | (count << 8)) || ops[o + 1] != (h & 0x7FFF)) {
      diff = "another register run";
    } else {
      for (uint32_t i = 0; i < count && !diff; ++i)
        if (ops[o + 2 + i] != rex::memory::load_and_swap<uint32_t>(Guest(base, p + 4 + 4 * i))) diff = "a value";
    }
    o += 2 + count;
    p += 4 * (1 + count);
  }
  if (!diff && o != n) diff = "fewer register runs";
  if (diff) {
    sa2::enh::VerifyDifference();
    g_flush_failed.store(true, std::memory_order_relaxed);
    REXLOG_ERROR("[native] DIFFERENCE: sa2_cpu_direct_flush ({}) does not match Direct3D's flush ({}; device+{} "
                 "{:08X} -> {:08X}); off from now on",
                 kind == FlushKind::kConstants ? "constants" : kind == FlushKind::kFetch ? "fetch" : "groups", diff,
                 diff_offset, diff_before, diff_after);
    return false;
  }
  return true;
}
}  // namespace

// sa2_cpu_direct_flush, game thread (the flush hooks, src/me/sa2_me_hooks.cpp): see me_native_system.h and the
// FlushState notes in namespace direct.
bool FlushRunsDirect(PPCContext& ctx, uint8_t* base, uint32_t device, uint64_t mask, uint32_t register_base,
                     uint32_t shadow, FlushKind kind, void (*original)(PPCContext&, uint8_t*)) {
  using namespace direct;
  FlushState* const f = t_flush;
  if (!f || !f->active || f->device != device || !mask) return false;
  const uint32_t per_bit = kind == FlushKind::kConstants ? 16 : kind == FlushKind::kFetch ? 6 : 1;
  // The original's fast path, followed on paper: every run must fit where it checks (the slow path makes room, may
  // submit, and the fetch flush then ends with other values), and the placeholder must fit too.
  const uint32_t wptr = rex::memory::load_and_swap<uint32_t>(base + device + 48);
  const uint32_t limit = rex::memory::load_and_swap<uint32_t>(base + device + 52);
  if (!(wptr + 12 < limit)) {
    Bump(g_flush_slow, 1);
    return false;
  }
  uint32_t w = wptr;
  uint32_t words = kind == FlushKind::kFetch ? 5 : 0;  // the fetch flush's end packet
  for (uint64_t m = mask; m;) {
    const int lz = std::countl_zero(m);
    m <<= lz;
    const int run = std::countl_one(m);
    const uint32_t count = uint32_t(run) * per_bit;
    bool fits;
    switch (kind) {
      case FlushKind::kConstants: {  // up to three fillers put the values on 16 bytes
        static constexpr uint32_t kFill[4] = {2, 1, 0, 3};
        fits = w + (count + 3) * 4 < limit;
        w += 4 * kFill[(w >> 2) & 3] + 4 + 4 * count;
        break;
      }
      case FlushKind::kFetch:  // one filler puts the values on 8 bytes
        fits = w + (count + 5) * 4 < limit;
        w += (w & 7) + 4 + 4 * count;
        break;
      default:
        fits = w + count * 4 < limit;
        w += 4 + 4 * count;
        break;
    }
    if (!fits) {
      Bump(g_flush_slow, 1);
      return false;
    }
    words += 2 + count;
    m = run == 64 ? 0 : m << run;
  }
  // Room: beside the call's other ops, and in the ring of records when every op of the call goes out on its own.
  if (f->size + words > kFlushWords || f->spots >= kFlushSpots) {
    Bump(g_flush_room, 1);
    return false;
  }
  const uint64_t used = g_head.load(std::memory_order_relaxed) - g_tail.load(std::memory_order_acquire);
  if (used + 2 * (uint64_t(f->size) + words + 2 * (f->spots + 1)) + 4096 > kRingWords) {
    Bump(g_flush_room, 1);
    return false;
  }
  // The call's first placeholder: from here to the end of the call this thread holds g_marking (a submission from
  // another thread waits for every placeholder to be published), unless something was submitted since the call began.
  if (!f->marking) {
    g_marking.store(true, std::memory_order_seq_cst);
    if (g_wptr_writes.load(std::memory_order_seq_cst) != f->submissions) {
      g_marking.store(false, std::memory_order_release);
      f->active = false;  // the rest of this call writes PM4
      Bump(g_flush_original, 1);
      return false;
    }
    f->marking = true;
  }
  // The ops: one register run per run of dirty bits, the values from the shadow in host order.
  const uint32_t begin = f->size;
  uint32_t* out = f->ops + begin;
  uint32_t reg = register_base, src = shadow;
  for (uint64_t m = mask; m;) {
    const int lz = std::countl_zero(m);
    m <<= lz;
    reg += uint32_t(lz) * per_bit;
    src += uint32_t(lz) * per_bit * 4;
    const int run = std::countl_one(m);
    const uint32_t count = uint32_t(run) * per_bit;
    *out++ = kRun | (count << 8);
    *out++ = reg;
    SwapWords(out, Guest(base, src), count);
    out += count;
    reg += count;
    src += 4 * count;
    m = run == 64 ? 0 : m << run;
  }
  if (kind == FlushKind::kFetch) {  // the end packet of the fast path: registers 0x5000-0x5002 = 0, 0x25000, 0
    *out++ = kRun | (3u << 8);
    *out++ = 0x5000;
    *out++ = 0;
    *out++ = 0x00025000;
    *out++ = 0;
  }
  const uint32_t end_ops = uint32_t(out - f->ops);
  // The self-check: the first calls, and every call in verify mode.
  if (g_flush_checks_left.load(std::memory_order_relaxed) > 0 || sa2::enh::VerifyMode()) {
    g_flush_checks_left.store(g_flush_checks_left.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
    Bump(g_flush_checked, 1);
    sa2::enh::VerifyCheck();
    if (!FlushCheck(ctx, base, device, shadow, f->ops + begin, end_ops - begin, original, kind))
      return false;  // the original
  }
  // The placeholder, where the original's first word would have gone: the command buffer advances by three words.
  f->base = base;
  f->spot[f->spots] = {wptr + 4, begin, end_ops, false};
  uint8_t* const nop = Guest(base, wptr + 4);
  rex::memory::store_and_swap<uint32_t>(nop, kFlushHeader);
  rex::memory::store_and_swap<uint32_t>(nop + 4, kFlushMagic);
  rex::memory::store_and_swap<uint32_t>(nop + 8, f->spots);
  rex::memory::store_and_swap<uint32_t>(base + device + 48, wptr + 12);
  ++f->spots;
  f->size = end_ops;
  Bump(g_flush_calls[int(kind)], 1);
  Bump(g_flush_words, end_ops - begin);
  g_direct_flush.UseSampled();
  return true;
}

// sa2_cpu_direct_state, game thread, right after the original draw function: its state packets become a host record
// and one NOP packet (namespace direct above).
void EndDrawDirect(uint8_t* base, const DrawDirect& d) {
  using namespace direct;
  if (!d.device) return;
  // sa2_cpu_direct_flush: on every way out, the placeholders the record below did not take become records of their
  // own, and g_marking is released. Declared first, so it runs last (after HeldPair, which then sees the marking held).
  struct FlushEnd {
    ~FlushEnd() {
      FlushState* const f = t_flush;
      if (!f) return;
      if (f->spots) PublishSpots(*f);
      f->spots = 0;
      f->size = 0;
      if (f->marking) {
        g_marking.store(false, std::memory_order_release);
        f->marking = false;
      }
      f->active = false;
    }
  } flush_end;
  FlushState* const flush = t_flush && t_flush->active && t_flush->device == d.device ? t_flush : nullptr;
  const bool flush_held = flush && flush->marking;  // this thread holds g_marking since its first placeholder
  // sa2_cpu_direct_pair: the call's held-back Draw* record is claimed for the direct record below, or pushed to the
  // queue on every path that leaves the packets as they are, under g_marking (a submission waits for it, so the record
  // is in the queue before the ring can see this draw).
  struct HeldPair {
    PendingPair* p;
    ~HeldPair() {
      if (!p) return;
      const bool held = FlushHoldsMarking();
      if (!held) g_marking.store(true, std::memory_order_seq_cst);
      if (p->state.exchange(PendingPair::kNone, std::memory_order_acq_rel) == PendingPair::kPending) {
        PushRecord(std::move(p->record), p->start);
        Bump(g_pairs_pushed_late, 1);
      }
      if (!held) g_marking.store(false, std::memory_order_release);
    }
  } held{d.pair ? t_pending : nullptr};
  if (g_wptr_writes.load(std::memory_order_relaxed) != d.submissions) {  // the ring may be reading that segment
    Bump(g_skip_submitted, 1);
    return;
  }
  const uint32_t end = rex::memory::load_and_swap<uint32_t>(base + d.device + 48);  // the last word written
  if (rex::memory::load_and_swap<uint32_t>(base + d.device + 52) != d.limit ||
      rex::memory::load_and_swap<uint32_t>(base + d.device + 56) != d.soft_limit) {  // another segment
    Bump(g_skip_segment, 1);
    return;
  }
  if (end <= d.start || end - d.start > 0x10000u) {
    Bump(g_skip_no_draw, 1);
    return;
  }
  const uint32_t first = d.start + 4;  // the first word the draw function wrote
  // The record is decoded in place, in the ring of records: contiguous, never waiting for the ring thread (its records
  // may be in segments not submitted yet); when there is no room, this draw keeps its packets. A packet's record is at
  // most twice its words (a one-word type-3 packet becomes four), plus the record's two header words, plus the ops of
  // sa2_cpu_direct_flush's placeholders.
  const uint32_t written = (end - first) / 4 + 1;
  const uint32_t room = 2 + 2 * written + (held.p ? 1 + kPairWords : 0) + 4 + (flush ? flush->size : 0);  // + kRestore
  if (!g_ring) [[unlikely]] g_ring = std::make_unique<uint32_t[]>(kRingWords);
  constexpr uint64_t mask = kRingWords - 1;
  const uint64_t pos = g_head.load(std::memory_order_relaxed);
  const uint64_t tail = g_tail.load(std::memory_order_acquire);
  uint64_t start_record = pos;
  if ((pos & mask) + room > kRingWords) start_record = (pos | mask) + 1;  // wrap
  if (start_record + room - tail > kRingWords || room > kRingWords / 4) {
    Bump(g_skip_full, 1);
    return;
  }
  uint32_t* const rec = &g_ring[start_record & mask];
  uint32_t* out = rec + 2;
  uint32_t draw_at = 0;
  // sa2_cpu_direct_flush: the placeholders taken into this record (marked only once it is published).
  std::array<uint8_t, kFlushSpots> taken{};
  uint32_t taken_n = 0;
  // sa2_cpu_direct_state_refs: packets starting at word 2 or later keep their payload in place (the marker writes words
  // 0-2 only); the first ones are copied.
  const bool refs = g_direct_refs.active;
  for (uint32_t p = first; p <= end;) {
    const uint8_t* const hp = Guest(base, p);
    const uint32_t h = rex::memory::load_and_swap<uint32_t>(hp);
    const uint32_t type = h >> 30;
    const uint32_t off = (p - first) / 4;
    if (type == 0) {
      if (!h) { p += 4; continue; }  // Packet() skips a zero word
      const uint32_t count = ((h >> 16) & 0x3FFF) + 1;
      if (p + 4 * count > end) break;
      if (refs && off >= 2) {
        *out++ = ((h >> 15) & 1 ? kRunSingleRef : kRunRef) | (count << 8);
        *out++ = h & 0x7FFF;
        *out++ = off + 1;
      } else {
        *out++ = ((h >> 15) & 1 ? kRunSingle : kRun) | (count << 8);
        *out++ = h & 0x7FFF;
        SwapWords(out, hp + 4, count);
        out += count;
      }
      p += 4 * (1 + count);
    } else if (type == 1) {
      if (p + 8 > end) break;
      if (refs && off >= 2) {
        *out++ = kType1Ref;
        *out++ = h;
        *out++ = off + 1;
      } else {
        *out++ = kType1;
        *out++ = h;
        SwapWords(out, hp + 4, 2);
        out += 2;
      }
      p += 12;
    } else if (type == 2) {
      p += 4;  // filler
    } else {
      const uint32_t opcode = (h >> 8) & 0x7F, count = ((h >> 16) & 0x3FFF) + 1;
      if (IsDraw(opcode)) {
        draw_at = p;
        break;
      }
      if ((h & 1) || !Carried(opcode)) {
        g_unknown_opcode.store(opcode, std::memory_order_relaxed);
        Bump(g_skip_unknown, 1);
        return;
      }
      if (p + 4 * count > end) break;
      if (opcode == xenos::PM4_NOP) {
        // sa2_cpu_direct_flush: a placeholder of this call: its ops go into the record at its place.
        if (const FlushSpot* s = PlaceholderAt(base, p, h, flush)) {
          std::memcpy(out, flush->ops + s->begin, size_t(s->end - s->begin) * 4);
          out += s->end - s->begin;
          taken[taken_n++] = uint8_t(s - flush->spot);
          p += 12;
          continue;
        }
        // The marker covers the region's first three words: a NOP there may hold data another packet points at.
        if (p - first < 12) {
          Bump(g_skip_nop_head, 1);
          return;
        }
        p += 4 * (1 + count);
        continue;
      }
      if (refs && off >= 2) {
        *out++ = kType3Ref | (count << 8);
        *out++ = h;
        *out++ = off;
      } else {
        *out++ = kType3 | (count << 8);
        *out++ = h;
        *out++ = p & 0x1FFFFFFFu;  // the packet's physical address (Type3 reads packet_address_)
        std::memcpy(out, hp + 4, size_t(count) * 4);  // guest byte order: Type3 reads them
        out += count;
      }
      p += 4 * (1 + count);
    }
  }
  if (!draw_at) {
    Bump(g_skip_no_draw, 1);
    return;
  }
  const uint32_t region = (draw_at - first) / 4;
  if (region < 3 || region - 1 > 0x4000) {
    Bump(g_skip_small, 1);
    return;
  }
  uint32_t need = uint32_t(out - rec);
  // The decoder's own check (first records, every record in verify mode).
  if (g_checks_left.load(std::memory_order_relaxed) > 0 || sa2::enh::VerifyMode()) {
    g_checks_left.store(g_checks_left.load(std::memory_order_relaxed) - 1, std::memory_order_relaxed);
    Bump(g_checked, 1);
    sa2::enh::VerifyCheck();
    if (!SameAsWritten(base, first, draw_at, rec + 2, need - 2, flush)) {
      sa2::enh::VerifyDifference();
      g_failed.store(true, std::memory_order_relaxed);
      REXLOG_ERROR("[native] DIFFERENCE: a direct-state record does not re-encode to the packets Direct3D wrote "
                   "(region {:08X}, {} words); sa2_cpu_direct_state off from now on (records already made are "
                   "still applied)", first, region);
      return;
    }
  }
  // From here to the marker's last word no submission may be published (g_marking, WriteMmio): checked once more now.
  // With sa2_cpu_direct_flush's marking already held since the first placeholder, another thread's submission waits
  // anyway, and this thread's own ones are counted in program order.
  if (!flush_held) g_marking.store(true, std::memory_order_seq_cst);
  if (g_wptr_writes.load(flush_held ? std::memory_order_relaxed : std::memory_order_seq_cst) != d.submissions) {
    if (!flush_held) g_marking.store(false, std::memory_order_release);
    Bump(g_skip_submitted, 1);
    return;  // the words decoded into the ring are not published: the next record overwrites them
  }
  // The held-back Draw* record joins the state record (no submission happened during the call, so none pushed it).
  if (held.p) {
    if (held.p->state.exchange(PendingPair::kNone, std::memory_order_acq_rel) == PendingPair::kPending) {
      const DrawRecord& r = held.p->record;
      uint32_t* q = rec + need;
      q[0] = kPair | (kPairWords << 8);
      q[1] = draw_at & 0x1FFFFFFFu;  // the draw packet it belongs to (physical, as the ring's packet_address_)
      q[2] = r.type;
      q[3] = r.r5;
      q[4] = r.r6;
      q[5] = r.r7;
      q[6] = r.vs;
      q[7] = r.ps;
      q[8] = r.start;
      q[9] = uint32_t(r.snapshot);
      q[10] = r.device;
      need += 1 + kPairWords;
      Bump(g_pairs_carried, 1);
    }
    held.p = nullptr;
  }
  if (g_direct_restore.active) {  // the three words the marker is about to replace, in guest order
    uint32_t* q = rec + need;
    q[0] = kRestore | (3u << 8);
    std::memcpy(q + 1, Guest(base, first), 12);
    need += 4;
  }
  if (start_record != pos) g_ring[pos & mask] = kPad;
  uint32_t seq = uint32_t(++g_seq);
  if (seq == kPad) seq = uint32_t(++g_seq);
  rec[0] = seq;
  rec[1] = need;
  g_head.store(start_record + need, std::memory_order_release);
  // The marker: one NOP covering the region; the old packets stay inside it as payload, the draw packet after it.
  uint8_t* marker = Guest(base, first);
  rex::memory::store_and_swap<uint32_t>(marker, 0xC0000000u | ((region - 2) << 16) | (xenos::PM4_NOP << 8));
  rex::memory::store_and_swap<uint32_t>(marker + 4, kMagic);
  rex::memory::store_and_swap<uint32_t>(marker + 8, seq);
  for (uint32_t i = 0; i < taken_n; ++i) flush->spot[taken[i]].taken = true;  // in this record: not published again
  Bump(g_flush_taken, taken_n);
  if (!flush_held) g_marking.store(false, std::memory_order_release);
  g_direct_state.UseSampled();
  Bump(g_converted, 1);
  Bump(g_words_covered, region);
  Bump(g_record_words, need);
}

bool Enabled() {
  // Horizon does not launch an NRO with a useful process environment: the toml decides. Native by default;
  // sa2_renderer_native = false + gpu_plugin = "xenos" runs ReXGlue's Xenos emulation instead.
  return REXCVAR_GET(sa2_renderer_native);
}

std::unique_ptr<rex::system::IGraphicsSystem> CreateGraphicsSystem() {
  return std::make_unique<NativeGraphicsSystem>();
}

}  // namespace me::native
