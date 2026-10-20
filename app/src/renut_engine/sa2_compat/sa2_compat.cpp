// sa2-nx (2026-10-20): what Renut's renderer (through banjo-nx) expects from banjo-nx's SDK and generated code, for
// SA2's. Whether the generated code marks the store cards (g_banjo_store_cards): src/cpu/sa2_hot_guest.cpp, which sees
// the generated header.
#include <cstdint>

#include <rex/cvar.h>

// mk9-nx (2026-10-20): MK9's generated code marks no store cards (sa2-nx: src/cpu/sa2_hot_guest.cpp decides).
extern "C" const bool g_banjo_store_cards = false;

// The store cards' host-side counters (banjo-nx's runtime, PC diagnostics): none.
extern "C" uint64_t RexStoreCardsHostBytes(int) { return 0; }

// Keys of banjo-nx's own native renderer (its src/me) that Renut's tuning reads too: defined here with banjo-nx's
// defaults and descriptions.
REXCVAR_DEFINE_INT32(banjo_background_compile_core, 2, "Banjo",
                     "Native renderer: the background compile threads (pipeline prewarm, banjo_gpu_bool_spec's "
                     "background compiles, the shader statistics copies, the shader preload) are pinned to this core, "
                     "so together they never take more than one core from the game (their priority cannot go below "
                     "the game threads' 0x3B on the console). -1 = not pinned (any core, as before)")
    .range(-1, 2)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(banjo_gpu_robustness_off, true, "Banjo",
                    "GPU (non-CPU enhancement): the game's pipelines are created with buffer and image robustness "
                    "disabled (VK_EXT_pipeline_robustness; vertex input as the device has it), so NVK drops its two "
                    "descriptor index clamps from every texture handle (nvk_nir_lower_descriptors.c, "
                    "clamp_desc_array_bounds). The renderer's indices are in range by construction. A pipeline key "
                    "bit (spec2 bit 8, no shader reads it) keeps the two forms apart. In-run A/B target 53")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(banjo_gpu_bool_spec, true, "Banjo",
                    "GPU (non-CPU enhancement): a pipeline is created with the values of the bool constants its shaders "
                    "test (specialization constants 1000 + N, BANJO_BOOL in shader_common.h), so the code behind a "
                    "bool that is off goes and the ones that are on lose their branch. Nuts & Bolts' material shaders "
                    "test 14 (twelve lights, two output paths): PS536 1,980 SPIR-V instructions unspecialised, 879 "
                    "with every bool off. The values are in the pipeline key (PipelineKey::bools). Exact: the value is "
                    "the one the shader reads at run time (verify compares). A new combination draws with the "
                    "generic pipeline while its own compiles (banjo_gpu_bool_spec_async). Needs a shader package from "
                    "2026-10-15 on. In-run A/B target 31")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(banjo_gpu_tess_grid, true, "Banjo",
                    "GPU (non-CPU enhancement): tessellated triangle patches (Nuts & Bolts' water: 33 patches at 16 "
                    "segments, 25,344 vertex invocations a draw, VS n414 1.8-2.9 ms a frame on the console) are drawn "
                    "indexed, one vertex invocation per grid point ((N + 1)(N + 2) / 2 a patch instead of 3 N^2; each "
                    "point is shared by up to six triangles). The index list, built once per patch count, level and "
                    "winding, gives each corner of the old triangle list the grid point it decodes to, in the same "
                    "order (same triangles, same corners, same first vertex). Exact: the domain of a point is "
                    "computed from the same (a, b) with the same operations (banjoTessDomain, bit 30 of the level "
                    "word). Needs a shader package from 2026-10-16 on. In-run A/B target 39")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(banjo_gpu_blend_skip, true, "Banjo",
                    "GPU (non-CPU enhancement, sa2-nx): transparent pixels whose blend would leave the RGBA8 target "
                    "unchanged are discarded before the blend (no depth/stencil write, op ADD, destination factor 1 "
                    "once the source is 0; a NaN output is never discarded)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);


// The same for keys Renut's tuning reads on the Switch only (its NVK paths: banjo-nx's me renderer defines them).
REXCVAR_DEFINE_BOOL(banjo_gpu_nvk_draw_contract, true, "Banjo",
                    "GPU / driver (non-CPU enhancement, sa2-nx): NVK's per-draw contract is also looked up under the "
                    "name the linked Mesa tree exports (nvk_switch_dibujo), so its per-draw paths (draw emission in one "
                    "go, only dirty cbuf slots and dynamic state groups, prefetch hints; each self-checked in NVK and "
                    "switched off by its guard on a difference) are requested. false = only the name nvk_switch_draw, "
                    "as before (never found: nothing requested)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(banjo_prewarm_threads, 1, "Banjo",
                     "Native renderer: threads that prewarm the pipeline list (masseffect_native_pipelines_prewarm), "
                     "all at the lowest priority. 1 = as before")
    .range(1, 3)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_INT32(banjo_ring_core, 1, "Banjo",
                     "Native renderer: the ring thread is pinned to this core (its affinity mask reduced to it). -1 = "
                     "the process's whole mask, as before. In-run A/B target 41 switches between the pin and the "
                     "whole mask")
    .range(-1, 2)
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

REXCVAR_DEFINE_BOOL(masseffect_native_nvk_cbufs, true, "MASSEFFECT",
                    "Native renderer: NVK only looks at the cbuf slots that can get dirty "
                    "and skips the cbuf flush when nothing is dirty (self-checked in NVK). false = as usual");

REXCVAR_DEFINE_BOOL(masseffect_native_nvk_dynamic, true, "MASSEFFECT",
                    "Native renderer: NVK only emits the dynamic state groups with "
                    "dirty bits (self-checked in NVK). false = as usual");

REXCVAR_DEFINE_BOOL(masseffect_native_nvk_emission, true, "MASSEFFECT",
                    "Native renderer: NVK writes each draw command in one go, with the "
                    "same bytes (self-checked in NVK). false = as usual");

REXCVAR_DEFINE_INT32(masseffect_native_nvk_measure, 64, "MASSEFFECT",
                     "Native renderer: measures by parts inside NVK on 1 in N calls "
                     "(power of 2; 'C6 NVK parts' report every 10 s). 0 = no measuring")
    .range(0, 4096);

REXCVAR_DEFINE_BOOL(masseffect_native_nvk_preload, true, "MASSEFFECT",
                    "Native renderer: cache hints (PRFM) for the pipeline and shaders before "
                    "they are used, and the pipeline handed to NVK after PipelineFor. Changes nothing visible. false = no hints");

REXCVAR_DEFINE_BOOL(masseffect_native_set4_differences, true, "MASSEFFECT",
                    "Native renderer: descriptor set 4 (UBO constants) by differences in "
                    "NVK: only the descriptors that change are sent and rebound. Same result (self-checked "
                    "in NVK). Needs the Mesa build with patch_nvk_set4; false = as usual");

