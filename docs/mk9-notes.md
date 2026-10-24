# Mortal Kombat Komplete Edition (Xbox 360): analysis notes

Facts about the game that the port relies on, each with where it came from. Addresses are guest addresses of
`default.xex` unless stated otherwise. The port starts from usf4-nx (Ultra Street Fighter IV, same ReXGlue method:
`/root/tmp/claudecode/switch-claude/xbox/360-recomp/usf4-nx`, its `docs/usf4-notes.md`), whose SDK fork, tools,
native renderer and Switch build are reused unchanged unless noted here.

## The disc

- `Mortal Kombat Komplete Edition [RF].iso`: XGD2 (game partition at `0x0FD90000`), 1,265 files, 7.27 GB
  (`tools/extract_iso.py --list`). Besides `$SystemUpdate/` (dashboard update, unused):
  - `default.xex` (21.3 MB, **lower case**: the app looks for both spellings), the only code module.
  - Unreal Engine 3 layout: `Asset/` 1,184 `.xxx` packages (4.0 GB: 193 `SND_*`, 179 `NIS_*` cinematics, 141 `ui_*`,
    92 `NPC_*`, 76 `CHAR_*` characters and their palettes, 69 `LoadScreen_*`, 34 `BGND_*` stages, per-character
    `_CharIntro`, `_Fatality1/2`, `_Ladder`, `_Victory`; `Core.xxx`, `Engine.xxx`, `FmodAudio.xxx`), `Config/Coalesced.ini`,
    `Localization/Coalesced.{eng,fra,ger,int,ita,por,spa}`.
  - `Movies/`: 66 `.wmv` files, 2.8 GB (story-mode cinematics `mk001`...`mk127`, `NRIntroLogo.wmv`, `WBIntroLogo.wmv`).
  - `AvatarAssetPack`, `nxeart`, `Splash/Splash.bmp`, `Xbox360TOC.txt`.

## default.xex

| | |
|---|---|
| Title ID / media ID | `575207FD` / `6579DAFA`, version 0.0.0.8, disc 1 of 1 |
| Image base / size | `0x82000000` / `0x01560000` (22.4 MB) |
| Entry point | `0x8236E580` |
| Encryption / compression | retail AES / basic (no LZX) |
| XDK | 2.0.20353 (`xdk-main-nov10` build paths in the image) |
| Sections | `.rdata` `0x82000400`, `.pdata` `0x822B3600` (0xA2500), **`.text` `0x82360000`-`0x83376864`** (16.1 MB), `.data` `0x83380000`, `.XBMOVIE`, `.idata`, `.XBLD`, `.reloc` |

The decrypted image for the analysis tools comes from `tools/xex_image.py` (new; usf4-nx dumped it from a running
game with `USF4_DUMP_IMAGE`): `python3 tools/xex_image.py assets/game_root/default.xex out/default_image.bin`.

Middleware (strings in the image): Unreal Engine 3 (NetherRealm's fork: `AMK9GamePlayerController`,
`UMKSeqAct_*`), **FMOD** (FMOD Ex + event system: `fmod_eventreverb.cpp`, `AFmodAmbientSound`), the XDK's WMV decoder
with its GPU shaders (`xdk-main-nov10\core\private\del\codecs\wmv\...\Shader_Deblock*`, as in USF4), the Avatar
library (`xtl\avatar\lib`), Kinect stubs (`AVAKinect_Dummy`).

## Code generation

First run (2026-10-20, rexglue 0.10.0 of the sa2-nx fork as built for usf4-nx, Linux, clang 19): 9 "target not in
any function" errors, all direct tail calls (`b`) to code without a `.pdata` entry, declared in `app/overrides.toml`
with their evidence: six function starts right after an unconditional terminator (four reached from `this`-adjustor
thunks, `addi r3,r3,N / b`), and three shared `li r3,0 / blr` return blocks that switch-case fragments of property
getters/setters branch back to (declared with `size = 0x8`). Second run: 539 files, 9 min.

Jump tables (`tools/switch_check.py`, code range `0x82360000:0x83376864`): 1,604 `bctr` (1,214 indirect, 277
absolute, 56 halfword, 57 byte tables); generated 389 switches and 857 indirect jumps; **0 truncated, 0 wrong
targets, 0 missed**. 28 tables have no bound in the code (candidates if a "Switch case out of range" appears; full list
in `out/switch_check.txt`). 23 odd shapes are indirect calls through function-pointer tables in `.data` (for example
17 at `0x8270A584`-`0x8270A878`, 4 at `0x828BDEE4`-`0x828BDFF0`): their targets are the pointer sweep's job.

Pointer-reached functions (`tools/pointer_fixpoint.py`, 2026-10-20): pass 1 over 37,227 unclaimed gaps found 253
candidates and accepted 228 (written to `app/overrides.toml` with where each pointer is stored); rejected: 18 tail
branches into a function body (funclets/chunks), 3 fragments that read an undefined register at entry, 2 with no
code, 1 extending beyond its gap, 1 without a terminator. Pass 2: no new candidates, fixpoint.
`tools/pointer_audit.py` afterwards: 98,790 registered functions. Pointers to non-starts, not acted on (watch for
them in boot crashes): 134 "vtable-like" ones, all right after a function epilogue (`addi r1,r31,N / b` to a
`restgprlr` helper) and stored in `.rdata` near EH data: C++ unwind funclets, not virtual functions; 14 built in code
(SEH handler bodies `lwz r3,N(r31)` after a branch, resume points after `bctr`); the rest are EH state maps (50,537),
jump tables (397), SEH scopes (173) and other data (595). Full list: `out/pointer_audit.txt`.

### setjmp / longjmp

Neither has a `.pdata` entry; found by shape (stores of f14-f31 into r3's buffer and the mirrored loads), the same
CRT code as USF4's:

- `setjmp` `0x82FE1340`: `lis r4,0x834D / lwz r0,-2400(r4)` (override pointer at `0x834CF6A0`), `bnectr`; else
  `mflr r0 / mfcr r4`, stfd f14-f31 at 0..136(r3), std r13-r31 at 152..296(r3), stvx128 of the vector registers from
  320(r3) on.
- `longjmp` `0x82FE1020` (registered as a function by discovery): `mr r7,r3`; with a non-zero 312(r3) it takes the
  unwinding path at `0x82FE130C`; else `bl 0x82B3A250`, lfd f14-f31 from 0..136(r7), ..., `lwz r5,308(r7) / mtlr r5 /
  ld r1,144(r7) / mtcrf from 304(r7) / mr r3,r6 / blr`.
- `setjmp` is called from 28 places in 7 generated files, so they are live.

## Direct3D

| Function | MK9 | Evidence |
|---|---|---|
| draw-time shader load | `sub_82BFEEE8` | the only Direct3D writer of `IM_LOAD` (`0xC0012700` at `0x82BFEF4C`, `0xC0012701` at `0x82BFF65C`/`0x82BFF6E4`); reads the bound **VS at device+12944 (`0x3290`), PS at device+12940 (`0x328C`)** (`lwz r30,12944(r31)` / `lwz r29,12940(r31)` at `0x82BFEF34`), then the VS object's +896 and +872: the same layout as USF4's (13072/13068 there) |
| Draw (auto index), candidates | `sub_82BEFEF8`, `sub_82BF09E0`, `sub_82BF1270` | each builds predicated `DRAW_INDX` with one data word (`0xC0012201`) at two sites. Which is DrawVertices / DrawVerticesUP: to check (USF4: the UP one calls the inline-data allocator) |
| Draw (indexed), candidates | `sub_82BF0400`, `sub_82BF0DF8`, `sub_82BF1710` | predicated `DRAW_INDX` with three data words (`0xC0032201`), two sites each; `sub_82BF5F90` has one more (with an `IM_LOAD_IMMEDIATE`, `0xC0192B00`: a Direct3D-internal draw, e.g. a clear or resolve) |
| common Draw* flush | | all six draw candidates call the same sequence: `sub_82BFDD28` (8 times), `sub_82BFE0C0` (twice), the shader load `sub_82BFEEE8`, `sub_82BFDF68`, `sub_82BFDDE8`, `sub_82BFD990`, `sub_82BFCD60`, `sub_82BFBE80` (`sub_82BF0400` also `sub_82BFBB78` twice). Direct call sites: `sub_82BEFEF8` 2, `sub_82BF09E0` 2, `sub_82BF1270` 0 (reached by pointer), `sub_82BF0400` 1, `sub_82BF0DF8` 8, `sub_82BF1710` 1. All six take the device in r3; `app/src/me/mk9_me_hooks.cpp` hooks all six with `NoteDrawCall` |
| shader container parse helper | `sub_82BE9B48` | 20 instructions, r3 = container, fills r4's descriptor {blob, microcode, size} (usf4-nx's `sub_821E5930`). Diagnostic run (out/diag, every function in `0x82BE0000`-`0x82C40000` checked for a `0x102A110x` container in r3-r5, 130 s to a fight): ~4,500 calls with a container (378 VS, 4,123 PS so far), nearly all distinct. Its callers `sub_826C07C0` / `sub_826C0978` are UE3's shader creation, with Direct3D's CreateVertexShader/CreatePixelShader **inlined** (`sub_82BEB838` / `sub_82BEB610` take VS / PS containers in r3 but ran once each: Direct3D's own startup shaders). Hooked for the shader dump (`mk9_dump_shaders`) |
| per-draw shader patching | `sub_82BFEC60`, `sub_82BFCD60` | r4 = container on every draw (484,501 calls in 130 s): Direct3D-internal, not hooked |
| Direct3D device | `0x40012980` | r3 of the shader load in a run (logged once by `mk9_me_hooks.cpp`); command-buffer pointer at device+48 as in usf4-nx (`lwz r3,48(r31)` at `0x82BFEF80`) |
| pixel-shader container checks | `sub_82BED4D0`, `sub_82C0B7E8`, `sub_82C13B48`, `sub_82C36AF0`, `sub_82CA06D8`, `sub_831AEE20`, `sub_8324BE50` | build the pixel container magic `0x102A1100`; CreatePixelShader is among them (USF4: 60-instruction function, r3 = container) |
| GPU wait iteration | `sub_82BF5680` | usf4-nx's `sub_821F54E0` shape: `cctpl`, a 4 x 8 `db16cyc` delay loop, then the hung bit `lbz 11069(device)`, the read-pointer word `lwz 11024(device)`, 5000-tick timeout. 216 bytes, FNV-1a 0x853791F298FEE8ED. `app/src/mk9_gpu_wait.cpp` (usf4-nx's `mk9_cpu_gpu_wait_sleep` enhancement) makes it sleep until the ring progresses: PC run, ~7-8 s asleep per 10 s on the waiting thread, no change in the image. `sub_82BF61E0` (864 bytes, 8 `db16cyc`) is a second wait, not hooked |
| `D3DDevice_Swap` | `sub_82BF6D58` | the only caller of `VdSwap` (import thunk `0x83376194`); called from `sub_82BF73E0`, `sub_826B57F8`, `sub_82BF1D18`, `sub_82C0A640`. Hooked by `app/src/mk9_frame_counter.cpp` for the `[fps]` line |

## App changes from usf4-nx

- Project renamed (`mk9_manifest.toml`, target `mk9`, `src/mk9_*`, cvars `mk9_snapshot_s`, `mk9_input_script`, ...).
  The native renderer (`src/me`, `src/cpu`) keeps its `sa2_*`/`usf4_*` file and cvar names so fixes from those ports
  can be taken over.
- **Not built** (USF4 addresses; in MK9 the same addresses are unrelated functions, so a `REX_HOOK_RAW(sub_X)` would
  replace the wrong code): `usf4_me_hooks.cpp` (shader load, Draw*, shader create), `usf4_tiling.cpp` (one tile
  instead of predicated tiling), `usf4_gpu_wait.cpp`, `usf4_job_wait.cpp`, `usf4_job_pop_yield.cpp`,
  `usf4_native_copy.cpp`. Each needs MK9's own function found first (stage 4/5).
- `app/mk9_switch_defaults.toml` keeps only usf4-nx's game-independent renderer settings.

## First boots on the PC (stage 3, emulated GPU)

Linux, hidden Xvfb display, `tools/linux/boot_test.sh`, 2026-10-20. **No code fixes were needed after stage 2**:

1. 60 s, Intel HD 530: no crash. Autosave notice, `WBIntroLogo.wmv` (the XDK's own WMV decoder, as in USF4),
   middleware logos (Beast, Dolby Digital, Turtle, PhysX, Unreal, Agora Games), title screen with PRESS START.
   [fps] 33-46 in boot and title.
2. Scripted gamepad (`mk9_input_script`), GTX 960M (`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json`),
   170 s: `32 START;38 A;44 A;52 A;58 A;64 A;70 A;76 A;84 A` goes title -> main menu ("1 PLAYER": Fight, Story Mode,
   Challenge Tower, Training, Xbox LIVE, Extras, Options) -> Fight -> Ladder -> character select (Scorpion, 32
   portraits) -> difficulty (Medium) -> the ladder tower -> VS screen (Scorpion vs Freddy Krueger: the Komplete
   Edition's DLC fighters are on the disc) -> the stage intro -> **a fight with HUD** (health bars, super meters), no
   crash.

Seen in the log, harmless so far: `UPDATE:` paths fail (no title update; the game falls back to the disc);
`D:\Config\Xe-MK9Engine.ini` not found (UE3 looks for a per-platform override first, then uses `Coalesced.ini`);
`KeSaveFloatingPointState` / `KeRestoreFloatingPointState` stubs called ~40 times a second by one thread (to check:
the SDK stub returns success without saving anything; the PPC FPSCR is per thread in the recompiled code anyway);
`XamVoiceSetMicArrayIdleUsers` stub.

Audio: `[audio] SDL in 10 s: 1876 frames with data and 0 of silence` throughout; XMA contexts decoded on the calling
thread (FMOD's XMA streams).

Frame rate (emulated GPU, RelWithDebInfo, classic codegen): GTX 960M 25-30 fps in menus, **13-19 fps in the fight**
(USF4: 30 at the same stage); Intel HD 530 ~40 fps on the title. As with USF4 the emulated GPU is the limit on this
machine; the native renderer (stage 4) replaces it.

Not verified yet: saving (no save prompt in 170 s and nothing written to `~/.local/share/mk9/<xuid>/`), the
story-mode movies, other modes, a full match to its end.

## Native renderer (stage 4)

`app/src/me/mk9_me_hooks.cpp` (2026-10-20): usf4-nx's hook file with MK9's addresses (table above): device shader
offsets 12944/12940, the shader load, the six draws, and the shader dump on the container parse helper.

Shader dump (`mk9_dump_shaders = true`, 200 s through boot, menus and a fight): **5,075 containers** (330 VS, 4,745
PS, 22 MB; USF4 had 865). UE3 compiles one pixel shader per material and light/fog permutation, so a fight on another
stage or with other characters will add more: the package needs the containers from the game files too (UE3 keeps
them in the shader caches inside the `.xxx` packages: `RefShaderCache`/`LocalShaderCache` objects), the counterpart
of usf4-nx's `tools/extract_game_shaders.py`.

### First native runs (2026-10-20, GTX 960M, 170 s scripted to a fight)

1. Package from the runtime dump only (4,998 shaders): no crash; menus, character select and the ladder drawn right,
   logo backgrounds white, 3D scenes mostly missing. `[native]` counters: 169 shader objects seen, 48 identified;
   ~84k "immediate" loads (Direct3D's own shaders) and 75 VS / 34 PS objects "not in the library": Direct3D rewrites
   a container when it binds it (vertex declaration patching), so the bound object's container differs from the one
   the parse helper saw.
2. Round 1 (usf4-nx's loop): `sa2_shader_misses = misses` keeps the rewritten containers (`missing/`, 83) and the raw
   ring microcode (`discovery/`, 61); `tools/wrap_discovered.py` wrapped 2 (Direct3D's own, no container anywhere) and
   skipped 59 that a real container (mostly in `missing/`) has. With those in the package: immediate loads 0, 103
   addresses / 75 codes mapped, and **the whole flow draws**: logos, menus, VS screen, the opponent's intro, the
   fight with HUD (Scorpion vs Nightwolf, "2 HITS 7% Damage"). 22-28 fps in the fight on this laptop GPU (emulated:
   13-19).

### Shaders from the game files

`shaders/tools/ue3_shader_scan` (masseffect-nx's UE3 package scanner) reads MK9's packages after one change: MK9's
compressed chunks start with the tag and the block size (`9E2A83C1 00020000 csize usize`, then (compressed,
uncompressed) per 128 KB block) where Mass Effect wrote the tag twice; the blocks are LZO1X as in Mass Effect. The 13
fully compressed packages (Engine.xxx, ...) begin with such a chunk at offset 0; the other 1,171 have a plain summary
(file version 472, licensee 46) and compressed chunks after it.

All 1,184 packages in 59 s: 6,381 chunks, **0 bad blocks**, 73,170 containers, **21,093 distinct** (1,352 VS, 19,741
PS) covering 19,829 distinct microcodes; **every one of the 4,531 microcodes of the runtime dump is among them**. So
the shader package can be made from the game files alone, as usf4-nx's browser installer does for USF4. What the
scan cannot have are the containers Direct3D rewrites at bind time (40 of the 83 `missing/` ones are in the scan by
microcode): those still come from `sa2_shader_misses` runs (or repairs of the variants).

### Full shader package (2026-10-20)

- `ue3_shader_scan` also scanned the raw (still compressed) file of every package and took a container magic that sits
  among LZO literals, with compressed bytes after it, for a container: 2,088 junk entries, all crashing XenosRecomp,
  none ever loaded by the game. The raw file is now scanned only for packages without compressed chunks: 19,005
  distinct containers, exactly the 2,088 fewer.
- XenosRecomp: 191 vertex shaders (77 of them created by the game in a run) list the same usage/index at two fetch
  addresses (their inputs are fetched again in a second code path); the translator declared each input twice and DXC
  failed (`redefinition of parameter 'iPosition0'`). Now one input per usage/index, every fetch address mapped to it
  (`shaders/XenosRecomp/shader_recompiler.cpp`). All 191 compile.
- `tools/build_shader_library.sh`: `SHADER_SPIRV_CACHE=<folder>` reuses compiled SPIR-V by container hash (a round
  takes ~6 min instead of ~45 for 24k containers).
- Package = runtime dump + scan + rounds 1-2 (`missing/` + wrapped): **19,071 shaders, 777 MB** (+33 MB index; the
  library keeps only the index in memory and reads SPIR-V lazily). Run to a fight (Scorpion vs Quan Chi, street
  stage): **unidentified loads: VS 0, PS 0, immediate 0**; 233 objects seen; no new discovery; 7 more rewritten
  containers in `missing/`. The street stage, both fighters, ROUND 1 and the HUD are drawn right. 21-22 fps on the GTX
  960M (GPU-bound laptop; usf4-nx's PC fights on this machine were also ~10-13 fps GPU-bound with the native
  renderer, while its console build reached 60).

## Switch (stage 5)

Built as usf4-nx's (docs/building.md, "Switch"): Docker `devkitpro/devkita64`, nfsmw-nx's Mesa 26.2.2 Horizon SDK
with libvulkan.a repacked (copied from usf4-nx's `out/tools/mesa-26.2.2-repacked`), glslangValidator built in the
image (`out/glslang-dkp`, copied).

- **0.0.1 (2026-10-20)**, classic codegen, **no LTO** (as usf4-nx's first NRO): 876 build steps, ~2 h 20 min on this
  machine (shared with PC tests); `out/nx/mk9.nro` 102 MB, ELF archived as `out/nx/mk9-0.0.1.elf` for symbolizing
  crash reports. Includes the native renderer hooks, the GPU-wait sleep, the FP-state stubs and the full shader
  package. Settings compiled in: `app/mk9_switch_defaults.toml` (native renderer, **1280x720 in both modes**, log to
  the SD card). SD-card zip: `../mk9-nx-switch-0.0.1.zip` (NRO, `sa2_shaders.mesp` + `.idx`, README; repacked with the filtered
  773 MB package, see below).
  **Not run on a console yet.**
- LTO build of the same sources: `out/nx-lto/mk9.nro` (123 MB, ELF `out/nx-lto/mk9-0.0.1-lto.elf`), zipped NRO only as
  `../mk9-nx-switch-0.0.1-lto-nro-only.zip` (drop-in replacement for the NRO of the full zip). Not run on a console.

What to look at in the first console run: whether it boots (memory: MK9's UE3 uses most of the 360's 512 MB, plus
the renderer and the 33 MB shader index on the host side), the `[fps]` lines in `logs/`, crash reports
(`atmosphere/crash_reports`, symbolize with `out/nx/mk9-0.0.1.elf`).

### Stage geometry broken by package duplicates; particles (2026-10-20)

A 450 s run through a whole match (Beginner, scripted attacks: the match is won, "JOHNNY CAGE WINS", the ladder goes
on) showed the rooftop, desert and Pit stages drawn as huge single-colored triangles, while the street stage had
been fine. Not the vertex cache (`usf4_vertex_cache = false`) nor any switchable enhancement (both masters off):
the package. With the dump-only package of round 1 the Pit stage had been right. The scan adds, for 5,305 runtime
shaders, the package copy of the same microcode with the header the game has not rewritten yet (usf4-nx's notes warn
about this for USF4's EMZ shaders), and identification by microcode can take that copy. `tools/filter_scanned_shaders.py`
keeps only scanned containers whose microcode no runtime container has (13,700 of 19,005); package 18,880 shaders,
773 MB. Same run: stages and fighters right (desert vs Noob Saibot, Soul Chamber vs Sonya), `unidentified loads: 0`.

Still wrong: **particles** (hit sparks, dust, blood) come out as long colored streaks and full-screen planes, with
the enhancements and the vertex cache off too. Not investigated yet; UE3's Xbox sprite particles expand each quad in
the vertex shader from per-particle data (vertex fetches with computed indices), a path USF4 did not use. Next:
trace a frame with sparks (`banjo_trace_frame_s` + `usf4_trace_until_s`), look at the particle draws' primitive type,
vertex fetches and the translated VS.

Bisection so far (same 260-300 s scripted match, GTX 960M; the ladder's opponents and stages differ between runs):
- `usf4_vertex_cache = false`: still broken. Both enhancement masters off: still broken.
- `banjo_native_skip_indexed = true` (skip draws whose VS fetches by a computed index): still broken, and the
  fighters are still drawn, so MK9's skinning does not use computed-index fetches.
- `sa2_native_dedupe_vertices = false`, `sa2_native_dedupe_sample = false`, `sa2_native_vertices_base_zero = false`:
  **much worse** (whole stages and fighters in garbage colors). The slower the ring thread, the worse the image:
  this points at a race between the game rewriting dynamic vertex data and the ring thread reading it (the copies
  are synchronous on the ring: `sa2_native_uploads_thread` is off). Mass Effect had the same class of bug for its
  transient UI buffers (comment at `sa2_native_uploads_thread`). Candidates: Direct3D's inline-data ring (the UP
  draws' vertices, reclaimed by the ring read pointer the renderer writes back) or a fence reported before the draws
  it covers are consumed. To check: which of the six draw hooks the particle draws come through, and when the
  renderer writes back the read pointer (`device+11024`) relative to consuming the indirect buffers.
- The emulated GPU draws the same effects correctly.

## The fights' triangles on the Switch: the render thread read particle data before the game had written it (2026-10-24)

(The diagnostics named here, `mk9_fetch_late_probe` and the other test switches, are not in the tree: they are in
`out/experiments_2026-10-24.patch`, which applies to commit c0c6f6b.) Found overnight on the console with a fixed scene (rig scripts: Training > Practice, arena The Dead Pool, Scorpion vs
Sub-Zero, seeded inputs; the ladder's random stages had made every earlier A/B meaningless).

- The worst shader on that stage, `63A3070E606FB209`, draws instanced mesh particles: `c18.x` is the mesh's vertex
  count (772, 276, 249), fetch 95 the mesh, fetch 92 80 bytes per instance, and the instance is
  `floor((vertex id + 0.5) / c18.x)`. Every captured bad vertex had a vertex id (an index value) whose instance lay
  beyond the instances in fetch 92's copy, so it read whatever followed it in the upload ring.
- The particle vertex and index buffers are the game's own dynamic memory (physical 0x0D03xxxx, 0x1FBxxxxx-0x1FCxxxxx),
  not Direct3D's inline-data ring (`[mk9] inline-data ring` diagnostics: 2 quarter moves a run, never waits).
- `mk9_fetch_late_probe` (diagnostic): of 253,000 window-shader buffers hashed at their draw and at every later
  KickOff and Swap, 46 % changed after the draw call, all of them by the frame's Swap, none afterwards, and no later
  draw reused the same bytes. MK9 fills a frame's particle buffers after issuing the draws, until the frame ends; the
  Xbox 360's GPU, a frame behind, reads them complete.
- Renut's render thread records draws as soon as it catches up with the game thread: it read the indices (and vertex
  copies) of some draws before the game had written them, so stale indices from an earlier, larger batch of particles.
- Not the cause: the frames the game may run ahead (`mk9_max_frames_ahead = 0`), copying the fetch data at the Swap only
  (`mk9_fetch_copy_at_frame_end`: indices are read at record time anyway; worse), the index buffer header moving after
  the draw (`mk9_index_address_at_draw`: 8 of 196,738 draws).
- Fix: `mk9_record_after_swap` (backend/render_thread.cpp): the render thread takes a frame's work only once the game
  has queued its Swap. A thread that waits for the render thread (fence waits, drains, IsFencePending polls) opens the
  gate at once; a 500 ms watchdog opens it otherwise (50 ms at first, which fired in one fight frame in three: they run 55 ms). On by default. The Dead Pool: 0 bad vertices in 153,469 window-shader draws
  (3,000-5,000 per 1,000 without), every screenshot clean, fire and sparks drawn.

- Console, after the fix (2026-10-24 morning, user's clocks one step lower; fps from the game's 5 s log lines, 90 s fights):
  The Dead Pool practice 18.3 avg (fix on, 500 ms watchdog) vs 18.2-18.4 off; Courtyard (Night) practice 10.9 on, 10.9
  off, 11.1 with 0.1.5 (game-thread bound: GPU 13 ms, render thread 6 ms, ~90 ms frames); an Arcade Ladder fight
  (Cage vs Jax) ~35. The game threads never wait for the render thread in a fight (0 fence waits, 0 ms at the
  Swap), gate on or off. Practice's red limbs are Practice Options > Collision Region (on by default), not a bug.

## Optimised codegen: the fights from 11 to 47-54 fps on the Switch (2026-10-24)

Console profile (switch_profile_stacks = true, Courtyard (Night) practice, 10.9 fps): the guest main thread used 28 %
of its core; a quarter of its samples waited in pthread_mutex_lock on the codegen's global lock (mtmsrd r13 ->
REX_ENTER_GLOBAL_LOCK, 669 sites in the classic code), against UE3's worker threads doing the same, and 15 % in
NtWaitForSingleObject. The same wall as USF4 0.5.5. MK9 was still on the classic codegen; `tools/codegen.sh --flavor
optimised` (perf_overrides.toml: skip_msr, registers as locals, no-volatile loads/stores + loop barriers, direct calls)
leaves no global lock. sa2_renut_core_layout = false (main thread off its own core) gave only +0.5 fps.

Console (user's lowered V1 clocks: CPU 1785, GPU 806, RAM 1733), 90 s fights, triangles fix on:

| Scene | classic | optimised |
|---|---|---|
| Courtyard (Night), practice | 10.9 | 47.2 (min 44.7) |
| The Dead Pool, practice | 18.3 | 53.6 (min 49.5) |
| Arcade Ladder (Cage vs Raiden, lava stage) | ~35 (another stage) | 40.4 (min 26.5) |

Screenshots clean in all three. The classic generated code is kept in out/generated_classic.

## Stage sweep on the optimised build (2026-10-24)

`rig/sweep.sh mk9_opt 0 19`: Practice, Arena Select DRIGHT x N, 20 s fights. 50-57 fps on most stages. Found:
- **Desert (arena 4)**: the stage is covered by an opaque sand-coloured layer (the fighters too on the Switch). PC
  Renut shows it as well; the emulated GPU (sa2_renderer=xenos) draws a light haze. The draw is a UE3 fog volume: PS
  A5A86E249D16F0B7 / VS 0ECBA41F038CB58A, a 24-vertex cube (+-128) the VS places (c18-c21 local-to-view, c0-c3 the
  projection, inverted depth: near 10, far ~10000), alpha 0.36 x (2(1 - W/SceneDepth))^2 x noise, scene depth from
  the 1280x720 24_8_FLOAT resolve 400D09D0. Ruled out: the depth values (mk9_diag_dump_depth: 0.0011-0.034, sane),
  the depth-words shortcuts, bool specialisation, chain reuse / snapshot deltas, the cull mode (mk9_diag_flip_cull_ps),
  the literals (c253-c255 from the cache). Skipping the PS (mk9_diag_skip_ps) shows the stage. Next: the cube's
  view-space placement / W against the emulated renderer.
- **Rooftop stage (arena 14)**: a black patch on the foreground floor (also in a ladder fight there).
- **Boot hang, 1 of ~25 boots**: the main thread inside svcMapProcessMemory (RexGmFaultIn -> MapChunkIntoView,
  sdk/src/core/guest_memory_switch.cpp) for 60+ s holding the guest-memory mutex, another guest thread waiting for it
  in RexGmQueryProtect (both in the data-abort handler). Dump: mk9-logs/rig/hang10/rex_crash.log.
- New diagnostics (draw.cpp): mk9_diag_skip_ps, mk9_diag_trace_ps ([texlog]: fetch constants, PS/VS constants, mode
  control), mk9_diag_flip_cull_ps, mk9_diag_dump_depth.

## Single-sample host targets (2026-10-24)
The game's fight targets (scene depth, shadow depth) are 2x. On deko3d the multisampled depth resolve
(depth_resolve_ms) reads 0: Rooftop (Dawn)'s floor shader (PS 7FCAC75ED7FAD8ED) saw an all-zero shadow map and
went black, and the Desert's fog sheet (PS A5A86E249D16F0B7) depth-tested against a zero scene depth and covered
the fighters. `mk9_msaa_as_1x` (default on) makes them single-sample on the host and `sa2_renut_msaa` now defaults
to 0. Both stages render as on PC; fights run 56-60 fps on the V1 (Courtyard (Night) 47 -> ~57).
The resolve itself is still broken on deko3d if MSAA is ever wanted back.

## RexGmFaultIn boot hang: instrumentation (2026-10-24)
Seen once (mk9-logs/rig/hang10, frame 63, on the older clocks): main thread inside svcMapProcessMemory for 60+ s
holding the guest-memory lock. The window's map/unmap calls now go through TracedSvc (guest_memory_switch.cpp) and
the hang report logs `[hang] guest memory svc:` twice a second apart (call count, failures, the call in flight and its
age): a still count = stuck in the kernel, a climbing one = a failing map retried. rig/bootloop.sh mk9_gmtrace 30 K0:
0 hangs in 30 boots after the user's RAM voltage change.

## Desert and Courtyard checked against real footage (2026-10-24)
The user compared with Xbox 360 gameplay: the Desert's thick sandstorm and Courtyard (Night)'s brighter look, as Renut
draws them, are correct. The emulated renderer (sa2_renderer=xenos) is wrong on both (it lightens the Desert's haze
and darkens the Courtyard), so it is not a reference for these stages.
