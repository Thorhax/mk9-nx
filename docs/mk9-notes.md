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
| common Draw* flush | | all six draw candidates call the same sequence: `sub_82BFDD28` (8 times), `sub_82BFE0C0` (twice), the shader load `sub_82BFEEE8`, `sub_82BFDF68`, `sub_82BFDDE8`, `sub_82BFD990`, `sub_82BFCD60`, `sub_82BFBE80` (`sub_82BF0400` also `sub_82BFBB78` twice). Direct call sites: `sub_82BEFEF8` 2, `sub_82BF09E0` 2, `sub_82BF1270` 0 (reached by pointer), `sub_82BF0400` 1, `sub_82BF0DF8` 8, `sub_82BF1710` 1. Stage 4 hooks all six with `NoteDrawCall` |
| pixel-shader container checks | `sub_82BED4D0`, `sub_82C0B7E8`, `sub_82C13B48`, `sub_82C36AF0`, `sub_82CA06D8`, `sub_831AEE20`, `sub_8324BE50` | build the pixel container magic `0x102A1100`; CreatePixelShader is among them (USF4: 60-instruction function, r3 = container) |
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
