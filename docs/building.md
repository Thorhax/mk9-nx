# Building mk9-nx

How to build and run the PC version from your own copy of the game. Windows is the main PC platform (the guide's
toolchain); Linux works too (section "Linux"). Nothing from the game is in this repository: you extract your disc,
translate its code and, for the native renderer, use a shader package made from your copy.

## What you need (Windows 10/11)

- **Your own disc image** of Mortal Kombat Komplete Edition for Xbox 360 (title 575207FD, `default.xex` version
  0.0.0.8; ~7.8 GB).
- **Visual Studio 2022 Build Tools** with the *Desktop development with C++* workload (MSVC libraries, Windows SDK;
  CMake and Ninja come with it). <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- **LLVM clang 20** (the SDK needs clang): the Windows x64 archive or installer from
  <https://github.com/llvm/llvm-project/releases> (e.g. `clang+llvm-20.1.8-x86_64-pc-windows-msvc.tar.xz`).
  Put it in a folder `_tools\llvm` **next to** the `mk9-nx` folder (so `..\_tools\llvm\bin\clang++.exe`), or point
  `$env:SA2_LLVM` at it.
- **Python 3** and **Git** on the PATH.
- About **30 GB** free: 8 GB for the extracted game, the rest for the build.
- A GPU with a current **Vulkan** driver (any NVIDIA/AMD/Intel card from the last ten years).

Run every command from **PowerShell** in the `mk9-nx` folder. Build commands go through `tools\win\run.ps1`, which
loads the compiler environment and keeps the whole output in a log (`out\*.log`).

## Steps

```powershell
# 0. Allow the scripts in this window (once per window)
Set-ExecutionPolicy -Scope Process Bypass

# 1. The SDK's third-party sources (~19,000 files from the ReXGlue SDK release, needs Git; 2-5 minutes)
python tools\fetch_thirdparty.py

# 2. Extract your disc (only reads the .iso; ~8 GB, a few minutes)
python tools\extract_iso.py "D:\path\to\Mortal Kombat Komplete Edition.iso" -o assets\game_root

# 3. Build the code generator (10-20 minutes the first time)
tools\win\run.ps1 out\host_configure.log "cmake -S sdk -B out/host -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_FLAGS=-march=x86-64-v2 -DCMAKE_CXX_FLAGS=-march=x86-64-v2"
tools\win\run.ps1 out\host_build.log "cmake --build out/host --target rexglue"

# 4. Translate the game's code to C++ (about 2 minutes; writes app\generated, 157 MB)
tools\codegen.ps1

# 5. glslangValidator, for the native renderer's own shaders (a few minutes)
tools\win\build_glslang.ps1

# 6. Build the game (20-40 minutes the first time: ~160 MB of generated C++)
tools\win\run.ps1 out\pc_configure.log "cmake --preset win-pc -S app"
tools\win\run.ps1 out\pc_build.log "cmake --build out/pc"
```

The result is `out\pc\mk9.exe`, with `mk9.toml` (settings) next to it.

**7. The shader package** for the native renderer: copy `sa2_shaders.mesp` and `sa2_shaders.mesp.idx` (made from
your copy of the game, see "Making the shader package") into `out\pc\`.

## Playing

```powershell
tools\win\play.ps1            # native Vulkan renderer, window, sound
tools\win\play.ps1 -Emulated  # ReXGlue's emulated GPU instead (slower; the reference image)
```

A controller (Xbox / XInput or any SDL-supported pad) works as player 1. The keyboard works too (`play.ps1` turns on
the SDK's keyboard driver, `mnk_mode = true`; click the window first):

| Key | Pad |
|---|---|
| Enter (or X) | Start (menus, skip movies) |
| Space (or ;) | A (confirm) |
| Backspace (or ') | B (back) |
| L | X |
| P | Y |
| 3 | RB |
| E (or O) | RT |
| W A S D | left stick |
| Shift + arrows | d-pad |
| Tab (or Z) | Back |

The attacks follow the game's own controller layout (see its Options menu).

When you close the game, `play.ps1` prints the frame-rate lines of the log (`[fps] ... fps over 5.0 s`, every 5
seconds). The full log is in `out\pc\logs\`.

`mk9.toml` starts muted and minimized for automated tests; `play.ps1` overrides both. To play without the script,
set `audio_mute = false`, `window_start_minimized = false` and `sa2_renderer_native = true` in `out\pc\mk9.toml`.

The Windows build itself is not tested yet for MK9 (usf4-nx's was, with the same scripts); the Linux build is
(section "Linux").

## Automated tests

```powershell
# 45 s muted and minimized, then the end of the log and the [fps] lines
tools\win\boot_test.ps1
# native renderer, snapshots at 10/20/30 s (out\pc\captures\snap_<s>s.ppm), scripted gamepad
tools\win\boot_test.ps1 -Native -Seconds 60 -Snapshots "10,20,30" -Script "20 START;26 START"
```

The scripted gamepad (`mk9_input_script`, `mk9_input_script_file`): events `<seconds> <BUTTON[+BUTTON]> [hold]`
separated by `;` or one per line, buttons `A B X Y START BACK LB RB LT RT UP DOWN LEFT RIGHT LS RS LS_UP ...`
(`app/src/mk9_script_input.cpp`).

## Making the shader package

The package holds the game's shaders translated ahead of time (XenosRecomp to HLSL, DXC to SPIR-V). MK9's shaders
are almost all in its Unreal packages, so the package is made from (docs/mk9-notes.md, "Shaders from the game files"):

1. **The game files:** `shaders/tools/build_extra_tools.sh ue3_shader_scan` (needs `XXHASH_DIR=sdk/thirdparty/xxHash`
   and `FMT_DIR=sdk/thirdparty/fmt/include`), then `out/tools/ue3_shader_scan out/ue3scan assets/game_root/Asset/*.xxx`
   (about a minute: ~19,000 containers).
2. **A run with `mk9_dump_shaders = true`** (boot, menus, a fight): every container the game parses goes to
   `out/pc/shaders/`. Optional: the scan already has their microcode.
3. **Containers Direct3D rewrites when it binds them, and Direct3D's own shaders:** run with
   `sa2_shader_misses = misses`; keep `misses/missing` (rewritten containers) and wrap what `misses/discovery` has:
   `python3 tools/wrap_discovered.py --raw <discovery folders> --containers <the folders above> --out <a NEW folder>`
   (it empties its output folder first: one folder per round). Repeat until the log says
   `unidentified loads: VS by address 0, PS by address 0, immediate 0`.

Then `SHADER_SPIRV_CACHE=out/shader_cache tools/build_shader_library.sh <all the folders>` writes
`out/shaders/sa2_shaders.mesp` + `.idx` (the cache keeps compiled SPIR-V by container hash, so later rounds only
translate what is new). The 2026-10-20 package has 19,071 shaders (777 MB; the index, 33 MB, is what stays in memory).

The file names keep sa2-nx's (`sa2_shaders.mesp`, `sa2_*` settings) so fixes can be taken over from it.

## Linux

Same steps with `tools/codegen.sh`, the `linux-pc` preset and `tools/linux/boot_test.sh` (Xvfb); needs clang >= 18,
CMake, Ninja, `libx11-xcb-dev` and `libwayland-dev`. The SDK's shared libraries are loaded from `sdk/out/linux-amd64`
(`LD_LIBRARY_PATH`, the boot test sets it). glslangValidator:
`cmake -S sdk/thirdparty/glslang -B out/glslang -G Ninja -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF && ninja -C out/glslang glslangValidator`.
DXC: Microsoft's Linux release unpacked into `out/tools/dxc`.

## Switch (stage 5)

Built on Linux in devkitPro's Docker image, with the repository mounted at the same path:

```sh
D=$PWD/..; R=$PWD
docker run --rm -v $D:$D -w $R devkitpro/devkita64 bash -c "cmake -S sdk/thirdparty/glslang -B out/glslang-dkp -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DENABLE_OPT=OFF -DGLSLANG_TESTS=OFF -DENABLE_HLSL=OFF && ninja -C out/glslang-dkp glslangValidator"
# The Mesa Horizon SDK (nfsmw-nx's Mesa 26.2.2), with libvulkan.a repacked (docs/mk9-notes.md, "Stage 5"):
python3 tools/switch/repack_archive.py <sdk>/lib/libvulkan.a <copy of the sdk>/lib/libvulkan.a --ar aarch64-none-elf-ar
docker run --rm -v $D:$D -w $R -e DEVKITPRO=/opt/devkitpro devkitpro/devkita64 bash -c "cmake -S app -B out/nx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$R/tools/switch/cmake/switch-devkitA64.cmake \
  -DREXGLUE_SWITCH_NVK_SDK=<copy of the sdk> -DSA2_GLSLANG_VALIDATOR=$R/out/glslang-dkp/StandAlone/glslangValidator \
  -DDEVKITPRO=/opt/devkitpro [-DMK9_SWITCH_LTO=OFF] && cmake --build out/nx -j 8"
tools/linux/package_switch.sh          # out/nx/mk9-nx-switch.zip
```

On the SD card: `sdmc:/switch/mk9-nx/` with the NRO and the shader package, and your extracted disc as
`sdmc:/switch/mk9-nx/game_root/` (without `$SystemUpdate`; ~7.3 GB, no file over 4 GB, so FAT32 cards work). Launch through a title takeover (hold R while starting a
game) or a forwarder with full memory.
