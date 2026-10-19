# Building usf4-nx

How to build and run the PC version from your own copy of the game. Windows is the main PC platform (the guide's
toolchain); Linux works too (section "Linux"). Nothing from the game is in this repository: you extract your disc,
translate its code and, for the native renderer, use a shader package made from your copy.

## What you need (Windows 10/11)

- **Your own disc image** of Ultra Street Fighter IV for Xbox 360 (`Ultra Street Fighter IV (World).iso`, title
  434307FC; ~8.7 GB).
- **Visual Studio 2022 Build Tools** with the *Desktop development with C++* workload (MSVC libraries, Windows SDK;
  CMake and Ninja come with it). <https://visualstudio.microsoft.com/visual-cpp-build-tools/>
- **LLVM clang 20** (the SDK needs clang): the Windows x64 archive or installer from
  <https://github.com/llvm/llvm-project/releases> (e.g. `clang+llvm-20.1.8-x86_64-pc-windows-msvc.tar.xz`).
  Put it in a folder `_tools\llvm` **next to** the `usf4-nx` folder (so `..\_tools\llvm\bin\clang++.exe`), or point
  `$env:SA2_LLVM` at it.
- **Python 3** and **Git** on the PATH.
- About **30 GB** free: 8 GB for the extracted game, the rest for the build.
- A GPU with a current **Vulkan** driver (any NVIDIA/AMD/Intel card from the last ten years).

Run every command from **PowerShell** in the `usf4-nx` folder. Build commands go through `tools\win\run.ps1`, which
loads the compiler environment and keeps the whole output in a log (`out\*.log`).

## Steps

```powershell
# 0. Allow the scripts in this window (once per window)
Set-ExecutionPolicy -Scope Process Bypass

# 1. The SDK's third-party sources (~19,000 files from the ReXGlue SDK release, needs Git; 2-5 minutes)
python tools\fetch_thirdparty.py

# 2. Extract your disc (only reads the .iso; ~8 GB, a few minutes)
python tools\extract_iso.py "D:\path\to\Ultra Street Fighter IV (World) (En,Ja,Fr,De,Es,It,Ko).iso" -o assets\game_root

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

The result is `out\pc\usf4.exe`, with `usf4.toml` (settings) next to it.

**7. The shader package** for the native renderer: copy `sa2_shaders.mesp` and `sa2_shaders.mesp.idx` (made from
your copy of the game, see "Making the shader package") into `out\pc\`.

## Playing

```powershell
tools\win\play.ps1            # native Vulkan renderer, window, sound
tools\win\play.ps1 -Emulated  # ReXGlue's emulated GPU instead (slower; the reference image)
```

A controller (Xbox / XInput or any SDL-supported pad) works as player 1. The keyboard works too (`play.ps1` turns on
the SDK's keyboard driver, `mnk_mode = true`; click the window first):

| Key | Pad | USF4 |
|---|---|---|
| Enter (or X) | Start | menus, skip movies |
| Space (or ;) | A | confirm, Light Kick |
| Backspace (or ') | B | back, Medium Kick |
| L | X | Light Punch |
| P | Y | Medium Punch |
| 3 | RB | Heavy Punch |
| E (or O) | RT | Heavy Kick |
| W A S D | left stick | move |
| Shift + arrows | d-pad | menus |
| Tab (or Z) | Back | |
 The first run asks to create save data:
choose **YES**. Saves go to `Documents\usf4\` (the SDK's user folder: `GetUserFolder()` / the app name; the `user_data_root` setting moves it).

When you close the game, `play.ps1` prints the frame-rate lines of the log (`[fps] ... fps over 5.0 s`, every 5
seconds). The full log is in `out\pc\logs\`.

`usf4.toml` starts muted and minimized for automated tests; `play.ps1` overrides both. To play without the script,
set `audio_mute = false`, `window_start_minimized = false` and `sa2_renderer_native = true` in `out\pc\usf4.toml`.

### First Windows test (2026-10-11)

Windows 11, Visual Studio 2026 Community (`18\Community`: set `$env:SA2_VSDIR` to it, the scripts default to the 2022
Build Tools folder), LLVM 20.1.8: every step built; the native renderer ran a fight at a steady **60 fps**. Long
commands copied from a chat window can arrive split over several lines: PowerShell then waits with `>>` (Ctrl+C);
building the command in a variable (`$cfg = "..."; $cfg += "..."`) avoids it.

### What to look at (state on 2026-10-11)

- Known wrong with the native renderer: horizontal stripes over the top-left HUD (player 1's health bar and name)
  and a flat yellow sky on the volcano stage. Anything else that looks different from the emulated GPU
  (`-Emulated`) is worth a note, with the place it happens.
- The frame rate in a fight (`[fps]` lines): on a desktop GPU the test machine's limits (an idle laptop GPU and a
  virtual display) are gone, so this is the first real measurement.

## Automated tests

```powershell
# 45 s muted and minimized, then the end of the log and the [fps] lines
tools\win\boot_test.ps1
# native renderer, snapshots at 10/20/30 s (out\pc\captures\snap_<s>s.ppm), scripted gamepad
tools\win\boot_test.ps1 -Native -Seconds 60 -Snapshots "10,20,30" -Script "20 START;26 START"
```

The scripted gamepad (`usf4_input_script`, `usf4_input_script_file`): events `<seconds> <BUTTON[+BUTTON]> [hold]`
separated by `;` or one per line, buttons `A B X Y START BACK LB RB LT RT UP DOWN LEFT RIGHT LS RS LS_UP ...`
(`app/src/usf4_script_input.cpp`).

## Making the shader package

The package holds the game's shaders translated ahead of time (XenosRecomp to HLSL, DXC to SPIR-V). A run only
creates the shaders of the stages and characters it loads, so the package is made from three sources:

1. **A run with `usf4_dump_shaders = true`** (boot, menus, a fight): every container the game creates goes to
   `out/pc/shaders/` (the game's in-memory form of each container).
2. **The game files:** `python3 tools/extract_game_shaders.py assets/game_root out/pc/shaders_game --skip out/pc/shaders`
   pulls every container out of the EMZ streams (`archive/shader/*.emz` and the `.eaf` archives: every stage and
   character). Keep only those whose microcode the dump does not already have (the game rewrites container headers
   when it loads them, so the same shader can differ in its header bytes).
3. **Direct3D's own shaders:** run with `sa2_shader_misses = misses` and wrap what it finds:
   `python3 tools/wrap_discovered.py --raw out/pc/misses/discovery --containers <the folders above> --out <a NEW folder>`
   (it empties its output folder first: one folder per round). Repeat until the log says
   `unidentified loads: ... immediate 0` and no `unresolved draw shader pairs` line appears.

Then `tools/build_shader_library.sh <all the folders>` (bash; on Windows it expects MSYS2's clang64 `clang++` and a
DXC build, see the variables at the top) writes `out/shaders/sa2_shaders.mesp` + `.idx`. The second package
(2026-10-11) has 865 shaders: 565 dumped, 290 from the game files, 9 wrapped Direct3D shaders, 1 rebuilt container.

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
# The Mesa Horizon SDK (nfsmw-nx's Mesa 26.2.2), with libvulkan.a repacked (docs/usf4-notes.md, "Stage 5"):
python3 tools/switch/repack_archive.py <sdk>/lib/libvulkan.a <copy of the sdk>/lib/libvulkan.a --ar aarch64-none-elf-ar
docker run --rm -v $D:$D -w $R -e DEVKITPRO=/opt/devkitpro devkitpro/devkita64 bash -c "cmake -S app -B out/nx -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=$R/tools/switch/cmake/switch-devkitA64.cmake \
  -DREXGLUE_SWITCH_NVK_SDK=<copy of the sdk> -DSA2_GLSLANG_VALIDATOR=$R/out/glslang-dkp/StandAlone/glslangValidator \
  -DDEVKITPRO=/opt/devkitpro [-DUSF4_SWITCH_LTO=OFF] && cmake --build out/nx -j 8"
tools/linux/package_switch.sh          # out/nx/usf4-nx-switch.zip
```

On the SD card: `sdmc:/switch/usf4-nx/` with the NRO and the shader package, and your extracted disc as
`sdmc:/switch/usf4-nx/game_root/` (without `$SystemUpdate`). Launch through a title takeover (hold R while starting a
game) or a forwarder with full memory.
