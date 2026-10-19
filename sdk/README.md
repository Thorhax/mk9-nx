# ReXGlue SDK fork with the Horizon (Nintendo Switch) layer

This is a fork of the [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) v0.10.0 (commit `c94f5eb`) used by the
Sonic Adventure 2 Xbox 360 to Nintendo Switch static-recompilation port (it came from masseffect-nx's fork, with the
fixes of the other ports). `patches/rexglue-sdk-v0.10.0.patch` at the root of the repository is the whole fork as one
diff against upstream. It adds:

- **Horizon layer** (`src/**/*_switch.*`, `src/ui/switch_*.cpp`, `src/input/switch/`, `src/audio/switch/`,
  `cmake/rexglue_switch.cmake`, `switch_compat/`): guest memory, exception handling, clock, threading, filesystem,
  input, audio, presentation and performance profile for devkitA64 + libnx. The Horizon layer is by StevensND (nfsmw-nx).
- **Codegen options** of the `rexglue` tool that shrink and speed up the recompiled C++ (arguments passed in
  registers, "diet" lowerings of single PPC instructions, registers as C++ locals, sharded registration, and so on).
  They are selected from the game's TOML configuration; see `src/codegen/codegen_flags.*` and
  `include/rex/codegen/config.h`.
- Console tuning of the Vulkan graphics backend, XMA audio, the VFS read caches and the kernel I/O path. The cvars
  added by the fork are prefixed `sa2_` (for example `sa2_io_ranges_mb`).

## Third-party sources

`thirdparty/` only holds the files this fork changed relative to upstream (FFmpeg build configuration and a table
cache in the FFT/MDCT code, the FFmpeg codec list, and `thirdparty/CMakeLists.txt`). Fetch the rest with:

    python tools/fetch_thirdparty.py                  # clones upstream v0.10.0 and its submodules
    python tools/fetch_thirdparty.py --source <dir>   # or copy from a local rexglue-sdk checkout

(`tools/` is at the root of the repository that contains this `sdk/` folder.)

## Host code generator

    cmake -S . -B out/host -G Ninja -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
    cmake --build out/host --target rexglue

## Licensing

BSD-3-Clause, see `LICENSE`. All upstream and third-party copyright notices are kept intact. The original upstream
README is in `README.upstream.md`.
