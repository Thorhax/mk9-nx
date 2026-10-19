# Third-party notices

This port is built on the work of other projects. Their licenses apply to their code and to the parts of this
repository derived from it. New code written for this port is copyright (c) 2026 ChanseyIsTheBest and is under the
license of the folder it is in (see below).

## Projects this port is derived from

| Project | Used for | License |
|---|---|---|
| [masseffect-nx](https://github.com/nebadasSwifty/masseffect-nx) by NebadasSwifty | The tree this port started from: the native renderer (`app/src/me`), the structure of `app/`, the shader library tools (`shaders/tools`), the CPU enhancement framework and many scripts in `tools/` | GPL-3.0 |
| [nfsmw-nx](https://github.com/StevensND/nfsmw-nx) by StevensND | The Horizon (Switch) layer of the SDK, the NVK driver build the NRO links, the profiling and PGO method | GPL-3.0; BSD-3-Clause for the SDK parts |
| [NFSMW Recompiled](https://github.com/madelrandel-blip/NFSMW-Recompiled) by madelrandel-blip | The recompilation project the ports above started from: the structure of `app/` (application hooks, configuration, code generator setup) and the build and analysis scripts in `tools/` | GPL-3.0 |
| [ReXGlue SDK](https://github.com/rexglue/rexglue-sdk) v0.10.0 by Tom Clay, with portions from [Xenia](https://xenia.jp) (Ben Vanik and the Xenia contributors) | `sdk/`: code generator, Xbox 360 kernel and runtime, GPU command processing, XMA audio, virtual file system | BSD-3-Clause (`sdk/LICENSE`) |
| Horizon (Switch) layer of the SDK by StevensND | `sdk/src/**/*_switch.*`, `sdk/src/ui/switch_*.cpp`, `sdk/src/input/switch/`, `sdk/src/audio/switch/`, `sdk/cmake/rexglue_switch.cmake`, `sdk/switch_compat/`: guest memory, exception handling, threading, input, audio, presentation, clocks | BSD-3-Clause (as part of the SDK) |
| Scripts and toolchain file by StevensND | `tools/find_gaps.py`, `tools/read_before_write.py`, `tools/direct_calls.py`, `tools/fetch_thirdparty.py`, `tools/switch/cmake/switch-devkitA64.cmake`, adapted for this game | GPL-3.0 |
| [XenosRecomp](https://github.com/hedge-dev/XenosRecomp) by hedge-dev and contributors, with the changes of nfsmw-nx, masseffect-nx and this port | `shaders/XenosRecomp`: Xenos microcode to HLSL translator | MIT (`shaders/LICENSE.md`) |
| [mesa-switch](https://github.com/danfromtico/mesa-switch) by danfromtico (with NaGaa95's updates), on top of [Mesa](https://mesa3d.org) | The Vulkan driver (NVK) and shader compiler (NAK) linked into the NRO: nfsmw-nx's build, which this port does not change (`mesa/README.md`) | MIT (Mesa's licenses per file) |

The Horizon (Switch) layer added to `sdk/` is under the SDK's BSD-3-Clause license, and the changes in `shaders/`
under XenosRecomp's MIT license, so they can be reused by other ports. The rest of this repository (`app/`, `tools/`,
`docs/`, `patches/`) is GPL-3.0 (`LICENSE`).

## Changes to third-party code

`patches/` holds every change this port makes to third-party code, as diffs against the upstream sources it was taken
from (`patches/README.md` names each base and how to apply it):

- **ReXGlue SDK** (`sdk/`): the whole fork against v0.10.0;
- **FFmpeg** and the SDK's FFmpeg build files (`sdk/thirdparty`): the files listed below;
- **XenosRecomp** (`shaders/XenosRecomp`): against the upstream commit it was taken from.

The changed files themselves are in the repository too, so it builds without applying anything. The Mesa driver is
not changed by this port.

## Libraries

Libraries used by the SDK, the NRO and the code generator. They are not stored in this repository except for the
files listed below: `tools/fetch_thirdparty.py` fetches them from ReXGlue SDK v0.10.0 (commit `c94f5ebdcb3c`) and its
submodules into `sdk/thirdparty`.

| Library | License |
|---|---|
| [FFmpeg](https://ffmpeg.org) (libavcodec, libavutil: the XMA and WMA audio decoders), from the FFmpeg submodule of ReXGlue SDK v0.10.0 | LGPL-2.1-or-later (configured without GPL parts) |
| [libmspack](https://www.cabextract.org.uk/libmspack/) by Stuart Caie (LZX decompression) | LGPL-2.1 |
| [glslang](https://github.com/KhronosGroup/glslang) | BSD-3-Clause and others (see its LICENSE.txt) |
| [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) | Apache-2.0 |
| [SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers), [Vulkan-Headers](https://github.com/KhronosGroup/Vulkan-Headers) | MIT, Apache-2.0 |
| [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator) | MIT |
| [fmt](https://github.com/fmtlib/fmt), [spdlog](https://github.com/gabime/spdlog) | MIT |
| [xxHash](https://github.com/Cyan4973/xxHash) | BSD-2-Clause |
| [SIMDe](https://github.com/simd-everywhere/simde) | MIT |
| [toml++](https://github.com/marzer/tomlplusplus) | MIT |
| [Dear ImGui](https://github.com/ocornut/imgui) | MIT |
| [o1heap](https://github.com/pavel-kirienko/o1heap) | MIT |
| [utfcpp](https://github.com/nemtrif/utfcpp) | BSL-1.0 |
| [CLI11](https://github.com/CLIUtils/CLI11) | BSD-3-Clause |
| [aes_128](https://github.com/openluopworld/aes_128) by LuoPeng, [tiny-AES-c](https://github.com/kokke/tiny-AES-c) | see the license file of each in the fetched tree (`sdk/thirdparty/aes_128`, `sdk/thirdparty/tiny-aes-c`) |
| disruptorplus, the RenderDoc API header and the PowerPC disassembler (`disasm`) as shipped with ReXGlue SDK v0.10.0 | see the license file of each in the fetched tree |
| [libnx](https://github.com/switchbrew/libnx) and the devkitA64 runtime (newlib, libstdc++) | ISC; newlib and GCC runtime licenses |
| zlib, Zstandard and Expat, from devkitPro's Switch portlibs (linked for Mesa) | zlib License; BSD-3-Clause; MIT |

Only for builds on a computer or for the tests, never in the Switch NRO: SDL3 (zlib), MoltenVK and the Vulkan loader
(Apache-2.0), Catch2 (BSL-1.0).

### FFmpeg and libmspack (LGPL)

This port changes these files of the SDK's third-party tree; the changed copies are in `sdk/thirdparty`, and
`patches/` has them as diffs:

- `FFmpeg/libavcodec/fft_template.c` and `FFmpeg/libavcodec/mdct_template.c`
- `FFmpeg/config.h`, plus `FFmpeg/config_switch_aarch64.h` and `FFmpeg/config_masseffect_wmv3.h` (new): the
  configuration for AArch64 Horizon, and the WMV3 decoder that masseffect-nx added (compiled, not enabled for SA2)
- `ffmpeg-overlay/codec_list.c`: the list of registered codecs
- `CMakeLists.txt`: builds the libraries for the Switch

The rest of FFmpeg and libmspack is the unmodified upstream source at the commits pinned by ReXGlue SDK v0.10.0.
Everything needed to rebuild and relink the NRO with a modified version of either library is in this repository and in
[docs/building.md](docs/building.md).

## The installer page

The installer page is its own repository, [sa2-nx-installer](https://github.com/ChanseyIsTheBest/sa2-nx-installer).
Its WebAssembly shader translator, packer and LZX decoder are built from `shaders/` and `sdk/thirdparty/libmspack` of
this repository (`shaders/wasm/build_wasm_tools.sh`, with [Emscripten](https://emscripten.org): MIT and the University
of Illinois/NCSA license). Its DXC is nfsmw-nx-installer's WebAssembly build of the unmodified
[DirectXShaderCompiler](https://github.com/microsoft/DirectXShaderCompiler) v2025.1 (commit `75a029d95`), with the
`dxc_web.cpp` wrapper that is also in `shaders/wasm/` (University of Illinois/NCSA Open Source License, with LLVM's
license terms; SPIRV-Tools Apache-2.0; SPIRV-Headers MIT).

## Trademarks

Sonic Adventure 2 and Sonic the Hedgehog are trademarks of SEGA; the icon (`app/icon.jpg`) is SEGA's artwork. Nintendo
Switch is a trademark of Nintendo. Xbox 360 and Xbox Live Arcade are trademarks of Microsoft. This project is not
affiliated with or endorsed by any of them.
