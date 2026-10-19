# Shaders

The game's shaders are Xbox 360 (Xenos) microcode. They are not shipped with this project: they are read from
your own copy of the game, translated ahead of time to SPIR-V, and stored in one local file, the **shader
package** `masseffect_shaders.mesp`, plus its **index** `masseffect_shaders.mesp.idx`. The Switch build loads
both at start-up and looks each shader up by a fingerprint of the original microcode container.

This folder holds the translator, the tools that build and check the package, and the WebAssembly recipes
the browser installer uses to run the same steps on a PC. Nothing here contains game data.

```
 game files (Unreal packages *.xxx, LZO-compressed chunks)
        |  ue3_shader_scan                       (also: app/src/me_shader_dump.cpp, a run-time dump)
        v
 containers   vs_<hash>.bin / ps_<hash>.bin       original Xbox 360 shader containers, big-endian
        |  masseffect_hlsl  (XenosRecomp + our changes)
        v
 HLSL         vs_<hash>.hlsl / ps_<hash>.hlsl     one self-contained file each (shader_common.h is pasted in)
        |  DXC  (-spirv, Vulkan 1.2)  + spirv-val
        v
 SPIR-V       vs_<hash>.spv / ps_<hash>.spv
        |  me_pack_shaders
        v
 masseffect_shaders.mesp   +   masseffect_shaders.mesp.idx
```

Every path below is relative to the repository root. Built programs go to `out/tools/`.

## Layout

| Path | What it is |
|---|---|
| `shaders/XenosRecomp/` | [XenosRecomp](https://github.com/hedge-dev/XenosRecomp) by hedge-dev (MIT, see `LICENSE.md`), vendored and changed for this game (below). Only `shader_recompiler.*`, `shader_common.h`, `shader.h`, `shader_code.h`, `constant_table.h` and `pch_min.h` are used. `main.cpp`, `dxc_compiler.*`, `pch.h` and `CMakeLists.txt` are upstream's own front end (DXC library, smol-v, zstd) and are **not built** here; they are kept so the tree can still be diffed against upstream. |
| `shaders/README.upstream.md` | Upstream's README, unchanged. |
| `shaders/LICENSE.md` | XenosRecomp's MIT licence. |
| `shaders/tools/env.sh` | Finds the compiler, xxHash, fmt, DXC and spirv-val (see "Prerequisites"). Sourced by the other scripts. |
| `shaders/tools/masseffect_hlsl.cpp` | Command line front end of the translator: a folder of containers in, a folder of `.hlsl` out. |
| `shaders/tools/build_xenos_hlsl.sh` | Builds it (`out/tools/xenos_hlsl`). |
| `shaders/tools/translate_all.py` | Translates a folder, one process per container (8 in parallel), so a crash in one container does not stop the rest; failures are listed in `translate_failed.txt`. |
| `shaders/tools/compile_spirv_all.sh`, `shaders/tools/compile_spirv_one.sh` | DXC + `spirv-val` for every `.hlsl`, 8 in parallel. A failure leaves `<name>.err`. These two files hold the exact DXC command line. |
| `shaders/tools/build_shader_spirv.sh` | The whole pipeline: containers -> HLSL -> SPIR-V -> package + index. |
| `shaders/tools/me_pack_shaders.cpp` | The packer. Also extends or replaces entries of an existing package (`--replace`, `--replace-exact`), and writes the `.idx`. |
| `shaders/tools/me_index_shaders.cpp`, `.sh` | Writes the `.idx` of an existing package. |
| `shaders/tools/validate_shader_package.cpp`, `.sh` | Runs every SPIR-V of a package through SPIRV-Tools. |
| `shaders/tools/extract_shader_package.cpp` | Package -> loose containers again (to search a shipped package). |
| `shaders/tools/package_reference.py` | An independent Python reader/writer of the package and the index, written from the format description below (needs `pip install xxhash`). It reproduces the C++ output byte for byte. |
| `shaders/tools/ue3_shader_scan.cpp` | Finds the containers in the game's Unreal packages (includes its own LZO1X decoder). |
| `shaders/tools/collect_shaders.sh` | Collects containers at run time instead (`MASSEFFECT_SHADER_DUMP`, needs the PC build of the game and a lot of play time). |
| `shaders/tools/find_shader_microcode.cpp`, `shaders/tools/wrap_raw_shader.py`, `shaders/tools/repair_vertex_variant_declarations.cpp` | Developer tools for run-time shaders that have no container (Direct3D's own): find, wrap in a container, repair a vertex declaration. |
| `shaders/tools/hlsl_fma.py` | Optional experiment: rewrites `a*b+c` as `mad()` in vertex HLSL. Not part of the default pipeline. |
| `shaders/tools/build_extra_tools.sh` | Builds `extract_shader_package`, `find_shader_microcode`, `repair_vertex_variant_declarations`, `ue3_shader_scan`. |
| `shaders/tests/` | `test_shader_precise_codegen.sh`, `test_shader_alu_parallel_codegen.sh` (translator code generation, no game files), `test_shader_index.cpp` (index load == full load), `test_vertex_variant_declarations.cpp`. |
| `shaders/wasm/` | WebAssembly build scripts for the browser installer (see "WebAssembly"). |

## Prerequisites

macOS or Linux, `clang++` (C++20), Python 3, and:

* xxHash (`xxhash.h`), fmt (header only), SPIRV-Tools, DXC, `spirv-val`.
  On macOS: `brew install xxhash fmt spirv-tools`, and DXC from the LunarG Vulkan SDK (or any `dxc` binary).
* `shaders/tools/env.sh` looks for them in `XXHASH_DIR`, `FMT_DIR`, `DXC`, `SPIRV_VAL` (environment), then in
  `../rexglue-sdk/thirdparty`, `thirdparty/`, Homebrew, `~/VulkanSDK/*/macOS/bin` and `PATH`.

## Quick start

```sh
# 1. containers from the game's Unreal packages (about 30 s for the whole disc, 118 MB of output)
shaders/tools/build_extra_tools.sh ue3_shader_scan
find /path/to/disc -type f -iname '*.xxx' -print0 | xargs -0 out/tools/ue3_shader_scan run/shaders

# 2. containers -> HLSL -> SPIR-V -> package + index (about 6 minutes on a 10-core Mac for 30,191 containers)
ALLOW_FAILURES=1 shaders/tools/build_shader_spirv.sh run/shaders run/shaderlib
#    -> run/shaderlib/masseffect_shaders.mesp  and  masseffect_shaders.mesp.idx

# 3. checks
shaders/tools/validate_shader_package.sh run/shaderlib/masseffect_shaders.mesp
shaders/tools/me_index_shaders.sh run/shaderlib/masseffect_shaders.mesp   # rebuilds the .idx and builds the index test
out/tools/test_shader_index run/shaderlib/masseffect_shaders.mesp run/shaderlib/masseffect_shaders.mesp.idx --all
shaders/tests/test_shader_precise_codegen.sh
shaders/tests/test_shader_alu_parallel_codegen.sh
```

Put both package files next to the game executable (or point `MASSEFFECT_SHADER_LIBRARY` to the `.mesp`; the
game looks for `<that path>.idx` beside it). On the Switch they go beside the `.nro`. The names are fixed.

## Step 1: finding the containers

The game's Unreal Engine 3 packages (`*.xxx`) hold a shader cache. Their compressed chunks start with the
tag `0x9E2A83C1` written twice (big-endian), then the total compressed size, the total uncompressed size, one
`(compressed, uncompressed)` pair of big-endian `u32` per 128 KB block, and the blocks. Mass Effect's blocks
are LZO1X (`CompressionFlags = 2`). `ue3_shader_scan` finds the chunks by that header (the chunk table
offsets vary), decompresses them, and looks in the result, and in the raw file, for every shader container.

A **container** is what the Xbox 360 shader compiler writes. All fields are big-endian:

| Offset | Field |
|---|---|
| 0 | signature `0x102A11xx`. Bit 0 of the low byte is set for a vertex shader and clear for a pixel shader. The rest of the low byte is the compiler tool's and is ignored. |
| 4 | virtual size (constant table, definitions, shader headers; at least 24) |
| 8 | physical size (microcode and the literal constants of the definition table) |

The container is `virtual size + physical size` bytes. `ue3_shader_scan` accepts a candidate when the
signature matches, `24 <= virtual <= 0x40000`, `0 < physical <= 0x40000` and the whole container fits in the
buffer, then writes it as `vs_<h>.bin` or `ps_<h>.bin`, where `h` is the 64-bit FNV-1a of the container
(only a file name; the stage prefix is what the later steps read). Duplicates are written once.
On a complete disc this finds 403,857 containers, **30,191 distinct**.

## Step 2: container -> HLSL

`out/tools/xenos_hlsl <containers> <hlsl out> shaders/XenosRecomp/shader_common.h` (the output folder must be
new or empty). Every `.bin` becomes a `.hlsl` that starts with a copy of `shader_common.h` (the helper
functions, constants and the constant-buffer layout every shader shares) followed by the translated shader,
so each `.hlsl` compiles on its own with no `#include`.

XenosRecomp is upstream's, with these changes (all marked `MASSEFFECT` in the source). The first three
are the patches this project always carried; the rest came later:

1. **Sampler arrays.** A sampler array (Unreal's `LightMapTextures`, `s5..s7`) covers several registers and the
   microcode fetches from each; upstream only defined the first. Every register now gets its definitions, and
   every fetch register that has no name gets an `sN_*` set.
2. **2008 containers** whose physical part is not a multiple of 12 bytes (844 and 856 byte ones exist) are
   accepted (the library checks in `sa2_shader_library.cpp`).
3. **Relocated literal definitions.** Float definitions whose physical range overlaps the microcode or lies
   outside the physical block are ignored and the real guest registers are read. Valid definitions are
   untouched.
4. Direct3D's own shaders (no constant table, or an empty one) are linked by hardware register instead of by
   semantic names; their constants not named anywhere are read straight from the constant buffer.
5. Constants come through a dynamic uniform buffer (`SPEC_CONSTANT_CONSTANTS_UBO`) instead of a 64-bit pointer;
   texture signs (unsigned-biased, gamma) are applied in the sampling helpers (the renderer packs them into
   bits 24..31 of the descriptor index); the alpha test has all eight comparison functions; the position is
   mapped to host clip space (`g_NdcScale`, `g_NdcOffset`); the vertex input remap (`g_InputRemap`); 1/size of
   a texture comes from the shared constants instead of a size query; `setp_inv` follows Xenia.
6. Simultaneous vector/scalar ALU results are computed with the operands as they were before the instruction
   (Xenos semantics), and an opt-in `precise` position contract (`MASSEFFECT_SHADER_PRECISE_POSITION=1`).

Code for other games has been removed from this copy where it was dead for Mass Effect: the `#ifdef`
variants of the renderer this one was built on are resolved to the Mass Effect side, and the shadow-map/PCF/blur helpers of
the other game (`tfetch2DShadow*`, specialization bits 15, 19 and 23) are gone from `shader_common.h`. The
Sonic `UNLEASHED_RECOMP` blocks of upstream are left alone (they are never compiled). The generated HLSL is
the same as before the cleanup: it was compared on 443 containers (identical except for generated temporary
names that depend on the header length), and DXC gives the same SPIR-V for the old and new `shader_common.h`.

**Containers that do not make it.** On the complete disc, 56 of the 30,191 containers crash the translator
(SIGSEGV/SIGBUS, probably malformed false positives of the scan) and 4 vertex shaders translate but DXC
rejects them (`redefinition of parameter 'iBlendWeight0'`). They are left out: the package then has **30,131**
shaders. None of the 56 crashing containers is in the project's production package either. An installer must
treat a failed container as "skip it", never as a fatal error. (`build_shader_spirv.sh` refuses to pack
unless `ALLOW_FAILURES=1`, so a regression is noticed.)

## Step 3: HLSL -> SPIR-V

The exact command, one file at a time (`shaders/tools/compile_spirv_one.sh`):

```
dxc -spirv -T ps_6_6 -E main -HV 2021 -fspv-target-env=vulkan1.2 -fvk-use-dx-layout \
    -Werror=parameter-usage  -Fo <name>.spv <name>.hlsl            # pixel shader (file name ps_*)
dxc -spirv -T vs_6_6 -E main ... same ... -fvk-invert-y ...        # vertex shader (anything else)
spirv-val --target-env vulkan1.2 --scalar-block-layout <name>.spv
```

A shader that fails either command is not packed, and the build script then refuses to write the package.
The tested DXC is 1.9 from the Vulkan SDK 1.4.363 (the browser build is pinned to the DXC commit named in
`wasm/link_dxc_wasm.sh`). A different DXC version gives different bytes but equally valid SPIR-V; the
package is keyed by container, not by SPIR-V, so that is fine.

## Step 4: the package (`masseffect_shaders.mesp`)

Everything is **little-endian** unless said otherwise. `XXH3` means XXH3 64-bit, seed 0, no secret (xxHash
0.8.x `XXH3_64bits`). Reference vectors: `XXH3("")` = `2D06800538D394C2`, `XXH3("a")` = `E6C632B61E964E1F`,
`XXH3("abc")` = `78AF5F94892F3950`, `XXH3(bytes 0..255 then 0..43, 300 bytes)` = `D44052F5A3485425`.

```
offset  size  field
0       8     magic        "MESSPV\0\0"  (4D 45 53 53 50 56 00 00)
8       4     version      1
12      4     count        number of entries, 1 .. 65536
16      8     checksum     XXH3 of every byte from offset 24 to the end of the file
24      ...   count entries, back to back, no padding:

   entry:
   +0   4     originalBytes   24 .. 65536
   +4   4     spirvWords      5 .. 1048576
   +8   8     key             XXH3 of the original container bytes (the whole container, as found on the disc)
   +16  originalBytes          the original container, unchanged
   +16+originalBytes  spirvWords*4   the SPIR-V module, little-endian 32-bit words
```

Rules the loader enforces (an invalid file is rejected as a whole):

* The file is at most 2^30 bytes. `checksum` matches. The file ends exactly at the last entry.
* Entries are sorted strictly ascending by `(key, container bytes compared as unsigned bytes,
  lexicographically)`. No container appears twice. (When the same container is packed twice with the same
  SPIR-V the packer keeps one; with different SPIR-V it fails.)
* `key` equals `XXH3(container)`.
* Container: the first four bytes read big-endian `s` satisfy `(s & 0xFFFFFF00) == 0x102A1100` (a 2005-layout
  `(s & ~1) == 0x102A0E00` with physical size a multiple of 12 is also accepted, unused for this game); the
  big-endian `u32` at 4 (virtual size) is at least 24; the one at 8 (physical size) is not 0; and
  `virtual + physical == originalBytes`. Vertex shader if bit 0 of `s` is set.
* SPIR-V: `word[0] == 0x07230203`, `word[4] == 0`, at most 4 MiB. Walking the instructions from `word[5]`
  (word count in the high 16 bits, opcode in the low 16; a count of 0 or one that overruns is an error)
  there is **exactly one** `OpEntryPoint` (opcode 15), with execution model 0 (Vertex) for a vertex
  container or 4 (Fragment) for a pixel container, at least 5 words, and the name `main` (words
  `0x6E69616D, 0`).

Lookup at run time: compute `XXH3` of the container the game hands to the driver, binary-search the sorted
entries by `key`, and compare the container bytes (a hash collision can never select another shader).

The packer (`me_pack_shaders <containers> <spirv> <out.mesp>`) packs every `<name>.bin` that has a
`<name>.spv`; it also writes `<out.mesp>.idx`, and reloads the result to check that every packed container
is found.

## Step 5: the index (`masseffect_shaders.mesp.idx`)

The full package is about 920 MB; reading all of it at start-up is slow and needs the RAM. The index lets the
game read only the table and the containers (about 48 MB) instead and fetch each SPIR-V from the package the first time it is used. It is derived
entirely from the package (`package_reference.py index` and the C++ `IndexShaders` give identical bytes),
it is bound to that package by size and checksum, and a package without a matching index is still loaded in
full.

```
offset  size  field
0       8     magic            "MESSIDX\0"  (4D 45 53 53 49 44 58 00)
8       4     version          1
12      4     count            same as the package
16      8     packageBytes     size of the package file
24      8     packageChecksum  the u64 at offset 16 of the package
32      8     originalsBytes   total size of the concatenated containers (last part of this file)
40      8     bodyHash         XXH3 of every byte of this file after the 56-byte header
48      8     reserved         0
56      count * 40   table, one row per package entry, in package order:
   +0   4     originalBytes
   +4   4     spirvWords
   +8   8     key              the entry's key
   +16  8     spirvHash        XXH3 of the entry's SPIR-V bytes exactly as stored in the package
   +24  4     kills            number of OpKill (252), OpTerminateInvocation (4416) and
                               OpDemoteToHelperInvocation (5380) in the module; 2 if it cannot be walked
   +28  4     flags            reserved, always 0
   +32  8     offset           file offset, in the package, of this entry's 16-byte header
56+40*count   originalsBytes   the original containers, concatenated in the same order
```

* `offset` of entry 0 is 24; entry `n+1` is at `offset(n) + 16 + originalBytes(n) + spirvWords(n)*4`; the last
  entry ends at `packageBytes`.
* The index file size is exactly `56 + 40*count + originalsBytes`.
* While loading, the game checks `count`, `packageBytes` and `packageChecksum` against the package, the
  `bodyHash`, the sort order, every key against its container, and that offsets are consistent. When a
  SPIR-V is read later it re-checks the entry's header, the container bytes, `spirvHash` and the SPIR-V
  framing; a mismatch aborts the game with a message (the package changed under the index).
* If the index is missing or does not match, the game loads the package without it, and the cvar
  `masseffect_shaders_index=false` turns the index off.

## What the installer must produce

Two files, byte-exact names, side by side in the folder the game reads them from (next to the executable /
`.nro`, or the path in `MASSEFFECT_SHADER_LIBRARY`):

* `masseffect_shaders.mesp`
* `masseffect_shaders.mesp.idx`

A browser implementation needs: the LZO1X chunk reader and container scanner (a port of
`shaders/tools/ue3_shader_scan.cpp`, no other dependency), the translator (`shaders/wasm/build_wasm_tools.sh hlsl`), DXC
(`shaders/wasm/link_dxc_wasm.sh`) with the command line above, XXH3-64, and a writer for the two formats above.
`shaders/tools/package_reference.py` is a compact model of the writer. Expect 30,191 containers and a package of
about 920 MB (30,131 shaders, index 48.5 MB); the package limit is 2^30 bytes.

An installer can check its work against a native build: the package is deterministic for a given set of
`(container, SPIR-V)` pairs, so the same pairs must give a byte-identical file.

## WebAssembly

`shaders/wasm/` holds the recipes to run the pipeline in a browser:

* `build_wasm_tools.sh` builds `hlsl.mjs` (translator), `pack.mjs` (packer) and `scan.mjs` (package scanner)
  with Emscripten. Each is a command line program: write the inputs into the module's in-memory file system
  (`FS`), call `callMain([...])` with the same arguments as on the command line, read the outputs back.
* `link_dxc_wasm.sh` + `dxc_web.cpp` link DXC (a WebAssembly build of microsoft/DirectXShaderCompiler) behind a
  single `compile(input, output, vertex)` function that uses the exact options of step 3.

Status (Emscripten 6.0.10 from Homebrew, run under Node 26):

* `hlsl.mjs`, `pack.mjs`, `scan.mjs`: **built and tested.** The translator's HLSL is byte-identical to the
  native build on 6 containers; the packer's `.mesp` and `.idx` are byte-identical to the native packer and to
  `shaders/tools/package_reference.py` for the same inputs; the scanner decompresses chunks with the same counts as the
  native one. Not tested: a whole 30k-container run in WebAssembly (memory: the packer holds the whole package,
  about 920 MB, in a 4 GB address space), nor in a real browser (only Node).
* `dxc_web.mjs`: **not built, untested.** Building DXC (LLVM/Clang based, with the SPIR-V backend) to
  WebAssembly needs the DXC sources with submodules, a native build for `llvm-tblgen`/`clang-tblgen`, and a long
  Emscripten build; that did not fit in this session. `link_dxc_wasm.sh` and `dxc_web.cpp` are the recipe
  carried over from another game of this family, adapted to `.sh`, with the same DXC options as
  `shaders/tools/compile_spirv_one.sh`. Treat them as a starting point.

## Licence and third parties

`XenosRecomp/` is MIT (hedge-dev and contributors, see `LICENSE.md`); our changes to it are under the same
terms. xxHash (BSD-2), fmt (MIT) and SPIRV-Tools / DXC (Apache-2.0 with LLVM exception) are used as external
dependencies and are not copied here.
