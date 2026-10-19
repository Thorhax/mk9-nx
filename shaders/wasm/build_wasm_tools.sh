#!/usr/bin/env bash
# Builds the WebAssembly modules of the browser installer (sa2-nx-installer), everything except DXC (see
# link_dxc_wasm.sh; the installer uses nfsmw-nx's build of the same unmodified DXC v2025.1 and dxc_web.cpp):
#   hlsl.mjs  the container -> HLSL translator (shaders/tools/masseffect_hlsl.cpp + XenosRecomp)
#   pack.mjs  the package + index packer (shaders/tools/me_pack_shaders.cpp + the app's library code)
#   lzx.mjs   the LZX decoder of default.xex (shaders/wasm/sa2_lzx.cpp + libmspack's lzxd.c and system.c)
#   scan.mjs  (not built by default) masseffect-nx's Unreal package scanner (shaders/tools/ue3_shader_scan.cpp);
#             SA2 keeps its shaders as plain files, so the installer does not use it
# Run it from a shell where Emscripten is active (em++ on PATH: `brew install emscripten` or emsdk_env.sh).
#
#   XXHASH_DIR  folder with xxhash.h            (default: found by shaders/tools/env.sh)
#   FMT_DIR     folder that contains fmt/       (same)
#   OUT         output folder (default shaders/wasm/out)
#   usage: shaders/wasm/build_wasm_tools.sh [hlsl] [pack] [lzx] [scan]     (default: hlsl pack lzx)
#
# The translator, the packer and the decoder are plain command line programs: the page writes the inputs into
# Emscripten's in-memory file system (FS) and calls callMain([...]) with the same arguments as on the
# command line, then reads the outputs back.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/../tools/env.sh"
command -v em++ >/dev/null || { echo "em++ not found: activate Emscripten first" >&2; exit 1; }
need_xxhash; need_fmt
OUT=${OUT:-$SHADERS/wasm/out}
APP=$ROOT/app/src/me/native
mkdir -p "$OUT"
COMMON=(-sEXPORT_ES6=1 -sMODULARIZE=1 -sINVOKE_RUN=0 -sEXIT_RUNTIME=0 -sALLOW_MEMORY_GROWTH=1
        -sSTACK_SIZE=4MB -sEXPORTED_RUNTIME_METHODS=FS,callMain -sENVIRONMENT=web,worker,node)
targets=${*:-hlsl pack lzx}
MSPACK=$ROOT/sdk/thirdparty/libmspack/libmspack/mspack

for t in $targets; do
  echo "=== $t"
  case $t in
    hlsl)
      em++ -std=c++20 -O1 -I"$SHADERS" -I"$XENOS" -I"$FMT_DIR" -I"$XXHASH_DIR" -include "$XENOS/pch_min.h" \
        -Wno-null-arithmetic -fms-extensions -DFMT_HEADER_ONLY -DXXH_INLINE_ALL -DMASSEFFECT_RECOMP -fwasm-exceptions \
        "$SHADERS/tools/masseffect_hlsl.cpp" "$XENOS/shader_recompiler.cpp" -o "$OUT/hlsl.mjs" \
        -sEXPORT_NAME=createHlslModule "${COMMON[@]}";;
      # usf4-nx: -fwasm-exceptions, so a container the translator rejects with an exception is skipped as in the
      # native build (USF4's package has one), instead of aborting the whole module.
    pack)
      # The full library (30k shaders, ~900 MB) needs the whole 4 GB address space of wasm32.
      em++ -std=c++20 -O2 -I"$XXHASH_DIR" "$SHADERS/tools/me_pack_shaders.cpp" "$APP/sa2_shader_library.cpp" \
        -o "$OUT/pack.mjs" -sEXPORT_NAME=createPackModule -sMAXIMUM_MEMORY=4GB "${COMMON[@]}";;
    lzx)
      emcc -O2 -I"$MSPACK" -c "$MSPACK/lzxd.c" -o "$OUT/lzxd.o"
      emcc -O2 -I"$MSPACK" -c "$MSPACK/system.c" -o "$OUT/system.o"
      em++ -std=c++20 -O2 -I"$MSPACK" "$SHADERS/wasm/sa2_lzx.cpp" "$OUT/lzxd.o" "$OUT/system.o" -o "$OUT/lzx.mjs" \
        -sEXPORT_NAME=createLzxModule "${COMMON[@]}"
      rm -f "$OUT/lzxd.o" "$OUT/system.o";;
    scan)
      em++ -std=c++20 -O2 "$SHADERS/tools/ue3_shader_scan.cpp" -o "$OUT/scan.mjs" \
        -sEXPORT_NAME=createScanModule "${COMMON[@]}";;
    *) echo "unknown target $t" >&2; exit 1;;
  esac
done
echo "done: $OUT"
