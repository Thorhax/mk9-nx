#!/usr/bin/env bash
# Builds the container -> HLSL translator (XenosRecomp + masseffect_hlsl.cpp) for the host.
# Output: out/tools/xenos_hlsl
#   usage: shaders/tools/build_xenos_hlsl.sh
#   then:  out/tools/xenos_hlsl <containers> <hlsl out> shaders/XenosRecomp/shader_common.h
# fmt (header only) and xxHash are found by env.sh.
set -e
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_xxhash; need_fmt
mkdir -p "$OUT_TOOLS"
EXTRA=()
# libc++ needs this for std::execution; other standard libraries do not know the flag.
"$CXX" -x c++ -std=c++20 -fexperimental-library -fsyntax-only /dev/null 2>/dev/null && EXTRA+=(-fexperimental-library)
"$CXX" -std=c++20 -O2 "${EXTRA[@]}" -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions \
  -include "$XENOS/pch_min.h" -I"$SHADERS" -I"$XENOS" -I"$FMT_DIR" -I"$XXHASH_DIR" \
  -DFMT_HEADER_ONLY -DXXH_INLINE_ALL -DMASSEFFECT_RECOMP \
  "$SHADERS/tools/masseffect_hlsl.cpp" "$XENOS/shader_recompiler.cpp" -o "$OUT_TOOLS/xenos_hlsl"
echo "$OUT_TOOLS/xenos_hlsl"
