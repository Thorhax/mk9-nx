#!/usr/bin/env bash
# Pure CPU translator regression (no GPU, no game files): links XenosRecomp's shader_recompiler.cpp.
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/../tools/env.sh"
need_xxhash; need_fmt
mkdir -p "$OUT_TOOLS"
EXTRA=()
"$CXX" -x c++ -std=c++20 -fexperimental-library -fsyntax-only /dev/null 2>/dev/null && EXTRA+=(-fexperimental-library)
"$CXX" -std=c++20 -O2 "${EXTRA[@]}" -Wno-switch -Wno-unused-variable -Wno-null-arithmetic -fms-extensions \
  -include "$XENOS/pch_min.h" -I"$SHADERS" -I"$XENOS" -I"$FMT_DIR" -I"$XXHASH_DIR" \
  -DFMT_HEADER_ONLY -DXXH_INLINE_ALL -DMASSEFFECT_RECOMP \
  "$SHADERS/tests/test_shader_alu_parallel_codegen.cpp" "$XENOS/shader_recompiler.cpp" \
  -o "$OUT_TOOLS/test_shader_alu_parallel_codegen"
"$OUT_TOOLS/test_shader_alu_parallel_codegen"
