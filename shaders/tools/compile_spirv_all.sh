#!/usr/bin/env bash
# DXC + spirv-val for every HLSL file, 8 in parallel.
#   usage: shaders/tools/compile_spirv_all.sh <hlsl dir> <spirv dir>
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_dxc
IN=$1; OUT=$2; mkdir -p "$OUT"
export DXC VAL=$SPIRV_VAL OUT=$(cd "$OUT" && pwd)
find "$IN" -name "*.hlsl" -print0 | xargs -0 -n 1 -P 8 "$SHADERS/tools/compile_spirv_one.sh"
echo "spv: $(find "$OUT" -name '*.spv' | wc -l | tr -d ' '), failed: $(find "$OUT" -name '*.err' | wc -l | tr -d ' ')"
