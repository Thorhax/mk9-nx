#!/usr/bin/env bash
# Containers -> HLSL -> SPIR-V -> package + index (masseffect_shaders.mesp and .mesp.idx).
#   usage: shaders/tools/build_shader_spirv.sh [containers=run/shaders] [out=run/shaderlib]
# <out> must not exist. Result: <out>/hlsl, <out>/spirv, <out>/masseffect_shaders.mesp and .idx,
# plus translate.log, dxc.log (the .err files of failed shaders are kept in <out>/spirv).
# Each container is translated in its own process (a crash in one does not stop the rest) and DXC runs
# 8 files in parallel. If any shader fails the package is NOT written, unless ALLOW_FAILURES=1
# (a full disc scan always has a few dozen containers the translator cannot handle, see README).
set -u
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_xxhash; need_dxc
cd "$ROOT"
IN=${1:-run/shaders}; OUT=${2:-run/shaderlib}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
"$SHADERS/tools/build_xenos_hlsl.sh" >/dev/null || exit 1
python3 "$SHADERS/tools/translate_all.py" "$IN" "$OUT/hlsl" | tee "$OUT/translate.log"
"$SHADERS/tools/compile_spirv_all.sh" "$OUT/hlsl" "$OUT/spirv" | tee "$OUT/dxc.log"
bad=$(find "$OUT/spirv" -name '*.err' | wc -l | tr -d ' ')
translate_bad=$(wc -l < "$OUT/translate_failed.txt" | tr -d ' ')
echo "containers: $(find "$IN" -name '*.bin' | wc -l | tr -d ' '), translation failures: $translate_bad, SPIR-V failures: $bad"
# Pack (library format described in shaders/README.md; key = XXH3 of the original container).
"$CXX" -std=c++20 -O2 -I"$XXHASH_DIR" "$SHADERS/tools/me_pack_shaders.cpp" \
  "$ROOT/app/src/me/native/sa2_shader_library.cpp" -o "$OUT_TOOLS/me_pack_shaders" || exit 1
if [ "$bad" -ne 0 ] || [ "$translate_bad" -ne 0 ]; then
  [ "${ALLOW_FAILURES:-0}" = 1 ] || { echo "not packing: some shaders failed (ALLOW_FAILURES=1 packs the rest)"; exit 1; }
  echo "ALLOW_FAILURES=1: packing without the failed ones (listed in $OUT/translate_failed.txt and the .err files)"
fi
"$OUT_TOOLS/me_pack_shaders" "$IN" "$OUT/spirv" "$OUT/masseffect_shaders.mesp"
