#!/bin/bash
# Builds the native renderer's shader package (sa2_shaders.mesp + .idx) from dumped shader containers, with the
# masseffect-nx toolchain as adapted by banjo-nx (shaders/: XenosRecomp with MASSEFFECT_RECOMP, masseffect_hlsl front
# end, me_pack_shaders). Same steps as banjo-nx's tools/build_shader_library.sh.
#
#   tools/build_shader_library.sh <dump folder> [<dump folder> ...]
#
# Dump folders: the "shaders" folder the game writes next to sa2.exe with sa2_dump_shaders = true (vs_*/ps_*.bin, the
# whole container each D3DDevice_CreateVertexShader/CreatePixelShader received). In-place shader objects, if SA2 has
# any, would come as v_*/p_*.bin (banjo-nx's scanner); both are accepted.
# Output: out/shaders/sa2_shaders.mesp (+ .idx). Copy both next to sa2.exe / the NRO.
# Needs: MSYS2 clang64 clang++ (CXX), DXC (DXC; default: the one in the UnleashedRecomp-NX tree).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
SHADERS="$ROOT/shaders"; XENOS="$SHADERS/XenosRecomp"; SDK="$ROOT/sdk"
OUT="$ROOT/out/shaders"
if [ "$(uname -s)" = Linux ]; then
  # Linux (usf4-nx): the system clang++ and Microsoft's Linux DXC release unpacked into out/tools/dxc.
  CXX=${CXX:-clang++}
  DXC=${DXC:-$ROOT/out/tools/dxc/bin/dxc}
  export LD_LIBRARY_PATH="$(dirname "$DXC")/../lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
else
  CXX=${CXX:-/c/devkitPro/msys2/clang64/bin/clang++.exe}
  DXC=${DXC:-/c/devkitPro/UnleashedRecomp-NX/tools/XenosRecomp/thirdparty/dxc-bin/bin/x64/dxc.exe}
fi
JOBS=${JOBS:-8}
[ $# -ge 1 ] || { echo "usage: $0 <dump folder> [...]"; exit 1; }
rm -rf "$OUT"; mkdir -p "$OUT/in" "$OUT/hlsl" "$OUT/spirv" "$OUT/bin" "$OUT/tmp"

# Runs "$1" on every remaining argument, $JOBS at a time, with bash job control. (xargs -P is not used: Git for
# Windows' xargs stalls on finished children now and then, which hung the whole build.)
parallel() {
  local fn=$1; shift
  local running=0
  for item in "$@"; do
    "$fn" "$item" &
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then wait -n; running=$((running - 1)); fi
  done
  wait
}

echo "== building tools"
"$CXX" -std=c++20 -O2 -static -w -fms-extensions -include "$XENOS/pch_min.h" -I"$SHADERS" -I"$XENOS" \
  -I"$SDK/thirdparty/fmt/include" -I"$SDK/thirdparty/xxHash" -DFMT_HEADER_ONLY -DXXH_INLINE_ALL -DMASSEFFECT_RECOMP \
  "$SHADERS/tools/masseffect_hlsl.cpp" "$XENOS/shader_recompiler.cpp" -o "$OUT/bin/xenos_hlsl.exe" \
  > "$OUT/build_tools.log" 2>&1 || { echo "translator build failed, see $OUT/build_tools.log"; exit 1; }
"$CXX" -std=c++20 -O2 -static -w -I"$SDK/thirdparty/xxHash" -I"$ROOT/app/src/me/native" \
  "$SHADERS/tools/me_pack_shaders.cpp" "$ROOT/app/src/me/native/sa2_shader_library.cpp" \
  -o "$OUT/bin/me_pack_shaders.exe" >> "$OUT/build_tools.log" 2>&1 || { echo "packer build failed, see $OUT/build_tools.log"; exit 1; }

echo "== gathering containers (vs_/ps_ names)"
for d in "$@"; do
  for f in "$d"/*.bin; do
    [ -e "$f" ] || continue
    b=$(basename "$f" .bin)
    case "$b" in
      vs_*|ps_*) cp "$f" "$OUT/in/$b.bin" ;;
      v_*) cp "$f" "$OUT/in/vs_i${b#v_}.bin" ;;
      p_*) cp "$f" "$OUT/in/ps_i${b#p_}.bin" ;;
    esac
  done
done
echo "   $(ls "$OUT/in" | wc -l) containers"

echo "== XenosRecomp (one process per container, so one crash does not stop the rest)"
translate() {
  local f=$1 b; b=$(basename "$f" .bin)
  local t="$OUT/tmp/$b"; mkdir -p "$t/i"; cp "$f" "$t/i/"
  # A time limit per container, so one shader that hangs the translator cannot stall the whole build.
  if timeout 120 "$OUT/bin/xenos_hlsl.exe" "$t/i" "$t/o" "$XENOS/shader_common.h" > "$t/log.txt" 2>&1 && [ -f "$t/o/$b.hlsl" ]; then
    mv "$t/o/$b.hlsl" "$OUT/hlsl/"; rm -rf "$t"
  else
    echo "$b $(tail -1 "$t/log.txt")" >> "$OUT/translate_failed.txt"
  fi
}
: > "$OUT/translate_failed.txt"
parallel translate "$OUT"/in/*.bin
echo "   $(ls "$OUT/hlsl" | wc -l) translated, $(wc -l < "$OUT/translate_failed.txt") failed"

echo "== DXC"
compile() {
  local f=$1 b t x; b=$(basename "$f" .hlsl)
  case $b in ps_*) t=ps_6_6; x="";; *) t=vs_6_6; x=-fvk-invert-y;; esac
  timeout 300 "$DXC" -spirv -T $t -E main -HV 2021 -fspv-target-env=vulkan1.2 -fvk-use-dx-layout -Werror=parameter-usage $x \
    -Fo "$OUT/spirv/$b.spv" "$f" > /dev/null 2> "$OUT/spirv/$b.err" && rm -f "$OUT/spirv/$b.err" || rm -f "$OUT/spirv/$b.spv"
}
parallel compile "$OUT"/hlsl/*.hlsl
echo "   $(ls "$OUT"/spirv/*.spv 2>/dev/null | wc -l) compiled, $(ls "$OUT"/spirv/*.err 2>/dev/null | wc -l) failed"

echo "== pack (only containers that compiled)"
mkdir -p "$OUT/in_ok"
for s in "$OUT"/spirv/*.spv; do b=$(basename "$s" .spv); cp "$OUT/in/$b.bin" "$OUT/in_ok/"; done
"$OUT/bin/me_pack_shaders.exe" "$OUT/in_ok" "$OUT/spirv" "$OUT/sa2_shaders.mesp" 2>&1 | tee "$OUT/pack.log"
ls -la "$OUT"/sa2_shaders.mesp* 2>/dev/null
