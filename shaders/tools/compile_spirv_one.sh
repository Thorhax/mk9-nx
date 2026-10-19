#!/bin/sh
# One HLSL file -> SPIR-V, validated with spirv-val (called by compile_spirv_all.sh; DXC, VAL and OUT
# are in the environment). A failure leaves <name>.err and no .spv.
f=$1; b=$(basename "$f" .hlsl)
[ -f "$OUT/$b.spv" ] && exit 0
case $b in ps_*) t=ps_6_6; x="";; *) t=vs_6_6; x=-fvk-invert-y;; esac
if "$DXC" -spirv -T $t -E main -HV 2021 -fspv-target-env=vulkan1.2 -fvk-use-dx-layout \
     -Werror=parameter-usage $x -Fo "$OUT/$b.spv" "$f" >/dev/null 2>"$OUT/$b.err" &&
   "$VAL" --target-env vulkan1.2 --scalar-block-layout "$OUT/$b.spv" 2>>"$OUT/$b.err"; then
  rm -f "$OUT/$b.err"
else
  rm -f "$OUT/$b.spv"
fi
