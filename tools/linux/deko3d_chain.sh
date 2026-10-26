#!/bin/bash
# mk9-nx: the deko3d shader chain for the Switch build (sa2-nx docs/renut-renderer.md, "Shaders for the Switch"), from
# the game's shaders (out/renut_shader_dump, docs/building.md) to the GLSL package the NRO embeds and the DKSH cache the
# console reads beside it. Everything it writes is made from your copy of the game and stays in the build tree.
#
#   tools/linux/deko3d_chain.sh            the whole chain (translate, GLSL, passes, package, cache)
#   tools/linux/deko3d_chain.sh cache      only the cache, from the GLSL already made
#
# Needs: out/pc/xenos_recomp (the PC build's renut_xenos_recomp target) and out/deko3d_tools (tools/deko3d_build_tools.sh).
set -e
cd "$(dirname "$0")/../.."
if [ "$1" != cache ]; then
  rm -rf out/deko3d_shaders/spirv out/deko3d_shaders/glsl out/deko3d_shaders/glsl_opt
  mkdir -p out/deko3d_shaders/spirv
  echo "== translate $(date)"
  BANJO_DEKO3D_SPIRV_DIR=out/deko3d_shaders/spirv BANJO_HLSL_OVERRIDE=out/pc/renut_gpu/early_alpha_hlsl \
    out/pc/xenos_recomp/build/XenosRecomp/XenosRecomp out/renut_shader_dump out/deko3d_shaders/spirv_pass.cpp \
    app/thirdparty/reblue-XenosRecomp/XenosRecomp/shader_common.h > out/deko3d_shaders/translate.log 2>&1 || true
  ls out/deko3d_shaders/spirv | wc -l
  echo "== glsl $(date)"
  python3 tools/deko3d_shaders.py --spirv out/deko3d_shaders/spirv --out out/deko3d_shaders/glsl --no-uam 2>&1 | tail -5
  echo "== passes + uam check $(date)"
  python3 tools/deko3d_glsl_opt.py --in out/deko3d_shaders/glsl/glsl --out out/deko3d_shaders/glsl_opt --check 2>&1 | tail -8
  echo "== package $(date)"
  python3 tools/deko3d_shader_package.py --glsl out/deko3d_shaders/glsl_opt --out app/out/renut_switch 2>&1 | tail -5
  python3 tools/renut_switch_shader_cache.py --pc out/pc 2>&1 | tail -3
fi
# The DKSH cache: every variant of the repository's pipeline list compiled from this GLSL (a few minutes), so a first
# session has no shader compiles where the list reaches. Goes on the SD card beside mk9.nro as deko3d_shaders.bin
# (tools/linux/package_switch.sh puts it there).
echo "== cache $(date)"
python3 tools/deko3d_variant_cache.py --list shaders/deko3d_pipeline_list.xz --glsl out/deko3d_shaders/glsl_opt \
  --out out/deko3d_shaders/deko3d_shaders.bin 2>&1 | grep -v "^  [0-9]*/"
echo "== done $(date)"
