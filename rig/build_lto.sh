#!/bin/sh
# build_lto.sh <out name> <glsl dir name>: build.sh on out/nx-dk-lto (configured with MK9_SWITCH_LTO=ON).
R=$(cd "$(dirname "$0")" && pwd); P=$(cd "$R/.." && pwd)
cd "$P" || exit 1
python3 tools/deko3d_shader_package.py --glsl out/deko3d_shaders/$2 --out app/out/renut_switch >/dev/null 2>&1 || exit 1
docker run --rm -v "$P:$P" -w "$P" -e DEVKITPRO=/opt/devkitpro devkitpro/devkita64 bash -c "cmake --build out/nx-dk-lto -j ${JOBS:-6}" > out/build_$1.log 2>&1
rc=$?
[ "$2" != glsl_opt ] && python3 tools/deko3d_shader_package.py --glsl out/deko3d_shaders/glsl_opt --out app/out/renut_switch >/dev/null 2>&1
if [ $rc -ne 0 ]; then grep -E "error" out/build_$1.log | cut -c1-300 | head -8; echo "BUILD FAILED"; exit 1; fi
cp out/nx-dk-lto/mk9.nro out/$1.nro && cp out/nx-dk-lto/mk9 out/$1.elf && echo "built out/$1.nro"
