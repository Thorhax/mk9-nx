#!/bin/sh
# build.sh <out name> <glsl dir name, e.g. glsl_diag2 | glsl_opt>: package the shaders, build out/nx-dk in devkitPro's
# devkita64 image, copy to out/<name>.nro and .elf; restores the glsl_opt package after. Exits non-zero (and
# copies nothing) when the build fails.
R=$(cd "$(dirname "$0")" && pwd); P=$(cd "$R/.." && pwd)
cd "$P" || exit 1
python3 tools/deko3d_shader_package.py --glsl out/deko3d_shaders/$2 --out app/out/renut_switch >/dev/null 2>&1 || exit 1
docker run --rm -v "$P:$P" -w "$P" -e DEVKITPRO=/opt/devkitpro devkitpro/devkita64 bash -c "cmake --build out/nx-dk -j ${JOBS:-6}" > out/build_$1.log 2>&1
rc=$?
[ "$2" != glsl_opt ] && python3 tools/deko3d_shader_package.py --glsl out/deko3d_shaders/glsl_opt --out app/out/renut_switch >/dev/null 2>&1
if [ $rc -ne 0 ]; then grep -E "error" out/build_$1.log | cut -c1-300 | head -8; echo "BUILD FAILED"; exit 1; fi
cp out/nx-dk/mk9.nro out/$1.nro && cp out/nx-dk/mk9 out/$1.elf && echo "built out/$1.nro"
