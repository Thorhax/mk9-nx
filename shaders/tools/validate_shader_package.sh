#!/usr/bin/env bash
# Builds and runs validate_shader_package (every SPIR-V of a package through SPIRV-Tools).
#   usage: shaders/tools/validate_shader_package.sh <package.mesp>
# Needs SPIRV-Tools with a pkg-config file (brew install spirv-tools).
set -e
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_xxhash
mkdir -p "$OUT_TOOLS"
c++ -std=c++23 -O2 \
  -I"$XXHASH_DIR" \
  $(pkg-config --cflags SPIRV-Tools) \
  "$SHADERS/tools/validate_shader_package.cpp" \
  "$ROOT/app/src/me/native/sa2_shader_library.cpp" \
  $(pkg-config --libs SPIRV-Tools) \
  -o "$OUT_TOOLS/validate_shader_package"
exec "$OUT_TOOLS/validate_shader_package" "$@"
