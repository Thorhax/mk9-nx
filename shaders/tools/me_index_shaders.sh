#!/usr/bin/env bash
# Builds me_index_shaders (and the host index test) and writes <package>.idx for each package given.
#   usage: shaders/tools/me_index_shaders.sh <package.mesp>...        -> <package.mesp>.idx next to each
#   test:  out/tools/test_shader_index <package.mesp> <package.mesp.idx> [--all] [--other <idx of another>]
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_xxhash
mkdir -p "$OUT_TOOLS"
for tool in tools/me_index_shaders tests/test_shader_index; do
  "$CXX" -std=c++20 -O2 -I"$XXHASH_DIR" "$SHADERS/$tool.cpp" \
    "$ROOT/app/src/me/native/sa2_shader_library.cpp" -o "$OUT_TOOLS/$(basename "$tool")"
done
for package in "$@"; do
  "$OUT_TOOLS/me_index_shaders" "$package"
done
