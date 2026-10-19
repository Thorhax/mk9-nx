#!/usr/bin/env bash
# Builds the standalone helper tools (out/tools/):
#   extract_shader_package   package.mesp -> loose containers (vs_/ps_<XXH3>.bin)
#   find_shader_microcode    find the container holding a microcode hash
#   repair_vertex_variant_declarations   offline DECL repair of a vertex container
#   ue3_shader_scan          collect containers straight from the game's Unreal packages
#   usage: shaders/tools/build_extra_tools.sh [tool...]   (default: all)
set -eu
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
need_xxhash
mkdir -p "$OUT_TOOLS"
LIB="$ROOT/app/src/me/native/sa2_shader_library.cpp"
tools=${*:-extract_shader_package find_shader_microcode repair_vertex_variant_declarations ue3_shader_scan}
for t in $tools; do
  case $t in
    extract_shader_package) "$CXX" -std=c++20 -O2 -I"$XXHASH_DIR" "$SHADERS/tools/$t.cpp" "$LIB" -o "$OUT_TOOLS/$t";;
    find_shader_microcode) "$CXX" -std=c++20 -O2 -I"$XXHASH_DIR" "$SHADERS/tools/$t.cpp" -o "$OUT_TOOLS/$t";;
    repair_vertex_variant_declarations) "$CXX" -std=c++20 -O2 "$SHADERS/tools/$t.cpp" -o "$OUT_TOOLS/$t";;
    ue3_shader_scan) "$CXX" -std=c++20 -O2 "$SHADERS/tools/$t.cpp" -o "$OUT_TOOLS/$t";;
    *) echo "unknown tool $t" >&2; exit 1;;
  esac
  echo "$OUT_TOOLS/$t"
done
