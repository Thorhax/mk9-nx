#!/bin/sh
# refresh_cache.sh: the console's pipeline list and shader cache in, every listed variant compiled on the PC
# (tools/deko3d_variant_cache.py, the console's fetch pruning included), the merged cache back as deko3d_shaders.bin.
# Closes the running title first (the game holds both files open).
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
W=$T/pcache; mkdir -p "$W"
close_title >/dev/null
ftp get /switch/mk9-nx/pipeline_list.bin "$W/list.bin" || exit 1
ftp get /switch/mk9-nx/deko3d_shaders.bin "$W/cache_in.bin" || exit 1
(cd "$P" && python3 tools/deko3d_variant_cache.py --list "$W/list.bin" --glsl out/deko3d_shaders/glsl_opt \
  --merge "$W/cache_in.bin" --out "$W/cache_out.bin" --jobs 8 | grep -v "^  [0-9]*/") || exit 1
ftp put "$W/cache_out.bin" /switch/mk9-nx/deko3d_shaders.bin && echo "cache refreshed"
