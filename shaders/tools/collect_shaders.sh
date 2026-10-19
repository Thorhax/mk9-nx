#!/usr/bin/env bash
# Collect shader containers (app/src/me_shader_dump.cpp) by booting straight into each map,
# headless, N seconds each. Output: run/shaders (game data, local only).
#   usage: shaders/tools/collect_shaders.sh BIOA_PRO00 BIOA_STA00 ...
# Environment (all required):
#   GAME_BIN       host (PC) build of the game that understands --headless
#   MAP_ROOT       folder with one game root per map (MAP_ROOT/<map>)
#   MAP_ROOT_CMD   command that prepares MAP_ROOT/<map> for one map and prints a line MASSEFFECT_COALESCED_SHA1=<sha>
. "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$ROOT"
secs=${COLLECT_SECONDS:-90}
D=$PWD/run/shaders
for map in "$@"; do
  out=$("${MAP_ROOT_CMD:?set MAP_ROOT_CMD}" "$map")
  sha=$(echo "$out" | grep MASSEFFECT_COALESCED_SHA1 | cut -d= -f2)
  before=$(ls $D 2>/dev/null | wc -l)
  MASSEFFECT_SHADER_DUMP=$D MASSEFFECT_COALESCED_SHA1=$sha "${GAME_BIN:?set GAME_BIN}" --headless \
    --game_data_root="${MAP_ROOT:?set MAP_ROOT}/$map" --user_data_root="$PWD/run/user-maps" \
    --cache_root="$PWD/run/cache-maps" --metadata_root="$PWD/run/metadata" \
    --log_file="$PWD/run/logs/collect-$map.log" >/dev/null 2>&1 & pid=$!
  sleep $secs
  kill $pid 2>/dev/null
  for i in $(seq 10); do kill -0 $pid 2>/dev/null || break; sleep 1; done
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  echo "$map: $(( $(ls $D | wc -l) - before )) new, $(ls $D | wc -l) total"
done
