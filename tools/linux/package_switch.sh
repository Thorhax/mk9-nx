#!/usr/bin/env bash
# Lays out the Switch SD card folder in out/nx/sdcard/switch/mk9-nx: the NRO (the deko3d renderer's GLSL is inside it),
# the DKSH cache tools/linux/deko3d_chain.sh compiled, the pipeline list (so the game builds the listed pipelines at
# start), plus a README. The game files are yours and stay out of it: copy your extracted disc (assets/game_root,
# WITHOUT its $SystemUpdate folder, which the game never reads) to sdmc:/switch/mk9-nx/game_root. Counterpart of
# sa2-nx's tools/win/package_switch.ps1.
#
#   tools/linux/package_switch.sh [build dir, default out/nx-dk-lto]   -> out/nx/sdcard/switch/mk9-nx and out/nx/mk9-nx-switch.zip
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${1:-$root/out/nx-dk-lto}"
nro="$build/mk9.nro"
cache="$root/out/deko3d_shaders/deko3d_shaders.bin"
[ -f "$nro" ] || { echo "$nro missing: build the Switch target first (docs/building.md, Switch)" >&2; exit 1; }
[ -f "$cache" ] || { echo "$cache missing: tools/linux/deko3d_chain.sh cache" >&2; exit 1; }
dest="$root/out/nx/sdcard/switch/mk9-nx"
rm -rf "$root/out/nx/sdcard"
mkdir -p "$dest"
cp "$nro" "$cache" "$dest/"
python3 -I -c "import lzma,sys; open(sys.argv[2],'wb').write(lzma.open(sys.argv[1]).read())" \
  "$root/shaders/deko3d_pipeline_list.xz" "$dest/pipeline_list.bin"
cat > "$dest/README.txt" <<'TXT'
mk9-nx: Mortal Kombat Komplete Edition (Xbox 360) on the Switch, test build.

Put this folder on the SD card as sdmc:/switch/mk9-nx, and copy your own extracted disc next to the NRO:
    sdmc:/switch/mk9-nx/game_root/      <- assets\game_root from the PC build, without its $SystemUpdate folder
                                           (default.xex, Asset\, Config\, Localization\, Movies\, ... ; about 7.3 GB)

Launch with a title takeover (hold R while starting a game, then open the Homebrew Menu), or a forwarder with full
memory. Not from the Album: it has too little memory.

The first start writes mk9.toml (your options) next to the NRO and keeps a log in logs/ (send it back after a test).
deko3d_shaders.bin holds the shaders already compiled for the scenes the project has played; a scene it does not have
is compiled the first time it is drawn (scenery can be missing for a moment) and added to it for the next time.
TXT
(cd "$root/out/nx/sdcard" && rm -f ../mk9-nx-switch.zip && python3 -I -c "
import os,zipfile
with zipfile.ZipFile('../mk9-nx-switch.zip','w',zipfile.ZIP_DEFLATED) as z:
    for d,_,fs in os.walk('switch'):
        for f in fs: z.write(os.path.join(d,f))
")
ls -la "$dest" "$root/out/nx/mk9-nx-switch.zip"
