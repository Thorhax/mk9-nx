#!/usr/bin/env bash
# Lays out the Switch SD card folder in out/nx/sdcard/switch/mk9-nx: the NRO and the shader package, plus a README.
# The game files are yours and stay out of it: copy your extracted disc (assets/game_root, WITHOUT its $SystemUpdate
# folder, which the game never reads) to sdmc:/switch/mk9-nx/game_root. Counterpart of sa2-nx's
# tools/win/package_switch.ps1.
#
#   tools/linux/package_switch.sh            -> out/nx/sdcard/switch/mk9-nx and out/nx/mk9-nx-switch.zip
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
nro="$root/out/nx/mk9.nro"
[ -f "$nro" ] || { echo "$nro missing: build the Switch target first (docs/building.md, Switch)" >&2; exit 1; }
for f in sa2_shaders.mesp sa2_shaders.mesp.idx; do
  [ -f "$root/out/shaders/$f" ] || { echo "out/shaders/$f missing: build the shader package first" >&2; exit 1; }
done
dest="$root/out/nx/sdcard/switch/mk9-nx"
rm -rf "$root/out/nx/sdcard"
mkdir -p "$dest"
cp "$nro" "$root/out/shaders/sa2_shaders.mesp" "$root/out/shaders/sa2_shaders.mesp.idx" "$dest/"
cat > "$dest/README.txt" <<'TXT'
mk9-nx: Mortal Kombat Komplete Edition (Xbox 360) on the Switch, test build.

Put this folder on the SD card as sdmc:/switch/mk9-nx, and copy your own extracted disc next to the NRO:
    sdmc:/switch/mk9-nx/game_root/      <- assets\game_root from the PC build, without its $SystemUpdate folder
                                           (default.xex, Asset\, Config\, Localization\, Movies\, ... ; about 7.3 GB)

Launch with a title takeover (hold R while starting a game, then open the Homebrew Menu), or a forwarder with full
memory. Not from the Album: it has too little memory.

The first start writes mk9.toml (your options) next to the NRO and keeps a log in logs/ (send it back after a test).
The first time each scene is drawn, the renderer compiles its pipelines: short stutters, saved in cache/ for the next
time.
TXT
(cd "$root/out/nx/sdcard" && rm -f ../mk9-nx-switch.zip && python3 -I -c "
import os,zipfile
with zipfile.ZipFile('../mk9-nx-switch.zip','w',zipfile.ZIP_DEFLATED) as z:
    for d,_,fs in os.walk('switch'):
        for f in fs: z.write(os.path.join(d,f))
")
ls -la "$dest" "$root/out/nx/mk9-nx-switch.zip"
