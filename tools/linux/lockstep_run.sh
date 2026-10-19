#!/usr/bin/env bash
# One lockstep recording for frame-by-frame A/B comparisons (the guide's "pixel-identical" check):
# vblank_per_swap (one guest vblank per presented frame), the training script on the game-frame clock, and sa2-nx's
# recorder hashing every frame from game frame FIRST for COUNT frames (XXH3 of the full-resolution image).
#
#   tools/linux/lockstep_run.sh LABEL [FIRST [COUNT]] [-- extra --cvar=value ...]
#     -> out/pc/lockstep/LABEL.hash (one line per frame) and LABEL.log
# Compare two: diff out/pc/lockstep/A.hash out/pc/lockstep/B.hash (identical = pixel-identical frames).
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
label=$1; shift
first=${1:-5000}; [ $# -gt 0 ] && shift
count=${1:-300}; [ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
mkdir -p "$root/out/pc/lockstep"
rm -f "$root"/out/pc/captures/video_*
script="$(cat "$root/out/pc/training_script.txt")"
: > "$root/out/pc/boot.log"
VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json} \
  "$root/tools/linux/boot_test.sh" -t "${LOCKSTEP_SECONDS:-420}" -- --usf4_input_script="$script" \
  --usf4_input_script_frames=true --vblank_per_swap=true --sa2_renderer_native=true \
  --sa2_video_s=5 --sa2_video_frame="$first" --sa2_video_frames="$count" "$@" > /dev/null 2>&1
h=$(ls "$root"/out/pc/captures/video_*.hash 2>/dev/null | head -1)
cp "$root/out/pc/boot.log" "$root/out/pc/lockstep/$label.log"
if [ -n "$h" ]; then
  cp "$h" "$root/out/pc/lockstep/$label.hash"
  cp "${h%.hash}.rgba" "$root/out/pc/lockstep/$label.rgba"; cp "${h%.hash}.txt" "$root/out/pc/lockstep/$label.txt"
  echo "$label: $(( $(stat -c%s "$h") / 8 )) frames hashed"
else echo "$label: no hash file (see out/pc/lockstep/$label.log, [capture] lines)"; fi
