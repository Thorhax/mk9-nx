#!/usr/bin/env bash
# One Training-mode fight on the PC build for performance A/B runs (no frame capture): the training script on the
# seconds clock (the frame clock and vblank_per_swap of lockstep_run.sh inflate the ring timings ~5x), native renderer, then the fight's report lines are kept in out/pc/fight/LABEL.log.
#
#   tools/linux/fight_run.sh LABEL [SECONDS] [-- extra --cvar=value ...]
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
label=$1; shift
seconds=${1:-200}; [ $# -gt 0 ] && shift
[ "${1:-}" = "--" ] && shift
mkdir -p "$root/out/pc/fight"
script="$(cat "$root/out/pc/training_script.txt")"
VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-/usr/share/vulkan/icd.d/nvidia_icd.json} \
  "$root/tools/linux/boot_test.sh" -t "$seconds" -- --usf4_input_script="$script" \
  --sa2_renderer_native=true "$@" > /dev/null 2>&1
cp "$root/out/pc/boot.log" "$root/out/pc/fight/$label.log"
grep "ring thread:" "$root/out/pc/fight/$label.log" | tail -3 | sed 's/.*ring thread: /ring: /'
grep "draw stages" "$root/out/pc/fight/$label.log" | tail -1 | sed 's/.*draw stages/stages/'
