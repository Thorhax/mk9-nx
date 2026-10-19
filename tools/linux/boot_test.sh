#!/usr/bin/env bash
# One boot test of the PC build on Linux: the game runs on its own hidden X display (Xvfb), muted, for a fixed time,
# then the end of the log is shown. It never touches the desktop (the window is shown only on the hidden display). Counterpart of tools/win/boot_test.ps1.
#
#   tools/linux/boot_test.sh [-t SECONDS] [-s "S1 S2 ..."] [-- extra --cvar=value ...]
#     -t  run time in seconds (default 45)
#     -s  snapshots of the game image at these seconds after start (out/pc/captures/snap_<s>s.png; taken by the
#         game from the presenter's guest output, cvar mk9_snapshot_s)
#
# The SDK's shared libraries are loaded from sdk/out/linux-amd64. Log: out/pc/boot.log (the whole run). Exit status: 0 if the game was still running at the end (then stopped),
# otherwise the game's own exit status.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
exe="$root/out/pc/mk9"
seconds=45
shots=""
while getopts "t:s:" opt; do
  case $opt in
    t) seconds=$OPTARG ;;
    s) shots=$OPTARG ;;
    *) sed -n 2,10p "$0"; exit 2 ;;
  esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift
[ -x "$exe" ] || { echo "$exe not built (cmake --build out/pc)" >&2; exit 2; }

log="$root/out/pc/boot.log"
mkdir -p "$root/out/pc/captures"
# A free display number for the hidden X server.
disp=99
while [ -e "/tmp/.X11-unix/X$disp" ] || [ -e "/tmp/.X$disp-lock" ]; do disp=$((disp + 1)); done
Xvfb ":$disp" -screen 0 1280x720x24 -nolisten tcp > "$root/out/pc/xvfb.log" 2>&1 &
xvfb=$!
trap 'kill $xvfb 2>/dev/null' EXIT
for _ in $(seq 50); do [ -e "/tmp/.X11-unix/X$disp" ] && break; sleep 0.1; done

# The guest memory is a /dev/shm/xenia_memory_* file of ~4.8 GB; a run that is killed leaves it behind and a full
# /dev/shm makes the next start die with SIGBUS. Note the ones that exist now and remove this run's afterwards.
shm_before="$(ls /dev/shm 2>/dev/null | grep '^xenia_memory_' || true)"
cd "$root/out/pc"
snap_args=()
[ -n "$shots" ] && snap_args=(--mk9_snapshot_s="$(echo $shots | tr ' ' ',')")
DISPLAY=":$disp" LD_LIBRARY_PATH="$root/sdk/out/linux-amd64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$exe" --audio_mute=true --window_start_minimized=false --log_file="$log" "${snap_args[@]}" "$@" > "$root/out/pc/stdout.log" 2>&1 &
game=$!
start=$(date +%s)
status=0
while kill -0 $game 2>/dev/null; do
  if [ $(( $(date +%s) - start )) -ge "$seconds" ]; then
    kill $game; sleep 2; kill -9 $game 2>/dev/null
    echo "== stopped after ${seconds}s (still running)"
    break
  fi
  sleep 1
done
if ! wait $game 2>/dev/null; then status=$?; fi
[ $(( $(date +%s) - start )) -lt "$seconds" ] && echo "== the game exited by itself after $(( $(date +%s) - start ))s, status $status"
for f in "$root"/out/pc/captures/snap_*s.ppm; do
  [ -e "$f" ] && ffmpeg -loglevel error -y -i "$f" "${f%.ppm}.png" && rm -f "$f"
done
for f in $(ls /dev/shm 2>/dev/null | grep '^xenia_memory_' || true); do
  echo "$shm_before" | grep -qx "$f" || rm -f "/dev/shm/$f"
done
echo "== end of $log"
tail -n 40 "$log" 2>/dev/null || tail -n 40 "$root/out/pc/stdout.log"
exit 0
