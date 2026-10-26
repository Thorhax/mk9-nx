#!/bin/sh
# krypt.sh <nro> <label> <seconds> [config]: boot, Extras, Krypt, then walk and look around with the sticks only (no
# confirm presses: in the Krypt confirm buys items with the save's Koins), screenshots every walk; then close the game
# and fetch the SD log (NONET style). For the pipeline list.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
put_cfg "$4"
launch_nro $1 > /dev/null
sleep 35
O=$L/$2; mkdir -p "$O"
# PLUS skips the last boot movie, confirm passes the title; 1 PLAYER: EXTRAS is the sixth entry; Extras opens on KRYPT;
# the Krypt's intro is skipped with confirm.
sb "click PLUS" "sleep 5" "click B" "sleep 5" "click DDOWN" "sleep 0.5" "click DDOWN" "sleep 0.5" "click DDOWN" "sleep 0.5" "click DDOWN" \
  "sleep 0.5" "click DDOWN" "sleep 0.5" "click B" "sleep 5" "click B" "sleep 15" "click B" "sleep 4"
t0=$(date +%s); n=0
while [ $(( $(date +%s) - t0 )) -lt $3 ]; do
  n=$((n+1))
  # Turns of different lengths so the walk does not end facing a wall.
  turn=$(( (n % 3) + 1 ))
  sb "setStick LEFT 0 0x7FFF" "sleep 5" "setStick LEFT 0 0" "setStick RIGHT 0x7FFF 0" "sleep $turn" "setStick RIGHT 0 0" >/dev/null
  sb shot "$O/k_$(printf %02d $n).jpg"
done
close_title
ftp get /switch/mk9-nx/logs/$(newest_sdlog) "$O/net.log"
fps_line "$O/net.log" 12
