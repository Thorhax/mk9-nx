#!/bin/sh
# practice.sh <nro name in out/, no .nro> <label> <fight secs> <arena: DRIGHT presses from Challenge Tower> [config]
# Closes the running title, relaunches <nro> via takeover + netloader, goes Training > Practice, picks the arena,
# Scorpion vs Sub-Zero, fights with screenshots, then copies the net log to <MK9_LOGS>/<label>/net.log and prints the
# posdiag metric and fps.
# NONET=1 (a config without log_network): the game is closed after the fight and the SD's newest log is fetched
# instead; a run that wrote no new SD log (a crash) fails.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
put_cfg "$5"
before=$(newest_net)
[ -n "$NONET" ] && sdbefore=$(newest_sdlog)
launch_nro $1
sleep 45
after=$(newest_net)
if [ -z "$NONET" ] && [ "$before" = "$after" ]; then echo "NO NEW LOG: game did not start"; exit 1; fi
sb "click PLUS" "sleep 6" "click DDOWN" "click DDOWN" "click DDOWN" "click B" "sleep 3" "click DDOWN" "click DDOWN" "click B" "sleep 6" "click X" "sleep 3"
n=0; while [ $n -lt $4 ]; do sb "click DRIGHT" "sleep 0.6"; n=$((n+1)); done
O=$L/$2; mkdir -p "$O"
sb "sleep 1" shot "$O/arena.jpg" 2>/dev/null; sb shot "$O/arena.jpg"
sb "click B" "sleep 2" "click B" "sleep 2" "click B" "sleep 85"
python3 "$R/fight.py" $3 "$O/f" > /dev/null 2>&1
if [ -n "$NONET" ]; then
  close_title
  sdlog=$(newest_sdlog)
  if [ "$sdlog" = "$sdbefore" ]; then echo "NO NEW LOG (newest is still $sdlog): run invalid"; exit 1; fi
  ftp get /switch/mk9-nx/logs/$sdlog "$O/net.log"
  fps_line "$O/net.log" 12
else
  cp "$NET/$after" "$O/net.log"
  sh "$R/metric.sh" "$O/net.log"
  fps_line "$O/net.log"
fi
