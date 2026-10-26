#!/bin/sh
# ladder.sh <nro name in out/> <label> <fight secs> [config]: relaunch <nro>, Arcade Ladder fight as Scorpion
# (tofight.sh), fight with screenshots, copy the net log, print fps.
# NONET=1 (a config without log_network): no net log; the SD's newest logs/mk9_*.log is fetched after the fight instead.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
put_cfg "$4"
before=$(newest_net)
launch_nro $1
sh "$R/tofight.sh"
after=$(newest_net)
if [ -z "$NONET" ] && [ "$before" = "$after" ]; then echo "NO NEW LOG: game did not start"; exit 1; fi
O=$L/$2; mkdir -p "$O"
python3 "$R/fight.py" $3 "$O/f" > /dev/null 2>&1
if [ -n "$NONET" ]; then
  close_title
  ftp get /switch/mk9-nx/logs/$(newest_sdlog) "$O/net.log"
else
  cp "$NET/$after" "$O/net.log"
fi
python3 "$R/sheet.py" "$O/sheet.jpg" "$O"/f_*.jpg
fps_line "$O/net.log"
