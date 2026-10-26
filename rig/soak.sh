#!/bin/sh
# soak.sh <nro name> <label> <seconds> [config]: one boot into an Arcade ladder (ladder.sh's start), then fight.py for
# <seconds> without relaunching (it mashes through win screens and continues), screenshots every ~3 s; then copies the
# net log, the SD's own log and rex_profile.log. Read <MK9_LOGS>/<label>/ with soakstat.py. NONET=1 as in ladder.sh.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
put_cfg "$4"
before=$(newest_net)
launch_nro $1
sh "$R/tofight.sh"
after=$(newest_net)
if [ -z "$NONET" ] && [ "$before" = "$after" ]; then echo "NO NEW LOG: game did not start"; exit 1; fi
O=$L/$2; mkdir -p "$O/shots"
echo "net log: net/$after"
SAFE=1 python3 "$R/fight.py" $3 "$O/shots/s" > /dev/null 2>&1
[ -z "$NONET" ] && cp "$NET/$after" "$O/net.log"
close_title
[ -z "$NONET" ] && cp "$NET/$after" "$O/net.log"
ftp get /switch/mk9-nx/logs/rex/rex_profile.log "$O/rex_profile.log"
# The SD's own log (the net log broke once: every line a spdlog "Invalid argument" over the network)
ftp get /switch/mk9-nx/logs/$(newest_sdlog) "$O/sd.log"
[ -n "$NONET" ] && cp "$O/sd.log" "$O/net.log"
echo done
