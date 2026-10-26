#!/bin/sh
# bootloop.sh <nro name in out/, no .nro> <boots> [config]: launches the NRO <boots> times (takeover + netloader),
# waits 50 s each, and reports any boot whose net log has a [hang] line (mk9_hang_dump_s, the RexGmFaultIn hang).
# Hung boots' logs and rex_crash.log go to <MK9_LOGS>/bootloop/<n>/.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
put_cfg "$3"
O=$L/bootloop; mkdir -p "$O"
i=1; hangs=0
while [ $i -le $2 ]; do
  before=$(newest_net)
  launch_nro $1 > /dev/null
  sleep 50
  after=$(newest_net)
  if [ "$before" = "$after" ]; then echo "boot $i: no log (did not start)"; i=$((i+1)); continue; fi
  if grep -q "\[hang\]" "$NET/$after"; then
    hangs=$((hangs+1)); mkdir -p "$O/$i"; cp "$NET/$after" "$O/$i/net.log"
    echo "boot $i: HANG"; grep "\[hang\]" "$NET/$after" | cut -c1-260
    sleep 45  # the 60 s dump
    cp "$NET/$after" "$O/$i/net.log"
    close_title
    ftp get /switch/mk9-nx/logs/rex/rex_crash.log "$O/$i/rex_crash.log"
  else
    echo "boot $i: ok ($(grep -c . "$NET/$after") lines)$(grep -o '\[gm\] .*' "$NET/$after" | cut -c1-400)"
  fi
  i=$((i+1))
done
echo "hangs: $hangs of $2"
