#!/bin/sh
# launch.sh <nro on PC> <sd path under /switch>: hbmenu title takeover, netloader, nxlink
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
sb "press R" "sleep 0.5" "click A" "sleep 2.5" "click A" "sleep 5" "release R" "sleep 1" "click Y" "sleep 2"
"$NXL" -a "$SWITCH_IP" -p "$2" "$1" 2>&1 | tail -2
