#!/bin/sh
# tofight.sh: from boot (after nxlink) to a ladder fight as Scorpion, Beginner
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
sleep 35
sb "click PLUS" "sleep 5" "click B" "sleep 3" "click B" "sleep 4" "click B" "sleep 2" "click DDOWN" "click DDOWN" "click B" "sleep 45"
