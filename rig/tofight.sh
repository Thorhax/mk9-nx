#!/bin/sh
# tofight.sh: from boot (after nxlink) to an Arcade Ladder fight. Menu: PLUS, then FIGHT, Arcade Ladder; the fighter
# select's cursor starts on Scorpion (top left). By default "down, down, confirm" picks Johnny Cage (every measurement
# before 2026-10-26 was made so) and the fight's mashing leaves the difficulty menu. PICK="DRIGHT DDOWN ..." moves the
# cursor from Scorpion instead, confirms, and picks Beginner (other fighters for the pipeline list).
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
sleep 35
sb "click PLUS" "sleep 5" "click B" "sleep 3" "click B" "sleep 4" "click B" "sleep 2"
if [ -z "$PICK" ]; then
  sb "click DDOWN" "click DDOWN" "click B" "sleep 45"
else
  for k in $PICK; do sb "click $k" "sleep 0.4"; done
  sb "click B" "sleep 2" "click DDOWN" "click DDOWN" "click B" "sleep 45"
fi
