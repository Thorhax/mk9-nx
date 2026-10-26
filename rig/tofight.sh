#!/bin/sh
# tofight.sh: from boot (after nxlink) to a fight. PLUS skips the last boot movie, the first confirm passes the title
# screen, then 1 PLAYER (FIGHT, STORY MODE, CHALLENGE TOWER, ...).
#   MODE unset: FIGHT, Arcade Ladder; the select cursor starts on Scorpion (top left). By default "down, down, confirm"
#     picks Johnny Cage (every measurement before 2026-10-26 was made so) and the fight's mashing leaves the difficulty
#     menu. PICK="DRIGHT DDOWN ..." moves the cursor from Scorpion instead, confirms, and picks Beginner.
#   MODE=tag: FIGHT, Tag Ladder: the fighter under the cursor (PICK moves it), the one to its right, Beginner.
#   MODE=tower: CHALLENGE TOWER, the current challenge (its instructions confirmed).
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
sleep 35
sb "click PLUS" "sleep 5"
case "$MODE" in
  tower)
    sb "click B" "sleep 5" "click DDOWN" "sleep 0.6" "click DDOWN" "sleep 0.6" "click B" "sleep 8" "click B" "sleep 12" \
      "click B" "sleep 3"
    exit 0 ;;
  tag)
    # Title, FIGHT, then down: Tag Ladder.
    sb "click B" "sleep 3" "click B" "sleep 4" "click DDOWN" "sleep 0.6" "click B" "sleep 2"
    for k in $PICK; do sb "click $k" "sleep 0.4"; done
    sb "click B" "sleep 1.5" "click DRIGHT" "sleep 0.4" "click B" "sleep 2" "click DDOWN" "click DDOWN" "click B" "sleep 45"
    exit 0 ;;
esac
sb "click B" "sleep 3" "click B" "sleep 4" "click B" "sleep 2"
if [ -z "$PICK" ]; then
  sb "click DDOWN" "click DDOWN" "click B" "sleep 45"
else
  for k in $PICK; do sb "click $k" "sleep 0.4"; done
  sb "click B" "sleep 2" "click DDOWN" "click DDOWN" "click B" "sleep 45"
fi
