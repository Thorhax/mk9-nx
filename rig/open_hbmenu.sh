#!/bin/sh
# open_hbmenu.sh: from an idle home menu (nothing running: after a reboot, or with the title closed), hbmenu through a
# title takeover (R held while launching the selected title). Run it before the other scripts in that state: they
# start by closing the running title (HOME, X, A), and on an idle home menu that A launches the title normally.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
sb "press R" "sleep 0.5" "click A" "sleep 2.5" "click A" "sleep 5" "release R" "sleep 3"
