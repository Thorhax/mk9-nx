#!/bin/sh
# metric.sh <net.log>: posdiag bad vertices (not finite + beyond 1e7) and window draws checked, summed over the run
grep "posdiag\] [0-9]* vertices not finite" "$1" | sed 's/.*posdiag\] //' | awk '{bad += $1 + $5; gsub(/[()]/, ""); for (i = 1; i <= NF; i++) if ($i == "window" && $(i+1) == "draws") w += $(i-1)} END {printf "bad vertices %d, window draws %d, bad per 1000 window draws %.1f\n", bad, w, w ? bad * 1000 / w : 0}'
