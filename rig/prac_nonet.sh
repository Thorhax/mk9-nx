#!/bin/sh
# prac_nonet.sh: practice.sh with NONET=1 (no network log; fps from the SD log). Same arguments.
NONET=1 exec sh "$(dirname "$0")/practice.sh" "$@"
