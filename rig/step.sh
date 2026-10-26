#!/bin/sh
# step.sh <shotname> <commands...>: run sys-botbase commands, then a screenshot to <RIG_TMP>/<shotname>.jpg
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
n=$1; shift
[ $# -gt 0 ] && sb "$@"
sb shot "$T/$n.jpg" && echo "$T/$n.jpg"
