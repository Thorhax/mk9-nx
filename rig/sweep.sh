#!/bin/sh
# sweep.sh <nro> <first> <last> [config]: practice.sh on arenas first..last (DRIGHT presses), 20 s fights, a sheet per
# arena.
R=$(cd "$(dirname "$0")" && pwd); . "$R/common.sh"
n=$2
while [ $n -le $3 ]; do
  sh "$R/practice.sh" $1 S_arena$n 20 $n ${4:-KN} > /dev/null 2>&1
  python3 "$R/sheet.py" "$L/S_arena$n/sheet.jpg" "$L/S_arena$n/arena.jpg" "$L/S_arena$n"/f_*.jpg 2>/dev/null
  echo "arena $n: $(ls "$L/S_arena$n"/f_*.jpg 2>/dev/null | wc -l) shots"
  n=$((n+1))
done
