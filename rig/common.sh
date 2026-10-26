# common.sh: sourced by the rig scripts after they set R to their own directory. Reads rig.env (rig.env.example).
[ -f "$R/rig.env" ] || { echo "rig: copy rig/rig.env.example to rig/rig.env and fill it in" >&2; exit 1; }
. "$R/rig.env"
P=$(cd "$R/.." && pwd)
L=${MK9_LOGS:-$(cd "$P/.." && pwd)/mk9-logs/rig}
NET=$L/net
T=${RIG_TMP:-$L/tmp}
NXL=${NXLINK:-nxlink}
mkdir -p "$NET" "$T"

sb() { python3 "$R/sb.py" "$@"; }
ftp() { python3 "$R/ftp.py" "$@"; }
# Closes the running title (HOME, X, A). On an idle home menu the A launches the selected title instead.
close_title() { sb "click HOME" "sleep 1.5" "click X" "sleep 1" "click A" "sleep 4"; }
# put_cfg <name | dir>: configs/<name>/mk9.toml (or <dir>/mk9.toml) to the SD, @PC_IP@ filled in. No-op without one.
put_cfg() {
  [ -n "$1" ] || return 0
  c=$R/configs/$1/mk9.toml; [ -f "$c" ] || c=$1/mk9.toml
  sed "s/@PC_IP@/$PC_IP/g" "$c" > "$T/mk9.toml" && ftp put "$T/mk9.toml" /switch/mk9-nx/mk9.toml
}
# launch_nro <name>: closes the running title, opens hbmenu through a title takeover (R held while launching the last
# title), its netloader (Y), and sends out/<name>.nro. A netloader that won't take the connection (seen after a
# reboot): B, Y reopens it.
launch_nro() {
  close_title
  sb "press R" "sleep 0.5" "click A" "sleep 2.5" "click A" "sleep 5" "release R" "sleep 2" "click Y" "sleep 3"
  if "$NXL" -a "$SWITCH_IP" -p mk9-nx/$1.nro "$P/out/$1.nro" 2>&1 | tail -1 | grep -q failed; then
    sb "click B" "sleep 2" "click Y" "sleep 4"
    "$NXL" -a "$SWITCH_IP" -p mk9-nx/$1.nro "$P/out/$1.nro" 2>&1 | tail -1
  fi
}
newest_net() { ls -t "$NET" 2>/dev/null | head -1; }
# The SD's newest game log, by number (mk9_<n>.log)
newest_sdlog() { ftp ls /switch/mk9-nx/logs 2>/dev/null | awk '/mk9_[0-9]+\.log/{print $NF}' | sort -t_ -k2 -n | tail -1; }
# fps_line <log> [n]: the last n (6) "fps over" averages
fps_line() { echo "fps: $(grep -o '[0-9.]* fps over' "$1" | tail -${2:-6} | cut -d' ' -f1 | tr '\n' ' ')"; }
