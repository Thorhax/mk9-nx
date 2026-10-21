#!/bin/sh
# mk9-nx (2026-10-21): runs a tool built on this Linux host (DXC, SPIRV-Cross, uam: glibc 2.38+) from inside the
# devkitPro container (glibc 2.36) for the Switch build: the host's root is mounted at /host (docker -v /:/host:ro) and
# the tool runs under the host's own loader and libraries. Outside the container it simply runs the tool.
#   hostrun.sh <tool> [args...]
tool="$1"; shift
if [ -x /host/lib64/ld-linux-x86-64.so.2 ] && [ ! -e /.mk9-host ]; then
  exec /host/lib64/ld-linux-x86-64.so.2 --library-path "$(dirname "$tool")/../lib:/host/lib/x86_64-linux-gnu:/host/usr/lib/x86_64-linux-gnu" "$tool" "$@"
fi
exec "$tool" "$@"
