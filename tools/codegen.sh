#!/usr/bin/env bash
# Translates default.xex to C++ (Linux counterpart of tools/codegen.ps1).
#   tools/codegen.sh [--flavor classic|optimised]   -> app/generated, log in out/codegen.log
# --flavor (default: keep what app/codegen_flavor.toml says): optimised copies app/perf_overrides.toml to
# app/codegen_flavor.toml (registers as locals, no MSR lock); classic empties it. Then tools/post_codegen.py runs the
# flavour's post-codegen steps (no-volatile pch + loop barriers + direct calls for optimised); the build appends the
# same script to its own codegen command (app/CMakeLists.txt).
# REXGLUE overrides the generator path (default: sdk/out/linux-amd64/rexglue, built from out/host).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
flavor=""
if [ "${1:-}" = "--flavor" ]; then flavor="${2:?--flavor classic|optimised}"; fi
case "$flavor" in
  optimised) { echo "# optimised codegen: written by tools/codegen.sh --flavor optimised from perf_overrides.toml."; \
               cat "$root/app/perf_overrides.toml"; } > "$root/app/codegen_flavor.toml" ;;
  classic) printf '# classic codegen: no extra options (tools/codegen.sh --flavor classic writes this file; --flavor optimised\n# copies perf_overrides.toml here).\n' > "$root/app/codegen_flavor.toml" ;;
  "") ;;
  *) echo "unknown flavor $flavor" >&2; exit 1 ;;
esac
rexglue="${REXGLUE:-$root/sdk/out/linux-amd64/rexglue}"
[ -x "$rexglue" ] || { echo "rexglue not found at $rexglue; build it: cmake --build out/host --target rexglue" >&2; exit 1; }
[ -f "$root/assets/game_root/default.xex" ] || { echo "assets/game_root/default.xex missing; run tools/extract_iso.py" >&2; exit 1; }
mkdir -p "$root/out"
cd "$root/app"
if ! REX_MAX_JUMP_TABLE_ENTRIES=1024 "$rexglue" codegen mk9_manifest.toml > "$root/out/codegen.log" 2>&1; then
  tail -30 "$root/out/codegen.log"; echo "codegen failed, see out/codegen.log" >&2; exit 1
fi
tail -3 "$root/out/codegen.log"
REXSDK_DIR="$root/sdk" python3 "$root/tools/post_codegen.py"
