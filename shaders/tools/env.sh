# Common settings of the shader tools. Sourced by the other scripts; sets:
#   ROOT        the repository root (everything is relative to it)
#   SHADERS     $ROOT/shaders
#   XENOS       $ROOT/shaders/XenosRecomp
#   OUT_TOOLS   $ROOT/out/tools (built binaries)
#   CXX         C++ compiler (default clang++)
#   XXHASH_DIR  folder with xxhash.h        (env XXHASH_DIR; else thirdparty/xxHash next to the repo, else brew)
#   FMT_DIR     folder that contains fmt/   (env FMT_DIR;     else thirdparty/fmt/include next to the repo, else brew)
#   DXC, SPIRV_VAL   DXC and spirv-val      (env DXC, SPIRV_VAL; else PATH, else ~/VulkanSDK/*/macOS/bin)
# Dependencies on macOS: brew install xxhash fmt spirv-tools   (and the LunarG Vulkan SDK or a dxc binary).
# A sibling checkout of the ReXGlue SDK (../rexglue-sdk/thirdparty/{xxHash,fmt}) is also found.
SHADERS_ENV_DIR=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
SHADERS=$(cd "$SHADERS_ENV_DIR/.." && pwd)
ROOT=$(cd "$SHADERS/.." && pwd)
XENOS=$SHADERS/XenosRecomp
OUT_TOOLS=$ROOT/out/tools
CXX=${CXX:-clang++}

_first_dir_with() {  # _first_dir_with <file> <dir>...
  local f=$1; shift
  for d in "$@"; do [ -n "$d" ] && [ -e "$d/$f" ] && { echo "$d"; return; }; done
}
_brew=$(command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null || true)
XXHASH_DIR=${XXHASH_DIR:-$(_first_dir_with xxhash.h "$ROOT/../rexglue-sdk/thirdparty/xxHash" "$ROOT/thirdparty/xxHash" "${_brew:+$_brew/include}" /usr/local/include /usr/include)}
FMT_DIR=${FMT_DIR:-$(_first_dir_with fmt/format.h "$ROOT/../rexglue-sdk/thirdparty/fmt/include" "$ROOT/thirdparty/fmt/include" "${_brew:+$_brew/include}" /usr/local/include /usr/include)}

_sdkbin=$(ls -d "$HOME"/VulkanSDK/*/macOS/bin 2>/dev/null | tail -1)
DXC=${DXC:-$(command -v dxc || true)}
[ -z "$DXC" ] && [ -n "$_sdkbin" ] && DXC=$_sdkbin/dxc
SPIRV_VAL=${SPIRV_VAL:-$(command -v spirv-val || true)}
[ -z "$SPIRV_VAL" ] && [ -n "$_sdkbin" ] && SPIRV_VAL=$_sdkbin/spirv-val

need_xxhash() { [ -n "$XXHASH_DIR" ] || { echo "xxhash.h not found: set XXHASH_DIR (brew install xxhash)" >&2; exit 1; }; }
need_fmt()    { [ -n "$FMT_DIR" ]    || { echo "fmt/format.h not found: set FMT_DIR (brew install fmt)" >&2; exit 1; }; }
need_dxc()    { [ -x "${DXC:-}" ] && [ -x "${SPIRV_VAL:-}" ] || { echo "dxc / spirv-val not found: set DXC and SPIRV_VAL (Vulkan SDK or brew install spirv-tools)" >&2; exit 1; }; }
