#!/usr/bin/env bash
# Links dxc_web.cpp with DXC compiled to WebAssembly: the HLSL to SPIR-V compiler of the browser installer.
# UNTESTED for this repository (see shaders/README.md, "WebAssembly"): the steps are the ones that
# produced the installer's DXC for another game of this family, kept here as a recipe.
#
# DXC is microsoft/DirectXShaderCompiler v2025.1 (commit 75a029d95e767f291885e081f71ed951acad0019), unmodified:
#   1. Build DXC natively for the host once, to get llvm-tblgen and clang-tblgen.
#   2. Configure a WebAssembly build from a shell where Emscripten is active:
#        emcmake cmake -G Ninja -S $DXC_SRC -B $DXC_BUILD -C $DXC_SRC/cmake/caches/PredefinedParams.cmake \
#          -DCMAKE_BUILD_TYPE=Release -DENABLE_SPIRV_CODEGEN=ON -DLLVM_TARGETS_TO_BUILD=None \
#          -DLLVM_ENABLE_THREADS=OFF -DLLVM_ENABLE_EH=ON -DLLVM_ENABLE_RTTI=ON -DLLVM_INCLUDE_TESTS=OFF \
#          -DCLANG_INCLUDE_TESTS=OFF -DHLSL_INCLUDE_TESTS=OFF -DSPIRV_BUILD_TESTS=OFF \
#          -DLLVM_TABLEGEN=<host build>/bin/llvm-tblgen -DCLANG_TABLEGEN=<host build>/bin/clang-tblgen \
#          -DCMAKE_C_FLAGS=-Wno-error \
#          "-DCMAKE_CXX_FLAGS=-Wno-invalid-specialization -Wno-error \
#            -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -D_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR \
#            -D_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS"
#   3. ninja -C $DXC_BUILD dxcompiler
#   4. This script: set DXC_SRC, DXC_BUILD (and optionally OUT), then run it from the same shell.
set -eu
: "${DXC_SRC:?set DXC_SRC to the DirectXShaderCompiler source tree}"
: "${DXC_BUILD:?set DXC_BUILD to its WebAssembly build folder}"
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT=${OUT:-$HERE/out}
mkdir -p "$OUT"
OBJS=("$DXC_BUILD"/tools/clang/tools/dxcompiler/CMakeFiles/dxcompiler.dir/*.o)
LIBS=("$DXC_BUILD"/lib/*.a)
em++ -O3 -std=gnu++17 -DNDEBUG -D_GNU_SOURCE -D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS -D__STDC_LIMIT_MACROS \
  -DENABLE_SPIRV_CODEGEN -D_LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION -D_LIBCPP_ENABLE_CXX17_REMOVED_AUTO_PTR \
  -D_LIBCPP_ENABLE_CXX20_REMOVED_TYPE_TRAITS -Wno-invalid-specialization -fms-extensions \
  "-I$DXC_SRC/include" "-I$DXC_BUILD/include" "-I$DXC_SRC/external/DirectX-Headers/include/directx" \
  "-I$DXC_SRC/external/DirectX-Headers/include/wsl/stubs" "$HERE/dxc_web.cpp" "${OBJS[@]}" \
  -Wl,--start-group "${LIBS[@]}" -Wl,--end-group -o "$OUT/dxc_web.mjs" \
  -sEXPORT_ES6=1 -sMODULARIZE=1 -sEXPORT_NAME=createDxcModule -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=128MB \
  -sSTACK_SIZE=8MB -sEXPORTED_FUNCTIONS=_compile -sEXPORTED_RUNTIME_METHODS=FS,ccall -sENVIRONMENT=web,worker,node
echo "done: $OUT/dxc_web.mjs"
