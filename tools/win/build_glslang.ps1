# Builds glslangValidator (host) from the SDK's glslang sources into out\glslang: the native renderer's EDRAM transfer
# shaders (app/src/me/native/shaders/*.comp|frag|vert) are compiled to SPIR-V initializers with it at build time.
# clang-cl, not clang++: glslang adds -Wl,-undefined,error for GNU-style drivers, which lld-link rejects.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\build_glslang.ps1
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$run = Join-Path $PSScriptRoot 'run.ps1'
& powershell -NoProfile -ExecutionPolicy Bypass -File $run out\glslang_configure.log ("cmake -S sdk/thirdparty/glslang -B out/glslang -G Ninja " +
    "-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=clang-cl -DCMAKE_CXX_COMPILER=clang-cl -DENABLE_OPT=OFF " +
    "-DGLSLANG_TESTS=OFF -DENABLE_GLSLANG_BINARIES=ON -DENABLE_HLSL=OFF -DGLSLANG_ENABLE_INSTALL=OFF " +
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5") | Select-Object -Last 2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& powershell -NoProfile -ExecutionPolicy Bypass -File $run out\glslang_build.log "cmake --build out/glslang --target glslangValidator -j 6" | Select-Object -Last 2
exit $LASTEXITCODE
