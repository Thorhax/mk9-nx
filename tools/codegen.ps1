# Translates the game (Default.xex) to C++ with the host code generator. It builds nothing: build the generator first
# (tools\win\run.ps1 out\host_build.log "cmake --build out/host --target rexglue").
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\codegen.ps1
#
# Input:   assets\game_root\Default.xex  (tools\extract_iso.py puts it there)
#          app\usf4_manifest.toml         (project, game paths, includes: overrides.toml)
# Output:  app\generated\                 (C++ of the game and generated\rexglue.cmake; not committed)
#          out\codegen.log                (log of the generator)
# Windows counterpart of tools/codegen.sh (classic codegen; the optimised flavour of sa2-nx is not set up for USF4).
param(
    [string]$Rexglue = ''
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$app = Join-Path $root 'app'
if (-not $Rexglue) {
    # The newest generator: rexglue.exe (host build) or rexgluerd.exe (the PC tree's RelWithDebInfo build).
    $Rexglue = Get-ChildItem (Join-Path $root 'sdk\out') -Recurse -Include rexglue.exe, rexgluerd.exe -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Rexglue -or -not (Test-Path $Rexglue)) { throw 'rexglue.exe not found; build it first (see the header)' }
$xex = Join-Path $root 'assets\game_root\Default.xex'
if (-not (Test-Path $xex)) { throw "$xex not found; extract your disc with tools\extract_iso.py first" }

New-Item -ItemType Directory -Force (Join-Path $root 'out') | Out-Null
$log = Join-Path $root 'out\codegen.log'
Write-Host "== codegen with $Rexglue"
Push-Location $app
try {
    $env:REX_MAX_JUMP_TABLE_ENTRIES = '1024'
    & $env:ComSpec /c "`"$Rexglue`" codegen usf4_manifest.toml > `"$log`" 2>&1"
    $code = $LASTEXITCODE
} finally { Pop-Location }
Get-Content $log -Tail 30
if ($code -ne 0) { throw "codegen failed (exit $code), see out\codegen.log" }
Write-Host "done: $app\generated"
