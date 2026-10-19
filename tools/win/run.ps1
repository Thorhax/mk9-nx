# Runs one command line in the Windows build environment (tools\win\env.ps1) from the repository root and writes its
# whole output to a log file. Windows PowerShell 5.1 stops a native command at its first stderr line when redirected
# with *>, so the redirect is done by cmd.exe instead.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\run.ps1 out\build.log "cmake --build out\pc"
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Log,
    [Parameter(Mandatory = $true, Position = 1)][string]$Cmd
)
. (Join-Path $PSScriptRoot 'env.ps1')
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
New-Item -ItemType Directory -Force (Split-Path $Log) | Out-Null
& $env:ComSpec /c "$Cmd > `"$Log`" 2>&1"
$code = $LASTEXITCODE
Get-Content $Log -Tail 40
"exit=$code"
exit $code
