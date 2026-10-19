# Plays the PC build in a normal window with sound: for testing by hand on a real screen, with a controller
# (any XInput / SDL pad) or the keyboard. Not muted, not minimized.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\play.ps1            native Vulkan renderer
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\play.ps1 -Emulated  ReXGlue's emulated GPU
#   ... -Exe out\pc\usf4.exe   another build
#
# The native renderer needs sa2_shaders.mesp + sa2_shaders.mesp.idx next to the exe. The log (with "[fps]" lines
# every 5 s) is written to logs\ next to the exe; the end of it is printed when the game is closed.
param(
    [switch]$Emulated,
    [string]$Exe = 'out\pc\usf4.exe'
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exePath = (Resolve-Path (Join-Path $root $Exe)).Path
$dir = Split-Path $exePath
if (-not $Emulated -and -not (Test-Path (Join-Path $dir 'sa2_shaders.mesp'))) {
    throw "sa2_shaders.mesp is not next to $exePath (copy the shader package there, or use -Emulated)"
}
$env:REX_AUDIO_MUTE = 'false'
# Keyboard as a controller (the SDK's mnk driver is off by default): Enter = Start, Space = A, Backspace = B, L = X,
# P = Y, 3 = RB, E = RT, WASD = left stick, Shift+arrows = d-pad (docs/building.md, "Playing").
$env:REX_MNK_MODE = 'true'
$env:REX_WINDOW_START_MINIMIZED = 'false'
$env:REX_SA2_RENDERER_NATIVE = if ($Emulated) { 'false' } else { 'true' }
$started = Get-Date
$p = Start-Process -FilePath $exePath -WorkingDirectory $dir -PassThru
$p.WaitForExit()
Remove-Item env:REX_AUDIO_MUTE, env:REX_MNK_MODE, env:REX_WINDOW_START_MINIMIZED, env:REX_SA2_RENDERER_NATIVE -ErrorAction SilentlyContinue
$log = Get-ChildItem (Join-Path $dir 'logs') -Filter *.log -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -ge $started } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) {
    "== $($log.FullName)"
    Select-String -Path $log.FullName -Pattern '\[fps\]' | Select-Object -Last 12 | ForEach-Object { $_.Line }
    Get-Content $log.FullName -Tail 15
}
