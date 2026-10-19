# Starts the PC build for a few seconds, muted and minimized, stops it, and prints the end of its newest log.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\boot_test.ps1 [-Seconds 45] [-Exe out\pc\usf4.exe]
#       [-Native] [-Snapshots "10,20,30"] [-Script "20 START;26 START"] [-ScriptFile path]
# -Native        the native Vulkan renderer (sa2_renderer_native; needs sa2_shaders.mesp next to the exe)
# -Snapshots     snapshots of the game image at these seconds (usf4_snapshot_s) -> out\pc\captures\snap_<s>s.ppm
# -Script(File)  scripted gamepad on player 1 (usf4_input_script / usf4_input_script_file)
param(
    [int]$Seconds = 45,
    [string]$Exe = 'out\pc\usf4.exe',
    [int]$Tail = 40,
    [switch]$Native,
    [string]$Snapshots = '',
    [string]$Script = '',
    [string]$ScriptFile = ''
)
$env:REX_SA2_RENDERER_NATIVE = if ($Native) { 'true' } else { 'false' }
$env:REX_USF4_SNAPSHOT_S = $Snapshots
$env:REX_USF4_INPUT_SCRIPT = $Script
$env:REX_USF4_INPUT_SCRIPT_FILE = $ScriptFile
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$exePath = (Resolve-Path (Join-Path $root $Exe)).Path
$dir = Split-Path $exePath
# A test instance left over from an interrupted run keeps the exe locked (the next link fails with "permission
# denied") and competes for the GPU: stop instances of this very executable only (other trees are left alone).
Get-CimInstance Win32_Process -Filter "Name='$(Split-Path $exePath -Leaf)'" -ErrorAction SilentlyContinue |
    Where-Object { $_.ExecutablePath -eq $exePath } | ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        Wait-Process -Id $_.ProcessId -Timeout 20 -ErrorAction SilentlyContinue
    }
$started = Get-Date
# Test runs are always muted and minimized (environment beats usf4.toml), so they never take over the desktop.
$env:REX_AUDIO_MUTE = 'true'
$env:REX_WINDOW_START_MINIMIZED = 'true'
$p = Start-Process -FilePath $exePath -WorkingDirectory $dir -PassThru -WindowStyle Minimized
$p.WaitForExit($Seconds * 1000) | Out-Null
if ($p.HasExited) {
    '{0} exited after {1:n1} s with code 0x{2:X8}' -f (Split-Path $exePath -Leaf), ((Get-Date) - $started).TotalSeconds, $p.ExitCode
} else {
    $p.Refresh()
    $status = 'still running after {0} s: cpu {1:n1} s, working set {2} MB; stopped' -f $Seconds, $p.TotalProcessorTime.TotalSeconds, [math]::Round($p.WorkingSet64 / 1MB)
    # Stop it BEFORE printing anything: a caller that pipes this script into Select-Object -First N closes the pipe
    # after its first line, which ends this script there and would leave the game running.
    Stop-Process -Id $p.Id -Force
    # Wait until it is really gone: a process still tearing down keeps usf4.exe locked and the next link fails.
    if (-not $p.WaitForExit(20000)) { $status += " (warning: process $($p.Id) did not exit within 20 s)" }
    $status
}
$log = Get-ChildItem (Join-Path $dir 'logs') -Filter *.log -ErrorAction SilentlyContinue |
    Where-Object { $_.LastWriteTime -ge $started.AddSeconds(-2) } | Sort-Object LastWriteTime | Select-Object -Last 1
if ($log) {
    "== $($log.FullName) ($($log.Length) bytes)"
    Get-Content $log.FullName -Tail $Tail
} else {
    'no log written'
}
Remove-Item env:REX_SA2_RENDERER_NATIVE, env:REX_USF4_SNAPSHOT_S, env:REX_USF4_INPUT_SCRIPT, env:REX_USF4_INPUT_SCRIPT_FILE -ErrorAction SilentlyContinue
# The frame rate the game reached ([fps] lines of src/usf4_frame_counter.cpp).
if ($log) { Select-String -Path $log.FullName -Pattern '\[fps\]' | Select-Object -Last 6 | ForEach-Object { $_.Line } }
