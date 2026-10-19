# A/B of native renderer settings under the same machine load: runs the game alternately with configuration A and B
# (each a comma-separated list of cvar=value, applied as REX_ environment variables), Rounds times each, and prints
# the game's Swaps and draws per 10 s from the log of every run. Alternating cancels most of the drift of other
# programs' load (the machine is shared).
#
#   powershell -File tools\win\ab_cvars.ps1 -A "" -B "sa2_native_edram4_clear_color0=false,..." -Seconds 40 -Rounds 2
param(
    [string]$A = '',
    [string]$B = '',
    [int]$Seconds = 40,
    [int]$Rounds = 2
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
function Set-Config([string]$list, [bool]$on) {
    foreach ($item in ($list -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })) {
        $name, $value = $item.Split('=', 2)
        if ($on) { Set-Item -Path "env:REX_$($name.ToUpper())" -Value $value }
        else { Remove-Item -Path "env:REX_$($name.ToUpper())" -ErrorAction SilentlyContinue }
    }
}
for ($round = 1; $round -le $Rounds; ++$round) {
    foreach ($label in 'A', 'B') {
        $list = if ($label -eq 'A') { $A } else { $B }
        $env:REX_SA2_RENDERER_NATIVE = 'true'
        Set-Config $list $true
        powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\boot_test.ps1 -Seconds $Seconds -Tail 0 | Out-Null
        Set-Config $list $false
        $env:REX_SA2_RENDERER_NATIVE = $null
        $log = Get-ChildItem out\pc\logs -Filter *.log | Sort-Object LastWriteTime | Select-Object -Last 1
        $lines = Select-String -Path $log.FullName -Pattern '\] ([0-9.]+) s: ([0-9]+) Swaps, ([0-9]+) draws'
        $swaps = ($lines | ForEach-Object { $_.Matches[0].Groups[2].Value }) -join ' '
        "round $round $label  swaps per report: $swaps   ($($log.Name))"
    }
}
