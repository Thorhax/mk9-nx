# Runs the native renderer once per cvar override and measures one recorded moment, to find which renderer switch
# causes a difference. The metric is the luminance of the darkest recorded frame (for a window that contains a
# black moment on the console, it must reach ~0).
#
#   powershell -File tools\win\cvar_sweep.ps1 -At 13 -Frames 6 -Cvars "sa2_native_edram4_clear_alias=false,..."
# (one comma-separated string: powershell -File would split a list into separate arguments)
param(
    [int]$At = 13,
    [int]$Frames = 6,
    [string]$Cvars = '',  # empty: the baseline run only
    [string]$Baseline = 'baseline'
)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $root
$runs = @($Baseline) + ($Cvars -split ',' | ForEach-Object { $_.Trim() } | Where-Object { $_ })
foreach ($run in $runs) {
    $env:REX_SA2_RENDERER_NATIVE = 'true'
    $name = $null
    if ($run -ne $Baseline) {
        $name, $value = $run.Split('=', 2)
        Set-Item -Path "env:REX_$($name.ToUpper())" -Value $value
    }
    powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\boot_test.ps1 -VideoAt $At -VideoFrames $Frames -Seconds ($At + 10) -Tail 0 | Out-Null
    if ($name) { Remove-Item -Path "env:REX_$($name.ToUpper())" -ErrorAction SilentlyContinue }
    $env:REX_SA2_RENDERER_NATIVE = $null
    $raw = Join-Path $root "out\pc\captures\video_${At}s.rgba"
    # The darkest frame of the window: a moment that is black on the console must reach ~0; a ghost of an earlier
    # screen keeps it above. Robust to the run-to-run timing drift of a single frame.
    $stats = & python -c "import numpy as np; r=np.fromfile(r'$raw', np.uint8).reshape(-1,360,640,4)[...,:3].reshape(-1,360*640*3).mean(1); print(f'darkest {r.min():.2f} (frame {r.argmin()}), mean {r.mean():.2f}')"
    '{0,-55} {1}' -f $run, $stats
}
