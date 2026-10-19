# Profile-guided optimisation, step 2 of 3 (2026-10-08). Step 1 is the instrumented NRO (build_switch.ps1 with
# -Pgo generate), played on the console: it writes pgo/sa2_pgo_<N>.stream next to the NRO (one per session, see
# app/src/sa2_pgo.cpp). Copy that pgo folder to the PC and run this: every session's counts are added into the .gcda
# files of the profile folder (gcov-tool merge-stream), then their summary is filled in (tools/pgo_summary.py: the
# stream API writes it as zero). Step 3: build_switch.ps1 -Pgo use.
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\pgo_merge.ps1 <folder with the .stream files> [-Out <profile folder>]
#
# The profile folder is pgo\ at the root of the repository (the build's SA2_PGO_DIR): one .gcda per object, named by
# the object's path in out\nx with # for each \ (GCC's names with -fprofile-prefix-path). The streams name the files
# they fill: an instrumented NRO built since then names the profile folder of the PC that built it (merge on that PC);
# the v1.0.0 instrumented NRO named the files next to the objects of out\nx, which this script moves to the profile
# folder under those names.
#
# merge-stream ADDS to .gcda files already there: a second run with the same streams would count them twice, so it
# stops when the profile folder or out\nx already has .gcda files (move them aside first; pgo\ holds the released
# NRO's profile).
param([Parameter(Mandatory = $true, Position = 0)][string]$Streams, [string]$Out = '')
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$build = Join-Path $root 'out\nx'
if (-not $Out) { $Out = Join-Path $root 'pgo' }
$tool = 'C:\devkitPro\devkitA64\bin\aarch64-none-elf-gcov-tool.exe'
if (-not (Test-Path $tool)) { throw "gcov-tool not found: $tool" }
$files = Get-ChildItem -Path $Streams -Filter 'sa2_pgo_*.stream' -File | Sort-Object Name
if (-not $files) { throw "no sa2_pgo_*.stream in $Streams" }
New-Item -ItemType Directory -Force $Out | Out-Null
$Out = (Resolve-Path $Out).Path
if (Get-ChildItem -Path $Out -Filter *.gcda -File | Select-Object -First 1) {
    throw "$Out already has .gcda files: move them aside first (merge-stream would add to them)"
}
if ((Test-Path $build) -and (Get-ChildItem -Path $build -Filter *.gcda -Recurse -File | Select-Object -First 1)) {
    throw "out\nx already has .gcda files: move them aside first (merge-stream would add to them)"
}
foreach ($f in $files) {
    "merging $($f.Name) ($([math]::Round($f.Length / 1MB, 1)) MB)"
    & $tool merge-stream $f.FullName
    if ($LASTEXITCODE) { throw "gcov-tool merge-stream failed on $($f.FullName)" }
}
# Streams of the v1.0.0 instrumented NRO: their files are next to the objects of out\nx.
$moved = 0
if (Test-Path $build) {
    foreach ($g in Get-ChildItem -Path $build -Filter *.gcda -Recurse -File) {
        $relative = $g.FullName.Substring($build.Length + 1)
        Move-Item -LiteralPath $g.FullName -Destination (Join-Path $Out ($relative -replace '\\', '#'))
        $moved++
    }
}
if ($moved) { "$moved file(s) moved from out\nx to $Out" }
$count = (Get-ChildItem -Path $Out -Filter *.gcda -File).Count
if (-not $count) { throw "no .gcda file in $Out after merging: the streams name another folder (merge on the PC that built the instrumented NRO)" }
& python (Join-Path $root 'tools\pgo_summary.py') $Out $files.Count
if ($LASTEXITCODE) { throw 'pgo_summary.py failed' }
"$($files.Count) session(s) merged into $count file(s) in $Out"
