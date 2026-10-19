# Lays out the Switch SD card folder in out\nx\sdcard\switch\sa2-nx: the NRO, its settings (app\sa2_switch.toml as
# sa2.toml), the testing notes and the shader package. The game files are yours and stay out of the repository:
#   -IncludeGame              adds them as the asset cache (assets\game_root.nxpack + .nxidx, built here with
#                             tools\build_asset_pack.py when missing or older than the folder) plus out\pc\dlc
#                             (the Battle DLC);
#   -IncludeGame -LooseGame   copies the game_root folder instead: the console builds the pack on its first boot.
# By hand:
#   sdmc:/switch/sa2-nx/game_root.nxpack + game_root.nxidx   <- python tools\build_asset_pack.py assets\game_root
#   sdmc:/switch/sa2-nx/dlc/                                 <- out\pc\dlc (optional: SA2 Battle)
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\win\package_switch.ps1 [-IncludeGame [-LooseGame]] [-AbPair]
param([switch]$IncludeGame, [switch]$LooseGame, [switch]$AbPair)
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$dest = Join-Path $root 'out\nx\sdcard\switch\sa2-nx'
New-Item -ItemType Directory -Force $dest | Out-Null
$nro = Join-Path $root 'out\nx\sa2.nro'
if (-not (Test-Path $nro)) { throw "no NRO at ${nro}: run tools\win\build_switch.ps1 first" }
Copy-Item -Force $nro (Join-Path $dest 'sa2.nro')
Copy-Item -Force (Join-Path $root 'app\sa2_switch.toml') (Join-Path $dest 'sa2.toml')
Copy-Item -Force (Join-Path $root 'docs\switch-testing.md') (Join-Path $dest 'README.md')
foreach ($f in 'sa2_shaders.mesp', 'sa2_shaders.mesp.idx') {
    Copy-Item -Force (Join-Path $root "out\shaders\$f") (Join-Path $dest $f)
}
# Since 2026-10-06 only the optimised-codegen NRO is delivered (the user's choice): build it from the optimised working
# tree and package WITHOUT -AbPair. The old codegen A/B pair, kept for reference: with -AbPair, sa2.nro is
# out\nx\ab\classic\sa2.nro and sa2_optimised.nro is out\nx\ab\optimised\sa2.nro.
if ($AbPair) {
    Copy-Item -Force (Join-Path $root 'out\nx\ab\classic\sa2.nro') (Join-Path $dest 'sa2.nro')
    Copy-Item -Force (Join-Path $root 'out\nx\ab\optimised\sa2.nro') (Join-Path $dest 'sa2_optimised.nro')
} else {
    Remove-Item -Force (Join-Path $dest 'sa2_optimised.nro') -ErrorAction SilentlyContinue
}
# Premade enhancement configurations (all off, CPU off, verify, ...): docs/enhancements.md. The folder is emptied first:
# a config that no longer exists (e.g. round 2's sa2_test_new_natives_on.toml) must not stay behind with stale keys.
Remove-Item -Force (Join-Path $dest 'configs\*.toml') -ErrorAction SilentlyContinue
python (Join-Path $root 'tools\make_enhancement_configs.py') --out (Join-Path $dest 'configs') | Out-Null
# The CPU sampler is switched by the configuration now (switch_profile_stacks in the A/B and profile configs): no
# logs/rex/stacks_profile.flag, which made every run pay for sampling.
Remove-Item -Force (Join-Path $dest 'logs\rex\stacks_profile.flag') -ErrorAction SilentlyContinue
if ($IncludeGame) {
    if ($LooseGame) {
        robocopy (Join-Path $root 'assets\game_root') (Join-Path $dest 'game_root') /E /NFL /NDL /NJH /NJS | Out-Null
        foreach ($f in 'game_root.nxpack', 'game_root.nxidx') {
            Remove-Item -Force (Join-Path $dest $f) -ErrorAction SilentlyContinue
        }
    } else {
        & python (Join-Path $root 'tools\build_asset_pack.py') (Join-Path $root 'assets\game_root') --if-stale
        if ($LASTEXITCODE) { throw 'asset pack build failed' }
        foreach ($f in 'game_root.nxpack', 'game_root.nxidx') {
            Copy-Item -Force (Join-Path $root "assets\$f") (Join-Path $dest $f)
        }
        # A loose folder from an earlier package is no longer read (the pack wins): drop it from the staging copy.
        if (Test-Path (Join-Path $dest 'game_root')) { Remove-Item -Recurse -Force (Join-Path $dest 'game_root') }
    }
    if (Test-Path (Join-Path $root 'out\pc\dlc')) {
        robocopy (Join-Path $root 'out\pc\dlc') (Join-Path $dest 'dlc') /E /NFL /NDL /NJH /NJS | Out-Null
    }
}
Get-ChildItem $dest | Select-Object Name, Length
