# Console test rig

Scripts that drive a Switch over the LAN to launch builds, play scripted fights and collect logs. Every measurement in
`docs/mk9-notes.md` (fps, late frames, soaks, profiles) came from them.

## What the console needs

- Atmosphère with **sys-botbase** (buttons, screenshots, memory reads). It has no authentication: LAN only, never
  port-forward it.
- An **FTP server** on the SD (ftpsrv or similar) for configs and logs.
- **hbmenu**, opened through a title takeover (R held while launching a title), with its netloader for `nxlink`.
- The game installed as in the main README, under `/switch/mk9-nx/`.

## Setup

```sh
cp rig/rig.env.example rig/rig.env   # Git ignores rig.env: the console address, FTP login and PC address go here
python3 rig/loglisten.py 28773 ../mk9-logs/rig/net &   # network logs (configs with log_network), one thread per connection
```

Runs land in `MK9_LOGS` (default `mk9-logs/rig` beside the checkout), one folder per label.

## Scripts

| Script | What it does |
|---|---|
| `build.sh <name> glsl_opt` / `build_lto.sh` | Package the shaders, build `out/nx-dk` (or `out/nx-dk-lto`) in devkitPro's Docker image, copy to `out/<name>.nro` |
| `open_hbmenu.sh` | From an idle home menu, open hbmenu through a takeover. Run it first when nothing is running: the other scripts start by closing the running title (HOME, X, A), and on an idle home menu that A launches the title normally |
| `ladder.sh <name> <label> <secs> [config]` | Relaunch `out/<name>.nro`, Arcade Ladder as Scorpion, fight, copy the log, print fps |
| `practice.sh <name> <label> <secs> <arena> [config]` | Training > Practice on an arena (`<arena>` right presses from Challenge Tower; 8 is the lava stage), fight, posdiag metric and fps |
| `soak.sh <name> <label> <secs> [config]` | One boot, a long ladder (win screens mashed through), then the net log, SD log and `rex_profile.log`; read with `soakstat.py <dir>` |
| `bootloop.sh <name> <boots> [config]` | Boot repeatedly and report hangs |
| `sweep.sh <name> <first> <last>` | `practice.sh` over a range of arenas, a contact sheet each |
| `refresh_cache.sh` | Fetch the console's pipeline list and shader cache, compile every listed variant on the PC, put the merged cache back |
| `rdcheck.py <guest base>` | Compare the running game's read-only data with `out/default_image.bin` (checks `tools/ro_constants.py`) |
| `proffn.py <elf> <rex_profile.log> [min] [top]` | Per-thread self time by function from a profiling run |
| `sb.py`, `ftp.py`, `step.sh` | sys-botbase commands and screenshots; FTP put/get/ls/rm; a command plus a screenshot |

`NONET=1` before `ladder.sh`, `practice.sh` (or `prac_nonet.sh`) and `soak.sh` takes the log from the SD after
closing the game instead of from the network, for configs without `log_network` (the network log changes timing).

## Configs

`configs/<name>/mk9.toml` is uploaded as `/switch/mk9-nx/mk9.toml`; `@PC_IP@` is replaced with `PC_IP` from `rig.env`.
A path to any folder holding a `mk9.toml` works too.

| Config | |
|---|---|
| `KN` | The test baseline: no MSAA, relaxed vsync, Xbox button layout, sideways Joy-Con |
| `KNNL` | KN plus the network log to this PC |
| `KNP` | KN plus the sampling profiler (`rex_profile.log`) |
| `KNCV` | KN plus the store-card check (every cached copy also hashed; misses logged) |
| `KNC0` | KN with store cards off |
| `KNCC` | KN plus the native constant copy checked against the guest code |
| `KNBV` | KN plus the native bone-matrix copy checked against the guest code |

## Cautions

- The game holds its files open (its log, the shader cache): close it before fetching them over FTP.
- Don't touch the console's clocks or anything key-related on the SD from these scripts.
