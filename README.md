# mk9-nx

A static recompilation of **Mortal Kombat Komplete Edition** (Xbox 360, title 575207FD) for PC and Nintendo Switch,
made with the ReXGlue SDK. It is built on usf4-nx (Ultra Street Fighter IV), which follows the method of
[sa2-nx](https://github.com/ChanseyIsTheBest/sa2-nx) (`docs/porting-a-360-game-prompt.md`): same SDK fork, tools,
native Vulkan renderer and Switch build.

**This repository holds no game data.** You need your own copy of the game. The code generated from `default.xex`,
the shaders extracted from the game and the binaries built from them are never committed; everyone builds them from
their own disc.

## Status

Early work: see [docs/mk9-notes.md](docs/mk9-notes.md) for what is done and every finding.

## Building

As usf4-nx ([docs/building.md](docs/building.md)), with `mk9` in place of `usf4`, the disc extracted to
`assets/game_root` (the executable is `default.xex`, lower case) and the decrypted image for the analysis tools made
by `python3 tools/xex_image.py assets/game_root/default.xex out/default_image.bin`.

## License

GPL-3.0, see [LICENSE](LICENSE). Third-party components: [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Mortal
Kombat is a trademark of Warner Bros. Entertainment; this project is not affiliated with or endorsed by Warner Bros.
or NetherRealm Studios.
