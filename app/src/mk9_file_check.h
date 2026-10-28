// mk9-nx 1.0.1: the game files on the SD card checked against the release's list (app/mk9_game_files.txt), so a file
// cut short or damaged on the way to the card is named on screen instead of crashing the game later (a tester of
// 1.0.0: BGND_CourtyardN.xxx 23,151,504 bytes of 27,328,512, Story mode's first fight crashed loading it).

#pragma once

#include <filesystem>

namespace mk9::file_check {

// Switch: every start, before the intro. The size of every file of the list (the game folders listed, no stat per
// file: a few milliseconds), and after a crash in the game's code (the marker Start leaves for it) the SHA-256 of the
// files the game had opened last. A missing or wrong file: the system's error dialog names it and the process ends
// (this does not return). Elsewhere, or without the compiled-in list, it does nothing.
void AtStart(const std::filesystem::path& game_root);

// Tests only (mk9_file_check_test): right after the intro started. Nothing by default.
void TestAfterIntroStarted();

}  // namespace mk9::file_check
