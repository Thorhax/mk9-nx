// mk9-nx (2026-10-26): Switch button icons in place of the Xbox ones MK9 draws (mk9_button_icons). After usf4-nx's
// usf4_button_icons (itself after the Mortal Kombat: Shaolin Monks port's port/vk/button_icons.cpp). MK9's menus take
// their prompts from three 128x128 BC3 atlases (Scaleform images, a grid of ~33.5-pixel cells: the face buttons,
// shoulders, triggers, START and BACK, with the game's own arrows, sticks and symbols). Renut redraws the Xbox cells
// at upload (render/textures.cpp), before the copy to the GPU: the atlas is decoded, the cells are drawn over and the
// blocks they touch are encoded again. With input_xbox_layout (the default) buttons are mapped by position, so each
// icon becomes the Switch button in that place: Xbox A (bottom) is the Switch's B, B (right) its A, X (left) its Y,
// Y (top) its X; LB/RB are L/R, LT/RT are ZL/ZR, START is + and BACK is -.
#pragma once

#include <cstdint>

namespace mk9::button_icons {

// Whether a texture's first level of this size can be one of the atlases (BC3, 128x128): only those are hashed.
bool Candidate(uint32_t width, uint32_t height);

// The atlas's first level as uploaded (BC3 blocks, 'row_pitch' bytes a row of blocks): redrawn in place when it is one
// of the Xbox atlases and mk9_button_icons is "switch". Returns whether it was.
bool Paint(uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch);

// FNV-1a 64 of the level's bytes without row padding (the atlases' table key; for diagnostics).
uint64_t Hash(const uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch);

}  // namespace mk9::button_icons
