// mk9-nx (2026-10-26): see mk9_button_icons.h.
#include "mk9_button_icons.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include <rex/cvar.h>

REXCVAR_DEFINE_STRING(mk9_button_icons, "switch", "MK9",
                      "The game's button prompts: \"switch\" draws Switch buttons in place of the Xbox ones (by "
                      "position with input_xbox_layout: Xbox A shows as B, B as A, X as Y, Y as X, LB/RB as L/R, LT/RT "
                      "as ZL/ZR, START as +, BACK as -), \"xbox\" keeps the game's own icons");

namespace mk9::button_icons {
namespace {

constexpr uint32_t kSize = 128;     // the atlases
constexpr float kCell = 33.5f;      // their grid (three columns and rows used)

enum class Kind { Face, Shoulder, Trigger, Small };

// A cell to draw over: the Xbox button there, and what it becomes by position and by letter.
struct Cell {
  int col, row;
  Kind kind;
  const char *by_position;
  const char *by_letter;
};

// FNV-1a 64 of each atlas's BC3 blocks, rows without padding (from the console's dumps, mk9_dump_textures_dir).
struct Atlas {
  uint64_t hash;
  Cell cells[6];
  int count;
};
constexpr Atlas kAtlases[] = {
    // A, START, an arrow / LB, BACK, RS / LS, RT, RS
    {0x639E358260AE66E6ull,
     {{0, 0, Kind::Face, "B", "A"},
      {1, 0, Kind::Small, "+", "+"},
      {0, 1, Kind::Shoulder, "L", "L"},
      {1, 1, Kind::Small, "-", "-"},
      {1, 2, Kind::Trigger, "ZR", "ZR"}},
     5},
    // arrows, Y, a helmet / arrows / an arrow, B, a warning sign
    {0x3B79FB3538601B95ull, {{1, 0, Kind::Face, "X", "Y"}, {1, 2, Kind::Face, "A", "B"}}, 2},
    // RS, RB, LS / X, +, - / LT, a sword
    {0x82D18F4A1FDFEB74ull,
     {{1, 0, Kind::Shoulder, "R", "R"}, {0, 1, Kind::Face, "Y", "X"}, {0, 2, Kind::Trigger, "ZL", "ZL"}},
     3},
};

uint64_t RowHash(const uint8_t *data, uint32_t rows, uint32_t row_bytes, uint32_t row_pitch) {
  uint64_t h = 0xCBF29CE484222325ull;
  for (uint32_t r = 0; r < rows; ++r)
    for (uint32_t i = 0; i < row_bytes; ++i) h = (h ^ data[size_t(r) * row_pitch + i]) * 0x100000001B3ull;
  return h;
}

// 5 x 7 letters, a row per byte, the high bit of the five on the left (as in usf4-nx)
struct Glyph {
  char c;
  uint8_t rows[7];
};
constexpr Glyph kLetters[] = {
    {'A', {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11}}, {'B', {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E}},
    {'X', {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11}}, {'Y', {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04}},
    {'L', {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F}}, {'R', {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11}},
    {'Z', {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F}},
};

// The atlas as RGBA
struct Canvas {
  uint8_t px[kSize * kSize * 4] = {};
  void Blend(int x, int y, const uint8_t rgba[4], float cover) {
    if (x < 0 || y < 0 || x >= int(kSize) || y >= int(kSize) || cover <= 0) return;
    uint8_t *p = px + (size_t(y) * kSize + size_t(x)) * 4;
    cover = std::min(cover, 1.0f) * rgba[3] / 255.0f;
    const float a = p[3] / 255.0f, out = cover + a * (1 - cover);
    for (int i = 0; i < 3; i++)
      p[i] = uint8_t(out > 0 ? std::lround((rgba[i] * cover + p[i] * a * (1 - cover)) / out) : 0);
    p[3] = uint8_t(std::lround(out * 255));
  }
};

// The Switch's buttons: dark grey with a light rim and a white label (as usf4-nx's)
constexpr uint8_t kFill[4] = {48, 48, 48, 255};
constexpr uint8_t kRing[4] = {205, 205, 205, 255};
constexpr uint8_t kInk[4] = {255, 255, 255, 255};
constexpr uint8_t kShadow[4] = {0, 0, 0, 110};

// coverage of a pixel by a shape, 4 x 4 samples
template <class Inside>
float Cover(int x, int y, Inside inside) {
  int n = 0;
  for (int sy = 0; sy < 4; sy++)
    for (int sx = 0; sx < 4; sx++) n += inside(x + (sx + 0.5f) / 4, y + (sy + 0.5f) / 4);
  return n / 16.0f;
}

// Only the pixels of one cell (x0..x1, y0..y1) are drawn.
struct Area {
  int x0, y0, x1, y1;
};

template <class Inside>
void Fill(Canvas &c, const Area &a, const uint8_t rgba[4], Inside inside) {
  for (int y = a.y0; y < a.y1; y++)
    for (int x = a.x0; x < a.x1; x++) c.Blend(x, y, rgba, Cover(x, y, inside));
}

// a label of one or two letters (5 x 7 scaled to 'height'), or '+' / '-' as bars, centred on (cx, cy)
void Label(Canvas &c, const Area &a, const char *text, float cx, float cy, float height) {
  if (!std::strcmp(text, "+") || !std::strcmp(text, "-")) {
    const float hw = height * 0.42f, hh = height * 0.11f;
    Fill(c, a, kInk, [&](float x, float y) {
      return (std::fabs(x - cx) <= hw && std::fabs(y - cy) <= hh) ||
             (text[0] == '+' && std::fabs(x - cx) <= hh && std::fabs(y - cy) <= hw);
    });
    return;
  }
  const size_t n = std::strlen(text);
  const float scale = height / 7.0f, gap = scale, width = n * 5 * scale + (n - 1) * gap;
  for (size_t i = 0; i < n; ++i) {
    const Glyph *g = nullptr;
    for (const Glyph &l : kLetters)
      if (l.c == text[i]) g = &l;
    if (!g) continue;
    const float x0 = cx - width / 2 + i * (5 * scale + gap), y0 = cy - 3.5f * scale;
    Fill(c, a, kInk, [&](float x, float y) {
      const int col = int(std::floor((x - x0) / scale)), row = int(std::floor((y - y0) / scale));
      return col >= 0 && col < 5 && row >= 0 && row < 7 && (g->rows[row] >> (4 - col) & 1);
    });
  }
}

// a round button where the Xbox ball was, with a soft shadow under it as the game's
void Round(Canvas &c, const Area &a, const char *text, float cx, float cy, float r, float label) {
  Fill(c, a, kShadow, [&](float x, float y) { return std::hypot(x - cx, y - (cy + 1.5f)) <= r + 0.5f; });
  Fill(c, a, kRing, [&](float x, float y) { return std::hypot(x - cx, y - cy) <= r; });
  Fill(c, a, kFill, [&](float x, float y) { return std::hypot(x - cx, y - cy) <= r - 1.6f; });
  Label(c, a, text, cx, cy, label);
}

// a rounded key with its label, where the LB / RB / LT / RT key was
void Key(Canvas &c, const Area &a, const char *text, float cx, float cy, float w, float h, float label) {
  const float x0 = cx - w / 2, x1 = cx + w / 2, r = h / 2;
  auto key = [=](float inset, float dy) {
    return [=](float x, float y) {
      const float ax = std::clamp(x, x0 + r, x1 - r);
      return std::hypot(x - ax, y - (cy + dy)) <= r - inset;
    };
  };
  Fill(c, a, kShadow, key(-0.5f, 1.5f));
  Fill(c, a, kRing, key(0, 0));
  Fill(c, a, kFill, key(1.6f, 0));
  Label(c, a, text, cx, cy, label);
}

// --- BC3 (DXT5): decoded to the canvas, and the blocks drawn over encoded again (from usf4-nx)
uint16_t To565(const uint8_t *p) { return uint16_t(((p[0] >> 3) << 11) | ((p[1] >> 2) << 5) | (p[2] >> 3)); }
void From565(uint16_t c, int out[3]) {
  out[0] = ((c >> 11) & 31) * 255 / 31;
  out[1] = ((c >> 5) & 63) * 255 / 63;
  out[2] = (c & 31) * 255 / 31;
}

void DecodeBc3Block(const uint8_t in[16], Canvas &c, uint32_t bx, uint32_t by) {
  const int a0 = in[0], a1 = in[1];
  int apal[8] = {a0, a1};
  if (a0 > a1)
    for (int i = 1; i <= 6; ++i) apal[i + 1] = ((7 - i) * a0 + i * a1) / 7;
  else {
    for (int i = 1; i <= 4; ++i) apal[i + 1] = ((5 - i) * a0 + i * a1) / 5;
    apal[6] = 0;
    apal[7] = 255;
  }
  uint64_t abits = 0;
  for (int i = 0; i < 6; ++i) abits |= uint64_t(in[2 + i]) << (8 * i);
  uint16_t c0, c1;
  uint32_t cbits;
  std::memcpy(&c0, in + 8, 2);
  std::memcpy(&c1, in + 10, 2);
  std::memcpy(&cbits, in + 12, 4);
  int e0[3], e1[3], pal[4][3];
  From565(c0, e0);
  From565(c1, e1);
  for (int k = 0; k < 3; ++k) {  // BC3's colour block is always the four-colour one
    pal[0][k] = e0[k];
    pal[1][k] = e1[k];
    pal[2][k] = (2 * e0[k] + e1[k]) / 3;
    pal[3][k] = (e0[k] + 2 * e1[k]) / 3;
  }
  for (int i = 0; i < 16; ++i) {
    uint8_t *p = c.px + (size_t(by * 4 + i / 4) * kSize + (bx * 4 + i % 4)) * 4;
    const int k = (cbits >> (2 * i)) & 3;
    for (int j = 0; j < 3; ++j) p[j] = uint8_t(pal[k][j]);
    p[3] = uint8_t(apal[(abits >> (3 * i)) & 7]);
  }
}

void EncodeBc3Block(const Canvas &c, uint32_t bx, uint32_t by, uint8_t out[16]) {
  const uint8_t *px[16];
  for (uint32_t i = 0; i < 16; ++i) px[i] = c.px + (size_t(by * 4 + i / 4) * kSize + (bx * 4 + i % 4)) * 4;
  // alpha: its extremes and the six steps between them
  uint8_t amax = 0, amin = 255;
  for (const uint8_t *p : px) amax = std::max(amax, p[3]), amin = std::min(amin, p[3]);
  int apal[8] = {amax, amin};
  for (int i = 1; i <= 6; ++i) apal[i + 1] = ((7 - i) * amax + i * amin) / 7;
  uint64_t abits = 0;
  for (int i = 0; i < 16; ++i) {
    int best = 0;
    for (int k = 1; k < 8; ++k)
      if (std::abs(apal[k] - px[i][3]) < std::abs(apal[best] - px[i][3])) best = k;
    abits |= uint64_t(amax == amin ? 0 : best) << (3 * i);
  }
  out[0] = amax;
  out[1] = amin;
  for (int i = 0; i < 6; ++i) out[2 + i] = uint8_t(abits >> (8 * i));
  // colour: the darkest and lightest of the visible pixels (all of them if none is)
  int lo = -1, hi = -1, llo = 1 << 30, lhi = -1;
  for (int pass = 0; pass < 2 && lo < 0; ++pass)
    for (int i = 0; i < 16; ++i) {
      if (pass == 0 && px[i][3] < 16) continue;
      const int l = px[i][0] * 3 + px[i][1] * 6 + px[i][2];
      if (l < llo) llo = l, lo = i;
      if (l > lhi) lhi = l, hi = i;
    }
  uint16_t c0 = To565(px[hi]), c1 = To565(px[lo]);
  if (c0 < c1) std::swap(c0, c1);
  int e0[3], e1[3];
  From565(c0, e0);
  From565(c1, e1);
  int pal[4][3];
  for (int k = 0; k < 3; ++k) {
    pal[0][k] = e0[k];
    pal[1][k] = e1[k];
    pal[2][k] = (2 * e0[k] + e1[k]) / 3;
    pal[3][k] = (e0[k] + 2 * e1[k]) / 3;
  }
  uint32_t cbits = 0;
  for (int i = 0; i < 16; ++i) {
    int best = 0, bestd = 1 << 30;
    for (int k = 0; k < (c0 == c1 ? 1 : 4); ++k) {
      int d = 0;
      for (int j = 0; j < 3; ++j) d += (pal[k][j] - px[i][j]) * (pal[k][j] - px[i][j]);
      if (d < bestd) bestd = d, best = k;
    }
    cbits |= uint32_t(best) << (2 * i);
  }
  std::memcpy(out + 8, &c0, 2);
  std::memcpy(out + 10, &c1, 2);
  std::memcpy(out + 12, &cbits, 4);
}

// The cell drawn over: cleared, then the Switch button centred where the Xbox one was.
void DrawCell(Canvas &c, const Cell &cell, bool by_position) {
  // The bottom row's keys and balls reach a few pixels past the cell; nothing is drawn under that row.
  const Area a{int(cell.col * kCell), int(cell.row * kCell), int((cell.col + 1) * kCell),
               int((cell.row + 1) * kCell) + (cell.row == 2 ? 4 : 0)};
  for (int y = a.y0; y < a.y1; ++y) std::memset(c.px + (size_t(y) * kSize + a.x0) * 4, 0, size_t(a.x1 - a.x0) * 4);
  const char *text = by_position ? cell.by_position : cell.by_letter;
  const float cx = cell.col * kCell + 16.0f, cy = cell.row * kCell + 16.0f;
  switch (cell.kind) {
    case Kind::Face: Round(c, a, text, cx, cy, 14.5f, 15.0f); break;
    case Kind::Small: Round(c, a, text, cx, cy, 11.5f, 13.0f); break;
    case Kind::Shoulder: Key(c, a, text, cx, cy, 29.0f, 19.0f, 11.0f); break;
    case Kind::Trigger: Key(c, a, text, cx, cy, 30.0f, 19.0f, 9.0f); break;
  }
}

bool SwitchIconsOn() {
  static const bool on = REXCVAR_GET(mk9_button_icons) != "xbox";
  return on;
}

}  // namespace

uint64_t Hash(const uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch) {
  return RowHash(data, (height + 3) / 4, (width + 3) / 4 * 16, row_pitch);
}

bool Candidate(uint32_t width, uint32_t height) { return width == kSize && height == kSize && SwitchIconsOn(); }

bool Paint(uint8_t *data, uint32_t width, uint32_t height, uint32_t row_pitch) {
  if (!Candidate(width, height)) return false;
  const uint64_t h = Hash(data, width, height, row_pitch);
  const Atlas *atlas = nullptr;
  for (const Atlas &a : kAtlases)
    if (a.hash == h) atlas = &a;
  if (!atlas) return false;
  // By position (input_xbox_layout, the default) the Xbox button shows as the Switch button in its place; by letter
  // it keeps its letter. (input_xbox_layout is the Switch input driver's: read by name, by position where there is
  // none.)
  static const bool by_position = rex::cvar::GetFlagByName("input_xbox_layout") != "false";
  static Canvas c;  // the render thread's uploads only
  constexpr uint32_t kBlocks = kSize / 4;
  for (uint32_t by = 0; by < kBlocks; ++by)
    for (uint32_t bx = 0; bx < kBlocks; ++bx) DecodeBc3Block(data + size_t(by) * row_pitch + bx * 16, c, bx, by);
  bool touched[kBlocks][kBlocks] = {};
  for (int i = 0; i < atlas->count; ++i) {
    const Cell &cell = atlas->cells[i];
    DrawCell(c, cell, by_position);
    for (uint32_t by = uint32_t(cell.row * kCell) / 4;
         by <= std::min(kBlocks - 1, (uint32_t((cell.row + 1) * kCell) + (cell.row == 2 ? 4 : 0)) / 4); ++by)
      for (uint32_t bx = uint32_t(cell.col * kCell) / 4;
           bx <= std::min(kBlocks - 1, uint32_t((cell.col + 1) * kCell) / 4); ++bx)
        touched[by][bx] = true;
  }
  for (uint32_t by = 0; by < kBlocks; ++by)
    for (uint32_t bx = 0; bx < kBlocks; ++bx)
      if (touched[by][bx]) EncodeBc3Block(c, bx, by, data + size_t(by) * row_pitch + bx * 16);
  return true;
}

}  // namespace mk9::button_icons
