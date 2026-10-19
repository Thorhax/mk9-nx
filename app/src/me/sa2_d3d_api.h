// Sonic Adventure 2: the game's Direct3D at its API boundary (see sa2_d3d_api.cpp).
#pragma once

#include <cstdint>

namespace sa2::d3d {

// The Direct3D device, from the Draw* calls' r3 (the census reads its command buffer pointers).
void NoteDevice(uint32_t device);
// Census: the device's flag bytes at a draw (a histogram in the report).
void NoteDrawFlags(const uint8_t* base, uint32_t device);

// sa2_diag_d3d_census: true when the census is on (read once).
bool CensusOn();
// Around a call of a census-hooked function (the renderer's own hooks call these too): counts the outermost call and
// the PM4 it wrote, or a nested call.
void CensusEnter(uint32_t address, const uint8_t* base);
void CensusLeave(const uint8_t* base);
// Around the command buffer's space function (a segment switch inside a call): the old segment's packets are decoded
// before it, the new segment is followed after it.
void CensusSegmentBefore(const uint8_t* base);
void CensusSegmentAfter(const uint8_t* base);

}  // namespace sa2::d3d

// The census build's hooks (generated into out/census, see sa2_d3d_api.cpp): the original runs unchanged.
#define SA2_D3D_CENSUS_HOOK(ADDR)                  \
  REX_EXTERN(__imp__sub_##ADDR);                   \
  REX_HOOK_RAW(sub_##ADDR) {                       \
    sa2::d3d::CensusEnter(0x##ADDR##u, base);      \
    __imp__sub_##ADDR(ctx, base);                  \
    sa2::d3d::CensusLeave(base);                   \
  }
#define SA2_D3D_CENSUS_SEGMENT_HOOK(ADDR)          \
  REX_EXTERN(__imp__sub_##ADDR);                   \
  REX_HOOK_RAW(sub_##ADDR) {                       \
    sa2::d3d::CensusSegmentBefore(base);           \
    __imp__sub_##ADDR(ctx, base);                  \
    sa2::d3d::CensusSegmentAfter(base);            \
  }
