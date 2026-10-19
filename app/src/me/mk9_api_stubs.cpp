// USF4: the parts of sa2-nx's renderer glue that are SA2-specific and not ported (the native Direct3D layer,
// src/me/sa2_api_*.cpp there, and the Direct3D census, sa2_d3d_api.cpp there; both rely on SA2's addresses). The
// renderer core still calls into them; these are their "off" behaviour.

#include <cstdint>

#include "me/sa2_api.h"
#include "me/sa2_d3d_api.h"

namespace sa2::api {
void Report() {}
}  // namespace sa2::api

namespace sa2::d3d {
void NoteDevice(uint32_t) {}
void NoteDrawFlags(const uint8_t*, uint32_t) {}
bool CensusOn() { return false; }
void CensusEnter(uint32_t, const uint8_t*) {}
void CensusLeave(const uint8_t*) {}
void CensusSegmentBefore(const uint8_t*) {}
void CensusSegmentAfter(const uint8_t*) {}
}  // namespace sa2::d3d
