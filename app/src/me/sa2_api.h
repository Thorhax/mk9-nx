// Sonic Adventure 2: the native Direct3D layer (the API-level renderer, docs/api-renderer.md).
//
// The game's Xbox 360 Direct3D (the XDK library linked into the executable) turned every call into PM4 packets for an
// emulated Xenos command processor. Here its entry points are native code: they keep the Direct3D device structure
// exactly as the game sees it (the game reads and writes parts of it inline), and hand the renderer host records of
// operations (src/me/me_native_system.cpp, the direct-state record format) instead of PM4. Each group of entry points
// has its own key (sa2_api_*); verify mode (sa2_cpu_verify) runs the original for real and the native code on an
// overlay of guest memory, and compares everything (device bytes, guest memory the native code wrote, and its ops
// against the original's packets decoded the same way).
#pragma once

#include <string>

#include <cstdint>

struct PPCContext;

namespace sa2::api {

// Draws (sa2_api_draws): D3DDevice_DrawVertices / DrawVerticesUP / DrawIndexedVertices and the inline-vertex draw,
// with Direct3D's draw-time state flush and shader load. Each returns false when it did not run (key off, native
// renderer off, a path the native code does not take): the hook then runs its old path.
bool DrawVertices(PPCContext& ctx, uint8_t* base);
// D3DDevice_BeginVertices (returns the inline vertices' address in r3) and D3DDevice_DrawVerticesUP.
bool BeginVertices(PPCContext& ctx, uint8_t* base);
bool DrawVerticesUP(PPCContext& ctx, uint8_t* base);
// Verify mode: Direct3D's inline-data allocator inside a shadow run of the original (a scratch address, no waiting).
bool ShadowInlineAlloc(PPCContext& ctx, uint8_t* base);
// D3DDevice_Clear (sa2_api_clears).
bool Clear(PPCContext& ctx, uint8_t* base);
// Direct3D's resolve core, sub_825645A0 (sa2_api_resolves); its counts for the [api] line.
bool Resolve(PPCContext& ctx, uint8_t* base);
std::string ResolveReport();
// Direct3D's window offset/scissor writer (registers 0x2080-0x2082; SetViewport, SetScissorRect, the surface setters).
bool WindowScissor(PPCContext& ctx, uint8_t* base);

// The [api] report line (every 10 s, with the renderer's report).
void Report();

}  // namespace sa2::api
