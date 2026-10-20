// The addresses both renderers hook (banjo-nx's tools/renut_gpu_import.py generated this list for N&B). sa2-nx
// (2026-10-20): SA2's. mk9-nx (2026-10-20): MK9's (docs/mk9-renut-census.md). mk9-nx's own hook there (src/me,
// src/*.cpp) calls renut_native_sub_X when Renut's renderer is chosen (sa2_renut_yield.h).
#pragma once

#include <rex/ppc.h>

extern "C" void renut_native_sub_82BFBB78(PPCContext& ctx, uint8_t* base);  // AllocateInlineData
extern "C" void renut_native_sub_82BF5680(PPCContext& ctx, uint8_t* base);  // CBlocker::Check (mk9_gpu_wait.cpp)
extern "C" void renut_native_sub_82BEFEF8(PPCContext& ctx, uint8_t* base);  // D3DDevice_BeginVertices
extern "C" void renut_native_sub_82BF09E0(PPCContext& ctx, uint8_t* base);  // D3DDevice_DrawVertices
extern "C" void renut_native_sub_82BF0DF8(PPCContext& ctx, uint8_t* base);  // D3DDevice_DrawIndexedVertices
extern "C" void renut_native_sub_82BF6D58(PPCContext& ctx, uint8_t* base);  // D3DDevice_Swap (mk9_frame_counter.cpp)
