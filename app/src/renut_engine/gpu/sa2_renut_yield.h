// sa2-nx (2026-10-20): for sa2-nx's own hooks at the Direct3D functions Renut's renderer hooks too, first thing in
// their body. With sa2_renderer = "renut": SA2_RENUT_YIELD_SHARED(sub_X) runs Renut's side (renut_native_sub_X,
// banjo_shared_hooks.h); SA2_RENUT_YIELD(sub_X) runs the game's original (the function Renut leaves to SA2's own
// Direct3D, whose callees it hooks). Nothing in a build without Renut's renderer (SA2_RENUT_RENDERER off).
#pragma once

#if SA2_RENUT_RENDERER
#include "renut_engine/gpu/banjo_renut_select.h"
#include "renut_engine/gpu/banjo_shared_hooks.h"
#define SA2_RENUT_YIELD_SHARED(subroutine) BANJO_RENUT_YIELD_SHARED(subroutine)
#define SA2_RENUT_YIELD(subroutine) BANJO_RENUT_YIELD(subroutine)
#else
#define SA2_RENUT_YIELD_SHARED(subroutine)
#define SA2_RENUT_YIELD(subroutine)
#endif
