// banjo-nx (2026-10-18): a colour resolve drawn instead of copied (draw.cpp, DrawResolveCopy). deko3d copies images on
// the copy engine, which needs the whole GPU idle before and after it; a draw is ordered like any other. The destination
// pixel plus g_Push.translate is the source texel, read with Load through a view of the surface's own format (the
// destination has the same format, so every texel is written back with its bits). present_vs draws the fullscreen
// triangle; the scissor limits it to the resolved rect.
#include "utility.hlsli"

float4 main(float4 position : SV_Position) : SV_Target0
{
    const int2 texel = int2(position.xy) + int2(g_Push.translate);
    return g_Textures[NonUniformResourceIndex(g_Push.textureIndex)].Load(int3(texel, 0));
}
