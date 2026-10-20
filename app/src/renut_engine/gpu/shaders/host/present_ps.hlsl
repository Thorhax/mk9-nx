#include "utility.hlsli"

struct PSInput
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

float4 main(PSInput input) : SV_Target0
{
    float4 color = SampleBindless(g_Push.textureIndex, g_Push.samplerIndex, input.uv);
    // sa2-nx (2026-10-20): param1's bits, when not 0, = the bindless slot of the game's display gamma ramp
    // (SetGammaRamp; the Xbox 360's display controller applies it at scan-out): 256 texels of red, green, blue in 0-1.
    // The same lookup as the emulated GPU's and sa2-nx's native renderer's output pass: each channel's 8-bit value picks
    // its entry. The slot is read as it is, with no arithmetic: an emulator (Ryujinx) follows a bindless handle only
    // back to a constant buffer load, and a computed one read zero (a black picture).
    const uint ramp = asuint(g_Push.param1);
    if (ramp != 0u)
    {
        const uint3 i = uint3(saturate(color.rgb) * 255.0 + 0.5);
        color.rgb = float3(g_Textures[NonUniformResourceIndex(ramp)].Load(int3(i.x, 0, 0)).x,
                           g_Textures[NonUniformResourceIndex(ramp)].Load(int3(i.y, 0, 0)).y,
                           g_Textures[NonUniformResourceIndex(ramp)].Load(int3(i.z, 0, 0)).z);
    }
    // param0 = output gamma exponent (1 = passthrough). sa2-nx: skipped at 1, where pow's exp2(log2) could move a
    // value by an ulp.
    if (g_Push.param0 != 1.0)
        color.rgb = pow(saturate(color.rgb), g_Push.param0);
    return float4(color.rgb, 1.0);
}
