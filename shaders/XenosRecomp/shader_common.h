#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)
// MASSEFFECT: constants arrive through a dynamic UBO (a constant bank on Maxwell) instead of being read
// through a 64-bit pointer. High bit so as not to clash with the UNLEASHED_RECOMP ones.
#define SPEC_CONSTANT_CONSTANTS_UBO    (1 << 8)
// The tfetch offset is scaled with 1/size taken from the shared constants instead of querying the
// texture for its size. A size query is another texture unit operation.
#define SPEC_CONSTANT_INV_TEX_SIZE    (1 << 9)
// sa2-nx (sa2_gpu_blend_skip): the pixel's blend would leave the target exactly as it was (a UNORM target, no depth or
// stencil write, a blend that keeps the destination when the source alpha, or the whole source, is 0 or less), so it
// is discarded before the blend. Set by the renderer per pipeline; a NaN output is never discarded.
#define SPEC_CONSTANT_BLEND_SKIP_ALPHA (1 << 10)
#define SPEC_CONSTANT_BLEND_SKIP_ZERO  (1 << 11)
// sa2-nx GPU round 2 (sa2_gpu_structured_cf): the shader's forward-only control flow as nested if/else instead of the
// per-instruction skipTo tests (the translator emits both when it has proven them equivalent; the renderer picks).
#define SPEC_CONSTANT_STRUCTURED_CF    (1 << 12)
// sa2-nx GPU round 2, per-pipeline facts the native renderer knows, so the shaders stop testing them per vertex or
// pixel (bits 5, 6 and 13 are free in the MASSEFFECT_RECOMP build):
// sa2_gpu_texcoords_unswapped: the renderer never swaps texcoords (g_SwappedTexcoords is always 0).
#define SPEC_CONSTANT_TEXCOORDS_UNSWAPPED (1 << 5)
// sa2_gpu_no_texture_signs: no texture of the draw has a signed, biased or gamma component (packed signs all 0).
#define SPEC_CONSTANT_NO_TEXTURE_SIGNS (1 << 6)
// sa2_gpu_identity_remap: every vertex input of the pipeline keeps its component order (g_InputRemap is 0xFFF).
#define SPEC_CONSTANT_IDENTITY_REMAP   (1 << 13)
// sa2_gpu_index_base_held: the translator proved this vertex shader never reads the vertex index r0 starts with: with
// the bit, r0 starts at 0 and the renderer may hold g_VertexIndexBase (and the shared block) across draws.
#define SPEC_CONSTANT_VERTEX_INDEX_UNUSED (1 << 19)
/*
 * The alpha test function, specialized (bits 16-18).
 *
 * g_AlphaFunction used to come from the shared constants, that is, at run time, so alphaTestValue
 * was a 7-case switch: ~8 branches in every pixel shader that does alpha testing. It was not needed
 * at all: the app already knows the function when it creates the pipeline (it comes from RB_COLORCONTROL, which is already in the key), so with three more bits the compiler
 * keeps the one comparison that applies and drops the other six. The output is bit-identical.
 */
// Bits 15, 19 and 23 are reserved (unused here; left free for the app's own use).
#define SPEC_CONSTANT_ALPHA_FUNC_SHIFT  16
#define SPEC_CONSTANT_ALPHA_FUNC_MASK   (7 << 16)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

#if !defined(__cplusplus) || defined(__INTELLISENSE__)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;

// MASSEFFECT: the same blocks of the upload buffer, also as dynamic UBOs in set 4. With
// -fvk-use-dx-layout each float4 takes 16 contiguous bytes, so any 4-byte word of the shared block is
// a component: v[B / 16][(B % 16) / 4], and asuint reads it without changing a bit.
struct MassEffectBlockVs { float4 v[256]; };
struct MassEffectBlockPs { float4 v[256]; };  // banjo-nx: 256 (Nuts & Bolts reads up to c255)
struct MassEffectSharedBlock { float4 v[35]; };  // banjo-nx: + bool constants, indexed fetch buffers, vertex index base
[[vk::binding(0, 4)]] ConstantBuffer<MassEffectBlockVs> g_UboVertex;
[[vk::binding(1, 4)]] ConstantBuffer<MassEffectBlockPs> g_UboPixel;
[[vk::binding(2, 4)]] ConstantBuffer<MassEffectSharedBlock> g_UboShared;
#define MASSEFFECT_UBO ((g_SpecConstants & SPEC_CONSTANT_CONSTANTS_UBO) != 0)
#define MASSEFFECT_SHARED_UINT(B)  asuint(g_UboShared.v[(B) / 16][((B) % 16) / 4])
#define MASSEFFECT_SHARED_FLOAT(B) g_UboShared.v[(B) / 16][((B) % 16) / 4]

#define g_Booleans                 (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(256) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256))
// banjo-nx: all 256 bool constants (the 8 SHADER_CONSTANT_BOOL words, 0x4900..0x4907) at bytes 488..519 of the
// shared block; bool N is bit N % 32 of word N / 32, as on the Xenos. Nuts & Bolts' shaders (most without a CTAB)
// test bools anywhere in that range (pixel shaders up to b254), beyond g_Booleans' 16 bits per stage.
#define g_BoolWord(I)              (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(488 + (I) * 4) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 488 + (I) * 4))
#define BANJO_BOOL(A)              ((g_BoolWord((A) >> 5) >> ((A) & 31)) & 1)
#define g_SwappedTexcoords         (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(260) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260))
#define g_HalfPixelOffset          (MASSEFFECT_UBO ? float2(MASSEFFECT_SHARED_FLOAT(264), MASSEFFECT_SHARED_FLOAT(268)) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264))
#define g_AlphaThreshold           (MASSEFFECT_UBO ? MASSEFFECT_SHARED_FLOAT(272) : vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272))
// MASSEFFECT: alpha test function (RB_COLORCONTROL.alpha_func): 0 never, 1 <, 2 ==, 3 <=,
// 4 >, 5 !=, 6 >=, 7 always.
#define g_AlphaFunction            (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(276) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 276))
// MASSEFFECT: position to host clip space, like ndc_scale/ndc_offset in the
// emulation (graphics/util/draw.cpp). (1, 1) and (0, 0) for normal draws;
// with Xenos clipping disabled it converts from pixels to NDC.
#define g_NdcScale                 (MASSEFFECT_UBO ? float2(MASSEFFECT_SHARED_FLOAT(280), MASSEFFECT_SHARED_FLOAT(284)) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 280))
#define g_NdcOffset                (MASSEFFECT_UBO ? float2(MASSEFFECT_SHARED_FLOAT(288), MASSEFFECT_SHARED_FLOAT(292)) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 288))
// MASSEFFECT: where each component of the vertex input at that location comes from
// (D3D patches the fetch swizzle according to the declaration). 0xFFF = as is.
#define g_InputRemap(LOC)          (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(296 + (LOC) * 4) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 296 + (LOC) * 4))
// banjo-nx: SV_VertexID + this = the guest vertex index, which a Xenos vertex shader finds in r0.x (byte 544).
#define g_VertexIndexBase          (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(544) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 544))

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

// banjo-nx: vertex fetches indexed by a computed register (bone palettes; classifyVertexFetches in
// shader_recompiler.cpp). The renderer uploads the buffer of each such fetch constant K (0..1) in host word order
// and passes its device address at bytes 520 + 8K of the shared block and its size in words at bytes 536 + 4K.
// Formats decode like the emulation's vertex fetch (graphics/pipeline/shader/spirv_translator_fetch.cpp).
uint banjoVfetchWord(uint K, int dword)
{
    const uint2 address = MASSEFFECT_UBO ? uint2(MASSEFFECT_SHARED_UINT(520 + K * 8), MASSEFFECT_SHARED_UINT(524 + K * 8))
                                         : vk::RawBufferLoad<uint2>(g_PushConstants.SharedConstants + 520 + K * 8, 8);
    const uint words = MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT(536 + K * 4)
                                      : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 536 + K * 4);
    if (dword < 0 || uint(dword) >= words)
        return 0;
    return vk::RawBufferLoad<uint>(((uint64_t(address.y) << 32) | uint64_t(address.x)) + uint64_t(uint(dword)) * 4, 4);
}

float4 banjoVfetchPacked(uint4 values, uint4 widths, uint isSigned, uint isInteger)
{
    float4 result;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        const uint width = widths[i];
        if (width == 0)
        {
            result[i] = i == 3 ? 1.0 : 0.0;
            continue;
        }
        const uint value = values[i] & ((1u << width) - 1u);
        if (isSigned != 0)
        {
            const int signedValue = int(value << (32 - width)) >> (32 - width);
            result[i] = isInteger != 0 ? float(signedValue)
                                       : max(float(signedValue) / float((1u << (width - 1)) - 1u), -1.0);
        }
        else
        {
            result[i] = isInteger != 0 ? float(value) : float(value) / float((1u << width) - 1u);
        }
    }
    return result;
}

float4 banjoVfetch(uint K, int dword, uint format, uint isSigned, uint isInteger, int expAdjust)
{
    float4 result = float4(0.0, 0.0, 0.0, 1.0);
    const uint w0 = banjoVfetchWord(K, dword);
    switch (format)
    {
    case 6:   // k_8_8_8_8
        result = banjoVfetchPacked(uint4(w0, w0 >> 8, w0 >> 16, w0 >> 24), uint4(8, 8, 8, 8), isSigned, isInteger);
        break;
    case 7:   // k_2_10_10_10
        result = banjoVfetchPacked(uint4(w0, w0 >> 10, w0 >> 20, w0 >> 30), uint4(10, 10, 10, 2), isSigned, isInteger);
        break;
    case 16:  // k_10_11_11
        result = banjoVfetchPacked(uint4(w0, w0 >> 11, w0 >> 22, 0), uint4(11, 11, 10, 0), isSigned, isInteger);
        break;
    case 17:  // k_11_11_10
        result = banjoVfetchPacked(uint4(w0, w0 >> 10, w0 >> 21, 0), uint4(10, 11, 11, 0), isSigned, isInteger);
        break;
    case 25:  // k_16_16
        result = banjoVfetchPacked(uint4(w0, w0 >> 16, 0, 0), uint4(16, 16, 0, 0), isSigned, isInteger);
        break;
    case 26:  // k_16_16_16_16
    {
        const uint w1 = banjoVfetchWord(K, dword + 1);
        result = banjoVfetchPacked(uint4(w0, w0 >> 16, w1, w1 >> 16), uint4(16, 16, 16, 16), isSigned, isInteger);
        break;
    }
    case 31:  // k_16_16_FLOAT
        result.xy = f16tof32(uint2(w0, w0 >> 16));
        return result;
    case 32:  // k_16_16_16_16_FLOAT
    {
        const uint w1 = banjoVfetchWord(K, dword + 1);
        result = f16tof32(uint4(w0, w0 >> 16, w1, w1 >> 16));
        return result;
    }
    case 33:  // k_32
    case 34:  // k_32_32
    case 35:  // k_32_32_32_32
    {
        const uint count = format - 32;
        [unroll] for (uint i = 0; i < 4; i++)
        {
            if (i >= count)
                break;
            const uint w = i == 0 ? w0 : banjoVfetchWord(K, dword + int(i));
            float value = isSigned != 0 ? float(int(w)) : float(w);
            if (isInteger == 0)
                value = isSigned != 0 ? max(value / 2147483647.0, -1.0) : value / 4294967295.0;
            result[i] = value;
        }
        break;
    }
    case 36:  // k_32_FLOAT
        return float4(asfloat(w0), 0.0, 0.0, 1.0);
    case 37:  // k_32_32_FLOAT
        return float4(asfloat(w0), asfloat(banjoVfetchWord(K, dword + 1)), 0.0, 1.0);
    case 57:  // k_32_32_32_FLOAT
        return float4(asfloat(w0), asfloat(banjoVfetchWord(K, dword + 1)), asfloat(banjoVfetchWord(K, dword + 2)), 1.0);
    case 38:  // k_32_32_32_32_FLOAT
        return float4(asfloat(w0), asfloat(banjoVfetchWord(K, dword + 1)), asfloat(banjoVfetchWord(K, dword + 2)),
                      asfloat(banjoVfetchWord(K, dword + 3)));
    default:
        break;
    }
    return expAdjust != 0 ? result * exp2(float(expAdjust)) : result;
}

#else

#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x); \
    uint g_AlphaFunction : packoffset(c17.y); \
    float2 g_NdcScale : packoffset(c17.z); \
    float2 g_NdcOffset : packoffset(c18.x);

uint g_SpecConstants();

#define g_InputRemap(LOC) 0xFFF
#define BANJO_BOOL(A) 0
#define g_VertexIndexBase 0u
float4 banjoVfetch(uint K, int dword, uint format, uint isSigned, uint isInteger, int expAdjust) { return float4(0.0, 0.0, 0.0, 1.0); }

#endif


// banjo-nx: the W of a MOVA's vector operand; the translator passes it already reduced by the write mask.
float banjoMovaW(float v) { return v; }
float banjoMovaW(float2 v) { return v.y; }
float banjoMovaW(float3 v) { return v.z; }
float banjoMovaW(float4 v) { return v.w; }
Texture2D<float4> g_Texture2DDescriptorHeap[] : register(t0, space0);
Texture3D<float4> g_Texture3DDescriptorHeap[] : register(t0, space1);
TextureCube<float4> g_TextureCubeDescriptorHeap[] : register(t0, space2);
SamplerState g_SamplerDescriptorHeap[] : register(s0, space3);

uint2 getTexture2DDimensions(Texture2D<float4> texture)
{
    uint2 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y);
    return dimensions;
}

// The native renderer packs the four post-swizzle Xenos TextureSign values into bits 24-31 of the
// descriptor index. Values: 0 unsigned, 1 signed (the renderer still warns for it), 2 unsigned biased,
// 3 Xenos piecewise-linear gamma. Descriptor heaps use the low 24 bits.
float masseffectGammaToLinear(float gamma)
{
    gamma = saturate(gamma);
    float scale;
    float offset;
    if (gamma >= 96.0 / 255.0)
    {
        if (gamma >= 192.0 / 255.0) { scale = 8.0 / 1024.0; offset = -1024.0; }
        else                         { scale = 4.0 / 1024.0; offset = -256.0; }
    }
    else
    {
        if (gamma >= 64.0 / 255.0) { scale = 2.0 / 1024.0; offset = -64.0; }
        else                        { scale = 1.0 / 1024.0; offset = 0.0; }
    }
    float linearValue = gamma * ((255.0 * 1024.0) * scale) + offset;
    linearValue += trunc(linearValue * scale);
    return linearValue * (1.0 / 1023.0);
}

float4 masseffectApplyTextureSigns(float4 value, uint packedSigns)
{
    // sa2-nx (sa2_gpu_no_texture_signs): the renderer sets the bit only when every texture of the draw has packed
    // signs 0, which is the early return below: the test goes with it.
    [branch] if (g_SpecConstants() & SPEC_CONSTANT_NO_TEXTURE_SIGNS)
        return value;
    // sa2-nx: no signed or gamma component (the usual case): one uniform branch (the signs come from the descriptor
    // index, the same for the whole draw) instead of four extract/compare/select steps per fetch. The loop below
    // changes nothing when packedSigns is 0, so the result is the same.
    [branch] if (packedSigns == 0)
        return value;
    for (uint i = 0; i < 4; ++i)
    {
        uint sign = (packedSigns >> (i * 2)) & 3;
        if (sign == 2)
            value[i] = value[i] * 2.0 - 1.0;
        else if (sign == 3)
            value[i] = masseffectGammaToLinear(value[i]);
    }
    return value;
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float2 invSize)
{
    uint packedSigns = resourceDescriptorIndex >> 24;
    resourceDescriptorIndex &= 0x00FFFFFF;
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    // With the bit set, the size comes from a constant and the texture does not have to be queried. It
    // is the same computation with the same number: the renderer writes 1/size of the host image. With a
    // ternary, DXC evaluates both branches and the size query stays: an if with [branch] is needed for
    // the dead branch to disappear when the pipeline is specialized.
    float2 displacement;
    [branch] if (g_SpecConstants() & SPEC_CONSTANT_INV_TEX_SIZE)
        displacement = offset * invSize;
    else
        displacement = offset / getTexture2DDimensions(texture);
    return masseffectApplyTextureSigns(
        texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + displacement), packedSigns);
}

// banjo-nx: a 1D texture is a row of the 2D heap (sN_Texture1DDescriptorIndex is the 2D index).
float4 tfetch1D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float texCoord)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], float2(texCoord, 0.5));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    resourceDescriptorIndex &= 0x00FFFFFF;
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}

float w0(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-a + 3.0f) - 3.0f) + 1.0f);
}

float w1(float a)
{
    return (1.0f / 6.0f) * (a * a * (3.0f * a - 6.0f) + 4.0f);
}

float w2(float a)
{
    return (1.0f / 6.0f) * (a * (a * (-3.0f * a + 3.0f) + 3.0f) + 1.0f);
}

float w3(float a)
{
    return (1.0f / 6.0f) * (a * a * a);
}

float g0(float a)
{
    return w0(a) + w1(a);
}

float g1(float a)
{
    return w2(a) + w3(a);
}

float h0(float a)
{
    return -1.0f + w1(a) / (w0(a) + w1(a)) + 0.5f;
}

float h1(float a)
{
    return 1.0f + w3(a) / (w2(a) + w3(a)) + 0.5f;
}

float4 tfetch2DBicubic(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    uint packedSigns = resourceDescriptorIndex >> 24;
    resourceDescriptorIndex &= 0x00FFFFFF;
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    uint2 dimensions = getTexture2DDimensions(texture);
    
    float x = texCoord.x * dimensions.x + offset.x;
    float y = texCoord.y * dimensions.y + offset.y;

    x -= 0.5f;
    y -= 0.5f;
    float px = floor(x);
    float py = floor(y);
    float fx = x - px;
    float fy = y - py;

    float g0x = g0(fx);
    float g1x = g1(fx);
    float h0x = h0(fx);
    float h1x = h1(fx);
    float h0y = h0(fy);
    float h1y = h1(fy);

    float4 r =
        g0(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h0y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h0y) / float2(dimensions))) +
        g1(fy) * (g0x * texture.Sample(samplerState, float2(px + h0x, py + h1y) / float2(dimensions)) +
            g1x * texture.Sample(samplerState, float2(px + h1x, py + h1y) / float2(dimensions)));

    return masseffectApplyTextureSigns(r, packedSigns);
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    uint packedSigns = resourceDescriptorIndex >> 24;
    resourceDescriptorIndex &= 0x00FFFFFF;
    return masseffectApplyTextureSigns(
        g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord),
        packedSigns);
}

struct CubeMapData
{
    float3 cubeMapDirections[2];
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    uint packedSigns = resourceDescriptorIndex >> 24;
    resourceDescriptorIndex &= 0x00FFFFFF;
    return masseffectApplyTextureSigns(
        g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]),
        packedSigns);
}

float4 tfetchR11G11B10(uint4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        return float4(
            (value.x & 0x00000400 ? -1.0 : 0.0) + ((value.x & 0x3FF) / 1024.0),
            (value.x & 0x00200000 ? -1.0 : 0.0) + (((value.x >> 11) & 0x3FF) / 1024.0),
            (value.x & 0x80000000 ? -1.0 : 0.0) + (((value.x >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    else
    {
        return asfloat(value);
    }
}

float4 tfetchTexcoord(uint swappedTexcoords, float4 value, uint semanticIndex)
{
    // sa2-nx (sa2_gpu_texcoords_unswapped): swappedTexcoords is always 0 in the native renderer.
    [branch] if (g_SpecConstants() & SPEC_CONSTANT_TEXCOORDS_UNSWAPPED)
        return value;
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}

// MASSEFFECT: 3 bits per component: 0-3 = data component, 4 = 0, 5 = 1, 7 = unchanged.
// sa2-nx: one component of remapInput as a select chain: value[source] with a runtime index can be lowered to
// scratch memory (local memory on Maxwell), the selects cannot. Same result: 0-3 pick a component, 4 is 0, 5 is 1,
// 6 and 7 keep the original component.
float masseffectRemapComponent(float4 value, float original, uint source)
{
    float result = original;
    result = source == 0 ? value.x : result;
    result = source == 1 ? value.y : result;
    result = source == 2 ? value.z : result;
    result = source == 3 ? value.w : result;
    result = source == 4 ? 0.0 : result;
    result = source == 5 ? 1.0 : result;
    return result;
}

float4 remapInput(float4 value, uint code)
{
    // sa2-nx (sa2_gpu_identity_remap): set only when every code of the pipeline's vertex input is 0xFFF, the early
    // return below.
    [branch] if (g_SpecConstants() & SPEC_CONSTANT_IDENTITY_REMAP)
        return value;
    // The code comes from the shared constants (the same for the whole draw): a uniform branch.
    [branch] if (code == 0xFFF)
        return value;

    return float4(masseffectRemapComponent(value, value.x, code & 7),
                  masseffectRemapComponent(value, value.y, (code >> 3) & 7),
                  masseffectRemapComponent(value, value.z, (code >> 6) & 7),
                  masseffectRemapComponent(value, value.w, (code >> 9) & 7));
}

// MASSEFFECT: Xenos alpha test with its comparison function. Positive if the pixel passes.
float alphaTestValue(float alpha)
{
    bool pass = true;
    switch ((g_SpecConstants() & SPEC_CONSTANT_ALPHA_FUNC_MASK) >> SPEC_CONSTANT_ALPHA_FUNC_SHIFT)
    {
    case 0: pass = false; break;
    case 1: pass = alpha < g_AlphaThreshold; break;
    case 2: pass = alpha == g_AlphaThreshold; break;
    case 3: pass = alpha <= g_AlphaThreshold; break;
    case 4: pass = alpha > g_AlphaThreshold; break;
    case 5: pass = alpha != g_AlphaThreshold; break;
    case 6: pass = alpha >= g_AlphaThreshold; break;
    default: break;
    }
    return pass ? 1.0 : -1.0;
}

float4 cube(float4 value, inout CubeMapData cubeMapData)
{
    uint index = cubeMapData.cubeMapIndex;
    cubeMapData.cubeMapDirections[index] = value.xyz;
    ++cubeMapData.cubeMapIndex;
    
    return float4(0.0, 0.0, 0.0, index);
}

float4 dst(float4 src0, float4 src1)
{
    float4 dest;
    dest.x = 1.0;
    dest.y = src0.y * src1.y;
    dest.z = src0.z;
    dest.w = src1.w;
    return dest;
}

float4 max4(float4 src0)
{
    return max(max(src0.x, src0.y), max(src0.z, src0.w));
}

float2 getPixelCoord(uint resourceDescriptorIndex, float2 texCoord)
{
    resourceDescriptorIndex &= 0x00FFFFFF;
    return getTexture2DDimensions(g_Texture2DDescriptorHeap[resourceDescriptorIndex]) * texCoord;
}

float computeMipLevel(float2 pixelCoord)
{
    float2 dx = ddx(pixelCoord);
    float2 dy = ddy(pixelCoord);
    float deltaMaxSqr = max(dot(dx, dx), dot(dy, dy));
    return max(0.0, 0.5 * log2(deltaMaxSqr));
}

#endif

#endif
