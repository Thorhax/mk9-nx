#ifndef SHADER_COMMON_H_INCLUDED
#define SHADER_COMMON_H_INCLUDED

#define SPEC_CONSTANT_R11G11B10_NORMAL  (1 << 0)
#define SPEC_CONSTANT_ALPHA_TEST        (1 << 1)

#ifdef UNLEASHED_RECOMP
    #define SPEC_CONSTANT_BICUBIC_GI_FILTER (1 << 2)
    #define SPEC_CONSTANT_ALPHA_TO_COVERAGE (1 << 3)
    #define SPEC_CONSTANT_REVERSE_Z         (1 << 4)
#endif

// SPIR-V vertex input locations, shared with reblue's host input-layout
// builder. A semantic missing here gets no location from the emitter.
#define REBLUE_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,  0,  0) \
    X(Position,  1,  1) \
    X(Position,  2,  2) \
    X(Position,  3,  3) \
    X(Position,  4,  4) \
    X(Normal,    0,  5) \
    X(Tangent,   0,  6) \
    X(TexCoord,  0,  7) \
    X(TexCoord,  1,  8) \
    X(TexCoord,  2,  9) \
    X(Color,     0, 10)

// SPIR-V vertex input locations for reNut, shared with its host input-layout
// builder. Covers every vertex element N&B's shaders declare. Like reblue's
// above, only a define: the recompiler reads it under RENUT_RECOMP.
#define RENUT_VERTEX_INPUT_LOCATIONS(X) \
    X(Position,  0,  0) \
    X(Position,  1,  1) \
    X(Position,  2,  2) \
    X(Position,  3,  3) \
    X(Position,  4,  4) \
    X(Normal,    0,  5) \
    X(Tangent,   0,  6) \
    X(Binormal,  0,  7) \
    X(TexCoord,  0,  8) \
    X(TexCoord,  1,  9) \
    X(TexCoord,  2, 10) \
    X(TexCoord,  3, 11) \
    X(Color,     0, 12) \
    X(Color,     1, 13) \
    X(BlendWeight,  0, 14) \
    X(BlendIndices, 0, 15) \
    X(TexCoord,  4, 16) \
    X(TexCoord,  5, 17) \
    X(TexCoord,  6, 18) \
    X(TexCoord,  7, 19) \
    X(TexCoord,  8, 20) \
    X(TexCoord,  9, 21) \
    X(TexCoord, 10, 22)

// SPEC_CONSTANTS_ONLY keeps the HLSL below out of host C++ TUs, which IntelliSense would otherwise parse.
#if (!defined(__cplusplus) || defined(__INTELLISENSE__)) && !defined(SHADER_COMMON_SPEC_CONSTANTS_ONLY)

#define FLT_MIN asfloat(0xff7fffff)
#define FLT_MAX asfloat(0x7f7fffff)

#ifdef __spirv__

// banjo-nx: 16-byte vectors of SharedConstants (render/draw.h, kSharedConstantsUboBytes / 16).
#define RENUT_SHARED_VECTORS 85  // banjo-nx (2026-10-19): 1360 bytes, with g_PixelOrigin

#ifndef BANJO_DEKO3D  // banjo-nx: the deko3d build has no push constants and no 64-bit addresses
struct PushConstants
{
    uint64_t VertexShaderConstants;
    uint64_t PixelShaderConstants;
    uint64_t SharedConstants;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> g_PushConstants;
#endif

#ifdef REBLUE_RECOMP
// 256-bit boolean register file (BD bool addresses reach ~158), then per-usage 16-bit-pair swap masks.
#define g_Booleans(i)              vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256 + (i)*4)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 288)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 292)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 300)
#define g_SwappedNormals           vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 304)
#define g_SwappedBinormals         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 308)
#define g_SwappedTangents          vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 312)
#define g_SwappedBlendWeights      vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 316)
#define g_SwappedPositions         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 320)
#define g_SintTexcoords            vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 324)
#define g_ShadowPcfScale           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 328)
#elif defined(RENUT_RECOMP)
// 0..383 texture descriptor indices (3 dimensions x 32 Xenos samplers), 384..511 sampler descriptor
// indices, 512..543 the 256-bit boolean register file (VS 0..127, PS 128..255), then render state:
// a draw normally has a host viewport made from its D3D viewport, and passes positions through;
// with the viewport transform off (D3DRS_VIEWPORTENABLE false) the vertex shader's positions are
// pixels, and 968 (g_ScreenSpace) asks it to map them to clip space: 568 and 544 hold what the
// transform left of scale and offset, 1328 the viewport (render target) size. For vfetches that
// read a buffer directly (no vertex element), the host copies each referenced
// vertex fetch constant's range into one per-draw allocation: 560 is that allocation's device
// address, and 576..959 hold one word per Xenos vertex fetch index (0..95): the byte offset of
// that constant's copy, with the constant's 2-bit endian mode in the low bits.
// banjo-nx (2026-10-17): the constants come from three dynamic uniform buffers (the vertex shader's
// float constants in set 5, the pixel shader's in set 6, SharedConstants in set 7), which NVK keeps in Maxwell's
// constant banks: a constant is a free operand of the instruction that reads it. Renut read them with
// vk::RawBufferLoad through the 64-bit addresses in g_PushConstants, which NAK compiles to global memory loads.
// Same bytes, same offsets. The host shaders (point_sprite_gs) still read SharedConstants through the push constant.
struct RenutUboFloat { float4 v[256]; };
struct RenutUboShared { uint4 v[RENUT_SHARED_VECTORS]; };
[[vk::binding(0, 5)]] ConstantBuffer<RenutUboFloat> g_UboVertex;
[[vk::binding(0, 6)]] ConstantBuffer<RenutUboFloat> g_UboPixel;
[[vk::binding(0, 7)]] ConstantBuffer<RenutUboShared> g_UboShared;
#define RENUT_SHARED_UINT(o)       (g_UboShared.v[uint(o) >> 4][(uint(o) >> 2) & 3u])
#define RENUT_SHARED_FLOAT(o)      asfloat(RENUT_SHARED_UINT(o))
#define RENUT_SHARED_FLOAT2(o)     float2(RENUT_SHARED_FLOAT(o), RENUT_SHARED_FLOAT((o) + 4))
#define RENUT_SHARED_UINT4(o)      (g_UboShared.v[uint(o) >> 4])
#define RENUT_SHARED_UINT64(o)     ((uint64_t(RENUT_SHARED_UINT((o) + 4)) << 32) | uint64_t(RENUT_SHARED_UINT(o)))
#define g_Booleans(i)              RENUT_SHARED_UINT(512 + (i)*4)
#define g_ScreenOffset             RENUT_SHARED_FLOAT2(544)
#define g_ScreenScale              RENUT_SHARED_FLOAT2(568)
#define g_ScreenSpace              RENUT_SHARED_UINT(968)
#define g_ViewportSize             RENUT_SHARED_FLOAT2(1328)
// 1336: host pixels per guest pixel on each axis (the render targets are that multiple of their
// guest size; g_ViewportSize and the screen-space values are in guest pixels).
#define g_ResolutionScale          RENUT_SHARED_FLOAT(1336)
// 1344 (Renut upstream 574b488, the Jinjo Bingo silhouette; ported 2026-10-19): in guest pixels, the corner of the tile
// a draw predicated to one tile renders to; the pixel position is relative to it, as the GPU's is to the EDRAM a tile
// renders in.
#define g_PixelOrigin              RENUT_SHARED_FLOAT2(1344)
#define g_AlphaThreshold           RENUT_SHARED_FLOAT(552)
#define g_AlphaTestFunc            RENUT_SHARED_UINT(556)
#define g_VertexFetchDataAddress   RENUT_SHARED_UINT64(560)
#define g_VertexFetchSlot(i)       RENUT_SHARED_UINT(576 + (i)*4)
// banjo-nx (package 27): 972 (was padding), the byte size of the draw's fetch data from g_VertexFetchDataAddress: the
// bound of vfetchMemoryClamped.
#define g_VertexFetchBytes         RENUT_SHARED_UINT(972)
// Memexport: 960 is the device address of the draw's export buffer, 976..1039 four stream
// windows (see memExport).
#define g_MemExportAddress         RENUT_SHARED_UINT64(960)
#define g_MemExportStream(i)       RENUT_SHARED_UINT4(976 + (i)*16)
// 1040: tessellated draws (see renutVertexRegisters).
#define g_Tessellation             RENUT_SHARED_UINT4(1040)
// 1056..1311: one element description per vertex input vfetchBound reads (see there).
#define g_BoundFetch(i)            RENUT_SHARED_UINT4(1056 + (i)*16)
// banjo-nx: the Xenos clamps' bounds (RCP, RSQ, LOG...: clamp(x, FLT_MIN, FLT_MAX), FLT_MIN being -FLT_MAX here)
// from SharedConstants' word at 1340, which the renderer sets to FLT_MAX: a constant-bank operand of NAK's min / max
// (with a negate modifier for FLT_MIN) instead of a 32-bit immediate NAK moves into a register before every clamp.
#undef FLT_MIN
#undef FLT_MAX
#define FLT_MAX                    RENUT_SHARED_FLOAT(1340)
#define FLT_MIN                    (-RENUT_SHARED_FLOAT(1340))
#else
#define g_Booleans                 vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 256)
#define g_SwappedTexcoords         vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + 260)
#define g_HalfPixelOffset          vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + 264)
#define g_AlphaThreshold           vk::RawBufferLoad<float>(g_PushConstants.SharedConstants + 272)
#endif

[[vk::constant_id(0)]] const uint g_SpecConstants = 0;

#define g_SpecConstants() g_SpecConstants

#else

#ifdef REBLUE_RECOMP
#define DEFINE_SHARED_CONSTANTS() \
    uint4 g_BooleansArr[2] : packoffset(c16); \
    uint g_SwappedTexcoords : packoffset(c18.x); \
    float2 g_HalfPixelOffset : packoffset(c18.y); \
    float g_AlphaThreshold : packoffset(c18.w); \
    uint g_SwappedNormals : packoffset(c19.x); \
    uint g_SwappedBinormals : packoffset(c19.y); \
    uint g_SwappedTangents : packoffset(c19.z); \
    uint g_SwappedBlendWeights : packoffset(c19.w); \
    uint g_SwappedPositions : packoffset(c20.x); \
    uint g_SintTexcoords : packoffset(c20.y); \
    float g_ShadowPcfScale : packoffset(c20.z);

#define g_Booleans(i) (g_BooleansArr[(i) / 4][(i) % 4])
#elif defined(RENUT_RECOMP)
#define DEFINE_SHARED_CONSTANTS() \
    uint4 g_BooleansArr[2] : packoffset(c32); \
    float2 g_ScreenOffset : packoffset(c34.x); \
    float g_AlphaThreshold : packoffset(c34.z); \
    uint g_AlphaTestFunc : packoffset(c34.w); \
    float2 g_ScreenScale : packoffset(c35.z); \
    uint4 g_VertexFetchSlotsArr[24] : packoffset(c36); \
    uint g_ScreenSpace : packoffset(c60.z); \
    uint g_VertexFetchBytes : packoffset(c60.w); \
    uint4 g_MemExportStreamsArr[4] : packoffset(c61); \
    uint4 g_Tessellation : packoffset(c65); \
    uint4 g_BoundFetchArr[16] : packoffset(c66); \
    float2 g_ViewportSize : packoffset(c83.x); \
    float g_ResolutionScale : packoffset(c83.z); \
    float2 g_PixelOrigin : packoffset(c84.x);

#define g_Booleans(i) (g_BooleansArr[(i) / 4][(i) % 4])
#define g_VertexFetchSlot(i) (g_VertexFetchSlotsArr[(i) / 4][(i) % 4])
#define g_MemExportStream(i) (g_MemExportStreamsArr[i])
#define g_BoundFetch(i) (g_BoundFetchArr[i])

// This draw's copies of the vertex fetch constants its shaders read directly (see above),
// bound as a root SRV.
ByteAddressBuffer g_VertexFetchData : register(t0, space5);

// The draw's memexport buffer (see memExport), bound as a root UAV.
RWByteAddressBuffer g_MemExport : register(u0, space6);
#else
#define DEFINE_SHARED_CONSTANTS() \
    uint g_Booleans : packoffset(c16.x); \
    uint g_SwappedTexcoords : packoffset(c16.y); \
    float2 g_HalfPixelOffset : packoffset(c16.z); \
    float g_AlphaThreshold : packoffset(c17.x);
#endif

uint g_SpecConstants();

#endif

#if defined(REBLUE_RECOMP) || defined(RENUT_RECOMP)
// Test Xenos boolean register N in the unified VS(0..127)/PS(128..255) file.
#define BOOL_BIT(n) ((g_Booleans((n) / 32u) & (1u << ((n) & 31u))) != 0)
#endif

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

#ifdef RENUT_RECOMP
// Descriptor indices carry flags above the slot: bits 29-31 the stacked texture flags (see
// RENUT_STACKED_TEXTURE), bits 26-28 the resolution scale code of a texture a resolve wrote. The
// host renders at a ratio of the guest resolution, and the textures resolved from its render
// targets are that ratio of their guest size (rounded); texel offsets and texel coordinates are in
// guest texels, so the sizes the fetches use are the guest's.
#define RENUT_DESCRIPTOR_INDEX_MASK 0x03FFFFFFu
#define RENUT_RESOLUTION_SCALE_SHIFT 26u

uint renutDescriptor(uint resourceDescriptorIndex)
{
    return resourceDescriptorIndex & RENUT_DESCRIPTOR_INDEX_MASK;
}

// Host texels per guest texel, by the host's RenderScale code (render_scale.h): 1x, 2/3 (480p),
// 1.5 (1080p), 2 (1440p), 3 (4K), 4, 6 and 4/3 (shadow maps and ambient occlusion rendered finer).
float renutTextureScale(uint resourceDescriptorIndex)
{
    const uint code = (resourceDescriptorIndex >> RENUT_RESOLUTION_SCALE_SHIFT) & 7u;
    // banjo-nx: select(), not a chain of ?: (a branch each; the code is the descriptor's, known only at run time).
    return select(code == 1u, 2.0 / 3.0,
           select(code == 2u, 1.5,
           select(code == 3u, 2.0,
           select(code == 4u, 3.0,
           select(code == 5u, 4.0,
           select(code == 6u, 6.0,
           select(code == 7u, 4.0 / 3.0, 1.0)))))));
}

// A scaled texture's guest width and height from its host ones.
uint2 renutGuestSize(uint2 hostSize, uint resourceDescriptorIndex)
{
    return uint2(round(float2(hostSize) / renutTextureScale(resourceDescriptorIndex)));
}

Texture2D<float4> getTexture2D(uint resourceDescriptorIndex)
{
    return g_Texture2DDescriptorHeap[renutDescriptor(resourceDescriptorIndex)];
}

// The guest size of a 2D texture.
uint2 getTexture2DSize(uint resourceDescriptorIndex)
{
    return renutGuestSize(getTexture2DDimensions(getTexture2D(resourceDescriptorIndex)), resourceDescriptorIndex);
}

float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    return getTexture2D(resourceDescriptorIndex).Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture2DSize(resourceDescriptorIndex));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DSize(resourceDescriptorIndex) + offset - 0.5));
}
#else
float4 tfetch2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return texture.Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord + offset / getTexture2DDimensions(texture));
}

float2 getWeights2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    return select(isnan(texCoord), 0.0, frac(texCoord * getTexture2DDimensions(texture) + offset - 0.5));
}
#endif

#ifdef REBLUE_RECOMP
// Bilinear-filtered shadow compare over the four neighboring depth texels.
float shadowCmp2D(uint resourceDescriptorIndex, float2 texCoord, float ref)
{
    Texture2D<float4> texture = g_Texture2DDescriptorHeap[resourceDescriptorIndex];
    int2 dimensions = int2(getTexture2DDimensions(texture));
    float2 coord = texCoord * dimensions - 0.5;
    float2 weights = frac(coord);
    int2 base = int2(floor(coord));
    int2 c0 = clamp(base, int2(0, 0), dimensions - 1);
    int2 c1 = clamp(base + 1, int2(0, 0), dimensions - 1);
    float4 taps = float4(
        texture.Load(int3(c0, 0)).x,
        texture.Load(int3(c1.x, c0.y, 0)).x,
        texture.Load(int3(c0.x, c1.y, 0)).x,
        texture.Load(int3(c1, 0)).x) > ref;
    return lerp(lerp(taps.x, taps.y, weights.x), lerp(taps.z, taps.w, weights.x), weights.y);
}
#endif

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

    return r;
}

float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord)
{
    return g_Texture3DDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord);
}

struct CubeMapData
{
	#ifdef REBLUE_RECOMP
    // BD's cube-shadow PCF issues up to 9 cube ops per shader (bd_mirror_cs_ps /
    // bd_glass_cs_ps); overflowing this array drops the stored directions and
    // every tfetchCube reads OOB -> point-light shadows vanish.
    float3 cubeMapDirections[9];
    #else
    float3 cubeMapDirections[2];
    #endif
    uint cubeMapIndex;
};

float4 tfetchCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
#ifdef RENUT_RECOMP
    resourceDescriptorIndex = renutDescriptor(resourceDescriptorIndex);
#endif
    return g_TextureCubeDescriptorHeap[resourceDescriptorIndex].Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]);
}

#ifdef RENUT_RECOMP
// Explicit-LOD fetches: vertex shaders have no derivatives, and setTexLOD
// fetches in pixel shaders sample the register LOD.
float4 tfetch2DLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset, float lod)
{
    return getTexture2D(resourceDescriptorIndex).SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture2DSize(resourceDescriptorIndex), lod);
}

uint3 getTexture3DDimensions(Texture3D<float4> texture)
{
    uint3 dimensions;
    texture.GetDimensions(dimensions.x, dimensions.y, dimensions.z);
    return dimensions;
}

Texture3D<float4> getTexture3D(uint resourceDescriptorIndex)
{
    return g_Texture3DDescriptorHeap[renutDescriptor(resourceDescriptorIndex)];
}

// The guest size of a 3D texture (see getTexture2DSize; slices are not scaled).
uint3 getTexture3DSize(uint resourceDescriptorIndex)
{
    const uint3 dimensions = getTexture3DDimensions(getTexture3D(resourceDescriptorIndex));
    return uint3(renutGuestSize(dimensions.xy, resourceDescriptorIndex), dimensions.z);
}

// Stacked textures (2D arrays) are fetched like 3D ones. The host binds them as 2D arrays and
// flags the 3D descriptor index: Xenos turns Z into a layer (Z times the layer count, plus the
// offset), keeps every layer in every mip, and blends the two nearest layers when the fetch
// constant's volume filter is linear. The layer is addressed with the fetch constant's Z mode,
// which the host sampler cannot do for an array index: wrapped when the mode is wrap (N&B's
// water scrolls Z through its animated noise and normal layers with time), else clamped.
Texture2DArray<float4> g_Texture2DArrayDescriptorHeap[] : register(t0, space4);
#define RENUT_STACKED_TEXTURE 0x80000000u
#define RENUT_STACKED_LAYER_LINEAR 0x40000000u
#define RENUT_STACKED_LAYER_WRAP 0x20000000u

bool isStackedTexture(uint resourceDescriptorIndex)
{
    return (resourceDescriptorIndex & RENUT_STACKED_TEXTURE) != 0u;
}

Texture2DArray<float4> getStackedTexture(uint resourceDescriptorIndex)
{
    return g_Texture2DArrayDescriptorHeap[resourceDescriptorIndex & RENUT_DESCRIPTOR_INDEX_MASK];
}

// The guest width and height (see getTexture2DSize) and the layer count.
uint3 getStackedTextureDimensions(uint resourceDescriptorIndex)
{
    uint3 dimensions;
    getStackedTexture(resourceDescriptorIndex).GetDimensions(dimensions.x, dimensions.y, dimensions.z);
    dimensions.xy = renutGuestSize(dimensions.xy, resourceDescriptorIndex);
    return dimensions;
}

// The two layers a stacked fetch reads, and the second one's weight.
struct StackedCoords
{
    float3 coord0;
    float3 coord1;
    float weight;
};

StackedCoords getStackedCoords(uint resourceDescriptorIndex, float3 texCoord, float3 offset)
{
    uint3 dimensions = getStackedTextureDimensions(resourceDescriptorIndex);
    float2 uv = texCoord.xy + offset.xy / float2(dimensions.xy);
    float layer = texCoord.z * float(dimensions.z) + offset.z;
    float first = floor(layer);
    StackedCoords coords;
    coords.weight = 0.0;
    if ((resourceDescriptorIndex & RENUT_STACKED_LAYER_LINEAR) != 0u)
    {
        first = floor(layer - 0.5);
        coords.weight = layer - 0.5 - first;
    }
    float layers = float(dimensions.z);
    float second = first + 1.0;
    if ((resourceDescriptorIndex & RENUT_STACKED_LAYER_WRAP) != 0u)
    {
        first -= layers * floor(first / layers);
        second -= layers * floor(second / layers);
    }
    float lastLayer = layers - 1.0;
    coords.coord0 = float3(uv, clamp(first, 0.0, lastLayer));
    coords.coord1 = float3(uv, clamp(second, 0.0, lastLayer));
    return coords;
}

// Both layers are always sampled: implicit derivatives need uniform control flow.
float4 tfetchStacked(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 offset)
{
    Texture2DArray<float4> texture = getStackedTexture(resourceDescriptorIndex);
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    StackedCoords coords = getStackedCoords(resourceDescriptorIndex, texCoord, offset);
    return lerp(texture.Sample(samplerState, coords.coord0), texture.Sample(samplerState, coords.coord1), coords.weight);
}

float4 tfetchStackedLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 offset, float lod)
{
    Texture2DArray<float4> texture = getStackedTexture(resourceDescriptorIndex);
    SamplerState samplerState = g_SamplerDescriptorHeap[samplerDescriptorIndex];
    StackedCoords coords = getStackedCoords(resourceDescriptorIndex, texCoord, offset);
    return lerp(texture.SampleLevel(samplerState, coords.coord0, lod), texture.SampleLevel(samplerState, coords.coord1, lod),
        coords.weight);
}

// 3D fetches with the instruction's texel offset (Xenos applies it natively).
float4 tfetch3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 offset)
{
    if (isStackedTexture(resourceDescriptorIndex))
        return tfetchStacked(resourceDescriptorIndex, samplerDescriptorIndex, texCoord, offset);
    return getTexture3D(resourceDescriptorIndex).Sample(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture3DSize(resourceDescriptorIndex));
}

float4 tfetch3DLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 offset, float lod)
{
    if (isStackedTexture(resourceDescriptorIndex))
        return tfetchStackedLevel(resourceDescriptorIndex, samplerDescriptorIndex, texCoord, offset, lod);
    return getTexture3D(resourceDescriptorIndex).SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture3DSize(resourceDescriptorIndex), lod);
}

// Fetches with denormalized coordinates (texCoordDenorm) address texels; sample with UVs instead.
float2 texelToUV1D(uint resourceDescriptorIndex, float texCoord)
{
    // 1D textures are uploaded as 2D with a height of 1.
    return float2(texCoord / getTexture2DSize(resourceDescriptorIndex).x, 0.5);
}

float2 texelToUV2D(uint resourceDescriptorIndex, float2 texCoord)
{
    return texCoord / getTexture2DSize(resourceDescriptorIndex);
}

float3 texelToUV3D(uint resourceDescriptorIndex, float3 texCoord)
{
    if (isStackedTexture(resourceDescriptorIndex))
        return texCoord / float3(getStackedTextureDimensions(resourceDescriptorIndex));
    return texCoord / getTexture3DSize(resourceDescriptorIndex);
}

// getCompTexLOD: the LOD the hardware would select for a fetch at this coordinate (result in .x).
float4 getCompTexLod2D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float2 texCoord, float2 offset)
{
    return float4(getTexture2D(resourceDescriptorIndex).CalculateLevelOfDetailUnclamped(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture2DSize(resourceDescriptorIndex)), 0.0, 0.0, 0.0);
}

float4 getCompTexLod3D(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, float3 offset)
{
    if (isStackedTexture(resourceDescriptorIndex))
    {
        uint3 dimensions = getStackedTextureDimensions(resourceDescriptorIndex);
        return float4(getStackedTexture(resourceDescriptorIndex).CalculateLevelOfDetailUnclamped(
            g_SamplerDescriptorHeap[samplerDescriptorIndex], texCoord.xy + offset.xy / float2(dimensions.xy)), 0.0, 0.0, 0.0);
    }
    return float4(getTexture3D(resourceDescriptorIndex).CalculateLevelOfDetailUnclamped(g_SamplerDescriptorHeap[samplerDescriptorIndex],
        texCoord + offset / getTexture3DSize(resourceDescriptorIndex)), 0.0, 0.0, 0.0);
}

float4 getCompTexLodCube(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData)
{
    return float4(g_TextureCubeDescriptorHeap[renutDescriptor(resourceDescriptorIndex)].CalculateLevelOfDetailUnclamped(
        g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z]), 0.0, 0.0, 0.0);
}

float4 tfetchCubeLevel(uint resourceDescriptorIndex, uint samplerDescriptorIndex, float3 texCoord, inout CubeMapData cubeMapData, float lod)
{
    return g_TextureCubeDescriptorHeap[renutDescriptor(resourceDescriptorIndex)].SampleLevel(g_SamplerDescriptorHeap[samplerDescriptorIndex], cubeMapData.cubeMapDirections[texCoord.z], lod);
}

// Xenos rcpf / rsqf: infinities become 0 rather than clamping to FLT_MAX.
// banjo-nx: select(), not ?: (HLSL 2021 short-circuits ?:, which DXC writes as a branch).
float rcpf(float value)
{
    float result = 1.0 / value;
    return select(isinf(result), 0.0, result);
}

float rsqf(float value)
{
    float result = rsqrt(value);
    return select(isinf(result), 0.0, result);
}

// Xenos alpha test (fixed-function in the ROP): RB_COLORCONTROL ALPHA_FUNC, 0 NEVER, 1 LESS,
// 2 EQUAL, 3 LEQUAL, 4 GREATER, 5 NOTEQUAL, 6 GEQUAL, 7 ALWAYS. The host passes 7 when the
// test is disabled. Arguments rather than globals: the cbuffer is declared after this file.
// banjo-nx (2026-10-17): the function picked with select() instead of a switch on it (a branch per case in every pixel
// shader with the test; the function is the draw's, from the shared constants). The same comparisons, NaN included.
// banjo-nx (banjo_gpu_early_alpha, package 27): what the early alpha copies tools/renut_early_alpha.py adds to the
// pixel shaders call (tools/early_alpha.py's forms, banjo-nx's translator): a copy of the statements the alpha needs,
// run where the shader has what it needs, and a 2x2 quad whose four pixels all fail the alpha test there is discarded
// before the rest of the shader (no pixel still shaded loses a neighbour its derivatives need).
#define banjoAnd(a, b) and(a, b)
#define banjoOr(a, b) or(a, b)
#define banjoSel(c, a, b) select(c, a, b)
#define BANJO_BOOL(n) BANJO_BOOL_##n

// Fails a pixel only when the test fails for every alpha within (|alpha| + |reference|) * 2^-12 + 1e-30 of the copy's
// (a multiply-add fused in one place and not the other, a flushed denormal): banjo-nx's banjoAlphaFailsEarly, with the
// draw's function and reference given at run time (Renut's renderer does not specialise them). Functions as the alpha
// test's (0 never ... 7 always); not-equal and always never fail early; NaN never does.
bool banjoAlphaFailsEarlyRt(float alpha, uint func, float reference)
{
    const float d = (abs(alpha) + abs(reference)) * (1.0 / 4096.0) + 1.0e-30;
    return select(func == 0u, true,
           select(func == 1u, alpha - d >= reference,
           select(func == 2u, abs(alpha - reference) > d,
           select(func == 3u, alpha - d > reference,
           select(func == 4u, alpha + d <= reference,
           select(func == 6u, alpha + d < reference, false))))));
}

// Discards the 2x2 quad when its four pixels all fail: fine derivatives of a 0/1 value, first along x (a pixel and its
// horizontal neighbour), then of that pair along y.
void banjoEarlyAlphaQuad(bool fails)
{
    const float f = select(fails, 1.0, 0.0);
    const float dx = ddx_fine(f);
    const float pair = select(and(f == 1.0, dx == 0.0), 1.0, 0.0);
    const float dy = ddy_fine(pair);
    if (pair == 1.0 && dy == 0.0)
        discard;
}

// banjo-nx (banjo_gpu_blend_skip, package 27, from banjo-nx's renderer / sa2-nx): the pipeline's blend would leave its
// one target exactly as it was, so the pixel is discarded before the blend. Specialization constant 900, set by the
// renderer from the pipeline key (BlendSkipMode in render/pipeline.cpp): 1 when the target keeps its value for a
// source alpha of 0 or less, 2 when it does for a whole source of 0 or less (a UNORM target, which clamps the source to
// [0, 1] before the blend; no depth or stencil write; no alpha to coverage). A NaN output is never discarded. Called at
// every exit of a pixel shader (shader_recompiler.cpp, after the alpha test). SPIR-V only.
#ifdef __spirv__
[[vk::constant_id(900)]] const uint g_banjoBlendSkip = 0;
#endif
void banjoBlendSkip(float4 c)
{
#ifdef __spirv__
    if ((g_banjoBlendSkip == 1u && c.w <= 0.0 && !any(isnan(c.xyz))) || (g_banjoBlendSkip == 2u && all(c <= 0.0)))
        discard;
#endif
}

// banjo-nx (banjo_gpu_output_trim, package 28; Unleashed / Marathon's trimmed pixel outputs): the colour channels the
// pipeline never uses are 0 from the shader's exits on, so the driver, which specialises the constant, drops what
// computed only them. Specialization constant 901, 4 bits a target (RGBA as the colour write mask), set by the renderer
// from the pipeline key (OutputTrimMask in render/pipeline.cpp): a channel the write mask or the target's format leaves
// out, and the alpha only when no written colour channel blends with it and there is no alpha to coverage. Called at
// every exit of a pixel shader after the alpha test and the blend skip. SPIR-V only.
#ifdef __spirv__
[[vk::constant_id(901)]] const uint g_banjoOutputTrim = 0;
// The test (banjo_renut_diag_trim_poison): the trimmed channels are 60000 instead of 0, so one trimmed wrongly shows.
[[vk::constant_id(902)]] const uint g_banjoTrimPoison = 0;
#endif
float4 banjoTrim(float4 c, uint target)
{
#ifdef __spirv__
    const uint m = (g_banjoOutputTrim >> (target * 4u)) & 0xFu;
    const float v = g_banjoTrimPoison != 0u ? 60000.0 : 0.0;
    return select(bool4((m & 1u) != 0u, (m & 2u) != 0u, (m & 4u) != 0u, (m & 8u) != 0u), float4(v, v, v, v), c);
#else
    return c;
#endif
}

void alphaTest(float alpha, uint func, float reference)
{
    const bool pass = select(func == 0u, false,
                      select(func == 1u, alpha < reference,
                      select(func == 2u, alpha == reference,
                      select(func == 3u, alpha <= reference,
                      select(func == 4u, alpha > reference,
                      select(func == 5u, alpha != reference,
                      select(func == 6u, alpha >= reference, true)))))));
    if (!pass)
        discard;
}

// Vertex fetch without a vertex element: N&B reads bone/instance data with computed indices, and
// some shaders without a vertex declaration read their stream by vertex index. The fetch reads the
// host's per-draw copy of the fetch constant's buffer, applies its endian swap and decodes the format.

#if defined(__spirv__) && defined(BANJO_DEKO3D)
// banjo-nx (deko3d renderer, docs/deko3d-renderer.md section 5): uam compiles no buffer references. The back end binds
// this storage buffer (set and binding 15, deko3d storage binding 15) over the 4 GB window of GPU addresses that holds
// every buffer a shader addresses (the window starts at an address with its low 32 bits zero), so an address's low
// word is the byte offset. Read only: vertex fetch data. memexport writes through its own (binding 14), which the
// renderer binds to the draw's memexport buffer (draw.cpp): a writable binding over every buffer made Ryujinx flush
// that whole range back over memory the heap had reused.
[[vk::binding(15, 15)]] ByteAddressBuffer g_DkAddressWindow;
[[vk::binding(14, 15)]] RWByteAddressBuffer g_DkMemExport;
#define BANJO_DK_ADDRESS_LOW(o)    RENUT_SHARED_UINT(o)
#endif

// banjo-nx (deko3d renderer): exact unsigned division. Maxwell has no integer divide, and uam lowers a non-constant one
// to float math that can be off by one. A float estimate scaled just below the quotient (so q * b never passes a), a
// second estimate from the remainder, then integer steps: at most two for any 32-bit a and b > 0. Other builds divide.
uint banjoUDiv(uint a, uint b)
{
    const float below = 0.99999;
    uint q = uint(float(a) / float(b) * below);
    uint r = a - q * b;
    q += uint(float(r) / float(b) * below);
    r = a - q * b;
    [loop] while (r >= b)
    {
        q += 1u;
        r -= b;
    }
    return q;
}
#ifdef BANJO_DEKO3D
#define BANJO_UDIV(a, b) banjoUDiv((a), (b))
#else
#define BANJO_UDIV(a, b) ((a) / (b))
#endif

uint loadVertexFetchData(uint byteOffset)
{
#if defined(__spirv__) && defined(BANJO_DEKO3D)
    return g_DkAddressWindow.Load(BANJO_DK_ADDRESS_LOW(560) + byteOffset);
#elif defined(__spirv__)
    return vk::RawBufferLoad<uint>(g_VertexFetchDataAddress + byteOffset, 4);
#else
    return g_VertexFetchData.Load(byteOffset);
#endif
}

// banjo-nx (package 27, banjo-nx's banjo_gpu_vfetch_block): four words in one 16-byte load. Only at a 16-byte aligned
// address: every copy of the draw's fetch data starts 16-byte aligned (UploadVertexFetchData), so a fetch whose offset
// and stride are whole rows (multiples of four words) is aligned at every index.
uint4 loadVertexFetchData4(uint byteOffset)
{
#if defined(__spirv__) && defined(BANJO_DEKO3D)
    return g_DkAddressWindow.Load4(BANJO_DK_ADDRESS_LOW(560) + byteOffset);
#elif defined(__spirv__)
    return vk::RawBufferLoad<uint4>(g_VertexFetchDataAddress + byteOffset, 16);
#else
    return g_VertexFetchData.Load4(byteOffset);
#endif
}

// banjo-nx (2026-10-17): without a branch. The endian mode comes from the draw's constants, so the if-chain was a
// branch per word (four words a fetch, twelve bone fetches in a skinning vertex shader); all three swaps are a few
// shifts, and select() keeps the one the mode names. Same value for every mode.
uint vfetchSwap(uint value, uint endian)
{
    const uint swap8in16 = ((value & 0x00FF00FFu) << 8) | ((value >> 8) & 0x00FF00FFu);
    const uint swap16in32 = (value << 16) | (value >> 16);
    const uint swap8in32 = (swap8in16 << 16) | (swap8in16 >> 16);
    return select(endian == 1u, swap8in16, select(endian == 2u, swap8in32, select(endian == 3u, swap16in32, value)));
}

// One packed integer component: signed/unsigned, normalized ("fraction") or integer.
float vfetchComponent(uint value, uint shift, uint bits, bool isSigned, bool isInteger, bool signedRfMode)
{
    uint raw = bits == 32 ? value : (value >> shift) & ((1u << bits) - 1u);
    if (isSigned)
    {
        int s = bits == 32 ? int(raw) : (int(raw << (32u - bits)) >> (32u - bits));
        if (isInteger)
            return float(s);
        float maxValue = bits == 32 ? 2147483647.0 : float((1u << (bits - 1u)) - 1u);
        return signedRfMode ? (float(s) * 2.0 + 1.0) / (maxValue * 2.0 + 1.0) : max(float(s) / maxValue, -1.0);
    }
    if (isInteger)
        return float(raw);
    return float(raw) / (bits == 32 ? 4294967295.0 : float((1u << bits) - 1u));
}

// slot = g_VertexFetchSlot(fetch index): copy byte offset | endian mode. Packed components count
// from the low bits of the swapped word, 16-bit ones too: under the usual 8in32 swap x is the
// second halfword in memory, which N&B's shaders swizzle back (.yx, .yxwz on 16-bit fetches).
// banjo-nx (package 27): 'rowsAllowed' (vfetchMemory, vfetchMemoryClamped: their offset and stride are literals at the
// call site, so the test below folds there) loads a whole-row fetch in one 16-byte load (BANJO_VFETCH_ROWS, on unless
// the build turns it off); 'clampAddress' (vfetchMemoryClamped) keeps the address inside the draw's fetch data
// ('fetchBytes', g_VertexFetchBytes, passed by the macro below: on DXIL the shared constants are declared after this
// header), so the predicated-block pass can run the fetch for every vertex and keep its result only where the
// predicate holds (banjo_pred_select.h): where it holds the address is already inside, and where it does not the value
// is discarded.
#ifndef BANJO_VFETCH_ROWS
#define BANJO_VFETCH_ROWS 1
#endif
float4 vfetchMemoryImpl(uint slot, float indexValue, bool roundIndex, int offsetDwords, uint strideDwords,
    uint format, bool isSigned, bool isInteger, bool signedRfMode, int expAdjust, bool rowsAllowed, bool clampAddress,
    uint fetchBytes)
{
    int index = int(roundIndex ? floor(indexValue + 0.5) : floor(indexValue));
    uint address = (slot & ~0x3u) + uint(index * int(strideDwords) + offsetDwords) * 4u;
    if (clampAddress)
        address = min(address, max(fetchBytes, 16u) - 16u);
    uint endian = slot & 0x3u;

    uint4 w;
    if (BANJO_VFETCH_ROWS && rowsAllowed && ((uint(offsetDwords) | strideDwords) & 3u) == 0u)
    {
        const uint4 v = loadVertexFetchData4(address);
        w = uint4(vfetchSwap(v.x, endian), vfetchSwap(v.y, endian), vfetchSwap(v.z, endian), vfetchSwap(v.w, endian));
    }
    else
    {
        w.x = vfetchSwap(loadVertexFetchData(address), endian);
        w.y = vfetchSwap(loadVertexFetchData(address + 4), endian);
        w.z = vfetchSwap(loadVertexFetchData(address + 8), endian);
        w.w = vfetchSwap(loadVertexFetchData(address + 12), endian);
    }

    float4 result = float4(0.0, 0.0, 0.0, 1.0);
    bool isFloat = false;
    switch (format)
    {
    case 6: // 8_8_8_8
        result = float4(vfetchComponent(w.x, 0, 8, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 8, 8, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.x, 16, 8, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 24, 8, isSigned, isInteger, signedRfMode));
        break;
    case 7: // 2_10_10_10
        result = float4(vfetchComponent(w.x, 0, 10, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 10, 10, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.x, 20, 10, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 30, 2, isSigned, isInteger, signedRfMode));
        break;
    case 16: // 10_11_11
        result.xyz = float3(vfetchComponent(w.x, 0, 11, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 11, 11, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.x, 22, 10, isSigned, isInteger, signedRfMode));
        break;
    case 17: // 11_11_10
        result.xyz = float3(vfetchComponent(w.x, 0, 10, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 10, 11, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.x, 21, 11, isSigned, isInteger, signedRfMode));
        break;
    case 25: // 16_16
        result.xy = float2(vfetchComponent(w.x, 0, 16, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 16, 16, isSigned, isInteger, signedRfMode));
        break;
    case 26: // 16_16_16_16
        result = float4(vfetchComponent(w.x, 0, 16, isSigned, isInteger, signedRfMode), vfetchComponent(w.x, 16, 16, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.y, 0, 16, isSigned, isInteger, signedRfMode), vfetchComponent(w.y, 16, 16, isSigned, isInteger, signedRfMode));
        break;
    case 31: // 16_16_FLOAT
        result.xy = f16tof32(uint2(w.x & 0xFFFF, w.x >> 16));
        isFloat = true;
        break;
    case 32: // 16_16_16_16_FLOAT
        result = f16tof32(uint4(w.x & 0xFFFF, w.x >> 16, w.y & 0xFFFF, w.y >> 16));
        isFloat = true;
        break;
    case 33: // 32
        result.x = vfetchComponent(w.x, 0, 32, isSigned, isInteger, signedRfMode);
        break;
    case 34: // 32_32
        result.xy = float2(vfetchComponent(w.x, 0, 32, isSigned, isInteger, signedRfMode), vfetchComponent(w.y, 0, 32, isSigned, isInteger, signedRfMode));
        break;
    case 35: // 32_32_32_32
        result = float4(vfetchComponent(w.x, 0, 32, isSigned, isInteger, signedRfMode), vfetchComponent(w.y, 0, 32, isSigned, isInteger, signedRfMode),
            vfetchComponent(w.z, 0, 32, isSigned, isInteger, signedRfMode), vfetchComponent(w.w, 0, 32, isSigned, isInteger, signedRfMode));
        break;
    case 36: // 32_FLOAT
        result.x = asfloat(w.x);
        isFloat = true;
        break;
    case 37: // 32_32_FLOAT
        result.xy = asfloat(w.xy);
        isFloat = true;
        break;
    case 38: // 32_32_32_32_FLOAT
        result = asfloat(w);
        isFloat = true;
        break;
    case 57: // 32_32_32_FLOAT
        result.xyz = asfloat(w.xyz);
        isFloat = true;
        break;
    }

    if (!isFloat && expAdjust != 0)
        result *= exp2(float(expAdjust));

    return result;
}

float4 vfetchMemory(uint slot, float indexValue, bool roundIndex, int offsetDwords, uint strideDwords,
    uint format, bool isSigned, bool isInteger, bool signedRfMode, int expAdjust)
{
    return vfetchMemoryImpl(slot, indexValue, roundIndex, offsetDwords, strideDwords, format, isSigned, isInteger,
        signedRfMode, expAdjust, true, false, 0u);
}

#define vfetchMemoryClamped(slot, indexValue, roundIndex, offsetDwords, strideDwords, format, isSigned, isInteger, \
    signedRfMode, expAdjust) \
    vfetchMemoryImpl(slot, indexValue, roundIndex, offsetDwords, strideDwords, format, isSigned, isInteger, \
        signedRfMode, expAdjust, true, true, g_VertexFetchBytes)

// Vertex fetch of a bound declaration's element by an index other than the vertex index (N&B's
// leaf cards read their per-card position and rotation at vertex index / 4). D3D patches the
// element's format, swizzle and stride into the fetch at bind time, so the microcode has none;
// the host describes the element instead. element = g_BoundFetch(input): x its stream's vertex
// fetch index, y its offset and z the stream stride in dwords, w its D3DDECLTYPE (bits 0-5
// format, 8 signed, 9 integer, 10-21 four 3-bit swizzles: 0-3 xyzw, 4 zero, 5 one).
// slot = g_VertexFetchSlot(element.x).
float4 vfetchBound(uint4 element, uint slot, float indexValue, bool roundIndex, bool signedRfMode, int expAdjust)
{
    // banjo-nx: word by word (its offset and stride come from the draw's constants, not the call site).
    float4 value = vfetchMemoryImpl(slot, indexValue, roundIndex, int(element.y), element.z, element.w & 0x3Fu,
        (element.w & 0x100u) != 0, (element.w & 0x200u) != 0, signedRfMode, expAdjust, false, false, 0u);
    float4 result;
    [unroll] for (uint i = 0; i < 4; i++)
    {
        uint swizzle = (element.w >> (10u + 3u * i)) & 7u;
        result[i] = swizzle < 4 ? value[swizzle] : (swizzle == 5 ? 1.0 : 0.0);
    }
    return result;
}

// What a vertex shader finds in r0 and r1 when it starts. For ordinary draws Xenos loads the
// vertex index into r0.x. Tessellated draws (triangle or quad patches) run the shader once per
// tessellated vertex instead; the host draws each patch as one instance of level^2 triangles
// (level^2 cells of two triangles for quads), with the per-patch vertex data as instance data,
// and this derives each vertex's patch and domain location from the vertex and instance IDs.
// tessellation = g_Tessellation, passed by the caller (on DXIL it lives in the shader's own
// constant buffer, declared after this header): x 0 ordinary, 1 triangle patches, 2 quad
// patches; y the segments per patch edge; z the first patch index; w, for ordinary draws, what
// the vertex ID lacks of the guest vertex index (the host draws the vertex range in use from 0).
// N&B tessellates only in adaptive mode (sub_823BA3A8 sets it before each of its
// DrawIndexedTessellatedVertices calls), and the prologue the XDK compiler writes for its patch
// shaders gives the layout. Triangle patches (vs_631d40df78b583f7): the barycentric weights of
// control points 2, 1, 0 in r0.xyz, the patch index in r1.x (it fetches the control points
// there), and in r1.y a corner permutation, 0 the identity. Quad patches (vs_b2740b93db9023c2,
// vs_c1cb11c6c0f59fb8): the patch index in r0.x, the domain (u, v) in r0.yz, and in r1.x a corner
// rotation, 0 the identity.
void renutVertexRegisters(uint vertexId, uint instanceId, uint4 tessellation, out float4 r0, out float4 r1)
{
    r0 = float4(vertexId + tessellation.w, 0.0, 0.0, 0.0);
    r1 = 0.0;
    if (tessellation.x == 0u)
        return;

    uint level = max(tessellation.y, 1u);
    float patchIndex = float(tessellation.z + instanceId);
    // banjo-nx (banjo_gpu_tess_grid, package 27, from banjo-nx's renderer): kinds 3 and 4 are kinds 1 and 2 drawn
    // indexed, the vertex ID a grid vertex rather than a triangle corner, so the shader runs once per grid vertex instead
    // of once per corner (about six times). The host's index list (TessellationGrid in render/draw.cpp) names the same
    // triangles in the same order with the same corners, and each grid vertex gets the same values as below.
    if (tessellation.x == 3u)
    {
        // Vertex (a, b), b <= a <= level, numbered a (a + 1) / 2 + b.
        uint a = uint((sqrt(float(8u * vertexId + 1u)) - 1.0) * 0.5);
        if (a * (a + 1u) / 2u > vertexId)
            --a;
        else if ((a + 1u) * (a + 2u) / 2u <= vertexId)
            ++a;
        uint2 vertex = uint2(a, vertexId - a * (a + 1u) / 2u);
        float3 weights = float3(level - vertex.x, vertex.x - vertex.y, vertex.y) / float(level);
        r0.xyz = weights.zyx;
        r1.x = patchIndex;
        return;
    }
    if (tessellation.x == 4u)
    {
        // Vertex (u, v), numbered v (level + 1) + u.
        const uint row = BANJO_UDIV(vertexId, level + 1u);  // banjo-nx: exact on Maxwell (deko3d)
        uint2 position = uint2(vertexId - row * (level + 1u), row);
        float2 uv = float2(position) / float(level);
        r0 = float4(patchIndex, uv, 0.0);
        return;
    }
    uint triangleIndex = vertexId / 3u;
    uint corner = vertexId % 3u;
    if (tessellation.x == 1u)
    {
        // Row r of the grid (from control point 0) holds 2r + 1 triangles, alternately pointing
        // away from and towards control point 0. Grid vertex (a, b), b <= a <= level, has
        // weights (level - a, a - b, b) / level for control points 0, 1, 2; every triangle keeps
        // the patch's winding.
        uint row = uint(sqrt(float(triangleIndex)));
        if (row * row > triangleIndex)
            --row;
        else if ((row + 1u) * (row + 1u) <= triangleIndex)
            ++row;
        uint k = triangleIndex - row * row;
        uint b = k / 2u;
        uint2 vertex;
        if ((k & 1u) == 0u)
            vertex = corner == 0u ? uint2(row, b) : (corner == 1u ? uint2(row + 1u, b) : uint2(row + 1u, b + 1u));
        else
            vertex = corner == 0u ? uint2(row, b) : (corner == 1u ? uint2(row + 1u, b + 1u) : uint2(row, b + 1u));
        float3 weights = float3(level - vertex.x, vertex.x - vertex.y, vertex.y) / float(level);
        r0.xyz = weights.zyx;
        r1.x = patchIndex;
    }
    else
    {
        uint cell = triangleIndex / 2u;
        const uint cellRow = BANJO_UDIV(cell, level);  // banjo-nx: exact on Maxwell (deko3d)
        uint2 position = uint2(cell - cellRow * level, cellRow);
        uint2 offset;
        if ((triangleIndex & 1u) == 0u)
            offset = corner == 0u ? uint2(0, 0) : (corner == 1u ? uint2(1, 0) : uint2(1, 1));
        else
            offset = corner == 0u ? uint2(0, 0) : (corner == 1u ? uint2(1, 1) : uint2(0, 1));
        float2 uv = float2(position + offset) / float(level);
        r0 = float4(patchIndex, uv, 0.0);
    }
}

// Memexport, as N&B's own export passes use it (sub_82317020 builds their stream constants before
// D3DDevice_EndExport): the shader computes eA = element index * (0, 1, 0, 0) + stream constant,
// so eA.x keeps the constant's raw bits (the GPU address in dwords, bits 0-29, tagged 01 in
// 30-31), eA.y and eA.w are floats 2^23 above the element index and the element count, and eA.z
// packs the endian (bits 0-2), format (8-13) and number format (16-18). eM# writes the element
// index + #. The game exports whole-dword elements in 8in32 order (32_32_32_32_FLOAT and
// 32_32_FLOAT, number format 7); formats of other element sizes are not written.
// The host binds one buffer per draw with a window for each stream constant the shader uses:
// g_MemExportStream(i) = (guest byte address, size in bytes, byte offset in the buffer, 0). The
// windows hold guest memory byte order; the host copies them back to guest memory.

void memExportStore(uint offset, uint value)
{
#if defined(__spirv__) && defined(BANJO_DEKO3D)
    g_DkMemExport.Store(offset, value);  // offset from the memexport buffer's start, which g_MemExportAddress is
#elif defined(__spirv__)
    vk::RawBufferStore<uint>(g_MemExportAddress + offset, value, 4);
#else
    g_MemExport.Store(offset, value);
#endif
}

// The element's dwords as the export writes them: float formats keep the float bits (halves for
// the 16-bit ones), the others scale each component to its bit width (bit 0 of the number format
// signed, bit 1 integer). Returns the dword count, 0 for a format not written.
uint memExportElement(float4 data, uint format, uint numFormat, out uint4 words)
{
    words = 0;
    switch (format)
    {
    case 36: // 32_FLOAT
        words.x = asuint(data.x);
        return 1;
    case 37: // 32_32_FLOAT
        words.xy = asuint(data.xy);
        return 2;
    case 38: // 32_32_32_32_FLOAT
        words = asuint(data);
        return 4;
    case 31: // 16_16_FLOAT
        words.x = f32tof16(data.x) | (f32tof16(data.y) << 16);
        return 1;
    case 32: // 16_16_16_16_FLOAT
        words.xy = f32tof16(data.xz) | (f32tof16(data.yw) << 16);
        return 2;
    }

    // Integer and normalized formats: component widths, in dword order.
    uint4 widths;
    uint dwords;
    switch (format)
    {
    case 6: widths = uint4(8, 8, 8, 8); dwords = 1; break;       // 8_8_8_8
    case 7: widths = uint4(10, 10, 10, 2); dwords = 1; break;    // 2_10_10_10
    case 25: widths = uint4(16, 16, 0, 0); dwords = 1; break;    // 16_16
    case 26: widths = uint4(16, 16, 16, 16); dwords = 2; break;  // 16_16_16_16
    case 33: widths = uint4(32, 0, 0, 0); dwords = 1; break;     // 32
    case 34: widths = uint4(32, 32, 0, 0); dwords = 2; break;    // 32_32
    case 35: widths = uint4(32, 32, 32, 32); dwords = 4; break;  // 32_32_32_32
    default: return 0;
    }
    const bool isSigned = (numFormat & 1u) != 0;
    const bool isInteger = (numFormat & 2u) != 0;
    uint bit = 0;
    for (uint c = 0; c < 4u; c++)
    {
        const uint width = widths[c];
        if (width == 0u)
            break;
        float v = data[c];
        uint packed;
        if (width == 32u)
        {
            packed = isInteger ? (isSigned ? uint(int(v)) : uint(max(v, 0.0)))
                               : (isSigned ? uint(int(clamp(v, -1.0, 1.0) * 2147483647.0))
                                           : uint(saturate(v) * 4294967295.0));
        }
        else
        {
            const float top = isSigned ? float((1u << (width - 1u)) - 1u) : float((1u << width) - 1u);
            const float scaled = isInteger ? v : round((isSigned ? clamp(v, -1.0, 1.0) : saturate(v)) * top);
            packed = uint(int(scaled)) & ((1u << width) - 1u);
        }
        const uint dword = bit >> 5;
        words[dword] |= packed << (bit & 31u);
        bit += width;
    }
    return dwords;
}

// The fetch/export endian swap of one dword (the 8in32 N&B uses, and the other 32-bit modes).
uint memExportSwap(uint value, uint endian)
{
    if (endian == 2u)
        return (value >> 24) | ((value >> 8) & 0xFF00u) | ((value << 8) & 0xFF0000u) | (value << 24);
    if (endian == 1u)
        return ((value >> 8) & 0x00FF00FFu) | ((value << 8) & 0xFF00FF00u);
    if (endian == 3u)
        return (value >> 16) | (value << 16);
    return value;
}

// stream0..3 = g_MemExportStream(0..3), passed by the caller: on DXIL they live in the shader's
// own constant buffer, declared after this header.
void memExport(float4 eA, float4 data, uint element, uint4 stream0, uint4 stream1, uint4 stream2,
    uint4 stream3)
{
    const float firstIndex = eA.y - 8388608.0;
    const float count = eA.w - 8388608.0;
    if (!(firstIndex >= 0.0) || !(count >= 0.0))
        return;
    const uint index = uint(firstIndex) + element;
    if (index >= uint(count))
        return;

    const uint info = asuint(eA.z);
    uint4 words;
    const uint dwords = memExportElement(data, (info >> 8) & 0x3Fu, (info >> 16) & 7u, words);
    if (dwords == 0u || (info & 7u) > 3u)
        return;

    const uint bytes = dwords * 4u;
    const uint address = (asuint(eA.x) & 0x3FFFFFFFu) * 4u + index * bytes;
    const uint4 streams[4] = { stream0, stream1, stream2, stream3 };
    for (uint i = 0; i < 4u; i++)
    {
        const uint4 stream = streams[i];
        if (address < stream.x || address + bytes > stream.x + stream.y)
            continue;
        const uint offset = stream.z + (address - stream.x);
        for (uint d = 0; d < dwords; d++)
            memExportStore(offset + d * 4u, memExportSwap(words[d], info & 7u));
        return;
    }
}
#endif

#ifdef REBLUE_RECOMP
// DEC3N normal decode; IA binds as R32_UINT so lane .x carries the raw bits (asuint recovers them).
float4 tfetchR11G11B10(float4 value)
{
    if (g_SpecConstants() & SPEC_CONSTANT_R11G11B10_NORMAL)
    {
        uint v = asuint(value.x);
        return float4(
            (v & 0x00000400 ? -1.0 : 0.0) + ((v & 0x3FF) / 1024.0),
            (v & 0x00200000 ? -1.0 : 0.0) + (((v >> 11) & 0x3FF) / 1024.0),
            (v & 0x80000000 ? -1.0 : 0.0) + (((v >> 22) & 0x1FF) / 512.0),
            0.0);
    }
    return value;
}

// Undo the engine bswap32 16-bit-pair swap (.yxwz) for any 16-bit-packed semantic flagged in the mask.
float4 swapFloats(uint swappedMask, float4 value, uint semanticIndex)
{
    return (swappedMask & (1u << semanticIndex)) != 0 ? value.yxwz : value;
}

// Recover X360 integer-cast-to-float TEXCOORDs from R16G16(B16A16)_UINT bindings (sign-extend low 16 bits).
float4 sintTexcoord(uint mask, float4 value, uint semanticIndex)
{
    if ((mask & (1u << semanticIndex)) != 0)
    {
        int4 si = (int4(asuint(value)) << 16) >> 16;
        return float4(si);
    }
    return value;
}
#else
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
    return (swappedTexcoords & (1ull << semanticIndex)) != 0 ? value.yxwz : value;
}
#endif

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
