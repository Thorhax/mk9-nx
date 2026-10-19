#include "shader_recompiler.h"
#include "shader_common.h"
#include <stdexcept>
#include <cstdlib>
#include <cstring>

// Opt-in diagnostic arithmetic contract, not complete Shader Model 3 numeric
// emulation: existing zero/denormal/NaN rules are deliberately unchanged.
// VS-only; OFF must preserve the baseline generated source byte for byte.
static bool precisePositionEnabled(bool isPixelShader)
{
    const char* value = std::getenv("MASSEFFECT_SHADER_PRECISE_POSITION");
    return !isPixelShader && value && std::strcmp(value, "1") == 0;
}

static constexpr char SWIZZLES[] = 
{ 
    'x',
    'y', 
    'z', 
    'w', 
    '0', 
    '1',
    '_',
    '_'
};

static constexpr const char* USAGE_TYPES[] =
{
    "float4", // POSITION
    "float4", // BLENDWEIGHT
    "uint4", // BLENDINDICES
    "float4", // NORMAL: the renderer supplies them already converted to float
    "float4", // PSIZE
    "float4", // TEXCOORD
    "float4", // TANGENT
    "float4", // BINORMAL
    "float4", // TESSFACTOR
    "float4", // POSITIONT
    "float4", // COLOR
    "float4", // FOG
    "float4", // DEPTH
    "float4", // SAMPLE
};

static constexpr const char* USAGE_VARIABLES[] =
{
    "Position",
    "BlendWeight",
    "BlendIndices",
    "Normal",
    "PointSize",
    "TexCoord",
    "Tangent",
    "Binormal",
    "TessFactor",
    "PositionT",
    "Color",
    "Fog",
    "Depth",
    "Sample"
};

static constexpr const char* USAGE_SEMANTICS[] =
{
    "POSITION",
    "BLENDWEIGHT",
    "BLENDINDICES",
    "NORMAL",
    "PSIZE",
    "TEXCOORD",
    "TANGENT",
    "BINORMAL",
    "TESSFACTOR",
    "POSITIONT",
    "COLOR",
    "FOG",
    "DEPTH",
    "SAMPLE"
};

struct DeclUsageLocation
{
    DeclUsage usage;
    uint32_t usageIndex;
    uint32_t location;
};

// NOTE: These are specialized Vulkan locations for Unleashed Recompiled. Change as necessary. Likely not going to work with other games.
static constexpr DeclUsageLocation USAGE_LOCATIONS[] =
{
    { DeclUsage::Position, 0, 0 },
    { DeclUsage::Normal, 0, 1 },
    { DeclUsage::Tangent, 0, 2 },
    { DeclUsage::Binormal, 0, 3 },
    { DeclUsage::TexCoord, 0, 4 },
    { DeclUsage::TexCoord, 1, 5 },
    { DeclUsage::TexCoord, 2, 6 },
    { DeclUsage::TexCoord, 3, 7 },
    { DeclUsage::Color, 0, 8 },
    { DeclUsage::BlendIndices, 0, 9 },
    { DeclUsage::BlendWeight, 0, 10 },
    { DeclUsage::Color, 1, 11 },
    { DeclUsage::TexCoord, 4, 12 },
    { DeclUsage::TexCoord, 5, 13 },
    { DeclUsage::TexCoord, 6, 14 },
    { DeclUsage::TexCoord, 7, 15 },
    { DeclUsage::Position, 1, 15 },
};

static constexpr std::pair<DeclUsage, size_t> INTERPOLATORS[] =
{
    { DeclUsage::TexCoord, 0 },
    { DeclUsage::TexCoord, 1 },
    { DeclUsage::TexCoord, 2 },
    { DeclUsage::TexCoord, 3 },
    { DeclUsage::TexCoord, 4 },
    { DeclUsage::TexCoord, 5 },
    { DeclUsage::TexCoord, 6 },
    { DeclUsage::TexCoord, 7 },
    { DeclUsage::TexCoord, 8 },
    { DeclUsage::TexCoord, 9 },
    { DeclUsage::TexCoord, 10 },
    { DeclUsage::TexCoord, 11 },
    { DeclUsage::TexCoord, 12 },
    { DeclUsage::TexCoord, 13 },
    { DeclUsage::TexCoord, 14 },
    { DeclUsage::TexCoord, 15 },
    { DeclUsage::Color, 0 },
    { DeclUsage::Color, 1 }
    // MASSEFFECT: the game shaders declare and consume COLOR2..5.
    , { DeclUsage::Color, 2 }
    , { DeclUsage::Color, 3 }
    , { DeclUsage::Color, 4 }
    , { DeclUsage::Color, 5 }
};

static constexpr std::string_view TEXTURE_DIMENSIONS[] = 
{
    "2D",
    "3D", 
    "Cube" 
};

static FetchDestinationSwizzle getDestSwizzle(uint32_t dstSwizzle, uint32_t index)
{
    return FetchDestinationSwizzle((dstSwizzle >> (index * 3)) & 0x7);
}

void ShaderRecompiler::printDstSwizzle(uint32_t dstSwizzle, bool operand)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            out += SWIZZLES[operand ? uint32_t(swizzle) : i];
    }
}

void ShaderRecompiler::printDstSwizzle01(uint32_t dstRegister, uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle == FetchDestinationSwizzle::Zero)
        {
            indent();
            println("r{}.{} = 0.0;", dstRegister, SWIZZLES[i]);
        }
        else if (swizzle == FetchDestinationSwizzle::One)
        {
            indent();
            println("r{}.{} = 1.0;", dstRegister, SWIZZLES[i]);
        }
    }
}

static uint32_t banjoSlotLocation(uint32_t slot);  // banjo-nx, defined with classifyVertexFetches

// sa2-nx: whether any destination component of a vertex fetch takes fetched data. Direct3D patches the swizzle of a
// fetch whose element the vertex declaration lacks to constants only (0, 0, 0, 1); such a fetch reads nothing, and
// printing it anyway gives "r1. = ....;", which does not compile.
static bool dstSwizzleReadsFetch(uint32_t dstSwizzle)
{
    for (size_t i = 0; i < 4; i++)
    {
        const auto swizzle = getDestSwizzle(dstSwizzle, i);
        if (swizzle >= FetchDestinationSwizzle::X && swizzle <= FetchDestinationSwizzle::W)
            return true;
    }
    return false;
}

void ShaderRecompiler::recompile(const VertexFetchInstruction& instr, uint32_t address)
{
    if (instr.isPredicated)
        openPredicate(instr.predicateCondition);
    else
        closePredicate();
    const size_t markPredicate = out.size();

    if (auto indexed = indexedFetches.find(address); indexed != indexedFetches.end())
    {
        // banjo-nx: a fetch indexed by a computed register: read the fetch constant's buffer (see
        // classifyVertexFetches). A full fetch computes the element index; its mini fetches reuse it.
        if (!instr.isMiniFetch)
        {
            indent();
            print("banjoVfetchIndex = int(floor(r{}.{}{})) * {};\n", instr.srcRegister,
                SWIZZLES[instr.srcSwizzle & 0x3], instr.isIndexRounded ? " + 0.5" : "", uint32_t(instr.stride));
        }
        if (dstSwizzleReadsFetch(instr.dstSwizzle))
        {
            indent();
            print("r{}.", instr.dstRegister);
            printDstSwizzle(instr.dstSwizzle, false);
            print(" = banjoVfetch({}, banjoVfetchIndex + {}, {}, {}, {}, {}).", indexed->second, int32_t(instr.offset),
                uint32_t(instr.format), uint32_t(instr.formatCompAll), uint32_t(instr.numFormatAll), int32_t(instr.expAdjust));
            printDstSwizzle(instr.dstSwizzle, true);
            out += ";\n";
        }
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closeIfWritesPredicate(markPredicate);
        return;
    }

    if (!dstSwizzleReadsFetch(instr.dstSwizzle))
    {
        // sa2-nx: constants only (see dstSwizzleReadsFetch).
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closeIfWritesPredicate(markPredicate);
        return;
    }

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";

    auto findResult = vertexElements.find(address);
    if (findResult == vertexElements.end() && linkByRegister)
    {
        // MASSEFFECT: a fetch Direct3D's own shader does not declare. Nothing binds a stream for it; the
        // shaders seen (Direct3D startup objects) only export it to an interpolator no pixel shader reads.
        out += "0.0;\n";
        printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);
        closeIfWritesPredicate(markPredicate);
        return;
    }
    if (findResult == vertexElements.end())
    {
        std::string declared;
        for (auto& [addr, element] : vertexElements)
            declared += fmt::format(" {}:{}{}", addr, uint32_t(element.usage), uint32_t(element.usageIndex));
        throw std::runtime_error(fmt::format("vertex FETCH without a declared element (address {}, declared{})", address, declared));
    }

    // MASSEFFECT: D3D patches the fetch swizzle according to the vertex declaration; the
    // native renderer passes the mapping through g_InputRemap.
    int32_t remapLocation = -1;
    if (linkByRegister && findResult->second.usage == DeclUsage::TexCoord)
        remapLocation = int32_t(banjoSlotLocation(uint32_t(findResult->second.usageIndex)));
    else
    for (auto& usageLocation : USAGE_LOCATIONS)
    {
        if (usageLocation.usage == findResult->second.usage && usageLocation.usageIndex == findResult->second.usageIndex)
        {
            remapLocation = int32_t(usageLocation.location);
            break;
        }
    }
    if (remapLocation >= 0)
        out += "remapInput(float4(";

    switch (findResult->second.usage)
    {
    // MASSEFFECT declares normals, tangents and binormals as float4 (USAGE_TYPES): the
    // native renderer supplies them already normalized through the Vulkan format.

    case DeclUsage::TexCoord:
        print("tfetchTexcoord(g_SwappedTexcoords, ");
        break;
    }

    print("i{}{}", USAGE_VARIABLES[uint32_t(findResult->second.usage)], uint32_t(findResult->second.usageIndex));

    switch (findResult->second.usage)
    {

    case DeclUsage::TexCoord:
        print(", {})", uint32_t(findResult->second.usageIndex));
        break;
    }

    if (remapLocation >= 0)
        print("), g_InputRemap({}))", remapLocation);

    out += '.';
    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    closeIfWritesPredicate(markPredicate);
}

void ShaderRecompiler::recompile(const TextureFetchInstruction& instr, bool bicubic)
{
    if (instr.opcode != FetchOpcode::TextureFetch && instr.opcode != FetchOpcode::GetTextureWeights)
        return;

    if (instr.isPredicated)
        openPredicate(instr.predCondition);
    else
        closePredicate();
    const size_t markPredicate = out.size();

    auto printSrcRegister = [&](size_t componentCount)
        {
            print("r{}.", instr.srcRegister);

            for (size_t i = 0; i < componentCount; i++)
                out += SWIZZLES[((instr.srcSwizzle >> (i * 2))) & 0x3];
        };

    std::string constName;
    const char* constNamePtr = nullptr;
#ifdef UNLEASHED_RECOMP
    bool subtractFromOne = false;
#endif

    auto findResult = samplers.find(instr.constIndex);
    if (findResult != samplers.end())
    {
        constNamePtr = findResult->second;

    #ifdef UNLEASHED_RECOMP
        subtractFromOne = hasMtxPrevInvViewProjection && strcmp(constNamePtr, "sampZBuffer") == 0;
    #endif
    }
    else
    {
        constName = fmt::format("s{}", instr.constIndex);
        constNamePtr = constName.c_str();
    }

#ifdef UNLEASHED_RECOMP
    if (instr.constIndex == 0 && instr.dimension == TextureDimension::Texture2D)
    {
        indent();
        print("pixelCoord = getPixelCoord({}_Texture2DDescriptorIndex, ", constNamePtr);
        printSrcRegister(2);
        out += ");\n";
    }
#endif

    indent();
    print("r{}.", instr.dstRegister);
    printDstSwizzle(instr.dstSwizzle, false);

    out += " = ";
    switch (instr.opcode)
    {
    case FetchOpcode::TextureFetch:
    {
    #ifdef UNLEASHED_RECOMP
        if (subtractFromOne)
            out += "1.0 - ";
    #endif

        out += "tfetch";
        break;
    }
    case FetchOpcode::GetTextureWeights:
    {
        out += "getWeights";
        break;
    }
    }

    std::string_view dimension;
    uint32_t componentCount = 0;

    switch (instr.dimension)
    {
    case TextureDimension::Texture1D:
        dimension = "1D";
        componentCount = 1;
        break;
    case TextureDimension::Texture2D:
        dimension = "2D";
        componentCount = 2;
        break;
    case TextureDimension::Texture3D:
        dimension = "3D";
        componentCount = 3;
        break;
    case TextureDimension::TextureCube:
        dimension = "Cube";
        componentCount = 3;
        break;
    }

    out += dimension;

#ifdef UNLEASHED_RECOMP
    if (bicubic)
        out += "Bicubic";
#endif

    print("({0}_Texture{1}DescriptorIndex, {0}_SamplerDescriptorIndex, ", constNamePtr, dimension);
    printSrcRegister(componentCount);

    switch (instr.dimension)
    {
    case TextureDimension::Texture2D:
        // The last argument is 1/size of the slot, so the texture does not have to be queried.
        print(", float2({}, {}), {}_InvSize", instr.offsetX * 0.5f, instr.offsetY * 0.5f, constNamePtr);
        break;
    case TextureDimension::TextureCube:
        out += ", cubeMapData";
        break;
    }

    out += ").";

    printDstSwizzle(instr.dstSwizzle, true);

    out += ";\n";

    printDstSwizzle01(instr.dstRegister, instr.dstSwizzle);

    closeIfWritesPredicate(markPredicate);
}

void ShaderRecompiler::recompile(const AluInstruction& instr)
{
    const bool precisePosition = precisePositionEnabled(isPixelShader);
    if (instr.isPredicated)
        openPredicate(instr.predicateCondition);
    else
        closePredicate();
    const size_t markPredicate = out.size();

    enum
    {
        VECTOR_0,
        VECTOR_1,
        VECTOR_2,
        SCALAR_0,
        SCALAR_1,
        SCALAR_CONSTANT_0,
        SCALAR_CONSTANT_1
    };

    std::string scalarOperandOverrides[7];
    bool forceFullMask = false;  // banjo-nx: MaxA reads src0.w whatever the write mask
    auto op = [&](size_t operand)
        {
            if (!scalarOperandOverrides[operand].empty())
                return scalarOperandOverrides[operand];
            size_t reg = 0;
            size_t swizzle = 0;
            bool select = true;
            bool negate = false;
            bool abs = false;

            switch (operand)
            {
            case SCALAR_CONSTANT_0:
                reg = instr.src3Register;
                swizzle = instr.src3Swizzle;
                select = false;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            case SCALAR_CONSTANT_1:
                reg = (uint32_t(instr.scalarOpcode) & 1) | (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C);
                swizzle = instr.src3Swizzle;
                select = true;
                negate = instr.src3Negate;
                abs = instr.absConstants;
                break;

            default:
                switch (operand)
                {
                case VECTOR_0:
                    reg = instr.src1Register;
                    swizzle = instr.src1Swizzle;
                    select = instr.src1Select;
                    negate = instr.src1Negate;
                    break;
                case VECTOR_1:
                    reg = instr.src2Register;
                    swizzle = instr.src2Swizzle;
                    select = instr.src2Select;
                    negate = instr.src2Negate;
                    break;
                case VECTOR_2:
                case SCALAR_0:
                case SCALAR_1:
                    reg = instr.src3Register;
                    swizzle = instr.src3Swizzle;
                    select = instr.src3Select;
                    negate = instr.src3Negate;
                    break;
                }

                if (select)
                {
                    abs = (reg & 0x80) != 0;
                    reg &= 0x3F;
                }
                else
                {
                    abs = instr.absConstants;
                }

                break;
            }

            std::string regFormatted;

            if (select)
            {
                regFormatted = fmt::format("r{}", reg);
            }
            else
            {
                // The bits correspond to the first constant and the following ones,
                // not to the whole instruction. Temporary operands do not count.
                const bool relative = operand == VECTOR_0 ? instr.const0Relative :
                    operand == VECTOR_1 ? (instr.src1Select ? instr.const0Relative : instr.const1Relative) :
                    (instr.src1Select && instr.src2Select ? instr.const0Relative : instr.const1Relative);
                auto findResult = float4Constants.find(reg);
                if (findResult != float4Constants.end())
                {
                    const char* constantName = reinterpret_cast<const char*>(constantTableData + findResult->second->name);
                    if (findResult->second->registerCount > 1)
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (hasMtxProjection && strcmp(constantName, "g_MtxProjection") == 0)
                        {
                            regFormatted = fmt::format("(iterationIndex == 0 ? mtxProjectionReverseZ[{0}] : mtxProjection[{0}])",
                                reg - findResult->second->registerIndex);
                        }
                        else
                    #endif
                        {
                            regFormatted = fmt::format("{}({}{})", constantName,
                                reg - findResult->second->registerIndex, relative ? (instr.constAddressRegisterRelative ? " + a0" : " + aL") : "");
                        }
                    }
                    else
                    {
                        if (relative)
                            throw std::runtime_error("relative addressing of a scalar constant is pending");
                        regFormatted = constantName;
                    }
                }
                else
                {
                    // MASSEFFECT: shaders without a constant table (Direct3D's own) read registers
                    // that no name covers: straight from the constant buffer, like the named ones.
                    // banjo-nx: relative addressing (skinning palettes, c[a0 + N]) indexes the same buffer, clamped
                    // to its 256 registers (literal definitions are not reachable that way).
                    if (relative)
                    {
                        const char* offsetRegister = instr.constAddressRegisterRelative ? "a0" : "aL";
                        regFormatted = fmt::format("(MASSEFFECT_UBO ? g_Ubo{0}.v[min(uint({1} + {2}), 255u)] : vk::RawBufferLoad<float4>(g_PushConstants.{0}ShaderConstants + min(uint({1} + {2}), 255u) * 16, 0x10))",
                            isPixelShader ? "Pixel" : "Vertex", reg, offsetRegister);
                    }
                    else if (literalFloat4.count(reg))
                        regFormatted = fmt::format("c{}", reg);
                    else
                        regFormatted = fmt::format("(MASSEFFECT_UBO ? g_Ubo{0}.v[{1}] : vk::RawBufferLoad<float4>(g_PushConstants.{0}ShaderConstants + {2}, 0x10))",
                            isPixelShader ? "Pixel" : "Vertex", reg, reg * 16);
                }
            }

            std::string result;

            if (negate)
                result += '-';

            if (abs)
                result += "abs(";

            result += regFormatted;
            result += '.';

            switch (operand)
            {
            case VECTOR_0:
            case VECTOR_1:
            case VECTOR_2:
            {
                uint32_t mask;

                switch (instr.vectorOpcode)
                {
                case AluVectorOpcode::Dp2Add:
                    mask = (operand == VECTOR_2) ? 0b1 : 0b11;
                    break;

                case AluVectorOpcode::Dp3:
                    mask = 0b111;
                    break;

                case AluVectorOpcode::Dp4:
                case AluVectorOpcode::Max4:
                    mask = 0b1111;
                    break;

                default:
                    mask = forceFullMask ? 0b1111 : instr.vectorWriteMask != 0 ? instr.vectorWriteMask : 0b1;
                    break;
                }

                for (size_t i = 0; i < 4; i++)
                {
                    if ((mask >> i) & 0x1)
                        result += SWIZZLES[((swizzle >> (i * 2)) + i) & 0x3];
                }

                break;
            }

            case SCALAR_0:
            case SCALAR_CONSTANT_0:
                result += SWIZZLES[((swizzle >> 6) + 3) & 0x3];
                break;

            case SCALAR_1:
            case SCALAR_CONSTANT_1:
                result += SWIZZLES[swizzle & 0x3];
                break;
            }

            if (abs)
                result += ")";

            return result;
        };

    switch (instr.vectorOpcode)
    {
    case AluVectorOpcode::KillEq:
        indent();
        println("clip(any({} == {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGt:
        indent();
        println("clip(any({} > {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillGe:
        indent();
        println("clip(any({} >= {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    
    case AluVectorOpcode::KillNe:
        indent();
        println("clip(any({} != {}) ? -1 : 1);", op(VECTOR_0), op(VECTOR_1));
        break;
    }

    bool closeIfBracket = false;

    std::string_view exportRegister;
    if (instr.exportData)
    {
        if (isPixelShader)
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::PSColor0:
                exportRegister = "oC0";
                break;        
            case ExportRegister::PSColor1:
                exportRegister = "oC1";
                break;        
            case ExportRegister::PSColor2:
                exportRegister = "oC2";
                break;            
            case ExportRegister::PSColor3:
                exportRegister = "oC3";
                break;           
            case ExportRegister::PSDepth:
                exportRegister = "oDepth4";  // banjo-nx: the X of the export is the depth (see shouldReturn)
                break;
            }
        }
        else
        {
            switch (ExportRegister(instr.vectorDest))
            {
            case ExportRegister::VSPosition:
                exportRegister = "oPos";

            #ifdef UNLEASHED_RECOMP
                if (hasMtxProjection)
                {
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_REVERSE_Z) == 0 || iterationIndex == 0)\n";
                    indent();
                    out += "{\n";
                    ++indentation;

                    closeIfBracket = true;
                }
            #endif

                break;

            default:
            {
                auto findResult = interpolators.find(instr.vectorDest);
                // banjo-nx: export 63 (point size / edge flag / kill vertex) and the memory export registers
                // 32..37 (eA, eM0..4; memexport is not emulated) have no interpolator: written to a dead local.
                if (findResult == interpolators.end() && (instr.vectorDest == 63 ||
                    (instr.vectorDest >= 32 && instr.vectorDest <= 37)))
                {
                    exportRegister = "oUnused";
                    break;
                }
                if (findResult == interpolators.end())
                    {
                    std::string declared;
                    for (auto& [register_value, name] : interpolators)
                        declared += fmt::format(" {}:{}", register_value, name);
                    throw std::runtime_error(fmt::format("vertex export without a declared interpolator (register {}, declared{})", uint32_t(instr.vectorDest), declared));
                }
                exportRegister = findResult->second;
                break;
            }
            }
        }
    }

    if (instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush)
    {
        indent();
        print("p0 = {} == 0.0 && {} ", op(VECTOR_0), op(VECTOR_1));

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::SetpEqPush:
            out += "==";
            break;
        case AluVectorOpcode::SetpNePush:
            out += "!=";
            break;
        case AluVectorOpcode::SetpGtPush:
            out += ">";
            break;
        case AluVectorOpcode::SetpGePush:
            out += ">=";
            break;
        }

        out += " 0.0;\n";
    }
    bool setsA0 = false;
    if (!(instr.vectorOpcode >= AluVectorOpcode::SetpEqPush && instr.vectorOpcode <= AluVectorOpcode::SetpGePush) &&
        instr.vectorOpcode >= AluVectorOpcode::MaxA)
    {
        // banjo-nx: a0 comes from src0.w (all four lanes, not the write-mask lanes). Like the emulation, the vector
        // operands of this instruction still read the old a0; the new one is visible from the scalar operation on.
        forceFullMask = true;
        const std::string source = op(VECTOR_0);
        forceFullMask = false;
        indent();
        println("banjoNextA0 = (int)clamp(floor(({}).w + 0.5), -256.0, 255.0);", source);
        setsA0 = true;
    }

    uint32_t vectorWriteMask = instr.vectorWriteMask;
    if (instr.exportData)
        vectorWriteMask &= ~instr.scalarWriteMask;

    bool scalarSnapshotScope = false;
    // Xenos computes scalar operands before committing vector TEMP results, but
    // after vector a0/p0 effects. Preserve only genuinely overlapping TEMP lanes;
    // relative constants must still observe the new vector a0, not an old snapshot.
    if (!instr.exportData && vectorWriteMask != 0)
    {
        bool used[7]{};
        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds: case AluScalarOpcode::Muls:
        case AluScalarOpcode::Maxs: case AluScalarOpcode::Mins:
        case AluScalarOpcode::MaxAs: case AluScalarOpcode::MaxAsf:
        case AluScalarOpcode::Subs:
            used[SCALAR_0] = used[SCALAR_1] = true;
            break;
        case AluScalarOpcode::Mulsc0: case AluScalarOpcode::Mulsc1:
        case AluScalarOpcode::Addsc0: case AluScalarOpcode::Addsc1:
        case AluScalarOpcode::Subsc0: case AluScalarOpcode::Subsc1:
            used[SCALAR_CONSTANT_1] = true;
            break;
        case AluScalarOpcode::SetpClr: case AluScalarOpcode::RetainPrev:
            break;
        default:
            used[SCALAR_0] = uint32_t(instr.scalarOpcode) <= 40 ||
                instr.scalarOpcode == AluScalarOpcode::Sin ||
                instr.scalarOpcode == AluScalarOpcode::Cos;
            break;
        }
        std::string capturedExpressions[7];
        for (size_t operand = SCALAR_0; operand <= SCALAR_CONSTANT_1; ++operand)
        {
            if (!used[operand]) continue;
            const bool special = operand == SCALAR_CONSTANT_1;
            if (!special && !instr.src3Select) continue;
            const uint32_t reg = special ? ((uint32_t(instr.scalarOpcode) & 1) |
                (instr.src3Select << 1) | (instr.src3Swizzle & 0x3C)) :
                (instr.src3Register & 0x3F);
            const uint32_t lane = operand == SCALAR_0 ?
                (((instr.src3Swizzle >> 6) + 3) & 3) : (instr.src3Swizzle & 3);
            if (reg != instr.vectorDest || !(vectorWriteMask & (1u << lane))) continue;
            const std::string expression = op(operand);
            for (size_t prior = SCALAR_0; prior < operand; ++prior)
                if (capturedExpressions[prior] == expression)
                    scalarOperandOverrides[operand] = scalarOperandOverrides[prior];
            if (!scalarOperandOverrides[operand].empty()) continue;
            capturedExpressions[operand] = expression;
            scalarOperandOverrides[operand] = fmt::format("me_alu_scalar_{}_{}", markPredicate, operand);
            // CF is emitted as switch cases. Initialized locals must not extend
            // to the next label, which DXC correctly rejects as a bypassed init.
            if (!scalarSnapshotScope)
            {
                indent();
                out += "{\n";
                ++indentation;
                scalarSnapshotScope = true;
            }
            indent();
            println("{}float {} = {};", precisePosition ? "precise " : "",
                    scalarOperandOverrides[operand], expression);
        }
    }

    if (vectorWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.vectorDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((vectorWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ";

        if (instr.vectorSaturate)
            out += "saturate(";

        switch (instr.vectorOpcode)
        {
        case AluVectorOpcode::Add:
            print("{} + {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Mul:
            print("{} * {}", op(VECTOR_0), op(VECTOR_1));
            break;

        /*
         * The Xenos MOV is a max with itself.
         *
         * The Xenos has no copy instruction: the assembler writes it as MAX dst, src, src
         * (and MAXs for the scalar channel). The translator emitted it as is, so the library
         * had 452 `max(a, a)` in the pixel shaders and 482 in the vertex shaders: almost a
         * thousand operations that are a copy. max(a, a) == a exactly (also with NaN and with
         * both zeros, which is the only thing that sets IEEE min/max apart), so the operand is
         * emitted directly.
         *
         * The `a0 = ...` of MaxA/MaxAs/MaxAsf is not touched: it is handled separately and
         * stays the same.
         */
        case AluVectorOpcode::Max:
        case AluVectorOpcode::MaxA:
        case AluVectorOpcode::Min:
        {
            std::string a = op(VECTOR_0);
            std::string b = op(VECTOR_1);
            if (a == b)
                out += a;
            else
                print("{}({}, {})", instr.vectorOpcode == AluVectorOpcode::Min ? "min" : "max", a, b);
            break;
        }

        case AluVectorOpcode::Seq:
            print("{} == {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sgt:
            print("{} > {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sge:
            print("{} >= {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Sne:
            print("{} != {}", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Frc:
            print("frac({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Trunc:
            print("trunc({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Floor:
            print("floor({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::Mad:
            print("{} * {} + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndEq:
            print("select({} == 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGe:
            print("select({} >= 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::CndGt:
            print("select({} > 0.0, {}, {})", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Dp4:
        case AluVectorOpcode::Dp3:
            if (precisePosition)
            {
                const std::string a = op(VECTOR_0), b = op(VECTOR_1);
                const uint32_t count = instr.vectorOpcode == AluVectorOpcode::Dp4 ? 4 : 3;
                std::string sum = fmt::format("(({}).x * ({}).x)", a, b);
                for (uint32_t component = 1; component < count; ++component)
                    sum = fmt::format("({} + (({}).{} * ({}).{}))", sum, a,
                                      SWIZZLES[component], b, SWIZZLES[component]);
                out += sum;
            }
            else print("dot({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dp2Add:
            if (precisePosition)
            {
                const std::string a = op(VECTOR_0), b = op(VECTOR_1);
                print("(((({}).x * ({}).x) + (({}).y * ({}).y)) + ({}))", a, b, a, b, op(VECTOR_2));
            }
            else print("dot({}, {}) + {}", op(VECTOR_0), op(VECTOR_1), op(VECTOR_2));
            break;

        case AluVectorOpcode::Cube:
            print("cube(r{}, cubeMapData)", instr.src1Register);
            break;

        case AluVectorOpcode::Max4:
            print("max4({})", op(VECTOR_0));
            break;

        case AluVectorOpcode::SetpEqPush:
        case AluVectorOpcode::SetpNePush:
        case AluVectorOpcode::SetpGtPush:
        case AluVectorOpcode::SetpGePush:
            print("p0 ? 0.0 : {} + 1.0", op(VECTOR_0));
            break;

        case AluVectorOpcode::KillEq:
            print("any({} == {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGt:
            print("any({} > {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillGe:
            print("any({} >= {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::KillNe:
            print("any({} != {})", op(VECTOR_0), op(VECTOR_1));
            break;

        case AluVectorOpcode::Dst:
            print("dst({}, {})", op(VECTOR_0), op(VECTOR_1));
            break;
        }

        if (instr.vectorSaturate)
            out += ')';

        out += ";\n";
    }

    if (setsA0)
    {
        indent();
        out += "a0 = banjoNextA0;\n";
    }

    if (instr.scalarOpcode != AluScalarOpcode::RetainPrev)
    {
        if (instr.scalarOpcode >= AluScalarOpcode::SetpEq && instr.scalarOpcode <= AluScalarOpcode::SetpRstr)
        {
            indent();
            out += "p0 = ";

            switch (instr.scalarOpcode)
            {
            case AluScalarOpcode::SetpEq:
                print("{} == 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpNe:
                print("{} != 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGt:
                print("{} > 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpGe:
                print("{} >= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpInv:
                print("{} == 1.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpPop:
                print("{} - 1.0 <= 0.0", op(SCALAR_0));
                break;

            case AluScalarOpcode::SetpClr:
                out += "false";
                break;

            case AluScalarOpcode::SetpRstr:
                print("{} == 0.0", op(SCALAR_0));
                break;
            }

            out += ";\n";
        }

        indent();
        out += "ps = ";
        if (instr.scalarSaturate)
            out += "saturate(";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::Adds:
            print("{} + {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::AddsPrev:
            print("{} + ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::Muls:
            print("{} * {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::MulsPrev:
        case AluScalarOpcode::MulsPrev2:
            print("{} * ps", op(SCALAR_0));
            break;

        // Same as above, for the scalar channel: MAXs dst, src, src is the Xenos MOV.
        case AluScalarOpcode::Maxs:
        case AluScalarOpcode::MaxAs:
        case AluScalarOpcode::MaxAsf:
        case AluScalarOpcode::Mins:
        {
            std::string a = op(SCALAR_0);
            std::string b = op(SCALAR_1);
            if (a == b)
                out += a;
            else
                print("{}({}, {})", instr.scalarOpcode == AluScalarOpcode::Mins ? "min" : "max", a, b);
            break;
        }

        case AluScalarOpcode::Seqs:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sgts:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sges:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Snes:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Frcs:
            print("frac({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Truncs:
            print("trunc({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Floors:
            print("floor({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Exp:
            print("exp2({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Logc:
        case AluScalarOpcode::Log:
            print("clamp(log2({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rcpc:
        case AluScalarOpcode::Rcpf:
        case AluScalarOpcode::Rcp:
            print("clamp(rcp({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Rsqc:
        case AluScalarOpcode::Rsqf:
        case AluScalarOpcode::Rsq:
            print("clamp(rsqrt({}), FLT_MIN, FLT_MAX)", op(SCALAR_0));
            break;

        case AluScalarOpcode::Subs:
            print("{} - {}", op(SCALAR_0), op(SCALAR_1));
            break;

        case AluScalarOpcode::SubsPrev:
            print("{} - ps", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpEq:
        case AluScalarOpcode::SetpNe:
        case AluScalarOpcode::SetpGt:
        case AluScalarOpcode::SetpGe:
            out += "p0 ? 0.0 : 1.0";
            break;

        case AluScalarOpcode::SetpInv:
            // Xenos: with src == 1 the counter goes back to 0 (Xenia's ucode.h, kSetpInv). Without that
            // case the counter stayed at 1 and later p0 blocks did not run.
            print("p0 ? 0.0 : ({0} == 0.0 ? 1.0 : {0})", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpPop:
            print("p0 ? 0.0 : ({} - 1.0)", op(SCALAR_0));
            break;

        case AluScalarOpcode::SetpClr:
            out += "FLT_MAX";
            break;

        case AluScalarOpcode::SetpRstr:
            print("p0 ? 0.0 : {}", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsEq:
            print("{} == 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGt:
            print("{} > 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsGe:
            print("{} >= 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsNe:
            print("{} != 0.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::KillsOne:
            print("{} == 1.0", op(SCALAR_0));
            break;

        case AluScalarOpcode::Sqrt:
            print("sqrt({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Mulsc0:
        case AluScalarOpcode::Mulsc1:
            print("{} * {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Addsc0:
        case AluScalarOpcode::Addsc1:
            print("{} + {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Subsc0:
        case AluScalarOpcode::Subsc1:
            print("{} - {}", op(SCALAR_CONSTANT_0), op(SCALAR_CONSTANT_1));
            break;

        case AluScalarOpcode::Sin:
            print("sin({})", op(SCALAR_0));
            break;

        case AluScalarOpcode::Cos:
            print("cos({})", op(SCALAR_0));
            break;
        }

        if (instr.scalarSaturate)
            out += ')';

        out += ";\n";

        switch (instr.scalarOpcode)
        {
        case AluScalarOpcode::MaxAs:
            indent();
            println("a0 = (int)clamp(floor({} + 0.5), -256.0, 255.0);", op(SCALAR_0));
            break;     
        case AluScalarOpcode::MaxAsf:
            indent();
            println("a0 = (int)clamp(floor({}), -256.0, 255.0);", op(SCALAR_0));
            break;
        }
    }

    uint32_t scalarWriteMask = instr.scalarWriteMask;
    if (instr.exportData)
        scalarWriteMask &= ~instr.vectorWriteMask;

    if (scalarWriteMask != 0)
    {
        indent();
        if (!exportRegister.empty())
        {
            out += exportRegister;
            out += '.';
        }
        else
        {
            print("r{}.", instr.scalarDest);
        }

        for (size_t i = 0; i < 4; i++)
        {
            if ((scalarWriteMask >> i) & 0x1)
                out += SWIZZLES[i];
        }

        out += " = ps;\n";
    }

    if (instr.exportData)
    {
        uint32_t zeroMask = instr.scalarDestRelative ? (0b1111 & ~(instr.vectorWriteMask | instr.scalarWriteMask)) : 0;
        uint32_t oneMask = instr.vectorWriteMask & instr.scalarWriteMask;

        for (size_t i = 0; i < 4; i++)
        {
            uint32_t mask = 1 << i;
            if (zeroMask & mask)
            {
                indent();
                println("{}.{} = 0.0;", exportRegister, SWIZZLES[i]);
            }
            else if (oneMask & mask)
            {
                indent();
                println("{}.{} = 1.0;", exportRegister, SWIZZLES[i]);
            }
        }
    }

    if (instr.scalarOpcode >= AluScalarOpcode::KillsEq && instr.scalarOpcode <= AluScalarOpcode::KillsOne)
    {
        indent();
        out += "clip(ps != 0.0 ? -1 : 1);\n";
    }

    if (scalarSnapshotScope)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    if (closeIfBracket)
    {
        --indentation;
        indent();
        out += "}\n";
    }

    closeIfWritesPredicate(markPredicate);
}

// MASSEFFECT: the full vertex fetches of a microcode (instruction index, destination register), walking the
// exec clauses of the control flow. Same walk as app/src/me/native/sa2_native_shaders.cpp
// FetchesOfVertices: Direct3D's own vertex shaders are linked by fetch destination register.
static std::vector<std::pair<uint32_t, uint32_t>> vertexFetchesByRegister(const std::vector<uint32_t>& w)
{
    std::vector<std::pair<uint32_t, uint32_t>> fetches;
    uint32_t end = uint32_t(w.size() / 3);
    for (uint32_t pair = 0; pair < end && pair * 3 + 2 < w.size(); pair++)
    {
        const uint64_t cf[2] = {
            uint64_t(w[pair * 3]) | (uint64_t(w[pair * 3 + 1] & 0xFFFF) << 32),
            uint64_t(w[pair * 3 + 1] >> 16) | (uint64_t(w[pair * 3 + 2]) << 16) };
        for (uint64_t c : cf)
        {
            const uint32_t opcode = uint32_t(c >> 44) & 0xF;
            const bool exec = (opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14;
            if (!exec)
                continue;
            const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
            const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
            if (address != 0)
                end = std::min(end, address);
            for (uint32_t i = 0; i < count; i++)
            {
                const uint32_t index = address + i;
                if (!((sequence >> (i * 2)) & 0x1) || index * 3 + 2 >= w.size())
                    continue;
                const uint32_t d0 = w[index * 3], d1 = w[index * 3 + 1];
                if ((d0 & 0x1F) == 0)
                    fetches.emplace_back(index, (d0 >> 12) & 0x3F);
            }
        }
    }
    return fetches;
}

// banjo-nx: whether r0.x still holds the vertex index at each full vertex fetch. A Xenos vertex shader starts with
// the vertex index in r0.x; Nuts & Bolts' instanced shaders overwrite r0 (e.g. r0 = floor(index * c132.x), the
// instance) before fetching "by r0.x", which then fetches by that computed index, not per vertex. Walks the exec
// clauses in control-flow order; any ALU or fetch write to r0.x (predicated or not) ends the vertex index.
static std::unordered_set<uint32_t> vertexFetchesAfterR0Write(const std::vector<uint32_t>& w)
{
    std::unordered_set<uint32_t> after;
    bool written = false;
    uint32_t end = uint32_t(w.size() / 3);
    for (uint32_t pair = 0; pair < end && pair * 3 + 2 < w.size(); pair++)
    {
        const uint64_t cf[2] = {
            uint64_t(w[pair * 3]) | (uint64_t(w[pair * 3 + 1] & 0xFFFF) << 32),
            uint64_t(w[pair * 3 + 1] >> 16) | (uint64_t(w[pair * 3 + 2]) << 16) };
        for (uint64_t c : cf)
        {
            const uint32_t opcode = uint32_t(c >> 44) & 0xF;
            if (!((opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14))
                continue;
            const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
            const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
            if (address != 0)
                end = std::min(end, address);
            for (uint32_t i = 0; i < count; i++)
            {
                const uint32_t index = address + i;
                if (index * 3 + 2 >= w.size())
                    continue;
                const uint32_t w0 = w[index * 3], w1 = w[index * 3 + 1];
                if ((sequence >> (i * 2)) & 0x1)
                {
                    if ((w0 & 0x1F) == 0 && written)
                        after.insert(index);  // a vertex fetch after r0.x stopped being the vertex index
                    // Any fetch (vertex or texture) into r0 that writes its x lane (dst swizzle 7 = keep).
                    if (((w0 >> 12) & 0x3F) == 0 && !((w0 >> 18) & 0x1) && (w1 & 0x7) != 7)
                        written = true;
                }
                else if (!((w0 >> 15) & 0x1))  // ALU, not an export
                {
                    if ((w0 & 0x3F) == 0 && !((w0 >> 6) & 0x1) && ((w0 >> 16) & 0x1))
                        written = true;  // vector result into r0.x
                    if (((w0 >> 8) & 0x3F) == 0 && !((w0 >> 14) & 0x1) && ((w0 >> 20) & 0x1))
                        written = true;  // scalar result into r0.x
                }
            }
        }
    }
    return after;
}

// banjo-nx: the full vertex fetches of a register-linked vertex shader, split in two:
//   - attributes: indexed by the vertex index (r0.x as the shader starts), one vertex input each. Input slot =
//     the destination register; a register fetched twice takes the lowest slot (0..15) no attribute register uses.
//     A mini fetch belongs to the full fetch before it.
//   - indexed: indexed by a computed register (bone matrices read by blend index); read from the buffer of their
//     fetch constant in the shader (banjoVfetch). Up to 2 distinct fetch constants.
// The native renderer applies the same rule (masseffect_native_shaders.cpp, ClassifyVertexFetches).
struct BanjoVertexFetchSplit
{
    std::vector<std::pair<uint32_t, uint32_t>> attributes;  // (instruction index, input slot)
    std::unordered_map<uint32_t, uint32_t> indexed;         // instruction index -> indexed buffer number
    std::vector<uint32_t> indexedConstants;                 // per indexed buffer number: vertex fetch constant 0..95
};

static BanjoVertexFetchSplit classifyVertexFetches(const std::vector<uint32_t>& w)
{
    BanjoVertexFetchSplit split;
    std::vector<std::pair<uint32_t, uint32_t>> attributeRegisters;
    bool lastFullIsAttribute = true;
    uint32_t lastFullBuffer = 0;
    const auto afterR0Write = vertexFetchesAfterR0Write(w);
    for (auto& [index, reg] : vertexFetchesByRegister(w))
    {
        const uint32_t d0 = w[index * 3], d1 = w[index * 3 + 1];
        const bool mini = (d1 >> 30) & 0x1;
        if (!mini)
        {
            const uint32_t src = (d0 >> 5) & 0x3F, srcRelative = (d0 >> 11) & 0x1, srcSwizzle = (d0 >> 30) & 0x3;
            lastFullIsAttribute = src == 0 && !srcRelative && srcSwizzle == 0 && !afterR0Write.count(index);
            if (!lastFullIsAttribute)
            {
                const uint32_t constant = ((d0 >> 20) & 0x1F) * 3 + ((d0 >> 25) & 0x3);
                auto it = std::find(split.indexedConstants.begin(), split.indexedConstants.end(), constant);
                lastFullBuffer = uint32_t(it - split.indexedConstants.begin());
                if (it == split.indexedConstants.end())
                {
                    if (split.indexedConstants.size() >= 2)
                        throw std::runtime_error("more than 2 indexed vertex fetch constants");
                    split.indexedConstants.push_back(constant);
                }
            }
        }
        if (lastFullIsAttribute)
            attributeRegisters.emplace_back(index, reg);
        else
            split.indexed.emplace(index, lastFullBuffer);
    }
    uint32_t registersUsed = 0;
    for (auto& [index, reg] : attributeRegisters)
    {
        if (reg >= 16)
            throw std::runtime_error("vertex fetch to a register without a location (>= r16)");
        registersUsed |= 1u << reg;
    }
    uint32_t slotsTaken = 0;
    for (auto& [index, reg] : attributeRegisters)
    {
        uint32_t slot = reg;
        if (slotsTaken & (1u << slot))
        {
            slot = 0;
            while (slot < 16 && ((registersUsed | slotsTaken) & (1u << slot)))
                slot++;
            if (slot >= 16)
                throw std::runtime_error("no free vertex input slot for a repeated fetch register");
        }
        slotsTaken |= 1u << slot;
        split.attributes.emplace_back(index, slot);
    }
    return split;
}

// banjo-nx: Vulkan location of input slot N of a register-linked vertex shader (TEXCOORDN). Slots 0..7 keep
// USAGE_LOCATIONS' texcoord locations; 8..15 take the locations of the usages these shaders never declare.
static uint32_t banjoSlotLocation(uint32_t slot)
{
    return slot < 4 ? 4 + slot : slot < 8 ? 12 + (slot - 4) : slot < 12 ? slot - 8 : 8 + (slot - 12);
}

/*
 * sa2-nx GPU round 2 (sa2_gpu_index_base_held): whether a vertex shader can read the value r0 starts with (the guest
 * vertex index, SV_VertexID + g_VertexIndexBase), walking its microcode in control flow order:
 *  - any read of r0 by an ALU, texture fetch or indexed vertex fetch instruction counts, whatever the component,
 *    except the Xenos zero idiom: the scalar SETGT of -|r0| (`-abs(r0.c) > 0.0`) is 0 for every value, NaN included;
 *  - the answer is "no" once r0.x is overwritten by an instruction that surely runs (not predicated, in a plain EXEC
 *    before any jump, loop or conditional EXEC), or when the shader surely ends before any read.
 * Everything else is "it may": conditional code, relative addressing or an unknown construct keep the index.
 * SA2's lighting vertex shaders read r0 only through the zero idiom before `r0.x = dot(...)`, so their r0 starts at 0
 * and their SPIR-V never loads the vertex index; the renderer then holds g_VertexIndexBase (and with it the whole
 * shared constant block) from one draw to the next.
 */
static bool vertexIndexMayBeRead(const be<uint32_t>* code, uint32_t shaderSize,
                                 const std::unordered_map<uint32_t, uint32_t>& indexedFetches)
{
    union { ControlFlowInstruction controlFlow[2]; struct { uint32_t code0, code1, code2, code3; } w; } cfu;
    uint32_t instrSize = shaderSize;
    // Pass 1: where the instructions start (the first EXEC address), as the translator's own scan finds it.
    for (uint32_t at = 0, word = 0; at < instrSize; at += 12, word += 3)
    {
        cfu.w.code0 = code[word];
        cfu.w.code1 = code[word + 1] & 0xFFFF;
        cfu.w.code2 = (code[word + 1] >> 16) | (code[word + 2] << 16);
        cfu.w.code3 = code[word + 2] >> 16;
        for (auto& cf : cfu.controlFlow)
        {
            uint32_t address = 0;
            switch (cf.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cf.exec.address;
                break;
            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cf.condExec.address;
                break;
            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cf.condExecPred.address;
                break;
            default:
                break;
            }
            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);
        }
    }
    bool uncertain = false;  // past a jump, loop or conditional EXEC: writes no longer surely run
    for (uint32_t at = 0, word = 0; at < instrSize; at += 12, word += 3)
    {
        cfu.w.code0 = code[word];
        cfu.w.code1 = code[word + 1] & 0xFFFF;
        cfu.w.code2 = (code[word + 1] >> 16) | (code[word + 2] << 16);
        cfu.w.code3 = code[word + 2] >> 16;
        for (auto& cf : cfu.controlFlow)
        {
            uint32_t address = 0, count = 0, sequence = 0;
            bool conditional = uncertain, ends = false;
            switch (cf.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cf.exec.address;
                count = cf.exec.count;
                sequence = cf.exec.sequence;
                ends = cf.opcode == ControlFlowOpcode::ExecEnd;
                break;
            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cf.condExec.address;
                count = cf.condExec.count;
                sequence = cf.condExec.sequence;
                conditional = true;
                break;
            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cf.condExecPred.address;
                count = cf.condExecPred.count;
                sequence = cf.condExecPred.sequence;
                conditional = true;
                break;
            case ControlFlowOpcode::CondJmp:
            case ControlFlowOpcode::LoopStart:
            case ControlFlowOpcode::LoopEnd:
                uncertain = true;
                continue;
            default:
                continue;  // no instructions, no control flow (Nop, Alloc, MarkVsFetchDone...)
            }
            for (uint32_t i = 0; i < count; ++i, sequence >>= 2)
            {
                const uint32_t index = address + i;
                if (size_t(index) * 12 + 12 > shaderSize)
                    return true;
                union { VertexFetchInstruction vertexFetch; TextureFetchInstruction textureFetch; AluInstruction alu;
                        struct { uint32_t code0, code1, code2; } w; } iu;
                iu.w.code0 = code[index * 3];
                iu.w.code1 = code[index * 3 + 1];
                iu.w.code2 = code[index * 3 + 2];
                if (sequence & 0x1)
                {
                    if (iu.vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        if (indexedFetches.count(index) && iu.vertexFetch.srcRegister == 0)
                            return true;  // an indexed fetch computes its element from r0
                        if (iu.vertexFetch.dstRegister == 0 && !iu.vertexFetch.dstRegisterAam &&
                            (iu.vertexFetch.dstSwizzle & 7) != 7 && !conditional && !iu.vertexFetch.isPredicated)
                            return false;  // r0.x overwritten for sure
                    }
                    else
                    {
                        if (iu.textureFetch.srcRegister == 0)
                            return true;  // texture coordinates from r0
                        if (iu.textureFetch.dstRegister == 0 && !iu.textureFetch.dstRegisterAm &&
                            (iu.textureFetch.dstSwizzle & 7) != 7 && !conditional && !iu.textureFetch.isPredicated)
                            return false;
                    }
                    continue;
                }
                const AluInstruction& alu = iu.alu;
                const auto reads0 = [](uint32_t select, uint32_t reg) { return select && (reg & 0x3F) == 0; };
                const uint32_t vop = uint32_t(alu.vectorOpcode);
                const bool vectorOneOperand = vop == 8 || vop == 9 || vop == 10 || vop == 19;
                const bool vectorThreeOperands = vop == 11 || vop == 12 || vop == 13 || vop == 14 || vop == 17;
                if (reads0(alu.src1Select, alu.src1Register))
                    return true;
                if (!vectorOneOperand && reads0(alu.src2Select, alu.src2Register))
                    return true;
                if (vectorThreeOperands && reads0(alu.src3Select, alu.src3Register))
                    return true;
                const uint32_t sop = uint32_t(alu.scalarOpcode);
                const bool zeroIdiom = alu.scalarOpcode == AluScalarOpcode::Sgts && alu.src3Negate &&
                                       (alu.src3Register & 0x80) != 0;
                if (sop >= 42 && sop <= 47)
                {
                    // Mulsc/Addsc/Subsc: the second operand is a temporary encoded in the opcode and src3 fields.
                    const uint32_t reg = (sop & 1) | (alu.src3Select << 1) | (alu.src3Swizzle & 0x3C);
                    if (reg == 0)
                        return true;
                }
                else if (alu.scalarOpcode != AluScalarOpcode::RetainPrev && !zeroIdiom &&
                         reads0(alu.src3Select, alu.src3Register))  // RetainPrev: no scalar operation, nothing read
                {
                    return true;
                }
                // Writes, after this instruction's reads.
                if (!conditional && !alu.isPredicated && !alu.exportData)
                {
                    // Only operations that surely write their destination: no predicate set/push, kill or MOVA.
                    const bool vectorWrites = vop < 20 || vop == 28;
                    if (vectorWrites && alu.vectorDest == 0 && !alu.vectorDestRelative && (alu.vectorWriteMask & 1))
                        return false;
                    const bool scalarWrites = (sop < 23 || sop == 25 || sop == 26 || sop == 40 || (sop >= 42 && sop <= 49));
                    if (scalarWrites && alu.scalarDest == 0 && !alu.scalarDestRelative && (alu.scalarWriteMask & 1))
                        return false;
                }
            }
            if (ends && !conditional)
                return false;  // the shader ends here for sure, and nothing read r0
        }
    }
    return true;  // reached the end of the control flow without a sure end: keep it
}

void ShaderRecompiler::recompile(const uint8_t* shaderData, const std::string_view& include)
{
    const auto shaderContainer = reinterpret_cast<const ShaderContainer*>(shaderData);

    /*
     * Mass Effect (2008 XDK) has 0x102A11xx; the low byte holds the stage flag.
     * Only the two high bytes are compared.
     */
    assert((shaderContainer->flags & 0xFFFF0000) == 0x102A0000);
    // banjo-nx: vertex shaders have no implicit derivatives (DXC rejects Sample there): the texture helpers of
    // shader_common.h sample level 0 instead, which is what a vertex fetch without an explicit LOD reads.
    if (shaderContainer->flags & 0x1)
        out += "#define Sample(s, c) SampleLevel(s, c, 0)\n";
    out += include;
    out += '\n';

    isPixelShader = (shaderContainer->flags & 0x1) == 0;
    const bool precisePosition = precisePositionEnabled(isPixelShader);
    const auto codeMetadata = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);
    const auto validFloatDefinition = [&](const Float4Definition& definition) {
        const uint64_t begin = definition.physicalOffset;
        const uint64_t end = begin + ((uint64_t(definition.count) + 3) / 4) * 16;
        const uint64_t codeBegin = codeMetadata->physicalOffset;
        const uint64_t codeEnd = codeBegin + uint32_t(codeMetadata->size);
        // Mass Effect's relocated / immediate shader wrappers can retain a
        // definition table whose physical offsets now point into instructions.
        // Those bytes are NOT literals. Original D3D supplies the real values
        // in the guest float registers; use the normal constant-buffer path.
        return end <= uint32_t(shaderContainer->physicalSize) &&
            (end <= codeBegin || begin >= codeEnd);
    };

    // MASSEFFECT: Direct3D's own shaders either have no constant table or carry an empty CTAB.
    // Both forms are linked by hardware register rather than by semantic names.
    static const ConstantTableContainer kEmptyConstantTable{};
    const auto constantTableContainer = shaderContainer->constantTableOffset != NULL
        ? reinterpret_cast<const ConstantTableContainer*>(shaderData + shaderContainer->constantTableOffset)
        : &kEmptyConstantTable;
    linkByRegister = shaderContainer->constantTableOffset == NULL ||
        constantTableContainer->constantTable.constants == 0;
    constantTableData = reinterpret_cast<const uint8_t*>(&constantTableContainer->constantTable);
    literalFloat4.clear();
    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitions = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset)->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            if (validFloatDefinition(*definition))
                for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
                    literalFloat4.insert(definition->registerIndex + i - (isPixelShader ? 256 : 0));
            definitions += 2;
        }
    }

    out += "#ifdef __spirv__\n\n";

#ifdef UNLEASHED_RECOMP
    bool isMetaInstancer = false;
    bool hasIndexCount = false;
#endif

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

    #ifdef UNLEASHED_RECOMP
        if (!isPixelShader)
        {
            if (strcmp(constantName, "g_MtxProjection") == 0)
                hasMtxProjection = true;
            else if (strcmp(constantName, "g_InstanceTypes") == 0)
                isMetaInstancer = true;
            else if (strcmp(constantName, "g_IndexCount") == 0)
                hasIndexCount = true;
        }
        else
        {
            if (strcmp(constantName, "g_MtxPrevInvViewProjection") == 0)
                hasMtxPrevInvViewProjection = true;
        }
    #endif

        switch (constantInfo->registerSet)
        {
        case RegisterSet::Float4:
        {
            const char* shaderName = isPixelShader ? "Pixel" : "Vertex";

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = 256 - constantInfo->registerIndex;  // banjo-nx: 256 in both stages

                // MASSEFFECT: dynamic UBO or pointer, depending on SPEC_CONSTANT_CONSTANTS_UBO.
                println("#define {}(INDEX) select((INDEX) < {}, (MASSEFFECT_UBO ? g_Ubo{}.v[{} + min(INDEX, {})] : vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + ({} + min(INDEX, {})) * 16, 0x10)), 0.0)",
                    constantName, tailCount, shaderName, constantInfo->registerIndex.get(), tailCount - 1,
                    shaderName, constantInfo->registerIndex.get(), tailCount - 1);
            }
            else
            {
                println("#define {} (MASSEFFECT_UBO ? g_Ubo{}.v[{}] : vk::RawBufferLoad<float4>(g_PushConstants.{}ShaderConstants + {}, 0x10))",
                    constantName, shaderName, constantInfo->registerIndex.get(), shaderName, constantInfo->registerIndex * 16);
            }
            
            for (uint16_t j = 0; j < constantInfo->registerCount; j++)
                float4Constants.emplace(constantInfo->registerIndex + j, constantInfo);

            break;
        }

        case RegisterSet::Sampler:
        {
            // MASSEFFECT: a sampler array (Unreal Engine 3's LightMapTextures, s5..s7) covers
            // registerCount registers, and the microcode fetches from each of them; only the
            // first used to get definitions, so the others ended up as undeclared sN_*.
            // Every register now gets its own set, named <name> for the first and sN after it.
            for (uint16_t reg = 0; reg < std::max<uint16_t>(1, constantInfo->registerCount); reg++)
            {
            const uint32_t registerIndex = constantInfo->registerIndex + reg;
            char fallbackName[16];
            std::snprintf(fallbackName, sizeof(fallbackName), "s%u", registerIndex);
            const char* samplerName = reg == 0 ? constantName : fallbackName;
            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("#define {}_Texture{}DescriptorIndex (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
                    samplerName, TEXTURE_DIMENSIONS[j], j * 64 + registerIndex * 4, j * 64 + registerIndex * 4);
            }

            println("#define {}_SamplerDescriptorIndex (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
                samplerName, std::size(TEXTURE_DIMENSIONS) * 64 + registerIndex * 4, std::size(TEXTURE_DIMENSIONS) * 64 + registerIndex * 4);

            // 1/size of the host image of that slot, which the renderer writes at byte
            // 360 + slot * 8 of the shared constants (right after g_InputRemap).
            {
                const uint32_t invBase = 360 + registerIndex * 8;
                println("#define {}_InvSize (MASSEFFECT_UBO ? float2(MASSEFFECT_SHARED_FLOAT({}), MASSEFFECT_SHARED_FLOAT({})) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + {}))",
                    samplerName, invBase, invBase + 4, invBase);
            }

            if (reg == 0)
                samplers.emplace(registerIndex, constantName);
            }
            break;
        }

        }
    }

    // MASSEFFECT: every fetch constant no table entry names gets the sN_* set (a sampler without a
    // name, or a shader without a table); #ifndef skips the ones already defined.
    for (uint32_t reg = 0; reg < 32; reg++)
    {
        println("#ifndef s{}_SamplerDescriptorIndex", reg);
        for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            println("#define s{}_Texture{}DescriptorIndex (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
                reg, TEXTURE_DIMENSIONS[j], j * 64 + reg * 4, j * 64 + reg * 4);
        println("#define s{}_SamplerDescriptorIndex (MASSEFFECT_UBO ? MASSEFFECT_SHARED_UINT({}) : vk::RawBufferLoad<uint>(g_PushConstants.SharedConstants + {}))",
            reg, std::size(TEXTURE_DIMENSIONS) * 64 + reg * 4, std::size(TEXTURE_DIMENSIONS) * 64 + reg * 4);
        println("#define s{0}_Texture1DDescriptorIndex s{0}_Texture2DDescriptorIndex", reg);
        println("#define s{}_InvSize (MASSEFFECT_UBO ? float2(MASSEFFECT_SHARED_FLOAT({}), MASSEFFECT_SHARED_FLOAT({})) : vk::RawBufferLoad<float2>(g_PushConstants.SharedConstants + {}))",
            reg, 360 + reg * 8, 364 + reg * 8, 360 + reg * 8);
        println("#endif");
    }

    out += "\n#else\n\n";

    println("cbuffer {}ShaderConstants : register(b{}, space4)", isPixelShader ? "Pixel" : "Vertex", isPixelShader ? 1 : 0);
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Float4)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            print("\tfloat4 {}", constantName);

            if (constantInfo->registerCount > 1)
                print("[{}]", constantInfo->registerCount.get());

            println(" : packoffset(c{});", constantInfo->registerIndex.get());

            if (constantInfo->registerCount > 1)
            {
                uint32_t tailCount = 256 - constantInfo->registerIndex;  // banjo-nx: 256 in both stages
                println("#define {0}(INDEX) select((INDEX) < {1}, {0}[min(INDEX, {2})], 0.0)", constantName, tailCount, tailCount - 1);
            }
        }
    }

    out += "};\n\n";

    out += "cbuffer SharedConstants : register(b2, space4)\n";
    out += "{\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Sampler)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);

            for (size_t j = 0; j < std::size(TEXTURE_DIMENSIONS); j++)
            {
                println("\tuint {}_Texture{}DescriptorIndex : packoffset(c{}.{});",
                    constantName, TEXTURE_DIMENSIONS[j], j * 4 + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
            }

            println("\tuint {}_SamplerDescriptorIndex : packoffset(c{}.{});",
                constantName, 4 * std::size(TEXTURE_DIMENSIONS) + constantInfo->registerIndex / 4, SWIZZLES[constantInfo->registerIndex % 4]);
        }
    }

    out += "\tDEFINE_SHARED_CONSTANTS();\n";
    out += "};\n\n";

    out += "#endif\n";

    for (uint32_t i = 0; i < constantTableContainer->constantTable.constants; i++)
    {
        const auto constantInfo = reinterpret_cast<const ConstantInfo*>(
            constantTableData + constantTableContainer->constantTable.constantInfo + i * sizeof(ConstantInfo));

        if (constantInfo->registerSet == RegisterSet::Bool)
        {
            const char* constantName = reinterpret_cast<const char*>(constantTableData + constantInfo->name);
            println("\t#define {} (1 << {})", constantName, constantInfo->registerIndex + (isPixelShader ? 16 : 0));
            // In MASSEFFECT CTAB numbers b0.. and CF uses 128.. for pixels.
            // The g_Booleans packing keeps its 16 bits per stage.
            boolConstants.emplace(constantInfo->registerIndex + (isPixelShader ? 128 : 0), constantName);
        }
    }

    out += '\n';

    const auto shader = reinterpret_cast<const Shader*>(shaderData + shaderContainer->shaderOffset);

    out += "#ifndef __spirv__\n";

    if (isPixelShader)
        out += "[shader(\"pixel\")]\n";
    else
        out += "[shader(\"vertex\")]\n";

    out += "#endif\n";

    out += "void main(\n";

    if (isPixelShader)
    {
        out += "\tin float4 iPos : SV_Position,\n";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            println("\tin float4 i{0}{1} : {2}{1},", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);

        out += "#ifdef __spirv__\n";
        out += "\tin bool iFace : SV_IsFrontFace\n";
        out += "#else\n";
        out += "\tin uint iFace : SV_IsFrontFace\n";
        out += "#endif\n";

        auto pixelShader = reinterpret_cast<const PixelShader*>(shader);
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR0)
            out += ",\n\tout float4 oC0 : SV_Target0";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR1)
            out += ",\n\tout float4 oC1 : SV_Target1";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR2)
            out += ",\n\tout float4 oC2 : SV_Target2";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_COLOR3)
            out += ",\n\tout float4 oC3 : SV_Target3";
        if (pixelShader->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
            out += ",\n\tout float oDepth : SV_Depth";
    }
    else
    {
        auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
        // MASSEFFECT: Direct3D's own shaders are linked by fetch destination register: one TEXCOORD<register>
        // input per full fetch, whatever the element table says (Direct3D rewrites these fetches by register).
        std::vector<std::pair<uint32_t, uint32_t>> fetchesByRegister;
        if (linkByRegister)
        {
            const auto* words = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);
            std::vector<uint32_t> microcode(shader->size / 4);
            for (size_t i = 0; i < microcode.size(); i++)
                microcode[i] = words[i];
            auto split = classifyVertexFetches(microcode);
            fetchesByRegister = std::move(split.attributes);
            indexedFetches = std::move(split.indexed);
        }
        const uint32_t elementCount = linkByRegister ? uint32_t(fetchesByRegister.size()) : uint32_t(vertexShader->vertexElementCount);
        for (uint32_t i = 0; i < elementCount; i++)
        {
            union
            {
                VertexElement vertexElement;
                uint32_t value;
            };

            if (linkByRegister)
            {
                value = 0;
                vertexElement.address = fetchesByRegister[i].first;
                vertexElement.usage = DeclUsage::TexCoord;
                vertexElement.usageIndex = fetchesByRegister[i].second;
            }
            else
                value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + i];

            const char* usageType = USAGE_TYPES[uint32_t(vertexElement.usage)];

        #ifdef UNLEASHED_RECOMP
            if ((vertexElement.usage == DeclUsage::TexCoord && vertexElement.usageIndex == 2 && isMetaInstancer) ||
                (vertexElement.usage == DeclUsage::Position && vertexElement.usageIndex == 1))
            {
                usageType = "uint4";
            }
        #endif

            out += '\t';

            if (linkByRegister)
                print("[[vk::location({})]] ", banjoSlotLocation(uint32_t(vertexElement.usageIndex)));
            else
            for (auto& usageLocation : USAGE_LOCATIONS)
            {
                if (usageLocation.usage == vertexElement.usage && usageLocation.usageIndex == vertexElement.usageIndex)
                {
                    print("[[vk::location({})]] ", usageLocation.location);
                    break;
                }
            }

            println("in {0} i{1}{2} : {3}{2},", usageType, USAGE_VARIABLES[uint32_t(vertexElement.usage)],
                uint32_t(vertexElement.usageIndex), USAGE_SEMANTICS[uint32_t(vertexElement.usage)]);

            vertexElements.emplace(uint32_t(vertexElement.address), vertexElement);
        }

    #ifdef UNLEASHED_RECOMP
        if (hasIndexCount)
        {
            out += "\tin uint iVertexId : SV_VertexID,\n";
            out += "\tin uint iInstanceId : SV_InstanceID,\n";
        }
    #else
        // banjo-nx: a Xenos vertex shader starts with the vertex index in r0.x (see the register initialization).
        out += "\tin uint iVertexId : SV_VertexID,\n";
    #endif

        out += precisePosition ? "\t[[vk::ext_decorate(18)]] out precise float4 oPos : SV_Position" : "\tout float4 oPos : SV_Position";

        for (auto& [usage, usageIndex] : INTERPOLATORS)
            print(",\n\tout float4 o{0}{1} : {2}{1}", USAGE_VARIABLES[uint32_t(usage)], usageIndex, USAGE_SEMANTICS[uint32_t(usage)]);
    }

    out += ")\n";
    out += "{\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
    {
        specConstantsMask |= SPEC_CONSTANT_REVERSE_Z;

        out += "\toPos = 0.0;\n";

        out += "\tfloat4x4 mtxProjection = float4x4(g_MtxProjection(0), g_MtxProjection(1), g_MtxProjection(2), g_MtxProjection(3));\n";
        out += "\tfloat4x4 mtxProjectionReverseZ = mul(mtxProjection, float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 1));\n";

        out += "\t[unroll] for (int iterationIndex = 0; iterationIndex < 2; iterationIndex++)\n";
        out += "\t{\n";
    }
#endif

    if (shaderContainer->definitionTableOffset != NULL)
    {
        auto definitionTable = reinterpret_cast<const DefinitionTable*>(shaderData + shaderContainer->definitionTableOffset);
        auto definitions = definitionTable->definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Float4Definition*>(definitions);
            if (!validFloatDefinition(*definition))
            {
                println("\t// Relocated definition c{}: read the guest registers, not overlapping microcode.",
                    definition->registerIndex - (isPixelShader ? 256 : 0));
                definitions += 2;
                continue;
            }
            auto value = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + definition->physicalOffset);
            for (uint16_t i = 0; i < (definition->count + 3) / 4; i++)
            {
                println("\tfloat4 c{} = asfloat(uint4(0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}));",
                    definition->registerIndex + i - (isPixelShader ? 256 : 0), value[0].get(), value[1].get(), value[2].get(), value[3].get());

                value += 4;
            }
            definitions += 2;
        }
        ++definitions;
        while (*definitions != 0)
        {
            auto definition = reinterpret_cast<const Int4Definition*>(definitions);
            for (uint16_t i = 0; i < definition->count; i++)
            {
                /*
                 * Named: GCC does not accept the anonymous local union member
                 * that MSVC and clang accept with -fms-extensions.
                 */
                union Bytes4
                {
                    uint32_t value;
                    struct { int8_t x, y, z, w; } b;
                } u;

                u.value = definition->values[i].get();

                println("\tint4 i{} = int4({}, {}, {}, {});",
                    (definition->registerIndex - 8992) / 4 + i, u.b.x, u.b.y, u.b.z, u.b.w);
            }
            definitions += 2;
            definitions += definition->count;
        }

        out += "\n";
    }

    bool printedRegisters[32]{};

    // sa2-nx: interpolators are ALWAYS linked by hardware register, whatever the constant table says. The GPU knows
    // no semantics: vertex shader export N reaches the pixel shader's interpolator N (Direct3D patches the export
    // registers when it links a pair, and the microcode the ring loads is the linked one). Linking some shaders by
    // semantic and others by register broke every mixed pair: a game vertex shader with no constants (an empty
    // CTAB, so "register-linked") drawing SA2's warning/SEGA/Sonic Team screens with a semantic pixel shader that
    // keeps COLOR0 in register 0 and TEXCOORD0 in register 1 got its texture coordinates from the colour (black).
    const bool interpolatorsByRegister = true;

    // MASSEFFECT: register linking: export register N of the vertex shader is TEXCOORDN, read back by the
    // pixel shader's register N.
    if (interpolatorsByRegister && !isPixelShader)
    {
        for (uint32_t i = 0; i < 16; i++)
            interpolators.emplace(i, fmt::format("oTexCoord{}", i));
    }

    uint32_t interpolatorCount = (shader->interpolatorInfo >> 5) & 0x1F;

    for (uint32_t i = 0; i < interpolatorCount; i++)
    {
        union
        {
            Interpolator interpolator;
            uint32_t value;
        };
    
        if (isPixelShader)
        {
            value = reinterpret_cast<const PixelShader*>(shader)->interpolators[i];
            if (interpolatorsByRegister)
                println("\tfloat4 r{0} = iTexCoord{0};", uint32_t(interpolator.reg));
            else
                println("\tfloat4 r{} = i{}{};", uint32_t(interpolator.reg), USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex));
            printedRegisters[interpolator.reg] = true;
        }
        else
        {
            auto vertexShader = reinterpret_cast<const VertexShader*>(shader);
            value = vertexShader->vertexElementsAndInterpolators[vertexShader->field18 + vertexShader->vertexElementCount + i];
            if (!interpolatorsByRegister)
                interpolators.emplace(i, fmt::format("o{}{}", USAGE_VARIABLES[uint32_t(interpolator.usage)], uint32_t(interpolator.usageIndex)));
        }
    }

    if (!isPixelShader)
    {
    #ifdef UNLEASHED_RECOMP
        if (!hasMtxProjection)
            out += "\toPos = 0.0;\n";
    #endif

        for (auto& [usage, usageIndex] : INTERPOLATORS)
        {
            // Xenos' default value for an interpolator which the VS does not export is
            // (0, 0, 0, 1). This matters for register-linked Direct3D shaders: Mass Effect's
            // UI pixel shader multiplies its alpha by r1.w while several paired vertex shaders
            // only export r0. Initializing every unused output to float4(0,0,0,0) made that UI
            // completely transparent.
            if (interpolatorsByRegister)
                println("\to{}{} = float4(0.0, 0.0, 0.0, 1.0);", USAGE_VARIABLES[uint32_t(usage)], usageIndex);
            else
                println("\to{}{} = 0.0;", USAGE_VARIABLES[uint32_t(usage)], usageIndex);
        }

        out += "\n";
    }

    for (size_t i = 0; i < 32; i++)
    {
        if (!printedRegisters[i])
        {
            if (precisePosition) print("\tprecise float4 r{} = ", i);
            else print("\tfloat4 r{} = ", i);
            // sa2-nx: register linking, as the hardware does it: the pixel shader's register N is loaded with
            // interpolator N whether or not the container declares it. A container rebuilt from a template around
            // ring microcode (MASSEFFECT_SHADER_MISSING) carries the template's interpolator list, not the shader's:
            // Direct3D's own Clear pixel shader (ps_ba62fb45, oC0 = r0) got an empty list, so r0 became the pixel
            // position and every clear wrote (x, y, 0, 0) instead of the clear colour. A shader that declares no
            // interpolator therefore reads them all; shaders that declare some keep their layout.
            if (isPixelShader && interpolatorsByRegister && interpolatorCount == 0 && i < 16)
            {
                out += fmt::format("iTexCoord{};\n", i);
            }
            else if (isPixelShader && i == ((shader->fieldC >> 8) & 0xFF))
            {
                out += "float4((iPos.xy - 0.5) * float2(iFace ? 1.0 : -1.0, 1.0), 0.0, 0.0);\n";
            }
        #ifdef UNLEASHED_RECOMP
            else if (!isPixelShader && hasIndexCount && i == 0)
            {
                out += "float4(iVertexId + g_IndexCount.x * iInstanceId, 0.0, 0.0, 0.0);\n";
            }
        #else
            else if (!isPixelShader && i == 0 && !vertexIndexMayBeRead(
                reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset),
                shader->size, indexedFetches))
            {
                // sa2-nx (sa2_gpu_index_base_held): nothing can read the vertex index r0 starts with (see
                // vertexIndexMayBeRead): with the bit, r0 starts at 0 like the other registers, the shader does not
                // load the index, and the renderer may hold g_VertexIndexBase.
                specConstantsMask |= SPEC_CONSTANT_VERTEX_INDEX_UNUSED;
                out += "(g_SpecConstants() & SPEC_CONSTANT_VERTEX_INDEX_UNUSED) != 0 ? float4(0.0, 0.0, 0.0, 0.0) : "
                       "float4(float(iVertexId + g_VertexIndexBase), 0.0, 0.0, 0.0);\n";
            }
            else if (!isPixelShader && i == 0)
            {
                out += "float4(float(iVertexId + g_VertexIndexBase), 0.0, 0.0, 0.0);\n";
            }
        #endif
            else
            {
                out += "0.0;\n";
            }
        }
    }

    out += "\tint a0 = 0;\n";
    out += "\tint banjoNextA0 = 0;\n";
    out += "\tint aL = 0;\n";
    if (!isPixelShader)
        out += "\tfloat4 oUnused = 0.0;\n";
    if (isPixelShader && (reinterpret_cast<const PixelShader*>(shader)->outputs & PIXEL_SHADER_OUTPUT_DEPTH))
        out += "\tfloat4 oDepth4 = 0.0;\n";
    if (!isPixelShader)
        out += "\tint banjoVfetchIndex = 0;\n";
    out += "\tbool p0 = false;\n";
    out += precisePosition ? "\tprecise float ps = 0.0;\n" : "\tfloat ps = 0.0;\n";
    if (isPixelShader)
    {
#ifdef UNLEASHED_RECOMP
        out += "\tfloat2 pixelCoord = 0.0;\n";
#endif
        out += "\tCubeMapData cubeMapData = (CubeMapData)0;\n";
    }

    const be<uint32_t>* code = reinterpret_cast<const be<uint32_t>*>(shaderData + shaderContainer->virtualSize + shader->physicalOffset);

    /*
     * Named, with references: GCC does not accept the anonymous struct inside a local
     * union that MSVC and clang accept. The references leave the rest of the body
     * untouched, which uses code0..code3 eighteen times.
     */
    union ControlFlowWords
    {
        ControlFlowInstruction controlFlow[2];
        struct { uint32_t code0, code1, code2, code3; } w;
    } cfu;
    uint32_t& code0 = cfu.w.code0;
    uint32_t& code1 = cfu.w.code1;
    uint32_t& code2 = cfu.w.code2;
    uint32_t& code3 = cfu.w.code3;

    auto controlFlowCode = code;
    uint32_t instrAddress = 0;
    uint32_t instrSize = shader->size;
    bool simpleControlFlow = true;
    // Mass Effect: every jump goes forward and there are no loops (the 34 skinned VS of the
    // game). Then no dispatcher loop is needed, see skipControlFlow below.
    bool forwardOnlyControlFlow = true;
    uint32_t scanPc = 0;
    // sa2-nx (sa2_gpu_structured_cf): every jump of the scan (pc -> unconditional, target), for the structuring below,
    // and every CF instruction that ends the shader (ExecEnd) or may end it (a CondExec*End, inside its own if).
    std::unordered_map<uint32_t, std::pair<bool, uint32_t>> scanJumps;
    std::unordered_set<uint32_t> scanConditionalReturns, scanReturns;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : cfu.controlFlow)
        {
            uint32_t address = 0;

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                if (cfInstr.opcode == ControlFlowOpcode::ExecEnd)
                    scanReturns.insert(scanPc);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                if (cfInstr.opcode == ControlFlowOpcode::CondExecEnd ||
                    cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd)
                    scanConditionalReturns.insert(scanPc);
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                if (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd)
                    scanConditionalReturns.insert(scanPc);
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional || cfInstr.condJmp.direction)
                    simpleControlFlow = false;
                else
                    ++ifEndLabels[cfInstr.condJmp.address];

                if (uint32_t(cfInstr.condJmp.address) <= scanPc)
                    forwardOnlyControlFlow = false;
                scanJumps[scanPc] = {bool(cfInstr.condJmp.isUnconditional), uint32_t(cfInstr.condJmp.address)};
                break;
            }

            case ControlFlowOpcode::LoopStart:
            case ControlFlowOpcode::LoopEnd:
                forwardOnlyControlFlow = false;
                break;
            }

            if (address != 0)
                instrSize = std::min<uint32_t>(instrSize, address * 12);
            ++scanPc;
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    /*
     * Mass Effect: forward-only jumps without loops are emitted as straight-line code: each CF
     * instruction runs under `if (skipTo <= its index)` and a jump sets skipTo to its target.
     * Exactly the same order and semantics as the pc dispatcher below, without its loop and switch.
     * The dispatcher made the skinned VS cost ~5 us per vertex on the Switch (NVK/Maxwell): every
     * case went back through the loop header, so nothing stayed in registers across the 80+ cases.
     * Like the explicit dispatch, it has no switch fall-through either (the Metal concern below).
     */
    bool skipControlFlow = !simpleControlFlow && forwardOnlyControlFlow;

    /*
     * sa2-nx GPU round 2 (sa2_gpu_structured_cf): the same forward-only control flow as nested if/else.
     *
     * The skipTo form puts every CF instruction under `if (skipTo <= pc)`. After two jumps, skipTo is a phi of three
     * or more constants, and NIR's bcsel folding (nir_opt_algebraic: a comparison of a bcsel of two constants) no
     * longer turns those tests into the jump conditions: SA2's lighting vertex shaders (72 CF instructions, 11 jumps)
     * keep a compare and a branch per CF instruction (SSY/BRA/SYNC on Maxwell, which has no uniform datapath), and the
     * wrapper text between two `if (p0)` blocks stops their merging (reopenClosing). Compiler-generated Xenos jumps
     * nest: a conditional jump skips to its target (`if (!jump) { ... }`), and an unconditional jump that is the last
     * instruction of such a region skips the code after the region up to its target: that code is the region's else.
     * When every jump of a shader fits that pattern, the structured form is emitted too, under
     * SPEC_CONSTANT_STRUCTURED_CF, and the renderer picks it per pipeline. Before emitting it, the translator checks
     * that it runs the same CF instructions in the same order as the skipTo form for every combination of jump and
     * conditional-return outcomes; a difference, or a pattern that does not nest, keeps the skipTo form alone.
     */
    std::unordered_map<uint32_t, uint32_t> structuredCloseAt;  // pc -> regions that close before it
    std::unordered_set<uint32_t> structuredElseAt;             // pc -> the innermost region turns into its else here
    uint32_t structuredOpenAtEnd = 0;                          // regions still open after the last CF instruction
    bool structured = false;
    const uint32_t cfCount = (instrSize / 12) * 2;
    if (skipControlFlow)
    {
        struct Region { uint32_t end; bool isElse; uint32_t opener; };
        std::vector<Region> stack;
        // pc -> the regions around it: (the conditional jump that opened the region, true for its else part)
        std::vector<std::vector<std::pair<uint32_t, bool>>> guards(cfCount);
        structured = true;
        for (uint32_t p = 0; p < cfCount && structured; ++p)
        {
            while (!stack.empty() && stack.back().end == p)
            {
                stack.pop_back();
                ++structuredCloseAt[p];
            }
            for (const Region& r : stack)
                guards[p].push_back({r.opener, r.isElse});
            const auto j = scanJumps.find(p);
            if (j == scanJumps.end())
                continue;
            const bool unconditional = j->second.first;
            const uint32_t target = j->second.second;
            if (target <= p)
            {
                structured = false;
            }
            else if (!unconditional)
            {
                if (!stack.empty() && target > stack.back().end)
                    structured = false;
                else
                    stack.push_back({target, false, p});
            }
            else if (stack.empty() || stack.back().end != p + 1 || stack.back().isElse || target <= p + 1 ||
                     (stack.size() >= 2 && target > stack[stack.size() - 2].end))
            {
                structured = false;
            }
            else
            {
                stack.back().end = target;
                stack.back().isElse = true;
                structuredElseAt.insert(p + 1);
            }
        }
        structuredOpenAtEnd = uint32_t(stack.size());

        // The proof: every outcome of the conditional jumps and conditional returns, in both forms.
        std::vector<uint32_t> decisions;  // CF instructions whose outcome is only known at run time
        for (uint32_t p = 0; p < cfCount; ++p)
        {
            const auto j = scanJumps.find(p);
            if ((j != scanJumps.end() && !j->second.first) || scanConditionalReturns.count(p))
                decisions.push_back(p);
        }
        if (structured && decisions.size() > 20)
            structured = false;  // more than a million outcomes: not checked, so not used
        for (uint64_t outcome = 0; structured && outcome < (uint64_t(1) << decisions.size()); ++outcome)
        {
            const auto taken = [&](uint32_t p) {
                for (size_t k = 0; k < decisions.size(); ++k)
                    if (decisions[k] == p)
                        return ((outcome >> k) & 1) != 0;
                return false;
            };
            std::vector<uint32_t> skipRun, structuredRun;
            uint32_t skipTo = 0;
            for (uint32_t p = 0; p < cfCount; ++p)
            {
                if (skipTo > p)
                    continue;
                skipRun.push_back(p);
                if (scanReturns.count(p) || (scanConditionalReturns.count(p) && taken(p)))
                    break;
                const auto j = scanJumps.find(p);
                if (j != scanJumps.end() && (j->second.first || taken(p)))
                    skipTo = j->second.second;
            }
            for (uint32_t p = 0; p < cfCount; ++p)
            {
                bool runs = true;
                for (const auto& [opener, inElse] : guards[p])
                    runs = runs && (taken(opener) == inElse);
                if (!runs)
                    continue;
                structuredRun.push_back(p);
                if (scanReturns.count(p) || (scanConditionalReturns.count(p) && taken(p)))
                    break;
            }
            if (skipRun != structuredRun)
            {
                structured = false;
                fprintf(stderr, "sa2_gpu_structured_cf: outcome %llu runs other CF instructions: skipTo form only\n",
                        (unsigned long long)outcome);
            }
        }
    }

    const int emitPasses = structured ? 2 : 1;
    bool structuredPass = false;
    uint32_t cfIndentation = 1;
    uint32_t pc = 0;
    for (int emitPass = 0; emitPass < emitPasses; ++emitPass)
    {
    if (structured)
    {
        // Each pass starts with no predicate block open or reopenable (the other pass's text comes in between).
        predOpen_ = -1;
        predGuaranteed_ = -1;
        markClosing_ = closingEnd_ = std::string::npos;
        cfIndentation = 2;
        indentation = 2;
        if (emitPass == 0)
        {
            specConstantsMask |= SPEC_CONSTANT_STRUCTURED_CF;
            out += "\n\t[branch] if (g_SpecConstants() & SPEC_CONSTANT_STRUCTURED_CF)\n\t{\n";
            simpleControlFlow = true;
            skipControlFlow = false;
            structuredPass = true;
        }
        else
        {
            out += "\t}\n\telse\n\t{\n\t\tuint skipTo = 0;\n";
            simpleControlFlow = false;
            skipControlFlow = true;
            structuredPass = false;
        }
    }
    else if (simpleControlFlow)
    {
        out += '\n';
        indentation = 1;
    }
    else if (skipControlFlow)
    {
        out += "\n\tuint skipTo = 0;\n";
        indentation = 1;
    }
    else
    {
        out += "\n\tuint pc = 0;\n";
        out += "\twhile (true)\n";
        out += "\t{\n";
        out += "\t\tswitch (pc)\n";
        out += "\t\t{\n";
    }

    controlFlowCode = code;
    instrAddress = 0;
    pc = 0;

    while (instrAddress < instrSize)
    {
        code0 = controlFlowCode[0];
        code1 = controlFlowCode[1] & 0xFFFF;
        code2 = (controlFlowCode[1] >> 16) | (controlFlowCode[2] << 16);
        code3 = controlFlowCode[2] >> 16;

        for (auto& cfInstr : cfu.controlFlow)
        {
            if (skipControlFlow)
            {
                indentation = cfIndentation;
                indent();
                println("if (skipTo <= {})", pc);
                indent();
                out += "{\n";
                ++indentation;
            }
            else if (!simpleControlFlow)
            {
                indentation = 3;
                println("\t\tcase {}:", pc);
            }
            else if (structuredPass)
            {
                if (const auto closes = structuredCloseAt.find(pc); closes != structuredCloseAt.end())
                {
                    for (uint32_t i = 0; i < closes->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
                if (structuredElseAt.count(pc))
                {
                    --indentation;
                    indent();
                    out += "}\n";
                    indent();
                    out += "else\n";
                    indent();
                    out += "{\n";
                    ++indentation;
                }
            }
            else
            {
                auto findResult = ifEndLabels.find(pc);
                if (findResult != ifEndLabels.end())
                {
                    for (uint32_t i = 0; i < findResult->second; i++)
                    {
                        --indentation;
                        indent();
                        out += "}\n";
                    }
                }
            }

            ++pc;

            uint32_t address = 0;
            uint32_t count = 0;
            uint32_t sequence = 0;
            bool shouldReturn = false;
            bool shouldCloseCurlyBracket = false;
            /*
             * If this CF opens its own `if (p0)` (CondExecPred), its condition. It serves
             * two purposes: reusing the identical block that was just closed, and leaving its
             * own brace recorded so the next one can reuse it. See reopenClosing().
             */
            int conditionPredicateCf = -1;

            // Control flow boundary. An open predicated block does not cross this point.
            closePredicate();

            switch (cfInstr.opcode)
            {
            case ControlFlowOpcode::Exec:
            case ControlFlowOpcode::ExecEnd:
                address = cfInstr.exec.address;
                count = cfInstr.exec.count;
                sequence = cfInstr.exec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::ExecEnd);
                break;

            case ControlFlowOpcode::CondExec:
            case ControlFlowOpcode::CondExecEnd:
            case ControlFlowOpcode::CondExecPredClean:
            case ControlFlowOpcode::CondExecPredCleanEnd:
                address = cfInstr.condExec.address;
                count = cfInstr.condExec.count;
                sequence = cfInstr.condExec.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecEnd || cfInstr.opcode == ControlFlowOpcode::CondExecPredCleanEnd);
                {
                    // The condition belongs to the whole EXEC block, including its
                    // return. Per-instruction ALU predication does not replace it.
                    const auto boolean = boolConstants.find(cfInstr.condExec.boolAddress);
                    indent();
                    if (boolean == boolConstants.end())
                        // banjo-nx: no CTAB entry: read the bool by its address from the whole bank.
                        println("if (BANJO_BOOL({}) {}= 0)", uint32_t(cfInstr.condExec.boolAddress), cfInstr.condExec.condition ? "!" : "=");
                    else
                        println("if ((g_Booleans & {}) {}= 0)", boolean->second, cfInstr.condExec.condition ? "!" : "=");
                    indent();
                    out += "{\n";
                    ++indentation;
                    shouldCloseCurlyBracket = true;
                }
                break;

            case ControlFlowOpcode::CondExecPred:
            case ControlFlowOpcode::CondExecPredEnd:
                address = cfInstr.condExecPred.address;
                count = cfInstr.condExecPred.count;
                sequence = cfInstr.condExecPred.sequence;
                shouldReturn = (cfInstr.opcode == ControlFlowOpcode::CondExecPredEnd);
                conditionPredicateCf = cfInstr.condExecPred.condition ? 1 : 0;
                /*
                 * The previous block was this same `if (p0)` and nothing was emitted between
                 * the two: its brace is deleted and this EXEC continues inside. This way runs of
                 * CondExecPred with the same predicate (the nine PCF taps, the six lights)
                 * end up in one basic block instead of three or four.
                 */
                if (!reopenClosing(conditionPredicateCf))
                {
                    indent();
                    println("if ({}p0)", conditionPredicateCf ? "" : "!");
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                shouldCloseCurlyBracket = true;
                // This `if` governs the whole block, so the predicated instructions inside
                // do not have to test it again. See predGuaranteed_.
                predGuaranteed_ = conditionPredicateCf;
                break;

            case ControlFlowOpcode::LoopStart:
                if (simpleControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    print("[unroll] ");
                #endif
                    println("for (aL = 0; aL < i{}.x; aL++)", uint32_t(cfInstr.loopStart.loopId));
                    indent();
                    out += "{\n";
                    ++indentation;
                }
                else 
                {
                    out += "\t\t\taL = 0;\n";
                }
                break;

            case ControlFlowOpcode::LoopEnd:
                if (simpleControlFlow)
                {
                    --indentation;
                    indent();
                    out += "}\n";
                }
                else
                {
                    out += "\t\t\t++aL;\n";
                    println("\t\t\tif (aL < i{}.x)", uint32_t(cfInstr.loopEnd.loopId));
                    out += "\t\t\t{\n";
                    println("\t\t\t\tpc = {};", uint32_t(cfInstr.loopEnd.address));
                    out += "\t\t\t\tcontinue;\n";
                    out += "\t\t\t}\n";
                }
                break;

            case ControlFlowOpcode::CondJmp:
            {
                if (cfInstr.condJmp.isUnconditional && structuredPass)
                {
                    // Nothing here: the region this jump ends turns into its else at the region's end (structuredElseAt).
                }
                else if (cfInstr.condJmp.isUnconditional)
                {
                    assert(!simpleControlFlow);
                    if (skipControlFlow)
                    {
                        indent();
                        println("skipTo = {};", uint32_t(cfInstr.condJmp.address));
                    }
                    else
                    {
                        println("\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\tcontinue;\n";
                    }
                }
                else
                {
                    indent();
                    if (cfInstr.condJmp.isPredicated)
                    {
                        println("if ({}p0)", cfInstr.condJmp.condition ^ simpleControlFlow ? "" : "!");
                    }
                    else
                    {
                        auto findResult = boolConstants.find(cfInstr.condJmp.boolAddress);
                        if (findResult != boolConstants.end())
                            println("if ((g_Booleans & {}) {}= 0)", findResult->second, cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");
                        else
                            println("if (BANJO_BOOL({}) {}= 0)", uint32_t(cfInstr.condJmp.boolAddress), cfInstr.condJmp.condition ^ simpleControlFlow ? "!" : "=");  // banjo-nx
                    }

                    if (simpleControlFlow)
                    {
                        indent();
                        out += "{\n";
                        ++indentation;
                    }
                    else if (skipControlFlow)
                    {
                        indent();
                        out += "{\n";
                        indent();
                        println("\tskipTo = {};", uint32_t(cfInstr.condJmp.address));
                        indent();
                        out += "}\n";
                    }
                    else
                    {
                        out += "\t\t\t{\n";
                        println("\t\t\t\tpc = {};", uint32_t(cfInstr.condJmp.address));
                        out += "\t\t\t\tcontinue;\n";
                        out += "\t\t\t}\n";
                    }
                }
                break;
            }
            }

            auto instructionCode = code + address * 3;
            
            for (uint32_t i = 0; i < count; i++)
            {
                /*
                 * Named, for the same reason as the other two: GCC does not accept
                 * the anonymous struct inside a local union.
                 */
                union InstrWords
                {
                    VertexFetchInstruction vertexFetch;
                    TextureFetchInstruction textureFetch;
                    AluInstruction alu;
                    struct { uint32_t code0, code1, code2; } w;
                } iu;
                VertexFetchInstruction& vertexFetch = iu.vertexFetch;
                TextureFetchInstruction& textureFetch = iu.textureFetch;
                AluInstruction& alu = iu.alu;

                iu.w.code0 = instructionCode[0];
                iu.w.code1 = instructionCode[1];
                iu.w.code2 = instructionCode[2];
            
                if ((sequence & 0x1) != 0)
                {
                    if (vertexFetch.opcode == FetchOpcode::VertexFetch)
                    {
                        recompile(vertexFetch, address + i);
                    }
                    else
                    {
                    #ifdef UNLEASHED_RECOMP
                        if (textureFetch.constIndex == 10) // g_GISampler
                        {
                            specConstantsMask |= SPEC_CONSTANT_BICUBIC_GI_FILTER;

                            indent();
                            out += "if (g_SpecConstants() & SPEC_CONSTANT_BICUBIC_GI_FILTER)";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, true);
                            --indentation;

                            indent();
                            out += "}";
                            indent();
                            out += "else";
                            indent();
                            out += '{';

                            ++indentation;
                            recompile(textureFetch, false);
                            --indentation;

                            indent();
                            out += '}';
                        }
                        else
                    #endif
                        {
                            recompile(textureFetch, false);
                        }
                    }
                }
                else
                {
                    recompile(alu);
                }
            
                sequence >>= 2;
                instructionCode += 3;
            }

            closePredicate();  // end of the EXEC instructions

            if (shouldReturn)
            {
                if (isPixelShader)
                {
                    if (reinterpret_cast<const PixelShader*>(shader)->outputs & PIXEL_SHADER_OUTPUT_DEPTH)
                    {
                        indent();
                        out += "oDepth = oDepth4.x;\n";
                    }
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TEST;

                    indent();
                    out += "[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST)";
                    indent();
                    out += '{';

                    indent();
                    // MASSEFFECT: the 8 comparison functions (alphaTestValue, shader_common.h).
                    out += "\tclip(alphaTestValue(oC0.w));\n";

                    indent();
                    out += "}";

                #ifdef UNLEASHED_RECOMP
                    specConstantsMask |= SPEC_CONSTANT_ALPHA_TO_COVERAGE;

                    indent();
                    out += "else if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TO_COVERAGE)";
                    indent();
                    out += '{';

                    indent();
                    out += "\toC0.w *= 1.0 + computeMipLevel(pixelCoord) * 0.25;\n";
                    indent();
                    out += "\toC0.w = 0.5 + (oC0.w - g_AlphaThreshold) / max(fwidth(oC0.w), 1e-6);\n";

                    indent();
                    out += '}';
                #endif
                #ifndef UNLEASHED_RECOMP
                    // sa2-nx (sa2_gpu_blend_skip): a transparent pixel whose blend would leave the target exactly as it
                    // was is discarded before the blend. The renderer sets these bits only for UNORM targets with no
                    // depth or stencil write and a blend that keeps the destination; a NaN is never discarded.
                    specConstantsMask |= SPEC_CONSTANT_BLEND_SKIP_ALPHA | SPEC_CONSTANT_BLEND_SKIP_ZERO;
                    out += "\n";
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ALPHA) != 0 && oC0.w <= 0.0) discard;\n";
                    indent();
                    out += "if ((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ZERO) != 0 && all(oC0 <= 0.0)) discard;\n";
                #endif
                }
                else
                {
                #ifdef UNLEASHED_RECOMP
                    if (!hasMtxProjection)
                #endif
                    {
                        // To host clip space (ndc_scale/ndc_offset from
                        // draw.cpp): draws in pixels with clipping disabled.
                        out += "\toPos.xy = oPos.xy * g_NdcScale + g_NdcOffset * oPos.w;\n";
                        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
                    }
                }

                if (simpleControlFlow || skipControlFlow)
                {
                    indent();
                #ifdef UNLEASHED_RECOMP
                    if (hasMtxProjection)
                    {
                        out += "continue;\n";
                    }
                    else
                #endif
                    {
                        out += "return;\n";
                    }
                }
                else
                {
                    out += "\t\t\tbreak;\n";
                }
            }

            if (shouldCloseCurlyBracket)
            {
                --indentation;
                const size_t markKeyCf = out.size();
                indent();
                out += "}\n";
                /*
                 * It is recorded in case the next CondExecPred is identical. Not with a
                 * `return` inside, since the block does not continue; and noteClosingCf() also
                 * refuses if something wrote p0 in here (predGuaranteed_ was set to -1): the
                 * next block has to test the new p0.
                 */
                noteClosingCf(markKeyCf, shouldReturn ? -1 : conditionPredicateCf);
            }
            predGuaranteed_ = -1;  // end of the EXEC block: it no longer guarantees anything
            // Do not rely on implicit switch fall-through for complex Xenos control flow. Large skinned
            // vertex shaders produce dozens of cases, and the Vulkan -> Metal path may lower the long
            // fall-through chain incorrectly. Dispatching the next CF instruction explicitly is exactly
            // equivalent, and also makes every case a closed basic block in SPIR-V/MSL.
            if (skipControlFlow)
            {
                --indentation;
                indent();
                out += "}\n";
            }
            else if (!simpleControlFlow && !shouldReturn)
            {
                println("\t\t\tpc = {};", pc);
                out += "\t\t\tcontinue;\n";
            }
        }

        controlFlowCode += 3;
        instrAddress += 12;
    }

    if (structuredPass)
    {
        for (uint32_t i = 0; i < structuredOpenAtEnd; i++)
        {
            --indentation;
            indent();
            out += "}\n";
        }
    }
    if (!simpleControlFlow && !skipControlFlow)
    {
        out += "\t\t\tbreak;\n";
        out += "\t\t}\n";
        out += "\t\tbreak;\n";
        out += "\t}\n";
    }
    }  // emitPass
    if (structured)
        out += "\t}\n";

#ifdef UNLEASHED_RECOMP
    if (hasMtxProjection)
        out += "\t}\n";

    if (!isPixelShader && hasMtxProjection)
        out += "\toPos.xy += g_HalfPixelOffset * oPos.w;\n";
#endif

    out += "}";
}
