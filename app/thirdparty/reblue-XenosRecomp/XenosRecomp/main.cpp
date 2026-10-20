#include "shader.h"
#include "shader_recompiler.h"
#include "dxc_compiler.h"
#ifdef RENUT_RECOMP
#include "banjo_pred_select.h"
#include "banjo_bool_spec.h"
#include <atomic>
// banjo-nx: rewritten shaders (BANJO_HLSL_OVERRIDE) that fell back to their own text.
static std::atomic<size_t> banjoOverrideFallbacks{0};
static std::atomic<size_t> banjoKillShaders{0}, banjoDepthShaders{0};  // banjo-nx (banjo_gpu_depth_only)
#endif

#include <mutex>
#include <vector>

static std::unique_ptr<uint8_t[]> readAllBytes(const char* filePath, size_t& fileSize)
{
    FILE* file = fopen(filePath, "rb");
    fseek(file, 0, SEEK_END);
    fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);
    auto data = std::make_unique<uint8_t[]>(fileSize);
    fread(data.get(), 1, fileSize, file);
    fclose(file);
    return data;
}

static void writeAllBytes(const char* filePath, const void* data, size_t dataSize)
{
    FILE* file = fopen(filePath, "wb");
    fwrite(data, 1, dataSize, file);
    fclose(file);
}

struct RecompiledShader
{
    uint8_t* data = nullptr;
    IDxcBlob* dxil = nullptr;
    std::vector<uint8_t> spirv;
    uint32_t specConstantsMask = 0;
#ifdef RENUT_RECOMP
    uint32_t vertexFetchMemoryMask = 0;
    uint32_t textureFetchMask = 0;
    uint32_t memExportConstants[8] = {};
    uint32_t floatConstants[8] = {};
    uint32_t shaderFlags = 0;  // banjo-nx (banjo_gpu_depth_only): ShaderRecompiler::shaderFlags
    std::vector<std::array<uint32_t, 5>> cbufLiterals;  // banjo-nx (banjo_gpu_literals_cbuf)
    std::vector<ShaderRecompiler::VertexInput> vertexInputs;
    // The file (and container index) the shader came from, naming its HLSL dump.
    std::string sourceName;
#endif
};

// Per-shader recompile failures, collected from the parallel loop and reported before exit.
struct ShaderFailure
{
    XXH64_hash_t hash;
    std::string reason;
};

static std::mutex g_failureMutex;
static std::vector<ShaderFailure> g_failures;

int main(int argc, char** argv)
{
#ifndef XENOS_RECOMP_INPUT
    if (argc < 4)
    {
#ifdef RENUT_RECOMP
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path] [optional: HLSL dump dir]");
#else
        printf("Usage: XenosRecomp [input path] [output path] [shader common header file path]");
#endif
        return 0;
    }
#endif

    const char* input =
#ifdef XENOS_RECOMP_INPUT 
        XENOS_RECOMP_INPUT
#else
        argv[1]
#endif
    ;

    const char* output =
#ifdef XENOS_RECOMP_OUTPUT 
        XENOS_RECOMP_OUTPUT
#else
        argv[2]
#endif
        ;
    
    const char* includeInput =
#ifdef XENOS_RECOMP_INCLUDE_INPUT
        XENOS_RECOMP_INCLUDE_INPUT
#else
        argv[3]
#endif
        ;

#ifdef RENUT_RECOMP
    std::string hlslDumpDir;
    if (argc > 4)
        hlslDumpDir = argv[4];
    else if (const char* env = std::getenv("XENOS_RECOMP_HLSL_DUMP"))
        hlslDumpDir = env;
#endif

    size_t includeSize = 0;
    auto includeData = readAllBytes(includeInput, includeSize);
    std::string_view include(reinterpret_cast<const char*>(includeData.get()), includeSize);

    if (std::filesystem::is_directory(input))
    {
        std::vector<std::unique_ptr<uint8_t[]>> files;
        std::map<XXH64_hash_t, RecompiledShader> shaders;

        for (auto& file : std::filesystem::recursive_directory_iterator(input))
        {
            if (std::filesystem::is_directory(file))
            {
                continue;
            }
            
            size_t fileSize = 0;
            auto fileData = readAllBytes(file.path().string().c_str(), fileSize);
            bool foundAny = false;
#ifdef RENUT_RECOMP
            int containerIndex = 0;
#endif

            for (size_t i = 0; fileSize > sizeof(ShaderContainer) && i < fileSize - sizeof(ShaderContainer) - 1;)
            {
                auto shaderContainer = reinterpret_cast<const ShaderContainer*>(fileData.get() + i);
                size_t dataSize = shaderContainer->virtualSize + shaderContainer->physicalSize;

                if ((shaderContainer->flags & 0xFFFFFF00) == 0x102A1100 &&
                    dataSize <= (fileSize - i) &&
                    shaderContainer->field1C == 0 &&
                    shaderContainer->field20 == 0)
                {
                    XXH64_hash_t hash = XXH3_64bits(shaderContainer, dataSize);
                    auto shader = shaders.try_emplace(hash);
                    if (shader.second)
                    {
                        shader.first->second.data = fileData.get() + i;
#ifdef RENUT_RECOMP
                        std::string stem = file.path().stem().string();
                        if (containerIndex > 0)
                            stem += fmt::format(".{}", containerIndex);
                        shader.first->second.sourceName = std::move(stem);
#endif
                        foundAny = true;
                    }

#ifdef RENUT_RECOMP
                    ++containerIndex;
#endif
                    i += dataSize;
                }
                else
                {
                    i += sizeof(uint32_t);
                }
            }

            if (foundAny)
                files.emplace_back(std::move(fileData));
        }

#ifdef RENUT_RECOMP
        if (!hlslDumpDir.empty())
            std::filesystem::create_directories(hlslDumpDir);
#endif

        std::atomic<uint32_t> progress = 0;

        std::for_each(std::execution::par_unseq, shaders.begin(), shaders.end(), [&](auto& hashShaderPair)
            {
                auto& shader = hashShaderPair.second;
                const XXH64_hash_t hash = hashShaderPair.first;

                auto recordFailure = [hash](const char* reason)
                {
                    std::lock_guard<std::mutex> lock(g_failureMutex);
                    g_failures.push_back({hash, reason});
                };

                thread_local ShaderRecompiler recompiler;
                recompiler = {};
                recompiler.recompile(shader.data, include);

                shader.specConstantsMask = recompiler.specConstantsMask;
#ifdef RENUT_RECOMP
                shader.vertexFetchMemoryMask = recompiler.vertexFetchMemoryMask;
                shader.textureFetchMask = recompiler.textureFetchMask;
                std::copy(std::begin(recompiler.memExportConstants), std::end(recompiler.memExportConstants),
                    std::begin(shader.memExportConstants));
                std::copy(std::begin(recompiler.floatConstants), std::end(recompiler.floatConstants),
                    std::begin(shader.floatConstants));
                shader.cbufLiterals = recompiler.cbufLiterals;  // banjo-nx
                shader.shaderFlags = recompiler.shaderFlags;    // banjo-nx (banjo_gpu_depth_only)
                if (recompiler.isPixelShader)
                {
                    // banjo-nx (banjo_gpu_depth_only): the flags against the shader's own text (a kill is its only
                    // clip(); the alpha test, the blend skip and the early alpha discard in shader_common.h). A draw
                    // without its pixel shader must not lose a kill or a depth export the flags missed.
                    const bool kills = recompiler.out.find("clip(") != std::string::npos;
                    const bool depth = recompiler.out.find("SV_Depth") != std::string::npos;
                    if (kills != ((shader.shaderFlags & ShaderRecompiler::SHADER_FLAG_PIXEL_KILL) != 0))
                        recordFailure("banjo_gpu_depth_only: the kill flag does not match the shader's clip()");
                    if (depth != ((shader.shaderFlags & ShaderRecompiler::SHADER_FLAG_PIXEL_DEPTH) != 0))
                        recordFailure("banjo_gpu_depth_only: the depth flag does not match the shader's SV_Depth");
                    if (kills)
                        ++banjoKillShaders;
                    if (depth)
                        ++banjoDepthShaders;
                }
                shader.vertexInputs = recompiler.vertexInputs;

                // banjo-nx (package 27): the first of the two runs around tools/renut_early_alpha.py only dumps the
                // pixel shaders (BANJO_HLSL_DUMP_ONLY): no compile, and its cache file is thrown away.
                const bool dumpOnly = std::getenv("BANJO_HLSL_DUMP_ONLY") != nullptr;
                if (!hlslDumpDir.empty() && (recompiler.isPixelShader || !dumpOnly))
                {
                    auto path = std::filesystem::path(hlslDumpDir) /
                        (shader.sourceName.empty()
                            ? fmt::format("{}_{:016X}", recompiler.isPixelShader ? "ps" : "vs", hash)
                            : shader.sourceName);
                    path += ".hlsl";

                    std::string contents = fmt::format(
                        "// {} shader  hash=0x{:016X}  specConstants=0x{:X}\n",
                        recompiler.isPixelShader ? "pixel" : "vertex",
                        hash, recompiler.specConstantsMask);
                    contents.append(recompiler.out);
                    writeAllBytes(path.string().c_str(), contents.data(), contents.size());
                }
                if (dumpOnly)
                    return;

                // banjo-nx (package 27): a pass over the HLSL between two runs of this tool (tools/renut_early_alpha.py,
                // banjo_gpu_early_alpha): the first run dumps (XENOS_RECOMP_HLSL_DUMP), the pass rewrites the dumps,
                // and the second run compiles the file of BANJO_HLSL_OVERRIDE with this shader's dump name instead of
                // its own text where there is one. If the rewritten text does not compile, the shader's own text is
                // compiled instead (a shader is never lost to the pass).
                std::string originalHlsl;
                if (const char* overrideDir = std::getenv("BANJO_HLSL_OVERRIDE"); overrideDir && *overrideDir)
                {
                    auto path = std::filesystem::path(overrideDir) /
                        (shader.sourceName.empty()
                            ? fmt::format("{}_{:016X}", recompiler.isPixelShader ? "ps" : "vs", hash)
                            : shader.sourceName);
                    path += ".hlsl";
                    std::error_code ec;
                    if (std::filesystem::is_regular_file(path, ec))
                    {
                        size_t size = 0;
                        auto bytes = readAllBytes(path.string().c_str(), size);
                        if (bytes && size)
                        {
                            originalHlsl = std::move(recompiler.out);
                            recompiler.out.assign(reinterpret_cast<const char*>(bytes.get()), size);
                        }
                    }
                }
#endif

                thread_local DxcCompiler dxcCompiler;

#ifdef XENOS_RECOMP_DXIL
                shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
#ifdef RENUT_RECOMP
                if (shader.dxil == nullptr && !originalHlsl.empty())
                {
                    // banjo-nx: the override did not compile; the shader's own text (above).
                    recompiler.out = std::move(originalHlsl);
                    originalHlsl.clear();
                    banjoOverrideFallbacks.fetch_add(1, std::memory_order_relaxed);
                    shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
                }
#endif
                if (shader.dxil == nullptr)
                {
                    recordFailure("dxc-dxil-compile-failed");
                    return;
                }
                if (*(reinterpret_cast<uint32_t*>(shader.dxil->GetBufferPointer()) + 1) == 0)
                {
                    recordFailure("dxil-not-signed");
                    return;
                }
#endif

                IDxcBlob* spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
#ifdef RENUT_RECOMP
                if (spirv == nullptr && !originalHlsl.empty())
                {
                    // banjo-nx: as above, for SPIR-V (and the DXIL is the override's: compile both from the original).
                    recompiler.out = std::move(originalHlsl);
                    originalHlsl.clear();
                    banjoOverrideFallbacks.fetch_add(1, std::memory_order_relaxed);
#ifdef XENOS_RECOMP_DXIL
                    shader.dxil = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, recompiler.specConstantsMask != 0, false);
#endif
                    spirv = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true);
                }
#endif
                if (spirv == nullptr)
                {
                    recordFailure("dxc-spirv-compile-failed");
                    return;
                }

                if (!smolv::Encode(spirv->GetBufferPointer(), spirv->GetBufferSize(), shader.spirv, smolv::kEncodeFlagStripDebugInfo))
                {
                    spirv->Release();
                    recordFailure("smolv-encode-failed");
                    return;
                }

                spirv->Release();

#ifdef RENUT_RECOMP
                // banjo-nx (deko3d renderer, docs/deko3d-renderer.md): with BANJO_DEKO3D_SPIRV_DIR, also the SPIR-V the
                // deko3d build converts (tools/deko3d_shaders.py): -DBANJO_DEKO3D, D3D clip space. <dir>/<vs|ps>_<hash>.spv,
                // and <hash>.fail with DXC's reason when it does not compile. Game data: the build tree only.
                if (const char* dekoDir = std::getenv("BANJO_DEKO3D_SPIRV_DIR"); dekoDir && *dekoDir)
                {
                    const std::string stem = fmt::format("{}_{:016X}", recompiler.isPixelShader ? "ps" : "vs", hash);
                    std::error_code ec;
                    std::filesystem::create_directories(dekoDir, ec);
                    IDxcBlob* deko = dxcCompiler.compile(recompiler.out, recompiler.isPixelShader, false, true, true);
                    if (deko != nullptr)
                    {
                        const std::string path = (std::filesystem::path(dekoDir) / (stem + ".spv")).string();
                        writeAllBytes(path.c_str(), deko->GetBufferPointer(), deko->GetBufferSize());
                        deko->Release();
                    }
                    else
                    {
                        const std::string path = (std::filesystem::path(dekoDir) / (stem + ".fail")).string();
                        writeAllBytes(path.c_str(), "dxc", 3);
                    }
                }
#endif

                size_t currentProgress = ++progress;
                if ((currentProgress % 10) == 0 || (currentProgress == shaders.size() - 1))
                    fmt::println("Recompiling shaders... {}%", currentProgress / float(shaders.size()) * 100.0f);
            });

        if (!g_failures.empty())
        {
            fmt::println(stderr, "Recompile failures ({}):", g_failures.size());
            for (const auto& failure : g_failures)
                fmt::println(stderr, "  hash=0x{:016X} reason={}", failure.hash, failure.reason);
            return 2;
        }

#ifdef RENUT_RECOMP
        fmt::println("banjo-nx: {} predicated blocks without a branch, {} memory fetches in them (banjo_pred_select.h)",
                     banjo_pred_select::g_blocks.load(), banjo_pred_select::g_fetches.load());
        fmt::println("banjo-nx: {} bool specialization constants in {} shaders (banjo_bool_spec.h)",
                     banjo_bool_spec::g_constants.load(), banjo_bool_spec::g_shaders.load());
        fmt::println("banjo-nx: {} rewritten shaders (BANJO_HLSL_OVERRIDE) did not compile and kept their own text",
                     banjoOverrideFallbacks.load());
        fmt::println("banjo-nx: {} pixel shaders kill pixels, {} write depth (banjo_gpu_depth_only keeps them)",
                     banjoKillShaders.load(), banjoDepthShaders.load());
#endif
        fmt::println("Creating shader cache...");

        StringBuffer f;
#ifdef REBLUE_RECOMP
        f.println("#include \"gpu/shaders/shader_cache.h\"");
#else
        f.println("#include \"shader_cache.h\"");
#endif
        f.println("ShaderCacheEntry g_shaderCacheEntries[] = {{");

        std::vector<uint8_t> dxil;
        std::vector<uint8_t> spirv;
#ifdef RENUT_RECOMP
        StringBuffer vertexInputs;
        size_t vertexInputCount = 0;
        StringBuffer literals;     // banjo-nx (banjo_gpu_literals_cbuf): g_shaderLiterals
        size_t literalCount = 0;
#endif

        for (auto& [hash, shader] : shaders)
        {
#ifdef RENUT_RECOMP
            const uint32_t* me = shader.memExportConstants;
            const uint32_t* fc = shader.floatConstants;
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {}, 0x{:X}, 0x{:X}, {{ 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X} }}, "
                "{{ 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X} }}, {}, {}, {}, {}, 0x{:X} }},",
                hash, dxil.size(), (shader.dxil != nullptr) ? shader.dxil->GetBufferSize() : 0, spirv.size(), shader.spirv.size(), shader.specConstantsMask,
                shader.vertexFetchMemoryMask, shader.textureFetchMask,
                me[0], me[1], me[2], me[3], me[4], me[5], me[6], me[7],
                fc[0], fc[1], fc[2], fc[3], fc[4], fc[5], fc[6], fc[7],
                vertexInputCount, shader.vertexInputs.size(), literalCount, shader.cbufLiterals.size(),
                shader.shaderFlags);
            for (const auto& l : shader.cbufLiterals)
                literals.println("\t{{ {}, {{ 0x{:X}, 0x{:X}, 0x{:X}, 0x{:X} }} }},", l[0], l[1], l[2], l[3], l[4]);
            literalCount += shader.cbufLiterals.size();
            for (const auto& input : shader.vertexInputs)
            {
                vertexInputs.println("\t{{ {}, {}, {}, {}, {}, {}, {} }},", input.usage, input.usageIndex, input.fetchIndex,
                    input.format, input.flags, input.strideDwords, input.offsetDwords);
            }
            vertexInputCount += shader.vertexInputs.size();
#else
            f.println("\t{{ 0x{:X}, {}, {}, {}, {}, {} }},",
                hash, dxil.size(), (shader.dxil != nullptr) ? shader.dxil->GetBufferSize() : 0, spirv.size(), shader.spirv.size(), shader.specConstantsMask);
#endif

            if (shader.dxil != nullptr)
            {
                dxil.insert(dxil.end(), reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()),
                    reinterpret_cast<uint8_t *>(shader.dxil->GetBufferPointer()) + shader.dxil->GetBufferSize());
            }
            
            spirv.insert(spirv.end(), shader.spirv.begin(), shader.spirv.end());
        }

        f.println("}};");

#ifdef RENUT_RECOMP
        // An empty array is not valid C++; the extra entry is never indexed.
        f.println("ShaderVertexInput g_shaderVertexInputs[] = {{");
        f.out += vertexInputs.out;
        f.println("\t{{ 0, 0, 0, 0, 0, 0, 0 }},");
        f.println("}};");
        // banjo-nx (banjo_gpu_literals_cbuf): the literal registers the shaders read from their constant buffers.
        f.println("ShaderLiteral g_shaderLiterals[] = {{");
        f.out += literals.out;
        f.println("\t{{ 0, {{ 0, 0, 0, 0 }} }},");
        f.println("}};");
#endif

        fmt::println("Compressing DXIL cache...");

        int level = ZSTD_maxCLevel();

#ifdef XENOS_RECOMP_DXIL
        std::vector<uint8_t> dxilCompressed(ZSTD_compressBound(dxil.size()));
        dxilCompressed.resize(ZSTD_compress(dxilCompressed.data(), dxilCompressed.size(), dxil.data(), dxil.size(), level));

        f.print("const uint8_t g_compressedDxilCache[] = {{");

        for (auto data : dxilCompressed)
            f.print("{},", data);

        f.println("}};");
        f.println("const size_t g_dxilCacheCompressedSize = {};", dxilCompressed.size());
        f.println("const size_t g_dxilCacheDecompressedSize = {};", dxil.size());
#endif

        fmt::println("Compressing SPIRV cache...");

        std::vector<uint8_t> spirvCompressed(ZSTD_compressBound(spirv.size()));
        spirvCompressed.resize(ZSTD_compress(spirvCompressed.data(), spirvCompressed.size(), spirv.data(), spirv.size(), level));

        f.print("const uint8_t g_compressedSpirvCache[] = {{");

        for (auto data : spirvCompressed)
            f.print("{},", data);

        f.println("}};");

        f.println("const size_t g_spirvCacheCompressedSize = {};", spirvCompressed.size());
        f.println("const size_t g_spirvCacheDecompressedSize = {};", spirv.size());
        f.println("const size_t g_shaderCacheEntryCount = {};", shaders.size());
#ifdef RENUT_RECOMP
        // Entries carry vertexFetchMemoryMask, textureFetchMask, memExportConstants,
        // floatConstants and their range of g_shaderVertexInputs; 6: shaders with the
        // screen-space viewport epilogue and the rewritten memExport; 7 (banjo-nx, package 28):
        // shaderFlags.
        f.println("extern const uint32_t g_shaderCacheLayout = 7;");
#endif

        writeAllBytes(output, f.out.data(), f.out.size());
    }
    else
    {
        ShaderRecompiler recompiler;
        size_t fileSize;
        recompiler.recompile(readAllBytes(input, fileSize).get(), include);
        writeAllBytes(output, recompiler.out.data(), recompiler.out.size());
    }

    return 0;
}
