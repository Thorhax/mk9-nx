/**
 * peggle2-nx (2026-10-20): command line front end of Renut's shader translator (app/thirdparty/reblue-XenosRecomp,
 * RENUT_RECOMP) for the web installer's WebAssembly build (wasm/src/deko3d/build_wasm_deko3d.sh in
 * peggle2-nx-installer). The translator's own main.cpp also compiles (DXC) and packages (smol-v, zstd); the page runs
 * DXC as its own module, so this front end only translates.
 *
 * Input:  a folder of shader containers (*.bin, big-endian, signature 0x102A11xx), one container per file.
 * Output: <vs|ps>_<XXH3 of the container, 16 upper-case hex digits>.hlsl per container, the text the translator hands
 *         DXC (its HLSL, shader_common.h first), and <same>.txt with the numbers the NRO's shader table holds for it
 *         (spec constants, fetch masks, memexport and float constant registers, vertex inputs, literals, flags), for
 *         the check against the table built into the NRO. The output folder must be new or empty.
 *
 * What the native build does between the translation and DXC and this one does not: the early alpha pass
 * (tools/renut_early_alpha.py) changes none of the shaders stored in Peggle 2's default.xex (checked 2026-10-20), and
 * the port's own shaders (tools/renut_port_shaders.py) are the ones the game compiles at boot, which the installer
 * never has.
 *
 * Usage:  renut_hlsl <input folder> <output folder> <shader_common.h>
 * Exit status: 0 if every container was translated, 2 if some were not, 1 on a usage or I/O error.
 */
#include "shader_recompiler.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<uint8_t> ReadAll(const std::filesystem::path& path) {
  std::vector<uint8_t> data;
  FILE* f = std::fopen(path.string().c_str(), "rb");
  if (!f) return data;
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n > 0) {
    data.resize(static_cast<size_t>(n));
    if (std::fread(data.data(), 1, data.size(), f) != data.size()) data.clear();
  }
  std::fclose(f);
  return data;
}

bool WriteAll(const std::filesystem::path& path, const std::string& text) {
  FILE* f = std::fopen(path.string().c_str(), "wb");
  if (!f) return false;
  const bool complete = std::fwrite(text.data(), 1, text.size(), f) == text.size();
  return std::fclose(f) == 0 && complete;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 4) {
    std::printf("usage: renut_hlsl <input folder> <output folder> <shader_common.h>\n");
    return 1;
  }
  const std::filesystem::path input = argv[1];
  const std::filesystem::path output = argv[2];
  const std::vector<uint8_t> common = ReadAll(argv[3]);
  if (common.empty()) {
    std::printf("cannot read %s\n", argv[3]);
    return 1;
  }
  const std::string_view include(reinterpret_cast<const char*>(common.data()), common.size());
  std::error_code ec;
  if (std::filesystem::exists(output) && !std::filesystem::is_empty(output)) {
    std::fprintf(stderr, "the output folder must be empty so old results are not mixed in\n");
    return 1;
  }
  std::filesystem::create_directories(output, ec);
  if (ec) throw std::runtime_error("cannot create the output folder");

  size_t total = 0, ok = 0;
  for (const auto& e : std::filesystem::directory_iterator(input)) {
    if (!e.is_regular_file() || e.path().extension() != ".bin") continue;
    ++total;
    const std::vector<uint8_t> data = ReadAll(e.path());
    if (data.size() < sizeof(ShaderContainer)) {
      std::printf("  %s: truncated\n", e.path().filename().string().c_str());
      continue;
    }
    // As the translator's main.cpp: the container's signature, and its hash over its whole extent.
    const auto* container = reinterpret_cast<const ShaderContainer*>(data.data());
    const size_t extent = container->virtualSize + container->physicalSize;
    if ((container->flags & 0xFFFFFF00) != 0x102A1100 || extent > data.size()) {
      std::printf("  %s: not a shader container\n", e.path().filename().string().c_str());
      continue;
    }
    const uint64_t hash = XXH3_64bits(container, extent);

    ShaderRecompiler recompiler;
    try {
      recompiler.recompile(data.data(), include);
    } catch (const std::exception& error) {
      std::printf("  %s: translation rejected: %s\n", e.path().filename().string().c_str(), error.what());
      continue;
    }
    if (recompiler.out.empty()) {
      std::printf("  %s: produced nothing\n", e.path().filename().string().c_str());
      continue;
    }
    const std::string stem = fmt::format("{}_{:016X}", recompiler.isPixelShader ? "ps" : "vs", hash);

    // The numbers of the NRO's shader table (main.cpp's g_shaderCacheEntries line, without the DXIL / SPIR-V offsets).
    std::string numbers = fmt::format("hash 0x{:X}\nspecConstantsMask 0x{:X}\nvertexFetchMemoryMask 0x{:X}\n"
                                      "textureFetchMask 0x{:X}\nshaderFlags 0x{:X}\nmemExportConstants",
                                      hash, recompiler.specConstantsMask, recompiler.vertexFetchMemoryMask,
                                      recompiler.textureFetchMask, recompiler.shaderFlags);
    for (uint32_t m : recompiler.memExportConstants) numbers += fmt::format(" 0x{:X}", m);
    numbers += "\nfloatConstants";
    for (uint32_t m : recompiler.floatConstants) numbers += fmt::format(" 0x{:X}", m);
    numbers += "\n";
    for (const auto& input : recompiler.vertexInputs)
      numbers += fmt::format("vertexInput {} {} {} {} {} {} {}\n", input.usage, input.usageIndex, input.fetchIndex,
                             input.format, input.flags, input.strideDwords, input.offsetDwords);
    for (const auto& l : recompiler.cbufLiterals)
      numbers += fmt::format("literal {} 0x{:X} 0x{:X} 0x{:X} 0x{:X}\n", l[0], l[1], l[2], l[3], l[4]);

    if (WriteAll(output / (stem + ".hlsl"), recompiler.out) && WriteAll(output / (stem + ".txt"), numbers)) {
      std::printf("  %s -> %s\n", e.path().filename().string().c_str(), stem.c_str());
      ++ok;
    }
  }
  std::printf("%zu containers: %zu translated\n", total, ok);
  return total > 0 && ok == total ? 0 : 2;
} catch (const std::exception& error) {
  std::fprintf(stderr, "error: %s\n", error.what());
  return 1;
}
