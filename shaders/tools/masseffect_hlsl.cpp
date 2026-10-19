/**
 * @file    masseffect_hlsl.cpp
 * @brief   Command line front end of the Xbox 360 shader translator (XenosRecomp) for Mass Effect.
 *
 * XenosRecomp's own main.cpp is not used. It does three things: translate to HLSL, compile to
 * DXIL/SPIR-V through the DXC library, and package the result with smol-v and zstd. Only the first is
 * needed here: DXC is run as a separate program (shaders/tools/compile_spirv_all.sh) and the packaging is our
 * own format (shaders/tools/me_pack_shaders.cpp).
 *
 * Input:  a folder of original shader containers (*.bin, big-endian, signature 0x102A11xx, the layout
 *         of the 2008 Xbox 360 XDK), as collected by the game's shader dump or ue3_shader_scan.
 * Output: one <name>.hlsl per container. The output folder must be new or empty.
 *
 * Usage
 *   masseffect_hlsl <input folder> <output folder> <shader_common.h>
 *
 * Exit status: 0 if every container was translated, 2 if some were skipped, 1 on a usage/IO error.
 */

#include "XenosRecomp/shader_recompiler.h"

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

// The container signature is big-endian: 0x102A11xx, where bit 0 of the low byte is the vertex
// shader flag. The low byte belongs to the compiler tool, so it is not compared.
bool ValidSignature(uint32_t flags_be) {
  return (__builtin_bswap32(flags_be) & 0xFFFFFF00u) == 0x102A1100u;
}

}  // namespace

int main(int argc, char** argv) try {
  if (argc < 4) {
    std::printf("usage: masseffect_hlsl <input folder> <output folder> <shader_common.h>\n");
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

  size_t total = 0, ok = 0, skipped = 0;
  for (const auto& e : std::filesystem::directory_iterator(input)) {
    if (!e.is_regular_file() || e.path().extension() != ".bin") continue;
    const std::string name = e.path().filename().string();
    const std::vector<uint8_t> data = ReadAll(e.path());
    ++total;
    if (data.size() < 24) {
      std::printf("  %s: unreadable file or truncated header\n", name.c_str());
      ++skipped;
      continue;
    }
    uint32_t flags_be = 0;
    std::memcpy(&flags_be, data.data(), 4);
    if (!ValidSignature(flags_be)) {
      std::printf("  %-20s unknown signature 0x%08X\n", name.c_str(), __builtin_bswap32(flags_be));
      ++skipped;
      continue;
    }

    std::printf("  %s\n", name.c_str());
    std::fflush(stdout);
    ShaderRecompiler recompiler;
    try {
      recompiler.recompile(data.data(), include);
    } catch (const std::exception& error) {
      std::printf("  %s: translation rejected: %s\n", name.c_str(), error.what());
      ++skipped;
      continue;
    }
    if (recompiler.out.empty()) {
      std::printf("  %-20s produced nothing\n", name.c_str());
      ++skipped;
      continue;
    }

    auto target = output / e.path().filename();
    target.replace_extension(".hlsl");
    FILE* f = std::fopen(target.string().c_str(), "wb");
    if (f) {
      const bool complete = std::fwrite(recompiler.out.data(), 1, recompiler.out.size(), f) == recompiler.out.size();
      const bool closed = std::fclose(f) == 0;
      if (complete && closed) ++ok;
      else ++skipped;
    } else {
      ++skipped;
    }
  }

  std::printf("\n%zu shaders: %zu translated, %zu skipped\n", total, ok, skipped);
  return total > 0 && ok == total ? 0 : 2;
} catch (const std::exception& error) {
  std::fprintf(stderr, "error: %s\n", error.what());
  return 1;
}
