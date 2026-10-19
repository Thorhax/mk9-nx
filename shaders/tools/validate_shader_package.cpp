// Structural and SPIR-V validation for a complete .mesp shader package.
//
// LibraryShaders checks the package checksum, ordering, container signatures,
// entry fingerprints and basic SPIR-V framing. This tool additionally runs every
// module through SPIRV-Tools using the Vulkan 1.3 environment used by the host
// renderer. It intentionally validates the package as shipped rather than the
// loose intermediate files.
#include "../../app/src/me/native/sa2_shader_library.h"

#include <spirv-tools/libspirv.hpp>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>

#define XXH_INLINE_ALL
#include <xxhash.h>

static uint32_t Be32(std::span<const uint8_t> bytes, size_t offset) {
  return uint32_t(bytes[offset]) << 24 | uint32_t(bytes[offset + 1]) << 16 |
         uint32_t(bytes[offset + 2]) << 8 | bytes[offset + 3];
}

static uint64_t MicrocodeFingerprint(const sa2::native::Shader& shader) {
  const auto& bytes = shader.original;
  const uint32_t virtual_size = Be32(bytes, 4);
  const uint32_t header = Be32(bytes, 24);
  const uint32_t physical_offset = Be32(bytes, header);
  const uint32_t size = Be32(bytes, header + 4);
  std::vector<uint32_t> words(size / 4);
  for (size_t i = 0; i < words.size(); ++i)
    words[i] = Be32(bytes, size_t(virtual_size) + physical_offset + i * 4);
  return XXH3_64bits(words.data(), words.size() * sizeof(uint32_t));
}

int main(int argc, char** argv) try {
  const bool extract = argc == 5 && std::string_view(argv[2]) == "--extract-entry";
  const bool find_fingerprint = argc == 4 && std::string_view(argv[2]) == "--find-fingerprint";
  const bool find_microcode = argc == 4 && std::string_view(argv[2]) == "--find-microcode";
  if (argc != 2 && !extract && !find_fingerprint && !find_microcode) {
    std::fprintf(stderr, "usage: validate_shader_package <package.mesp> "
                         "[--extract-entry <index> <output-prefix> | "
                         "--find-fingerprint <hex> | --find-microcode <hex>]\n");
    return 2;
  }

  sa2::native::LibraryShaders library;
  library.Load(std::filesystem::path(argv[1]));
  if (find_microcode) {
    const uint64_t fingerprint = std::stoull(argv[3], nullptr, 16);
    for (size_t index = 0; index < library.shaders().size(); ++index) {
      const auto& shader = library.shaders()[index];
      if (MicrocodeFingerprint(shader) != fingerprint) continue;
      std::printf("entry=%zu type=%s container=%016llX microcode=%016llX original=%zu spirv=%zu\n",
                  index, shader.vertices ? "VS" : "PS",
                  static_cast<unsigned long long>(shader.fingerprint),
                  static_cast<unsigned long long>(fingerprint), shader.original.size(),
                  shader.spirv.size() * sizeof(uint32_t));
    }
    return 0;
  }
  if (find_fingerprint) {
    const uint64_t fingerprint = std::stoull(argv[3], nullptr, 16);
    for (size_t index = 0; index < library.shaders().size(); ++index) {
      const auto& shader = library.shaders()[index];
      if (shader.fingerprint != fingerprint) continue;
      std::printf("entry=%zu type=%s fingerprint=%016llX original=%zu spirv=%zu\n", index,
                  shader.vertices ? "VS" : "PS",
                  static_cast<unsigned long long>(shader.fingerprint), shader.original.size(),
                  shader.spirv.size() * sizeof(uint32_t));
      return 0;
    }
    std::fprintf(stderr, "fingerprint not found: %016llX\n",
                 static_cast<unsigned long long>(fingerprint));
    return 1;
  }
  if (extract) {
    const size_t index = std::stoull(argv[3]);
    if (index >= library.shaders().size()) {
      throw std::runtime_error("entry index out of range");
    }
    const auto& shader = library.shaders()[index];
    const std::filesystem::path prefix = argv[4];
    std::ofstream original(prefix.string() + ".bin", std::ios::binary);
    std::ofstream spirv(prefix.string() + ".spv", std::ios::binary);
    original.write(reinterpret_cast<const char*>(shader.original.data()), shader.original.size());
    spirv.write(reinterpret_cast<const char*>(shader.spirv.data()),
                shader.spirv.size() * sizeof(uint32_t));
    if (!original || !spirv) {
      throw std::runtime_error("failed to write extracted entry");
    }
    std::printf("entry=%zu type=%s fingerprint=%016llX original=%zu spirv=%zu\n", index,
                shader.vertices ? "VS" : "PS", static_cast<unsigned long long>(shader.fingerprint),
                shader.original.size(), shader.spirv.size() * sizeof(uint32_t));
    return 0;
  }

  spvtools::SpirvTools validator(SPV_ENV_VULKAN_1_3);
  std::string diagnostic;
  validator.SetMessageConsumer([&](spv_message_level_t, const char*, const spv_position_t& position,
                                   const char* message) {
    diagnostic = "word " + std::to_string(position.index) + ": " + message;
  });

  size_t vertex = 0;
  size_t pixel = 0;
  size_t invalid = 0;
  uint64_t original_bytes = 0;
  uint64_t spirv_bytes = 0;
  for (size_t i = 0; i < library.shaders().size(); ++i) {
    const auto& shader = library.shaders()[i];
    shader.vertices ? ++vertex : ++pixel;
    original_bytes += shader.original.size();
    spirv_bytes += shader.spirv.size() * sizeof(uint32_t);
    diagnostic.clear();
    if (!validator.Validate(shader.spirv.data(), shader.spirv.size())) {
      ++invalid;
      std::fprintf(stderr, "entry %zu (%s, %016llx): %s\n", i,
                   shader.vertices ? "VS" : "PS",
                   static_cast<unsigned long long>(shader.fingerprint), diagnostic.c_str());
    }
  }

  std::printf("package=%s shaders=%zu vertex=%zu pixel=%zu original_bytes=%llu "
              "spirv_bytes=%llu invalid=%zu\n",
              argv[1], library.shaders().size(), vertex, pixel,
              static_cast<unsigned long long>(original_bytes),
              static_cast<unsigned long long>(spirv_bytes), invalid);
  return invalid ? 1 : 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
