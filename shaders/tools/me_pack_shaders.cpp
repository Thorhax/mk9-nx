// Pack the collected shader containers and their SPIR-V into a shader library, with
// the library format of app/src/me/native/sa2_shader_library.cpp (XXH3 of the whole
// original container as the key; the format is described in shaders/README.md). The containers are
// the 2008 layout and are packed as they are.
//   usage: me_pack_shaders <containers> <validated spirv> <new output file>
//          me_pack_shaders --replace <base package> <containers> <validated spirv> <new output file>
// It also writes <new output file>.idx, the index the game loads instead of the whole package (U1;
// shaders/tools/me_index_shaders makes it for an existing package).
#include "../../app/src/me/native/sa2_shader_library.h"  // banjo-nx path

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <span>

namespace fs = std::filesystem;

static std::vector<uint8_t> Read(const fs::path& path) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f || f.tellg() < 0 || f.tellg() > 64 * 1024 * 1024)
    throw std::runtime_error("unreadable input: " + path.string());
  std::vector<uint8_t> d(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(d.data()), d.size())) throw std::runtime_error("short read");
  return d;
}

static uint32_t Be32(std::span<const uint8_t> d, size_t offset) {
  if (offset + 4 > d.size()) throw std::runtime_error("truncated shader container");
  return uint32_t(d[offset]) << 24 | uint32_t(d[offset + 1]) << 16 |
         uint32_t(d[offset + 2]) << 8 | uint32_t(d[offset + 3]);
}

static std::span<const uint8_t> Physical(const std::vector<uint8_t>& container) {
  const uint32_t virtual_size = Be32(container, 4);
  const uint32_t physical_size = Be32(container, 8);
  if (virtual_size > container.size() || physical_size > container.size() - virtual_size)
    throw std::runtime_error("invalid shader container sizes");
  return std::span<const uint8_t>(container).subspan(virtual_size, physical_size);
}

int main(int argc, char** argv) try {
  const bool replace_exact = argc == 6 && std::string_view(argv[1]) == "--replace-exact";
  const bool replace = argc == 6 && (std::string_view(argv[1]) == "--replace" || replace_exact);
  if (argc != 4 && !replace)
    throw std::runtime_error("usage: me_pack_shaders [--replace <base>] <containers> <spirv> <output>");
  const fs::path base = replace ? argv[2] : fs::path{};
  const fs::path containers = argv[replace ? 3 : 1];
  const fs::path spirv = argv[replace ? 4 : 2];
  const fs::path output = argv[replace ? 5 : 3];
  if (fs::exists(output)) throw std::runtime_error("output exists");
  std::vector<fs::path> paths;
  for (const auto& e : fs::directory_iterator(containers))
    if (e.is_regular_file() && e.path().extension() == ".bin") paths.push_back(e.path());
  std::sort(paths.begin(), paths.end());
  std::vector<sa2::native::Shader> replacements;
  std::vector<fs::path> packed;
  size_t missing = 0;
  for (const auto& path : paths) {
    const fs::path spv_path = spirv / (path.stem().string() + ".spv");
    if (!fs::exists(spv_path)) {  // translation or DXC failed for it (listed by the build)
      ++missing;
      continue;
    }
    packed.push_back(path);
    sa2::native::Shader s;
    s.original = Read(path);
    auto spv = Read(spv_path);
    if (spv.size() % 4) throw std::runtime_error("misaligned SPIR-V");
    for (size_t i = 0; i < spv.size(); i += 4)
      s.spirv.push_back(uint32_t(spv[i]) | uint32_t(spv[i + 1]) << 8 | uint32_t(spv[i + 2]) << 16 |
                        uint32_t(spv[i + 3]) << 24);
    replacements.push_back(std::move(s));
  }
  std::vector<sa2::native::Shader> shaders;
  size_t removed = 0;
  if (replace) {
    sa2::native::LibraryShaders old;
    old.Load(base);
    for (const auto& shader : old.shaders()) {
      const auto physical = Physical(shader.original);
      const bool superseded = std::any_of(replacements.begin(), replacements.end(), [&](const auto& newer) {
        if (replace_exact) return shader.original == newer.original;
        const auto candidate = Physical(newer.original);
        return physical.size() == candidate.size() && std::equal(physical.begin(), physical.end(), candidate.begin());
      });
      if (superseded) ++removed;
      else shaders.push_back(shader);
    }
  }
  shaders.insert(shaders.end(), std::make_move_iterator(replacements.begin()),
                 std::make_move_iterator(replacements.end()));
  auto package = sa2::native::PackShaders(std::move(shaders));
  sa2::native::LibraryShaders library;
  library.Load(package);
  for (const auto& path : packed)
    if (!library.Search(Read(path))) throw std::runtime_error("a shader is missing after packing");
  std::ofstream f(output, std::ios::binary);
  if (!f.write(reinterpret_cast<const char*>(package.data()), package.size()) || !f.flush())
    throw std::runtime_error("write failed");
  const auto index = sa2::native::IndexShaders(package);
  fs::path index_path = output;
  index_path += ".idx";
  std::ofstream fi(index_path, std::ios::binary | std::ios::trunc);
  if (!fi.write(reinterpret_cast<const char*>(index.data()), index.size()) || !fi.flush())
    throw std::runtime_error("index write failed");
  std::printf("index %s: %zu bytes\n", index_path.string().c_str(), index.size());
  std::printf("%zu containers, %zu without SPIR-V, %zu old entries replaced -> %zu shaders, %zu bytes; "
              "all packed ones found\n", paths.size(), missing, removed, library.shaders().size(), package.size());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
