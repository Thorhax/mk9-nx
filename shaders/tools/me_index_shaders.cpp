// U1: writes the index (<package>.idx) of an existing shader package, which the game then loads instead of
// the whole package (LibraryShaders::LoadIndexed). The package is fully loaded and verified first,
// and the index is checked by loading it back.
//   usage: me_index_shaders <package.mesp> [<output .idx>, default <package.mesp>.idx]
#include "../../app/src/me/native/sa2_shader_library.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace fs = std::filesystem;

int main(int argc, char** argv) try {
  if (argc != 2 && argc != 3) throw std::runtime_error("usage: me_index_shaders <package.mesp> [<output.idx>]");
  const fs::path package = argv[1];
  fs::path output = argc == 3 ? fs::path(argv[2]) : fs::path(package);
  if (argc == 2) output += ".idx";
  const auto t0 = std::chrono::steady_clock::now();
  std::ifstream f(package, std::ios::binary | std::ios::ate);
  if (!f) throw std::runtime_error("cannot open the package");
  std::vector<uint8_t> data(static_cast<size_t>(f.tellg()));
  f.seekg(0);
  if (!f.read(reinterpret_cast<char*>(data.data()), data.size())) throw std::runtime_error("short read");
  const auto index = sa2::native::IndexShaders(data);
  const fs::path temporary = fs::path(output) += ".tmp";
  {
    std::ofstream o(temporary, std::ios::binary | std::ios::trunc);
    if (!o.write(reinterpret_cast<const char*>(index.data()), index.size()) || !o.flush())
      throw std::runtime_error("write failed");
  }
  sa2::native::LibraryShaders check;
  check.LoadIndexed(package, temporary);
  fs::rename(temporary, output);
  std::printf("%s: %zu entries, %zu bytes (%.1f s)\n", output.string().c_str(), check.shaders().size(), index.size(),
              std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  return 0;
} catch (const std::exception& e) {
  std::fprintf(stderr, "%s\n", e.what());
  return 1;
}
