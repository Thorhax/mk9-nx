// Writes every original container of a .mesp package to a folder as <vs|ps>_<XXH3>.bin, so
// container-level tools (find_shader_microcode, wrap_raw_shader.py, matchers) can search the
// shipped package after loose container folders are gone. Game data: keep the output local.
//   extract_shader_package package.mesp out_dir
#define XXH_INLINE_ALL
#include <xxhash.h>
#include "../../app/src/me/native/sa2_shader_library.h"
#include <cstdio>
#include <filesystem>
#include <fstream>
int main(int argc, char** argv) {
  if (argc != 3) { std::fprintf(stderr, "usage: %s package.mesp out_dir\n", argv[0]); return 2; }
  sa2::native::LibraryShaders library;
  library.Load(std::filesystem::path(argv[1]));
  if (library.shaders().empty()) { std::fprintf(stderr, "package did not load\n"); return 1; }
  std::filesystem::create_directories(argv[2]);
  size_t n = 0;
  for (const auto& s : library.shaders()) {
    char name[64];
    std::snprintf(name, sizeof(name), "%s_%016llx.bin", s.vertices ? "vs" : "ps",
                  (unsigned long long)XXH3_64bits(s.original.data(), s.original.size()));
    std::ofstream(std::filesystem::path(argv[2]) / name, std::ios::binary)
        .write(reinterpret_cast<const char*>(s.original.data()), std::streamsize(s.original.size()));
    ++n;
  }
  std::printf("%zu containers\n", n);
}
