// sa2_lzx: LZX decompression of an XEX image, for the browser installer (wasm/lzx.mjs). The same library and
// parameters as the ReXGlue runtime's XEX loader: libmspack's lzxd (sdk/thirdparty/libmspack), window of the XEX's
// compression info, 32 KB frames. Same command line as nfsmw-nx's nfsmw_lzx.cpp, so the page code is shared.
//
//   sa2_lzx <input> <new output> <window bits> <output bytes>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mspack.h>
#include <lzx.h>

extern "C" struct mspack_system* mspack_default_system;

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: sa2_lzx <input> <new output> <window bits> <output bytes>\n");
    return 1;
  }
  const int bits = std::atoi(argv[3]);
  const long bytes = std::atol(argv[4]);
  if (bits < 15 || bits > 21 || bytes <= 0 || bytes > 512 * 1024 * 1024 || std::filesystem::exists(argv[2])) {
    return 1;
  }
  mspack_system* system = mspack_default_system;
  mspack_file* input = system->open(system, argv[1], MSPACK_SYS_OPEN_READ);
  if (!input) {
    return 1;
  }
  mspack_file* output = system->open(system, argv[2], MSPACK_SYS_OPEN_WRITE);
  if (!output) {
    system->close(input);
    return 1;
  }
  lzxd_stream* stream = lzxd_init(system, input, output, bits, 0, 0x8000, bytes, 0);
  const int status = stream ? lzxd_decompress(stream, bytes) : -1;
  if (stream) {
    lzxd_free(stream);
  }
  system->close(input);
  system->close(output);
  if (status || std::filesystem::file_size(argv[2]) != size_t(bytes)) {
    std::fprintf(stderr, "LZX failed: %d; the output is not valid\n", status);
    return 2;
  }
  std::printf("LZX: %ld bytes decompressed\n", bytes);
  return 0;
}
