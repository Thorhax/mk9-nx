/* 64-bit FNV-1a, for tools/build_asset_pack.py (ctypes): the asset pack's data checksum covers gigabytes, which
 * pure Python hashes at about 1 MB/s. Built on first use with the project's clang into out/tools/. */
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
__declspec(dllexport)
#endif
uint64_t fnv64(uint64_t hash, const uint8_t* data, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}
