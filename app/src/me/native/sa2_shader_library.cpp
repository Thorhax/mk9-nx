#include "sa2_shader_library.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
/*
 * xxHash reads with memcpy (XXH_FORCE_MEMORY_ACCESS 0). With the method it picks for GCC (1) it reads
 * through 64- and 32-bit pointers without may_alias, and GCC may hoist that read above the write of the
 * data being hashed (strict aliasing). That made the texture key read keys[4] before writing it, and
 * the same texture was created several times. Same fingerprint values; on AArch64, the same LDR.
 */
#if defined(XXH_IMPLEM_13a8737387)
#error "xxhash.h was already included with its implementation before this point: XXH_FORCE_MEMORY_ACCESS 0 would come too late"
#endif
#undef XXH_FORCE_MEMORY_ACCESS
#define XXH_FORCE_MEMORY_ACCESS 0
#define XXH_INLINE_ALL
#include <xxhash.h>

namespace sa2::native {

// The package of an indexed load, kept open for the lazy SPIR-V reads.
struct FileShaders {
  std::filesystem::path path;
  std::FILE* principal = nullptr;  // only under `read`
  std::mutex read;              // on-demand reads (Shader::Spirv): one entry at a time
  std::mutex publication;          // held only to publish one entry (never across a read)
  ~FileShaders() {
    if (principal) std::fclose(principal);
  }
};

namespace {
constexpr std::array<uint8_t, 8> kSignature{'M','E','S','S','P','V',0,0};
// Mass Effect has 30,191 shader containers; the complete library is ~700 MB of SPIR-V.
constexpr size_t kMaxFile = size_t(1) << 30;
constexpr size_t kMaxShaders = 65536;
constexpr size_t kMaxOriginal = 64 * 1024;
constexpr size_t kMaxSpirv = 4 * 1024 * 1024;

/*
 * Index of a package (<package>.idx), little-endian:
 *   header (56 bytes): "MESSIDX\0", u32 version (1), u32 count, u64 package size, u64 package checksum
 *     (the XXH3 of the package body stored at +16 in its header), u64 bytes of originals, u64 XXH3 of
 *     everything after this header, u64 reserved (0).
 *   count entries of 40 bytes, in package order: u32 original bytes, u32 SPIR-V words, u64 container
 *     XXH3 (fingerprint), u64 SPIR-V XXH3, u32 kills (CountKillsSpirv), u32 flags (reserved, 0),
 *     u64 offset of the entry in the package (its 16-byte header).
 *   the original containers, concatenated in the same order.
 * The index is generated from a fully loaded and verified package and bound to it by size and checksum;
 * every lazily read entry is checked again against it (header, container bytes, SPIR-V hash).
 */
constexpr std::array<uint8_t, 8> kSignatureIndex{'M','E','S','S','I','D','X',0};
constexpr uint32_t kVersionIndex = 1;
constexpr size_t kHeaderIndex = 56;
constexpr size_t kEntryIndex = 40;
constexpr size_t kMaxIndex = 64 + kMaxShaders * kEntryIndex + size_t(256) * 1024 * 1024;

void Require(bool condition, const char* error) {
  if (!condition) throw std::runtime_error(error);
}

uint32_t BE(const uint8_t* p) {
  return uint32_t(p[0]) << 24 | uint32_t(p[1]) << 16 |
         uint32_t(p[2]) << 8 | p[3];
}

uint32_t LE32(const uint8_t* p) {
  return uint32_t(p[0]) | uint32_t(p[1]) << 8 | uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
}
uint64_t LE64(const uint8_t* p) { return LE32(p) | uint64_t(LE32(p + 4)) << 32; }

struct Reader {
  std::span<const uint8_t> data;
  size_t position = 0;
  std::span<const uint8_t> Take(size_t n) {
    Require(n <= data.size() - position, "Shader package truncated");
    auto r = data.subspan(position, n);
    position += n;
    return r;
  }
  uint32_t U32() {
    auto p = Take(4);
    return uint32_t(p[0]) | uint32_t(p[1]) << 8 |
           uint32_t(p[2]) << 16 | uint32_t(p[3]) << 24;
  }
  uint64_t U64() {
    const uint64_t low = U32();
    return low | uint64_t(U32()) << 32;
  }
};

void U32(std::vector<uint8_t>& d, uint32_t v) {
  for (unsigned i = 0; i < 4; ++i) d.push_back(uint8_t(v >> (i * 8)));
}
void U64(std::vector<uint8_t>& d, uint64_t v) {
  U32(d, uint32_t(v)); U32(d, uint32_t(v >> 32));
}

// Signature, lengths and fingerprint of the original container; sets vertices and fingerprint.
void ValidateContainer(Shader& s) {
  Require(s.original.size() >= 24 && s.original.size() <= kMaxOriginal,
         "Container length out of bounds");
  const uint32_t signature = BE(s.original.data());
  // Two container layouts are accepted (0x102A0Exx and 0x102A11xx); bit 0 is the vertex shader flag in both.
  Require((signature & ~1u) == 0x102A0E00 || (signature & 0xFFFFFF00u) == 0x102A1100,
         "Unknown container signature");
  const uint32_t virtual_value = BE(s.original.data() + 4);
  const uint32_t physical = BE(s.original.data() + 8);
  // In the 0x102A11xx layout the physical part is not only 12-byte microcode words
  // (Mass Effect has 844- and 856-byte ones); the shader header gives the microcode size.
  const bool is2008 = (signature & 0xFFFFFF00u) == 0x102A1100;
  Require(virtual_value >= 24 && physical && (is2008 || !(physical % 12)) &&
             uint64_t(virtual_value) + physical == s.original.size(),
         "Inconsistent container lengths");
  s.vertices = (signature & 1) != 0;
  s.fingerprint = XXH3_64bits(s.original.data(), s.original.size());
}

// The SPIR-V wrapper: header, stage and main entry point matching the container.
void ValidateSpirv(std::span<const uint32_t> spirv, bool vertices) {
  Require(spirv.size() >= 5 && spirv.size() <= kMaxSpirv / 4,
         "SPIR-V length out of bounds");
  Require(spirv[0] == 0x07230203 && spirv[4] == 0,
         "Wrong SPIR-V header");
  // This checks the wrapper; it does not replace spirv-val in the packager.
  bool entry = false;
  for (size_t i = 5; i < spirv.size();) {
    const uint32_t words = spirv[i] >> 16, op = spirv[i] & 65535;
    Require(words && words <= spirv.size() - i, "SPIR-V instruction truncated");
    if (op == 15) {  // OpEntryPoint: execution model, id, name and interface.
      Require(!entry && words >= 5 && spirv[i+1] == (vertices ? 0u : 4u) &&
                 spirv[i+3] == 0x6e69616d && spirv[i+4] == 0,
             "SPIR-V stage or main entry point does not match the container");
      entry = true;
    }
    i += words;
  }
  Require(entry, "SPIR-V has no main entry point");
}

void Validate(Shader& s) {
  ValidateContainer(s);
  ValidateSpirv(s.spirv, s.vertices);
}

bool Smaller(const Shader& a, const Shader& b) {
  if (a.fingerprint != b.fingerprint) return a.fingerprint < b.fingerprint;
  return a.original < b.original;
}

void WordsLE(const uint8_t* bytes, size_t words, std::vector<uint32_t>& output) {
  output.resize(words);
  if constexpr (std::endian::native == std::endian::little) {
    std::memcpy(output.data(), bytes, words * 4);
  } else {
    for (size_t j = 0; j < words; ++j) output[j] = LE32(bytes + j * 4);
  }
}

bool ExactRead(std::FILE* f, uint64_t displacement, uint8_t* target, size_t bytes) {
  if (std::fseek(f, long(displacement), SEEK_SET) != 0) return false;
  return std::fread(target, 1, bytes, f) == bytes;
}

// One entry of an indexed package: read, compared with the index and validated. nullptr if correct.
const char* ReadEntry(std::FILE* f, const Shader& s, std::vector<uint8_t>& temporal,
                        std::vector<uint32_t>& output) {
  const size_t o = s.original.size();
  temporal.resize(16 + o + size_t(s.words) * 4);
  if (!ExactRead(f, s.displacement, temporal.data(), temporal.size())) {
    return "reading the shader package failed";
  }
  const uint8_t* p = temporal.data();
  if (LE32(p) != o || LE32(p + 4) != s.words || LE64(p + 8) != s.fingerprint ||
      std::memcmp(p + 16, s.original.data(), o) != 0) {
    return "package entry does not match the index";
  }
  const uint8_t* code = p + 16 + o;
  if (XXH3_64bits(code, size_t(s.words) * 4) != s.spirv_fingerprint) {
    return "package SPIR-V was altered (fingerprint differs from the index)";
  }
  WordsLE(code, s.words, output);
  try {
    ValidateSpirv(output, s.vertices);
  } catch (const std::exception&) {
    return "package SPIR-V is not valid";
  }
  return nullptr;
}

void Publish(FileShaders& a, const Shader& s, std::vector<uint32_t>&& spirv) {
  std::lock_guard<std::mutex> l(a.publication);
  if (s.ready.v.load(std::memory_order_relaxed)) return;
  s.spirv_lazy = std::move(spirv);
  s.ready.v.store(true, std::memory_order_release);
}
}  // namespace

const std::vector<uint32_t>& Shader::Spirv() const {
  if (!file || ready.v.load(std::memory_order_acquire)) return file ? spirv_lazy : spirv;
  std::lock_guard<std::mutex> l(file->read);
  if (ready.v.load(std::memory_order_acquire)) return spirv_lazy;  // the prefetch published it meanwhile
  std::vector<uint8_t> temporal;
  std::vector<uint32_t> data;
  if (const char* error = ReadEntry(file->principal, *this, temporal, data)) {
    std::fprintf(stderr, "[sa2] shader library %s: entry %016llX: %s\n", file->path.string().c_str(),
                 static_cast<unsigned long long>(fingerprint), error);
    std::fflush(stderr);
    std::abort();
  }
  Publish(*file, *this, std::move(data));
  return spirv_lazy;
}

uint32_t CountKillsSpirv(std::span<const uint32_t> spirv) {
  if (spirv.size() < 5 || spirv[0] != 0x07230203u) {
    return 2;
  }
  uint32_t kills = 0;
  for (size_t i = 5; i < spirv.size();) {
    const uint32_t words = spirv[i] >> 16;
    const uint32_t code = spirv[i] & 0xFFFF;
    if (words == 0 || i + words > spirv.size()) {
      return 2;
    }
    if (code == 252 || code == 4416 || code == 5380) {
      ++kills;
    }
    i += words;
  }
  return kills;
}

LibraryShaders::LibraryShaders() = default;
LibraryShaders::~LibraryShaders() = default;

void LibraryShaders::Load(std::span<const uint8_t> file) {
  Require(file.size() >= 24 && file.size() <= kMaxFile,
         "Shader package size out of bounds");
  Reader l{file};
  auto signature = l.Take(8);
  Require(std::equal(signature.begin(), signature.end(), kSignature.begin()), "Unknown package signature");
  Require(l.U32() == 1, "Unsupported package version");
  const uint32_t count = l.U32();
  Require(count && count <= kMaxShaders, "Shader count out of bounds");
  const uint64_t fingerprint = l.U64();
  Require(XXH3_64bits(file.data() + 24, file.size() - 24) == fingerprint,
         "Shader package altered");
  std::vector<Shader> new_values;
  new_values.reserve(count);
  for (uint32_t i = 0; i < count; ++i) {
    const uint32_t original = l.U32(), words = l.U32();
    const uint64_t expected = l.U64();
    Require(original >= 24 && original <= kMaxOriginal &&
               words >= 5 && words <= kMaxSpirv / 4, "Entry too large");
    Shader s;
    auto data = l.Take(original);
    s.original.assign(data.begin(), data.end());
    // Reading the whole range first avoids allocations with a truncated file.
    auto code = l.Take(size_t(words) * 4);
    WordsLE(code.data(), words, s.spirv);
    Validate(s);
    Require(s.fingerprint == expected, "Wrong container fingerprint");
    if (!new_values.empty()) Require(Smaller(new_values.back(), s), "Duplicate or unsorted entries");
    new_values.push_back(std::move(s));
  }
  Require(l.position == file.size(), "Trailing data in the shader package");
  shaders_ = std::move(new_values);
  file_.reset();
}

void LibraryShaders::Load(const std::filesystem::path& file) {
  std::ifstream f(file, std::ios::binary | std::ios::ate);
  Require(bool(f), "Could not open the shader package");
  const auto n = f.tellg();
  Require(n >= 24 && n <= std::streamoff(kMaxFile), "Package size out of bounds");
  std::vector<uint8_t> data(static_cast<size_t>(n));
  f.seekg(0);
  Require(bool(f.read(reinterpret_cast<char*>(data.data()), data.size())), "Incomplete package read");
  Load(data);
}

void LibraryShaders::LoadIndexed(const std::filesystem::path& package,
                                       const std::filesystem::path& index) {
  auto new_value = std::make_unique<FileShaders>();
  new_value->path = package;
  new_value->principal = std::fopen(package.string().c_str(), "rb");
  Require(new_value->principal != nullptr, "Could not open the shader package");
  std::FILE* p = new_value->principal;
  Require(std::fseek(p, 0, SEEK_END) == 0, "Could not measure the shader package");
  const long package_size = std::ftell(p);
  Require(package_size >= 24 && uint64_t(package_size) <= kMaxFile,
         "Package size out of bounds");
  std::array<uint8_t, 24> header_package;
  Require(ExactRead(p, 0, header_package.data(), header_package.size()), "Incomplete package read");
  Require(std::equal(kSignature.begin(), kSignature.end(), header_package.begin()), "Unknown package signature");
  Require(LE32(header_package.data() + 8) == 1, "Unsupported package version");
  const uint32_t count = LE32(header_package.data() + 12);
  Require(count && count <= kMaxShaders, "Shader count out of bounds");

  // The whole index in one sequential read.
  std::vector<uint8_t> data;
  {
    std::FILE* fi = std::fopen(index.string().c_str(), "rb");
    Require(fi != nullptr, "No index for the shader package");
    const bool measured = std::fseek(fi, 0, SEEK_END) == 0;
    const long n = measured ? std::ftell(fi) : -1;
    const bool valid = n >= long(kHeaderIndex) && uint64_t(n) <= kMaxIndex;
    if (valid) {
      data.resize(size_t(n));
      if (!ExactRead(fi, 0, data.data(), data.size())) data.clear();
    }
    std::fclose(fi);
    Require(valid, "Shader index size out of bounds");
    Require(!data.empty(), "Incomplete shader index read");
  }
  const uint8_t* c = data.data();
  Require(std::equal(kSignatureIndex.begin(), kSignatureIndex.end(), c), "Unknown index signature");
  Require(LE32(c + 8) == kVersionIndex, "Unsupported index version");
  Require(LE32(c + 12) == count, "The index belongs to another package (count)");
  Require(LE64(c + 16) == uint64_t(package_size), "The index belongs to another package (size)");
  Require(LE64(c + 24) == LE64(header_package.data() + 16), "The index belongs to another package (fingerprint)");
  const uint64_t bytes_original = LE64(c + 32);
  Require(data.size() == kHeaderIndex + size_t(count) * kEntryIndex + bytes_original,
         "Shader index truncated");
  Require(XXH3_64bits(c + kHeaderIndex, data.size() - kHeaderIndex) == LE64(c + 40),
         "Shader index altered");

  std::vector<Shader> new_values;
  new_values.reserve(count);
  const uint8_t* originals = c + kHeaderIndex + size_t(count) * kEntryIndex;
  uint64_t position = 24, consumed = 0;
  for (uint32_t i = 0; i < count; ++i) {
    const uint8_t* e = c + kHeaderIndex + size_t(i) * kEntryIndex;
    const uint32_t original = LE32(e), words = LE32(e + 4);
    Require(original >= 24 && original <= kMaxOriginal &&
               words >= 5 && words <= kMaxSpirv / 4, "Entry too large");
    Require(original <= bytes_original - consumed, "Shader index truncated");
    Require(LE64(e + 32) == position, "Inconsistent entry offset");
    Shader s;
    s.original.assign(originals + consumed, originals + consumed + original);
    consumed += original;
    ValidateContainer(s);
    Require(s.fingerprint == LE64(e + 8), "Wrong container fingerprint");
    if (!new_values.empty()) Require(Smaller(new_values.back(), s), "Duplicate or unsorted entries");
    s.file = new_value.get();
    s.displacement = position;
    s.words = words;
    s.spirv_fingerprint = LE64(e + 16);
    s.kills = LE32(e + 24);
    position += 16 + uint64_t(original) + uint64_t(words) * 4;
    Require(position <= uint64_t(package_size), "Shader package truncated");
    new_values.push_back(std::move(s));
  }
  Require(consumed == bytes_original, "Trailing data in the shader index");
  Require(position == uint64_t(package_size), "Trailing data in the shader package");
  shaders_ = std::move(new_values);
  file_ = std::move(new_value);
}

const Shader* LibraryShaders::Search(std::span<const uint8_t> original) const {
  if (original.size() < 24 || original.size() > kMaxOriginal) return nullptr;
  const uint64_t fingerprint = XXH3_64bits(original.data(), original.size());
  auto it = std::lower_bound(shaders_.begin(), shaders_.end(), fingerprint,
      [](const Shader& s, uint64_t h) { return s.fingerprint < h; });
  // A hash collision can never pick another shader: the bytes are compared.
  for (; it != shaders_.end() && it->fingerprint == fingerprint; ++it)
    if (it->original.size() == original.size() &&
        std::equal(original.begin(), original.end(), it->original.begin())) return &*it;
  return nullptr;
}

StatsPreload LibraryShaders::Preload(std::vector<const Shader*> shaders,
                                                  const std::atomic<bool>* stop) const {
  StatsPreload r;
  if (!file_) return r;
  FileShaders* const a = file_.get();
  std::erase_if(shaders, [a](const Shader* s) { return !s || s->file != a; });
  std::sort(shaders.begin(), shaders.end(),
            [](const Shader* x, const Shader* y) { return x->displacement < y->displacement; });
  shaders.erase(std::unique(shaders.begin(), shaders.end()), shaders.end());
  std::FILE* f = std::fopen(a->path.string().c_str(), "rb");  // its own handle: never takes `read`
  if (!f) {
    r.error = true;
    return r;
  }
  std::vector<uint8_t> temporal;
  for (const Shader* s : shaders) {
    if (stop && stop->load(std::memory_order_relaxed)) break;
    if (s->ready.v.load(std::memory_order_acquire)) {
      ++r.already_resident;
      continue;
    }
    std::vector<uint32_t> data;
    if (ReadEntry(f, *s, temporal, data)) {
      r.error = true;  // Spirv() will report it (and stop) if that entry is ever used
      break;
    }
    ++r.inputs;
    r.bytes += uint64_t(s->words) * 4;
    Publish(*a, *s, std::move(data));
  }
  std::fclose(f);
  return r;
}

std::vector<uint8_t> PackShaders(std::vector<Shader> shaders) {
  Require(!shaders.empty() && shaders.size() <= kMaxShaders, "Shader count out of bounds");
  for (auto& s : shaders) {
    if (s.indexed()) {  // from an indexed library: the SPIR-V is read now
      s.spirv = s.Spirv();
      s.file = nullptr;
    }
    Validate(s);
  }
  std::sort(shaders.begin(), shaders.end(), Smaller);
  std::vector<uint8_t> body;
  uint32_t count = 0;
  const Shader* previous = nullptr;
  for (const auto& s : shaders) {
    if (previous && previous->original == s.original) {
      Require(previous->spirv == s.spirv, "One container has two different translations");
      continue;
    }
    const size_t n = 16 + s.original.size() + s.spirv.size() * 4;
    Require(n <= kMaxFile - 24 - body.size(), "Package too large");
    U32(body, uint32_t(s.original.size())); U32(body, uint32_t(s.spirv.size()));
    U64(body, s.fingerprint);
    body.insert(body.end(), s.original.begin(), s.original.end());
    for (uint32_t word : s.spirv) U32(body, word);
    ++count;
    previous = &s;
  }
  std::vector<uint8_t> file(kSignature.begin(), kSignature.end());
  U32(file, 1); U32(file, count);
  U64(file, XXH3_64bits(body.data(), body.size()));
  file.insert(file.end(), body.begin(), body.end());
  return file;
}

std::vector<uint8_t> IndexShaders(std::span<const uint8_t> package) {
  LibraryShaders complete;
  complete.Load(package);  // verifies checksum, order and every entry
  const auto& shaders = complete.shaders();
  std::vector<uint8_t> table, original;
  table.reserve(shaders.size() * kEntryIndex);
  uint64_t position = 24;
  for (const Shader& s : shaders) {
    U32(table, uint32_t(s.original.size()));
    U32(table, uint32_t(s.spirv.size()));
    U64(table, s.fingerprint);
    // The bytes as stored in the package (little-endian words), as ReadEntry hashes them.
    std::vector<uint8_t> code;
    code.reserve(s.spirv.size() * 4);
    for (uint32_t word : s.spirv) U32(code, word);
    U64(table, XXH3_64bits(code.data(), code.size()));
    U32(table, CountKillsSpirv(s.spirv));
    U32(table, 0u);  // flags: reserved
    U64(table, position);
    position += 16 + s.original.size() + s.spirv.size() * 4;
    original.insert(original.end(), s.original.begin(), s.original.end());
  }
  Require(position == package.size(), "Trailing data in the shader package");
  std::vector<uint8_t> body = std::move(table);
  body.insert(body.end(), original.begin(), original.end());
  std::vector<uint8_t> index(kSignatureIndex.begin(), kSignatureIndex.end());
  U32(index, kVersionIndex);
  U32(index, uint32_t(shaders.size()));
  U64(index, package.size());
  U64(index, LE64(package.data() + 16));
  U64(index, original.size());
  U64(index, XXH3_64bits(body.data(), body.size()));
  U64(index, 0);
  index.insert(index.end(), body.begin(), body.end());
  return index;
}
}  // namespace sa2::native
