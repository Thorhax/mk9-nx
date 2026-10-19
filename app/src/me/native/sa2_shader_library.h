#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sa2::native {

struct FileShaders;  // open package of an indexed load (sa2_shader_library.cpp)

// The original container also identifies constants and interface. The physical
// code alone is not enough. It is looked up before the driver creates its object.
struct Shader {
  std::vector<uint8_t> original;
  // Full load (and shaders built by the tools): always filled. Indexed load (LoadIndexed): empty; the
  // SPIR-V is read from the package on the first Spirv(). The game uses Spirv(); `spirv` directly only
  // where the shader comes from a full load (tools) or is being packed.
  std::vector<uint32_t> spirv;
  uint64_t fingerprint = 0;
  bool vertices = false;

  // --- Indexed load only ---------------------------------------------------------
  // Precomputed by IndexShaders over the SPIR-V. In a full load indexed() is false and they are not
  // filled (callers compute them from the SPIR-V, as before).
  FileShaders* file = nullptr;  // package the SPIR-V is read from
  uint64_t displacement = 0;        // file offset of the entry (its 16-byte header)
  uint32_t words = 0;              // SPIR-V words
  uint64_t spirv_fingerprint = 0;          // XXH3 of the SPIR-V bytes as stored (little-endian)
  uint32_t kills = 0;                 // CountKillsSpirv
  bool indexed() const { return file != nullptr; }

  // The SPIR-V. With an indexed load the first call reads and verifies the entry (one seek and one read
  // under the package's read lock); if the package changed under the index or the read fails there is
  // nothing valid to return and the process aborts with a message. Thread safe; once returned, the vector
  // never changes or moves.
  const std::vector<uint32_t>& Spirv() const;

  // Copyable flag (std::atomic is not): set once the lazily read `spirv_lazy` is published.
  struct Flag {
    std::atomic<bool> v{false};
    Flag() = default;
    Flag(const Flag& o) : v(o.v.load(std::memory_order_acquire)) {}
    Flag& operator=(const Flag& o) {
      v.store(o.v.load(std::memory_order_acquire), std::memory_order_release);
      return *this;
    }
  };
  mutable Flag ready;
  // Written once, under the package's publication lock, before `ready` is set.
  mutable std::vector<uint32_t> spirv_lazy;
};

struct StatsPreload {
  uint32_t inputs = 0;       // read and published by this call
  uint32_t already_resident = 0;  // already loaded by someone else
  uint64_t bytes = 0;          // SPIR-V bytes read by this call
  bool error = false;          // could not open its own handle, or an entry failed verification
};

class LibraryShaders {
 public:
  LibraryShaders();
  ~LibraryShaders();
  LibraryShaders(const LibraryShaders&) = delete;
  LibraryShaders& operator=(const LibraryShaders&) = delete;

  // Transactional load: an invalid file does not replace the library.
  // Pointers returned by Search last until the next successful load.
  void Load(std::span<const uint8_t> file);
  void Load(const std::filesystem::path& file);
  // Only the index (entry table and the original containers, one sequential read) is
  // loaded; each SPIR-V is read from the package on its first Spirv(), and the package stays open. Same
  // entries, order and Search results as Load(package). Throws (library unchanged) if the index is
  // missing, damaged or belongs to another package (size and header checksum).
  void LoadIndexed(const std::filesystem::path& package, const std::filesystem::path& index);
  const Shader* Search(std::span<const uint8_t> original) const;
  const std::vector<Shader>& shaders() const { return shaders_; }
  bool is_indexed() const { return file_ != nullptr; }

  // Reads the SPIR-V of these shaders (of this library) in file order with a file handle of its own, so
  // Spirv() on other threads never waits for it: it only takes the publication lock to publish each entry.
  // `stop` is polled between entries. Nothing to do for a full load.
  StatsPreload Preload(std::vector<const Shader*> shaders, const std::atomic<bool>* stop) const;

 private:
  std::vector<Shader> shaders_;
  std::unique_ptr<FileShaders> file_;
};

// Versioned local format, little-endian, with integrity checks.
// It contains private data derived from the game; it is not embedded in the executable.
std::vector<uint8_t> PackShaders(std::vector<Shader> shaders);

// The index (<package>.idx) of a package, given the bytes of the whole .mesp (fully loaded
// and verified here). Format: HeaderIndex in sa2_shader_library.cpp.
std::vector<uint8_t> IndexShaders(std::span<const uint8_t> package);

// Facts about a SPIR-V module the native renderer needs at start-up (stored in the index).
// OpKill/OpTerminateInvocation/OpDemoteToHelperInvocation count; 2 if the module cannot be walked.
uint32_t CountKillsSpirv(std::span<const uint32_t> spirv);

}  // namespace sa2::native
