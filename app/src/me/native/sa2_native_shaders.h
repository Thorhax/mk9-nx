// SA2 - native renderer, shader identification: identify in the native shader library the
// shaders the game's D3D uploads to the ring (PM4_IM_LOAD and
// IM_LOAD_IMMEDIATE).
//
// The ring only carries the microcode. The library stores the original
// compiled container (header, constant table, definitions and microcode)
// with its SPIR-V translation made by XenosRecomp.
//
// Pixel shaders arrive as is and are identified by their microcode. Vertex
// shaders do not: D3D reorders the fetches, changes their swizzles and
// nulls the outputs the pixel shader does not read (measured), so their
// identity comes from the device objects (the D3D object table) and only
// the vertex input is read from the patched microcode.

#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <vector>

namespace sa2::native {
struct Shader;
struct StatsPreload;
}

namespace sa2::native {

// Vertex declaration element of the container (XenosRecomp shader.h).
struct ElementVertex {
  uint16_t instruction = 0;  // index of the fetch instruction in the microcode
  uint8_t usage = 0;           // DeclUsage: 0 position, 3 normal, 5 texcoord, 10 color...
  uint8_t usage_index = 0;
};

// Sampler from the constant table: the register is the texture's fetch constant
// in pixel shaders.
struct SamplerShader {
  uint16_t register_value = 0;
  uint16_t type = 0;  // D3DXPARAMETER_TYPE: 10-12 = 2D, 13 = 3D, 14 = cube
};

struct ShaderEntry {
  const sa2::native::Shader* shader = nullptr;  // original container and SPIR-V
  bool vertices = false;
  // Direct3D's own shader (no constant table): vertex input and interpolators linked by
  // register. Direct3D rewrites these vertex shaders per vertex declaration, so the variant the ring loads
  // is its own library entry (found by microcode), not the one of the object.
  bool binding_per_register = false;
  uint32_t number = 0;  // position in the library, for the reports
  std::vector<uint32_t> microcode;  // in host byte order, with the fetches masked
  uint64_t fingerprint = 0;                // XXH3 of that masked microcode
  std::vector<ElementVertex> elements;
  std::vector<SamplerShader> samplers;
  uint32_t outputs = 0;  // pixel shader: bits COLOR0..3 y DEPTH (PixelShaderOutputs)
  // banjo-nx: register-linked vertex shader: the vertex fetch constants (0..95) read by fetches indexed by a computed
  // register (bone palettes), in the order XenosRecomp numbers them (banjoVfetch K). See ClassifyVertexFetches.
  std::vector<uint8_t> indexed_constants;
  // banjo-nx: vertex shader whose ALU code may read r0, which starts as the vertex index (conservative: any ALU
  // register operand r0). Its draws pass the index base in the shared constants and do not take the zero-base binding.
  bool reads_r0 = true;
  // OpKill in the SPIR-V. XenosRecomp puts one in every pixel shader for the alpha test (guarded by
  // SPEC_CONSTANT_ALPHA_TEST), so it only really discards if there is more than one.
  uint32_t kills = 0;
  bool discards = false;
  // Bytes of its float constant buffer that the SPIR-V reads (16 per register).
  uint32_t constants_bytes = 256 * 16;
};

struct StatsShaders {
  uint64_t loads = 0;           // IM_LOAD packets received
  uint64_t distinct = 0;        // distinct microcodes (with their patches)
  uint64_t identified = 0;    // of the distinct ones
  uint64_t no_identify = 0;  // of the distinct ones
  uint64_t ambiguous = 0;         // distinct ones with more than one possible container
  uint64_t swizzle_variants = 0;  // identified only with a destination swizzle D3D patched (futari-nx)
  uint64_t reordered = 0;  // sa2-nx: identified only with their declared fetches reordered by D3D (IdentifyReordered)
};

class ShadersNative {
 public:
  ShadersNative();
  ~ShadersNative();

  // false, with a warning in the log, if the library is missing or damaged. With use_index and a
  // matching <file>.idx only the index is loaded and each SPIR-V is read on first use
  // (Shader::Spirv); otherwise the whole package.
  bool Load(const std::filesystem::path& file, bool use_index = true);
  bool loaded() const;
  bool is_indexed() const;  // loaded from the index: SPIR-V read on demand

  // Indexed library: reads now, on the calling thread, the SPIR-V of the entries with these container
  // fingerprints (PerFingerprint), with a file handle of its own. Other threads' Spirv() never wait for it
  // beyond publishing one entry. Nothing to do with a full load.
  sa2::native::StatsPreload PreloadSpirv(std::span<const uint64_t> fingerprints,
                                                     const std::atomic<bool>* stop) const;

  // Full original container (the one the D3D constructors receive).
  // nullptr if it is not in the library. Can be called from any thread
  // once the library is loaded.
  const ShaderEntry* IdentifyContainer(std::span<const uint8_t> container) const;

  // The entry with that number (its position in the library), or nullptr. Like IdentifyContainer, from any
  // thread once loaded: entries do not change after Load. Pipeline prewarming (sa2_native_draws.cpp) uses
  // it to recreate the pipelines of the previous session.
  const ShaderEntry* PerNumber(uint32_t number) const;

  // Stable container fingerprint. Unlike library position this survives adding newly discovered shaders,
  // so pipeline prewarming can reuse records produced by an older (smaller) package.
  const ShaderEntry* PerFingerprint(uint64_t fingerprint) const;

  // Microcode already in host byte order. nullptr if it is not in the library.
  // Only the ring thread uses it.
  const ShaderEntry* Identify(bool vertices, std::span<const uint32_t> microcode);

  // sa2-nx (sa2_gpu_vs_fetch_reorder): a vertex program that no entry matches in its own fetch order, but that is a
  // library shader with its declared fetches moved to each other's slots by Direct3D's patcher
  // (me_vertex_fetch_reorder.h): that entry (of several, a container of the game's own before a register-linked
  // one, then the fewest swizzle components to remap), or nullptr. Always nullptr while the key is off. Only the ring
  // thread uses it.
  const ShaderEntry* IdentifyReordered(std::span<const uint32_t> microcode);

  StatsShaders Stats() const;

 private:
  struct Data;
  std::unique_ptr<Data> data_;
};

// Short name of a DeclUsage for the reports.
const char* UsageName(uint8_t usage);

// Whether a patched microcode from the ring can belong to that vertex shader: every
// declaration element has, at one of the element positions, a fetch that writes the
// same temporary register as the original. D3D reorders the fetches and changes
// formats and swizzles, but not the registers.
bool FetchCoherent(const ShaderEntry& vs, std::span<const uint32_t> patched);

// sa2-nx (sa2_gpu_vs_fetch_reorder): the draw-time guard of an entry IdentifyReordered chose: the same proof again, on
// the loaded words (counted as a use). false while the key is off.
bool FetchReordered(const ShaderEntry& vs, std::span<const uint32_t> loaded);

}  // namespace sa2::native
