// SA2 - native renderer: shader identification (see sa2_native_shaders.h).
//
// Shader container format: see Read(). All fields are big-endian.

#include "sa2_native_shaders.h"

#include "sa2_shader_library.h"
#include "../me_shader_identity.h"
#include "../me_vertex_fetch_reorder.h"
#include "cpu/sa2_enhancements.h"

#include <rex/logging.h>

#include <algorithm>
#include <unordered_set>
#include <chrono>
#include <exception>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>

/*
 * xxHash reads with memcpy (XXH_FORCE_MEMORY_ACCESS 0). With the method it picks for GCC (1) it reads through
 * 64- and 32-bit pointers without may_alias, and GCC may move that read ahead of the store of the data being
 * hashed (strict aliasing). With that, the texture key read keys[4] before writing it and the same texture was
 * created several times. Same fingerprint values; on AArch64, the same LDR.
 */
#if defined(XXH_IMPLEM_13a8737387)
#error "xxhash.h was already included with its implementation before this point: XXH_FORCE_MEMORY_ACCESS 0 would come too late"
#endif
#undef XXH_FORCE_MEMORY_ACCESS
#define XXH_FORCE_MEMORY_ACCESS 0
#define XXH_INLINE_ALL
#include <xxhash.h>

// sa2-nx (2026-10-07): vertex programs Direct3D loads with their declared fetches moved to each other's slots for the
// vertex declaration (me_vertex_fetch_reorder.h). Non-CPU group: a correctness fix; off = such draws are dropped, as
// before, unless the package carries a container repaired offline for that very order.
SA2_ENHANCEMENT(g_vs_fetch_reorder, sa2_gpu_vs_fetch_reorder, sa2::enh::kNonCpu, true,
                "Renderer: a vertex shader Direct3D loads with its declared fetches moved to each other's slots (for the "
                "vertex declaration) is identified as the library shader it was made from, by the offline repair "
                "tool's proof at run time, when no library entry has that order (such draws were dropped: the Chao "
                "garden's stats window)");

SA2_ENHANCEMENT(g_constants_exact_extent, usf4_cpu_constants_exact_extent, sa2::enh::kCpu, true,
                "Shaders without relative constant operands upload their float constants up to the last declared "
                "register instead of all 256 when they have an array or matrix");

namespace sa2::native {
namespace {

// sa2_gpu_vs_fetch_reorder: the proof disagreed with its brute-force self-check (IdentifyReordered): off for the
// session.
std::atomic<bool> g_reorder_off{false};

// Bits of a vertex fetch instruction that D3D does not touch
// (VertexFetchInstruction in XenosRecomp shader_code.h): opcode, source and
// destination registers, destination swizzle and predicate. The rest (fetch
// constant, format, stride, offset, numeric modes and isMiniFetch come from the declaration. In
// particular D3D may turn a full fetch from the container into a mini fetch in the ring.
constexpr uint32_t kFetchKeeps[3] = {0x0007FFFF, 0x80000FFF, 0x80000000};

constexpr uint32_t kMaxWarnings = 48;

constexpr const char* kUses[] = {"position",   "weight",   "indices",    "normal",
                                 "point_size", "texcoord", "tangent",    "binormal",
                                 "tessellation", "position_t", "color",  "fog",
                                 "depth",      "sample",   "usage14",    "usage15"};

struct Container {
  const std::vector<uint8_t>& o;

  bool Has(size_t position, size_t bytes) const {
    return position <= o.size() && bytes <= o.size() - position;
  }
  uint32_t U32(size_t p) const {
    return uint32_t(o[p]) << 24 | uint32_t(o[p + 1]) << 16 | uint32_t(o[p + 2]) << 8 | o[p + 3];
  }
  uint16_t U16(size_t p) const { return uint16_t(uint32_t(o[p]) << 8 | o[p + 1]); }
};

// The full vertex fetches of a microcode (instruction index, destination register), walking the
// exec clauses of the control flow. Same walk as XenosRecomp vertexFetchesByRegister:
// Direct3D's own vertex shaders (no constant table) are linked by fetch destination
// register, one TEXCOORD<register> element per fetch, because Direct3D rewrites those fetches by register.
std::vector<std::pair<uint32_t, uint32_t>> FetchesOfVertices(const std::vector<uint32_t>& w) {
  std::vector<std::pair<uint32_t, uint32_t>> fetches;
  uint32_t end = uint32_t(w.size() / 3);
  for (uint32_t par = 0; par < end && par * 3 + 2 < w.size(); ++par) {
    const uint64_t cf[2] = {uint64_t(w[par * 3]) | (uint64_t(w[par * 3 + 1] & 0xFFFF) << 32),
                            uint64_t(w[par * 3 + 1] >> 16) | (uint64_t(w[par * 3 + 2]) << 16)};
    for (uint64_t c : cf) {
      const uint32_t opcode = uint32_t(c >> 44) & 0xF;
      if (!((opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14)) continue;
      const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
      const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
      if (address != 0) end = std::min(end, address);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = address + i;
        if (!((sequence >> (i * 2)) & 0x1) || size_t(index) * 3 + 2 >= w.size()) continue;
        const uint32_t d0 = w[index * 3];
        if ((d0 & 0x1F) != 0) continue;
        fetches.emplace_back(index, (d0 >> 12) & 0x3F);
      }
    }
  }
  return fetches;
}

// banjo-nx (same as XenosRecomp vertexFetchesAfterR0Write): whether r0.x still holds the vertex index at each full vertex fetch. A Xenos vertex shader starts with
// the vertex index in r0.x; Nuts & Bolts' instanced shaders overwrite r0 (e.g. r0 = floor(index * c132.x), the
// instance) before fetching "by r0.x", which then fetches by that computed index, not per vertex. Walks the exec
// clauses in control-flow order; any ALU or fetch write to r0.x (predicated or not) ends the vertex index.
std::unordered_set<uint32_t> VertexFetchesAfterR0Write(const std::vector<uint32_t>& w) {
  std::unordered_set<uint32_t> after;
  bool written = false;
  uint32_t end = uint32_t(w.size() / 3);
  for (uint32_t par = 0; par < end && par * 3 + 2 < w.size(); ++par) {
    const uint64_t cf[2] = {uint64_t(w[par * 3]) | (uint64_t(w[par * 3 + 1] & 0xFFFF) << 32),
                            uint64_t(w[par * 3 + 1] >> 16) | (uint64_t(w[par * 3 + 2]) << 16)};
    for (uint64_t c : cf) {
      const uint32_t opcode = uint32_t(c >> 44) & 0xF;
      if (!((opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14)) continue;
      const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
      const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
      if (address != 0) end = std::min(end, address);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = address + i;
        if (size_t(index) * 3 + 2 >= w.size()) continue;
        const uint32_t w0 = w[size_t(index) * 3], w1 = w[size_t(index) * 3 + 1];
        if ((sequence >> (i * 2)) & 0x1) {
          if ((w0 & 0x1F) == 0 && written) after.insert(index);
          if (((w0 >> 12) & 0x3F) == 0 && !((w0 >> 18) & 0x1) && (w1 & 0x7) != 7) written = true;
        } else if (!((w0 >> 15) & 0x1)) {
          if ((w0 & 0x3F) == 0 && !((w0 >> 6) & 0x1) && ((w0 >> 16) & 0x1)) written = true;
          if (((w0 >> 8) & 0x3F) == 0 && !((w0 >> 14) & 0x1) && ((w0 >> 20) & 0x1)) written = true;
        }
      }
    }
  }
  return after;
}

// banjo-nx: the vertex fetches of a register-linked vertex shader split like XenosRecomp's classifyVertexFetches
// (shader_recompiler.cpp), which must stay identical:
//   - attributes, indexed by the vertex index (full fetch from r0.x; a mini fetch follows its full fetch): input slot =
//     destination register, or for a register fetched again the lowest slot 0..15 no attribute register uses;
//   - indexed by a computed register (bone matrices by blend index): read in the shader from the buffer of their
//     fetch constant, at most 2 distinct constants.
struct SplitVertexFetches {
  std::vector<std::pair<uint32_t, uint32_t>> attributes;  // (instruction index, input slot)
  std::vector<uint8_t> indexed_constants;
  const char* error = nullptr;
};

SplitVertexFetches ClassifyVertexFetches(const std::vector<uint32_t>& w) {
  SplitVertexFetches split;
  std::vector<std::pair<uint32_t, uint32_t>> registers;
  bool last_full_attribute = true;
  const auto after_r0_write = VertexFetchesAfterR0Write(w);
  for (const auto& [index, register_value] : FetchesOfVertices(w)) {
    const uint32_t d0 = w[size_t(index) * 3], d1 = w[size_t(index) * 3 + 1];
    if (!((d1 >> 30) & 0x1)) {
      last_full_attribute = ((d0 >> 5) & 0x3F) == 0 && !((d0 >> 11) & 0x1) && ((d0 >> 30) & 0x3) == 0 &&
                            !after_r0_write.count(index);
      if (!last_full_attribute) {
        const uint8_t constant = uint8_t(((d0 >> 20) & 0x1F) * 3 + ((d0 >> 25) & 0x3));
        if (std::find(split.indexed_constants.begin(), split.indexed_constants.end(), constant) ==
            split.indexed_constants.end()) {
          if (split.indexed_constants.size() >= 2) {
            split.error = "more than 2 indexed vertex fetch constants";
            return split;
          }
          split.indexed_constants.push_back(constant);
        }
      }
    }
    if (last_full_attribute) registers.emplace_back(index, register_value);
  }
  uint32_t registers_used = 0;
  for (const auto& [index, register_value] : registers) {
    if (register_value >= 16) {
      split.error = "vertex fetch into a register with no binding slot (>= r16)";
      return split;
    }
    registers_used |= 1u << register_value;
  }
  uint32_t taken = 0;
  for (const auto& [index, register_value] : registers) {
    uint32_t slot = register_value;
    if (taken & (1u << slot)) {
      slot = 0;
      while (slot < 16 && ((registers_used | taken) & (1u << slot))) ++slot;
      if (slot >= 16) {
        split.error = "no free vertex input slot for a repeated fetch register";
        return split;
      }
    }
    taken |= 1u << slot;
    split.attributes.emplace_back(index, slot);
  }
  return split;
}

// banjo-nx: whether any ALU instruction of the exec clauses takes r0 as a register operand (conservative: unused
// operand fields count too). XenosRecomp starts r0 as the guest vertex index (SV_VertexID + g_VertexIndexBase).
bool AluReadsR0(const std::vector<uint32_t>& w) {
  uint32_t end = uint32_t(w.size() / 3);
  for (uint32_t par = 0; par < end && par * 3 + 2 < w.size(); ++par) {
    const uint64_t cf[2] = {uint64_t(w[par * 3]) | (uint64_t(w[par * 3 + 1] & 0xFFFF) << 32),
                            uint64_t(w[par * 3 + 1] >> 16) | (uint64_t(w[par * 3 + 2]) << 16)};
    for (uint64_t c : cf) {
      const uint32_t opcode = uint32_t(c >> 44) & 0xF;
      if (!((opcode >= 1 && opcode <= 6) || opcode == 13 || opcode == 14)) continue;
      const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
      const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
      if (address != 0) end = std::min(end, address);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = address + i;
        if (((sequence >> (i * 2)) & 0x1) || size_t(index) * 3 + 2 >= w.size()) continue;  // fetch
        const uint32_t w2 = w[size_t(index) * 3 + 2];
        for (uint32_t k = 0; k < 3; ++k) {
          const bool is_register = (w2 >> (29 + k)) & 0x1;      // src3, src2, src1 select
          const uint32_t reg = (w2 >> (k * 8)) & 0x3F;          // src3, src2, src1 register
          if (is_register && reg == 0) return true;
        }
      }
    }
  }
  return false;
}

// Texture fetch constant registers referenced by a table-less pixel shader. Direct3D's generated
// Scaleform shaders have an empty CTAB even though their microcode samples s0; without this scan the
// native renderer leaves descriptor index 0 in the shared block and every glyph samples the transparent
// fallback texture. The control-flow walk is the same one used above for vertex fetches.
// usf4-nx (usf4_cpu_constants_exact_extent): whether any ALU instruction of the shader has a relative constant
// operand (dword 1 bits 30-31, XenosRecomp AluInstruction const1Relative / const0Relative). Only those read a
// constant array past its declared registers (up to register 255, shader_recompiler.cpp); the bits are checked on
// every ALU instruction, whatever its operands, so a doubt counts as relative.
bool UsesRelativeConstants(const std::vector<uint32_t>& w) {
  uint32_t end = uint32_t(w.size() / 3);
  for (uint32_t par = 0; par < end && par * 3 + 2 < w.size(); ++par) {
    const uint64_t cf[2] = {uint64_t(w[par * 3]) | (uint64_t(w[par * 3 + 1] & 0xFFFF) << 32),
                            uint64_t(w[par * 3 + 1] >> 16) | (uint64_t(w[par * 3 + 2]) << 16)};
    for (uint64_t c : cf) {
      const uint32_t opcode_cf = uint32_t(c >> 44) & 0xF;
      if (!((opcode_cf >= 1 && opcode_cf <= 6) || opcode_cf == 13 || opcode_cf == 14)) continue;
      const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
      const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
      if (address != 0) end = std::min(end, address);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = address + i;
        if (((sequence >> (i * 2)) & 0x1) || size_t(index) * 3 + 2 >= w.size()) continue;  // fetch
        if (w[index * 3 + 1] & 0xC0000000u) return true;
      }
    }
  }
  return false;
}

std::vector<uint32_t> FetchesOfTextures(const std::vector<uint32_t>& w) {
  std::vector<uint32_t> registers;
  uint32_t end = uint32_t(w.size() / 3);
  for (uint32_t par = 0; par < end && par * 3 + 2 < w.size(); ++par) {
    const uint64_t cf[2] = {uint64_t(w[par * 3]) | (uint64_t(w[par * 3 + 1] & 0xFFFF) << 32),
                            uint64_t(w[par * 3 + 1] >> 16) | (uint64_t(w[par * 3 + 2]) << 16)};
    for (uint64_t c : cf) {
      const uint32_t opcode_cf = uint32_t(c >> 44) & 0xF;
      if (!((opcode_cf >= 1 && opcode_cf <= 6) || opcode_cf == 13 || opcode_cf == 14)) continue;
      const uint32_t address = uint32_t(c) & 0xFFF, count = uint32_t(c >> 12) & 0x7;
      const uint32_t sequence = uint32_t(c >> 16) & 0xFFF;
      if (address != 0) end = std::min(end, address);
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t index = address + i;
        if (!((sequence >> (i * 2)) & 0x1) || size_t(index) * 3 + 2 >= w.size()) continue;
        const uint32_t d0 = w[index * 3];
        const uint32_t opcode_fetch = d0 & 0x1F;
        // TextureFetch and GetTextureWeights. Both need the texture and sampler descriptors.
        if (opcode_fetch != 1 && opcode_fetch != 19) continue;
        const uint32_t register_value = (d0 >> 20) & 0x1F;
        if (std::find(registers.begin(), registers.end(), register_value) == registers.end()) {
          registers.push_back(register_value);
        }
      }
    }
  }
  return registers;
}

// Reads what is needed from a shader container. Two layouts are understood: the 0x102A11xx one of
// XenosRecomp shader.h (Mass Effect), and an older 0x102A0Exx one (24-byte header with signature, virtual
// part, physical part, definitions (+12), CTAB (+16) and shader header (+20); the microcode is the whole
// physical part), which the game does not use. Returns the reason if it cannot be read.
const char* Read(const sa2::native::Shader& shader, ShaderEntry& e) {
  const Container c{shader.original};
  if (!c.Has(0, 24)) return "container too short";
  const uint32_t virtual_value = c.U32(4);
  const uint32_t table = c.U32(16);
  // D3D's built-in shaders carry a syntactically valid CTAB at 0x24, but its constant count is
  // zero. Semantically this is the same as no table: inputs, outputs, constants and texture fetches
  // are linked by their hardware register numbers. Checking only a null offset misclassified every
  // such Mass Effect shader as a normal package shader.
  const bool empty_table = !table || (c.Has(size_t(table) + 4, 16) && c.U32(size_t(table) + 16) == 0);
  // The 0x102A11xx layout (XenosRecomp shader.h ShaderContainer) keeps the shader
  // header at +0x18 and the microcode at virtualSize + shader.physicalOffset, shader.size bytes; the
  // physical part can hold more than the microcode.
  const bool is2008 = (c.U32(0) & 0xFFFFFF00u) == 0x102A1100u;
  const uint32_t header = is2008 ? c.U32(24) : c.U32(20);
  if (is2008 && !c.Has(header, 8)) return "shader header outside the container";
  const uint32_t microcode_start = is2008 ? virtual_value + c.U32(header) : virtual_value;
  const uint32_t physical = is2008 ? c.U32(header + 4) : c.U32(8);
  if (!physical || (physical % 4) || !c.Has(microcode_start, physical)) {
    return "microcode outside the container";
  }
  if (header >= virtual_value || size_t(header) + (shader.vertices ? 40 : 32) > virtual_value) {
    return "shader header outside the virtual part";
  }

  e.vertices = shader.vertices;
  e.microcode.resize(physical / 4);
  for (size_t i = 0; i < e.microcode.size(); ++i) {
    e.microcode[i] = c.U32(size_t(microcode_start) + i * 4);
  }

  if (e.vertices) {
    // List at +0x28, after skipping the words at +0x18; +0x1C = element count.
    const uint32_t previous = c.U32(header + 24);
    const uint32_t count = c.U32(header + 28);
    const size_t beginning = size_t(header) + (is2008 ? 36 : 40) + size_t(previous) * 4;
    if (count > 64 || previous > 1024 || beginning + size_t(count) * 4 > virtual_value) {
      return "vertex elements outside the virtual part";
    }
    for (uint32_t i = 0; i < count; ++i) {
      const uint32_t value = c.U32(beginning + size_t(i) * 4);
      ElementVertex element;
      element.instruction = uint16_t(value & 0xFFF);
      element.usage = uint8_t((value >> 12) & 0xF);
      element.usage_index = uint8_t((value >> 16) & 0xF);
      if ((size_t(element.instruction) + 1) * 3 > e.microcode.size()) {
        return "vertex element points outside the microcode";
      }
      e.elements.push_back(element);
    }
    if (empty_table) {
      e.binding_per_register = true;
      // Linked by fetch destination register (FetchesOfVertices).
      e.elements.clear();
      const SplitVertexFetches split = ClassifyVertexFetches(e.microcode);
      if (split.error) return split.error;
      for (const auto& [index, slot] : split.attributes) {
        ElementVertex element;
        element.instruction = uint16_t(index);
        element.usage = 5;  // texcoord
        element.usage_index = uint8_t(slot);
        e.elements.push_back(element);
      }
      e.indexed_constants = split.indexed_constants;
      e.reads_r0 = AluReadsR0(e.microcode);
    }
    // Normalize only AFTER selecting the authoritative fetch set. Table-less
    // D3D shaders may carry stale header DECL entries pointing at ALU triples;
    // masking those first would irreversibly corrupt strict ALU/CF identity.
    for (const auto& element : e.elements) {
      for (size_t j = 0; j < 3; ++j)
        e.microcode[size_t(element.instruction) * 3 + j] &= kFetchKeeps[j];
    }
  } else {
    if (!c.Has(header + 24, 8)) return "pixel shader header too short";
    e.outputs = c.U32(header + 28);
  }

  // Constant table: only the samplers (register and type).
  // Direct3D's own shaders (clears, copies) have none: no named samplers, and the whole
  // float buffer, since XenosRecomp reads their registers by number.
  if (empty_table) {
    e.binding_per_register = true;
    e.constants_bytes = 256 * 16;
    if (!e.vertices) {
      for (uint32_t register_value : FetchesOfTextures(e.microcode)) {
        SamplerShader sampler;
        sampler.register_value = uint16_t(register_value);
        e.samplers.push_back(sampler);
      }
    }
    e.fingerprint = XXH3_64bits(e.microcode.data(), e.microcode.size() * sizeof(uint32_t));
    return nullptr;
  }
  if (!c.Has(table + 4, 28)) return "no constant table";
  const size_t base = size_t(table) + 4;
  const uint32_t constants = c.U32(base + 12);
  const uint32_t info = c.U32(base + 16);
  if (constants > 1024 || !c.Has(base + info, size_t(constants) * 20)) {
    return "constant table outside the container";
  }
  // Float registers the SPIR-V reads: the highest in the table, or up to the end of the
  // buffer if there is an array with relative indexing (XenosRecomp shader_recompiler.cpp:
  // 1172-1183: tailCount = 256 in VS and 224 in PS).
  // usf4-nx (usf4_cpu_constants_exact_extent): without relative operands, an array is read only inside its own
  // registers, so the extent is the end of the last one (a 4x4 matrix no longer means all 256 registers: ~85 % of
  // USF4's draws copied the whole 4 KB VS bank).
  uint32_t registers_float = 0, registers_declared = 0;
  for (uint32_t i = 0; i < constants; ++i) {
    const size_t p = base + info + size_t(i) * 20;
    if (c.U16(p + 4) == 2) {  // RegisterSet::Float4
      const uint32_t index = c.U16(p + 6);
      const uint32_t how_many = c.U16(p + 8);
      registers_float = std::max(registers_float, how_many > 1 ? 256u
                                                              : index + 1);
      registers_declared = std::max(registers_declared, index + std::max<uint32_t>(how_many, 1));
    }
  }
  if (registers_float > registers_declared && g_constants_exact_extent.active &&
      !UsesRelativeConstants(e.microcode)) {
    g_constants_exact_extent.Use();
    registers_float = registers_declared;
  }
  e.constants_bytes = std::min<uint32_t>(std::max<uint32_t>(registers_float, 1), 256) * 16;
  for (uint32_t i = 0; i < constants; ++i) {
    const size_t p = base + info + size_t(i) * 20;
    if (c.U16(p + 4) != 3) continue;  // RegisterSet::Sampler
    SamplerShader sampler;
    sampler.register_value = c.U16(p + 6);
    const uint32_t type = c.U32(p + 12);
    sampler.type = c.Has(base + type, 4) ? c.U16(base + type + 2) : 0;
    e.samplers.push_back(sampler);
  }

  // Mass Effect has shaders whose CTAB omits anonymous hardware samplers. The movie YUV shader is
  // one: tex0-tex2 are named in the table, while s3 is still sampled by the microcode and supplies
  // the movie alpha. XenosRecomp emits the s3 descriptor access, so leaving it out here binds slot 0
  // accidentally. Treat the microcode as authoritative and add every referenced fetch register that
  // the table did not name. This is also safe for ordinary shaders, where the two sets are identical.
  if (!e.vertices) {
    for (uint32_t register_value : FetchesOfTextures(e.microcode)) {
      const auto existing = std::find_if(e.samplers.begin(), e.samplers.end(), [register_value](const auto& s) {
        return s.register_value == register_value;
      });
      if (existing == e.samplers.end()) {
        SamplerShader sampler;
        sampler.register_value = uint16_t(register_value);
        e.samplers.push_back(sampler);
      }
    }
  }

  e.fingerprint = XXH3_64bits(e.microcode.data(), e.microcode.size() * sizeof(uint32_t));
  return nullptr;
}

struct RawKey {
  uint64_t fingerprint;
  uint32_t words;
  bool vertices;
  bool operator==(const RawKey&) const = default;
};

struct HashRawKey {
  size_t operator()(const RawKey& c) const {
    return size_t(c.fingerprint ^ (uint64_t(c.words) << 1) ^ uint64_t(c.vertices));
  }
};

}  // namespace

struct ShadersNative::Data {
  sa2::native::LibraryShaders library;
  std::vector<ShaderEntry> inputs;
  std::unordered_map<const sa2::native::Shader*, uint32_t> per_shader;
  std::unordered_map<uint64_t, uint32_t> per_fingerprint;  // container XXH3 -> entries (PerFingerprint)
  // (vertices, words) -> candidate entries.
  std::map<std::pair<bool, uint32_t>, std::vector<uint32_t>> candidates;
  std::unordered_map<RawKey, const ShaderEntry*, HashRawKey> cache;
  std::vector<uint32_t> temporal;
  StatsShaders stats;
  uint32_t warnings = 0;
  bool loaded = false;
  // sa2_gpu_vs_fetch_reorder (IdentifyReordered): its answer per distinct program, and the proofs still to hold
  // against the brute-force placements outside verify mode (every one in verify mode).
  std::unordered_map<RawKey, const ShaderEntry*, HashRawKey> reorder_cache;
  uint32_t reorder_checks = 64;
};

ShadersNative::ShadersNative() : data_(std::make_unique<Data>()) {}
ShadersNative::~ShadersNative() = default;

bool ShadersNative::loaded() const {
  return data_->loaded;
}

bool FetchCoherent(const ShaderEntry& vs, std::span<const uint32_t> patched) {
  if (!vs.vertices || patched.size() != vs.microcode.size()) {
    return false;
  }
  for (const ElementVertex& element : vs.elements) {
    const size_t p = size_t(element.instruction) * 3;
    const uint32_t register_value = (vs.microcode[p] >> 12) & 0x3F;
    bool found = false;
    for (const ElementVertex& other : vs.elements) {
      const uint32_t d0 = patched[size_t(other.instruction) * 3];
      if (((d0 >> 12) & 0x3F) == register_value && (d0 & 0x1F) == 0) {
        found = true;
        break;
      }
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

bool FetchReordered(const ShaderEntry& vs, std::span<const uint32_t> loaded) {
  if (!g_vs_fetch_reorder.active || g_reorder_off.load(std::memory_order_relaxed) || !vs.vertices) return false;
  const bool proven = bool(me::native::ProveVertexFetchReorder(
      loaded, vs.microcode, vs.elements, [](const ElementVertex& element) { return element.instruction; }));
  if (proven) g_vs_fetch_reorder.Use();
  return proven;
}

bool ShadersNative::Load(const std::filesystem::path& file, bool use_index) {
  Data& d = *data_;
  const auto start = std::chrono::steady_clock::now();
  // With <package>.idx only the index is read (entry table and original containers) and
  // each SPIR-V is read from the package on first use. Without it, or if it does not match the package,
  // the whole package is loaded as before.
  bool is_indexed = false;
  if (use_index) {
    std::filesystem::path index = file;
    index += ".idx";
    std::error_code ec;
    if (std::filesystem::exists(index, ec)) {
      try {
        d.library.LoadIndexed(file, index);
        is_indexed = true;
      } catch (const std::exception& e) {
        REXLOG_WARN("[native] shaders: index {} not used ({}); loading the whole package", index.string(),
                    e.what());
      }
    } else {
      REXLOG_INFO("[native] shaders: no index {}: loading the whole package", index.string());
    }
  }
  if (!is_indexed) {
    try {
      d.library.Load(file);
    } catch (const std::exception& e) {
      REXLOG_WARN("[native] shaders: shader library not available ({}): {}", file.string(),
                  e.what());
      return false;
    }
  }
  const auto& shaders = d.library.shaders();
  uint32_t with_kill = 0, no_kill = 0;
  d.inputs.clear();
  d.inputs.reserve(shaders.size());
  uint32_t vertex = 0, pixel = 0;
  for (uint32_t i = 0; i < shaders.size(); ++i) {
    ShaderEntry e;
    e.shader = &shaders[i];
    e.number = i;
    if (const char* reason = Read(shaders[i], e)) {
      REXLOG_WARN("[native] shaders: library container {} ignored: {}", i, reason);
      continue;
    }
    if (!e.vertices) {
      e.kills = shaders[i].indexed() ? shaders[i].kills : sa2::native::CountKillsSpirv(shaders[i].spirv);
      e.discards = e.kills != 1;                // 1 = only the alpha test one
      (e.discards ? with_kill : no_kill) += 1;
    }
    (e.vertices ? vertex : pixel) += 1;
    d.inputs.push_back(std::move(e));
  }
  d.candidates.clear();
  d.per_shader.clear();
  d.per_fingerprint.clear();
  std::map<std::tuple<bool, uint32_t, uint64_t>, std::vector<uint32_t>> groups;
  for (uint32_t i = 0; i < d.inputs.size(); ++i) {
    const ShaderEntry& e = d.inputs[i];
    const uint32_t words = uint32_t(e.microcode.size());
    d.candidates[{e.vertices, words}].push_back(i);
    d.per_shader[e.shader] = i;
    d.per_fingerprint.emplace(e.shader->fingerprint, i);  // the first in library order, as the linear search did
    groups[{e.vertices, words, e.fingerprint}].push_back(i);
  }
  uint32_t repeated = 0, repeated_distinct = 0;
  for (const auto& [key, members] : groups) {
    if (members.size() < 2) continue;
    ++repeated;
    // Indexed: the SPIR-V is compared by its length and XXH3 (stored in the index) instead of read.
    const sa2::native::Shader& first = *d.inputs[members[0]].shader;
    for (uint32_t m : members) {
      const sa2::native::Shader& other = *d.inputs[m].shader;
      if (first.indexed() ? (other.words != first.words || other.spirv_fingerprint != first.spirv_fingerprint)
                             : other.spirv != first.spirv) {
        ++repeated_distinct;
        break;
      }
    }
  }
  d.cache.clear();
  d.loaded = !d.inputs.empty();
  REXLOG_INFO("[native] shaders: library with {} shaders ({} vertex, {} pixel); {} groups sharing the "
              "same microcode, {} of them with different SPIR-V",
              d.inputs.size(), vertex, pixel, repeated, repeated_distinct);
  REXLOG_INFO("[native] shaders: library loaded in {} ms ({}, {:.1f} MB of SPIR-V resident)",
              std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count(),
              is_indexed ? "index only; SPIR-V on demand" : "whole package",
              is_indexed ? 0.0 : [&] {
                size_t bytes = 0;
                for (const auto& s : shaders) bytes += s.spirv.size() * 4;
                return double(bytes) / (1024.0 * 1024.0);
              }());
  REXLOG_INFO("[native] shaders: pixel shaders that can discard pixels {} of {} (the rest only carry the "
              "alpha-test kill: no color to write, so their stage can be dropped)",
              with_kill, with_kill + no_kill);
  return d.loaded;
}

const ShaderEntry* ShadersNative::Identify(bool vertices,
                                                 std::span<const uint32_t> microcode) {
  Data& d = *data_;
  ++d.stats.loads;
  const RawKey key{XXH3_64bits(microcode.data(), microcode.size_bytes()),
                         uint32_t(microcode.size()), vertices};
  if (auto it = d.cache.find(key); it != d.cache.end()) {
    return it->second;
  }
  ++d.stats.distinct;

  const ShaderEntry* chosen = nullptr;
  uint32_t matches = 0;
  if (auto c = d.candidates.find({vertices, uint32_t(microcode.size())});
      c != d.candidates.end()) {
    for (uint32_t index : c->second) {
      const ShaderEntry& e = d.inputs[index];
      d.temporal.assign(microcode.begin(), microcode.end());
      for (const ElementVertex& element : e.elements) {
        for (size_t j = 0; j < 3; ++j) {
          d.temporal[size_t(element.instruction) * 3 + j] &= kFetchKeeps[j];
        }
      }
      if (XXH3_64bits(d.temporal.data(), d.temporal.size() * sizeof(uint32_t)) != e.fingerprint ||
          d.temporal != e.microcode) {
        continue;
      }
      if (!chosen) {
        chosen = &e;
      }
      ++matches;
    }
    // futari-nx: D3D also rewrites a fetch's destination swizzle for the declaration (float3 -> xyz1, D3DCOLOR ->
    // zyxw, float2 -> xy01), and the fingerprint above keeps it, so a variant of a library shader for another
    // declaration was not identified (Mushihimesama Futari: vs_6090 at 1FC95000 is the container vs_9fb4 with 0x688 -> 0xA88
    // and 0x60A). The draw path already applies such swizzles through g_InputRemap: accept, after no exact match, the
    // candidate that the draw-time predicate (me_shader_identity.h) accepts.
    // Of several such candidates the closest wins (fewest fetch swizzle components to remap; sa2-nx's suggestion), so
    // a library variant built for this very declaration is preferred over a remapped container.
    if (!chosen && vertices) {
      uint32_t best_remaps = UINT32_MAX;
      for (uint32_t index : c->second) {
        const ShaderEntry& e = d.inputs[index];
        if (!me::native::VertexShaderIdentityMatches(
                {me::native::ShaderIdentityStage::Vertex, microcode},
                {me::native::ShaderIdentityStage::Vertex, e.microcode}, e.elements,
                [](const ElementVertex& element) { return element.instruction; })) {
          continue;
        }
        uint32_t remaps = 0;
        for (const ElementVertex& element : e.elements) {
          const uint32_t original = e.microcode[size_t(element.instruction) * 3 + 1] & 0xFFF;
          const uint32_t patched = microcode[size_t(element.instruction) * 3 + 1] & 0xFFF;
          for (uint32_t k = 0; k < 4; ++k) remaps += ((original >> (k * 3)) & 7) != ((patched >> (k * 3)) & 7);
        }
        if (remaps < best_remaps) {
          best_remaps = remaps;
          chosen = &e;
        }
        ++matches;
      }
      if (chosen && ++d.stats.swizzle_variants <= 16) {
        REXLOG_INFO("[native] shaders: VS n{} identified with destination swizzles patched by D3D ({} words, {} "
                    "components remapped, {} candidates)", chosen->number, microcode.size(), best_remaps, matches);
      }
    }
  }

  // Vertex shaders arrive patched: a mismatch is normal.
  const bool warn = d.warnings < kMaxWarnings && !vertices;
  if (chosen) {
    ++d.stats.identified;
    if (matches > 1) {
      ++d.stats.ambiguous;
    }
    if (warn && vertices) {
      // What D3D has patched in each fetch: it becomes the vertex input.
      ++d.warnings;
      std::string detail;
      for (const ElementVertex& element : chosen->elements) {
        const size_t p = size_t(element.instruction) * 3;
        const uint32_t d0 = microcode[p], d1 = microcode[p + 1], d2 = microcode[p + 2];
        const uint32_t fetch = ((d0 >> 20) & 0x1F) * 3 + ((d0 >> 25) & 0x3);
        const int32_t offset = int32_t(d2 << 1) >> 9;
        detail += fmt::format(" {}{}:f{}/fmt{}/z{}/o{}{}", kUses[element.usage & 0xF],
                               element.usage_index, fetch, (d1 >> 16) & 0x3F, d2 & 0xFF, offset,
                               ((d1 >> 30) & 0x1) ? "/mini" : "");
      }
      REXLOG_INFO("[native] shaders: VS n{} ({} words, {} matches):{}", chosen->number,
                  microcode.size(), matches, detail);
    }
  } else {
    ++d.stats.no_identify;
    if (warn) {
      ++d.warnings;
      const auto c = d.candidates.find({vertices, uint32_t(microcode.size())});
      REXLOG_WARN("[native] shaders: unidentified {} shader: {} words, fingerprint {:016X} "
                  "({} containers of that type and length)",
                  vertices ? "vertex" : "pixel", microcode.size(), key.fingerprint,
                  c != d.candidates.end() ? c->second.size() : 0);
      // Diagnostics: which words change against the containers of the same
      // length (incoming / container original, without the mask).
      for (size_t k = 0; c != d.candidates.end() && k < c->second.size() && k < 2; ++k) {
        const ShaderEntry& e = d.inputs[c->second[k]];
        const auto& o = e.shader->original;
        const size_t virtual_value = size_t(o[4]) << 24 | size_t(o[5]) << 16 | size_t(o[6]) << 8 | o[7];
        std::string differences;
        uint32_t how_many = 0;
        for (size_t i = 0; i < microcode.size(); ++i) {
          uint32_t incoming = microcode[i];
          for (const ElementVertex& element : e.elements) {
            const size_t p = size_t(element.instruction) * 3;
            if (i >= p && i < p + 3) {
              incoming &= kFetchKeeps[i - p];
            }
          }
          if (incoming == e.microcode[i]) {
            continue;
          }
          if (++how_many <= 24) {
            const size_t b = virtual_value + i * 4;
            const uint32_t original = uint32_t(o[b]) << 24 | uint32_t(o[b + 1]) << 16 |
                                      uint32_t(o[b + 2]) << 8 | o[b + 3];
            differences += fmt::format(" {}:{:08X}/{:08X}", i, microcode[i], original);
          }
        }
        std::string fetch;
        for (const ElementVertex& element : e.elements) {
          fetch += fmt::format(" {}", element.instruction);
        }
        REXLOG_WARN("[native] shaders:   versus n{}: {} words differ; fetch at "
                    "instructions{}:{}",
                    e.number, how_many, fetch, differences);
      }
    }
  }
  d.cache.emplace(key, chosen);
  return chosen;
}

const ShaderEntry* ShadersNative::IdentifyReordered(std::span<const uint32_t> microcode) {
  Data& d = *data_;
  if (!g_vs_fetch_reorder.active || g_reorder_off.load(std::memory_order_relaxed) || !d.loaded || microcode.empty()) {
    return nullptr;
  }
  const RawKey key{XXH3_64bits(microcode.data(), microcode.size_bytes()), uint32_t(microcode.size()), true};
  if (auto it = d.reorder_cache.find(key); it != d.reorder_cache.end()) {
    return it->second;
  }
  const ShaderEntry* chosen = nullptr;
  me::native::VertexFetchReorder best;
  uint32_t proven = 0;
  if (auto c = d.candidates.find({true, uint32_t(microcode.size())}); c != d.candidates.end()) {
    for (uint32_t index : c->second) {
      const ShaderEntry& e = d.inputs[index];
      const auto proof = me::native::ProveVertexFetchReorder(
          microcode, e.microcode, e.elements, [](const ElementVertex& element) { return element.instruction; });
      if (!proof) {
        continue;
      }
      ++proven;
      // Of several (the library shader, containers repaired for other orders, a register-linked wrap of a dump), a
      // container of the game's own first (its constant table: the same constant upload as the shader in its own
      // order), then the fewest swizzle components the draw path has to remap, then library order. Any of them
      // computes the same (the proof); this only keeps the choice predictable.
      if (!chosen || (chosen->binding_per_register && !e.binding_per_register) ||
          (chosen->binding_per_register == e.binding_per_register && proof.remaps < best.remaps)) {
        chosen = &e;
        best = proof;
      }
    }
  }
  if (chosen && (d.reorder_checks > 0 || sa2::enh::VerifyMode())) {
    // Self-check: the same question by brute force (every placement of the declared triples against the ordered
    // predicate); exactly the proven one may pass.
    if (d.reorder_checks > 0) --d.reorder_checks;
    sa2::enh::VerifyCheck();
    if (!me::native::VertexFetchReorderAgreesWithPlacements(microcode, chosen->microcode, best, d.temporal)) {
      sa2::enh::VerifyDifference();
      REXLOG_ERROR("[native] DIFFERENCE: sa2_gpu_vs_fetch_reorder: the proof for VS n{} ({} words) disagrees with the "
                   "brute-force placements; off from now on", chosen->number, microcode.size());
      g_reorder_off.store(true, std::memory_order_relaxed);
      return nullptr;
    }
  }
  if (chosen && ++d.stats.reordered <= 16) {
    std::string order;
    for (uint32_t k = 0; k < best.count; ++k) {
      order += fmt::format(" {}->{}", best.from[k], best.to[k]);
    }
    REXLOG_INFO("[native] sa2_gpu_vs_fetch_reorder: VS n{} identified with its fetches reordered by Direct3D ({} "
                "words; fetch slots{}; {} swizzle components remapped; {} entries proven)",
                chosen->number, microcode.size(), order, best.remaps, proven);
  }
  d.reorder_cache.emplace(key, chosen);
  return chosen;
}

const ShaderEntry* ShadersNative::IdentifyContainer(
    std::span<const uint8_t> container) const {
  const Data& d = *data_;
  if (!d.loaded) {
    return nullptr;
  }
  const sa2::native::Shader* shader = d.library.Search(container);
  if (!shader) {
    return nullptr;
  }
  const auto it = d.per_shader.find(shader);
  return it != d.per_shader.end() ? &d.inputs[it->second] : nullptr;
}

// Entries are in library order (Load), with gaps where a container was skipped.
const ShaderEntry* ShadersNative::PerNumber(uint32_t number) const {
  const Data& d = *data_;
  if (!d.loaded) {
    return nullptr;
  }
  const auto it = std::lower_bound(d.inputs.begin(), d.inputs.end(), number,
                                   [](const ShaderEntry& e, uint32_t n) { return e.number < n; });
  return it != d.inputs.end() && it->number == number ? &*it : nullptr;
}

const ShaderEntry* ShadersNative::PerFingerprint(uint64_t fingerprint) const {
  const Data& d = *data_;
  if (!d.loaded || !fingerprint) {
    return nullptr;
  }
  const auto it = d.per_fingerprint.find(fingerprint);
  return it != d.per_fingerprint.end() ? &d.inputs[it->second] : nullptr;
}

bool ShadersNative::is_indexed() const {
  return data_->loaded && data_->library.is_indexed();
}

sa2::native::StatsPreload ShadersNative::PreloadSpirv(std::span<const uint64_t> fingerprints,
                                                                   const std::atomic<bool>* stop) const {
  const Data& d = *data_;
  if (!d.loaded || !d.library.is_indexed()) {
    return {};
  }
  std::vector<const sa2::native::Shader*> shaders;
  shaders.reserve(fingerprints.size());
  for (uint64_t fingerprint : fingerprints) {
    if (const ShaderEntry* e = PerFingerprint(fingerprint)) {
      shaders.push_back(e->shader);
    }
  }
  return d.library.Preload(std::move(shaders), stop);
}

const char* UsageName(uint8_t usage) {
  return kUses[usage & 0xF];
}

StatsShaders ShadersNative::Stats() const {
  return data_->stats;
}

}  // namespace sa2::native
