#pragma once

// Deliberately narrow CPU admission adapter, not a general guest interpreter.
// Call on the guest submission thread while the fetch memory cannot be freed.
// Input words are the currently loaded HOST-order program, not a package copy.
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <rex/graphics/pipeline/shader/interpreter.h>
#include <rex/platform.h>
#include <rex/system/xmemory.h>
#include <rex/ui/graphics_util.h>

namespace me::native {

class NativeDrawExtentEstimator {
 public:
  struct Diagnostics {
    std::string reason;  // empty when ReasonAsPointer(true): reason_text holds it then
    // sa2-nx (sa2_cpu_ring_estimator_reason): the reason as the literal it was set from, always set.
    const char* reason_text = "";
    std::string_view Reason() const { return reason.empty() ? std::string_view(reason_text) : std::string_view(reason); }
    std::array<std::array<float, 4>, 3> position{};
    // x/y of `position` are already raster space (the viewport transform was applied here); z/w are raw.
    bool xy_raster = false;
    uint32_t rejected_format = 0;  // vertex format of a fetch-format rejection (diagnostics)
    // sa2-nx: interpolator 0 as each vertex exported it (rectangle lists only; all four components), for a clear
    // whose pixel shader writes interpolator 0 (ProvePixelColor0IsR0).
    std::array<std::array<float, 4>, 3> interpolator0{};
    bool interpolator0_valid = false;
  };
  // Clip enabled with every vertex inside the clip volume (no user planes) clips nothing, and an XY viewport
  // transform is then applied here: full-screen passes written in clip space (the depth restore) qualify.
  void AllowClipInside(bool allow) { allow_clip_inside_ = allow; }
  // Two triangles (6-vertex list or 4-vertex strip, auto-indexed) whose corners are exactly the 4 corners of
  // an axis-aligned rectangle split along one diagonal cover it exactly like a rectangle (fill rule).
  void AllowQuadTriangles(bool allow) { allow_quad_triangles_ = allow; }
  void AllowAlu(bool allow) { allow_alu_ = allow; }
  // sa2-nx: the reason is kept only as a pointer to its literal (Diagnostics::reason_text), not copied into
  // Diagnostics::reason: a heap allocation per call for the reasons over 15 characters, the success one included.
  void ReasonAsPointer(bool on) { reason_pointer_ = on; }

  explicit NativeDrawExtentEstimator(const rex::memory::Memory& memory) : memory_(memory) {}

  // Whether `length` bytes of guest physical memory at `physical` can be read through physical_membase().
  bool PhysicalReadable(uint32_t physical, uint32_t length) const {
    if (!length || physical >= 0x20000000u || length > 0x20000000u - physical) return false;
#if defined(_WIN32) || defined(__linux__) || REX_PLATFORM_SWITCH
    // usf4 0.6.7: the Switch too; its QueryProtect (RexGmQueryProtect) was ~1 % of the console ring thread.
    // VirtualQuery computes the region size by walking the page tables across every page with the same attributes
    // (hundreds of MB of the physical mapping), which was 87 % of the ring thread on PC. The physical heap's page
    // table records the same commit and protection for physical_membase().
    // usf4-nx: on Linux, rex::memory::QueryProtect reads and parses all of /proc/self/maps (thousands of mappings once
    // guest pages are protected one by one): 179 times in 4 s of a fight, most of the ring thread (strace -k,
    // 2026-10-11). Same page-table answer as on Windows.
    auto* heap = const_cast<rex::memory::Memory&>(memory_).GetPhysicalHeap();
    return uint32_t(heap->QueryRangeAccess(physical, physical + length - 1)) &
           uint32_t(rex::memory::PageAccess::kReadOnly);
#else
    // Query the actual physical mapping, not a possibly different virtual alias.
    size_t readable_length = 0;
    rex::memory::PageAccess access;
    return rex::memory::QueryProtect(const_cast<uint8_t*>(memory_.physical_membase() + physical), readable_length,
                                     access) &&
           (uint32_t(access) & uint32_t(rex::memory::PageAccess::kReadOnly)) && readable_length >= length;
#endif
  }

  std::optional<uint32_t> Estimate(std::span<const uint32_t> register_words,
                                   std::span<const uint32_t> host_code,
                                   Diagnostics* diagnostics = nullptr) const {
    using namespace rex::graphics;
    Diagnostics scratch;
    Diagnostics& d = diagnostics ? *diagnostics : scratch;
    d = {};
    auto reject = [&](const char* reason) -> std::optional<uint32_t> {
      d.reason_text = reason;
      if (!reason_pointer_) d.reason = reason;
      return std::nullopt;
    };
    if (register_words.size() < RegisterFile::kRegisterCount) return reject("register-file-short");
    if (host_code.empty() || host_code.size() > 128 || host_code.size() % 3)
      return reject("program-size");
    // The register words already have RegisterFile's layout (one array of kRegisterCount words), and the
    // estimator and ShaderInterpreter only read them: view them in place. Copying the 80 KB register file
    // on every candidate quad was ~5 % of the ring thread (memmove under Estimate in the Switch profile).
    static_assert(sizeof(RegisterFile) == sizeof(uint32_t) * RegisterFile::kRegisterCount,
                  "RegisterFile must be exactly its register array");
    const RegisterFile& regs = *reinterpret_cast<const RegisterFile*>(register_words.data());
    auto draw = regs.Get<reg::VGT_DRAW_INITIATOR>();
    const bool rect = draw.prim_type == xenos::PrimitiveType::kRectangleList && draw.num_indices == 3;
    const bool list = allow_quad_triangles_ && draw.prim_type == xenos::PrimitiveType::kTriangleList &&
                       draw.num_indices == 6;
    const bool strip = allow_quad_triangles_ && draw.prim_type == xenos::PrimitiveType::kTriangleStrip &&
                      draw.num_indices == 4;
    const bool dma = (list || strip) && draw.source_select == xenos::SourceSelect::kDMA;
    if ((draw.source_select != xenos::SourceSelect::kAutoIndex && !dma) || !(rect || list || strip) ||
        regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select == xenos::VGTOutputPath::kTessellationEnable)
      return reject("not-three-auto-rectangle");
    const uint32_t n_vertices = draw.num_indices;
    const auto clip = regs.Get<reg::PA_CL_CLIP_CNTL>();
    const bool clip_on = !clip.clip_disable;
    if (clip_on && (!allow_clip_inside_ || clip.ucp_ena)) return reject("clip-enabled");
    const uint32_t offset = regs.Get<reg::VGT_INDX_OFFSET>().indx_offset;
    const uint32_t minimum = regs.Get<reg::VGT_MIN_VTX_INDX>().min_indx;
    const uint32_t maximum = regs.Get<reg::VGT_MAX_VTX_INDX>().max_indx;
    std::array<uint32_t, 6> indices{};
    for (uint32_t i = 0; i < n_vertices; ++i)
      indices[i] = std::min(maximum, std::max(minimum, (i + offset) & 0xFFFFFF));
    if (dma) {
      // Indexed quad: the indices come from the index buffer (VGT_DMA_BASE, guest endianness).
      const auto dma_size = regs.Get<reg::VGT_DMA_SIZE>();
      const bool i32 = draw.index_size == xenos::IndexFormat::kInt32;
      const uint32_t dma_base = regs.values[XE_GPU_REG_VGT_DMA_BASE] & 0x1FFFFFFFu;
      if (dma_size.num_words < n_vertices || !dma_base) return reject("index-buffer-size");
      const uint8_t* bytes = memory_.physical_membase() + dma_base;
      if (!PhysicalReadable(dma_base, n_vertices * (i32 ? 4u : 2u))) return reject("index-buffer-unreadable");
      for (uint32_t i = 0; i < n_vertices; ++i) {
        uint32_t raw;
        if (i32) {
          std::memcpy(&raw, bytes + i * 4, 4);
          raw = xenos::GpuSwap(raw, dma_size.swap_mode);
        } else {
          uint16_t v16;
          std::memcpy(&v16, bytes + i * 2, 2);
          raw = xenos::GpuSwap(v16, dma_size.swap_mode);
        }
        indices[i] = std::min(maximum, std::max(minimum, (raw + offset) & 0xFFFFFF));
      }
    }

    // Check all reachable CF and instruction addresses BEFORE SDK AnalyzeUcode.
    // No branches, predicates, loops, calls, textures or memexports admitted.
    std::array<uint8_t, 64> defined{};
    std::array<uint8_t, 64> equivalent{};
    defined[0] = 1;  // Only r0.x is initialized by the SDK extent estimator.
    equivalent[0] = 1;
    bool ended = false, has_position = false;
    uint32_t body_start = uint32_t(host_code.size() / 3);
    uint32_t executed = 0;
    std::optional<ucode::VertexFetchInstruction> last_full_fetch;
    bool index_temp_valid = true;
    uint32_t end_cf_index = 0;
    for (uint32_t cf_index = 0; cf_index < host_code.size() / 3 * 2 && !ended; ++cf_index) {
      if (cf_index / 2 >= body_start) return reject("cf-overlaps-body");
      ucode::ControlFlowInstruction pair[2];
      ucode::UnpackControlFlowInstructions(host_code.data() + (cf_index / 2) * 3, pair);
      const auto& cf = pair[cf_index & 1];
      switch (cf.opcode()) {
        case ucode::ControlFlowOpcode::kNop:
        case ucode::ControlFlowOpcode::kMarkVsFetchDone: break;
        case ucode::ControlFlowOpcode::kAlloc:
          if (cf.alloc.alloc_type() == ucode::AllocType::kMemory) return reject("memory-export-alloc");
          break;
        case ucode::ControlFlowOpcode::kExec:
        case ucode::ControlFlowOpcode::kExecEnd: {
          const auto& e = cf.exec;
          body_start = std::min(body_start, e.address());
          if ((cf_index / 2) >= body_start || !e.count() ||
              uint64_t(e.address()) + e.count() > host_code.size() / 3 ||
              (executed += e.count()) > 64) return reject("exec-bounds");
          for (uint32_t j = 0; j < e.count(); ++j) {
            const uint32_t* words = host_code.data() + (e.address() + j) * 3;
            if ((e.sequence() >> (j * 2)) & 1) {
              ucode::VertexFetchInstruction fetch;
              std::memcpy(&fetch, words, sizeof(fetch));
              if (fetch.opcode() != ucode::FetchOpcode::kVertexFetch || fetch.is_predicated() ||
                  fetch.is_dest_relative() || fetch.offset() < 0 || fetch.exp_adjust())
                return reject("fetch-profile");
              if (!fetch.is_mini_fetch()) {
                if (!index_temp_valid || fetch.is_src_relative() || fetch.src() != 0 ||
                    fetch.src_swizzle() != 0 || fetch.fetch_constant_index() >= 96)
                  return reject("full-fetch-index");
                last_full_fetch = fetch;
              } else if (!last_full_fetch) return reject("mini-without-full");
              const bool float4 = fetch.data_format() == xenos::VertexFormat::k_32_32_32_32_FLOAT;
              const bool float3 = fetch.data_format() == xenos::VertexFormat::k_32_32_32_FLOAT;
              // float2 (full-screen quads): z/w must then come from explicit 0/1 selectors (checked below).
              const bool float2 = allow_quad_triangles_ && fetch.data_format() == xenos::VertexFormat::k_32_32_FLOAT;
              if (!float4 && !float3 && !float2) {
                d.rejected_format = uint32_t(fetch.data_format());
                return reject("fetch-format");
              }
              const uint32_t components = float4 ? 4 : float3 ? 3 : 2;
              const auto binding = regs.GetVertexFetch(last_full_fetch->fetch_constant_index());
              if (binding.type != xenos::FetchConstantType::kVertex) return reject("fetch-constant-type");
              const uint64_t base = binding.address;
              const uint64_t end = base + binding.size;
              if (!binding.size || end > (uint64_t(0x20000000) / 4)) return reject("fetch-buffer-bounds");
              for (uint32_t index : indices) {
                const uint64_t first = base + uint64_t(last_full_fetch->stride()) * index + uint32_t(fetch.offset());
                if (first < base || first + components > end) return reject("fetch-index-bounds");
                const auto* bytes = memory_.physical_membase() + first * 4;
                if (!PhysicalReadable(uint32_t(first * 4), components * 4)) return reject("fetch-memory-unreadable");
                for (uint32_t c = 0; c < components; ++c) {
                  uint32_t bits;
                  std::memcpy(&bits, bytes + c * 4, 4);
                  bits = xenos::GpuSwap(bits, binding.endian);
                  if (!std::isfinite(std::bit_cast<float>(bits))) return reject("fetch-nonfinite");
                }
              }
              uint8_t mask = defined[fetch.dest()];
              uint8_t eq = equivalent[fetch.dest()];
              for (uint32_t c = 0; c < 4; ++c) {
                const uint32_t selector = (fetch.dest_swizzle() >> (c * 3)) & 7;
                // SDK initializes float3's unavailable W to zero. Position may
                // use XYZ plus an explicit 0/1 selector, not that missing W.
                if (selector == 6)
                  return reject("fetch-swizzle");
                if (selector != 7) {
                  mask |= uint8_t(1 << c);
                  if (selector < components || selector == 4 || selector == 5) eq |= uint8_t(1 << c);
                  else eq &= uint8_t(~(1 << c));
                }
              }
              defined[fetch.dest()] = mask;
              equivalent[fetch.dest()] = eq;
              if (fetch.dest() == 0 && (mask & 1)) index_temp_valid = false;
            } else {
              ucode::AluInstruction alu;
              std::memcpy(&alu, words, sizeof(alu));
              if (allow_alu_) {
                // Relaxed profile: any non-kill ALU with non-relative constants; the SDK interpreter computes
                // it on the live constant bank. Every temp source must be fully defined.
                const auto vop = uint32_t(alu.vector_opcode()), sop = uint32_t(alu.scalar_opcode());
                const bool scale = alu.scalar_opcode() != ucode::AluScalarOpcode::kRetainPrev;
                if (alu.is_predicated() || alu.is_vector_dest_relative() ||
                    (scale && !alu.is_export() && alu.is_scalar_dest_relative()) ||
                    alu.is_const_0_addressed() || alu.is_const_1_addressed() ||
                    (vop >= 24 && vop <= 27) || (sop >= 35 && sop <= 39)) return reject("alu-profile");
                const uint32_t vmask = alu.GetVectorOpResultWriteMask();
                const uint32_t smask = scale ? alu.GetScalarOpResultWriteMask() : 0u;
                // Only the source components the operations actually read must be defined.
                const auto read = [&](size_t operand, uint32_t comps) -> bool {
                  if (!alu.src_is_temp(operand) || !comps) return true;
                  if (ucode::AluInstruction::is_src_temp_relative(alu.src_reg(operand))) return false;
                  const uint8_t def = defined[ucode::AluInstruction::src_temp_reg(alu.src_reg(operand))];
                  for (uint32_t c = 0; c < 4; ++c)
                    if ((comps >> c) & 1 &&
                        !((def >> ucode::AluInstruction::GetSwizzledComponentIndex(alu.src_swizzle(operand), c)) & 1))
                      return false;
                  return true;
                };
                const auto& vinfo = ucode::GetAluVectorOpcodeInfo(alu.vector_opcode());
                const uint32_t vusado = alu.is_export() ? 15u : vmask;
                for (uint32_t operand = 1; operand <= vinfo.GetOperandCount(); ++operand)
                  if (!read(operand, ucode::GetAluVectorOpNeededSourceComponents(alu.vector_opcode(), operand, vusado)))
                    return reject("alu-undefined-temp");
                if (scale) {
                  const auto& sinfo = ucode::GetAluScalarOpcodeInfo(alu.scalar_opcode());
                  const uint32_t scomps = sinfo.operand_count == 0 ? 0u
                      : sinfo.operand_count == 2 ? 0b0001u
                      : sinfo.single_operand_is_two_component ? 0b1001u : 0b1000u;
                  if (!read(3, scomps)) return reject("alu-undefined-temp");
                }
                if (alu.is_export()) {
                  if (alu.vector_dest() != uint32_t(ucode::ExportRegister::kVSPosition) &&
                      alu.vector_dest() >= 32) return reject("unsupported-export");
                  if (alu.vector_dest() == uint32_t(ucode::ExportRegister::kVSPosition)) {
                    if ((vmask | smask | alu.GetConstant0WriteMask() | alu.GetConstant1WriteMask()) != 15)
                      return reject("partial-position-export");
                    has_position = true;
                  }
                } else {
                  // r0.x is the vertex index: a later full fetch would read the overwritten value (rejected there).
                  if ((alu.vector_dest() == 0 && (vmask & 1)) || (scale && alu.scalar_dest() == 0 && (smask & 1)))
                    index_temp_valid = false;
                  defined[alu.vector_dest()] |= uint8_t(vmask);
                  equivalent[alu.vector_dest()] |= uint8_t(vmask);
                  if (smask) {
                    defined[alu.scalar_dest()] |= uint8_t(smask);
                    equivalent[alu.scalar_dest()] |= uint8_t(smask);
                  }
                }
                continue;
              }
              if (alu.is_predicated() || alu.vector_opcode() != ucode::AluVectorOpcode::kMax ||
                  alu.scalar_opcode() != ucode::AluScalarOpcode::kRetainPrev ||
                  alu.scalar_write_mask() || alu.is_vector_dest_relative() ||
                  alu.is_const_0_addressed() || alu.is_const_1_addressed()) return reject("alu-profile");
              uint8_t eq_result = 15;
              if (alu.GetVectorOpResultWriteMask()) {
                for (size_t operand = 1; operand <= 2; ++operand) {
                  if (alu.src_is_temp(operand)) {
                    if (ucode::AluInstruction::is_src_temp_relative(alu.src_reg(operand)) ||
                        defined[ucode::AluInstruction::src_temp_reg(alu.src_reg(operand))] != 15)
                      return reject("alu-undefined-temp");
                    for (uint32_t c = 0; c < 4; ++c)
                      if (!(equivalent[ucode::AluInstruction::src_temp_reg(alu.src_reg(operand))] &
                            (1 << ucode::AluInstruction::GetSwizzledComponentIndex(alu.src_swizzle(operand), c))))
                        eq_result &= uint8_t(~(1 << c));
                  } else {
                    // The compiled package may bake DEF values unlike the live
                    // guest constant bank. Initial profile avoids that boundary.
                    return reject("alu-constant-operand");
                  }
                }
              }
              if (alu.is_export()) {
                if (alu.vector_dest() != uint32_t(ucode::ExportRegister::kVSPosition) &&
                    alu.vector_dest() >= 32) return reject("unsupported-export");
                if (alu.vector_dest() == uint32_t(ucode::ExportRegister::kVSPosition)) {
                  if ((eq_result & alu.GetVectorOpResultWriteMask()) != alu.GetVectorOpResultWriteMask())
                    return reject("position-undefined-fetch-components");
                  if ((alu.GetVectorOpResultWriteMask() | alu.GetConstant0WriteMask() |
                       alu.GetConstant1WriteMask()) != 15) return reject("partial-position-export");
                  has_position = true;
                }
              } else {
                if (alu.vector_dest() == 0 && alu.vector_write_mask()) return reject("index-temp-overwrite");
                defined[alu.vector_dest()] |= uint8_t(alu.vector_write_mask());
                equivalent[alu.vector_dest()] = uint8_t((equivalent[alu.vector_dest()] &
                    ~alu.vector_write_mask()) | (eq_result & alu.vector_write_mask()));
              }
            }
          }
          ended = cf.opcode() == ucode::ControlFlowOpcode::kExecEnd;
          if (ended) end_cf_index = cf_index;
          break;
        }
        default: return reject("unsafe-control-flow");
      }
    }
    if (!ended || !has_position) return reject("no-end-or-position");
    // AnalyzeUcode visits even unreachable CF padding up to the first body.
    // Admit only inert padding there, so analysis cannot dereference unvalidated
    // exec addresses or build a cyclic predecessor graph.
    for (uint32_t i = end_cf_index + 1; i < body_start * 2; ++i) {
      ucode::ControlFlowInstruction pair[2];
      ucode::UnpackControlFlowInstructions(host_code.data() + (i / 2) * 3, pair);
      if (pair[i & 1].opcode() != ucode::ControlFlowOpcode::kNop &&
          pair[i & 1].opcode() != ucode::ControlFlowOpcode::kMarkVsFetchDone)
        return reject("non-inert-cf-padding");
    }
    // The admission checks above are deliberately stronger than the SDK's
    // CanInterpretShader (which does not bound CF execution). Use the raw
    // HOST-order entry point; do not require the GPU plugin's shader analyzer.
    struct Sink final : ShaderInterpreter::ExportSink {
      std::array<float, 4> value{}, interpolator0{};
      uint32_t mask = 0, mask_interpolator0 = 0;
      void Export(ucode::ExportRegister reg, const float* v, uint32_t m) override {
        if (reg == ucode::ExportRegister::kVSInterpolator0) {
          for (uint32_t c = 0; c < 4; ++c) if (m & (1 << c)) interpolator0[c] = v[c];
          mask_interpolator0 |= m;
          return;
        }
        if (reg != ucode::ExportRegister::kVSPosition) return;
        for (uint32_t c = 0; c < 4; ++c) if (m & (1 << c)) value[c] = v[c];
        mask |= m;
      }
    } sink;
    ShaderInterpreter interpreter(regs, memory_);
    interpreter.SetShader(xenos::ShaderType::kVertex, host_code.data());
    interpreter.SetExportSink(&sink);
    std::array<std::array<float, 4>, 6> vertices{};
    bool interpolator0_all = rect;
    for (uint32_t i = 0; i < n_vertices; ++i) {
      sink = Sink{};
      std::fill_n(interpreter.temp_registers(), 64 * 4, 0.0f);
      interpreter.temp_registers()[0] = float(indices[i]);
      interpreter.Execute();
      if (sink.mask != 15) return reject("missing-position");
      for (float v : sink.value) if (!std::isfinite(v)) return reject("position-nonfinite");
      vertices[i] = sink.value;
      if (rect && i < 3) {
        d.interpolator0[i] = sink.interpolator0;
        interpolator0_all = interpolator0_all && sink.mask_interpolator0 == 15;
      }
    }
    d.interpolator0_valid = interpolator0_all;
    if (rect) {
      for (uint32_t i = 0; i < 3; ++i) d.position[i] = vertices[i];
    } else {
      // Exactly two distinct x and two distinct y, every vertex a corner, equal z and w everywhere.
      float xa = vertices[0][0], xb = xa, ya = vertices[0][1], yb = ya;
      for (uint32_t i = 0; i < n_vertices; ++i) {
        const auto& v = vertices[i];
        if (v[2] != vertices[0][2] || v[3] != vertices[0][3]) return reject("quad-z-w");
        if (v[0] != xa && v[0] != xb) { if (xa != xb) return reject("quad-not-rect"); xb = v[0]; }
        if (v[1] != ya && v[1] != yb) { if (ya != yb) return reject("quad-not-rect"); yb = v[1]; }
      }
      if (xa == xb || ya == yb) return reject("quad-degenerate");
      const auto corner = [&](const std::array<float, 4>& v) { return (v[0] == xb ? 1 : 0) | (v[1] == yb ? 2 : 0); };
      // Each triangle has 3 distinct corners; the corners they lack are opposite (they share a diagonal).
      const uint32_t t0[3] = {0, 1, 2}, t1[3] = {list ? 3u : 1u, list ? 4u : 2u, list ? 5u : 3u};
      int missing[2];
      for (uint32_t t = 0; t < 2; ++t) {
        const uint32_t* tri = t ? t1 : t0;
        uint32_t mask = 0;
        for (uint32_t k = 0; k < 3; ++k) mask |= 1u << corner(vertices[tri[k]]);
        if (std::popcount(mask) != 3) return reject("quad-triangle-degenerate");
        missing[t] = std::countr_zero(~mask & 15u);
      }
      if ((missing[0] ^ missing[1]) != 3) return reject("quad-not-split-on-diagonal");
      const float z = vertices[0][2], w = vertices[0][3];
      d.position[0] = {xa, ya, z, w};
      d.position[1] = {xb, ya, z, w};
      d.position[2] = {xa, yb, z, w};
    }
    // The SDK estimator sees only the three input vertices, not native rectangle
    // expansion's fourth vertex. Exact axis-aligned corners with equal positive
    // W prove that missing corner cannot add a larger Y. No epsilon admission.
    if (!(d.position[0][3] > 0) || d.position[0][3] != d.position[1][3] ||
        d.position[0][3] != d.position[2][3]) return reject("rectangle-w");
    unsigned horizontal = 0, vertical = 0;
    for (uint32_t i = 0; i < 3; ++i) for (uint32_t j = i + 1; j < 3; ++j) {
      if (d.position[i][0] == d.position[j][0] && d.position[i][1] != d.position[j][1]) ++vertical;
      if (d.position[i][1] == d.position[j][1] && d.position[i][0] != d.position[j][0]) ++horizontal;
    }
    if (horizontal != 1 || vertical != 1) return reject("not-axis-aligned-corners");
    std::array<float, 3> length{};
    for (uint32_t i = 0; i < 3; ++i) {
      const float dx = d.position[(i + 1) % 3][0] - d.position[(i + 2) % 3][0];
      const float dy = d.position[(i + 1) % 3][1] - d.position[(i + 2) % 3][1];
      // Volatile splits reproduce the wrapper's individual binary32 operations
      // rather than a host FMA. Reject ties/overflow, don't approximate rotation.
      volatile float xx = dx * dx, yy = dy * dy;
      length[i] = xx + yy;
      if (!std::isfinite(length[i])) return reject("rectangle-distance-range");
    }
    const uint32_t first = length[0] > length[1] && length[0] > length[2] ? 0 :
                            (length[1] > length[2] ? 1 : 2);
    if (!(length[first] > length[(first + 1) % 3] &&
          length[first] > length[(first + 2) % 3])) return reject("rectangle-distance-tie");
    volatile float fourth_delta_y = d.position[(first + 1) % 3][1] - d.position[first][1];
    const float fourth_y = fourth_delta_y + d.position[(first + 2) % 3][1];
    const float min_y = std::min({d.position[0][1], d.position[1][1], d.position[2][1]});
    const float max_y = std::max({d.position[0][1], d.position[1][1], d.position[2][1]});
    if (!std::isfinite(fourth_y) || fourth_y < min_y || fourth_y > max_y)
      return reject("rectangle-fourth-y-outside");
    const auto vte = regs.Get<reg::PA_CL_VTE_CNTL>();
    const bool vport = vte.vport_x_scale_ena || vte.vport_x_offset_ena ||
                       vte.vport_y_scale_ena || vte.vport_y_offset_ena;
    if (vport && !(allow_clip_inside_ && vte.vport_x_scale_ena && vte.vport_x_offset_ena &&
                   vte.vport_y_scale_ena && vte.vport_y_offset_ena)) return reject("xy-viewport-transform");
    // Clipping: Z must be inside (a constant-Z rectangle is then either whole or gone); X/Y past the clip
    // planes only crop the rectangle to the viewport box (applied after the viewport transform below).
    bool clip_xy = false;
    if (clip_on) {
      for (const auto& p : d.position) {
        const float w = p[3];
        if (!(w > 0) || (clip.dx_clip_space_def ? (p[2] < 0 || p[2] > w) : std::abs(p[2]) > w))
          return reject("clip-outside");
        if (std::abs(p[0]) > w || std::abs(p[1]) > w) clip_xy = true;
      }
      if (clip_xy && !vport) return reject("clip-outside");
    }
    if (vport) {
      const auto f = [&](uint32_t r) { return std::bit_cast<float>(regs.values[r]); };
      const float xs = f(XE_GPU_REG_PA_CL_VPORT_XSCALE), xo = f(XE_GPU_REG_PA_CL_VPORT_XOFFSET);
      const float ys = f(XE_GPU_REG_PA_CL_VPORT_YSCALE), yo = f(XE_GPU_REG_PA_CL_VPORT_YOFFSET);
      for (auto& p : d.position) {
        const float x = vte.vtx_xy_fmt ? p[0] : p[0] / p[3];
        const float y = vte.vtx_xy_fmt ? p[1] : p[1] / p[3];
        p[0] = x * xs + xo;
        p[1] = y * ys + yo;
        if (!std::isfinite(p[0]) || !std::isfinite(p[1])) return reject("viewport-nonfinite");
        if (clip_xy) {
          // The clip planes x = ±w / y = ±w are the viewport box edges in raster space. With D3D pixel centers
          // on integers the box [a, b) passes through centers; the top-left rule takes the left/top ones, so it
          // covers exactly the pixels a..b-1, i.e. [a - 0.5, b - 0.5] in the proofs' pixel-edge convention.
          const float off = regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero ? 0.5f : 0.0f;
          p[0] = std::clamp(p[0], xo - std::abs(xs) - off, xo + std::abs(xs) - off);
          p[1] = std::clamp(p[1], yo - std::abs(ys) - off, yo + std::abs(ys) - off);
        }
      }
      d.xy_raster = true;
    }
    float raster_max_y = -8192.0f;
    for (const auto& p : d.position) {
      const float y = vte.vtx_xy_fmt || d.xy_raster ? p[1] : p[1] / p[3];
      if (!std::isfinite(y) || std::abs(y) > 8192) return reject("raster-y-range");
      raster_max_y = std::max(raster_max_y, y);
    }
    // Raster conversion derived from SDK DrawExtentEstimator::EstimateVertexMaxY
    // (Xenia, Copyright 2022 Ben Vanik; BSD 3-Clause, SDK LICENSE). The same
    // exported FloatToD3D11Fixed16p8 helper preserves its round-to-nearest-even
    // conversion. Three post-VS corners suffice only after the fourth-Y proof.
    // No SDK estimator instance / GPU cvars / second execution are needed.
    int32_t max_y_24p8 = rex::ui::FloatToD3D11Fixed16p8(raster_max_y);
    if (regs.Get<reg::PA_SU_VTX_CNTL>().pix_center == xenos::PixelCenter::kD3DZero)
      max_y_24p8 += 128;
    if (regs.Get<reg::PA_SU_SC_MODE_CNTL>().vtx_window_offset_enable)
      max_y_24p8 += regs.Get<reg::PA_SC_WINDOW_OFFSET>().window_y_offset * 256;
    // Top-left rule: .5 exclusive for 1x, 1.0 exclusive for multisampling.
    const uint32_t result = (uint32_t(std::max(int32_t(0), max_y_24p8)) +
        (regs.Get<reg::RB_SURFACE_INFO>().msaa_samples == xenos::MsaaSamples::k1X ? 127u : 255u)) >> 8;
    if (result > 8192) return reject("sdk-estimate-range");
    d.reason_text = "axis-aligned-sdk-estimate";
    if (!reason_pointer_) d.reason = d.reason_text;
    return result;
  }

 private:
  const rex::memory::Memory& memory_;
  bool allow_clip_inside_ = false;
  bool allow_quad_triangles_ = false;
  bool allow_alu_ = false;
  bool reason_pointer_ = false;
};

}  // namespace me::native
