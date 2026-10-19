#pragma once

#include "me_depth.h"
#include <cmath>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace me::native {

// CPU contract of the integer operations emitted below. NaN/nonpositive ->0;
// +infinity saturates to the largest finite FLOAT24, not guest 1.0.
inline uint32_t PackDepthQuantize20e4(float guest, bool round_float24) {
  if (!(guest > 0.0f)) return 0;
  uint32_t bits = std::bit_cast<uint32_t>(guest);
  if (bits >= 0x3FFFFFF8u) return 0xFFFFFF;
  if (bits < 0x38800000u)
    bits = ((bits & 0x7FFFFFu) | 0x800000u) >>
           std::min(113u - (bits >> 23), 24u);
  else bits += 0xC8000000u;
  if (round_float24) bits += 3u + ((bits >> 3) & 1u);
  return (bits >> 3) & 0xFFFFFFu;
}

inline float QuantizeIncomingDepthCPU(float host, bool round_float24 = false) {
  return DecodeDepthFloat20e4(PackDepthQuantize20e4(host * 2.0f, round_float24)) * 0.5f;
}
inline float QuantizeDepthHalfCpu(float host, bool round_float24 = false) {
  return QuantizeIncomingDepthCPU(host, round_float24);
}

inline std::vector<uint32_t> MakeDepthOnlyFragmentForQuantization() {
  return {0x07230203, 0x00010300, 0, 5, 0,
      (2u << 16) | 17, 1,  // Shader.
      (3u << 16) | 14, 0, 1,  // Logical GLSL450.
      (5u << 16) | 15, 4, 3, 0x6E69616D, 0,
      (3u << 16) | 16, 3, 7,  // OriginUpperLeft.
      (2u << 16) | 19, 1,
      (3u << 16) | 33, 2, 1,
      (5u << 16) | 54, 1, 3, 0, 2,
      (2u << 16) | 248, 4,
      (1u << 16) | 253,
      (1u << 16) | 56};
}

// Experimental late-Z quantization of raster depth, AFTER half-range composition.
// Existing FragCoord reads remain byte-for-byte unchanged. Injected loads read
// raw host FragCoord.Z, not the guest-space values produced by existing PS code.
// Scope: one Fragment entry, direct scalar/vector BuiltIns. Existing direct
// FragDepth stores (including helpers) are quantized instead of injecting raster
// depth. Reads/derived pointers/escapes of that output are unsupported. Without
// an existing output, entry returns get stores; helper returns/kills do not.
// Optional native2x raster reconstruction is dormant until hardware MSAA is
// enabled by the caller. It captures raw center Z + fine derivatives in the
// first entry block, before any material divergence, then reconstructs standard
// host sample0(+.25,+.25) / sample1(-.25,-.25). It never changes material
// FragCoord reads or interpolator decorations. Explicit guest FragDepth stores
// retain the existing per-fragment quantizer, without forced sample shading.
inline bool TransformDepthQuantizeIncoming(const std::vector<uint32_t>& input,
    std::vector<uint32_t>& output, std::string& reason, bool round_float24 = false,
    bool native2x_raster = false) {
  if (&input == &output) { reason = "input/output alias"; return false; }
  output.clear(); reason.clear();
  const auto fail = [&](const char* why) { output.clear(); reason = why; return false; };
  if (input.size() < 5 || input[0] != 0x07230203 || !input[3] || input[4])
    return fail("invalid SPIR-V header");
  struct Ins { size_t at; uint32_t wc, op; };
  std::vector<Ins> instructions;
  std::map<uint32_t, uint32_t> floats, builtins;
  std::map<uint32_t, uint32_t> integer_widths;
  struct Vec { uint32_t type, count; };
  struct Ptr { uint32_t storage, type; };
  struct Var { uint32_t type, storage, count; };
  std::map<uint32_t, Vec> vectors;
  std::map<uint32_t, Ptr> pointers;
  std::map<uint32_t, Var> variables;
  uint32_t entry = 0, coord = 0, float_type = 0, vector_type = 0;
  uint32_t existing_uint = 0, existing_bool = 0;
  uint32_t existing_depth = 0;
  uint32_t sample_id = 0, sample_id_type = 0;
  bool sample_rate_capability = false, derivative_control_capability = false;
  bool depth_replacing = false;
  std::vector<uint32_t> entry_interfaces;
  size_t first_type = input.size(), first_function = input.size(), execution_end = 0;
  size_t capability_end = input.size();
  uint32_t function = 0, returns = 0;
  size_t entry_body = input.size();
  bool entry_first_block = false, entry_prologue = false;
  bool entry_found = false;
  for (size_t at = 5; at < input.size();) {
    const uint32_t wc = input[at] >> 16, op = input[at] & 65535;
    if (!wc || wc > input.size() - at) return fail("malformed instruction");
    const auto* w = input.data() + at;
    instructions.push_back({at, wc, op});
    if (op == 17 && wc == 2 && w[1] == 35) sample_rate_capability = true;
    if (op == 17 && wc == 2 && w[1] == 51) derivative_control_capability = true;
    if (op != 17 && capability_end == input.size()) capability_end = at;
    if (op == 15) {
      if (wc < 4 || entry || w[1] != 4) return fail("requires one Fragment entry");
      // Validate the entry name terminator before appending interface IDs.
      bool terminated = false;
      uint32_t name_end = 3;
      for (; name_end < wc && !terminated; ++name_end)
        for (uint32_t b = 0; b < 4; ++b) if (!((w[name_end] >> (b * 8)) & 255)) terminated = true;
      if (!terminated) return fail("unterminated entry name");
      entry_interfaces.assign(w + name_end, w + wc);
      entry = w[2];
    }
    if (op == 15 || op == 16 || op == 331) execution_end = at + wc;
    if (op == 16 && wc >= 3 && w[2] == 12) depth_replacing = true;
    if ((op == 16 || op == 331) && wc >= 3 &&
        (w[2] == 14 || w[2] == 15 || w[2] == 16))
      return fail("existing depth execution mode unsupported");
    if (op == 22 && wc == 3) floats[w[1]] = w[2];
    if (op == 20 && wc == 2 && !existing_bool) existing_bool = w[1];
    if (op == 21 && wc == 4 && w[2] == 32 && w[3] == 0 && !existing_uint) existing_uint = w[1];
    if (op == 21 && wc == 4) integer_widths[w[1]] = w[2];
    if (op == 23 && wc == 4) vectors[w[1]] = {w[2], w[3]};
    if (op == 32 && wc == 4) pointers[w[1]] = {w[2], w[3]};
    if (op == 71 && wc >= 4 && w[2] == 11) {
      if (wc != 4) return fail("malformed BuiltIn");
      builtins[w[1]] = w[3];
      if (w[3] == 22) {
        if (existing_depth) return fail("multiple FragDepth variables");
        existing_depth = w[1];
      }
      if (w[3] == 18) {
        if (sample_id) return fail("multiple SampleId variables");
        sample_id = w[1];
      }
    }
    if (op == 72 && wc >= 5 && w[3] == 11 &&
        (w[4] == 15 || w[4] == 22 || (native2x_raster && w[4] == 18)))
      return fail("depth BuiltIn blocks unsupported");
    if (op == 74 || op == 75) return fail("group decorations unsupported");
    if (op >= 19 && op <= 39 && first_type == input.size()) first_type = at;
    if (op == 54) {
      if (wc != 5 || function) return fail("malformed function");
      function = w[2];
      if (first_function == input.size()) first_function = at;
      if (function == entry) {
        if (entry_found) return fail("duplicate entry function");
        entry_found = true;
      }
    }
    if (op == 59 && !function) {
      if (wc < 4) return fail("malformed variable");
      variables[w[2]] = {w[1], w[3], wc};
    }
    if (function == entry && op == 248 && !entry_first_block) {
      if (wc != 2) return fail("malformed entry label");
      entry_first_block = entry_prologue = true;
    } else if (function == entry && entry_prologue && op != 59 && op != 8 && op != 317) {
      // Function OpVariables must remain at the start of the entry block.
      // Inject before the first executable instruction, including branch/kill.
      entry_body = at;
      entry_prologue = false;
    }
    if (op == 253 && function == entry) {
      if (wc != 1) return fail("malformed return");
      ++returns;
    }
    if (op == 254 && function == entry) return fail("entry ReturnValue unsupported");
    if (op == 56) {
      if (!function || wc != 1) return fail("malformed function end");
      function = 0;
    }
    at += wc;
  }
  if (!entry || !entry_found || function || (!returns && !existing_depth) || first_type == input.size())
    return fail("missing entry/types/returns");
  if (existing_depth) {
    if (!variables.count(existing_depth)) return fail("FragDepth not direct variable");
    const auto v = variables[existing_depth];
    if (v.count != 4 || v.storage != 3 || !pointers.count(v.type) ||
        pointers[v.type].storage != 3 || !floats.count(pointers[v.type].type) ||
        floats[pointers[v.type].type] != 32) return fail("FragDepth not Output float32");
    float_type = pointers[v.type].type;
    uint32_t stores = 0;
    for (const auto& ins : instructions) if (ins.at >= first_function) {
      const auto* w = input.data() + ins.at;
      if (ins.op == 8 || ins.op == 317) continue;
      for (uint32_t k = 1; k < ins.wc; ++k) {
        if ((ins.op == 79 && k >= 5) || (ins.op == 81 && k >= 4) ||
            (ins.op == 82 && k >= 5) || (ins.op == 12 && k == 4) ||
            (ins.op == 54 && k == 3) || (ins.op == 246 && k >= 3) ||
            (ins.op == 247 && k == 2) || (ins.op == 250 && k >= 4) ||
            (ins.op == 251 && k >= 3 && (k & 1)) ||
            // OpVariable StorageClass is a literal. Function == 7 must not
            // alias a FragDepth result ID of 7. Its optional initializer (4)
            // remains an ID operand and must still be checked for escapes.
            (ins.op == 59 && k == 3) ||
            (ins.op == 61 && k >= 4) || (ins.op == 62 && k >= 3)) continue;
        if (w[k] != existing_depth) continue;
        if (ins.op != 62 || k != 1 || ins.wc < 3)
          return fail("FragDepth pointer read/escape/derived use unsupported");
        ++stores;
      }
    }
    if (!stores) return fail("existing FragDepth has no stores");
    returns = stores;  // Also bound ID allocation for helper-function stores.
  } else if (depth_replacing) return fail("DepthReplacing without FragDepth");
  const bool reconstruct_samples = native2x_raster && !existing_depth;
  if (reconstruct_samples && entry_body == input.size())
    return fail("missing entry-block insertion point");
  if (reconstruct_samples && sample_id) {
    if (!variables.count(sample_id)) return fail("SampleId not direct variable");
    const auto v = variables[sample_id];
    if (v.count != 4 || v.storage != 1 || !pointers.count(v.type) ||
        pointers[v.type].storage != 1 || !integer_widths.count(pointers[v.type].type) ||
        integer_widths[pointers[v.type].type] != 32) return fail("SampleId not Input int32");
    sample_id_type = pointers[v.type].type;
  }
  for (const auto& [id, builtin] : builtins) if (builtin == 15) {
    if (coord || !variables.count(id)) return fail("FragCoord not one direct variable");
    const auto v = variables[id];
    if (v.storage != 1 || !pointers.count(v.type) || pointers[v.type].storage != 1)
      return fail("FragCoord not Input");
    const uint32_t type = pointers[v.type].type;
    if (!vectors.count(type) || vectors[type].count != 4 ||
        !floats.count(vectors[type].type) || floats[vectors[type].type] != 32)
      return fail("FragCoord not vec4<float32>");
    if (float_type && float_type != vectors[type].type) return fail("incompatible float types");
    coord = id; vector_type = type; float_type = vectors[type].type;
  }
  if (!float_type) for (const auto& [id, width] : floats)
    if (width == 32) { float_type = id; break; }
  if (!vector_type && float_type) for (const auto& [id, v] : vectors)
    if (v.type == float_type && v.count == 4) { vector_type = id; break; }
  if (uint64_t(input[3]) + uint64_t(returns) * 40 + 80 >= UINT32_MAX)
    return fail("ID bound overflow");
  uint32_t next = input[3];
  const bool new_float = !float_type;
  if (new_float) float_type = next++;
  const bool new_vector = !vector_type;
  if (new_vector) vector_type = next++;
  const uint32_t uint_type = existing_uint ? existing_uint : next++;
  const uint32_t bool_type = existing_bool ? existing_bool : next++;
  if (!sample_id_type) sample_id_type = uint_type;
  uint32_t ptr_input = 0, ptr_output = 0;
  for (const auto& [id, p] : pointers) {
    if (p.storage == 1 && p.type == vector_type && !ptr_input) ptr_input = id;
    if (p.storage == 3 && p.type == float_type && !ptr_output) ptr_output = id;
  }
  const bool new_ptr_input = !ptr_input, new_ptr_output = !ptr_output;
  if (new_ptr_input) ptr_input = next++;
  if (new_ptr_output) ptr_output = next++;
  uint32_t ptr_sample_input = 0;
  const bool new_sample_id = reconstruct_samples && !sample_id;
  if (new_sample_id) {
    for (const auto& [id, p] : pointers)
      if (p.storage == 1 && p.type == sample_id_type) { ptr_sample_input = id; break; }
  }
  const bool new_ptr_sample_input = new_sample_id && !ptr_sample_input;
  if (new_ptr_sample_input) ptr_sample_input = next++;
  if (new_sample_id) sample_id = next++;
  const bool add_sample_interface = reconstruct_samples &&
      std::find(entry_interfaces.begin(), entry_interfaces.end(), sample_id) == entry_interfaces.end();
  const bool new_coord = !coord && !existing_depth;
  if (new_coord) coord = next++;
  const bool add_coord_interface = !existing_depth &&
      std::find(entry_interfaces.begin(), entry_interfaces.end(), coord) == entry_interfaces.end();
  const uint32_t depth = existing_depth ? existing_depth : next++;
  const bool add_depth_interface = std::find(entry_interfaces.begin(), entry_interfaces.end(), depth) == entry_interfaces.end();
  std::vector<uint32_t> declarations;
  const auto emit = [](std::vector<uint32_t>& dst, uint32_t op,
                      std::initializer_list<uint32_t> args) {
    dst.push_back((uint32_t(args.size() + 1) << 16) | op);
    dst.insert(dst.end(), args);
  };
  if (new_float) emit(declarations, 22, {float_type, 32});
  if (new_vector) emit(declarations, 23, {vector_type, float_type, 4});
  if (!existing_uint) emit(declarations, 21, {uint_type, 32, 0});
  if (!existing_bool) emit(declarations, 20, {bool_type});
  if (new_ptr_input) emit(declarations, 32, {ptr_input, 1, vector_type});
  if (new_ptr_output) emit(declarations, 32, {ptr_output, 3, float_type});
  if (new_ptr_sample_input) emit(declarations, 32, {ptr_sample_input, 1, sample_id_type});
  std::map<uint32_t, uint32_t> integers;
  for (uint32_t value : {0u, 1u, 3u, 8u, 23u, 24u, 113u, 0x7FFFFFu,
                         0x800000u, 0xFFFFFFu, 0x100000u, 0xC8000000u,
                         0x38800000u, 0x3FFFFFF8u, 0x38000000u}) {
    integers[value] = next++;
    emit(declarations, 43, {uint_type, integers[value], value});
  }
  const uint32_t zero = next++, twice = next++, half = next++, subnormal_unit = next++;
  emit(declarations, 43, {float_type, zero, 0});
  emit(declarations, 43, {float_type, twice, 0x40000000});
  emit(declarations, 43, {float_type, half, 0x3F000000});
  emit(declarations, 43, {float_type, subnormal_unit, 0x2E800000});  // 2^-34.
  const uint32_t quarter = reconstruct_samples ? next++ : 0;
  const uint32_t negative_quarter = reconstruct_samples ? next++ : 0;
  if (reconstruct_samples) {
    emit(declarations, 43, {float_type, quarter, 0x3E800000});
    emit(declarations, 43, {float_type, negative_quarter, 0xBE800000});
  }
  if (new_sample_id) emit(declarations, 59, {ptr_sample_input, sample_id, 1});
  if (new_coord) emit(declarations, 59, {ptr_input, coord, 1});
  if (!existing_depth) emit(declarations, 59, {ptr_output, depth, 3});
  std::vector<uint32_t> capture, precise_ids;
  uint32_t captured_host = 0;
  if (reconstruct_samples) {
    const auto unary = [&](uint32_t op, uint32_t type, uint32_t value) {
      const uint32_t id = next++; emit(capture, op, {type, id, value}); return id;
    };
    const auto binary = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
      const uint32_t id = next++; emit(capture, op, {type, id, a, b}); return id;
    };
    const uint32_t raw_coord = unary(61, vector_type, coord);
    const uint32_t center = next++; emit(capture, 81, {float_type, center, raw_coord, 2});
    const uint32_t dx = unary(210, float_type, center);  // OpDPdxFine.
    const uint32_t dy = unary(211, float_type, center);  // OpDPdyFine.
    uint32_t sample = unary(61, sample_id_type, sample_id);
    if (sample_id_type != uint_type) sample = unary(124, uint_type, sample);
    const uint32_t is_zero = binary(170, bool_type, sample, integers.at(0));
    const uint32_t offset = next++;
    emit(capture, 169, {float_type, offset, is_zero, quarter, negative_quarter});
    const uint32_t dx_offset = binary(133, float_type, dx, offset);
    const uint32_t dy_offset = binary(133, float_type, dy, offset);
    const uint32_t delta = binary(129, float_type, dx_offset, dy_offset);
    captured_host = binary(129, float_type, center, delta);
    precise_ids = {dx_offset, dy_offset, delta, captured_host};
  }
  output.assign(input.begin(), input.begin() + 5);
  function = 0;
  for (const auto& ins : instructions) {
    const auto* w = input.data() + ins.at;
    if (ins.at == capability_end && reconstruct_samples) {
      // Capabilities precede extensions/imports as well as the memory model.
      if (!sample_rate_capability) emit(output, 17, {35});
      if (!derivative_control_capability) emit(output, 17, {51});
    }
    if (ins.at == execution_end && !depth_replacing) emit(output, 16, {entry, 12});
    if (ins.at == first_type) {
      if (new_coord) emit(output, 71, {coord, 11, 15});
      if (!existing_depth) emit(output, 71, {depth, 11, 22});
      if (new_sample_id) {
        emit(output, 71, {sample_id, 11, 18});
        emit(output, 71, {sample_id, 14});  // Flat integer Fragment input.
      }
      for (uint32_t id : precise_ids) emit(output, 71, {id, 42});  // NoContraction.
    }
    if (ins.at == first_function) output.insert(output.end(), declarations.begin(), declarations.end());
    if (reconstruct_samples && ins.at == entry_body)
      output.insert(output.end(), capture.begin(), capture.end());
    if (ins.op == 54) function = w[2];
    const bool quantize_store = existing_depth && ins.op == 62 && ins.wc >= 3 && w[1] == depth;
    if (quantize_store || (!existing_depth && ins.op == 253 && function == entry)) {
      const auto unary = [&](uint32_t op, uint32_t type, uint32_t value) {
        const uint32_t id = next++; emit(output, op, {type, id, value}); return id;
      };
      const auto binary = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
        const uint32_t id = next++; emit(output, op, {type, id, a, b}); return id;
      };
      const auto select = [&](uint32_t type, uint32_t cond, uint32_t a, uint32_t b) {
        const uint32_t id = next++; emit(output, 169, {type, id, cond, a, b}); return id;
      };
      const auto u = [&](uint32_t v) { return integers.at(v); };
      uint32_t host = quantize_store ? w[2] : captured_host;
      if (!quantize_store && !reconstruct_samples) {
        const uint32_t loaded = unary(61, vector_type, coord);
        host = next++; emit(output, 81, {float_type, host, loaded, 2});
      }
      const uint32_t guest = binary(133, float_type, host, twice);
      const uint32_t bits = unary(124, uint_type, guest);
      const uint32_t exponent = binary(194, uint_type, bits, u(23));
      const uint32_t shift = binary(130, uint_type, u(113), exponent);
      const uint32_t limited = select(uint_type,
          binary(176, bool_type, shift, u(24)), shift, u(24));
      const uint32_t mantissa = binary(197, uint_type,
          binary(199, uint_type, bits, u(0x7FFFFF)), u(0x800000));
      const uint32_t denorm = binary(194, uint_type, mantissa, limited);
      const uint32_t normal = binary(128, uint_type, bits, u(0xC8000000));
      uint32_t encoded = select(uint_type,
          binary(176, bool_type, bits, u(0x38800000)), denorm, normal);
      if (round_float24) encoded = binary(128, uint_type, encoded,
          binary(128, uint_type, u(3),
                 binary(199, uint_type, binary(194, uint_type, encoded, u(3)), u(1))));
      uint32_t code = binary(199, uint_type, binary(194, uint_type, encoded, u(3)), u(0xFFFFFF));
      code = select(uint_type, binary(174, bool_type, bits, u(0x3FFFFFF8)), u(0xFFFFFF), code);
      code = select(uint_type, binary(186, bool_type, guest, zero), code, u(0));
      const uint32_t sub_float = binary(133, float_type, unary(112, float_type, code), subnormal_unit);
      const uint32_t normal_float = unary(124, float_type,
          binary(128, uint_type, binary(196, uint_type, code, u(3)), u(0x38000000)));
      const uint32_t decoded = select(float_type,
          binary(176, bool_type, code, u(0x100000)), sub_float, normal_float);
      const uint32_t result = binary(133, float_type, decoded, half);
      if (quantize_store) {
        const size_t at = output.size(); output.insert(output.end(), w, w + ins.wc);
        output[at + 2] = result;
      } else emit(output, 62, {depth, result});
    }
    if (ins.op == 15) {
      const size_t at = output.size();
      output.insert(output.end(), w, w + ins.wc);
      if (add_coord_interface) output.push_back(coord);
      if (add_depth_interface) output.push_back(depth);
      if (add_sample_interface) output.push_back(sample_id);
      output[at] = (uint32_t(ins.wc + uint32_t(add_depth_interface) + uint32_t(add_coord_interface) +
                            uint32_t(add_sample_interface)) << 16) | 15;
    } else if (!quantize_store && !((ins.op == 16 || ins.op == 331) && ins.wc >= 3 && w[1] == entry && w[2] == 9))
      output.insert(output.end(), w, w + ins.wc);
    if (ins.op == 56) function = 0;
  }
  output[3] = next;
  return true;
}

}  // namespace me::native
