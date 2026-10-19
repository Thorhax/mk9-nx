#pragma once

#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace me::native {

// Convert a fragment module's depth interface from guest [0,2) to host [0,1).
// No new depth quantization is introduced. Inline stores preserve kill paths,
// helper functions and multiple returns. This must be composed ONCE per module,
// after / before independent color transforms, never applied to an already
// half-range module. Failure clears output; callers must not use the original PS.
// Supported pointers: direct BuiltIn variables, constant FragCoord components,
// CopyObject. Input/Output pointer Phi/Select are not Vulkan logical pointers.
// BuiltIn blocks, dynamic access,
// pointer escape / copy-memory / atomics are deliberately rejected.
inline bool TransformDepthHalf(const std::vector<uint32_t>& input,
                               std::vector<uint32_t>& output, std::string& reason) {
  // An aliased input/output vector would be cleared before it can be inspected.
  if (&input == &output) { reason = "input/output alias"; return false; }
  output.clear(); reason.clear();
  const auto fail = [&](const char* why) { output.clear(); reason = why; return false; };
  if (input.size() < 5 || input[0] != 0x07230203 || !input[3] || input[4])
    return fail("invalid SPIR-V header");
  struct Instruction { size_t at; uint32_t count, op; };
  std::vector<Instruction> instructions;
  std::map<uint32_t, uint32_t> floats, integers, vectors, builtins, constants;
  struct PointerType { uint32_t storage, type; };
  std::map<uint32_t, PointerType> pointer_types;
  struct Variable { uint32_t type, storage, count; };
  std::map<uint32_t, Variable> variables;
  uint32_t entry = 0, coord = 0, depth = 0, float_type = 0, vector_type = 0;
  bool depth_replacing = false, in_function = false;
  size_t first_function = input.size(), first_type = input.size(), execution_end = 0;
  for (size_t at = 5; at < input.size();) {
    const uint32_t count = input[at] >> 16, op = input[at] & 65535;
    if (!count || count > input.size() - at) return fail("malformed instruction");
    instructions.push_back({at, count, op});
    const auto* w = input.data() + at;
    if (op == 15) {
      if (count < 4 || entry || w[1] != 4) return fail("requires one fragment entry point");
      entry = w[2];
    } else if (op == 16 && count >= 3 && w[2] == 12) {
      depth_replacing = true;
    } else if (op == 22 && count == 3) floats[w[1]] = w[2];
    else if (op == 21 && count == 4) integers[w[1]] = w[2];
    else if (op == 23 && count == 4) vectors[w[1]] = w[2] | (w[3] << 24);
    else if (op == 32 && count == 4) pointer_types[w[1]] = {w[2], w[3]};
    else if (op == 43 && count == 4 && integers.count(w[1]) && integers[w[1]] == 32)
      constants[w[2]] = w[3];
    else if (op == 71 && count >= 4 && w[2] == 11) {
      if (count != 4) return fail("malformed BuiltIn decoration");
      builtins[w[1]] = w[3];
    } else if (op == 72 && count >= 5 && w[3] == 11 && (w[4] == 15 || w[4] == 22))
      return fail("FragCoord/FragDepth BuiltIn blocks unsupported");
    else if (op == 74 || op == 75) return fail("group decorations unsupported");
    if (op == 15 || op == 16 || op == 331) execution_end = at + count;
    if (op >= 19 && op <= 39 && first_type == input.size()) first_type = at;
    if (op == 54) { in_function = true; if (first_function == input.size()) first_function = at; }
    if (op == 59 && !in_function) {
      if (count < 4) return fail("malformed global variable");
      variables[w[2]] = {w[1], w[3], count};
    }
    at += count;
  }
  if (!entry || first_function == input.size() || first_type == input.size())
    return fail("missing fragment entry point/types/functions");
  enum Kind : uint32_t { CoordVector, CoordX, CoordY, CoordZ, CoordW, DepthScalar };
  std::map<uint32_t, Kind> pointers;
  for (const auto& [id, builtin] : builtins) {
    if (builtin != 15 && builtin != 22) continue;
    const auto variable = variables.find(id);
    if (variable == variables.end()) return fail("depth BuiltIn is not a direct variable");
    const auto pt = pointer_types.find(variable->second.type);
    if (pt == pointer_types.end()) return fail("missing BuiltIn pointer type");
    const auto type = pt->second.type;
    if (builtin == 15) {
      if (coord || variable->second.storage != 1 || pt->second.storage != 1 ||
          !vectors.count(type) || (vectors[type] >> 24) != 4 ||
          !floats.count(vectors[type] & 0xFFFFFF) || floats[vectors[type] & 0xFFFFFF] != 32)
        return fail("FragCoord must be one Input vec4<float32>");
      coord = id; vector_type = type;
      const uint32_t scalar = vectors[type] & 0xFFFFFF;
      if (float_type && float_type != scalar) return fail("incompatible float types");
      float_type = scalar; pointers[id] = CoordVector;
    } else {
      if (depth || variable->second.storage != 3 || pt->second.storage != 3 ||
          !floats.count(type) || floats[type] != 32 || variable->second.count != 4)
        return fail("FragDepth must be one uninitialized Output float32");
      if (float_type && float_type != type) return fail("incompatible float types");
      depth = id; float_type = type; pointers[id] = DepthScalar;
    }
  }
  // No depth interface: the exact original is already a valid half-range PS.
  if (!coord && !depth) { output = input; return true; }
  const auto kind_type = [&](Kind k) { return k == CoordVector ? vector_type : float_type; };
  const auto valid_pointer = [&](uint32_t type, Kind k) {
    const auto pt = pointer_types.find(type);
    return pt != pointer_types.end() && pt->second.type == kind_type(k) &&
           pt->second.storage == (k == DepthScalar ? 3u : 1u);
  };
  // Fixed-point propagation permits CopyObject chains independent of ID order.
  bool changed = true;
  for (size_t pass = 0; changed && pass <= instructions.size(); ++pass) {
    changed = false;
    for (const auto& ins : instructions) {
      if (ins.at < first_function) continue;
      const auto* w = input.data() + ins.at;
      if (ins.op == 65 || ins.op == 66) {
        if (ins.count < 4 || !pointers.count(w[3])) continue;
        Kind kind = pointers[w[3]];
        if (ins.count != 4) {
          if (kind != CoordVector || ins.count != 5 || !constants.count(w[4]) || constants[w[4]] > 3)
            return fail("dynamic or nested depth interface access unsupported");
          kind = Kind(CoordX + constants[w[4]]);
        }
        if (!valid_pointer(w[1], kind)) return fail("incompatible derived depth pointer");
        if (!pointers.count(w[2])) { pointers[w[2]] = kind; changed = true; }
      } else if (ins.op == 83 && ins.count == 4 && pointers.count(w[3])) {
        if (!valid_pointer(w[1], pointers[w[3]])) return fail("incompatible copied depth pointer");
        if (!pointers.count(w[2])) { pointers[w[2]] = pointers[w[3]]; changed = true; }
      }
    }
  }
  // Reject every unhandled use of a tracked pointer. Unknown instruction operands
  // are conservatively treated as IDs (may reject a colliding literal); never
  // silently leave a reachable unconverted load/store behind.
  for (const auto& ins : instructions) {
    if (ins.at < first_function) continue;
    const auto* w = input.data() + ins.at;
    if (ins.op == 8 || ins.op == 317) continue;  // OpLine / OpNoLine: debug literals.
    for (uint32_t k = 1; k < ins.count; ++k) {
      // Literal operands are not IDs. DXC assigns low IDs to BuiltIns; in
      // particular FragCoord ID 3 must not collide with a shuffle's W index.
      if ((ins.op == 79 && k >= 5) || (ins.op == 81 && k >= 4) ||
          (ins.op == 82 && k >= 5) || (ins.op == 12 && k == 4) ||
          (ins.op == 54 && k == 3) || (ins.op == 246 && k >= 3) ||
          (ins.op == 247 && k == 2) || (ins.op == 250 && k >= 4) ||
          (ins.op == 251 && k >= 3 && (k & 1)) ||
          (ins.op == 61 && k >= 4) || (ins.op == 62 && k >= 3)) continue;
      if (!pointers.count(w[k])) continue;
      bool allowed = false;
      if (ins.op == 61) allowed = k == 3 && ins.count >= 4 && w[1] == kind_type(pointers[w[k]]);
      if (ins.op == 62) allowed = k == 1 && ins.count >= 3 && pointers[w[k]] == DepthScalar;
      if (ins.op == 65 || ins.op == 66 || ins.op == 83)
        allowed = (k == 2 || k == 3) && pointers.count(w[2]);
      if (!allowed) {
        reason = "depth interface pointer escape or unsupported use: opcode " +
                 std::to_string(ins.op) + " operand " + std::to_string(k);
        return false;
      }
    }
  }
  uint32_t next = input[3];
  if (next > UINT32_MAX - instructions.size() * 5u - 2u) return fail("ID bound overflow");
  const uint32_t half = next++, twice = next++;
  const auto emit = [](std::vector<uint32_t>& dst, uint32_t op,
                       std::initializer_list<uint32_t> args) {
    dst.push_back((uint32_t(args.size() + 1) << 16) | op);
    dst.insert(dst.end(), args);
  };
  output.assign(input.begin(), input.begin() + 5);
  for (const auto& ins : instructions) {
    const auto* w = input.data() + ins.at;
    if (ins.at == execution_end && depth && !depth_replacing) emit(output, 16, {entry, 12});
    if (ins.at == first_function) {
      emit(output, 43, {float_type, half, 0x3F000000});
      emit(output, 43, {float_type, twice, 0x40000000});
    }
    if (ins.op == 61 && ins.count >= 4 && pointers.count(w[3]) &&
        (pointers[w[3]] == CoordVector || pointers[w[3]] == CoordZ || pointers[w[3]] == DepthScalar)) {
      const uint32_t raw = next++;
      const size_t at = output.size();
      output.insert(output.end(), w, w + ins.count); output[at + 2] = raw;
      if (pointers[w[3]] == CoordVector) {
        const uint32_t z = next++, guest_z = next++;
        emit(output, 81, {float_type, z, raw, 2});
        emit(output, 133, {float_type, guest_z, z, twice});
        emit(output, 82, {vector_type, w[2], guest_z, raw, 2});
      } else emit(output, 133, {float_type, w[2], raw, twice});
    } else if (ins.op == 62 && ins.count >= 3 && pointers.count(w[1])) {
      const uint32_t host_depth = next++;
      emit(output, 133, {float_type, host_depth, w[2], half});
      const size_t at = output.size();
      output.insert(output.end(), w, w + ins.count); output[at + 2] = host_depth;
    } else output.insert(output.end(), w, w + ins.count);
  }
  output[3] = next;
  return true;
}

}  // namespace me::native
