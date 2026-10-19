#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <vector>

namespace me::native {

struct RectangleShader {
  std::vector<uint32_t> words;
  // Original Location, followed by the three new Locations for that input.
  std::vector<std::array<uint32_t, 4>> inputs;
};

// Portable post-VS rectangle expansion, following rexglue's rectangle geometry
// shader: rotate to the corner opposite the longest XY edge, then extrapolate
// v3 = (v1 - v0) + v2. Each host vertex receives all three guest input records.
// No geometry-shader support or per-game shader replacement is required.
// Reject unfamiliar interfaces / side effects instead of silently approximating.
inline RectangleShader ExpandRectangleShader(const std::vector<uint32_t>& s) {
  RectangleShader result;
  if (s.size() < 5 || s[0] != 0x07230203 || s[1] < 0x00010400) return result;
  struct Var { uint32_t ptr, id, storage, type; };
  std::vector<Var> vars;
  std::map<uint32_t, uint32_t> locations, builtins, pointees, private_ptrs;
  std::map<uint32_t, uint32_t> vectors;
  uint32_t entry = 0, void_type = 0, int_type = 0, bool_type = 0, float_type = 0, fn_type = 0;
  size_t functions = 0, first_variable = 0;
  for (size_t i = 5; i < s.size();) {
    const uint32_t n = s[i] >> 16, op = s[i] & 65535;
    if (!n || i + n > s.size()) return {};
    if (op == 15) { if (entry || s[i+1] != 0) return {}; entry = s[i+2]; }
    if (op == 19) void_type = s[i+1];
    if (op == 20) bool_type = s[i+1];
    if (op == 21 && s[i+2] == 32 && s[i+3] == 1) int_type = s[i+1];
    if (op == 22 && s[i+2] == 32) float_type = s[i+1];
    if (op == 23 && s[i+2] == float_type) vectors[s[i+1]] = s[i+3];
    if (op == 32) { pointees[s[i+1]] = s[i+3]; if (s[i+2] == 6) private_ptrs[s[i+3]] = s[i+1]; }
    if (op == 33 && n == 3 && s[i+2] == void_type) fn_type = s[i+1];
    if (op == 59 && !functions) {
      if (!first_variable) first_variable = i;
      vars.push_back({s[i+1], s[i+2], s[i+3], pointees[s[i+1]]});
    }
    if (op == 71 && n >= 4) {
      if (s[i+2] == 30) locations[s[i+1]] = s[i+3];
      if (s[i+2] == 11) builtins[s[i+1]] = s[i+3];
    }
    if (op == 54 && !functions) functions = i;
    // Memory exports / atomics cannot be duplicated across host invocations.
    if ((op >= 227 && op <= 242) || op == 99 || op == 4416) return {};
    i += n;
  }
  if (!entry || !void_type || !fn_type || !float_type || !functions) return {};
  std::vector<Var> inputs, outputs;
  uint32_t position = 0;
  for (const auto& v : vars) {
    if (v.storage == 1) {
      if (builtins.count(v.id) || !locations.count(v.id) || !vectors.count(v.type)) return {};
      inputs.push_back(v);
    } else if (v.storage == 3) {
      if (!vectors.count(v.type)) return {};
      if (builtins.count(v.id)) {
        if (builtins[v.id] != 0 || vectors[v.type] != 4) return {};
        position = v.id;
      } else if (!locations.count(v.id)) return {};
      outputs.push_back(v);
    } else if (v.storage == 6) {
      // Persistent private state would need reinitializing for each guest VS.
      return {};
    }
  }
  if (!position || inputs.size() > 10) return {};
  std::map<uint32_t, bool> writable;
  for (const auto& v : outputs) writable[v.id] = true;
  for (size_t i = functions; i < s.size();) {
    const uint32_t n = s[i] >> 16, op = s[i] & 65535;
    if (op == 59 && s[i+3] == 7) writable[s[i+2]] = true;
    if (op == 65 || op == 66 || op == 67) {
      // Inputs converted to Private must be loaded directly, not accessed via
      // an old Input pointer type. Complex interfaces require a separate path.
      for (const auto& v : inputs) if (s[i+3] == v.id) return {};
      if (writable.count(s[i+3])) writable[s[i+2]] = true;
    }
    if ((op == 62 || op == 63 || op == 64) && !writable.count(s[i+1])) return {};
    i += n;
  }
  uint32_t next = s[3];
  const auto id = [&]() { return next++; };
  std::vector<uint32_t> declarations, decorations, wrapper;
  const auto emit = [](std::vector<uint32_t>& dst, uint32_t op, std::initializer_list<uint32_t> args) {
    dst.push_back((uint32_t(args.size()+1) << 16) | op);
    dst.insert(dst.end(), args);
  };
  if (!int_type) { int_type = id(); emit(declarations, 21, {int_type, 32, 1}); }
  if (!bool_type) { bool_type = id(); emit(declarations, 20, {bool_type}); }
  const uint32_t input_int_ptr = id(), vertex_index = id();
  emit(declarations, 32, {input_int_ptr, 1, int_type});
  std::array<uint32_t, 7> ints;
  for (uint32_t k = 0; k < ints.size(); ++k) { ints[k] = id(); emit(declarations, 43, {int_type, ints[k], k}); }
  std::map<uint32_t, std::array<uint32_t, 3>> new_inputs;
  for (uint32_t k = 0; k < inputs.size(); ++k) {
    const auto& v = inputs[k];
    if (!private_ptrs.count(v.type)) {
      const uint32_t p = id(); private_ptrs[v.type] = p;
      emit(declarations, 32, {p, 6, v.type});
    }
    std::array<uint32_t, 4> mapping{locations[v.id], k*3, k*3+1, k*3+2};
    result.inputs.push_back(mapping);
    for (uint32_t c = 0; c < 3; ++c) {
      const uint32_t ni = id(); new_inputs[v.id][c] = ni;
      emit(decorations, 71, {ni, 30, mapping[c+1]});
    }
  }
  emit(decorations, 71, {vertex_index, 11, 42});
  // Put all types/constants before the new global variables.
  for (const auto& v : inputs)
    for (uint32_t c = 0; c < 3; ++c) emit(declarations, 59, {v.ptr, new_inputs[v.id][c], 1});
  emit(declarations, 59, {input_int_ptr, vertex_index, 1});
  const uint32_t new_entry = id();
  emit(wrapper, 54, {void_type, new_entry, 0, fn_type});
  emit(wrapper, 248, {id()});
  const auto load = [&](uint32_t type, uint32_t pointer) {
    const uint32_t r = id(); emit(wrapper, 61, {type, r, pointer}); return r;
  };
  const auto binary = [&](uint32_t op, uint32_t type, uint32_t a, uint32_t b) {
    const uint32_t r = id(); emit(wrapper, op, {type, r, a, b});
    if (op == 129 || op == 131) emit(decorations, 71, {r, 42}); // NoContraction, like rexglue's extrapolation.
    return r;
  };
  const auto select = [&](uint32_t type, uint32_t cond, uint32_t a, uint32_t b) {
    const uint32_t r = id(); emit(wrapper, 169, {type, r, cond, a, b}); return r;
  };
  std::map<uint32_t, std::array<uint32_t, 3>> values;
  for (uint32_t c = 0; c < 3; ++c) {
    for (const auto& v : inputs) emit(wrapper, 62, {v.id, load(v.type, new_inputs[v.id][c])});
    emit(wrapper, 57, {void_type, id(), entry});
    for (const auto& v : outputs) values[v.id][c] = load(v.type, v.id);
  }
  std::array<uint32_t, 3> lengths;
  for (uint32_t c = 0; c < 3; ++c) {
    std::array<uint32_t, 2> delta;
    for (uint32_t axis = 0; axis < 2; ++axis) {
      const uint32_t a = id(), b = id();
      emit(wrapper, 81, {float_type, a, values[position][(c+1)%3], axis});
      emit(wrapper, 81, {float_type, b, values[position][(c+2)%3], axis});
      const uint32_t d = binary(131, float_type, a, b);
      delta[axis] = binary(133, float_type, d, d);
    }
    lengths[c] = binary(129, float_type, delta[0], delta[1]);
  }
  const uint32_t d0 = binary(167, bool_type,
      binary(186, bool_type, lengths[0], lengths[1]), binary(186, bool_type, lengths[0], lengths[2]));
  const uint32_t first = select(int_type, d0, ints[0],
      select(int_type, binary(186, bool_type, lengths[1], lengths[2]), ints[1], ints[2]));
  const uint32_t local = binary(139, int_type, load(int_type, vertex_index), ints[6]);
  // Two consistently wound triangles equivalent to strip 0,1,2,3.
  const uint32_t corner = select(int_type, binary(170, bool_type, local, ints[5]), ints[3],
      select(int_type, binary(170, bool_type, local, ints[4]), ints[1],
      select(int_type, binary(170, bool_type, local, ints[3]), ints[2], local)));
  std::array<uint32_t, 3> rotated;
  for (uint32_t c = 0; c < 3; ++c) rotated[c] = binary(139, int_type, binary(128, int_type, first, ints[c]), ints[3]);
  for (const auto& v : outputs) {
    const auto& vs = values[v.id];
    const auto at = [&](uint32_t index) {
      return select(v.type, binary(170, bool_type, index, ints[0]), vs[0],
          select(v.type, binary(170, bool_type, index, ints[1]), vs[1], vs[2]));
    };
    const uint32_t a = at(rotated[0]), b = at(rotated[1]), c = at(rotated[2]);
    const uint32_t fourth = binary(129, v.type, binary(131, v.type, b, a), c);
    const uint32_t value = select(v.type, binary(170, bool_type, corner, ints[3]), fourth,
        select(v.type, binary(170, bool_type, corner, ints[0]), a,
        select(v.type, binary(170, bool_type, corner, ints[1]), b, c)));
    emit(wrapper, 62, {v.id, value});
  }
  emit(wrapper, 253, {}); emit(wrapper, 56, {});
  auto& out = result.words;
  out.assign(s.begin(), s.begin()+5);
  bool decorated = false;
  for (size_t i = 5; i < s.size();) {
    const uint32_t n = s[i] >> 16, op = s[i] & 65535;
    if (!decorated && (op >= 19 && op <= 39)) {
      out.insert(out.end(), decorations.begin(), decorations.end()); decorated = true;
    }
    if (i == first_variable) out.insert(out.end(), declarations.begin(), declarations.end());
    const size_t start = out.size();
    if (op == 71 && new_inputs.count(s[i+1])) {
      // Input Location and interpolation decorations don't apply to Private.
    } else {
      out.insert(out.end(), s.begin()+i, s.begin()+i+n);
      if (op == 15) {
        out[start+2] = new_entry;
        for (const auto& v : inputs) for (uint32_t ni : new_inputs[v.id]) out.push_back(ni);
        out.push_back(vertex_index);
        out[start] = (uint32_t(out.size()-start) << 16) | op;
      } else if (op == 16 && s[i+1] == entry) out[start+1] = new_entry;
      else if (op == 59 && new_inputs.count(s[i+2])) {
        out[start+1] = private_ptrs[pointees[s[i+1]]]; out[start+3] = 6;
      }
    }
    i += n;
  }
  out.insert(out.end(), wrapper.begin(), wrapper.end()); out[3] = next;
  return result;
}

}  // namespace me::native
