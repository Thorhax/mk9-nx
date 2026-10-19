// Compile with the same pch_min/fmt/MASSEFFECT_RECOMP flags as build_xenos_hlsl.sh,
// linking shaders/XenosRecomp/shader_recompiler.cpp. Pure CPU codegen.
#include "../XenosRecomp/shader_recompiler.h"
#include <cstdlib>
#include <cstring>
#include <cstdio>

static std::string Emit(AluVectorOpcode opcode, const char* setting, bool pixel = false) {
  if (setting) setenv("MASSEFFECT_SHADER_PRECISE_POSITION", setting, 1);
  else unsetenv("MASSEFFECT_SHADER_PRECISE_POSITION");
  ShaderRecompiler compiler;
  compiler.isPixelShader = pixel;
  AluInstruction instruction{};
  instruction.vectorOpcode = opcode;
  instruction.scalarOpcode = AluScalarOpcode::RetainPrev;
  instruction.vectorDest = 4;
  instruction.vectorWriteMask = 1;
  instruction.src1Select = instruction.src2Select = instruction.src3Select = 1;
  instruction.src1Register = 1; instruction.src2Register = 2; instruction.src3Register = 3;
  compiler.recompile(instruction);
  return compiler.out;
}

static std::string EmitContainer(const char* setting) {
  if (setting) setenv("MASSEFFECT_SHADER_PRECISE_POSITION", setting, 1);
  else unsetenv("MASSEFFECT_SHADER_PRECISE_POSITION");
  std::vector<uint8_t> container(120);
  const auto put = [&](size_t offset, uint32_t word) {
    for (unsigned i = 0; i < 4; ++i) container[offset + i] = uint8_t(word >> (24 - i * 8));
  };
  put(0, 0x102A1101); put(4, 96); put(8, 24); put(24, 40);
  put(44, 24); // Header: code offset0, length24, no declarations/interpolators.
  const uint64_t exec = (uint64_t(2) << 44) | (1u << 12) | 1u;
  put(96, uint32_t(exec)); put(100, uint32_t(exec >> 32));
  AluInstruction instruction{};
  instruction.vectorOpcode = AluVectorOpcode::Max;
  instruction.scalarOpcode = AluScalarOpcode::RetainPrev;
  instruction.vectorDest = 62; instruction.exportData = 1; instruction.vectorWriteMask = 15;
  instruction.src1Select = instruction.src2Select = 1;
  instruction.src1Register = instruction.src2Register = 1;
  uint32_t words[3]; std::memcpy(words, &instruction, sizeof(words));
  for (unsigned i = 0; i < 3; ++i) put(108 + i * 4, words[i]);
  ShaderRecompiler compiler;
  compiler.recompile(container.data(), "");
  return compiler.out;
}

int main() {
  assert(Emit(AluVectorOpcode::Dp3, nullptr) == "r4.x = dot(r1.xyz, r2.xyz);\n");
  assert(Emit(AluVectorOpcode::Dp4, "0") == "r4.x = dot(r1.xyzw, r2.xyzw);\n");
  assert(Emit(AluVectorOpcode::Dp2Add, "off") == "r4.x = dot(r1.xy, r2.xy) + r3.x;\n");
  assert(Emit(AluVectorOpcode::Dp3, "1") ==
         "r4.x = ((((r1.xyz).x * (r2.xyz).x) + ((r1.xyz).y * (r2.xyz).y)) + ((r1.xyz).z * (r2.xyz).z));\n");
  assert(Emit(AluVectorOpcode::Dp4, "1") ==
         "r4.x = (((((r1.xyzw).x * (r2.xyzw).x) + ((r1.xyzw).y * (r2.xyzw).y)) + ((r1.xyzw).z * (r2.xyzw).z)) + ((r1.xyzw).w * (r2.xyzw).w));\n");
  assert(Emit(AluVectorOpcode::Dp2Add, "1") ==
         "r4.x = ((((r1.xy).x * (r2.xy).x) + ((r1.xy).y * (r2.xy).y)) + (r3.x));\n");
  assert(Emit(AluVectorOpcode::Dp4, "1", true) == Emit(AluVectorOpcode::Dp4, "0", true));
  const auto baseline = EmitContainer(nullptr);
  assert(baseline == EmitContainer("0"));
  assert(baseline == EmitContainer("false"));
  assert(baseline.find("precise") == std::string::npos);
  const auto precise = EmitContainer("1");
  assert(precise.find("[[vk::ext_decorate(18)]] out precise float4 oPos : SV_Position") != std::string::npos);
  assert(precise.find("precise float4 r0 = ") != std::string::npos);
  assert(precise.find("precise float ps = 0.0;") != std::string::npos);
  assert(precise.find("oPos.xyzw = r1.xyzw;") != std::string::npos);
  unsetenv("MASSEFFECT_SHADER_PRECISE_POSITION");
  std::puts("shader precise VS codegen: PASS (not GPU/numerical conformance)");
}
