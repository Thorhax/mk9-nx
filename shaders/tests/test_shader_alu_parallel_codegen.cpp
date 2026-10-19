// Pure CPU codegen regression; link XenosRecomp with its normal pch_min flags.
#include "../XenosRecomp/shader_recompiler.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

static std::string Emit(const AluInstruction& instruction, bool pixel = false) {
  ShaderRecompiler compiler;
  compiler.isPixelShader = pixel;
  compiler.literalFloat4.insert(255);
  compiler.recompile(instruction);
  return compiler.out;
}
static AluInstruction Base() {
  AluInstruction i{};
  i.vectorOpcode = AluVectorOpcode::Add;
  i.vectorDest = 2; i.vectorWriteMask = 7;
  i.scalarOpcode = AluScalarOpcode::Maxs;
  i.scalarDest = 0; i.scalarWriteMask = 4;
  i.src1Select = i.src2Select = i.src3Select = 1;
  i.src1Register = 0; i.src2Register = 1; i.src3Register = 2;
  i.src3Swizzle = 0xC2; // Both scalar sources are old r2.z.
  return i;
}
static size_t Count(const std::string& text, const std::string& needle) {
  size_t n = 0, pos = 0;
  while ((pos = text.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
  return n;
}
int main(int argc, char** argv) {
  unsetenv("MASSEFFECT_SHADER_PRECISE_POSITION");
  const uint32_t raw[3]{0x14470002, 0x046C6CC6, 0xA580FF02};
  AluInstruction real{};
  static_assert(sizeof(real) == sizeof(raw));
  std::memcpy(&real, raw, sizeof(raw));
  const auto actual = Emit(real);
  assert(actual.front() == '{' && actual.back() == '\n');
  assert(actual.find("float me_alu_scalar_0_3 = r2.z;\n") != std::string::npos);
  assert(actual.rfind("}\n") == actual.size() - 2);
  assert(actual.find("r2.xyz = ") > actual.find(" = r2.z;"));
  assert(actual.find("ps = me_alu_scalar_0_3;") != std::string::npos);
  assert(Count(actual, "float me_alu_scalar_") == 1); // MOV dedup preserves old value.
  assert(Emit(real, true) == actual);

  // Exact lane overlap, not merely matching TEMP number. Exhaust all lane pairs/masks.
  for (unsigned a = 0; a < 4; ++a) for (unsigned b = 0; b < 4; ++b)
    for (unsigned mask = 0; mask < 16; ++mask) {
      auto i = Base(); i.scalarOpcode = AluScalarOpcode::Adds;
      i.src3Swizzle = (((a + 1) & 3) << 6) | b; i.vectorWriteMask = mask;
      const unsigned expected = ((mask >> a) & 1) + (b != a && ((mask >> b) & 1));
      assert(Count(Emit(i), "float me_alu_scalar_") == expected);
    }
  auto i = Base(); i.src3Register |= 0x80; i.src3Negate = 1;
  assert(Emit(i).find("= -abs(r2.z);") != std::string::npos);
  i = Base(); i.scalarOpcode = AluScalarOpcode::AddsPrev;
  assert(Emit(i).find("ps = me_alu_scalar_0_3 + ps;") != std::string::npos);
  i.scalarOpcode = AluScalarOpcode::RetainPrev;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);
  i.scalarOpcode = AluScalarOpcode::SetpClr;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);
  i = Base(); i.src3Register = 3;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);
  i = Base(); i.exportData = 1; i.vectorDest = 62;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);

  // Special const/TEMP scalar opcodes use a differently encoded TEMP register.
  for (unsigned opcode = 42; opcode <= 47; ++opcode) {
    i = Base(); i.scalarOpcode = AluScalarOpcode(opcode);
    i.src3Swizzle = 2; i.src3Register = 255;
    i.vectorDest = (opcode & 1) | 2; i.absConstants = 1; i.src3Negate = 1;
    const auto text = Emit(i);
    assert(text.find("= -abs(r" + std::to_string(i.vectorDest) + ".z);") != std::string::npos);
    assert(Count(text, "float me_alu_scalar_") == 1);
    assert(text.find("ps = -abs(c255.") != std::string::npos); // Constant remains late.
  }
  // Vector a0 update precedes capture; scalar predicate reads snapshot, not new TEMP.
  i = Base(); i.vectorOpcode = AluVectorOpcode::MaxA; i.scalarOpcode = AluScalarOpcode::SetpEq;
  const auto state = Emit(i);
  assert(state.find("a0 = ") < state.find("float me_alu_scalar_"));
  assert(state.find("r2.xyz = ") < state.find("p0 = me_alu_scalar_"));
  // Scalar constants are never captured, including unused RetainPrev sources.
  i = Base(); i.src3Select = 0; i.src3Register = 255;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);
  i.scalarOpcode = AluScalarOpcode::RetainPrev; i.const0Relative = 1;
  assert(Emit(i).find("me_alu_scalar_") == std::string::npos);
  // A real declared relative constant is evaluated after vector a0 updates,
  // while the special second operand's overlapping TEMP is captured beforehand.
  {
    ShaderRecompiler compiler;
    const uint8_t name[]{'T','e','s','t','C','o','n','s','t','a','n','t','s',0};
    ConstantInfo info{}; info.registerCount.value = byteSwap(uint16_t(256));
    compiler.constantTableData = name; compiler.float4Constants[255] = &info;
    i = Base(); i.vectorOpcode = AluVectorOpcode::MaxA;
    i.scalarOpcode = AluScalarOpcode::Mulsc0; i.src3Register = 255;
    i.src3Swizzle = 2; i.const0Relative = 1; i.constAddressRegisterRelative = 1;
    compiler.recompile(i);
    const auto& text = compiler.out;
    assert(text.find("a0 = ") < text.find("float me_alu_scalar_"));
    assert(text.find("float me_alu_scalar_") < text.find("r2.xyz = "));
    assert(text.find("r2.xyz = ") < text.find("ps = TestConstants(255 + a0)."));
    assert(Count(text, "TestConstants(") == 1);
  }
  // Multiple instructions and predication must retain scoped, unique snapshots.
  {
    ShaderRecompiler compiler;
    i = Base(); i.isPredicated = 1; i.predicateCondition = 1;
    compiler.recompile(i); const size_t firstSize = compiler.out.size();
    compiler.recompile(i);
    assert(Count(compiler.out, "float me_alu_scalar_") == 2);
    assert(compiler.out.find("float me_alu_scalar_" + std::to_string(firstSize) + "_3") != std::string::npos);
    assert(Count(compiler.out, "if (p0)") == 1);
  }
  setenv("MASSEFFECT_SHADER_PRECISE_POSITION", "1", 1);
  assert(Emit(real).find("precise float me_alu_scalar_") != std::string::npos);
  assert(Emit(real, true) == actual);
  unsetenv("MASSEFFECT_SHADER_PRECISE_POSITION");
  // Optional actual-HLSL fixture for DXC: a following case would bypass the
  // introduced initialization if the per-ALU snapshot scope were omitted.
  if (argc == 2) {
    std::ofstream fixture(argv[1]);
    fixture << "float4 main(uint id : SV_VertexID) : SV_Position {\n"
            << "float4 r0=0, r1=1, r2=float4(1,2,3,4), c255=float4(0,1,2,3); float ps=0;\n"
            << "switch(id) { case 0:\n" << actual
            << "break; case 1: r0=r1; break; default: r0=r2; break; }\nreturn r0; }\n";
    fixture.close(); assert(fixture.good());
  }
  std::puts("shader parallel ALU TEMP-WAR codegen: PASS (not GPU conformance)");
}
