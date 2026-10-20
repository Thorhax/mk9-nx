// Mortal Kombat Komplete Edition hooks for the native renderer (src/me, sa2-nx's copy of masseffect-nx's renderer
// through banjo-nx and usf4-nx), with MK9's addresses (docs/mk9-notes.md, "Direct3D": found by the PM4 packets each
// function writes and a diagnostic run over every function in 0x82BE0000-0x82C40000, out/diag).
//
// The renderer reads the game's PM4 ring by itself; these hooks only feed it what the ring cannot say:
//
//   - Shader objects. NoteShaderObject reads the container copy inside a D3D shader object and its microcode pointer.
//     MK9's draw-time shader load is sub_82BFEEE8 (r3 = device): it reads the bound VS from device+12944 and the PS
//     from device+12940 (usf4-nx: 13072/13068), so the objects are noted there, before the original patches the VS
//     for the vertex declaration.
//   - Draw records, a hint for pairing (NoteDrawCall: r3 = device, then r4..r7 as D3D received them). MK9 has six
//     Direct3D draw functions, each writing predicated DRAW_INDX packets and calling the same state flush: three with
//     one data word (auto index: sub_82BEFEF8, sub_82BF09E0, sub_82BF1270) and three with three (indexed:
//     sub_82BF0400, sub_82BF0DF8, sub_82BF1710). Which is which D3D call is not established; every one takes the
//     device in r3.
//   - Shader containers for the library (mk9_dump_shaders = true). Direct3D's CreateVertexShader/CreatePixelShader
//     are inlined into UE3's shader creation (sub_826C07C0, sub_826C0978), so the dump hooks the XDK's container
//     parse helper sub_82BE9B48 instead (usf4-nx's sub_821E5930; 20 instructions, r3 = container, r4 = descriptor
//     it fills): in a 130 s run it received ~4,500 containers (378 VS, 4,123 PS), nearly all distinct. The blob
//     describes itself: [0] magic (0x102A1100 pixel, 0x102A1101 vertex), [1] header size (the microcode starts
//     there), [2] microcode size. Written whole to shaders/vs_<hash>.bin / ps_<hash>.bin next to the executable, once
//     each; tools/build_shader_library.sh turns the folder into the shader package.
//
// Every hook calls the original and changes no PPC register.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <system_error>
#include <unordered_set>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/filesystem.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc.h>

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "me/me_native_system.h"
#include "renut_engine/gpu/sa2_renut_yield.h"  // sa2_renderer = "renut"

REXCVAR_DEFINE_BOOL(mk9_dump_shaders, false, "MK9",
                    "Write every shader container the game creates to shaders/ next to the executable "
                    "(vs_<hash>.bin / ps_<hash>.bin), for tools/build_shader_library.sh");

namespace {

inline uint32_t Load32(const uint8_t* base, uint32_t address) {
  uint32_t v;
  std::memcpy(&v, base + address, 4);
  return __builtin_bswap32(v);
}

constexpr uint32_t kDeviceVertexShader = 12944;  // 0x3290 (sub_82BFEEE8: lwz r30,12944(r31) at 0x82BFEF34)
constexpr uint32_t kDevicePixelShader = 12940;   // 0x328C (sub_82BFEEE8: lwz r29,12940(r31) at 0x82BFEF3C)

// The draw records (NoteDrawCall) read the bound shaders from the same device fields.
const bool g_device_layout_set = [] {
  me::native::SetDeviceShaderOffsets(kDeviceVertexShader, kDevicePixelShader);
  return true;
}();

std::mutex g_dump_mutex;
std::unordered_set<uint64_t> g_dumped;

void DumpShaderBlob(const uint8_t* base, uint32_t blob_address) {
  if (!REXCVAR_GET(mk9_dump_shaders) || !blob_address) {
    return;
  }
  const uint32_t magic = Load32(base, blob_address);
  if ((magic & 0xFFFFFFFEu) != 0x102A1100u) {
    return;
  }
  const bool vertex = magic & 1;
  const uint32_t header = Load32(base, blob_address + 4);
  const uint32_t ucode = Load32(base, blob_address + 8);
  if (header < 12 || header > (1u << 20) || !ucode || ucode > (1u << 20)) {
    REXLOG_WARN("[shaders] implausible {} blob at {:08X} (header {}, microcode {}), skipped", vertex ? "VS" : "PS",
                blob_address, header, ucode);
    return;
  }
  const uint8_t* blob = base + blob_address;
  const size_t size = size_t(header) + ucode;
  const uint64_t hash = XXH3_64bits(blob, size);
  {
    std::lock_guard lock(g_dump_mutex);
    if (!g_dumped.insert(hash).second) {
      return;
    }
  }
  const auto folder = rex::filesystem::GetExecutableFolder() / "shaders";
  std::error_code ec;
  std::filesystem::create_directories(folder, ec);
  const auto path = folder / fmt::format("{}_{:016x}.bin", vertex ? "vs" : "ps", hash);
  std::ofstream(path, std::ios::binary).write(reinterpret_cast<const char*>(blob), std::streamsize(size));
}

}  // namespace

// Draw-time shader load (called by every Draw*'s state flush).
REX_EXTERN(__imp__sub_82BFEEE8);
REX_HOOK_RAW(sub_82BFEEE8) {
  SA2_RENUT_YIELD(sub_82BFEEE8)
  const uint32_t device = ctx.r3.u32;
  static std::atomic<bool> logged{false};
  if (!logged.exchange(true, std::memory_order_relaxed)) {
    REXLOG_INFO("[native] mk9: Direct3D device at {:08X}", device);
  }
  if (device >= 0x40000000 && device < 0x7F000000) {
    const uint32_t vs = Load32(base, device + kDeviceVertexShader), ps = Load32(base, device + kDevicePixelShader);
    if (!me::native::ShaderObjectsFinished(vs, ps)) {
      me::native::NoteShaderObject(base, vs, true);
      me::native::NoteShaderObject(base, ps, false);
    }
  }
  __imp__sub_82BFEEE8(ctx, base);
}

// mk9-nx (2026-10-20): with sa2_renderer = "renut", Renut's side at the addresses it hooks too (_SHARED), the game's
// original at the others. Which is which, from the Renut census (docs/mk9-renut-census.md): BeginVertices,
// DrawVertices, DrawIndexedVertices, BeginIndexedVertices and the two tessellated draws.
#define MK9_DRAW_HOOK(addr, yield)                                                                         \
  REX_EXTERN(__imp__sub_##addr);                                                                          \
  REX_HOOK_RAW(sub_##addr) {                                                                              \
    yield(sub_##addr)                                                                                     \
    me::native::NoteDrawCall(base, ctx.r3.u32, ctx.r4.u32, ctx.r5.u32, ctx.r6.u32, ctx.r7.u32, nullptr); \
    __imp__sub_##addr(ctx, base);                                                                         \
  }

MK9_DRAW_HOOK(82BEFEF8, SA2_RENUT_YIELD_SHARED)  // DRAW_INDX, auto index (BeginVertices)
MK9_DRAW_HOOK(82BF09E0, SA2_RENUT_YIELD_SHARED)  // DRAW_INDX, auto index (DrawVertices)
MK9_DRAW_HOOK(82BF1270, SA2_RENUT_YIELD)  // DRAW_INDX, auto index (no direct caller: reached by pointer; tessellated?)
MK9_DRAW_HOOK(82BF0400, SA2_RENUT_YIELD)  // DRAW_INDX, indexed (BeginIndexedVertices: Renut hooks DrawIndexedVerticesUP)
MK9_DRAW_HOOK(82BF0DF8, SA2_RENUT_YIELD_SHARED)  // DRAW_INDX, indexed (DrawIndexedVertices)
MK9_DRAW_HOOK(82BF1710, SA2_RENUT_YIELD)  // DRAW_INDX, indexed (tessellated?)

REX_EXTERN(__imp__sub_82BE9B48);
REX_HOOK_RAW(sub_82BE9B48) {  // XDK shader container parse helper (r3 = the container blob)
  DumpShaderBlob(base, ctx.r3.u32);
  __imp__sub_82BE9B48(ctx, base);
}
