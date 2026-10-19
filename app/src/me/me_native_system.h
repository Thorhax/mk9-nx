// Native graphics system (see me_native_system.cpp).
#pragma once

#include <memory>

#include <rex/system/interfaces/graphics.h>

struct PPCContext;  // rex/ppc/context.h (FlushRunsDirect)

namespace me::native {

// true when SA2_NATIVE=1: the app installs the native system instead of Xenos emulation.
bool Enabled();

std::unique_ptr<rex::system::IGraphicsSystem> CreateGraphicsSystem();

// Called on the game thread when Direct3D binds a shader object (sub_8222F850), before it patches the
// vertex shader: records which library entry the microcode at that physical address belongs to, so the
// ring can identify IM_LOADs by address No-op without the native system.
void NoteShaderObject(const uint8_t* base, uint32_t object, bool vertex);

// True when the shader object was matched to a library entry (diagnostics).
bool HasEntry(uint32_t object);

// sa2_cpu_note_shader_pair: true when both objects' NoteShaderObject work is already finished (the calls would return
// at once), so a draw hook can skip both calls. False without the key.
bool ShaderObjectsFinished(uint32_t vs, uint32_t ps);

struct DrawDirect;

// Called on the game thread by the Direct3D draw functions (the game's hooks file): the device they were called on
// (r3), the call's primitive type and counts. The record also takes the bound VS and PS objects from the device and
// its command-buffer position (device+48). The ring pairs each DRAW_INDX with these records in order, because the
// ring's own shader loads can be immediate copies Direct3D patched. No-op without the native system.
// With `direct` (the call's BeginDrawDirect result, sa2_cpu_direct_pair) the record is held back for the call's direct
// record; EndDrawDirect sends it one way or the other.
void NoteDrawCall(const uint8_t* base, uint32_t device, uint32_t type, uint32_t r5, uint32_t r6, uint32_t r7,
                  DrawDirect* direct = nullptr);

// sa2_cpu_direct_state (the direct renderer, step 1): the draw hooks call BeginDrawDirect before Direct3D's draw
// function and EndDrawDirect right after it. The state packets the draw function wrote before its draw packet become a
// host record the ring applies directly (no PM4 parsing for them), covered in the command buffer by one NOP packet.
// See namespace direct in me_native_system.cpp. No-op without the native system or with the key off.
struct DrawDirect {
  uint32_t device = 0;       // 0 = nothing to do after the call
  uint32_t start = 0;        // Direct3D's command buffer write pointer before the call (the last word written)
  uint32_t limit = 0;        // the command buffer segment's limits before the call (device+52, +56): a segment
  uint32_t soft_limit = 0;   //   change during the call leaves the packets as they are
  uint64_t submissions = 0;  // the ring write pointer's guest writes before the call
  bool pair = false;         // NoteDrawCall held the call's Draw* record back (sa2_cpu_direct_pair)
};
DrawDirect BeginDrawDirect(const uint8_t* base, uint32_t device);
void EndDrawDirect(uint8_t* base, const DrawDirect& d);

// sa2_cpu_direct_flush (the direct renderer, step 2): inside a converted draw call (BeginDrawDirect .. EndDrawDirect),
// Direct3D's flush of register runs from a device shadow writes the call's direct record itself instead of PM4: the
// `kind` registers of each dirty bit of `mask` (bit 63 first) from the shadow at guest `shadow` (one big-endian word per
// register) to the registers from `register_base`, plus the fetch flush's end packet. The command buffer gets one
// 3-word placeholder NOP where the packets would have been. Returns false WITHOUT having changed anything when it does
// not apply (no converted call, the original would take its slow path, no room): the hook then runs the original.
// `original` is the recompiled function, which the self-check runs into a scratch buffer and compares.
enum class FlushKind : uint8_t {
  kConstants,  // 16 registers per bit (Direct3D's shader constant flush, SA2 sub_82555CC8)
  kFetch,      // 6 registers per bit, then the 0x5000 end packet (fetch constant flush, SA2 sub_82555B70)
  kGroups,     // 1 register per bit (register group flush, SA2 sub_82555930)
};
bool FlushRunsDirect(PPCContext& ctx, uint8_t* base, uint32_t device, uint64_t mask, uint32_t register_base,
                     uint32_t shadow, FlushKind kind, void (*original)(PPCContext&, uint8_t*));

// True when the ring has processed everything the game submitted (lockstep video captures: with the game held at its
// next flip, the presenter then shows the frame just flipped). Always true without the native system.
bool RingIdle();

// XE_SWAP packets the ring has presented so far (one per game flip): lockstep video captures wait for a frame's swap.
uint64_t PresentedSwaps();

// True while the native renderer runs (the emulated GPU needs the game's own Direct3D: the native layer stays off).
bool NativeActive();

// The native Direct3D layer (src/me/sa2_api_*.cpp, keys sa2_api_*): a record of ops (the direct-state record format:
// kRun, kType3, kPair, ...) published to the same ring of records as the direct-state records, so the ring applies it
// in order where the caller then writes its marker NOP [0xC0011000]['SA2D'][seq]. Game thread (the one drawing thread).
// Returns the record's sequence number, or 0 when there is no room or records are off (the caller runs the original).
uint32_t PublishApiRecord(const uint32_t* ops, uint32_t n);
// The same record written in place (no copy): ReserveApiRecord returns where its ops go (room for `max_words`), or
// nullptr (no room: build them elsewhere and use PublishApiRecord); CommitApiRecord publishes the first `n` words and
// returns the sequence number, or 0 when the reservation was lost (another record was published meanwhile: one
// producer is expected, so this is a failure, not a retry).
uint32_t* ReserveApiRecord(uint32_t max_words);
uint32_t CommitApiRecord(const uint32_t* ops, uint32_t n);

// Called on the ring thread right after each presented swap, with PresentedSwaps(); nullptr = none.
void SetPresentHook(void (*hook)(uint64_t presented));

// Where the game's Direct3D device keeps the bound vertex and pixel shader objects; it moves between XDK versions
// (Mass Effect, the default: +12416 / +12412; Banjo-Kazooie N&B, 2008: +12696 / +12692; SA2, 2012: +13072 / +13068).
// Set once by the game's hooks file, before the first draw.
void SetDeviceShaderOffsets(uint32_t vertex_shader_offset, uint32_t pixel_shader_offset);

// Called by the ring thread after it writes the read pointer back: wakes the game's D3D GPU wait
// (src/native/me_ring_wait.cpp, sub_8222FA98).
void NotifyRingProgress();

// The progress counter NotifyRingProgress bumps, and a sleep until it moves past `seen` (at most max_us; true when it
// moved). For the game's D3D GPU waits (src/cpu/sa2_gpu_wait.cpp).
uint32_t RingProgress();
bool WaitRingProgress(uint32_t seen, uint32_t max_us);

}  // namespace me::native
