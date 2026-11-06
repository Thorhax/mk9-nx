// mk9-nx 1.0.5: the player's profile (MK9PSET) read at Press Start. On a 360 a save's "not signed in" check passes
// because the profile has been read; this port reached the menus without it: the game only asks for it on entering a
// mode that needs it (Practice, Story, the Krypt). Until then the profile block at 0x833D1C48 (4452 bytes a player,
// its first word the size in the save) is 0, and sub_824C0DF0, saving a change made in Options, and the Fatality
// Tutorial's Retry, showed "You are not signed in" (sub_829D62E8). "Continue without saving" then turned saving off
// for the session (the manager's slot: +3152 = 0, +3156 = 1), so settings were lost.
//
// sub_824A7978, the script's "this player pressed Start", marks the player's slot as able to save and picks the
// storage device. Right after it, the first time, this asks for the profile as those modes do: sub_824C0420 (r3 = 0:
// the profile, not the settings; r4 = the user), which queues the read and its callback (sub_82522138).

#include <cstdint>
#include <cstring>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc.h>
#include <rex/system/kernel_state.h>
#include <rex/system/xmemory.h>

REXCVAR_DEFINE_BOOL(mk9_load_profile_at_start, true, "MK9",
                    "Read the player's profile at Press Start, so changes in Options are saved (no \"You are not "
                    "signed in\" box)")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);

namespace {

constexpr uint32_t kProfileBlock = 0x833D1C48;
constexpr uint32_t kSaveManager = 0x834261C8;
constexpr uint32_t kEngine = 0x833848D4;  // the object whose +92 is the player, +4 of that the user

uint32_t Load32(uint32_t address) {
  uint32_t v;
  std::memcpy(&v, rex::system::kernel_state()->memory()->TranslateVirtual(address), 4);
  return __builtin_bswap32(v);
}

bool g_asked = false;

}  // namespace

REX_EXTERN(__imp__sub_824A7978);
REX_EXTERN(__imp__sub_824C0420);
REX_HOOK_RAW(sub_824A7978) {
  __imp__sub_824A7978(ctx, base);
  if (g_asked || !REXCVAR_GET(mk9_load_profile_at_start)) {
    return;
  }
  const uint32_t user = Load32(Load32(Load32(kEngine) + 92) + 4);
  if (user != 0) {  // only user 0 is signed in
    return;
  }
  const uint32_t can_save = Load32(kSaveManager + 3152), no_save = Load32(kSaveManager + 3156);
  const uint32_t device = Load32(kSaveManager + 3160), block = Load32(kProfileBlock);
  if (!can_save || no_save || !device || block) {
    REXLOG_INFO("[profile] not read at Press Start: can save {}, not saving {}, device {:08X}, profile {:08X}", can_save,
                no_save, device, block);
    return;
  }
  g_asked = true;
  const uint64_t r3 = ctx.r3.u64, r4 = ctx.r4.u64;
  ctx.r3.u64 = 0;
  ctx.r4.u64 = user;
  __imp__sub_824C0420(ctx, base);
  REXLOG_INFO("[profile] the player's profile asked for at Press Start: {}", ctx.r3.u32);
  ctx.r3.u64 = r3;
  ctx.r4.u64 = r4;
}
