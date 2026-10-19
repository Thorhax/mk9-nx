/*
 * Publish FPS and resolution where the console monitor reads them, and obey Reverse-NX.
 *
 * The FPS counter of those overlays (Status Monitor Overlay and its forks, such as "Horizon OC
 * Monitor") is not computed by them: it is read from shared memory published by SaltyNX. That
 * memory is filled in by NX-FPS, which hooks the game's present call (nvnQueuePresentTexture,
 * eglSwapBuffers or vkQueuePresentKHR). Our game goes through none of the three (NVK is linked
 * inside the NRO and talks to nvdrv), so there is nothing to hook and the field stays empty.
 *
 * Here we write the block ourselves. The format is SaltyNX's (struct NxFpsSharedBlock, 174 packed
 * bytes, magic 0x465053), and the overlay finds it by scanning the shared memory 4 bytes at a time.
 *
 * The Reverse-NX block (magic "NXRT") is read as well; that one does publish its state even though
 * it cannot hook us: this way the game can follow the handheld/docked mode chosen in its overlay
 * without putting the console in the dock.
 *
 * If SaltyNX is not installed, the port does not exist and nothing is done here.
 *
 * References: masagrator/SaltyNX (saltysd_core), masagrator/Status-Monitor-Overlay
 * (source/Utils.hpp) and masagrator/ReverseNX-RT (Overlay/include/SaltyNX.h).
 */

#include "rex/ui/switch_saltynx.h"

#if REX_PLATFORM_SWITCH

#include <switch.h>

#include <atomic>
#include <cstring>

#include <cstdio>

/*
 * This object is compiled separately (rexglue_switch_startup) and has no fmt, so no cvars and no
 * REXLOG: warnings go to stderr, which on the console ends up in rex_stderr.log.
 */

namespace rex::ui::switch_saltynx {
namespace {

constexpr uint32_t kMagicFps = 0x465053;          // "SPF": NX-FPS block
constexpr uint32_t kMagicReverseNx = 0x5452584E;  // "NXRT" in little-endian
constexpr size_t kSharedSize = 0x1000;      // SaltyNX maps one page

/* Resolution calls as the overlay reads them: width, height and how many times. */
struct ResolutionCalls {
  uint16_t width;
  uint16_t height;
  uint16_t calls;
} __attribute__((packed));

/* SaltyNX's struct NxFpsSharedBlock. The overlay checks that it is 174 bytes. */
struct BlockFps {
  uint32_t magic;
  uint8_t fps;
  float fps_average;
  bool plugin_active;
  uint8_t fps_blocked;
  uint8_t fps_mode;
  uint8_t zero_sync;
  uint8_t applied_patch;
  uint8_t api;
  uint32_t ticks[10];
  uint8_t buffers;
  uint8_t buffers_set;
  uint8_t buffers_active;
  uint8_t buffers_active_set;
  uint8_t display_sync;
  ResolutionCalls render[8];
  ResolutionCalls viewport[8];
  bool force_refresh_original;
  bool no_force_60_in_base;
  bool force_suspension;
  uint8_t refresh_current;
  float read_per_second;
  uint8_t fps_blocked_base;
  uint64_t frame_number;
  int8_t buffers_expected;
} __attribute__((packed));

static_assert(sizeof(BlockFps) == 174, "the overlay expects 174 bytes");

/* ReverseNX-RT struct Shared (9 bytes). */
struct BlockReverseNx {
  uint32_t magic;
  bool in_base;
  bool by_default;
  bool plugin_active;
  uint8_t resolutions;
  bool usage_ddr;
} __attribute__((packed));

static_assert(sizeof(BlockReverseNx) == 9, "ReverseNX-RT uses 9 bytes");

/* API declared by the block: 0 unknown, 1 NVN, 2 GL, 3 Vulkan. */
constexpr uint8_t kApiVulkan = 3;

/* Warnings to stderr, which on the console ends up in rex_stderr.log. */
void Warning(const char* text) { std::fprintf(stderr, "%s\n", text); }
void WarningNum(const char* text, long long number) {
  std::fprintf(stderr, "%s 0x%llX (%lld)\n", text, (unsigned long long)number, number);
}
void WarningTwo(const char* text, long long a, long long b) {
  std::fprintf(stderr, "%s %lld / %lld\n", text, a, b);
}

SharedMemory g_memory{};
bool g_mapped = false;
uint8_t* g_base = nullptr;   // start of the shared area, whether from IPC or from the scan
size_t g_bytes = 0;
/*
 * Written by the profiler thread (Start/Update) and read by the ring thread on every present
 * (Beat), hence atomic.
 */
std::atomic<BlockFps*> g_fps{nullptr};
uint64_t g_tick_previous = 0;   // ring thread only
unsigned g_tick_pos = 0;      // same
uint64_t g_frames = 0;    // same
std::atomic<BlockReverseNx*> g_reverse{nullptr};
std::atomic<bool> g_reverse_in_base{false};
std::atomic<bool> g_reverse_active{false};
/* Turned on by the game (cvar sa2_switch_saltynx). On by default. */
std::atomic<int> g_enabled{1};
/*
 * Connection attempts left. The sysmodule may take longer than us to start, so if it is not there at
 * first, it is retried once per second during the first minute.
 */
int g_attempts = 0;  // how many attempts so far

/*
 * The last values published. If the block appears late (because the overlay was opened later,
 * SaltyNX was slow to hand out the memory, or the first allocation did not fit), it has to be seeded
 * right away with these values: otherwise the overlay shows no FPS until the next second and no
 * resolution until the next present, and during loading there may be neither for several seconds.
 * Only the profiler thread touches them (Update / Start).
 */
uint8_t g_last_fps = 0;
float g_last_average = 0.0f;
uint16_t g_last_width = 0;
uint16_t g_last_height = 0;
uint64_t g_last_frames = 0;

/*
 * Tick of the first attempt. The profiler thread calls Start() right at startup (it does not wait
 * for the 8 s of kStartDelayNs), so it serves as the "startup" reference for the success message.
 */
uint64_t g_tick_first_attempt = 0;
/* warnings issued only once; repeating them on every retry filled the log. */
bool g_warned_published = false;
bool g_warned_connection = false;
bool g_warned_block = false;
bool g_warned_no_site = false;
bool g_warned_reverse = false;
bool g_warned_no_memory = false;

/*
 * --- SaltySD IPC. Modern libnx no longer ships the old ipc.h API, but the service speaks plain
 * CMIF, so serviceDispatch works: the header carries the same SFCI and the body the same fields. ---
 */

Result ReserveMemory(Service* s, uint64_t size, uint64_t* displacement) {
  return serviceDispatchInOut(s, 6, size, *displacement, .in_send_pid = true);
}

Result RequestHandler(Service* s, Handle* output) {
  return serviceDispatch(s, 7, .in_send_pid = true,
                         .out_handle_attrs = {SfOutHandleAttr_HipcCopy}, .out_handles = output);
}

Result Finish(Service* s) {
  const uint64_t zero = 0;
  return serviceDispatchIn(s, 0, zero, .in_send_pid = true);
}

/* Looks for a magic by scanning the page 4 bytes at a time, which is how the overlay does it. */
void* SearchMarkIn(uint8_t* base, size_t bytes, uint32_t mark) {
  if (!base) {
    return nullptr;
  }
  for (size_t offset = 0; offset + sizeof(uint32_t) <= bytes; offset += 4) {
    uint32_t read = 0;
    std::memcpy(&read, base + offset, sizeof(read));
    if (read == mark) {
      return base + offset;
    }
  }
  return nullptr;
}

void* SearchMark(uint32_t mark) { return SearchMarkIn(g_base, g_bytes, mark); }

/*
 * The block is only valid while it keeps its magic. The SaltySD page is shared by several clients
 * and its allocation is not reset when we start, so a good pointer can go bad. Checking it is a
 * 4-byte read, and it is done on every use.
 */
bool ValidBlock(const BlockFps* block) { return block != nullptr && block->magic == kMagicFps; }

/*
 * Writes into the block, right away, everything the overlay needs to show its two rows (FPS and
 * RES): the alive mark, the API, the frames per second and the resolution. The overlay shows neither
 * row until the corresponding field has something, so seeding it as soon as the block exists is
 * what makes the data appear without restarting the game.
 */
void SeedBlock(BlockFps* block) {
  block->plugin_active = true;  // the overlay sets it to false to check that we are still alive
  block->api = kApiVulkan;
  block->fps = g_last_fps;
  block->fps_average = g_last_average;
  block->frame_number = g_last_frames;
  if (g_last_width && g_last_height) {
    // `calls` cannot be 0xFFFF: that is the mark the overlay uses to ask whether we know the resolution.
    const uint16_t how_many = g_last_fps ? uint16_t(g_last_fps) : uint16_t(1);
    const ResolutionCalls r = {g_last_width, g_last_height, how_many};
    block->render[0] = r;
    block->viewport[0] = r;
  }
  // The overlay computes its average from ticks[], not from fps_average: with the array at zero it shows
  // "inf". A freshly created block has it at zero, so if a rate has already been measured it is
  // filled in by hand. Beat corrects it with real times as soon as there are two presents.
  if (g_last_fps != 0 && block->ticks[0] == 0) {
    // By index, not by reference: the block is packed and ticks[] sits at an odd offset; taking the
    // address of an element would give an unaligned pointer (-Waddress-of-packed-member).
    const uint32_t per_frame = uint32_t(armGetSystemTickFreq() / g_last_fps);
    for (unsigned i = 0; i < 10; ++i) {
      block->ticks[i] = per_frame;
    }
  }
}

/*
 * A single log line when publishing succeeds, with the seconds since startup. Without it there is
 * no way to know whether the overlay is late because of us or because of it.
 */
void WarnPublished(const BlockFps* block) {
  if (g_warned_published) {
    return;
  }
  g_warned_published = true;
  const uint64_t freq = armGetSystemTickFreq();
  const double seconds =
      freq ? double(armGetSystemTick() - g_tick_first_attempt) / double(freq) : 0.0;
  const unsigned displacement =
      g_base ? unsigned(reinterpret_cast<const uint8_t*>(block) - static_cast<const uint8_t*>(g_base)) : 0u;
  std::fprintf(stderr,
               "[saltynx] PUBLISHED %.2f s after startup: FPS=%u and RES=%ux%u at offset 0x%X\n",
               seconds, unsigned(g_last_fps), unsigned(g_last_width), unsigned(g_last_height),
               displacement);
}

/*
 * With the page already mapped, looks for both magics and keeps them. It is cheap (1024 comparisons
 * of 4 bytes on memory that is already there) and uses no port session, so it can be repeated every
 * second while the block is missing. Returns true if there is an FPS block.
 */
bool Hook() {
  auto* block = static_cast<BlockFps*>(SearchMark(kMagicFps));
  if (block) {
    g_fps.store(block, std::memory_order_release);
    SeedBlock(block);
    WarnPublished(block);
  }
  if (!g_reverse.load(std::memory_order_acquire)) {
    if (auto* reverse = static_cast<BlockReverseNx*>(SearchMark(kMagicReverseNx))) {
      g_reverse.store(reverse, std::memory_order_release);
    }
  }
  return block != nullptr;
}

/*
 * If SaltyNX was injected into this process, its shared memory is already mapped here. The memory
 * map is walked with svcQueryMemory, looking inside the readable shared memory regions.
 */
bool SearchSharedOwnMemory() {
  uint64_t address = 0;
  for (int regions = 0; regions < 4096; ++regions) {
    MemoryInfo info{};
    u32 pages = 0;
    if (R_FAILED(svcQueryMemory(&info, &pages, address))) {
      return false;
    }
    if (info.size == 0) {
      return false;
    }
    if (info.type == MemType_SharedMem && (info.perm & Perm_R) != 0 && info.size >= kSharedSize) {
      auto* base = reinterpret_cast<uint8_t*>(uintptr_t(info.addr));
      const size_t bytes = size_t(info.size) < 0x10000 ? size_t(info.size) : 0x10000;
      if (SearchMarkIn(base, bytes, kMagicFps) || SearchMarkIn(base, bytes, kMagicReverseNx)) {
        g_base = base;
        g_bytes = bytes;
        return true;
      }
    }
    const uint64_t next = info.addr + info.size;
    if (next <= address) {
      return false;
    }
    address = next;
  }
  return false;
}

/*
 * A connection error can come from two very different things that look much alike:
 *   - the single session of the SaltyNX port is busy (its limit, fixed by retrying), or
 *   - we are the ones out of sessions (0x10801 is "resource exhausted", and the kernel also returns
 *     it when the process cannot reserve another session).
 * They are told apart with a control: connecting to "sm:", which always exists and accepts many
 * sessions. If "sm:" fails too, the limit is ours; and the used sessions and the process cap are
 * printed right there.
 */
void Diagnostic(Result rc_saltysd) {
  if (rc_saltysd == 0) {
    Warning("[saltynx] not even attempted: the process has no room for another port session");
  } else {
    WarningNum("[saltynx] cannot connect to the SaltyNX ports. Last error", rc_saltysd);
  }

  // Our own title ID: SaltyNX rejects those above 0x01FFFFFFFFFFFFFF ("is a homebrew application"),
  // which is exactly the range forwarders fall in. Knowing it saves a question.
  u64 title = 0;
  if (R_SUCCEEDED(svcGetInfo(&title, InfoType_ProgramId, CUR_PROCESS_HANDLE, 0))) {
    std::fprintf(stderr, "[saltynx] our TID: %016llX (SaltyNX accepts <= 01FFFFFFFFFFFFFF and without 0x1F00)\n",
                 (unsigned long long)title);
  }

  // Each port separately: "does not exist" (0xF201, not installed) is not the same as "resource exhausted".
  static const char* const kTwoPorts[2] = {"InjectServ", "SaltySD"};
  for (const char* p : kTwoPorts) {
    Handle h = INVALID_HANDLE;
    const Result rc = svcConnectToNamedPort(&h, p);
    if (R_SUCCEEDED(rc)) {
      svcCloseHandle(h);
      Warning(p[0] == 'I' ? "[saltynx] InjectServ port: connects OK" : "[saltynx] SaltySD port: connects OK");
    } else {
      WarningNum(p[0] == 'I' ? "[saltynx] InjectServ port: error" : "[saltynx] SaltySD port: error", rc);
    }
  }

  Handle control = INVALID_HANDLE;
  const Result rc_sm = svcConnectToNamedPort(&control, "sm:");
  if (R_SUCCEEDED(rc_sm)) {
    svcCloseHandle(control);
    Warning("[saltynx] control: 'sm:' DOES give a new session, so the limit is on the SaltyNX port");
  } else {
    WarningNum("[saltynx] control: 'sm:' does not give a session either, so the limit is OURS. Error", rc_sm);
  }

  u64 raw = 0;
  Result rc_lim = svcGetInfo(&raw, InfoType_ResourceLimit, INVALID_HANDLE, 0);
  if (R_FAILED(rc_lim)) {
    rc_lim = svcGetInfo(&raw, InfoType_ResourceLimit, CUR_PROCESS_HANDLE, 0);
  }
  if (R_FAILED(rc_lim)) {
    WarningNum("[saltynx] cannot read the process limits. Error", rc_lim);
    return;
  }
  const Handle limit = static_cast<Handle>(raw);
  struct Resource {
    const char* name;
    LimitableResource which;
  };
  const Resource kResources[] = {
      {"[saltynx] sessions used / cap:", LimitableResource_Sessions},
      {"[saltynx] events used / cap:", LimitableResource_Events},
      {"[saltynx] threads used / cap:", LimitableResource_Threads},
      {"[saltynx] transfer memories used / cap:", LimitableResource_TransferMemories},
  };
  for (const Resource& r : kResources) {
    s64 now = 0;
    s64 cap = 0;
    if (R_SUCCEEDED(svcGetResourceLimitCurrentValue(&now, limit, r.which)) &&
        R_SUCCEEDED(svcGetResourceLimitLimitValue(&cap, limit, r.which))) {
      WarningTwo(r.name, (long long)now, (long long)cap);
    }
  }
  svcCloseHandle(limit);
}

/*
 * --- Making room for a session. ---------------------------------------------------------------------
 * Measured on the console: this process (forwarder + hbloader) has a resource limit with "sessions
 * used / cap: 1 / 1". A single port session, and libnx already uses it for "sm:". That is why
 * both SaltyNX ports and also the "sm:" control failed, all three with 0x10801 (resource
 * exhausted). It is not SaltyNX's fault.
 *
 * Two ways to make room, in this order:
 *   1. Raise the cap of our own resource limit. The kernel requires no privilege for that: only that
 *      the new cap is not lower than what is already in use. But the call (SVC 0x7E) may not be
 *      allowed in the process, and using a forbidden SVC kills the game, so envIsSyscallHinted is
 *      asked first.
 *   2. If that is not possible, release "sm:" for a moment (smExit) and bring it back afterwards. It
 *      is only needed once: the shared memory handle stays with us even if the session is closed.
 */

Handle OpenLimitOfResources() {
  u64 raw = 0;
  if (R_SUCCEEDED(svcGetInfo(&raw, InfoType_ResourceLimit, INVALID_HANDLE, 0))) {
    return static_cast<Handle>(raw);
  }
  if (R_SUCCEEDED(svcGetInfo(&raw, InfoType_ResourceLimit, CUR_PROCESS_HANDLE, 0))) {
    return static_cast<Handle>(raw);
  }
  return INVALID_HANDLE;
}

bool HasSiteForASession(Handle limit) {
  s64 now = 0;
  s64 cap = 0;
  if (R_FAILED(svcGetResourceLimitCurrentValue(&now, limit, LimitableResource_Sessions)) ||
      R_FAILED(svcGetResourceLimitLimitValue(&cap, limit, LimitableResource_Sessions))) {
    return true;  // if it cannot be read, try anyway
  }
  return now < cap;
}

/*
 * Sets *sm_closed to true if "sm:" had to be released (it has to be brought back later).
 * Releasing "sm:" is not tried on every attempt: it is a window of a few milliseconds without the
 * name service and should not be repeated once per second forever. Raising the cap, on the other
 * hand, is permanent and done only once.
 */
bool MakeSite(bool* sm_closed, bool allow_drop_sm) {
  *sm_closed = false;
  const Handle limit = OpenLimitOfResources();
  if (limit == INVALID_HANDLE) {
    return true;
  }
  bool site = HasSiteForASession(limit);

  if (!site) {
    if (envIsSyscallHinted(0x7E)) {  // svcSetResourceLimitLimitValue
      s64 cap = 0;
      svcGetResourceLimitLimitValue(&cap, limit, LimitableResource_Sessions);
      const Result rc = svcSetResourceLimitLimitValue(limit, LimitableResource_Sessions,
                                                      static_cast<u64>(cap + 4));
      if (R_SUCCEEDED(rc)) {
        site = HasSiteForASession(limit);
        WarningTwo("[saltynx] raised the process session cap:", (long long)cap, (long long)(cap + 4));
      } else {
        WarningNum("[saltynx] raising the session cap is not allowed. Error", rc);
      }
    } else {
      Warning("[saltynx] the loader does not allow svcSetResourceLimitLimitValue (SVC 0x7E)");
    }
  }

  if (!site && allow_drop_sm) {
    smExit();  // libnx reference-counts it; if it really closes, there is room
    if (HasSiteForASession(limit)) {
      *sm_closed = true;
      site = true;
      Warning("[saltynx] released 'sm:' for a moment to make room");
    } else {
      smInitialize();  // it did not close: restore the count and leave it as it was
      Warning("[saltynx] even releasing 'sm:' there is no room for a session");
    }
  }

  svcCloseHandle(limit);
  return site;
}

/* Brings "sm:" back on exit, whatever happens. */
struct ReturnSm {
  bool active = false;
  ~ReturnSm() {
    if (active) {
      smInitialize();
    }
  }
};

}  // namespace

void Start() {
  if (!g_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  if (g_tick_first_attempt == 0) {
    g_tick_first_attempt = armGetSystemTick();
  }

  /*
   * The exit condition is not "the page is mapped" but "we have a good block". Otherwise, if
   * mapping succeeded but there was no room for the block (or the magic disappeared because another
   * client rewrote the page), g_mapped stayed true and this was never retried: the only way out was
   * restarting the game with the overlay already on. The magic is checked and, if missing, the block
   * is attached again.
   */
  BlockFps* previous_block = g_fps.load(std::memory_order_acquire);
  if (!ValidBlock(previous_block)) {
    g_fps.store(nullptr, std::memory_order_release);
  } else if (g_reverse.load(std::memory_order_acquire) != nullptr) {
    return;  // both in place: nothing to do
  }

  // The cheap part first: if the page is already mapped, it is enough to search for the magics again.
  // It uses no sessions and no IPC, so it can be done once per second forever.
  if (g_mapped && Hook()) {
    return;
  }

  // Path 1: IPC. SaltyNX creates two ports and each accepts a single session, so both are tried, with
  // a few retries: the sysmodule may take longer than us to start.
  // First there has to be room: this process only allows one port session and libnx uses it for "sm:".
  ReturnSm return_value;
  // Raising the cap is tried from the very start (it is permanent and bothers nobody). Releasing "sm:"
  // only from the third attempt on, when startup has already opened its services.
  // That window used to close at attempt 15 and never reopen: if the port was busy during that quarter
  // of a minute, the only option was restarting the game. After the window, it is retried once per
  // minute: it is still a pause of a few milliseconds without the name service, but it is no longer
  // abandoned.
  const bool drop_sm = g_attempts >= 3 && (g_attempts < 15 || (g_attempts % 60) == 0);
  const bool with_site = MakeSite(&return_value.active, drop_sm);
  // With "sm:" closed it has to be quick: a single pass. On the first attempt it insists (the
  // sysmodule may be starting up); in the per-second retries two passes are enough, since the overlay
  // also needs the port's only session.
  const int laps = return_value.active ? 1 : (g_attempts == 0 ? 20 : 2);
  bool ready = false;

  /*
   * "SaltySD" first. Both ports accept connections, but the one that serves commands 6 and 7 (shared
   * memory slot and handle) is "SaltySD": it is the one the overlays talk to. "InjectServ" connects
   * and then answers by closing the session (0xF601, ConnectionClosed). An earlier version tried it
   * first, took it as good and spent the whole attempt: 65 seconds looking at the wrong port, once per
   * second, until by chance it was busy and the right one was used. Now, if a port connects but does
   * not give the memory, it is closed and the next one is tried in the same pass.
   */
  static const char* const kPorts[2] = {"SaltySD", "InjectServ"};

  uint64_t displacement = 0;
  bool reserved = false;
  uint64_t nx_displacement = 0;
  bool reserved_nx = false;
  Result rc_port = 0;
  Result rc_handler = 0;
  const char* name = nullptr;

  for (int i = 0; i < laps && !ready && with_site; ++i) {
    for (const char* candidate : kPorts) {
      Handle port = INVALID_HANDLE;
      rc_port = svcConnectToNamedPort(&port, candidate);
      if (R_FAILED(rc_port)) {
        continue;
      }
      Service service{};
      service.session = port;

      // First the handle and the mapping, and only then allocate what is missing. SaltySD hands out the
      // page with a counter that it only resets when it injects into a game (hijack_bootstrap), and it
      // never injects into us: allocating on every boot would eat the 4 KB in twenty sessions. By
      // checking first whether the block is already there, later boots reuse the same slot.
      // If the page was already mapped by an earlier attempt, the handle is not requested again (that
      // would map the same memory twice): it goes straight to allocating what is missing.
      bool have_page = g_mapped;
      if (!have_page) {
        Handle memory = INVALID_HANDLE;
        rc_handler = RequestHandler(&service, &memory);
        if (R_SUCCEEDED(rc_handler)) {
          shmemLoadRemote(&g_memory, memory, kSharedSize, Perm_Rw);
          if (R_SUCCEEDED(shmemMap(&g_memory))) {
            g_base = static_cast<uint8_t*>(shmemGetAddr(&g_memory));
            g_bytes = kSharedSize;
            g_mapped = true;
            have_page = true;
          }
        }
      }
      if (have_page) {
        ready = true;
        name = candidate;
        if (!SearchMark(kMagicFps)) {
          reserved = R_SUCCEEDED(ReserveMemory(&service, sizeof(BlockFps), &displacement));
        }
        if (!SearchMark(kMagicReverseNx)) {
          reserved_nx = R_SUCCEEDED(ReserveMemory(&service, sizeof(BlockReverseNx), &nx_displacement));
        }
      }

      Finish(&service);  // command 0: the server closes its side
      // And our end has to be released too. The port accepts a single session
      // (svcManageNamedPort(..., 1)), so leaving the handle open would keep the overlay from ever
      // connecting again. The shared memory handle is already ours and does not depend on the session.
      svcCloseHandle(port);
      if (ready) {
        break;
      }
    }
    if (!ready) {
      svcSleepThread(10 * 1000 * 1000);  // 10 ms
    }
  }

  if (return_value.active) {  // room again: bring "sm:" back without waiting for the end
    smInitialize();
    return_value.active = false;
  }

  if (name && !g_warned_connection) {
    g_warned_connection = true;  // once; this is retried every second and used to fill the log
    Warning(name[0] == 'S' ? "[saltynx] connected through SaltySD" : "[saltynx] connected through InjectServ");
  }
  if (!g_mapped) {
    if (R_FAILED(rc_handler) && !g_warned_no_memory) {
      // A port connected but did not serve the memory. The attempt is not lost: the other one has already
      // been tried. Only once: this is retried every second for the whole session and used to fill the log.
      g_warned_no_memory = true;
      WarningNum("[saltynx] some port connects but does not give the shared memory; error", rc_handler);
    }
    // Path 2: no free session. If SaltyNX was injected into this process, its shared memory is already mapped here.
    if (!SearchSharedOwnMemory()) {
      // It is retried for the whole session, not just one minute. Error 0x10801 (LimitReached) says
      // that the port exists but its only session is busy, so it may be freed later.
      ++g_attempts;
      // The first diagnosis is done on attempt 5, not 1: the first attempt happens at second zero of
      // startup, and it makes no sense to diagnose before having tried releasing "sm:".
      if (g_attempts == 5 || g_attempts == 30) {
        Diagnostic(rc_port);
      } else if (g_attempts % 600 == 0) {
        WarningNum("[saltynx] still retrying without success. Error", rc_port);
      }
      return;
    }
    g_mapped = true;
    Warning("[saltynx] no session on the ports, but their shared memory was already mapped here");
  }

  // The FPS block: if SaltyNX already left one in this process it is overwritten (its fields would be
  // 0, since it has nothing to hook); otherwise the slot allocated through IPC is used.
  auto* existing = static_cast<BlockFps*>(SearchMark(kMagicFps));
  if (existing) {
    g_fps.store(existing, std::memory_order_release);
    if (!g_warned_block) {
      g_warned_block = true;
      Warning("[saltynx] FPS block already present: writing over it");
    }
  } else if (reserved) {
    auto* block = reinterpret_cast<BlockFps*>(g_base + displacement);
    std::memset(block, 0, sizeof(*block));
    block->magic = kMagicFps;
    g_fps.store(block, std::memory_order_release);
    WarningNum("[saltynx] FPS block created at offset", (long long)displacement);
  } else if (!g_warned_no_site) {
    g_warned_no_site = true;
    Warning("[saltynx] no room for the FPS block; retrying once per second");
  }
  if (BlockFps* block = g_fps.load(std::memory_order_acquire)) {
    // FPS and resolution at once, without waiting for the next second or the next present.
    SeedBlock(block);
    WarnPublished(block);
    g_attempts = 0;
  } else {
    // Mapped but without a block. This is not taken as success: the counter keeps going up so the next
    // tick can ask for room and a session again.
    ++g_attempts;
  }

  /*
   * Reverse-NX. Its block is not created by the overlay either: it is created by the plugin SaltyNX
   * injects into the game, and it is not injected into us, so its overlay said "ReverseNX-RT is not
   * running!". We create it here like the FPS one and its overlay starts working: the player picks
   * handheld or docked and the game obeys. The overlay requires no handshake for this (it only looks
   * for the "NXRT" magic), but it does require `pluginActive` to show the controls, and that means
   * "the game has asked for the mode": it is set in BaseMode.
   */
  auto* reverse = static_cast<BlockReverseNx*>(SearchMark(kMagicReverseNx));
  if (!reverse && reserved_nx) {
    reverse = reinterpret_cast<BlockReverseNx*>(g_base + nx_displacement);
    std::memset(reverse, 0, sizeof(*reverse));
    reverse->magic = kMagicReverseNx;
    reverse->by_default = true;  // the system decides until the player says otherwise
    WarningNum("[saltynx] Reverse-NX block created at offset", (long long)nx_displacement);
  } else if (!g_warned_reverse) {
    g_warned_reverse = true;  // once, since this is retried every second
    Warning(reverse ? "[saltynx] Reverse-NX block already present" : "[saltynx] no room for the Reverse-NX block");
  }
  // Only stored if there is one. Otherwise a retry would write nullptr over a good pointer.
  if (reverse) {
    g_reverse.store(reverse, std::memory_order_release);
  }
}

void Update(double fps_second, double fps_average, uint32_t width, uint32_t height, uint64_t frames) {
  // Remembered before publishing. If the block appears later (overlay opened afterwards, or SaltyNX
  // slow to hand out the memory), it is seeded with these values the moment it exists.
  const double cap = fps_second < 0.0 ? 0.0 : (fps_second > 255.0 ? 255.0 : fps_second);
  g_last_fps = uint8_t(cap + 0.5);
  g_last_average = float(fps_average);
  if (width && height) {
    g_last_width = uint16_t(width);
    g_last_height = uint16_t(height);
  }
  g_last_frames = frames;

  /*
   * The block may be missing (SaltyNX was not handing out memory yet, or the allocation did not fit)
   * or may no longer be ours (the page is shared by several clients). The magic is checked on every
   * tick and, if it is missing, the block is attached again. This is what allows the overlay to be
   * opened at any time without restarting the game: retrying only while the page was unmapped meant
   * that once mapped it was never looked at again.
   */
  BlockFps* block = g_fps.load(std::memory_order_acquire);
  // This starts at second zero, before the game applies its cvars, so the switch has to be checked here
  // too: if it is off, the block is released and Beat stops writing as well. Turning it back on
  // recovers it on the next tick.
  if (!g_enabled.load(std::memory_order_relaxed)) {
    if (block) {
      block->plugin_active = false;
      g_fps.store(nullptr, std::memory_order_release);
    }
    return;
  }
  if (!ValidBlock(block)) {
    Start();
    block = g_fps.load(std::memory_order_acquire);
  }
  if (block) {
    // Both values, always: FPS and resolution. Leaving the resolution to Beat is not enough, because
    // Beat only runs while the game is presenting: during loading the overlay was left without the RES row.
    SeedBlock(block);
  }
  if (BlockReverseNx* reverse = g_reverse.load(std::memory_order_acquire)) {
    g_reverse_active.store(!reverse->by_default, std::memory_order_relaxed);
    g_reverse_in_base.store(reverse->in_base, std::memory_order_relaxed);
  }
}

/*
 * The heartbeat, once per frame from the ring thread.
 *
 * The overlay does not believe the game is alive until it answers two handshakes, and both are very
 * short (Status-Monitor-Overlay, source/Utils.hpp and source/modes/Resolutions.hpp):
 *
 *   NxFps->pluginActive = false;  svcSleepThread(100'000'000);  if (NxFps->pluginActive) GameRunning = true;
 *   NxFps->renderCalls[0].calls = 0xFFFF;  ... if (renderCalls[0].calls != 0xFFFF) resolutionLookup = 2;
 *
 * With one update per second it was not enough: the overlay said "Game is not running or it's
 * incompatible". Also, it computes the FPS average from FPSticks[10] (frequency / average of the
 * ticks), so with the array at zero it showed "inf". Here it is filled with the real time between
 * presents.
 */
void Beat(uint32_t width, uint32_t height) {
  BlockFps* block = g_fps.load(std::memory_order_acquire);
  // If the magic is gone, the pointer is not valid. It is dropped and the profiler thread's one-second
  // tick recovers it: this runs on the ring thread on every frame, so no IPC and no searching here.
  if (!ValidBlock(block)) {
    if (block) {
      g_fps.store(nullptr, std::memory_order_release);
    }
    return;
  }
  block->plugin_active = true;
  block->api = kApiVulkan;

  // If the block is a different one (just attached because the overlay was opened now), the previous
  // time is minutes old: that first measurement is discarded, or the overlay would show a very long
  // frame in the average.
  static const BlockFps* last_seen = nullptr;  // only the ring thread touches it
  if (last_seen != block) {
    last_seen = block;
    g_tick_previous = 0;
  }

  const uint64_t now = armGetSystemTick();
  if (g_tick_previous != 0) {
    const uint64_t jump = now - g_tick_previous;
    block->ticks[g_tick_pos] = uint32_t(jump > 0xFFFFFFFFull ? 0xFFFFFFFFull : jump);
    g_tick_pos = (g_tick_pos + 1) % 10;
    ++g_frames;
    block->frame_number = g_frames;
  }
  g_tick_previous = now;

  if (width && height) {
    // `calls` cannot be 0xFFFF: that is the mark the overlay asks with. It is set to the frames of the
    // last second, which is what NX-FPS counts.
    const uint16_t how_many = block->fps ? block->fps : uint16_t(1);
    const ResolutionCalls r = {uint16_t(width), uint16_t(height), how_many};
    block->render[0] = r;
    block->viewport[0] = r;
  }
}

void Enable(bool enabled) { g_enabled.store(enabled ? 1 : 0, std::memory_order_relaxed); }

bool BaseMode(bool real) {
  BlockReverseNx* reverse = g_reverse.load(std::memory_order_acquire);
  if (!reverse) {
    return real;
  }
  // In Reverse-NX, `pluginActive` means "the game has asked for the mode"; without it its overlay says
  // "Game didn't check any mode!" and does not show the controls.
  reverse->plugin_active = true;
  if (reverse->by_default) {
    reverse->in_base = real;  // the system decides: mirror it so the overlay shows the real mode
    return real;
  }
  return reverse->in_base;
}

ReverseStateNx ReverseState() {
  const BlockReverseNx* reverse = g_reverse.load(std::memory_order_acquire);
  if (!reverse) {
    return {false, false, false, false};
  }
  return {true, reverse->in_base, reverse->by_default, reverse->plugin_active};
}

}  // namespace rex::ui::switch_saltynx

/* Switch from the game, before the profiler thread starts. */
extern "C" void RexSwitchSaltyNxEnable(int enabled) {
  rex::ui::switch_saltynx::Enable(enabled != 0);
}

#endif  // REX_PLATFORM_SWITCH
