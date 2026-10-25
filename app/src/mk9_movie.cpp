// mk9-nx (2026-10-24): MK9's cutscenes (Movies\*.wmv: WMV3 Main 1280x720 30 fps with B frames, WMA2) are played by the
// XDK's XMV player (XMedia2, 0x82CE7xxx), whose video codec is Microsoft's embedded WMV9 decoder (wmvdec.c).
// Recompiled, that decoder ran 23-35 fps on a whole core of a V1 at 1785 MHz, so the picture fell behind the sound
// (~4 s by the end of a Story cutscene). The Switch's FFmpeg decodes the same file at 124 fps on one core.
//
// The codec is a plug-in of the player. A factory (sub_82D25910) fills a table of entry points, each called with
// r3 = the table:
//   [0] Init 82D255C0 (table, media type, ?, data callback, its context, player, &handle): the media type's +8 points
//       to the format (+8 width, +12 height, +20 FOURCC, +48 u16 private data size, +52 its address: WMV3's 4-byte
//       sequence header). The original allocates its 4240-byte state from the player (82CF94C0(player, 25, size,
//       &p)), keeps it at table+32 and ends by registering the codec's message handler 82D25580 with it
//       (82CF9018(player, handler, state, &state->4232)), returning that handle in *r9. The player then sends the
//       handler 0x80000 (the output format: 82D25470 writes width, height and frame bytes at state+32/+36/+108 and
//       answers 0x90000) before it asks for any frame. The replacement does the same with that original handler.
//   [1] Close 82D25280 (table): the original also zeroes table words 0-5 and 8.
//   [2] Decode 82D25330 (table, &bytes ready, ?): decodes one frame; *bytes = width * height * 3 / 2 when one is ready.
//   [3] GetOutput 82D25390 (table, buffer, capacity, &size, &flags, &time u64, &one): the frame as planar Y, U, V
//       (I420, no padding), its time in ms; *one = 1. 0x80500006 when the buffer is too small.
//   [4] Reset 82D25228 (table).
// The data callback (XMV's sub_82CFA1E0) is called (context, &ptr, &size, &u32, &time u64, &has_time, &frame size,
// &u32): each call gives the next piece of the compressed stream (up to 512 bytes), the first piece of a frame with
// has_time and the time in ms; it returns 0x00500001 for the piece that ends a frame, a negative HRESULT on error.
// Traced on PC (mk9_movie_trace) from the original's calls; the first frame of WBIntroLogo.wmv came out of both
// decoders with the same Y and U bytes.

#include <atomic>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include <rex/cvar.h>
#include <rex/hook.h>
#include <rex/logging.h>
#include <rex/ppc.h>
#include <rex/system/xmemory.h>

extern "C" {
#include "libavcodec/avcodec.h"
}

REXCVAR_DEFINE_BOOL(mk9_movie_ffmpeg, true, "MK9",
                    "Cutscenes: the WMV video decoded by the host's FFmpeg instead of the game's own decoder")
    .lifecycle(rex::cvar::Lifecycle::kInitOnly);
REXCVAR_DEFINE_INT32(mk9_movie_trace, 0, "MK9",
                     "Log this many calls of the cutscene codec's interface (XMV's WMV decoder), each kind")
    .range(0, 100000);

namespace {

constexpr uint32_t kBufferSmall = 0x80500006u;  // GetOutput: the buffer is too small (the original's code)
constexpr uint32_t kFail = 0x80004005u;         // E_FAIL

uint8_t* Raw(uint8_t* base, uint32_t a) { return base + a + (a >= 0xE0000000u ? 0x1000u : 0u); }
uint32_t Ld32(uint8_t* base, uint32_t a) {
  uint32_t v;
  std::memcpy(&v, Raw(base, a), 4);
  return __builtin_bswap32(v);
}
uint64_t Ld64(uint8_t* base, uint32_t a) {
  uint64_t v;
  std::memcpy(&v, Raw(base, a), 8);
  return __builtin_bswap64(v);
}
// Every write of guest memory here marks its store cards (rex/system/xmemory.h), as the guest's stores would: the
// decoded frames become textures Renut proves by them.
void Mark(uint8_t* base, uint32_t a, size_t n) { rex::memory::MarkStoreCardsAt(base, a, uint32_t(n)); }
void St32(uint8_t* base, uint32_t a, uint32_t v) {
  v = __builtin_bswap32(v);
  std::memcpy(Raw(base, a), &v, 4);
  Mark(base, a, 4);
}
void St64(uint8_t* base, uint32_t a, uint64_t v) {
  v = __builtin_bswap64(v);
  std::memcpy(Raw(base, a), &v, 8);
  Mark(base, a, 8);
}

bool Trace(std::atomic<int>& n) {
  const int limit = REXCVAR_GET(mk9_movie_trace);
  return limit > 0 && n.fetch_add(1, std::memory_order_relaxed) < limit;
}

struct Frame {
  std::vector<uint8_t> yuv;  // I420, width x height, no padding
  int64_t time_ms = 0;
};

struct Decoder {
  uint32_t player = 0, state = 0;  // the guest state block (the original's layout, for its message handler)
  AVCodecContext* av = nullptr;
  AVFrame* frame = nullptr;
  AVPacket* packet = nullptr;
  uint32_t width = 0, height = 0, callback = 0, callback_ctx = 0;
  std::vector<uint8_t> data;  // the frame being gathered from the callback's pieces
  std::deque<Frame> ready;
  bool eof = false;
  uint64_t frames_in = 0, frames_out = 0;
  // Output times: the data callback gives a frame's time only now and then (the original decoder fills the rest
  // in: 0, 33, 66, 100 ms ...). A known time anchors the count; the others are anchor + frames * 1000 / fps.
  int64_t anchor_ms = 0, since_anchor = 0;
  uint32_t fps = 30;

  ~Decoder() {  // the guest state is released by Close
    av_packet_free(&packet);
    av_frame_free(&frame);
    avcodec_free_context(&av);
  }
  uint32_t FrameBytes() const { return width * height * 3 / 2; }
};

std::mutex g_mutex;
std::unordered_map<uint32_t, std::unique_ptr<Decoder>> g_decoders;  // by codec table (guest address)

Decoder* Find(uint32_t table) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_decoders.find(table);
  return it == g_decoders.end() ? nullptr : it->second.get();
}

// A guest function called from the host: arguments in r3-r10, out values in a scratch frame below the guest stack
// pointer (out = its +0x100; the guest ABI lets a callee use the area above its own frame only for its caller's
// saves). Returns r3.
constexpr uint32_t kOut = 0x100;
uint32_t CallGuest(PPCContext& ctx, uint8_t* base, uint32_t fn, std::initializer_list<uint32_t> args,
                   uint32_t* out_area = nullptr) {
  const uint64_t r1 = ctx.r1.u64, lr = ctx.lr;
  const uint32_t frame = (ctx.r1.u32 - 0x200) & ~0xFu;
  for (uint32_t i = 0; i < 0x40; i += 4) St32(base, frame + kOut + i, 0);
  if (out_area) *out_area = frame + kOut;
  ctx.r1.u64 = frame;
  PPCRegister* regs[] = {&ctx.r3, &ctx.r4, &ctx.r5, &ctx.r6, &ctx.r7, &ctx.r8, &ctx.r9, &ctx.r10};
  size_t i = 0;
  for (uint32_t a : args) regs[i++]->u64 = a == ~0u - 1 ? frame + kOut : a;  // ~0u - 1: the out area
  rex::runtime::ResolveIndirectFunction(fn)(ctx, base);
  const uint32_t result = ctx.r3.u32;
  ctx.r1.u64 = r1;
  ctx.lr = lr;
  return result;
}
constexpr uint32_t kOutArg = ~0u - 1;

constexpr uint32_t kPlayerAlloc = 0x82CF94C0, kPlayerFree = 0x82CF9518;        // (player, 25, size, &p) / (player, 25, &p)
constexpr uint32_t kPlayerRegister = 0x82CF9018, kPlayerUnregister = 0x82CF90B0;  // (player, fn, ctx, &h) / (player, h)
constexpr uint32_t kCodecHandler = 0x82D25580, kStateSize = 4240;

// The player's data callback, called from the host.
struct Piece {
  uint32_t result = 0, ptr = 0, size = 0, has_time = 0, whole = 0;
  uint64_t time_ms = 0;
};
Piece CallData(PPCContext& ctx, uint8_t* base, const Decoder& d) {
  const uint64_t r1 = ctx.r1.u64, lr = ctx.lr;
  const uint32_t frame = (ctx.r1.u32 - 0x200) & ~0xFu, out = frame + 0x100;
  for (uint32_t i = 0; i < 0x40; i += 4) St32(base, out + i, 0);
  ctx.r1.u64 = frame;
  ctx.r3.u64 = d.callback_ctx;
  ctx.r4.u64 = out + 0x00;   // ptr
  ctx.r5.u64 = out + 0x04;   // size
  ctx.r6.u64 = out + 0x08;   // u32
  ctx.r7.u64 = out + 0x10;   // time (u64)
  ctx.r8.u64 = out + 0x18;   // has time
  ctx.r9.u64 = out + 0x1C;   // the frame's whole size
  ctx.r10.u64 = out + 0x20;  // u32
  rex::runtime::ResolveIndirectFunction(d.callback)(ctx, base);
  Piece p;
  p.result = ctx.r3.u32;
  p.ptr = Ld32(base, out + 0x00);
  p.size = Ld32(base, out + 0x04);
  p.time_ms = Ld64(base, out + 0x10);
  p.has_time = Ld32(base, out + 0x18);
  p.whole = Ld32(base, out + 0x1C);
  ctx.r1.u64 = r1;
  ctx.lr = lr;
  return p;
}

void Collect(Decoder& d) {
  while (avcodec_receive_frame(d.av, d.frame) == 0) {
    Frame f;
    const int64_t known = d.frame->best_effort_timestamp != AV_NOPTS_VALUE ? d.frame->best_effort_timestamp : d.frame->pts;
    if (known != AV_NOPTS_VALUE) {
      d.anchor_ms = known;
      d.since_anchor = 0;
    }
    f.time_ms = d.anchor_ms + d.since_anchor * 1000 / d.fps;
    d.since_anchor++;
    f.yuv.resize(d.FrameBytes());
    uint8_t* dst = f.yuv.data();
    for (int plane = 0; plane < 3; ++plane) {
      const uint32_t w = plane ? d.width / 2 : d.width, h = plane ? d.height / 2 : d.height;
      for (uint32_t y = 0; y < h; ++y, dst += w)
        std::memcpy(dst, d.frame->data[plane] + size_t(y) * d.frame->linesize[plane], w);
    }
    d.ready.push_back(std::move(f));
    d.frames_out++;
    av_frame_unref(d.frame);
  }
}

// One compressed frame from the callback into FFmpeg. False at the end of the data (or an error). A frame (an ASF
// media object) comes in pieces: its first one carries the time and the frame's whole size; 0x00500001 only ends an
// ASF payload, which can stop in the middle of a frame (WBIntroLogo's 185811-byte key frame took 16 of them).
bool FeedOne(PPCContext& ctx, uint8_t* base, Decoder& d) {
  d.data.clear();
  int64_t time_ms = AV_NOPTS_VALUE;
  uint32_t pieces = 0, whole = 0;
  for (;;) {
    const Piece p = CallData(ctx, base, d);
    if (static_cast<int32_t>(p.result) < 0) {
      static std::atomic<int> logged{0};
      if (logged.fetch_add(1) < 8)
        REXLOG_INFO("[movie] data callback {:08X} after {} frames in, {} out", p.result, d.frames_in, d.frames_out);
      return false;
    }
    if (pieces++ == 0) {
      whole = p.whole;
      if (p.has_time) time_ms = static_cast<int64_t>(p.time_ms);
    }
    if (p.size) {
      const size_t at = d.data.size();
      d.data.resize(at + p.size);
      std::memcpy(d.data.data() + at, Raw(base, p.ptr), p.size);
    }
    if (d.data.size() >= whole) break;
  }
  const size_t n = d.data.size();
  {
    static std::atomic<int> logged{0};
    if (logged.fetch_add(1) < REXCVAR_GET(mk9_movie_trace))
      REXLOG_INFO("[movie] frame {}: {} bytes in {} pieces (whole {}), time {}", d.frames_in, n, pieces, whole, time_ms);
  }
  d.data.resize(n + AV_INPUT_BUFFER_PADDING_SIZE, 0);
  av_packet_unref(d.packet);
  d.packet->data = d.data.data();
  d.packet->size = static_cast<int>(n);
  d.packet->pts = time_ms;
  d.packet->dts = AV_NOPTS_VALUE;
  if (const int err = avcodec_send_packet(d.av, d.packet); err < 0) {
    static std::atomic<int> logged{0};
    char text[64];
    av_strerror(err, text, sizeof(text));
    if (logged.fetch_add(1) < 8)
      REXLOG_WARN("[movie] FFmpeg refused frame {} ({} bytes, time {}): {} ({}); first bytes {:02X}{:02X}{:02X}{:02X}",
                  d.frames_in, n, time_ms, text, err, d.data[0], d.data[1], d.data[2], d.data[3]);
  }
  d.frames_in++;
  Collect(d);
  return true;
}

}  // namespace

REX_EXTERN(__imp__sub_82D255C0);
REX_HOOK_RAW(sub_82D255C0) {  // Init
  static std::atomic<int> n{0};
  const uint32_t table = ctx.r3.u32, media = ctx.r4.u32, callback = ctx.r6.u32, callback_ctx = ctx.r7.u32;
  const uint32_t player = ctx.r8.u32, handle_out = ctx.r9.u32;
  const uint32_t format = Ld32(base, media + 8);
  const uint32_t width = Ld32(base, format + 8), height = Ld32(base, format + 12), fourcc = Ld32(base, format + 20);
  const uint32_t extra_n = Ld32(base, format + 48) >> 16, extra_p = Ld32(base, format + 52);
  const AVCodecID id = fourcc == 0x574D5633u ? AV_CODEC_ID_WMV3 : fourcc == 0x57564331u ? AV_CODEC_ID_VC1 : AV_CODEC_ID_NONE;
  const AVCodec* codec = id != AV_CODEC_ID_NONE ? avcodec_find_decoder(id) : nullptr;
  if (!REXCVAR_GET(mk9_movie_ffmpeg) || !codec || !width || !height || (width | height) & 1 || extra_n > 4096) {
    if (REXCVAR_GET(mk9_movie_ffmpeg))
      REXLOG_WARN("[movie] {:08X}: FOURCC {:08X} {}x{}: the game's own decoder", table, fourcc, width, height);
    __imp__sub_82D255C0(ctx, base);
    return;
  }
  auto d = std::make_unique<Decoder>();
  d->width = width;
  d->height = height;
  d->callback = callback;
  d->callback_ctx = callback_ctx;
  d->av = avcodec_alloc_context3(codec);
  d->av->width = static_cast<int>(width);
  d->av->height = static_cast<int>(height);
  d->av->thread_count = 1;
  d->av->pkt_timebase = AVRational{1, 1000};
  if (extra_n) {
    d->av->extradata = static_cast<uint8_t*>(av_mallocz(extra_n + AV_INPUT_BUFFER_PADDING_SIZE));
    std::memcpy(d->av->extradata, Raw(base, extra_p), extra_n);
    d->av->extradata_size = static_cast<int>(extra_n);
  }
  d->frame = av_frame_alloc();
  d->packet = av_packet_alloc();
  if (avcodec_open2(d->av, codec, nullptr) < 0) {
    REXLOG_WARN("[movie] {:08X}: FFmpeg could not open {}; the game's own decoder", table, codec->name);
    __imp__sub_82D255C0(ctx, base);
    return;
  }
  // The original's state block and message handler (see the header).
  uint32_t out = 0;
  if (static_cast<int32_t>(CallGuest(ctx, base, kPlayerAlloc, {player, 25, kStateSize, kOutArg}, &out)) < 0) {
    REXLOG_WARN("[movie] {:08X}: the player could not allocate the codec state; the game's own decoder", table);
    __imp__sub_82D255C0(ctx, base);
    return;
  }
  d->player = player;
  d->state = Ld32(base, out);
  std::memset(Raw(base, d->state), 0, kStateSize);
  Mark(base, d->state, kStateSize);
  St32(base, d->state + 0, callback);
  St32(base, d->state + 4, callback_ctx);
  St32(base, d->state + 4228, player);
  const uint32_t reg = CallGuest(ctx, base, kPlayerRegister, {player, kCodecHandler, d->state, d->state + 4232});
  if (static_cast<int32_t>(reg) < 0) {
    REXLOG_WARN("[movie] {:08X}: registering the codec's handler failed ({:08X})", table, reg);
    ctx.r3.u64 = reg;
    return;
  }
  if (handle_out) St32(base, handle_out, Ld32(base, d->state + 4232));
  St32(base, table + 32, d->state);
  REXLOG_INFO("[movie] {:08X}: {} {}x{} ({} bytes of private data) decoded by FFmpeg", table, codec->name, width, height,
              extra_n);
  if (Trace(n))
    REXLOG_INFO("[movie] Init({:08X} media {:08X} cb {:08X} ctx {:08X} player {:08X}) state {:08X} handle {}", table,
                media, callback, callback_ctx, player, d->state, Ld32(base, d->state + 4232));
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g_decoders[table] = std::move(d);
  }
  ctx.r3.u64 = 0;
}

REX_EXTERN(__imp__sub_82D25330);
REX_HOOK_RAW(sub_82D25330) {  // Decode
  static std::atomic<int> n{0};
  const uint32_t table = ctx.r3.u32, out = ctx.r4.u32;
  Decoder* d = Find(table);
  if (!d) {
    __imp__sub_82D25330(ctx, base);
    return;
  }
  while (d->ready.empty() && !d->eof) {
    if (!FeedOne(ctx, base, *d)) {
      d->eof = true;
      avcodec_send_packet(d->av, nullptr);  // drain: the frames FFmpeg holds back for reordering
      Collect(*d);
    }
  }
  const bool have = !d->ready.empty();
  if (out) St32(base, out, have ? d->FrameBytes() : 0);
  ctx.r3.u64 = have ? 0 : kFail;
  if (Trace(n))
    REXLOG_INFO("[movie] Decode({:08X}) = {:08X}, {} ready, in {} out {}", table, ctx.r3.u32, d->ready.size(),
                d->frames_in, d->frames_out);
}

REX_EXTERN(__imp__sub_82D25390);
REX_HOOK_RAW(sub_82D25390) {  // GetOutput
  static std::atomic<int> n{0};
  const uint32_t table = ctx.r3.u32, buffer = ctx.r4.u32, capacity = ctx.r5.u32;
  const uint32_t size_out = ctx.r6.u32, flags_out = ctx.r7.u32, time_out = ctx.r8.u32, one_out = ctx.r9.u32;
  Decoder* d = Find(table);
  if (!d) {
    __imp__sub_82D25390(ctx, base);
    return;
  }
  {
    // The size the player negotiated (written by the original handler) against the decoder's.
    static std::atomic<int> checked{0};
    const uint32_t w = Ld32(base, d->state + 32), h = Ld32(base, d->state + 36), bytes = Ld32(base, d->state + 108);
    if ((w != d->width || h != d->height || bytes != d->FrameBytes()) && checked.fetch_add(1) < 4)
      REXLOG_WARN("[movie] {:08X}: the player negotiated {}x{} ({} bytes), the stream is {}x{}", table, w, h, bytes,
                  d->width, d->height);
  }
  if (d->FrameBytes() > capacity) {
    ctx.r3.u64 = kBufferSmall;
    return;
  }
  if (d->ready.empty()) {
    if (size_out) St32(base, size_out, 0);
    if (flags_out) St32(base, flags_out, 0);
    ctx.r3.u64 = 0;
    return;
  }
  Frame f = std::move(d->ready.front());
  d->ready.pop_front();
  std::memcpy(Raw(base, buffer), f.yuv.data(), f.yuv.size());
  Mark(base, buffer, f.yuv.size());
  if (size_out) St32(base, size_out, static_cast<uint32_t>(f.yuv.size()));
  if (flags_out) St32(base, flags_out, 0);
  if (time_out) St64(base, time_out, static_cast<uint64_t>(f.time_ms));
  if (one_out) St32(base, one_out, 1);
  ctx.r3.u64 = 0;
  if (Trace(n)) REXLOG_INFO("[movie] GetOutput({:08X}) time {} ms, {} left", table, f.time_ms, d->ready.size());
}

REX_EXTERN(__imp__sub_82D25228);
REX_HOOK_RAW(sub_82D25228) {  // Reset
  const uint32_t table = ctx.r3.u32;
  Decoder* d = Find(table);
  if (!d) {
    __imp__sub_82D25228(ctx, base);
    return;
  }
  avcodec_flush_buffers(d->av);
  d->ready.clear();
  d->eof = false;
  REXLOG_INFO("[movie] {:08X}: reset after {} frames in, {} out", table, d->frames_in, d->frames_out);
  ctx.r3.u64 = 0;
}

REX_EXTERN(__imp__sub_82D25280);
REX_HOOK_RAW(sub_82D25280) {  // Close
  const uint32_t table = ctx.r3.u32;
  std::unique_ptr<Decoder> d;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_decoders.find(table);
    if (it != g_decoders.end()) {
      d = std::move(it->second);
      g_decoders.erase(it);
    }
  }
  if (!d) {
    __imp__sub_82D25280(ctx, base);
    return;
  }
  REXLOG_INFO("[movie] {:08X}: closed after {} frames in, {} out", table, d->frames_in, d->frames_out);
  CallGuest(ctx, base, kPlayerUnregister, {d->player, Ld32(base, d->state + 4232)});
  St32(base, table + 32, d->state);
  CallGuest(ctx, base, kPlayerFree, {d->player, 25, table + 32});  // &p: table+32 holds the state's address
  for (uint32_t i = 0; i < 6; ++i) St32(base, table + 4 * i, 0);
  St32(base, table + 32, 0);
  ctx.r3.u64 = 0;
}
