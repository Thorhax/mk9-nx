// SA2 - counters of what the ring thread and its helper threads spend time on, read by the "[hitch]"
// lines that sa2_native_targets.cpp writes for frames that run long.

#pragma once

#include <atomic>
#include <cstdint>

namespace sa2::waits {

// What the ring re-checks of the textures (XXH3 of the guest bytes) and how many checks are postponed by
// the per-frame budget (sa2_native_fingerprints_kb_frame).
inline std::atomic<uint64_t> g_bytes_fingerprint{0};
inline std::atomic<uint64_t> g_postponed_fingerprints{0};

// What the ring does in the stutter frame: the texture checks stay under 8 MB, this shows where the ring's
// time goes. Only the ring thread writes all of this.
inline std::atomic<uint64_t> g_ns_textures{0};         // check, detile and prepare the textures in use
inline std::atomic<uint64_t> g_textures_uploads{0};    // textures that changed and are uploaded again
inline std::atomic<uint64_t> g_bytes_uploaded{0};
inline std::atomic<uint64_t> g_created_textures{0};
// Of that texture time, what goes into the two fingerprints (XXH3 of the guest memory and of the already
// prepared data).
inline std::atomic<uint64_t> g_ns_raw_fingerprint{0};
inline std::atomic<uint64_t> g_ns_data_fingerprint{0};
// How long the ring thread waits for the vertex copy thread (WaitUploads in sa2_native_draws.cpp:
// before each submit and before returning the read pointer to the game), and how many times it really waits.
// Only the ring thread writes it.
inline std::atomic<uint64_t> g_ns_waiting_copies{0};
inline std::atomic<uint64_t> g_waits_copies{0};
// What the ring copies itself during those waits, from what the thread had not taken yet, and how many
// copies (sa2_native_uploads_help). g_ns_waiting_copies is therefore only the wait. Ring thread only.
inline std::atomic<uint64_t> g_ns_helping_copies{0};
inline std::atomic<uint64_t> g_helped_copies{0};
// What the ring spends creating textures (image, pool memory and view; "textures X ms" starts after they
// are created and did not include it), how long it waits for the binding thread and how many textures that
// thread bound (sa2_native_textures_binding_thread in sa2_native_draws.cpp). Only the ring thread writes it.
inline std::atomic<uint64_t> g_ns_create_textures{0};
inline std::atomic<uint64_t> g_ns_waiting_bindings{0};
inline std::atomic<uint64_t> g_bound_textures_thread{0};

}  // namespace sa2::waits
