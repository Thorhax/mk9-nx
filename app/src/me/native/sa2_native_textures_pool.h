// SA2 - native renderer: sub-allocation pool for textures.
//
// Why it exists (measured, not estimated)
//
//   Without the pool, every texture gets its own dedicated allocation: CreateTexture ->
//   rex::ui::vulkan::util::CreateDedicatedAllocationImage -> vkAllocateMemory with
//   VkMemoryDedicatedAllocateInfo. On Horizon that is not a cheap call.
//
//   What every texture pays, following the chain in the Mesa tree:
//     1. nvk_AllocateMemory (nvk_device_memory.c:133); in the __SWITCH__ branch
//        (:178-193) it sets alignment, tile_mode and pte_kind.
//     2. nvkmd_switch_dev_alloc_mem_impl (nvkmd/switch/nvkmd_switch_dev.c:750-838):
//        memory_create -> memory_map -> alloc_va (VA #1) -> va_bind_mem (mapping #1).
//     3. nouveau_horizon_memory_create (horizon/nouveau_horizon_memory.c:468-663):
//        memalign + nvMapCreate (:572) and, always, memset(size) + armDCacheClean(size)
//        (:612-620), even though Vulkan does not ask for ZERO_INITIALIZE.
//     4. nvk_image_plane_bind (nvk_image.c:1697-1707): since our textures are
//        SAMPLED|TRANSFER_DST, nvk_image_can_compress returns false (nvk_image.c:843-846),
//        so the dedicated shortcut is not taken: a VA #2 is reserved with the block-linear
//        pte_kind and a mapping #2 is made.
//
//   In short: the dedicated allocation costs twice the VAs and mappings and gives nothing back.
//   Its only benefit is compression, and our textures are never compressible.
//
// The measurement (console, entering a new zone)
//
//   From the profile log (translated): "new NvMaps 17.6 with cache
//   (2.8 MB) and 0.3 without cache (0.0 MB), 7.8 ms | GPU addresses 35.4 (4.0 ms), mappings
//   35.4 (0.0 MB, 19.2 ms)"
//   -> 31.0 ms/s of CPU just allocating and mapping. And 35.4/17.9 = 1.98: two VAs and two
//      mappings per NvMap, exactly what the code above predicts.
//
//   Medians per call, 80 intervals with >=5 allocations/s across several logs
//   (the counters come from the --wrap wrappers in the Switch perf shim):
//     nvMapCreate                422 us
//     nvAddressSpaceAllocFixed   127 us
//     MapBufferEx                626 us
//     One texture (1 + 2 + 2)   ~1.9 ms
//
//   Real peaks: 159 ms/s (16 % of a core) and 298.7 ms/s (30 % of a core).
//
//   The driver's BO cache does not save this: in a long measurement it hits 88 % and
//   everything is still paid, because recycling saves the nvMapCreate but not the VA or the
//   mapping, which are the expensive two thirds.
//
// What this pool saves
//
//   Per texture only the plane's VA and its partial mapping remain: 1.9 ms -> ~0.75 ms (-60 %).
//     Bad stretch:                  31.0 ms/s -> 18.8 ms/s   (-12.2 ms/s)
//     Average frame (~27 FPS):       1.15 ms  ->  0.70 ms    (-0.45 ms)
//     Burst of 15 textures:         28.5 ms   -> 11.3 ms     (-17.2 ms)
//   It is a variance fix, not an average one: allocation explains 9 % of the average gap
//   but 17 of the 44 ms of the worst frame, which is what is noticeable when turning.
//
//   As a bonus: live NvMap handles drop from ~3,600 to ~12, and with them the ceiling of
//   the area shared with nvdrv goes away (error 0x235C, which showed up as a black screen).
//
// Why not a single pixel changes
//
//   Binding at offset != 0 into non-dedicated memory is supported and keeps the block-linear
//   layout: NVK reserves a VA of its own for the plane with the right pte_kind and maps the
//   sub-range (nvk_image.c -> nouveau_horizon_vm.c).
//   The size vkGetImageMemoryRequirements returns is already rounded to 64 KiB precisely for
//   this (nvk_image.c:1422-1428), and requiresDedicatedAllocation is never true on Switch
//   (nvk_image.c:1444-1460). The slabs are allocated without VkMemoryDedicatedAllocateInfo,
//   so their layout.valid is false and they accept any pte_kind (nvkmd_switch_dev.c:787-795,
//   nouveau_horizon_memory.c:1056-1073).
//   No compression is lost because there was none (nvk_image.c:843-846).
//
// Safety invariant (mandatory, read before touching eviction)
//
//   Without the pool, if the GPU read a texture that was just destroyed it would read memory
//   that was freed but still mapped: harmless garbage. With the pool it would read the pixels
//   of another texture, because the physical block is reused immediately.
//
//   That is only safe while both of these conditions hold. The current code meets them, and
//   they are mandatory:
//     a) Normal eviction only releases textures unused for 120 frames
//        (kFramesNoUsageForDrop in sa2_native_draws.cpp). No pending work
//        can reference them.
//     b) Emergency eviction (DropTexturesPerMissingOfMemory) calls
//        WaitGpuOfTheAll() first.
//   If that threshold of 120 is ever lowered or that wait removed, this pool is no longer
//   safe and releasing a block must be delayed by a few frames.
//
// Single thread
//
//   All of this lives on the ring thread (PrepareTexture and the render targets),
//   so there is no lock. If a texture is ever created from another thread, a lock must be
//   added here first.

#pragma once

#include <rex/ui/vulkan/device.h>

#include <cstdint>
#include <string>
#include <vector>

namespace sa2::native {

// Opaque identifier of a pool block. This value means "not from the pool" and is the
// default value ImageNative::pool_block must hold.
inline constexpr uint32_t kBlockPoolInvalid = UINT32_MAX;

// Pool granularity. Horizon binds with 64 KiB alignment
// (NOUVEAU_HORIZON_BIND_ALIGN_B, horizon/nouveau_horizon_private.h:21) and NVK already
// rounds image sizes to that value, so no memory is lost: it is the same rounding the
// dedicated allocation already pays.
inline constexpr uint64_t kUnitPoolBytes = 64ull * 1024ull;

// For the NoteMemoryOfTheGpu report.
struct PoolStateTextures {
  bool active = false;
  uint32_t memory_type = UINT32_MAX;
  uint32_t slabs = 0;
  uint64_t bytes_reserved = 0;  // sum of the slab sizes
  uint64_t bytes_in_usage = 0;      // space used by live textures
  uint64_t bytes_free = 0;
  uint64_t bytes_greater_gap = 0;  // largest contiguous free range: measures fragmentation
  uint64_t live_textures = 0;
  uint64_t placed_textures = 0;   // running total that went into the pool
  uint64_t dedicated_textures = 0;   // running total that fell back to the dedicated path
  uint64_t failed_gaps = 0;      // times it did not fit in any slab
  uint64_t slabs_in_hot = 0;    // slabs created after prewarming
  uint64_t slabs_failed = 0;       // times vkAllocateMemory refused a slab
};

/*
 * GPU memory sub-allocator for the native renderer's textures.
 *
 * Intended use, all from the ring thread:
 *   Start(vulkan_device, mb_cache_max)      once, at renderer initialization, before drawing
 *   PerFrame(frame)                         once per frame, to grow with headroom
 *   Reserve(requirements, mem, offset, blk) in CreateTexture, before vkBindImageMemory
 *   NoteDedicated()                         when a texture falls back to the dedicated path
 *   Release(block)                          in DestroyImage
 *   Finish()                                in the destructor, after the images are destroyed
 */
class PoolTextures {
 public:
  PoolTextures() = default;
  ~PoolTextures();

  PoolTextures(const PoolTextures&) = delete;
  PoolTextures& operator=(const PoolTextures&) = delete;

  /*
   * Prepares the pool and prewarms the slabs for the steady state.
   *
   * `mb_cache_max` is sa2_native_textures_mb_max: it gives the initial MB (half of it,
   * between 64 and 256) and the cap (the cache limit plus one slab of headroom).
   *
   * Returns false if the cvar is off, if there is no valid memory type or if not even one
   * slab could be created. In that case Reserve() always fails and everything keeps using
   * the usual dedicated path, exactly the behavior without the pool.
   */
  bool Start(const rex::ui::vulkan::VulkanDevice* vulkan_device, int32_t mb_cache_max);

  // Frees every slab. The images must already be destroyed.
  void Finish();

  bool Active() const { return active_; }

  /*
   * Finds room for an already created image.
   *
   * `requirements` is what vkGetImageMemoryRequirements returns for that VkImage.
   * requirements.alignment is honored exactly, it is not assumed to be 64 KiB: a 3D texture
   * can ask for up to 512 KB because of the tile mode's z_log2 (nil/image.rs:430).
   *
   * If it returns true, the caller must call vkBindImageMemory(image, memory_out,
   * offset_out) and store block_out in ImageNative::pool_block.
   * If it returns false, nothing has been touched: the caller uses the dedicated path.
   */
  bool Reserve(const VkMemoryRequirements& requirements, VkDeviceMemory& memory_out,
                VkDeviceSize& offset_out, uint32_t& block_out);

  // Returns the block to the pool. Accepts kBlockPoolInvalid and does nothing with it.
  void Release(uint32_t block);

  /*
   * Growth with headroom. Called once per frame.
   *
   * Creates a new slab when less than kSlackBytes is free, never at the moment it is
   * needed: nouveau_horizon_memory_create does a memset + dcache clean of the whole slab
   * and 32 MB of that in the middle of play would be a 10-15 ms stutter, the
   * very problem this pool solves. It also creates at most one every kFramesBetweenSlabs.
   */
  void PerFrame(uint64_t frame);

  // A texture did not fit (or the pool is off) and took the dedicated path.
  void NoteDedicated() { ++dedicated_textures_; }

  PoolStateTextures State() const;

  // Line for the texture-memory report written by NoteMemoryOfTheGpu. No trailing newline.
  std::string Summary() const;

 private:
  struct Slab {
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t units = 0;
    uint32_t units_in_usage = 0;
    std::vector<uint64_t> busy;  // bitmap, one unit per bit
    std::vector<uint32_t> long_value;     // units reserved from each start (0 = not a start)
  };

  // 8 MB of headroom: below that a new slab is requested.
  static constexpr uint64_t kSlackBytes = 8ull << 20;
  // Never two slabs in a row: gives eviction time to free memory before growing further.
  static constexpr uint64_t kFramesBetweenSlabs = 120;
  // The slab index goes in the top 8 bits of the block identifier.
  static constexpr uint32_t kMaxSlabs = 255;

  bool CreateSlab(bool in_hot);
  bool ChooseMemoryType(uint32_t& out_type, uint64_t& alignment_view_out) const;

  const rex::ui::vulkan::VulkanDevice* vulkan_device_ = nullptr;
  VkDevice device_ = VK_NULL_HANDLE;
  bool active_ = false;
  uint32_t memory_type_ = UINT32_MAX;
  uint64_t slab_bytes_ = 0;
  uint32_t slab_units_ = 0;
  uint32_t slabs_cap_ = 0;
  uint64_t last_growth_ = 0;
  std::vector<Slab> slabs_;

  uint64_t live_textures_ = 0;
  uint64_t placed_textures_ = 0;
  uint64_t dedicated_textures_ = 0;
  uint64_t failed_gaps_ = 0;
  uint64_t slabs_in_hot_ = 0;
  uint64_t slabs_failed_ = 0;
};

}  // namespace sa2::native
