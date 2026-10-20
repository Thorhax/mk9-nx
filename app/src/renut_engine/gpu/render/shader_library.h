/**
 * @file    gpu/render/shader_library.h
 * @brief   The XenosRecomp shader cache at runtime, and capture of shaders it lacks.
 */
#pragma once

#include <cstdint>
#include <memory>

#include <plume_render_interface.h>

#include "shader_cache.h"

namespace renut::gpu {

// XXH3_64 of a whole guest shader container (virtual + physical), the key the
// cache and tools/dump_shaders.py use.
uint64_t HashShaderContainer(const uint8_t *container);

const ShaderCacheEntry *FindShaderEntry(uint64_t hash);

// Builds the active backend's shader from a cache entry (decompressing the
// cache on first use). nullptr on failure.
std::unique_ptr<plume::RenderShader> CreateCachedShader(plume::RenderDevice *device,
                                                        const ShaderCacheEntry &entry);

// For a shader FindShaderEntry did not find (the game compiles a few at
// runtime, which no dump of its files has): writes its container to the
// capture directory (renut_shader_capture_dir), named as tools/dump_shaders.py
// names them, and logs it. The next build copies the captures into the shader
// dump directory and recompiles them into the cache. Once per hash.
void CaptureMissingShader(const uint8_t *container, bool pixel_shader, uint64_t hash);

// For the memory report: the decompressed shader cache kept in memory.
uint64_t ShaderCacheHostBytes();

// banjo-nx (banjo_gpu_bool_spec): the bools whose specialization constants (SpecId 1000 + N) the shader's SPIR-V
// declares, bit N of mask (8 words). All zero on D3D12, for a shader without, or one not in the cache.
void ShaderBoolSpecMask(const ShaderCacheEntry &entry, uint32_t mask[8]);

} // namespace renut::gpu
