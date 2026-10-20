// banjo-nx (2026-10-17): Renut's renderer on plume's deko3d back end (renut_gpu_backend = "deko3d",
// docs/deko3d-renderer.md). The game shaders' GLSL package in the NRO (tools/deko3d_shader_package.py) and the DKSH
// cache of compiled variants on the SD card.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace plume {
struct RenderDevice;
struct RenderShader;
}  // namespace plume

namespace banjo::renut_gpu::deko3d {

// Whether the NRO carries the GLSL package (Switch builds made with one).
bool PackageAvailable();

// The deko3d shader (plume::Deko3DShaderBlob) of a game shader by its container hash, or null (logged once).
std::unique_ptr<plume::RenderShader> CreateGameShader(plume::RenderDevice *device, uint64_t hash);

// The boolean specialization constants (ids 1000..1255) a game shader has, as ShaderBoolSpecMask gives them.
bool BoolSpecMask(uint64_t hash, uint32_t mask[8]);

// peggle2-nx (2026-10-20): how many game shaders of the shader table have no GLSL in any package (the NRO's, or
// deko3d_glsl.bin beside it): a build without translated game shaders before the installer's file is there.
size_t MissingGameShaders();

// The DKSH cache next to the NRO (deko3d_shaders.bin): what it holds is read at start; variants the console compiles
// are added to it (written by Save, and by a periodic save while the game runs). Installed into plume before the
// device is made.
void InstallShaderCache();
void SaveShaderCache();

// One line for the renderer's report: compiles, cache hits, memory.
std::string Report();

}  // namespace banjo::renut_gpu::deko3d
