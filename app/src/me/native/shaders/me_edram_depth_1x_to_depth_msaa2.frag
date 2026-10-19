#version 450

// Bounded production utility: actual Vulkan 1x -> 2x PHYSICAL alias, NOT guest resolve.
// Same guest depth encoding / half-range flags preserve the raw host depth.
// Both views are depth: NO 40-word column swap, averaging or quantization.
// Caller must provide a 2x attachment / pipeline with full sample shading,
// LOAD both aspects, and the same nine-pass stencil replacement as the 1x path.
layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(set = 0, binding = 1) uniform usampler2D source_stencil;
// Exact 72-byte ABI of me_edram_depth_to_depth.frag.
layout(push_constant) uniform Constants {
  uvec2 source_size;
  uvec2 target_size;
  uint pitch_tiles_source;
  uint pitch_tiles_target;
  uint tile_source_start;
  uint tile_target_start;
  uint tiles_count;
  uint source_format;
  uint target_format;
  uint source_64bpp;
  uint target_64bpp;
  uint source_msaa_x, source_msaa_y;
  uint target_msaa_x, target_msaa_y;
  uint stencil_mask;
  // sa2-nx render resolution: the EDRAM tile in host samples (80x16 at the game's 1280x720; 20q x 4q at
  // q/4 of it, me_resolution_scale.h). Every tile size below is relative to it.
  uint tile_width;
  uint tile_height;
} c;

void main() {
  // Only D24S8(0), D24FS8(1), or half-range D24FS8(257). Metadata cannot
  // silently request mixed encoding, grid X, collapsed Y or raw 64bpp.
  if ((c.source_format != 0u && c.source_format != 1u && c.source_format != 257u) ||
      c.source_format != c.target_format || c.source_64bpp != 0u || c.target_64bpp != 0u ||
      c.source_msaa_x != 0u || c.target_msaa_x != 0u ||
      c.source_msaa_y != 0u || c.target_msaa_y != 1u ||
      c.pitch_tiles_source == 0u || c.pitch_tiles_target == 0u ||
      c.pitch_tiles_source > 2048u || c.pitch_tiles_target > 2048u ||
      c.tiles_count == 0u || c.tiles_count > 2048u ||
      c.tile_source_start >= 2048u || c.tile_target_start >= 2048u ||
      c.tiles_count > 2048u - c.tile_source_start ||
      c.tiles_count > 2048u - c.tile_target_start ||
      (c.stencil_mask != 0u &&
       (c.stencil_mask > 128u || (c.stencil_mask & (c.stencil_mask - 1u)) != 0u))) discard;
  if (any(notEqual(uvec2(textureSize(source_depth, 0)), c.source_size)) ||
      any(notEqual(uvec2(textureSize(source_stencil, 0)), c.source_size))) discard;
  uvec2 pixel = uvec2(gl_FragCoord.xy);
  if (any(greaterThanEqual(pixel, c.target_size))) discard;
  // Native Vulkan2x top guest0=host1, bottom guest1=host0. Each destination
  // sample reads its own physical source row; no averaging or replication.
  uint sample_host = uint(gl_SampleID);
  if (sample_host > 1u) discard;
  uvec2 physical = uvec2(pixel.x, pixel.y * 2u + (sample_host ^ 1u));
  uint tile = (physical.y / c.tile_height) * c.pitch_tiles_target + physical.x / c.tile_width;
  if (tile < c.tile_target_start || tile - c.tile_target_start >= c.tiles_count) discard;
  uint source_tile = c.tile_source_start + tile - c.tile_target_start;
  uvec2 source_physical = uvec2((source_tile % c.pitch_tiles_source) * c.tile_width + physical.x % c.tile_width,
                               (source_tile / c.pitch_tiles_source) * c.tile_height + physical.y % c.tile_height);
  ivec2 source_pixel = ivec2(source_physical);
  if (any(greaterThanEqual(uvec2(source_pixel), c.source_size))) discard;
  uint stencil = texelFetch(source_stencil, source_pixel, 0).r & 255u;
  if (c.stencil_mask != 0u && (stencil & c.stencil_mask) == 0u) discard;
  gl_FragDepth = texelFetch(source_depth, source_pixel, 0).r;
}


