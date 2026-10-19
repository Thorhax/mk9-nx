#version 450

// Mode-4 EDRAM transfer between two 64-bit (FP16 x4) color views as a fragment pass: raw UINT texels are copied
// (all half-float bit patterns survive, NaN payloads included), only the tile addressing changes. Same mapping
// as me_edram_raw64_to_raw64.comp driven by the destination texel; fragments outside the run's tile range
// discard (loadOp = LOAD keeps them). Source: R16G16B16A16_UINT view; destination: R16G16B16A16_UINT view
// of the destination image as a color attachment.
layout(set = 0, binding = 0) uniform usampler2D source;
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
  uint source_64bpp;     // always 1 here
  uint target_64bpp;    // always 1 here
  uint source_msaa_x, source_msaa_y;
  uint target_msaa_x, target_msaa_y;
  // sa2-nx render resolution: the EDRAM tile in host samples (80x16 at the game's 1280x720; 20q x 4q at
  // q/4 of it, me_resolution_scale.h). Every tile size below is relative to it.
  uint tile_width;
  uint tile_height;
} c;
layout(location = 0) out uvec4 output_value;

void main() {
  uvec2 pixel = uvec2(gl_FragCoord.xy);
  if (any(greaterThanEqual(pixel, c.target_size))) discard;
  // 64-bit tiles are 40 pixels wide (80 words); the compute writer of a texel is the physical sample with zero
  // sub-sample bits.
  uvec2 physical = pixel << uvec2(c.target_msaa_x, c.target_msaa_y);
  uint tile_target = (physical.y / c.tile_height) * c.pitch_tiles_target + physical.x / (c.tile_width >> 1u);
  if (tile_target < c.tile_target_start || tile_target - c.tile_target_start >= c.tiles_count) discard;
  uint tile_source = c.tile_source_start + (tile_target - c.tile_target_start);
  uvec2 p = uvec2((tile_source % c.pitch_tiles_source) * (c.tile_width >> 1u) + physical.x % (c.tile_width >> 1u),
                  (tile_source / c.pitch_tiles_source) * c.tile_height + physical.y % c.tile_height) >>
            uvec2(c.source_msaa_x, c.source_msaa_y);
  output_value = any(greaterThanEqual(p, c.source_size)) ? uvec4(0u) : texelFetch(source, ivec2(p), 0);
}
