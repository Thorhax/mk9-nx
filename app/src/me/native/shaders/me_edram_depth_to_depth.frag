#version 450

// Depth->depth EDRAM transfer. Same guest encoding preserves host depth
// precision (SDK host-target contract), including across sample layouts.
// Different guest formats reinterpret the packed 24-bit depth + stencil word.
// Both source and destination are depth: do NOT swap the 40-word tile columns.
layout(set = 0, binding = 0) uniform sampler2D source_depth;
layout(set = 0, binding = 1) uniform usampler2D source_stencil;
layout(constant_id = 0) const bool round_float24 = false;
layout(push_constant) uniform Constants {
  uvec2 source_size;
  uvec2 target_size;
  uint pitch_tiles_source;
  uint pitch_tiles_target;
  uint tile_source_start;
  uint tile_target_start;
  uint tiles_count;
  uint source_format; // Depth: 0 D24S8, 1 D24FS8.
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

uint A20e4(float z) {
  if (!(z > 0.0)) return 0u;
  uint bits = floatBitsToUint(z);
  if (bits >= 0x3FFFFFF8u) return 0xFFFFFFu;
  if (bits < 0x38800000u) {
    bits = ((bits & 0x7FFFFFu) | 0x800000u) >> min(113u - (bits >> 23u), 24u);
  } else bits += 0xC8000000u;
  if (round_float24) bits += 3u + ((bits >> 3u) & 1u);
  return (bits >> 3u) & 0xFFFFFFu;
}

float From20e4(uint bits) {
  bits &= 0xFFFFFFu;
  if (bits == 0u) return 0.0;
  uint mantissa = bits & 0xFFFFFu;
  uint exponent = bits >> 20u;
  if (exponent == 0u) {
    uint shift = 20u - uint(findMSB(mantissa));
    exponent = 1u - shift;
    mantissa = (mantissa << shift) & 0xFFFFFu;
  }
  return uintBitsToFloat(((exponent + 112u) << 23u) | (mantissa << 3u));
}

void main() {
  if (c.source_64bpp != 0u || c.target_64bpp != 0u || c.tiles_count == 0u ||
      c.pitch_tiles_source == 0u || c.pitch_tiles_target == 0u ||
      (c.source_format & 255u) > 1u || (c.target_format & 255u) > 1u) discard;
  uvec2 pixel = uvec2(gl_FragCoord.xy);
  if (any(greaterThanEqual(pixel, c.target_size))) discard;
  uvec2 physical = pixel << uvec2(c.target_msaa_x, c.target_msaa_y);
  uint tile = (physical.y / c.tile_height) * c.pitch_tiles_target + physical.x / c.tile_width;
  if (tile < c.tile_target_start || tile - c.tile_target_start >= c.tiles_count) discard;
  uint source_tile = c.tile_source_start + tile - c.tile_target_start;
  uvec2 source_pixel = uvec2((source_tile % c.pitch_tiles_source) * c.tile_width + physical.x % c.tile_width,
                            (source_tile / c.pitch_tiles_source) * c.tile_height + physical.y % c.tile_height) >>
                       uvec2(c.source_msaa_x, c.source_msaa_y);
  if (any(greaterThanEqual(source_pixel, c.source_size))) discard;
  float depth_value = texelFetch(source_depth, ivec2(source_pixel), 0).r;
  uint stencil = texelFetch(source_stencil, ivec2(source_pixel), 0).r & 255u;
  if (c.stencil_mask != 0u && (stencil & c.stencil_mask) == 0u) discard;
  if ((c.source_format & 255u) == (c.target_format & 255u)) {
    // Avoid quantizing D32 to guest float24/unorm24 merely because the physical
    // sample layout changed. Matching half flags require no arithmetic at all.
    if ((c.source_format & 256u) != (c.target_format & 256u))
      depth_value *= (c.source_format & 256u) != 0u ? 2.0 : 0.5;
    gl_FragDepth = depth_value;
    return;
  }
  if ((c.source_format & 256u) != 0u) depth_value *= 2.0;
  uint depth24 = (c.source_format & 255u) == 0u
      ? uint(roundEven(clamp(depth_value, 0.0, 1.0) * 16777215.0)) : A20e4(depth_value);
  // Format bit8 halves FLOAT24 guest [0,2) into the valid host [0,1) range.
  gl_FragDepth = clamp(((c.target_format & 255u) == 0u ? float(depth24) / 16777215.0 : From20e4(depth24)) * ((c.target_format & 256u) != 0u ? 0.5 : 1.0), 0.0, 1.0);
}
