#version 450

// ReXGlue/Xenia host-target transfer contract, native unhalved depth range.
// Pass 0 writes depth and REPLACE stencil=0. Passes 1..8 discard zero stencil
// bits and REPLACE only the selected bit through dynamic stencil write masks.
// No shader-stencil-export feature is required. Host MSAA remains an explicit
// single-sample approximation; physical sample coordinates select one sample.
layout(set = 0, binding = 0) uniform usampler2D source;
layout(push_constant) uniform Constants {
  uvec2 source_size;
  uvec2 target_size;
  uint pitch_tiles_source;
  uint pitch_tiles_target;
  uint tile_source_start;
  uint tile_target_start;
  uint tiles_count;
  uint source_format; // Existing color class: 0 RGBA8, 2 UNORM10, 3 7e3.
  uint target_format; // Guest depth format: 0 D24S8, 1 D24FS8.
  uint source_64bpp;
  uint target_64bpp;
  uint source_msaa_x, source_msaa_y;
  uint target_msaa_x, target_msaa_y;
  uint stencil_mask; // 0 for the initial depth+zero-stencil pass, otherwise one bit.
  // sa2-nx render resolution: the EDRAM tile in host samples (80x16 at the game's 1280x720; 20q x 4q at
  // q/4 of it, me_resolution_scale.h). Every tile size below is relative to it.
  uint tile_width;
  uint tile_height;
} c;

uint A7e3(float value) {
  uint bits = floatBitsToUint(clamp(value, 0.0, 31.875));
  uint denormal = ((bits & 0x7FFFFFu) | 0x800000u) >> min(125u - (bits >> 23u), 24u);
  uint normal = bits - (124u << 23u);
  uint biased = bits < 0x3E800000u ? denormal : normal;
  return ((biased + 0x7FFFu + ((biased >> 16u) & 1u)) >> 16u) & 0x3FFu;
}

uint Pack(vec4 value) {
  if (c.source_format == 0u) {
    uvec4 n = uvec4(roundEven(clamp(value, 0.0, 1.0) * 255.0));
    return n.r | (n.g << 8u) | (n.b << 16u) | (n.a << 24u);
  }
  if (c.source_format == 2u) {
    uvec4 n = uvec4(roundEven(clamp(value, 0.0, 1.0) * vec4(1023.0, 1023.0, 1023.0, 3.0)));
    return n.r | (n.g << 10u) | (n.b << 20u) | (n.a << 30u);
  }
  return A7e3(value.r) | (A7e3(value.g) << 10u) | (A7e3(value.b) << 20u) |
         (uint(roundEven(clamp(value.a, 0.0, 1.0) * 3.0)) << 30u);
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
  if (c.source_64bpp != 1u || c.target_64bpp != 0u || c.tiles_count == 0u ||
      c.pitch_tiles_source == 0u || c.pitch_tiles_target == 0u || (c.target_format & 255u) > 1u ||
      c.source_format != 7u) discard;
  uvec2 pixel = uvec2(gl_FragCoord.xy);
  if (any(greaterThanEqual(pixel, c.target_size))) discard;
  uvec2 physical = pixel << uvec2(c.target_msaa_x, c.target_msaa_y);
  uint tile = (physical.y / c.tile_height) * c.pitch_tiles_target + physical.x / c.tile_width;
  if (tile < c.tile_target_start || tile - c.tile_target_start >= c.tiles_count) discard;
  uint source_tile = c.tile_source_start + tile - c.tile_target_start;
  uint word_x = physical.x % c.tile_width;
  // Color and depth permute the two 40-word columns of each physical tile.
  word_x = word_x < (c.tile_width >> 1u) ? word_x + (c.tile_width >> 1u) : word_x - (c.tile_width >> 1u);
  uvec2 source_pixel = uvec2((source_tile % c.pitch_tiles_source) * (c.tile_width >> 1u) + (word_x >> 1u),
                            (source_tile / c.pitch_tiles_source) * c.tile_height + physical.y % c.tile_height) >>
                       uvec2(c.source_msaa_x, c.source_msaa_y);
  if (any(greaterThanEqual(source_pixel, c.source_size))) discard;
  uvec4 raw = texelFetch(source, ivec2(source_pixel), 0);
  uint packed = (word_x & 1u) == 0u ? raw.r | (raw.g << 16u) : raw.b | (raw.a << 16u);
  if (c.stencil_mask != 0u && (packed & c.stencil_mask) == 0u) discard;
  uint depth24 = packed >> 8u;
  // Vulkan accepts [0,1]; native doesn't implement SDK's complete [0,2) remap.
  gl_FragDepth = clamp(((c.target_format & 255u) == 0u ? float(depth24) / 16777215.0 : From20e4(depth24)) * ((c.target_format & 256u) != 0u ? 0.5 : 1.0), 0.0, 1.0);
}
