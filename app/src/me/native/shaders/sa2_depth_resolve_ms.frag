#version 450

// sa2-nx (sa2_msaa): a depth resolve from one of the game's single-sample depth targets drawn with host samples. The
// guest has one sample per pixel: sample 0 of each pixel is written as the depth of the single-sample resolved texture
// (destination texel (x, y) = source pixel (x, y) + delta). Its stencil is left as it was: the game samples a resolved
// depth texture as k_24_8 depth only. The viewport and scissor are the destination rectangle (loadOp LOAD keeps the
// rest of the texture).
layout(set = 0, binding = 0) uniform sampler2DMS source_depth;
layout(push_constant) uniform Constants {
  ivec2 delta;
} c;

void main() {
  gl_FragDepth = texelFetch(source_depth, ivec2(gl_FragCoord.xy) + c.delta, 0).r;
}
