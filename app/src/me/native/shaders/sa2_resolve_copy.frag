#version 450

// sa2-nx (sa2_gpu_resolve_draw): a same-format RGBA8 color resolve as one fragment pass on the 3D engine, instead of
// vkCmdCopyImage, which NVK records on the copy engine (NV90B5): every resolve switched engines, with a wait for idle
// each way. Destination texel (x, y) = source texel (x, y) + delta. UNORM8 values round-trip exactly (n / 255 written
// back is n), as in me_edram_color_to_color.frag. The viewport and scissor are the destination rectangle (loadOp LOAD
// keeps the rest of the texture).
layout(set = 0, binding = 0) uniform sampler2D source;
layout(push_constant) uniform Constants {
  ivec2 delta;
} c;
layout(location = 0) out vec4 output_value;

void main() {
  output_value = texelFetch(source, ivec2(gl_FragCoord.xy) + c.delta, 0);
}
