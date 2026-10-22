#!/usr/bin/env python3
"""mk9-nx (2026-10-21): a diagnostic copy of the deko3d GLSL whose vertex shaders report bad positions.

After every `gl_Position = ...;` the vertex shader checks the value: a NaN or infinity, or a component beyond 1e7, is
counted in storage binding 13 (plume_deko3d's diagnostic buffer, read back by the renderer: "[posdiag]" log lines) in
the slot of the shader's hash (low 8 bits), with the hash itself. For the fights' huge flat-coloured triangles on the
Switch and in Ryujinx (never on the PC): it names the vertex shaders whose output explodes. Test builds only.

  python3 tools/deko3d_diag_positions.py --in out/deko3d_shaders/glsl_opt --out out/deko3d_shaders/glsl_diag
"""
import argparse
import glob
import os
import re
import shutil

DECL = """#define CAPTURE 1028u
layout(binding = 13, std430) buffer type_BanjoPosDiag { uint w[]; } g_banjoPosDiag;
void banjoPosDiag(vec4 p)
{
    bool nonFinite = any(isnan(p)) || any(isinf(p));
    bool huge = any(greaterThan(abs(p), vec4(1e7)));
    if (nonFinite || huge)
    {
        uint slot = 4u + (%(lo)du & 255u) * 4u;
        g_banjoPosDiag.w[slot] = %(lo)du;
        g_banjoPosDiag.w[slot + 1u] = %(hi)du;
        atomicAdd(g_banjoPosDiag.w[slot + (nonFinite ? 2u : 3u)], 1u);
        atomicAdd(g_banjoPosDiag.w[nonFinite ? 0u : 1u], 1u);
        // The first bad vertex of the window: the shader's inputs, its position and constants 0-63.
        if (atomicCompSwap(g_banjoPosDiag.w[CAPTURE], 0u, 1u) == 0u)
        {
            g_banjoPosDiag.w[CAPTURE + 1u] = %(lo)du;
            g_banjoPosDiag.w[CAPTURE + 2u] = %(hi)du;
            for (uint c = 0u; c < 4u; c++) g_banjoPosDiag.w[CAPTURE + 4u + c] = floatBitsToUint(p[c]);
%(inputs)s            for (uint r = 0u; r < 64u; r++)
                for (uint c = 0u; c < 4u; c++)
                    g_banjoPosDiag.w[CAPTURE + 72u + r * 4u + c] = floatBitsToUint(g_UboVertex.v[r][c]);
        }
    }
}
"""
WINDOW_CAPTURE = """            g_banjoPosDiag.w[CAPTURE + 3u] = 1u;
            {
                uint vid = uint(gl_VertexID) + g_UboShared.v[65u].w;
                uint low = g_UboShared.v[35u].x;
                g_banjoPosDiag.w[CAPTURE + 72u] = vid;
                g_banjoPosDiag.w[CAPTURE + 73u] = low;
                g_banjoPosDiag.w[CAPTURE + 74u] = uint(gl_VertexID);
                for (uint e = 0u; e < 16u; e++)
                {
                    uvec4 el = g_UboShared.v[66u + e];
                    uint b = CAPTURE + 76u + e * 10u;
                    g_banjoPosDiag.w[b] = el.x; g_banjoPosDiag.w[b + 1u] = el.y;
                    g_banjoPosDiag.w[b + 2u] = el.z; g_banjoPosDiag.w[b + 3u] = el.w;
                    if (el.x < 96u)
                    {
                        uint slot = g_UboShared.v[36u + el.x / 4u][el.x % 4u];
                        g_banjoPosDiag.w[b + 4u] = slot;
                        uint at = (low + (slot & ~3u) + (vid * el.z + el.y) * 4u) / 4u;
                        for (uint k = 0u; k < 4u; k++) g_banjoPosDiag.w[b + 5u + k] = g_DkAddressWindow._m0[at + k];
                    }
                }
            }
"""
# Window shaders, every vertex: the canary UploadVertexFetchData wrote after the draw's fetch copies (mk9_diag_bad_vertices;
# its serial in SharedConstants' padding, v[84].z, its position in v[84].w). A window that does not hold it is counted
# (word 2; word 3 counts the draws checked) and the first such vertex captured from word CANARY: the GPU read other bytes than the CPU wrote there.
CANARY_DECL = """#define CANARY 1270u
void banjoCanary()
{
    uvec4 cn = g_UboShared.v[84u];
    if (cn.z == 0u)
        return;
    uint i = (g_UboShared.v[35u].x + cn.w) / 4u;
    uint a = g_DkAddressWindow._m0[i];
    uint b = g_DkAddressWindow._m0[i + 1u];
    if (gl_VertexID == 0)
        atomicAdd(g_banjoPosDiag.w[3], 1u);  // draws checked (their first vertex)
    if (a == cn.z && b == ~cn.z)
        return;
    atomicAdd(g_banjoPosDiag.w[2], 1u);
    if (atomicCompSwap(g_banjoPosDiag.w[CANARY], 0u, 1u) == 0u)
    {
        g_banjoPosDiag.w[CANARY + 1u] = %(lo)du;
        g_banjoPosDiag.w[CANARY + 2u] = %(hi)du;
        g_banjoPosDiag.w[CANARY + 3u] = cn.z;
        g_banjoPosDiag.w[CANARY + 4u] = cn.w;
        g_banjoPosDiag.w[CANARY + 5u] = a;
        g_banjoPosDiag.w[CANARY + 6u] = b;
        g_banjoPosDiag.w[CANARY + 7u] = g_DkAddressWindow._m0[i + 2u];
        g_banjoPosDiag.w[CANARY + 8u] = g_DkAddressWindow._m0[i + 3u];
        g_banjoPosDiag.w[CANARY + 9u] = g_UboShared.v[35u].x;
        g_banjoPosDiag.w[CANARY + 10u] = uint(gl_VertexID);
        g_banjoPosDiag.w[CANARY + 11u] = g_UboShared.v[66u].x;
        g_banjoPosDiag.w[CANARY + 12u] = g_UboShared.v[36u + g_UboShared.v[66u].x / 4u][g_UboShared.v[66u].x %% 4u];
    }
}
"""
INPUT = re.compile(r'^layout\(location = \d+\) in (\w+) (in_var_\w+);', re.M)
ASSIGN = re.compile(r'^(\s*)gl_Position = [^;]+;\s*$', re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    done = skipped = 0
    for path in sorted(glob.glob(os.path.join(a.src, '*.glsl'))):
        name = os.path.basename(path)
        text = open(path).read()
        m = re.match(r'vs_([0-9A-Fa-f]{16})\.glsl$', name)
        if m and 'void main()' in text and ASSIGN.search(text):
            h = int(m.group(1), 16)
            # Up to 16 inputs, each as 4 words at CAPTURE + 8 + 4 * i; their names in the log come from the GLSL.
            inputs = ''
            for i, (typ, var) in enumerate(INPUT.findall(text)[:16]):
                conv = {'vec4': var, 'vec3': f'vec4({var}, 0.0)', 'vec2': f'vec4({var}, 0.0, 0.0)',
                        'float': f'vec4({var}, 0.0, 0.0, 0.0)', 'uvec4': f'uintBitsToFloat({var})',
                        'ivec4': f'intBitsToFloat({var})'}.get(typ)
                if conv:
                    inputs += (f'            for (uint c = 0u; c < 4u; c++) g_banjoPosDiag.w[CAPTURE + {8 + 4 * i}u + c] = '
                               f'floatBitsToUint(({conv})[c]);\n')
            if 'g_DkAddressWindow' in text and 'g_UboShared' in text:
                # Window shaders (vfetchBound / vfetchMemory, no vertex attributes): the vertex index, the window's
                # offset and per bound element its descriptor, its stream's slot and the 4 words at the element.
                inputs += WINDOW_CAPTURE
            decl = DECL % {'lo': h & 0xFFFFFFFF, 'hi': h >> 32, 'inputs': inputs}
            window = WINDOW_CAPTURE in inputs
            if 'g_UboVertex' not in text:
                decl = decl.replace('g_UboVertex.v[r][c]', '0.0')
            if WINDOW_CAPTURE in decl:
                decl = decl.replace('for (uint r = 0u; r < 64u; r++)', 'for (uint r = 0u; r < 0u; r++)')
            if window:
                decl += CANARY_DECL % {'lo': h & 0xFFFFFFFF, 'hi': h >> 32}
            text = text.replace('void main()', decl + '\nvoid main()', 1)
            if window:
                text = re.sub(r'(void main\(\)\s*\{)', r'\1\n    banjoCanary();', text, count=1)
            text = ASSIGN.sub(lambda mm: mm.group(0).rstrip() + ' banjoPosDiag(gl_Position);', text)

            def ssbo(mm):
                return 'ssbo=0x%x' % (int(mm.group(1), 16) | (1 << 13))
            text, n = re.subn(r'ssbo=0x([0-9a-fA-F]+)', ssbo, text, count=1)
            if not n:
                text = re.sub(r'(// banjo-dk [^\n]*)', r'\1 ssbo=0x2000', text, count=1)
            done += 1
        else:
            skipped += 1
        open(os.path.join(a.out, name), 'w').write(text)
    print(f'deko3d_diag_positions: {done} vertex shaders checked, {skipped} files copied as they were')


if __name__ == '__main__':
    main()
