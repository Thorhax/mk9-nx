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
            decl = DECL % {'lo': h & 0xFFFFFFFF, 'hi': h >> 32, 'inputs': inputs}
            if 'g_UboVertex' not in text:
                decl = decl.replace('g_UboVertex.v[r][c]', '0.0')
            text = text.replace('void main()', decl + '\nvoid main()', 1)
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
