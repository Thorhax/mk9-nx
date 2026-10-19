#!/usr/bin/env python3
"""The game's Direct3D API surface, from the generated code (phase 1 of the API-level renderer,
docs/api-renderer-assessment.md).

The Xbox 360 XDK links Direct3D statically as one block of functions. Calls from game code (outside the block) into
it are the entry points the game uses; inline XDK functions do not appear as calls (they touch the device directly).

  python tools/d3d_surface.py [--lo 0x82540000 --hi 0x82570000] [--edges]

  --edges   prints, around both ends of the range, each function with how many callers it has inside and outside the
            range, to place the library's boundaries.
Output: every entry point (a function inside the range with callers outside it), its outside call sites and calling
functions, its size, the imports and MMIO it reaches, and the names known from docs/sa2-notes.md.
"""
import argparse
import collections
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, 'app', 'generated', 'default')
DEF = re.compile(r'^DEFINE_REX_FUNC\((sub_[0-9A-F]{8})\) \{$')
CALL = re.compile(r'(?<![\w])(?:__imp__)?(sub_[0-9A-F]{8}|__imp__[A-Za-z]\w+)\(ctx, base\);')
INDIRECT = re.compile(r'REX_CALL_INDIRECT_FUNC')
MMIO = re.compile(r'REX_MM_(?:STORE|LOAD)')

KNOWN = {  # docs/sa2-notes.md, "Named D3D functions" (tools/d3d_match.py against Banjo's named XDK functions)
    0x82558E98: 'D3DDevice_Swap', 0x82565C60: 'D3DDevice_DrawVertices', 0x82566078: 'D3DDevice_DrawIndexedVertices',
    0x82565C18: 'D3DDevice_DrawVerticesUP', 0x82556A60: 'draw-time shader load (FlushState part)',
    0x8255E750: 'D3DDevice_CreateVertexShader', 0x8255E568: 'D3DDevice_CreatePixelShader',
    0x8255E198: 'D3DDevice_CreateVertexDeclaration', 0x82563AB0: 'D3DDevice_Clear', 0x825645A0: 'D3DDevice_Resolve?',
    0x825576D0: 'D3DDevice_BeginTiling?', 0x82557A40: 'D3DDevice_EndTiling?', 0x82548DF8: 'D3DDevice_CreateTexture',
    0x82548F18: 'D3DDevice_CreateSurface', 0x82549730: 'D3DDevice_CreateVertexBuffer',
    0x82549808: 'D3DDevice_CreateIndexBuffer', 0x825527D8: 'D3DDevice_SetGammaRamp?', 0x82552708: 'D3DDevice_SetGammaRamp?',
    0x82549650: 'D3DResource_AddRef', 0x8254A4A8: 'D3DResource_Release', 0x82549358: 'D3DResource_GetType',
    0x82557FC0: 'present (flip)', 0x8254F400: 'GPU wait loop', 0x8255D990: 'shader constants setter (VS 376+?)',
    0x8255D8B8: 'shader constants setter (120+?)',
}


def parse():
    funcs = {}
    for path in sorted(glob.glob(os.path.join(GEN, '*_recomp.*.cpp'))):
        cur = None
        for line in open(path, encoding='utf-8', errors='replace'):
            m = DEF.match(line)
            if m:
                cur = int(m.group(1)[4:], 16)
                funcs[cur] = {'calls': [], 'indirect': 0, 'mmio': 0, 'lines': 0}
                continue
            if cur is None:
                continue
            f = funcs[cur]
            f['lines'] += 1
            for c in CALL.findall(line):
                f['calls'].append(c)
            if INDIRECT.search(line):
                f['indirect'] += 1
            if MMIO.search(line):
                f['mmio'] += 1
    return funcs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lo', default='0x82540000')
    ap.add_argument('--hi', default='0x82570000')
    ap.add_argument('--edges', action='store_true')
    a = ap.parse_args()
    lo, hi = int(a.lo, 16), int(a.hi, 16)
    funcs = parse()
    addrs = sorted(funcs)
    callers = collections.defaultdict(list)  # callee -> [caller addr]
    for f, d in funcs.items():
        for c in d['calls']:
            if c.startswith('sub_'):
                callers[int(c[4:], 16)].append(f)
    inside = lambda x: lo <= x < hi
    if a.edges:
        for edge in (lo, hi):
            i = next(k for k, x in enumerate(addrs) if x >= edge)
            for x in addrs[max(0, i - 25):i + 25]:
                cs = callers.get(x, [])
                n_in = sum(1 for c in cs if inside(c))
                imports = sorted({c for c in funcs[x]['calls'] if c.startswith('__imp__')})
                print(f'{x:08X} {"IN " if inside(x) else "out"} callers in {n_in:3d} out {len(cs) - n_in:3d} '
                      f'lines {funcs[x]["lines"]:5d} mmio {funcs[x]["mmio"]:2d} {",".join(imports)[:70]}')
            print('---')
        return
    entries = []
    for x in addrs:
        if not inside(x):
            continue
        outside = [c for c in callers.get(x, []) if not inside(c)]
        if outside:
            entries.append((x, outside))
    print(f'{len([x for x in addrs if inside(x)])} functions in [{lo:08X}, {hi:08X}); {len(entries)} called from '
          f'outside (the API surface)')
    for x, outside in sorted(entries, key=lambda e: -len(e[1])):
        d = funcs[x]
        sites = len(outside)
        distinct = len(set(outside))
        sub = sorted({c for c in d['calls'] if c.startswith('sub_') and inside(int(c[4:], 16))})
        imports = sorted({c[7:] for c in d['calls'] if c.startswith('__imp__')})
        print(f'{x:08X} {KNOWN.get(x, ""):42s} sites {sites:4d} from {distinct:3d} fns  lines {d["lines"]:5d} '
              f'calls-in-D3D {len(sub):2d} mmio {d["mmio"]:2d} ind {d["indirect"]:2d} {",".join(imports)[:60]}')


if __name__ == '__main__':
    main()
