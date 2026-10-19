#!/usr/bin/env python3
"""Finds the Xbox 360 XDK Direct3D functions in SA2 from a game where they are already named.

Every XDK title links the same Direct3D library, so a function like D3DDevice_DrawIndexedVertices has the same shape
in both games: the same instructions and register use, with different immediates (addresses, and struct offsets
that move between XDK versions). This compares functions by that shape:

  token = primary opcode, extended opcode and register fields of each instruction; immediates and branch
          displacements are left out

and, for each named function of the reference game, ranks SA2's functions: a shortlist by size and opcode
histogram, then difflib's ratio on the token sequences.

  python tools/d3d_match.py --ref-image out/banjo_image.bin --ref-gen <banjo generated/default> \
      --ref-names <renut_gpu_funcs.toml> [--image out/default_image.bin] [--gen app/generated/default]
"""
import argparse
import collections
import difflib
import glob
import math
import os
import re
import struct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0x82000000
MAX_INSNS = 600


def function_starts(gen):
    starts = set()
    for path in glob.glob(os.path.join(gen, '*_recomp.*.cpp')):
        for m in re.finditer(r'^DEFINE_REX_FUNC\(sub_([0-9A-F]+)\)', open(path, errors='replace').read(), re.M):
            starts.add(int(m.group(1), 16))
    return sorted(starts)


def token(x):
    op = x >> 26
    rd, ra, rb = (x >> 21) & 31, (x >> 16) & 31, (x >> 11) & 31
    if op in (16, 18):  # bc / b: keep the link bit and, for bc, BO/BI
        return (op, x & 1, rd if op == 16 else 0, ra if op == 16 else 0)
    if op in (19, 31, 63):
        return (op, (x >> 1) & 0x3FF, rd, ra, rb)
    if op == 59:
        return (op, (x >> 1) & 0x1F, rd, ra, rb)
    if op == 30:
        return (op, (x >> 2) & 7, rd, ra)
    if op == 4:
        return (op, x & 0x7FF, rd)
    return (op, rd, ra)  # D-form: immediate/displacement left out


def functions(image, starts):
    out = {}
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else s + 4 * MAX_INSNS
        n = min((e - s) // 4, MAX_INSNS)
        off = s - BASE
        if off < 0 or off + 4 * n > len(image):
            continue
        out[s] = [token(struct.unpack_from('>I', image, off + 4 * k)[0]) for k in range(n)]
    return out


def histogram(tokens):
    return collections.Counter(t[0] if t[0] not in (19, 31, 59, 63) else (t[0], t[1]) for t in tokens)


def cosine(a, b):
    num = sum(a[k] * b[k] for k in a if k in b)
    den = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ref-image', required=True)
    ap.add_argument('--ref-gen', required=True)
    ap.add_argument('--ref-names', required=True, help='toml with "0xADDR = { name = ... }" lines')
    ap.add_argument('--image', default=os.path.join(ROOT, 'out', 'default_image.bin'))
    ap.add_argument('--gen', default=os.path.join(ROOT, 'app', 'generated', 'default'))
    ap.add_argument('--shortlist', type=int, default=60)
    a = ap.parse_args()

    names = {int(m.group(1), 16): m.group(2) for m in
             re.finditer(r'(0x[0-9A-Fa-f]+)\s*=\s*\{\s*name\s*=\s*"([^"]+)"', open(a.ref_names).read())}
    ref = functions(open(a.ref_image, 'rb').read(), function_starts(a.ref_gen))
    ours = functions(open(a.image, 'rb').read(), function_starts(a.gen))
    ours_hist = {s: histogram(t) for s, t in ours.items()}
    print(f'reference: {len(ref)} functions, {len(names)} named; ours: {len(ours)} functions')
    for addr, name in sorted(names.items(), key=lambda t: t[1]):
        rt = ref.get(addr)
        if not rt:
            print(f'{name:40} reference 0x{addr:08X}: not a function start in the reference tree')
            continue
        rh = histogram(rt)
        cands = [(cosine(rh, ours_hist[s]), s) for s, t in ours.items() if 0.6 <= len(t) / len(rt) <= 1.6]
        cands.sort(reverse=True)
        scored = []
        for _, s in cands[: a.shortlist]:
            sm = difflib.SequenceMatcher(None, rt, ours[s], autojunk=False)
            scored.append((sm.ratio(), s))
        scored.sort(reverse=True)
        best = scored[0] if scored else (0, 0)
        second = scored[1] if len(scored) > 1 else (0, 0)
        flag = '' if best[0] >= 0.85 and best[0] - second[0] >= 0.05 else '  (check)'
        print(f'{name:40} ref 0x{addr:08X} ({len(rt):3} ins) -> sub_{best[1]:08X} {best[0]:.2f}'
              f'   next sub_{second[1]:08X} {second[0]:.2f}{flag}')


if __name__ == '__main__':
    main()
