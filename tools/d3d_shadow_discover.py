#!/usr/bin/env python3
"""Where does the game's Direct3D keep each GPU register? (phase 1 of the API-level renderer,
docs/api-renderer-assessment.md).

Input: logs/api_discover.bin from a PC run with sa2_api_discover_draws = N (src/me/me_native_system.cpp, namespace
discover): for N draws, the Direct3D device struct as the game called the draw (before Direct3D flushed it into PM4)
and the register file the ring rebuilt from the PM4 for the same draw.

For every GPU register whose value varies across the samples, the device dwords holding exactly that value in every
sample are its shadow. Registers that never vary cannot be placed this way (listed apart). Varying registers with no
exact shadow are computed by Direct3D when it flushes (from several fields), or come from elsewhere (command buffer
contents, resource headers): they need its flush code.

  python tools/d3d_shadow_discover.py [out/pc/logs/api_discover.bin] [--json out/d3d_shadow_map.json]
"""
import argparse
import collections
import hashlib
import json
import os
import struct
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANGES = [  # Xenos register file (xenia's register_table.inc / the renderer's names)
    (0x0000, 0x2000, 'control and misc'),
    (0x2000, 0x4000, 'context: render state (RB_, PA_, VGT_, SQ_)'),
    (0x4000, 0x4800, 'ALU constants (512 float4)'),
    (0x4800, 0x4900, 'fetch constants (32 textures x 6 dwords / vertex fetch)'),
    (0x4900, 0x4908, 'bool constants'),
    (0x4908, 0x4928, 'loop constants'),
    (0x4928, 0x6000, 'other'),
]


def name_of(reg):
    for lo, hi, n in RANGES:
        if lo <= reg < hi:
            return n
    return '?'


def load(path):
    data = open(path, 'rb').read()
    pos = 0
    regs, devs, heads = [], [], []
    while pos + 56 <= len(data):
        h = struct.unpack_from('<14I', data, pos)
        if h[0] != 0x53413244:
            sys.exit(f'bad sample magic at {pos}')
        count, dev_bytes = h[12], h[13]
        if pos + 56 + 4 * count + dev_bytes > len(data):
            break  # the last sample of a run that ended before the file was closed
        pos += 56
        regs.append(np.frombuffer(data, '<u4', count, pos)); pos += 4 * count
        devs.append(np.frombuffer(data, '>u4', dev_bytes // 4, pos).astype(np.uint32)); pos += dev_bytes
        heads.append(h)
    return np.array(regs), np.array(devs), heads


def column_key(col):
    return hashlib.blake2b(col.tobytes(), digest_size=16).digest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?', default=os.path.join(ROOT, 'out', 'pc', 'logs', 'api_discover.bin'))
    ap.add_argument('--json', default=os.path.join(ROOT, 'out', 'd3d_shadow_map.json'))
    a = ap.parse_args()
    R, D, heads = load(a.path)
    S = len(R)
    devices = collections.Counter(h[11] for h in heads)
    print(f'{S} samples; registers {R.shape[1]:#x}; device bytes {D.shape[1] * 4:#x}; devices '
          + ', '.join(f'{d:08X} x{n}' for d, n in devices.most_common()))
    # Device columns indexed by their value vector.
    by_key = collections.defaultdict(list)
    dev_varying = 0
    for j in range(D.shape[1]):
        col = D[:, j]
        if col.min() != col.max():
            dev_varying += 1
        by_key[column_key(col)].append(j)
    mapping, unmatched, constant = {}, [], collections.defaultdict(int)
    for i in range(R.shape[1]):
        col = R[:, i]
        if col.min() == col.max():
            if col[0]:
                constant[name_of(i)] += 1
            continue
        offs = by_key.get(column_key(col), [])
        if offs:
            mapping[i] = [o * 4 for o in offs]
        else:
            unmatched.append(i)
    print(f'device dwords that vary: {dev_varying}')
    per_range = collections.defaultdict(lambda: [0, 0])
    for i in mapping:
        per_range[name_of(i)][0] += 1
    for i in unmatched:
        per_range[name_of(i)][1] += 1
    print('varying registers by range: mapped / unmatched')
    for lo, hi, n in RANGES:
        m, u = per_range.get(n, [0, 0])
        if m or u or constant.get(n):
            print(f'  {n:55s} {m:5d} / {u:5d}   (non-zero constant: {constant.get(n, 0)})')
    # Contiguous runs: register i -> device offset o with o - 4*i constant over a run (arrays such as constants).
    runs = []
    for i in sorted(mapping):
        o = min(mapping[i])
        if runs and i == runs[-1][1] + 1 and o - 4 * i == runs[-1][2]:
            runs[-1][1] = i
        else:
            runs.append([i, i, o - 4 * i])
    print('mapped runs (register range -> device offset range):')
    for lo, hi, d in runs:
        if hi - lo >= 3:
            print(f'  {lo:#06x}-{hi:#06x} ({hi - lo + 1:4d}) -> +{lo * 4 + d:#06x}  [{name_of(lo)}]')
    singles = [(i, mapping[i]) for lo, hi, d in runs if hi - lo < 3 for i in range(lo, hi + 1)]
    print(f'mapped single registers: {len(singles)}')
    for i, offs in singles[:200]:
        print(f'  reg {i:#06x} -> ' + ', '.join(f'+{o:#06x}' for o in offs[:4]) + (' ...' if len(offs) > 4 else ''))
    print(f'unmatched varying registers: {len(unmatched)}')
    for i in unmatched[:200]:
        vals = collections.Counter(R[:, i].tolist())
        print(f'  reg {i:#06x} [{name_of(i)}] {len(vals)} values, e.g. ' +
              ', '.join(f'{v:08X}' for v, _ in vals.most_common(4)))
    json.dump({'samples': S, 'mapping': {f'{i:#06x}': offs for i, offs in mapping.items()},
               'unmatched': [f'{i:#06x}' for i in unmatched]}, open(a.json, 'w'), indent=1)
    print('->', a.json)


if __name__ == '__main__':
    main()
