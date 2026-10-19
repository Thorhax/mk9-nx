#!/usr/bin/env python3
"""Finds every raw (headerless) Xenos shader in the module images and writes it the way a discovery run would
(<stage>_<hash>.ucode + .txt, SA2_SHADER_DISCOVERY format), so tools/wrap_discovered.py can put them in the
shader library without having to play every mode first.

Mushihimesama Futari keeps a block of raw shaders in .rdata (default.xex 0x82003268.., every core 0x88001680.. and
0x88002A38..): Direct3D's own (clear, gamma, ...) and the game's, loaded with IM_LOAD_IMMEDIATE. They are packed back to
back, each followed by one zero instruction, and the immediate load takes that padding too: the four found by
discovery runs (vs_e35e, vs_6d38, vs_1436, ps_d40d) are all (instructions + 1) * 3 words. A program is recognised by
its control flow alone:
  - the first exec starts right after the control-flow triples (address == number of CF triples),
  - the execs cover the instructions back to back and the last one is an EXEC_END,
  - only exec / alloc / nop opcodes, at least one alloc.
The stage comes from the allocs: a vertex shader allocates the position (alloc type 1), a pixel shader never does.

  python tools/sweep_raw_shaders.py out/default_image.bin out/reco_normal_image.bin ... --out out/pc/shaders_sweep

The hash in the file name is XXH3-64 of the words in host (little-endian) order, as the renderer names its dumps.
"""
import argparse
import os
import struct
import sys

import xxhash

EXEC_OPS = {1, 2, 3, 4, 5, 6, 13, 14}  # exec, exec_end, cond_exec(_end), cond_pred_exec(_end), pred_clean(_end)
END_OPS = {2, 4, 6, 14}
ALLOC, NOP = 12, 0
MAX_CF_TRIPLES = 16
MAX_INSTRUCTIONS = 256


def cf_pair(w0, w1, w2):
    a = w0 | ((w1 & 0xFFFF) << 32)
    b = (w1 >> 16) | (w2 << 16)
    return a, b


def parse(words, start):
    """(instructions, stage) of a program starting at word index start, or None."""
    cfs = []
    triples = 0
    ended = False
    while triples < MAX_CF_TRIPLES and not ended:
        i = start + triples * 3
        if i + 3 > len(words):
            return None
        for cf in cf_pair(*words[i:i + 3]):
            cfs.append(cf)
            if (cf >> 44) & 0xF in END_OPS:
                ended = True
                break
        triples += 1
    if not ended:
        return None
    expected = triples
    allocs = []
    for cf in cfs:
        op = (cf >> 44) & 0xF
        if op in EXEC_OPS:
            address, count = cf & 0xFFF, (cf >> 12) & 0x7
            if address != expected or count == 0:
                return None
            expected += count
        elif op == ALLOC:
            allocs.append((cf >> 41) & 0x3)
        elif op != NOP or cf != 0:
            return None
    if not allocs or expected > MAX_INSTRUCTIONS or expected == triples:
        return None
    stage = 'vs' if 1 in allocs else 'ps'
    return expected, stage


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='+', help='module images (REXGLUE_DUMP_IMAGE dumps, out/<module>_image.bin)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=lambda s: int(s, 0), default=0x10000,
                    help='only the first bytes of each image are swept (the raw block sits near the start of .rdata)')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    found = {}
    for path in a.images:
        data = open(path, 'rb').read()[:a.limit]
        words = list(struct.unpack('>%dI' % (len(data) // 4), data[:len(data) // 4 * 4]))
        base = 0x82000000 if 'default' in os.path.basename(path) else 0x88000000
        i = 0
        while i < len(words):
            # A program starts after zeros (padding of the previous one) or at an exec / alloc triple.
            if words[i] == 0 and (i + 1 >= len(words) or (words[i + 1] >> 16) == 0):
                i += 1
                continue
            r = parse(words, i)
            if not r:
                i += 1
                continue
            instructions, stage = r
            n = (instructions + 1) * 3
            body = words[i:i + n]
            if len(body) != n or any(body[instructions * 3:]):
                i += 1  # the padding instruction must be zero, otherwise this is not a packed raw program
                continue
            host = struct.pack('<%dI' % n, *body)
            h = xxhash.xxh3_64_intdigest(host)
            name = f'{stage}_{h:016x}'
            where = f'{os.path.basename(path)}:{base + i * 4:08X}'
            if name not in found:
                found[name] = (body, [where])
                with open(os.path.join(a.out, name + '.ucode'), 'wb') as f:
                    f.write(struct.pack('>%dI' % n, *body))
            else:
                found[name][1].append(where)
            i += instructions * 3
    for name, (body, places) in sorted(found.items()):
        with open(os.path.join(a.out, name + '.txt'), 'w') as f:
            f.write(f'stage={name[:2]}\naddress=00000000\nwords={len(body)}\nhash={name[3:].upper()}\n')
            f.write('found=' + ' '.join(places) + '\n')
            for k, w in enumerate(body):
                f.write(f'{k:04d} {w:08X}\n')
        print(f'{name} {len(body):3d} words at {places[0]}' + (f' (+{len(places) - 1} more)' if len(places) > 1 else ''))
    print(f'{len(found)} distinct raw shaders')


if __name__ == '__main__':
    sys.exit(main())
