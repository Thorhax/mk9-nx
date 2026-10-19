#!/usr/bin/env python3
"""Prints the instructions that differ between the same function in two generated trees (two games, same library
code): the struct offsets and constants a newer XDK moved.

  python tools/listing_diff.py GEN_A sub_A GEN_B sub_B
"""
import difflib
import glob
import os
import re
import sys


def listing(gen, func):
    for path in glob.glob(os.path.join(gen, '*_recomp.*.cpp')):
        text = open(path, errors='replace').read()
        i = text.find(f'DEFINE_REX_FUNC({func})')
        if i < 0:
            continue
        body = text[i:text.find('\n}', i)]
        return [m.group(1).strip() for m in re.finditer(r'^\t// (.+)$', body, re.M)]
    sys.exit(f'{func} not found in {gen}')


a, b = listing(sys.argv[1], sys.argv[2]), listing(sys.argv[3], sys.argv[4])
norm = lambda s: re.sub(r'0x[0-9a-f]+', 'ADDR', s)
sm = difflib.SequenceMatcher(None, [norm(x) for x in a], [norm(x) for x in b], autojunk=False)
print(f'{sys.argv[2]}: {len(a)} instructions, {sys.argv[4]}: {len(b)}, similarity {sm.ratio():.2f}')
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == 'equal':
        continue
    for k in range(max(i2 - i1, j2 - j1)):
        left = a[i1 + k] if i1 + k < i2 else ''
        right = b[j1 + k] if j1 + k < j2 else ''
        print(f'  {i1 + k:4}  {left:40} | {right}')
