#!/usr/bin/env python3
"""Prints the size and FNV-1a 64 hash of guest functions, for the code checks of the native replacements
(app/src/cpu: a native engages only when the loaded game code is byte for byte the code it was written from).

  python tools/guest_code_hash.py 827F05C0 827EFEA0 ...   [--image out/default_image.bin] [--gen app/generated/default]

A function's size is the distance to the next function start of the generated code. The image is the loaded guest
image (SA2_DUMP_IMAGE, see src/sa2_app.h), based at 0x82000000.
"""
import argparse
import glob
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = 0x82000000


def fnv1a64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('addresses', nargs='+')
    ap.add_argument('--image', default=os.path.join(ROOT, 'out', 'default_image.bin'))
    ap.add_argument('--gen', default=os.path.join(ROOT, 'app', 'generated', 'default'))
    a = ap.parse_args()
    starts = set()
    for path in glob.glob(os.path.join(a.gen, '*_recomp.*.cpp')):
        for m in re.finditer(r'^DEFINE_REX_FUNC\(sub_([0-9A-F]+)\)', open(path, errors='replace').read(), re.M):
            starts.add(int(m.group(1), 16))
    starts = sorted(starts)
    image = open(a.image, 'rb').read()
    for text in a.addresses:
        addr = int(text.replace('sub_', ''), 16)
        following = [s for s in starts if s > addr]
        end = following[0] if following else addr + 4
        code = image[addr - BASE:end - BASE]
        print(f'{addr:08X} bytes {len(code):5d} fnv 0x{fnv1a64(code):016X}')


if __name__ == '__main__':
    main()
