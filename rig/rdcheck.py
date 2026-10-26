#!/usr/bin/env python3
"""rdcheck.py <guest virtual base hex> [lo hi]: reads the running game's read-only data (default 82000400-822B3404,
the range tools/ro_constants.py folds) through sys-botbase's peekAbsolute and compares it with the code generator's
image dump (out/default_image.bin), the XEX's import records excepted. Every differing word is a folded
constant that would be wrong."""
import os, sys, binascii
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb
P = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(P, 'tools'))
import ro_constants
vbase = int(sys.argv[1], 16)
lo = int(sys.argv[2], 16) if len(sys.argv) > 2 else 0x82000400
hi = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0x822B3404
image = open(os.path.join(P, 'out', 'default_image.bin'), 'rb').read()
records = ro_constants.import_records(os.path.join(P, 'assets', 'game_root', 'default.xex'))
CH = 0x2000
diff = []
a = lo
while a < hi:
    n = min(CH, hi - a)
    live = binascii.unhexlify(sb.query(f'peekAbsolute 0x{vbase + a:X} {n}'))
    ref = image[a - 0x82000000:a - 0x82000000 + n]
    if live != ref:
        for w in range(0, n - 3, 4):
            if live[w:w + 4] != ref[w:w + 4] and (a + w) not in records:
                diff.append((a + w, ref[w:w + 4].hex(), live[w:w + 4].hex()))
    a += n
print(f'{(hi - lo) // 1024} KB compared, {len(records)} import records skipped, {len(diff)} words differ')
for d in diff[:40]:
    print('  %08X image %s live %s' % d)
