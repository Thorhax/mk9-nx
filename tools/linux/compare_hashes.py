#!/usr/bin/env python3
"""Compares two lockstep recordings' per-frame hashes (sa2-nx's recorder: one little-endian XXH3-64 per frame).

  python3 tools/linux/compare_hashes.py out/pc/lockstep/A.hash out/pc/lockstep/B.hash
Prints how many frames match and the first ones that differ (frame index from the start of the recording)."""
import struct
import sys

a, b = (open(p, "rb").read() for p in sys.argv[1:3])
ha = struct.unpack(f"<{len(a) // 8}Q", a[:len(a) // 8 * 8])
hb = struct.unpack(f"<{len(b) // 8}Q", b[:len(b) // 8 * 8])
# Lockstep runs are the same frames shifted by their loading-screen frames: try every shift and keep the best.
best = max(range(-120, 121), key=lambda k: sum(1 for i in range(len(ha)) if 0 <= i + k < len(hb) and ha[i] == hb[i + k]))
pairs = [(i, i + best) for i in range(len(ha)) if 0 <= i + best < len(hb)]
print(f"best shift: B = A {best:+d} frames")
n = len(pairs)
diff = [i for i, j in pairs if ha[i] != hb[j]]
print(f"{n} frames compared ({len(ha)} / {len(hb)}): {n - len(diff)} identical, {len(diff)} different")
if diff:
    print("first differing frames:", diff[:20])
