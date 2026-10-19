#!/usr/bin/env python3
"""Compares two raw captures of the same window (sa2_video_s recordings, e.g. the emulated GPU and the native
renderer started with the same -VideoAt): every Nth frame of A is paired with the most similar frame of B within a
search window (the two runs drift by a few frames), and the pair's mean absolute difference is reported. Writes the
worst pairs side by side (A | B | 4x difference) so the remaining rendering differences can be looked at.

  python tools/compare_raw.py emu_8s.rgba nat_8s.rgba [--step 30] [--window 90] [--worst 8] [--out worst.png]

--follow centres each search on the previous match instead of the same index: for two vblank_per_swap runs, whose
frames are the same but shifted by a different number of loading-screen frames after every load.

Each .rgba needs its .txt beside it (width height fps count). Needs numpy and Pillow.
"""
import argparse
import os

import numpy as np
from PIL import Image


def open_raw(path):
    w, h, _fps, _count = open(os.path.splitext(path)[0] + '.txt').read().split()[:4]
    w, h = int(w), int(h)
    n = os.path.getsize(path) // (w * h * 4)
    return np.memmap(path, np.uint8, 'r', shape=(n, h, w, 4))


def thumb(frame):
    g = frame[..., :3].astype(np.float32).mean(axis=2)
    h, w = g.shape
    return g[: h - h % 90, : w - w % 160].reshape(90, h // 90, 160, w // 160).mean(axis=(1, 3))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--step', type=int, default=30, help='compare every Nth frame of A')
    ap.add_argument('--window', type=int, default=90, help='search B within +-N frames of the same index')
    ap.add_argument('--worst', type=int, default=8, help='pairs written to --out')
    ap.add_argument('--out', default=None)
    ap.add_argument('--follow', action='store_true', help='search around the previous match (drifting offset)')
    a = ap.parse_args()

    fa, fb = open_raw(a.a), open_raw(a.b)
    n = min(len(fa), len(fb))
    tb = [None] * len(fb)

    def thumb_b(i):
        if tb[i] is None:
            tb[i] = thumb(fb[i])
        return tb[i]

    rows = []
    centre = 0
    for i in range(0, len(fa) if a.follow else n, a.step):
        ta = thumb(fa[i])
        c = centre + a.step if a.follow and rows else i
        lo, hi = max(0, c - a.window), min(len(fb), c + a.window + 1)
        if lo >= hi:
            break
        best = min(range(lo, hi), key=lambda j: float(np.abs(ta - thumb_b(j)).mean()))
        centre = best
        full = float(np.abs(fa[i, ..., :3].astype(np.int16) - fb[best, ..., :3].astype(np.int16)).mean())
        rows.append((i, best, full))
        print(f'A {i:5d}  B {best:5d} ({best - i:+4d})  MAD {full:6.2f}')
    mads = np.array([r[2] for r in rows])
    print(f'{len(rows)} pairs: mean MAD {mads.mean():.2f}, median {np.median(mads):.2f}, max {mads.max():.2f}')
    if a.out:
        worst = sorted(rows, key=lambda r: -r[2])[: a.worst]
        tiles = []
        for i, j, _ in sorted(worst):
            x, y = fa[i, ..., :3], fb[j, ..., :3]
            d = np.clip(np.abs(x.astype(np.int16) - y.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            tiles.append(np.concatenate([x, y, d], 1))
        Image.fromarray(np.concatenate(tiles, 0)).save(a.out)
        print('worst pairs (A frame, B frame):', ', '.join(f'{i}/{j}' for i, j, _ in sorted(worst)), '->', a.out)


if __name__ == '__main__':
    main()
