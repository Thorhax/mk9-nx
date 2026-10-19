#!/usr/bin/env python3
"""Lines up a recording of this port (raw sa2_video_s capture) against a reference video of the same scenes (for
example a console recording) and writes side-by-side pairs, so rendering differences can be checked by eye.

For each reference timestamp it finds our most similar frame (mean absolute difference of 160x90 grayscale
thumbnails, searched over a frame range of our recording) and writes reference | ours at 640x360 each.

  python tools/compare_video.py REF.mp4 OURS.rgba --times 132,136,140 [--range 5600:11153] [--out pairs.png]

OURS.rgba needs its .txt beside it (width height fps count, as written by the recorder). Needs ffmpeg
(C:/ffmpeg/ffmpeg.exe or $FFMPEG), numpy and Pillow.
"""
import argparse
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

FFMPEG = os.environ.get('FFMPEG', 'C:/ffmpeg/ffmpeg.exe')
TW, TH = 160, 90


def thumb(rgb):
    h, w = rgb.shape[:2]
    g = rgb[..., :3].astype(np.float32).mean(axis=2)
    return g[: h - h % TH, : w - w % TW].reshape(TH, h // TH, TW, w // TW).mean(axis=(1, 3))


def ref_frame(path, t):
    out = subprocess.run([FFMPEG, '-v', 'error', '-ss', str(t), '-i', path, '-frames:v', '1', '-vf', 'scale=640:360',
                          '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(360, 640, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('ours')
    ap.add_argument('--times', required=True, help='reference timestamps in seconds, comma separated')
    ap.add_argument('--range', default='', help='frame range of our recording to search, a:b')
    ap.add_argument('--out', default='pairs.png')
    a = ap.parse_args()

    w, h, fps, count = open(a.ours[: -len('.rgba')] + '.txt').read().split()
    w, h, count = int(w), int(h), int(count)
    lo, hi = (int(x) for x in a.range.split(':')) if a.range else (0, count)
    frame_bytes = w * h * 4

    times = [float(t) for t in a.times.split(',')]
    refs = [ref_frame(a.ref, t) for t in times]
    ref_thumbs = [thumb(r) for r in refs]
    best = [(1e9, -1)] * len(refs)
    with open(a.ours, 'rb') as f:
        f.seek(lo * frame_bytes)
        for n in range(lo, hi):
            buf = f.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            t = thumb(np.frombuffer(buf, np.uint8).reshape(h, w, 4))
            for i, rt in enumerate(ref_thumbs):
                d = float(np.abs(t - rt).mean())
                if d < best[i][0]:
                    best[i] = (d, n)

    sheet = Image.new('RGB', (1280, 360 * len(refs)))
    draw = ImageDraw.Draw(sheet)
    with open(a.ours, 'rb') as f:
        for i, ((d, n), t, r) in enumerate(zip(best, times, refs)):
            f.seek(n * frame_bytes)
            ours = np.frombuffer(f.read(frame_bytes), np.uint8).reshape(h, w, 4)[..., :3]
            sheet.paste(Image.fromarray(r), (0, 360 * i))
            sheet.paste(Image.fromarray(ours).resize((640, 360)), (640, 360 * i))
            label = f'reference {int(t // 60)}:{t % 60:04.1f}   |   ours frame {n} (difference {d:.1f})'
            draw.rectangle((0, 360 * i, 640, 360 * i + 22), fill=(0, 0, 0))
            draw.text((6, 360 * i + 5), label, fill=(255, 255, 0))
            print(f'ref {t:7.1f} s -> ours frame {n:6d}  mean abs diff {d:.1f}')
    sheet.save(a.out)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
