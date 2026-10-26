#!/usr/bin/env python3
"""PNGs of the textures mk9_dump_textures_dir wrote (game data: keep them in out/, never in the repository).

    python tools/texture_dump_png.py DUMP_DIR OUT_DIR [--sheets]

Names: <hash>_<w>x<h>_f<guest format>_h<plume format>_b<host block bytes>_p<row pitch>_d<decoded>.bin. Decodes BC1,
BC2, BC3 (plume 55, 58, 61), R8 (50) and RGBA8 (20); others are skipped. --sheets also writes contact sheets
(sheet_NN.png) with each texture's hash under it.
"""
import os, re, sys, struct
import numpy as np
from PIL import Image, ImageDraw

NAME = re.compile(r"([0-9A-F]{16})_(\d+)x(\d+)_f(\d+)_h(\d+)_b(\d+)_p(\d+)_d(\d)\.bin")

def rgb565(c):
    r, g, b = (c >> 11) & 31, (c >> 5) & 63, c & 31
    return np.array([r * 255 // 31, g * 255 // 63, b * 255 // 31], np.int32)

def color_block(b, opaque4):
    c0, c1, idx = struct.unpack_from("<HHI", b, 0)
    p0, p1 = rgb565(c0), rgb565(c1)
    if c0 > c1 or opaque4:
        pal = [p0, p1, (2 * p0 + p1) // 3, (p0 + 2 * p1) // 3]; alpha = [255] * 4
    else:
        pal = [p0, p1, (p0 + p1) // 2, np.zeros(3, np.int32)]; alpha = [255, 255, 255, 0]
    out = np.zeros((4, 4, 4), np.uint8)
    for i in range(16):
        k = (idx >> (2 * i)) & 3
        out[i // 4, i % 4, :3] = pal[k]; out[i // 4, i % 4, 3] = alpha[k]
    return out

def alpha_bc3(b):
    a0, a1 = b[0], b[1]
    bits = int.from_bytes(b[2:8], "little")
    if a0 > a1: pal = [a0, a1] + [((7 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    else: pal = [a0, a1] + [((5 - i) * a0 + i * a1) // 5 for i in range(1, 5)] + [0, 255]
    return np.array([pal[(bits >> (3 * i)) & 7] for i in range(16)], np.uint8).reshape(4, 4)

def decode(data, w, h, fmt, pitch):
    if fmt in (55, 58, 61):
        bb = 8 if fmt == 55 else 16
        bw, bh = (w + 3) // 4, (h + 3) // 4
        img = np.zeros((bh * 4, bw * 4, 4), np.uint8)
        for by in range(bh):
            row = data[by * pitch: by * pitch + bw * bb]
            for bx in range(bw):
                blk = row[bx * bb:(bx + 1) * bb]
                if fmt == 55:
                    px = color_block(blk, False)
                else:
                    px = color_block(blk[8:], True)
                    if fmt == 61: px[:, :, 3] = alpha_bc3(blk[:8])
                    else: px[:, :, 3] = np.array([((int.from_bytes(blk[:8], "little") >> (4 * i)) & 15) * 17 for i in range(16)], np.uint8).reshape(4, 4)
                img[by * 4:by * 4 + 4, bx * 4:bx * 4 + 4] = px
        return img[:h, :w]
    if fmt == 50:
        a = np.frombuffer(data, np.uint8)[: pitch * h].reshape(h, pitch)[:, :w]
        return np.dstack([a, a, a, np.full_like(a, 255)])
    if fmt == 20:
        return np.frombuffer(data, np.uint8)[: pitch * h].reshape(h, pitch)[:, : w * 4].reshape(h, w, 4).copy()
    return None

def main():
    src, dst = sys.argv[1], sys.argv[2]; sheets = "--sheets" in sys.argv
    os.makedirs(dst, exist_ok=True); thumbs = []
    for n in sorted(os.listdir(src)):
        m = NAME.match(n)
        if not m: continue
        hsh, w, h, gf, hf, bb, pitch, dec = m.groups(); w, h, hf, pitch = int(w), int(h), int(hf), int(pitch)
        if hf == 50 and (w, h) in ((640, 360), (1280, 720)): continue  # movie planes
        img = decode(open(os.path.join(src, n), "rb").read(), w, h, hf, pitch)
        if img is None: continue
        im = Image.fromarray(img, "RGBA"); out = os.path.join(dst, n[:-4] + ".png"); im.save(out)
        if sheets: thumbs.append((hsh, w, h, im))
    if sheets:
        T = 192; cols = 8; per = 48
        for s in range(0, len(thumbs), per):
            chunk = thumbs[s:s + per]; rows = (len(chunk) + cols - 1) // cols
            sheet = Image.new("RGB", (cols * T, rows * (T + 14)), (255, 0, 255)); d = ImageDraw.Draw(sheet)
            for i, (hsh, w, h, im) in enumerate(chunk):
                t = im.copy(); t.thumbnail((T, T)); bg = Image.new("RGB", t.size, (40, 40, 40)); bg.paste(t, mask=t.split()[3])
                x, y = (i % cols) * T, (i // cols) * (T + 14); sheet.paste(bg, (x, y)); d.text((x + 2, y + T), f"{hsh[:6]} {w}x{h}", fill=(255, 255, 255))
            sheet.save(os.path.join(dst, f"sheet_{s // per:02d}.png"))

main()
