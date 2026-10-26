#!/usr/bin/env python3
"""sheet.py out.jpg files...: 4-column contact sheet at 320x180 per image"""
import sys
from PIL import Image, ImageDraw
fs = sys.argv[2:]; W, H = 400, 225; cols = 4; rows = (len(fs) + cols - 1) // cols
sh = Image.new("RGB", (W * cols, H * rows))
for i, f in enumerate(fs):
    im = Image.open(f).resize((W, H)); ImageDraw.Draw(im).text((4, 4), str(i + 1), fill=(255, 255, 0))
    sh.paste(im, ((i % cols) * W, (i // cols) * H))
sh.save(sys.argv[1], quality=85)
