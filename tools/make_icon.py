# The original mk9-nx icon (recoloured from usf4-nx's generator (app/icon.jpg): no game artwork). Drawn at 1024 px, saved at 256.
# Usage: python3 tools/make_icon.py   (writes app/icon.jpg; needs Pillow and the Noto Sans fonts)
import math, random
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageChops
S = 1024
random.seed(4)
F = '/usr/share/fonts/truetype/noto/'
# Background: near black, with a blood-red radial glow behind the text.
bg = Image.new('RGB', (S, S))
px = bg.load()
for y in range(S):
    for x in range(S):
        d = math.hypot(x - S * 0.5, y - S * 0.46) / (S * 0.72)
        g = max(0.0, 1.0 - d)
        r = int(12 + 190 * g ** 2.0)
        gg = int(6 + 18 * g ** 2.6)
        b = int(8 + 10 * g)
        px[x, y] = (min(r, 255), min(gg, 255), b)
# Energy burst: thin rays from the centre.
rays = Image.new('L', (S, S), 0)
dr = ImageDraw.Draw(rays)
cx, cy = S * 0.5, S * 0.46
for i in range(56):
    a = random.uniform(0, 2 * math.pi)
    w = random.uniform(0.006, 0.02)
    L = random.uniform(0.45, 0.8) * S
    dr.polygon([(cx, cy), (cx + L * math.cos(a - w), cy + L * math.sin(a - w)),
                (cx + L * math.cos(a + w), cy + L * math.sin(a + w))], fill=random.randint(40, 110))
rays = rays.filter(ImageFilter.GaussianBlur(6))
bg = Image.composite(Image.new('RGB', (S, S), (255, 60, 30)), bg, rays)
# Diagonal slash streaks (brush-like), deep red.
sl = Image.new('L', (S, S), 0)
ds = ImageDraw.Draw(sl)
for k, (off, th) in enumerate([(-120, 70), (40, 34), (150, 18)]):
    pts = []
    for t in range(0, 101):
        u = t / 100
        x = -60 + u * (S + 120)
        y = S * 0.80 - u * S * 0.62 + off + math.sin(u * 3.1) * 30
        pts.append((x, y))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        u = x0 / S
        wdt = th * (0.25 + math.sin(min(max(u, 0), 1) * math.pi))
        ds.line([(x0, y0), (x1, y1)], fill=255, width=max(2, int(wdt)))
sl = sl.filter(ImageFilter.GaussianBlur(3))
grad = Image.linear_gradient('L').rotate(90).resize((S, S))
fire = Image.merge('RGB', (Image.new('L', (S, S), 255), ImageChops.multiply(grad, Image.new('L', (S, S), 60)).point(lambda v: v + 10), Image.new('L', (S, S), 10)))
bg = Image.composite(fire, bg, sl.point(lambda v: int(v * 0.55)))
# Text: heavy italic "MK9" with dark outline and glow, "NX" badge below.
def text_layer(txt, font, pos, fill, stroke, sw):
    lay = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    ImageDraw.Draw(lay).text(pos, txt, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke, anchor='mm')
    return lay
big = ImageFont.truetype(F + 'NotoSans-CondensedBlackItalic.ttf', 380)
glow = text_layer('MK9', big, (S * 0.5, S * 0.44), (255, 40, 10, 255), (255, 40, 10, 255), 30).filter(ImageFilter.GaussianBlur(28))
img = bg.convert('RGBA')
img = Image.alpha_composite(img, glow)
# Text fill: vertical gold-to-bronze gradient.
mask = Image.new('L', (S, S), 0)
ImageDraw.Draw(mask).text((S * 0.5, S * 0.44), 'MK9', font=big, fill=255, anchor='mm')
tg = Image.new('RGB', (S, S))
tp = ImageDraw.Draw(tg)
for y in range(S):
    t = min(max((y - S * 0.30) / (S * 0.28), 0), 1)
    tp.line([(0, y), (S, y)], fill=(int(255 - 60 * t), int(220 - 110 * t), int(110 - 90 * t)))
outline = text_layer('MK9', big, (S * 0.5, S * 0.44), (0, 0, 0, 0), (20, 10, 30, 255), 14)
img = Image.alpha_composite(img, outline)
img.paste(tg, (0, 0), mask)
# White edge highlight on the top of the letters.
hi = ImageChops.subtract(mask, mask.transform(mask.size, Image.AFFINE, (1, 0, 0, 0, 1, -8)))
img.paste((255, 255, 240), (0, 0), hi.point(lambda v: int(v * 0.8)))
# NX badge.
small = ImageFont.truetype(F + 'NotoSans-BlackItalic.ttf', 120)
bx, by, bw, bh = S * 0.5, S * 0.775, 300, 150
badge = Image.new('RGBA', (S, S), (0, 0, 0, 0))
bd = ImageDraw.Draw(badge)
bd.rounded_rectangle([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2], radius=34, fill=(230, 20, 40, 255), outline=(255, 255, 255, 255), width=10)
img = Image.alpha_composite(img, badge)
img = Image.alpha_composite(img, text_layer('NX', small, (bx, by - 6), (255, 255, 255, 255), (120, 0, 10, 255), 4))
# Vignette.
vig = Image.new('L', (S, S), 0)
ImageDraw.Draw(vig).ellipse([-S * 0.25, -S * 0.25, S * 1.25, S * 1.25], fill=255)
vig = vig.filter(ImageFilter.GaussianBlur(120))
img = Image.composite(img, Image.new('RGBA', (S, S), (0, 0, 0, 255)), vig)
img.convert('RGB').resize((256, 256), Image.LANCZOS).save(__import__('os').path.join(__import__('os').path.dirname(__file__), '..', 'app', 'icon.jpg'), quality=95)
