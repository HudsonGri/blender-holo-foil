"""Composite build/gallery/{render,map}_<tile>.png into docs/images/pattern_gallery.jpg, and write square asset
previews to build/previews/ (scripts/build_holo_foil.py embeds them). Pillow, system python.
    python3 scripts/compose_gallery.py [--tiles a b c] [--out path] [--cols N] [--width W]
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TD = os.path.join(ROOT, 'build', 'gallery')
ORDER = ['linear', 'sheen', 'radial', 'scratched', 'glitter', 'cosmos', 'cracked', 'stars', 'beams',
         'dotmatrix', 'image', 'card_holo', 'card_reverse', 'card_layer']
ap = argparse.ArgumentParser()
ap.add_argument('--tiles', nargs='*', default=ORDER)
ap.add_argument('--prefix', default='render_')
ap.add_argument('--out', default=os.path.join(ROOT, 'docs', 'images', 'pattern_gallery.jpg'))
ap.add_argument('--cols', type=int, default=5)
ap.add_argument('--title', default='')
ap.add_argument('--tiledir', default=TD)
ap.add_argument('--width', type=int, default=0, help='downscale tiles to this width')
args = ap.parse_args()
RD = args.tiledir



def font(size, bold=False):
    names = ['Inter-SemiBold.ttf' if bold else 'Inter-Regular.ttf', 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
    dirs = [os.environ.get('HOLO_FONT_DIR', ''), os.path.expanduser('~/.local/share/fonts/inter'),
            '/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/TTF', '/Library/Fonts', 'C:/Windows/Fonts']
    for d in dirs:
        for n in names:
            if d and os.path.exists(os.path.join(d, n)):
                return ImageFont.truetype(os.path.join(d, n), size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


tiles = [t for t in args.tiles if os.path.exists(os.path.join(RD, f'{args.prefix}{t}.png'))]
first = Image.open(os.path.join(RD, f'{args.prefix}{tiles[0]}.png'))
W, H = first.size
if args.width:
    W, H = args.width, int(round(H * args.width / W))
LAB = max(40, W // 9)
f1 = font(max(13, W // 22), bold=True)
f2 = font(max(11, W // 30))
cells = len(tiles) + 1
cols = args.cols
rows = (cells + cols - 1) // cols
G = Image.new('RGB', (cols * W, rows * (H + LAB)), (12, 12, 14))
d = ImageDraw.Draw(G)
infos = []
for i, t in enumerate(tiles):
    x, y = (i % cols) * W, (i // cols) * (H + LAB)
    im = Image.open(os.path.join(RD, f'{args.prefix}{t}.png')).convert('RGB')
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    G.paste(im, (x, y))
    mp = os.path.join(RD, f'map_{t}.png')
    if os.path.exists(mp):
        m = Image.open(mp).convert('RGB')
        bbox = m.point(lambda v: 255 if v > 8 else 0).convert('L').getbbox()
        if bbox:
            m = m.crop(bbox)
        mh = int(H * 0.24)
        m = m.resize((max(1, int(m.width * mh / m.height)), mh), Image.LANCZOS)
        d.rectangle([x + 5, y + 5, x + 8 + m.width, y + 8 + m.height], fill=(200, 200, 200))
        G.paste(m, (x + 7, y + 7))
    js = os.path.join(RD, f'{args.prefix}{t}.json')
    info = json.load(open(js)) if os.path.exists(js) else {'label': t, 'note': ''}
    infos.append(info)
    d.text((x + 8, y + H + 3), info['label'], font=f1, fill=(240, 240, 240))
    d.text((x + 8, y + H + 5 + f1.size), info['note'], font=f2, fill=(170, 170, 175))
# legend cell
i = len(tiles)
x, y = (i % cols) * W, (i // cols) * (H + LAB)
inf = infos[0]
lines = [
    ("Holo pattern library", f1, (255, 255, 255)),
    ("stock Blender 5.2 Cycles, no OSL", f2, (200, 200, 200)),
    (f"BSDF: {inf.get('bsdf', '')}", f2, (200, 200, 200)),
    (f"{inf.get('device', 'CPU')}, {inf.get('spp', '?')} spp" + (" + OIDN" if inf.get('denoise') else ""), f2, (200, 200, 200)),
    ("63 x 88 mm card, 2 small lamps", f2, (200, 200, 200)),
    ("+ dim studio HDRI", f2, (200, 200, 200)),
    ("", f2, (0, 0, 0)),
    ("inset: pattern map", f1, (255, 255, 255)),
    ("hue = grating direction (mod 180)", f2, (200, 200, 200)),
    ("dark = plain foil (Mask 0)", f2, (200, 200, 200)),
    ("darker = no foil / opaque ink", f2, (200, 200, 200)),
]
if args.title:
    lines.insert(0, (args.title, f1, (255, 220, 120)))
yy = y + 20
for txt, f, c in lines:
    d.text((x + 16, yy), txt, font=f, fill=c)
    yy += f.size + 8
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
if args.out.lower().endswith(('.jpg', '.jpeg')):
    G.save(args.out, quality=88, optimize=True, progressive=True)
else:
    G.save(args.out, optimize=True)
print('wrote', args.out, G.size)


# ---------------------------------------------------------------------------------------- asset previews
import re  # noqa: E402

DST = os.path.join(ROOT, 'build', 'previews')
MAP = {
    'Diffraction Grating BSDF': 'sheen', 'Holo Pattern: Linear Rainbow': 'linear', 'Holo Pattern: Radial CD': 'radial',
    'Holo Pattern: Scratched Metal': 'scratched', 'Holo Pattern: Glitter Patches': 'glitter',
    'Holo Pattern: Cosmos': 'cosmos', 'Holo Pattern: Cracked Ice': 'cracked', 'Holo Pattern: Sparkle Stars': 'stars',
    'Holo Pattern: Diagonal Beams': 'beams', 'Holo Pattern: Dot Matrix': 'dotmatrix',
    'Holo Pattern: Image Driven': 'image', 'Holo Pattern: Layer': 'card_layer', 'Holo Card Shader': 'card_reverse',
    'Holo Card': 'card_holo', 'Holo Foil Demo': 'cracked',
}
os.makedirs(DST, exist_ok=True)
for asset, tile in MAP.items():
    p = os.path.join(RD, 'render_%s.png' % tile)
    if not os.path.exists(p):
        print('missing', p)
        continue
    im = Image.open(p).convert('RGB')
    s = min(im.size)
    x0, y0 = (im.width - s) // 2, (im.height - s) // 2 + (im.height - s) // 6
    im = im.crop((x0, y0, x0 + s, y0 + s)).resize((256, 256), Image.LANCZOS)
    out = os.path.join(DST, re.sub(r'[^a-z0-9]+', '_', asset.lower()).strip('_') + '.jpg')
    im.save(out, quality=90, optimize=True)
    print(out)
