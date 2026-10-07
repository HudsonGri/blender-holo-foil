"""
Geometry, textbook marks and the printed screen texture of test 3 (laser + CD). Pure Python; the texture
needs PIL, so make it with the system python before building the scene:
    python3 scripts/tests/lib/laser_geom.py      -> scripts/tests/assets/laser_screen.png
"""
import math, os, sys

D = 0.300
PITCH = 1600.0
SCREEN_W, SCREEN_H = 0.94, 0.22
RES = (2400, 562)
CAM_Y, CAM_Z = -0.050, 0.120
BEAM = (0.008, 0.020)
SPREAD_DEG = 0.5
LAMS = (450.0, 532.0, 650.0)
TEX_PX_PER_M = 5000


def font(size, bold=False):
    """Inter if installed (HOLO_FONT_DIR or ~/.local/share/fonts/inter), else DejaVu Sans, else PIL's default."""
    from PIL import ImageFont
    names = ['Inter-SemiBold.ttf' if bold else 'Inter-Medium.ttf', 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf']
    dirs = [os.environ.get('HOLO_FONT_DIR', ''), os.path.expanduser('~/.local/share/fonts/inter'),
            '/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/TTF', '/Library/Fonts', 'C:/Windows/Fonts']
    for d in dirs:
        for n in names:
            p = os.path.join(d, n)
            if d and os.path.exists(p):
                return ImageFont.truetype(p, size)
    try:
        return ImageFont.load_default(size)
    except TypeError:
        return ImageFont.load_default()


def textbook_marks(d=PITCH, D=D):
    marks = []
    for m in (1, 2, 3):
        for lam in LAMS:
            s = m * lam / d
            if s >= 1:
                marks.append(dict(m=m, lam=lam, sin=s, theta_deg=None, x_m=None))
                continue
            th = math.asin(s)
            marks.append(dict(m=m, lam=lam, sin=round(s, 5), theta_deg=round(math.degrees(th), 3),
                              x_m=round(D * math.tan(th), 5)))
    return marks


def make_texture(path, marks):
    from PIL import Image, ImageDraw
    W, H = int(SCREEN_W * TEX_PX_PER_M), int(SCREEN_H * TEX_PX_PER_M)
    im = Image.new('RGB', (W, H), (255, 255, 255))
    d = ImageDraw.Draw(im)
    f_num = font(38, bold=True)
    f_small = font(30)
    f_title = font(34, bold=True)

    def X(x):     # metres from the beam -> texture px
        return W / 2 + x * TEX_PX_PER_M

    def Z(z):     # metres above the beam -> texture px (top row first)
        return H / 2 - z * TEX_PX_PER_M
    # ruler band at the bottom
    z0 = -0.080
    for mm in range(-460, 461):
        x = X(mm / 1000.0)
        L = 0.012 if mm % 10 == 0 else (0.008 if mm % 5 == 0 else 0.004)
        d.line([(x, Z(z0)), (x, Z(z0 - L))], fill=(20, 20, 20), width=3 if mm % 10 == 0 else 2)
        if mm % 50 == 0:
            t = '%d' % abs(mm // 10)
            tw = d.textlength(t, font=f_num)
            d.text((x - tw / 2, Z(z0 - 0.014)), t, font=f_num, fill=(20, 20, 20))
    d.line([(X(-0.46), Z(z0)), (X(0.46), Z(z0))], fill=(20, 20, 20), width=3)
    d.text((X(0.405), Z(z0 - 0.0215)), 'cm', font=f_small, fill=(20, 20, 20))
    ink = {450.0: (40, 70, 230), 532.0: (20, 160, 40), 650.0: (225, 30, 30)}
    for mk in marks:
        if mk['x_m'] is None or mk['m'] > 2:
            continue
        for sgn in (1, -1):
            x = X(sgn * mk['x_m'])
            if not (0 < x < W):
                continue
            for za, zb in ((0.052, 0.068), (-0.068, -0.052)):
                d.line([(x, Z(za)), (x, Z(zb))], fill=ink[mk['lam']], width=5)
            t = '%d' % mk['lam']
            tw = d.textlength(t, font=f_small)
            d.text((x - tw / 2, Z(0.052) - 4), t, font=f_small, fill=ink[mk['lam']])
    for m in (1, 2):
        xs = [mk['x_m'] for mk in marks if mk['m'] == m and mk['x_m'] is not None]
        xc = (min(xs) + max(xs)) / 2
        for sgn in (1, -1):
            t = 'm = %+d' % (sgn * m)
            tw = d.textlength(t, font=f_num)
            d.text((X(sgn * xc) - tw / 2, Z(0.098)), t, font=f_num, fill=(20, 20, 20))
    t = 'm = 0 (beam)'
    tw = d.textlength(t, font=f_small)
    d.text((X(0) - tw / 2, Z(0.098)), t, font=f_small, fill=(20, 20, 20))
    im.save(path)
    return path




TEX_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'laser_screen.png')

if __name__ == '__main__':
    os.makedirs(os.path.dirname(TEX_PATH), exist_ok=True)
    print(make_texture(TEX_PATH, textbook_marks()))
    for mk in textbook_marks():
        print(mk)
