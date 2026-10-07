"""
Test 3b: the same white-beam "laser" experiment for a CD, a DVD and a Blu-ray (1.6 / 0.74 / 0.32 um), with a
wider screen closer to the disc so the DVD's large angles fit. Geometry, textbook marks and one printed screen
texture per disc. Make the textures with the system python before building the scene:
    python3 scripts/tests/lib/laser_discs_geom.py   -> scripts/tests/assets/laser_screen_{cd,dvd,bd}.png
"""
import math, os

D = 0.200                       # disc -> screen (m)
SCREEN_W, SCREEN_H = 1.24, 0.22
RES = (3200, 568)
CAM_Y, CAM_Z = -0.050, 0.120
BEAM = (0.008, 0.020)
SPREAD_DEG = 0.5
LAMS = (450.0, 532.0, 650.0)
TEX_PX_PER_M = 4000
DISCS = (('CD', 1600.0, 'cd'), ('DVD', 740.0, 'dvd'), ('Blu-ray', 320.0, 'bd'))
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets')


def tex_path(key):
    return os.path.join(ASSETS, 'laser_screen_%s.png' % key)


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


def textbook_marks(d, D=D):
    marks = []
    for m in (1, 2, 3):
        for lam in LAMS:
            s = m * lam / d
            if s >= 1:
                marks.append(dict(m=m, lam=lam, sin=round(s, 5), theta_deg=None, x_m=None))
                continue
            th = math.asin(s)
            marks.append(dict(m=m, lam=lam, sin=round(s, 5), theta_deg=round(math.degrees(th), 3),
                              x_m=round(D * math.tan(th), 5)))
    return marks


def make_texture(path, name, d):
    from PIL import Image, ImageDraw
    marks = textbook_marks(d)
    W, H = int(SCREEN_W * TEX_PX_PER_M), int(SCREEN_H * TEX_PX_PER_M)
    im = Image.new('RGB', (W, H), (255, 255, 255))
    dr = ImageDraw.Draw(im)
    f_num = font(32, bold=True)
    f_small = font(26)
    f_big = font(40, bold=True)

    def X(x):
        return W / 2 + x * TEX_PX_PER_M

    def Z(z):
        return H / 2 - z * TEX_PX_PER_M
    z0 = -0.080
    half_mm = int(SCREEN_W * 500) - 5
    for mm in range(-half_mm, half_mm + 1):
        x = X(mm / 1000.0)
        L = 0.012 if mm % 10 == 0 else (0.008 if mm % 5 == 0 else 0.004)
        dr.line([(x, Z(z0)), (x, Z(z0 - L))], fill=(20, 20, 20), width=3 if mm % 10 == 0 else 1)
        if mm % 50 == 0:
            t = '%d' % abs(mm // 10)
            tw = dr.textlength(t, font=f_num)
            dr.text((x - tw / 2, Z(z0 - 0.014)), t, font=f_num, fill=(20, 20, 20))
    dr.line([(X(-half_mm / 1000), Z(z0)), (X(half_mm / 1000), Z(z0))], fill=(20, 20, 20), width=3)
    ink = {450.0: (40, 70, 230), 532.0: (20, 160, 40), 650.0: (225, 30, 30)}
    shown = [mk for mk in marks if mk['x_m'] is not None and mk['m'] <= 2 and mk['x_m'] < SCREEN_W / 2]
    for mk in shown:
        for sgn in (1, -1):
            x = X(sgn * mk['x_m'])
            for za, zb in ((0.052, 0.068), (-0.068, -0.052)):
                dr.line([(x, Z(za)), (x, Z(zb))], fill=ink[mk['lam']], width=5)
            t = '%d' % mk['lam']
            tw = dr.textlength(t, font=f_small)
            dr.text((x - tw / 2, Z(0.052) - 2), t, font=f_small, fill=ink[mk['lam']])
    for m in (1, 2):
        xs = [mk['x_m'] for mk in shown if mk['m'] == m]
        if not xs:
            continue
        xc = (min(xs) + max(xs)) / 2
        for sgn in (1, -1):
            t = 'm = %+d' % (sgn * m)
            tw = dr.textlength(t, font=f_num)
            dr.text((X(sgn * xc) - tw / 2, Z(0.098)), t, font=f_num, fill=(20, 20, 20))
    t = '%s, d = %g µm: printed marks = textbook sin θ = mλ/d' % (name, d / 1000)
    dr.text((X(-SCREEN_W / 2 + 0.01), Z(0.104)), t, font=f_small, fill=(20, 20, 20))
    if not shown:
        for sgn in (1, -1):
            t = 'no first order: λ/d > 1 for all visible λ'
            tw = dr.textlength(t, font=f_big)
            dr.text((X(sgn * 0.33) - tw / 2, Z(0.075)), t, font=f_big, fill=(200, 30, 30))
    im.save(path)
    return marks


if __name__ == '__main__':
    os.makedirs(ASSETS, exist_ok=True)
    for name, d, key in DISCS:
        mk = make_texture(tex_path(key), name, d)
        print(tex_path(key), [(m['m'], m['lam'], m['x_m']) for m in mk if m['x_m'] is not None and m['m'] <= 2])
