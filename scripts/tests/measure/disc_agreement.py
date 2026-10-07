"""How well each render agrees with the textbook prediction image (disc_textbook.py), on the data area:
* lit-area overlap (IoU) of 'diffracted light present' masks (pixel luminance > 8 % of that image's 99.5th
  percentile, after subtracting the image's dark-disc level);
* median hue difference (degrees) on pixels lit in both.
(ulimit -v 8000000; python3 scripts/tests/measure/disc_agreement.py RENDER_DIR) -> disc_agreement.json"""
import json, os, sys
import numpy as np
from PIL import Image
import colorsys
d = sys.argv[1]


def lin(path):
    a = np.asarray(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
    return a


def hue(a):
    mx, mn = a.max(-1), a.min(-1)
    c = mx - mn + 1e-6
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    h = np.where(mx == r, ((g - b) / c) % 6, np.where(mx == g, (b - r) / c + 2, (r - g) / c + 4))
    return h * 60.0, c


out = {}
for view in ('std', 'graze'):
    for suf, name in (('', 'CD'), ('_dvd', 'DVD'), ('_bd', 'Blu-ray')):
        if not os.path.exists(os.path.join(d, 'p2_discs_%s_textbook%s.png' % (view, suf))):
            continue
        tb = lin(os.path.join(d, 'p2_discs_%s_textbook%s.png' % (view, suf)))
        ht, ct = hue(tb)
        mt = ct > 0.08 * np.percentile(ct, 99.5) if np.percentile(ct, 99.5) > 0.05 else np.zeros(ct.shape, bool)
        row = dict(textbook_lit_px=int(mt.sum()))
        for v in ('balanced', 'ours'):
            f = os.path.join(d, 'p2_discs_%s_%s%s.png' % (view, v, suf))
            if not os.path.exists(f):
                continue
            a = lin(f)
            h, c = hue(a)
            thr = max(0.08 * np.percentile(c, 99.5), 0.06)
            m = c > thr
            inter, union = (m & mt).sum(), (m | mt).sum()
            both = m & mt
            dh = np.abs((h[both] - ht[both] + 180) % 360 - 180)
            row[v] = dict(lit_px=int(m.sum()), iou=round(float(inter / union), 3) if union else None,
                          median_hue_diff_deg=round(float(np.median(dh)), 1) if both.any() else None,
                          lit_where_textbook_dark_px=int((m & ~mt).sum()))
        out['%s %s' % (view, name)] = row
        print(view, name, row)
json.dump(out, open(os.path.join(d, 'disc_agreement.json'), 'w'), indent=1)
