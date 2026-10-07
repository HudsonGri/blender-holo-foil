"""
Test 4, rainbow order: where the textbook grating equation and Balanced's lobe law put each colour along the
centre line of the banding scene (scripts/tests/scenes/s5_banding.py, wide 'CompareCam' view), and what colour each
render (s5_banding_{balanced,ours}.png from scripts/tests/render.py, any square resolution) shows there.

Scene: flat plate, 1D grating d = 1459.08 nm with its grating vector along y, one 3 cm lamp at (0, 0.285, 2.986),
camera (0, -3, 5.6) -> (0, 0, 0), 50 mm, vertical sensor fit 24 mm.
Textbook: inside every order blue sits nearest the mirror highlight and red farthest, and no first-order colour
can sit closer to the mirror direction than |sin ti + sin to| = 380/d.
(ulimit -v 8000000; python3 scripts/tests/measure/rainbow_order.py RENDER_DIR) -> RENDER_DIR/rainbow_order.json
"""
import json, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import physics as PH  # noqa

RDIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', '..', 'build', 'tests', 'renders')
imgs = {k: np.asarray(Image.open(os.path.join(RDIR, 's5_banding_%s.png' % k)).convert('RGB')).astype(float)
        for k in ('balanced', 'ours') if os.path.exists(os.path.join(RDIR, 's5_banding_%s.png' % k))}
if not imgs:
    sys.exit('no s5_banding_*.png in ' + RDIR)
H_, W_ = next(iter(imgs.values())).shape[:2]
CAM, TGT, LENS, RES = (0, -3.0, 5.6), (0, 0, 0), 50.0, (W_, H_)
C0, C1 = W_ // 2 - 4, W_ // 2 + 5                 # centre column band
LAMP = np.array([0, 0.285, 2.986])
D = 1459.08
cam = PH.PinholeCam(CAM, TGT, LENS, RES, sensor=24.0)
ys = np.linspace(-1.4, 1.4, 56001)
P = np.stack([0 * ys, ys, 0 * ys], -1)
v = PH.norm(np.array(CAM) - P)
l = PH.norm(LAMP - P)
N = np.broadcast_to([0, 0, 1.0], P.shape)
g = np.broadcast_to([0, 1.0, 0], P.shape)
T = np.broadcast_to([1.0, 0, 0], P.shape)


def px(y):
    return float(cam.project(np.array([0, y, 0]))[0][1])


out = dict(scene='s5_banding CompareCam', pitch_nm=D, res=list(RES))
s = (l + v)[:, 1]
y0 = float(ys[np.argmin(abs(s))])
out['mirror'] = dict(y_m=round(y0, 4), px_y=round(px(y0), 1))
tb = {}
for m in (-1, -2, -3):
    lam_y, _ = PH.real_lam(l, v, N, g, D, m)
    for lam in (400, 450, 470, 530, 550, 620, 650, 700):
        j = np.argmin(abs(lam_y - lam))
        if abs(lam_y[j] - lam) < 2:
            tb['m=%d %d' % (m, lam)] = dict(y_m=round(float(ys[j]), 4), px_y=round(px(ys[j]), 1))
out['textbook'] = tb
bal = {}
for m in (1, -1, 2, -2, 3, -3):
    for i, lam in enumerate(PH.BAL_LAMS):
        ang, valid, cl = PH.bal_lobe_angle(l, v, N, T, lam, D, m)
        ang = np.where(valid, ang, 9)
        j = int(np.argmin(ang))
        if ang[j] < 0.002:
            py = px(ys[j])
            yi = int(round(py))
            q = dict(y_m=round(float(ys[j]), 4), px_y=round(py, 1))
            if 'balanced' in imgs and 0 <= yi < H_:
                q['rendered_srgb_there'] = [int(x) for x in imgs['balanced'][max(0, yi - 4):yi + 5, C0:C1].reshape(-1, 3).mean(0)]
            bal['m=%+d %d' % (m, lam)] = q
out['balanced_lobes'] = bal
# rendered colour at the textbook positions in OURS
for k, q in tb.items():
    yi = int(round(q['px_y']))
    if 'ours' in imgs and 0 <= yi < H_:
        q['ours_rendered_srgb_there'] = [int(x) for x in imgs['ours'][max(0, yi - 4):yi + 5, C0:C1].reshape(-1, 3).mean(0)]
json.dump(out, open(os.path.join(RDIR, 'rainbow_order.json'), 'w'), indent=1)
print('mirror', out['mirror'])
for k, q in tb.items():
    print('textbook', k, q)
for k, q in bal.items():
    print('balanced', k, q)
