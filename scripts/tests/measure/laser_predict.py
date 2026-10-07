"""Where a real 1.6 um grating and Balanced's lobe law put each colour on the laser-experiment screen
(normal incidence, screen parallel to the CD at D). Exact for the beam centre (z = 0). No renderer.
(ulimit -v 8000000; python3 scripts/tests/measure/laser_predict.py [OUT_DIR]) -> OUT_DIR/laser_predict.json"""
import sys, os, json, math
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import physics as PH  # noqa
from laser_geom import D, PITCH  # noqa
N = np.array([0, 1.0, 0]); T = np.array([0, 0, 1.0]); g = np.array([1.0, 0, 0])
xs = np.linspace(-0.47, 0.47, 94001)
P = np.stack([xs, np.full_like(xs, D), 0 * xs], -1)
v = PH.norm(P); l = np.broadcast_to(np.array([0, 1.0, 0]), P.shape)
Nn = np.broadcast_to(N, P.shape); Tn = np.broadcast_to(T, P.shape)
out = dict(real={}, balanced={})
for m in (1, 2, 3):
    for lam in (450.0, 470.0, 490.0, 530.0, 532.0, 570.0, 620.0, 650.0):
        s = m * lam / PITCH
        if s < 1:
            out['real']['m=%d %d' % (m, lam)] = round(D * math.tan(math.asin(s)) * 100, 2)
for mm in (1, -1, 2, -2, 3, -3):
    for lam in PH.BAL_LAMS:
        ang, valid, cl = PH.bal_lobe_angle(l, v, Nn, Tn, lam, PITCH, mm)
        ang = np.where(valid, ang, 9.0)
        # local minima below 2 mrad
        idx = [i for i in range(1, len(xs) - 1) if ang[i] < 0.002 and ang[i] <= ang[i - 1] and ang[i] <= ang[i + 1]]
        for i in idx:
            out['balanced'].setdefault('m=%+d %d' % (mm, lam), []).append(round(xs[i] * 100, 2))
print(json.dumps(out, indent=1))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'build', 'tests', 'measure')
os.makedirs(OUT, exist_ok=True)
json.dump(out, open(os.path.join(OUT, 'laser_predict.json'), 'w'), indent=1)
