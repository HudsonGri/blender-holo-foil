"""Sphere luminance vs view angle in the white-room renders (theta from the sphere's apparent centre, orthographic
approximation sin(theta) = r / R; the camera is 7.2 m away, so the error is < 1 deg).
(ulimit -v 8000000; python3 scripts/tests/measure/furnace_rim.py RENDER_DIR) -> RENDER_DIR/furnace_rim.json"""
import sys, os, json
import numpy as np
d = sys.argv[1]
idx = None
for f in sorted(os.listdir(d)):
    if f.startswith('p1_furnace') and f.endswith('.npz'):
        z0 = np.load(os.path.join(d, f))
        if 'index' in z0:
            idx = z0['index'].astype(np.float32)
            break
if idx is None:
    sys.exit('no p1_furnace_*.npz with an object-index pass in ' + d + ' (render with --index 1)')
m = np.abs(idx - 3) < 0.25
ys, xs = np.nonzero(m)
cy, cx = ys.mean(), xs.mean()
R = np.sqrt(m.sum() / np.pi)
rr = np.hypot(np.arange(idx.shape[0])[:, None] - cy, np.arange(idx.shape[1])[None, :] - cx)
th = np.degrees(np.arcsin(np.clip(rr / R, 0, 1)))
bands = [(0, 30), (30, 60), (60, 75), (75, 85), (85, 90)]
out = {}
for v in ('balanced', 'ours', 'ours_strict', 'ours_al'):
    p = os.path.join(d, 'p1_furnace_%s.npz' % v)
    if not os.path.exists(p):
        continue
    z = np.load(p)
    img = (z['noisy'] if 'noisy' in z else z['rgb']).astype(np.float32)
    row = {}
    for a, b in bands:
        sel = m & (th >= a) & (th < b) & (rr < R - 1.5)
        px = img[sel]
        row['%d-%d' % (a, b)] = dict(rgb=[round(float(x), 3) for x in px.mean(0)], lum=round(float(px.mean(0) @ [.2126, .7152, .0722]), 3), n=int(sel.sum()))
    out[v] = row
    print(v, {k: (x['lum'], x['rgb']) for k, x in row.items()})
json.dump(out, open(os.path.join(d, 'furnace_rim.json'), 'w'), indent=1)
