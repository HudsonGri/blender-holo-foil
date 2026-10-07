"""
Measure the white-furnace renders: mean linear RGB of every object (object-index masks, eroded 3 px so no
pixel touches the white background), from the raw (un-denoised) render.

(ulimit -v 8000000; python3 scripts/tests/measure/furnace_measure.py DIR)  -> DIR/furnace_measure.json
"""
import json, os, sys
import numpy as np

NAMES = {1: 'card', 2: 'pouch', 3: 'sphere', 4: 'control mirror', 5: 'control diffuse'}


def erode(m, k=3):
    out = m.copy()
    for _ in range(k):
        e = out.copy()
        e[1:] &= out[:-1]; e[:-1] &= out[1:]; e[:, 1:] &= out[:, :-1]; e[:, :-1] &= out[:, 1:]
        out = e
    return out


def measure(npz, index=None):
    z = np.load(npz)
    img = (z['noisy'] if 'noisy' in z else z['rgb']).astype(np.float32)
    idx = (z['index'] if 'index' in z else index).astype(np.float32)
    res = {}
    for k, nm in NAMES.items():
        m = erode(np.abs(idx - k) < 0.25)
        if m.sum() < 20:
            continue
        px = img[m]
        res[nm] = dict(mean_rgb=[round(float(x), 4) for x in px.mean(0)],
                       luminance=round(float(px.mean(0) @ [0.2126, 0.7152, 0.0722]), 4),
                       b_over_r=round(float(px.mean(0)[2] / max(px.mean(0)[0], 1e-9)), 3),
                       p5_p95_lum=[round(float(np.percentile(px @ [0.2126, 0.7152, 0.0722], q)), 3) for q in (5, 95)],
                       pixels=int(m.sum()))
    bg = erode(idx < 0.25, 6)
    res['background'] = dict(mean_rgb=[round(float(x), 4) for x in img[bg].mean(0)], pixels=int(bg.sum()))
    return res


if __name__ == '__main__':
    d = sys.argv[1]
    out = {}
    index = None
    files = sorted(f for f in os.listdir(d) if f.startswith('p1_furnace') and f.endswith('.npz'))
    for f in files:
        z = np.load(os.path.join(d, f))
        if 'index' in z:
            index = z['index']
    for f in files:
        out[f[:-4]] = measure(os.path.join(d, f), index)
        print(f[:-4])
        for k, v in out[f[:-4]].items():
            print('   %-16s %s' % (k, v))
    json.dump(out, open(os.path.join(d, 'furnace_measure.json'), 'w'), indent=1)
