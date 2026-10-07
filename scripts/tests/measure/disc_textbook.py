"""
Textbook prediction images for the CD / DVD / Blu-ray lineup (no renderer): for every pixel that sees the disc's
data area, where would a real reflection grating of that track pitch send light from the lamp into the camera?
  l_t + v_t = (m lam / d) g      (g = radial unit vector, tracks are circles)
* exact scene geometry from the .json written by scripts/tests/render.py (camera, lamp, disc), same resolution;
* the 3 mm lamp is integrated with 37 points; each point contributes where the groove-direction mismatch is within
  a Gaussian of sigma 0.006 (the lobe width of the CD roughness, alpha .0036, in direction cosines);
* orders +-1..+-3 (what both shaders simulate), weight 0.457^(|m|-1) (the test's falloff), colour = CIE 1931
  x D65 -> linear Rec.709 of lam (scripts/spectral.py, identical to a spectrometer-style rainbow), schematic
  brightness;
* it also reports, per disc, the range of |sin ti + sin to| over the data area (the textbook condition for any
  first order: lam/d <= that value).
(ulimit -v 8000000; python3 scripts/tests/measure/disc_textbook.py RENDER_DIR) -> RENDER_DIR/textbook_*.png, disc_textbook.json
"""
import json, math, os, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import physics as PH  # noqa

DISCS = (('CD', 1600.0, ''), ('DVD', 740.0, '_dvd'), ('Blu-ray', 320.0, '_bd'))
R0, R1 = 0.022, 0.058
SIG = 0.006
FALL = 0.4574


def lamp_points(c, size, n_ring=(1, 6, 12, 18)):
    c = np.asarray(c, float)
    nrm = PH.norm(-c)                                   # the lamp faces the disc centre
    a = PH.norm(np.cross(nrm, [0, 0, 1.0]) if abs(nrm[2]) < 0.99 else np.cross(nrm, [1.0, 0, 0]))
    b = np.cross(nrm, a)
    pts = []
    rings = len(n_ring)
    for k, n in enumerate(n_ring):
        r = size / 2 * (k / (rings - 1))
        for j in range(n):
            t = 2 * math.pi * j / n
            pts.append(c + r * (math.cos(t) * a + math.sin(t) * b))
    return np.array(pts)


def predict(meta):
    cam_m, lamp_m = meta['camera'], meta['lamp']
    W, H = cam_m['res']
    cam = PH.PinholeCam(cam_m['loc'], cam_m['target'], cam_m['lens_mm'], (W, H), sensor=cam_m.get('sensor_mm', 36.0))
    rays = cam.rays()
    zc = meta.get('disc_centre', [0, 0, 0.0012])[2]
    t = (zc - cam.loc[2]) / rays[..., 2]
    P = cam.loc + t[..., None] * rays
    r = np.hypot(P[..., 0], P[..., 1])
    on = (t > 0) & (r > R0) & (r < R1)
    disc_body = (t > 0) & (r > 0.0075) & (r < 0.060)
    Pq = P[on]
    N = np.array([0, 0, 1.0])
    g = PH.norm(np.concatenate([Pq[:, :2], np.zeros((len(Pq), 1))], -1))
    T = np.cross(N, g)
    v = PH.norm(cam.loc - Pq)
    lps = lamp_points(lamp_m['loc'], lamp_m['size_m'])
    out = {}
    imgs = {}
    for name, d, suf in DISCS:
        acc = np.zeros((len(Pq), 3))
        smin, smax = 9.0, 0.0
        rng = {}
        for lp in lps:
            l = PH.norm(lp - Pq)
            s = l + v
            s = s - (s @ N)[:, None] * N
            sg, sT = np.sum(s * g, 1), np.sum(s * T, 1)
            sm = np.linalg.norm(s, axis=1)
            smin, smax = min(smin, float(sm.min())), max(smax, float(sm.max()))
            w0 = np.exp(-0.5 * (sT / SIG) ** 2)
            for m in (1, -1, 2, -2, 3, -3):
                lam = d * sg / m
                ok = (lam >= 380) & (lam <= 780) & (w0 > 0.05)
                if ok.any():
                    lo, hi = float(lam[ok].min()), float(lam[ok].max())
                    q = rng.setdefault('m=%+d' % m, [lo, hi])
                    q[0], q[1] = min(q[0], lo), max(q[1], hi)
                acc += (w0 * FALL ** (abs(m) - 1))[:, None] * PH.c_ours(lam) / len(lps)
        img = np.zeros((H, W, 3))
        img[disc_body] = 0.01
        img[on] += acc
        imgs[name] = img
        need = 380.0 / d
        out[name] = dict(pitch_nm=d, first_order_needs_s_at_least=round(need, 3),
                         s_range_on_data_area=[round(smin, 3), round(smax, 3)],
                         first_order_possible=bool(smax >= need),
                         visible_orders_nm={k: [int(round(a)), int(round(b))] for k, (a, b) in sorted(rng.items())})
    return imgs, out


def tonemap(img, k):
    x = img * k
    x = x / (1 + x)
    return (np.clip(x, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8)


if __name__ == '__main__':
    d = sys.argv[1]
    res = {}
    for view in ('std', 'graze'):
        jf = os.path.join(d, 'p2_discs_%s_ours.json' % view)
        if not os.path.exists(jf):
            continue
        meta = json.load(open(jf))['proof_meta']
        imgs, out = predict(meta)
        res[view] = out
        k = 1.0 / max(np.percentile(np.concatenate([i.reshape(-1, 3) for i in imgs.values()]).max(1), 99.95), 1e-6)
        for name, suf in (('CD', ''), ('DVD', '_dvd'), ('Blu-ray', '_bd')):
            Image.fromarray(tonemap(imgs[name], 2 * k)).save(os.path.join(d, 'p2_discs_%s_textbook%s.png' % (view, suf)))
        print(view, json.dumps(out, indent=1))
    json.dump(res, open(os.path.join(d, 'disc_textbook.json'), 'w'), indent=1)
