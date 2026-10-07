"""
"Run the experiment backwards": measure where each colour lands on the laser-experiment screen and compute the
track pitch a student would infer, d = m * lam / sin(theta).

* Profile: the un-denoised render ('noisy', unbiased), rows within +-5 mm of the beam height, averaged; paper
  background (10th percentile per channel) subtracted; 2 mm Gaussian smoothing along x.
* Pixel -> screen x is exactly linear (camera perpendicular to the screen, its half-width = SCREEN_W/2).
  theta = atan(|x| / D).
* Spots, per colour channel (as if each channel were one laser colour): local maxima beyond the beam spot
  (|x| > 3 cm), >= 6 % of that channel's maximum on that side, where that channel dominates the others (>= 0.5x);
  maxima closer than 2.5 cm are one spot. Like a student, the k-th spot out from the beam is order m = k.
* In each spectrum the peak of each channel (parabolic refinement) is assigned the wavelength at which THAT
  shader's own colour for the channel peaks: ours (CIE x D65 -> Rec.709, scripts/spectral.py) B 450, G 535,
  R 605 nm; Balanced (its Wavelength-node lines, weighted by its lobe weights) B 470, G 530, R 620 nm.
(ulimit -v 8000000; python3 scripts/tests/measure/laser_measure.py DIR) -> DIR/laser_measure.json
"""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
from laser_geom import D, PITCH, SCREEN_W, SCREEN_H, textbook_marks  # noqa

LAM_CH = {'ours': {'R': 605.0, 'G': 535.0, 'B': 450.0}, 'balanced': {'R': 620.0, 'G': 530.0, 'B': 470.0}}


def gauss_smooth(p, sigma_px):
    k = np.arange(-int(4 * sigma_px), int(4 * sigma_px) + 1)
    w = np.exp(-0.5 * (k / sigma_px) ** 2)
    w /= w.sum()
    return np.stack([np.convolve(p[:, c], w, mode='same') for c in range(p.shape[1])], -1)


GEOM = dict(D=D, PITCH=PITCH, SCREEN_W=SCREEN_W)


def profile(npz, key='noisy', half_mm=5.0, geom=None):
    SCREEN_W = (geom or GEOM)['SCREEN_W']
    z = np.load(npz)
    img = z[key if key in z else 'rgb'].astype(np.float32)
    H, W, _ = img.shape
    px_per_m = W / SCREEN_W
    r = int(round(half_mm / 1000 * px_per_m))
    band = img[H // 2 - r:H // 2 + r].mean(0)
    band = band - np.percentile(band, 10, axis=0)
    x = (np.arange(W) + 0.5 - W / 2) / (W / 2) * (SCREEN_W / 2)
    return x, gauss_smooth(band, 0.002 * px_per_m), px_per_m


def refine(x, y, i):
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den < 0:
            return x[i] + 0.5 * (a - c) / den * (x[1] - x[0])
    return x[i]


def find_spots(x, p, shader, side, x_min=0.030, thr=0.06, dominance=0.5, cluster=0.025, abs_min=0.008):
    """Per channel: local maxima beyond the beam spot, >= thr x that channel's max on this side, where the channel
    dominates (>= dominance x each other channel), clustered within 2.5 cm, sorted outward."""
    sel = (x > x_min) if side == '+' else (x < -x_min)
    out = {}
    for c, ch in enumerate('RGB'):
        pc = p[:, c]
        mx = pc[sel].max()
        cand = []
        for i in np.nonzero(sel)[0]:
            if 0 < i < len(x) - 1 and pc[i] >= pc[i - 1] and pc[i] > pc[i + 1] and pc[i] >= max(thr * mx, abs_min):
                others = [p[i, k] for k in range(3) if k != c]
                if pc[i] >= dominance * max(others):
                    cand.append((abs(refine(x, pc, i)), float(pc[i] / mx)))
        cand.sort()
        spots = []
        for xc, h in cand:
            if spots and xc - spots[-1][0] < cluster:
                if h > spots[-1][1]:
                    spots[-1] = (xc, h)
            else:
                spots.append((xc, h))
        out[ch] = spots
    return out


def measure(npz, shader, key='noisy', geom=None):
    geom = geom or GEOM
    D, PITCH = geom['D'], geom['PITCH']
    x, p, ppm = profile(npz, key, geom=geom)
    res = {}
    for side in ('+', '-'):
        spots = find_spots(x, p, shader, side)
        res[side] = {}
        for ch, sp in spots.items():
            lam = LAM_CH[shader][ch]
            rows = []
            for k, (xp, h) in enumerate(sp[:3]):
                m = k + 1
                th = math.atan(xp / D)
                rows.append(dict(spot=m, x_cm=round(xp * 100, 2), rel_height=round(h, 3), theta_deg=round(math.degrees(th), 2),
                                 lam_nm=lam, d_nm=round(m * lam / math.sin(th), 1),
                                 textbook_x_cm=round(D * math.tan(math.asin(m * lam / PITCH)) * 100, 2)
                                 if m * lam / PITCH < 1 else None))
            res[side][ch] = rows
    d1 = [r[0]['d_nm'] for s in ('+', '-') for r in res[s].values() if r]
    d2 = [r[1]['d_nm'] for s in ('+', '-') for r in res[s].values() if len(r) > 1]
    res['measured_pitch_spot1_nm'] = (dict(mean=round(float(np.mean(d1)), 1), min=min(d1), max=max(d1), n=len(d1))
                                      if d1 else dict(mean=None, n=0, note='no diffracted spot found'))
    if d2:
        res['measured_pitch_spot2_nm'] = dict(mean=round(float(np.mean(d2)), 1), min=min(d2), max=max(d2), n=len(d2))
    return res, (x, p)


def main_discs(d):
    import laser_discs_geom as G
    out = {}
    for name, pitch, key in G.DISCS:
        geom = dict(D=G.D, PITCH=pitch, SCREEN_W=G.SCREEN_W)
        suf = {'cd': '', 'dvd': '_dvd', 'bd': '_bd'}[key]
        for shader in ('balanced', 'ours'):
            f = os.path.join(d, 'p3b_laser_discs_%s%s.npz' % (shader, suf))
            if not os.path.exists(f):
                continue
            r, _ = measure(f, shader, geom=geom)
            out['%s %s' % (name, shader)] = r
            print('==', name, shader, '| pitch from 1st spot', r['measured_pitch_spot1_nm'], '| 2nd', r.get('measured_pitch_spot2_nm'))
            for s_ in ('+', '-'):
                for ch, rows in r[s_].items():
                    print('  side %s %s:' % (s_, ch), ' | '.join('spot %d x=%.2f cm (h %.2f) -> d=%.0f nm [textbook x %s]' % (
                        q['spot'], q['x_cm'], q['rel_height'], q['d_nm'], q['textbook_x_cm']) for q in rows))
    out['textbook'] = {name: dict(pitch_nm=pitch, D_m=G.D, marks=G.textbook_marks(pitch)) for name, pitch, _ in G.DISCS}
    json.dump(out, open(os.path.join(d, 'laser_discs_measure.json'), 'w'), indent=1)


if __name__ == '__main__' and '--discs' in sys.argv:
    main_discs(sys.argv[1])
elif __name__ == '__main__':
    d = sys.argv[1]
    out = {}
    prof = {}
    for shader in ('balanced', 'ours'):
        if not os.path.exists(os.path.join(d, 'p3_laser_%s.npz' % shader)):
            continue
        r, pr = measure(os.path.join(d, 'p3_laser_%s.npz' % shader), shader)
        out[shader] = r
        prof[shader] = pr
        print('==', shader, '| pitch from 1st spot per colour', r['measured_pitch_spot1_nm'], '| from 2nd spot',
              r.get('measured_pitch_spot2_nm'))
        for s in ('+', '-'):
            for ch, rows in r[s].items():
                print('  side %s %s:' % (s, ch), ' | '.join('spot %d x=%.2f cm (h %.2f) -> d=%.0f nm [textbook x %s]' % (
                    q['spot'], q['x_cm'], q['rel_height'], q['d_nm'], q['textbook_x_cm']) for q in rows))
        rd, _ = measure(os.path.join(d, 'p3_laser_%s.npz' % shader), shader, key='rgb')
        print('   (same on the denoised image: spot1 %s, spot2 %s)' % (rd['measured_pitch_spot1_nm'], rd.get('measured_pitch_spot2_nm')))
        out[shader + '_denoised_check'] = {k: v for k, v in rd.items() if k.startswith('measured')}
    out['textbook'] = dict(pitch_nm=PITCH, D_m=D, marks=textbook_marks())
    json.dump(out, open(os.path.join(d, 'laser_measure.json'), 'w'), indent=1)
    np.savez_compressed(os.path.join(d, 'laser_profiles.npz'), **{k + '_x': v[0] for k, v in prof.items()},
                        **{k + '_p': v[1] for k, v in prof.items()})
