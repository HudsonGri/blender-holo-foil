"""
Test 4 (general check): colour order and spread inside each diffraction order, real grating vs Balanced's lobe law.

A real grating (textbook): for a fixed view direction v, order m of wavelength lam is lit by a lamp whose
tangential direction cosine is l_t = m lam / d - v_t. Its distance from the mirror direction (l_t = -v_t) is
|m| lam / d, so inside every order BLUE sits nearest the mirror highlight and RED farthest, and order 2 is spread
exactly twice as wide as order 1 (in direction cosines).
Balanced: solve where its lobe (rotated normal N' = half vector) puts each of its 5 wavelengths for the same
view, and read off the same quantities. View in the dispersion plane, polar angle theta_v; d = 1459.08 nm
(Andrew's pouch value) and 1600 nm (CD).

(ulimit -v 8000000; python3 scripts/tests/measure/order_check.py [OUT_DIR]) -> OUT_DIR/order_check.json
"""
import json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import physics as PH  # noqa

N = np.array([0, 0, 1.0]); g = np.array([0, 1.0, 0]); T = np.cross(N, g)   # T = groove direction (x)


def balanced_lt(theta_v_deg, lam, d, m):
    """Tangential direction cosine (along g) of the lamp direction that lights Balanced's (m, lam) lobe peak for a
    view at theta_v in the dispersion plane, and whether it is a valid (above-horizon, unclamped) lobe."""
    t = math.radians(theta_v_deg)
    v = np.array([0.0, math.sin(t), math.cos(t)])
    Np, clamped, a = PH.bal_normal(N, T, v, lam, d, m)
    l = 2 * np.dot(v, Np) * Np - v
    ok = (l[2] > 0) and (np.dot(Np, v) > 0) and not clamped
    return float(l[1]), float(v[1]), bool(ok), bool(clamped)


def table(d):
    rows = []
    for th in range(0, 71, 5):
        row = dict(theta_v=th)
        for m in (1, -1, 2, -2):
            dist = []
            for lam in PH.BAL_LAMS:
                lt, vt, ok, cl = balanced_lt(th, lam, d, m)
                dist.append((lam, abs(lt + vt), ok, cl))
            valid = [x for x in dist if x[2]]
            row['m%+d' % m] = dict(dist_from_mirror={int(x[0]): round(x[1], 4) for x in dist},
                                   valid={int(x[0]): x[2] for x in dist})
            if len(valid) >= 2 and dist[0][2] and dist[-1][2]:
                row['m%+d' % m]['blue_nearest'] = bool(dist[0][1] < dist[-1][1])
                row['m%+d' % m]['spread_470_620'] = round(abs(dist[-1][1] - dist[0][1]), 4)
        for s in ('+', '-'):
            a, b = row.get('m%s1' % s, {}), row.get('m%s2' % s, {})
            if 'spread_470_620' in a and 'spread_470_620' in b and a['spread_470_620'] > 1e-6:
                row['ratio_m%s2_over_m%s1' % (s, s)] = round(b['spread_470_620'] / a['spread_470_620'], 3)
        rows.append(row)
    return rows


if __name__ == '__main__':
    out = {}
    for d in (1459.08, 1600.0):
        rows = table(d)
        out[str(d)] = rows
        print('\n=== d = %g nm. Real grating: blue nearest the mirror in every order; m=2 spread / m=1 spread = 2.000; '
              'real m=1 470->620 spread = %.4f' % (d, 150 / d))
        print('theta_v | m=+1 blue nearest? spread | m=-1 blue nearest? spread | ratio m2/m1 (+,-)')
        for r in rows:
            def f(k):
                x = r.get(k, {})
                if 'blue_nearest' not in x:
                    return '   (no valid pair)   '
                return '%-5s %.3f' % ('yes' if x['blue_nearest'] else 'NO', x['spread_470_620'])
            print('  %2d    | %-18s | %-18s | %s %s' % (r['theta_v'], f('m+1'), f('m-1'),
                                                       r.get('ratio_m+2_over_m+1', '-'), r.get('ratio_m-2_over_m-1', '-')))
    OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'build', 'tests', 'measure')
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, 'order_check.json'), 'w'), indent=1)
