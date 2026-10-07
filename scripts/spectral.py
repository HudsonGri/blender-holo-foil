"""
Spectral data for the v2 Diffraction Grating BSDF (numpy only, so it also runs inside Blender).

c(lam)  RGB (scene-linear Rec.709, D65 white) of a D65 white light diffracted at wavelength lam:
        CIE 1931 2-degree CMFs x D65 SPD -> XYZ -> linear Rec.709, negative channels clipped (no negative
        lobes, so overlapping orders never subtract), then white-balanced per channel so that the
        average over 380..780 nm is exactly (1, 1, 1).  Hue-preserving alternatives (desaturate toward the
        equal-luminance grey in xy or in Oklab) were tested: after the white balance they turn deep reds
        pink (docs/HOW_IT_WORKS.md).
p(lam)  wavelength pdf proportional to R+G+B of c(lam) (+10 % floor), so every sample's colour weight w = c/p
        has R+G+B <= 3.3 (bounded fireflies, ~1.7x lower spike variance than uniform sampling).
Tables  u -> lam(u) and u -> w(u)/3 are baked into a Float Curve and an RGB Curves node.
Bessel  J_m(x)^2 lookup image for the physical (sinusoidal groove) efficiency model.

CIE tables: Cycles' own 5 nm data (intern/cycles/kernel/tables.h, CIE public data).
"""
import math
import numpy as np

LAM_MIN, LAM_MAX = 380.0, 780.0
LUM = np.array([0.2126, 0.7152, 0.0722])          # Rec.709 luminance
M_XYZ_709 = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]])
CMF = [
    (0.0014, 0.0000, 0.0065),
    (0.0022, 0.0001, 0.0105),
    (0.0042, 0.0001, 0.0201),
    (0.0076, 0.0002, 0.0362),
    (0.0143, 0.0004, 0.0679),
    (0.0232, 0.0006, 0.1102),
    (0.0435, 0.0012, 0.2074),
    (0.0776, 0.0022, 0.3713),
    (0.1344, 0.0040, 0.6456),
    (0.2148, 0.0073, 1.0391),
    (0.2839, 0.0116, 1.3856),
    (0.3285, 0.0168, 1.6230),
    (0.3483, 0.0230, 1.7471),
    (0.3481, 0.0298, 1.7826),
    (0.3362, 0.0380, 1.7721),
    (0.3187, 0.0480, 1.7441),
    (0.2908, 0.0600, 1.6692),
    (0.2511, 0.0739, 1.5281),
    (0.1954, 0.0910, 1.2876),
    (0.1421, 0.1126, 1.0419),
    (0.0956, 0.1390, 0.8130),
    (0.0580, 0.1693, 0.6162),
    (0.0320, 0.2080, 0.4652),
    (0.0147, 0.2586, 0.3533),
    (0.0049, 0.3230, 0.2720),
    (0.0024, 0.4073, 0.2123),
    (0.0093, 0.5030, 0.1582),
    (0.0291, 0.6082, 0.1117),
    (0.0633, 0.7100, 0.0782),
    (0.1096, 0.7932, 0.0573),
    (0.1655, 0.8620, 0.0422),
    (0.2257, 0.9149, 0.0298),
    (0.2904, 0.9540, 0.0203),
    (0.3597, 0.9803, 0.0134),
    (0.4334, 0.9950, 0.0087),
    (0.5121, 1.0000, 0.0057),
    (0.5945, 0.9950, 0.0039),
    (0.6784, 0.9786, 0.0027),
    (0.7621, 0.9520, 0.0021),
    (0.8425, 0.9154, 0.0018),
    (0.9163, 0.8700, 0.0017),
    (0.9786, 0.8163, 0.0014),
    (1.0263, 0.7570, 0.0011),
    (1.0567, 0.6949, 0.0010),
    (1.0622, 0.6310, 0.0008),
    (1.0456, 0.5668, 0.0006),
    (1.0026, 0.5030, 0.0003),
    (0.9384, 0.4412, 0.0002),
    (0.8544, 0.3810, 0.0002),
    (0.7514, 0.3210, 0.0001),
    (0.6424, 0.2650, 0.0000),
    (0.5419, 0.2170, 0.0000),
    (0.4479, 0.1750, 0.0000),
    (0.3608, 0.1382, 0.0000),
    (0.2835, 0.1070, 0.0000),
    (0.2187, 0.0816, 0.0000),
    (0.1649, 0.0610, 0.0000),
    (0.1212, 0.0446, 0.0000),
    (0.0874, 0.0320, 0.0000),
    (0.0636, 0.0232, 0.0000),
    (0.0468, 0.0170, 0.0000),
    (0.0329, 0.0119, 0.0000),
    (0.0227, 0.0082, 0.0000),
    (0.0158, 0.0057, 0.0000),
    (0.0114, 0.0041, 0.0000),
    (0.0081, 0.0029, 0.0000),
    (0.0058, 0.0021, 0.0000),
    (0.0041, 0.0015, 0.0000),
    (0.0029, 0.0010, 0.0000),
    (0.0020, 0.0007, 0.0000),
    (0.0014, 0.0005, 0.0000),
    (0.0010, 0.0004, 0.0000),
    (0.0007, 0.0002, 0.0000),
    (0.0005, 0.0002, 0.0000),
    (0.0003, 0.0001, 0.0000),
    (0.0002, 0.0001, 0.0000),
    (0.0002, 0.0001, 0.0000),
    (0.0001, 0.0000, 0.0000),
    (0.0001, 0.0000, 0.0000),
    (0.0001, 0.0000, 0.0000),
    (0.0000, 0.0000, 0.0000),
]
D65 = [
    49.9755, 52.3118, 54.6482, 68.7015, 82.7549, 87.1204, 91.4860, 92.4589, 93.4318,
    90.0570, 86.6823, 95.7736, 104.8650, 110.9360, 117.0080, 117.4100, 117.8120, 116.3360,
    114.8610, 115.3920, 115.9230, 112.3670, 108.8110, 109.0820, 109.3540, 108.5780, 107.8020,
    106.2960, 104.7900, 106.2390, 107.6890, 106.0470, 104.4050, 104.2250, 104.0460, 102.0230,
    100.0000, 98.1671, 96.3342, 96.0611, 95.7880, 92.2368, 88.6856, 89.3459, 90.0062,
    89.8026, 89.5991, 88.6489, 87.6987, 85.4936, 83.2886, 83.4939, 83.6992, 81.8630,
    80.0268, 80.1207, 80.2146, 81.2462, 82.2778, 80.2810, 78.2842, 74.0027, 69.7213,
    70.6652, 71.6091, 72.9790, 74.3490, 67.9765, 61.6040, 65.7448, 69.8856, 72.4863,
    75.0870, 69.3398, 63.5927, 55.0054, 46.4182, 56.6118, 66.8054, 65.0941, 63.3828,
]
CMF = np.array(CMF)
D65 = np.array(D65)
LAM5 = LAM_MIN + 5.0 * np.arange(len(CMF))


def raw_rgb(lam):
    """Linear Rec.709 of D65-weighted monochromatic light (unclipped, arbitrary scale)."""
    lam = np.asarray(lam, float)
    xyz = np.stack([np.interp(lam, LAM5, CMF[:, i]) for i in range(3)], -1)
    xyz = xyz * np.interp(lam, LAM5, D65)[..., None]
    return xyz @ M_XYZ_709.T


_FINE = np.linspace(LAM_MIN, LAM_MAX, 16001)
_K = None


def colour(lam):
    """c(lam): clipped, white-balanced (mean over [380, 780] nm is exactly (1,1,1))."""
    global _K
    if _K is None:
        _K = 1.0 / np.maximum(raw_rgb(_FINE), 0).mean(0)
    return np.maximum(raw_rgb(lam), 0) * _K


PDF_FLOOR = 0.1     # pdf floor (fraction of the mean): keeps the baked u -> lambda table fine at the spectrum ends


PDF_MODE = 'sum'


def pdf(lam):
    """Wavelength pdf (density w.r.t. the uniform measure on [380,780], i.e. mean 1): R+G+B of c plus a 10 % floor.
    Without the floor the 256-knot inverse-CDF table has a 105 nm wide last cell (675-780 nm) that smears deep red
    out to 780 nm (a +35 % red bias in the faint rainbow ends); with it the widest cell is 17 nm and E[w^2] grows 1-3 %."""
    if PDF_MODE == 'uniform':
        return np.ones_like(np.asarray(lam, float))
    c, c0 = colour(lam), colour(_FINE)
    if PDF_MODE == 'lumsum':        # half luminance, half R+G+B (less luminance noise, a bit more chroma noise)
        s = 0.5 * (c @ LUM) / (c0 @ LUM).mean() + 0.5 * c.sum(-1) / c0.sum(-1).mean()
        s0 = 1.0
    else:
        s = c.sum(-1)
        s0 = c0.sum(-1).mean()
    return (s + PDF_FLOOR * s0) / ((1 + PDF_FLOOR) * s0)


def inverse_cdf(u):
    p = pdf(_FINE)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (p[1:] + p[:-1]))])
    cdf /= cdf[-1]
    return np.interp(u, cdf, _FINE)


def curve_tables(n=256):
    """Knots u_i = i/(n-1): normalised lambda (lam-380)/400 and the colour weight w/3 at the knots."""
    u = np.linspace(0.0, 1.0, n)
    lam = inverse_cdf(u)
    return u, (lam - LAM_MIN) / (LAM_MAX - LAM_MIN), weights_for(lam)


def weights_for(lam_knots):
    """Given the knot wavelengths that the renderer will interpolate linearly in u, return knot weights w_i
    (RGB, /3 so they fit a curve's 0..1 range) so that E_u[w(u) f(lam(u))] = mean_lam[c(lam) f(lam)].
    Each cell's integral of c over [lam_i, lam_i+1] is matched by the trapezoid of the knot weights
    (solved by a sweep), then the result is rescaled so E_u[w] = (1,1,1) exactly."""
    lam_knots = np.asarray(lam_knots, float)
    n = len(lam_knots)
    h = 1.0 / (n - 1)
    # target cell integrals  int_cell c(lam) dlam / 400  (fine trapezoid)
    tgt = np.zeros((n - 1, 3))
    for i in range(n - 1):
        x = np.linspace(lam_knots[i], lam_knots[i + 1], 33)
        cx = colour(x)
        tgt[i] = np.trapezoid(cx, x, axis=0) / (LAM_MAX - LAM_MIN)
    # least-squares fit of knot values w_i s.t. (w_i + w_{i+1})/2 * h ~= tgt_i, with a small smoothness term
    A = np.zeros((2 * (n - 1) + 1, n))
    for i in range(n - 1):
        A[i, i] = A[i, i + 1] = 0.5 * h
    lam_s = 2e-3 * h
    for i in range(n - 2):
        A[n - 1 + i, i], A[n - 1 + i, i + 1], A[n - 1 + i, i + 2] = lam_s, -2 * lam_s, lam_s
    w = np.zeros((n, 3))
    for ch in range(3):
        b = np.concatenate([tgt[:, ch], np.zeros(n)])
        w[:, ch] = np.linalg.lstsq(A, b, rcond=None)[0]
    w = np.maximum(w, 0.0)
    mean = (0.5 * (w[1:] + w[:-1])).mean(0)
    w /= mean
    return w / 3.0


def cumulative_colour(n=256):
    """Knots u_i = i/(n-1) over lam = 380 + 400 u: C(lam) = fraction of each channel's white energy below lam
    (0 at 380 nm, 1 at 780 nm). Used for the closed-form share of a diffraction order that propagates."""
    u = np.linspace(0.0, 1.0, n)
    lam = LAM_MIN + (LAM_MAX - LAM_MIN) * u
    c = colour(_FINE)
    cum = np.concatenate([[np.zeros(3)], np.cumsum(0.5 * (c[1:] + c[:-1]), 0)])
    cum /= cum[-1]
    return u, np.stack([np.interp(lam, _FINE, cum[:, i]) for i in range(3)], 1)


def order0_quadrature(nq=6):
    """Fixed wavelengths + RGB weights to integrate eta_0(lam) * c(lam) for the physical order-0 colour.
    Bins of equal R+G+B mass; node = c-weighted centroid; weight = bin integral of c (sums to white)."""
    c = colour(_FINE)
    s = c.sum(1)
    cdf = np.cumsum(s) / s.sum()
    lam_q, w_q = [], []
    for j in range(nq):
        m = (cdf >= j / nq) & (cdf < (j + 1) / nq + (1e-9 if j == nq - 1 else 0))
        lam_q.append(float((_FINE[m] * s[m]).sum() / s[m].sum()))
        w_q.append(c[m].sum(0) / c.sum(0))
    return lam_q, np.array(w_q)


def gt_wavelengths(n=48):
    """Midpoint-rule wavelengths for the brute-force ground truth and their colour weights (sum = white)."""
    lam = LAM_MIN + (np.arange(n) + 0.5) * (LAM_MAX - LAM_MIN) / n
    # integrate c over each bin exactly (fine) so the set sums to white
    w = []
    for j in range(n):
        x = np.linspace(LAM_MIN + j * (LAM_MAX - LAM_MIN) / n, LAM_MIN + (j + 1) * (LAM_MAX - LAM_MIN) / n, 41)
        w.append(np.trapezoid(colour(x), x, axis=0) / (LAM_MAX - LAM_MIN))
    return lam, np.array(w)


def bessel_j(m, x, n=4096):
    """J_m(x) by the Bessel integral (1/pi) int_0^pi cos(m t - x sin t) dt (spectrally accurate)."""
    t = (np.arange(n) + 0.5) * math.pi / n
    x = np.asarray(x, float)
    return np.cos(m * t[None, :] - x[:, None] * np.sin(t)[None, :]).mean(1)


BESSEL_W, BESSEL_H, BESSEL_AMAX = 1024, 16, 24.0


def bessel_lut():
    """(H, W) array: row m = J_m(a)^2 for a = col * AMAX/(W-1)."""
    a = np.arange(BESSEL_W) * BESSEL_AMAX / (BESSEL_W - 1)
    return np.stack([bessel_j(m, a) ** 2 for m in range(BESSEL_H)])


if __name__ == '__main__':
    u, ln, w = curve_tables()
    print('lam knots', (LAM_MIN + 400 * ln[[0, 1, 2, 128, 253, 254, 255]]).round(1))
    print('w*3 at u=.5', (3 * w[128]).round(3), 'sum', (3 * w[128]).sum().round(3), 'max', (3 * w).max().round(3))
    lq, wq = order0_quadrature()
    print('order0 quadrature', np.round(lq, 1), wq.sum(0).round(4))
    L = bessel_lut()
    print('sum_m J_m^2 (m=-15..15) at a=0,3,10:', [(L[0, i] + 2 * L[1:, i].sum()).round(6) for i in (0, 128, 427)])
