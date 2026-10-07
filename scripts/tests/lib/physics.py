"""
Textbook grating physics and an exact numpy model of Secrop's "Balanced" lobe law, for the proof renders.
numpy only (runs locally and inside Blender).

Conventions: unit vectors l (surface -> lamp) and v (surface -> camera), surface normal N, grating vector g
(unit, in the surface, perpendicular to the grooves), groove direction T = N x g.
Real reflection grating (Stam 1999; any optics text): the tangential parts obey
        l_t + v_t = (m * lam / d) * g          (lam, d in nm)
so along g: sin(theta_i) + sin(theta_o) = m lam / d  (signed in-plane angles), and the groove component of
l_t + v_t must vanish. At normal incidence this is the familiar  sin(theta_m) = m lam / d.

Balanced (Andrew's 3Solutions "Diffraction_Example", NormalSolver as wired):
        theta = acos(N . v)        (unsigned)
        s     = sin(theta) - |m| lam / d
        alpha = sign(m) * asin(s)  (Blender's ARCSINE clamps |s| > 1 to +-90 deg)
        N'    = N rotated about the Tangent input (= groove direction T) by alpha
and each of its 30 glossy lobes (m = +-1..3 x lam = 470/490/530/570/620 nm) peaks where the half vector
H = normalize(l + v) equals N'.
"""
import math
import numpy as np

BAL_LAMS = np.array([470.0, 490.0, 530.0, 570.0, 620.0])
# per-order lobe weights of Andrew's Balanced (read from its node tree), same for +m and -m
BAL_W = {1: np.array([.1296, .1296, .1296, .0912, .1200]),
         2: np.array([.0605, .0605, .0605, .0426, .0560]),
         3: np.array([.0259, .0259, .0259, .0182, .0240])}
# Cycles Wavelength-node colours of those lines (measured by rendering the node)
WL_NODE = np.array([[0.0, 0.014, 0.536], [0.0, 0.148, 0.182], [0.0, 0.574, 0.0],
                    [0.365, 0.43, 0.0], [0.896, 0.0, 0.0]])
BAL_COLOUR_NAMES = ['blue 470', 'cyan 490', 'green 530', 'yellow 570', 'red 620']


def norm(a):
    a = np.asarray(a, float)
    return a / np.linalg.norm(a, axis=-1, keepdims=True)


def dot(a, b):
    return np.sum(np.asarray(a) * np.asarray(b), axis=-1)


# ---------------------------------------------------------------------------------------------- real
def first_order_exists(d, lam, smax=2.0):
    """A first order can propagate only if |sin ti + sin to| = lam/d is reachable (<= 2, or <= smax)."""
    return lam / d <= smax


def normal_incidence_angle(m, lam, d):
    x = m * lam / d
    return math.degrees(math.asin(x)) if abs(x) <= 1 else None


def real_lam(l, v, N, g, d, m):
    """Wavelength (nm) that order m of a real grating sends from l into v, and the groove-direction mismatch
    (direction-cosine units; must be ~0 for the order to be seen)."""
    s = l + v
    s = s - dot(s, N)[..., None] * N
    T = np.cross(N, g)
    lam = d * dot(s, g) / m
    return lam, dot(s, T)


# ---------------------------------------------------------------------------------------------- Balanced
def bal_normal(N, T, v, lam, d, m):
    """Balanced's rotated normal (NormalSolver), and whether its asin argument was clamped."""
    c = np.clip(dot(N, v), -1.0, 1.0)
    s = np.sqrt(np.maximum(0.0, 1.0 - c * c)) - abs(m) * lam / d
    clamped = np.abs(s) > 1.0
    a = np.sign(m) * np.arcsin(np.clip(s, -1.0, 1.0))
    TxN = np.cross(T, N)
    Np = N * np.cos(a)[..., None] + TxN * np.sin(a)[..., None]
    return Np, clamped, a


def bal_lobe_angle(l, v, N, T, lam, d, m):
    """Angle (rad) between the half vector and Balanced's rotated normal (0 = lobe peak), validity mask."""
    Np, clamped, a = bal_normal(N, T, v, lam, d, m)
    H = norm(l + v)
    ang = np.arccos(np.clip(dot(H, Np), -1.0, 1.0))
    valid = (dot(Np, v) > 0) & (dot(Np, l) > 0)
    return ang, valid, clamped


# ---------------------------------------------------------------------------------------------- colour
def c_ours(lam):
    """Our colour of a D65 white light diffracted at lam (scripts/spectral.py)."""
    import os, sys
    here = os.path.dirname(os.path.abspath(__file__))
    sp = os.path.join(here, '..', '..')            # scripts/ (spectral.py)
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import spectral
    lam = np.asarray(lam, float)
    out = np.zeros(lam.shape + (3,))
    ok = (lam >= 380) & (lam <= 780)
    if ok.any():
        out[ok] = spectral.colour(lam[ok])
    return out


# ---------------------------------------------------------------------------------------------- camera
class PinholeCam:
    """Blender-equivalent perspective camera (sensor fit AUTO, 36 mm sensor, look-at with world-Z up)."""

    def __init__(self, loc, target, lens, res, sensor=36.0, ortho_scale=None):
        self.loc = np.array(loc, float)
        f = norm(np.array(target, float) - self.loc)
        r = norm(np.cross(f, [0, 0, 1.0]))
        u = np.cross(r, f)
        self.f, self.r, self.u = f, r, u
        self.lens, self.res, self.sensor = lens, res, sensor
        self.ortho_scale = ortho_scale

    def _k(self):
        W, H = self.res
        half = self.sensor / 2.0 / self.lens        # tan(half fov) of the larger side
        return (half, half * H / W) if W >= H else (half * W / H, half)

    def rays(self):
        W, H = self.res
        kx, ky = self._k()
        xs = ((np.arange(W) + 0.5) / W * 2 - 1) * kx
        ys = (1 - (np.arange(H) + 0.5) / H * 2) * ky
        X, Y = np.meshgrid(xs, ys)
        d = self.f + X[..., None] * self.r + Y[..., None] * self.u
        return norm(d)

    def project(self, P):
        """World points -> pixel coords (x right, y down), and depth."""
        P = np.asarray(P, float)
        q = P - self.loc
        z = dot(q, self.f)
        kx, ky = self._k()
        x = dot(q, self.r) / z / kx
        y = dot(q, self.u) / z / ky
        W, H = self.res
        return np.stack([(x + 1) / 2 * W, (1 - y) / 2 * H], -1), z


def lam_range_text(lo, hi):
    return '%d-%d nm' % (lo, hi)
