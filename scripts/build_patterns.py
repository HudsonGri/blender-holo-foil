"""
Holo foil PATTERN library for the "Diffraction Grating BSDF" (stock Blender, pure SVM, GPU-safe, no OSL).

    blender -b --factory-startup --python scripts/build_patterns.py [-- OUT.blend]
    -> build/holo_patterns.blend   (scripts/build_holo_foil.py merges everything into holo_foil.blend)

Real holographic foils (trading cards, banknotes, packaging) are not one uniform grating: the
embossing shim is a mosaic of regions whose groove direction and pitch differ. Each pattern
below is a shader node group that describes such a mosaic. Every pattern group shares one
output contract (see docs/INPUTS.md, "Patterns"):

    Angle                 float, radians. Direction of the grating vector g (perpendicular to the
                          grooves), counter-clockwise about N from the UV tangent (U direction).
                          Feed it to the BSDF's "Grating Angle". Gratings are symmetric, so only
                          Angle mod pi matters.
    Tangent               the same direction as a world-space unit vector (UV tangent rotated by
                          Angle about N), for BSDFs that take a tangent instead of an angle.
                          Use EITHER Angle (with the UV tangent on the BSDF) OR Tangent (with
                          Angle = 0), not both.
    Pitch Multiplier      float, ~1. Plug into the BSDF's "Pitch Multiplier".
    Mask                  0..1, where the embossed grating is present (elsewhere: plain foil).
    Roughness Multiplier  float, ~1. Multiplies the BSDF's roughness.

Inputs: "Vector" (unconnected = UV), "Scale" and pattern-specific parameters. Feed Vector in a
space with physical (isotropic) proportions whose x axis follows U, e.g. UV * (1, 88/63) on a
63 x 88 mm card; otherwise angles are measured in a stretched space.

Also builds:
    "Holo Pattern: Layer"   stacks two patterns (top wins where its mask > 0.5)
    "Holo Card Shader"      printed card over foil + laminate, wraps the grating BSDF
    material "Holo Card"    template with procedurally generated demo art, foil mask and
                            white-ink map (packed into the .blend)

The grating BSDF is used from the current file if it is already loaded (build_holo_foil.py),
otherwise appended from $HOLO_BSDF_BLEND, build/diffraction_grating.blend or holo_foil.blend
(first that exists).
"""
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..'))
TEX = os.path.join(ROOT, 'build', 'textures')
OUT_BLEND = os.environ.get('HOLO_OUT') or os.path.join(ROOT, 'build', 'holo_patterns.blend')
if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 1:
    OUT_BLEND = os.path.abspath(sys.argv[sys.argv.index('--') + 1])
PI = math.pi

# ----------------------------------------------------------------------------------------------
# grating BSDF adapter: the only place that knows the BSDF's socket names
# ----------------------------------------------------------------------------------------------
BSDF_GROUP = "Diffraction Grating BSDF"
BSDF_OUT = 'BSDF'
_CANDIDATES = [os.environ.get('HOLO_BSDF_BLEND', ''), os.path.join(ROOT, 'build', 'diffraction_grating.blend'),
               os.path.join(ROOT, 'holo_foil.blend')]
BSDF_BLEND = next((p for p in _CANDIDATES if p and os.path.exists(p)), _CANDIDATES[1])
# logical name -> socket name on the BSDF group (see docs/INPUTS.md)
BSDF_SOCK = {
    'color': 'Color', 'pitch': 'Pitch (nm)', 'pitch_mult': 'Pitch Multiplier', 'angle': 'Grating Angle',
    'type': 'Grating Type', 'mask': 'Mask', 'orders': 'Orders', 'eta0': 'Order 0 Strength',
    'falloff': 'Efficiency Falloff', 'depth': 'Groove Depth (nm)', 'rough': 'Roughness',
    'sat': 'Spectral Saturation', 'coat': 'Coat', 'coat_rough': 'Coat Roughness', 'coat_ior': 'Coat IOR',
    'coat_tint': 'Coat Tint', 'seed': 'Seed', 'tangent': None, 'normal': 'Normal',
}


def load_bsdf():
    ng = bpy.data.node_groups.get(BSDF_GROUP)
    if ng:
        return ng
    with bpy.data.libraries.load(BSDF_BLEND, link=False) as (src, dst):
        dst.node_groups = [BSDF_GROUP]
    ng = bpy.data.node_groups[BSDF_GROUP]
    ng.use_fake_user = True
    return ng


# ----------------------------------------------------------------------------------------------
# tiny node DSL
# ----------------------------------------------------------------------------------------------
class NB:
    def __init__(self, tree):
        self.t = tree

    def node(self, idname, **props):
        n = self.t.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, src, dst):
        self.t.links.new(src, dst)

    def _in(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.link(val, sock)
        elif val is not None:
            if isinstance(val, (int, float)) and hasattr(sock, 'default_value') and \
                    hasattr(sock.default_value, '__len__'):
                n = len(sock.default_value)
                sock.default_value = [float(val)] * n if n != 4 else [float(val)] * 3 + [1.0]
            else:
                sock.default_value = val

    def math(self, op, a, b=None, c=None, clamp=False):
        n = self.node('ShaderNodeMath', operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    # shorthands
    def add(self, a, b): return self.math('ADD', a, b)
    def sub(self, a, b): return self.math('SUBTRACT', a, b)
    def mul(self, a, b): return self.math('MULTIPLY', a, b)
    def div(self, a, b): return self.math('DIVIDE', a, b)
    def madd(self, a, b, c): return self.math('MULTIPLY_ADD', a, b, c)   # a*b + c
    def gt(self, a, b): return self.math('GREATER_THAN', a, b)
    def lt(self, a, b): return self.math('LESS_THAN', a, b)
    def mx(self, a, b): return self.math('MAXIMUM', a, b)
    def mn(self, a, b): return self.math('MINIMUM', a, b)
    def absf(self, a): return self.math('ABSOLUTE', a)

    def vmath(self, op, a, b=None, c=None, scale=None):
        n = self.node('ShaderNodeVectorMath', operation=op)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        if scale is not None:
            self._in(n.inputs['Scale'], scale)
        return n.outputs['Value'] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs['Vector']

    def sep(self, v):
        n = self.node('ShaderNodeSeparateXYZ')
        self.link(v, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z=0.0):
        n = self.node('ShaderNodeCombineXYZ')
        self._in(n.inputs[0], x)
        self._in(n.inputs[1], y)
        self._in(n.inputs[2], z)
        return n.outputs[0]

    def _mix(self, dtype, fac, a, b):
        n = self.node('ShaderNodeMix', data_type=dtype, clamp_factor=True)
        ins = {s.identifier: s for s in n.inputs}
        suf = {'FLOAT': 'Float', 'VECTOR': 'Vector', 'RGBA': 'Color'}[dtype]
        self._in(ins['Factor_Float'], fac)
        self._in(ins['A_' + suf], a)
        self._in(ins['B_' + suf], b)
        return {s.identifier: s for s in n.outputs}['Result_' + suf]

    def mixf(self, fac, a, b): return self._mix('FLOAT', fac, a, b)
    def mixv(self, fac, a, b): return self._mix('VECTOR', fac, a, b)
    def mixc(self, fac, a, b): return self._mix('RGBA', fac, a, b)

    def smooth(self, x, e0, e1):
        """smoothstep(e0, e1, x) with e0 < e1 (0 below e0, 1 above e1)."""
        n = self.node('ShaderNodeMapRange', interpolation_type='SMOOTHSTEP', clamp=True)
        self._in(n.inputs['Value'], x)
        self._in(n.inputs['From Min'], e0)
        self._in(n.inputs['From Max'], e1)
        n.inputs['To Min'].default_value = 0.0
        n.inputs['To Max'].default_value = 1.0
        return n.outputs['Result']

    def voronoi(self, vec, feature='F1', rand=1.0, dims='2D', metric='EUCLIDEAN'):
        n = self.node('ShaderNodeTexVoronoi', voronoi_dimensions=dims, feature=feature,
                      distance=metric)
        self._in(n.inputs['Vector'], vec)
        n.inputs['Scale'].default_value = 1.0
        self._in(n.inputs['Randomness'], rand)
        return n

    def white(self, vec=None, w=None, dims='3D'):
        n = self.node('ShaderNodeTexWhiteNoise', noise_dimensions=dims)
        if vec is not None:
            self._in(n.inputs['Vector'], vec)
        if w is not None:
            self._in(n.inputs['W'], w)
        return n

    def noise(self, vec, scale, detail=2.0, rough=0.5):
        n = self.node('ShaderNodeTexNoise', noise_dimensions='3D')
        self._in(n.inputs['Vector'], vec)
        self._in(n.inputs['Scale'], scale)
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        return n.outputs['Fac']

    def group(self, ng, **inputs):
        n = self.node('ShaderNodeGroup')
        n.node_tree = ng
        for k, v in inputs.items():
            self._in(n.inputs[k.replace('_', ' ')], v)
        return n

    # helpers -------------------------------------------------------------------------------
    def rand_signed(self, r, amp):
        """(r - 0.5) * 2 * amp   for r in [0, 1)"""
        return self.mul(self.sub(r, 0.5), self.mul(amp, 2.0))

    def select(self, cond, a, b):
        """cond > 0.5 ? b : a   (hard switch, anti-aliased by pixel supersampling)"""
        return self.mixf(self.gt(cond, 0.5), a, b)

    def over(self, top, base):
        """1 where a soft-edged top mask covers more than what the base mask leaves: top > base (1 - top).
        Base fully embossed -> switch at top = 0.5; base plain foil (0) -> any top coverage wins, so the
        soft rim of a star/beam keeps the star's own grating instead of borrowing the base angle."""
        return self.gt(top, self.mul(base, self.sub(1.0, top)))


def layout(tree):
    """Left-to-right layout by longest path depth (keeps groups readable when opened)."""
    nodes = list(tree.nodes)
    preds = {n: [] for n in nodes}
    for l in tree.links:
        preds[l.to_node].append(l.from_node)
    depth = {}

    def d(n, stack=()):
        if n in depth:
            return depth[n]
        if n in stack:
            return 0
        depth[n] = 0 if not preds[n] else 1 + max(d(p, stack + (n,)) for p in preds[n])
        return depth[n]

    for n in nodes:
        d(n)
    cols = {}
    for n in nodes:
        if n.bl_idname == 'NodeGroupOutput':
            depth[n] = max(depth.values()) + 1
        cols.setdefault(depth[n], []).append(n)
    for k, ns in cols.items():
        for i, n in enumerate(ns):
            n.location = (k * 230.0, -i * 190.0)


# ----------------------------------------------------------------------------------------------
# group interface helpers
# ----------------------------------------------------------------------------------------------
STD_OUT = [
    ("Angle", 'NodeSocketFloat', 'ANGLE', "Grating direction about N (radians, from the UV tangent); -> BSDF Grating Angle"),
    ("Tangent", 'NodeSocketVector', 'NONE', "Same direction as a world-space tangent (UV tangent rotated by Angle about N)"),
    ("Pitch Multiplier", 'NodeSocketFloat', 'NONE', "Multiplies the BSDF pitch (~1)"),
    ("Mask", 'NodeSocketFloat', 'FACTOR', "1 where the embossed grating is present, 0 = plain foil"),
    ("Roughness Multiplier", 'NodeSocketFloat', 'NONE', "Multiplies the BSDF roughness (~1)"),
]


STD_OUT_N = STD_OUT + [
    ("Normal", 'NodeSocketVector', 'NONE', "Optional embossing: tilted world normal (zero-tilt = geometry "
     "normal). Plug into Holo Card Shader > Foil Normal (or the BSDF's Normal)"),
]


def new_group(name, inputs, outputs=STD_OUT, desc=""):
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old)
    ng = bpy.data.node_groups.new(name, 'ShaderNodeTree')
    ng.description = desc
    it = ng.interface
    for nm, typ, sub, d in outputs:
        s = it.new_socket(nm, in_out='OUTPUT', socket_type=typ)
        if sub != 'NONE':
            s.subtype = sub
        s.description = d
    for spec in inputs:
        nm, typ, default = spec[0], spec[1], spec[2]
        opt = spec[3] if len(spec) > 3 else {}
        s = it.new_socket(nm, in_out='INPUT', socket_type=typ)
        if opt.get('sub'):
            s.subtype = opt['sub']
        if default is not None:
            s.default_value = default
        if 'min' in opt:
            s.min_value = opt['min']
        if 'max' in opt:
            s.max_value = opt['max']
        s.hide_value = opt.get('hide', False)
        s.description = opt.get('desc', "")
    ng.use_fake_user = True
    b = NB(ng)
    gi = b.node('NodeGroupInput')
    go = b.node('NodeGroupOutput')
    return ng, b, gi, go


VEC_IN = ("Vector", 'NodeSocketVector', (0.0, 0.0, 0.0),
          dict(hide=True, desc="Pattern coordinates (unconnected = UV). Use physical proportions, x along U"))


def F(name, default, lo=None, hi=None, sub=None, desc=""):
    o = dict(desc=desc)
    if lo is not None:
        o['min'] = lo
    if hi is not None:
        o['max'] = hi
    if sub:
        o['sub'] = sub
    return (name, 'NodeSocketFloat', default, o)


SEED = F("Seed", 0.0, desc="Changes the random layout")

_helpers = {}


def helper_coords():
    """Vector (or UV if unconnected) * Scale + seed offset."""
    if 'coords' in _helpers:
        return _helpers['coords']
    ng, b, gi, go = new_group(
        "HoloPat Coords", [VEC_IN, F("Scale", 1.0), SEED],
        outputs=[("Vector", 'NodeSocketVector', 'NONE', "")],
        desc="internal: Vector (UV if unconnected) * Scale + seed offset")
    tc = b.node('ShaderNodeTexCoord')
    v = gi.outputs['Vector']
    use_in = b.gt(b.vmath('LENGTH', v), 1e-9)
    p = b.mixv(use_in, tc.outputs['UV'], v)
    p = b.vmath('SCALE', p, scale=gi.outputs['Scale'])
    seed = gi.outputs['Seed']
    off = b.comb(b.mul(seed, 17.31), b.mul(seed, 29.17), 0.0)
    b.link(b.vmath('ADD', p, off), go.inputs['Vector'])
    layout(ng)
    _helpers['coords'] = ng
    return ng


def helper_tangent():
    """Angle -> world tangent: UV tangent, Gram-Schmidt against N, rotated by Angle about N."""
    if 'tangent' in _helpers:
        return _helpers['tangent']
    ng, b, gi, go = new_group(
        "HoloPat Angle to Tangent", [F("Angle", 0.0, sub='ANGLE')],
        outputs=[("Tangent", 'NodeSocketVector', 'NONE', "")],
        desc="internal: UV tangent rotated by Angle about the shading normal")
    geo = b.node('ShaderNodeNewGeometry')
    tan = b.node('ShaderNodeTangent', direction_type='UV_MAP')
    N = geo.outputs['Normal']
    T = tan.outputs['Tangent']
    Tp = b.vmath('NORMALIZE', b.vmath('SUBTRACT', T, b.vmath('SCALE', N, scale=b.vmath('DOT_PRODUCT', T, N))))
    rot = b.node('ShaderNodeVectorRotate', rotation_type='AXIS_ANGLE')
    b.link(Tp, rot.inputs['Vector'])
    b.link(N, rot.inputs['Axis'])
    b.link(gi.outputs['Angle'], rot.inputs['Angle'])
    b.link(rot.outputs[0], go.inputs['Tangent'])
    layout(ng)
    _helpers['tangent'] = ng
    return ng


def helper_tilt():
    """Geometry normal rotated by Tilt about the in-plane axis at Axis Angle (from the UV tangent)."""
    if 'tilt' in _helpers:
        return _helpers['tilt']
    ng, b, gi, go = new_group(
        "HoloPat Tilt Normal", [F("Axis Angle", 0.0, sub='ANGLE'), F("Tilt", 0.0, sub='ANGLE')],
        outputs=[("Normal", 'NodeSocketVector', 'NONE', "")],
        desc="internal: geometry normal rotated by Tilt about an in-plane axis (embossed relief)")
    geo = b.node('ShaderNodeNewGeometry')
    ax = b.group(helper_tangent(), Angle=gi.outputs['Axis Angle'])
    rot = b.node('ShaderNodeVectorRotate', rotation_type='AXIS_ANGLE')
    b.link(geo.outputs['Normal'], rot.inputs['Vector'])
    b.link(ax.outputs['Tangent'], rot.inputs['Axis'])
    b.link(gi.outputs['Tilt'], rot.inputs['Angle'])
    b.link(rot.outputs[0], go.inputs['Normal'])
    layout(ng)
    _helpers['tilt'] = ng
    return ng


def finish(b, go, angle, pitch=1.0, mask=1.0, rough=1.0, normal=None):
    if normal is not None:
        b.link(normal, go.inputs['Normal'])
    t = b.group(helper_tangent(), Angle=angle)
    b._in(go.inputs['Angle'], angle)
    b.link(t.outputs['Tangent'], go.inputs['Tangent'])
    b._in(go.inputs['Pitch Multiplier'], pitch)
    b._in(go.inputs['Mask'], mask)
    b._in(go.inputs['Roughness Multiplier'], rough)
    layout(b.t)


def coords(b, gi, scale=None, seed=True):
    g = b.group(helper_coords())
    b.link(gi.outputs['Vector'], g.inputs['Vector'])
    b._in(g.inputs['Scale'], gi.outputs['Scale'] if scale is None else scale)
    if seed:
        b.link(gi.outputs['Seed'], g.inputs['Seed'])
    return g.outputs['Vector']


def rot2(b, x, y, ang):
    """rotate (x, y) by ang -> (x', y')"""
    c, s = b.math('COSINE', ang), b.math('SINE', ang)
    return b.sub(b.mul(x, c), b.mul(y, s)), b.add(b.mul(x, s), b.mul(y, c))


# ----------------------------------------------------------------------------------------------
# 1. Linear Rainbow
# ----------------------------------------------------------------------------------------------
def build_linear():
    ng, b, gi, go = new_group("Holo Pattern: Linear Rainbow", [
        VEC_IN, F("Scale", 1.0, 0.0),
        F("Angle", 0.0, sub='ANGLE', desc="Grating direction (0 = grooves along V, rainbow along U)"),
        F("Wobble", 0.0, 0.0, PI, sub='ANGLE', desc="Slow random angle variation (embossing-roller imperfections)"),
        F("Wobble Scale", 2.0, 0.0, desc="Size of the wobble noise (per Vector unit)"),
        F("Sweep Direction", 0.0, sub='ANGLE', desc="Direction along which the sweeps below vary"),
        F("Angle Sweep", 0.0, desc="Angle change per Vector unit along Sweep Direction (kinematic 'sheen' sweep)"),
        F("Pitch Sweep", 0.0, desc="Relative pitch change per Vector unit (chirped grating: colour band travels)"),
        SEED], desc="Uniform 1D grating (rainbow film, plain holo foil, XY sheen, SV mirage)")
    p = coords(b, gi)
    x, y, _ = b.sep(p)
    sd = gi.outputs['Sweep Direction']
    s = b.add(b.mul(x, b.math('COSINE', sd)), b.mul(y, b.math('SINE', sd)))
    n = b.noise(p, gi.outputs['Wobble Scale'], detail=1.0)
    ang = b.add(gi.outputs['Angle'], b.mul(b.sub(n, 0.5), b.mul(gi.outputs['Wobble'], 3.0)))
    ang = b.madd(gi.outputs['Angle Sweep'], s, ang)
    pitch = b.mx(b.madd(gi.outputs['Pitch Sweep'], s, 1.0), 0.2)
    finish(b, go, ang, pitch, 1.0, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# 2. Radial / CD
# ----------------------------------------------------------------------------------------------
def build_radial():
    ng, b, gi, go = new_group("Holo Pattern: Radial CD", [
        VEC_IN, F("Scale", 1.0, 0.0, desc="Multiplies the radius"),
        ("Center", 'NodeSocketVector', (0.5, 0.5, 0.0), dict(desc="Centre in Vector space")),
        F("Twist", 0.0, sub='ANGLE', desc="0 = concentric grooves (CD); 90 deg = radial grooves (sunburst / starburst foil)"),
        F("Spiral", 0.0, desc="Extra angle per unit radius (swirl / kinetic spiral foils)"),
        F("Inner Radius", 0.0, 0.0, desc="Mask hole radius (CD: 0.19 of the outer diameter... in scaled units)"),
        F("Outer Radius", 100.0, 0.0, desc="Mask outer radius (scaled units)"),
        F("Edge Softness", 0.004, 0.0),
        F("Pitch Ramp", 0.0, desc="Relative pitch change per unit radius"),
    ], desc="Concentric (CD, 1.6 um) or radial grooves around a centre")
    tc = coords(b, gi, scale=1.0, seed=False)
    q = b.vmath('SCALE', b.vmath('SUBTRACT', tc, gi.outputs['Center']), scale=gi.outputs['Scale'])
    qx, qy, _ = b.sep(q)
    r = b.vmath('LENGTH', b.comb(qx, qy, 0.0))
    phi = b.math('ARCTAN2', qy, qx)
    ang = b.add(b.add(phi, gi.outputs['Twist']), b.mul(gi.outputs['Spiral'], r))
    e = b.mx(gi.outputs['Edge Softness'], 1e-5)
    ri, ro = gi.outputs['Inner Radius'], gi.outputs['Outer Radius']
    m_in = b.smooth(r, b.sub(ri, e), b.add(ri, e))
    m_out = b.sub(1.0, b.smooth(r, b.sub(ro, e), b.add(ro, e)))
    mask = b.mul(m_in, m_out)
    pitch = b.mx(b.madd(gi.outputs['Pitch Ramp'], r, 1.0), 0.2)
    finish(b, go, ang, pitch, mask, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# 3. Scratched Metal
# ----------------------------------------------------------------------------------------------
def scratch_layer(b, gi, p, k):
    """One Voronoi layer: one straight scratch segment through each cell's feature point."""
    vor = b.voronoi(p, rand=1.0)
    c = vor.outputs['Position']
    r1, r2, r3 = b.sep(vor.outputs['Color'])
    wn = b.white(c, dims='3D')
    r4, r5, r6 = b.sep(wn.outputs['Color'])
    dx, dy, _ = b.sep(b.vmath('SUBTRACT', p, c))
    theta = b.add(gi.outputs['Angle'], b.rand_signed(r1, b.mul(gi.outputs['Angle Spread'], PI / 2)))
    along, across = rot2(b, dx, dy, b.mul(theta, -1.0))
    half_l = b.mul(b.mul(gi.outputs['Length'], 0.5), b.madd(r2, 0.6, 0.4))
    wid = b.mul(gi.outputs['Width'], b.madd(r3, 1.0, 0.5))
    soft = gi.outputs['Softness']
    m_ac = b.sub(1.0, b.smooth(b.absf(across), b.mul(wid, b.sub(1.0, soft)), wid))
    m_al = b.sub(1.0, b.smooth(b.absf(along), b.mul(half_l, 0.7), half_l))
    present = b.lt(r4, gi.outputs['Density'])
    m = b.mul(b.mul(m_ac, m_al), present)
    ang = b.add(theta, PI / 2)                      # grating vector is perpendicular to the scratch
    pitch = b.madd(b.rand_signed(r5, gi.outputs['Pitch Jitter']), 1.0, 1.0)
    return m, ang, pitch


def build_scratched():
    ng, b, gi, go = new_group("Holo Pattern: Scratched Metal", [
        VEC_IN, F("Scale", 30.0, 0.0, desc="Scratch cells per Vector unit"),
        F("Density", 0.85, 0.0, 1.0, sub='FACTOR', desc="Fraction of cells holding a scratch"),
        F("Length", 1.1, 0.0, desc="Scratch length in cell units (two layers overlap)"),
        F("Width", 0.05, 0.0, desc="Scratch half-width in cell units"),
        F("Angle", 0.0, sub='ANGLE', desc="Mean scratch direction"),
        F("Angle Spread", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = random directions (polish swirl, halo); 0 = all parallel (brushed)"),
        F("Pitch Jitter", 0.3, 0.0, 0.9, desc="Per-scratch relative pitch variation"),
        F("Softness", 0.5, 0.0, 1.0, sub='FACTOR', desc="Edge softness of the scratch profile"),
        SEED], desc="Random short scratch segments; a lamp lights the ones tangent to circles around its "
                    "highlight, giving the circular scratch halo of polished metal")
    p = coords(b, gi)
    m1, a1, p1 = scratch_layer(b, gi, p, 0)
    p2 = b.vmath('ADD', b.vmath('SCALE', p, scale=1.37), (5.31, 2.77, 0.0))
    m2, a2, pp2 = scratch_layer(b, gi, p2, 1)
    top = b.gt(m2, m1)
    ang = b.select(top, a1, a2)
    pitch = b.select(top, p1, pp2)
    mask = b.mx(m1, m2)
    finish(b, go, ang, pitch, mask, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# 4. Glitter Patches
# ----------------------------------------------------------------------------------------------
def build_glitter():
    ng, b, gi, go = new_group("Holo Pattern: Glitter Patches", [
        VEC_IN, F("Scale", 12.0, 0.0, desc="Cells per Vector unit"),
        F("Randomness", 1.0, 0.0, 1.0, sub='FACTOR', desc="0 = regular hex-like cells, 1 = random"),
        F("Angle", 0.0, sub='ANGLE', desc="Mean grating direction"),
        F("Angle Spread", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = uniformly random angle per cell"),
        F("Pitch Jitter", 0.15, 0.0, 0.9, desc="Per-cell relative pitch variation"),
        F("Roughness Jitter", 0.0, 0.0, 0.9, desc="Per-cell relative roughness variation"),
        F("Coverage", 1.0, 0.0, 1.0, sub='FACTOR', desc="Fraction of cells that are embossed"),
        F("Gap", 0.0, 0.0, 0.5, desc="Unembossed border between cells (cell units; sequin look)"),
        SEED], desc="Voronoi mosaic, random grating angle and pitch per cell (holo glitter, sequin, "
                    "'mosaic' foils of Toisoul et al. 2018)")
    p = coords(b, gi)
    vor = b.voronoi(p, rand=gi.outputs['Randomness'])
    r1, r2, r3 = b.sep(vor.outputs['Color'])
    wn = b.white(vor.outputs['Position'])
    r4 = b.sep(wn.outputs['Color'])[0]
    ang = b.add(gi.outputs['Angle'], b.rand_signed(r1, b.mul(gi.outputs['Angle Spread'], PI / 2)))
    pitch = b.add(1.0, b.rand_signed(r2, gi.outputs['Pitch Jitter']))
    rough = b.add(1.0, b.rand_signed(r3, gi.outputs['Roughness Jitter']))
    ved = b.voronoi(p, feature='DISTANCE_TO_EDGE', rand=gi.outputs['Randomness'])
    gap = gi.outputs['Gap']
    gm = b.smooth(ved.outputs['Distance'], b.mul(gap, 0.6), b.add(gap, 1e-4))
    mask = b.mul(b.lt(r4, gi.outputs['Coverage']), gm)
    finish(b, go, ang, pitch, mask, rough)
    return ng


# ----------------------------------------------------------------------------------------------
# 5. Cosmos
# ----------------------------------------------------------------------------------------------
def disc_layer(b, gi, p, size, density, ring_frac=None, rand=0.75):
    vor = b.voronoi(p, rand=rand)
    dist = vor.outputs['Distance']
    r1, r2, r3 = b.sep(vor.outputs['Color'])
    wn = b.white(vor.outputs['Position'])
    r4, r5, _ = b.sep(wn.outputs['Color'])
    R = b.mul(size, b.madd(r2, 0.65, 0.35))
    soft = gi.outputs['Softness']
    disc = b.sub(1.0, b.smooth(dist, b.mul(R, b.sub(1.0, soft)), R))
    if ring_frac is not None:
        Ri = b.mul(R, 0.55)
        hole = b.smooth(dist, b.mul(Ri, b.sub(1.0, soft)), Ri)
        disc = b.mul(disc, b.mixf(b.lt(r3, ring_frac), 1.0, hole))
    m = b.mul(disc, b.lt(r1, density))
    ang = b.mul(r4, PI)
    pitch = b.add(1.0, b.rand_signed(r5, gi.outputs['Pitch Jitter']))
    return m, ang, pitch


def build_cosmos():
    ng, b, gi, go = new_group("Holo Pattern: Cosmos", [
        VEC_IN, F("Scale", 5.0, 0.0, desc="Planet cells per Vector unit"),
        F("Angle", 0.0, sub='ANGLE', desc="Background grating direction"),
        F("Background Wobble", 0.35, 0.0, PI, sub='ANGLE', desc="Swirl of the background grating"),
        F("Wobble Scale", 1.2, 0.0),
        F("Background Foil", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = background is embossed too, 0 = plain foil between circles"),
        F("Planet Density", 0.55, 0.0, 1.0, sub='FACTOR'),
        F("Planet Size", 0.36, 0.0, 0.5, desc="Max planet radius (cell units)"),
        F("Ring Fraction", 0.3, 0.0, 1.0, sub='FACTOR', desc="Fraction of planets that are rings"),
        F("Dot Scale", 3.5, 0.1, desc="Dot cells per planet cell"),
        F("Dot Density", 0.6, 0.0, 1.0, sub='FACTOR'),
        F("Dot Size", 0.28, 0.0, 0.5),
        F("Pitch Jitter", 0.2, 0.0, 0.9),
        F("Softness", 0.12, 0.0, 1.0, sub='FACTOR'),
        SEED], desc="Trading-card 'cosmos'/'galaxy' holo: circles and dots of different sizes, each its own "
                    "grating, over a swirling background grating")
    p = coords(b, gi)
    mp, ap, pp = disc_layer(b, gi, p, gi.outputs['Planet Size'], gi.outputs['Planet Density'],
                            ring_frac=gi.outputs['Ring Fraction'])
    pd = b.vmath('ADD', b.vmath('SCALE', p, scale=gi.outputs['Dot Scale']), (7.13, 3.71, 0.0))
    md, ad, pdm = disc_layer(b, gi, pd, gi.outputs['Dot Size'], gi.outputs['Dot Density'])
    n = b.noise(p, b.div(gi.outputs['Wobble Scale'], b.mx(gi.outputs['Scale'], 1e-3)), detail=1.0)
    abg = b.add(gi.outputs['Angle'], b.mul(b.sub(n, 0.5), b.mul(gi.outputs['Background Wobble'], 3.0)))
    bg = gi.outputs['Background Foil']
    sd, sp = b.over(md, bg), b.over(mp, b.mx(bg, md))
    ang = b.mixf(sp, b.mixf(sd, abg, ad), ap)
    pitch = b.mixf(sp, b.mixf(sd, 1.0, pdm), pp)
    mask = b.mx(gi.outputs['Background Foil'], b.mx(mp, md))
    finish(b, go, ang, pitch, mask, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# 6. Cracked Ice
# ----------------------------------------------------------------------------------------------
def build_cracked():
    ng, b, gi, go = new_group("Holo Pattern: Cracked Ice", [
        VEC_IN, F("Scale", 4.0, 0.0, desc="Coarse crack regions per Vector unit"),
        F("Shard Detail", 2.6, 0.1, desc="Shards per coarse region (fine / coarse cell ratio)"),
        F("Stretch", 1.8, 0.1, desc="Elongation of the shards (direction random per region)"),
        F("Angle", 0.0, sub='ANGLE'),
        F("Angle Spread", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = uniformly random angle per shard: large jumps"),
        F("Pitch Jitter", 0.25, 0.0, 0.9),
        F("Roughness Jitter", 0.2, 0.0, 0.9),
        SEED], desc="Shattered-glass shards: a fine Voronoi whose lattice is shifted, rotated and "
                    "stretched per coarse Voronoi region, so straight cracks cut cells into slivers")
    p = coords(b, gi)
    vc = b.voronoi(p, rand=1.0)
    cc = vc.outputs['Color']
    c1, c2, c3 = b.sep(cc)
    rot = b.node('ShaderNodeVectorRotate', rotation_type='Z_AXIS')
    b.link(b.vmath('SCALE', p, scale=gi.outputs['Shard Detail']), rot.inputs['Vector'])
    b.link(b.mul(c1, 2 * PI), rot.inputs['Angle'])
    q = b.vmath('MULTIPLY', rot.outputs[0], b.comb(gi.outputs['Stretch'], 1.0, 1.0))
    q = b.vmath('ADD', q, b.vmath('SCALE', cc, scale=13.7))
    vf = b.voronoi(q, rand=1.0)
    key = b.vmath('ADD', vf.outputs['Color'], b.vmath('SCALE', cc, scale=3.17))
    r1, r2, r3 = b.sep(b.white(key).outputs['Color'])
    ang = b.add(gi.outputs['Angle'], b.rand_signed(r1, b.mul(gi.outputs['Angle Spread'], PI / 2)))
    pitch = b.add(1.0, b.rand_signed(r2, gi.outputs['Pitch Jitter']))
    rough = b.add(1.0, b.rand_signed(r3, gi.outputs['Roughness Jitter']))
    finish(b, go, ang, pitch, 1.0, rough)
    return ng


# ----------------------------------------------------------------------------------------------
# 7. Sparkle Stars
# ----------------------------------------------------------------------------------------------
def star_layer(b, gi, p, size):
    vor = b.voronoi(p, rand=0.8)
    c = vor.outputs['Position']
    dist = vor.outputs['Distance']
    r1, r2, r3 = b.sep(vor.outputs['Color'])
    r4, r5, _ = b.sep(b.white(c).outputs['Color'])
    dx, dy, _ = b.sep(b.vmath('SUBTRACT', p, c))
    phi = b.math('ARCTAN2', dy, dx)
    n = gi.outputs['Points']
    # superformula star: r(phi) = R * (|cos(n phi'/4)|^q + |sin(n phi'/4)|^q)^(-1/q)
    # q = 2 circle, 1 straight-sided diamond, < 1 concave sparkle (2/3 = astroid)
    rotj = b.mul(r3, b.mul(gi.outputs['Rotation Jitter'], 2 * PI))
    t = b.mul(b.mul(b.add(phi, rotj), n), 0.25)
    q = b.mixf(gi.outputs['Sharpness'], 2.0, 0.3)
    su = b.add(b.math('POWER', b.absf(b.math('COSINE', t)), q),
               b.math('POWER', b.absf(b.math('SINE', t)), q))
    lobe = b.math('POWER', su, b.div(-1.0, q))
    sv = gi.outputs['Size Variation']
    R = b.mul(size, b.mixf(sv, 1.0, b.mul(r2, r2)))
    shape = b.mul(R, lobe)
    soft = gi.outputs['Softness']
    m = b.sub(1.0, b.smooth(dist, b.mul(shape, b.sub(1.0, soft)), shape))
    m = b.mul(m, b.lt(r1, gi.outputs['Density']))
    a_flat = b.add(gi.outputs['Angle'], b.rand_signed(r4, b.mul(gi.outputs['Angle Spread'], PI / 2)))
    ang = b.mixf(gi.outputs['Arm Grooves'], a_flat, b.add(phi, PI / 2))
    pitch = b.mul(gi.outputs['Star Pitch'], b.add(1.0, b.rand_signed(r5, gi.outputs['Pitch Jitter'])))
    # faceted dome: normal leans outward, more towards the tips (axis = tangential direction)
    tilt = b.mul(gi.outputs['Dome Tilt'], b.mn(b.div(dist, b.mx(shape, 1e-6)), 1.0))
    return m, ang, pitch, tilt, b.add(phi, PI / 2)


def build_stars():
    ng, b, gi, go = new_group("Holo Pattern: Sparkle Stars", [
        VEC_IN, F("Scale", 6.0, 0.0, desc="Big-star cells per Vector unit"),
        F("Density", 0.65, 0.0, 1.0, sub='FACTOR', desc="Fraction of cells holding a star"),
        F("Size", 0.45, 0.0, 0.5, desc="Max star radius (cell units)"),
        F("Size Variation", 0.75, 0.0, 1.0, sub='FACTOR', desc="Skews sizes towards small stars"),
        F("Small Star Scale", 2.6, 0.1, desc="Second, finer star layer (cells per big cell)"),
        F("Points", 4.0, 2.0, 12.0, desc="Star points (4 = sparkle, 5 = star, 8 = burst)"),
        F("Sharpness", 0.85, 0.0, 1.0, sub='FACTOR', desc="0 = round, 0.6 = straight sides, 1 = needle spikes"),
        F("Rotation Jitter", 0.15, 0.0, 1.0, sub='FACTOR', desc="Random rotation of each star (0 = all upright)"),
        F("Angle", 0.0, sub='ANGLE', desc="Grating direction inside the stars"),
        F("Angle Spread", 1.0, 0.0, 1.0, sub='FACTOR', desc="Random grating direction per star"),
        F("Arm Grooves", 0.0, 0.0, 1.0, sub='FACTOR', desc="1 = grooves run along each arm, so arms flash one by one as the card turns"),
        F("Star Pitch", 0.64, 0.1, desc="Pitch multiplier inside stars (Blender Guru's card: 1000 / 1564 nm)"),
        F("Pitch Jitter", 0.1, 0.0, 0.9),
        F("Softness", 0.15, 0.0, 1.0, sub='FACTOR'),
        F("Dome Tilt", 0.0, 0.0, 1.0, sub='ANGLE', desc="Embossed relief: normal leans outward towards the tips "
          "(-> Normal output), so every star catches a point light somewhere"),
        SEED], outputs=STD_OUT_N,
        desc="Star-shaped sparkles of mixed size, each its own grating (starlight holo, the holo "
             "card in Blender Guru's video); Mask = 0 outside the stars")
    p = coords(b, gi)
    m1, a1, p1, t1, x1 = star_layer(b, gi, p, gi.outputs['Size'])
    p2 = b.vmath('ADD', b.vmath('SCALE', p, scale=gi.outputs['Small Star Scale']), (3.3, 9.1, 0.0))
    m2, a2, pp2, t2, x2 = star_layer(b, gi, p2, gi.outputs['Size'])
    top = b.gt(m2, m1)
    mm = b.mx(m1, m2)
    on = b.gt(mm, 1e-4)                               # base is plain foil (mask 0)
    ang = b.mixf(on, gi.outputs['Angle'], b.select(top, a1, a2))
    pitch = b.mixf(on, 1.0, b.select(top, p1, pp2))
    tilt = b.mul(b.gt(mm, 0.5), b.select(top, t1, t2))
    nrm = b.group(helper_tilt(), Axis_Angle=b.select(top, x1, x2), Tilt=tilt).outputs['Normal']
    finish(b, go, ang, pitch, mm, 1.0, normal=nrm)
    return ng


# ----------------------------------------------------------------------------------------------
# 8. Diagonal Beams
# ----------------------------------------------------------------------------------------------
def build_beams():
    ng, b, gi, go = new_group("Holo Pattern: Diagonal Beams", [
        VEC_IN, F("Scale", 7.0, 0.0, desc="Stripes per Vector unit (across the stripes)"),
        F("Direction", PI / 4, sub='ANGLE', desc="Direction the stripes run along"),
        F("Duty", 0.35, 0.0, 1.0, sub='FACTOR', desc="Beam width as a fraction of the stripe period"),
        F("Width Jitter", 0.7, 0.0, 1.0, sub='FACTOR', desc="Random beam widths"),
        F("Beam Angle", PI / 4, sub='ANGLE', desc="Grating direction inside the beams"),
        F("Gap Angle", PI / 4 + 0.45, sub='ANGLE', desc="Grating direction between the beams"),
        F("Angle Jitter", 0.06, 0.0, PI, sub='ANGLE', desc="Random angle offset per beam"),
        F("Beam Pitch", 1.0, 0.1),
        F("Gap Pitch", 0.8, 0.1),
        F("Gap Foil", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = gaps are embossed (Gap Angle), 0 = plain foil"),
        F("Softness", 0.04, 0.0, 0.5, desc="Edge softness (stripe-period units)"),
        F("Ridge Tilt", 0.0, 0.0, 1.0, sub='ANGLE', desc="Embossed ridge per beam: normal tilts across the beam "
          "by +-this angle (-> Normal output), so each beam lights along its length"),
        SEED], outputs=STD_OUT_N,
        desc="Parallel stripes alternating between two gratings (Blender Guru's diagonal "
             "rainbow beams, 'line' and 'tinsel' holo foils)")
    p = coords(b, gi, seed=False)
    x, y, _ = b.sep(p)
    d = gi.outputs['Direction']
    s = b.add(b.mul(x, b.mul(b.math('SINE', d), -1.0)), b.mul(y, b.math('COSINE', d)))
    k = b.math('FLOOR', s)
    f = b.sub(s, k)
    kw = b.madd(gi.outputs['Seed'], 101.3, k)
    rk = b.white(w=kw, dims='1D').outputs['Value']
    rk2 = b.white(w=b.add(kw, 0.37), dims='1D').outputs['Value']
    duty = b.mul(gi.outputs['Duty'], b.add(1.0, b.rand_signed(rk, gi.outputs['Width Jitter'])))
    duty = b.mn(b.mx(duty, 0.02), 0.98)
    e = b.mx(gi.outputs['Softness'], 1e-4)
    inb = b.mul(b.smooth(f, 0.0, e), b.sub(1.0, b.smooth(f, b.sub(duty, e), duty)))
    a_beam = b.add(gi.outputs['Beam Angle'], b.rand_signed(rk2, gi.outputs['Angle Jitter']))
    sel = b.over(inb, gi.outputs['Gap Foil'])
    ang = b.mixf(sel, gi.outputs['Gap Angle'], a_beam)
    pitch = b.mixf(sel, gi.outputs['Gap Pitch'], gi.outputs['Beam Pitch'])
    mask = b.mixf(inb, gi.outputs['Gap Foil'], 1.0)
    u = b.mn(b.mx(b.sub(b.mul(b.div(f, duty), 2.0), 1.0), -1.0), 1.0)      # -1..1 across the beam
    nrm = b.group(helper_tilt(), Axis_Angle=d, Tilt=b.mul(b.mul(gi.outputs['Ridge Tilt'], u), inb)).outputs['Normal']
    finish(b, go, ang, pitch, mask, 1.0, normal=nrm)
    return ng


# ----------------------------------------------------------------------------------------------
# 9. Dot-Matrix 2D
# ----------------------------------------------------------------------------------------------
def build_dotmatrix():
    ng, b, gi, go = new_group("Holo Pattern: Dot Matrix", [
        VEC_IN, F("Scale", 40.0, 0.0, desc="Dots per Vector unit"),
        F("Dot Size", 0.85, 0.0, 1.0, sub='FACTOR', desc="Dot diameter / dot spacing"),
        F("Squareness", 0.0, 0.0, 1.0, sub='FACTOR', desc="0 = round dots, 1 = square pixels"),
        F("Kinematic", 0.6, 0.0, 1.0, sub='FACTOR', desc="0 = random angle per dot (sparkle), 1 = smooth angle field (rotating/expanding animation on tilt)"),
        ("Center", 'NodeSocketVector', (0.5, 0.5, 0.0), dict(desc="Centre of the kinematic field (Vector space)")),
        F("Twist", PI / 2, sub='ANGLE', desc="Field angle offset (0 = radial grating vectors, 90 deg = concentric)"),
        F("Ring Frequency", 3.0, desc="Field turns per Vector unit of radius (spiral / ring kinematics)"),
        F("Angle Levels", 12.0, 0.0, 64.0, desc="Quantise to this many orientations (dot-matrix originators use a fixed set); 0 = continuous"),
        F("Angle", 0.0, sub='ANGLE'),
        F("Pitch Jitter", 0.1, 0.0, 0.9),
        F("Softness", 0.15, 0.0, 1.0, sub='FACTOR'),
        SEED], desc="Square grid of holo-pixels, each a small grating with its own angle (dot-matrix "
                    "security holograms; real pixels are 20-100 um)")
    v = coords(b, gi, scale=1.0, seed=False)
    p = b.vmath('SCALE', v, scale=gi.outputs['Scale'])
    cell = b.vmath('FLOOR', p)
    lx, ly, _ = b.sep(b.vmath('SUBTRACT', b.vmath('SUBTRACT', p, cell), (0.5, 0.5, 0.0)))
    cc = b.vmath('SCALE', b.vmath('ADD', cell, (0.5, 0.5, 0.0)),
                 scale=b.div(1.0, b.mx(gi.outputs['Scale'], 1e-6)))
    dcx, dcy, _ = b.sep(b.vmath('SUBTRACT', cc, gi.outputs['Center']))
    rc = b.vmath('LENGTH', b.comb(dcx, dcy, 0.0))
    phic = b.math('ARCTAN2', dcy, dcx)
    seedv = b.comb(b.mul(gi.outputs['Seed'], 7.77), 0.0, 0.0)
    r1, r2, _ = b.sep(b.white(b.vmath('ADD', cell, seedv)).outputs['Color'])
    a_field = b.add(b.add(phic, gi.outputs['Twist']), b.mul(gi.outputs['Ring Frequency'], b.mul(rc, 2 * PI)))
    ang = b.add(gi.outputs['Angle'], b.mixf(gi.outputs['Kinematic'], b.mul(r1, PI), a_field))
    L = gi.outputs['Angle Levels']
    inc = b.div(PI, b.mx(L, 1.0))
    snapped = b.math('SNAP', b.add(ang, b.mul(inc, 0.5)), inc)
    ang = b.mixf(b.gt(L, 0.5), ang, snapped)
    R = b.mul(gi.outputs['Dot Size'], 0.5)
    soft = gi.outputs['Softness']
    d_round = b.vmath('LENGTH', b.comb(lx, ly, 0.0))
    d_sq = b.mx(b.absf(lx), b.absf(ly))
    dd = b.mixf(gi.outputs['Squareness'], d_round, d_sq)
    mask = b.sub(1.0, b.smooth(dd, b.mul(R, b.sub(1.0, soft)), R))
    pitch = b.add(1.0, b.rand_signed(r2, gi.outputs['Pitch Jitter']))
    finish(b, go, ang, pitch, mask, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# 10. Image-Driven
# ----------------------------------------------------------------------------------------------
def build_image():
    ng, b, gi, go = new_group("Holo Pattern: Image Driven", [
        ("Angle Map", 'NodeSocketColor', (1.0, 0.0, 0.0, 1.0),
         dict(desc="Painted colour: hue = grating direction. Use Closest interpolation for crisp regions")),
        F("Mask Map", 1.0, 0.0, 1.0, sub='FACTOR', desc="Where the foil is embossed (e.g. an image's alpha/BW)"),
        F("Pitch Map", 0.5, 0.0, 1.0, sub='FACTOR', desc="0 -> Pitch Min, 1 -> Pitch Max"),
        F("Pitch Min", 0.8, 0.1),
        F("Pitch Max", 1.2, 0.1),
        F("Angle", 0.0, sub='ANGLE', desc="Offset added to the painted angle"),
        F("Hue Range", PI, sub='ANGLE', desc="Angle span of the hue wheel (180 deg covers every grating direction)"),
        F("Saturation Mask", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = grey/white paint means plain foil (no grating)"),
        F("Saturation Threshold", 0.12, 0.0, 1.0, sub='FACTOR'),
    ], desc="Artist-painted foil: grating direction from an image's hue, mask from an image")
    hsv = b.node('ShaderNodeSeparateColor', mode='HSV')
    b.link(gi.outputs['Angle Map'], hsv.inputs[0])
    H, S, _ = hsv.outputs
    ang = b.madd(H, gi.outputs['Hue Range'], gi.outputs['Angle'])
    th = gi.outputs['Saturation Threshold']
    sm = b.smooth(S, th, b.add(th, 0.08))
    mask = b.mul(gi.outputs['Mask Map'], b.mixf(gi.outputs['Saturation Mask'], 1.0, sm))
    pitch = b.mixf(gi.outputs['Pitch Map'], gi.outputs['Pitch Min'], gi.outputs['Pitch Max'])
    finish(b, go, ang, pitch, mask, 1.0)
    return ng


# ----------------------------------------------------------------------------------------------
# Layer (combine two patterns)
# ----------------------------------------------------------------------------------------------
def build_layer():
    ins = []
    for side in ("Base", "Top"):
        ins += [F(f"{side} Angle", 0.0, sub='ANGLE'), F(f"{side} Pitch", 1.0),
                F(f"{side} Mask", 1.0 if side == "Base" else 0.0, 0.0, 1.0, sub='FACTOR'),
                F(f"{side} Roughness", 1.0)]
    ins.append(F("Top Opacity", 1.0, 0.0, 1.0, sub='FACTOR'))
    ins += [("Base Normal", 'NodeSocketVector', (0.0, 0.0, 0.0), dict(hide=True, desc="optional")),
            ("Top Normal", 'NodeSocketVector', (0.0, 0.0, 0.0), dict(hide=True, desc="optional"))]
    ng, b, gi, go = new_group("Holo Pattern: Layer", ins, outputs=STD_OUT_N,
                              desc="Stack two patterns: Top wins where its Mask x Opacity covers more than the base leaves "
                                   "(e.g. Sparkle Stars over Diagonal Beams = the holo card in Blender Guru's video)")
    tm = b.mul(gi.outputs['Top Mask'], gi.outputs['Top Opacity'])
    sel = b.over(tm, gi.outputs['Base Mask'])
    ang = b.mixf(sel, gi.outputs['Base Angle'], gi.outputs['Top Angle'])
    pitch = b.mixf(sel, gi.outputs['Base Pitch'], gi.outputs['Top Pitch'])
    rough = b.mixf(sel, gi.outputs['Base Roughness'], gi.outputs['Top Roughness'])
    nrm = b.mixv(sel, gi.outputs['Base Normal'], gi.outputs['Top Normal'])
    finish(b, go, ang, pitch, b.mx(gi.outputs['Base Mask'], tm), rough, normal=nrm)
    return ng


PATTERN_BUILDERS = [build_linear, build_radial, build_scratched, build_glitter, build_cosmos,
                    build_cracked, build_stars, build_beams, build_dotmatrix, build_image]


# ----------------------------------------------------------------------------------------------
# Holo Card Shader: printed card over foil, glossy laminate on top
# ----------------------------------------------------------------------------------------------
def bsdf_inputs(bsdf):
    return {it.name for it in bsdf.interface.items_tree
            if getattr(it, 'item_type', '') == 'SOCKET' and it.in_out == 'INPUT'}


def bsdf_instance(b, bsdf, **vals):
    """vals: logical names of BSDF_SOCK -> value/socket. Unknown or absent sockets are skipped."""
    g = b.group(bsdf)
    S = BSDF_SOCK
    for key, val in vals.items():
        if S.get(key) and S[key] in g.inputs:
            b._in(g.inputs[S[key]], val)
    if S.get('tangent') and S['tangent'] in g.inputs:
        tan = b.node('ShaderNodeTangent', direction_type='UV_MAP')
        b.link(tan.outputs['Tangent'], g.inputs[S['tangent']])
    return g.outputs[BSDF_OUT]


def build_card_shader(bsdf):
    v2 = {'Mask', 'Pitch Multiplier', 'Coat Tint'} <= bsdf_inputs(bsdf)
    ng, b, gi, go = new_group("Holo Card Shader", [
        ("Art Color", 'NodeSocketColor', (0.8, 0.8, 0.8, 1.0), dict(desc="Printed card art (Image Texture)")),
        F("Foil Mask", 1.0, 0.0, 1.0, sub='FACTOR', desc="1 = metallised holo foil under the print, 0 = card stock"),
        F("Ink Opacity", 0.0, 0.0, 1.0, sub='FACTOR', desc="Opaque white-ink underlay: 1 hides the foil (reads as solid print)"),
        F("Pattern Angle", 0.0, sub='ANGLE'), F("Pattern Pitch", 1.0), F("Pattern Mask", 1.0, 0.0, 1.0, sub='FACTOR'),
        F("Pattern Roughness", 1.0),
        ("Foil Color", 'NodeSocketColor', (0.92, 0.92, 0.92, 1.0), dict(desc="Bare foil colour (silver ~0.9, gold ~(1, 0.75, 0.35))")),
        F("Ink Tint", 1.0, 0.0, 3.0, desc="How strongly the translucent ink tints the foil (art colour ^ Ink Tint)"),
        F("Ink Scatter", 0.12, 0.0, 1.0, sub='FACTOR', desc="Diffuse share of the ink layer over foil"),
        F("Pitch (nm)", 1200.0, 200.0, 100000.0),
        F("Orders", 3.0, 1.0, 8.0),
        F("Roughness", 0.08, 0.0, 1.0, sub='FACTOR', desc="Grating lobe roughness"),
        F("Order 0 Strength", 0.3, 0.0, 1.0, sub='FACTOR'),
        F("Efficiency Falloff", 0.6, 0.02, 0.98, sub='FACTOR'),
        F("Groove Depth (nm)", 0.0, 0.0, 1000.0, desc="v2 BSDF only: > 0 = physical Bessel efficiencies"),
        F("Spectral Saturation", 1.0, 0.0, 1.0, sub='FACTOR'),
        F("Grating Type", 0.0, 0.0, 1.0),
        F("Plain Foil Roughness", 0.12, 0.0, 1.0, sub='FACTOR', desc="Unembossed foil (Pattern Mask = 0); v1 BSDF only"),
        F("Laminate", 1.0, 0.0, 1.0, sub='FACTOR', desc="Strength of the glossy top coat"),
        F("Laminate Roughness", 0.05, 0.0, 1.0, sub='FACTOR'),
        F("Laminate IOR", 1.5, 1.0, 3.0),
        F("Seed", 0.0),
        ("Foil Normal", 'NodeSocketVector', (0.0, 0.0, 0.0),
         dict(hide=True, desc="Embossed foil normal (a pattern's Normal output); unconnected = geometry normal")),
    ], outputs=[("Shader", 'NodeSocketShader', 'NONE', "")],
        desc="Card print over holo foil + laminate. v1 BSDF: diffuse + plain foil + grating (5) + laminate "
             "= 8 closures; v2 BSDF: diffuse + paper laminate + grating (<= 12) = 14. Cycles' cap is 64")
    I = gi.outputs
    art = I['Art Color']
    k = I['Ink Tint']
    tint = b.vmath('POWER', b.vmath('MAXIMUM', art, (1e-4, 1e-4, 1e-4)), b.comb(k, k, k))
    rough = b.mn(b.mul(I['Roughness'], I['Pattern Roughness']), 1.0)
    diff = b.node('ShaderNodeBsdfDiffuse')
    b.link(art, diff.inputs['Color'])
    foil_eff = b.mul(I['Foil Mask'], b.sub(1.0, I['Ink Opacity']))
    fr = b.node('ShaderNodeFresnel')
    b.link(I['Laminate IOR'], fr.inputs['IOR'])
    coat = b.node('ShaderNodeBsdfGlossy', distribution='GGX')
    coat.inputs['Color'].default_value = (1, 1, 1, 1)
    b.link(I['Laminate Roughness'], coat.inputs['Roughness'])
    geo = b.node('ShaderNodeNewGeometry')
    fn = I['Foil Normal']
    n_eff = b.mixv(b.gt(b.vmath('LENGTH', fn), 0.5), geo.outputs['Normal'], b.vmath('NORMALIZE', fn))
    common = dict(normal=n_eff, pitch=I['Pitch (nm)'], angle=I['Pattern Angle'], rough=rough, orders=I['Orders'],
                  eta0=I['Order 0 Strength'], falloff=I['Efficiency Falloff'], sat=I['Spectral Saturation'],
                  type=I['Grating Type'], seed=I['Seed'])
    if v2:
        # v2 BSDF: Mask, Pitch Multiplier and the laminate (Coat) live inside the BSDF; translucent
        # ink over foil = Coat Tint (INTERFACE.md "Notes for layering")
        foil = bsdf_instance(b, bsdf, color=I['Foil Color'], pitch_mult=I['Pattern Pitch'],
                             mask=I['Pattern Mask'], depth=I['Groove Depth (nm)'], coat=I['Laminate'],
                             coat_rough=I['Laminate Roughness'], coat_ior=I['Laminate IOR'],
                             coat_tint=tint, **common)
        paper = b.node('ShaderNodeMixShader')                 # laminate over card stock / opaque ink
        b.link(b.mul(fr.outputs[0], I['Laminate']), paper.inputs[0])
        b.link(diff.outputs[0], paper.inputs[1])
        b.link(coat.outputs[0], paper.inputs[2])
        top = b.node('ShaderNodeMixShader')
        b.link(b.mul(foil_eff, b.sub(1.0, I['Ink Scatter'])), top.inputs[0])
        b.link(paper.outputs[0], top.inputs[1])
        b.link(foil, top.inputs[2])
    else:
        foilcol = b.vmath('MULTIPLY', tint, I['Foil Color'])
        common['pitch'] = b.mul(I['Pitch (nm)'], I['Pattern Pitch'])
        grating = bsdf_instance(b, bsdf, color=foilcol, **common)
        plain = b.node('ShaderNodeBsdfGlossy', distribution='GGX')
        b.link(foilcol, plain.inputs['Color'])
        b.link(I['Plain Foil Roughness'], plain.inputs['Roughness'])
        b.link(n_eff, plain.inputs['Normal'])
        foil = b.node('ShaderNodeMixShader')
        b.link(I['Pattern Mask'], foil.inputs[0])
        b.link(plain.outputs[0], foil.inputs[1])
        b.link(grating, foil.inputs[2])
        base = b.node('ShaderNodeMixShader')
        b.link(b.mul(foil_eff, b.sub(1.0, I['Ink Scatter'])), base.inputs[0])
        b.link(diff.outputs[0], base.inputs[1])
        b.link(foil.outputs[0], base.inputs[2])
        top = b.node('ShaderNodeMixShader')
        b.link(b.mul(fr.outputs[0], I['Laminate']), top.inputs[0])
        b.link(base.outputs[0], top.inputs[1])
        b.link(coat.outputs[0], top.inputs[2])
    b.link(top.outputs[0], go.inputs['Shader'])
    layout(ng)
    return ng


# ----------------------------------------------------------------------------------------------
# demo textures + "Holo Card" material template
# ----------------------------------------------------------------------------------------------
def prepare_textures():
    """Writes the demo card maps (numpy only, small): a made-up card design (no third-party art),
    its foil mask (white = card body, black = art box) and a white-ink map (opaque where the
    'subject' of the artwork is printed, translucent sky so the foil shimmers through)."""
    import numpy as np
    os.makedirs(TEX, exist_ok=True)
    W, H = 630, 880                                   # 63 x 88 mm at 10 px/mm
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (x + 0.5) / W, (y + 0.5) / H               # v = 0 at the top (image rows top first)

    def box(x0, y0, x1, y1, r=0.0, soft=1.5):
        """Rounded-rectangle coverage in pixels (anti-aliased)."""
        cx, cy = (x0 + x1) / 2 * W, (y0 + y1) / 2 * H
        hx, hy = (x1 - x0) / 2 * W - r, (y1 - y0) / 2 * H - r
        qx, qy = np.maximum(np.abs(x - cx) - hx, 0), np.maximum(np.abs(y - cy) - hy, 0)
        d = np.hypot(qx, qy) - r
        inside = np.minimum(np.maximum(np.abs(x - cx) - hx, np.abs(y - cy) - hy), 0)
        return np.clip(0.5 - (d + inside) / soft, 0, 1)

    def disc(cx, cy, rad, soft=1.5):
        return np.clip(0.5 - (np.hypot(x - cx * W, y - cy * H) - rad * W) / soft, 0, 1)

    def over(img, col, a):
        return img * (1 - a[..., None]) + np.asarray(col, np.float32) * a[..., None]

    art = np.zeros((H, W, 3), np.float32)
    art[:] = (0.08, 0.10, 0.22)                                       # card edge
    art = over(art, (0.80, 0.82, 0.86), box(0.03, 0.022, 0.97, 0.978, r=14))   # silver frame
    body = box(0.055, 0.04, 0.945, 0.96, r=10)
    grad = (1 - v)[..., None] * np.array([0.30, 0.42, 0.62]) + v[..., None] * np.array([0.16, 0.20, 0.38])
    art = art * (1 - body[..., None]) + grad * body[..., None]
    for x0, x1 in ((0.09, 0.47), (0.74, 0.91)):                       # title bar "text"
        art = over(art, (0.93, 0.93, 0.95), box(x0, 0.062, x1, 0.098, r=6))
    abox = box(0.09, 0.12, 0.91, 0.56, r=4)                           # art box
    sky_t = np.clip((v - 0.12) / 0.44, 0, 1)[..., None]
    sky = (1 - sky_t) * np.array([0.10, 0.06, 0.30]) + sky_t * np.array([0.85, 0.45, 0.55])
    art = art * (1 - abox[..., None]) + sky * abox[..., None]
    rng = np.random.default_rng(7)
    stars = np.zeros((H, W), np.float32)
    for sx, sy, sr in zip(rng.uniform(0.11, 0.89, 60), rng.uniform(0.13, 0.40, 60), rng.uniform(0.6, 1.8, 60)):
        stars = np.maximum(stars, disc(sx, sy, sr / W, soft=1.0))
    art = over(art, (1.0, 0.97, 0.85), stars * abox)
    planet = disc(0.64, 0.25, 0.13) * abox
    shade = np.clip(0.55 + 0.45 * ((0.58 * W - x) + (0.20 * H - y)) / (0.2 * W), 0.25, 1.0)[..., None]
    art = over(art, np.array([0.98, 0.72, 0.35]) * shade, planet)
    ring = np.clip(1 - np.abs(np.hypot((x - 0.64 * W) / 1.0, (y - 0.25 * H) / 0.28) - 0.21 * W) / 3.0, 0, 1)
    ring *= (1 - planet * (y < 0.25 * H)) * abox
    art = over(art, (0.95, 0.90, 0.80), ring)
    ridge = 0.43 + 0.05 * np.sin(u * 9.0 + 1.0) + 0.035 * np.sin(u * 23.0) + 0.06 * np.abs(np.sin(u * 4.1))
    mount = np.clip((v - ridge) * H / 1.5 + 0.5, 0, 1) * abox
    art = over(art, (0.10, 0.08, 0.20), mount)
    art = over(art, (0.88, 0.90, 0.95), box(0.09, 0.575, 0.91, 0.61, r=6) * 0.85)   # info strip
    for i, y0 in enumerate((0.66, 0.70, 0.74, 0.80, 0.84, 0.88)):                 # text lines
        x1 = (0.86, 0.80, 0.55, 0.88, 0.83, 0.62)[i]
        art = over(art, (0.90, 0.92, 0.96), box(0.12, y0, x1, y0 + 0.016, r=4) * 0.9)
    art = over(art, (0.98, 0.80, 0.30), disc(0.12, 0.627, 0.018))
    art = np.clip(art, 0, 1)

    mask = 1.0 - abox                                                 # white = card body
    subject = np.maximum(planet, np.maximum(mount, ring))
    k = 5                                                             # soft edge (box blur)
    pad = np.pad(subject, k, mode='edge')
    ink = sum(pad[k + dy:k + dy + H, k + dx:k + dx + W] for dy in range(-k, k + 1) for dx in range(-k, k + 1))
    ink = np.clip(ink / (2 * k + 1) ** 2 * abox, 0, 1)

    def save_png(path, img):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
        im = bpy.data.images.new(os.path.basename(path), W, H, alpha=True)
        px = np.ones((H, W, 4), np.float32)
        px[..., :3] = img[::-1]                                       # Blender rows are bottom first
        im.pixels.foreach_set(px.ravel())
        im.filepath_raw = path
        im.file_format = 'PNG'
        im.save()
        bpy.data.images.remove(im)
        return path

    return dict(art=save_png(os.path.join(TEX, 'demo_card_art.png'), art),
                mask=save_png(os.path.join(TEX, 'demo_card_foil_mask.png'), mask),
                ink=save_png(os.path.join(TEX, 'demo_card_ink_opacity.png'), ink))


def make_paint_angle_demo():
    """Painted angle map for the Image Driven demo (gallery): hue = grating direction, grey = plain foil.
    Load it as Non-Color with Closest interpolation."""
    import numpy as np
    os.makedirs(TEX, exist_ok=True)

    def save_png(path, rgb):
        h, w = rgb.shape[:2]
        img = bpy.data.images.new(os.path.basename(path), w, h, alpha=True)
        px = np.ones((h, w, 4), np.float32)
        px[..., :3] = rgb[::-1]
        img.pixels.foreach_set(px.ravel())
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        bpy.data.images.remove(img)

    # painted angle map for the Image Driven demo: hue = grating direction, grey = plain foil
    p = os.path.join(TEX, 'paint_angle_demo.png')
    W, H = 512, 716
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = x / W, (H - 1 - y) / W                        # isotropic, v up
    hue = np.full((H, W), 0.0, np.float32)
    sat = np.full((H, W), 0.0, np.float32)
    # background: hue sweeps slowly left->right (kinematic sweep)
    hue[:] = (0.15 + 0.5 * u) % 1.0
    sat[:] = 0.9
    # 5-point star whose hue follows the polar angle (rotational kinematic)
    cx, cy = 0.5, 0.95
    dx, dy = u - cx, v - cy
    r = np.hypot(dx, dy)
    ph = np.arctan2(dy, dx)
    R = 0.36 * (0.42 + 0.58 * np.abs(np.cos(2.5 * (ph - np.pi / 2))) ** 1.6)
    star = r < R
    hue[star] = ((ph[star] / np.pi) % 1.0)
    # grey outline ring around the star (plain foil)
    ring = (r >= R) & (r < R + 0.025)
    sat[ring] = 0.0
    # three horizontal bands with fixed hues, separated by grey lines
    for i, h0 in enumerate((0.0, 0.33, 0.66)):
        y0 = 0.22 + i * 0.1
        band = (np.abs(v - y0) < 0.035) & (u > 0.08) & (u < 0.92)
        hue[band] = h0
        edge = (np.abs(np.abs(v - y0) - 0.045) < 0.008) & (u > 0.08) & (u < 0.92)
        sat[edge] = 0.0
    rgb = np.zeros((H, W, 3), np.float32)
    hh = hue * 6.0
    i = np.floor(hh).astype(int) % 6
    f = hh - np.floor(hh)
    vv = np.ones_like(hue)
    pp, q, t = vv * (1 - sat), vv * (1 - sat * f), vv * (1 - sat * (1 - f))
    for k, (a, bb, c) in enumerate(((vv, t, pp), (q, vv, pp), (pp, vv, t), (pp, q, vv), (t, pp, vv), (vv, pp, q))):
        sel = i == k
        rgb[sel, 0], rgb[sel, 1], rgb[sel, 2] = a[sel], bb[sel], c[sel]
    rgb[sat == 0] = 0.5
    save_png(p, rgb)    # load as Non-Color so the hue reaches the shader unchanged
    return p


def load_image(path, colorspace='sRGB'):
    img = bpy.data.images.load(path, check_existing=True)
    img.colorspace_settings.name = colorspace
    img.pack()                    # self-contained asset: the maps travel with the material
    return img


def card_vector(b):
    """UV * (1, 88/63): isotropic card coordinates (x along U, 1 unit = card width)."""
    tc = b.node('ShaderNodeTexCoord')
    return b.vmath('MULTIPLY', tc.outputs['UV'], (1.0, 88.0 / 63.0, 1.0)), tc


def build_card_material(card_shader, patterns, tex, name="Holo Card", pattern="Holo Pattern: Cosmos",
                        reverse=False, pattern_inputs=None, shader_inputs=None):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    b = NB(nt)
    out = b.node('ShaderNodeOutputMaterial')
    vec, tc = card_vector(b)
    art = b.node('ShaderNodeTexImage', interpolation='Cubic')
    art.image = load_image(tex['art'])
    art.label = "Card art"
    b.link(tc.outputs['UV'], art.inputs['Vector'])
    msk = b.node('ShaderNodeTexImage', interpolation='Linear')
    msk.image = load_image(tex['mask'], 'Non-Color')
    msk.label = "Foil mask (white = card body)"
    b.link(tc.outputs['UV'], msk.inputs['Vector'])
    rev = b.node('ShaderNodeValue')
    rev.label = "Reverse Holo (1 = foil on card body, 0 = foil in art box)"
    rev.outputs[0].default_value = 1.0 if reverse else 0.0
    mv = b.sep(msk.outputs['Color'])[0]
    foil = b.mixf(rev.outputs[0], b.sub(1.0, mv), mv)
    pat = b.group(patterns[pattern])
    pat.label = "Pattern (swap for any Holo Pattern group)"
    b.link(vec, pat.inputs['Vector'])
    for k, v in (pattern_inputs or {}).items():
        b._in(pat.inputs[k], v)
    cs = b.group(card_shader)
    b.link(art.outputs['Color'], cs.inputs['Art Color'])
    b.link(foil, cs.inputs['Foil Mask'])
    if os.path.exists(tex['ink']):
        ink = b.node('ShaderNodeTexImage', interpolation='Linear')
        ink.image = load_image(tex['ink'], 'Non-Color')
        ink.label = "Ink opacity (white ink underlay)"
        b.link(tc.outputs['UV'], ink.inputs['Vector'])
        b.link(b.mul(b.sep(ink.outputs['Color'])[0], b.sub(1.0, rev.outputs[0])), cs.inputs['Ink Opacity'])
    for a, c in (('Angle', 'Pattern Angle'), ('Pitch Multiplier', 'Pattern Pitch'),
                 ('Mask', 'Pattern Mask'), ('Roughness Multiplier', 'Pattern Roughness'), ('Normal', 'Foil Normal')):
        if a in pat.outputs:
            b.link(pat.outputs[a], cs.inputs[c])
    for k, v in (shader_inputs or {}).items():
        b._in(cs.inputs[k], v)
    b.link(cs.outputs['Shader'], out.inputs['Surface'])
    layout(nt)
    mat.use_fake_user = True
    return mat


def build_all():
    bsdf = load_bsdf()
    pats = {}
    for f in PATTERN_BUILDERS:
        ng = f()
        pats[ng.name] = ng
    pats["Holo Pattern: Layer"] = build_layer()
    cs = build_card_shader(bsdf)
    tex = prepare_textures()
    build_card_material(cs, pats, tex)
    return pats, cs, tex


if __name__ == "__main__":
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    pats, cs, tex = build_all()
    print("BSDF:", BSDF_GROUP, "from", BSDF_BLEND, "| interface ok:",
          {'Mask', 'Pitch Multiplier', 'Coat Tint'} <= bsdf_inputs(load_bsdf()))
    for name in sorted(pats):
        ng = pats[name]
        n_all = len(ng.nodes) + sum(len(n.node_tree.nodes) for n in ng.nodes if n.type == 'GROUP')
        print(f"group {name:40s} nodes {len(ng.nodes):4d} (incl. helpers {n_all})")
    os.makedirs(os.path.dirname(OUT_BLEND), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=OUT_BLEND, compress=True, relative_remap=True)
    print("saved", OUT_BLEND)
