"""
Build the v2 "Diffraction Grating BSDF" node group for stock Blender 5.x Cycles (pure SVM, no OSL).

    blender -b --factory-startup --python scripts/build_shader.py [-- OUT.blend]
    -> build/diffraction_grating.blend   (scripts/build_holo_foil.py builds the full holo_foil.blend)
       node group  "Diffraction Grating BSDF"  (+ its "DG ..." sub-groups and the packed Bessel LUT image)
       material    "Holo Foil (v2)"            (demo: cracked-ice holo patches, aluminium under a clear coat)
       compositor  "DG Foil Denoise"           (OIDN with a clean albedo on the foil)
       scene       "Holo Card Demo"            (card + lamps + camera, compositor already wired)

How it works (derivation: docs/HOW_IT_WORKS.md):
  * Physics core: order (m, n) at wavelength lam leaves along
    wi_t = lam (m g + n g2)/d - wo_t, and a GGX lobe whose normal is h = normalize(wi + wo) puts its
    mirror direction exactly there. Every lobe weight is efficiency x colour weight / probability.
  * Per shading point: O order groups x H wavelengths = K diffracted lobes (Quality 1..4 -> 4, 8, 12, 16),
    plus the mirror order and the clear-coat lobe. Unused lobe slots sit behind Mix Shader nodes with
    factor 0, which Cycles' SVM skips together with every node that only feeds them.
    - orders: stratified ("systematic") sampling of a two-sided truncated geometric distribution,
      u_a = (a + xi)/O; a 2D lattice draws n from a second stratified dimension;
    - wavelengths: importance-sampled, p(lam) ~ R+G+B of the spectral colour (+10 % floor); the inverse CDF is
      baked into a Float Curve, the colour/pdf weight into an RGB Curves node; stratified over all K lobes.
  * Colour: CIE 1931 x D65 -> linear Rec.709, negative channels clipped, white-balanced (spectral.py).
  * Energy: in art mode the mirror order takes the energy of every diffracted order that is evanescent at the
    current view angle and wavelength (closed form, 1D and 2D), so a white furnace stays white at any angle, pitch
    and Order 0 Strength.
  * Metal Fresnel (Schlick on "Color") per lobe at its own half-vector angle; optional clear coat
    (dielectric reflection + Fresnel transmission at both passes + tint).
  * Physical efficiency option: sinusoidal grooves, eta_m = J_m(a)^2, a = pi h n_c (cos_i' + cos_o') / lam,
    from a packed J^2 lookup image, per lobe; the mirror order gets J_0^2 integrated over 6 wavelengths.
  * Random numbers: White Noise 4D of the hit position (bit hash, fresh every sample; see docs/HOW_IT_WORKS.md).

Imported by scripts/build_holo_foil.py and tests/.
"""
import math
import os
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import spectral as SP  # noqa: E402

GROUP_NAME = "Diffraction Grating BSDF"
SUB = "DG "                      # prefix of the internal sub-groups
LUT_NAME = "DG Bessel J2 LUT"
DENOISE_GROUP = "DG Foil Denoise"
VERSION = "2.0"
ALUMINIUM = (0.913, 0.922, 0.924, 1.0)     # F0 of aluminium, linear Rec.709
O_MAX, H_MAX = 4, 4
QUALITY_LOBES = {1: (2, 2), 2: (4, 2), 3: (4, 3), 4: (4, 4)}   # Quality -> (order groups, wavelengths per group)
RENORM_ORDERS = 8                # orders included in the closed-form propagating share (= max of the Orders input)
DEFAULTS = dict(dist='GGX')


# ==============================================================================================
# node-building DSL with sections (frames) and a left-to-right layout
# ==============================================================================================
class NB:
    def __init__(self, tree):
        self.t = tree
        self.section = 'main'
        self.order = []          # (node, section)

    def node(self, idname, label=None, hide=False, **props):
        n = self.t.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        if label:
            n.label = label
        n.hide = hide
        self.order.append((n, self.section))
        return n

    def link(self, src, dst):
        self.t.links.new(src, dst)

    def _in(self, sock, val):
        if isinstance(val, bpy.types.NodeSocket):
            self.link(val, sock)
        elif val is not None:
            if isinstance(val, (int, float)) and hasattr(sock.default_value, '__len__'):
                n = len(sock.default_value)
                val = [float(val)] * n if n != 4 else [float(val)] * 3 + [1.0]
            sock.default_value = val

    def math(self, op, a, b=None, c=None, clamp=False, label=None):
        n = self.node('ShaderNodeMath', label=label, hide=True, operation=op, use_clamp=clamp)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        return n.outputs[0]

    def vmath(self, op, a, b=None, c=None, scale=None, label=None):
        n = self.node('ShaderNodeVectorMath', label=label, hide=True, operation=op)
        self._in(n.inputs[0], a)
        if b is not None:
            self._in(n.inputs[1], b)
        if c is not None:
            self._in(n.inputs[2], c)
        if scale is not None:
            self._in(n.inputs['Scale'], scale)
        return n.outputs['Value'] if op in ('DOT_PRODUCT', 'LENGTH', 'DISTANCE') else n.outputs['Vector']

    def add(self, a, b, **k): return self.math('ADD', a, b, **k)
    def sub(self, a, b, **k): return self.math('SUBTRACT', a, b, **k)
    def mul(self, a, b, **k): return self.math('MULTIPLY', a, b, **k)
    def div(self, a, b, **k): return self.math('DIVIDE', a, b, **k)
    def madd(self, a, b, c, **k): return self.math('MULTIPLY_ADD', a, b, c, **k)
    def gt(self, a, b, **k): return self.math('GREATER_THAN', a, b, **k)
    def lt(self, a, b, **k): return self.math('LESS_THAN', a, b, **k)
    def vscale(self, v, s, **k): return self.vmath('SCALE', v, scale=s, **k)

    def mix(self, fac, a, b, dtype='FLOAT', label=None):
        n = self.node('ShaderNodeMix', label=label, hide=True, data_type=dtype)
        if dtype == 'RGBA':
            n.blend_type = 'MIX'
        ins = {s.identifier: s for s in n.inputs}
        suf = {'FLOAT': 'Float', 'VECTOR': 'Vector', 'RGBA': 'Color'}[dtype]
        self._in(ins['Factor_Float'], fac)
        self._in(ins['A_' + suf], a)
        self._in(ins['B_' + suf], b)
        return {s.identifier: s for s in n.outputs}['Result_' + suf]

    def combine(self, x, y, z, label=None):
        n = self.node('ShaderNodeCombineXYZ', label=label, hide=True)
        for i, v in enumerate((x, y, z)):
            self._in(n.inputs[i], v)
        return n.outputs[0]

    def separate(self, v, label=None):
        n = self.node('ShaderNodeSeparateXYZ', label=label, hide=True)
        self.link(v, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def group(self, ng, label=None, **inputs):
        n = self.node('ShaderNodeGroup', label=label)
        n.node_tree = ng
        n.width = 200
        for k, v in inputs.items():
            self._in(n.inputs[k.replace('_', ' ')], v)
        return n

    def glossy(self, color, rough, normal, dist='GGX', label=None):
        n = self.node('ShaderNodeBsdfGlossy', label=label, distribution=dist)
        self._in(n.inputs['Color'], color)
        self._in(n.inputs['Roughness'], rough)
        self._in(n.inputs['Normal'], normal)
        return n.outputs[0]

    def mix_shader(self, fac, a, b, label=None):
        n = self.node('ShaderNodeMixShader', label=label)
        self._in(n.inputs[0], fac)
        if a is not None:
            self.link(a, n.inputs[1])
        self.link(b, n.inputs[2])
        return n.outputs[0]

    def add_shaders(self, shaders, label='Add'):
        """Balanced tree of Add Shader nodes."""
        while len(shaders) > 1:
            nxt = []
            for i in range(0, len(shaders) - 1, 2):
                n = self.node('ShaderNodeAddShader', label=label, hide=True)
                self.link(shaders[i], n.inputs[0])
                self.link(shaders[i + 1], n.inputs[1])
                nxt.append(n.outputs[0])
            if len(shaders) % 2:
                nxt.append(shaders[-1])
            shaders = nxt
        return shaders[0]

    def lut(self, x_norm, row, label=None):
        """J_|m|^2 from the packed Bessel image: x_norm = a / AMAX, row = |m| (exact row centre)."""
        W, H = SP.BESSEL_W, SP.BESSEL_H
        x = self.madd(x_norm, (W - 1.0) / W, 0.5 / W)
        y = self.madd(row, 1.0 / H, 0.5 / H)
        n = self.node('ShaderNodeTexImage', label=label or 'J^2 lookup', interpolation='Linear',
                      extension='EXTEND', hide=True)
        n.image = bessel_image()
        self.link(self.combine(x, y, 0.0), n.inputs['Vector'])
        return n.outputs['Color']      # R = G = B = J^2, read back as a float (luminance weights sum to 1)

    # ---- layout
    @staticmethod
    def _height(n):
        if n.hide:
            return 32
        if n.bl_idname == 'NodeGroupInput':
            return 40 + 22 * len([o for o in n.outputs if not o.hide])
        vis_in = [i for i in n.inputs if i.enabled and not i.hide]
        h = 46 + 22 * (len(vis_in) + len([o for o in n.outputs if o.enabled]))
        if n.bl_idname in ('ShaderNodeFloatCurve', 'ShaderNodeRGBCurve'):
            h += 240
        if n.bl_idname.startswith('ShaderNodeBsdf'):
            h += 40
        return h

    def layout(self, dx=200, split_inputs=True):
        """One horizontal band per section; inside a band, columns follow dependency depth (compacted);
        each band gets a frame and its own Group Input node showing only the sockets it uses."""
        nodes = [n for n, _ in self.order]
        sec_of = {n.name: s for n, s in self.order}
        incoming = {}
        for l in self.t.links:
            incoming.setdefault(l.to_node.name, []).append(l.from_node.name)
        depth = {}

        def d(name, stack=()):
            if name in depth:
                return depth[name]
            if name in stack:
                return 0
            src = [s for s in incoming.get(name, []) if self.t.nodes[s].bl_idname != 'NodeGroupInput']
            v = 0 if not src else 1 + max(d(s, stack + (name,)) for s in src)
            depth[name] = v
            return v

        for n in self.t.nodes:
            d(n.name)
        sections = []
        for _, s in self.order:
            if s not in sections and s != 'io':
                sections.append(s)
        gin = [n for n in self.t.nodes if n.bl_idname == 'NodeGroupInput']
        gout = [n for n in self.t.nodes if n.bl_idname == 'NodeGroupOutput']
        y0 = 0.0
        max_x = 0.0
        for s in sections:
            members = [n for n in nodes if sec_of.get(n.name) == s]
            if not members:
                continue
            cols = sorted(set(depth[n.name] for n in members))
            x0 = min(cols) * dx * 0.35
            band_h = 0.0
            for ci, c in enumerate(cols):
                y = y0 - 40
                for n in [m for m in members if depth[m.name] == c]:
                    n.location = (x0 + ci * dx + (40 if n.bl_idname == 'ShaderNodeGroup' else 0), y)
                    y -= self._height(n) + 14
                band_h = max(band_h, y0 - y)
                max_x = max(max_x, x0 + ci * dx + 260)
            # this band's own Group Input
            if split_inputs and gin:
                used = [l for l in self.t.links if l.from_node == gin[0] and sec_of.get(l.to_node.name) == s]
                if used:
                    g2 = self.t.nodes.new('NodeGroupInput')
                    pairs = [(l.from_socket.identifier, l.to_socket) for l in used]
                    for ident, to in pairs:          # a new link into an input replaces the old one
                        self.t.links.new(g2.outputs[ident], to)
                    for o in g2.outputs:
                        o.hide = not o.is_linked
                    g2.location = (x0 - 230, y0 - 40)
                    members.append(g2)
                    band_h = max(band_h, 40 + self._height(g2) + 40)
            fr = self.t.nodes.new('NodeFrame')
            fr.label = s
            fr.label_size = 16
            fr.shrink = True
            for n in members:
                loc = n.location.copy()
                n.parent = fr
                n.location_absolute = loc
            y0 -= band_h + 110
        for g in gin:
            if split_inputs and not any(l.from_node == g for l in self.t.links) and len(gin) >= 1:
                self.t.nodes.remove(g)
            else:
                g.location = (-300, 0)
        for g in gout:
            g.location = (max_x + 120, 0)


# ==============================================================================================
# interfaces
# ==============================================================================================
def new_tree(name, kind='ShaderNodeTree'):
    old = bpy.data.node_groups.get(name)
    if old:
        bpy.data.node_groups.remove(old)
    return bpy.data.node_groups.new(name, kind)


def sock(ng, io, name, typ, default=None, lo=None, hi=None, subtype=None, desc='', hide=False, panel=None):
    kw = dict(in_out=io, socket_type=typ)
    if panel is not None:
        kw['parent'] = panel
    s = ng.interface.new_socket(name, **kw)
    if subtype:
        s.subtype = subtype
    if default is not None and hasattr(s, 'default_value'):
        s.default_value = default
    if lo is not None:
        s.min_value = lo
    if hi is not None:
        s.max_value = hi
    s.description = desc
    if hide:
        s.hide_value = True
    return s


T = {'f': 'NodeSocketFloat', 'v': 'NodeSocketVector', 'c': 'NodeSocketColor', 's': 'NodeSocketShader'}


def simple_group(name, inputs, outputs, desc=''):
    """inputs/outputs: (name, type key[, default[, description]]). Vector inputs hide their value field."""
    ng = new_tree(name)
    ng.description = desc
    for spec in outputs:
        sock(ng, 'OUTPUT', spec[0], T[spec[1]], desc=spec[3] if len(spec) > 3 else '')
    for spec in inputs:
        sock(ng, 'INPUT', spec[0], T[spec[1]], default=spec[2] if len(spec) > 2 else None,
             desc=spec[3] if len(spec) > 3 else '', hide=(spec[1] == 'v'))
    b = NB(ng)
    b.section = 'io'
    gi = b.node('NodeGroupInput')
    go = b.node('NodeGroupOutput')
    b.section = 'main'
    return ng, b, gi.outputs, go.inputs


# ==============================================================================================
# data blocks: Bessel LUT image
# ==============================================================================================
def bessel_image():
    img = bpy.data.images.get(LUT_NAME)
    if img is not None:
        return img
    import numpy as np
    lut = SP.bessel_lut()                                   # (H, W): row m, column a
    H, W = lut.shape
    img = bpy.data.images.new(LUT_NAME, W, H, float_buffer=True, alpha=False)
    img.colorspace_settings.name = 'Non-Color'
    px = np.ones((H, W, 4), np.float32)
    px[..., 0] = px[..., 1] = px[..., 2] = lut
    img.pixels.foreach_set(px.ravel())
    tmp = os.path.join(bpy.app.tempdir or '/tmp', 'dg_bessel_lut.exr')
    img.file_format = 'OPEN_EXR'
    img.filepath_raw = tmp
    img.save()
    img.pack()
    img.filepath_raw = '//' + LUT_NAME + '.exr'
    img.use_fake_user = True
    try:
        os.remove(tmp)
    except OSError:
        pass
    return img


def set_curve(mapping, curve, xs, ys):
    """Piecewise-linear curve through (xs, ys) (VECTOR handles, no clipping, flat extension)."""
    pts = curve.points
    while len(pts) > 2:
        pts.remove(pts[1])
    pts[0].location = (float(xs[0]), float(ys[0]))
    pts[1].location = (float(xs[-1]), float(ys[-1]))
    for x, y in zip(xs[1:-1], ys[1:-1]):
        pts.new(float(x), float(y))
    for p in pts:
        p.handle_type = 'VECTOR'
    mapping.use_clip = False
    mapping.extend = 'HORIZONTAL'
    mapping.update()


# ==============================================================================================
# sub-groups
# ==============================================================================================
def grp_frame():
    ng, b, I, O = simple_group(
        SUB + "Grating Frame",
        [('Normal', 'v'), ('Tangent', 'v'), ('Angle', 'f', 0.0)],
        [('Normal', 'v'), ('Grating Vector', 'v', None, 'g: perpendicular to the grooves'),
         ('Groove Vector', 'v', None, 'g2 = N x g'), ('Incoming', 'v', None, 'wo: towards the viewer'),
         ('View', 'v', None, '(-wo.g, -wo.g2, wo.N)')],
        "Shading frame: N, grating vector g (perpendicular to the grooves, rotated by Angle) and g2 = N x g")
    b.section = 'Normal: input if connected, else geometry normal'
    geo = b.node('ShaderNodeNewGeometry', hide=True)
    use_n = b.gt(b.vmath('LENGTH', I['Normal']), 0.5)
    N = b.vmath('NORMALIZE', b.mix(use_n, geo.outputs['Normal'], I['Normal'], 'VECTOR'), label='N')
    b.section = 'Tangent: input, else UV tangent, else world X (Y if the surface faces X)'
    uvt = b.node('ShaderNodeTangent', direction_type='UV_MAP', hide=True).outputs['Tangent']
    nx = b.separate(N)[0]
    fallback = b.mix(b.gt(b.math('ABSOLUTE', nx), 0.9), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), 'VECTOR')
    t1 = b.mix(b.gt(b.vmath('LENGTH', uvt), 0.1), fallback, uvt, 'VECTOR')
    Tn = b.mix(b.gt(b.vmath('LENGTH', I['Tangent']), 0.1), t1, I['Tangent'], 'VECTOR', label='T')
    b.section = 'g = T projected on the surface, rotated by Angle about N;  g2 = N x g'
    tdn = b.vmath('DOT_PRODUCT', Tn, N)
    Tp = b.vmath('NORMALIZE', b.vmath('SUBTRACT', Tn, b.vscale(N, tdn)))
    rot = b.node('ShaderNodeVectorRotate', rotation_type='AXIS_ANGLE', hide=True)
    b.link(Tp, rot.inputs['Vector'])
    b.link(N, rot.inputs['Axis'])
    b.link(I['Angle'], rot.inputs['Angle'])
    g = rot.outputs[0]
    g2 = b.vmath('CROSS_PRODUCT', N, g, label='g2')
    b.section = 'view direction in the grating frame'
    wo = geo.outputs['Incoming']
    view = b.combine(b.mul(b.vmath('DOT_PRODUCT', wo, g), -1.0), b.mul(b.vmath('DOT_PRODUCT', wo, g2), -1.0),
                     b.math('MAXIMUM', b.vmath('DOT_PRODUCT', wo, N), 1e-4), label='View')
    for k, v in (('Normal', N), ('Grating Vector', g), ('Groove Vector', g2), ('Incoming', wo), ('View', view)):
        b.link(v, O[k])
    b.layout()
    return ng


def grp_order_sampler():
    ng, b, I, O = simple_group(
        SUB + "Order Sampler",
        [('Random', 'f', 0.5), ('Distribution A', 'v', None, '(z, z threshold, S)'),
         ('Distribution B', 'v', None, '(-(1-F)/2F, ln F, F)'), ('Max Order', 'f', 3.0)],
        [('Order', 'f'), ('Abs Order', 'f'), ('Probability', 'f')],
        "Inverse CDF of the two-sided truncated geometric P(m) = (m == 0 ? z : F^|m|) / S, |m| <= M")
    b.section = 'unpack;  sign and folded uniform t = |2u - 1|'
    z, zt, S = b.separate(I['Distribution A'])
    negc1, lnf, F = b.separate(I['Distribution B'])
    u = I['Random']
    sgn = b.madd(b.gt(u, 0.5), 2.0, -1.0, label='sign')
    t = b.math('ABSOLUTE', b.madd(u, 2.0, -1.0))
    b.section = '|m| = 0 if t S <= z, else 1 + floor(ln(1 - (t S - z)(1 - F)/2F) / ln F)'
    ts = b.mul(t, S)
    nz = b.gt(ts, zt, label='nonzero')
    A = b.madd(b.sub(ts, z), negc1, 1.0)
    k = b.math('FLOOR', b.div(b.math('LOGARITHM', A, math.e), lnf))
    j = b.mul(b.math('MINIMUM', b.add(k, 1.0), I['Max Order']), nz, label='|m|')
    b.section = 'probability of the signed order'
    p = b.div(b.mix(b.gt(j, 0.5), z, b.math('POWER', F, j)), S)
    b.link(b.mul(j, sgn), O['Order'])
    b.link(j, O['Abs Order'])
    b.link(p, O['Probability'])
    b.layout()
    return ng


def grp_spectrum(pdf_mode='sum', curves=None):
    """u -> wavelength (nm) and colour weight c(lam)/p(lam)/3, with spectral saturation."""
    ng, b, I, O = simple_group(
        SUB + "Spectrum",
        [('Random', 'f', 0.5), ('Saturation', 'f', 1.0)],
        [('Wavelength', 'f'), ('Colour', 'c', None, 'c(lambda)/p(lambda)/3, saturation applied')],
        "Importance-sampled wavelength (pdf ~ R+G+B, 10 % floor) and its colour weight / 3; Saturation mixes toward "
        "the same-luminance grey")
    b.section = 'inverse CDF of p(lambda), 380..780 nm (Float Curve)'
    fc = b.node('ShaderNodeFloatCurve', label='lambda(u)')
    b.link(I['Random'], fc.inputs['Value'])
    lam = b.madd(fc.outputs[0], SP.LAM_MAX - SP.LAM_MIN, SP.LAM_MIN, label='nm')
    b.section = 'colour weight c(lambda)/p(lambda)/3 (RGB Curves), saturation'
    rc = b.node('ShaderNodeRGBCurve', label='weight(u)')
    b.link(I['Random'], rc.inputs['Color'])
    lum = b.vmath('DOT_PRODUCT', rc.outputs[0], tuple(SP.LUM), label='luminance')
    col = b.mix(I['Saturation'], lum, rc.outputs[0], 'RGBA', label='saturation')
    b.link(lam, O['Wavelength'])
    b.link(col, O['Colour'])
    fill_spectrum_curves(fc, rc, curves, pdf_mode)
    b.layout()
    return ng


def fill_spectrum_curves(fc, rc, curves=None, pdf_mode='sum'):
    """Bake lambda(u) and w(u)/3 into the curve nodes, read back exactly what Cycles samples (256 points),
    and fit the weights to that effective wavelength table so the estimator stays exact."""
    import numpy as np
    SP.PDF_MODE = pdf_mode
    u, lam_n, _ = SP.curve_tables(256)
    SP.PDF_MODE = 'sum'
    set_curve(fc.mapping, fc.mapping.curves[0], u, lam_n)
    lam_eff = np.array([fc.mapping.evaluate(fc.mapping.curves[0], float(t)) for t in u])
    lam_eff = SP.LAM_MIN + (SP.LAM_MAX - SP.LAM_MIN) * lam_eff
    w = SP.weights_for(lam_eff)
    for ch in range(3):
        set_curve(rc.mapping, rc.mapping.curves[ch], u, w[:, ch])
    rc.mapping.update()
    if curves is not None:
        curves['lam_eff'] = lam_eff
        curves['w_eff'] = np.array([[rc.mapping.evaluate(rc.mapping.curves[ch], float(t)) for ch in range(3)]
                                    for t in u])


def grp_coat(dist):
    ng, b, I, O = simple_group(
        SUB + "Coat",
        [('Normal', 'v'), ('View', 'v'), ('Coat', 'f', 0.0), ('Coat Roughness', 'f', 0.03),
         ('Coat IOR', 'f', 1.5), ('Coat Tint', 'c', (1, 1, 1, 1))],
        [('BSDF', 's'), ('Base Factor', 'c', None, 'transmission at the view angle x tint'),
         ('Coat Optics', 'v', None, '(n^2 - 1, 1/n, cos of the view angle inside the coat)'),
         ('Coat Exit', 'v', None, '(Schlick base, Schlick negative slope, transmission at the view angle)')],
        "Clear coat: dielectric reflection lobe, Fresnel transmission and tint for the foil underneath")
    c = I['Coat']
    b.section = 'coat reflection (dielectric Fresnel at the view angle)'
    fr = b.node('ShaderNodeFresnel', hide=True)
    b.link(I['Coat IOR'], fr.inputs['IOR'])
    b.link(I['Normal'], fr.inputs['Normal'])
    fd = b.mul(fr.outputs[0], c, label='Coat x F')
    b.link(b.glossy(fd, I['Coat Roughness'], I['Normal'], dist, label='Coat reflection'), O['BSDF'])
    b.section = 'transmission into the foil, effective index, tint'
    cos_o = b.separate(I['View'])[2]
    t_o = b.sub(1.0, fd, label='T view')
    n = b.madd(c, b.sub(I['Coat IOR'], 1.0), 1.0, label='n_c')
    n2m1 = b.madd(n, n, -1.0)
    inv_n = b.div(1.0, n)
    cosi = b.mul(b.math('SQRT', b.madd(cos_o, cos_o, n2m1)), inv_n, label="cos' view")
    tint = b.vmath('POWER', I['Coat Tint'], b.div(1.0, cosi))
    tint = b.mix(c, (1.0, 1.0, 1.0, 1.0), tint, 'RGBA', label='tint')
    f0 = b.math('POWER', b.div(b.sub(I['Coat IOR'], 1.0), b.add(I['Coat IOR'], 1.0)), 2.0, label='F0 coat')
    b.link(b.vscale(tint, t_o), O['Base Factor'])
    b.link(b.combine(n2m1, inv_n, cosi), O['Coat Optics'])
    b.link(b.combine(b.sub(1.0, b.mul(c, f0)), b.mul(b.mul(c, b.sub(1.0, f0)), -1.0), t_o), O['Coat Exit'])
    b.layout()
    return ng


LOBE_INPUTS = [('Wavelength', 'f', 550.0), ('Colour', 'c', (1, 1, 1, 1)), ('Order', 'v', None, '(m/d, n/d, -)'),
               ('Abs Order', 'v', None, '(|m|, |n|, -)'), ('Lattice Vector', 'v', None, 'Q = (m g + n g2)/d'),
               ('Normal', 'v'), ('Incoming', 'v'), ('View', 'v'), ('Roughness', 'f', 0.1),
               ('Art Weight', 'f', 0.0), ('Physical Weight', 'f', 0.0), ('Mode', 'v', None, '(2D, physical, phase)'),
               ('Coat Optics', 'v'), ('Coat Exit', 'v'), ('Fresnel A', 'c', (1, 1, 1, 1)),
               ('Fresnel B', 'c', (0, 0, 0, 1))]


def grp_lobe(dist):
    ng, b, I, O = simple_group(SUB + "Diffraction Lobe", LOBE_INPUTS, [('BSDF', 's')],
                               "One diffraction order at one wavelength: grating-equation direction, half-vector "
                               "GGX lobe, metal Fresnel, coat transmission, art or physical (Bessel) efficiency")
    lam = I['Wavelength']
    b.section = 'grating equation: wi_t = lam q - wo_t ;  wi_z = sqrt(1 - |wi_t|^2)'
    qx, qy, _ = b.separate(I['Order'])
    nox, noy, cos_o = b.separate(I['View'])
    wix = b.madd(lam, qx, nox, label='wi.g')
    wiy = b.madd(lam, qy, noy, label='wi.g2')
    s2 = b.madd(wix, wix, b.mul(wiy, wiy))
    wiz = b.math('SQRT', b.sub(1.0, s2), label='wi.N')
    fade = b.mul(wiz, 25.0, clamp=True, label='evanescent fade')
    b.section = 'half vector h = normalize(lam Q + (cos_o + wi.N) N)'
    hn = b.vscale(I['Normal'], b.add(cos_o, wiz))
    h = b.vmath('NORMALIZE', b.vmath('MULTIPLY_ADD', I['Lattice Vector'], lam, hn), label='h')
    b.section = 'metal Fresnel at the lobe angle, coat transmission at the exit angle'
    t5 = b.math('POWER', b.sub(1.0, b.vmath('DOT_PRODUCT', I['Incoming'], h)), 5.0)
    fk = b.vmath('MULTIPLY_ADD', I['Fresnel B'], t5, I['Fresnel A'], label='F(wo.h)')
    sb, sn, _ = b.separate(I['Coat Exit'])
    ti = b.madd(b.math('POWER', b.sub(1.0, wiz), 5.0), sn, sb, label='T exit')
    scal = b.mul(fade, ti)
    col = b.vmath('MULTIPLY', I['Colour'], fk)
    b.section = 'art efficiency'
    art = b.glossy(b.vscale(col, b.mul(scal, I['Art Weight'])), I['Roughness'], h, dist, label='Lobe (art)')
    b.section = "physical efficiency: J_|m|(a)^2 J_|n|(a)^2,  a = pi h n (cos_i' + cos_o') / lam"
    is2d, phys, phase = b.separate(I['Mode'])
    n2m1, inv_n, cos_oi = b.separate(I['Coat Optics'])
    am, an, _ = b.separate(I['Abs Order'])
    ci = b.mul(b.math('SQRT', b.madd(wiz, wiz, n2m1)), inv_n, label="cos' exit")
    a = b.mul(b.div(b.add(ci, cos_oi), lam), phase, label='a / AMAX')
    jm = b.lut(a, am, label='J_m^2')
    jn = b.madd(is2d, b.sub(b.lut(a, an, label='J_n^2'), 1.0), 1.0)
    eta = b.mul(b.mul(jm, jn), b.mul(scal, I['Physical Weight']), label='eta / P')
    phy = b.glossy(b.vscale(col, eta), I['Roughness'], h, dist, label='Lobe (physical)')
    b.section = 'art or physical (the unused branch is skipped)'
    b.link(b.mix_shader(phys, art, phy, label='art | physical'), O['BSDF'])
    b.layout()
    return ng


def grp_mirror(dist, quad=None, shares=None):
    """Order 0. Art mode: strength 1 - D f_prop, i.e. its own share 1 - D plus the energy D (1 - f_prop) of every
    diffracted order that is evanescent at this view angle and wavelength (exact, closed form; 1D and 2D branches,
    only the active one runs). Physical mode: J_0(a)^2 integrated over the spectrum at fixed wavelengths."""
    ins = [('Normal', 'v'), ('View', 'v'), ('Roughness', 'f', 0.1), ('Fresnel A', 'c', (1, 1, 1, 1)),
           ('Fresnel B', 'c', (0, 0, 0, 1)), ('Coat Exit', 'v'), ('Coat Optics', 'v'), ('Mask', 'f', 1.0),
           ('Mode', 'v'), ('Evanescence', 'v', None, '(d, F, D)'), ('Max Order', 'f', 3.0),
           ('Saturation', 'f', 1.0)]
    name = SUB + "Mirror Order" + ('' if quad is None else ' (GT)')
    ng, b, I, O = simple_group(name, ins, [('BSDF', 's')],
                               "Order 0: mirror lobe with metal Fresnel and coat transmission. It also takes the energy "
                               "of the diffracted orders that are evanescent (art mode)")
    b.section = 'metal Fresnel at the view angle, coat transmission (in and out)'
    cos_o = b.separate(I['View'])[2]
    t_o = b.separate(I['Coat Exit'])[2]
    t5 = b.math('POWER', b.sub(1.0, cos_o), 5.0)
    base = b.vscale(b.vmath('MULTIPLY_ADD', I['Fresnel B'], t5, I['Fresnel A']), t_o)
    D = b.separate(I['Evanescence'])[2]
    kw = dict(View=I['View'], Evanescence=I['Evanescence'], Max_Order=I['Max Order'], Saturation=I['Saturation'])
    b.section = 'art, 1D: strength = 1 - D x (propagating share)'
    f1 = b.group(shares['1d'], label='Propagating Share 1D', **kw)
    art1 = b.glossy(b.vmath('MULTIPLY', base, b.vmath('SUBTRACT', (1, 1, 1), b.vscale(f1.outputs['Share'], D))),
                    I['Roughness'], I['Normal'], dist, label='Order 0 (art, 1D)')
    b.section = 'art, 2D: strength = 1 - D x (propagating share)'
    f2 = b.group(shares['2d'], label='Propagating Share 2D', **kw)
    art2 = b.glossy(b.vmath('MULTIPLY', base, b.vmath('SUBTRACT', (1, 1, 1), b.vscale(f2.outputs['Share'], D))),
                    I['Roughness'], I['Normal'], dist, label='Order 0 (art, 2D)')
    lams, ws = quad if quad is not None else SP.order0_quadrature(6)
    b.section = 'physical: sum_j W_j J_0(a_j)^2 (squared again for a 2D lattice)'
    is2d, phys, phase = b.separate(I['Mode'])
    cos_oi = b.separate(I['Coat Optics'])[2]
    a0 = b.mul(b.mul(cos_oi, 2.0), phase)
    acc = None
    for lam, w in zip(lams, ws):
        j = b.lut(b.mul(a0, 1.0 / float(lam)), 0.0, label='J_0^2 @ %.0f nm' % lam)
        j = b.mul(j, b.madd(is2d, b.sub(j, 1.0), 1.0))
        term = b.vscale(tuple(float(x) for x in w), j)
        acc = term if acc is None else b.vmath('ADD', acc, term)
    eff = b.vmath('ADD', b.vscale(acc, I['Mask']), b.sub(1.0, I['Mask']), label='1 - Mask (1 - eta_0)')
    phy = b.glossy(b.vmath('MULTIPLY', base, eff), I['Roughness'], I['Normal'], dist, label='Order 0 (physical)')
    b.section = 'art 1D | art 2D | physical (only the active branch runs)'
    art = b.mix_shader(is2d, art1, art2, label='1D | 2D')
    b.link(b.mix_shader(phys, art, phy, label='art | physical'), O['BSDF'])
    b.layout()
    return ng


def cumulative_curve_node(b):
    """Returns f(u_socket, label) -> RGB socket: cumulative white energy per channel below lambda(u)."""
    u, C = SP.cumulative_colour(256)

    def f(x, label):
        rc = b.node('ShaderNodeRGBCurve', label=label, hide=True)
        b.link(x, rc.inputs['Color'])
        for ch in range(3):
            set_curve(rc.mapping, rc.mapping.curves[ch], u, C[:, ch])
        return rc.outputs[0]
    return f


def grp_params():
    """Per-shading-point scalars from the user inputs: order distribution, lobe weights, Fresnel factors and the
    constants the mirror order needs to take over the energy of evanescent orders."""
    ins = [('Pitch', 'f', 1500.0), ('Pitch Multiplier', 'f', 1.0), ('Orders', 'f', 3.0), ('Grating Type', 'f', 0.0),
           ('Mask', 'f', 1.0), ('Order 0 Strength', 'f', 0.3), ('Efficiency Falloff', 'f', 0.6),
           ('Groove Depth', 'f', 0.0), ('Lobes', 'f', 8.0), ('Color', 'c', ALUMINIUM), ('View', 'v'),
           ('Coat Optics', 'v'), ('Base Factor', 'c', (1, 1, 1, 1))]
    outs = [('Distribution A', 'v'), ('Distribution B', 'v'), ('Max Order', 'f'),
            ('Weights', 'v', None, '(art weight, physical weight, 1/d)'),
            ('Mode', 'v', None, '(2D, physical, pi h n / AMAX)'),
            ('Evanescence', 'v', None, '(d in nm, art falloff F, diffracted share D = Mask (1 - eta_0))'),
            ('Fresnel A', 'c'), ('Fresnel B', 'c')]
    ng, b, I, O = simple_group(SUB + "Parameters", ins, outs,
                               "Order distribution, lobe weights, physical-mode constants, Fresnel factors")
    b.section = 'grating'
    d = b.math('MAXIMUM', b.mul(I['Pitch'], I['Pitch Multiplier']), 50.0, label='d (nm)')
    inv_d = b.div(1.0, d, label='1/d')
    M = b.math('MINIMUM', b.math('MAXIMUM', b.math('ROUND', I['Orders']), 1.0), float(RENORM_ORDERS), label='M')
    is2d = b.gt(I['Grating Type'], 0.5, label='2D')
    phys = b.gt(I['Groove Depth'], 0.0, label='physical')
    b.section = 'physical mode: phase constant and an order proposal fitted to J at 550 nm'
    _, inv_n, cos_oi = b.separate(I['Coat Optics'])
    kp = b.mul(b.div(I['Groove Depth'], inv_n), math.pi / SP.BESSEL_AMAX, label='pi h n / AMAX')
    aref = b.mul(b.mul(cos_oi, 2.0 / 550.0), kp)
    j0 = b.add(b.lut(aref, 0.0, label='J0^2 ref'), 1e-4)
    j1 = b.add(b.lut(aref, 1.0, label='J1^2 ref'), 1e-4)
    j2 = b.add(b.lut(aref, 2.0, label='J2^2 ref'), 1e-4)
    fp = b.math('MINIMUM', b.math('MAXIMUM', b.div(j2, j1), 0.05), 0.9, label='F proposal')
    zp = b.mul(b.div(j0, j1), fp, label='z proposal')
    b.section = 'order distribution P(m) = (m == 0 ? z : F^|m|) / S'
    fart = b.math('MINIMUM', b.math('MAXIMUM', I['Efficiency Falloff'], 0.02), 0.98, label='F art')
    F = b.mix(phys, fart, fp, label='F')
    z = b.mul(is2d, b.mix(phys, 1.0, zp), label='z (0 in 1D)')
    t1 = b.div(b.mul(F, b.sub(1.0, b.math('POWER', F, M))), b.sub(1.0, F), label='sum F^j')
    S = b.madd(t1, 2.0, z, label='S')
    negc1 = b.div(b.sub(F, 1.0), b.mul(F, 2.0))
    b.section = 'lobe weights (x3 undoes the colour weight / 3; / number of lobes)'
    diff = b.mul(I['Mask'], b.sub(1.0, I['Order 0 Strength']), label='diffracted share D')
    s2 = b.mul(S, S)
    art_w = b.mul(diff, b.madd(is2d, b.sub(b.div(s2, b.sub(s2, 1.0)), 1.0), 1.0))
    art_w = b.mul(art_w, b.div(3.0, I['Lobes']), label='art weight')
    phys_w = b.mul(I['Mask'], b.div(3.0, I['Lobes']), label='physical weight')
    b.section = 'metal Schlick Fresnel  F = A + B (1 - cos)^5, coat folded in'
    fa = b.vmath('MULTIPLY', I['Color'], I['Base Factor'])
    fb = b.vmath('MULTIPLY', b.vmath('SUBTRACT', (1.0, 1.0, 1.0), I['Color']), I['Base Factor'])
    b.link(b.combine(z, b.sub(z, 1e-6), S), O['Distribution A'])
    b.link(b.combine(negc1, b.math('LOGARITHM', F, math.e), F), O['Distribution B'])
    b.link(M, O['Max Order'])
    b.link(b.combine(art_w, phys_w, inv_d), O['Weights'])
    b.link(b.combine(is2d, phys, kp), O['Mode'])
    b.link(b.combine(d, fart, diff), O['Evanescence'])
    b.link(fa, O['Fresnel A'])
    b.link(fb, O['Fresnel B'])
    b.layout()
    return ng


SHARE_INPUTS = [('View', 'v'), ('Evanescence', 'v', None, '(d, F, D)'), ('Max Order', 'f', 3.0),
                ('Saturation', 'f', 1.0)]


def grp_share_1d():
    """Share of the diffracted (art-mode) energy that propagates, 1D: order +-j propagates for
    lambda < d (sqrt(1 - oy^2) +- ox) / j, so the share is sum_j eta_j [C(lambda_j+) + C(lambda_j-)], C = cumulative
    white spectrum per channel. Exact for |m| <= 8."""
    ng, b, I, O = simple_group(SUB + "Propagating Share 1D", SHARE_INPUTS, [('Share', 'c')],
                               "Fraction of the diffracted energy (per RGB channel) in orders that propagate (1D)")
    b.section = 'order +-j propagates below lambda = d (sqrt(1 - oy^2) +- ox) / j'
    nox, noy, _ = b.separate(I['View'])
    d, F, _ = b.separate(I['Evanescence'])
    M = I['Max Order']
    r = b.math('SQRT', b.madd(noy, b.mul(noy, -1.0), 1.0), label='sqrt(1 - oy^2)')
    rp = b.sub(r, nox, label='r + ox')          # nox = -ox
    rm = b.add(r, nox, label='r - ox')
    t1 = b.div(b.mul(F, b.sub(1.0, b.math('POWER', F, M))), b.sub(1.0, F), label='sum F^j')
    cum = cumulative_curve_node(b)
    acc = None
    for j in range(1, RENORM_ORDERS + 1):
        b.section = 'order +-%d' % j
        k = b.mul(d, 1.0 / ((SP.LAM_MAX - SP.LAM_MIN) * j))
        cp = cum(b.madd(rp, k, -SP.LAM_MIN / (SP.LAM_MAX - SP.LAM_MIN)), 'C(+%d)' % j)
        cm = cum(b.madd(rm, k, -SP.LAM_MIN / (SP.LAM_MAX - SP.LAM_MIN)), 'C(-%d)' % j)
        eta = b.mul(b.div(b.math('POWER', F, float(j)), b.mul(t1, 2.0)), b.lt(float(j) - 0.5, M))
        term = b.vscale(b.vmath('ADD', cp, cm), eta)
        acc = term if acc is None else b.vmath('ADD', acc, term)
    b.section = 'desaturated lobes carry luminance-weighted energy'
    b.link(b.mix(I['Saturation'], b.vmath('DOT_PRODUCT', acc, tuple(SP.LUM)), acc, 'RGBA'), O['Share'])
    b.layout()
    return ng


def grp_share_2d():
    """2D square lattice: order (m, n) propagates for lambda < d (s - b')/k with k = m^2 + n^2,
    b' = m (-ox) + n (-oy), s = sqrt(b'^2 + k cos_o^2) (and d (s + b')/k for (-m, -n)). Exact for |m|, |n| <= 3; the
    (small) energy of higher orders, when Orders > 3, is assumed to propagate like the |m| or |n| = 3 ring.
    Cumulative colours are summed per efficiency class (|m|, |n|) first and scaled once per class."""
    ng, b, I, O = simple_group(SUB + "Propagating Share 2D", SHARE_INPUTS, [('Share', 'c')],
                               "Fraction of the diffracted energy (per RGB channel) in orders that propagate (2D)")
    b.section = 'constants: eta(m,n) = F^(|m|+|n|) / (S^2 - 1), S = 1 + 2 sum F^j'
    nox, noy, cos_o = b.separate(I['View'])
    d, F, _ = b.separate(I['Evanescence'])
    M = I['Max Order']
    c = b.mul(cos_o, cos_o, label='cos_o^2')
    t1 = b.div(b.mul(F, b.sub(1.0, b.math('POWER', F, M))), b.sub(1.0, F))
    S = b.madd(t1, 2.0, 1.0)
    norm = b.div(1.0, b.madd(S, S, -1.0), label='1/(S^2 - 1)')
    nn = {n: b.mul(noy, float(n)) for n in range(1, 4)}
    kc, kd = {}, {}
    cum = cumulative_curve_node(b)
    off = -SP.LAM_MIN / (SP.LAM_MAX - SP.LAM_MIN)
    classes = {}                          # (|m|, |n|) sorted -> list of summed C(+) + C(-) sockets
    pairs = [(m, n) for m in range(0, 4) for n in range(-3, 4) if (m > 0 or n > 0)]
    for m, n in pairs:
        b.section = 'orders (%d, %d) and (%d, %d)' % (m, n, -m, -n)
        k = m * m + n * n
        if k not in kc:
            kc[k] = b.mul(c, float(k))
            kd[k] = b.mul(d, 1.0 / ((SP.LAM_MAX - SP.LAM_MIN) * k))
        if m == 0:
            bp = nn[n]
        elif n == 0:
            bp = b.mul(nox, float(m))
        else:
            bp = b.madd(nox, float(m), nn[abs(n)] if n > 0 else b.mul(nn[abs(n)], -1.0))
        sq = b.math('SQRT', b.madd(bp, bp, kc[k]))
        c1 = cum(b.madd(b.sub(sq, bp), kd[k], off), 'C(%d,%d)' % (m, n))
        c2 = cum(b.madd(b.add(sq, bp), kd[k], off), 'C(%d,%d)' % (-m, -n))
        key = tuple(sorted((abs(m), abs(n))))
        classes.setdefault(key, []).append(b.vmath('ADD', c1, c2))
    b.section = 'efficiency classes (|m|, |n|): eta x sum of C over the class; orders beyond |3| like the |3| ring'
    ind = {0: 1.0}
    for j in range(1, 4):
        ind[j] = b.lt(float(j) - 0.5, M)
    acc_in = acc3 = eta3 = eta_all = None
    for (i, j), socks in sorted(classes.items()):
        tot = socks[0]
        for x in socks[1:]:
            tot = b.vmath('ADD', tot, x)
        e = b.mul(b.math('POWER', F, float(i + j)), norm)
        gate = ind[j] if i == 0 else b.mul(ind[i], ind[j])
        e = b.mul(e, gate, label='eta(%d,%d)' % (i, j))
        term = b.vscale(tot, e)
        cnt = 2 * len(socks)                 # orders in the class
        ec = b.mul(e, float(cnt))
        eta_all = ec if eta_all is None else b.add(eta_all, ec)
        if max(i, j) == 3:
            acc3 = term if acc3 is None else b.vmath('ADD', acc3, term)
            eta3 = ec if eta3 is None else b.add(eta3, ec)
        else:
            acc_in = term if acc_in is None else b.vmath('ADD', acc_in, term)
    beyond = b.math('MAXIMUM', b.sub(1.0, eta_all), 0.0, label='energy beyond |3|')
    ring3 = b.vscale(acc3, b.div(beyond, b.math('MAXIMUM', eta3, 1e-9)))
    share = b.vmath('ADD', b.vmath('ADD', acc_in, acc3), ring3)
    b.section = 'desaturated lobes carry luminance-weighted energy'
    b.link(b.mix(I['Saturation'], b.vmath('DOT_PRODUCT', share, tuple(SP.LUM)), share, 'RGBA'), O['Share'])
    b.layout()
    return ng


def grp_order_group(G):
    """One stratified order (m, n) and up to H_MAX wavelengths for it. Lobes 3 and 4 are skipped below Quality 3/4."""
    ins = [('Index', 'f', 0.0, 'order group a = 0..3'), ('Stratum', 'f', 0.0, 'van der Corput offset for n'),
           ('Random', 'v', None, '(xi m, xi n, xi lambda)'), ('Budget', 'v', None, '(1/O, 1/H, H)'),
           ('Normal', 'v'), ('Grating Vector', 'v'), ('Groove Vector', 'v'), ('Incoming', 'v'), ('View', 'v'),
           ('Roughness', 'f', 0.1), ('Saturation', 'f', 1.0), ('Distribution A', 'v'), ('Distribution B', 'v'),
           ('Max Order', 'f', 3.0), ('Weights', 'v'), ('Mode', 'v'), ('Coat Optics', 'v'), ('Coat Exit', 'v'),
           ('Fresnel A', 'c', (1, 1, 1, 1)), ('Fresnel B', 'c', (0, 0, 0, 1))]
    ng, b, I, O = simple_group(SUB + "Order Group", ins, [('BSDF', 's')],
                               "Stratified order (m, n) shared by up to four stratified, importance-sampled "
                               "wavelengths; returns the sum of their lobes")
    b.section = 'stratified order: m from (a + xi_m)/O, n from frac(xi_n + stratum)'
    xm, xn, xl = b.separate(I['Random'])
    inv_o, inv_h, h_act = b.separate(I['Budget'])
    art_w, phys_w, inv_d = b.separate(I['Weights'])
    is2d = b.separate(I['Mode'])[0]
    samp = dict(Distribution_A=I['Distribution A'], Distribution_B=I['Distribution B'], Max_Order=I['Max Order'])
    sm = b.group(G['sampler'], label='Order m', Random=b.mul(b.add(xm, I['Index']), inv_o), **samp)
    sn = b.group(G['sampler'], label='Order n (2D)', Random=b.math('FRACT', b.add(xn, I['Stratum'])), **samp)
    b.section = 'lattice vector Q = (m g + n g2)/d and weights / P(m, n)'
    n_ = b.mul(sn.outputs['Order'], is2d)
    an = b.mul(sn.outputs['Abs Order'], is2d)
    pn = b.madd(is2d, b.sub(sn.outputs['Probability'], 1.0), 1.0, label='P(n) (1 in 1D)')
    nonzero = b.gt(b.add(sm.outputs['Abs Order'], an), 0.5, label='(m,n) != 0')
    qx = b.mul(sm.outputs['Order'], inv_d, label='m/d')
    qy = b.mul(n_, inv_d, label='n/d')
    Q = b.vmath('MULTIPLY_ADD', I['Groove Vector'], qy, b.vscale(I['Grating Vector'], qx), label='Q')
    order = b.combine(qx, qy, 0.0, label='Order')
    absorder = b.combine(sm.outputs['Abs Order'], an, 0.0, label='Abs Order')
    artw = b.mul(art_w, nonzero)
    physw = b.div(b.mul(phys_w, nonzero), b.mul(sm.outputs['Probability'], pn), label='1/P')
    lobes = []
    for hh in range(H_MAX):
        b.section = 'wavelength %d: stratified u = frac(xi + (%d + a/O)/H)' % (hh + 1, hh)
        off = b.mul(b.madd(I['Index'], inv_o, float(hh)), inv_h)
        sp = b.group(G['spectrum'], label='Spectrum %d' % (hh + 1), Random=b.math('FRACT', b.add(xl, off)),
                     Saturation=I['Saturation'])
        lb = b.group(G['lobe'], label='Lobe %d' % (hh + 1), Wavelength=sp.outputs['Wavelength'],
                     Colour=sp.outputs['Colour'], Order=order, Abs_Order=absorder, Lattice_Vector=Q,
                     Normal=I['Normal'], Incoming=I['Incoming'], View=I['View'], Roughness=I['Roughness'],
                     Art_Weight=artw, Physical_Weight=physw, Mode=I['Mode'], Coat_Optics=I['Coat Optics'],
                     Coat_Exit=I['Coat Exit'], Fresnel_A=I['Fresnel A'], Fresnel_B=I['Fresnel B'])
        out = lb.outputs['BSDF']
        if hh >= 2:
            out = b.mix_shader(b.lt(float(hh), h_act), None, out, label='only if Quality >= %d' % (hh + 1))
        lobes.append(out)
    b.section = 'sum'
    b.link(b.add_shaders(lobes), O['BSDF'])
    b.layout(dx=230)
    return ng


LOBES_INPUTS = [('Random', 'v', None, '(xi m, xi n, xi lambda)'), ('Budget', 'v', None, '(1/O, 1/H, H)'),
                ('Normal', 'v'), ('Grating Vector', 'v'), ('Groove Vector', 'v'), ('Incoming', 'v'), ('View', 'v'),
                ('Roughness', 'f', 0.1), ('Saturation', 'f', 1.0), ('Distribution A', 'v'), ('Distribution B', 'v'),
                ('Max Order', 'f', 3.0), ('Weights', 'v'), ('Mode', 'v'), ('Coat Optics', 'v'), ('Coat Exit', 'v'),
                ('Fresnel A', 'c', (1, 1, 1, 1)), ('Fresnel B', 'c', (0, 0, 0, 1))]


def grp_lobes(G):
    """All diffracted lobes: O_MAX order groups; groups 3 and 4 are skipped when Quality = 1 (O = 2)."""
    ng, b, I, O = simple_group(SUB + "Diffraction Lobes", LOBES_INPUTS, [('BSDF', 's')],
                               "Sum of the stratified diffraction lobes: %d order groups x up to %d wavelengths"
                               % (O_MAX, H_MAX))
    groups = []
    o_act = None
    for a in range(O_MAX):
        b.section = 'Order group %d%s' % (a + 1, '' if a < 2 else '  (skipped when Quality = 1)')
        kw = {k.replace(' ', '_'): I[k] for k, *_ in LOBES_INPUTS}
        og = b.group(G['order'], label='Order Group %d' % (a + 1), Index=float(a), Stratum=vdc(a), **kw)
        out = og.outputs['BSDF']
        if a >= 2:
            if o_act is None:
                o_act = b.div(1.0, b.separate(I['Budget'])[0], label='O')
            out = b.mix_shader(b.lt(float(a), o_act), None, out, label='only if O = 4')
        groups.append(out)
    b.section = 'Sum'
    b.link(b.add_shaders(groups), O['BSDF'])
    b.layout(dx=240)
    return ng


def vdc(i):
    """van der Corput radical inverse (base 2)."""
    r, f = 0.0, 0.5
    while i:
        if i & 1:
            r += f
        i >>= 1
        f *= 0.5
    return r


# ==============================================================================================
# the main group
# ==============================================================================================
def main_interface(ng):
    it = ng.interface
    sock(ng, 'OUTPUT', 'BSDF', 'NodeSocketShader', desc='The foil: metal grating (all orders) plus the optional coat')
    sock(ng, 'INPUT', 'Color', 'NodeSocketColor', ALUMINIUM,
         desc='Metal reflectance at normal incidence (F0); every order gets Schlick Fresnel at its own angle. '
              'Default: aluminium. White = v1 behaviour')
    p = it.new_panel('Grating', description='Groove spacing and direction', default_closed=False)
    sock(ng, 'INPUT', 'Pitch (nm)', 'NodeSocketFloat', 1500.0, 200.0, 100000.0, panel=p,
         desc='Groove / hole spacing d in nanometres. Holo foil 700-1500, CD 1600, "49 copies" films 1600-5000')
    sock(ng, 'INPUT', 'Pitch Multiplier', 'NodeSocketFloat', 1.0, 0.05, 20.0, panel=p,
         desc="Multiplies the pitch (plug a pattern's Pitch Multiplier here)")
    sock(ng, 'INPUT', 'Grating Angle', 'NodeSocketFloat', 0.0, subtype='ANGLE', panel=p,
         desc="Rotates the grating vector (perpendicular to the grooves) about the normal, starting from "
              "Tangent (plug a pattern's Angle here)")
    sock(ng, 'INPUT', 'Tangent', 'NodeSocketVector', hide=True, panel=p,
         desc='Grating vector before rotation (perpendicular to the grooves). Unconnected: the UV tangent, or '
              'world X projected on the surface when there are no UVs')
    sock(ng, 'INPUT', 'Grating Type', 'NodeSocketFloat', 0.0, 0.0, 1.0, panel=p,
         desc='0 = 1D grooves (orders m); 1 = 2D square lattice (orders m, n: rainbow copies in a grid)')
    sock(ng, 'INPUT', 'Mask', 'NodeSocketFloat', 1.0, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc="Where the grating is embossed: 0 = plain mirror foil, 1 = full grating (plug a pattern's Mask here)")
    p = it.new_panel('Orders', description='How energy is split between the diffraction orders', default_closed=False)
    sock(ng, 'INPUT', 'Orders', 'NodeSocketFloat', 3.0, 1.0, 8.0, panel=p,
         desc='Highest order |m| (and |n|) that is simulated')
    sock(ng, 'INPUT', 'Order 0 Strength', 'NodeSocketFloat', 0.3, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc='Art mode: share of the energy in the mirror reflection; the rest is diffracted. Energy of orders that '
              'cannot exist at the current angle (evanescent) also goes to the mirror. Ignored when Groove Depth > 0')
    sock(ng, 'INPUT', 'Efficiency Falloff', 'NodeSocketFloat', 0.6, 0.02, 0.98, subtype='FACTOR', panel=p,
         desc='Art mode: efficiency of order k+1 relative to order k. Ignored when Groove Depth > 0')
    sock(ng, 'INPUT', 'Groove Depth (nm)', 'NodeSocketFloat', 0.0, 0.0, 600.0, panel=p,
         desc='0 = art mode (the two inputs above). > 0 = physical mode: sinusoidal grooves of this '
              'peak-to-valley depth, Bessel J_m^2 efficiencies that depend on wavelength and angle. Foil ~100-200')
    p = it.new_panel('Appearance', description='Blur, colour and clear coat', default_closed=False)
    sock(ng, 'INPUT', 'Roughness', 'NodeSocketFloat', 0.15, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc='GGX roughness of every lobe (how much each copy of a light is blurred)')
    sock(ng, 'INPUT', 'Spectral Saturation', 'NodeSocketFloat', 1.0, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc='1 = spectral colours, 0 = luminance only (silver diffraction). The average stays white at any value')
    sock(ng, 'INPUT', 'Coat', 'NodeSocketFloat', 0.0, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc='Clear lacquer / laminate on top: dielectric reflection plus the matching Fresnel loss on every order')
    sock(ng, 'INPUT', 'Coat Roughness', 'NodeSocketFloat', 0.03, 0.0, 1.0, subtype='FACTOR', panel=p,
         desc='Roughness of the coat reflection')
    sock(ng, 'INPUT', 'Coat IOR', 'NodeSocketFloat', 1.5, 1.0, 3.0, panel=p, desc='Index of refraction of the coat')
    sock(ng, 'INPUT', 'Coat Tint', 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0), panel=p,
         desc='Colour of the coat at normal incidence (round trip). Yellow over aluminium = gold foil; '
              'translucent ink over foil')
    p = it.new_panel('Advanced', description='Noise / speed trade-off, randomness, shading normal', default_closed=True)
    sock(ng, 'INPUT', 'Quality', 'NodeSocketFloat', 3.0, 1.0, 4.0, panel=p,
         desc='Diffraction lobes per sample: 1 = 4, 2 = 8, 3 = 12 (default), 4 = 16. More lobes = less colour noise '
              'under lamps for ~5 % more render time per extra lobe; HDRI-only lighting gains little above 2')
    sock(ng, 'INPUT', 'Seed', 'NodeSocketFloat', 0.0, panel=p,
         desc='Decorrelates the per-sample random numbers of stacked gratings')
    sock(ng, 'INPUT', 'Normal', 'NodeSocketVector', hide=True, panel=p,
         desc='Shading normal (unconnected = geometry normal)')


def build_subgroups(dist='GGX', curves=None, lam_pdf='sum'):
    shares = {'1d': grp_share_1d(), '2d': grp_share_2d()}
    G = dict(frame=grp_frame(), sampler=grp_order_sampler(), spectrum=grp_spectrum(lam_pdf, curves),
             coat=grp_coat(dist), lobe=grp_lobe(dist), mirror=grp_mirror(dist, shares=shares), params=grp_params(),
             shares=shares)
    G['order'] = grp_order_group(G)
    G['lobes'] = grp_lobes(G)
    return G


def _front(b, gi, G, lobes):
    """Frame, coat and parameter groups shared by the production group and the ground truth."""
    b.section = 'Shading frame and clear coat'
    fr = b.group(G['frame'], label='Grating Frame', Normal=gi['Normal'], Tangent=gi['Tangent'],
                 Angle=gi['Grating Angle'])
    co = b.group(G['coat'], label='Coat', Normal=fr.outputs['Normal'], View=fr.outputs['View'], Coat=gi['Coat'],
                 Coat_Roughness=gi['Coat Roughness'], Coat_IOR=gi['Coat IOR'], Coat_Tint=gi['Coat Tint'])
    b.section = 'Order distribution, weights, Fresnel'
    pa = b.group(G['params'], label='Parameters', Pitch=gi['Pitch (nm)'], Pitch_Multiplier=gi['Pitch Multiplier'],
                 Orders=gi['Orders'], Grating_Type=gi['Grating Type'], Mask=gi['Mask'],
                 Order_0_Strength=gi['Order 0 Strength'], Efficiency_Falloff=gi['Efficiency Falloff'],
                 Groove_Depth=gi['Groove Depth (nm)'], Lobes=lobes, Color=gi['Color'], View=fr.outputs['View'],
                 Coat_Optics=co.outputs['Coat Optics'], Base_Factor=co.outputs['Base Factor'])
    return fr, co, pa


def build_group(name=GROUP_NAME, dist='GGX', curves=None, lam_pdf='sum', aov=True):
    """The production group: Quality 1..4 -> 4, 8, 12 or 16 diffracted lobes + order 0 + coat."""
    G = build_subgroups(dist, curves, lam_pdf)
    ng = new_tree(name)
    ng.description = ('Holographic foil / diffraction grating BSDF (v%s): grating equation per order, continuous '
                      'importance-sampled spectrum, 4-16 stratified lobes per sample (Quality). Pure SVM, GPU OK.'
                      % VERSION)
    main_interface(ng)
    b = NB(ng)
    b.section = 'io'
    gi = b.node('NodeGroupInput').outputs
    go = b.node('NodeGroupOutput').inputs
    b.section = 'Lobe budget: Quality 1..4 -> (order groups O, wavelengths per group H)'
    q = b.math('ROUND', b.math('MINIMUM', b.math('MAXIMUM', gi['Quality'], 1.0), 4.0), label='Quality')
    o_act = b.madd(b.gt(q, 1.5), 2.0, 2.0, label='O = 2 or 4')
    h_act = b.math('MAXIMUM', q, 2.0, label='H = 2..4')
    budget = b.combine(b.div(1.0, o_act), b.div(1.0, h_act), h_act, label='Budget')
    fr, co, pa = _front(b, gi, G, b.mul(o_act, h_act, label='lobes'))
    b.section = 'Random numbers: bit hash of the hit position (fresh every sample)'
    geo = b.node('ShaderNodeNewGeometry', hide=True)
    wn = b.node('ShaderNodeTexWhiteNoise', noise_dimensions='4D', label='xi (m, n, lambda)')
    b.link(geo.outputs['Position'], wn.inputs['Vector'])
    b.link(gi['Seed'], wn.inputs['W'])
    po = pa.outputs
    b.section = 'Diffraction lobes (Quality 1..4: 4, 8, 12 or 16 lobes)'
    lo = b.group(G['lobes'], label='Diffraction Lobes', Random=wn.outputs['Color'], Budget=budget,
                 Normal=fr.outputs['Normal'], Grating_Vector=fr.outputs['Grating Vector'],
                 Groove_Vector=fr.outputs['Groove Vector'], Incoming=fr.outputs['Incoming'], View=fr.outputs['View'],
                 Roughness=gi['Roughness'], Saturation=gi['Spectral Saturation'], Distribution_A=po['Distribution A'],
                 Distribution_B=po['Distribution B'], Max_Order=po['Max Order'], Weights=po['Weights'],
                 Mode=po['Mode'], Coat_Optics=co.outputs['Coat Optics'], Coat_Exit=co.outputs['Coat Exit'],
                 Fresnel_A=po['Fresnel A'], Fresnel_B=po['Fresnel B'])
    groups = [lo.outputs['BSDF']]
    b.section = 'Order 0 (mirror), coat, sum'
    mi = b.group(G['mirror'], label='Order 0', Normal=fr.outputs['Normal'], View=fr.outputs['View'],
                 Roughness=gi['Roughness'], Fresnel_A=po['Fresnel A'], Fresnel_B=po['Fresnel B'],
                 Coat_Exit=co.outputs['Coat Exit'], Coat_Optics=co.outputs['Coat Optics'], Mask=gi['Mask'],
                 Mode=po['Mode'], Evanescence=po['Evanescence'], Max_Order=po['Max Order'],
                 Saturation=gi['Spectral Saturation'])
    b.link(b.add_shaders([mi.outputs['BSDF']] + groups + [co.outputs['BSDF']]), go['BSDF'])
    if aov:
        b.section = 'AOV "dg_foil" for the foil-aware denoise (DG Foil Denoise)'
        av = b.node('ShaderNodeOutputAOV', label='AOV dg_foil')
        av.aov_name = 'dg_foil'
        av.inputs['Value'].default_value = 1.0
        b.link(gi['Color'], av.inputs['Color'])
    b.layout(dx=240)
    ng['dg_version'] = VERSION
    return ng


def build_ground_truth_chunk(name, k, chunk=56, n_lam=48, M=3, dist='GGX', **unused):
    """Brute-force reference: every 1D order +-1..+-M at n_lam fixed wavelengths (midpoint rule, colour = exact
    bin integral of c), no randomness. Chunk k holds lobes [k*chunk, (k+1)*chunk); chunk 0 also holds order 0
    (integrated over the same n_lam wavelengths in physical mode) and the coat. Uses the production lobe,
    parameter, frame and coat sub-groups."""
    lams, cw = SP.gt_wavelengths(n_lam)
    G = build_subgroups(dist)
    Gm = grp_mirror(dist, quad=(lams, cw), shares=G['shares'])
    ng = new_tree(name)
    main_interface(ng)
    b = NB(ng)
    b.section = 'io'
    gi = b.node('NodeGroupInput').outputs
    go = b.node('NodeGroupOutput').inputs
    fr, co, pa = _front(b, gi, G, 3.0)      # Lobes = 3 cancels the colour-weight /3: art weight = (1 - eta0) Mask
    po = pa.outputs
    art_w, phys_w, inv_d = b.separate(po['Weights'])
    F = b.math('MINIMUM', b.math('MAXIMUM', gi['Efficiency Falloff'], 0.02), 0.98)
    t1 = None
    for j in range(1, M + 1):
        term = b.math('POWER', F, float(j))
        t1 = term if t1 is None else b.add(t1, term)
    lobes = [(m, li) for m in [s * j for j in range(1, M + 1) for s in (1, -1)] for li in range(n_lam)]
    sel = lobes[k * chunk:(k + 1) * chunk]
    shaders = []
    eta_m = {}
    for m, li in sel:
        if abs(m) not in eta_m:
            eta_m[abs(m)] = b.div(b.mul(art_w, b.math('POWER', F, float(abs(m)))), b.mul(t1, 2.0))
        qx = b.mul(inv_d, float(m))
        lb = b.group(G['lobe'], Wavelength=float(lams[li]), Colour=tuple(float(x) for x in cw[li]) + (1.0,),
                     Order=b.combine(qx, 0.0, 0.0), Abs_Order=(float(abs(m)), 0.0, 0.0),
                     Lattice_Vector=b.vscale(fr.outputs['Grating Vector'], qx), Normal=fr.outputs['Normal'],
                     Incoming=fr.outputs['Incoming'], View=fr.outputs['View'], Roughness=gi['Roughness'],
                     Art_Weight=eta_m[abs(m)], Physical_Weight=gi['Mask'], Mode=po['Mode'],
                     Coat_Optics=co.outputs['Coat Optics'], Coat_Exit=co.outputs['Coat Exit'],
                     Fresnel_A=po['Fresnel A'], Fresnel_B=po['Fresnel B'])
        shaders.append(lb.outputs['BSDF'])
    if k == 0:
        mi = b.group(Gm, Normal=fr.outputs['Normal'], View=fr.outputs['View'], Roughness=gi['Roughness'],
                     Fresnel_A=po['Fresnel A'], Fresnel_B=po['Fresnel B'], Coat_Exit=co.outputs['Coat Exit'],
                     Coat_Optics=co.outputs['Coat Optics'], Mask=gi['Mask'], Mode=po['Mode'],
                     Evanescence=po['Evanescence'], Max_Order=po['Max Order'], Saturation=gi['Spectral Saturation'])
        shaders += [mi.outputs['BSDF'], co.outputs['BSDF']]
    if not shaders:
        shaders = [b.node('ShaderNodeBsdfTransparent').outputs[0]]
    b.link(b.add_shaders(shaders), go['BSDF'])
    n_chunks = int(math.ceil(len(lobes) / chunk))
    return ng, dict(gt_chunks=n_chunks, gt_lobes=len(sel))


# ==============================================================================================
# materials, compositor, demo scene
# ==============================================================================================
def make_test_material(name, group, patches=True, params=None, cell_scale=6.0):
    """Same layout as v1's test material: UV Voronoi cells, random grating angle per cell."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    b = NB(nt)
    out = b.node('ShaderNodeOutputMaterial')
    g = b.node('ShaderNodeGroup')
    g.node_tree = group
    for k, v in (params or {}).items():
        if k in g.inputs:
            g.inputs[k].default_value = v
    if patches:
        tc = b.node('ShaderNodeTexCoord')
        vor = b.node('ShaderNodeTexVoronoi', feature='F1', distance='EUCLIDEAN')
        vor.inputs['Scale'].default_value = cell_scale
        b.link(tc.outputs['UV'], vor.inputs['Vector'])
        r = b.separate(vor.outputs['Color'])[0]
        b.link(b.mul(r, 2 * math.pi), g.inputs['Grating Angle'])
    b.link(g.outputs['BSDF'], out.inputs['Surface'])
    b.layout(split_inputs=False)
    return mat


def make_demo_material(group, name="Holo Foil (v2)"):
    """Cracked-ice holo foil: Voronoi shards with a random grating angle and +-12 % pitch each, aluminium
    under a clear coat, narrow un-embossed borders between the shards."""
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    nt = mat.node_tree
    nt.nodes.clear()
    N, L = nt.nodes, nt.links

    def put(idname, x, y, label=None, **props):
        n = N.new(idname)
        n.location = (x, y)
        for k, v in props.items():
            setattr(n, k, v)
        if label:
            n.label = label
        return n
    tc = put('ShaderNodeTexCoord', -1250, 120)
    vor = put('ShaderNodeTexVoronoi', -1000, 300, 'Shards', feature='F1', distance='EUCLIDEAN')
    edge = put('ShaderNodeTexVoronoi', -1000, -120, 'Shard edges', feature='DISTANCE_TO_EDGE')
    for v in (vor, edge):
        v.inputs['Scale'].default_value = 7.0
        v.inputs['Randomness'].default_value = 0.9
        L.new(tc.outputs['UV'], v.inputs['Vector'])
    sep = put('ShaderNodeSeparateColor', -760, 260)
    L.new(vor.outputs['Color'], sep.inputs['Color'])
    ang = put('ShaderNodeMath', -560, 320, 'random angle', operation='MULTIPLY')
    ang.inputs[1].default_value = 2 * math.pi
    L.new(sep.outputs['Red'], ang.inputs[0])
    pm = put('ShaderNodeMath', -560, 160, 'pitch x0.88..1.12', operation='MULTIPLY_ADD')
    pm.inputs[1].default_value = 0.24
    pm.inputs[2].default_value = 0.88
    L.new(sep.outputs['Green'], pm.inputs[0])
    mk = put('ShaderNodeMath', -560, -120, 'mask: plain foil along cracks', operation='MULTIPLY', use_clamp=True)
    mk.inputs[1].default_value = 80.0
    L.new(edge.outputs['Distance'], mk.inputs[0])
    g = put('ShaderNodeGroup', -280, 260, 'Diffraction Grating BSDF')
    g.node_tree = group
    g.width = 260
    g.inputs['Pitch (nm)'].default_value = 1200.0
    g.inputs['Roughness'].default_value = 0.12
    g.inputs['Coat'].default_value = 1.0
    L.new(ang.outputs[0], g.inputs['Grating Angle'])
    L.new(pm.outputs[0], g.inputs['Pitch Multiplier'])
    L.new(mk.outputs[0], g.inputs['Mask'])
    out = put('ShaderNodeOutputMaterial', 80, 260)
    L.new(g.outputs['BSDF'], out.inputs['Surface'])
    fr = N.new('NodeFrame')
    fr.label = 'Pattern: random angle + pitch per shard'
    fr.label_size = 16
    for n in (tc, vor, edge, sep, ang, pm, mk):
        loc = n.location.copy()
        n.parent = fr
        n.location_absolute = loc
    return mat


def build_denoise_group(name=DENOISE_GROUP):
    """Compositor group: OpenImageDenoise with a clean albedo on the foil.
    The foil's glossy lobes give Cycles' denoising albedo almost nothing to work with (features are deferred to
    the next bounce, usually a lamp or the world), and OIDN then darkens sparse rainbow highlights. The grating
    group writes AOV 'dg_foil' (value 1); here the albedo becomes Foil Albedo where that mask is set."""
    ng = new_tree(name, 'CompositorNodeTree')
    ng.description = 'OIDN with a clean albedo on holo foil (needs AOV "dg_foil", type Value, and Denoising Data)'
    it = ng.interface
    it.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    for nm, typ, dflt, desc in (('Image', 'NodeSocketColor', None, 'Noisy image'),
                                ('Albedo', 'NodeSocketColor', None, 'Denoising Albedo pass'),
                                ('Normal', 'NodeSocketVector', None, 'Denoising Normal pass'),
                                ('Foil Mask', 'NodeSocketFloat', 0.0, 'AOV dg_foil'),
                                ('Foil Albedo', 'NodeSocketColor', (1.0, 1.0, 1.0, 1.0),
                                 'Albedo given to the denoiser on the foil (white works best)')):
        sk = it.new_socket(nm, in_out='INPUT', socket_type=typ)
        sk.description = desc
        if dflt is not None:
            sk.default_value = dflt
    b = NB(ng)
    b.section = 'io'
    gi = b.node('NodeGroupInput').outputs
    go = b.node('NodeGroupOutput').inputs
    b.section = "albedo: Cycles' denoising albedo, replaced by a clean constant on the foil"
    alb = b.mix(gi['Foil Mask'], gi['Albedo'], gi['Foil Albedo'], 'RGBA', label='albedo')
    dn = b.node('CompositorNodeDenoise', label='OpenImageDenoise')
    b.link(gi['Image'], dn.inputs['Image'])
    b.link(alb, dn.inputs['Albedo'])
    b.link(gi['Normal'], dn.inputs['Normal'])
    b.link(dn.outputs['Image'], go['Image'])
    b.layout(dx=240, split_inputs=False)
    ng.use_fake_user = True
    return ng


def setup_scene_denoise(scene, view_layer=None):
    """Render denoising off; add AOV dg_foil + denoising data passes; route the image through DG Foil Denoise."""
    vl = view_layer or scene.view_layers[0]
    if 'dg_foil' not in [a.name for a in vl.aovs]:
        a = vl.aovs.add()
        a.name = 'dg_foil'
        a.type = 'VALUE'
    vl.cycles.denoising_store_passes = True
    scene.cycles.use_denoising = False
    grp = bpy.data.node_groups.get(DENOISE_GROUP) or build_denoise_group()
    ng = bpy.data.node_groups.new('Holo Compositing', 'CompositorNodeTree')
    ng.interface.new_socket('Image', in_out='OUTPUT', socket_type='NodeSocketColor')
    rl = ng.nodes.new('CompositorNodeRLayers')
    rl.location = (-320, 0)
    rl.scene = scene
    g = ng.nodes.new('CompositorNodeGroup')
    g.node_tree = grp
    g.location = (0, 0)
    g.width = 200
    go = ng.nodes.new('NodeGroupOutput')
    go.location = (300, 0)
    ng.links.new(rl.outputs['Image'], g.inputs['Image'])
    ng.links.new(rl.outputs['Denoising Albedo'], g.inputs['Albedo'])
    ng.links.new(rl.outputs['Denoising Normal'], g.inputs['Normal'])
    ng.links.new(rl.outputs['dg_foil'], g.inputs['Foil Mask'])
    ng.links.new(g.outputs['Image'], go.inputs[0])
    scene.compositing_node_group = ng
    scene.render.use_compositing = True
    return ng


def build_demo_scene(mat, name="Holo Card Demo"):
    """A 63 x 88 mm card (scaled x10) on a dark table, a small key lamp and a softbox, camera, compositor."""
    from mathutils import Vector
    sc = bpy.data.scenes.get(name) or bpy.data.scenes.new(name)
    sc.render.engine = 'CYCLES'
    sc.cycles.samples = 256
    sc.cycles.use_adaptive_sampling = True
    sc.cycles.adaptive_threshold = 0.02
    sc.cycles.max_bounces = 6
    sc.render.resolution_x, sc.render.resolution_y = 1080, 1350
    sc.view_settings.view_transform = 'AgX'
    sc.world = sc.world or bpy.data.worlds.new(name + ' World')
    sc.world.color = (0.0015, 0.0015, 0.002)
    if sc.world.node_tree and sc.world.node_tree.nodes.get('Background'):
        sc.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.0015, 0.0015, 0.002, 1.0)

    def obj(o):
        sc.collection.objects.link(o)
        return o

    me = bpy.data.meshes.new('Card')
    w, h = 0.63, 0.88
    me.from_pydata([(-w / 2, -h / 2, 0), (w / 2, -h / 2, 0), (w / 2, h / 2, 0), (-w / 2, h / 2, 0)], [],
                   [(0, 1, 2, 3)])
    uv = me.uv_layers.new(name='UVMap')
    for i, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)]):
        uv.data[i].uv = (x, y)
    me.materials.append(mat)
    card = obj(bpy.data.objects.new('Card', me))
    card.location = (0.0, 0.0, 0.46)
    card.rotation_euler = (math.radians(78), math.radians(0), math.radians(-14))

    def look(o, target):
        o.rotation_euler = (Vector(target) - o.location).to_track_quat('-Z', 'Y').to_euler()

    # three small lamps around the camera, off the mirror direction, so the camera sees their diffracted rainbows
    for nm, loc, size, power in (('Key', (1.1, -1.2, 1.3), 0.04, 30.0), ('Rim', (-1.2, -1.0, 0.9), 0.04, 22.0),
                                 ('Top', (-0.2, -0.6, 1.9), 0.06, 25.0)):
        ld = bpy.data.lights.new(nm, 'AREA')
        ld.shape = 'DISK'
        ld.size = size
        ld.energy = power
        lo = obj(bpy.data.objects.new(nm, ld))
        lo.location = loc
        look(lo, (0, 0, 0.46))
    cd = bpy.data.cameras.new('Camera')
    cd.lens = 60
    cam = obj(bpy.data.objects.new('Camera', cd))
    cam.location = (0.25, -2.0, 0.75)
    look(cam, (0, 0.0, 0.45))
    sc.camera = cam
    setup_scene_denoise(sc)
    return sc


def count_nodes(ng):
    """Executable nodes after inlining sub-groups (frames, reroutes, group IO excluded)."""
    n = 0
    for nd in ng.nodes:
        if nd.bl_idname in ('NodeFrame', 'NodeReroute', 'NodeGroupInput', 'NodeGroupOutput'):
            continue
        if nd.bl_idname == 'ShaderNodeGroup' and nd.node_tree:
            n += count_nodes(nd.node_tree)
        else:
            n += 1
    return n


def count_closures(ng):
    n = 0
    for nd in ng.nodes:
        if nd.bl_idname == 'ShaderNodeGroup' and nd.node_tree:
            n += count_closures(nd.node_tree)
        elif nd.bl_idname.startswith('ShaderNodeBsdf'):
            n += 1
    return n


def build_file(path):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    for m in list(bpy.data.materials):
        bpy.data.materials.remove(m)
    ng = build_group(GROUP_NAME, **DEFAULTS)
    ng.use_fake_user = True
    mat = make_demo_material(ng)
    mat.use_fake_user = True
    build_denoise_group()
    old = list(bpy.data.scenes)
    demo = build_demo_scene(mat)
    for w in bpy.context.window_manager.windows:
        w.scene = demo
    for sc in old:
        if sc != demo:
            bpy.data.scenes.remove(sc)
    for im in list(bpy.data.images):
        if im.name != LUT_NAME and im.type != 'RENDER_RESULT':
            bpy.data.images.remove(im)
    print('group', ng.name, 'nodes (inlined)', count_nodes(ng), 'closure nodes', count_closures(ng))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backups
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True)
    print('saved', path)


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(HERE), 'build', 'diffraction_grating.blend')
    if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 1:
        out = sys.argv[sys.argv.index('--') + 1]
    build_file(out)
