"""
Optical-disc helpers for the disc test scenes: 120 mm disc, 15 mm hole, clear hub, data area r = 22..58 mm,
concentric tracks (grating vector radial), identical order-0 lobe (eta0 0.3) and hub/rim materials in both
variants, Balanced SlitDistance = ours Pitch = the disc's track pitch, ours x the established CD luminance
gain. Only the track pitch differs between CD (1600 nm), DVD (740 nm) and Blu-ray (320 nm).
"""
import bpy, bmesh, math
import common as CC

R_HOLE, R_HUB, R_DATA0, R_DATA1, R_OUT = 0.0075, 0.0165, 0.022, 0.058, 0.060
PITCH = {'CD': 1600.0, 'DVD': 740.0, 'Blu-ray': 320.0}


def disc_mesh(name='Disc', seg=512):
    me = bpy.data.meshes.new(name)
    bm = bmesh.new()
    radii = [R_HOLE, 0.0118, R_HUB, 0.0195, R_DATA0, 0.03, 0.04, 0.05, R_DATA1, 0.0595, R_OUT]
    rings = [[bm.verts.new((r * math.cos(2 * math.pi * i / seg), r * math.sin(2 * math.pi * i / seg), 0.0))
              for i in range(seg)] for r in radii]
    for a, b in zip(rings[:-1], rings[1:]):
        for i in range(seg):
            j = (i + 1) % seg
            bm.faces.new((a[i], a[j], b[j], b[i]))
    bm.to_mesh(me)
    bm.free()
    for p in me.polygons:
        p.use_smooth = True
    return me


def add_disc(sc, name='Disc', location=(0, 0, 0.0012), rotation=(0, 0, 0)):
    ob = bpy.data.objects.new(name, disc_mesh(name))
    sc.collection.objects.link(ob)
    ob.location = location
    ob.rotation_euler = rotation
    sol = ob.modifiers.new('thickness', 'SOLIDIFY')
    sol.thickness = 0.0012
    sol.offset = -1.0
    return ob


def inputs(nt):
    """Normal = geometry normal, Tangent = circle tangent (along the tracks)."""
    tc = nt.nodes.new('ShaderNodeTexCoord')
    cr = nt.nodes.new('ShaderNodeVectorMath'); cr.operation = 'CROSS_PRODUCT'
    cr.inputs[0].default_value = (0, 0, 1)
    nt.links.new(tc.outputs['Object'], cr.inputs[1])
    vt = nt.nodes.new('ShaderNodeVectorTransform')
    vt.vector_type = 'VECTOR'; vt.convert_from = 'OBJECT'; vt.convert_to = 'WORLD'
    nt.links.new(cr.outputs[0], vt.inputs[0])
    nrm = nt.nodes.new('ShaderNodeVectorMath'); nrm.operation = 'NORMALIZE'
    nt.links.new(vt.outputs[0], nrm.inputs[0])
    nrm.label = 'circle tangent (along the grooves)'
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    return {'normal': geo.outputs['Normal'], 'tangent': nrm.outputs[0]}


def wrap(nt, diff):
    """data area -> diffraction; hub, stacking ring and rim -> plain plastic/metal (identical)."""
    tc = nt.nodes.new('ShaderNodeTexCoord')
    sep = nt.nodes.new('ShaderNodeSeparateXYZ'); nt.links.new(tc.outputs['Object'], sep.inputs[0])
    cmb = nt.nodes.new('ShaderNodeCombineXYZ')
    nt.links.new(sep.outputs[0], cmb.inputs[0]); nt.links.new(sep.outputs[1], cmb.inputs[1])
    ln = nt.nodes.new('ShaderNodeVectorMath'); ln.operation = 'LENGTH'
    nt.links.new(cmb.outputs[0], ln.inputs[0])
    r = ln.outputs['Value']

    def step(edge, x):
        m = nt.nodes.new('ShaderNodeMath'); m.operation = 'GREATER_THAN'
        nt.links.new(x, m.inputs[0]); m.inputs[1].default_value = edge
        return m.outputs[0]

    def mul(a, b):
        m = nt.nodes.new('ShaderNodeMath'); m.operation = 'MULTIPLY'
        nt.links.new(a, m.inputs[0]); nt.links.new(b, m.inputs[1])
        return m.outputs[0]

    def inv(a):
        m = nt.nodes.new('ShaderNodeMath'); m.operation = 'SUBTRACT'
        m.inputs[0].default_value = 1.0; nt.links.new(a, m.inputs[1])
        return m.outputs[0]

    data = mul(step(R_DATA0 + 0.0005, r), inv(step(R_DATA1, r)))
    hub = inv(step(R_HUB, r))
    metal = nt.nodes.new('ShaderNodeBsdfGlossy'); metal.distribution = 'GGX'
    metal.inputs['Roughness'].default_value = 0.06
    metal.inputs['Color'].default_value = (0.8, 0.8, 0.82, 1)
    plastic = nt.nodes.new('ShaderNodeBsdfPrincipled')
    plastic.inputs['Base Color'].default_value = (0.02, 0.02, 0.02, 1)
    plastic.inputs['Roughness'].default_value = 0.08
    plastic.inputs['Transmission Weight'].default_value = 0.9
    plastic.inputs['IOR'].default_value = 1.58
    m1 = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(hub, m1.inputs[0]); nt.links.new(metal.outputs[0], m1.inputs[1])
    nt.links.new(plastic.outputs[0], m1.inputs[2])
    m2 = nt.nodes.new('ShaderNodeMixShader')
    nt.links.new(data, m2.inputs[0]); nt.links.new(m1.outputs[0], m2.inputs[1]); nt.links.new(diff, m2.inputs[2])
    m2.label = 'data area r 22..58 mm'
    return m2.outputs[0]


def disc_material_pair(kind):
    """(balanced or None, ours) materials for 'CD' | 'DVD' | 'Blu-ray': the CD pair with the pitch swapped.
    Ours keeps the CD luminance gain (measured at 1600 nm) so the three discs are the same material."""
    pitch = PITCH[kind]
    key = 'cd' if kind == 'CD' else 'disc_%d' % int(pitch)
    if key not in CC.PRESETS:
        CC.PRESETS[key] = dict(CC.PRESETS['cd'], slit=pitch)
    return CC.material_pair(kind, key, inputs, eta0=0.3, wrap=wrap, gain=CC.load_gains()['cd'])
