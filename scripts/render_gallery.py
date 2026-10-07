"""
Render one pattern-gallery tile (pattern or card demo) from holo_foil.blend.

    blender -b holo_foil.blend -t 6 --python scripts/render_gallery.py -- \
        --tile cosmos --mode render --res 384 512 --spp 64 --out build/gallery/render_cosmos
modes: render  (Cycles path tracing: 2 small lamps + dim studio HDRI, oblique camera)
       map     (flat false-colour map: hue = Angle mod 180 deg, brightness = Mask)
Writes <out>.png and <out>.json (timing, settings). scripts/render_gallery.sh renders every tile and composes
docs/images/pattern_gallery.jpg.
Card-sized plane (63 x 88 mm) in metres, two small lamps and a dim studio HDRI (ships with Blender).
"""
import argparse
import json
import math
import os
import sys
import time

import bpy
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_patterns as BP  # noqa: E402

PI = math.pi
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--tile', required=True)
ap.add_argument('--mode', default='render', choices=['render', 'map'])
ap.add_argument('--res', type=int, nargs=2, default=[384, 512])
ap.add_argument('--spp', type=int, default=96)
ap.add_argument('--denoise', type=int, default=1)
ap.add_argument('--device', default='CPU')
ap.add_argument('--threads', type=int, default=6)
ap.add_argument('--hdri', default='studio.exr')
ap.add_argument('--hdri_strength', type=float, default=0.12)
ap.add_argument('--hdri_rot', type=float, nargs=2, default=[-55.0, 150.0])
ap.add_argument('--exposure', type=float, default=0.0)
ap.add_argument('--clamp', type=float, default=0.0)
ap.add_argument('--out', required=True)
ap.add_argument('--save_blend', default='')
ap.add_argument('--exr', type=int, default=0, help='also write linear <out>.exr')
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--bypass', type=int, default=0, help='cost A/B: pattern outputs replaced by constants')
args = ap.parse_args(argv)

K = 0.0315          # test scene (2 x 2.8 units) scaled to a 63 x 88 mm card
ASPECT = 88.0 / 63.0

# ------------------------------------------------------------------------------------- presets
# pitch: real-world values. pin = pattern group inputs.
T = {
    'linear': dict(group='Holo Pattern: Linear Rainbow', label='Linear Rainbow',
                   note='rainbow film / plain holo foil, 1000 nm',
                   pin={'Angle': 0.0, 'Wobble': 0.06, 'Wobble Scale': 2.0}, pitch=1000.0),
    'sheen': dict(group='Holo Pattern: Linear Rainbow', label='Linear Rainbow, sheen sweep',
                  note='XY "sheen": angle sweeps along the card',
                  pin={'Angle': -0.2, 'Sweep Direction': PI / 2, 'Angle Sweep': 0.45}, pitch=1000.0),
    'radial': dict(group='Holo Pattern: Radial CD', label='Radial / CD',
                   note='concentric grooves, 1600 nm (CD track pitch)',
                   pin={'Center': (0.5, 0.5 * ASPECT, 0.0), 'Inner Radius': 0.07, 'Outer Radius': 0.49,
                        'Edge Softness': 0.003}, pitch=1600.0),
    'scratched': dict(group='Holo Pattern: Scratched Metal', label='Scratched Metal',
                      note='random scratches -> circular halo; 4 um, 8 orders',
                      pin={'Scale': 30.0, 'Density': 0.95, 'Width': 0.045, 'Length': 1.5},
                      pitch=4000.0, orders=8.0, falloff=0.85, eta0=0.35, rough=0.14,
                      plain_rough=0.05, foil=(0.75, 0.75, 0.77, 1), lights='mirror'),
    'glitter': dict(group='Holo Pattern: Glitter Patches', label='Glitter Patches',
                    note='Voronoi mosaic, random angle + pitch per cell, 1200 nm',
                    pin={'Scale': 9.0, 'Pitch Jitter': 0.15}, pitch=1200.0),
    'cosmos': dict(group='Holo Pattern: Cosmos', label='Cosmos',
                   note='circles + dots over a swirling base grating, 1200 nm',
                   pin={'Scale': 5.0}, pitch=1200.0),
    'cracked': dict(group='Holo Pattern: Cracked Ice', label='Cracked Ice',
                    note='shards with large angle jumps, 1000 nm +-25 %',
                    pin={'Scale': 2.6}, pitch=1000.0),
    'stars': dict(group='Holo Pattern: Sparkle Stars', label='Sparkle Stars',
                  note='4-point stars, own grating, 0.64 x pitch',
                  pin={'Scale': 5.0, 'Angle': 0.05, 'Angle Spread': 0.3, 'Density': 0.8},
                  pitch=1564.0),
    'beams': dict(group='Holo Pattern: Diagonal Beams', label='Diagonal Beams',
                  note='stripes alternating two gratings, 1564 nm',
                  pin={'Scale': 6.0, 'Beam Angle': 0.14, 'Gap Angle': -0.42, 'Gap Pitch': 0.85},
                  pitch=1564.0),
    'dotmatrix': dict(group='Holo Pattern: Dot Matrix', label='Dot Matrix',
                      note='holo-pixels, kinematic angle field, 1000 nm',
                      pin={'Scale': 28.0, 'Center': (0.5, 0.5 * ASPECT, 0.0), 'Kinematic': 0.8,
                           'Ring Frequency': 2.0}, pitch=1000.0),
    'image': dict(group='Holo Pattern: Image Driven', label='Image Driven',
                  note='hue of a painted map = grating direction', pin={}, pitch=1200.0),
    # card demos (Holo Card Shader with the procedural demo art)
    'card_holo': dict(card='holo', label='Holo Card: holo', note='cosmos in the art box; white-ink subject'),
    'card_reverse': dict(card='reverse', label='Holo Card: reverse holo', note='cracked ice on the card body'),
    'card_layer': dict(card='layer', label='Holo Card: stars over beams',
                       note='Layer: Sparkle Stars over Diagonal Beams, full-art foil'),
}


# ------------------------------------------------------------------------------------- scene
def look_at(obj, target):
    d = Vector(target) - obj.location
    obj.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
cy = sc.cycles
cy.device = args.device
if args.device == 'GPU':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for typ in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = typ
            prefs.get_devices()
            if any(d.type == typ for d in prefs.devices):
                for d in prefs.devices:
                    d.use = d.type == typ
                break
        except Exception:
            pass
cy.samples = args.spp if args.mode == 'render' else 16
cy.use_adaptive_sampling = False
cy.use_denoising = bool(args.denoise) and args.mode == 'render'
if cy.use_denoising:
    cy.denoiser = 'OPENIMAGEDENOISE'
    cy.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
cy.max_bounces = 4
cy.seed = args.seed
cy.glossy_bounces = 4
cy.diffuse_bounces = 2
cy.transmission_bounces = 2
cy.sample_clamp_direct = args.clamp
cy.sample_clamp_indirect = args.clamp
sc.render.resolution_x, sc.render.resolution_y = args.res
sc.render.resolution_percentage = 100
sc.render.threads_mode = 'FIXED'
sc.render.threads = args.threads
sc.view_settings.view_transform = 'AgX' if args.mode == 'render' else 'Standard'
sc.view_settings.exposure = args.exposure
sc.render.film_transparent = False

bpy.ops.mesh.primitive_plane_add(size=1.0)
card = bpy.context.object
card.name = 'Card'
card.scale = (0.063, 0.088, 1.0)
bpy.ops.object.transform_apply(scale=True)

cam_d = bpy.data.cameras.new('Cam')
cam = bpy.data.objects.new('Cam', cam_d)
sc.collection.objects.link(cam)
sc.camera = cam
cam_d.sensor_fit = 'VERTICAL'
cam_d.sensor_height = 24.0
cam_d.lens = 44.0
cam_d.clip_start = 0.005
cam.location = (0.0, -3.0 * K, 5.6 * K)
look_at(cam, (0, 0, 0))

world = bpy.data.worlds.new('W')
sc.world = world
wt = world.node_tree
bg = wt.nodes['Background']


def add_area_light(name, loc, size, power):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.shape = 'DISK'
    ld.size = size
    ld.energy = power
    lo = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(lo)
    lo.location = loc
    look_at(lo, (0, 0, 0))
    return lo


if args.mode == 'render':
    rig = T[args.tile].get('lights', 'two')
    if rig == 'mirror':
        # lamp in the mirror direction of the camera: its highlight sits on the card (scratch halo)
        add_area_light('Mirror', (0.0, 1.42 * K * 1.6, 2.65 * K * 1.6), 0.04 * K, 12.0 * K * K)
        args.hdri_strength *= 0.3
    else:
        add_area_light('Key', (1.95 * K, 1.9 * K, 2.6 * K), 0.15 * K, 60.0 * K * K)
        add_area_light('Fill', (-1.8 * K, 2.25 * K, 2.25 * K), 0.12 * K, 40.0 * K * K)
    hdri_dir = bpy.utils.system_resource('DATAFILES', path=os.path.join('studiolights', 'world'))
    env = wt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(os.path.join(hdri_dir, args.hdri))
    mapn = wt.nodes.new('ShaderNodeMapping')
    mapn.inputs['Rotation'].default_value = (math.radians(args.hdri_rot[0]), 0, math.radians(args.hdri_rot[1]))
    tcw = wt.nodes.new('ShaderNodeTexCoord')
    wt.links.new(tcw.outputs['Generated'], mapn.inputs['Vector'])
    wt.links.new(mapn.outputs[0], env.inputs['Vector'])
    wt.links.new(env.outputs['Color'], bg.inputs['Color'])
    bg.inputs['Strength'].default_value = args.hdri_strength
    # camera sees a near-black backdrop; reflections see the HDRI
    lp = wt.nodes.new('ShaderNodeLightPath')
    bg2 = wt.nodes.new('ShaderNodeBackground')
    bg2.inputs['Color'].default_value = (0.012, 0.012, 0.014, 1)
    mix = wt.nodes.new('ShaderNodeMixShader')
    wt.links.new(lp.outputs['Is Camera Ray'], mix.inputs[0])
    wt.links.new(bg.outputs[0], mix.inputs[1])
    wt.links.new(bg2.outputs[0], mix.inputs[2])
    wt.links.new(mix.outputs[0], wt.nodes['World Output'].inputs['Surface'])
else:
    bg.inputs['Color'].default_value = (0.0, 0.0, 0.0, 1)

# ------------------------------------------------------------------------------------- material
pats = {ng.name: ng for ng in bpy.data.node_groups if ng.name.startswith('Holo Pattern')}
cs_group = bpy.data.node_groups['Holo Card Shader']
spec = T[args.tile]
tex = dict(BP.prepare_textures(), paint=BP.make_paint_angle_demo())


def pattern_material(spec):
    mat = bpy.data.materials.new('Tile')
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    b = BP.NB(nt)
    out = b.node('ShaderNodeOutputMaterial')
    vec, tc = BP.card_vector(b)
    pat = b.group(pats[spec['group']])
    if spec['group'] != 'Holo Pattern: Image Driven':
        b.link(vec, pat.inputs['Vector'])
    else:
        im = b.node('ShaderNodeTexImage', interpolation='Closest')
        im.image = BP.load_image(tex['paint'], 'Non-Color')
        b.link(tc.outputs['UV'], im.inputs['Vector'])
        b.link(im.outputs['Color'], pat.inputs['Angle Map'])
    for k, v in spec.get('pin', {}).items():
        b._in(pat.inputs[k], v)
    cs = b.group(cs_group)
    for a, c in (('Angle', 'Pattern Angle'), ('Pitch Multiplier', 'Pattern Pitch'),
                 ('Mask', 'Pattern Mask'), ('Roughness Multiplier', 'Pattern Roughness')):
        b.link(pat.outputs[a], cs.inputs[c])
    sets = {'Art Color': (1, 1, 1, 1), 'Foil Mask': 1.0, 'Ink Opacity': 0.0, 'Ink Tint': 0.0,
            'Ink Scatter': 0.0, 'Laminate': 0.0, 'Pitch (nm)': spec.get('pitch', 1200.0),
            'Orders': spec.get('orders', 3.0), 'Roughness': spec.get('rough', 0.08),
            'Order 0 Strength': spec.get('eta0', 0.3), 'Efficiency Falloff': spec.get('falloff', 0.6),
            'Plain Foil Roughness': spec.get('plain_rough', 0.12),
            'Foil Color': spec.get('foil', (0.92, 0.92, 0.92, 1))}
    for k, v in sets.items():
        b._in(cs.inputs[k], v)
    b.link(cs.outputs['Shader'], out.inputs['Surface'])
    return mat


def layer_material():
    """Sparkle Stars layered over Diagonal Beams, full-art foil under the demo print."""
    mat = BP.build_card_material(cs_group, pats, tex, name='Tile', pattern='Holo Pattern: Diagonal Beams',
                                 reverse=True,
                                 pattern_inputs={'Scale': 6.0, 'Beam Angle': 0.14, 'Gap Angle': -0.42,
                                                 'Gap Pitch': 0.85, 'Ridge Tilt': 0.3},
                                 shader_inputs={'Pitch (nm)': 1564.0})
    nt = mat.node_tree
    b = BP.NB(nt)
    beams = next(n for n in nt.nodes if n.type == 'GROUP' and n.node_tree.name == 'Holo Pattern: Diagonal Beams')
    cs = next(n for n in nt.nodes if n.type == 'GROUP' and n.node_tree.name == 'Holo Card Shader')
    stars = b.group(pats['Holo Pattern: Sparkle Stars'], Scale=5.0, Angle=0.05)
    stars.inputs['Angle Spread'].default_value = 0.3
    b.link(beams.inputs['Vector'].links[0].from_socket, stars.inputs['Vector'])
    lay = b.group(pats['Holo Pattern: Layer'])
    for a, c in (('Angle', 'Angle'), ('Pitch Multiplier', 'Pitch'), ('Mask', 'Mask'), ('Roughness Multiplier', 'Roughness')):
        b.link(beams.outputs[a], lay.inputs['Base ' + c])
        b.link(stars.outputs[a], lay.inputs['Top ' + c])
    b.link(beams.outputs['Normal'], lay.inputs['Base Normal'])
    b.link(stars.outputs['Normal'], lay.inputs['Top Normal'])
    for a, c in (('Angle', 'Pattern Angle'), ('Pitch Multiplier', 'Pattern Pitch'),
                 ('Mask', 'Pattern Mask'), ('Roughness Multiplier', 'Pattern Roughness'), ('Normal', 'Foil Normal')):
        b.link(lay.outputs[a], cs.inputs[c])
    # full-art: foil everywhere, the white-ink subject stays solid
    for l in list(cs.inputs['Foil Mask'].links):
        nt.links.remove(l)
    cs.inputs['Foil Mask'].default_value = 1.0
    ink_link = cs.inputs['Ink Opacity'].links
    if ink_link:
        src = ink_link[0].from_node                      # multiply(ink, 1 - reverse)
        nt.links.remove(ink_link[0])
        ink_img = next(n for n in nt.nodes if n.type == 'TEX_IMAGE' and n.image and 'ink' in n.image.name)
        b.link(b.sep(ink_img.outputs['Color'])[0], cs.inputs['Ink Opacity'])
    return mat


if 'card' in spec:
    if spec['card'] == 'holo':
        mat = BP.build_card_material(cs_group, pats, tex, name='Tile', pattern='Holo Pattern: Cosmos',
                                     reverse=False, pattern_inputs={'Scale': 9.0})
    elif spec['card'] == 'reverse':
        mat = BP.build_card_material(cs_group, pats, tex, name='Tile', pattern='Holo Pattern: Cracked Ice',
                                     reverse=True, pattern_inputs={'Scale': 3.0},
                                     shader_inputs={'Pitch (nm)': 1000.0})
    else:
        mat = layer_material()
else:
    mat = pattern_material(spec)

if args.mode == 'map':
    nt = mat.node_tree
    b = BP.NB(nt)
    cs = next(n for n in nt.nodes if n.type == 'GROUP' and n.node_tree.name == 'Holo Card Shader')

    def src(name, default):
        ls = cs.inputs[name].links
        return ls[0].from_socket if ls else default

    ang, mask = src('Pattern Angle', 0.0), src('Pattern Mask', 1.0)
    foil = b.mul(src('Foil Mask', cs.inputs['Foil Mask'].default_value),
                 b.sub(1.0, src('Ink Opacity', cs.inputs['Ink Opacity'].default_value)))
    hue = b.math('FRACT', b.div(ang, PI))
    val = b.mul(b.madd(mask, 0.85, 0.15), b.madd(foil, 0.8, 0.2))
    hsv = b.node('ShaderNodeCombineColor', mode='HSV')
    b.link(hue, hsv.inputs[0])
    hsv.inputs[1].default_value = 0.85
    b.link(val, hsv.inputs[2])
    em = b.node('ShaderNodeEmission')
    b.link(hsv.outputs[0], em.inputs['Color'])
    out = next(n for n in nt.nodes if n.type == 'OUTPUT_MATERIAL')
    b.link(em.outputs[0], out.inputs['Surface'])

if args.bypass:
    cs = next(n for n in mat.node_tree.nodes if n.type == 'GROUP' and n.node_tree.name == 'Holo Card Shader')
    for nm in ('Pattern Angle', 'Pattern Pitch', 'Pattern Mask', 'Pattern Roughness'):
        for l in list(cs.inputs[nm].links):
            mat.node_tree.links.remove(l)
    cs.inputs['Pattern Mask'].default_value = 1.0
card.data.materials.clear()
card.data.materials.append(mat)

# ------------------------------------------------------------------------------------- render
if args.save_blend:
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend), compress=True, copy=True)
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
import resource  # noqa: E402
r0 = resource.getrusage(resource.RUSAGE_SELF)
t0 = time.time()
bpy.ops.render.render(write_still=False)
dt = time.time() - t0
r1 = resource.getrusage(resource.RUSAGE_SELF)
img = bpy.data.images['Render Result']
sc.render.image_settings.file_format = 'PNG'
sc.render.image_settings.color_depth = '8'
img.save_render(os.path.abspath(args.out) + '.png', scene=sc)
if args.exr:
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.render.image_settings.color_depth = '32'
    img.save_render(os.path.abspath(args.out) + '.exr', scene=sc)


def count_closures(tree, seen=None):
    n = 0
    for nd in tree.nodes:
        if nd.type.startswith('BSDF') or nd.type in ('EMISSION',):
            n += 1
        elif nd.type == 'GROUP' and nd.node_tree:
            n += count_closures(nd.node_tree)
    return n


info = dict(tile=args.tile, mode=args.mode, label=spec['label'], note=spec['note'], res=args.res,
            spp=cy.samples, denoise=cy.use_denoising, device=args.device, wall_s=round(dt, 2),
            cpu_s=round((r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime), 1),
            closures_in_tree=count_closures(mat.node_tree), bsdf=BP.BSDF_GROUP)
json.dump(info, open(os.path.abspath(args.out) + '.json', 'w'), indent=1)
print('TILE', json.dumps(info))
