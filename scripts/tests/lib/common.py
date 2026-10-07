"""
Shared helpers for the accuracy tests (run inside Blender): "Balanced" vs "Diffraction Grating BSDF".

Fairness rules (enforced here):
  * One .blend per test scene. Every variant lives in it as a separate material; set_variant() swaps material
    slots and nothing else. Camera, lights, world, samples, seed, denoiser and colour management are shared.
  * "Balanced" is Secrop's `Diffraction_Example` node group exactly as Andrew Price ships it (CC-BY 4.0) in
    BlenderGuru_DiffractionGrating_3Solutions.blend. It is NOT part of this repo: download the .blend from
    https://www.blenderguru.com/posts/2026/8/13/why-you-cant-render-pokemon-cards-in-blender and point the
    environment variable BALANCED_BLEND at it. Without it, only the "ours" variants are built.
  * "Ours" is appended from holo_foil.blend (override with HOLO_FOIL_BLEND). Balanced's parameters are mapped
    1:1 by map_balanced(): same pitch number, same microfacet alpha, the same grating direction and the same
    first-order share.

Import from a Blender script:
    import sys; sys.path.insert(0, "<repo>/scripts/tests/lib"); import common as CC
"""
import json
import math
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(TESTS))
OURS_BLEND = os.environ.get('HOLO_FOIL_BLEND') or os.path.join(REPO, 'holo_foil.blend')
OURS_GROUP = 'Diffraction Grating BSDF'
BALANCED_BLEND = os.environ.get('BALANCED_BLEND', '')
BALANCED_GROUP = 'Diffraction_Example'
BUILD = os.environ.get('HOLO_TEST_OUT') or os.path.join(REPO, 'build', 'tests')
BLEND_OUT = os.path.join(BUILD, 'blend')
ASSETS = os.path.join(TESTS, 'assets')
GAINS_JSON = os.path.join(TESTS, 'data', 'gains.json')

CREDITS = ("'Balanced' diffraction shader: Miguel 'Secrop' Porces, with modifications by Andrew Price "
           "(CC-BY 4.0). Pouch mesh: Andrew Price / Blender Guru (CC-BY 4.0).")

# canonical parameter name -> socket name on our group
OURS_SOCKETS = {
    'color': 'Color', 'pitch_nm': 'Pitch (nm)', 'angle': 'Grating Angle', 'gtype': 'Grating Type',
    'orders': 'Orders', 'roughness': 'Roughness', 'eta0': 'Order 0 Strength', 'falloff': 'Efficiency Falloff',
    'saturation': 'Spectral Saturation', 'seed': 'Seed', 'normal': 'Normal', 'tangent': 'Tangent',
}


def log(*a):
    print('[tests]', *a, flush=True)


def has_balanced():
    return bool(BALANCED_BLEND) and os.path.exists(BALANCED_BLEND)


# ------------------------------------------------------------------------------------------------
# appending node groups
# ------------------------------------------------------------------------------------------------
def _append_group(path, name):
    if not os.path.exists(path):
        raise SystemExit('missing %s (build it: blender -b --factory-startup --python scripts/build_holo_foil.py)'
                         % path)
    with bpy.data.libraries.load(path, link=False) as (src, dst):
        if name not in src.node_groups:
            raise RuntimeError('%s not in %s (has %s)' % (name, path, list(src.node_groups)))
        dst.node_groups = [name]
    grp = dst.node_groups[0]
    grp.use_fake_user = True
    log('appended', repr(grp.name), 'from', os.path.basename(path))
    return grp


def ours_group():
    return bpy.data.node_groups.get(OURS_GROUP) or _append_group(OURS_BLEND, OURS_GROUP)


def balanced_group():
    if not has_balanced():
        return None
    return bpy.data.node_groups.get(BALANCED_GROUP) or _append_group(BALANCED_BLEND, BALANCED_GROUP)


def use_source():
    """Append our group once and record it on every scene (kept for the render logs)."""
    g = ours_group()
    for sc in bpy.data.scenes:
        sc['ours_group'] = g.name
        sc['ours_blend'] = os.path.basename(OURS_BLEND)
        sc['balanced_available'] = has_balanced()
    if not has_balanced():
        log('BALANCED_BLEND not set or missing: building the "ours" variants only')
    return g


# ------------------------------------------------------------------------------------------------
# Balanced -> ours parameter mapping
# ------------------------------------------------------------------------------------------------
def first_order_share_to_falloff(share):
    """Our efficiency model (orders 1..3): eta_k ~ F^(k-1). Pick F so order 1 carries `share` of the
    diffracted energy, like Balanced's hard-coded weights. Solves 1 + F + F^2 = 1/share."""
    c = 1.0 / share
    return (-1.0 + math.sqrt(1.0 - 4.0 * (1.0 - c))) / 2.0


def map_balanced(rough_in, slit_nm, rotation, first_order_share):
    """Balanced group inputs -> our canonical parameters.
    * Roughness: Balanced feeds sqrt(Roughness) to Beckmann glossies (Blender alpha = r^2), so alpha =
      Roughness. Ours is GGX with alpha = r^2, so r = sqrt(Roughness) gives the same alpha.
    * SlitDistance (nm) -> Pitch (nm), the same number.
    * Balanced rotates the normal about the tangent, so its grooves run along the tangent; its Rotation
      (turns, opposite sense) maps to Grating Angle = 2*pi*(0.25 - Rotation) on the same tangent.
    * Efficiency Falloff: the first order carries the same share of the diffracted energy."""
    return dict(roughness=math.sqrt(max(rough_in, 0.0)), pitch_nm=slit_nm,
                angle=2.0 * math.pi * (0.25 - rotation), falloff=first_order_share_to_falloff(first_order_share))


# Balanced settings per test. 'pouch' = Andrew's exact values for the pouch in the 3Solutions file.
# Synthetic scenes: lobe alpha chosen per scene (identical for both shaders), Anisotropy 0 (ours is isotropic).
PRESETS = {
    'pouch':   dict(rough=0.0254, slit=1459.0797, rot=0.0, aniso=0.6650, share=0.60),
    'cd':      dict(rough=0.0036, slit=1600.0, rot=0.0, aniso=0.0, share=0.60),
    'banding': dict(rough=0.0064, slit=1459.0797, rot=0.0, aniso=0.0, share=0.60),
    # 49 copies: ours as a 2D square lattice (7x7 orders), falloff .75; Balanced is 1D only, its best effort
    # is two crossed copies mixed 50/50 (s3_grid49.py).
    'grid49':  dict(rough=0.0036, slit=5000.0, rot=0.0, aniso=0.0, share=0.60, ours_extra=dict(gtype=1.0, falloff=0.75)),
}
# Balanced has no zero order. In the synthetic scenes BOTH variants get the identical external order-0 lobe
# (zero_order_mix), so ours' own order 0 is off there.
OURS_DEFAULTS = dict(eta0=0.0, orders=3, gtype=0.0, saturation=1.0, seed=0.0)


def balanced_node(tree, preset, links, color=None, location=(0, 0)):
    p = PRESETS[preset]
    nd = tree.nodes.new('ShaderNodeGroup')
    nd.node_tree = balanced_group()
    nd.location = location
    nd.label = 'Secrop Balanced (%s)' % preset
    nd['test_role'] = 'balanced'
    nd.inputs['ColorFilter'].default_value = color if color is not None else p.get('color', (1, 1, 1, 1))
    nd.inputs['Roughness'].default_value = p['rough']
    nd.inputs['SlitDistance'].default_value = p['slit']
    nd.inputs['Rotation'].default_value = p['rot']
    if p['aniso'] is not None and 'Anisotropy' in nd.inputs:
        nd.inputs['Anisotropy'].default_value = p['aniso']
    sock = {'normal': 'Normal', 'tangent': 'Tangent', 'color': 'ColorFilter'}
    for k, src in links.items():
        tree.links.new(src, nd.inputs[sock[k]])
    return nd


def ours_params(preset, color=None, **extra):
    p = PRESETS[preset]
    d = dict(OURS_DEFAULTS)
    d.update(map_balanced(p['rough'], p.get('ours_pitch', p['slit']), p['rot'], p['share']))
    c = color if color is not None else p.get('color', (1, 1, 1, 1))
    d['color'] = list(c[:3]) + [1.0]
    d.update(p.get('ours_extra', {}))
    d.update(extra)
    return d


def add_ours_node(tree, params, links=None, location=(0, 0), label='Ours'):
    nd = tree.nodes.new('ShaderNodeGroup')
    nd.node_tree = ours_group()
    nd.location = location
    nd.label = label
    nd['test_role'] = 'ours'
    nd['test_params'] = json.dumps(params)
    for canon, val in params.items():
        s = nd.inputs.get(OURS_SOCKETS.get(canon, canon))
        if s is None:
            log('  (ours has no input %r, skipped)' % canon)
            continue
        if isinstance(val, (list, tuple)) and hasattr(s.default_value, '__len__'):
            s.default_value = tuple(val)[:len(s.default_value)]
        else:
            s.default_value = val
    for canon, src in (links or {}).items():
        tree.links.new(src, nd.inputs[OURS_SOCKETS[canon]])
    return nd


def ours_node(tree, preset, links, gain, color=None, location=(0, 0), **extra):
    """Ours mapped from the same preset, with the same colour input as Balanced. The luminance-match gain is a
    closure weight: Mix Shader (Fac = 1 - gain) between ours and an empty closure = gain x ours."""
    params = ours_params(preset, color, **extra)
    links = dict(links)
    if 'color' in links:
        params.pop('color')
    nd = add_ours_node(tree, params, links, location, 'Ours (%s)' % preset)
    if gain > 1.0 + 1e-6:
        raise ValueError('gain %.3f > 1: a closure weight cannot amplify' % gain)
    mix = tree.nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = 1.0 - gain
    mix.label = 'luminance match x%.3f (white furnace)' % gain
    mix.location = (location[0] + 220, location[1] + 120)
    tree.links.new(nd.outputs[0], mix.inputs[1])          # inputs[2] stays empty = no closure
    return nd, mix.outputs[0]


def zero_order_mix(tree, diff_socket, eta0, roughness, normal=None, location=(0, 0)):
    """Mix(eta0) of the diffraction closure with a plain GGX mirror: the same node setup in both variants."""
    gl = tree.nodes.new('ShaderNodeBsdfGlossy')
    gl.distribution = 'GGX'
    gl.inputs['Roughness'].default_value = roughness
    gl.inputs['Color'].default_value = (1, 1, 1, 1)
    gl.label = 'order 0 (identical in both)'
    if normal is not None:
        tree.links.new(normal, gl.inputs['Normal'])
    mix = tree.nodes.new('ShaderNodeMixShader')
    mix.inputs[0].default_value = eta0
    gl.location = (location[0], location[1] - 200)
    mix.location = (location[0] + 250, location[1])
    tree.links.new(diff_socket, mix.inputs[1])
    tree.links.new(gl.outputs[0], mix.inputs[2])
    return mix.outputs[0]


def load_gains():
    """Luminance-match gains (scripts/tests/measure_gains.py): ours x gain reflects as much white light as Balanced at
    0 and 30 degrees, so brightness is not a confound in the look comparisons."""
    return json.load(open(GAINS_JSON))['gain']


# ------------------------------------------------------------------------------------------------
# variant switch: swaps material slots, nothing else
# ------------------------------------------------------------------------------------------------
def register_variant(scene, obj, slot, balanced_mat, ours_mat, **more):
    """Variants: 'balanced', 'ours' and any extra name=material. None = not available (no BALANCED_BLEND)."""
    reg = json.loads(scene.get('test_variants', '[]'))
    mats = {k: m for k, m in dict(balanced=balanced_mat, ours=ours_mat, **more).items() if m is not None}
    reg.append([obj.name, slot, {k: m.name for k, m in mats.items()}])
    scene['test_variants'] = json.dumps(reg)
    for m in mats.values():
        m.use_fake_user = True
    if len(obj.material_slots) <= slot or obj.material_slots[slot].material is None:
        first = mats.get('ours') or next(iter(mats.values()))
        if len(obj.material_slots) <= slot:
            obj.data.materials.append(first)
        else:
            obj.material_slots[slot].material = first


def variants(scene):
    reg = json.loads(scene.get('test_variants', '[]'))
    names = None
    for _, _, d in reg:
        names = set(d) if names is None else names & set(d)
    return sorted(names or [])


def set_variant(scene, variant):
    reg = json.loads(scene.get('test_variants', '[]'))
    if variant not in variants(scene):
        raise SystemExit('variant %r not in this file (%s). Balanced variants need BALANCED_BLEND when the scene '
                         'is built.' % (variant, variants(scene)))
    for oname, slot, d in reg:
        bpy.data.objects[oname].material_slots[slot].material = bpy.data.materials[d[variant]]
    scene['test_variant'] = variant
    log('variant ->', variant, '(%d slots)' % len(reg))


SWITCH_TEXT = '''# Variant switch. Edit VARIANT and run this text block (Alt+P).
# It only swaps material slots; camera, lights, world and render settings are shared.
VARIANT = "ours"   # see the scene property "test_variants" for what this file contains
import bpy, json
sc = bpy.context.scene
for oname, slot, d in json.loads(sc["test_variants"]):
    bpy.data.objects[oname].material_slots[slot].material = bpy.data.materials[d[VARIANT]]
sc["test_variant"] = VARIANT
print("variant ->", VARIANT)
'''


def add_switch_text():
    t = bpy.data.texts.get('variant_switch.py') or bpy.data.texts.new('variant_switch.py')
    t.clear()
    t.write(SWITCH_TEXT)
    c = bpy.data.texts.get('CREDITS') or bpy.data.texts.new('CREDITS')
    c.clear()
    c.write(CREDITS + '\n"Ours": Diffraction Grating BSDF (github.com/HudsonGri/blender-holo-foil).\n')


# ------------------------------------------------------------------------------------------------
# render settings shared by every variant
# ------------------------------------------------------------------------------------------------
def common_render_settings(scene, spp=32, res=(480, 480), denoise=True, seed=1, device='CPU', threads=None):
    scene.render.engine = 'CYCLES'
    c = scene.cycles
    c.device = device
    c.samples = spp
    c.use_adaptive_sampling = False
    c.seed = seed
    c.use_animated_seed = False
    c.use_denoising = denoise
    if denoise:
        c.denoiser = 'OPENIMAGEDENOISE'
        c.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
        try:
            c.denoising_prefilter = 'ACCURATE'
        except Exception:  # noqa: BLE001
            pass
    c.shading_system = False          # SVM only: both shaders are pure nodes (GPU-capable)
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    if threads:
        scene.render.threads_mode = 'FIXED'
        scene.render.threads = threads


def look_at(obj, target, up='Y'):
    from mathutils import Vector
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat('-Z', up).to_euler()


def save(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0      # no .blend1 backups
    add_switch_text()
    bpy.ops.wm.save_as_mainfile(filepath=path, compress=True, relative_remap=True)
    log('saved', path, '| variants:', variants(bpy.context.scene))


def new_synthetic_scene(res=(1080, 1080), world_rgb=(0.004, 0.004, 0.005), look='None'):
    for o in list(bpy.data.objects):
        bpy.data.objects.remove(o)
    sc = bpy.context.scene
    w = bpy.data.worlds.new('World')
    sc.world = w
    bg = w.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = tuple(world_rgb) + (1.0,)
    bg.inputs['Strength'].default_value = 1.0
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = look
    c = sc.cycles
    c.max_bounces = 8
    c.glossy_bounces = 4
    c.diffuse_bounces = 2
    c.transmission_bounces = 4
    c.caustics_reflective = True
    c.caustics_refractive = True
    c.sample_clamp_direct = 0.0
    c.sample_clamp_indirect = 10.0
    c.blur_glossy = 0.0
    common_render_settings(sc, spp=32, res=res, denoise=True, threads=6)
    return sc


def new_camera(sc, loc, target, lens=50.0, name='CompareCam'):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    sc.collection.objects.link(cam)
    sc.camera = cam
    cam.location = loc
    look_at(cam, target)
    cam.data.lens = lens
    cam.data.clip_start = 0.001
    return cam


def area_light(sc, loc, target, size, power, shape='DISK', name='Lamp'):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.shape = shape
    ld.size = size
    ld.energy = power
    lo = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(lo)
    lo.location = loc
    look_at(lo, target)
    return lo


def material_pair(name, preset, inputs_fn, eta0=0.3, balanced_builder=None, wrap=None, gain=None):
    """Two materials with identical structure: [diffraction] -> Mix(eta0) with the same GGX order-0 lobe
    -> (optional wrap(nt, shader_socket) -> socket) -> output. inputs_fn(nt) -> dict with 'normal'/'tangent'
    sockets, built identically in each tree. Returns (balanced or None, ours)."""
    gain = load_gains()[preset] if gain is None else gain
    p = PRESETS[preset]
    mats = []
    for variant in ('balanced', 'ours'):
        if variant == 'balanced' and not has_balanced():
            mats.append(None)
            continue
        m = bpy.data.materials.new('%s [%s]' % (name, variant))
        nt = m.node_tree
        nt.nodes.clear()
        out = nt.nodes.new('ShaderNodeOutputMaterial')
        out.location = (1200, 0)
        L = inputs_fn(nt)
        if variant == 'balanced':
            diff = balanced_builder(nt, L) if balanced_builder else balanced_node(nt, preset, L, location=(400, 0)).outputs[0]
        else:
            diff = ours_node(nt, preset, L, gain, location=(400, 0))[1]
        if eta0 > 0:
            diff = zero_order_mix(nt, diff, eta0, math.sqrt(p['rough']), L.get('normal'), location=(700, 0))
        if wrap:
            diff = wrap(nt, diff)
        nt.links.new(diff, out.inputs['Surface'])
        m['test_note'] = '%s, preset %s%s' % (variant, preset, ', gain %.4f' % gain if variant == 'ours' else '')
        mats.append(m)
    return mats
