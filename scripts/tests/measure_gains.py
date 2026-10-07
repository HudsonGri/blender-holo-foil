"""
White-furnace albedo of Secrop's Balanced group vs ours for every preset in scripts/tests/lib/common.py PRESETS,
and the luminance-match gain the scenes apply to ours' Color so both variants reflect the same
total amount of white light (brightness is then not a confound; colour/placement differences remain).

A large flat plane under a uniform white world (strength 1, 1 bounce), Standard view transform;
the camera looks at it from 0, 30 and 60 degrees; the mean pixel value = directional albedo.
gain = Y(Balanced) / Y(ours, gain 1), Y = Rec.709 luminance averaged over the 0 and 30 degree views.

Needs Balanced: BALANCED_BLEND=/path/to/BlenderGuru_DiffractionGrating_3Solutions.blend
blender -b --factory-startup -t 6 --python scripts/tests/measure_gains.py [-- OUT.json]     (default: scripts/tests/data/gains.json)
About 1-3 minutes on a CPU.
"""
import bpy, sys, os, json, math, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import common as CC  # noqa
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = os.path.abspath(argv[0]) if argv else CC.GAINS_JSON
TMP = os.path.join(tempfile.gettempdir(), '_holo_furnace.exr')
if not CC.has_balanced():
    sys.exit('set BALANCED_BLEND to BlenderGuru_DiffractionGrating_3Solutions.blend (see scripts/README.md)')

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
CC.common_render_settings(sc, spp=128, res=(40, 40), denoise=False, threads=6)
sc.view_settings.view_transform = 'Standard'
sc.cycles.max_bounces = 1
sc.cycles.glossy_bounces = 1
sc.cycles.sample_clamp_indirect = 0
w = bpy.data.worlds.new('W'); sc.world = w
w.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
w.node_tree.nodes['Background'].inputs['Strength'].default_value = 1.0
bpy.ops.mesh.primitive_plane_add(size=200.0)
plane = bpy.context.object
cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')); sc.collection.objects.link(cam)
sc.camera = cam; cam.data.lens = 200

CC.use_source()


def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    tan = nt.nodes.new('ShaderNodeTangent'); tan.direction_type = 'UV_MAP'
    return m, nt, out, {'normal': geo.outputs['Normal'], 'tangent': tan.outputs[0]}


def mat_balanced(preset):
    m, nt, out, L = new_mat('B_' + preset)
    if preset == 'grid49':      # best Balanced can do for a 2D lattice: two crossed 1D copies, 50/50
        a = CC.balanced_node(nt, preset, L)
        b = CC.balanced_node(nt, preset, L); b.inputs['Rotation'].default_value = 0.25
        mx = nt.nodes.new('ShaderNodeMixShader'); mx.inputs[0].default_value = 0.5
        nt.links.new(a.outputs[0], mx.inputs[1]); nt.links.new(b.outputs[0], mx.inputs[2])
        nt.links.new(mx.outputs[0], out.inputs['Surface'])
    else:
        nt.links.new(CC.balanced_node(nt, preset, L).outputs[0], out.inputs['Surface'])
    return m


def mat_ours(preset, gain=1.0):
    m, nt, out, L = new_mat('O_' + preset)
    nd, o = CC.ours_node(nt, preset, {'normal': L['normal'], 'tangent': L['tangent']}, gain)
    nt.links.new(o, out.inputs['Surface'])
    return m


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


res, gains = {}, {}
for preset in CC.PRESETS:
    mats = {'balanced': mat_balanced(preset), 'ours': mat_ours(preset)}
    for th in (0, 30, 60):
        t = math.radians(th)
        cam.location = (0, -60 * math.sin(t), 60 * math.cos(t))
        CC.look_at(cam, (0, 0, 0))
        for v, m in mats.items():
            plane.data.materials.clear(); plane.data.materials.append(m)
            bpy.ops.render.render()
            sc.render.image_settings.file_format = 'OPEN_EXR'
            bpy.data.images['Render Result'].save_render(TMP, scene=sc)
            im = bpy.data.images.load(TMP, check_existing=False)
            px = np.array(im.pixels[:]).reshape(-1, 4)[:, :3]
            bpy.data.images.remove(im)
            res.setdefault(preset, {}).setdefault(v, {})[th] = [round(float(x), 4) for x in px.mean(0)]
            print('FURNACE', preset, v, th, res[preset][v][th], flush=True)
    yb = (lum(res[preset]['balanced'][0]) + lum(res[preset]['balanced'][30])) / 2
    yo = (lum(res[preset]['ours'][0]) + lum(res[preset]['ours'][30])) / 2
    gains[preset] = round(yb / yo, 4)
    print('GAIN', preset, gains[preset], flush=True)
# verification: ours WITH its gain applied (closure weight) must now match Balanced's luminance
check = {}
for preset in CC.PRESETS:
    mats = {'balanced': mat_balanced(preset), 'ours_matched': mat_ours(preset, gains[preset])}
    for th in (0, 30, 60):
        t = math.radians(th)
        cam.location = (0, -60 * math.sin(t), 60 * math.cos(t))
        CC.look_at(cam, (0, 0, 0))
        ys = {}
        for v, m in mats.items():
            if v == 'balanced':
                ys[v] = lum(res[preset]['balanced'][th]); continue
            plane.data.materials.clear(); plane.data.materials.append(m)
            bpy.ops.render.render()
            sc.render.image_settings.file_format = 'OPEN_EXR'
            bpy.data.images['Render Result'].save_render(TMP, scene=sc)
            im = bpy.data.images.load(TMP, check_existing=False)
            px = np.array(im.pixels[:]).reshape(-1, 4)[:, :3]
            bpy.data.images.remove(im)
            ys[v] = lum(px.mean(0))
        check.setdefault(preset, {})[th] = round(ys['ours_matched'] / ys['balanced'], 3)
    print('VERIFY', preset, 'Y(ours x gain)/Y(Balanced) at 0/30/60 deg:', check[preset], flush=True)
json.dump(dict(note='gain = Y(Balanced)/Y(ours) (Rec.709 luminance, mean of the 0 and 30 degree views) under a '
                    'uniform white world; ours is multiplied by it in the look comparisons (closure weight). '
                    'Regenerate: scripts/tests/measure_gains.py (needs BALANCED_BLEND).',
               gain=gains, albedo_rgb=res, verify_ratio=check), open(OUT, 'w'), indent=1)
print('wrote', OUT)
