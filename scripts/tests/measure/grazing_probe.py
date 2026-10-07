"""White-furnace probe of ours at grazing views (flat plane, white world, 24x24 px, CPU, ~1-3 min).
A lossless foil must read (1, 1, 1) at every angle. Views in the dispersion plane and along the grooves, 60-85 deg.
blender -b --factory-startup -t 6 --python scripts/tests/measure/grazing_probe.py [-- OUT_DIR]   -> OUT_DIR/grazing_probe.json"""
import bpy, sys, os, math, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'lib'))
import common as CC  # noqa
import numpy as np
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
CC.common_render_settings(sc, spp=512, res=(24, 24), denoise=False, threads=6)
sc.view_settings.view_transform = 'Standard'
sc.cycles.max_bounces = 1; sc.cycles.glossy_bounces = 1; sc.cycles.sample_clamp_indirect = 0; sc.cycles.sample_clamp_direct = 0
w = bpy.data.worlds.new('W'); sc.world = w
w.node_tree.nodes['Background'].inputs['Color'].default_value = (1, 1, 1, 1)
bpy.ops.mesh.primitive_plane_add(size=400.0)
plane = bpy.context.object
cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam')); sc.collection.objects.link(cam)
sc.camera = cam; cam.data.lens = 2000
g = CC.use_source()
TMP = os.path.join(tempfile.gettempdir(), '_holo_grazing_probe.exr')
res = {}
def mat(eta0, rough, pitch):
    m = bpy.data.materials.new('m'); nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial')
    geo = nt.nodes.new('ShaderNodeNewGeometry'); tan = nt.nodes.new('ShaderNodeTangent'); tan.direction_type = 'UV_MAP'
    p = CC.ours_params('pouch', color=(1, 1, 1, 1), eta0=eta0, roughness=rough, pitch_nm=pitch)
    nd = CC.add_ours_node(nt, p, {'normal': geo.outputs['Normal'], 'tangent': tan.outputs[0]})
    nt.links.new(nd.outputs[0], out.inputs['Surface'])
    return m
cases = [(0.0, math.sqrt(0.0254), 1459.08), (0.3, math.sqrt(0.0254), 1459.08), (0.0, 0.06, 1459.08), (0.0, math.sqrt(0.0254), 5000.0)]
for eta0, rough, pitch in cases:
    m = mat(eta0, rough, pitch)
    plane.data.materials.clear(); plane.data.materials.append(m)
    for plane_name, phi in (('dispersion', 90), ('grooves', 0)):
        for th in (60, 70, 75, 80, 85):
            t, p = math.radians(th), math.radians(phi)
            cam.location = (60 * math.sin(t) * math.cos(p), -60 * math.sin(t) * math.sin(p), 60 * math.cos(t))
            CC.look_at(cam, (0, 0, 0))
            bpy.ops.render.render()
            sc.render.image_settings.file_format = 'OPEN_EXR'
            bpy.data.images['Render Result'].save_render(TMP, scene=sc)
            im = bpy.data.images.load(TMP, check_existing=False)
            px = np.array(im.pixels[:]).reshape(-1, 4)[:, :3].mean(0); bpy.data.images.remove(im)
            key = 'eta0=%.1f r=%.3f d=%g %s %d' % (eta0, rough, pitch, plane_name, th)
            res[key] = [round(float(x), 3) for x in px]
            print('PROBE', key, res[key], flush=True)
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT = argv[0] if argv else os.path.join(CC.BUILD, 'measure')
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, 'grazing_probe.json'), 'w'), indent=1)
print('wrote', os.path.join(OUT, 'grazing_probe.json'))
