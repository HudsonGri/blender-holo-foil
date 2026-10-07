"""
Smoke test: append "Diffraction Grating BSDF" (and one pattern) from holo_foil.blend into an empty scene,
render a tiny image on the CPU and check that it shows coloured diffraction (not black, no NaNs).

    blender -b --factory-startup -t 6 --python scripts/tests/smoke_test.py [-- --blend holo_foil.blend --spp 32 --out build/smoke_test.png]

Exit code 0 = pass. About 10-60 s on a laptop CPU.
"""
import argparse
import math
import os
import sys

import bpy
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--blend', default=os.path.join(ROOT, 'holo_foil.blend'))
ap.add_argument('--spp', type=int, default=32)
ap.add_argument('--res', type=int, nargs=2, default=(320, 240))
ap.add_argument('--device', default='CPU')
ap.add_argument('--out', default=os.path.join(ROOT, 'build', 'smoke_test.png'))
a = ap.parse_args(argv)

GROUP, PATTERN = 'Diffraction Grating BSDF', 'Holo Pattern: Glitter Patches'
with bpy.data.libraries.load(a.blend, link=False) as (src, dst):
    missing = [n for n in (GROUP, PATTERN) if n not in src.node_groups]
    if missing:
        sys.exit('FAIL: %s not in %s' % (missing, a.blend))
    dst.node_groups = [GROUP, PATTERN]
bsdf_group, pattern_group = bpy.data.node_groups[GROUP], bpy.data.node_groups[PATTERN]

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
sc.render.engine = 'CYCLES'
sc.cycles.device = a.device
sc.cycles.samples = a.spp
sc.cycles.use_adaptive_sampling = False
sc.cycles.use_denoising = False
sc.render.resolution_x, sc.render.resolution_y = a.res
sc.render.resolution_percentage = 100
sc.view_settings.view_transform = 'AgX'
sc.world = bpy.data.worlds.new('World')
sc.world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.01, 0.01, 0.012, 1)

bpy.ops.mesh.primitive_plane_add(size=2.0)
plane = bpy.context.object
mat = bpy.data.materials.new('Smoke Foil')
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new('ShaderNodeOutputMaterial')
pat = nt.nodes.new('ShaderNodeGroup')
pat.node_tree = pattern_group
g = nt.nodes.new('ShaderNodeGroup')
g.node_tree = bsdf_group
g.inputs['Pitch (nm)'].default_value = 1200.0
g.inputs['Roughness'].default_value = 0.12
for o, i in (('Angle', 'Grating Angle'), ('Pitch Multiplier', 'Pitch Multiplier'), ('Mask', 'Mask')):
    nt.links.new(pat.outputs[o], g.inputs[i])
nt.links.new(g.outputs['BSDF'], out.inputs['Surface'])
plane.data.materials.append(mat)

cam = bpy.data.objects.new('Cam', bpy.data.cameras.new('Cam'))
sc.collection.objects.link(cam)
sc.camera = cam
cam.location = (0.0, -3.2, 2.4)
cam.rotation_euler = (math.radians(53), 0, 0)
for name, loc in (('L1', (1.6, -1.0, 2.0)), ('L2', (-1.8, -0.4, 1.6))):
    ld = bpy.data.lights.new(name, 'AREA')
    ld.size, ld.energy = 0.15, 60.0
    lo = bpy.data.objects.new(name, ld)
    sc.collection.objects.link(lo)
    lo.location = loc
    lo.rotation_euler = (lo.location * -1).to_track_quat('-Z', 'Y').to_euler()

bpy.ops.render.render()
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
sc.render.image_settings.file_format = 'PNG'
bpy.data.images['Render Result'].save_render(a.out, scene=sc)
img = bpy.data.images.load(a.out)
px = np.array(img.pixels[:], np.float32).reshape(-1, 4)[:, :3]
mean = px.mean(0)
chroma = float((px.max(1) - px.min(1)).mean())
print('SMOKE mean sRGB %s, mean chroma %.4f, NaN %s -> %s' % (np.round(mean, 4), chroma, bool(np.isnan(px).any()), a.out))
if np.isnan(px).any() or mean.max() < 0.01 or chroma < 0.005:
    sys.exit('FAIL: image is black, NaN or colourless')
print('PASS')
