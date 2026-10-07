"""
Test 2: CD vs DVD vs Blu-ray (track pitch 1.6 / 0.74 / 0.32 um) under one small lamp.

Each disc is rendered in the SAME spot under the SAME lamp and camera (one .blend per view, the variant switch
swaps only the disc material), so the three columns differ only in track pitch. Disc material = the established
s4 CD pair (scripts/tests/lib/cdlib.py): Balanced SlitDistance = ours Pitch = the real track pitch, identical
order-0 lobe (eta0 0.3), lobe alpha .0036, ours x the CD luminance gain for all three discs.

Views (--view):
  std   : camera 28.6 cm away, 25 deg from the disc normal (50 mm lens); a 3 mm LED 7.7 cm away on the far side,
          placed so its mirror image sits in the disc's centre hole (you look at the lamp's reflection in the
          disc). Physics: |sin ti + sin to| on the data area = 0.26-0.91.
  graze : camera 29 cm away, 15 deg above the table (75 mm lens); the same 3 mm LED 14 cm away behind the disc,
          45 deg up (lamp and camera on opposite sides). Physics: |sin ti + sin to| = 0.10-0.68.
A first order of wavelength lam can reach the camera only where |sin ti + sin to| = lam/d, i.e. >= 380/d:
  CD 0.24, DVD 0.51, Blu-ray 1.19 (> 1, so only when lamp AND camera are low on the SAME side).
The lamp is hidden from the camera (visible_camera off).

blender -b --factory-startup -t 6 --python scripts/tests/scenes/p2_discs.py -- --view std|graze
-> build/tests/blend/p2_discs_<view>.blend   (variants: balanced, ours (CD), balanced_dvd, ours_dvd, balanced_bd, ours_bd)
"""
import bpy, sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa
import cdlib  # noqa

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
view = argv[argv.index('--view') + 1] if '--view' in argv else 'std'
VIEWS = {
    'std': dict(cam=(0.0, -0.12, 0.26), lens=50.0, lamp=(0.0, 0.032, 0.07), lamp_w=0.0015, res=(1080, 1080)),
    'graze': dict(cam=(0.0, -0.28, 0.075), lens=75.0, lamp=(0.0, 0.10, 0.10), lamp_w=0.008, res=(1440, 720)),
}
V = VIEWS[view]
if '--lamp-w' in argv:
    V['lamp_w'] = float(argv[argv.index('--lamp-w') + 1])

sc = CC.new_synthetic_scene(res=V['res'], world_rgb=(0.012, 0.012, 0.014))
CC.use_source()
disc = cdlib.add_disc(sc, 'Disc')

bpy.ops.mesh.primitive_plane_add(size=40.0)
table = bpy.context.object
table.name = 'Table'
tm = bpy.data.materials.new('Table')
tb = tm.node_tree.nodes['Principled BSDF']
tb.inputs['Base Color'].default_value = (0.004, 0.004, 0.0045, 1)
tb.inputs['Roughness'].default_value = 1.0
tb.inputs['Specular IOR Level'].default_value = 0.05
table.data.materials.append(tm)

cam = CC.new_camera(sc, V['cam'], (0.0, 0.0, 0.0), lens=V['lens'])
lamp = CC.area_light(sc, V['lamp'], (0.0, 0.0, 0.0), size=0.003, power=V['lamp_w'], shape='DISK', name='LED')
lamp.visible_camera = False

mats = {k: cdlib.disc_material_pair(k) for k in ('CD', 'DVD', 'Blu-ray')}
CC.register_variant(sc, disc, 0, mats['CD'][0], mats['CD'][1],
                    balanced_dvd=mats['DVD'][0], ours_dvd=mats['DVD'][1],
                    balanced_bd=mats['Blu-ray'][0], ours_bd=mats['Blu-ray'][1])
CC.set_variant(sc, 'ours')
sc['test_scene'] = 'p2_discs_' + view
sc['proof_meta'] = json.dumps(dict(
    test='CD / DVD / Blu-ray lineup', view=view, camera=dict(loc=V['cam'], target=[0, 0, 0], lens_mm=V['lens'],
                                                             sensor_mm=36.0, res=list(V['res'])),
    lamp=dict(loc=V['lamp'], shape='disk', size_m=0.003, power_W=V['lamp_w']),
    pitch_nm=cdlib.PITCH, data_area_r_m=[cdlib.R_DATA0, cdlib.R_DATA1], disc_centre=[0, 0, 0.0012],
    ours_gain_all_discs=CC.load_gains()['cd'], eta0_external=0.3, lobe_alpha=0.0036))
CC.save(os.path.join(CC.BLEND_OUT, 'p2_discs_%s.blend' % view))
