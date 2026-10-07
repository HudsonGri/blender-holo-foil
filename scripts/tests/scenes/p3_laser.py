"""
Test 3: the high-school "laser + CD" experiment, with a white beam.

Setup (all lengths in metres, world axes: x right, y away from the camera, z up):
  * CD (the CD material, 1.6 um tracks) standing in the plane y = 0, face towards +y. Its centre
    is at x = -0.040, so the beam hits the data area 40 mm from the centre, where the tracks run vertically
    (grating vector = x): the orders spread left/right.
  * The "laser": a collimated white beam, a rectangular area light 8 mm (x) x 20 mm (z), Spread 0.5 deg, just in
    front of the screen at y = D - 3 mm, pointing straight at the CD (normal incidence). Invisible to the camera.
  * A white paper screen parallel to the CD at y = D = 0.300 m, facing the CD, with a printed ruler (cm) and
    the TEXTBOOK positions printed as thin coloured marks: x = D * tan(asin(m * lam / d)) for lam = 450 / 532 /
    650 nm, m = +-1, +-2 (computed from this geometry; written to proof_meta).
  * A dim uniform room light (world 0.05) so the paper and ruler are readable. No other lamps.
  * The camera sits just above/behind the CD (it does not block light in Cycles; the CD is out of frame) and
    looks perpendicular at the screen (lens shift, no tilt), so screen x maps LINEARLY to pixels.
Light path on the screen = caustic (camera -> paper -> CD -> lamp): identical settings for both variants,
no clamping, many samples.
Variants (CD material slot only): balanced | ours, the CD pair (cdlib.disc_material_pair('CD')).

blender -b --factory-startup -t 6 --python scripts/tests/scenes/p3_laser.py -> build/tests/blend/p3_laser.blend
"""
import bpy, sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa
import cdlib  # noqa

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
from laser_geom import *  # noqa  (D, PITCH, SCREEN_W/H, RES, CAM_Y/Z, BEAM, SPREAD_DEG, LAMS, textbook_marks, TEX_PATH)


# ------------------------------------------------------------------------------------------------ scene
sc = CC.new_synthetic_scene(res=RES, world_rgb=(0.05, 0.05, 0.05))
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
c = sc.cycles
c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
c.max_bounces = 8; c.diffuse_bounces = 4; c.glossy_bounces = 4
c.caustics_reflective = True
c.blur_glossy = 0.0
sc.render.resolution_x, sc.render.resolution_y = RES
CC.use_source()

disc = cdlib.add_disc(sc, 'CD', location=(-0.040, 0.0, 0.0), rotation=(math.radians(-90.0), 0.0, 0.0))
bal, ours = cdlib.disc_material_pair('CD')
CC.register_variant(sc, disc, 0, bal, ours)

marks = textbook_marks()
tex = TEX_PATH
if not os.path.exists(tex):
    raise SystemExit('run python3 scripts/tests/lib/laser_geom.py first (makes the screen texture)')
bpy.ops.mesh.primitive_plane_add(size=1.0)
scr = bpy.context.object
scr.name = 'Screen'
scr.scale = (SCREEN_W, SCREEN_H, 1.0)
bpy.ops.object.transform_apply(scale=True)
scr.location = (0.0, D, 0.0)
scr.rotation_euler = (math.radians(90.0), 0.0, 0.0)      # normal -y: faces the CD and the camera
sm = bpy.data.materials.new('Screen paper')
nt = sm.node_tree
bsdf = nt.nodes['Principled BSDF']
bsdf.inputs['Roughness'].default_value = 1.0
bsdf.inputs['Specular IOR Level'].default_value = 0.0
img = bpy.data.images.load(tex)
img.pack()
ti = nt.nodes.new('ShaderNodeTexImage'); ti.image = img; ti.interpolation = 'Cubic'
mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
mul.inputs['Factor'].default_value = 1.0
mul.inputs[6].default_value = (0.8, 0.8, 0.8, 1.0)
nt.links.new(ti.outputs['Color'], mul.inputs[7])
nt.links.new(mul.outputs[2], bsdf.inputs['Base Color'])
scr.data.materials.append(sm)

lamp = CC.area_light(sc, (0.0, D - 0.003, 0.0), (0.0, 0.0, 0.0), size=BEAM[0], power=1.0, shape='RECTANGLE', name='Beam')
lamp.data.size_y = BEAM[1]
lamp.data.spread = math.radians(SPREAD_DEG)
lamp.visible_camera = False
lamp.data.energy = float(argv[argv.index('--power') + 1]) if '--power' in argv else 0.02

lens = 18.0 / ((SCREEN_W / 2) / (D - CAM_Y))
cam = CC.new_camera(sc, (0.0, CAM_Y, CAM_Z), (0.0, 1.0, CAM_Z), lens=lens)
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36.0
cam.data.shift_y = -(CAM_Z / (D - CAM_Y)) * lens / 36.0

CC.set_variant(sc, 'ours')
sc['test_scene'] = 'p3_laser'
sc['proof_meta'] = json.dumps(dict(
    test='laser + CD (white beam)', D_m=D, pitch_nm=PITCH, beam_m=BEAM, spread_deg=SPREAD_DEG,
    beam_hits_cd_at_radius_m=0.040, incidence='normal', screen_w_m=SCREEN_W, screen_h_m=SCREEN_H,
    cam=dict(loc=[0.0, CAM_Y, CAM_Z], lens_mm=lens, sensor_w_mm=36.0, shift_y=cam.data.shift_y, res=RES),
    pixel_mapping='screen x (m) = (px + 0.5 - W/2) / (W/2) * (SCREEN_W/2); screen z = CAM_Z + (D-CAM_Y) * '
                  '((H/2 - py - 0.5) / (W/2) * 18/lens + shift_y*36/lens)',
    textbook_marks=marks, ours_gain=CC.load_gains()['cd']))
CC.save(os.path.join(CC.BLEND_OUT, 'p3_laser.blend'))
