"""
Test 3b: the white-beam "laser" experiment for a CD, a DVD and a Blu-ray (1.6 / 0.74 / 0.32 um).

Same construction as p3_laser.py (collimated 8 x 20 mm white beam, Spread 0.5 deg, normal incidence 40 mm from
the disc centre, printed screen, camera perpendicular to the screen, dim room light), but the screen is closer
(D = 20 cm) and wider (+-62 cm) so the DVD's large angles fit. One printed screen per disc (its own textbook marks;
the Blu-ray screen says that no first order exists). Disc materials: cdlib.disc_material_pair (the established CD
pair with the pitch swapped; ours x the CD brightness match).
Variants: balanced | ours (CD), balanced_dvd | ours_dvd, balanced_bd | ours_bd. A variant swaps the disc material
and the matching screen print, nothing else.

python3 scripts/tests/lib/laser_discs_geom.py      (textures first)
blender -b --factory-startup -t 6 --python scripts/tests/scenes/p3b_laser_discs.py -> build/tests/blend/p3b_laser_discs.blend
"""
import bpy, sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa
import cdlib  # noqa
import laser_discs_geom as G  # noqa

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
POWER = float(argv[argv.index('--power') + 1]) if '--power' in argv else 0.009

sc = CC.new_synthetic_scene(res=G.RES, world_rgb=(0.05, 0.05, 0.05))
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
c = sc.cycles
c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
c.max_bounces = 8; c.diffuse_bounces = 4; c.glossy_bounces = 4
c.caustics_reflective = True
c.blur_glossy = 0.0
sc.render.resolution_x, sc.render.resolution_y = G.RES
CC.use_source()

disc = cdlib.add_disc(sc, 'Disc', location=(-0.040, 0.0, 0.0), rotation=(math.radians(-90.0), 0.0, 0.0))

bpy.ops.mesh.primitive_plane_add(size=1.0)
scr = bpy.context.object
scr.name = 'Screen'
scr.scale = (G.SCREEN_W, G.SCREEN_H, 1.0)
bpy.ops.object.transform_apply(scale=True)
scr.location = (0.0, G.D, 0.0)
scr.rotation_euler = (math.radians(90.0), 0.0, 0.0)


def screen_mat(key):
    sm = bpy.data.materials.new('Screen paper (%s)' % key)
    nt = sm.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    img = bpy.data.images.load(G.tex_path(key))
    img.pack()
    ti = nt.nodes.new('ShaderNodeTexImage'); ti.image = img; ti.interpolation = 'Cubic'
    mul = nt.nodes.new('ShaderNodeMix'); mul.data_type = 'RGBA'; mul.blend_type = 'MULTIPLY'
    mul.inputs['Factor'].default_value = 1.0
    mul.inputs[6].default_value = (0.8, 0.8, 0.8, 1.0)
    nt.links.new(ti.outputs['Color'], mul.inputs[7])
    nt.links.new(mul.outputs[2], bsdf.inputs['Base Color'])
    return sm


mats = {name: cdlib.disc_material_pair(name) for name, _, _ in G.DISCS}
scr_m = {key: screen_mat(key) for _, _, key in G.DISCS}
scr.data.materials.append(scr_m['cd'])
CC.register_variant(sc, disc, 0, mats['CD'][0], mats['CD'][1], balanced_dvd=mats['DVD'][0], ours_dvd=mats['DVD'][1],
                    balanced_bd=mats['Blu-ray'][0], ours_bd=mats['Blu-ray'][1])
CC.register_variant(sc, scr, 0, scr_m['cd'], scr_m['cd'], balanced_dvd=scr_m['dvd'], ours_dvd=scr_m['dvd'],
                    balanced_bd=scr_m['bd'], ours_bd=scr_m['bd'])

lamp = CC.area_light(sc, (0.0, G.D - 0.003, 0.0), (0.0, 0.0, 0.0), size=G.BEAM[0], power=POWER, shape='RECTANGLE',
                     name='Beam')
lamp.data.size_y = G.BEAM[1]
lamp.data.spread = math.radians(G.SPREAD_DEG)
lamp.visible_camera = False

lens = 18.0 / ((G.SCREEN_W / 2) / (G.D - G.CAM_Y))
cam = CC.new_camera(sc, (0.0, G.CAM_Y, G.CAM_Z), (0.0, 1.0, G.CAM_Z), lens=lens)
cam.data.sensor_fit = 'HORIZONTAL'
cam.data.sensor_width = 36.0
cam.data.shift_y = -(G.CAM_Z / (G.D - G.CAM_Y)) * lens / 36.0
cam.data.clip_start = 0.001

CC.set_variant(sc, 'ours')
sc['test_scene'] = 'p3b_laser_discs'
sc['proof_meta'] = json.dumps(dict(
    test='laser experiment, CD / DVD / Blu-ray', D_m=G.D, beam_m=G.BEAM, spread_deg=G.SPREAD_DEG,
    beam_hits_disc_at_radius_m=0.040, incidence='normal', screen_w_m=G.SCREEN_W, screen_h_m=G.SCREEN_H,
    cam=dict(loc=[0.0, G.CAM_Y, G.CAM_Z], lens_mm=lens, sensor_w_mm=36.0, shift_y=cam.data.shift_y, res=G.RES),
    power_W=POWER, pitch_nm={name: d for name, d, _ in G.DISCS},
    textbook_marks={name: G.textbook_marks(d) for name, d, _ in G.DISCS}, ours_gain=CC.load_gains()['cd']))
CC.save(os.path.join(CC.BLEND_OUT, 'p3b_laser_discs.blend'))
