"""
Scene 3: "One light -> 49 copies" (the video, 1:20: "if you hold this up to the light, you'll see
that a single light source appears 49 times, while also stretching into rainbows the further out
from the center").

A square holo film (2D hole-array foil, 5 um pitch) in a dark room, one small disk lamp
(3 cm, 20 W) placed at the mirror position of the camera so its reflection sits at the film centre.
Reflection geometry (we render the reflection of the lamp; the video holds the film up and looks
through it, transmission, which follows the same grating equation).

Variants (material slot of 'HoloFilm'), identical node structure around the diffraction part and an
identical order-0 GGX lobe (eta0 0.3) in both:
  balanced : Secrop's group can only do 1D gratings; its best effort at a 2D lattice is two crossed
             copies (Rotation 0 and 0.25) mixed 50/50 (60 closures + 1, just under Cycles' 64).
  ours     : Grating Type = 2D square lattice, orders 0..+-3 in m and n (7x7 = 49), falloff 0.75.
  Same 5000 nm pitch, same lobe alpha (.0036), ours Color x luminance-match gain.

blender -b --factory-startup -t 6 --python scripts/tests/scenes/s3_grid49.py
-> build/tests/blend/s3_grid49.blend
"""
import bpy, sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa


sc = CC.new_synthetic_scene(res=(1080, 1080))
CC.use_source()

bpy.ops.mesh.primitive_plane_add(size=1.0)
film = bpy.context.object
film.name = 'HoloFilm'
film.scale = (2.8, 2.8, 1.0)
bpy.ops.object.transform_apply(scale=True)
sol = film.modifiers.new('thickness', 'SOLIDIFY')
sol.thickness = 0.004
sol.offset = -1.0

# camera and the lamp at its mirror position w.r.t. the film normal (+Z)
el = math.atan2(5.6, 3.0)                      # camera elevation, as in prototype/grid2d
CC.new_camera(sc, (0.0, -3.0, 5.6), (0, 0, 0), lens=50.0)
sc.camera.data.sensor_fit = 'VERTICAL'
sc.camera.data.sensor_height = 24.0
CC.area_light(sc, (0.0, 3.0 * math.cos(el), 3.0 * math.sin(el)), (0, 0, 0), size=0.03, power=20.0)


def inputs(nt):
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    tan = nt.nodes.new('ShaderNodeTangent'); tan.direction_type = 'UV_MAP'
    return {'normal': geo.outputs['Normal'], 'tangent': tan.outputs[0]}


def balanced_crossed(nt, L):
    a = CC.balanced_node(nt, 'grid49', L, location=(200, 200))
    b = CC.balanced_node(nt, 'grid49', L, location=(200, -200))
    b.inputs['Rotation'].default_value = 0.25
    mx = nt.nodes.new('ShaderNodeMixShader'); mx.inputs[0].default_value = 0.5
    mx.label = 'two crossed 1D copies (Balanced is 1D only)'
    nt.links.new(a.outputs[0], mx.inputs[1]); nt.links.new(b.outputs[0], mx.inputs[2])
    return mx.outputs[0]


bal, ours = CC.material_pair('HoloFilm', 'grid49', inputs, eta0=0.3, balanced_builder=balanced_crossed)
CC.register_variant(sc, film, 0, bal, ours)
CC.set_variant(sc, 'ours')
sc['test_scene'] = 's3_grid49'
sc['test_caption'] = ('"If you hold this up to the light, you\'ll see that a single light source appears 49 '
                         'times, while also stretching into rainbows the further out from the center." '
                         '(Blender Guru, 1:20)')
CC.save(os.path.join(CC.BLEND_OUT, 's3_grid49.blend'))
