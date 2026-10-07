"""
Scene 5: close-up of one lamp's diffracted reflection ("banding").

A flat foil plate with one uniform 1D grating (Andrew's pouch SlitDistance, 1459 nm), one small
disk lamp (3 cm) above, dark room. The camera frames the lamp's first-order reflection: Secrop's
Balanced draws it as 5 separate copies of the lamp (its 5 fixed wavelengths, 470/490/530/570/620
nm), ours as one continuous spectrum. Same lobe alpha (.0064 = our GGX r .08), Anisotropy 0, identical
order-0 lobe (eta0 0.3), ours Color x luminance-match gain.

Cameras: 'CompareCam' (whole plate, scene camera) and 'CloseCam' (the lamp's first/second-order
copies): scripts/tests/render.py --camera CloseCam.

blender -b --factory-startup -t 6 --python scripts/tests/scenes/s5_banding.py
-> build/tests/blend/s5_banding.blend
"""
import bpy, sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa


sc = CC.new_synthetic_scene(res=(1080, 1080))
CC.use_source()

bpy.ops.mesh.primitive_plane_add(size=1.0)
plate = bpy.context.object
plate.name = 'FoilPlate'
plate.scale = (2.0, 2.8, 1.0)
bpy.ops.object.transform_apply(scale=True)

wide = CC.new_camera(sc, (0.0, -3.0, 5.6), (0, 0, 0), lens=50.0, name='CompareCam')
wide.data.sensor_fit = 'VERTICAL'; wide.data.sensor_height = 24.0
close = CC.new_camera(sc, (0.0, -3.0, 5.6), (0, 0.5, 0), lens=100.0, name='CloseCam')
close.data.sensor_fit = 'VERTICAL'; close.data.sensor_height = 24.0
sc.camera = wide
# lamp near the zenith on the plane of incidence: its order -1 lands on the plate centre at ~550 nm
CC.area_light(sc, (0.0, 0.285, 2.986), (0, 0, 0), size=0.03, power=10.0)


def inputs(nt):
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    tan = nt.nodes.new('ShaderNodeTangent'); tan.direction_type = 'UV_MAP'
    return {'normal': geo.outputs['Normal'], 'tangent': tan.outputs[0]}


bal, ours = CC.material_pair('FoilPlate', 'banding', inputs, eta0=0.3)
CC.register_variant(sc, plate, 0, bal, ours)
CC.set_variant(sc, 'ours')
sc['test_scene'] = 's5_banding'
CC.save(os.path.join(CC.BLEND_OUT, 's5_banding.blend'))
