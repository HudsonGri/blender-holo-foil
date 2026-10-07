"""
Test 1: the WHITE ROOM (white furnace) test.

Physics: an object that loses no energy, placed inside a perfectly uniform white environment, receives
radiance 1 from every direction and must send radiance 1 back in every direction, so it becomes
invisible (Kirchhoff / the standard "white furnace test" of rendering). Any grey = lost energy, any
colour = an unbalanced spectrum.

Scene: uniform white world (strength 1, no lamps), Standard view transform (scene value 1.0 = display
white), no compositor, 64 bounces of every kind, no clamping. Objects:
  * a flat foil card, Andrew Price's crumpled pouch mesh (from his 3Solutions file, if BALANCED_BLEND is set;
    otherwise a procedurally crumpled sheet stands in) and a foil sphere, all with the shader under test;
  * controls that MUST vanish: a perfect mirror ball (Glossy, roughness 0, white) and a white diffuse
    ball (albedo 1). They are identical in every variant.
Variants (only the foil material changes):
  balanced    : Secrop's Diffraction_Example exactly as Andrew ships it in the 3Solutions file, with his
                pouch settings (Roughness 0.0254, SlitDistance 1459.08, Rotation 0, Anisotropy 0.665);
                ColorFilter = white (1,1,1) instead of his label texture (the texture can only darken it).
  ours        : "Diffraction Grating BSDF", the 1:1 mapping of those settings
                (Pitch 1459.08 nm, GGX r = sqrt(0.0254), Grating Angle pi/2, Efficiency Falloff for a 0.60
                first-order share), Color = white (1,1,1), i.e. metal Fresnel off = lossless, and ours' own
                default Order 0 Strength 0.3 (its zero order). NO brightness-match gain: this test IS the energy.
  ours_strict : the same with Order 0 Strength 0 (the strict 1:1 swap used in Andrew's scenes, because Balanced
                has no zero order). Expected to lose energy where every diffracted order is evanescent
                (grazing views along the grooves): with order 0 switched off the energy has nowhere to go.
  ours_al     : like 'ours' with the shipped default Color = aluminium F0 (0.913, 0.922, 0.924): a real
                aluminium foil, which should read ~0.92 at normal view and whiter at grazing.
Object pass_index (for the masks): card 1, pouch 2, sphere 3, mirror 4, diffuse 5.

blender -b --factory-startup -t 6 --python scripts/tests/scenes/p1_furnace.py -> build/tests/blend/p1_furnace.blend
"""
import bpy, sys, os, math, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'lib'))
import common as CC  # noqa
from mathutils import Vector

for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)
sc = bpy.context.scene
w = bpy.data.worlds.new('WhiteFurnace'); sc.world = w
bg = w.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (1, 1, 1, 1)
bg.inputs['Strength'].default_value = 1.0
sc.view_settings.view_transform = 'Standard'
sc.view_settings.look = 'None'
sc.view_settings.exposure = 0.0
sc.view_settings.gamma = 1.0
sc.render.use_compositing = False
c = sc.cycles
c.max_bounces = 64; c.diffuse_bounces = 64; c.glossy_bounces = 64; c.transmission_bounces = 64
c.transparent_max_bounces = 8
c.caustics_reflective = True; c.caustics_refractive = True
c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
c.blur_glossy = 0.0
c.use_light_tree = True
CC.common_render_settings(sc, spp=64, res=(1920, 1080), denoise=False, threads=6)
c.sample_clamp_direct = 0.0; c.sample_clamp_indirect = 0.0
sc.render.film_transparent = False
sc.render.filter_size = 1.5
CC.use_source()

# ---------------------------------------------------------------- objects
if CC.has_balanced():
    with bpy.data.libraries.load(CC.BALANCED_BLEND, link=False) as (src, dst):
        dst.objects = ['Balanced']
    pouch = dst.objects[0]
    pouch.name = 'Pouch (Andrew Price mesh)'
    sc.collection.objects.link(pouch)
    pouch.data.materials.clear()
else:   # stand-in: a crumpled sheet of the same size (the furnace result does not depend on the shape)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=160, y_subdivisions=220, size=1.0)
    pouch = bpy.context.object
    pouch.name = 'Crumpled sheet (stand-in for the pouch)'
    pouch.scale = (1.15, 1.6, 1.0)
    pouch.rotation_euler = (math.radians(80.0), 0.0, 0.0)
    tex = bpy.data.textures.new('crumple', 'VORONOI')
    tex.noise_scale = 0.18
    disp = pouch.modifiers.new('crumple', 'DISPLACE')
    disp.texture = tex
    disp.strength = 0.06
    bpy.ops.object.modifier_apply(modifier='crumple')
    bpy.ops.object.transform_apply(scale=True)
    bpy.ops.object.shade_smooth()
bpy.context.view_layer.update()
bb = [pouch.matrix_world @ Vector(v) for v in pouch.bound_box]
ctr = sum(bb, Vector()) / 8.0
pouch.location -= ctr - Vector((0.0, 0.0, 0.15))

bpy.ops.mesh.primitive_plane_add(size=1.0)
card = bpy.context.object
card.name = 'Card'
card.scale = (0.95, 1.33, 1.0)
bpy.ops.object.transform_apply(scale=True)
card.location = (-1.55, 0.1, 0.15)
card.rotation_euler = (math.radians(72.0), 0.0, math.radians(-22.0))

bpy.ops.mesh.primitive_uv_sphere_add(segments=96, ring_count=48, radius=0.6, location=(1.55, 0.1, 0.15))
sph = bpy.context.object
sph.name = 'FoilSphere'
bpy.ops.object.shade_smooth()

bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.26, location=(-0.42, -1.25, -0.78))
mirror = bpy.context.object
mirror.name = 'Control mirror ball'
bpy.ops.object.shade_smooth()
bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=0.26, location=(0.42, -1.25, -0.78))
diffuse = bpy.context.object
diffuse.name = 'Control white diffuse ball'
bpy.ops.object.shade_smooth()
for i, o in enumerate((card, pouch, sph, mirror, diffuse)):
    o.pass_index = i + 1

cam = CC.new_camera(sc, (0.0, -7.2, 0.55), (0.0, 0.0, -0.12), lens=50.0)
cam.data.sensor_fit = 'HORIZONTAL'

# ---------------------------------------------------------------- materials
def inputs(nt):
    geo = nt.nodes.new('ShaderNodeNewGeometry')
    tan = nt.nodes.new('ShaderNodeTangent'); tan.direction_type = 'UV_MAP'
    return {'normal': geo.outputs['Normal'], 'tangent': tan.outputs[0]}


def new_mat(name):
    m = bpy.data.materials.new(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new('ShaderNodeOutputMaterial'); out.location = (900, 0)
    return m, nt, out


bal = None
if CC.has_balanced():
    bal, nt, out = new_mat('Foil [balanced]')
    nd = CC.balanced_node(nt, 'pouch', inputs(nt), color=(1, 1, 1, 1), location=(400, 0))
    nt.links.new(nd.outputs[0], out.inputs['Surface'])
    bal['test_note'] = "Secrop/Andrew Balanced, Andrew's pouch settings, ColorFilter white"


def ours_mat(name, color, note, eta0=0.3):
    m, nt, out = new_mat(name)
    p = CC.ours_params('pouch', color=color, eta0=eta0)
    nd = CC.add_ours_node(nt, p, inputs(nt), (400, 0), 'Ours (pouch mapping)')
    nt.links.new(nd.outputs[0], out.inputs['Surface'])
    m['test_note'] = note
    return m, p


ours, p_ours = ours_mat('Foil [ours]', (1, 1, 1, 1), 'ours, pouch mapping, Color white (lossless), Order 0 0.3, no gain')
ours_strict, p_strict = ours_mat('Foil [ours_strict]', (1, 1, 1, 1),
                                 'ours, pouch mapping, Color white, Order 0 Strength 0 (strict 1:1), no gain', eta0=0.0)
ours_al, p_al = ours_mat('Foil [ours_al]', (0.913, 0.922, 0.924, 1), 'ours, pouch mapping, default aluminium Color')

mm, nt, out = new_mat('Control: perfect mirror')
gl = nt.nodes.new('ShaderNodeBsdfGlossy'); gl.distribution = 'GGX'
gl.inputs['Roughness'].default_value = 0.0; gl.inputs['Color'].default_value = (1, 1, 1, 1)
nt.links.new(gl.outputs[0], out.inputs['Surface'])
dm, nt, out = new_mat('Control: white diffuse')
df = nt.nodes.new('ShaderNodeBsdfDiffuse'); df.inputs['Color'].default_value = (1, 1, 1, 1)
df.inputs['Roughness'].default_value = 0.0
nt.links.new(df.outputs[0], out.inputs['Surface'])
mirror.data.materials.append(mm)
diffuse.data.materials.append(dm)

for o in (card, pouch, sph):
    CC.register_variant(sc, o, 0, bal, ours, ours_strict=ours_strict, ours_al=ours_al)
CC.set_variant(sc, 'ours')
sc['test_scene'] = 'p1_furnace'
sc['proof_meta'] = json.dumps(dict(
    test='white furnace', world='uniform white RGB (1,1,1), strength 1, no lamps', view='Standard, exposure 0',
    objects={1: 'card', 2: pouch.name, 3: 'foil sphere', 4: 'control mirror', 5: 'control diffuse'},
    balanced=dict(group='Diffraction_Example (3Solutions file)', Roughness=0.0254, SlitDistance=1459.0797, Rotation=0.0,
                  Anisotropy=0.665, ColorFilter=[1, 1, 1]),
    ours={k: v for k, v in p_ours.items()}, ours_strict={k: v for k, v in p_strict.items()},
    ours_al={k: v for k, v in p_al.items()},
    gain='none (this test measures energy)'))
CC.save(os.path.join(CC.BLEND_OUT, 'p1_furnace.blend'))
