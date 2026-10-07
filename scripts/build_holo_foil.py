"""
Build holo_foil.blend, the asset file shipped with this repo, from scratch.

    blender -b --factory-startup --python scripts/build_holo_foil.py [-- OUT.blend]
    -> holo_foil.blend (repo root) and build/diffraction_grating.blend (the shader on its own)

Steps
  1. build_shader.py   node group "Diffraction Grating BSDF" (+ its "DG ..." sub-groups and the packed
                       Bessel lookup image), material "Holo Foil Demo", compositor group "DG Foil Denoise",
                       scene "Holo Card Demo"
  2. build_patterns.py the "Holo Pattern: ..." groups, "Holo Card Shader" and the "Holo Card" material
                       template (procedural demo art, packed), plus a second scene "Holo Card Template Demo"
  3. marks the user-facing data-blocks as assets, sets fake users and saves compressed, without .blend1 backups.

Asset previews: build/previews/<asset>.jpg ("Holo Pattern: Cosmos" -> holo_pattern_cosmos.jpg), made by
scripts/render_gallery.sh, are embedded when present (run this script again after the gallery).
"""
import os
import re
import sys

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OUT = os.path.join(ROOT, 'holo_foil.blend')
if '--' in sys.argv and len(sys.argv) > sys.argv.index('--') + 1:
    OUT = os.path.abspath(sys.argv[sys.argv.index('--') + 1])
SHADER_BLEND = os.path.join(ROOT, 'build', 'diffraction_grating.blend')
PREVIEWS = os.path.join(ROOT, 'build', 'previews')

AUTHOR = 'blender-holo-foil (github.com/HudsonGri/blender-holo-foil)'
LICENSE = 'CC-BY-4.0'
TAGS = ['holographic', 'holo', 'foil', 'diffraction', 'grating', 'rainbow', 'cycles']

# 1. shader ------------------------------------------------------------------------------------------
import build_shader as BS  # noqa: E402

BS.build_file(SHADER_BLEND)
bpy.ops.wm.open_mainfile(filepath=SHADER_BLEND)

# 2. patterns + card template ------------------------------------------------------------------------
import build_patterns as BP  # noqa: E402

pats, card_shader, _tex = BP.build_all()
demo_mat = next(m for m in bpy.data.materials if m.name.startswith('Holo Foil'))
demo_mat.name = 'Holo Foil Demo'
card_mat = bpy.data.materials['Holo Card']
BS.build_demo_scene(card_mat, name='Holo Card Template Demo')


# 3. assets ------------------------------------------------------------------------------------------
def preview_path(name):
    stem = re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')
    for ext in ('.jpg', '.png'):
        p = os.path.join(PREVIEWS, stem + ext)
        if os.path.exists(p):
            return p
    return None


def mark(idb, description, extra_tags=()):
    idb.use_fake_user = True
    idb.asset_mark()
    ad = idb.asset_data
    ad.description = description
    ad.author = AUTHOR
    ad.license = LICENSE
    ad.copyright = '2026 blender-holo-foil contributors'
    for t in list(TAGS) + list(extra_tags):
        ad.tags.new(t, skip_if_exists=True)
    p = preview_path(idb.name)
    if p:
        try:
            with bpy.context.temp_override(id=idb):
                bpy.ops.ed.lib_id_load_custom_preview(filepath=p)
        except Exception as e:  # noqa: BLE001  (previews are cosmetic)
            print('preview skipped for', idb.name, ':', e)


bsdf = bpy.data.node_groups[BS.GROUP_NAME]
mark(bsdf, 'Physically based holographic foil / diffraction grating BSDF: grating equation, continuous '
     'spectrum, energy conserving. Pure SVM (no OSL), CPU and GPU. Plug a "Holo Pattern" into Grating Angle, '
     'Pitch Multiplier and Mask.', ('bsdf', 'shader'))
for name, ng in sorted(pats.items()):
    mark(ng, ng.description or name, ('pattern',))
mark(card_shader, card_shader.description or 'Printed card over holo foil plus laminate.', ('card',))
den = bpy.data.node_groups.get(BS.DENOISE_GROUP)
if den:
    mark(den, 'OIDN denoise with a clean albedo on the foil (uses the dg_foil AOV written by the '
         'Diffraction Grating BSDF). Stops OIDN from darkening sparse rainbow highlights.', ('denoise',))
mark(demo_mat, 'Demo: cracked-ice holo patches, aluminium foil under a clear coat.', ('material',))
mark(card_mat, 'Holo card template: printed art over holo foil (Cosmos pattern), foil mask with a '
     'Reverse Holo switch, white-ink map, laminate. Swap the pattern node for any Holo Pattern.', ('material', 'card'))

# tidy: packed images get neutral relative paths (no build-machine paths in the file), drop unused data,
# show the demo scene first, save
for img in bpy.data.images:
    if img.packed_file:
        ext = '.exr' if img.file_format == 'OPEN_EXR' else '.png'
        rel = '//textures/' + re.sub(r'[^A-Za-z0-9_.-]+', '_', os.path.splitext(img.name)[0]) + ext
        img.filepath_raw = rel
        for pf in img.packed_files:
            pf.filepath = rel
for scr in bpy.data.screens:                 # the startup file's file-browser area remembers a home directory
    for area in scr.areas:
        for sp in area.spaces:
            if sp.type == 'FILE_BROWSER' and sp.params:
                sp.params.directory = b'_' * 1000      # overwrite the whole fixed-size buffer, then clear it
                sp.params.directory = b''
bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
demo = bpy.data.scenes.get('Holo Card Demo')
for w in bpy.context.window_manager.windows:
    w.scene = demo
bpy.context.preferences.filepaths.save_version = 0
bpy.ops.wm.save_as_mainfile(filepath=OUT, compress=True, relative_remap=False)
print('assets:', sorted(i.name for i in list(bpy.data.node_groups) + list(bpy.data.materials) if i.asset_data))
print('scenes:', [s.name for s in bpy.data.scenes])
print('saved', OUT)
