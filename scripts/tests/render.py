"""
Render one variant of a test scene with explicit, logged settings.

blender -b build/tests/blend/<scene>.blend -t 6 --python scripts/tests/render.py -- \
    --variant ours|balanced|... [--spp 1024] [--maxdim 0] [--denoise 1|0|scene] [--device CPU|GPU]
    [--camera NAME] [--seed 1] [--out DIR] [--tag T] [--index 1] [--keep-exr 0]

Writes <out>/<scene>[_<camera>]_<variant>[_<tag>]:
  .png   display-referred (the scene's view transform)
  .npz   linear scene-referred float16 arrays, top row first: rgb (the image as rendered), and when present
         noisy (the same render before the denoiser) and index (object pass_index, for masks)
  .json  every setting + timings + Blender/GPU info
Render every variant of a test with identical arguments.
"""
import bpy, sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))
import common as CC  # noqa
import exrlite  # noqa
import numpy as np

argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument('--variant', required=True)
ap.add_argument('--spp', type=int, default=256)
ap.add_argument('--maxdim', type=int, default=0, help='scale so max(w,h)=this (0 = keep the scene resolution)')
ap.add_argument('--denoise', default='scene')
ap.add_argument('--device', default='CPU')
ap.add_argument('--camera', default=None)
ap.add_argument('--seed', type=int, default=1)
ap.add_argument('--out', default=os.path.join(CC.BUILD, 'renders'))
ap.add_argument('--tag', default='')
ap.add_argument('--index', type=int, default=0, help='also store the object-index pass (masks)')
ap.add_argument('--keep-exr', type=int, default=0)
ap.add_argument('--name', default=None)
a = ap.parse_args(argv)
scene_id = a.name or os.path.splitext(os.path.basename(bpy.data.filepath))[0]

sc = bpy.context.scene
CC.set_variant(sc, a.variant)
if a.camera:
    sc.camera = bpy.data.objects[a.camera]
c = sc.cycles
c.device = a.device
gpu_names = []
if a.device == 'GPU':
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for backend in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
        try:
            prefs.compute_device_type = backend
            prefs.get_devices()
            if any(d.type == backend for d in prefs.devices):
                for d in prefs.devices:
                    d.use = (d.type == backend)
                gpu_names = ['%s (%s)' % (d.name, backend) for d in prefs.devices if d.use]
                break
        except TypeError:
            continue
c.samples = a.spp
c.use_adaptive_sampling = False
c.seed = a.seed
c.use_animated_seed = False
c.time_limit = 0
c.shading_system = False
if a.denoise != 'scene':
    c.use_denoising = a.denoise == '1'
    if c.use_denoising:
        c.denoiser = 'OPENIMAGEDENOISE'
        c.denoising_input_passes = 'RGB_ALBEDO_NORMAL'
vl = sc.view_layers[0]
vl.use_pass_combined = True
vl.use_pass_object_index = bool(a.index)
vl.cycles.denoising_store_passes = bool(c.use_denoising)
r = sc.render
if a.maxdim:
    k = a.maxdim / max(r.resolution_x, r.resolution_y)
    r.resolution_x = max(2, int(round(r.resolution_x * k / 2)) * 2)
    r.resolution_y = max(2, int(round(r.resolution_y * k / 2)) * 2)
r.resolution_percentage = 100
r.use_persistent_data = True
os.makedirs(a.out, exist_ok=True)
base = '%s%s_%s%s' % (scene_id, ('_' + a.camera) if a.camera else '', a.variant, ('_' + a.tag) if a.tag else '')

stats = {}
bpy.app.handlers.render_stats.append(lambda s: stats.__setitem__('last', s))
t0 = time.time()
bpy.ops.render.render(write_still=False)
dt = time.time() - t0
img = bpy.data.images['Render Result']
s = r.image_settings
exr = os.path.join(a.out, base + '.exr')
if hasattr(s, 'media_type'):                     # Blender 5.x: multilayer is a media type
    s.media_type = 'MULTI_LAYER_IMAGE'
    s.file_format = 'OPEN_EXR_MULTILAYER'
else:
    s.file_format = 'OPEN_EXR_MULTILAYER'
s.color_depth = '32'
s.exr_codec = 'NONE'
img.save_render(exr, scene=sc)
if hasattr(s, 'media_type'):
    s.media_type = 'IMAGE'
s.file_format = 'PNG'
s.color_depth = '8'
s.color_mode = 'RGB'
img.save_render(os.path.join(a.out, base + '.png'), scene=sc)

ch = exrlite.read_exr(exr)
arrs = {}
layer = [k.split('.')[0] for k in ch if '.Combined.' in k or k.endswith('Combined.R')]
pre = (layer[0] + '.') if layer else ''
arrs['rgb'] = exrlite.stack(ch, pre + 'Combined').astype(np.float16)
for nm in ('Noisy Image', 'Noisy'):
    if (pre + nm + '.R') in ch:
        arrs['noisy'] = exrlite.stack(ch, pre + nm).astype(np.float16)
        break
idx = [k for k in ch if k.startswith(pre + 'IndexOB') or k.startswith(pre + 'Object Index')]
if idx:
    arrs['index'] = ch[idx[0]].astype(np.float16)
np.savez_compressed(os.path.join(a.out, base + '.npz'), **arrs)
channels = sorted(ch)
del ch
if not a.keep_exr:
    os.remove(exr)

info = dict(scene=scene_id, camera=sc.camera.name, variant=a.variant, spp=a.spp, res=[r.resolution_x, r.resolution_y],
            gpu=', '.join(gpu_names) or None, denoise=c.use_denoising,
            denoiser=c.denoiser if c.use_denoising else None, device=a.device, seed=a.seed, seconds=round(dt, 2),
            view=[sc.view_settings.view_transform, sc.view_settings.look, sc.view_settings.exposure],
            bounces=dict(max=c.max_bounces, diffuse=c.diffuse_bounces, glossy=c.glossy_bounces,
                         transmission=c.transmission_bounces, clamp_direct=c.sample_clamp_direct,
                         clamp_indirect=c.sample_clamp_indirect, caustics_reflective=c.caustics_reflective,
                         blur_glossy=c.blur_glossy),
            npz_arrays=sorted(arrs), exr_channels=channels,
            ours_group=sc.get('ours_group'), ours_blend=sc.get('ours_blend'),
            proof_meta=json.loads(sc.get('proof_meta', '{}')),
            blender=bpy.app.version_string, stats=stats.get('last', ''))
json.dump(info, open(os.path.join(a.out, base + '.json'), 'w'), indent=1)
print('RENDERED', json.dumps({k: info[k] for k in ('scene', 'variant', 'spp', 'res', 'seconds', 'gpu', 'npz_arrays')}),
      flush=True)
