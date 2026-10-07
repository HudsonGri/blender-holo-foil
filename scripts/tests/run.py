"""
Build the accuracy-test scenes, render them and run the measurements, all locally.

    python3 scripts/tests/run.py                               # every test, quick settings, "ours" (+ Balanced if available)
    python3 scripts/tests/run.py --tests laser --variants ours # one test, our shader only
    BALANCED_BLEND=/path/to/BlenderGuru_DiffractionGrating_3Solutions.blend python3 scripts/tests/run.py --quality full --device GPU

Tests: white_room, laser, laser_discs, discs, grid49, banding, rainbow_order (no render: needs banding),
order_check (no render), grazing (tiny CPU renders). --list shows them with their settings.

Needs: Blender 5.2+ (`blender` on PATH, or --blender / $BLENDER), and a Python 3 with numpy + Pillow for the
measurements. Outputs: build/tests/{blend,renders/<test>,measure}/ . Renders write .png (display), .npz (linear
float16, incl. the un-denoised pass) and .json (every setting, time, device).

"quick" = small resolution and samples, minutes per test on a CPU: enough to see the result and to read the laser
pitch back. "full" = the published settings (hours on a CPU; use --device GPU).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))
M = os.path.join(TESTS_DIR, 'measure')

DISC_VARIANTS = ['balanced', 'ours', 'balanced_dvd', 'ours_dvd', 'balanced_bd', 'ours_bd']
# name: scene builds [(script, args, blend stem)], renders [(blend stem, variants, quick args, full args)],
#       measures [[python script + args]] ({R} = render dir, {O} = measure dir)
TESTS = {
    'white_room': dict(
        title='White room (white furnace): a lossless foil must vanish',
        scenes=[('p1_furnace.py', [], 'p1_furnace')],
        renders=[('p1_furnace', ['balanced', 'ours', 'ours_strict', 'ours_al'],
                  ['--spp', '64', '--denoise', '0', '--index', '1', '--maxdim', '640'],
                  ['--spp', '4096', '--denoise', '0', '--index', '1'])],
        measures=[['furnace_measure.py', '{R}'], ['furnace_rim.py', '{R}']]),
    'laser': dict(
        title='Laser + CD, run backwards: read the track pitch off the screen',
        scenes=[('p3_laser.py', [], 'p3_laser')],
        renders=[('p3_laser', ['balanced', 'ours'], ['--spp', '1024', '--denoise', '1', '--maxdim', '1200'],
                  ['--spp', '32768', '--denoise', '1'])],
        measures=[['laser_measure.py', '{R}'], ['laser_predict.py', '{R}']]),
    'laser_discs': dict(
        title='Laser experiment for CD / DVD / Blu-ray',
        scenes=[('p3b_laser_discs.py', [], 'p3b_laser_discs')],
        renders=[('p3b_laser_discs', DISC_VARIANTS, ['--spp', '1024', '--denoise', '1', '--maxdim', '1600'],
                  ['--spp', '32768', '--denoise', '1'])],
        measures=[['laser_measure.py', '{R}', '--discs']]),
    'discs': dict(
        title='CD vs DVD vs Blu-ray under one small lamp, against the textbook image',
        scenes=[('p2_discs.py', ['--view', 'std'], 'p2_discs_std'), ('p2_discs.py', ['--view', 'graze'], 'p2_discs_graze')],
        renders=[('p2_discs_std', DISC_VARIANTS, ['--spp', '128', '--denoise', '1', '--maxdim', '540'],
                  ['--spp', '2048', '--denoise', '1']),
                 ('p2_discs_graze', DISC_VARIANTS, ['--spp', '128', '--denoise', '1', '--maxdim', '540'],
                  ['--spp', '2048', '--denoise', '1'])],
        measures=[['disc_textbook.py', '{R}'], ['disc_agreement.py', '{R}']]),
    'grid49': dict(
        title='One light, 49 copies (2D lattice)',
        scenes=[('s3_grid49.py', [], 's3_grid49')],
        renders=[('s3_grid49', ['balanced', 'ours'], ['--spp', '64', '--denoise', '1', '--maxdim', '540'],
                  ['--spp', '2048', '--denoise', '1', '--maxdim', '2160'])],
        measures=[]),
    'banding': dict(
        title='One lamp up close: 5 coloured copies vs a continuous spectrum',
        scenes=[('s5_banding.py', [], 's5_banding')],
        renders=[('s5_banding', ['balanced', 'ours'], ['--spp', '64', '--denoise', '1', '--maxdim', '540'],
                  ['--spp', '2048', '--denoise', '1', '--maxdim', '2160']),
                 ('s5_banding', ['balanced', 'ours'],
                  ['--spp', '64', '--denoise', '1', '--maxdim', '540', '--camera', 'CloseCam'],
                  ['--spp', '2048', '--denoise', '1', '--maxdim', '2160', '--camera', 'CloseCam'])],
        measures=[['rainbow_order.py', '{R}']]),
    'rainbow_order': dict(
        title='Rainbow order next to the highlight (analysis of the banding renders)',
        scenes=[], renders=[], measures=[['rainbow_order.py', '{RB}']]),
    'order_check': dict(
        title="Balanced's colour order and spread vs the textbook, any view (no renderer)",
        scenes=[], renders=[], measures=[['order_check.py', '{O}']]),
    'grazing': dict(
        title='White-furnace probe at grazing views (tiny CPU renders)',
        scenes=[], renders=[], measures=[], blender_measures=[['grazing_probe.py', '--', '{O}']]),
}
DEFAULT = ['white_room', 'laser', 'laser_discs', 'discs', 'grid49', 'banding', 'order_check', 'grazing']


def limit(gb):
    def f():
        if gb:
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (int(gb * 2 ** 30), int(gb * 2 ** 30)))
    return f if (gb and os.name == 'posix') else None


def run(cmd, mem_gb=0, log=None):
    print('$', ' '.join(cmd), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, preexec_fn=limit(mem_gb), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    keep = [ln for ln in p.stdout.splitlines()
            if ln.startswith(('[tests]', 'RENDERED', 'PROBE', 'FURNACE', 'GAIN', 'VERIFY', '==', '  side', 'wrote'))
            or (('Error' in ln or 'Traceback' in ln or 'FAIL' in ln) and 'CUEW' not in ln and 'EGL' not in ln)]
    out = p.stdout if (p.returncode or not keep or cmd[0].endswith(('python', 'python3', sys.executable))) else '\n'.join(keep)
    print(out.rstrip(), '\n   (%.0f s)' % (time.time() - t0), flush=True)
    if log:
        with open(log, 'a') as fh:
            fh.write('$ %s\n%s\n' % (' '.join(cmd), p.stdout))
    if p.returncode:
        raise SystemExit('command failed (exit %d)' % p.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tests', nargs='+', default=DEFAULT, choices=sorted(TESTS))
    ap.add_argument('--variants', nargs='+', default=None,
                    help="'ours' and/or 'balanced' (families: ours_strict, ours_dvd, ... follow them). "
                         "Default: ours, plus balanced when BALANCED_BLEND is set")
    ap.add_argument('--quality', choices=('quick', 'full'), default='quick')
    ap.add_argument('--device', default='CPU', help='CPU or GPU (OptiX/CUDA/HIP/Metal/oneAPI, first found)')
    ap.add_argument('--threads', type=int, default=0, help='CPU render threads (0 = Blender default)')
    ap.add_argument('--blender', default=os.environ.get('BLENDER', 'blender'))
    ap.add_argument('--python', default=sys.executable, help='Python with numpy + Pillow for the measurements')
    ap.add_argument('--mem-gb', type=float, default=12.0, help='address-space cap per process on Linux/macOS (0 = off)')
    ap.add_argument('--skip-build', action='store_true', help='reuse existing scene .blend files')
    ap.add_argument('--skip-render', action='store_true', help='only run the measurements')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()
    if a.list:
        for k in sorted(TESTS):
            t = TESTS[k]
            print('%-14s %s' % (k, t['title']))
            for r in t['renders']:
                print('%16s %-16s %s | quick %s | full %s' % ('', r[0], ' '.join(r[1]), ' '.join(r[2]), ' '.join(r[3])))
        return
    if shutil.which(a.blender) is None and not os.path.exists(a.blender):
        sys.exit('Blender not found: pass --blender /path/to/blender or set $BLENDER')
    if not os.path.exists(os.environ.get('HOLO_FOIL_BLEND') or os.path.join(ROOT, 'holo_foil.blend')):
        sys.exit('holo_foil.blend missing: blender -b --factory-startup --python scripts/build_holo_foil.py')
    have_bal = bool(os.environ.get('BALANCED_BLEND')) and os.path.exists(os.environ['BALANCED_BLEND'])
    fams = set(a.variants or (['ours', 'balanced'] if have_bal else ['ours']))
    if 'balanced' in fams and not have_bal:
        sys.exit('--variants balanced needs BALANCED_BLEND=/path/to/BlenderGuru_DiffractionGrating_3Solutions.blend '
                 '(download: https://www.blenderguru.com/posts/2026/8/13/why-you-cant-render-pokemon-cards-in-blender)')
    out = os.environ.get('HOLO_TEST_OUT') or os.path.join(ROOT, 'build', 'tests')
    os.makedirs(os.path.join(out, 'measure'), exist_ok=True)
    log = os.path.join(out, 'run_log.txt')
    blender = [a.blender, '-b', '--factory-startup'] + (['-t', str(a.threads)] if a.threads else [])
    bmem = a.mem_gb if a.device.upper() == 'CPU' else 0          # GPU drivers reserve huge address space
    t_all = time.time()
    for name in a.tests:
        t = TESTS[name]
        print('\n=== %s: %s' % (name, t['title']), flush=True)
        rdir = os.path.join(out, 'renders', name)
        os.makedirs(rdir, exist_ok=True)
        if not a.skip_build and not a.skip_render:
            for script, args, _ in t['scenes']:
                run(blender + ['--python', os.path.join(TESTS_DIR, 'scenes', script), '--'] + args, bmem, log)
        if not a.skip_render:
            for stem, variants, quick, full in t['renders']:
                for v in variants:
                    if v.split('_')[0] not in fams:
                        continue
                    run([a.blender, '-b', os.path.join(out, 'blend', stem + '.blend')] +
                        (['-t', str(a.threads)] if a.threads else []) +
                        ['--python', os.path.join(TESTS_DIR, 'render.py'), '--', '--variant', v, '--device', a.device,
                         '--out', rdir] + (quick if a.quality == 'quick' else full), bmem, log)
        subst = {'{R}': rdir, '{O}': os.path.join(out, 'measure'), '{RB}': os.path.join(out, 'renders', 'banding')}
        for cmd in t.get('blender_measures', []):
            run(blender + ['--python', os.path.join(M, cmd[0])] + [subst.get(x, x) for x in cmd[1:]], bmem, log)
        for cmd in t['measures']:
            run([a.python, os.path.join(M, cmd[0])] + [subst.get(x, x) for x in cmd[1:]], min(a.mem_gb, 8) or 0, log)
    print('\nall done in %.1f min; renders and JSON in %s' % ((time.time() - t_all) / 60, out))


if __name__ == '__main__':
    main()
