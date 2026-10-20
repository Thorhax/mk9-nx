"""banjo-nx (deko3d renderer, docs/deko3d-renderer.md section 5): the game shaders' deko3d SPIR-V -> uam GLSL -> DKSH.

    python tools/deko3d_shaders.py --spirv out/deko3d_shaders/spirv --out out/deko3d_shaders [--jobs 8] [--limit N]

The SPIR-V comes from Renut's XenosRecomp with BANJO_DEKO3D_SPIRV_DIR set (one <vs|ps>_<hash>.spv per shader). For each
shader: SPIRV-Cross (Vulkan GLSL 4.60, NaN checks relaxed), tools/deko3d_glsl.py, then uam for the default variant
(every specialization constant at its default) as a check that it compiles. Writes <out>/glsl/<stem>.glsl,
<out>/dksh/<stem>.dksh and <out>/report.txt (failures by reason, uam's warnings, DKSH sizes). Game data: the build tree
only.
"""
import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deko3d_glsl  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPIRV_CROSS = os.path.join(ROOT, 'out', 'deko3d_tools', 'spirv-cross-build', 'spirv-cross.exe')
UAM = os.path.join(ROOT, 'out', 'deko3d_tools', 'uam-build', 'uam.exe')
NO_UAM = False


def one(spv, out):
    stem = os.path.splitext(os.path.basename(spv))[0]
    stage = 'frag' if stem.startswith('ps_') else 'vert'
    vk = os.path.join(out, 'vk', stem + '.vk.glsl')
    glsl = os.path.join(out, 'glsl', stem + '.glsl')
    dksh = os.path.join(out, 'dksh', stem + '.dksh')
    r = subprocess.run([SPIRV_CROSS, spv, '--version', '460', '--vulkan-semantics', '--relax-nan-checks',
                        '--output', vk], capture_output=True, text=True)
    if r.returncode != 0:
        return stem, 'spirv-cross', (r.stderr or r.stdout).strip()[:300], 0, []
    try:
        text, _ = deko3d_glsl.convert(open(vk, encoding='utf-8').read(), stage, deko3d_glsl.DEFAULT_UBO_MAP, 'handle')
    except deko3d_glsl.Unsupported as e:
        return stem, 'convert', str(e)[:300], 0, []
    with open(glsl, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)
    if NO_UAM:
        return stem, None, '', 0, []
    r = subprocess.run([UAM, '-s', stage, '-o', dksh, glsl], capture_output=True, text=True)
    log = (r.stdout + r.stderr).strip()
    warnings = [l.strip() for l in log.split('\n') if 'warning' in l.lower()]
    if r.returncode != 0 or not os.path.isfile(dksh):
        first = next((l for l in log.split('\n') if 'error' in l.lower()), log[:300])
        return stem, 'uam', first.strip()[:300], 0, warnings
    return stem, None, '', os.path.getsize(dksh), warnings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spirv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--no-uam', action='store_true', help='convert only (the GLSL package needs no DKSH)')
    a = ap.parse_args()
    global NO_UAM
    NO_UAM = a.no_uam
    for d in ('vk', 'glsl', 'dksh'):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    spvs = sorted(os.path.join(a.spirv, f) for f in os.listdir(a.spirv) if f.endswith('.spv'))
    if a.limit:
        spvs = spvs[:a.limit]
    fails = collections.defaultdict(list)
    warns = collections.Counter()
    warned = collections.defaultdict(list)
    sizes = {'vs': [], 'ps': []}
    with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
        for i, (stem, reason, detail, size, warnings) in enumerate(ex.map(lambda s: one(s, a.out), spvs)):
            if reason:
                fails[reason].append((stem, detail))
            else:
                sizes[stem[:2]].append(size)
            for w in set(warnings):
                key = re.sub(r'\d+', 'N', w)[:160]
                warns[key] += 1
                warned[key].append(stem)
            if (i + 1) % 500 == 0:
                print(f'{i + 1}/{len(spvs)}', flush=True)
    lines = [f'{len(spvs)} shaders: {sum(len(v) for v in sizes.values())} compiled, '
             f'{sum(len(v) for v in fails.values())} failed']
    for k in ('vs', 'ps'):
        v = sorted(sizes[k])
        if v:
            lines.append(f'{k}: {len(v)} DKSH, median {v[len(v) // 2]} bytes, max {v[-1]}')
    for reason, items in sorted(fails.items(), key=lambda kv: -len(kv[1])):
        lines.append(f'\n[{reason}] {len(items)}')
        by_detail = collections.Counter(re.sub(r'_[0-9A-Fa-f]{6,}|\d+', 'N', d)[:200] for _, d in items)
        for d, n in by_detail.most_common(15):
            ex_stem = next(s for s, dd in items if re.sub(r'_[0-9A-Fa-f]{6,}|\d+', 'N', dd)[:200] == d)
            lines.append(f'  {n:5}  {d}   (e.g. {ex_stem})')
    if warns:
        lines.append('\n[uam warnings]')
        for w, n in warns.most_common(20):
            lines.append(f'  {n:5}  {w}   (e.g. {warned[w][0]})')
    report = '\n'.join(lines)
    with open(os.path.join(a.out, 'report.txt'), 'w', encoding='utf-8') as f:
        f.write(report + '\n')
    print(report)


if __name__ == '__main__':
    main()
