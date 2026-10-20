#!/usr/bin/env python3
"""banjo-nx (2026-10-18, docs/deko3d-shader-optimisations.md): what uam makes of the game shaders, per shader, weighted by
the console's GPU time.

    python tools/deko3d_shader_stats.py --list pipeline_list.bin [--list ...] --gpu-log renut_001.log [--gpu-log ...]
        [--glsl out/deko3d_shaders/glsl] [--base-glsl DIR | --base-json FILE] [--json out.json] [--top 40]

The variants are those of the pipeline lists, made as tools/deko3d_variant_cache.py makes them (the text plume_deko3d
compiles), and each is compiled by the PC build of uam (the console's patches). From the DKSH: the instructions (the
code after the 0x80-byte shader program header, without the scheduling words and the trailing padding), the GPRs, the
scratch a warp; from the Maxwell code: the instruction classes, by the opcodes the GM107 emitter of
app/thirdparty/uam/mesa-imported/codegen/nv50_ir_emit_gm107.cpp writes (texture fetches, constant loads, scratch
traffic, branches, and so on).

The weights: the "GPU profile" blocks of the logs (renut_gpu_profile with banjo_renut_diag_gpu_shaders), the ms of each
pixel shader summed over every block (a vertex shader gets the ms of the pixel shaders it is drawn with). A
shader's weighted change is its instruction change times its share of that time: the expected GPU change if the time
went as the instructions (texture-bound shaders gain less, ALU-bound more). The console's profile is the measurement.

--base-glsl compares with another GLSL tree (before a change); --base-json with the --json of an earlier run.
Compiled results are kept in out/deko3d_shaders/stats_cache.json (by the variant text's FNV-1a 64).
Game data: the build tree only.
"""
import argparse
import collections
import concurrent.futures
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deko3d_variant_cache as dvc  # noqa: E402
import deko3d_glsl_opt as opt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UAM = dvc.UAM
EMITTER = os.path.join(ROOT, 'app', 'thirdparty', 'uam', 'mesa-imported', 'codegen', 'nv50_ir_emit_gm107.cpp')
CACHE = os.path.join(ROOT, 'out', 'deko3d_shaders', 'stats_cache.json')
CACHE_VERSION = 3

# Emitter function -> class.
CLASSES = {
    'tex': 'TEX TEXS TLD TLD4 TXD TMML TXQ',
    'ldc': 'LDC',
    'scratch': 'LDL STL',
    'mem': 'LD ST LDG STG LDS STS ATOM ATOMS RED CCTL SUREDx SULDx SUSTx',
    'attr': 'ALD AST IPA ISBERD AL2P OUT PIXLD',
    'mufu': 'MUFU RRO',
    'f32': 'FADD FMUL FFMA FMNMX FSET FSETP FCMP FSWZADD FADD32I FMUL32I FFMA32I',
    'f64': 'DADD DMUL DFMA DMNMX DSET DSETP',
    'int': 'IADD IMUL IMAD XMAD ISCADD IMNMX ISET ISETP ICMP SHL SHR SHF LOP LOP3 NOT POPC BFE BFI FLO IADD3 LEA',
    'conv': 'F2F F2I I2F I2I',
    'move': 'MOV SEL PRMT SHFL S2R CS2R PSETP P2R R2P CSET CSETP',
    'flow': 'BRA JMP JMX BRX CAL JCAL RET PRET SSY SYNC PBK BRK PCNT CONT EXIT KIL BAR DEPBAR SAM RAM',
    'nop': 'NOP',
}
CLASS_OF = {name: cls for cls, names in CLASSES.items() for name in names.split()}


def opcode_table():
    """(value, prefix bits, name) for each opcode the emitter writes; the prefix ends at the value's lowest set bit of
    its top 16 bits (the opcode field; a longer prefix that also matches wins)."""
    table = []
    func = None
    for line in open(EMITTER, encoding='utf-8'):
        m = re.match(r'\s*CodeEmitterGM107::emit(\w+)\s*\(', line)
        if m:
            func = m.group(1)
        if 'emitInsn' not in line or func is None:
            continue
        comment = re.search(r'//\s*([A-Z][A-Z0-9]+)', line)
        for v in re.findall(r'0x[0-9a-fA-F]{8}', line):
            value = int(v, 16)
            top = value >> 16
            if top == 0:
                continue
            bits = 16 - ((top & -top).bit_length() - 1)
            table.append((value >> 16, bits, comment.group(1) if comment else func))
    table.sort(key=lambda e: -e[1])
    return table


TABLE = None


def classify(hi):
    top = hi >> 16
    if top >> 12 == 0x3:
        top &= ~0x0100  # immediate forms (0011...): bit 56 is the immediate's sign, not opcode

    for value, bits, name in TABLE:
        if (top >> (16 - bits)) == (value >> (16 - bits)):
            return name
    return '?'


def dksh_stats(dksh):
    magic, header_sz, control_sz, code_sz, programs_off, num_programs = struct.unpack_from('<6I', dksh, 0)
    ptype, entry, gprs, cb_off, cb_sz, scratch = struct.unpack_from('<6I', dksh, programs_off)
    start = control_sz + entry + 0x50
    end = control_sz + (cb_off if cb_sz else code_sz)
    code = dksh[start:end]
    ops = collections.Counter()
    count = 0
    # Groups of 32 bytes: a scheduling word, then three instructions.
    last_real = 0
    names = []
    for g in range(0, len(code) - 31, 32):
        for i in range(1, 4):
            lo, hi = struct.unpack_from('<II', code, g + i * 8)
            names.append(None if (lo == 0 and hi == 0) else classify(hi))
    # The code ends with EXIT then a branch to itself and NOPs to the group's end: count up to the last EXIT... not
    # before the last non-NOP, non-zero instruction.
    while names and names[-1] in (None, 'NOP'):
        names.pop()
    if names and names[-1] in ('BRA', 'JMP'):
        names.pop()
    while names and names[-1] in (None, 'NOP'):
        names.pop()
    for n in names:
        if n is None:
            n = 'NOP'
        ops[n] += 1
    count = len(names)
    classes = collections.Counter()
    for n, c in ops.items():
        classes[CLASS_OF.get(n, 'other')] += c
    return {'instr': count, 'gprs': gprs, 'scratch': scratch, 'ops': dict(ops), 'classes': dict(classes)}


def uam_id(path, flags=None):
    """The compiler's identity in the stats cache: its size and modification time (and the banjo flags it ran with,
    UAM_BANJO_FLAGS, when not its default)."""
    st = os.stat(path)
    return f'{st.st_size:x}-{int(st.st_mtime):x}' + (f'-f{flags}' if flags is not None else '')


def compile_stats(item, uam=None, flags=None):
    key, stage, text = item
    env = dict(os.environ, UAM_BANJO_FLAGS=str(flags)) if flags is not None else None
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, 'v.glsl')
        out = os.path.join(tmp, 'v.dksh')
        with open(src, 'wb') as f:
            f.write(text)
        r = subprocess.run([uam or UAM, '-s', stage, '-o', out, src], capture_output=True, text=True, env=env)
        if r.returncode != 0 or not os.path.isfile(out):
            return key, None, (r.stdout + r.stderr).strip()[:400]
        return key, dksh_stats(open(out, 'rb').read()), ''


def gpu_weights(paths):
    """ms per PS hash summed over the GPU profile blocks of the logs, and the blocks' total."""
    ms = collections.Counter()
    total = 0.0
    line_re = re.compile(r'^\s+([0-9.]+) ms\s+(.*)$')
    head_re = re.compile(r'GPU profile: ([0-9.]+) ms per frame over (\d+) frames')
    for path in paths:
        in_block = False
        for line in open(path, encoding='utf-8', errors='replace'):
            h = head_re.search(line)
            if h:
                in_block = float(h.group(1)) > 10.0  # the gameplay windows (not the menus)
                if in_block:
                    total += float(h.group(1))
                continue
            if not in_block:
                continue
            m = line_re.match(line)
            if not m:
                in_block = False
                continue
            ps = re.search(r'PS ([0-9A-F]{16})', m.group(2))
            if ps:
                ms[ps.group(1)] += float(m.group(1))
    return ms, total


def collect(glsl_dir, lists, blend_skip, passes=None, ps_spec=None):
    """{(prefix, hash): [(key, stage, text), ...]} for the lists' variants, and the pipelines' (vs, ps) pairs.
    passes: tools/deko3d_glsl_opt.py passes applied to each shader first; ps_spec: constants added to every pixel
    shader variant the list does not give them for (an alpha function for a list without it)."""
    pipelines = []
    for path in lists:
        pipelines += dvc.read_list(path, blend_skip)
    cache = {}
    variants = collections.defaultdict(list)
    pairs = collections.Counter()
    seen = set()
    def shader_of(prefix, h):
        name = f'{prefix}_{h}.glsl'
        if name not in cache:
            p = os.path.join(glsl_dir, name)
            text = open(p, 'rb').read().decode('utf-8') if os.path.isfile(p) else None
            if text is not None and passes:
                text = opt.optimise(text, passes, 'frag' if prefix == 'ps' else 'vert')
            cache[name] = (text, dvc.spec_defaults(text)) if text is not None else None
        return cache[name]

    for (vs_hash, ps_hash), spec in pipelines:
        pairs[(vs_hash, ps_hash)] += 1
        vs_shader = shader_of('vs', vs_hash)
        no_ps = ps_hash == '0000000000000000'
        ps_shader = None if no_ps else shader_of('ps', ps_hash)
        for prefix, h, stage in (('vs', vs_hash, 'vert'), ('ps', ps_hash, 'frag')):
            if h == '0000000000000000':
                continue
            shader = vs_shader if prefix == 'vs' else ps_shader
            if shader is None:
                continue
            vspec = dvc.stage_spec(prefix, spec, vs_shader[0] if vs_shader else None,
                                   ps_shader[0] if ps_shader else None, not no_ps)
            if prefix == 'ps' and ps_spec:
                merged = dict(ps_spec)
                merged.update(vspec)
                vspec = merged
            defines = dvc.variant_defines(shader[1], vspec)
            if (prefix, h, defines) in seen:
                continue
            seen.add((prefix, h, defines))
            text = dvc.variant_text(shader[0], defines).encode('utf-8')
            variants[(prefix, h)].append((dvc.fnv1a64(text), stage, text))
    return variants, pairs


def measure(variants, jobs, cache, uam=None, flags=None):
    uam = uam or UAM
    tag = uam_id(uam, flags)
    work = {}
    for items in variants.values():
        for key, stage, text in items:
            if f'{tag}:{key:016X}' not in cache:
                work[key] = (key, stage, text)
    if work:
        print(f'deko3d_shader_stats: compiling {len(work)} variants', flush=True)
        failed = 0
        with concurrent.futures.ThreadPoolExecutor(jobs) as ex:
            for i, (key, stats, err) in enumerate(ex.map(lambda it: compile_stats(it, uam, flags), work.values())):
                if stats is None:
                    failed += 1
                    if failed <= 5:
                        print(f'  uam failed for {key:016X}: {err}')
                    continue
                cache[f'{tag}:{key:016X}'] = stats
                if (i + 1) % 500 == 0:
                    print(f'  {i + 1}/{len(work)}', flush=True)
    # Per shader: the mean over its used variants (each variant draws).
    out = {}
    for (prefix, h), items in variants.items():
        rows = [cache[f'{tag}:{k:016X}'] for k, _, _ in items if f'{tag}:{k:016X}' in cache]
        if not rows:
            continue
        n = len(rows)
        classes = collections.Counter()
        for r in rows:
            classes.update(r['classes'])
        out[f'{prefix}_{h}'] = {
            'variants': n,
            'instr': sum(r['instr'] for r in rows) / n,
            'gprs': max(r['gprs'] for r in rows),
            'scratch': max(r['scratch'] for r in rows),
            'classes': {c: v / n for c, v in classes.items()},
        }
    return out


def main():
    global TABLE
    ap = argparse.ArgumentParser()
    ap.add_argument('--list', action='append', default=[])
    ap.add_argument('--gpu-log', action='append', default=[])
    ap.add_argument('--glsl', default=os.path.join(ROOT, 'out', 'deko3d_shaders', 'glsl'))
    ap.add_argument('--base-glsl', default=None)
    ap.add_argument('--base-json', default=None)
    ap.add_argument('--json', default=None)
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--blend-skip', action='store_true')
    ap.add_argument('--passes', default='', help='tools/deko3d_glsl_opt.py passes applied to --glsl first')
    ap.add_argument('--base-passes', default='', help='the same for --base-glsl')
    ap.add_argument('--uam', default=UAM, help='the uam to measure with')
    ap.add_argument('--base-uam', default=None, help='the uam of the comparison (default --uam)')
    ap.add_argument('--flags', type=int, default=None, help='UAM_BANJO_FLAGS for --uam (default: its own)')
    ap.add_argument('--base-flags', type=int, default=None,
                    help='UAM_BANJO_FLAGS of the comparison; without --base-glsl it compares the same GLSL')
    ap.add_argument('--ps-spec', default='', help='id=value,... added to every pixel shader variant (e.g. 903=8)')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    TABLE = opcode_table()
    cache = {}
    if os.path.isfile(CACHE):
        c = json.load(open(CACHE, encoding='utf-8'))
        if c.get('version') == CACHE_VERSION:
            cache = c['entries']

    ps_spec = {int(k): int(v) for k, v in (p.split('=') for p in a.ps_spec.split(',') if p)}
    variants, pairs = collect(a.glsl, a.list, a.blend_skip, a.passes, ps_spec)
    now = measure(variants, a.jobs, cache, a.uam, a.flags)
    base = None
    if a.base_glsl:
        bvariants, _ = collect(a.base_glsl, a.list, a.blend_skip, a.base_passes)
        base = measure(bvariants, a.jobs, cache, a.base_uam or a.uam, a.base_flags)
    elif a.base_flags is not None or a.base_uam:
        base = measure(variants, a.jobs, cache, a.base_uam or a.uam, a.base_flags)
    elif a.base_json:
        base = json.load(open(a.base_json, encoding='utf-8'))['shaders']
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    json.dump({'version': CACHE_VERSION, 'entries': cache}, open(CACHE + '.tmp', 'w', encoding='utf-8'))
    os.replace(CACHE + '.tmp', CACHE)

    ms, total = gpu_weights(a.gpu_log) if a.gpu_log else ({}, 0.0)
    # A vertex shader's weight: the ms of the pixel shaders it is drawn with, split by how many pipelines pair them.
    ps_pipes = collections.Counter()
    for (vs, ps), n in pairs.items():
        ps_pipes[ps] += n
    vs_ms = collections.Counter()
    for (vs, ps), n in pairs.items():
        if ps in ms and ps_pipes[ps]:
            vs_ms[vs] += ms[ps] * n / ps_pipes[ps]
    weight = {}
    for name in now:
        prefix, h = name.split('_', 1)
        weight[name] = (ms.get(h, 0.0) if prefix == 'ps' else vs_ms.get(h, 0.0))
    wsum = {p: sum(w for n, w in weight.items() if n.startswith(p)) or 1.0 for p in ('ps', 'vs')}

    def summary(data, prefix):
        rows = [(n, d) for n, d in data.items() if n.startswith(prefix)]
        if not rows:
            return {}
        instr = sorted(d['instr'] for _, d in rows)
        gprs = sorted(d['gprs'] for _, d in rows)
        wi = sum(d['instr'] * weight.get(n, 0.0) for n, d in rows) / wsum[prefix]
        classes = collections.Counter()
        for n, d in rows:
            for c, v in d['classes'].items():
                classes[c] += v * weight.get(n, 0.0) / wsum[prefix]
        return {'shaders': len(rows), 'median_instr': instr[len(instr) // 2], 'p90_instr': instr[len(instr) * 9 // 10],
                'median_gprs': gprs[len(gprs) // 2], 'max_gprs': gprs[-1],
                'scratch_shaders': sum(1 for _, d in rows if d['scratch']),
                'weighted_instr': wi, 'weighted_classes': dict(classes)}

    report = {'gpu_ms_total': total, 'summary': {p: summary(now, p) for p in ('ps', 'vs')}, 'shaders': now}
    if base is not None:
        report['base_summary'] = {p: summary(base, p) for p in ('ps', 'vs')}
    for p in ('ps', 'vs'):
        s = report['summary'][p]
        if not s:
            continue
        line = (f'{p}: {s["shaders"]} shaders, median {s["median_instr"]:.0f} instructions (p90 {s["p90_instr"]:.0f}), '
                f'GPRs median {s["median_gprs"]} max {s["max_gprs"]}, {s["scratch_shaders"]} with scratch; '
                f'GPU-time weighted {s["weighted_instr"]:.0f} instructions')
        if base is not None and report['base_summary'][p]:
            b = report['base_summary'][p]
            line += (f' (before: median {b["median_instr"]:.0f}, weighted {b["weighted_instr"]:.0f}: '
                     f'{100.0 * (s["weighted_instr"] - b["weighted_instr"]) / max(b["weighted_instr"], 1):+.1f} %)')
        print(line)
        wc = s['weighted_classes']
        print('   weighted classes: ' + ', '.join(f'{c} {v:.0f}' for c, v in sorted(wc.items(), key=lambda e: -e[1])))
    if total:
        print(f'GPU profile: {total:.0f} ms of frames summed; the pixel shaders below are the heaviest by it')
    heavy = sorted((n for n in now if n.startswith('ps')), key=lambda n: -weight.get(n, 0.0))[:a.top]
    for n in heavy:
        d = now[n]
        cl = d['classes']
        row = (f'  {n}  {weight.get(n, 0.0):8.1f} ms  {d["instr"]:6.0f} instr  {d["gprs"]:2d} GPRs'
               f'{"  scratch " + str(d["scratch"]) if d["scratch"] else ""}  tex {cl.get("tex", 0):.0f}'
               f'  ldc {cl.get("ldc", 0):.0f}  f32 {cl.get("f32", 0):.0f}  int {cl.get("int", 0):.0f}'
               f'  move {cl.get("move", 0):.0f}  flow {cl.get("flow", 0):.0f}')
        if base is not None and n in base:
            b = base[n]
            row += f'   (before {b["instr"]:.0f} / {b["gprs"]} GPRs: {d["instr"] - b["instr"]:+.0f})'
        print(row)
    if a.json:
        json.dump(report, open(a.json, 'w', encoding='utf-8'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
