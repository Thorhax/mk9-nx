#!/usr/bin/env python3
"""banjo-nx (banjo_gpu_early_alpha for Renut's renderer, package 27): tools/early_alpha.py on Renut's translator's HLSL.

Renut's translator (thirdparty/reblue-XenosRecomp, RENUT_RECOMP) writes a pixel shader's alpha test as
`alphaTest(oC0.w, g_AlphaTestFunc, g_AlphaThreshold);` at each exit, its constants as g_C(N), its predicated blocks as
`{ const bool pbN = ...; r = select(pbN, (T)(...), r); }` (banjo_pred_select.h) and its bools as BANJO_BOOL_N
(banjo_bool_spec.h). This wraps early_alpha.py without changing it: each shader's text is put into the forms that
tool reads (one line for one, but the alpha test, two), the tool runs (the slice, the copy and its self-check against
the original on random inputs, as for banjo-nx's translator), and only what it adds, the copy and its call site, goes
back into the shader's own text, in Renut's forms:

  - the copy's `return banjoAlphaFailsEarly(oC0.w);` tests the draw's alpha function and reference at run time
    (banjoAlphaFailsEarlyRt in shader_common.h: the same table and the same margin as banjo-nx's), where banjo-nx's
    renderer specialises them;
  - the call site runs under `g_AlphaTestFunc != 7u` (a uniform branch: the alpha test is on) instead of banjo-nx's
    specialization bit; banjoEarlyAlphaQuad discards a 2x2 quad only when its four pixels all fail.

Usage: renut_early_alpha.py <folder of dumped ps_*.hlsl> [--trials N]   (rewrites the shaders that get a copy in place;
the translator's second run compiles them, BANJO_HLSL_OVERRIDE). Writes <folder>/../renut_early_alpha.txt.
"""
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import early_alpha as ea  # noqa: E402

# Renut's constant registers and its scalar locals.
ea.CONST_RE = re.compile(r'g_C\((\d+)\)')
ea.const_text = lambda n: f'g_C({n})'
_var_width = ea.var_width


# Renut's per-draw values main() reads directly (shared constants): random per trial in the self-check, as the tool
# treats its own per-draw vectors.
# banjo-nx (2026-10-19): g_PixelOrigin, the Jinjo Bingo fix's tile corner (Renut 574b488). Missing from this list, every
# shader reading the pixel position lost its early alpha copy ("unsupported: variable g_PixelOrigin", 3604 shaders):
# package 35's fur shells about twice as slow on the console.
UNIFORM_WIDTHS = {'g_ResolutionScale': 1, 'g_ViewportSize': 2, 'g_ScreenOffset': 2, 'g_ScreenScale': 2, 'g_PixelOrigin': 2}
ea.UNIFORM_VECTORS.update(UNIFORM_WIDTHS)


def var_width(name):
    if name in UNIFORM_WIDTHS:
        return UNIFORM_WIDTHS[name]
    if name == 'lod' or re.fullmatch(r'pb\d+', name):
        return 1
    if re.fullmatch(r'i(?:Normal|Tangent|Binormal)\d', name):
        return 4
    if name == 'iPointCoord':
        return 2
    return _var_width(name)


ea.var_width = var_width
ea.CONSTRUCTORS = dict(ea.CONSTRUCTORS, bool=1)
# The block predicates are assigned in the copies too (`pbN = p0;`).
ea.ASSIGN_RE = re.compile(r'^(r\d+|oC\d|ps|p0|banjoTmp\d+|pb\d+)(?:\.([xyzw]{1,4}))?\s*=\s*(.*);$')

# asfloat(0x...u): a literal, as early_alpha's own literal declarations hold them.
_primary = ea.Parser.primary


def primary(self):
    k, v = self.peek()
    if k == 'id' and v == 'asfloat' and self.i + 3 < len(self.t) and self.t[self.i + 1] == ('op', '(') and \
            self.t[self.i + 2][0] == 'hex' and self.t[self.i + 3] == ('op', ')'):
        word = self.t[self.i + 2][1]
        self.i += 4
        return ('num', f'asfloat({word if word.endswith("u") else word + "u"})')
    return _primary(self)


ea.Parser.primary = primary

ALPHA_TEST = 'alphaTest(oC0.w, g_AlphaTestFunc, g_AlphaThreshold);'
BLEND_SKIP = 'banjoBlendSkip(oC0);'
TRIM_RE = re.compile(r'oC([0-3]) = banjoTrim\(oC\1, \1u\);')  # banjo_gpu_output_trim (package 28)
CAST_RE = re.compile(r'\((float[234]?|bool)\)\(')


def to_tool(lines):
    """The tool's forms, and for each of its lines the index of the shader's own line it came from."""
    out, origin = [], []
    for i, l in enumerate(lines):
        s = l.strip()
        indent = l[:len(l) - len(l.lstrip())]
        if s == ALPHA_TEST:
            out += [indent + '[branch] if (g_SpecConstants() & SPEC_CONSTANT_ALPHA_TEST) { clip(alphaTestValue(oC0.w));',
                    indent + '}']
            origin += [i, i]
            continue
        if s == BLEND_SKIP:
            out.append(indent + 'if ((g_SpecConstants() & SPEC_CONSTANT_BLEND_SKIP_ALPHA) != 0 && true) discard;')
            origin.append(i)
            continue
        if TRIM_RE.fullmatch(s):
            # After the alpha test: not part of what the alpha needs. A comment for the tool; the shader keeps it.
            out.append(indent + '// ' + s)
            origin.append(i)
            continue
        t = re.sub(r'^const bool (pb\d+) = ', r'bool \1 = ', s)
        # A kill: clip(select(c, -1.0, 1.0)) is the tool's clip(banjoSel(c, -1, 1)).
        km = re.fullmatch(r'clip\(select\((.*), -1\.0, 1\.0\)\);', t)
        if km:
            t = f'clip(banjoSel({km.group(1)}, -1, 1));'
        t = CAST_RE.sub(r'\1(', t)
        t = re.sub(r'\bBANJO_BOOL_(\d+)\b', r'BANJO_BOOL(\1)', t)
        out.append(indent + t if t != s else l)
        origin.append(i)
    return out, origin


def to_renut(text):
    """The copy and its call site, from the tool's forms to Renut's."""
    text = text.replace('return banjoAlphaFailsEarly(oC0.w);',
                        'return banjoAlphaFailsEarlyRt(oC0.w, g_AlphaTestFunc, g_AlphaThreshold);')
    text = text.replace('[branch] if (g_SpecConstants() & SPEC_CONSTANT_EARLY_ALPHA)', 'if (g_AlphaTestFunc != 7u)')
    # A constructor of one scalar is a broadcast: HLSL writes it as a cast.
    text = re.sub(r'\b(float[234])\(([^(),]+)\)', lambda m: m.group(0) if ',' in m.group(2) else
                  f'(({m.group(1)})({m.group(2)}))', text)
    if 'g_SpecConstants' in text or 'SPEC_CONSTANT' in text:
        raise ea.Unsupported('banjo-nx form left in the copy')
    return text


def process(path, trials):
    own = open(path, encoding='utf-8').read()
    lines = own.split('\n')
    tool_lines, origin = to_tool(lines)
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(tool_lines))
    try:
        outcome, info = ea.process(path, trials)
        if outcome != 'copied':
            # No copy: the shader's own text back (not the tool's forms, which only the tool reads).
            open(path, 'w', encoding='utf-8', newline='\n').write(own)
            return outcome, info
        done = open(path, encoding='utf-8').read().split('\n')
        # The copy: from the marker line to the closing brace (and its blank line); the site: its marker to '\t}'.
        f0 = next(i for i, l in enumerate(done) if l.startswith(f'// {ea.MARK} (tools/early_alpha.py)'))
        f1 = next(i for i in range(f0, len(done)) if done[i] == '}')
        s0 = next(i for i, l in enumerate(done) if l == f'\t// {ea.MARK}: a 2x2 quad whose pixels all fail the alpha '
                                                         'test stops here.')
        s1 = next(i for i in range(s0, len(done)) if done[i] == '\t}')
        func = to_renut('\n'.join(done[f0:f1 + 1]))
        site = to_renut('\n'.join(done[s0:s1 + 1]))
        # Where the site went in the tool's lines: after s0 - (lines added before it: the copy and its blank line).
        tool_insert = s0 - (f1 + 1 - f0) - 1
        start = next(i for i, l in enumerate(lines) if l.startswith('void main('))
        own_insert = origin[tool_insert] if tool_insert < len(origin) else len(lines)
        # The tool's insertion point falls after a whole statement; an alpha test's two lines map to one.
        if tool_insert > 0 and origin[tool_insert - 1] == own_insert:
            own_insert += 1
        out = lines[:start] + func.split('\n') + [''] + lines[start:own_insert] + site.split('\n') + lines[own_insert:]
        open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
        return outcome, info
    except Exception:
        open(path, 'w', encoding='utf-8', newline='\n').write(own)
        raise


def one(job):
    path, trials = job
    name = os.path.basename(path)[:-5]
    try:
        outcome, info = process(path, trials)
    except ea.Unsupported as e:
        outcome, info = 'unsupported', dict(reason=str(e))
    except (RecursionError, StopIteration, IndexError) as e:
        outcome, info = 'unsupported', dict(reason=type(e).__name__)
    return name, outcome, info


def main():
    args = sys.argv[1:]
    folder = args[0]
    trials = int(args[args.index('--trials') + 1]) if '--trials' in args else 24
    jobs = int(args[args.index('--jobs') + 1]) if '--jobs' in args else max(1, (os.cpu_count() or 2) // 2)
    paths = sorted(glob.glob(os.path.join(folder, 'ps_*.hlsl')))
    if jobs > 1:
        import multiprocessing
        with multiprocessing.Pool(jobs) as pool:
            results = pool.map(one, [(p, trials) for p in paths], chunksize=16)
    else:
        results = [one((p, trials)) for p in paths]
    counts, report = {}, []
    for name, outcome, info in results:
        counts[outcome] = counts.get(outcome, 0) + 1
        report.append(f'{name} {outcome} {info}')
    with open(os.path.join(folder, '..', 'renut_early_alpha.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print('banjo-nx: early alpha (renut_early_alpha.py): ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))


if __name__ == '__main__':
    main()
