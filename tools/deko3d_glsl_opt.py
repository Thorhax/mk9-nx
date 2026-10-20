#!/usr/bin/env python3
"""banjo-nx (2026-10-18, docs/deko3d-shader-optimisations.md): GLSL passes for uam, after tools/deko3d_glsl.py.

    python tools/deko3d_glsl_opt.py --in out/deko3d_shaders/glsl --out out/deko3d_shaders/glsl_opt [--passes a,b,...]
    python tools/deko3d_glsl_opt.py --list

uam is Mesa's GLSL compiler and nouveau's codegen (GLSL IR -> TGSI -> nv50 IR), not NIR: what NAK and the D3D
compilers fold for themselves it often leaves in. Each pass rewrites a form of SPIRV-Cross's output (of Renut's
XenosRecomp SPIR-V) into one that uam compiles to less code, with the same results. tools/deko3d_shader_stats.py
measures each (--passes there too). The passes and why are in PASSES below and in the document.

Only the game shaders (vs_/ps_<hash>.glsl) are rewritten; the text is what the console's uam compiles, so the
DKSH cache (tools/deko3d_variant_cache.py --glsl <out>) and the package (tools/deko3d_shader_package.py --glsl <out>)
take the output directory. Game data: the build tree only.
"""
import argparse
import os
import re
import types
import sys

# ---------------------------------------------------------------------------------------------------------------------
# helpers


def matching(text, open_index, open_ch, close_ch):
    depth = 0
    for i in range(open_index, len(text)):
        c = text[i]
        if c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
    return -1


def main_span(text):
    """(start, end) of main()'s body, or None."""
    m = re.search(r'\nvoid main\(\)\n\{', text)
    if not m:
        return None
    open_i = m.end() - 1
    close_i = matching(text, open_i, '{', '}')
    if close_i < 0:
        return None
    return open_i + 1, close_i


def insert_before_main(text, code):
    at = text.find('\nvoid main()')
    if at < 0:
        return text
    return text[:at + 1] + code + text[at + 1:]


def insert_globals(text, code):
    """After the #version line and the banjo-dk / extension / define lines that follow it."""
    lines = text.split('\n')
    k = 1
    while k < len(lines) and (lines[k].startswith('//') or lines[k].startswith('#extension') or
                              lines[k].startswith('#define dkh')):
        k += 1
    lines[k:k] = code.rstrip('\n').split('\n')
    return '\n'.join(lines)


def map_main(text, fn):
    span = main_span(text)
    if not span:
        return text
    a, b = span
    return text[:a] + fn(text[a:b]) + text[b:]


# ---------------------------------------------------------------------------------------------------------------------
# pass: sel — `c ? a : b` -> dk_sel(c, a, b)

SEL_TYPES = [('float', 'mix(b, a, c)'), ('vec2', 'mix(b, a, bvec2(c))'), ('vec3', 'mix(b, a, bvec3(c))'),
             ('vec4', 'mix(b, a, bvec4(c))'), ('int', 'mix(b, a, c)'), ('ivec2', 'mix(b, a, bvec2(c))'),
             ('ivec3', 'mix(b, a, bvec3(c))'), ('ivec4', 'mix(b, a, bvec4(c))'), ('uint', 'mix(b, a, c)'),
             ('uvec2', 'mix(b, a, bvec2(c))'), ('uvec3', 'mix(b, a, bvec3(c))'), ('uvec4', 'mix(b, a, bvec4(c))'),
             ('bool', 'mix(b, a, c)'), ('bvec2', 'mix(b, a, bvec2(c))'), ('bvec3', 'mix(b, a, bvec3(c))'),
             ('bvec4', 'mix(b, a, bvec4(c))')]
SEL_HELPERS = ''.join(f'{t} dk_sel(bool c, {t} a, {t} b) {{ return {body}; }}\n' for t, body in SEL_TYPES)

def _is_assignment(s, i):
    """The '=' at i is an assignment (=, +=, <<=, ...), not part of ==, !=, <= or >=."""
    if i + 1 < len(s) and s[i + 1] == '=':
        return False  # the first of '=='
    prev = s[i - 1] if i > 0 else ''
    if prev in ('=', '!'):
        return False  # '==' or '!='
    if prev in ('<', '>'):
        return i > 1 and s[i - 2] == prev  # '<<=' / '>>=' assign, '<=' / '>=' compare
    return True  # '=' or a compound assignment


def _ternary_at(s, q):
    """For the '?' at q: (start, colon, end) of the conditional expression, or None."""
    # condition: back to the enclosing bracket, a comma, an assignment, a ternary operator, or the statement start
    depth = 0
    i = q - 1
    while i >= 0:
        c = s[i]
        if c in ')]':
            depth += 1
        elif c in '([':
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if c in ',;{}?:':
                break
            if c == '=' and _is_assignment(s, i):
                break
        i -= 1
    start = i + 1
    if re.match(r'\s*return\b', s[start:q]):
        start = start + s[start:q].index('return') + len('return')
    # then-part: to the ':' that pairs with this '?'
    depth = 0
    pending = 0
    j = q + 1
    colon = -1
    while j < len(s):
        c = s[j]
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0:
            if c == '?':
                pending += 1
            elif c == ':':
                if pending == 0:
                    colon = j
                    break
                pending -= 1
            elif c in ',;':
                return None
        j += 1
    if colon < 0:
        return None
    # else-part: to the enclosing bracket's close, a comma, ';', or a ':' of an enclosing conditional
    depth = 0
    pending = 0
    k = colon + 1
    while k < len(s):
        c = s[k]
        if c in '([':
            depth += 1
        elif c in ')]':
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if c in ',;':
                break
            if c == '?':
                pending += 1
            elif c == ':':
                if pending == 0:
                    break
                pending -= 1
        k += 1
    return start, colon, k


def pass_sel(text):
    """Every conditional expression a select: uam turns a ?: whose operands are not plain expressions (another ?:, a
    vector) into an if/else (SSY, branches, SYNC and moves), and a mix() with a bool selector into SEL/ICMP. The two
    operands of SPIRV-Cross's ?: are values SPIR-V's OpSelect computed both of anyway."""
    count = 0

    def body(s):
        nonlocal count
        lines = s.split('\n')
        for n, line in enumerate(lines):
            if '?' not in line or line.lstrip().startswith('#') or line.lstrip().startswith('//'):
                continue
            guard = 0
            while '?' in line and guard < 400:
                guard += 1
                done = False
                # the innermost first: a '?' whose three parts hold no other '?'
                for q in [m.start() for m in re.finditer(r'\?', line)]:
                    t = _ternary_at(line, q)
                    if t is None:
                        continue
                    start, colon, end = t
                    cond, a, b = line[start:q], line[q + 1:colon], line[colon + 1:end]
                    if '?' in cond or '?' in a or '?' in b:
                        continue
                    lead = cond[:len(cond) - len(cond.lstrip())]
                    trail = b[len(b.rstrip()):]
                    line = (line[:start] + lead + f'dk_sel({cond.strip()}, {a.strip()}, {b.strip()})' + trail +
                            line[end:])
                    count += 1
                    done = True
                    break
                if not done:
                    break
            lines[n] = line
        return '\n'.join(lines)

    out = map_main(text, body)
    if count:
        out = insert_before_main(out, SEL_HELPERS)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# pass: alpha — the alpha test function as specialization constant 903

ALPHA_WORD = 'g_UboShared.v[34u].w'
ALPHA_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_903\n#define SPIRV_CROSS_CONSTANT_ID_903 0u\n#endif\n'
              'const uint g_banjoAlphaFunc = SPIRV_CROSS_CONSTANT_ID_903;\n'
              '#define DK_ALPHA_FUNC ((g_banjoAlphaFunc != 0u) ? (g_banjoAlphaFunc - 1u) : g_UboShared.v[34u].w)\n')


def pass_alpha(text):
    """The alpha test's compare function (SharedConstants.alpha_test_func) as specialization constant 903, its value
    + 1 (0: read at run time): the pipeline key has it (render/pipeline.cpp), so a draw without the alpha test (7,
    always) has no compare chain and no discard, and a pixel shader without discard and depth output keeps the GPU's
    early depth test (the SPH's kills_pixels bit is off). The early-alpha block (translator: the quad's alpha test
    before the texture fetches) folds away with it."""
    if ALPHA_WORD not in text:
        return text
    span = main_span(text)
    if not span:
        return text
    a, b = span
    body = text[a:b].replace(ALPHA_WORD, 'DK_ALPHA_FUNC')
    text = text[:a] + body + text[b:]
    return insert_before_main(text, ALPHA_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: scale1 — every texture and target at the console's resolution

SCALE_CHAIN_RE = re.compile(
    r'\((\w+) == 1u\) \? 0\.666666686534881591796875 : \(\(\1 == 2u\) \? 1\.5 : \(\(\1 == 3u\) \? 2\.0 : '
    r'\(\(\1 == 4u\) \? 3\.0 : \(\(\1 == 5u\) \? 4\.0 : \(\(\1 == 6u\) \? 6\.0 : '
    r'\(\(\1 == 7u\) \? 1\.33333337306976318359375 : 1\.0\)\)\)\)\)\)')
SCALE_CODE_RE = re.compile(r'^(\s*uint (_\d+) = )\(([^;]*?) >> 26u\) & 7u;$', re.M)
ROUND_RE = re.compile(r'roundEven\((vec[234])\((uvec[234])\(textureSize\(([\w]+, \d+)\)\)\) / vec[234]\(1\.0\)\)')
ROUNDTRIP_RE = re.compile(r'(uvec[234])\((vec[234])\(\1\(textureSize\(([\w]+, \d+)\)\)\)\)')
SCALE_WORD = 'uintBitsToFloat(g_UboShared.v[83u].z)'


def pass_scale1(text):
    """The deko3d renderer draws every texture at the console's scale (render/surfaces.cpp and graphics_options.cpp
    force 1x on deko3d): a texture word's scale code (bits 26-28) is 0. The per-texture scale divisor (a seven-way
    select on the code, before every sample with a texel offset and every texel-coordinate fetch) and the roundEven of
    size / 1 with its float and integer conversions go."""
    text = SCALE_CODE_RE.sub(lambda m: f'{m.group(1)}0u;', text)
    text = SCALE_CHAIN_RE.sub('1.0', text)
    text = ROUND_RE.sub(lambda m: f'{m.group(1)}({m.group(2)}(textureSize({m.group(3)})))', text)
    text = ROUNDTRIP_RE.sub(lambda m: f'{m.group(1)}(textureSize({m.group(3)}))', text)
    return text


def pass_fragscale(text):
    """SharedConstants.resolution_scale is 1 on deko3d (1280x720 targets only): the gl_FragCoord / resolution_scale
    of every shader that reads the pixel position (the Xenos VPOS input: the depth of field, the screen-space
    shadows and effects) is gl_FragCoord, without its reciprocal and multiplies."""
    span = main_span(text)
    if span:
        a, b = span
        text = text[:a] + text[a:b].replace(SCALE_WORD, '1.0') + text[b:]
    return text


# ---------------------------------------------------------------------------------------------------------------------
# pass: alphamask — the generic alpha test as a bit test

ALPHA_HEAD = '!(dk_sel((DK_ALPHA_FUNC == 0u), false, (dk_sel((DK_ALPHA_FUNC == 1u), ('
ALPHA_CMP_RE = re.compile(r'^\(dk_sel\(\(DK_ALPHA_FUNC == 0u\), false, \(dk_sel\(\(DK_ALPHA_FUNC == 1u\), \((\S+) < (\S+)\), '
                          r'\(dk_sel\(\(DK_ALPHA_FUNC == 2u\), \(\1 == \2\), \(dk_sel\(\(DK_ALPHA_FUNC == 3u\), \(\1 <= \2\), '
                          r'\(dk_sel\(\(DK_ALPHA_FUNC == 4u\), \(\1 > \2\), \(dk_sel\(\(DK_ALPHA_FUNC == 5u\), \(\1 != \2\), '
                          r'\(dk_sel\(\(DK_ALPHA_FUNC == 6u\), \(\1 >= \2\), true\)\)[)]*$')
ALPHA_HELPER = ('bool dk_alpha_pass(uint f, float a, float r)\n{\n'
                '    // D3D compare functions are a mask: 1 less, 2 equal, 4 greater (0 never, 7 always); with a NaN only\n'
                '    // not-equal (5) and always (7) pass.\n'
                '    uint m = mix(mix(mix(5u, 2u, a == r), 4u, a > r), 1u, a < r);\n'
                '    return mix((f & m) != 0u, (f & 5u) == 5u, m == 5u);\n}\n')


def pass_alphamask(text):
    """The alpha test of a pipeline whose function is read at run time (constant 903 at 0: the generic pipeline a draw
    uses while its own is built, or banjo_deko3d_alpha_spec off): the seven-way compare chain (seven compares and
    selects) as the compare function's bit mask (D3D's functions are less | equal | greater bits). Runs after sel."""
    out = []
    pos = 0
    n = 0
    while True:
        at = text.find(ALPHA_HEAD, pos)
        if at < 0:
            out.append(text[pos:])
            break
        close = matching(text, at + 1, '(', ')')
        inner = text[at + 1:close + 1]  # the whole parenthesised chain after the '!'
        m = ALPHA_CMP_RE.match(inner)
        if close < 0 or not m:
            out.append(text[pos:at + 1])
            pos = at + 1
            continue
        out.append(text[pos:at])
        # Only where the function is read at run time: with constant 903 the chain folds to one compare (or to
        # nothing, and no discard, for 7), which the bit test would hide from uam (package 33 kept the discard).
        out.append(f'!((g_banjoAlphaFunc != 0u) ? {inner} : '
                   f'dk_alpha_pass(g_UboShared.v[34u].w, {m.group(1)}, {m.group(2)}))')
        pos = close + 1
        n += 1
    if not n:
        return text
    return insert_before_main(''.join(out), ALPHA_HELPER)


# ---------------------------------------------------------------------------------------------------------------------
# pass: pred — a condition known inside its own if

def pass_pred(text):
    """Inside `if (X) { ... }` (X a bool the block does not assign), an inner `if (X)` is `if (true)` and `if (!X)`
    `if (false)`, X anywhere else in the block is true (a bvec4(X) of a mix, a ?:), and the same in the else block
    the other way round. In the translator's predicated blocks the same predicate is tested again and again (the inner
    else blocks, their copies and the phis are dead), which Mesa's GLSL optimiser does not see."""
    span = main_span(text)
    if not span:
        return text
    a, b = span
    lines = text[a:b].split('\n')
    bools = set(re.findall(r'^\s*bool (_\d+)\b', text[a:b], re.M))
    if not bools:
        return text
    # the line of each block's closing brace, and the lines assigning each bool
    close_of = {}
    opened = []
    for n, line in enumerate(lines):
        st = line.strip()
        if st == '{':
            opened.append(n)
        elif (st == '}' or st.startswith('} while')) and opened:
            close_of[opened.pop()] = n
    assign_at = {}
    for n, line in enumerate(lines):
        for m in re.finditer(r'(?<![\w.])(_\d+)\s*=[^=]', line):
            if m.group(1) in bools:
                assign_at.setdefault(m.group(1), []).append(n)

    def stable(var, open_line):
        close = close_of.get(open_line, len(lines))
        return not any(open_line < k < close for k in assign_at.get(var, ()))

    stack = []      # (indent of the block's braces, var or None, value inside)
    pending = None  # (indent, var, value) of an if or else whose '{' comes next
    last_if = {}    # indent -> (var, value) of the if just closed (for its else)
    changed = 0
    for n, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        known = {var: val for _, var, val in stack if var is not None}
        m = re.match(r'^(\s*)(?:else )?if \((!?)(_\d+)\)$', line)
        if m and m.group(3) in known:
            val = known[m.group(3)] != (m.group(2) == '!')
            lines[n] = f'{m.group(1)}{"else " if stripped.startswith("else") else ""}if ({"true" if val else "false"})'
            changed += 1
        if m and m.group(3) in bools:
            pending = (indent, m.group(3), m.group(2) != '!')
            continue
        if stripped == 'else' and indent in last_if:
            var, val = last_if.pop(indent)
            pending = (indent, var, not val)
            continue
        if stripped == '{':
            if pending and pending[0] == indent and stable(pending[1], n):
                stack.append(pending)
            else:
                stack.append((indent, None, None))
            pending = None
            continue
        pending = None
        if stripped == '}' or stripped.startswith('} while'):
            if stack:
                ind, var, val = stack.pop()
                if var is not None and stripped == '}':
                    last_if[ind] = (var, val)
            continue
        last_if = {k: v for k, v in last_if.items() if k < indent}
        if known:
            def use(mm):
                v = mm.group(1)
                if v in known:
                    return 'true' if known[v] else 'false'
                return mm.group(0)
            new = re.sub(r'(?<![\w.])(_\d+)(?![\w\[])(?!\s*=[^=])', use, line)
            if new != line:
                lines[n] = new
                changed += 1
    if not changed:
        return text
    return text[:a] + '\n'.join(lines) + text[b:]


# ---------------------------------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------------------------------
# pass: tess — the vertex shader's tessellation kind as specialization constant 904

TESS_WORD = 'g_UboShared.v[65u].x'
TESS_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_904\n#define SPIRV_CROSS_CONSTANT_ID_904 0u\n#endif\n'
             'const uint g_banjoTessKind = SPIRV_CROSS_CONSTANT_ID_904;\n'
             '#define DK_TESS_KIND ((g_banjoTessKind != 0u) ? (g_banjoTessKind - 1u) : g_UboShared.v[65u].x)\n')


def pass_tess(text):
    """SharedConstants.tessellation[0] (the patch kind: 0 for a draw that is not tessellated) as specialization
    constant 904, its value + 1 (0: read at run time): render/pipeline.cpp gives 1 to every pipeline whose key is not a
    tessellated draw's (per_instance_streams 0), so the vertex shader's patch-grid code (the domain point from the
    vertex index: square roots, divisions, the four kinds' branches) goes, and its registers with it. No new
    pipelines: the key decides it."""
    if TESS_WORD not in text:
        return text
    span = main_span(text)
    if not span:
        return text
    a, b = span
    text = text[:a] + text[a:b].replace(TESS_WORD, 'DK_TESS_KIND') + text[b:]
    return insert_before_main(text, TESS_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: vsout — the vertex shader's outputs the pixel shader reads, as specialization constant 905

VSOUT_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_905\n#define SPIRV_CROSS_CONSTANT_ID_905 4294967295u\n#endif\n'
              'const uint g_banjoVsOutputs = SPIRV_CROSS_CONSTANT_ID_905;\n')
VSOUT_RE = re.compile(r'^layout\(location = (\d+)\) out (\w+) (\w+);$', re.M)


def pass_vsout(text):
    """Each write of a vertex shader output (layout location N) under `(g_banjoVsOutputs & (1u << N)) != 0u`,
    specialization constant 905: plume_deko3d gives a pipeline's vertex shader the locations its pixel shader reads (none
    for a depth-only pipeline: the shadow maps), so the outputs nothing reads, the arithmetic and the vertex fetches
    behind them go. An output the shader reads back is left alone."""
    outs = [(int(m.group(1)), m.group(3)) for m in VSOUT_RE.finditer(text) if int(m.group(1)) < 32]
    if not outs:
        return text
    span = main_span(text)
    if not span:
        return text
    a, b = span
    body = text[a:b]
    wrapped = 0
    for loc, name in outs:
        uses = list(re.finditer(r'(?<![\w.])' + re.escape(name) + r'\b', body))
        writes = re.findall(r'^\s*' + re.escape(name) + r'(?:\.[xyzw]+)?\s*=[^=][^;]*;$', body, re.M)
        if not writes or len(writes) != len(uses):
            continue  # read back somewhere, or written in a form not handled: kept
        body = re.sub(r'^(\s*)(' + re.escape(name) + r'(?:\.[xyzw]+)?\s*=[^=][^;]*;)$',
                      lambda m: f'{m.group(1)}if ((g_banjoVsOutputs & {1 << loc}u) != 0u) {{ {m.group(2)} }}',
                      body, flags=re.M)
        wrapped += 1
    if not wrapped:
        return text
    text = text[:a] + body + text[b:]
    return insert_before_main(text, VSOUT_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# passes: vszero / pszero — an input its vertex shader writes as zero (constant 906)

ZERO_RE = re.compile(r'^(?:(?:vec[234])\()?0\.0\)?$')
PSIN_RE = re.compile(r'^layout\(location = (\d+)\) in (\w+) (\w+);$', re.M)
PSZERO_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_906\n#define SPIRV_CROSS_CONSTANT_ID_906 0u\n#endif\n'
               'const uint g_banjoPsZeroIn = SPIRV_CROSS_CONSTANT_ID_906;\n')


def pass_vszero(text):
    """Records in a `// banjo-vszero=0x...` line the outputs the vertex shader writes as zero (one write of 0.0 or
    vecN(0.0), nothing else touching it): plume_deko3d gives the pipeline's pixel shader those of its inputs as constant
    906 (pszero). Renut's vertex shaders write every output; the point coordinate (location 23) is zero except under
    the point-sprite geometry shader, and 2124 pixel shaders read it."""
    span = main_span(text)
    if not span:
        return text
    body = text[span[0]:span[1]]
    mask = 0
    for m in VSOUT_RE.finditer(text):
        loc, name = int(m.group(1)), m.group(3)
        if loc >= 32:
            continue
        uses = re.findall(r'(?<![\w.])' + re.escape(name) + r'\b', body)
        writes = re.findall(r'^\s*' + re.escape(name) + r' = ([^;]*);$', body, re.M)
        if len(uses) == 1 and len(writes) == 1 and ZERO_RE.match(writes[0].strip()):
            mask |= 1 << loc
    if not mask:
        return text
    lines = text.split('\n')
    lines.insert(1, f'// banjo-vszero=0x{mask:x}')
    return '\n'.join(lines)


def pass_pszero(text):
    """Each input read through `DK_IN_<name>`: `vecN(0.0)` when bit <location> of constant 906 is set (its vertex
    shader writes it as zero, vszero), else the input. The interpolation and what depends on it go."""
    ins = [(int(m.group(1)), m.group(2), m.group(3)) for m in PSIN_RE.finditer(text) if int(m.group(1)) < 32]
    if not ins:
        return text
    span = main_span(text)
    if not span:
        return text
    a, b = span
    body = text[a:b]
    macros = ''
    for loc, typ, name in ins:
        if not re.search(r'(?<![\w.])' + re.escape(name) + r'\b', body):
            continue
        body = re.sub(r'(?<![\w.])' + re.escape(name) + r'\b', f'DK_IN_{name}', body)
        macros += (f'#define DK_IN_{name} (((g_banjoPsZeroIn & {1 << loc}u) != 0u) ? {typ}(0.0) : {name})\n')
    if not macros:
        return text
    text = text[:a] + body + text[b:]
    return insert_before_main(text, PSZERO_DECL + macros)


# ---------------------------------------------------------------------------------------------------------------------
# pass: bonerow — a relative constant read past the registers reads the zero register

BONEROW_RE = re.compile(r'mix\(vec4\(0\.0\), (g_Ubo(?:Vertex|Pixel))\.v\[min\((_\d+), 255u\)\], bvec4\(\2 < 256u\)\)')


def pass_bonerow(text):
    """Renut's relative constant reads (the bone rows of skinning: c[a0 + n]) are `index < 256 ? c[min(index, 255)] :
    0`, a select per component and a compare. The float constant uploads have a zero register 256 after the 256
    registers (render/upload_ring.h, kFloatConstantsUboBytes), so `c[min(index, 256)]` is the same value with no
    select: an index past the registers reads zeros."""
    out, n = BONEROW_RE.subn(lambda m: f'{m.group(1)}.v[min({m.group(2)}, 256u)]', text)
    if not n:
        return text
    for block in set(m.group(1) for m in BONEROW_RE.finditer(text)):
        out = re.sub(r'(uniform type_ConstantBuffer_RenutUboFloat\n\{\n\s*vec4 v\[)256(\];\n\} ' + block + ';)',
                     r'\g<1>257\2', out)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# pass: predmul — a predicated scale by one value

PREDMUL_RE = re.compile(r'mix\((_\d+(?:\.[xyzw]+)?), \1 \* (_\d+\.([xyzw])\3{1,3}|vec[234]\([0-9.e+-]+\)), '
                        r'bvec([234])\((_\d+)\)\)')


def pass_predmul(text):
    """The translator's predicated vector multiply by one scalar (a broadcast swizzle or a constant):
    `mix(a, a * k, bvec3(p))` (three multiplies and three selects) is `a * mix(vec3(1.0), k, bvec3(p))`: the selects
    are of the same two values, one select. Exact: a * 1.0 is a (uam's multiplies keep denormals; only a NaN's payload
    may change)."""
    def rep(m):
        n = m.group(4)
        return f'({m.group(1)} * mix(vec{n}(1.0), {m.group(2)}, bvec{n}({m.group(5)})))'
    return map_main(text, lambda body: PREDMUL_RE.sub(rep, body))


# ---------------------------------------------------------------------------------------------------------------------
# pass: sext — a signed bitfield as one extract

SEXT_RE = re.compile(r'int\(\(\((_\d+) >> (\d+)u\) & (\d+)u\) << (\d+)u\) >> (\d+)')


def pass_sext(text):
    """The vertex fetch's signed fields (10:10:10:2 normals, signed shorts): `int(((x >> s) & m) << k) >> k`, a shift,
    a mask, a shift and an arithmetic shift, is `bitfieldExtract(int(x), s, n)` (one BFE; GLSL's extract of an int
    sign-extends from bit n - 1, as the shift pair does). Only when m is n low bits and k = 32 - n."""
    def rep(m):
        x, sh, mask, k1, k2 = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        n = mask.bit_length()
        if mask != (1 << n) - 1 or k1 != k2 or k1 != 32 - n or sh + n > 32 or n == 0:
            return m.group(0)
        return f'bitfieldExtract(int({x}), {sh}, {n})'
    return map_main(text, lambda body: SEXT_RE.sub(rep, body))


# ---------------------------------------------------------------------------------------------------------------------
# pass: bool01 — a select of the two constants of a type is a conversion

BOOL01 = {('1.0', '0.0'): 'float({c})', ('0.0', '1.0'): 'float(!({c}))', ('1u', '0u'): 'uint({c})',
          ('0u', '1u'): 'uint(!({c}))', ('1', '0'): 'int({c})', ('0', '1'): 'int(!({c}))',
          ('true', 'false'): '({c})', ('false', 'true'): '(!({c}))'}


def pass_bool01(text):
    """`c ? 1.0 : 0.0` is float(c) (one FSET; the select is a compare and two moves), and the same for 0.0 / 1.0, the
    integers and the bools: the translator's predicates turned into numbers (setp results, the Xenos kill and pred
    instructions) are written so."""
    count = 0

    def body(s):
        nonlocal count
        lines = s.split('\n')
        for n, line in enumerate(lines):
            if '?' not in line:
                continue
            guard = 0
            while guard < 200:
                guard += 1
                done = False
                for q in [m.start() for m in re.finditer(r'\?', line)]:
                    t = _ternary_at(line, q)
                    if t is None:
                        continue
                    start, colon, end = t
                    cond, x, y = line[start:q], line[q + 1:colon].strip(), line[colon + 1:end].strip()
                    form = BOOL01.get((x, y))
                    if form is None or '?' in cond:
                        continue
                    lead = cond[:len(cond) - len(cond.lstrip())]
                    trail = line[colon + 1:end][len(line[colon + 1:end].rstrip()):]
                    line = line[:start] + lead + form.format(c=cond.strip()) + trail + line[end:]
                    count += 1
                    done = True
                    break
                if not done:
                    break
            lines[n] = line
        return '\n'.join(lines)

    return map_main(text, body)


# ---------------------------------------------------------------------------------------------------------------------
# pass: earlyhoist — the main body's first texture region ahead of Renut's early alpha test

EARLY_IF_RE = re.compile(r'\n([ \t]*)if \(DK_ALPHA_FUNC != 7u\)\n[ \t]*\{')
ASSIGNED_RE = re.compile(r'\b(_\d+)(?:\.[xyzw]+)?\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)')


def _next_statement(body, at):
    """(start, end) of the statement at 'at' (after blanks): a simple one up to its ';', or a `do { ... } while(false);`;
    None for anything else (if, for, a block, the end of the enclosing block)."""
    n = len(body)
    i = at
    while i < n and body[i] in ' \t\n':
        i += 1
    if i >= n or body[i] == '}':
        return None
    if re.match(r'do\s*\{', body[i:i + 8]):
        open_i = body.index('{', i)
        close_i = matching(body, open_i, '{', '}')
        if close_i < 0:
            return None
        m = re.match(r'\s*while\s*\(\s*false\s*\)\s*;', body[close_i + 1:])
        if not m:
            return None
        return i, close_i + 1 + m.end()
    if re.match(r'(if|for|while|switch|return|discard|break|continue|else)\b', body[i:]) or body[i] == '{':
        return None
    depth = 0
    for j in range(i, n):
        c = body[j]
        if c in '({[':
            depth += 1
        elif c in ')}]':
            depth -= 1
            if depth < 0:
                return None
        elif c == ';' and depth == 0:
            return i, j + 1
        elif c == '{':
            return None
    return None


def pass_earlyhoist(text):
    """Renut's early alpha test (a quad whose pixels all fail is discarded before the main body) computes the alpha as
    the main body does, and the main body's first region then fetches the same textures through the same uniform
    branches (the fur shader: a texture array or a 2D texture by a constant-buffer bit). uam's value numbering
    replaces what an equivalent dominating computation already did, so the main body's region goes first: its
    statements up to its first `do { } while(false)` (pure: no discard, no output, nothing the early test assigns), when
    that region fetches a texture the early test also fetches. uam then finds the early test's alpha in the region's
    results and drops the early copy; were the two not equivalent, the region would only run earlier (pure code).
    Specialization constant 907 (1 by default; banjo_deko3d_early_hoist off gives 0) picks the place: the region is
    written at both, its top-level declarations ahead of the first."""
    changed = []

    def body(b):
        m = EARLY_IF_RE.search(b)
        if not m:
            return b
        open_i = b.index('{', m.end() - 1)
        close_i = matching(b, open_i, '{', '}')
        if close_i < 0:
            return b
        early = b[m.start() + 1:close_i + 1]
        stmts = []
        at = close_i + 1
        while True:
            st = _next_statement(b, at)
            if not st:
                return b
            stmts.append(st)
            at = st[1]
            if b[st[0]:st[0] + 2] == 'do':
                break
            if len(stmts) > 12:
                return b
        start, end = stmts[0][0], stmts[-1][1]
        region = b[start:end]
        if re.search(r'\b(discard|return)\b|\bout_var_|gl_FragDepth|\bdFd[xy]|\bfwidth', region):
            return b
        # nothing the early test assigns (declared before it) may be read by the region
        declared_in_early = set(re.findall(r'\b(?:float|int|uint|bool|[biu]?vec[234]|mat[234])\s+(_\d+)\b', early))
        assigned = {v for v in ASSIGNED_RE.findall(early)} - declared_in_early
        if assigned & set(re.findall(r'\b_\d+\b', region)):
            return b
        # worth it only when the region fetches what the early test fetches
        tex_region = set(re.findall(r'\btexture\w*\((dk_tex_b\d+)', region))
        tex_early = set(re.findall(r'\btexture\w*\((dk_tex_b\d+)', early))
        if not (tex_region & tex_early):
            return b
        indent = m.group(1)
        # the region's top-level declarations go ahead of both places; in each it assigns
        decls, inner = [], []
        for a0, a1 in stmts:
            st = b[a0:a1]
            d = re.match(r'((?:float|int|uint|bool|[biu]?vec[234]|mat[234])\s+)(_\d+)\s*(=|;)', st)
            if not d and re.match(r'(const\s+)?(float|int|uint|bool|[biu]?vec[234]|mat[234]|[biu]?vec\d|\w+\s+_\d+\s*\[)\b',
                                  st):
                return b  # a declaration this does not split (const, array): left where it is
            if d:
                decls.append(f'{d.group(1)}{d.group(2)};')
                st = '' if d.group(3) == ';' else st[len(d.group(1)):]
            if st:
                inner.append(st)
        inner_text = ('\n' + indent + '    ').join(x.replace('\n', '\n    ') for x in inner)

        def place(cond):
            return f'{indent}if (g_banjoEarlyHoist {cond} 0u)\n{indent}{{\n{indent}    {inner_text}\n{indent}}}'

        changed.append(True)
        head = ''.join(f'{indent}{d}\n' for d in decls)
        return (b[:m.start() + 1] + head + place('!=') + '\n' + b[m.start() + 1:close_i + 1] + '\n' + place('==') +
                b[end:])
    out = map_main(text, body)
    if changed:
        out = insert_globals(out, '#ifndef SPIRV_CROSS_CONSTANT_ID_907\n#define SPIRV_CROSS_CONSTANT_ID_907 1u\n#endif\n'
                                  'const uint g_banjoEarlyHoist = SPIRV_CROSS_CONSTANT_ID_907;\n')
    return out


# ---------------------------------------------------------------------------------------------------------------------
# pass: ifmerge — consecutive ifs on the same unchanged condition variable as one

IF_VAR_RE = re.compile(r'\n([ \t]*)if \((!?)(_\d+)\)\n\1\{')
DECL_ONLY_RE = re.compile(r'\n([ \t]*)(?:float|int|uint|bool|[biu]?vec[234]|mat[234]) _\d+;')


def _if_end(b, m):
    """For an if matched by IF_VAR_RE: (then body, else body or None, end index)."""
    open_i = b.index('{', m.end() - 1)
    close_i = matching(b, open_i, '{', '}')
    if close_i < 0:
        return None
    then_body = b[open_i + 1:close_i]
    indent = m.group(1)
    em = re.match(r'\n' + re.escape(indent) + r'else\n' + re.escape(indent) + r'\{', b[close_i + 1:])
    if not em:
        return then_body, None, close_i + 1
    eo = close_i + 1 + em.end() - 1
    ec = matching(b, eo, '{', '}')
    if ec < 0:
        return None
    return then_body, b[eo + 1:ec], ec + 1


def pass_ifmerge(text):
    """SPIRV-Cross writes Renut's per-pixel branches as one if per value: the fur shader's shadow cascades test `_678`
    in four ifs in a row and `_914` in three (each its own branches and, the condition being per pixel, its own
    reconvergence: SSY, BRA, SYNC). Two ifs at the same level on the same condition variable (or its negation, the
    same both times), with only declarations between them and nothing in the first assigning the variable, are one if:
    the first's then and the second's then, the first's else and the second's else, in that order (the same statements
    run in the same order on each path). The declarations go ahead of the first."""
    def body(b):
        pos = 0
        while True:
            m = IF_VAR_RE.search(b, pos)
            if not m:
                return b
            first = _if_end(b, m)
            if not first:
                pos = m.end()
                continue
            then1, else1, end1 = first
            indent, neg, var = m.group(1), m.group(2), m.group(3)
            # declarations only between
            at = end1
            decls = []
            while True:
                dm = DECL_ONLY_RE.match(b, at)
                if not dm or dm.group(1) != indent:
                    break
                decls.append(b[at + 1:dm.end()].strip())
                at = dm.end()
            m2 = IF_VAR_RE.match(b, at)
            assigned = re.compile(r'\b' + re.escape(var) + r'\s*(?:[-+*/%&|^]|<<|>>)?=(?!=)')
            if (not m2 or m2.group(1) != indent or m2.group(2) != neg or m2.group(3) != var or
                    assigned.search(then1) or (else1 is not None and assigned.search(else1))):
                pos = m.end()
                continue
            second = _if_end(b, m2)
            if not second:
                pos = m.end()
                continue
            then2, else2, end2 = second
            merged = (''.join(f'\n{indent}{d}' for d in decls) +
                      f'\n{indent}if ({neg}{var})\n{indent}{{{then1.rstrip()}{then2.rstrip()}\n{indent}}}')
            if else1 is not None or else2 is not None:
                merged += f'\n{indent}else\n{indent}{{{(else1 or "").rstrip()}{(else2 or "").rstrip()}\n{indent}}}'
            b = b[:m.start()] + merged + b[end2:]
            pos = m.start()  # a third may follow
    return map_main(text, body)


# ---------------------------------------------------------------------------------------------------------------------
# pass: dowhile — `do { ... break; } while(false)` early exits as if / else

DO_RE = re.compile(r'\n([ \t]*)do\n\1\{')
LOOP_START_RE = re.compile(r'(do|for|while|switch)\b')


def _statements(body):
    """The top-level statements of a block's text: (start, end) spans; None when it has one this does not parse."""
    out = []
    i, n = 0, len(body)
    while True:
        while i < n and body[i] in ' \t\n':
            i += 1
        if i >= n:
            return out
        if body[i] == '{':
            j = matching(body, i, '{', '}')
            if j < 0:
                return None
            out.append((i, j + 1))
            i = j + 1
            continue
        m = re.match(r'(if|for|while|switch)\s*\(', body[i:])
        if m:
            p0 = i + m.end() - 1
            p1 = matching(body, p0, '(', ')')
            if p1 < 0:
                return None
            k = p1 + 1
            while k < n and body[k] in ' \t\n':
                k += 1
            if k >= n or body[k] != '{':
                return None
            e = matching(body, k, '{', '}')
            if e < 0:
                return None
            end = e + 1
            if m.group(1) == 'if':
                em = re.match(r'\s*else\b', body[end:])
                while em:
                    k = end + em.end()
                    while k < n and body[k] in ' \t\n':
                        k += 1
                    if body.startswith('if', k):  # else if: one more if statement
                        q0 = body.index('(', k)
                        q1 = matching(body, q0, '(', ')')
                        k = q1 + 1
                        while k < n and body[k] in ' \t\n':
                            k += 1
                    if k >= n or body[k] != '{':
                        return None
                    e = matching(body, k, '{', '}')
                    if e < 0:
                        return None
                    end = e + 1
                    em = re.match(r'\s*else\b', body[end:])
            out.append((i, end))
            i = end
            continue
        if body.startswith('do', i) and re.match(r'do\s*\{', body[i:]):
            k = body.index('{', i)
            e = matching(body, k, '{', '}')
            if e < 0:
                return None
            m2 = re.match(r'\s*while\s*\([^;]*\)\s*;', body[e + 1:])
            if not m2:
                return None
            out.append((i, e + 1 + m2.end()))
            i = e + 1 + m2.end()
            continue
        depth = 0
        j = i
        while j < n:
            c = body[j]
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == '{':
                return None
            elif c == ';' and depth == 0:
                break
            j += 1
        if j >= n:
            return None
        out.append((i, j + 1))
        i = j + 1


def _breaks_outside_loops(text):
    """break / continue statements of 'text' that are not inside a nested loop or switch (they would leave ours)."""
    found = 0
    i = 0
    while True:
        m = re.search(r'\b(break|continue|do|for|while|switch)\b', text[i:])
        if not m:
            return found
        k = i + m.start()
        w = m.group(1)
        if w in ('break', 'continue'):
            found += 1
            i = k + len(w)
            continue
        # a nested loop or switch: skip its body (and a do's `while (...);`)
        b = text.find('{', k)
        if b < 0:
            return found + 1
        e = matching(text, b, '{', '}')
        if e < 0:
            return found + 1
        i = e + 1
        if w == 'do':
            wm = re.match(r'\s*while\s*\([^;]*\)\s*;', text[i:])
            if wm:
                i += wm.end()


def pass_dowhile(text):
    """SPIRV-Cross writes the early exits of Renut's shaders (the texture-array choice, the early alpha test, the main
    body's three fog / colour paths) as `do { ...; if (c) { ...; break; } ...; break; } while(false);`: uam compiles it
    as a loop (PBK, BRK, a loop for the value numbering of 2.26). When its breaks are only the last statement of ifs at
    its top level (no else) and its last statement, it is if / else: `if (c) { A } else { rest }`, in a block of its
    own. The same statements run on each path in the same order."""
    def convert(body):
        stmts = _statements(body)
        if stmts is None:
            return None
        pieces = []
        for idx, (a, b) in enumerate(stmts):
            st = body[a:b]
            ls = body.rfind('\n', 0, a) + 1
            ind = body[ls:a] if not body[ls:a].strip() else ''
            last = idx == len(stmts) - 1
            if st.strip() == 'break;':
                if not last:
                    return None
                pieces.append(('end', None))
                continue
            if st.startswith('if'):
                p0 = st.index('(')
                p1 = matching(st, p0, '(', ')')
                k = st.index('{', p1)
                e = matching(st, k, '{', '}')
                inner = st[k + 1:e]
                rest = st[e + 1:].strip()
                istmts = _statements(inner)
                if istmts and inner[istmts[-1][0]:istmts[-1][1]].strip() == 'break;' and not rest:
                    head = inner[:istmts[-1][0]]
                    if _breaks_outside_loops(head):
                        return None
                    pieces.append(('exit', (ind, st[:p1 + 1], head)))
                    continue
            if _breaks_outside_loops(st):
                return None
            pieces.append(('stmt', ind + st))
        if any(k == 'end' for k, _ in pieces[:-1]):
            return None
        # nest from the last exit outwards
        out = ''
        for kind, val in reversed(pieces):
            if kind == 'end':
                continue
            if kind == 'stmt':
                out = val + '\n' + out
            else:
                ind, cond, head = val
                out = f'{ind}{cond}\n{ind}{{{head.rstrip()}\n{ind}}}\n{ind}else\n{ind}{{\n{out}{ind}}}\n'
        return out

    def body(b):
        pos = 0
        while True:
            m = DO_RE.search(b, pos)
            if not m:
                return b
            open_i = b.index('{', m.end() - 1)
            close_i = matching(b, open_i, '{', '}')
            if close_i < 0:
                return b
            wm = re.match(r'\s*while\s*\(\s*false\s*\)\s*;', b[close_i + 1:])
            if not wm:
                pos = m.end()
                continue
            inner = b[open_i + 1:close_i]
            if 'continue' in inner and _breaks_outside_loops(inner) and re.search(r'\bcontinue\b', inner):
                pos = m.end()
                continue
            new = convert(inner)
            if new is None:
                pos = m.end()
                continue
            ind = m.group(1)
            b = b[:m.start()] + f'\n{ind}{{\n' + new + f'{ind}}}' + b[close_i + 1 + wm.end():]  # its own block
            pos = m.start()
    return map_main(text, body)


# ---------------------------------------------------------------------------------------------------------------------
# pass: selsame — a select inside a select on the same condition

def _split_args(text):
    """Top-level comma-separated arguments of 'text' (the inside of a call's parentheses)."""
    out, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if c in '([':
            depth += 1
        elif c in ')]':
            depth -= 1
        elif c == ',' and depth == 0:
            out.append(text[start:i])
            start = i + 1
    out.append(text[start:])
    return out


def _unparen(e):
    e = e.strip()
    while e.startswith('(') and matching(e, 0, '(', ')') == len(e) - 1:
        e = e[1:-1].strip()
    return e


def pass_selsame(text):
    """Renut's lights select each term on the light's flag twice: `dk_sel(c, (dk_sel(c, exp2(a), b)), d)`. With the same
    condition text, the inner select's other value can never be chosen: `dk_sel(c, x, d)` (and `dk_sel(c, a, dk_sel(c, x,
    y))` is `dk_sel(c, a, y)`). The conditions are variables or pure expressions (no calls with effects in GLSL)."""
    def body(b):
        changed = True
        while changed:
            changed = False
            out, pos = [], 0
            for m in re.finditer(r'\bdk_sel\(', b):
                if m.start() < pos:
                    continue
                o = m.end() - 1
                c = matching(b, o, '(', ')')
                if c < 0:
                    continue
                args = _split_args(b[o + 1:c])
                if len(args) != 3:
                    continue
                cond = args[0].strip()
                new = list(args)
                for k, keep in ((1, 1), (2, 2)):
                    inner = _unparen(args[k])
                    if inner.startswith('dk_sel(') and matching(inner, 6, '(', ')') == len(inner) - 1:
                        ia = _split_args(inner[7:-1])
                        if len(ia) == 3 and ia[0].strip() == cond:
                            new[k] = ' (' + ia[keep].strip() + ')'
                if new != args:
                    out.append(b[pos:m.start()])
                    out.append('dk_sel(' + ','.join(new) + ')')
                    pos = c + 1
                    changed = True
            if changed:
                out.append(b[pos:])
                b = ''.join(out)
        return b
    return map_main(text, body)


# ---------------------------------------------------------------------------------------------------------------------
# pass: txq — a texture's level-0 size from the texture size buffer instead of a texture query

TXQ_RE = re.compile(r'textureSize\((dk_tex_b(\d+)), 0\)')
TXQ_TYPES = {'sampler2D': ('ivec2', 'xy'), 'samplerCube': ('ivec2', 'xy'), 'sampler2DArray': ('ivec3', 'xyz'),
             'sampler3D': ('ivec3', 'xyz')}
TXQ_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_908\n#define SPIRV_CROSS_CONSTANT_ID_908 1u\n#endif\n'
            'const uint g_banjoTxq = SPIRV_CROSS_CONSTANT_ID_908;\n'
            'layout(binding = 3, std140) uniform type_DkTexSize\n{\n    uvec4 v[64];\n} g_UboTexSize;\n'
            'bool g_dkTxqBad = false;\n'
            'ivec2 dk_txq_check(ivec2 c, ivec2 q) { g_dkTxqBad = g_dkTxqBad || any(notEqual(c, q)); return q; }\n'
            'ivec3 dk_txq_check(ivec3 c, ivec3 q) { g_dkTxqBad = g_dkTxqBad || any(notEqual(c, q)); return q; }\n')


def pass_txq(text):
    """A texture query of level 0 (`textureSize(t, 0)`: Renut's half-texel offsets, the stacked textures' layer count)
    is a texture instruction per pixel, as costly as a fetch in the texture unit: A5E2 (the second heaviest pixel shader
    of the console's profile) runs 4 fetches and 4 queries a pixel over the whole screen. The size of each queried
    binding comes from the texture size buffer instead (UBO binding 3, entry k for binding k), which plume deko3d fills
    from the view it wrote into the descriptor (Deko3DDescriptorSet::imageSizes) and updates when it changes; the
    banjo-dk line gets txq=<binding mask> and UBO bit 3. Specialization constant 908: 1 (default) the buffer, 0 the
    query as before (banjo_deko3d_txq_constants off), 2 both, the query's value used and the pixel magenta where the two
    differ (banjo_deko3d_txq_verify: the GPU checks plume's sizes). Pixel shaders only."""
    if 'textureSize(' not in text or 'g_banjoTxq' in text:
        return text
    types = {m.group(2): m.group(1) for m in re.finditer(r'uniform (\w+) dk_tex_b(\d+);', text)}
    mask = 0

    def repl(m):
        nonlocal mask
        b = m.group(2)
        t = TXQ_TYPES.get(types.get(b, ''))
        if not t or int(b) >= 32:
            return m.group(0)
        mask |= 1 << int(b)
        vec, comps = t
        c = f'{vec}(g_UboTexSize.v[{b}u].{comps})'
        q = m.group(0)
        return f'((g_banjoTxq == 1u) ? {c} : ((g_banjoTxq == 2u) ? dk_txq_check({c}, {q}) : {q}))'

    span = main_span(text)
    if not span:
        return text
    a, b = span
    body = TXQ_RE.sub(repl, text[a:b])
    if not mask:
        return text
    # The verify variant: magenta where a size differed (before main's closing brace).
    if 'out_var_SV_Target0' in text:
        body = body.rstrip() + '\n    if (g_banjoTxq == 2u && g_dkTxqBad)\n    {\n' \
               '        out_var_SV_Target0 = vec4(1.0, 0.0, 1.0, 1.0);\n    }\n'
    text = text[:a] + body + text[b:]
    text = insert_before_main(text, TXQ_DECL)

    def header(m):
        line = m.group(0)
        ubo = re.search(r'ubo=0x([0-9a-fA-F]+)', line)
        if ubo:
            line = line.replace(ubo.group(0), 'ubo=0x%x' % (int(ubo.group(1), 16) | 8))
        return line + ' txq=0x%x' % mask
    return re.sub(r'^// banjo-dk [^\n]*', header, text, count=1, flags=re.M)


# ---------------------------------------------------------------------------------------------------------------------
# pass: earlyz — depth and stencil tested before the shader where the draw writes neither

EARLYZ_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_909\n#define SPIRV_CROSS_CONSTANT_ID_909 0u\n#define DK_EARLY_Z 0\n'
               '#else\n#define DK_EARLY_Z 1\n#endif\n#if DK_EARLY_Z\nlayout(early_fragment_tests) in;\n#endif\n')


def pass_earlyz(text):
    """A pixel shader that discards (the alpha test, Renut's early alpha test of a quad) has its SPH's kills_pixels
    bit, and the GPU may then test depth after it: every covered pixel runs the shader, also those the depth test
    rejects. Where the draw writes neither depth nor stencil (blended draws: the fur shells, particles), testing first
    gives the same pixels: the discard has nothing to undo, and nothing counts samples (the renderer has no occlusion
    queries). Specialization constant 909 = 1 (render/pipeline.cpp, banjo_deko3d_early_z: depth writes and stencil off
    in the key) declares `layout(early_fragment_tests)`; its default 0 leaves the shader as it was. The constant is
    tested by the preprocessor: plume writes `#define SPIRV_CROSS_CONSTANT_ID_909 1u` ahead of the text."""
    if 'discard' not in text or 'gl_FragDepth' in text or 'SPIRV_CROSS_CONSTANT_ID_909' in text:
        return text
    return insert_globals(text, EARLYZ_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: gpr32 — the listed shaders get constant 910: compiled with 32 GPRs at most when it is 1

GPR32_LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gpr32_shaders.txt')
GPR32_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_910\n#define SPIRV_CROSS_CONSTANT_ID_910 0u\n#endif\n'
              'const uint g_banjoGpr32 = SPIRV_CROSS_CONSTANT_ID_910;\n')
_GPR32 = None


def pass_gpr32(text, name=None):
    """On Maxwell a warp's registers come in blocks of 8 per thread: 33-40 GPRs leave about 48 of the SM's 64 warp
    slots, 32 all of them, to hide texture latency with. The pixel shaders tools/gpr32_shaders.txt lists (33-40 GPRs:
    the fur family, the top of the console's GPU profile) declare constant 910; with 910 = 1 (banjo_deko3d_gpr32,
    off by default: whether the spills cost less than the occupancy gives is for the console to measure) uam
    compiles them with a 32-GPR register file (compiler_iface.cpp CompileBest: the define's text), spilling the rest
    (the fur shader: 15 loads, 5 stores). The constant changes no code by itself."""
    global _GPR32
    if _GPR32 is None:
        _GPR32 = set()
        if os.path.isfile(GPR32_LIST):
            for line in open(GPR32_LIST, encoding='utf-8'):
                line = line.strip()
                if line and not line.startswith('#'):
                    _GPR32.add(line.upper())
    if name is None or name.upper() not in _GPR32 or 'SPIRV_CROSS_CONSTANT_ID_910' in text:
        return text
    return insert_before_main(text, GPR32_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: bonesel — a predicated bone's data select under its own accumulate select (2026-10-20)

BONESEL_NAMED_RE = re.compile(r'mix\((_\d+), \((_\d+) \* ([\w.]+)\) \+ \1, (_\d+)\)')


def pass_bonesel(text):
    """Skinning with predicated bones (pred-select): bone k's rows are `D = mix(P, L, c)` (the fetch L when the bone's
    weight is not zero, else the register's old value P) and its sum `mix(A, (D * w) + A, c)`. Under the same c the
    sum keeps A whenever D would be P, so the sum reads L itself: `mix(A, (L * w) + A, c)` (the same multiply-add of
    the same values when c holds). D stays defined for any other reader; with none left (the next bone's data select
    goes the same way) the compiler drops it: a select per component less (36 a vertex for 3 predicated bones). Both
    forms: D a variable (`vec4 D = mix(P, L, c);`, then `vec4 D_bl = L;` ahead of it) and the data select written
    inside the sum."""
    span = main_span(text)
    if not span:
        return text
    a, b = span
    body = text[a:b]
    count = 0
    # Named: D defined by a select on the sum's own condition.
    defs = {}
    for m in re.finditer(r'^([ \t]*)vec4 (_\d+) = mix\((.*)\);$', body, re.M):
        args = _split_args(m.group(3))
        if len(args) == 3 and re.fullmatch(r'_\d+', args[2].strip()) and re.fullmatch(r'_\d+', args[0].strip()):
            defs[m.group(2)] = (m, args[1].strip(), args[2].strip())
    renamed = {}

    def named(mm):
        nonlocal count
        acc, d, w, c = mm.groups()
        info = defs.get(d)
        if not info or info[2] != c:
            return mm.group(0)
        renamed[d] = info
        count += 1
        return f'mix({acc}, ({d}_bl * {w}) + {acc}, {c})'
    body = BONESEL_NAMED_RE.sub(named, body)
    for d, (m, load, c) in renamed.items():
        line = re.compile(r'^([ \t]*)vec4 ' + d + r' = mix\(', re.M).search(body)
        if not line:
            return text
        body = body[:line.start()] + f'{line.group(1)}vec4 {d}_bl = {load};\n' + body[line.start():]
        # the data select itself reads D_bl (one load, as before)
        body = body.replace(f'vec4 {d} = mix({m.group(3)});', f'vec4 {d} = mix({_split_args(m.group(3))[0].strip()}, '
                            f'{d}_bl, {c});', 1)
    # Inline: mix(A, (mix(P, L, c) * w) + A, c).
    out, pos = [], 0
    for mm in re.finditer(r'mix\((_\d+), \(mix\(', body):
        if mm.start() < pos:
            continue
        acc = mm.group(1)
        inner_open = mm.end() - 1
        inner_close = matching(body, inner_open, '(', ')')
        if inner_close < 0:
            continue
        iargs = _split_args(body[inner_open + 1:inner_close])
        if len(iargs) != 3:
            continue
        c = iargs[2].strip()
        tail = re.match(r' \* ([\w.]+)\) \+ ' + re.escape(acc) + r', ' + re.escape(c) + r'\)', body[inner_close + 1:])
        if not re.fullmatch(r'_\d+', c) or not tail:
            continue
        out.append(body[pos:mm.start()])
        out.append(f'mix({acc}, ({iargs[1].strip()} * {tail.group(1)}) + {acc}, {c})')
        pos = inner_close + 1 + tail.end()
        count += 1
    if pos:
        out.append(body[pos:])
        body = ''.join(out)
    if not count:
        return text
    return text[:a] + body + text[b:]


# ---------------------------------------------------------------------------------------------------------------------
# pass: deadcomp — a constructor's component overwritten before anything reads it (2026-10-20)

_DC_CTOR = re.compile(r'^([ \t]*)vec([234]) (_\d+) = vec\2\((.*)\);$')


def pass_deadcomp(text):
    """`vec4 x = vec4(a, b, c, d);` then, in the same block before any read of x, `x.w = e;`: the constructor's d is
    never read, so it is 0.0 (a value the next light's vector carries from the light before goes, which guardsel then
    sees). Only plain single-component writes at the same level whose right side does not mention x count; any other
    occurrence of x, a deeper or shallower line, ends the search."""
    span = main_span(text)
    if not span:
        return text
    a, b = span
    lines = text[a:b].split('\n')
    count = 0
    for i, ln in enumerate(lines):
        m = _DC_CTOR.match(ln)
        if not m:
            continue
        ind, n, x = m.group(1), int(m.group(2)), m.group(3)
        args = _split_args(m.group(4))
        if len(args) != n:
            continue
        name_re = re.compile(r'(?<![\w.])' + re.escape(x) + r'\b')
        dead = set()
        for j in range(i + 1, len(lines)):
            lj = lines[j]
            if not lj.strip():
                continue
            if not lj.startswith(ind) or lj[len(ind):len(ind) + 1] in (' ', '\t') or lj.strip() in ('{', '}'):
                break
            w = re.match(r'^[ \t]*' + re.escape(x) + r'\.([xyzw]) = (.*);$', lj)
            if w and not name_re.search(w.group(2)):
                c = 'xyzw'.index(w.group(1))
                if c < n:
                    dead.add(c)
                continue
            if name_re.search(lj):
                break
        dead = {c for c in dead if args[c].strip() != '0.0'}
        if not dead:
            continue
        for c in dead:
            args[c] = ' 0.0' if c else '0.0'
        lines[i] = f'{ind}vec{n} {x} = vec{n}({",".join(args)});'
        count += len(dead)
    if not count:
        return text
    return text[:a] + '\n'.join(lines) + text[b:]


# ---------------------------------------------------------------------------------------------------------------------
# pass: guardsel — a select whose other arm nobody reads (2026-10-20)

_GS_VAR = re.compile(r'(?<![\w.])(_\d+)\b')
_GS_TYPES = r'(?:[biu]?vec[234]|float|int|uint|bool)'
_GS_ASSIGN = re.compile(r'^([ \t]*)(?:' + _GS_TYPES + r' )?(_\d+)((?:\.[xyzw]+)?) = (.*);$')
_GS_DECL = re.compile(r'^\s*' + _GS_TYPES + r' _\d+;$')


def _gs_selects(expr, booleans):
    """(condition, true arm span, false arm span, call span) of each select in expr: dk_sel(c, a, b), and mix(b, a, s)
    where s is bvecN(x) or a bool / bvec variable (a mix by a float is a blend, never taken)."""
    out = []
    for m in re.finditer(r'(?<![\w.])(dk_sel|mix)\(', expr):
        op = m.end() - 1
        cl = matching(expr, op, '(', ')')
        if cl < 0:
            continue
        args = _split_args(expr[op + 1:cl])
        if len(args) != 3:
            continue
        spans, pos = [], op + 1
        for x in args:
            spans.append((pos, pos + len(x)))
            pos += len(x) + 1
        if m.group(1) == 'dk_sel':
            out.append((args[0], spans[1], spans[2], (m.start(), cl + 1)))
            continue
        sel = args[2].strip()
        mm = re.fullmatch(r'bvec[234]\((_\d+|true|false)\)', sel)
        if mm and (mm.group(1) in booleans or mm.group(1) in ('true', 'false')):
            out.append((mm.group(1), spans[1], spans[0], (m.start(), cl + 1)))
        elif sel in booleans:
            out.append((sel, spans[1], spans[0], (m.start(), cl + 1)))
    return out


def _gs_analyse(text):
    """pass_guardsel's analysis (also pass_lightbranch's): None, or the main body's lines and what is known
    about them (see pass_guardsel)."""
    span = main_span(text)
    if not span:
        return None
    a, b = span
    body = text[a:b]
    if re.search(r'\b(for|while|do)\b', body):
        return None
    lines = body.split('\n')
    booleans = set(re.findall(r'\b(?:bool|bvec[234]) (_\d+)\b', text))
    assigns = {}
    for i, ln in enumerate(lines):
        m = _GS_ASSIGN.match(ln)
        if m:
            assigns.setdefault(m.group(2), []).append(i)
    alias = {}
    for v, idx in assigns.items():
        if len(idx) == 1:
            m = (re.match(r'^\s*bvec[234] ' + v + r' = bvec[234]\((_\d+|true|false)\);$', lines[idx[0]]) or
                 re.match(r'^\s*bool ' + v + r' = (_\d+)(?: && \1)?;$', lines[idx[0]]))
            if m:
                alias[v] = m.group(1)

    def norm(c):
        c = _unparen(c)
        while c in alias:
            c = alias[c]
        return c

    def block_close(k):  # the line of the } closing the { at line k
        d = 0
        for e in range(k, len(lines)):
            s2 = lines[e].strip()
            if s2 == '{':
                d += 1
            elif s2 == '}':
                d -= 1
                if d == 0:
                    return e
        return -1

    def block_of(i):  # (open, close) of the innermost block holding line i
        depth = 0
        for k in range(i - 1, -1, -1):
            s2 = lines[k].strip()
            if s2 == '}':
                depth += 1
            elif s2 == '{':
                if depth == 0:
                    return k, block_close(k)
                depth -= 1
        return None

    def exclusive(i, j):  # line i in an if's then-block, line j in its else-block
        blk = block_of(i)
        if not blk or blk[0] < 1 or not lines[blk[0] - 1].strip().startswith('if ('):
            return False
        k = blk[1] + 1
        if k + 1 >= len(lines) or lines[k].strip() != 'else' or lines[k + 1].strip() != '{':
            return False
        return k + 1 < j < block_close(k + 1)

    single_cache = {}

    def single(c):
        if re.fullmatch(r'_\d+', c) is None:
            return False
        if c not in single_cache:
            idx = assigns.get(c, [])
            single_cache[c] = len(idx) == 1 or (len(idx) == 2 and exclusive(idx[0], idx[1]))
        return single_cache[c]

    # a float that is a condition's 0 / 1 (the bool01 pass's float(c) and float(!c))
    as_float = {}
    for v, idx in assigns.items():
        if len(idx) == 1:
            m = re.match(r'^\s*float ' + v + r' = float\((!?)\(?(_\d+)\)?\);$', lines[idx[0]])
            if m:
                as_float[v] = (m.group(2), not m.group(1))

    def if_guard(cond_line):  # the guard of an `if (c)` / `if (!c)` / `if (!(c))` / `if (f == 0.0)` line's block
        m = re.fullmatch(r'\s*if \((!?)\(?(_\d+)\)?\)', cond_line)
        if m:
            c = norm(m.group(2))
            return (c, not m.group(1)) if single(c) else None
        m = re.fullmatch(r'\s*if \((_\d+) == ([01])\.0\)', cond_line)
        if m and m.group(1) in as_float:
            c, pol = as_float[m.group(1)]
            c = norm(c)
            return (c, pol == (m.group(2) == '1')) if single(c) else None
        return None

    # the if / else blocks around each line (guards of the block conditions)
    block_guards = []
    stack, last_closed = [], None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if s == '{':
            prev = lines[i - 1].strip() if i else ''
            if prev.startswith('if ('):
                stack.append(if_guard(lines[i - 1]))
            elif prev == 'else' and last_closed is not None:
                stack.append((last_closed[0], not last_closed[1]))
            else:
                stack.append(None)
            last_closed = None
        elif s == '}':
            last_closed = stack.pop() if stack else None
        elif s != 'else':
            last_closed = None
        block_guards.append(frozenset(g for g in stack if g is not None))

    # implications between conditions: (x, true) => (a, true) for x = a && b; (x, p) => (a, !p) for x = !a; a
    # condition assigned once in each arm of an if / else implies what both arms imply (an arm assigning false, or
    # true for the false side, implies everything: it never gives that value)
    imp = {}

    def add_imp(k, g):
        imp.setdefault(k, set()).add(g)
    every = object()
    two_arm = {}
    for v, idx in assigns.items():
        if not single(v):
            continue
        if len(idx) == 1:
            m = re.match(r'^\s*bool ' + v + r' = (.+);$', lines[idx[0]])
            if not m:
                continue
            e = _unparen(m.group(1))
            parts = [_unparen(x) for x in e.split(' && ')] if ' && ' in e and ' || ' not in e else None
            if parts and all(re.fullmatch(r'_\d+', x) for x in parts):
                for x in parts:
                    x = norm(x)
                    if single(x):
                        add_imp((v, True), (x, True))
                continue
            parts = [_unparen(x) for x in e.split(' || ')] if ' || ' in e and ' && ' not in e else None
            if parts and all(re.fullmatch(r'_\d+', x) for x in parts):
                for x in parts:
                    x = norm(x)
                    if single(x):
                        add_imp((v, False), (x, False))
                continue
            m2 = re.fullmatch(r'!\(?(_\d+)\)?', e)
            if m2 and single(norm(m2.group(1))):
                x = norm(m2.group(1))
                add_imp((v, True), (x, False))
                add_imp((v, False), (x, True))
        else:
            arms = []
            for i in idx:
                m = re.match(r'^\s*' + v + r' = (.+);$', lines[i])
                arms.append((_unparen(m.group(1)) if m else None, i))
            if all(e is not None for e, _ in arms):
                two_arm[v] = arms

    def closure_now(gs):
        out, todo = set(gs), list(gs)
        while todo:
            g = todo.pop()
            for h in imp.get(g, ()):
                if h not in out:
                    out.add(h)
                    todo.append(h)
        return out

    # the two-arm rule needs what the arms imply: to a fixpoint
    grown = True
    while grown:
        grown = False
        for v, arms in two_arm.items():
            for pol in (True, False):
                sets = []
                for e, i in arms:
                    if e == ('false' if pol else 'true'):
                        sets.append(every)
                        continue
                    s = set()
                    if re.fullmatch(r'_\d+', e) and single(norm(e)):
                        s |= closure_now({(norm(e), pol)})
                    blk = block_of(i)
                    if blk and blk[0] >= 1:
                        s |= closure_now(block_guards[blk[0]] - block_guards[blk[0] - 1])
                    sets.append(s)
                real = [s for s in sets if s is not every]
                common = set.intersection(*real) if real else set()
                for g in common:
                    if g not in imp.get((v, pol), ()):
                        add_imp((v, pol), g)
                        grown = True

    closure_cache = {}

    def closure(gs):
        if gs in closure_cache:
            return closure_cache[gs]
        out, todo = set(gs), list(gs)
        while todo:
            g = todo.pop()
            for h in imp.get(g, ()):
                if h not in out:
                    out.add(h)
                    todo.append(h)
        out = frozenset(out)
        closure_cache[gs] = out
        return out

    uses = {}  # variable -> [(guards at the read, the variable the read's statement assigns or None)]
    parsed = []
    for i, ln in enumerate(lines):
        if _GS_DECL.match(ln):
            parsed.append(None)
            continue
        m = _GS_ASSIGN.match(ln)
        if m:
            target, start = m.group(2), len(ln) - len(m.group(4)) - 1
        else:
            target, start = None, 0
        expr = ln[start:-1] if m else ln
        sels = _gs_selects(expr, booleans)
        parsed.append((target, start) if m else None)
        for vm in _GS_VAR.finditer(expr):
            p = vm.start()
            g = set(block_guards[i])
            dead = False
            for cond, ts, fs, _ in sels:
                c = norm(cond)
                if c in ('true', 'false'):  # the arm a literal never takes reads nothing
                    dead = dead or (fs if c == 'true' else ts)[0] <= p < (fs if c == 'true' else ts)[1]
                    continue
                if not single(c):
                    continue
                if ts[0] <= p < ts[1]:
                    g.add((c, True))
                elif fs[0] <= p < fs[1]:
                    g.add((c, False))
            if not dead:
                uses.setdefault(vm.group(1), []).append((frozenset(g), target))
    top = None
    guard = {v: top for v in assigns}
    changed = True
    while changed:
        changed = False
        for v in assigns:
            acc = top
            for g, target in uses.get(v, []):
                if target is not None and target in assigns:
                    tg = guard[target]
                    s = top if tg is top else closure(g | tg)
                else:
                    s = closure(g)
                if s is top or any((c, not p) in s for c, p in s):  # never read, or read under a contradiction
                    continue
                acc = s if acc is top else (acc & s)
            if acc != guard[v]:
                guard[v] = acc
                changed = True
    return types.SimpleNamespace(a=a, b=b, lines=lines, parsed=parsed, guard=guard, uses=uses, imp=imp,
                                 closure=closure, norm=norm, single=single, booleans=booleans,
                                 assigns=assigns, block_guards=block_guards)


def pass_guardsel(text):
    """A value read only where a condition c holds (inside the chosen arm of selects on c, directly or through the
    values it feeds) never shows what a select on c inside its own definition gives when c does not hold: that select
    is its chosen arm. Renut's lights select every term on the light's flag (`dk_sel(f, clamp(...), old)`) and then
    add it only under the same flag (`mix(acc, (t * k) + acc, bvec3(f))`): the selects go (20 in the fur shader).
    A greatest fixpoint over every variable's readers; the conditions are variables with one value a run: assigned
    once, or once in each arm of one if / else. Not in a body with loops (a condition could change between the reads).
    Exact: a value changes only in runs where nothing reads it."""
    an = _gs_analyse(text)
    if an is None:
        return text
    a, b, lines, parsed, guard, norm, booleans = an.a, an.b, an.lines, an.parsed, an.guard, an.norm, an.booleans
    count = 0
    for i, info in enumerate(parsed):
        if not info:
            continue
        target, start = info
        g = guard.get(target)
        if not g:
            continue
        expr = lines[i][start:-1]
        while True:
            for cond, ts, fs, (cs, ce) in _gs_selects(expr, booleans):
                c = norm(cond)
                if (c, True) in g:
                    keep = expr[ts[0]:ts[1]]
                elif (c, False) in g:
                    keep = expr[fs[0]:fs[1]]
                else:
                    continue
                expr = expr[:cs] + '(' + keep.strip() + ')' + expr[ce:]
                count += 1
                break
            else:
                break
        lines[i] = lines[i][:start] + expr + ';'
    if not count:
        return text
    return text[:a] + '\n'.join(lines) + text[b:]


# ---------------------------------------------------------------------------------------------------------------------
# pass: lightbranch — a light's work skipped where the light does not reach (2026-10-20, constant 911, A/B)

LIGHTBRANCH_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_911\n#define SPIRV_CROSS_CONSTANT_ID_911 0u\n#endif\n'
                    'const uint g_banjoLightBranch = SPIRV_CROSS_CONSTANT_ID_911;\n')
_LB_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lightbranch_shaders.txt')
_LB_FORBIDDEN = re.compile(r'\b(texture\w*|texelFetch\w*|dFd\w+|fwidth\w*|discard|out_var_\w+|gl_FragDepth|return|'
                           r'imageStore|atomic\w*)\b')


def _lb_list():
    try:
        return {ln.split('#')[0].strip().upper() for ln in open(_LB_LIST_PATH, encoding='utf-8')
                if ln.split('#')[0].strip()}
    except OSError:
        return set()


_LIGHTBRANCH = _lb_list()


def pass_lightbranch(text, name=None):
    """After guardsel and deadcomp a light's work is read only where its attenuation test c holds, except the sums it
    adds to (`acc2 = mix(acc, x + acc, bvecN(f))` with f => c, which keep acc where c does not hold). The statements
    after `bool c = ...;` up to the last one whose results are read later go into `if (g_banjoLightBranch == 0u || c)`
    with an else that gives each sum its previous value (the declarations of what is read later move ahead of it).
    Specialization constant 911 (0 by default: the if folds away, the code as before; banjo_deko3d_light_branch gives
    1): warps whose pixels are all out of a light's reach skip it. Exact: with c true the same statements run; with c
    false nothing reads their results but the sums, which get the value their select gives. Not over texture fetches,
    derivatives, discards or outputs. Only the shaders of tools/lightbranch_shaders.txt (an A/B for the console)."""
    if name is None or name.upper() not in _LIGHTBRANCH or 'SPIRV_CROSS_CONSTANT_ID_911' in text:
        return text
    an = _gs_analyse(text)
    if an is None:
        return text
    lines, guard, closure, norm, single = an.lines, an.guard, an.closure, an.norm, an.single
    ind_of = [len(ln) - len(ln.lstrip(' \t')) for ln in lines]
    # every read of each variable: (line, assigning target or None)
    reads = {}
    for i, ln in enumerate(lines):
        m = _GS_ASSIGN.match(ln)
        expr = ln[len(ln) - len(m.group(4)) - 1:-1] if m else ln
        if _GS_DECL.match(ln):
            continue
        for vm in _GS_VAR.finditer(expr):
            reads.setdefault(vm.group(1), []).append(i)
    edits = []  # (start, end, cond, hoisted [(type, var)], else [(var, value)])
    taken_until = -1
    for i, ln in enumerate(lines):
        m = re.match(r'^([ \t]*)bool (_\d+) = (.+);$', ln)
        if not m or i <= taken_until or not single(m.group(2)):
            continue
        ind, c = m.group(1), m.group(2)
        # the block's statements after the condition: whole lines at this level (nested blocks taken whole)
        ends = []
        j = i + 1
        while j < len(lines):
            s = lines[j].strip()
            if not s:
                j += 1
                continue
            if ind_of[j] < len(ind) or s == '}' and ind_of[j] == len(ind):
                break
            if ind_of[j] > len(ind):
                break  # not expected right after a statement at this level
            k = j
            if s.startswith('if (') or s == 'else':
                # take the if / else blocks whole
                k = j + 1
                while k < len(lines):
                    if lines[k].strip() == '{' and ind_of[k] == len(ind):
                        d = 0
                        for e in range(k, len(lines)):
                            if lines[e].strip() == '{':
                                d += 1
                            elif lines[e].strip() == '}':
                                d -= 1
                                if d == 0:
                                    k = e
                                    break
                        if k + 1 < len(lines) and lines[k + 1].strip() == 'else':
                            k += 2
                            continue
                        break
                    k += 1
            ends.append(k)
            j = k + 1
        best = None
        for e in ends:
            region = range(i + 1, e + 1)
            if any(_LB_FORBIDDEN.search(lines[r]) for r in region):
                break
            ok, hoist, elses = True, [], []
            assigned = {}
            for r in region:
                mm = _GS_ASSIGN.match(lines[r])
                dm = re.match(r'^\s*(' + _GS_TYPES + r') (_\d+);$', lines[r])
                if mm:
                    assigned.setdefault(mm.group(2), r)
                elif dm:
                    assigned.setdefault(dm.group(2), r)
            for v, first in assigned.items():
                later = [r for r in reads.get(v, []) if r > e]
                if not later:
                    continue
                g = guard.get(v)
                if g is not None and (c, True) in closure(g):
                    tm = re.match(r'^\s*(' + _GS_TYPES + r') ' + v + r'\b', lines[first])
                    hoist.append((tm.group(1) if tm else None, v))
                    continue
                # a sum: `T v = mix(P, X, bvecN(f));` assigned once in the region, at its level, with f => c
                defs = [r for r in region if (_GS_ASSIGN.match(lines[r]) or [None, None, None])[2] == v]
                sm = re.match(r'^[ \t]*(vec[234]|float) ' + v + r' = mix\((.*)\);$', lines[defs[0]]) if len(
                    defs) == 1 and ind_of[defs[0]] == len(ind) else None
                if sm:
                    args = _split_args(sm.group(2))
                    sel = args[2].strip() if len(args) == 3 else ''
                    fm = re.fullmatch(r'bvec[234]\((_\d+)\)', sel)
                    f = norm(fm.group(1) if fm else sel)
                    prev = args[0].strip()
                    # a region-local copy of an outside value (`vec3 p = acc.xyz;`, assigned once): its expression
                    pm = re.fullmatch(r'(_\d+)', prev)
                    if pm and pm.group(1) in assigned:
                        pdefs = [r for r in region if (_GS_ASSIGN.match(lines[r]) or [None, None, None])[2] ==
                                 pm.group(1)]
                        pd = _GS_ASSIGN.match(lines[pdefs[0]]) if len(pdefs) == 1 else None
                        if pd and not pd.group(3):
                            prev = '(' + pd.group(4).strip() + ')'
                    prev_vars = _GS_VAR.findall(prev)
                    if (re.fullmatch(r'_\d+', f) and single(f) and (c, True) in closure(frozenset([(f, True)])) and
                            all(pv not in assigned for pv in prev_vars)):
                        hoist.append((sm.group(1), v))
                        elses.append((v, prev))
                        continue
                ok = False
                break
            if not ok:
                continue
            if any(t is None for t, _ in hoist):
                continue
            n_stmt = sum(1 for r in region if _GS_ASSIGN.match(lines[r]))
            if n_stmt >= 8:
                best = (i + 1, e, c, hoist, elses)
        if best:
            edits.append((best[0], best[1], best[2], best[3], best[4], ind))
            taken_until = best[1]
    if not edits:
        return text
    out = list(lines)
    for start, end, c, hoist, elses, ind in reversed(edits):
        names = {v for _, v in hoist}
        body = []
        for r in range(start, end + 1):
            ln = out[r]
            mm = re.match(r'^([ \t]*)(' + _GS_TYPES + r') (_\d+)( = .*;|;)$', ln)
            if mm and mm.group(3) in names:
                if mm.group(4) == ';':
                    continue  # declared ahead
                ln = f'{mm.group(1)}{mm.group(3)}{mm.group(4)}'
            body.append('    ' + ln if ln.strip() else ln)
        block = [f'{ind}{t} {v};' for t, v in hoist]
        block.append(f'{ind}if (g_banjoLightBranch == 0u || {c})')
        block.append(f'{ind}{{')
        block.extend(body)
        block.append(f'{ind}}}')
        if elses:
            block.append(f'{ind}else')
            block.append(f'{ind}{{')
            block.extend(f'{ind}    {v} = {p};' for v, p in elses)
            block.append(f'{ind}}}')
        out[start:end + 1] = block
    text = text[:an.a] + '\n'.join(out) + text[an.b:]
    return insert_before_main(text, LIGHTBRANCH_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: nullps — a diagnostic: the listed pixel shaders discard at once (2026-10-20, constant 912)

NULLPS_DECL = ('#ifndef SPIRV_CROSS_CONSTANT_ID_912\n#define SPIRV_CROSS_CONSTANT_ID_912 0u\n#endif\n'
               'const uint g_banjoNullPs = SPIRV_CROSS_CONSTANT_ID_912;\n')
_NULLPS_LIST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nullps_shaders.txt')


def _nullps_list():
    try:
        return {ln.split('#')[0].strip().upper() for ln in open(_NULLPS_LIST_PATH, encoding='utf-8')
                if ln.split('#')[0].strip()}
    except OSError:
        return set()


_NULLPS = _nullps_list()


def pass_nullps(text, name=None):
    """Diagnostic (banjo_renut_diag_null_ps, constant 912 = 1): the listed pixel shaders (tools/nullps_shaders.txt, the
    fur family, which discard already) discard as their first statement, so the console's per-shader GPU profile shows
    what their draws cost without the pixel work (vertex shading, rasterisation, the pixel launches): the fur's vertex
    share. 0 (default): the branch folds away, the shader as before. Not a picture to keep (the fur vanishes)."""
    if name is None or name.upper() not in _NULLPS or 'SPIRV_CROSS_CONSTANT_ID_912' in text or 'discard' not in text:
        return text
    m = re.search(r'\nvoid main\(\)\n\{\n', text)
    if not m:
        return text
    text = text[:m.end()] + '    if (g_banjoNullPs != 0u)\n    {\n        discard;\n    }\n' + text[m.end():]
    return insert_before_main(text, NULLPS_DECL)


# ---------------------------------------------------------------------------------------------------------------------
# pass: aw4 — a bone row's four address-window words as one 16-byte load (2026-10-20)

AW4_DECL = ('layout(binding = 15, std430) readonly buffer type_ByteAddressBuffer4\n{\n    uvec4 _m0[];\n} '
            'g_DkAddressWindow4;\n')
AW4_INDEX_RE = re.compile(r'\(g_UboShared\.v\[35u\]\.x \+ (?:min\(\((_\d+) \+ \(uint\((_\d+)(?: \+ (\d+))?\) \* 4u\)\), '
                          r'(_\d+)\)|\((_\d+) \+ \(uint\((_\d+)(?: \+ (\d+))?\) \* 4u\)\))\) >> 2u')


def pass_aw4(text):
    """The vertex fetch from memory (g_DkAddressWindow, shader_common.h vfetchMemory) reads a bone row as four words
    W, W + 1, W + 2, W + 3 (four 32-bit loads and the index adds). Where W is provably a multiple of 4 the row is one
    uvec4 load of the same binding (a second view, g_DkAddressWindow4) at W >> 2. W = (window offset + slot base +
    4 * (I + k)) >> 2, and each part is a multiple of 16 bytes: the window offset (SharedConstants.vertex_fetch_data's
    low byte) and the slot base (vertex_fetch_slots, minus the endian bits) because render/draw.cpp's fetch copies
    start at 16-byte multiples of 256-byte aligned allocations; 4 * (I + k) because I is `int(floor(x)) * K` with K a
    multiple of 4 and k one too; the clamp min(..., C) because C is vertex_fetch_bytes - 16 (a multiple of 16). Only
    rows whose four loads and index definitions are all in one block at the same level (the loads run there anyway)."""
    span = main_span(text)
    if not span or 'g_DkAddressWindow._m0[' not in text or 'g_DkAddressWindow4' in text:
        return text
    a, b = span
    body = text[a:b]
    lines = body.split('\n')
    defs = {}  # name -> (line index, indentation, expression)
    for i, ln in enumerate(lines):
        m = re.match(r'^([ \t]*)(?:uint|int) (_\d+) = (.+);$', ln)
        if m:
            defs[m.group(2)] = (i, m.group(1), m.group(3))

    def mult16_int(name):  # I with 4 * I a multiple of 16
        d = defs.get(name)
        if not d:
            return False
        m = re.fullmatch(r'int\(floor\([^;]*\)\) \* (\d+)', d[2])
        return bool(m) and int(m.group(1)) % 4 == 0

    def slot_base(name):
        d = defs.get(name)
        return bool(d) and re.fullmatch(r'g_UboShared\.v\[\d+u\]\.[xyzw] & 4294967292u', d[2]) is not None

    def clamp_bound(name):
        d = defs.get(name)
        return bool(d) and d[2] == 'max(g_UboShared.v[60u].w, 16u) - 16u'

    load_lines = {}
    for i, ln in enumerate(lines):
        for v in re.findall(r'g_DkAddressWindow\._m0\[(_\d+)\]', ln):
            load_lines.setdefault(v, []).append(i)
    inserts = {}  # line index -> lines to insert after it
    replace = {}  # word index variable -> 'Q.x'
    rows = 0
    for w, (wi, ind, expr) in defs.items():
        m = AW4_INDEX_RE.fullmatch(expr)
        if not m or w not in load_lines:
            continue
        if m.group(1):
            base, i_var, k, bound = m.group(1), m.group(2), m.group(3), m.group(4)
            if not clamp_bound(bound):
                continue
        else:
            base, i_var, k = m.group(5), m.group(6), m.group(7)
        if not slot_base(base) or not mult16_int(i_var) or (k is not None and int(k) % 4):
            continue
        sibs = {}
        for s, (si, sind, sexpr) in defs.items():
            mm = re.fullmatch(re.escape(w) + r' \+ ([123])u', sexpr)
            if mm and sind == ind and s in load_lines:
                sibs[int(mm.group(1))] = (s, si)
        if len(sibs) != 3:
            continue
        group = [(w, wi)] + [sibs[n] for n in (1, 2, 3)]
        last = max(i for _, i in group)
        first_load = min(min(load_lines[v]) for v, _ in group)
        if first_load <= last:
            continue
        # every load at the definitions' level of the same block: no line between leaves the block or nests deeper
        ok = True
        end = max(max(load_lines[v]) for v, _ in group)
        for j in range(wi, end + 1):
            lj = lines[j]
            if not lj.strip():
                continue
            lind = len(lj) - len(lj.lstrip(' \t'))
            if lind < len(ind) or (lind > len(ind) and any(j in load_lines[v] for v, _ in group)):
                ok = False
                break
        if not ok:
            continue
        q = f'{w}_q'
        # the row index straight from the byte offset (uam keeps `(x >> 2u) >> 2u` as two shifts)
        inserts.setdefault(last, []).append(f'{ind}uvec4 {q} = g_DkAddressWindow4._m0[{expr[:-len(" >> 2u")]} >> 4u];')
        for n, (v, _) in enumerate(group):
            replace[v] = f'{q}.{"xyzw"[n]}'
        rows += 1
    if not rows:
        return text
    out = []
    for i, ln in enumerate(lines):
        if any(v in replace for v in re.findall(r'g_DkAddressWindow\._m0\[(_\d+)\]', ln)):
            ln = re.sub(r'g_DkAddressWindow\._m0\[(_\d+)\]',
                        lambda mm: replace.get(mm.group(1), mm.group(0)), ln)
        out.append(ln)
        out.extend(inserts.get(i, []))
    text = text[:a] + '\n'.join(out) + text[b:]
    decl = re.search(r'\} g_DkAddressWindow;\n', text)
    if not decl:
        return text
    return text[:decl.end()] + '\n' + AW4_DECL + text[decl.end():]


PASSES = [
    ('scale1', pass_scale1),
    ('fragscale', pass_fragscale),
    ('alpha', pass_alpha),
    ('tess', pass_tess),
    ('bonerow', pass_bonerow),
    ('vszero', pass_vszero),
    ('vsout', pass_vsout),
    ('pszero', pass_pszero),
    ('pred', pass_pred),
    ('predmul', pass_predmul),
    ('sext', pass_sext),
    ('bool01', pass_bool01),
    ('sel', pass_sel),
    ('alphamask', pass_alphamask),
    ('earlyhoist', pass_earlyhoist),
    ('ifmerge', pass_ifmerge),
    ('dowhile', pass_dowhile),
    ('selsame', pass_selsame),
    ('deadcomp', pass_deadcomp),
    ('guardsel', pass_guardsel),
    ('lightbranch', pass_lightbranch),
    ('nullps', pass_nullps),
    ('txq', pass_txq),
    ('earlyz', pass_earlyz),
    ('gpr32', pass_gpr32),
    ('bonesel', pass_bonesel),
    ('aw4', pass_aw4),
]
DEFAULT = ','.join(name for name, _ in PASSES)


# The passes for one stage only.
STAGE_OF = {'bonesel': 'vert', 'aw4': 'vert', 'lightbranch': 'frag', 'nullps': 'frag', 'gpr32': 'frag', 'earlyz': 'frag', 'txq': 'frag', 'alpha': 'frag', 'alphamask': 'frag', 'earlyhoist': 'frag', 'tess': 'vert', 'vsout': 'vert', 'vszero': 'vert', 'pszero': 'frag'}


def optimise(text, passes=DEFAULT, stage=None, name=None):
    """stage: 'vert' or 'frag' (from the text when not given: a vertex shader writes gl_Position). name: the shader's
    hash (the gpr32 pass's list)."""
    names = [p for p in passes.split(',') if p] if isinstance(passes, str) else list(passes)
    if stage is None:
        stage = 'vert' if re.search(r'\bgl_Position\b', text) else 'frag'
    for pname, fn in PASSES:
        if pname in names and STAGE_OF.get(pname, stage) == stage:
            text = fn(text, name) if fn in (pass_gpr32, pass_lightbranch, pass_nullps) else fn(text)
    return text


UAM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'out', 'deko3d_tools', 'uam-build',
                   'uam.exe')


def check_one(item):
    """uam compiles the shader's default variant (and with the alpha test's function 7 and 8): '' or the error."""
    import subprocess
    import tempfile
    name, text = item
    stage = 'frag' if name.startswith('ps_') else 'vert'
    variants = [text]
    if 'SPIRV_CROSS_CONSTANT_ID_903' in text:
        nl = text.find('\n') + 1
        variants += [text[:nl] + f'#define SPIRV_CROSS_CONSTANT_ID_903 {f}u\n' + text[nl:] for f in (7, 8)]
    with tempfile.TemporaryDirectory() as tmp:
        for k, v in enumerate(variants):
            src = os.path.join(tmp, f'v{k}.glsl')
            out = os.path.join(tmp, f'v{k}.dksh')
            with open(src, 'w', encoding='utf-8', newline='\n') as f:
                f.write(v)
            r = subprocess.run([UAM, '-s', stage, '-o', out, src], capture_output=True, text=True)
            if r.returncode != 0 or not os.path.isfile(out):
                return name, (r.stdout + r.stderr).strip()[:300]
    return name, ''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='src')
    ap.add_argument('--out')
    ap.add_argument('--passes', default=DEFAULT)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--check', action='store_true',
                    help='compile each output with uam; a shader that fails keeps its text as it came in')
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()
    if a.list:
        for name, fn in PASSES:
            print(f'{name}: {fn.__doc__.split(".")[0].strip()}')
        return 0
    os.makedirs(a.out, exist_ok=True)
    sources = {}
    texts = {}
    for name in sorted(os.listdir(a.src)):
        if not re.fullmatch(r'(vs|ps)_[0-9A-F]{16}\.glsl', name):
            continue
        sources[name] = open(os.path.join(a.src, name), encoding='utf-8').read()
        texts[name] = optimise(sources[name], a.passes, 'frag' if name.startswith('ps_') else 'vert', name[3:19])
    failed = []
    if a.check:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(a.jobs) as ex:
            for name, err in ex.map(check_one, texts.items()):
                if err:
                    failed.append(name)
                    if len(failed) <= 10:
                        print(f'  {name}: uam failed, kept as it was: {err}')
                    texts[name] = sources[name]
    for name, out in texts.items():
        with open(os.path.join(a.out, name), 'w', encoding='utf-8', newline='\n') as f:
            f.write(out)
    print(f'deko3d_glsl_opt: {len(texts)} shaders, passes {a.passes} -> {a.out}' +
          (f' ({len(failed)} kept as they were: uam failed)' if a.check else ''))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
