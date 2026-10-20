#!/usr/bin/env python3
"""banjo-nx (banjo_gpu_early_alpha): the alpha test of the translated pixel shaders, run as early as the alpha allows.

An alpha-tested Xenos pixel shader computes its whole colour before the alpha test at its end throws the pixel away.
When the alpha needs only a few early texture fetches, a pixel that fails could stop there, before the expensive rest
(in Nuts & Bolts' material shaders: three shadow cascades of 8 taps and the light loop, ~30 ms a frame on the
console for that family). For every ps_*.hlsl XenosRecomp wrote, this pass:

  1. parses main() (the translator's regular statement forms; anything else leaves the shader as it is);
  2. slices it backwards from the alpha clips: the statements, and of each only the components, the final oC0.w
     needs (a statement inside an if needs the if's condition; a return ends a path);
  3. takes the earliest top-level point after which that slice has no texture fetch (nor any other call it cannot
     copy), and emits
       bool banjoEarlyAlphaFails(<the registers live there>)   the slice after that point, returning
                                                               banjoAlphaFailsEarly(oC0.w) where main() clips
     and at that point in main(), under SPEC_CONSTANT_EARLY_ALPHA (the renderer sets it with the alpha test:
     banjo_gpu_early_alpha), banjoEarlyAlphaQuad(banjoEarlyAlphaFails(...)): the 2x2 quad is discarded when its four
     invocations all fail, so no pixel that is still shaded loses a neighbour its derivatives need.

Repeated fetches. A texture fetch the alpha needs may be repeated in the copy when, in main(), every if around it tests
uniform values only and no lane can have left before it (no kill, alpha clip or per-pixel return): all four lanes of
every quad run it there and at the early point, so the coordinates, their implicit derivatives and the texels are
the same (and the compiler may merge the two). In Nuts & Bolts' in-place material shaders the alpha texture is
sampled after the shadow cascades.

Runtime zeros. A product `x = k * y` whose factor k holds a pixel shader constant component (or a literal 0 of the
shader's own constants: no guard then) and whose other factor y depends on a texture fetch and is provably finite (comparison results, saturated values, constants guarded to |c| <= 2^40, sums and products of
those) is +-0 when k == 0 at run time. The copy uses 0 there, guarded by `k == 0 && |c| <= 2^40` (tested per draw,
uniform), which drops y's whole slice: Nuts & Bolts' shadow-receiving materials add c12.x times the shadow-map
sum to their alpha, and c12.x is 0 in 92 % of their draws (banjo_diag_ps_alpha). A zeroed value may reach the
alpha only through +, -, *, saturate, abs, min, max, clamp, dot, selections and comparisons, where +0 and -0 agree.

Exactness: the copy is the same operations on the same values, and banjoAlphaFailsEarly fails a pixel only when the
test fails for every alpha within (|alpha| + |ref|) * 2^-12 + 1e-30 of the copy's, which covers a multiply-add the
compiler fuses in one place and not the other and a flushed denormal. Without the bit the shader is what it was.

Self-check: every copy is executed against the original main() on random registers, fetch results, constants and
bool constants (an interpreter for the same statement forms, float32 arithmetic): wherever the guards hold and the
original reaches an alpha clip, the copy must return the bit-identical alpha (or both NaN, or both zero), and
nothing where the original does not clip. A shader whose copy differs once keeps no copy, and the report says so.

usage: early_alpha.py <hlsl folder> [--trials N] [--only NAME]
(rewrites ps_*.hlsl in place; a file an earlier version processed is stripped back to the translator's output first)
"""
import glob
import math
import os
import random
import re
import struct
import sys
import zlib

try:
    import numpy as np
except ImportError:  # the self-check needs numpy; without it no shader gets a copy
    np = None

MARK = 'banjo_gpu_early_alpha'
BIG = 2.0 ** 40          # the guard on constants a bound relies on
FINITE = 2.0 ** 100      # a bound below this is finite in float32
MAX_COPY = 160           # statements in a copy
MIN_FETCHES_SKIPPED = 2  # texture fetches after the early point, at least
MIN_STATEMENTS_SKIPPED = 24

CONST_RE = re.compile(r'\(MASSEFFECT_UBO \? g_UboPixel\.v\[(\d+)\] : vk::RawBufferLoad<float4>\('
                      r'g_PushConstants\.PixelShaderConstants \+ (\d+), 0x10\)\)')


def const_text(n):
    return f'(MASSEFFECT_UBO ? g_UboPixel.v[{n}] : vk::RawBufferLoad<float4>(g_PushConstants.PixelShaderConstants + {n * 16}, 0x10))'


class Unsupported(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Expressions
TOKEN_RE = re.compile(r'\s*(?:(?P<hex>0x[0-9A-Fa-f]+u?)|(?P<num>(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?)|'
                      r'(?P<id>[A-Za-z_]\w*)|(?P<op>&&|\|\||==|!=|<=|>=|[-+*/?:<>!(),.]))')
LEVELS = [('||',), ('&&',), ('==', '!='), ('<', '<=', '>', '>='), ('+', '-'), ('*', '/')]
COMPARISONS = {'==', '!=', '<', '<=', '>', '>='}
FETCHES = {'tfetch1D', 'tfetch2D', 'tfetch3D', 'tfetchCube', 'tfetch2DBicubic', 'getWeights2D'}
COMPONENTWISE = {'saturate', 'abs', 'rsqrt', 'rcp', 'sqrt', 'exp2', 'log2', 'frac', 'floor', 'ceil', 'trunc', 'round',
                 'sign', 'sin', 'cos', 'clamp', 'min', 'max', 'select', 'isnan'}
CONSTRUCTORS = {'float': 1, 'float2': 2, 'float3': 3, 'float4': 4}
# banjo_gpu_clamp_cbuf (package 25): the translator's clamp bounds, the same two values from the shared block.
SCALAR_NAMES = {'FLT_MIN': -3.4028234663852886e38, 'FLT_MAX': 3.4028234663852886e38,
                'BANJO_FLT_LO': -3.4028234663852886e38, 'BANJO_FLT_HI': 3.4028234663852886e38}


def tokenize(s):
    out, pos, s = [], 0, s.rstrip()
    while pos < len(s):
        m = TOKEN_RE.match(s, pos)
        if not m:
            raise Unsupported('token: ' + s[pos:pos + 24])
        pos = m.end()
        out.append((m.lastgroup, m.group(m.lastgroup)))
    return out


class Parser:
    def __init__(self, text):
        self.t, self.i = tokenize(text), 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def at(self, v):
        k, x = self.peek()
        return k == 'op' and x == v

    def take(self, v=None):
        k, x = self.peek()
        if k is None or (v is not None and x != v):
            raise Unsupported(f'expected {v}, found {x}')
        self.i += 1
        return k, x

    def whole(self):
        n = self.expr()
        if self.i != len(self.t):
            raise Unsupported('trailing ' + str(self.peek()))
        return n

    def expr(self):
        c = self.binary(0)
        if self.at('?'):
            self.take('?')
            a = self.expr()
            self.take(':')
            return ('tern', c, a, self.expr())
        return c

    def binary(self, level):
        if level == len(LEVELS):
            return self.unary()
        a = self.binary(level + 1)
        while self.peek()[0] == 'op' and self.peek()[1] in LEVELS[level]:
            op = self.take()[1]
            a = ('bin', op, a, self.binary(level + 1))
        return a

    def unary(self):
        if self.at('-') or self.at('!'):
            op = self.take()[1]
            return ('un', op, self.unary())
        n = self.primary()
        while self.at('.'):
            self.take('.')
            k, v = self.take()
            if k != 'id' or not re.fullmatch('[xyzw]{1,4}', v):
                raise Unsupported('member ' + str(v))
            n = ('swz', n, v)
        return n

    def primary(self):
        k, v = self.peek()
        if k == 'op' and v == '(':
            self.take()
            n = self.expr()
            self.take(')')
            return n
        if k in ('num', 'hex'):
            self.take()
            return ('num', v)
        if k == 'id':
            self.take()
            if self.at('('):
                self.take('(')
                args = []
                if not self.at(')'):
                    args.append(self.expr())
                    while self.at(','):
                        self.take(',')
                        args.append(self.expr())
                self.take(')')
                # The translator's branch-free forms (shader_common.h): the same trees as && / || / ?: before.
                if v in ('and', 'or', 'banjoAnd', 'banjoOr') and len(args) == 2:
                    return ('bin', '&&' if v in ('and', 'banjoAnd') else '||', args[0], args[1])
                if v == 'banjoSel' and len(args) == 3:
                    return ('tern', args[0], args[1], args[2])
                if v == 'banjoSel0' and len(args) == 2:
                    return ('tern', args[0], ('num', '0.0'), args[1])
                return ('call', v, tuple(args))
            m = re.fullmatch(r'__C(\d+)', v)
            if m:
                return ('const', int(m.group(1)))
            if v in ('true', 'false'):
                return ('num', '1.0' if v == 'true' else '0.0')
            return ('var', v)
        raise Unsupported(f'primary {k} {v}')


UNIFORM_VECTORS = {'ColorFactor', 'SolidColor'}  # per-draw constants of some shaders


def is_uniform_name(name):
    """Texture slot macros (descriptor indices, 1/size), cubeMapData (arguments of opaque calls only) and the per-draw
    constant vectors: never register state."""
    return re.match(r's\d+_', name) is not None or name == 'cubeMapData' or name in UNIFORM_VECTORS


def var_width(name):
    if name.endswith('_InvSize') and is_uniform_name(name):
        return 2
    if name in UNIFORM_VECTORS:
        return 4
    if name in ('ps', 'p0', 'iFace') or name in SCALAR_NAMES or name.startswith('banjoTmp') or \
            name.startswith('me_alu_') or is_uniform_name(name):
        return 1
    if re.fullmatch(r'(r\d+|c\d+|oC\d|iTexCoord\d+|iColor\d+|iPos)', name):
        return 4
    raise Unsupported('variable ' + name)


def width(n):
    t = n[0]
    if t == 'num':
        return 1
    if t == 'var':
        return var_width(n[1])
    if t == 'const':
        return 4
    if t == 'swz':
        width(n[1])
        return len(n[2])
    if t == 'un':
        return width(n[2])
    if t == 'bin':
        if n[1] in ('&&', '||'):
            if width(n[2]) != 1 or width(n[3]) != 1:
                raise Unsupported('vector logic')
            return 1
        return max(width(n[2]), width(n[3]))
    if t == 'tern':
        if width(n[1]) != 1:
            raise Unsupported('vector condition')
        return max(width(n[2]), width(n[3]))
    if t == 'call':
        f, args = n[1], n[2]
        if f in FETCHES:
            return 4 if f != 'getWeights2D' else 2
        if f in ('dot', 'max4', 'BANJO_BOOL'):
            return 1
        if f in CONSTRUCTORS:
            if sum(width(a) for a in args) != CONSTRUCTORS[f] and not (len(args) == 1 and width(args[0]) == 1):
                raise Unsupported('constructor')
            return CONSTRUCTORS[f]
        if f in COMPONENTWISE:
            return max(width(a) for a in args)
        return 4  # an unknown call (cube, dst, ...): opaque, never copied
    raise Unsupported('node ' + t)


def bc(n, i):
    """Component i of n, a scalar broadcast."""
    return comp(n, 0) if width(n) == 1 else comp(n, i)


def comp(n, i):
    """The scalar expression of component i of n. Leaves: ('vc', var, c), ('sv', var), ('cc', const, c), ('num', t),
    ('bb', n); opaque: ('opq', call node, i) (fetches and unknown calls)."""
    t = n[0]
    if t == 'num':
        return n
    if t == 'var':
        return ('sv', n[1]) if var_width(n[1]) == 1 else ('vc', n[1], 'xyzw'[i])
    if t == 'const':
        return ('cc', n[1], 'xyzw'[i])
    if t == 'swz':
        inner = n[1]
        return comp(inner, 'xyzw'.index(n[2][i]))
    if t == 'un':
        return ('un', n[1], bc(n[2], i))
    if t == 'bin':
        return ('bin', n[1], bc(n[2], i), bc(n[3], i))
    if t == 'tern':
        return ('tern', comp(n[1], 0), bc(n[2], i), bc(n[3], i))
    if t == 'call':
        f, args = n[1], n[2]
        if f == 'BANJO_BOOL':
            if len(args) != 1 or args[0][0] != 'num':
                raise Unsupported('BANJO_BOOL')
            return ('bb', int(args[0][1]))
        if f == 'dot':
            a, b = args
            w = width(a)
            if width(b) != w:
                raise Unsupported('dot widths')
            return ('dot', tuple(comp(a, j) for j in range(w)), tuple(comp(b, j) for j in range(w)))
        if f == 'max4':
            return ('max4', tuple(bc(args[0], j) for j in range(4)))
        if f in CONSTRUCTORS:
            if len(args) == 1 and width(args[0]) == 1:
                return comp(args[0], 0)
            flat = [comp(a, j) for a in args for j in range(width(a))]
            return flat[i]
        if f in COMPONENTWISE:
            return ('call', f, tuple(bc(a, i) for a in args))
        return ('opq', n, i)
    raise Unsupported('node ' + t)


def full_reads(n):
    """Every component a (vector) node reads."""
    out = set()
    for i in range(width(n)):
        out |= reads(comp(n, i))
    return out


def reads(s):
    t = s[0]
    if t == 'vc':
        return set() if is_uniform_name(s[1]) else {f'{s[1]}.{s[2]}'}
    if t == 'sv':
        return set() if s[1] in SCALAR_NAMES or is_uniform_name(s[1]) else {s[1]}
    if t in ('cc', 'num', 'bb'):
        return set()
    if t == 'opq':
        out = set()
        for a in s[1][2]:
            out |= full_reads(a)
        return out
    if t == 'un':
        return reads(s[2])
    if t == 'bin':
        return reads(s[2]) | reads(s[3])
    if t == 'tern':
        return reads(s[1]) | reads(s[2]) | reads(s[3])
    if t == 'call':
        return set().union(*(reads(a) for a in s[2]))
    if t == 'dot':
        return set().union(*(reads(a) for a in s[1] + s[2]))
    if t == 'max4':
        return set().union(*(reads(a) for a in s[1]))
    raise Unsupported('reads ' + t)


# Fetches a copy may repeat (see mark_fetch_ok): the same texture, sampler and coordinates give the same texels.
HOISTABLE = {'tfetch1D', 'tfetch2D', 'tfetch3D'}


def copyable(s, fetch_ok=False):
    t = s[0]
    if t == 'opq':
        return fetch_ok and s[1][1] in HOISTABLE and all(
            copyable(comp(a, j), fetch_ok) for a in s[1][2] for j in range(width(a)))
    if t in ('vc', 'sv', 'cc', 'num', 'bb'):
        return True
    if t == 'un':
        return copyable(s[2], fetch_ok)
    if t == 'bin':
        return copyable(s[2], fetch_ok) and copyable(s[3], fetch_ok)
    if t == 'tern':
        return all(copyable(x, fetch_ok) for x in s[1:])
    if t == 'call':
        return all(copyable(a, fetch_ok) for a in s[2])
    if t == 'dot':
        return all(copyable(a, fetch_ok) for a in s[1] + s[2])
    if t == 'max4':
        return all(copyable(a, fetch_ok) for a in s[1])
    return False


def emit_vector(n):
    if n[0] == 'var' and is_uniform_name(n[1]):
        return n[1]
    w = width(n)
    if w == 1:
        return emit(comp(n, 0))
    return f'float{w}({", ".join(emit(comp(n, j)) for j in range(w))})'


def mark_fetch_ok(stmts, enclosing_uniform, state):
    """A fetch can be repeated earlier, with the same implicit derivatives, when every if around it tests uniform
    values only and no lane can have left before it (no kill, no alpha clip, no return under a per-pixel condition):
    then all four lanes of every quad run it, in the original and in the copy."""
    for s in stmts:
        s.fetch_ok = enclosing_uniform and not state['blocked']
        if s.kind in ('kill', 'aclip'):
            state['blocked'] = True
        elif s.kind == 'if':
            uniform = not reads(s.cond)
            mark_fetch_ok(s.body, enclosing_uniform and uniform, state)
            if not uniform and any(x.kind in ('ret', 'aclip', 'kill') for x in walk(s.body)):
                state['blocked'] = True


def emit(s):
    t = s[0]
    if t == 'num':
        return s[1]
    if t == 'vc':
        return f'{s[1]}.{s[2]}'
    if t == 'sv':
        return s[1]
    if t == 'cc':
        return f'__C{s[1]}.{s[2]}'
    if t == 'bb':
        return f'BANJO_BOOL({s[1]})'
    if t == 'un':
        return f'({s[1]}{emit(s[2])})'
    if t == 'bin':
        if s[1] in ('&&', '||'):  # no short-circuit branch (see banjoSel in shader_common.h)
            return f'{"banjoAnd" if s[1] == "&&" else "banjoOr"}({emit(s[2])}, {emit(s[3])})'
        return f'({emit(s[2])} {s[1]} {emit(s[3])})'
    if t == 'tern':
        return f'banjoSel({emit(s[1])}, {emit(s[2])}, {emit(s[3])})'
    if t == 'call':
        return f'{s[1]}({", ".join(emit(a) for a in s[2])})'
    if t == 'dot':
        w = len(s[1])
        if w == 1:
            return f'({emit(s[1][0])} * {emit(s[2][0])})'
        return (f'dot(float{w}({", ".join(emit(a) for a in s[1])}), '
                f'float{w}({", ".join(emit(a) for a in s[2])}))')
    if t == 'max4':
        return f'max4(float4({", ".join(emit(a) for a in s[1])}))'
    if t == 'opq' and s[1][1] in HOISTABLE:
        return f'{s[1][1]}({", ".join(emit_vector(a) for a in s[1][2])}).{"xyzw"[s[2]]}'
    raise Unsupported('emit ' + t)


# ---------------------------------------------------------------------------------------------------------------------
# Statements
ALPHA_CLIP_RE = re.compile(r'^\[branch\] if \(g_SpecConstants\(\) & SPEC_CONSTANT_ALPHA_TEST\)\s*\{\s*'
                           r'clip\(alphaTestValue\(oC0\.w\)\);$')
BLEND_SKIP_RE = re.compile(r'^if \(\(g_SpecConstants\(\) & SPEC_CONSTANT_BLEND_SKIP_(ALPHA|ZERO)\) != 0 && .*\) discard;$')
KILL_RE = re.compile(r'^clip\((?:banjoSel\((.*), -1, 1\)|(.*) \? -1 : 1)\);$')
DECL_RE = re.compile(r'^(float4|float|bool|int|uint|CubeMapData)\s+(\w+)\s*=\s*(.*);$')
ASSIGN_RE = re.compile(r'^(r\d+|oC\d|ps|p0|banjoTmp\d+)(?:\.([xyzw]{1,4}))?\s*=\s*(.*);$')
# (banjo_gpu_literals_cbuf, package 26: the translator writes BANJO_LITERAL_PS / _VS(N, x, y, z, w, mask) for a literal
# register with components NAK cannot encode; the same four words, so the copy keeps them as literals.)
LITERAL_RE = re.compile(r'^(?:asfloat\(uint4\(|BANJO_LITERAL_(?:PS|VS)\(\d+, )(0x[0-9A-Fa-f]+), (0x[0-9A-Fa-f]+), '
                        r'(0x[0-9A-Fa-f]+), (0x[0-9A-Fa-f]+)(?:\)\)|, \d+\))$')


class Stmt:
    __slots__ = ('kind', 'dest', 'comps', 'cond', 'body', 'line', 'end', 'pre', 'literal', 'decl', 'fetch_ok')

    def __init__(self, kind, line, **kw):
        self.kind, self.line, self.end = kind, line, line
        self.dest = self.comps = self.cond = self.body = self.pre = self.literal = None
        self.decl = False
        self.fetch_ok = False
        for k, v in kw.items():
            setattr(self, k, v)


def make_assign(dest, mask, text, line):
    node = Parser(text).whole()
    w = width(node)
    if mask is None:
        mask = '' if var_width(dest) == 1 else 'xyzw'
    if mask == '':
        if w != 1:
            raise Unsupported('scalar from vector')
        return Stmt('assign', line, dest=dest, comps={'': comp(node, 0)})
    if w != 1 and w < len(mask):
        raise Unsupported(f'width {w} into .{mask}')
    # (a wider value is truncated to its first components, as HLSL does)
    return Stmt('assign', line, dest=dest, comps={c: bc(node, j) for j, c in enumerate(mask)})


def parse_body(lines, first, last):
    """Statements of lines[first:last] (the inside of main's braces). Returns the top-level list."""
    root, stack, pending_if, i = [], [], None, first
    current = root
    while i < last:
        raw = lines[i]
        s = CONST_RE.sub(lambda m: f'__C{m.group(1)}', raw.strip())
        if s == '' or s.startswith('//'):
            i += 1
            continue
        if pending_if is not None:
            if s != '{':
                raise Unsupported('if without brace')
            stack.append((current, pending_if))
            current = pending_if.body
            pending_if = None
        elif s == '{' and i + 1 < last and lines[i + 1].strip().startswith('// banjo_gpu_pred_select'):
            # The translator's flattened predicated block (shader_recompiler.cpp banjoFlattenPredicated): read as the
            # `if (p0) { ... }` it replaces (the statements between the saved copies and the selects), which has the
            # same values.
            pm = re.match(r'^const bool banjoPred = (!?)p0;$', lines[i + 2].strip())
            if not pm or not lines[i + 3].strip().startswith('[branch] if (or(banjoPred, (g_SpecConstants() & '
                                                             'SPEC_CONSTANT_PRED_BRANCHES') \
                    or lines[i + 4].strip() != '{':
                raise Unsupported('pred_select block')
            first_body = i + 5
            while first_body < last and lines[first_body].strip().startswith('const ') and \
                    'banjoKeep_' in lines[first_body]:
                first_body += 1
            k = first_body
            while k < last and not re.match(r'^\w+ = select\((?:bool4\()?banjoPred\b', lines[k].strip()) and \
                    lines[k].strip() != '}':
                k += 1
            depth, end = 0, None
            for e in range(i, last):
                t = lines[e].strip()
                depth += t.count('{') - t.count('}')
                if depth == 0:
                    end = e
                    break
            if end is None:
                raise Unsupported('pred_select block not closed')
            cond = Parser(('!' if pm.group(1) else '') + 'p0').whole()
            st = Stmt('if', i, cond=comp(cond, 0), body=parse_body(lines, first_body, k))
            st.end = end
            current.append(st)
            i = end
        elif s == '{':  # a scope for the translator's temporaries (unique names): its statements stay in this list
            stack.append((current, None))
        elif s == '}':
            if not stack:
                raise Unsupported('unbalanced')
            current, owner = stack.pop()
            if owner is not None:
                owner.end = i
        elif ALPHA_CLIP_RE.match(s):
            j = i + 1
            while lines[j].strip() == '':
                j += 1
            if lines[j].strip() != '}':
                raise Unsupported('alpha clip block')
            current.append(Stmt('aclip', i))
            current[-1].end = j
            i = j
        elif BLEND_SKIP_RE.match(s):
            current.append(Stmt('kill', i, cond=None))
        elif KILL_RE.match(s):
            km = KILL_RE.match(s)
            current.append(Stmt('kill', i, cond=comp(Parser(km.group(1) or km.group(2)).whole(), 0)))
        elif s == 'return;' or s == 'return false;':
            current.append(Stmt('ret', i))
        elif s == 'return banjoAlphaFailsEarly(oC0.w);':  # in a copy
            current.append(Stmt('aclip', i))
        elif s.startswith('if (') and s.endswith(')'):
            cond = Parser(s[3:]).whole()
            if width(cond) != 1:
                raise Unsupported('vector if')
            pending_if = Stmt('if', i, cond=comp(cond, 0), body=[])
            current.append(pending_if)
        elif DECL_RE.match(s):
            m = DECL_RE.match(s)
            ty, name, text = m.groups()
            if ty in ('int', 'uint', 'CubeMapData'):
                if name not in ('a0', 'banjoNextA0', 'aL', 'cubeMapData'):
                    raise Unsupported('declaration ' + name)
            elif ty == 'float4' and LITERAL_RE.match(text):
                words = [int(x, 16) for x in LITERAL_RE.match(text).groups()]
                st = Stmt('assign', i, dest=name,
                          comps={c: ('num', f'asfloat({w:#x}u)') for c, w in zip('xyzw', words)})
                st.literal = [struct.unpack('<f', struct.pack('<I', w))[0] for w in words]
                st.decl = not stack
                current.append(st)
            else:
                st = make_assign(name, None, text, i)
                st.decl = not stack and not name.startswith('me_alu_')
                current.append(st)
        elif ASSIGN_RE.match(s):
            dest, mask, text = ASSIGN_RE.match(s).groups()
            current.append(make_assign(dest, mask, text, i))
        else:
            raise Unsupported('statement: ' + s[:60])
        i += 1
    if stack or pending_if is not None:
        raise Unsupported('unclosed block')
    return root


def find_main(lines):
    start = next((i for i, l in enumerate(lines) if l.startswith('void main(')), None)
    if start is None:
        raise Unsupported('no main')
    open_ = next(i for i in range(start, len(lines)) if lines[i].strip() == '{')
    depth = 0
    for i in range(open_, len(lines)):
        s = lines[i].strip()
        depth += s.count('{') - s.count('}')
        if depth == 0:
            return start, open_, i
    raise Unsupported('main not closed')


def walk(stmts):
    for s in stmts:
        yield s
        if s.kind == 'if':
            yield from walk(s.body)


# ---------------------------------------------------------------------------------------------------------------------
# Forward facts: per component, the pixel shader constant it holds (if it is exactly one) and a bound on its magnitude
# (with the constants that bound relies on).
class Fact:
    __slots__ = ('uniform', 'bound', 'guards', 'fetchy', 'value')

    def __init__(self, uniform=None, bound=None, guards=frozenset(), fetchy=False, value=None):
        self.uniform, self.bound, self.guards, self.fetchy = uniform, bound, guards, fetchy
        self.value = value  # the literal it holds, when the shader's own constants (or a literal) put one there


def literal_of(s, st):
    if s[0] == 'num':
        try:
            return float(s[1]) if not s[1].startswith('asfloat') else \
                struct.unpack('<f', struct.pack('<I', int(s[1][8:-2], 16)))[0]
        except ValueError:
            return None
    if s[0] in ('vc', 'sv'):
        r = reads(s)
        f = st.get(r.pop()) if r else None
        return f.value if f is not None else None
    return None


def fetchy(s, st):
    """Whether scalar expression s depends on a texture fetch (or another call the copy cannot repeat)."""
    if any(x[0] == 'opq' for x in iter_nodes(s)):
        return True
    return any(st[r].fetchy for r in reads(s) if r in st)


def bound(s, st):
    """(bound or None, guards) of scalar expression s in state st."""
    t = s[0]
    if t == 'num':
        try:
            v = abs(float(s[1])) if not s[1].startswith('asfloat') else abs(
                struct.unpack('<f', struct.pack('<I', int(s[1][8:-2], 16)))[0])
        except ValueError:
            return None, frozenset()
        return (v, frozenset()) if math.isfinite(v) else (None, frozenset())
    if t == 'bb':
        return 1.0, frozenset()
    if t == 'cc':
        return BIG, frozenset({('abs', s[1], s[2])})
    if t in ('vc', 'sv'):
        if t == 'sv' and s[1] in SCALAR_NAMES:
            return abs(SCALAR_NAMES[s[1]]), frozenset()
        f = st.get(reads(s).pop() if reads(s) else '')
        return (f.bound, f.guards) if f is not None else (None, frozenset())
    if t == 'un':
        return (1.0, frozenset()) if s[1] == '!' else bound(s[2], st)
    if t == 'bin':
        if s[1] in COMPARISONS or s[1] in ('&&', '||'):
            return 1.0, frozenset()
        (a, ga), (b, gb) = bound(s[2], st), bound(s[3], st)
        if a is None or b is None or s[1] == '/':
            return None, frozenset()
        return (a + b if s[1] in '+-' else a * b), ga | gb
    if t == 'tern':
        (a, ga), (b, gb) = bound(s[2], st), bound(s[3], st)
        return (None, frozenset()) if a is None or b is None else (max(a, b), ga | gb)
    if t == 'dot':
        total, g = 0.0, frozenset()
        for x, y in zip(s[1], s[2]):
            (a, ga), (b, gb) = bound(x, st), bound(y, st)
            if a is None or b is None:
                return None, frozenset()
            total, g = total + a * b, g | ga | gb
        return total, g
    if t == 'call':
        f = s[1]
        parts = [bound(a, st) for a in s[2]]
        if any(p[0] is None for p in parts):
            return None, frozenset()
        g = frozenset().union(*(p[1] for p in parts))
        if f in ('saturate', 'frac'):
            return 1.0, g
        if f in ('abs', 'min', 'max', 'clamp', 'select'):
            return max(p[0] for p in parts), g
        if f in ('floor', 'ceil', 'trunc', 'round'):
            return parts[0][0] + 1.0, g
        if f == 'sign' or f == 'isnan':
            return 1.0, g
        return None, frozenset()
    if t == 'max4':
        parts = [bound(a, st) for a in s[1]]
        if any(p[0] is None for p in parts):
            return None, frozenset()
        return max(p[0] for p in parts), frozenset().union(*(p[1] for p in parts))
    return None, frozenset()


def uniform_of(s, st):
    if s[0] == 'cc':
        return (s[1], s[2])
    if s[0] in ('vc', 'sv'):
        r = reads(s)
        f = st.get(r.pop()) if r else None
        return f.uniform if f is not None else None
    return None


def forward(stmts, st):
    """Annotates every assignment with its pre-state (for the zero rule); returns the state after stmts."""
    for s in stmts:
        if s.kind == 'assign':
            s.pre = st
            new = dict(st)
            for c, node in s.comps.items():
                name = s.dest + ('.' + c if c else '')
                if s.literal is not None:
                    v = s.literal['xyzw'.index(c)]
                    new[name] = Fact(None, abs(v) if math.isfinite(v) else None, value=v)
                    continue
                b, g = bound(node, st)
                new[name] = Fact(uniform_of(node, st), b, g, fetchy(node, st), literal_of(node, st))
            st = new
        elif s.kind == 'if':
            inner = forward(s.body, st)
            merged = {}
            cond_fetchy = fetchy(s.cond, st)
            for k in set(st) | set(inner):
                a, b = st.get(k), inner.get(k)
                if a is None or b is None:
                    merged[k] = Fact(fetchy=True)
                    continue
                if a is b:
                    merged[k] = a
                    continue
                u = a.uniform if a.uniform == b.uniform else None
                bd = None if a.bound is None or b.bound is None else max(a.bound, b.bound)
                v = a.value if a.value is not None and a.value == b.value and \
                    math.copysign(1, a.value) == math.copysign(1, b.value) else None
                merged[k] = Fact(u, bd, a.guards | b.guards if bd is not None else frozenset(),
                                 a.fetchy or b.fetchy or cond_fetchy, v)
            st = merged
    return st


def initial_state():
    st = {}
    for r in range(32):
        for c in 'xyzw':
            st[f'r{r}.{c}'] = Fact()
    st['ps'] = Fact(None, 0.0)
    st['p0'] = Fact(None, 1.0)
    return st


# ---------------------------------------------------------------------------------------------------------------------
# Backward slice
class Ctx:
    def __init__(self, allowed):
        self.allowed = allowed  # the (line, component) products that may be zeroed; None: every candidate
        self.guards = set()
        self.zeroed = set()  # (stmt line, component name)


def zero_rule(s, name, node, ctx):
    """The guards under which this product is +-0, for a constant factor times a finite value that depends on a
    texture fetch (only those can move the early point), or None."""
    if node[0] != 'bin' or node[1] != '*' or s.pre is None:
        return None
    if ctx.allowed is not None and (s.line, name) not in ctx.allowed:
        return None
    for k, y in ((node[2], node[3]), (node[3], node[2])):
        if not fetchy(y, s.pre):
            continue
        b, g = bound(y, s.pre)
        if b is None or b > FINITE:
            continue
        if literal_of(k, s.pre) == 0.0:  # a literal 0 (the shader's own constant): no run-time guard for it
            return frozenset({('literal',)}) | g
        u = uniform_of(k, s.pre)
        if u is not None:
            return frozenset({('zero', u[0], u[1])}) | g
    return None


def slice_block(stmts, live_out, ctx):
    live, out = set(live_out), []
    for s in reversed(stmts):
        if s.kind == 'ret':
            live = set()
            out.append((s, None))
        elif s.kind == 'aclip':
            live = live | {'oC0.w'}
            out.append((s, None))
        elif s.kind == 'kill':
            continue
        elif s.kind == 'assign':
            needed = {}
            for c, node in s.comps.items():
                name = s.dest + ('.' + c if c else '')
                if name in live:
                    needed[c] = node
            if not needed:
                continue
            for c in needed:
                live.discard(s.dest + ('.' + c if c else ''))
            kept = {}
            for c, node in needed.items():
                z = zero_rule(s, s.dest + ('.' + c if c else ''), node, ctx)
                if z is not None:
                    kept[c] = ('zero', node)
                    ctx.guards |= {g for g in z if g[0] != 'literal'}
                    ctx.zeroed.add((s.line, s.dest + ('.' + c if c else '')))
                else:
                    kept[c] = node
                    live |= reads(node)
            out.append((s, kept))
        elif s.kind == 'if':
            inner_live, inner = slice_block(s.body, live, ctx)
            if inner:
                live = live | inner_live | reads(s.cond)
                out.append((s, inner))
    out.reverse()
    return live, out


def item_copyable(item):
    s, extra = item
    if s.kind == 'assign':
        return all(copyable(n, s.fetch_ok) for n in extra.values() if n[0] != 'zero')
    if s.kind == 'if':
        return copyable(s.cond, s.fetch_ok) and all(item_copyable(sub) for sub in extra)
    return True


def copy_fetches(items):
    n = 0
    for s, extra in walk_items(items):
        nodes = [x for x in extra.values() if x[0] != 'zero'] if s.kind == 'assign' else (
            [s.cond] if s.kind == 'if' else [])
        seen = set()
        for node in nodes:
            for x in iter_nodes(node):
                if x[0] == 'opq' and id(x[1]) not in seen:
                    seen.add(id(x[1]))
                    n += 1
    return n


def sliced_nodes(item):
    s, extra = item
    if s.kind == 'assign':
        return [n for n in extra.values() if n[0] != 'zero']
    if s.kind == 'if':
        return [s.cond] + [n for sub in extra for n in sliced_nodes(sub)]
    return []


def count_copy(items):
    n = 0
    for s, extra in items:
        n += 1 if s.kind != 'if' else 1 + count_copy(extra)
    return n


# Operations a runtime zero may pass through on its way to the alpha (where +0 and -0 give the same test).
SIGN_SAFE_CALLS = {'saturate', 'abs', 'min', 'max', 'clamp', 'select'}


def sign_safe(s, tainted):
    """True when the tainted components s reads reach its value only through sign-insensitive operations."""
    if not (reads(s) & tainted):
        return True
    t = s[0]
    if t in ('vc', 'sv'):
        return True
    if t == 'un':
        return s[1] == '-' and sign_safe(s[2], tainted) or s[1] == '!' and sign_safe(s[2], tainted)
    if t == 'bin':
        if s[1] in ('+', '-', '*') or s[1] in COMPARISONS or s[1] in ('&&', '||'):
            return sign_safe(s[2], tainted) and sign_safe(s[3], tainted)
        return False
    if t == 'tern':
        return all(sign_safe(x, tainted) for x in s[1:])
    if t == 'call':
        return s[1] in SIGN_SAFE_CALLS and all(sign_safe(a, tainted) for a in s[2])
    if t == 'dot':
        return all(sign_safe(a, tainted) for a in s[1] + s[2])
    if t == 'max4':
        return all(sign_safe(a, tainted) for a in s[1])
    return False


def check_taint(items, tainted):
    for s, extra in items:
        if s.kind == 'assign':
            for c, node in extra.items():
                name = s.dest + ('.' + c if c else '')
                if node[0] == 'zero':
                    tainted.add(name)
                elif reads(node) & tainted:
                    if not sign_safe(node, tainted):
                        return False
                    tainted.add(name)
                else:
                    tainted.discard(name)
        elif s.kind == 'if':
            if not sign_safe(s.cond, tainted):
                return False
            inner = set(tainted)
            if not check_taint(extra, inner):
                return False
            tainted |= inner
    return True


# ---------------------------------------------------------------------------------------------------------------------
# Emission
def emit_items(items, out, indent):
    pad = '\t' * indent
    for s, extra in items:
        if s.kind == 'ret':
            out.append(f'{pad}return false;')
            return  # the rest of the block is not reached
        elif s.kind == 'aclip':
            out.append(f'{pad}return banjoAlphaFailsEarly(oC0.w);')
            return
        elif s.kind == 'if':
            out.append(f'{pad}if ({emit(s.cond)})')
            out.append(f'{pad}{{')
            emit_items(extra, out, indent + 1)
            out.append(f'{pad}}}')
        elif s.kind == 'assign':
            names = {c: s.dest + ('.' + c if c else '') for c in extra}
            written = set()
            conflict = False
            for c, node in extra.items():
                if node[0] != 'zero' and reads(node) & written:
                    conflict = True
                written.add(names[c])
            if not conflict:
                for c, node in extra.items():
                    out.append(f'{pad}{names[c]} = {emit_value(node)};')
            else:
                temps = []
                for c, node in extra.items():
                    t = f'banjoTmp{s.line}_{len(temps)}'
                    temps.append((names[c], t))
                    out.append(f'{pad}float {t} = {emit_value(node)};')
                for name, t in temps:
                    out.append(f'{pad}{name} = {t};')


def emit_value(node):
    if node[0] == 'zero':
        return '0.0'
    return emit(node)


def param_decl(name):
    if name in ('p0', 'iFace'):
        return f'bool {name}'
    if var_width(name) == 1:
        return f'float {name}'
    return f'float4 {name}'


# ---------------------------------------------------------------------------------------------------------------------
# Self-check: an interpreter for the same statement forms, float32.
class Run:
    def __init__(self, seed, consts, bools):
        self.seed, self.consts, self.bools = seed, consts, bools
        self.v = {}

    def get(self, name):
        if name in SCALAR_NAMES:
            return np.float32(SCALAR_NAMES[name])
        if name.split('.')[0] in UNIFORM_VECTORS:
            return np.float32(random.Random(zlib.crc32(name.encode()) ^ self.seed).uniform(-2, 2))
        if name not in self.v:
            return np.float32(np.nan)  # never written: undefined in the shader too
        return self.v[name]

    def opaque(self, node, i):
        vals = []
        for a in node[2]:
            for j in range(width(a)):
                vals.append(float(self.ev(comp(a, j))))
        slot = node[2][0][1] if node[2] and node[2][0][0] == 'var' else None  # the texture (descriptor) argument
        key = zlib.crc32(repr((node[1], slot, vals)).encode()) ^ self.seed
        rnd = random.Random(key)
        k = rnd.random()
        out = [rnd.choice([0.0, 1.0, rnd.random(), rnd.random(), rnd.uniform(-2, 2)]) for _ in range(4)]
        return np.float32(out[i]) if k > 0.02 else np.float32(0.0)

    def ev(self, s):
        t = s[0]
        with np.errstate(all='ignore'):
            if t == 'num':
                x = s[1]
                if x.startswith('asfloat('):
                    return np.float32(struct.unpack('<f', struct.pack('<I', int(x[8:-2], 16)))[0])
                if x.startswith('0x'):
                    return np.float32(int(x.rstrip('u'), 16))
                return np.float32(float(x))
            if t == 'vc':
                return self.get(f'{s[1]}.{s[2]}')
            if t == 'sv':
                return self.get(s[1])
            if t == 'cc':
                return self.consts[s[1]]['xyzw'.index(s[2])]
            if t == 'bb':
                return np.float32(self.bools[s[1]])
            if t == 'opq':
                return self.opaque(s[1], s[2])
            if t == 'un':
                x = self.ev(s[2])
                return -x if s[1] == '-' else np.float32(1.0 if x == 0 else 0.0)
            if t == 'bin':
                op = s[1]
                if op == '&&':
                    return np.float32(1.0 if self.ev(s[2]) != 0 and self.ev(s[3]) != 0 else 0.0)
                if op == '||':
                    return np.float32(1.0 if self.ev(s[2]) != 0 or self.ev(s[3]) != 0 else 0.0)
                a, b = self.ev(s[2]), self.ev(s[3])
                if op == '+':
                    return np.float32(a + b)
                if op == '-':
                    return np.float32(a - b)
                if op == '*':
                    return np.float32(a * b)
                if op == '/':
                    return np.float32(a / b)
                return np.float32(1.0 if {'==': a == b, '!=': a != b, '<': a < b, '<=': a <= b, '>': a > b,
                                          '>=': a >= b}[op] else 0.0)
            if t == 'tern':
                return self.ev(s[2]) if self.ev(s[1]) != 0 else self.ev(s[3])
            if t == 'dot':
                acc = np.float32(0.0)
                for j, (x, y) in enumerate(zip(s[1], s[2])):
                    p = np.float32(self.ev(x) * self.ev(y))
                    acc = p if j == 0 else np.float32(acc + p)
                return acc
            if t == 'max4':
                a, b, c, d = (self.ev(x) for x in s[1])
                return np.float32(np.fmax(np.fmax(a, b), np.fmax(c, d)))
            if t == 'call':
                f, a = s[1], [self.ev(x) for x in s[2]]
                x = a[0]
                r = {
                    'saturate': lambda: np.float32(0.0) if np.isnan(x) else np.float32(min(max(x, 0.0), 1.0)),
                    'abs': lambda: np.abs(x),
                    'rsqrt': lambda: np.float32(1.0) / np.sqrt(x),
                    'rcp': lambda: np.float32(1.0) / x,
                    'sqrt': lambda: np.sqrt(x),
                    'exp2': lambda: np.exp2(x),
                    'log2': lambda: np.log2(x),
                    'frac': lambda: x - np.floor(x),
                    'floor': lambda: np.floor(x),
                    'ceil': lambda: np.ceil(x),
                    'trunc': lambda: np.trunc(x),
                    'round': lambda: np.rint(x),
                    'sign': lambda: np.sign(x),
                    'sin': lambda: np.sin(x),
                    'cos': lambda: np.cos(x),
                    'isnan': lambda: np.float32(1.0 if np.isnan(x) else 0.0),
                    'clamp': lambda: np.fmin(np.fmax(x, a[1]), a[2]) if len(a) == 3 else None,
                    'min': lambda: np.fmin(x, a[1]) if len(a) == 2 else None,
                    'max': lambda: np.fmax(x, a[1]) if len(a) == 2 else None,
                    'select': lambda: a[1] if x != 0 else a[2],
                }[f]()
                return np.float32(r)
        raise Unsupported('ev ' + t)

    def block(self, stmts, snap_at=None, snaps=None):
        for idx, s in enumerate(stmts):
            if snap_at is not None and idx == snap_at:
                snaps.append(dict(self.v))
            if s.kind == 'assign':
                vals = {c: self.ev(node) for c, node in s.comps.items()}
                for c, v in vals.items():
                    self.v[s.dest + ('.' + c if c else '')] = np.float32(v)
            elif s.kind == 'if':
                if self.ev(s.cond) != 0:
                    r = self.block(s.body)
                    if r is not None:
                        return r
            elif s.kind == 'aclip':
                return ('clip', self.get('oC0.w'))
            elif s.kind == 'kill':
                if s.cond is not None and self.ev(s.cond) != 0:
                    return ('kill',)
            elif s.kind == 'ret':
                return ('ret',)
        if snap_at is not None and snap_at == len(stmts):
            snaps.append(dict(self.v))
        return None


def same(a, b):
    if np.isnan(a) and np.isnan(b):
        return True
    if a == 0 and b == 0:
        return True
    return np.float32(a).tobytes() == np.float32(b).tobytes()


def self_check(body, point, copy_stmts, params, guards, trials, name):
    """None if the copy agrees with main() on every trial where its guards hold, else a description."""
    consts_used = set()
    for s in walk(body):
        nodes = list(s.comps.values()) if s.kind == 'assign' else ([s.cond] if s.cond is not None else [])
        for n in nodes:
            for x in iter_nodes(n):
                if x[0] == 'cc':
                    consts_used.add(x[1])
    bools_used = set()
    for s in walk(body):
        nodes = list(s.comps.values()) if s.kind == 'assign' else ([s.cond] if s.cond is not None else [])
        for n in nodes:
            for x in iter_nodes(n):
                if x[0] == 'bb':
                    bools_used.add(x[1])
    zero_consts = {(g[1], g[2]) for g in guards if g[0] == 'zero'}
    checked = 0
    for trial in range(trials):
        rnd = random.Random(zlib.crc32(name.encode()) * 1000 + trial)
        consts = {}
        for n in consts_used:
            consts[n] = [np.float32(rnd.choice([0.0, 1.0, -1.0, 0.5, rnd.uniform(-2, 2), rnd.uniform(-2, 2),
                                                rnd.uniform(0, 1)])) for _ in range(4)]
        force = trial % 4 != 3
        for n, c in zero_consts:
            if force:
                consts.setdefault(n, [np.float32(0)] * 4)
                consts[n]['xyzw'.index(c)] = np.float32(0.0)
        bools = {b: rnd.random() < 0.5 for b in bools_used}
        run = Run(rnd.getrandbits(31), consts, bools)
        for r in range(16):
            vals = [rnd.choice([0.0, 1.0, rnd.uniform(-2, 2), rnd.uniform(-2, 2), rnd.uniform(0, 1)]) for _ in range(4)]
            for c, v in zip('xyzw', vals):
                run.v[f'iTexCoord{r}.{c}'] = np.float32(v)
        for c in 'xyzw':
            run.v[f'iPos.{c}'] = np.float32(rnd.uniform(0, 1280))
            for k in range(6):
                run.v[f'iColor{k}.{c}'] = np.float32(rnd.uniform(0, 1))
        run.v['iFace'] = np.float32(rnd.random() < 0.5)
        snaps = []
        result = run.block(body, snap_at=point, snaps=snaps)
        if not snaps:
            continue  # the original ended before the point (cannot happen at top level without a return)
        holds = all((consts[g[1]]['xyzw'.index(g[2])] == 0) if g[0] == 'zero'
                    else (abs(float(consts[g[1]]['xyzw'.index(g[2])])) <= BIG) for g in guards)
        if not holds:
            continue
        early = Run(run.seed, consts, bools)
        early.v = {k: v for k, v in snaps[0].items() if any(k == p or k.startswith(p + '.') for p in params)}
        copy_result = early.block(copy_stmts)
        checked += 1
        if result is None or result[0] == 'ret':
            if copy_result is not None and copy_result[0] == 'clip':
                return f'trial {trial}: the copy clips where main() does not', checked
        elif result[0] == 'clip':
            if copy_result is None or copy_result[0] != 'clip':
                return f'trial {trial}: main() clips with alpha {result[1]}, the copy returns {copy_result}', checked
            if not same(result[1], copy_result[1]):
                return f'trial {trial}: alpha {result[1]!r} in main(), {copy_result[1]!r} in the copy', checked
    return None, checked


def iter_nodes(n):
    yield n
    t = n[0]
    if t == 'un':
        yield from iter_nodes(n[2])
    elif t == 'bin':
        yield from iter_nodes(n[2])
        yield from iter_nodes(n[3])
    elif t == 'tern':
        for x in n[1:]:
            yield from iter_nodes(x)
    elif t == 'call':
        for x in n[2]:
            yield from iter_nodes(x)
    elif t == 'dot':
        for x in n[1] + n[2]:
            yield from iter_nodes(x)
    elif t == 'max4':
        for x in n[1]:
            yield from iter_nodes(x)
    elif t == 'opq':
        for a in n[1][2]:
            for j in range(width(a)):
                yield from iter_nodes(comp(a, j))


# ---------------------------------------------------------------------------------------------------------------------
def count_fetches(stmts):
    n = 0
    for s in walk(stmts):
        nodes = list(s.comps.values()) if s.kind == 'assign' else ([s.cond] if s.cond is not None else [])
        seen = set()
        for node in nodes:
            for x in iter_nodes(node):
                if x[0] == 'opq' and x[1][1] in FETCHES and id(x[1]) not in seen:
                    seen.add(id(x[1]))
                    n += 1
    return n


def plan(body, allowed):
    """The earliest top-level point after which the alpha's slice can be copied (zeroing the products `allowed`
    lets), and the slice after it: (point, live there, items, ctx)."""
    ctx = Ctx(allowed)
    _, items = slice_block(body, set(), ctx)
    by_stmt = {id(s): extra for s, extra in items}
    last_blocked = -1
    for idx, s in enumerate(body):
        if id(s) in by_stmt and not item_copyable((s, by_stmt[id(s)])):
            last_blocked = idx
    point = max(last_blocked + 1, first_real(body))  # never inside the register declarations
    ctx2 = Ctx(allowed)
    live_at, copy_items = slice_block(body[point:], set(), ctx2)
    return point, live_at, copy_items, ctx2


def best_plan(body):
    """Without runtime zeros, or with the fewest that give the earliest point."""
    none = plan(body, set())
    full = plan(body, None)
    if full[0] >= none[0] or not check_taint(full[2], set()):
        return none
    zeros = set(full[3].zeroed)
    for z in sorted(zeros):
        trial = plan(body, zeros - {z})
        if trial[0] <= full[0] and check_taint(trial[2], set()):
            zeros.discard(z)
    best = plan(body, zeros)
    return best if check_taint(best[2], set()) else none


def first_real(body):
    """Index of the first statement that is not one of main()'s declarations."""
    i = 0
    while i < len(body) and body[i].decl:
        i += 1
    return i


def guard_text(guards):
    parts = []
    for g in sorted(guards, key=lambda g: (g[0] != 'zero', g[1], g[2])):
        c = f'{const_text(g[1])}.{g[2]}'
        parts.append(f'{c} == 0.0' if g[0] == 'zero' else f'abs({c}) <= 1099511627776.0')
    text = parts[0]
    for p in parts[1:]:
        text = f'banjoAnd({text}, {p})'  # no short-circuit branches (banjo_gpu_branchless_select)
    return text


def strip(text):
    """The file as XenosRecomp wrote it: without an earlier run's copy and call site (so a new version of this pass
    can be run on a folder an older one already processed)."""
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        l = lines[i]
        if l.startswith(f'// {MARK} (tools/early_alpha.py)'):
            while lines[i] != '}':
                i += 1
            i += 2 if i + 1 < len(lines) and lines[i + 1] == '' else 1
            continue
        if l == f'\t// {MARK}: a 2x2 quad whose pixels all fail the alpha test stops here.':
            while lines[i] != '\t}':
                i += 1
            i += 1
            continue
        out.append(l)
        i += 1
    return '\n'.join(out)


def process(path, trials):
    text = open(path, encoding='utf-8').read()
    if 'bool banjoEarlyAlphaFails(' in text:
        text = strip(text)
        if 'banjoEarlyAlpha' in text.split('void main(')[1]:
            raise Unsupported('could not strip the earlier copy')
        open(path, 'w', encoding='utf-8', newline='\n').write(text)
    lines = text.split('\n')
    start, open_, close = find_main(lines)
    body = parse_body(lines, open_ + 1, close)
    forward(body, initial_state())
    mark_fetch_ok(body, True, {'blocked': False})
    if not any(s.kind == 'aclip' for s in walk(body)):
        raise Unsupported('no alpha clip')
    point, live_at, copy_items, ctx = best_plan(body)
    if not any(s.kind == 'aclip' for s, _ in walk_items(copy_items)):
        raise Unsupported('no clip in the copy')
    statements_after = sum(1 for _ in walk(body[point:]))
    hoisted = copy_fetches(copy_items)
    fetches_after = count_fetches(body[point:]) - hoisted
    size = count_copy(copy_items)
    if size > MAX_COPY:
        raise Unsupported(f'copy of {size} statements')
    if fetches_after < MIN_FETCHES_SKIPPED and statements_after < MIN_STATEMENTS_SKIPPED:
        return 'little', dict(point=point, fetches=fetches_after, statements=statements_after)
    # oC0 read before main() writes it (the final clip after the return paths, which no path reaches): a local.
    oc_written = any(s.kind == 'assign' and s.dest.startswith('oC') for s in walk(body[:point]))
    params = sorted({n.split('.')[0] for n in live_at if oc_written or not n.startswith('oC')},
                    key=lambda v: (re.sub(r'\d', '', v), int(re.sub(r'\D', '', v) or 0)))
    written = set()
    for s, extra in walk_items(copy_items):
        if s.kind == 'assign':
            written.add(s.dest)
    locals_ = sorted((written | {n.split('.')[0] for n in live_at}) - set(params))
    body_out = []
    emit_items(copy_items, body_out, 1)
    func = ['', f'// {MARK} (tools/early_alpha.py): the alpha main() clips with, from its registers after line '
                f'{body[point - 1].end + 1 if point else open_ + 1}', '// (a slice of the statements after it; see '
                'banjoAlphaFailsEarly / banjoEarlyAlphaQuad).',
            f'bool banjoEarlyAlphaFails({", ".join(param_decl(p) for p in params)})', '{']
    for l in locals_:
        if l == 'oC0' or re.fullmatch(r'r\d+', l):
            func.append(f'\tfloat4 {l} = 0.0;')
        elif l == 'ps':
            func.append('\tfloat ps = 0.0;')
        elif l == 'p0':
            func.append('\tbool p0 = false;')
        elif l.startswith('banjoTmp'):
            pass
        elif l.startswith('me_alu_'):
            func.append(f'\tfloat {l} = 0.0;')
        elif re.fullmatch(r'pb\d+', l):  # Renut's translator's block predicates (tools/renut_early_alpha.py)
            func.append(f'\tbool {l} = false;')
        else:
            raise Unsupported('local ' + l)
    func += body_out
    func += ['\treturn false;', '}']
    func_text = '\n'.join(func)
    func_text = re.sub(r'__C(\d+)', lambda m: const_text(int(m.group(1))), func_text)
    # self-check the emitted copy
    copy_body = parse_body(func_text.split('\n'), func.index('{') + 1, len(func) - 1)
    if np is None:
        raise Unsupported('numpy missing: no self-check')
    problem, checked = self_check(body, point, copy_body, params, ctx.guards, trials, os.path.basename(path))
    if problem:
        return 'differs', dict(problem=problem)
    call = f'banjoEarlyAlphaFails({", ".join(params)})'
    insert_at = body[point - 1].end + 1 if point else open_ + 1
    site = ['\t// ' + MARK + ': a 2x2 quad whose pixels all fail the alpha test stops here.',
            '\t[branch] if (g_SpecConstants() & SPEC_CONSTANT_EARLY_ALPHA)', '\t{']
    if ctx.guards:
        site += [f'\t\tconst bool banjoEarlyFails = {call};',
                 f'\t\t[branch] if ({guard_text(ctx.guards)})',
                 '\t\t\tbanjoEarlyAlphaQuad(banjoEarlyFails);']
    else:
        site += [f'\t\tbanjoEarlyAlphaQuad({call});']
    site += ['\t}']
    out = lines[:start] + func_text.split('\n')[1:] + [''] + lines[start:insert_at] + site + lines[insert_at:]
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    return 'copied', dict(point=point, fetches=fetches_after, hoisted=hoisted, statements=statements_after, size=size,
                          guards=len(ctx.guards), zeroed=len(ctx.zeroed), checked=checked)


def walk_items(items):
    for s, extra in items:
        yield s, extra
        if s.kind == 'if':
            yield from walk_items(extra)


def main():
    args = sys.argv[1:]
    folder = args[0]
    trials = int(args[args.index('--trials') + 1]) if '--trials' in args else 24
    only = args[args.index('--only') + 1] if '--only' in args else None
    counts = {}
    report = []
    for path in sorted(glob.glob(os.path.join(folder, 'ps_*.hlsl'))):
        name = os.path.basename(path)[:-5]
        if only and name != only:
            continue
        try:
            outcome, info = process(path, trials)
        except Unsupported as e:
            outcome, info = 'unsupported', dict(reason=str(e))
        except RecursionError:
            outcome, info = 'unsupported', dict(reason='recursion')
        counts[outcome] = counts.get(outcome, 0) + 1
        report.append(f'{name} {outcome} {info}')
    with open(os.path.join(folder, '..', 'early_alpha.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report) + '\n')
    print('early alpha: ' + ', '.join(f'{k} {v}' for k, v in sorted(counts.items())))


if __name__ == '__main__':
    main()
