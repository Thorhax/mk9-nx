#!/usr/bin/env python3
"""Rewrite a*b + c (and c + a*b) as mad(a, b, c) in translated vertex shader HLSL (backlog S23).

The translator marks oPos `precise`; DXC then gives the whole position chain (bones, dp4) NoContraction, so NAK
cannot fuse it into FFMA. DXC lowers a `precise` mad() to GLSL Fma: still invariant (the same expression gives the
same bits in every shader), but fused. Only statement right-hand sides `<lhs> = <expr>;` in main() are touched;
anything the small parser does not understand is left as is.

  usage: hlsl_fma.py <in dir> <out dir>
"""
import os, re, sys

TOKEN = re.compile(r'\s*(?:(\d+\.\d*(?:[eE][-+]?\d+)?|\d*\.\d+(?:[eE][-+]?\d+)?|0x[0-9A-Fa-f]+u?|\d+u?)|([A-Za-z_]\w*)|(==|!=|<=|>=|&&|\|\||<<|>>|[-+*/%<>(),.\[\]?:!&|^~]))')

class P:
    def __init__(s, text):
        s.toks = []; pos = 0
        while pos < len(text):
            m = TOKEN.match(text, pos)
            if not m or m.end() == pos:
                if text[pos:].strip() == '': break
                raise ValueError('tok')
            s.toks.append(m.group(1) or m.group(2) or m.group(3)); pos = m.end()
        s.i = 0
    def peek(s): return s.toks[s.i] if s.i < len(s.toks) else None
    def take(s, t=None):
        v = s.peek()
        if t is not None and v != t: raise ValueError('expected ' + t)
        s.i += 1; return v
    # precedence climbing for the operators the translator emits
    PREC = {'||': 1, '&&': 2, '|': 3, '^': 4, '&': 5, '==': 6, '!=': 6, '<': 7, '>': 7, '<=': 7, '>=': 7,
            '<<': 8, '>>': 8, '+': 9, '-': 9, '*': 10, '/': 10, '%': 10}
    def expr(s, minp=0):
        lhs = s.unary()
        while True:
            op = s.peek()
            if op == '?' and minp == 0:
                s.take('?'); a = s.expr(); s.take(':'); b = s.expr()
                lhs = ('?', lhs, a, b); continue
            p = P.PREC.get(op)
            if p is None or p < minp: return lhs
            s.take(); rhs = s.expr(p + 1)
            lhs = ('bin', op, lhs, rhs)
    def unary(s):
        t = s.peek()
        if t in ('-', '!', '~', '+'):
            s.take(); return ('un', t, s.unary())
        return s.postfix(s.primary())
    def primary(s):
        t = s.take()
        if t == '(':
            e = s.expr(); s.take(')'); return ('par', e)
        if t is None: raise ValueError('eof')
        if re.match(r'[A-Za-z_]', t) and s.peek() == '(':
            s.take('('); args = []
            if s.peek() != ')':
                args.append(s.expr())
                while s.peek() == ',':
                    s.take(','); args.append(s.expr())
            s.take(')'); return ('call', t, args)
        if re.match(r'[A-Za-z_0-9.]', t): return ('atom', t)
        raise ValueError('primary ' + t)
    def postfix(s, e):
        while True:
            t = s.peek()
            if t == '.':
                s.take('.'); e = ('mem', e, s.take())
            elif t == '[':
                s.take('['); i = s.expr(); s.take(']'); e = ('idx', e, i)
            else: return e

def strip(e):
    while e[0] == 'par': e = e[1]
    return e

def fma(e, count):
    k = e[0]
    if k == 'bin':
        a, b = fma(e[2], count), fma(e[3], count)
        if e[1] == '+':
            sa, sb = strip(a), strip(b)
            if sb[0] == 'bin' and sb[1] == '*':      # c + (x*y)
                count[0] += 1; return ('call', 'mad', [sb[2], sb[3], a])
            if sa[0] == 'bin' and sa[1] == '*':      # (x*y) + c
                count[0] += 1; return ('call', 'mad', [sa[2], sa[3], b])
        return ('bin', e[1], a, b)
    if k == 'par': return ('par', fma(e[1], count))
    if k == 'call': return ('call', e[1], [fma(x, count) for x in e[2]])
    if k == 'un': return ('un', e[1], fma(e[2], count))
    if k == 'mem': return ('mem', fma(e[1], count), e[2])
    if k == 'idx': return ('idx', fma(e[1], count), fma(e[2], count))
    if k == '?': return ('?', fma(e[1], count), fma(e[2], count), fma(e[3], count))
    return e

def emit(e):
    k = e[0]
    if k == 'atom': return e[1]
    if k == 'par': return '(' + emit(e[1]) + ')'
    if k == 'call': return e[1] + '(' + ', '.join(emit(x) for x in e[2]) + ')'
    if k == 'un': return e[1] + emit(e[2])
    if k == 'mem': return emit(e[1]) + '.' + e[2]
    if k == 'idx': return emit(e[1]) + '[' + emit(e[2]) + ']'
    if k == 'bin': return '(' + emit(e[2]) + ' ' + e[1] + ' ' + emit(e[3]) + ')'
    if k == '?': return '(' + emit(e[1]) + ' ? ' + emit(e[2]) + ' : ' + emit(e[3]) + ')'
    raise ValueError(k)

ASSIGN = re.compile(r'^(\s*(?:precise\s+)?(?:float[1-4]?\s+)?[A-Za-z_]\w*(?:\.[xyzw]+)?\s*=\s*)(.*);(\s*)$')

def process(text):
    out = []; inmain = False; n = 0
    for line in text.split('\n'):
        if re.match(r'\s*void main\s*\(', line): inmain = True
        m = ASSIGN.match(line) if inmain else None
        if m and ('*' in m.group(2)) and ('+' in m.group(2)):
            try:
                cnt = [0]; e = fma(P(m.group(2)).expr(), cnt)
                if cnt[0]:
                    line = m.group(1) + emit(e) + ';' + m.group(3); n += cnt[0]
            except ValueError:
                pass
        out.append(line)
    return '\n'.join(out), n

def main():
    src, dst = sys.argv[1], sys.argv[2]
    os.makedirs(dst, exist_ok=True); total = 0
    for f in sorted(os.listdir(src)):
        if not f.endswith('.hlsl'): continue
        t, n = process(open(os.path.join(src, f)).read()) if f.startswith('vs_') else (open(os.path.join(src, f)).read(), 0)
        open(os.path.join(dst, f), 'w').write(t); total += n
    print('mad rewrites:', total)

if __name__ == '__main__':
    main()
