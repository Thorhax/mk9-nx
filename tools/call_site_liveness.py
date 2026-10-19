#!/usr/bin/env python3
# From masseffect-nx tests/hot_fuzz/liveness.py (generated file pattern made generic for sa2-nx).
"""liveness.py GEN ADDR [ADDR...]  : for every direct call site `bl 0xADDR` in the generated code, find which volatile
registers the caller reads after the call before redefining them (r3/f1/v1 return regs are reported too)."""
import re, sys, glob, collections

gen = sys.argv[1]
targets = [a.lower().replace('sub_', '').replace('0x', '') for a in sys.argv[2:]]
VOL_R = {0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
VOL_F = set(range(0, 14))
VOL_V = set(range(0, 14))


def parse_ops(s):
    return [x.strip() for x in s.split(',')] if s.strip() else []


def regs_of(op):
    return [(m.group(1), int(m.group(2))) for m in re.finditer(r'\b([rfv])(\d+)\b', op)]


READ_ALL = ('st', 'dcb', 'icb', 'tw', 'td', 'cmp', 'fcmp', 'mtctr', 'mtlr', 'mtspr', 'mtcr', 'mtfsf', 'mtfsb', 'sync', 'eieio',
            'isync', 'lwsync')


def analyze(mn, ops):
    reads = set()
    writes = set()
    b = mn.rstrip('.')
    all_regs = [regs_of(o) for o in ops]
    if (b.startswith(('l', 'st', 'dcb', 'addi')) and not b.startswith(('lis',))) or b == 'addis':
        # rA == r0 means literal 0 in D-form (d(r0)) and in X-form (rD,r0,rB) memory ops and addi/addis
        if len(ops) == 3 and all_regs[1] == [('r', 0)]:
            all_regs[1] = []
        if ops and re.search(r'\(r0\)', ops[-1]):
            all_regs[-1] = [r for r in all_regs[-1] if r != ('r', 0)]
    flat = [r for rs in all_regs for r in rs]
    if b.startswith(READ_ALL):
        for r in flat:
            reads.add(r)
        if re.match(r'^st(w|d|b|h|fs|fd)u(x)?$', b):
            m = re.search(r'\((r\d+)\)', ops[-1]) if ops else None
            if m:
                writes.add(('r', int(m.group(1)[1:])))
        return reads, writes
    if b.startswith('b'):
        for r in flat:
            reads.add(r)
        return reads, writes
    if ops:
        for r in all_regs[0]:
            writes.add(r)
        for rs in all_regs[1:]:
            for r in rs:
                reads.add(r)
        if re.match(r'^l(wz|bz|hz|ha|fs|fd|d)(u|ux)$', b):
            m = re.search(r'\((r\d+)\)', ops[-1])
            if m:
                writes.add(('r', int(m.group(1)[1:])))
        if b.startswith(('rlwimi', 'rldimi')):
            for r in all_regs[0]:
                reads.add(r)
    return reads, writes


def regset_vol(rs):
    return [(k, n) for k, n in rs if (k == 'r' and n in VOL_R) or (k == 'f' and n in VOL_F) or (k == 'v' and n in VOL_V)]


def func_instrs(lines):
    ins = []
    labels = {}
    for ln in lines:
        m = re.match(r'^(loc_[0-9A-F]+):', ln)
        if m:
            labels[m.group(1)[4:].lower()] = len(ins)
            continue
        m = re.match(r'^\t// (\S+)\s*(.*)$', ln)
        if m:
            ins.append((m.group(1), m.group(2).strip()))
    return ins, labels


ALLVOL = frozenset({('r', n) for n in VOL_R} | {('f', n) for n in VOL_F} | {('v', n) for n in VOL_V})
res = {t: collections.Counter() for t in targets}
nsites = collections.Counter()
for f in sorted(glob.glob(gen + '/*_recomp.*.cpp')):
    txt = open(f, errors='ignore').read()
    if not any(('// bl 0x' + t) in txt for t in targets):
        continue
    for m in re.finditer(r'^DEFINE_REX_FUNC\((sub_[0-9A-F]{8})\) \{\n(.*?)^\}\n', txt, re.S | re.M):
        body = m.group(2)
        if not any(('// bl 0x' + t) in body for t in targets):
            continue
        ins, labels = func_instrs(body.split('\n'))
        for i, (mn, ops) in enumerate(ins):
            if mn != 'bl':
                continue
            t = ops.replace('0x', '').lower()
            if t not in targets:
                continue
            nsites[t] += 1
            seen = set()
            found = collections.Counter()
            stack = [(i + 1, frozenset())]
            while stack:
                pc, defd = stack.pop()
                while pc < len(ins):
                    k2 = (pc, defd)
                    if k2 in seen:
                        break
                    seen.add(k2)
                    mn2, ops2 = ins[pc]
                    o = parse_ops(ops2)
                    if mn2 == 'blr':
                        # return registers are live-out
                        for r in (('r', 3), ('f', 1), ('v', 1), ('r', 4)):
                            pass
                        break
                    if mn2 in ('bl', 'bctrl'):
                        defd = defd | ALLVOL
                        pc += 1
                        continue
                    if mn2 == 'bctr':
                        break
                    if mn2 == 'b':
                        tgt = ops2.replace('0x', '').lower()
                        if tgt in labels:
                            pc = labels[tgt]
                            continue
                        break
                    reads, writes = analyze(mn2, o)
                    for r in regset_vol(reads):
                        if r not in defd:
                            found[r] += 1
                    if mn2.startswith('b') and not mn2.startswith('bic'):
                        mt = re.search(r'0x([0-9a-f]{8})', ops2)
                        if mt and mt.group(1) in labels:
                            stack.append((labels[mt.group(1)], defd))
                        pc += 1
                        continue
                    defd = defd | frozenset(writes)
                    pc += 1
            for r in found:
                res[t][r] += 1

for t in targets:
    print('sub_%s: %d direct call sites' % (t.upper(), nsites[t]))
    items = sorted(res[t].items(), key=lambda kv: (-kv[1], kv[0]))
    print('  volatile regs read after the call before redefinition (reg:sites):',
          ', '.join('%s%d:%d' % (k, n, c) for (k, n), c in items) or '(none)')
