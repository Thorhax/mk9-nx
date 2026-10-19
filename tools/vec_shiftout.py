#!/usr/bin/env python3
"""Byte-level proof that a function's results do not depend on the entry values of v32-v63.

non_argument_as_local turns v32-v63 into C++ locals that start at zero. tools/non_argument_audit.py lists three SA2
functions that read some of them before writing them: the audio resamplers sub_828A0DB0 (32-bit samples),
sub_828A2A60 (16-bit) and sub_828A3220 (8-bit). Their software-pipelined loops build each input vector in a shift
register (`vsldoi128 vD,vOld,vNew,4` per sample, after an unaligned `lvlx128`), so the first iteration starts from
whatever the registers held; those bytes are shifted out before the vector is used.

This walks the PPC disassembly (the `// insn` comments of the generated code) with a definedness state per byte of
every vector register: v32-v63 start undefined, everything else defined. Bytes flow through the instructions with
their exact mapping (vsldoi shifts bytes, vspltw splats a word, element-wise arithmetic needs every byte of its source
elements, loads and splat-immediates define everything). A vector store, or an instruction it does not model, that
may see an undefined byte is reported. Must-defined dataflow (intersection at joins, loops to a fixpoint); the count
register is tracked as a constant from `li`/`mtctr`, so a loop that always runs N times is unrolled (sub_828A2A60's
inner loop runs exactly twice: 2 x 4 shifts of 2 bytes = 16).

  python tools/vec_shiftout.py app/generated/default sub_828A0DB0 sub_828A2A60 sub_828A3220
Exit status 0 when no undefined byte reaches a sink in any of them. Generated code with or without the option.
"""
import glob
import os
import re
import sys

FULL = 0xFFFF


def listing(gen, name):
    pat = re.compile(r'^DEFINE_REX_FUNC\(' + re.escape(name) + r'\) \{\n(.*?)^\}\n', re.S | re.M)
    for f in sorted(glob.glob(os.path.join(gen, '*_recomp.*.cpp'))):
        m = pat.search(open(f, encoding='utf-8', errors='replace').read())
        if not m:
            continue
        items = []
        for ln in m.group(1).split('\n'):
            if ln.startswith('loc_'):
                items.append(('label', ln.strip()[:-1].lower().replace('loc_', '')))
            elif ln.startswith('\t// '):
                p = ln[4:].strip().split(None, 1)
                items.append(('insn', p[0], [x.strip() for x in p[1].split(',')] if len(p) > 1 else []))
        return items
    sys.exit(f'{name} not found in {gen}')


def vreg(x):
    m = re.fullmatch(r'v(\d+)', x)
    return int(m.group(1)) if m else None


def word_mask(m):
    """the words (PPC order: word k = bytes 4k..4k+3) whose four bytes are all defined"""
    out = 0
    for k in range(4):
        w = 0xF << (4 * k)
        if m & w == w:
            out |= w
    return out


def step(state, mn, ops, report, where):
    b = mn.rstrip('.')
    V = [vreg(o) for o in ops]
    if not b.startswith(('v', 'lv', 'stv')):
        return  # scalar instruction: no vector effect
    if b.startswith(('lvlx', 'lvrx', 'lvx', 'lvsl', 'lvsr', 'lvebx', 'lvehx', 'lvewx')) or b.startswith('vspltis'):
        state[V[0]] = FULL
        return
    if b.startswith('stv'):
        if state[V[0]] != FULL:
            report.append(f'{where}: {mn} {",".join(ops)} stores undefined bytes {state[V[0]] ^ FULL:04x}')
        return
    if b in ('vsldoi128', 'vsldoi'):
        a, c, sh = state[V[1]], state[V[2]], int(ops[3])
        cat = [(a >> i) & 1 for i in range(16)] + [(c >> i) & 1 for i in range(16)]
        state[V[0]] = sum(cat[sh + i] << i for i in range(16))
        return
    if b in ('vspltw128', 'vspltw'):
        w = (state[V[1]] >> (4 * int(ops[2]))) & 0xF
        state[V[0]] = FULL if w == 0xF else 0
        return
    if b in ('vor128', 'vor', 'vand', 'vand128', 'vxor', 'vxor128', 'vandc128', 'vandc', 'vnor128', 'vnor'):
        state[V[0]] = FULL if b.startswith('vxor') and ops[1] == ops[2] else state[V[1]] & state[V[2]]
        return
    if b in ('vupkhsb128', 'vupklsb128'):  # each word from the high (low) half of the source
        half = 0x00FF if b == 'vupkhsb128' else 0xFF00
        state[V[0]] = FULL if state[V[1]] & half == half else 0
        return
    if b in ('vcsxwfp128', 'vcuxwfp128', 'vcfsx', 'vcfux', 'vctsxs', 'vctuxs', 'vcfpsxws128', 'vcfpuxws128'):
        state[V[0]] = word_mask(state[V[1]])
        return
    if b in ('vmaddfp', 'vmaddfp128', 'vnmsubfp', 'vnmsubfp128'):
        state[V[0]] = word_mask(state[V[1]] & state[V[2]] & state[V[3]])
        return
    if b in ('vadduwm', 'vsubuwm', 'vaddfp', 'vaddfp128', 'vsubfp', 'vsubfp128', 'vmulfp128', 'vsrw', 'vsrw128',
             'vslw', 'vslw128', 'vsraw', 'vsraw128', 'vmaxfp', 'vmaxfp128', 'vminfp', 'vminfp128'):
        state[V[0]] = word_mask(state[V[1]] & state[V[2]])
        return
    report.append(f'{where}: {mn} not modelled (sink)')
    if V and V[0] is not None:
        state[V[0]] = 0


def analyze(gen, name):
    items = listing(gen, name)
    labels = {it[1]: i for i, it in enumerate(items) if it[0] == 'label'}
    n = len(items)
    succ = []
    for i, it in enumerate(items):
        nxt = [i + 1] if i + 1 < n else []
        if it[0] == 'label':
            succ.append(nxt)
            continue
        mn, ops = it[1], it[2]
        m = re.fullmatch(r'0x([0-9a-f]+)', ops[-1]) if ops else None
        tgt = m.group(1) if m else None
        if mn == 'b':
            succ.append([labels[tgt]] if tgt in labels else [])  # a tail call ends the path
        elif mn in ('blr', 'bctr'):
            succ.append([])
        elif mn.startswith('b') and mn not in ('bl', 'bctrl', 'blrl') and tgt is not None:
            succ.append(nxt + ([labels[tgt]] if tgt in labels else []))
        else:
            succ.append(nxt)
    entry = [FULL] * 128
    for v in range(32, 64):
        entry[v] = 0
    # facts keyed by (instruction, known count register); definedness and GPR constants merge by intersection
    IN = {(0, None): (entry, {})}
    work = [(0, None)]
    report, seen = [], set()
    while work:
        key = work.pop()
        i, ctr = key
        vst, consts = IN[key]
        st, cs = list(vst), dict(consts)
        it = items[i]
        rep = []
        nexts = succ[i]
        if it[0] == 'insn':
            mn, ops = it[1], it[2]
            step(st, mn, ops, rep, f'{i}')
            dst = ops[0] if ops else ''
            if mn == 'li' and re.fullmatch(r'r\d+', dst):
                cs[dst] = int(ops[1], 0)
            elif mn == 'mtctr':
                ctr = cs.get(ops[0])
            elif mn in ('bdnz', 'bdz'):
                if ctr is not None:
                    ctr -= 1
                    taken = (ctr != 0) if mn == 'bdnz' else (ctr == 0)
                    nexts = [s for s in succ[i] if (s != i + 1) == taken] or succ[i][:1]
            elif re.fullmatch(r'r\d+', dst) and not mn.startswith(('st', 'cmp', 'b', 'dcb', 'mt', 'tw', 'td')):
                cs.pop(dst, None)
            if mn in ('bl', 'bctrl', 'blrl'):
                cs, ctr = {}, None
        for r in rep:
            if r not in seen:
                seen.add(r)
                report.append(r)
        for j in nexts:
            k2 = (j, ctr)
            if k2 not in IN:
                IN[k2] = (st, cs)
                work.append(k2)
            else:
                ost, ocs = IN[k2]
                new = [a & b for a, b in zip(ost, st)]
                ncs = {r: v for r, v in ocs.items() if cs.get(r) == v}
                if new != ost or ncs != ocs:
                    IN[k2] = (new, ncs)
                    work.append(k2)
    insns = sum(1 for it in items if it[0] == 'insn')
    stores = sum(1 for it in items if it[0] == 'insn' and it[1].startswith('stv'))
    print(f'{name}: {insns} instructions, {stores} vector stores, {len({i for i, _ in IN})}/{n} reachable;',
          'no entry byte of v32-v63 reaches a store' if not report else f'{len(report)} REPORTS')
    for r in report:
        print('   ', r)
    return not report


if __name__ == '__main__':
    ok = all([analyze(sys.argv[1], nm) for nm in sys.argv[2:]])
    sys.exit(0 if ok else 1)
