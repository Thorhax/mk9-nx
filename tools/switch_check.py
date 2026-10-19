#!/usr/bin/env python3
"""Checks every jump table of a module against the generated code, and writes the [[switch_tables]] entries that fix
the ones the code generator got wrong.

The generator turns each `bctr` through a jump table into a C++ switch whose default is __builtin_trap()
("Switch case out of range"). If its analysis sized a table too small, the game runs fine until it takes one of the
missing cases, then dies with an illegal instruction (SA2: sub_824D6468, 8 cases generated of 68, on loading the
Battle DLC's desert demo, stage 48). If it missed a table altogether, the bctr becomes an indirect tail call to a label
inside the function ("Call to invalid or unregistered function"). If it read a table with the wrong shape, cases jump
to the wrong place.

The truth is in the PPC code. This tool reads every bctr of the code range and follows its target register back:

    absolute    lis/addi rT, table ; rlwinm rS, rI, 2, 0, 29 ; lwzx rX, rT, rS ; mtctr rX ; bctr
    byte        lis/addi rT, table ; lbzx rO, rT, rI ; [rlwinm rO, rO, sh] ; lis/addi rB, base ; add rX, rB, rO
    halfword    lis/addi rT, table ; rlwinm rS, rI, 1, 0, 30 ; lhzx rO, rT, rS ; [rlwinm] ; lis/addi rB ; add rX, rB, rO

and finds the bound of the index rI on the straight-line code that leads to the bctr: `cmplwi crK, rI', N` +
`bgt crK` (N + 1 cases; rI' holds the same value as rI, `mr` copies followed) or `bge` (N cases), or a
`clrlwi`/`andi.` mask. That code starts at a join; every way into it is checked (the fall-through and each branch to
it, a taken `ble`/`blt` bounding too), and the bound is the largest over them. An absolute table placed in the code
right after its bctr ends at its lowest target. Then it compares case by case with the switch the generator wrote at
that bctr (addresses from the labels plus one per instruction comment).

The generator itself has used the same value-numbering rule since 2026-10-05 (straightLineSwitchBound in
sdk/src/codegen/function_scanner.cpp); its old rule, the first cmplwi/clrlwi naming the register on the way back, is
what truncated the four tables of overrides.toml. This check stays the safety net after each codegen.

  python tools/switch_check.py [--image out/default_image.bin] [--gen app/generated/default] [--toml out.toml]
  python tools/switch_check.py --image out/default_image.Data_DLL.bin --base 0x88000000 --code 88010000:88014470 \\
      --gen app/generated/data_dll

Report sections: truncated / wrong targets / missed (bctr emitted as an indirect call) are crash risks and go to
--toml; "extra" means the generator has more cases than the bound allows (harmless: never taken); "no bound" lists the
tables whose size the code does not state, with the generator's count, for review by hand.
"""
import argparse
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_mini import Image, decode, is_nop  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_RUN = 160


def ins_at(img, a):
    return decode(img.word(a), a)


def find_def(img, reg, before, limit=48):
    """The nearest instruction before `before` that writes `reg`, on the straight-line code (None past a terminator)."""
    a = before - 4
    for _ in range(limit):
        if not img.contains(a):
            return None
        i = ins_at(img, a)
        if i.unconditional:
            return None
        if reg in i.writes:
            return i
        a -= 4
    return None


def const_value(img, reg, before, depth=0):
    """The constant in `reg` just before `before`, if it was built with lis/li/addi/ori/addis/mr (else None)."""
    if depth > 6:
        return None
    d = find_def(img, reg, before)
    if d is None:
        return None
    if d.op in ('lis', 'li'):
        return d.imm & 0xFFFFFFFF
    if d.op in ('addi', 'addis'):
        v = const_value(img, d.ra, d.addr, depth + 1)
        return None if v is None else (v + d.imm) & 0xFFFFFFFF
    if d.op == 'ori':
        v = const_value(img, d.rt, d.addr, depth + 1)  # ori: source is the RS field (rt)
        return None if v is None else v | d.imm
    if d.op == 'mr':
        return const_value(img, d.rt, d.addr, depth + 1)
    return None


def unshift(img, reg, before, scale):
    """reg was made by `rlwinm reg, rI, scale, 0, 31 - scale` (slwi): returns (rI, address); else (None, None)."""
    d = find_def(img, reg, before)
    if d and d.op == 'rlwinm' and d.sh == scale and d.mb == 0 and d.me == 31 - scale:
        return d.rt, d.addr
    return None, None


def table_shape(img, bctr):
    """Follows the CTR value of a bctr back to a jump table. Returns a dict or None (not a table jump)."""
    a, mt = bctr - 4, None
    for _ in range(4):
        i = ins_at(img, a)
        if i.op == 'mtctr':
            mt = i
            break
        if not is_nop(i):
            return None
        a -= 4
    if mt is None:
        return None
    x = find_def(img, mt.rt, mt.addr)
    if x is None:
        return None
    if x.op == 'lwzx':
        for t_reg, s_reg in ((x.ra, x.rb), (x.rb, x.ra)):
            table = const_value(img, t_reg, x.addr)
            if table is None:
                continue
            idx, at = unshift(img, s_reg, x.addr, 2)
            if idx is None:
                return {'kind': 'absolute?', 'table': table, 'load': x.addr}
            return {'kind': 'absolute', 'table': table, 'index': idx, 'index_use': at, 'load': x.addr, 'width': 4}
        return None
    if x.op == 'add':
        for b_reg, o_reg in ((x.ra, x.rb), (x.rb, x.ra)):
            base = const_value(img, b_reg, x.addr)
            if base is None:
                continue
            d = find_def(img, o_reg, x.addr)
            shift = 0
            if d is not None and d.op == 'rlwinm' and d.mb == 0 and d.me == 31 - d.sh:
                shift = d.sh
                d = find_def(img, d.rt, d.addr)
            if d is None or d.op not in ('lbzx', 'lhzx'):
                return {'kind': 'offset?', 'base': base, 'load': x.addr}
            width = 1 if d.op == 'lbzx' else 2
            for t_reg, s_reg in ((d.ra, d.rb), (d.rb, d.ra)):
                table = const_value(img, t_reg, d.addr)
                if table is None:
                    continue
                if width == 1:
                    idx, at = s_reg, d.addr
                else:
                    idx, at = unshift(img, s_reg, d.addr, 1)
                    if idx is None:
                        return {'kind': 'offset?', 'base': base, 'load': d.addr}
                return {'kind': 'byte' if width == 1 else 'half', 'table': table, 'base': base, 'shift': shift,
                        'index': idx, 'index_use': at, 'load': d.addr, 'width': width}
            return {'kind': 'offset?', 'base': base, 'load': d.addr}
        return None
    return None


def straight_run(img, end, lo, branches):
    """Addresses of the straight-line code that falls through to `end` (exclusive), oldest first. It starts after an
    unconditional branch or at a join (an address some branch targets), so every path to `end` runs all of it."""
    run, a = [], end - 4
    while len(run) < MAX_RUN and a >= lo and img.contains(a):
        i = ins_at(img, a)
        if i.unconditional:
            break
        run.append(a)
        if branches.preds(a):
            break
        a -= 4
    run.reverse()
    return run


class Branches:
    """Index of every direct branch target in the code range: target -> [branch addresses]."""

    def __init__(self, img, lo, hi):
        self.by_target = {}
        for a in range(lo, hi, 4):
            w = img.word(a)
            op = w >> 26
            if op in (16, 18) and not w & 1:
                i = decode(w, a)
                self.by_target.setdefault(i.target, []).append(a)

    def preds(self, t):
        return self.by_target.get(t, [])


def path_bound(img, seq, entry, index):
    """Bound of `index` at the end of the instruction sequence `seq` (forward value numbering: `mr` copies a value,
    any other write makes a new one). `entry` is the position in `seq` of the branch into the run, taken on this
    path (or None)."""
    val = {r: ('init', r) for r in range(32)}
    fresh = [0]

    def new():
        fresh[0] += 1
        return ('v', fresh[0])

    crdef = {}
    bounds = {}  # value -> (cases, where)

    def add_bound(v, n, where):
        if v is not None and (v not in bounds or n < bounds[v][0]):
            bounds[v] = (n, where)

    for pos, a in enumerate(seq):
        i = ins_at(img, a)
        if i.op in ('bc', 'bclr') and not i.link and (i.bo & 0x14) != 0x14 and i.bo & 0x04:
            # Condition test without CTR decrement. Falls through when the condition is false; the branch into the
            # run is the one taken.
            taken = pos == entry
            want = bool(i.bo & 0x08)  # branch if the CR bit equals this
            bit, field = i.bi & 3, i.bi >> 2
            holds = want if taken else not want  # the CR bit's value on the path to the bctr
            c = crdef.get(field)
            if c and c[0] == 'cmpli':  # unsigned compares only: a signed bound lets negative indexes through
                _, v, n, where = c
                if bit == 1 and not holds:  # not GT: index <= n
                    add_bound(v, n + 1, where)
                elif bit == 0 and holds:  # LT: index < n
                    add_bound(v, n, where)
            continue
        if i.op == 'rlwinm' and i.sh == 0 and i.me == 31 and i.mb > 0:
            v = new()
            val[i.ra] = v
            add_bound(v, 1 << (32 - i.mb), a)
            if i.rc:
                crdef[0] = ('other',)
            continue
        if i.op == 'andi.' and i.imm & (i.imm + 1) == 0:
            v = new()
            val[i.ra] = v
            add_bound(v, i.imm + 1, a)
            crdef[0] = ('other',)
            continue
        if i.op in ('cmpli', 'cmpi') and not i.sh:
            crdef[i.rt >> 2] = (i.op, val[i.ra], i.imm, a)
            continue
        if i.op == 'mr':
            val[i.ra] = val[i.rt]
            if i.crw:
                crdef[0] = ('other',)
            continue
        for f in i.crw:
            crdef[f] = ('other',)
        for r in i.writes:
            val[r] = new()
    return bounds.get(val[index], (None, None))


def bound_of(img, branches, bctr, shape, lo):
    """Number of cases the code allows for the switch index (None if some path to the bctr does not state it).

    The straight-line run before the bctr starts at a join or after an unconditional branch. Every way into it is
    checked: the fall-through from the code above, and each branch to its first instruction (a dispatch loop jumps
    back with `cmplwi` + `ble`, SA2's sub_8284C850). The bound is the largest over those paths."""
    run = straight_run(img, bctr, lo, branches)
    if not run:
        return None, None
    paths = []
    if not ins_at(img, run[0] - 4).unconditional and run[0] - 4 >= lo:
        paths.append((straight_run(img, run[0], lo, branches), None))
    for p in branches.preds(run[0]):
        prefix = straight_run(img, p, lo, branches) + [p]
        paths.append((prefix, len(prefix) - 1))
    if not paths:
        paths.append(([], None))
    best = (0, None)
    for prefix, entry in paths:
        n, where = path_bound(img, prefix + run, entry, shape['index'])
        if n is None:
            return None, None
        if n > best[0]:
            best = (n, where)
    return best


def inline_count(img, shape, lo, hi):
    """An absolute table placed in the code right after its bctr ends where the code resumes, at its lowest target
    (SA2: sub_82141700 and sub_82436E48 switch on an argument with no bound check). None if it is not that shape."""
    t = shape['table']
    if shape['kind'] != 'absolute' or not lo <= t < hi:
        return None
    k, low = 0, None
    while low is None or t + 4 * k < low:
        v = img.word(t + 4 * k)
        if v & 3 or not lo <= v < hi or v < t:
            return None
        low = v if low is None else min(low, v)
        k += 1
    return k if t + 4 * k == low else None


def read_targets(img, shape, n):
    out = []
    for k in range(n):
        if shape['kind'] == 'absolute':
            out.append(img.word(shape['table'] + 4 * k))
        elif shape['kind'] == 'byte':
            out.append((shape['base'] + (img.byte(shape['table'] + k) << shape['shift'])) & 0xFFFFFFFF)
        else:
            out.append((shape['base'] + (img.half(shape['table'] + 2 * k) << shape['shift'])) & 0xFFFFFFFF)
    return out


def generated_bctrs(gen):
    """{bctr address: [(function, kind, register, [case targets])]} from the generated code. kind is 'switch' or
    'indirect' (REX_CALL_INDIRECT_FUNC). Addresses come from the labels plus one per instruction comment."""
    out, starts = {}, set()
    for path in sorted(glob.glob(os.path.join(gen, '*_recomp.*.cpp'))):
        func = addr = None
        lines = open(path, errors='replace').read().split('\n')
        n = 0
        while n < len(lines):
            line = lines[n]
            m = re.match(r'DEFINE_REX_FUNC\(sub_([0-9A-F]+)\)', line)
            if m:
                func = addr = int(m.group(1), 16)
                starts.add(func)
                n += 1
                continue
            if func is None:
                n += 1
                continue
            if line.startswith('}'):
                func = None
                n += 1
                continue
            m = re.match(r'^loc_([0-9A-F]+):', line)
            if m:
                addr = int(m.group(1), 16)
                n += 1
                continue
            m = re.match(r'^\t// (\S.*)$', line)
            if m and not m.group(1).startswith(('PPC_', '0x')):
                here = addr
                addr += 4
                if m.group(1).split()[0] == 'bctr':
                    n += 1
                    while n < len(lines) and lines[n].strip() == '':
                        n += 1
                    sm = re.match(r'\tswitch \(ctx\.r(\d+)\.u32\) \{', lines[n])
                    if sm:
                        reg, cases, cur = int(sm.group(1)), [], None
                        n += 1
                        while not lines[n].startswith('\tdefault'):
                            cm = re.match(r'\tcase (\d+):', lines[n])
                            if cm:
                                cur = int(cm.group(1))
                                assert cur == len(cases), (hex(here), cur)
                                cases.append(None)
                            else:
                                t = re.search(r'goto loc_([0-9A-F]+);', lines[n]) or \
                                    re.search(r'\b(?:sub|__imp__sub)_([0-9A-F]+)\(ctx, base\);', lines[n])
                                if t and cases[cur] is None:
                                    cases[cur] = int(t.group(1), 16)
                                elif 'jump to null value' in lines[n] and cases[cur] is None:
                                    cases[cur] = 0
                            n += 1
                        out.setdefault(here, []).append((func, 'switch', reg, cases))
                    elif 'REX_CALL_INDIRECT_FUNC' in '\n'.join(lines[n:n + 3]):
                        out.setdefault(here, []).append((func, 'indirect', None, []))
                    continue
            n += 1
    return out, starts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--image', default=os.path.join(ROOT, 'out', 'default_image.bin'))
    ap.add_argument('--base', default='0x82000000', help='image base (hex)')
    ap.add_argument('--gen', default=os.path.join(ROOT, 'app', 'generated', 'default'))
    ap.add_argument('--code', default='82360000:83376864', help='code range lo:hi (hex)')
    ap.add_argument('--toml', help='write [[switch_tables]] entries for the tables to fix here')
    ap.add_argument('-v', '--verbose', action='store_true', help='also list the tables with extra cases')
    a = ap.parse_args()
    lo, hi = (int(x, 16) for x in a.code.split(':'))
    img = Image(a.image, int(a.base, 16))
    branches = Branches(img, lo, hi)
    gen, starts = generated_bctrs(a.gen)

    stats = {}
    fixes, report = [], {'truncated': [], 'wrong targets': [], 'missed': [], 'extra': [], 'no bound': [],
                         'not generated': [], 'odd shape': []}
    for bctr in range(lo, hi, 4):
        if img.word(bctr) != 0x4E800420:
            continue
        shape = table_shape(img, bctr)
        if shape is None:
            stats['indirect'] = stats.get('indirect', 0) + 1
            continue
        if shape['kind'].endswith('?'):
            report['odd shape'].append((bctr, shape, gen.get(bctr)))
            continue
        stats[shape['kind']] = stats.get(shape['kind'], 0) + 1
        # The index register must still hold the index at the bctr (the switch reads it there).
        for k in range(shape['index_use'] + 4, bctr, 4):
            if shape['index'] in ins_at(img, k).writes:
                report['odd shape'].append((bctr, dict(shape, why='index register rewritten'), gen.get(bctr)))
                break
        else:
            n, where = bound_of(img, branches, bctr, shape, lo)
            if n is None:
                n = inline_count(img, shape, lo, hi)
                where = shape['table'] if n else None
            g = gen.get(bctr)
            if not g:
                report['not generated'].append((bctr, shape, n))
                continue
            if n is None:
                report['no bound'].append((bctr, shape, g))
                continue
            truth = read_targets(img, shape, n)
            if any(t & 3 or not lo <= t < hi for t in truth):
                report['odd shape'].append((bctr, dict(shape, why=f'target outside code with {n} cases'), g))
                continue
            for func, kind, reg, cases in g:
                entry = (bctr, func, shape, n, where, kind, reg, cases, truth)
                if kind == 'indirect':
                    report['missed'].append(entry)
                    fixes.append(entry)
                elif cases[:n] != truth[:len(cases)]:
                    report['wrong targets'].append(entry)
                    fixes.append(entry)
                elif len(cases) < n:
                    report['truncated'].append(entry)
                    fixes.append(entry)
                elif reg != shape['index']:
                    report['wrong targets'].append(entry)
                    fixes.append(entry)
                elif len(cases) > n:
                    report['extra'].append(entry)

    total = sum(len(v) for v in gen.values())
    nsw = sum(1 for v in gen.values() for e in v if e[1] == 'switch')
    print(f'bctr in the binary: {stats}; generated: {nsw} switches, {total - nsw} indirect')
    for key in ('truncated', 'wrong targets', 'missed'):
        print(f'{key}: {len(report[key])}')
        for bctr, func, shape, n, where, kind, reg, cases, truth in report[key]:
            diff = [k for k in range(min(n, len(cases))) if cases[k] != truth[k]]
            print(f'  sub_{func:08X} bctr 0x{bctr:08X} {shape["kind"]} table 0x{shape["table"]:08X} r{shape["index"]}'
                  f' bound {n} (0x{where:08X}); generated {kind} r{reg} with {len(cases)} cases'
                  + (f', first wrong case {diff[0]}: 0x{cases[diff[0]] or 0:08X} instead of 0x{truth[diff[0]]:08X}'
                     if diff else ''))
    print(f'extra cases (generated more than the bound, never taken): {len(report["extra"])}')
    if a.verbose:
        for bctr, func, shape, n, where, kind, reg, cases, truth in report['extra']:
            print(f'  sub_{func:08X} bctr 0x{bctr:08X} {shape["kind"]} bound {n}, generated {len(cases)}')
    print(f'no bound in the code: {len(report["no bound"])}')
    for bctr, shape, g in report['no bound']:
        for func, kind, reg, cases in g:
            note = ''
            if kind == 'indirect' and not lo <= shape['table'] < hi:
                # A table of function pointers in data (a state machine): the indirect call is right as long as
                # every entry is a function start.
                ents, k = [], 0
                while True:
                    v = img.word(shape['table'] + 4 * k)
                    if v & 3 or not lo <= v < hi:
                        break
                    ents.append(v)
                    k += 1
                bad = [v for v in ents if v not in starts]
                note = (f'; function-pointer table, {len(ents)} entries, '
                        + (f'NOT function starts: {", ".join(f"0x{v:08X}" for v in bad)}' if bad
                           else 'all function starts'))
            print(f'  sub_{func:08X} bctr 0x{bctr:08X} {shape["kind"]} table 0x{shape["table"]:08X} r{shape["index"]}:'
                  f' generated {kind} with {len(cases)} cases{note}')
    print(f'table jumps with no generated code (unreachable or not discovered): {len(report["not generated"])}')
    for bctr, shape, n in report['not generated']:
        print(f'  bctr 0x{bctr:08X} {shape["kind"]} table 0x{shape["table"]:08X} bound {n}')
    print(f'odd shapes (review by hand): {len(report["odd shape"])}')
    for bctr, shape, g in report['odd shape']:
        print(f'  bctr 0x{bctr:08X} {shape}' + (f' generated {[(hex(f), k, len(c)) for f, k, r, c in g]}' if g else ''))

    if a.toml and fixes:
        with open(a.toml, 'w') as f:
            for bctr, func, shape, n, where, kind, reg, cases, truth in fixes:
                what = ('emitted as an indirect call' if kind == 'indirect' else
                        f'generated {len(cases)} cases' + ('' if cases[:n] == truth[:len(cases)] else ', wrong targets'))
                f.write(f'\n# sub_{func:08X}: {shape["kind"]} table at 0x{shape["table"]:08X}, {n} cases (bound at '
                        f'0x{where:08X}); {what} (tools/switch_check.py)\n')
                f.write(f'[[switch_tables]]\naddress = 0x{bctr:08X}\nregister = {shape["index"]}\nlabels = [\n')
                for k in range(0, len(truth), 8):
                    f.write('    ' + ', '.join(f'0x{t:08X}' for t in truth[k:k + 8]) + ',\n')
                f.write(']\n')
        print('wrote', a.toml)


if __name__ == '__main__':
    main()
