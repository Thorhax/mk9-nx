#!/usr/bin/env python3
"""Audits the code pointers of a module against the functions the generator registered: every 4-aligned data word
(outside the code range) whose value lands in the code range of this module or of another one, and every address
the code builds with lis + addi/ori, when the target is not a registered function start.

tools/pointer_fixpoint.py only looks at targets in unclaimed gaps of default.xex. This tool also reports targets
inside a function body (an entry point the generator merged into the code before it), the stage DLL, and pointers
from one module into the other. Most hits are not callable pointers, so it sorts them:

  vtable-like     at least 2 of the 6 neighbouring words are registered functions: almost certainly a callable
                  entry. Every one of these is a crash risk ("Call to invalid or unregistered function").
  EH state map    a (code address, small state number) pair of a C++ exception IP-to-state table: never called.
  SEH scope       an address next to the `mr r8,r8` marker MSVC puts around __try blocks: a scope record of an
                  SEH table (begin, end, filter, handler), used only when an exception is raised.
  jump table      a target of a jump table the generator already handles (tools/switch_check.py).
  other           data that happens to look like a code address (stage data, floats, SEH scope records); for
                  review only.

  python tools/pointer_audit.py                         default.xex, with the DLL's data as a second source
  python tools/pointer_audit.py --module dll            the stage DLL
  python tools/pointer_audit.py --module dll -v         also list the 'other' hits

Needs the image dumps written by codegen with REXGLUE_DUMP_IMAGE=out/default_image.bin (the DLL goes to
out/default_image.Data_DLL.bin) and the generated trees.
"""
import argparse
import collections
import glob
import os
import re
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ppc_mini import Image, decode, text  # noqa: E402
import switch_check as sc  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = {
    # name: (image dump, base, code range lo, hi (code only: the DLL's import thunks start at 0x88014470), gen dir)
    'default': ('out/default_image.bin', 0x82000000, 0x82360000, 0x83376864, 'app/generated/default'),
    'dll': ('out/default_image.Data_DLL.bin', 0x88000000, 0x88010000, 0x88014470, 'app/generated/data_dll'),
}


def registered(gen):
    """Function starts of the module's function table (DEFINE_REX_FUNC bodies, imports, save/restore helpers)."""
    out = set()
    for p in glob.glob(os.path.join(gen, '*_init.cpp')):
        out |= {int(m.group(1), 16) for m in re.finditer(r'\{ 0x([0-9A-F]+), \w+ \}', open(p).read())}
    return out


def owners(gen):
    """address -> function whose generated body holds it."""
    out = {}
    for path in glob.glob(os.path.join(gen, '*recomp*.cpp')):
        cur = addr = None
        for line in open(path, errors='replace'):
            m = re.match(r'DEFINE_REX_FUNC\(sub_([0-9A-F]+)\)', line)
            if m:
                cur = addr = int(m.group(1), 16)
                continue
            if cur is None:
                continue
            if line.startswith('}'):
                cur = None
                continue
            m = re.match(r'loc_([0-9A-F]+):', line)
            if m:
                addr = int(m.group(1), 16)
                continue
            m = re.match(r'\t// (\S.*)$', line)
            if m and not m.group(1).startswith(('PPC_', '0x')):
                out.setdefault(addr, cur)
                addr += 4
    return out


def jump_targets(img, lo, hi):
    br = sc.Branches(img, lo, hi)
    out = set()
    for bctr in range(lo, hi, 4):
        if img.word(bctr) != 0x4E800420:
            continue
        shape = sc.table_shape(img, bctr)
        if not shape or shape['kind'].endswith('?'):
            continue
        n, _ = sc.bound_of(img, br, bctr, shape, lo)
        n = n or sc.inline_count(img, shape, lo, hi)
        if n:
            out |= set(sc.read_targets(img, shape, n))
    return out


def built_addresses(img, lo, hi):
    """(value, address) for lis rX,hi followed within 8 instructions by addi/ori rY,rX,lo (same rule as
    tools/pointer_fixpoint.py)."""
    for a in range(lo, hi - 4, 4):
        x = img.word(a)
        if x >> 26 != 15 or (x >> 16) & 31:
            continue
        rx, upper = (x >> 21) & 31, (x & 0xFFFF) << 16
        for k in range(1, 9):
            y = img.word(a + 4 * k)
            op, rt, ra, imm = y >> 26, (y >> 21) & 31, (y >> 16) & 31, y & 0xFFFF
            if op in (14, 24) and ra == rx:
                yield (upper + (imm - 0x10000 if op == 14 and imm & 0x8000 else imm)) & 0xFFFFFFFF, a + 4 * k
                if rt == rx:
                    break
                continue
            if op == 18 or (op == 19 and (y >> 1) & 0x3FF in (16, 528)):
                break
            if rt == rx and op not in (36, 37, 38, 44, 45, 47, 54, 55):
                break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--module', choices=sorted(MODULES), default='default', help='module whose code is checked')
    ap.add_argument('-v', '--verbose', action='store_true', help="also list the 'other' hits")
    a = ap.parse_args()

    dump, base, lo, hi, gen = MODULES[a.module]
    img = Image(os.path.join(ROOT, dump), base)
    gen = os.path.join(ROOT, gen)
    starts, own = registered(gen), owners(gen)
    jt = jump_targets(img, lo, hi)
    sources = [(name, Image(os.path.join(ROOT, m[0]), m[1]), m[2], m[3]) for name, m in MODULES.items()
               if os.path.exists(os.path.join(ROOT, m[0]))]

    hits = collections.defaultdict(list)  # target -> [(source module, where, kind)]
    for name, src, slo, shi in sources:
        for off in range(0, len(src.data) - 3, 4):
            where = src.base + off
            if slo <= where < shi:
                continue  # code words are instructions, not pointers
            v = struct.unpack_from('>I', src.data, off)[0]
            if v & 3 or not lo <= v < hi or v in starts:
                continue
            # A vtable or callback table: registered functions around it, and neither small numbers nor other
            # pointers into function bodies (those are the scope records of SEH tables: count, begin, end, filter,
            # handler, and the IP-to-state maps of C++ EH: address, state).
            nb = [src.word(where + 4 * k) for k in (-3, -2, -1, 1, 2, 3) if src.contains(where + 4 * k)]
            vt = (sum(1 for x in nb if x in starts) >= 2 and not any(0 < x < 0x10000 or x == 0xFFFFFFFF for x in nb)
                  and sum(1 for x in nb if lo <= x < hi and x not in starts) <= 1)
            state = src.word(where + 4) if src.contains(where + 8) else None
            near = [src.word(where + d) for d in (-8, 8) if src.contains(where + d)]
            eh = (state is not None and (state < 0x400 or state == 0xFFFFFFFF)
                  and any(lo <= x < hi for x in near))
            # MSVC brackets the __try blocks of SEH with `mr r8,r8` (0x7D084378); scope records point next to it.
            seh = 0x7D084378 in (img.word(v), img.word(v - 4))
            kind = ('SEH scope' if seh else 'vtable-like' if vt else 'jump table' if v in jt else
                    'EH state map' if eh else 'other')
            hits[v].append((name, where, kind))
    for v, where in built_addresses(img, lo, hi):
        if not v & 3 and lo <= v < hi and v not in starts:
            nxt = decode(img.word(where + 4), where + 4)
            kind = 'jump table' if nxt.op in ('lwzx', 'lbzx', 'lhzx', 'add') or v in jt else 'built in code'
            hits[v].append((a.module, where, kind))

    by_kind = collections.defaultdict(list)
    for v, refs in sorted(hits.items()):
        kinds = {k for _, _, k in refs}
        kind = next(k for k in ('vtable-like', 'built in code', 'jump table', 'EH state map', 'SEH scope', 'other')
                    if k in kinds)
        by_kind[kind].append((v, refs))
    print(f'{a.module}: code {lo:08X}-{hi:08X}, {len(starts)} registered functions; pointers to non-starts: '
          + ', '.join(f'{k} {len(v)}' for k, v in by_kind.items()))
    for kind in ('vtable-like', 'built in code', 'other'):
        if kind == 'other' and not a.verbose:
            continue
        print(f'== {kind}')
        for v, refs in by_kind.get(kind, []):
            where = ', '.join(f'{n}:0x{w:08X}' for n, w, _ in refs[:3]) + (f' (+{len(refs) - 3})' if len(refs) > 3 else '')
            inside = f'inside sub_{own[v]:08X}' if v in own else 'in a gap'
            prev = text(decode(img.word(v - 4), v - 4))
            print(f'  0x{v:08X} {inside}, from {where} | before: {prev} | {text(decode(img.word(v), v))}')


if __name__ == '__main__':
    main()
