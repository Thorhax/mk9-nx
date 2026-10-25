#!/usr/bin/env python3
"""Guest single-precision loads byte-swapped in the vector unit on AArch64 (post-codegen step of the "optimised"
codegen flavour; docs/enhancements.md, "The second CPU round of 2026-10-12"). Works on any ReXGlue port.

Guest memory is big-endian, so an `lfs` is a 32-bit load, a byte swap and a conversion to double. The generator
writes it through an integer:
    temp.u32 = REX_LOAD_D_U32(r3.u32, 172);      ldr w0, [..] ; rev w0, w0 ; fmov s0, w0
    f13.f64 = double(temp.f32);                  fcvt d13, s0
and the move from the general registers to the vector unit (`fmov s0, w0`) is the slow step on the Cortex-A57: 5
cycles, after the load's 4 and the swap's 1. Loaded straight into the vector unit and swapped there it is
    ldr s0, [..] ; rev32 v0.8b, v0.8b            5 + 3 cycles, one instruction less
The same four bytes in the same order: REX_F32_FROM_BE() below, checked by sdk/tests/float_loads_test.cpp against the
integer form through simde's model of the instruction.

This step rewrites the two statements into
    f13.f64 = double(REX_LOAD_D_F32(r3.u32, 172));
The generated pch defines REX_LOAD_D_F32 / REX_LOAD_F32 as the vector form on AArch64 and as the integer form
elsewhere (the PC build is unchanged) or with REX_FLOAT_LOADS_INTEGER defined.

Not rewritten: loads that feed an any-mode helper (they test the bits as an integer), stores (the value comes from
a conversion in the vector unit and needs the same number of instructions either way), and a load from an address a
float was stored to earlier in the same straight line of code (no label, call or loop barrier in between): there the
compiler hands the stored value on without any load, which it can only do while both sides are the integer form.

  python tools/float_loads.py --gen app/generated/default [--check] [--undo]
Idempotent.
"""
import argparse
import os
import re
import sys

STORE = re.compile(r'^\tREX_STORE(?:_D)?_U32\((.*), (?:ctx\.)?temp\.u32\);$')
FENCE = re.compile(r'^loc_|\(ctx, base[,)]|REX_CALL_INDIRECT_|REX_LOOP_BARRIER|^DEFINE_REX_FUNC|^\}')
LOAD = re.compile(r'^\t((?:ctx\.)?temp)\.u32 = (REX_LOAD(?:_D)?)_U32\((.*)\);$')
USE = re.compile(r'^\t((?:ctx\.)?[fl_]+\d+)\.f64 = double\(((?:ctx\.)?temp)\.f32\);$')
NEW = re.compile(r'^\t((?:ctx\.)?[fl_]+\d+)\.f64 = double\((REX_LOAD(?:_D)?)_F32\((.*)\)\);$')
MARKER = '// rexglue ports: single-precision loads swapped in the vector unit (tools/float_loads.py)'
MACROS = MARKER + '''
#ifndef REX_LOAD_F32
#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(REX_FLOAT_LOADS_INTEGER)
[[gnu::always_inline]] static inline float RexFloatFromBigEndian(const void* p) noexcept {
  float f;
  __builtin_memcpy(&f, p, 4);  // ldr s
  const float32x2_t v = vset_lane_f32(f, vdup_n_f32(0.0f), 0);
  return vget_lane_f32(vreinterpret_f32_u8(vrev32_u8(vreinterpret_u8_f32(v))), 0);  // rev32
}
#define REX_LOAD_F32(x) RexFloatFromBigEndian(base + (u32)(x) + REX_PHYS_HOST_OFFSET(x))
#define REX_LOAD_D_F32(r, d) RexFloatFromBigEndian(REX_ADDR_D(r, d))
#else
#define REX_LOAD_F32(x) std::bit_cast<float>(u32(REX_LOAD_U32(x)))
#define REX_LOAD_D_F32(r, d) std::bit_cast<float>(u32(REX_LOAD_D_U32(r, d)))
#endif
#endif
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', required=True, help='generated folder (*_recomp.N.cpp and *_pch.h)')
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--undo', action='store_true')
    a = ap.parse_args()
    files = sorted(f for f in os.listdir(a.gen) if re.match(r'\w+_recomp\.\d+\.cpp$', f))
    pchs = [f for f in os.listdir(a.gen) if f.endswith('_pch.h')]
    if not files or len(pchs) != 1:
        sys.exit('ERROR (post-codegen patch): float_loads: no *_recomp.N.cpp or no single *_pch.h in %s' % a.gen)
    sites = present = changed = forwarded = 0
    for f in files:
        path = os.path.join(a.gen, f)
        lines = open(path, encoding='utf-8').read().split('\n')
        out = []
        i = 0
        n = 0
        stored = set()  # addresses of the float stores since the last label, call or barrier
        while i < len(lines):
            line = lines[i]
            if FENCE.search(line):
                stored.clear()
            else:
                st = STORE.match(line)
                if st:
                    stored.add(st.group(1))
            if a.undo:
                m = NEW.match(line)
                if m:
                    out.append('\ttemp.u32 = %s_U32(%s);' % (m.group(2), m.group(3)))
                    out.append('\t%s.f64 = double(temp.f32);' % m.group(1))
                    n += 1
                    i += 1
                    continue
            else:
                if NEW.match(line):
                    present += 1
                m = LOAD.match(line)
                u = USE.match(lines[i + 1]) if m and i + 1 < len(lines) else None
                if m and u and u.group(2) == m.group(1) and m.group(3) in stored:
                    forwarded += 1
                elif m and u and u.group(2) == m.group(1):
                    out.append('\t%s.f64 = double(%s_F32(%s));' % (u.group(1), m.group(2), m.group(3)))
                    n += 1
                    i += 2
                    continue
            out.append(line)
            i += 1
        if n:
            sites += n
            changed += 1
            if not a.check:
                open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    pch = os.path.join(a.gen, pchs[0])
    text = open(pch, encoding='utf-8').read()
    if not a.undo and MARKER not in text:
        # After the load macros it builds on (REX_ADDR_D, REX_LOAD_D_U32).
        anchor = '#define REX_MEMORY_SIZE'
        if anchor not in text:
            sys.exit('ERROR (post-codegen patch): float_loads: no REX_MEMORY_SIZE in %s: the generated pch changed' % pch)
        if not a.check:
            open(pch, 'w', encoding='utf-8', newline='\n').write(text.replace(anchor, MACROS + anchor, 1))
    print('float_loads%s: %d single-precision loads %s in %d files (%d already in that form; %d kept as integers '
          'because a float was just stored to the same address)' %
          (' (check only)' if a.check else '', sites, 'restored' if a.undo else 'rewritten', changed, present,
           forwarded))


if __name__ == '__main__':
    main()
