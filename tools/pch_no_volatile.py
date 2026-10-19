#!/usr/bin/env python3
"""Guest loads/stores without `volatile` in a generated pch (post-codegen step of the "optimised" codegen flavour,
tools\\codegen.ps1; from masseffect-nx's tools/pch_no_volatile.py, measured there at +15-20 % fps).

The generated REX_LOAD_*/REX_STORE_* (and the D-form variants) read guest memory through volatile pointers, which stops
the compiler from keeping values in registers or pairing and scheduling accesses. MMIO macros (REX_MM_*) keep volatile.
Defining REX_GUEST_VOLATILE as `volatile` restores the generated behaviour. Idempotent.

Risk (why it is a build flavour with an A/B NRO, not the default): a guest loop that polls memory with no call or
atomic inside could have its load hoisted out of the loop. masseffect-nx shipped it without loop barriers; the classic
NRO is the comparison.

  python tools/pch_no_volatile.py app/generated/default/sa2_pch.h [app/generated/data_dll/sa2_pch.h ...] [--check]
"""
import sys

MARK = 'REX_GUEST_VOLATILE'
PREFIXES = ('#define REX_LOAD_U', '#define REX_STORE_U', '#define REX_LOAD_D_U', '#define REX_STORE_D_U')


def patch(path, check):
    text = open(path, encoding='utf-8').read()
    if MARK in text:
        print(f'{path}: already without volatile')
        return True
    if check:
        print(f'{path}: still volatile')
        return False
    lines = text.split('\n')
    changed = 0
    for i, line in enumerate(lines):
        if line.startswith(PREFIXES):
            new = line.replace('(volatile u', '(REX_GUEST_VOLATILE u')
            changed += new != line
            lines[i] = new
    if not changed or '#define REX_LOAD_U8(x)' not in text:
        sys.exit(f'{path}: no "(volatile u" REX_LOAD/REX_STORE macros found; the generated pch changed')
    text = '\n'.join(lines).replace('#define REX_LOAD_U8(x)', '''// sa2-nx optimised codegen: guest loads/stores without volatile (tools/pch_no_volatile.py); define
// REX_GUEST_VOLATILE as volatile to restore the generated behaviour.
#ifndef REX_GUEST_VOLATILE
#define REX_GUEST_VOLATILE
#endif
#define REX_LOAD_U8(x)''', 1)
    open(path, 'w', encoding='utf-8', newline='\n').write(text)
    print(f'{path}: {changed} macros without volatile')
    return True


def main():
    check = '--check' in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not paths:
        sys.exit(__doc__)
    ok = all([patch(p, check) for p in paths])
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
