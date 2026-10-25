#!/usr/bin/env python3
"""The trap helper of the generated pch out of line and cold (post-codegen step of the "optimised" codegen flavour;
docs/enhancements.md, "The second CPU round of 2026-10-12"). Works on any ReXGlue port.

The game's compiler put a trap before every integer division (`twllei rX, 0`: divisor zero) and at its assertions:
1,219 sites in SA2, each `if (<condition>) ppc_trap(ctx, base, <type>);`. ppc_trap is an inline function of the pch
that logs a warning, so the compiler may put that logging code into the body of every function that divides, and it
knows nothing about how rarely the branch is taken. As a cold function that is never inlined there is one copy of
it, the branch to it is laid out as unlikely, and the function's hot path is the division alone. The same function
runs when a trap fires.

With the Switch compiler three recompiled files changed by -0.37 %, -0.81 % and +0.25 % instructions (4 to 8 sites
each: the rest is the inliner deciding differently with a smaller unit), so this is a layout change, not a size one.
-DREX_TRAP_ATTR= (compile time) gives the inline function back.

  python tools/pch_cold_traps.py app/generated/default/sa2_pch.h [more pch files] [--check]
Idempotent.
"""
import sys

OLD = 'inline void ppc_trap(PPCContext& ctx, u8* base, u16 trap_type) {'
MARKER = '// rexglue ports: the trap helper out of line and cold (tools/pch_cold_traps.py)'
NEW = (MARKER + '\n#ifndef REX_TRAP_ATTR\n#define REX_TRAP_ATTR [[gnu::cold, gnu::noinline]]\n#endif\n'
       'REX_TRAP_ATTR ' + OLD)


def main():
    check = '--check' in sys.argv
    paths = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not paths:
        sys.exit(__doc__)
    for path in paths:
        text = open(path, encoding='utf-8').read()
        if MARKER in text:
            print('pch_cold_traps: already in %s' % path)
            continue
        if text.count(OLD) != 1:
            sys.exit('ERROR (post-codegen patch): pch_cold_traps: no single ppc_trap definition in %s: the generated '
                     'pch changed' % path)
        if not check:
            open(path, 'w', encoding='utf-8', newline='\n').write(text.replace(OLD, NEW, 1))
        print('pch_cold_traps: ppc_trap is cold and out of line in %s%s' % (path, ' (check only)' if check else ''))


if __name__ == '__main__':
    main()
