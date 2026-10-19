#!/usr/bin/env python3
"""Checks a Switch build before it is packaged (from sasrt-nx's check_switch_build.py and nfsmw-nx's lessons).

  python tools/check_switch_build.py [--elf out/nx/sa2] [--nro out/nx/sa2.nro]

Checks (exit 1 if any fails):
  - _start is at address 0 (nfsmw-nx: a section-ordering file that moved the start-up code made the NRO unbootable);
  - the build names the optimised codegen flavour (the only one delivered);
  - the NRO links the nfsmw NVK driver of the SDK (its "Mesa x.y.z" string; the generic portlibs Mesa once got in);
  - branch-to-self loops in recompiled functions are only in the functions with a known guest halt (KNOWN_HALTS; a
    polling loop GCC turned into "b ." with non-volatile memory means a missing loop barrier);
  - no vector fmla/fmls in recompiled functions: the generated code has no vector fused multiply-add (VMX multiply-adds
    are a multiply and an add), so any would be contraction that leaked past -ffp-contract=off (e.g. at an LTO link).
    Scalar fmadd/fmsub/fnmadd/fnmsub come from the guest's own std::fma and are only counted.
"""
import argparse
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# The 15 functions whose original PowerPC code has a "b ." halt (the non-LTO optimised build, 2026-10-06). Optimisation
# may duplicate a halt instruction (the LTO build has two adjacent ones in sub_828DB988 and sub_8293FA08), so the check
# is by function: a branch-to-self anywhere else is a polling loop that lost its barrier.
KNOWN_HALTS = ['sub_8215CB08', 'sub_8216E7F8', 'sub_82176F68', 'sub_821B15E0', 'sub_822BA028', 'sub_8235DF50',
               'sub_82374728', 'sub_823BD350', 'sub_823E96D0', 'sub_82431930', 'sub_8252B7C8', 'sub_8252C0F8',
               'sub_828DB988', 'sub_829296A8', 'sub_8293FA08']
DEVKITA64 = next((d + '/devkitA64/bin/' for d in (os.environ.get('DEVKITPRO', ''), 'C:/devkitPro', '/opt/devkitpro')
                  if d and os.path.isdir(d + '/devkitA64/bin')), 'C:/devkitPro/devkitA64/bin/')
EXE = '.exe' if os.name == 'nt' else ''


def generated_bodies(names):
    """name -> generated C++ body, for the given recompiled functions (app/generated/*/ *_recomp.*.cpp)."""
    import glob
    out = {}
    head = re.compile(r'^DEFINE_REX_FUNC\((sub_[0-9A-F]{8})\) \{$', re.M)
    for path in glob.glob(os.path.join(ROOT, 'app', 'generated', '*', '*_recomp.*.cpp')):
        text = open(path, encoding='utf-8', errors='replace').read()
        for m in head.finditer(text):
            if m.group(1) in names:
                end = text.find('\n}\n', m.end())
                out[m.group(1)] = text[m.end():end if end >= 0 else len(text)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--elf', default=os.path.join(ROOT, 'out', 'nx', 'sa2'))
    ap.add_argument('--nro', default=os.path.join(ROOT, 'out', 'nx', 'sa2.nro'))
    ap.add_argument('--halts', default=','.join(KNOWN_HALTS),
                    help='the recompiled functions with a genuine guest halt (b . in the original PowerPC code)')
    ap.add_argument('--sdk-vulkan', default='C:/devkitPro/mesa-switch-main-nfsmw-source/mesa-switch-main/'
                    'mesa-unified-install/opt/devkitpro/portlibs/switch/lib/libvulkan.a')
    a = ap.parse_args()
    failures = []

    nm = subprocess.run([DEVKITA64 + 'aarch64-none-elf-nm' + EXE, a.elf], capture_output=True, text=True).stdout
    start = next((int(l.split()[0], 16) for l in nm.splitlines() if l.endswith(' _start')), None)
    print(f'_start at {start:#x}' if start is not None else '_start not found')
    if start != 0:
        failures.append('_start is not at address 0')

    nro = open(a.nro, 'rb').read()
    if b'optimised codegen' not in nro:
        failures.append('the NRO does not name the optimised codegen flavour')
    else:
        print('flavour: optimised codegen')
    mesa = sorted(set(re.findall(rb'Mesa \d+\.\d+\.\d+ \(git-[0-9a-f]+\)', nro)))
    sdk = sorted(set(re.findall(rb'Mesa \d+\.\d+\.\d+ \(git-[0-9a-f]+\)', open(a.sdk_vulkan, 'rb').read())))
    print('Mesa in the NRO:', [m.decode() for m in mesa], '| in the SDK:', [m.decode() for m in sdk])
    if not mesa or mesa != sdk:
        failures.append('the NRO does not carry the SDK driver\'s Mesa string')

    # One streaming pass over the disassembly: per function, b-to-self and fused multiply-adds.
    proc = subprocess.Popen([DEVKITA64 + 'aarch64-none-elf-objdump' + EXE, '-d', '--no-show-raw-insn', a.elf],
                            stdout=subprocess.PIPE, text=True, errors='replace', bufsize=1 << 20)
    func = None
    self_loops = []
    vector_fma = collections.Counter()
    scalar_fma = 0
    head = re.compile(r'^[0-9a-f]+ <([^>]+)>:$')
    ins = re.compile(r'^\s+([0-9a-f]+):\s+(\S+)\s*(.*)$')
    for line in proc.stdout:
        m = head.match(line)
        if m:
            func = m.group(1)
            continue
        m = ins.match(line)
        if not m or not func:
            continue
        addr, op, args = m.groups()
        guest = func.startswith(('sub_8', '__imp__sub_8'))
        if op == 'b' and args.split(' ')[0] == addr:
            if guest:
                self_loops.append(func)
        elif op in ('fmla', 'fmls') and guest:
            vector_fma[func] += 1
        elif op in ('fmadd', 'fmsub', 'fnmadd', 'fnmsub') and guest:
            scalar_fma += 1
    proc.wait()
    known = set(a.halts.split(','))
    functions = {f.replace('__imp__', '') for f in self_loops}
    unknown = sorted(functions - known)
    # Round 7: GCC may inline a known halt into a caller (non_volatile_as_local made the callees smaller: the halts of
    # sub_828DB988 and sub_8293FA08 then appear in sub_828D71E8 and sub_8293FE28). A function whose generated body
    # calls a known-halt function directly is accepted, and named.
    inlined = []
    if unknown:
        bodies = generated_bodies(set(unknown))
        for f in list(unknown):
            calls = set(re.findall(r'\b(?:__imp__)?(sub_[0-9A-F]{8})\(ctx, base\)', bodies.get(f, '')))
            if calls & known:
                inlined.append(f'{f} (inlines {", ".join(sorted(calls & known))})')
                unknown.remove(f)
    print(f'branch-to-self in recompiled functions: {len(self_loops)} instructions in {len(functions)} functions, '
          f'{len(functions & known)} of them known guest halts'
          + (f'; {len(inlined)} more calling one that GCC inlined: {"; ".join(inlined)}' if inlined else ''))
    if unknown:
        failures.append('branch-to-self outside the known guest halts (a polling loop without its barrier?): '
                        + ', '.join(unknown[:20]))
    print(f'fused multiply-adds in recompiled functions: scalar {scalar_fma} (the guest\'s own), vector '
          f'{sum(vector_fma.values())} in {len(vector_fma)} functions (must be 0)')
    if vector_fma:
        failures.append('vector fmla/fmls in recompiled code (contraction leaked): '
                        + ', '.join(f'{k} x{v}' for k, v in vector_fma.most_common(10)))
    if failures:
        print('\nFAILED:\n  - ' + '\n  - '.join(failures))
        sys.exit(1)
    print('\nall checks passed')


if __name__ == '__main__':
    main()
