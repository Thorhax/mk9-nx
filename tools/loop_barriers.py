#!/usr/bin/env python3
"""Compiler barriers at every loop header of the generated code (post-codegen step of the "optimised" codegen flavour,
REQUIRED together with tools/pch_no_volatile.py; from futari-nx tools/perf_codegen_post.py pass 3).

With guest loads that are not volatile, GCC -O3 (the Switch build) hoists the load of a guest polling loop
(`while (device[x] != 0) {}`: a real wait on the Xbox 360, another thread or the GPU changes the word) out of the loop
and turns it into a branch to itself: the thread spins forever. clang on the PC did not, so PC runs looked fine. Found
by futari-nx on its console; tools/find_self_loops.py counts these in an ELF (SA2 optimised build without barriers:
40 in recompiled functions, e.g. sub_8255A760 in Direct3D).

A loop header is a label `loc_X:` that a later `goto loc_X;` of the same function jumps back to. The line after it gets
REX_LOOP_BARRIER(), a compiler memory barrier (defined in the pch): no guest load is reused across iterations, also
inside inlined callees; straight-line code keeps the plain loads. Idempotent.

  python tools/loop_barriers.py app/generated/default [app/generated/data_dll ...] [--check]
"""
import os
import re
import sys

LABEL = re.compile(r'^(loc_[0-9A-F]{8}):\s*$')
GOTO = re.compile(r'goto (loc_[0-9A-F]{8});')
BARRIER = '\tREX_LOOP_BARRIER();'
MARKER = '// sa2-nx optimised codegen: loop barrier (tools/loop_barriers.py)'
MACRO = (MARKER + '\n#ifndef REX_LOOP_BARRIER\n'
         '#define REX_LOOP_BARRIER() __asm__ __volatile__("" ::: "memory")\n#endif\n')


def process(folder, check):
    headers_total = files = 0
    for f in sorted(os.listdir(folder)):
        if not re.match(r'\w+_recomp\.\d+\.cpp$', f):
            continue
        path = os.path.join(folder, f)
        lines = open(path, encoding='utf-8').read().split('\n')
        headers, seen = set(), set()
        for line in lines:
            if line.startswith('DEFINE_REX_FUNC('):
                seen = set()
                continue
            m = LABEL.match(line)
            if m:
                seen.add(m.group(1))
                continue
            for target in GOTO.findall(line):
                if target in seen:
                    headers.add(target)
        out = []
        for i, line in enumerate(lines):
            out.append(line)
            m = LABEL.match(line)
            if m and m.group(1) in headers and (i + 1 >= len(lines) or lines[i + 1] != BARRIER):
                out.append(BARRIER)
        headers_total += len(headers)
        if out != lines:
            files += 1
            if not check:
                open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    pch = next((os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('_pch.h')), None)
    if not pch:
        sys.exit(f'loop barriers: no *_pch.h in {folder}')
    text = open(pch, encoding='utf-8').read()
    pch_changed = False
    if MARKER not in text:
        if '#define REX_LOAD_U8(x)' not in text:
            sys.exit(f'loop barriers: no REX_LOAD_U8 in {pch}: the generated pch changed')
        text = text.replace('#define REX_LOAD_U8(x)', MACRO + '#define REX_LOAD_U8(x)', 1)
        pch_changed = True
        if not check:
            open(pch, 'w', encoding='utf-8', newline='\n').write(text)
    print(f'{folder}: {headers_total} loop headers, {files} files {"to change" if check else "changed"}'
          f'{", pch macro added" if pch_changed else ""}')
    return files == 0 and not pch_changed


def main():
    check = '--check' in sys.argv
    folders = [a for a in sys.argv[1:] if not a.startswith('--')]
    if not folders:
        sys.exit(__doc__)
    ok = all([process(f, check) for f in folders])
    sys.exit(0 if ok or not check else 1)


if __name__ == '__main__':
    main()
