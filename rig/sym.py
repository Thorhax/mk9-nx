#!/usr/bin/env python3
# sym.py <elf> <file|->: replaces every image+0xOFF in the text with function+0xoff (nm of the ELF, cached).
import bisect, os, re, subprocess, sys, pickle
elf = os.path.abspath(sys.argv[1])
cache = elf + '.syms'
if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(elf):
    A, N = pickle.load(open(cache, 'rb'))
else:
    nm = subprocess.run(['docker', 'run', '--rm', '-v', f'{os.path.dirname(elf)}:{os.path.dirname(elf)}', 'devkitpro/devkita64',
                         '/opt/devkitpro/devkitA64/bin/aarch64-none-elf-nm', '-n', '-C', '--defined-only', elf],
                        capture_output=True, text=True).stdout
    A, N = [], []
    for l in nm.splitlines():
        p = l.split(None, 2)
        if len(p) == 3 and p[1] in 'tTwW':
            A.append(int(p[0], 16)); N.append(p[2])
    pickle.dump((A, N), open(cache, 'wb'))
def s(m):
    x = int(m.group(1), 16); i = bisect.bisect_right(A, x) - 1
    n = N[i] if i >= 0 else '?'
    n = re.sub(r'\((?!anonymous namespace\)).*', '', n)[:60]
    return f'{n}+{x - A[i]:#x}'
src = sys.stdin if sys.argv[2] == '-' else open(sys.argv[2])
for l in src:
    sys.stdout.write(re.sub(r'image\+0x([0-9a-f]+)', s, l))
