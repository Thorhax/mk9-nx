#!/usr/bin/env python3
# proffn.py <elf> <rex_profile.log> [min_cpu_total] [topN]: per-thread self time by function, summed over the 10 s
# blocks whose CPU total >= min (the fight). Sample counts from "pc image+X xN" lines.
import bisect, collections, os, pickle, re, subprocess, sys
elf = os.path.abspath(sys.argv[1]); cache = elf + '.syms'
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
def fn(x):
    i = bisect.bisect_right(A, x) - 1
    return re.sub(r'\(.*', '', N[i])[:70] if i >= 0 else '?'
mn = float(sys.argv[3]) if len(sys.argv) > 3 else 250; top = int(sys.argv[4]) if len(sys.argv) > 4 else 25
tot = collections.defaultdict(collections.Counter); tsamp = collections.Counter(); tcpu = collections.defaultdict(list)
on = False; th = None; blocks = 0
for l in open(sys.argv[2], errors='replace'):
    if l.startswith('==== '):
        m = re.search(r'CPU total (\d+)%', l); on = bool(m) and int(m.group(1)) >= mn; th = None; blocks += on
        continue
    if not on: continue
    if l.startswith('== during the hitches'): th = None; on_h = True; continue
    m = re.match(r'-- thread \d+ "([^"]*)": CPU ([\d.]+)%', l)
    if m: th = m.group(1); tcpu[th].append(float(m.group(2))); continue
    if l.startswith('-- thread'): th = None; continue
    m = re.match(r'\s+[\d.]+%\s+pc image\+0x([0-9a-f]+) x(\d+)', l)
    if m and th:
        n = int(m.group(2)); tot[th][fn(int(m.group(1), 16))] += n; tsamp[th] += n
print(f'{blocks} blocks')
for th, c in sorted(tcpu.items(), key=lambda kv: -sum(kv[1]) / len(kv[1]))[:9]:
    print(f'\n## {th}: CPU {sum(c)/len(c):.1f}% avg, {tsamp[th]} listed pc samples')
    for f, n in tot[th].most_common(top):
        print(f'  {100*n/max(1,tsamp[th]):5.1f}%  {f}')
