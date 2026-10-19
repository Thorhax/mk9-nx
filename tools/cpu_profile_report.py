#!/usr/bin/env python3
"""Ranks the hottest functions of a PC CPU profile (logs/perfil_pc.csv, written by app/src/profiler/sa2_cpu_profile_pc
with sa2_cpu_profile_from_s / sa2_cpu_profile_duration_s).

For each busy thread it prints:
  - self time: the function the program counter was in;
  - guest time: the innermost recompiled game function (sub_XXXXXXXX) on the sampled stack, so time spent in an SDK
    helper or the C runtime called from game code is charged to the game function that called it.

  python tools/cpu_profile_report.py out/pc/logs/perfil_pc.csv [--exe out/pc/sa2.exe] [--top 30] [--thread Main]

Addresses are symbolized with llvm-symbolizer (the project's LLVM, or SA2_LLVM) against the executable and its PDB.
"""
import argparse
import collections
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_symbolizer():
    candidates = [os.environ.get('SA2_LLVM') and os.path.join(os.environ['SA2_LLVM'], 'bin', 'llvm-symbolizer.exe'),
                  os.path.join(ROOT, '..', '_tools', 'llvm', 'bin', 'llvm-symbolizer.exe'),
                  shutil.which('llvm-symbolizer')]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    sys.exit('llvm-symbolizer not found (set SA2_LLVM)')


def image_base(exe):
    # PE optional header ImageBase (PE32+).
    with open(exe, 'rb') as f:
        data = f.read(4096)
    pe = int.from_bytes(data[0x3C:0x40], 'little')
    return int.from_bytes(data[pe + 24 + 24:pe + 24 + 32], 'little')


def symbolize(exe, rvas):
    base = image_base(exe)
    names = {}
    rvas = sorted(rvas)
    tool = find_symbolizer()
    for i in range(0, len(rvas), 2000):
        chunk = rvas[i:i + 2000]
        proc = subprocess.run([tool, '--functions=linkage', '--no-inlines', '--output-style=GNU', '--obj', exe],
                              input='\n'.join(hex(base + r) for r in chunk), capture_output=True, text=True)
        lines = proc.stdout.splitlines()
        # GNU style: function line, then file:line, per address.
        for j, rva in enumerate(chunk):
            names[rva] = lines[2 * j] if 2 * j < len(lines) else '??'
    return names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('csv')
    ap.add_argument('--exe', default=os.path.join(ROOT, 'out', 'pc', 'sa2.exe'))
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--thread', action='append', help='only threads whose name contains this (repeatable)')
    a = ap.parse_args()

    modules, threads, samples = {}, {}, []
    for line in open(a.csv, encoding='utf-8', errors='ignore'):
        line = line.strip()
        if line.startswith('#modulo;'):
            _, idx, name, base, size = line.split(';')[:5]
            modules[int(idx)] = name
        elif line.startswith('#hilo;'):
            parts = line.split(';')
            threads[int(parts[1])] = (parts[3], int(parts[4]), float(parts[5]))
        elif line and not line.startswith('#'):
            parts = line.split(';')
            frames = [(int(parts[i]), int(parts[i + 1], 16)) for i in range(1, len(parts) - 1, 2)]
            samples.append((int(parts[0]), frames))
    exe_module = next(i for i, n in modules.items() if n.lower() == os.path.basename(a.exe).lower())

    rvas = {rva for _, frames in samples for m, rva in frames if m == exe_module}
    names = symbolize(a.exe, rvas)

    def label(m, rva):
        if m == exe_module:
            return names.get(rva, '??')
        return f'[{modules.get(m, m)}]'

    per_thread = collections.defaultdict(list)
    for t, frames in samples:
        per_thread[t].append(frames)
    for t, frame_lists in sorted(per_thread.items(), key=lambda kv: -len(kv[1])):
        name, count, cpu = threads.get(t, (str(t), len(frame_lists), 0))
        if a.thread and not any(s.lower() in name.lower() for s in a.thread):
            continue
        n = len(frame_lists)
        if n < 100:
            continue
        self_counts = collections.Counter(label(*frames[0]) for frames in frame_lists if frames)
        guest_counts = collections.Counter()
        for frames in frame_lists:
            guest = next((label(m, r) for m, r in frames if m == exe_module and 'sub_8' in label(m, r)), None)
            guest_counts[guest or '(no game function on the stack)'] += 1
        print(f'== {name}: {n} samples, CPU {cpu:.2f} cores')
        print('   self (program counter):')
        for fn, c in self_counts.most_common(a.top):
            print(f'     {100.0 * c / n:5.1f} %  {fn}')
        print('   innermost game function on the stack:')
        for fn, c in guest_counts.most_common(a.top):
            print(f'     {100.0 * c / n:5.1f} %  {fn}')
        print()


if __name__ == '__main__':
    main()
