#!/usr/bin/env python3
"""Ranks the hottest functions of a console CPU profile (logs/rex/rex_profile.log, written by sdk/src/ui/switch_perf.cpp
with switch_profile_stacks = true).

The log has one block per 10 s interval ("==== 10.0 s | mode: normal | game 32.4 fps | ...", then "clocks: CPU
1020.0 MHz ..."), and in it one section per thread with:
  - "pc image+0xOFF xN": the raw count of samples whose program counter was there (running samples only);
  - "stack image+0x.. <- image+0x.. ...": the sampled call stacks (innermost first), as a percentage of all samples.

The report keeps only the intervals that match the filters (default: normal mode, CPU 1020 MHz, under 58 fps, i.e. the
CPU-bound gameplay at stock clocks), then prints per thread:
  - self: the function each sampled pc was in;
  - guest: the innermost recompiled game function (sub_XXXXXXXX) on each sampled stack, so time in an SDK helper or the
    C runtime is charged to the game function that called it (running stacks only; the kernel-wait stacks are dropped).

  python tools/switch_profile_report.py <rex_profile.log> --elf out/nx/ab/optimised/sa2.elf [--top 40]
         [--mhz 1020] [--mem 1600] [--max-fps 58] [--thread Main]

The ELF must be the one the NRO was built from (out/nx/ab/<flavour>/sa2.elf); image offsets are ELF addresses.
"""
import argparse
import bisect
import collections
import os
import re
import subprocess
import sys

DEVKITA64 = next((d + '/devkitA64/bin/' for d in (os.environ.get('DEVKITPRO', ''), 'C:/devkitPro', '/opt/devkitpro')
                  if d and os.path.isdir(d + '/devkitA64/bin')), 'C:/devkitPro/devkitA64/bin/')


def load_symbols(elf):
    nm = DEVKITA64 + 'aarch64-none-elf-nm' + ('.exe' if os.name == 'nt' else '')
    out = subprocess.run([nm, '-n', '--defined-only', '-S', elf], capture_output=True, text=True, check=True).stdout
    addrs, names = [], []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[2] in 'tTwW':
            addrs.append(int(parts[0], 16))
            names.append(parts[3])
        elif len(parts) == 3 and parts[1] in 'tTwW':
            addrs.append(int(parts[0], 16))
            names.append(parts[2])
    return addrs, names


def demangle(names):
    filt = DEVKITA64 + 'aarch64-none-elf-c++filt' + ('.exe' if os.name == 'nt' else '')
    if not os.path.exists(filt):
        return {n: n for n in names}
    names = sorted(set(names))
    out = subprocess.run([filt], input='\n'.join(names), capture_output=True, text=True).stdout.splitlines()
    return dict(zip(names, out)) if len(out) == len(names) else {n: n for n in names}


GUEST = re.compile(r'^(?:__imp__)?sub_([0-9A-F]{8})$')
OFF = re.compile(r'image\+0x([0-9a-f]+)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('log')
    ap.add_argument('--elf', required=True)
    ap.add_argument('--top', type=int, default=40)
    ap.add_argument('--mhz', type=float, default=1020.0, help='keep intervals at this CPU clock (0 = any)')
    ap.add_argument('--mem', type=float, default=0.0, help='keep intervals at this memory clock (0 = any)')
    ap.add_argument('--max-fps', type=float, default=58.0, help='keep intervals under this frame rate')
    ap.add_argument('--min-fps', type=float, default=5.0)
    ap.add_argument('--thread', action='append', help='thread name substring (repeatable; default: all)')
    ap.add_argument('--guest-list', help='write the guest ranking (sub_ addresses) to this file')
    a = ap.parse_args()

    addrs, names = load_symbols(a.elf)

    def sym(off):
        i = bisect.bisect_right(addrs, off) - 1
        return names[i] if i >= 0 else '?'

    self_cnt = collections.defaultdict(collections.Counter)   # thread -> name -> samples
    guest_cnt = collections.defaultdict(collections.Counter)  # thread -> guest fn -> samples (innermost on stack)
    incl_cnt = collections.defaultdict(collections.Counter)   # thread -> guest fn -> samples (anywhere on stack)
    run_samples = collections.Counter()
    kept, skipped = [], 0

    keep = False
    fps = mhz = 0.0
    thread = None
    samples = 0
    pending_header = None
    with open(a.log, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('==== '):
                m = re.search(r'mode: (\w+) \| game ([\d.]+) fps', line)
                pending_header = (m.group(1), float(m.group(2))) if m else ('?', 0.0)
                keep = False
                thread = None
                continue
            if pending_header and 'clocks: CPU' in line:
                mode, fps = pending_header
                mhz = float(re.search(r'clocks: CPU ([\d.]+) MHz', line).group(1))
                mm = re.search(r'memory ([\d.]+) MHz', line)
                mem = float(mm.group(1)) if mm else 0.0
                pending_header = None
                keep = (mode == 'normal' and a.min_fps <= fps < a.max_fps and (a.mhz == 0 or abs(mhz - a.mhz) < 1) and
                        (a.mem == 0 or abs(mem - a.mem) < 1))
                if keep:
                    kept.append((fps, mhz))
                else:
                    skipped += 1
                continue
            if not keep:
                continue
            if line.startswith('== '):  # the hitch summary inside the interval: its thread lines have no id
                thread = None
                continue
            m = re.match(r'-- thread \d+ "(.*?)": CPU', line)
            if m:
                thread = m.group(1)
                if a.thread and not any(t in thread for t in a.thread):
                    thread = None
                continue
            if thread is None:
                continue
            m = re.match(r'\s+(\d+) samples, (\d+)% waiting', line)
            if m:
                samples = int(m.group(1))
                continue
            m = re.match(r'\s+[\d.]+%\s+pc image\+0x([0-9a-f]+) x(\d+)', line)
            if m:
                n = int(m.group(2))
                self_cnt[thread][sym(int(m.group(1), 16))] += n
                run_samples[thread] += n
                continue
            m = re.match(r'\s+([\d.]+)%\s+stack (.*)$', line)
            if m:
                frames = [sym(int(x, 16)) for x in OFF.findall(m.group(2))]
                if not frames or frames[0].startswith('__syscall') or 'svc' in frames[0].lower():
                    continue
                n = float(m.group(1)) * samples / 100.0
                inner = next((fn for fn in frames if GUEST.match(fn)), None)
                if inner:
                    guest_cnt[thread][GUEST.sub(r'sub_\1', inner)] += n
                for fn in {GUEST.sub(r'sub_\1', fn) for fn in frames if GUEST.match(fn)}:
                    incl_cnt[thread][fn] += n

    print(f'{len(kept)} intervals kept ({skipped} skipped): fps ' +
          ', '.join(f'{fps:.1f}' for fps, _ in kept))
    pretty = demangle([n for c in self_cnt.values() for n in c])
    order = sorted(run_samples, key=lambda t: -run_samples[t])
    all_guest = collections.Counter()
    for t in order:
        tot = run_samples[t]
        if tot < 20:
            continue
        print(f'\n=== thread "{t}": {tot} running samples (pc)')
        print('  -- self (pc):')
        for name, n in self_cnt[t].most_common(a.top):
            print(f'    {100.0 * n / tot:5.1f}%  {n:5d}  {pretty.get(name, name)[:150]}')
        gt = sum(guest_cnt[t].values())
        if gt:
            print(f'  -- innermost guest function on running stacks ({gt:.0f} samples):')
            for name, n in guest_cnt[t].most_common(a.top):
                print(f'    {100.0 * n / gt:5.1f}%  {n:7.1f}  {name}   (inclusive {100.0 * incl_cnt[t][name] / gt:5.1f}%)')
        for name, n in guest_cnt[t].items():
            all_guest[name] += n
    if a.guest_list:
        with open(a.guest_list, 'w') as f:
            for name, n in all_guest.most_common():
                f.write(f'{name} {n:.1f}\n')


if __name__ == '__main__':
    main()
