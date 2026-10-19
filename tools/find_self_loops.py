#!/usr/bin/env python3
"""Infinite loops GCC made out of guest polling loops (Switch ELF, AArch64).

A loop that only reads guest memory and branches back (`while (*flag == 0) {}`) is a real wait on the Xbox 360: another
thread or the GPU changes the word. With guest loads that are not volatile (tools/perf_codegen_post.py) the compiler
may hoist the load out of the loop, and the loop becomes `b .` (a branch to itself) or a conditional branch to itself:
the thread spins at 100 % forever. This lists every branch-to-self instruction in the ELF's code with the symbol it is
in, so two builds can be compared.

  python tools/find_self_loops.py out/nx/sa2.elf [another.elf]   (from futari-nx)
"""
import bisect
import struct
import subprocess
import sys

NM = r'C:\devkitPro\devkitA64\bin\aarch64-none-elf-nm.exe'


def sections(path):
    data = open(path, 'rb').read()
    assert data[:4] == b'\x7fELF' and data[4] == 2
    shoff, = struct.unpack_from('<Q', data, 0x28)
    shentsize, shnum = struct.unpack_from('<HH', data, 0x3A)
    out = []
    for i in range(shnum):
        name, type_, flags, addr, offset, size = struct.unpack_from('<IIQQQQ', data, shoff + i * shentsize)
        if type_ == 1 and flags & 0x4 and size:  # PROGBITS, executable
            out.append((addr, data[offset:offset + size]))
    return out


def symbols(path):
    text = subprocess.run([NM, '-n', path], capture_output=True, text=True).stdout
    addrs, names = [], []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[1] in 'tTwW':
            addrs.append(int(parts[0], 16))
            names.append(parts[2])
    return addrs, names


def self_branches(code):
    for i in range(0, len(code) - 3, 4):
        w, = struct.unpack_from('<I', code, i)
        if w == 0x14000000:                                   # b .
            yield i, 'b .'
        elif (w & 0xFF00001F) in (0x54000000 | c for c in range(16)) and ((w >> 5) & 0x7FFFF) == 0:
            yield i, 'b.cond .'                               # b.cond .
        elif (w & 0x7E000000) == 0x34000000 and ((w >> 5) & 0x7FFFF) == 0:
            yield i, 'cbz/cbnz .'
        elif (w & 0x7E000000) == 0x36000000 and ((w >> 5) & 0x3FFF) == 0:
            yield i, 'tbz/tbnz .'


def scan(path):
    addrs, names = symbols(path)
    found = {}
    for base, code in sections(path):
        for off, kind in self_branches(code):
            a = base + off
            k = bisect.bisect_right(addrs, a) - 1
            name = names[k] if k >= 0 else '?'
            found.setdefault(name, []).append((a, kind))
    return found


def main():
    results = {p: scan(p) for p in sys.argv[1:]}
    for p, found in results.items():
        guest = {n: v for n, v in found.items() if 'sub_' in n}
        print(f'== {p}: {sum(len(v) for v in found.values())} self-branches, {len(guest)} in recompiled functions')
        for n, v in sorted(guest.items()):
            print(f'   {n}: ' + ', '.join(f'{a:#x} {k}' for a, k in v))


if __name__ == '__main__':
    main()
