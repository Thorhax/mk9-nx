"""Address -> source line for the Switch ELF from an `llvm-dwarfdump --debug-line` dump, ignoring the line sequences
the linker left at address 0 (debug info of discarded sections: they overlap the hot functions placed first in .text
and make addr2line/llvm-symbolizer answer with the wrong file).
   python nxline_map.py <dump.txt> <hex address> [...]          (or import: build(dump) -> lookup(addr))"""
import bisect, re, sys

ROW = re.compile(r'^0x([0-9a-f]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s+\d+\s+\d+\s*(.*)$')


def build(path):
    files = {}
    seqs = []  # (start, end, rows[(addr, file, line)])
    cur = []
    file_names = {}
    include_dirs = {}
    for line in open(path, encoding='utf-8', errors='replace'):
        if line.startswith('debug_line['):
            file_names = {}
            include_dirs = {}
            cur = []
            continue
        m = re.match(r'^include_directories\[\s*(\d+)\] = "(.*)"', line)
        if m:
            include_dirs[int(m.group(1))] = m.group(2)
            continue
        m = re.match(r'^file_names\[\s*(\d+)\]:', line)
        if m:
            last_file = int(m.group(1))
            continue
        m = re.match(r'^\s+name: "(.*)"', line)
        if m:
            file_names[last_file] = m.group(1)
            continue
        m = ROW.match(line)
        if m:
            addr, ln, col, fidx = int(m.group(1), 16), int(m.group(2)), int(m.group(3)), int(m.group(4))
            flags = m.group(5)
            cur.append((addr, file_names.get(fidx, '?'), ln))
            if 'end_sequence' in flags:
                if cur and cur[0][0] >= 0x1000:  # a sequence at 0 belongs to a discarded section
                    seqs.append((cur[0][0], cur[-1][0], cur))
                cur = []
    seqs.sort()
    starts = [s[0] for s in seqs]
    return seqs, starts


def lookup(index, addr):
    seqs, starts = index
    i = bisect.bisect_right(starts, addr) - 1
    while i >= 0:
        start, end, rows = seqs[i]
        if start <= addr < end:
            addrs = [r[0] for r in rows]
            j = bisect.bisect_right(addrs, addr) - 1
            return rows[j][1], rows[j][2]
        if end < addr - 0x100000:
            break
        i -= 1
    return '?', 0


if __name__ == '__main__':
    idx = build(sys.argv[1])
    for a in sys.argv[2:]:
        f, l = lookup(idx, int(a, 16))
        print(a, f, l)
