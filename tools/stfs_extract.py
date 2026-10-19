"""Extracts an Xbox 360 STFS package (LIVE/PIRS/CON) into a folder.

SA2 ships on XBLA as a LIVE package (content type 0x000D0000, the game) and a
second LIVE package (content type 0x00000002, the "Battle" DLC). Both are
read-only STFS volumes. This only reads; it never writes into the package.

Usage: python tools/stfs_extract.py <package> <out_dir> [--list]
"""
import argparse
import os
import struct
import sys

BLOCK = 0x1000
PER_TABLE = 0xAA  # 170 hash entries (blocks) per hash table
LEVEL_BASE = (PER_TABLE, PER_TABLE ** 2, PER_TABLE ** 3)


def u24le(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16)


def u24be(b, o):
    return (b[o] << 16) | (b[o + 1] << 8) | b[o + 2]


class Stfs:
    def __init__(self, path):
        self.f = open(path, 'rb')
        hdr = self.f.read(0xA000)
        magic = hdr[:4]
        if magic not in (b'LIVE', b'PIRS', b'CON '):
            raise ValueError(f'not an STFS package (magic {magic!r})')
        self.header_size = struct.unpack_from('>I', hdr, 0x340)[0]
        self.content_type = struct.unpack_from('>I', hdr, 0x344)[0]
        self.title_id = struct.unpack_from('>I', hdr, 0x360)[0]
        if hdr[0x3A9] != 0:
            raise ValueError('volume is SVOD, not STFS')
        vd = 0x379
        flags = hdr[vd + 2]
        self.read_only = bool(flags & 1)
        # A read-only volume keeps one copy of each hash table, a writable one two.
        self.tables_per_level = 1 if self.read_only else 2
        self.ft_blocks = struct.unpack_from('<H', hdr, vd + 3)[0]
        self.ft_start = u24le(hdr, vd + 5)
        self.alloc_blocks = struct.unpack_from('>I', hdr, vd + 0x1C)[0]
        self.data_base = (self.header_size + BLOCK - 1) & ~(BLOCK - 1)
        name_raw = hdr[0x411:0x411 + 0x80]
        self.display_name = name_raw.decode('utf-16-be', 'replace').split('\0')[0]

    # Data block number -> physical block number, skipping the interleaved hash tables
    # (same arithmetic as Xenia's BlockToOffsetSTFS).
    def physical(self, block):
        out = block
        for base in LEVEL_BASE:
            out += ((block + base) // base) * self.tables_per_level
            if block < base:
                break
        return out

    def offset(self, block):
        return self.data_base + self.physical(block) * BLOCK

    def read_block(self, block):
        self.f.seek(self.offset(block))
        return self.f.read(BLOCK)

    def next_block(self, block):
        # The level-0 hash table of a group sits right before the group's first data block.
        group_first = (block // PER_TABLE) * PER_TABLE
        table_phys = self.physical(group_first) - self.tables_per_level
        self.f.seek(self.data_base + table_phys * BLOCK + (block % PER_TABLE) * 0x18)
        entry = self.f.read(0x18)
        return u24be(entry, 0x15)

    def chain(self, start, count, consecutive):
        block = start
        for _ in range(count):
            yield block
            block = block + 1 if consecutive else self.next_block(block)

    def entries(self):
        raw = b''.join(self.read_block(b) for b in
                       self.chain(self.ft_start, self.ft_blocks, False))
        out = []
        for i in range(0, len(raw), 0x40):
            e = raw[i:i + 0x40]
            nlen = e[0x28] & 0x3F
            if nlen == 0:
                continue
            out.append({
                'index': i // 0x40,
                'name': e[:nlen].decode('ascii', 'replace'),
                'dir': bool(e[0x28] & 0x80),
                'consecutive': bool(e[0x28] & 0x40),
                'blocks': u24le(e, 0x29),
                'start': u24le(e, 0x2F),
                'parent': struct.unpack_from('>h', e, 0x32)[0],
                'size': struct.unpack_from('>I', e, 0x34)[0],
            })
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('package')
    ap.add_argument('out_dir')
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    p = Stfs(a.package)
    print(f'{p.display_name!r} title {p.title_id:08X} content {p.content_type:08X} '
          f'read_only={p.read_only} blocks={p.alloc_blocks}')
    ents = p.entries()
    by_index = {e['index']: e for e in ents}

    def path_of(e):
        parts = [e['name']]
        while e['parent'] != -1:
            e = by_index[e['parent']]
            parts.append(e['name'])
        return os.path.join(*reversed(parts))

    total = 0
    for e in ents:
        rel = path_of(e)
        if a.list:
            print(f"{'<dir>' if e['dir'] else e['size']:>12}  {rel}")
            continue
        dst = os.path.join(a.out_dir, rel)
        if e['dir']:
            os.makedirs(dst, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        left = e['size']
        with open(dst, 'wb') as out:
            for b in p.chain(e['start'], e['blocks'], e['consecutive']):
                if left <= 0:
                    break
                data = p.read_block(b)[:min(BLOCK, left)]
                out.write(data)
                left -= len(data)
        if left:
            sys.exit(f'short read on {rel}: {left} bytes missing')
        total += e['size']
    if not a.list:
        print(f'{len(ents)} entries, {total / 2**20:.1f} MiB written to {a.out_dir}')


if __name__ == '__main__':
    main()
