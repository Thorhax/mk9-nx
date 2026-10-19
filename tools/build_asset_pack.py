#!/usr/bin/env python3
"""Builds the asset cache of a game data folder on the PC: <folder>.nxpack + <folder>.nxidx next to it, the format
of the SDK's PackDevice (sdk/include/rex/filesystem/devices/pack_device.h), which is Fruit Ninja NX's asset pack
version 2 byte for byte. The console then mounts the pack instead of opening thousands of files on the SD card, and
does not have to build it on its first boot.

  python tools/build_asset_pack.py assets/game_root [--out-dir DIR] [--if-stale] [--verify]

Layout: PackHeader (40 bytes) then every file 16-byte aligned, in path order (strcmp of the UTF-8 '/'-separated
relative path); the index is IndexHeader (56 bytes), DiskEntry[count] (24 bytes each) and the NUL-terminated paths.
Checksums are 64-bit FNV-1a. --verify rereads a pack and checks the data checksum (asset_pack.c only compares the
headers at mount time).
"""
import argparse
import os
import struct
import sys
import time

FNV_BASIS = 1469598103934665603
FNV_PRIME = 1099511628211
MASK = (1 << 64) - 1
CHUNK = 1 << 20


def _fnv_python(h, data):
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & MASK
    return h


def _load_fnv_c():
    """tools/fnv64.c compiled once into out/tools (ctypes); None if no C compiler is found."""
    import ctypes
    import shutil
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    lib = os.path.join(root, 'out', 'tools', 'fnv64' + ('.dll' if os.name == 'nt' else '.so'))
    if not os.path.exists(lib):
        os.makedirs(os.path.dirname(lib), exist_ok=True)
        candidates = [os.environ.get('SA2_LLVM') and os.path.join(os.environ['SA2_LLVM'], 'bin', 'clang.exe'),
                      os.path.join(root, '..', '_tools', 'llvm', 'bin', 'clang.exe'),
                      shutil.which('clang'), shutil.which('gcc'), shutil.which('cc')]
        for cc in [c for c in candidates if c and os.path.exists(c)]:
            if subprocess.run([cc, '-O2', '-shared', '-o', lib, os.path.join(here, 'fnv64.c')] +
                              ([] if os.name == 'nt' else ['-fPIC']), capture_output=True).returncode == 0:
                break
        else:
            return None
    dll = ctypes.CDLL(lib)
    dll.fnv64.restype = ctypes.c_uint64
    dll.fnv64.argtypes = [ctypes.c_uint64, ctypes.c_char_p, ctypes.c_size_t]
    return lambda h, data: dll.fnv64(h, bytes(data), len(data))


_FNV_C = _load_fnv_c()
if _FNV_C is None:
    print('warning: no C compiler for tools/fnv64.c: hashing in Python (slow)', file=sys.stderr)


def fnv(h, data):
    return _FNV_C(h, data) if _FNV_C else _fnv_python(h, data)


fnv_fast = fnv


def valid_relative(path):
    if not path or path.startswith('/') or path.endswith('/'):
        return False
    for part in path.split('/'):
        if part in ('', '.', '..') or '\\' in part or ':' in part:
            return False
    return True


def collect(folder):
    items = []
    for root, _, files in os.walk(folder):
        for name in files:
            full = os.path.join(root, name)
            rel = os.path.relpath(full, folder).replace(os.sep, '/')
            if not valid_relative(rel):
                sys.exit(f'invalid path for the pack: {rel}')
            items.append([rel.encode('utf-8'), full, os.path.getsize(full), 0])
    items.sort(key=lambda item: item[0])
    return items


def build(folder, out_dir):
    folder = os.path.abspath(folder)
    base = os.path.join(out_dir or os.path.dirname(folder), os.path.basename(folder))
    pack_path, index_path = base + '.nxpack', base + '.nxidx'
    items = collect(folder)
    if not items or len(items) > 100000:
        sys.exit('no files, or too many files, to pack')
    pack_size, paths_size, total = 40, 0, 0
    pack_id = FNV_BASIS
    for item in items:
        pack_size = (pack_size + 15) & ~15
        item[3] = pack_size
        pack_size += item[2]
        paths_size += len(item[0]) + 1
        total += item[2]
        pack_id = fnv(pack_id, item[0])
        pack_id = fnv(pack_id, struct.pack('<Q', item[2]))
    if pack_size >= 4 << 30:  # one plain file: FAT32 (the usual Switch SD format) cannot hold 4 GB or more
        sys.exit(f'the pack would be {pack_size >> 20} MB, over the FAT32 4 GB file limit: not built')
    nonce = ((int(time.time()) << 32) ^ time.perf_counter_ns()) & MASK
    pack_id = fnv(pack_id, struct.pack('<Q', nonce)) or 1

    data_checksum = FNV_BASIS
    t0 = time.time()
    with open(pack_path + '.tmp', 'wb') as out:
        out.write(b'\0' * 40)
        pos, done, last = 40, 0, -1
        for rel, full, size, offset in items:
            if offset > pos:
                out.write(b'\0' * (offset - pos))
            pos = offset
            with open(full, 'rb') as f:
                while True:
                    piece = f.read(CHUNK)
                    if not piece:
                        break
                    out.write(piece)
                    data_checksum = fnv_fast(data_checksum, piece)
                    pos += len(piece)
                    done += len(piece)
            percent = done * 100 // total if total else 100
            if percent // 10 != last:
                last = percent // 10
                print(f'  packing {percent} %  ({done >> 20} / {total >> 20} MB, {time.time() - t0:.0f} s)')
        out.seek(0)
        out.write(b'SSNXPAK1' + struct.pack('<IIQQQ', 2, 0, pack_id, pack_size, data_checksum))

    entries = bytearray()
    paths = bytearray()
    for rel, full, size, offset in items:
        entries += struct.pack('<QQII', offset, size, len(paths), len(rel))
        paths += rel + b'\0'
    checksum = fnv(fnv(FNV_BASIS, bytes(entries)), bytes(paths))
    with open(index_path + '.tmp', 'wb') as out:
        out.write(b'SSNXIDX1' + struct.pack('<IIQQQQQ', 2, len(items), pack_id, pack_size, len(paths), checksum,
                                            data_checksum))
        out.write(entries)
        out.write(paths)
    for final in (pack_path, index_path):
        if os.path.exists(final):
            os.remove(final)
    os.rename(pack_path + '.tmp', pack_path)
    os.rename(index_path + '.tmp', index_path)
    print(f'{pack_path}: {len(items)} files, {pack_size >> 20} MB, data checksum {data_checksum:016x}')
    return pack_path, index_path


def verify(pack_path, index_path):
    with open(index_path, 'rb') as f:
        head = f.read(56)
        magic, version, count, pack_id, pack_size, paths_size, checksum, data_checksum = struct.unpack(
            '<8sIIQQQQQ', head)
        entries = f.read(count * 24)
        paths = f.read(paths_size)
    assert magic == b'SSNXIDX1' and version == 2, 'not an asset index'
    assert fnv(fnv(FNV_BASIS, entries), paths) == checksum, 'index checksum mismatch'
    h = FNV_BASIS
    with open(pack_path, 'rb') as f:
        pmagic, pversion, _, ppack_id, pfile_size, pdata = struct.unpack('<8sIIQQQ', f.read(40))
        assert pmagic == b'SSNXPAK1' and ppack_id == pack_id and pfile_size == pack_size == os.path.getsize(pack_path)
        for i in range(count):
            offset, size, _, _ = struct.unpack_from('<QQII', entries, i * 24)
            f.seek(offset)
            while size:
                piece = f.read(min(size, CHUNK))
                h = fnv_fast(h, piece)
                size -= len(piece)
    assert h == data_checksum == pdata, f'data checksum mismatch: {h:016x} != {data_checksum:016x}'
    print(f'{pack_path}: {count} files verified')


def stale(folder, out_dir):
    base = os.path.join(out_dir or os.path.dirname(os.path.abspath(folder)), os.path.basename(os.path.abspath(folder)))
    if not (os.path.exists(base + '.nxpack') and os.path.exists(base + '.nxidx')):
        return True
    built = min(os.path.getmtime(base + '.nxpack'), os.path.getmtime(base + '.nxidx'))
    for root, dirs, files in os.walk(folder):
        for name in files + dirs:
            if os.path.getmtime(os.path.join(root, name)) > built:
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('folder')
    ap.add_argument('--out-dir', help='where to write <name>.nxpack/.nxidx (default: next to the folder)')
    ap.add_argument('--if-stale', action='store_true', help='only when missing or older than a file of the folder')
    ap.add_argument('--verify', action='store_true', help='reread the pack and check its data checksum')
    a = ap.parse_args()
    if a.if_stale and not stale(a.folder, a.out_dir):
        print('asset pack is up to date')
        base = os.path.join(a.out_dir or os.path.dirname(os.path.abspath(a.folder)), os.path.basename(os.path.abspath(a.folder)))
        paths = (base + '.nxpack', base + '.nxidx')
    else:
        paths = build(a.folder, a.out_dir)
    if a.verify:
        verify(*paths)


if __name__ == '__main__':
    main()
