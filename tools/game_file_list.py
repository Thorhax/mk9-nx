#!/usr/bin/env python3
"""game_file_list.py <game folder> <out.txt>: the release's list of game files, one "<size> <sha256> <path>" line each
(path relative to game_root, '/' separated, sorted; $SystemUpdate left out as the installer does). Compiled into the
Switch NRO (src/mk9_file_check.cpp): every start compares the sizes on the SD with it, and the start after a crash in
the game's code checks the SHA-256 of the files it had opened. Sizes and hashes only: no game data."""
import hashlib, os, sys

root, out = sys.argv[1], sys.argv[2]
rows = []
for base, dirs, files in os.walk(root):
    dirs.sort()
    for name in files:
        full = os.path.join(base, name)
        rel = os.path.relpath(full, root).replace(os.sep, '/')
        if rel.lower().startswith('$systemupdate/'):
            continue
        h = hashlib.sha256()
        with open(full, 'rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        rows.append((rel, os.path.getsize(full), h.hexdigest()))
rows.sort()
with open(out, 'w', newline='\n') as f:
    f.write('# mk9-nx: the game files of Mortal Kombat Komplete Edition (Xbox 360 disc, default.xex SHA-256\n'
            '# e51751328751287b2a6113c815b99f470d1ea0c88f5d9d9f9b1942a28b3e7f6d): size, SHA-256, path in game_root.\n'
            '# Written by tools/game_file_list.py; sizes and hashes only.\n')
    for rel, size, digest in rows:
        f.write(f'{size} {digest} {rel}\n')
print(f'{out}: {len(rows)} files, {sum(r[1] for r in rows)} bytes')
