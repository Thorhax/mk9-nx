#!/usr/bin/env python3
"""Writes the shader recipe of mk9-nx's installer page: where each shader container of a release's deko3d package is in
a player's own copy of the game, so the page can rebuild them (mk9-nx-installer, lib/shaders.js).

MK9's containers are in its Unreal packages (Asset/*.xxx), inside LZO-compressed chunks; a few are Direct3D's own, in
default.xex's decompressed image; two Direct3D builds at run time and are on no disc (the recipe carries those few
hundred bytes). Sources:

    {"ue3": [package, file offset of the chunk header], "at": n}   a slice of a compressed chunk's uncompressed data
    {"image": n}                                                   a slice of default.xex's decompressed image
    "inline": hex                                                  the container itself (Direct3D's run-time ones)

The recipe holds offsets, sizes and SHA-1s only (and the inline few): the page reads the rest from the player's files.

    out/tools/ue3_shader_locate assets/game_root assets/game_root/Asset/*.xxx > out/ue3_locations.tsv
    python3 tools/installer_recipe_mk9.py out/ue3_locations.tsv out/default_image.bin out/renut_shader_dump \\
        out/deko3d_shaders/glsl_opt release/recipe-<version>.json

The dump folder (the containers the release's package was made from) is indexed by each file's XXH3, whatever it is
named (the disc scan's files keep ue3_shader_scan's FNV names); every container is checked against its copy.
"""
import hashlib
import json
import os
import re
import struct
import sys

import xxhash

CONTAINER = re.compile(rb'\x10\x2a\x11[\x00\x01]')


def name_of(data):
    return ('vs_' if data[3] & 1 else 'ps_') + xxhash.xxh3_64_hexdigest(data)


def image_containers(image):
    """Every container in the decompressed default.xex image by its translator name (first offset)."""
    found = {}
    for m in CONTAINER.finditer(image):
        o = m.start()
        if o % 4:
            continue
        vs, ps = struct.unpack_from('>II', image, o + 4)
        if vs < 24 or vs > 0x40000 or ps == 0 or ps > 0x40000 or o + vs + ps > len(image):
            continue
        found.setdefault(name_of(image[o:o + vs + ps]), o)
    return found


def main():
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    locations, image_path, dump, package, out = sys.argv[1:]
    ue3 = {}
    for line in open(locations, encoding='utf-8'):
        name, path, size, chunks, at = line.rstrip('\n').split('\t')
        if chunks == '-' or ',' in chunks:
            sys.exit(f'{name}: not inside one compressed chunk ({chunks}); the page reads only that form')
        ue3[name] = (path, int(chunks.split(':')[0]), int(at), int(size))
    image = open(image_path, 'rb').read()
    in_image = image_containers(image)

    copies = {}
    for f in os.listdir(dump):
        if f.endswith('.bin'):
            path = os.path.join(dump, f)
            copies.setdefault(name_of(open(path, 'rb').read()), path)
    names = sorted(f[:3] + f[3:-5].lower() for f in os.listdir(package) if f.endswith('.glsl'))
    containers, kinds = [], {'ue3': 0, 'image': 0, 'inline': 0}
    for name in names:
        if name not in copies:
            sys.exit(f'{name}: not in the dump folder')
        data = open(copies[name], 'rb').read()
        entry = {'name': name + '.bin', 'size': len(data), 'sha1': hashlib.sha1(data).hexdigest()}
        if name in ue3:
            path, header, at, size = ue3[name]
            if size != len(data):
                sys.exit(f'{name}: the package\'s copy has another size')
            entry['source'] = {'ue3': [path, header], 'at': at}
            kinds['ue3'] += 1
        elif name in in_image:
            entry['source'] = {'image': in_image[name]}
            kinds['image'] += 1
        else:
            entry['inline'] = data.hex()
            kinds['inline'] += 1
        containers.append(entry)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'version': 1, 'game': 'mk9', 'containers': containers}, f, separators=(',', ':'))
    print(f'{len(containers)} containers: {kinds["ue3"]} from the packages\' chunks, {kinds["image"]} from default.xex, '
          f'{kinds["inline"]} carried ({sum(c["size"] for c in containers if "inline" in c)} bytes) -> {out} '
          f'({os.path.getsize(out) >> 10} KB)')


if __name__ == '__main__':
    main()
