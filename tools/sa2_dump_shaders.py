#!/usr/bin/env python3
"""Dumps every Xbox 360 shader container SA2 has, for Renut's translator (the deko3d renderer, sa2_renderer = "renut").

SA2's shaders are in two places: compiled into default.xex (Direct3D's own and the game's), and the Ninja shader objects
in Shader/xbox360/obj (*.wpu pixel, *.wvu vertex), each a whole container. The executable is read from its loaded
image (out/default_image.bin, written by a run with SA2_DUMP_IMAGE=<file>: SA2's XEX uses a compression reNut's reader
does not take), so nothing here decrypts or decompresses.

Output: <out>/{vs,ps}_<xxh3_64 of the container>.bin, each a complete container (virtual part, then physical part: the
layout XenosRecomp reads), and <out>/manifest.csv. Same naming as reNut's tools/dump_shaders.py (masterspike52), whose
container check this copies.

  python tools/sa2_dump_shaders.py --image out/default_image.bin --assets assets/game_root --out out/renut_shader_dump
"""
import argparse
import csv
import os
import re
import struct
import sys

try:
    import xxhash
except ImportError:
    sys.exit("error: the 'xxhash' module is required (pip install xxhash)")

SHADER_MAGIC = re.compile(rb"\x10\x2a\x11[\x00\x01]")


def find_shader_headers(data):
    """reNut's check: a container's sizes, tables and its physical part's (offset, size) agree."""
    found = []
    for m in SHADER_MAGIC.finditer(data):
        o = m.start()
        if o % 4 or o + 0x24 > len(data):
            continue
        virt, phys = struct.unpack_from(">II", data, o + 4)
        ct, dt, so = struct.unpack_from(">III", data, o + 0x10)
        if not (0x24 <= virt < 0x10000 and 0 < phys < 0x100000 and o + virt + phys <= len(data)):
            continue
        if ct >= virt or dt >= virt or not (0x24 <= so < virt - 8):
            continue
        poff, psize = struct.unpack_from(">II", data, o + so)
        if poff + psize != phys:
            continue
        found.append(o)
    return found


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--image", default=os.path.join(root, "out", "default_image.bin"),
                    help="the loaded default.xex image (SA2_DUMP_IMAGE)")
    ap.add_argument("--image-base", type=lambda s: int(s, 0), default=0x82000000)
    ap.add_argument("--assets", default=os.path.join(root, "assets", "game_root"))
    ap.add_argument("--out", default=os.path.join(root, "out", "renut_shader_dump"))
    args = ap.parse_args()

    sources = []
    if os.path.isfile(args.image):
        sources.append((args.image, args.image_base))
    else:
        print("warning: no %s: the executable's shaders are left out" % args.image)
    obj = os.path.join(args.assets, "Shader", "xbox360", "obj")
    if os.path.isdir(obj):
        for name in sorted(os.listdir(obj)):
            sources.append((os.path.join(obj, name), 0))
    if not sources:
        sys.exit("no inputs")

    os.makedirs(args.out, exist_ok=True)
    seen = set()
    manifest = []
    for path, base in sources:
        data = open(path, "rb").read()
        count = 0
        for h in find_shader_headers(data):
            flags, virt, phys = struct.unpack_from(">III", data, h)
            blob = data[h:h + virt + phys]
            kind = "vs" if flags & 1 else "ps"
            name = "%s_%s" % (kind, xxhash.xxh3_64_hexdigest(blob))
            manifest.append((name, kind, virt, phys, os.path.basename(path), "0x%08x" % (base + h)))
            count += 1
            if name in seen:
                continue
            seen.add(name)
            with open(os.path.join(args.out, name + ".bin"), "wb") as fh:
                fh.write(blob)
        print("%-40s %4d shaders" % (os.path.basename(path), count))
    manifest.sort()
    with open(os.path.join(args.out, "manifest.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "type", "virtual_size", "physical_size", "source", "asset"])
        w.writerows(manifest)
    ps = sum(1 for n in seen if n.startswith("ps_"))
    print("%d unique shaders (%d ps, %d vs) -> %s" % (len(seen), ps, len(seen) - ps, args.out))


if __name__ == "__main__":
    main()
