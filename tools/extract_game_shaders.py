#!/usr/bin/env python3
"""Extracts every Xbox 360 shader container from USF4's game files, for the native renderer's shader package.

A shader dump from a run (usf4_dump_shaders) only has the shaders of the stages and characters that run loaded; the
game files have all of them. USF4 keeps them in EMB archives, compressed as EMZ ("#EMZ", checksum, unpacked size,
data offset, then raw deflate): archive/shader/{Vs,Ps}{User,Bg}Shader.emz, and more EMZ streams inside the .eaf
archives (stages, characters, DLC). This tool walks the game folder, unpacks every EMZ stream it finds (whole files
and streams embedded in other files) and writes each container once:

    <out>/vs_g<sha1-16>.bin   magic 0x102A1101 (vertex)
    <out>/ps_g<sha1-16>.bin   magic 0x102A1100 (pixel)

A container is [0] magic, [1] header size (the microcode starts there), [2] microcode size, the same layout the
CreateVertexShader / CreatePixelShader hooks dump (app/src/me/usf4_me_hooks.cpp); its bytes are header + microcode.
Containers already in --skip folders (byte-identical) are not written again.

  python3 tools/extract_game_shaders.py assets/game_root out/pc/shaders_game [--skip out/pc/shaders]

Pure Python 3, read only; nothing from the game is part of the repository (the output goes under out/).
"""
import argparse
import hashlib
import os
import struct
import sys
import zlib

VS_MAGIC = b"\x10\x2a\x11\x01"
PS_MAGIC = b"\x10\x2a\x11\x00"


def emz_streams(data):
    """Yields the unpacked bytes of every EMZ stream in `data` (whole file or embedded)."""
    pos = data.find(b"#EMZ")
    while pos >= 0:
        if pos + 16 <= len(data):
            size, offset = struct.unpack_from("<II", data, pos + 8)
            if 0 < size < (256 << 20) and 16 <= offset < 4096:
                try:
                    out = zlib.decompressobj(-15).decompress(data[pos + offset:pos + offset + size + (1 << 20)], size)
                    if len(out) == size:
                        yield out
                except zlib.error:
                    pass
        pos = data.find(b"#EMZ", pos + 4)


def containers(blob):
    """Yields (vertex, bytes) for every plausible shader container in `blob`."""
    for magic, vertex in ((VS_MAGIC, True), (PS_MAGIC, False)):
        pos = blob.find(magic)
        while pos >= 0:
            if pos % 4 == 0 and pos + 12 <= len(blob):
                header, ucode = struct.unpack_from(">II", blob, pos + 4)
                total = header + ucode
                if 12 <= header <= (1 << 20) and 0 < ucode <= (1 << 20) and ucode % 4 == 0 and pos + total <= len(blob):
                    yield vertex, blob[pos:pos + total]
            pos = blob.find(magic, pos + 4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game_root")
    ap.add_argument("out")
    ap.add_argument("--skip", nargs="*", default=[], help="folders of containers already collected (vs_*/ps_*.bin)")
    a = ap.parse_args()

    known = set()
    for folder in a.skip:
        for name in os.listdir(folder):
            if name.endswith(".bin") and name[:3] in ("vs_", "ps_"):
                known.add(hashlib.sha1(open(os.path.join(folder, name), "rb").read()).digest())
    os.makedirs(a.out, exist_ok=True)
    written = {True: 0, False: 0}
    seen = set()
    files = streams = 0
    for root, _, names in os.walk(a.game_root):
        for name in sorted(names):
            path = os.path.join(root, name)
            if name.lower().endswith((".wmv", ".xex")) or "$SystemUpdate" in path:
                continue
            files += 1
            data = open(path, "rb").read()
            sources = list(emz_streams(data))
            streams += len(sources)
            sources.append(data)  # uncompressed containers, if any
            for blob in sources:
                for vertex, c in containers(blob):
                    digest = hashlib.sha1(c).digest()
                    if digest in seen or digest in known:
                        continue
                    seen.add(digest)
                    prefix = "vs" if vertex else "ps"
                    with open(os.path.join(a.out, f"{prefix}_g{digest.hex()[:16]}.bin"), "wb") as f:
                        f.write(c)
                    written[vertex] += 1
            del data
    print(f"{files} files, {streams} EMZ streams; new containers: {written[True]} vertex, {written[False]} pixel "
          f"({len(known)} already known were skipped) -> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
