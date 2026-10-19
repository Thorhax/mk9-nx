#!/usr/bin/env python3
"""Writes the shader recipe of the installer page: how to rebuild, from a player's own copy of the game, every shader
container of a tested shader package.

The package (tools/build_shader_library.sh) is made from containers of three kinds: those the game files hold (EMZ
streams of archive/ and the .eaf archives), those a run dumped (the game's in-memory form: most are byte-identical to
a game-file container, a few have header bytes the game rewrote at load time, a few have microcode Direct3D patched
for the vertex declaration) and Direct3D's own shaders, whose containers sit in the decompressed default.xex.

For every container of the package's input folder this finds a source of the same size in the player's files and
records the bytes that differ, as runs of (offset, hex). A source is one of:

    {"emz": [path, offset of "#EMZ"], "at": n}   a slice of an unpacked EMZ stream
    {"file": path, "at": n}                     a slice of a game file as it is on the disc
    {"image": n}                                a slice of default.xex's decompressed image (based at 0x82000000)

The recipe holds offsets and the few patched bytes only, never a container: the page reads the rest from the
player's files. The best source is the one with the fewest differing bytes (ties: the smallest game file, then the
path), so patches stay at the bytes the game or Direct3D changed.

    python3 tools/installer_recipe.py assets/game_root out/default_image.bin out/shaders/in out/installer/recipe.json

Then tools/installer_recipe.py --check rebuilds every container from the recipe and compares it with the input folder.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import zlib

MAGICS = (b"\x10\x2a\x11\x01", b"\x10\x2a\x11\x00")


def emz_streams(data):
    """Yields (offset of "#EMZ", unpacked bytes) for every EMZ stream in `data` (as tools/extract_game_shaders.py)."""
    pos = data.find(b"#EMZ")
    while pos >= 0:
        if pos + 16 <= len(data):
            size, offset = struct.unpack_from("<II", data, pos + 8)
            if 0 < size < (256 << 20) and 16 <= offset < 4096:
                try:
                    out = zlib.decompressobj(-15).decompress(data[pos + offset:pos + offset + size + (1 << 20)], size)
                    if len(out) == size:
                        yield pos, out
                except zlib.error:
                    pass
        pos = data.find(b"#EMZ", pos + 4)


def containers(blob):
    """Yields (offset, size) of every plausible shader container in `blob`."""
    for magic in MAGICS:
        pos = blob.find(magic)
        while pos >= 0:
            if pos % 4 == 0 and pos + 12 <= len(blob):
                header, ucode = struct.unpack_from(">II", blob, pos + 4)
                total = header + ucode
                if 12 <= header <= (1 << 20) and 0 < ucode <= (1 << 20) and ucode % 4 == 0 and pos + total <= len(blob):
                    yield pos, total
            pos = blob.find(magic, pos + 4)


def game_files(root):
    for top, _, names in os.walk(root):
        for name in sorted(names):
            path = os.path.join(top, name)
            rel = os.path.relpath(path, root).replace(os.sep, "/")
            if name.lower().endswith((".wmv", ".xex")) or rel.startswith("$SystemUpdate"):
                continue
            yield rel, path


def index_sources(root, image):
    """Every candidate source: (size -> list of (rank, locator, bytes)). Rank orders ties: smaller files first."""
    by_size = {}
    files = sorted(game_files(root), key=lambda rp: (os.path.getsize(rp[1]), rp[0]))
    for rank, (rel, path) in enumerate(files):
        data = open(path, "rb").read()
        for emz_at, blob in emz_streams(data):
            for at, size in containers(blob):
                by_size.setdefault(size, []).append(((0, rank, rel, emz_at, at), {"emz": [rel, emz_at], "at": at},
                                                    blob[at:at + size]))
        for at, size in containers(data):
            by_size.setdefault(size, []).append(((1, rank, rel, 0, at), {"file": rel, "at": at}, data[at:at + size]))
    for at, size in containers(image):
        by_size.setdefault(size, []).append(((2, 0, "", 0, at), {"image": at}, image[at:at + size]))
    return by_size


def diff_runs(a, b):
    """Runs of differing bytes as [offset, hex of b]."""
    runs, i = [], 0
    while i < len(a):
        if a[i] == b[i]:
            i += 1
            continue
        j = i
        while j < len(a) and a[j] != b[j]:
            j += 1
        runs.append([i, b[i:j].hex()])
        i = j
    return runs


def split_source(by_size, want):
    """A container whose whole bytes are nowhere: its header from the nearest container with the same header size
    and kind, its microcode from the nearest microcode of the same size (Direct3D rewrites the fetches of a vertex
    shader for a vertex declaration; its own shaders get a wrapper header). Returns (header locator, microcode
    locator, patch runs over the whole container) or None."""
    header, ucode = struct.unpack_from(">II", want, 4)
    best_h = best_m = None
    for size, items in by_size.items():
        for key, locator, have in items:
            h, u = struct.unpack_from(">II", have, 4)
            if h == header and have[:4] == want[:4]:
                cost = sum(x != y for x, y in zip(have[:h], want[:header]))
                if best_h is None or (cost, key) < (best_h[0], best_h[1]):
                    best_h = (cost, key, dict(locator, len=h), have[:h])
            if u == ucode:
                cost = sum(x != y for x, y in zip(have[h:h + u], want[header:]))
                if best_m is None or (cost, key) < (best_m[0], best_m[1]):
                    loc = dict(locator)
                    if "image" in loc:
                        loc["image"] += h
                    else:
                        loc["at"] += h
                    best_m = (cost, key, dict(loc, len=u), have[h:h + u])
    if best_h is None:
        return None
    if best_m is None:
        # No microcode of that size anywhere (a shader Direct3D builds itself): carried inline.
        return best_h[2], {"inline": want[header:].hex()}, diff_runs(best_h[3], want[:header])
    return best_h[2], best_m[2], diff_runs(best_h[3] + best_m[3], want)


def build(a):
    image = open(a.image, "rb").read()
    print("indexing the game files ...", flush=True)
    by_size = index_sources(a.game_root, image)
    entries, unsolved = [], []
    for name in sorted(os.listdir(a.containers)):
        if not name.endswith(".bin"):
            continue
        want = open(os.path.join(a.containers, name), "rb").read()
        best = None
        for key, locator, have in by_size.get(len(want), []):
            if have == want:
                best = (0, key, locator, [])
                break
            if best is not None and best[0] == 0:
                break
            # Same container kind only (vertex / pixel magic), and the same header size.
            if have[:8] != want[:8]:
                continue
            runs = diff_runs(have, want)
            cost = sum(len(r[1]) // 2 for r in runs)
            if best is None or (cost, key) < (best[0], best[1]):
                best = (cost, key, locator, runs)
        if best is None:
            split = split_source(by_size, want)
            if split is None:
                # Nothing of its header size nor of its microcode size anywhere: Direct3D made it (a register-
                # linked wrapper or a vertex shader rewritten for a declaration). Carried inline.
                entries.append({"name": name, "size": len(want), "sha1": hashlib.sha1(want).hexdigest(),
                                "inline": want.hex()})
                continue
            entry = {"name": name, "size": len(want), "sha1": hashlib.sha1(want).hexdigest(), "header": split[0],
                     "microcode": split[1]}
            if split[2]:
                entry["patch"] = split[2]
            entries.append(entry)
            continue
        entry = {"name": name, "size": len(want), "sha1": hashlib.sha1(want).hexdigest(), "source": best[2]}
        if best[3]:
            entry["patch"] = best[3]
        entries.append(entry)
    patched = [e for e in entries if "patch" in e]
    split = [e for e in entries if "header" in e]
    inline = [e for e in split if "inline" in e["microcode"]]
    whole = [e for e in entries if "inline" in e]
    print(f"{len(whole)} whole containers inline: {sum(e['size'] for e in whole)} bytes")
    print(f"{len(inline)} with their microcode inline: {sum(len(e['microcode']['inline']) // 2 for e in inline)} bytes")
    print(f"{len(split)} split (header and microcode from different sources): "
          f"{sum(sum(len(r[1]) // 2 for r in e.get('patch', [])) for e in split)} patched bytes of "
          f"{sum(e['size'] for e in split)}")
    print(f"{len(entries)} containers: {len(entries) - len(patched)} as they are in the player's files, "
          f"{len(patched)} with patches ({sum(sum(len(r[1]) // 2 for r in e['patch']) for e in patched)} bytes); "
          f"{len(unsolved)} without a source: {unsolved[:10]}")
    files = sorted({loc["emz"][0] if "emz" in loc else loc.get("file") for e in entries
                    for loc in (e.get("source"), e.get("header"), e.get("microcode")) if loc} - {None})
    print(f"read from {len(files)} game files and the image")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump({"version": 1, "containers": entries}, f, indent=0)
    return 1 if unsolved else 0


def read_locator(root, image, src, size, cache):
    if "emz" in src:
        key = tuple(src["emz"])
        if key not in cache:
            path, emz_at = src["emz"]
            data = open(os.path.join(root, path), "rb").read()
            n, offset = struct.unpack_from("<II", data, emz_at + 8)
            cache[key] = zlib.decompressobj(-15).decompress(data[emz_at + offset:emz_at + offset + n + (1 << 20)], n)
        blob = cache[key]
    elif "file" in src:
        key = ("file", src["file"])
        if key not in cache:
            cache[key] = open(os.path.join(root, src["file"]), "rb").read()
        blob = cache[key]
    else:
        blob = image
    at = src["at"] if "at" in src else src["image"]
    return blob[at:at + size]


def rebuild(root, image, entry, cache):
    if "inline" in entry:
        return bytes.fromhex(entry["inline"])
    if "header" in entry:
        m = entry["microcode"]
        ucode = bytes.fromhex(m["inline"]) if "inline" in m else read_locator(root, image, m, m["len"], cache)
        out = bytearray(read_locator(root, image, entry["header"], entry["header"]["len"], cache) + ucode)
        for off, hexbytes in entry.get("patch", []):
            b = bytes.fromhex(hexbytes)
            out[off:off + len(b)] = b
        return bytes(out)
    src = entry["source"]
    if "emz" in src:
        key = tuple(src["emz"])
        if key not in cache:
            path, emz_at = src["emz"]
            data = open(os.path.join(root, path), "rb").read()
            size, offset = struct.unpack_from("<II", data, emz_at + 8)
            cache[key] = zlib.decompressobj(-15).decompress(data[emz_at + offset:emz_at + offset + size + (1 << 20)], size)
        blob = cache[key]
    elif "file" in src:
        key = ("file", src["file"])
        if key not in cache:
            cache[key] = open(os.path.join(root, src["file"]), "rb").read()
        blob = cache[key]
    else:
        blob = image
    at = src.get("at", src.get("image", 0))
    out = bytearray(blob[at:at + entry["size"]])
    for off, hexbytes in entry.get("patch", []):
        b = bytes.fromhex(hexbytes)
        out[off:off + len(b)] = b
    return bytes(out)


def check(a):
    image = open(a.image, "rb").read()
    recipe = json.load(open(a.out))
    cache, bad = {}, 0
    for entry in recipe["containers"]:
        got = rebuild(a.game_root, image, entry, cache)
        want = open(os.path.join(a.containers, entry["name"]), "rb").read()
        if got != want or hashlib.sha1(got).hexdigest() != entry["sha1"]:
            bad += 1
            print("MISMATCH", entry["name"])
    print(f"{len(recipe['containers'])} containers rebuilt, {bad} mismatches")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("game_root")
    ap.add_argument("image", help="default.xex's decompressed image (out/default_image.bin)")
    ap.add_argument("containers", help="the package's input folder (out/shaders/in)")
    ap.add_argument("out", help="recipe JSON to write (or to check, with --check)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    return check(a) if a.check else build(a)


if __name__ == "__main__":
    sys.exit(main())
