#!/usr/bin/env python3
"""Independent reference reader/writer of the shader package (.mesp) and its index (.mesp.idx).

Written from the format description in shaders/README.md, not from the C++ code, so it doubles as a
check of that description and as a model for other implementations (for example the browser installer).
Needs the `xxhash` module (pip install xxhash): XXH3 64-bit, seed 0.

  package_reference.py index  <package.mesp> [<out.idx>]   write the index (default <package>.idx)
  package_reference.py check  <package.mesp> <package.idx> verify that the .idx is byte-identical to the one
                                                           this script derives from the package
  package_reference.py build  <containers dir> <spirv dir> <out.mesp>   pack (<name>.bin + <name>.spv)
"""
import os
import struct
import sys

import xxhash

PKG_MAGIC = b"MESSPV\0\0"
IDX_MAGIC = b"MESSIDX\0"
MAX_ORIGINAL = 64 * 1024
MAX_SPIRV_WORDS = (4 * 1024 * 1024) // 4
MAX_PACKAGE = 1 << 30
MAX_SHADERS = 65536


def xxh3(data):
    return xxhash.xxh3_64_intdigest(data)


def be32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def validate_container(c):
    """Returns True if the container is a vertex shader (bit 0 of the signature)."""
    if not (24 <= len(c) <= MAX_ORIGINAL):
        raise ValueError("container length")
    sig, virt, phys = be32(c, 0), be32(c, 4), be32(c, 8)
    if not ((sig & ~1) == 0x102A0E00 or (sig & 0xFFFFFF00) == 0x102A1100):
        raise ValueError("container signature")
    is2008 = (sig & 0xFFFFFF00) == 0x102A1100
    if not (virt >= 24 and phys and (is2008 or phys % 12 == 0) and virt + phys == len(c)):
        raise ValueError("container sizes")
    return bool(sig & 1)


def walk_spirv(words):
    """Yields (opcode, start, wordcount) for every instruction after the 5-word header."""
    i = 5
    while i < len(words):
        wc, op = words[i] >> 16, words[i] & 0xFFFF
        if wc == 0 or wc > len(words) - i:
            raise ValueError("truncated instruction")
        yield op, i, wc
        i += wc


def validate_spirv(words, vertex):
    if not (5 <= len(words) <= MAX_SPIRV_WORDS) or words[0] != 0x07230203 or words[4] != 0:
        raise ValueError("SPIR-V header")
    entry = False
    for op, i, wc in walk_spirv(words):
        if op == 15:  # OpEntryPoint
            if entry or wc < 5 or words[i + 1] != (0 if vertex else 4) or words[i + 3] != 0x6E69616D or words[i + 4] != 0:
                raise ValueError("entry point")
            entry = True
    if not entry:
        raise ValueError("no main entry point")


def count_kills(words):
    if len(words) < 5 or words[0] != 0x07230203:
        return 2
    n = 0
    try:
        for op, i, wc in walk_spirv(words):
            if op in (252, 4416, 5380):  # OpKill, OpTerminateInvocation, OpDemoteToHelperInvocation
                n += 1
    except ValueError:
        return 2
    return n


def read_package(data):
    if len(data) < 24 or len(data) > MAX_PACKAGE or data[:8] != PKG_MAGIC:
        raise ValueError("package header")
    version, count, checksum = struct.unpack_from("<IIQ", data, 8)
    if version != 1 or not (0 < count <= MAX_SHADERS):
        raise ValueError("package version/count")
    if xxh3(data[24:]) != checksum:
        raise ValueError("package checksum")
    pos, entries, prev = 24, [], None
    for _ in range(count):
        n_orig, n_words, fingerprint = struct.unpack_from("<IIQ", data, pos)
        offset = pos
        pos += 16
        orig = bytes(data[pos:pos + n_orig]); pos += n_orig
        spv = bytes(data[pos:pos + n_words * 4]); pos += n_words * 4
        if pos > len(data):
            raise ValueError("truncated")
        vertex = validate_container(orig)
        words = struct.unpack("<%dI" % n_words, spv)
        validate_spirv(words, vertex)
        if fingerprint != xxh3(orig):
            raise ValueError("container hash")
        key = (fingerprint, orig)
        if prev is not None and not prev < key:
            raise ValueError("entries not strictly ascending")
        prev = key
        entries.append(dict(offset=offset, orig=orig, spv=spv, words=words, fingerprint=fingerprint))
    if pos != len(data):
        raise ValueError("trailing data")
    return entries, checksum


def build_package(items):
    """items: list of (container bytes, spirv bytes). Returns the package bytes."""
    prepared = []
    for orig, spv in items:
        vertex = validate_container(orig)
        words = struct.unpack("<%dI" % (len(spv) // 4), spv)
        validate_spirv(words, vertex)
        prepared.append((xxh3(orig), orig, spv))
    prepared.sort(key=lambda t: (t[0], t[1]))
    body, count, last = bytearray(), 0, None
    for fingerprint, orig, spv in prepared:
        if last is not None and last[1] == orig:
            if last[2] != spv:
                raise ValueError("one container with two translations")
            continue
        body += struct.pack("<IIQ", len(orig), len(spv) // 4, fingerprint) + orig + spv
        count += 1
        last = (fingerprint, orig, spv)
    return PKG_MAGIC + struct.pack("<IIQ", 1, count, xxh3(bytes(body))) + bytes(body)


def build_index(package):
    entries, checksum = read_package(package)
    table, originals = bytearray(), bytearray()
    for e in entries:
        table += struct.pack("<IIQQIIQ", len(e["orig"]), len(e["words"]), e["fingerprint"], xxh3(e["spv"]),
                             count_kills(e["words"]), 0, e["offset"])
        originals += e["orig"]
    body = bytes(table) + bytes(originals)
    return IDX_MAGIC + struct.pack("<IIQQQQQ", 1, len(entries), len(package), checksum, len(originals), xxh3(body), 0) + body


def main(argv):
    if len(argv) >= 3 and argv[1] == "index":
        pkg = open(argv[2], "rb").read()
        out = argv[3] if len(argv) > 3 else argv[2] + ".idx"
        open(out, "wb").write(build_index(pkg))
        print("wrote", out)
    elif len(argv) == 4 and argv[1] == "check":
        pkg, idx = open(argv[2], "rb").read(), open(argv[3], "rb").read()
        mine = build_index(pkg)
        print("identical" if mine == idx else "DIFFERENT")
        return 0 if mine == idx else 1
    elif len(argv) == 5 and argv[1] == "build":
        items = []
        for name in sorted(os.listdir(argv[2])):
            if name.endswith(".bin"):
                spv = os.path.join(argv[3], name[:-4] + ".spv")
                if os.path.exists(spv):
                    items.append((open(os.path.join(argv[2], name), "rb").read(), open(spv, "rb").read()))
        pkg = build_package(items)
        open(argv[4], "wb").write(pkg)
        open(argv[4] + ".idx", "wb").write(build_index(pkg))
        print("wrote", argv[4], len(items), "containers")
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
