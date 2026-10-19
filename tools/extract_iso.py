#!/usr/bin/env python3
"""Extracts an Xbox 360 game disc image (.iso, XDVDFS) into a folder and prints the information of its default.xex.

Pure Python 3.8+, no dependencies and no external extraction tool. Stage 1 of the build: the result,
assets/game_root, is the game folder that app/sa2_manifest.toml and the console build read.

Usage:
    python3 tools/extract_iso.py "path/to/Mass Effect.iso"                 extract into assets/game_root
    python3 tools/extract_iso.py "path/to/Mass Effect.iso" -o some/folder  extract somewhere else
    python3 tools/extract_iso.py "path/to/Mass Effect.iso" --list          list the files, extract nothing
    python3 tools/extract_iso.py "path/to/Mass Effect.iso" --xex-only      extract only default.xex
    python3 tools/extract_iso.py assets/game_root/default.xex --info       print the XEX header of an extracted game

What it does:
  1. Finds where the game partition starts (XGD1, XGD2, XGD3 or a raw partition image).
  2. Walks the XDVDFS directory tree and lists or extracts every file.
  3. Reads the XEX2 header of default.xex: title id, media id, version, image base address and entry point.

It does not decrypt or decompress the executable: the code generator (rexglue) does that. Your own disc dump is the
only input; nothing from the game is part of this repository. Compressed or container formats (CCI, GOD, ZAR) must be
converted to a plain .iso first.
"""

import argparse
import os
import struct
import sys

SECTOR = 2048
XDVDFS_MAGIC = b"MICROSOFT*XBOX*MEDIA"
ATTR_DIRECTORY = 0x10
MASS_EFFECT_TITLE_ID = 0x4D5307E8
MAX_DIRECTORY_BYTES = 256 << 20     # a directory table larger than this is a damaged image

# Offsets where the game partition starts, by disc type (the volume descriptor is 32 sectors further).
KNOWN_BASES = [
    (0x00000000, "raw partition / trimmed image"),
    (0x0FD90000, "XGD2 (most Xbox 360 discs)"),
    (0x02080000, "XGD3 (late titles)"),
    (0x18300000, "XGD1 (original Xbox layout)"),
]


class ExtractError(Exception):
    pass


# ---------------------------------------------------------------------------------------------------------------
# XDVDFS
# ---------------------------------------------------------------------------------------------------------------

def _magic_at(fh, base):
    """True if a volume descriptor sits at partition offset `base`."""
    try:
        fh.seek(base + 32 * SECTOR)
    except OSError:
        return False
    return fh.read(len(XDVDFS_MAGIC)) == XDVDFS_MAGIC


def find_partition(fh, scan_limit=1 << 30):
    """Returns (base, description) of the game partition."""
    for base, description in KNOWN_BASES:
        if _magic_at(fh, base):
            return base, description

    # None of the known layouts: scan the first gigabyte for the volume descriptor magic.
    fh.seek(0, os.SEEK_END)
    limit = min(fh.tell(), scan_limit)
    chunk = 16 << 20
    overlap = len(XDVDFS_MAGIC)
    pos = 0
    while pos < limit:
        fh.seek(pos)
        buf = fh.read(chunk + overlap)
        if not buf:
            break
        idx = buf.find(XDVDFS_MAGIC)
        while idx != -1:
            absolute = pos + idx
            if absolute % SECTOR == 0 and absolute >= 32 * SECTOR and _magic_at(fh, absolute - 32 * SECTOR):
                return absolute - 32 * SECTOR, "found by scanning (non-standard offset)"
            idx = buf.find(XDVDFS_MAGIC, idx + 1)
        pos += chunk
    raise ExtractError("No XDVDFS file system found in the image. Check that it is an Xbox 360 .iso and not a "
                       "compressed CCI/GOD/ZAR image.")


def read_volume_descriptor(fh, base):
    """Returns (root_sector, root_size)."""
    fh.seek(base + 32 * SECTOR)
    vd = fh.read(SECTOR)
    if len(vd) < SECTOR or vd[:20] != XDVDFS_MAGIC:
        raise ExtractError("Invalid volume descriptor.")
    if vd[0x7EC:0x7EC + 20] != XDVDFS_MAGIC:
        print("  warning: the closing magic at 0x7EC is missing (truncated image?)", file=sys.stderr)
    root_sector, root_size = struct.unpack_from("<II", vd, 0x14)
    return root_sector, root_size


def _directory_entries(table):
    """Walks the binary search tree of one directory table. Yields a dict per entry."""
    seen = set()
    stack = [0]
    while stack:
        off = stack.pop()
        if off in seen or off + 14 > len(table):
            continue
        seen.add(off)
        left, right, sector, size, attrs, name_len = struct.unpack_from("<HHIIBB", table, off)
        for child in (left, right):
            if child not in (0, 0xFFFF):      # 0 and 0xFFFF both mean "no child"
                stack.append(child * 4)
        end = off + 14 + name_len
        if name_len == 0 or end > len(table):
            continue
        yield {
            "name": table[off + 14:end].decode("latin-1"),
            "sector": sector,
            "size": size,
            "is_dir": bool(attrs & ATTR_DIRECTORY),
        }


def walk(fh, base, sector, size, prefix=""):
    """Walks the directory tree depth first. Yields (relative path, entry)."""
    if size == 0 or size > MAX_DIRECTORY_BYTES:
        return
    fh.seek(base + sector * SECTOR)
    table = fh.read(size)
    if len(table) < size:
        print("  warning: truncated directory table in /%s" % prefix, file=sys.stderr)
    for entry in sorted(_directory_entries(table), key=lambda e: e["name"].lower()):
        path = prefix + "/" + entry["name"] if prefix else entry["name"]
        yield path, entry
        if entry["is_dir"]:
            yield from walk(fh, base, entry["sector"], entry["size"], path)


def _safe_destination(out_dir, rel_path):
    """Destination path inside out_dir; refuses names that would escape it."""
    parts = rel_path.split("/")
    if any(p in ("", ".", "..") or "\\" in p or ":" in p for p in parts):
        raise ExtractError("Unsafe file name in the image: %r" % rel_path)
    dest = os.path.join(out_dir, *parts)
    root = os.path.abspath(out_dir)
    if os.path.commonpath([root, os.path.abspath(dest)]) != root:
        raise ExtractError("Unsafe file name in the image: %r" % rel_path)
    return dest


def extract_file(fh, base, entry, dest):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    fh.seek(base + entry["sector"] * SECTOR)
    remaining = entry["size"]
    with open(dest, "wb") as out:
        while remaining > 0:
            block = fh.read(min(1 << 20, remaining))
            if not block:
                raise ExtractError("Unexpected end of file while reading %s. Is the image incomplete?"
                                   % entry["name"])
            out.write(block)
            remaining -= len(block)


# ---------------------------------------------------------------------------------------------------------------
# XEX2 header
# ---------------------------------------------------------------------------------------------------------------

COMPRESSION = {0: "none", 1: "basic", 2: "normal (LZX)", 3: "delta"}
ENCRYPTION = {0: "none", 1: "normal (AES-128)"}


def _u32(buf, off):
    return struct.unpack_from(">I", buf, off)[0]


def print_xex_info(path):
    with open(path, "rb") as fh:
        header = fh.read(0x1000)
        if header[:4] != b"XEX2":
            raise ExtractError("%s does not start with the magic 'XEX2': not an Xbox 360 executable." % path)
        module_flags = _u32(header, 0x04)
        pe_offset = _u32(header, 0x08)
        security_offset = _u32(header, 0x10)
        optional_count = _u32(header, 0x14)

        print("== XEX2 header ==")
        print("  file                  : %s (%s bytes)" % (path, f"{os.path.getsize(path):,}"))
        print("  module flags          : 0x%08X" % module_flags)
        print("  PE data offset        : 0x%08X" % pe_offset)
        print("  security info offset  : 0x%08X" % security_offset)
        print("  optional headers      : %d" % optional_count)

        fh.seek(0x18)
        raw = fh.read(optional_count * 8)
        optional = {}
        for i in range(optional_count):
            key, value = struct.unpack_from(">II", raw, i * 8)
            optional[key] = value

        print()
        print("== Key data ==")
        title_id = None
        if 0x00040006 in optional:                    # execution info
            fh.seek(optional[0x00040006])
            info = fh.read(24)
            if len(info) == 24:
                media_id, version, _base_version, title_id = struct.unpack_from(">IIII", info, 0)
                _platform, _exe_table, disc_number, disc_count = struct.unpack_from(">BBBB", info, 0x10)
                print("  Title ID              : %08X" % title_id)
                print("  Media ID              : %08X" % media_id)
                print("  Version               : %d.%d.%d.%d" % ((version >> 28) & 0xF, (version >> 16) & 0xFFF,
                                                                  (version >> 8) & 0xFF, version & 0xFF))
                print("  Disc                  : %d of %d" % (disc_number, disc_count))
        else:
            print("  Title ID              : (no execution info)")
        if 0x00010201 in optional:
            print("  Image base address    : 0x%08X" % optional[0x00010201])
        if 0x00010100 in optional:
            print("  Entry point           : 0x%08X" % optional[0x00010100])

        fh.seek(security_offset)
        security = fh.read(0x184)
        if len(security) >= 0x114:
            print("  Load address          : 0x%08X" % _u32(security, 0x110))
            print("  Image size            : %s bytes" % f"{_u32(security, 0x004):,}")

        if 0x000003FF in optional:                    # file format info
            fh.seek(optional[0x000003FF])
            fmt = fh.read(8)
            if len(fmt) == 8:
                _size, enc, comp = struct.unpack(">IHH", fmt)
                print("  Encryption            : %s (%d)" % (ENCRYPTION.get(enc, "unknown"), enc))
                print("  Compression           : %s (%d)" % (COMPRESSION.get(comp, "unknown"), comp))
        if 0x00020200 in optional:
            print("  Default stack size    : %s bytes" % f"{optional[0x00020200]:,}")

    print()
    if title_id == MASS_EFFECT_TITLE_ID:
        print("  >> The Title ID is that of Mass Effect (Xbox 360). Good.")
    elif title_id is not None:
        print("  >> Note: this port is written for Mass Effect 1 (Title ID %08X); this executable has %08X."
              % (MASS_EFFECT_TITLE_ID, title_id))
    print("The image base address and the entry point are the numbers the manifest and the overrides refer to.")


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------

def default_output():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "assets", "game_root")


def extract_image(image, out_dir, list_only=False, xex_only=False):
    with open(image, "rb") as fh:
        base, description = find_partition(fh)
        print("Game partition at offset 0x%08X  (%s)" % (base, description))
        root_sector, root_size = read_volume_descriptor(fh, base)
        print("Root directory: sector %d, %s bytes" % (root_sector, f"{root_size:,}"))

        entries = list(walk(fh, base, root_sector, root_size))
        if not entries:
            raise ExtractError("The file system is empty. Corrupt image?")
        files = [(p, e) for p, e in entries if not e["is_dir"]]
        print("%d files, %d directories, %s bytes in total\n"
              % (len(files), len(entries) - len(files), f"{sum(e['size'] for _, e in files):,}"))

        if list_only:
            for path, entry in entries:
                print("  [dir]  %s" % path if entry["is_dir"] else "  %12s  %s" % (f"{entry['size']:,}", path))
            return None

        wanted = files
        if xex_only:
            wanted = [(p, e) for p, e in files if p.lower().endswith((".xex", ".xexp"))]
            if not wanted:
                raise ExtractError("There is no .xex file in the image.")

        # Create the directories too, so empty ones exist as on the disc.
        if not xex_only:
            for path, entry in entries:
                if entry["is_dir"]:
                    os.makedirs(_safe_destination(out_dir, path), exist_ok=True)
        done = 0
        for path, entry in wanted:
            extract_file(fh, base, entry, _safe_destination(out_dir, path))
            done += 1
            if done % 50 == 0:
                print("  ... %d files" % done)
        print("\n%d files extracted into %s" % (done, out_dir))
    return os.path.join(out_dir, "default.xex")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Extracts an Xbox 360 .iso and prints the XEX information.")
    parser.add_argument("input", help="the .iso, or a .xex (with --info)")
    parser.add_argument("-o", "--output", default=default_output(),
                        help="destination folder (default: assets/game_root of this repository)")
    parser.add_argument("--list", action="store_true", help="list the contents, extract nothing")
    parser.add_argument("--xex-only", action="store_true", help="extract only default.xex (and .xexp patches)")
    parser.add_argument("--info", action="store_true", help="the input is a .xex: print its header and exit")
    args = parser.parse_args(argv)

    try:
        if not os.path.exists(args.input):
            raise ExtractError("File not found: %s" % args.input)
        if args.info or args.input.lower().endswith(".xex"):
            print_xex_info(args.input)
            return 0
        xex = extract_image(args.input, args.output, list_only=args.list, xex_only=args.xex_only)
        if xex and os.path.exists(xex):
            print()
            print_xex_info(xex)
        elif xex:
            print("\nWarning: no default.xex at the root of the image. Check the listing with --list.")
    except ExtractError as error:
        print("error: %s" % error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
