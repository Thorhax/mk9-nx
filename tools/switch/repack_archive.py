#!/usr/bin/env python3
"""Rebuilds a static library with unique member names and a fresh symbol index, for the Mesa Horizon SDK's
libvulkan.a.

Why: the SDK's libvulkan.a bundles two Rust static libraries (NAK and NIL), each with its own copy of Rust's runtime
crates, so the archive holds members with the same name (core-....cgu.0.rcgu.o twice, std, alloc, ...). devkitA64's
GNU ld 2.45.1 then fails to pull the runtime objects the index lists: 4,939 "undefined reference to
core::panicking::panic ..." and "hidden symbol __rustc::__rdl_realloc isn't defined", even for a program that only
calls vkCreateInstance (usf4-nx, 2026-10-11). Passing the same objects explicitly resolves them. Repacking the
members, in their original order, under unique names gives an archive ld reads correctly. No object is changed.

  python3 tools/switch/repack_archive.py <in.a> <out.a> [--ar aarch64-none-elf-ar]

Reads the GNU/SysV ar format directly (global header, 60-byte member headers, the "//" long-name table, the "/" or
"/SYM64/" symbol table is dropped and rebuilt by `ar rcs`).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile


def members(data):
    if data[:8] != b"!<arch>\n":
        sys.exit("not an ar archive")
    pos, longnames = 8, b""
    while pos + 60 <= len(data):
        header = data[pos:pos + 60]
        name = header[0:16].decode("latin1").rstrip()
        size = int(header[48:58].decode("latin1").strip())
        body = data[pos + 60:pos + 60 + size]
        pos += 60 + size + (size & 1)
        if name in ("/", "/SYM64/"):
            continue  # symbol index: rebuilt
        if name == "//":
            longnames = body
            continue
        if name.startswith("/") and name[1:].isdigit():
            start = int(name[1:])
            end = longnames.index(b"/\n", start)
            name = longnames[start:end].decode("latin1")
        elif name.endswith("/"):
            name = name[:-1]
        yield name, body


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--ar", default="ar")
    a = ap.parse_args()
    data = open(a.input, "rb").read()
    work = tempfile.mkdtemp(prefix="repack_")
    try:
        names, seen = [], {}
        for index, (name, body) in enumerate(members(data)):
            count = seen.get(name, 0)
            seen[name] = count + 1
            base, ext = os.path.splitext(os.path.basename(name))
            unique = f"{index:05d}_{base}{'_dup' + str(count) if count else ''}{ext}"
            with open(os.path.join(work, unique), "wb") as f:
                f.write(body)
            names.append(unique)
        duplicates = sum(1 for n, c in seen.items() if c > 1)
        if os.path.exists(a.output):
            os.remove(a.output)
        # In order, in chunks (command-line length), then one index for all.
        out = os.path.abspath(a.output)
        for i in range(0, len(names), 200):
            subprocess.run([a.ar, "q", out] + names[i:i + 200], cwd=work, check=True)
        subprocess.run([a.ar, "s", out], check=True)
        print(f"{len(names)} members ({duplicates} names that were duplicated) -> {a.output}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
