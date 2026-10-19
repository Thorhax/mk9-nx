#!/usr/bin/env python3
"""Wrap exact ring microcode in a conservative register-linked 2008 container.

This is for D3D-generated shaders for which no original container survived.  It does not guess a
shader identity: VS fetches/exports and PS interpolators are exposed by hardware register number.
"""

import argparse
import pathlib
import struct


def u32(data: bytes | bytearray, offset: int) -> int:
    return struct.unpack_from(">I", data, offset)[0]


def put(data: bytearray, offset: int, value: int) -> None:
    struct.pack_into(">I", data, offset, value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw", type=pathlib.Path)
    parser.add_argument("template", type=pathlib.Path)
    parser.add_argument("output", type=pathlib.Path)
    parser.add_argument("--stage", choices=("vs", "ps"), required=True)
    parser.add_argument("--depth", action="store_true", help="declare a pixel depth export")
    args = parser.parse_args()

    raw = args.raw.read_bytes()
    if not raw or len(raw) % 4:
        raise SystemExit("raw microcode must contain whole big-endian words")
    source = args.template.read_bytes()
    if len(source) < 36 or (u32(source, 0) & 0xFFFFFF00) != 0x102A1100:
        raise SystemExit("template is not a 2008 Xenos shader container")
    shader = u32(source, 24)
    if shader + (36 if args.stage == "vs" else 32) > u32(source, 4):
        raise SystemExit("template shader header is truncated")

    # Keep the known-good empty CTAB and definition area.  The physical part is replaced entirely.
    virtual_size = u32(source, 4)
    if args.stage == "ps":
        virtual_size = max(virtual_size, shader + 32 + 16 * 4)
    container = bytearray(virtual_size + len(raw))
    container[: min(u32(source, 4), len(source))] = source[: min(u32(source, 4), len(source))]
    put(container, 4, virtual_size)
    put(container, 8, len(raw))
    put(container, shader, 0)
    put(container, shader + 4, len(raw))

    # No semantic table is trusted. XenosRecomp's empty-CTAB path links these registers directly.
    put(container, shader + 20, (16 << 5) if args.stage == "ps" else 0)
    if args.stage == "vs":
        put(container, shader + 24, 0)
        put(container, shader + 28, 0)
        put(container, shader + 32, 0)
    else:
        put(container, shader + 24, 0)
        put(container, shader + 28, 0x11 if args.depth else 0x01)
        for register in range(16):
            put(container, shader + 32 + register * 4, register << 8)
    container[virtual_size:] = raw
    # sa2-nx: no constant table and no definition table, whatever the template had. With a template from the game
    # (a real CTAB) XenosRecomp would not link the wrapped shader by register (it then rejects every undeclared
    # fetch), and the template's DEF literals would be baked into a shader they do not belong to. Direct3D's own
    # shaders get their constants from the ring (SET_CONSTANT), which the native renderer passes through.
    put(container, 16, 0)  # constantTableOffset
    put(container, 20, 0)  # definitionTableOffset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(container)


if __name__ == "__main__":
    main()
