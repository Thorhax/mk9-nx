#!/usr/bin/env python3
"""Find code pointers stored in data that land in unclaimed code gaps.

  usage: python3 tools/pointer_scan.py IMAGE_DUMP GAPS.toml [CODE_BASE CODE_END]

Input: a guest image dump (SA2_DUMP_IMAGE at runtime, or REXGLUE_DUMP_IMAGE during codegen) and the candidate list
that tools/find_gaps.py writes (out/gaps/candidates.toml). Scans every 4-byte aligned big-endian word outside the code
range; keeps values that are 4-aligned, inside the code range and inside a gap.
These are candidate function entries (vtables, callback tables, the CRT's static initializer table) that static
discovery missed. Candidates must still pass gap_classify validation.
The code range defaults to SA2's (the runtime logs it: "Function table initialized for module: code=...").
"""
import re, struct, sys
IMAGE_BASE = 0x82000000
CODE_BASE = int(sys.argv[3], 16) if len(sys.argv) > 3 else 0x82360000
CODE_SIZE = (int(sys.argv[4], 16) if len(sys.argv) > 4 else 0x83376864) - CODE_BASE
image = open(sys.argv[1], "rb").read()
gaps = []
for l in open(sys.argv[2]):
    m = re.match(r'"(0x[0-9A-F]+)" = \{ \}\s*#\s*(\d+) bytes', l)
    if m: gaps.append((int(m.group(1), 16), int(m.group(2))))
gaps.sort()
import bisect
starts = [g[0] for g in gaps]
def gap_of(v):
    i = bisect.bisect_right(starts, v) - 1
    if i >= 0 and gaps[i][0] <= v < gaps[i][0] + gaps[i][1]: return gaps[i]
    return None
code_lo, code_hi = CODE_BASE - IMAGE_BASE, CODE_BASE - IMAGE_BASE + CODE_SIZE
hits = {}
for off in range(0, len(image) - 3, 4):
    if code_lo <= off < code_hi: continue
    v = struct.unpack_from(">I", image, off)[0]
    if v & 3 or not (CODE_BASE <= v < CODE_BASE + CODE_SIZE): continue
    g = gap_of(v)
    if g: hits.setdefault(v, []).append(IMAGE_BASE + off)
for v in sorted(hits):
    g = gap_of(v)
    print(f"0x{v:08X} gap=0x{g[0]:08X}+{v-g[0]} size={g[1]} refs={len(hits[v])} first_ref=0x{hits[v][0]:08X}")
print(f"# {len(hits)} distinct pointer targets in gaps", file=sys.stderr)
