#!/usr/bin/env python3
"""List the PM4 type-3 packet headers a generated function builds (lis/ori constants).
  tools/pm4_ops.py sub_XXXXXXXX [...]"""
import re, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import callgraph as c

OPS = {0x10: "NOP", 0x22: "DRAW_INDX", 0x36: "DRAW_INDX_2", 0x2D: "SET_CONSTANT", 0x55: "SET_CONSTANT2",
       0x2F: "LOAD_ALU_CONSTANT", 0x56: "SET_SHADER_CONSTANTS", 0x27: "IM_LOAD", 0x2B: "IM_LOAD_IMMEDIATE",
       0x46: "EVENT_WRITE", 0x3B: "INVALIDATE_STATE", 0x3C: "WAIT_REG_MEM", 0x3D: "MEM_WRITE",
       0x3E: "REG_RMW", 0x48: "ME_INIT", 0x58: "EVENT_WRITE_SHD", 0x59: "EVENT_WRITE_EXT",
       0x21: "WAIT_FOR_IDLE", 0x3F: "COND_WRITE", 0x5A: "CONTEXT_UPDATE", 0x5E: "VIZ_QUERY",
       0x35: "INDIRECT_BUFFER_PFD", 0x3F: "INDIRECT_BUFFER", 0x44: "SET_BIN_MASK_LO", 0x57: "SET_BIN_SELECT_LO",
       0x4B: "SET_SHADER_BASES?", 0x2A: "INVALIDATE_STATE?", 0x26: "INTERRUPT"}
LIS = re.compile(r"// lis (r\d+),(-?\d+)$")
ORI = re.compile(r"// ori (r\d+),(r\d+),(\d+)$")
ADDI = re.compile(r"// addi (r\d+),(r\d+),(-?\d+)$")

F = c.load()
for name in sys.argv[1:]:
    hi = {}
    found = []
    for line in c.listing(name, F[name]):
        s = line.strip()
        m = LIS.match(s)
        if m:
            hi[m.group(1)] = (int(m.group(2)) & 0xFFFF) << 16
            continue
        m = ORI.match(s) or ADDI.match(s)
        if m and m.group(2) in hi:
            v = (hi[m.group(2)] + (int(m.group(3)) if s.startswith("// addi") else int(m.group(3)))) & 0xFFFFFFFF
            if (v >> 30) == 3:
                op = (v >> 8) & 0x7F
                found.append(f"{OPS.get(op, hex(op))}[{((v >> 16) & 0x3FFF) + 1}]")
    print(name, " ".join(found) if found else "-")
