#!/usr/bin/env python3
"""Constants the game loads from its read-only data, folded at build time (post-codegen step of the "optimised"
codegen flavour; docs/enhancements.md, "The second CPU round of 2026-10-12"). Works on any ReXGlue port.

The 360's compiler keeps every float literal in .rdata and loads it through an address it builds with `lis`:
    lis r11,-32241               r11.s64 = -2112946176;
    lfs f31,-21952(r11)          temp.u32 = REX_LOAD_D_U32(r11.u32, -21952);  f31.f64 = double(temp.f32);
The recompiled code really loads it from guest memory every time: a load, a byte swap, a move to the vector unit and
a conversion (about 15 cycles on the Cortex-A57 before the value can be used), for a number that never changes:
.rdata is read-only in the executable (the section's flags say so). One kind of word in it is not constant all the
same: the XEX's import records, which stand at the start of .rdata and get, for an imported kernel variable, the
address the loader's kernel gave it. With --xex they are read from the XEX header and a load of one stays a load.

This step follows the constants `lis` / `addi` / `ori` put in registers along each straight line of a function (a
label forgets everything, a call forgets the registers that live in ctx, any other write to a register forgets it)
and, where a D-form load's address is then known and lies wholly inside a read-only data section of the image,
replaces the load by REX_RO_U32(address, value) / REX_RO_U64 / ... with the bytes found there:
  - on the Switch: the value. The compiler then folds the conversion too and often the arithmetic behind it;
  - elsewhere (the PC build): the load as before, compared with the value (RexRoMismatch reports a difference, a
    counter counts the checks): every PC run tests every folded constant it reaches, and behaves exactly as before;
  - with REX_RO_CONSTANTS_OFF: the plain load.
Float (`lfs`), double (`lfd`) and, with --ints (the default), integer loads (`lwz` / `lhz` / `lbz` / `ld`) are
folded; an integer folded into a register is itself a known constant for what follows.

Third round (2026-10-12): vector loads too. `lvx vN, rA, rB` with both registers known reads 16 bytes at a known
address; when they lie in the read-only data the load and its byte reversal become REX_RO_V128(address, high, low):
the reversed bytes as a constant on the Switch (one load from the literal pool instead of a guest load, REV64 and
EXT), the load compared with it in the PC build. Vector code keeps its masks and float constants this way.

  python tools/ro_constants.py --gen app/generated/default --image out/default_image.bin [--check] [--no-ints]
The image is the dump the code generator writes with REXGLUE_DUMP_IMAGE=out/default_image.bin (tools/codegen.ps1 sets
it); its size must be the pch's REX_IMAGE_SIZE. Idempotent (a folded load no longer matches). Run it before
tools/float_loads.py and the localizers.
"""
import argparse
import os
import re
import struct
import sys

FUNC = re.compile(r'^DEFINE_REX_FUNC\((\w+)\) \{$')
LABEL = re.compile(r'^loc_[0-9A-Fa-f]+:')
REGNAME = r'(?:ctx\.)?r\d+'
CONST = re.compile(r'^\t(' + REGNAME + r')\.s64 = (-?\d+);$')
ADDI = re.compile(r'^\t(' + REGNAME + r')\.s64 = (' + REGNAME + r')\.s64 \+ (-?\d+);$')
ORI = re.compile(r'^\t(' + REGNAME + r')\.u64 = (' + REGNAME + r')\.u64 \| (\d+);$')
FLOAT = re.compile(r'^\t((?:ctx\.)?temp)\.u32 = REX_LOAD_D_U32\((' + REGNAME + r')\.u32, (-?\d+)\);$')
FLOAT_USE = re.compile(r'\.f64 = (?:double\((?:ctx\.)?temp\.f32\)|rex::ppc::SingleToDoubleAnyMode\((?:ctx\.)?temp\.u32\));$')
DOUBLE = re.compile(r'^\t((?:ctx\.)?f\d+)\.u64 = REX_LOAD_D_U64\((' + REGNAME + r')\.u32, (-?\d+)\);$')
INT = re.compile(r'^\t(' + REGNAME + r')\.u64 = REX_LOAD_D_(U8|U16|U32|U64)\((' + REGNAME + r')\.u32, (-?\d+)\);$')
# lvx: the aligned effective address, then the load with its byte reversal.
EA_SUM = re.compile(r'^\tea = \((' + REGNAME + r')\.u32 \+ (' + REGNAME + r')\.u32\) & ~0xF;$')
EA_ONE = re.compile(r'^\tea = \((' + REGNAME + r')\.u32\) & ~0xF;$')
VECTOR = 'REX_BSWAP128(simde_mm_load_si128((simde__m128i*)REX_RAW_ADDR(ea)))'
# Every mention of a general register, with what follows it: a field and maybe an assignment operator.
MENTION = re.compile(r'(?<![\w.])(' + REGNAME + r')(?!\d)(\.\w+)?(\s*(?:[-+*/|&^%]|<<|>>)?=(?!=))?')
CALL = re.compile(r'\(ctx, base[,)]|REX_CALL_INDIRECT_|ppc_trap\(')
STRUCTURED = re.compile(r'^\t+(?:switch|for|while|do|else|case|default|try|catch)\b|[{}]\s*$|^\t*\}')
SIZE = {'U8': 1, 'U16': 2, 'U32': 4, 'U64': 8}
MARKER = '// rexglue ports: constants of the read-only data folded (tools/ro_constants.py)'
MACROS = MARKER + '''
#ifndef REX_RO_U32
#if defined(REX_RO_CONSTANTS_OFF)
#define REX_RO_U8(a, v) REX_LOAD_U8(a)
#define REX_RO_U16(a, v) REX_LOAD_U16(a)
#define REX_RO_U32(a, v) REX_LOAD_U32(a)
#define REX_RO_U64(a, v) REX_LOAD_U64(a)
#elif defined(__SWITCH__)
#define REX_RO_U8(a, v) ((u8)(v))
#define REX_RO_U16(a, v) ((u16)(v))
#define REX_RO_U32(a, v) ((u32)(v))
#define REX_RO_U64(a, v) ((u64)(v))
#else
extern "C" void RexRoMismatch(unsigned address) noexcept;
extern "C" volatile unsigned long long g_rex_ro_checked;
template <typename T>
[[gnu::always_inline]] static inline T RexRoCheck(T memory, T constant, unsigned address) noexcept {
  g_rex_ro_checked = g_rex_ro_checked + 1;
  if (memory != constant) [[unlikely]] RexRoMismatch(address);
  return memory;
}
#define REX_RO_U8(a, v) RexRoCheck<u8>(REX_LOAD_U8(a), (u8)(v), (a))
#define REX_RO_U16(a, v) RexRoCheck<u16>(REX_LOAD_U16(a), (u16)(v), (a))
#define REX_RO_U32(a, v) RexRoCheck<u32>(REX_LOAD_U32(a), (u32)(v), (a))
#define REX_RO_U64(a, v) RexRoCheck<u64>(REX_LOAD_U64(a), (u64)(v), (a))
#endif
#endif
'''
MARKER_V = '// rexglue ports: vector constants of the read-only data folded (tools/ro_constants.py)'
MACROS_V = MARKER_V + '''
#ifndef REX_RO_V128
#if defined(REX_RO_CONSTANTS_OFF)
#define REX_RO_V128(a, hi, lo) REX_BSWAP128(simde_mm_load_si128((simde__m128i*)REX_RAW_ADDR(a)))
#elif defined(__SWITCH__)
#define REX_RO_V128(a, hi, lo) simde_mm_set_epi64x((int64_t)(hi), (int64_t)(lo))
#else
[[gnu::always_inline]] static inline simde__m128i RexRoCheckV128(simde__m128i memory, simde__m128i constant,
                                                                 unsigned address) noexcept {
  g_rex_ro_checked = g_rex_ro_checked + 1;
  if (simde_mm_movemask_epi8(simde_mm_cmpeq_epi8(memory, constant)) != 0xFFFF) [[unlikely]] RexRoMismatch(address);
  return memory;
}
#define REX_RO_V128(a, hi, lo)                                                        \\
  RexRoCheckV128(REX_BSWAP128(simde_mm_load_si128((simde__m128i*)REX_RAW_ADDR(a))),   \\
                 simde_mm_set_epi64x((int64_t)(hi), (int64_t)(lo)), (a))
#endif
#endif
'''


def read_only_ranges(image):
    """[(start, end)] of the image's read-only data sections, as guest addresses, and the image base."""
    pe = struct.unpack_from('<I', image, 0x3C)[0]
    if image[:2] != b'MZ' or image[pe:pe + 4] != b'PE\0\0':
        sys.exit('ERROR (post-codegen patch): ro_constants: the image dump is not a PE image')
    sections, opt_size = struct.unpack_from('<H', image, pe + 6)[0], struct.unpack_from('<H', image, pe + 20)[0]
    base = struct.unpack_from('<I', image, pe + 24 + 28)[0]
    table = pe + 24 + opt_size
    ranges = []
    for i in range(sections):
        name, vsize, va, _, _, _, _, _, _, flags = struct.unpack_from('<8sIIIIIIHHI', image, table + 40 * i)
        name = name.rstrip(b'\0').decode('latin-1')
        # Readable, not writable, not code; .rdata only (the unwind and relocation tables hold no constants).
        if name == '.rdata' and flags & 0x40000000 and not flags & 0x80000000 and not flags & 0x20000020:
            ranges.append((base + va, base + va + vsize))
    return ranges, base


def import_records(xex):
    """The addresses of a XEX's import records (optional header 0x000103FF): words inside the image, in these
    executables at the start of .rdata, that the loader writes at run time: for an imported kernel VARIABLE the
    address the kernel gave it in that process (sdk/src/system/xex_module.cpp). The image dump holds what the code
    generator's own process got, so such a word is no constant of the game (found by the futari-nx session,
    2026-10-14: sa2-nx's and banjo-nx's trees had 86 and 77 such loads folded, this one 79)."""
    d = open(xex, 'rb').read()
    if d[:4] != b'XEX2':
        sys.exit('ERROR (post-codegen patch): ro_constants: %s is not a XEX2 file' % xex)
    out = set()
    for i in range(struct.unpack_from('>I', d, 20)[0]):
        key, value = struct.unpack_from('>II', d, 24 + 8 * i)
        if key != 0x000103FF:
            continue
        _, strings, libraries = struct.unpack_from('>III', d, value)
        p = value + 12 + strings
        for _ in range(libraries):
            size = struct.unpack_from('>I', d, p)[0]
            count = struct.unpack_from('>H', d, p + 38)[0]
            out.update(struct.unpack_from('>%dI' % count, d, p + 40))
            p += size
    return out


RO_USE = re.compile(r'REX_RO_(U8|U16|U32|U64)\(0x([0-9A-Fa-f]{8}), 0x[0-9A-Fa-f]+(?:ull|u)\)')
RO_VECTOR = re.compile(r'REX_RO_V128\(0x([0-9A-Fa-f]{8}), ')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gen', required=True, help='generated folder (*_recomp.N.cpp and *_pch.h)')
    ap.add_argument('--image', required=True, help="the code generator's image dump (REXGLUE_DUMP_IMAGE)")
    ap.add_argument('--xex', help="the module's XEX: loads of its import records stay loads (and a tree folded "
                                  "without this option gets them back)")
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--no-ints', action='store_true', help='fold float and double loads only')
    a = ap.parse_args()
    records = import_records(a.xex) if a.xex else set()

    def on_record(address, size):
        return any(((address + k) & ~3) in records for k in range(size))
    files = sorted(f for f in os.listdir(a.gen) if re.match(r'\w+_recomp\.\d+\.cpp$', f))
    pchs = [f for f in os.listdir(a.gen) if f.endswith('_pch.h')]
    if not files or len(pchs) != 1:
        sys.exit('ERROR (post-codegen patch): ro_constants: no *_recomp.N.cpp or no single *_pch.h in %s' % a.gen)
    if not os.path.exists(a.image):
        sys.exit('ERROR (post-codegen patch): ro_constants: %s not found: run tools\\codegen.ps1 (it writes the dump)'
                 % a.image)
    image = open(a.image, 'rb').read()
    pch_path = os.path.join(a.gen, pchs[0])
    pch = open(pch_path, encoding='utf-8').read()
    ranges, base = read_only_ranges(image)
    m = re.search(r'#define REX_IMAGE_BASE (0x[0-9A-Fa-f]+)', pch)
    s = re.search(r'#define REX_IMAGE_SIZE (0x[0-9A-Fa-f]+)', pch)
    if not m or not s or int(m.group(1), 16) != base or int(s.group(1), 16) != len(image) or not ranges:
        sys.exit('ERROR (post-codegen patch): ro_constants: %s is not the image this code was generated from' % a.image)

    def constant(address, size):
        address &= 0xFFFFFFFF
        if on_record(address, size):
            return None
        for start, end in ranges:
            if start <= address and address + size <= end:
                return address, int.from_bytes(image[address - base:address - base + size], 'big')
        return None

    stats = dict(floats=0, doubles=0, ints=0, vectors=0, already=0, changed=0, records=0)

    def unfold(m):
        if not on_record(int(m.group(2), 16), SIZE[m.group(1)]):
            return m.group(0)
        stats['records'] += 1
        return 'REX_LOAD_%s(0x%s)' % (m.group(1), m.group(2))

    for f in files:
        path = os.path.join(a.gen, f)
        lines = open(path, encoding='utf-8').read().split('\n')
        known = {}
        ea = None  # the address `ea` holds, when both registers of an lvx were known
        dirty = False
        for i, line in enumerate(lines):
            if not line.startswith('\t'):
                known.clear()  # a label, a function's first or last line
                ea = None
                continue
            if line.startswith('\t//'):
                continue
            if records and 'REX_RO_' in line:
                # A tree folded before --xex existed: a load of an import record goes back to a load.
                new = RO_USE.sub(unfold, line)
                m = RO_VECTOR.search(new)
                if m and on_record(int(m.group(1), 16), 16):
                    sys.exit('ERROR (post-codegen patch): ro_constants: a folded vector load at %s covers an import '
                             'record (%s): regenerate the code' % (m.group(1), f))
                if new != line:
                    lines[i] = line = new
                    dirty = True
            if 'REX_RO_U' in line or 'REX_RO_V128' in line:
                stats['already'] += 1
            if line.startswith('\tea = '):
                m = EA_SUM.match(line)
                if m and m.group(1) in known and m.group(2) in known:
                    ea = (known[m.group(1)] + known[m.group(2)]) & 0xFFFFFFF0
                else:
                    m = EA_ONE.match(line)
                    ea = known[m.group(1)] & 0xFFFFFFF0 if m and m.group(1) in known else None
                continue
            if ea is not None and VECTOR in line:
                c = constant(ea, 16)
                if c:
                    lines[i] = line.replace(VECTOR, 'REX_RO_V128(0x%08X, 0x%016Xull, 0x%016Xull)' %
                                            (c[0], c[1] >> 64, c[1] & 0xFFFFFFFFFFFFFFFF))
                    stats['vectors'] += 1
                    dirty = True
                    line = lines[i]
            m = CONST.match(line)
            if m:
                known[m.group(1)] = int(m.group(2))
                continue
            m = ADDI.match(line)
            if m and m.group(2) in known:
                known[m.group(1)] = known[m.group(2)] + int(m.group(3))
                continue
            m = ORI.match(line)
            if m and m.group(2) in known:
                known[m.group(1)] = known[m.group(2)] | int(m.group(3))
                continue
            if STRUCTURED.search(line):
                known.clear()
                ea = None
                continue
            m = FLOAT.match(line)
            if m and m.group(2) in known and i + 1 < len(lines) and FLOAT_USE.search(lines[i + 1]):
                c = constant(known[m.group(2)] + int(m.group(3)), 4)
                if c:
                    lines[i] = '\t%s.u32 = REX_RO_U32(0x%08X, 0x%08X);' % (m.group(1), c[0], c[1])
                    stats['floats'] += 1
                    dirty = True
                continue
            m = DOUBLE.match(line)
            if m and m.group(2) in known:
                c = constant(known[m.group(2)] + int(m.group(3)), 8)
                if c:
                    lines[i] = '\t%s.u64 = REX_RO_U64(0x%08X, 0x%016Xull);' % (m.group(1), c[0], c[1])
                    stats['doubles'] += 1
                    dirty = True
                continue
            m = INT.match(line)
            if m and not a.no_ints and m.group(3) in known:
                c = constant(known[m.group(3)] + int(m.group(4)), SIZE[m.group(2)])
                if c:
                    lines[i] = '\t%s.u64 = REX_RO_%s(0x%08X, 0x%X%s);' % (
                        m.group(1), m.group(2), c[0], c[1], 'ull' if m.group(2) == 'U64' else 'u')
                    known[m.group(1)] = c[1]  # the register now holds a known value too
                    stats['ints'] += 1
                    dirty = True
                    continue
            # Anything else: forget every register this statement may write (an assignment to it, or a mention
            # of the whole register), and after a call the registers that live in ctx.
            for mm in MENTION.finditer(line):
                if mm.group(1) in known and (mm.group(3) or not mm.group(2)):
                    del known[mm.group(1)]
            if 'REX_LR_READ(' in line:
                for name in re.findall(r'REX_LR_READ\((' + REGNAME + r')\)', line):
                    known.pop(name, None)
            if CALL.search(line):
                for name in [k for k in known if k.startswith('ctx.')]:
                    del known[name]
        if dirty:
            stats['changed'] += 1
            if not a.check:
                open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
    anchor = '#define REX_MEMORY_SIZE'
    if MARKER not in pch or MARKER_V not in pch:
        if anchor not in pch:
            sys.exit('ERROR (post-codegen patch): ro_constants: no REX_MEMORY_SIZE in %s: the generated pch changed'
                     % pch_path)
        new = pch
        # Each block goes right in front of the anchor, so the scalar one first: the vector block uses its names.
        if MARKER not in new:
            new = new.replace(anchor, MACROS + anchor, 1)
        if MARKER_V not in new:
            new = new.replace(anchor, MACROS_V + anchor, 1)
        if not a.check:
            open(pch_path, 'w', encoding='utf-8', newline='\n').write(new)
    print('ro_constants%s: %d float, %d double, %d integer and %d vector loads from read-only data folded (%s; %d '
          'already folded), %d files changed' %
          (' (check only)' if a.check else '', stats['floats'], stats['doubles'], stats['ints'], stats['vectors'],
           ', '.join('%08X-%08X' % r for r in ranges), stats['already'], stats['changed']))
    if a.xex:
        print('ro_constants: %d import records in %s; %d folded loads of one put back as loads' %
              (len(records), os.path.basename(a.xex), stats['records']))
    else:
        print('ro_constants: WARNING: no --xex: loads of import records (kernel variables, whose addresses the '
              'loader writes at run time) are folded like constants')


if __name__ == '__main__':
    main()
