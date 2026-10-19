#!/usr/bin/env python3
"""A small PowerPC (Xenon) decoder for the analysis tools: just enough of the integer instruction set to follow how a
register gets its value (jump tables, bounds checks, constants built with lis/addi), and to tell which registers and
condition fields an instruction writes. Vector and floating-point instructions are reported as writing no GPR.

  from ppc_mini import Image, decode
  img = Image('out/default_image.bin', 0x82000000)
  ins = decode(img.word(0x82141714), 0x82141714)   # ins.op == 'bctr'
"""
import struct


class Image:
    def __init__(self, path, base):
        self.data = open(path, 'rb').read()
        self.base = base

    def contains(self, addr, size=4):
        return self.base <= addr and addr + size <= self.base + len(self.data)

    def word(self, addr):
        return struct.unpack_from('>I', self.data, addr - self.base)[0]

    def half(self, addr):
        return struct.unpack_from('>H', self.data, addr - self.base)[0]

    def byte(self, addr):
        return self.data[addr - self.base]


def _s16(x):
    return x - 0x10000 if x & 0x8000 else x


class Ins:
    """op: mnemonic family; rt/ra/rb/rs: fields; imm: immediate; writes: GPRs written; crw: CR fields written;
    target: branch target; bo/bi: branch fields; link: bl-type call."""
    __slots__ = ('addr', 'word', 'op', 'rt', 'ra', 'rb', 'imm', 'sh', 'mb', 'me', 'bo', 'bi', 'target', 'link',
                 'writes', 'crw', 'rc')

    def __init__(self, addr, word):
        self.addr, self.word = addr, word
        self.op = '?'
        self.rt = self.ra = self.rb = self.imm = self.sh = self.mb = self.me = None
        self.bo = self.bi = self.target = None
        self.link = False
        self.writes = ()
        self.crw = ()
        self.rc = False

    def __repr__(self):
        return f'{self.addr:08X} {self.op} rt={self.rt} ra={self.ra} rb={self.rb} imm={self.imm}'

    @property
    def unconditional(self):
        """b/blr/bctr (BO = 1z1zz): control never falls through."""
        if self.op == 'b':
            return not self.link
        if self.op in ('bclr', 'bcctr', 'bc'):
            return (self.bo & 0x14) == 0x14 and not self.link
        return False


# X-form (primary 31) extended opcodes whose destination is RA (logical, shifts, extends) vs RT (arithmetic, loads).
_X_DEST_RA = {28, 60, 124, 284, 316, 412, 444, 476, 24, 536, 792, 824, 954, 922, 986, 26, 27, 539, 794, 58}
_X_LOADS = {23: 'lwzx', 87: 'lbzx', 279: 'lhzx', 343: 'lhax', 21: 'ldx', 341: 'lwax', 55: 'lwzux', 119: 'lbzux',
            311: 'lhzux', 375: 'lhaux', 53: 'ldux', 373: 'lwaux', 20: 'lwarx', 84: 'ldarx', 534: 'lwbrx',
            790: 'lhbrx', 532: 'ldbrx'}
_X_UPDATE_LOADS = {55, 119, 311, 375, 53, 373}
_XO_ARITH = {266, 40, 104, 235, 491, 459, 138, 202, 136, 10, 8, 75, 11, 234, 200, 232, 233, 457, 489, 73, 9}
_X_STORE_UPDATE = {183, 247, 439, 181, 695, 759}  # stwux stbux sthux stdux stfsux stfdux: write RA


def decode(word, addr):
    i = Ins(addr, word)
    op = word >> 26
    rt, ra, rb = (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31
    imm = word & 0xFFFF
    i.rt, i.ra, i.rb = rt, ra, rb
    if op == 18:  # b
        li = word & 0x03FFFFFC
        if li & 0x02000000:
            li -= 0x04000000
        i.op = 'b'
        i.target = (li if word & 2 else addr + li) & 0xFFFFFFFF
        i.link = bool(word & 1)
        if i.link:
            i.writes = tuple([0] + list(range(3, 13)))
        return i
    if op == 16:  # bc
        bd = word & 0xFFFC
        if bd & 0x8000:
            bd -= 0x10000
        i.op = 'bc'
        i.bo, i.bi = rt, ra
        i.target = (bd if word & 2 else addr + bd) & 0xFFFFFFFF
        i.link = bool(word & 1)
        return i
    if op == 19:
        xo = (word >> 1) & 0x3FF
        i.bo, i.bi = rt, ra
        i.link = bool(word & 1)
        if xo == 16:
            i.op = 'bclr'
        elif xo == 528:
            i.op = 'bcctr'
            if i.link:
                i.writes = tuple([0] + list(range(3, 13)))
        elif xo in (257, 449, 193, 225, 33, 129, 289, 417):  # crand cror crxor crnand crnor crandc creqv crorc
            i.op = 'crop'
            i.crw = (rt >> 2,)
        elif xo == 0:
            i.op = 'mcrf'
            i.crw = (rt >> 2,)
        else:
            i.op = 'op19'
        return i
    if op == 15:
        i.op = 'lis' if ra == 0 else 'addis'
        i.imm = (imm << 16) & 0xFFFFFFFF
        i.writes = (rt,)
        return i
    if op == 14:
        i.op = 'li' if ra == 0 else 'addi'
        i.imm = _s16(imm)
        i.writes = (rt,)
        return i
    if op in (24, 25, 26, 27, 28, 29):  # ori oris xori xoris andi. andis.: RS=rt, dest RA
        i.op = {24: 'ori', 25: 'oris', 26: 'xori', 27: 'xoris', 28: 'andi.', 29: 'andis.'}[op]
        i.imm = imm
        i.writes = () if word == 0x60000000 else (ra,)  # nop = ori r0,r0,0 changes nothing
        if op in (28, 29):
            i.crw = (0,)
        return i
    if op in (20, 21, 23):  # rlwimi rlwinm rlwnm: RS=rt, dest RA
        i.op = {20: 'rlwimi', 21: 'rlwinm', 23: 'rlwnm'}[op]
        i.sh, i.mb, i.me = rb, (word >> 6) & 31, (word >> 1) & 31
        i.writes = (ra,)
        i.rc = bool(word & 1)
        if i.rc:
            i.crw = (0,)
        return i
    if op == 30:
        i.op = 'rld'
        i.writes = (ra,)
        if word & 1:
            i.crw = (0,)
        return i
    if op == 10 or op == 11:
        i.op = 'cmpli' if op == 10 else 'cmpi'
        i.imm = imm if op == 10 else _s16(imm)
        i.crw = (rt >> 2,)
        i.sh = (word >> 21) & 1  # L: 64-bit compare
        return i
    if op in (7, 8, 12, 13):  # mulli subfic addic addic.
        i.op = {7: 'mulli', 8: 'subfic', 12: 'addic', 13: 'addic.'}[op]
        i.imm = _s16(imm)
        i.writes = (rt,)
        if op == 13:
            i.crw = (0,)
        return i
    if op in (32, 33, 34, 35, 40, 41, 42, 43):  # lwz lwzu lbz lbzu lhz lhzu lha lhau
        i.op = {32: 'lwz', 33: 'lwzu', 34: 'lbz', 35: 'lbzu', 40: 'lhz', 41: 'lhzu', 42: 'lha', 43: 'lhau'}[op]
        i.imm = _s16(imm)
        i.writes = (rt, ra) if op & 1 else (rt,)
        return i
    if op == 58:  # ld ldu lwa
        i.op = 'ld'
        i.writes = (rt, ra) if word & 3 == 1 else (rt,)
        return i
    if op == 46:  # lmw
        i.op = 'lmw'
        i.writes = tuple(range(rt, 32))
        return i
    if op in (37, 39, 45, 53, 55, 62):  # stwu stbu sthu stfsu stfdu stdu
        i.op = 'stu'
        i.writes = (ra,) if op != 62 or word & 3 == 1 else ()
        return i
    if op in (36, 38, 44, 47, 52, 54, 62):
        i.op = 'st'
        return i
    if op in (48, 49, 50, 51):  # lfs lfsu lfd lfdu
        i.op = 'lf'
        i.writes = (ra,) if op & 1 else ()
        return i
    if op == 31:
        xo = (word >> 1) & 0x3FF
        xo9 = (word >> 1) & 0x1FF
        rc = bool(word & 1)
        if xo == 467:
            spr = ((word >> 16) & 31) | (((word >> 11) & 31) << 5)
            i.op = 'mtctr' if spr == 9 else ('mtlr' if spr == 8 else 'mtspr')
            return i
        if xo == 339:
            i.op = 'mfspr'
            i.writes = (rt,)
            return i
        if xo in (0, 32):
            i.op = 'cmp' if xo == 0 else 'cmpl'
            i.crw = (rt >> 2,)
            return i
        if xo == 144:
            i.op = 'mtcrf'
            i.crw = tuple(range(8))
            return i
        if xo == 19:
            i.op = 'mfcr'
            i.writes = (rt,)
            return i
        if xo in _X_LOADS:
            i.op = _X_LOADS[xo]
            i.writes = (rt, ra) if xo in _X_UPDATE_LOADS else (rt,)
            return i
        if xo in _X_STORE_UPDATE:
            i.op = 'stux'
            i.writes = (ra,)
            return i
        if xo == 444 and rt == rb:
            i.op = 'mr'
            i.writes = (ra,)
            if rc:
                i.crw = (0,)
            return i
        if xo in _X_DEST_RA or (word >> 2) & 0x1FF == 413:  # ... or sradi (XS-form)
            i.op = {28: 'and', 444: 'or', 316: 'xor', 24: 'slw', 536: 'srw', 792: 'sraw', 824: 'srawi',
                    954: 'extsb', 922: 'extsh', 986: 'extsw', 26: 'cntlzw'}.get(xo, 'logic')
            i.sh = rb
            i.writes = (ra,)
            if rc:
                i.crw = (0,)
            return i
        if xo9 in _XO_ARITH:
            i.op = {266: 'add', 40: 'subf', 104: 'neg'}.get(xo9, 'arith')
            i.writes = (rt,)
            if rc:
                i.crw = (0,)
            return i
        if xo in (4, 68):  # tw td
            i.op = 'tw'
            return i
        i.op = 'x31'
        return i
    if op in (2, 3):
        i.op = 'twi'
        return i
    i.op = 'other'
    return i


def is_nop(ins):
    return ins.word == 0x60000000


_BC_NAMES = {(12, 0): 'blt', (4, 0): 'bge', (12, 1): 'bgt', (4, 1): 'ble', (12, 2): 'beq', (4, 2): 'bne',
             (12, 3): 'bso', (4, 3): 'bns'}


def text(ins):
    """A rough disassembly line, for reports (not every form is spelled out)."""
    o, w = ins.op, ins.word
    if w == 0x60000000:
        return 'nop'
    if o == 'b':
        return f'{"bl" if ins.link else "b"} 0x{ins.target:08X}'
    if o in ('bc', 'bclr', 'bcctr'):
        suffix = {'bc': '', 'bclr': 'lr', 'bcctr': 'ctr'}[o] + ('l' if ins.link else '')
        if (ins.bo & 0x14) == 0x14:
            name = 'b' + suffix
        elif ins.bo & 0x04:
            name = _BC_NAMES.get((ins.bo & 0x1C, ins.bi & 3), f'bc{ins.bo},{ins.bi}') + suffix
            name += f' cr{ins.bi >> 2}'
        else:
            name = ('bdnz' if not ins.bo & 2 else 'bdz') + suffix + f' (bo={ins.bo},bi={ins.bi})'
        return name + (f', 0x{ins.target:08X}' if o == 'bc' else '')
    if o in ('lis', 'li'):
        return f'{o} r{ins.rt},0x{ins.imm & 0xFFFFFFFF:X}' if o == 'lis' else f'li r{ins.rt},{ins.imm}'
    if o in ('addi', 'addis', 'mulli', 'subfic', 'addic', 'addic.'):
        return f'{o} r{ins.rt},r{ins.ra},{ins.imm}'
    if o in ('ori', 'oris', 'xori', 'xoris', 'andi.', 'andis.'):
        return f'{o} r{ins.ra},r{ins.rt},0x{ins.imm:X}'
    if o in ('rlwinm', 'rlwimi', 'rlwnm'):
        return f'{o}{"." if ins.rc else ""} r{ins.ra},r{ins.rt},{ins.sh},{ins.mb},{ins.me}'
    if o in ('cmpli', 'cmpi'):
        return f'{"cmplwi" if o == "cmpli" else "cmpwi"}{"(64)" if ins.sh else ""} cr{ins.rt >> 2},r{ins.ra},{ins.imm}'
    if o in ('cmp', 'cmpl'):
        return f'{o}w cr{ins.rt >> 2},r{ins.ra},r{ins.rb}'
    if o in ('lwz', 'lwzu', 'lbz', 'lbzu', 'lhz', 'lhzu', 'lha', 'lhau'):
        return f'{o} r{ins.rt},{ins.imm}(r{ins.ra})'
    if o in ('lwzx', 'lbzx', 'lhzx', 'lhax', 'ldx', 'lwax', 'add', 'subf'):
        return f'{o} r{ins.rt},r{ins.ra},r{ins.rb}'
    if o == 'mr':
        return f'mr r{ins.ra},r{ins.rt}'
    if o in ('mtctr', 'mtlr'):
        return f'{o} r{ins.rt}'
    if o == 'mfspr':
        spr = ((w >> 16) & 31) | (((w >> 11) & 31) << 5)
        return {8: f'mflr r{ins.rt}', 9: f'mfctr r{ins.rt}'}.get(spr, f'mfspr r{ins.rt},{spr}')
    p = w >> 26
    if p in (36, 37, 38, 39, 44, 45):
        n = {36: 'stw', 37: 'stwu', 38: 'stb', 39: 'stbu', 44: 'sth', 45: 'sth'}[p]
        return f'{n} r{ins.rt},{(w & 0xFFFF) - (0x10000 if w & 0x8000 else 0)}(r{ins.ra})'
    if p in (58, 62):
        ds = (w & 0xFFFC) - (0x10000 if w & 0x8000 else 0)
        n = ('ld', 'ldu', 'lwa')[w & 3] if p == 58 else ('std', 'stdu', '?')[min(w & 3, 2)]
        return f'{n} r{ins.rt},{ds}(r{ins.ra})'
    if p in (48, 49, 50, 51, 52, 53, 54, 55):
        n = {48: 'lfs', 49: 'lfsu', 50: 'lfd', 51: 'lfdu', 52: 'stfs', 53: 'stfsu', 54: 'stfd', 55: 'stfdu'}[p]
        return f'{n} f{ins.rt},{(w & 0xFFFF) - (0x10000 if w & 0x8000 else 0)}(r{ins.ra})'
    return f'{o} [{w:08X}]'
