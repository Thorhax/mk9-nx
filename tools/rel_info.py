"""Reads SA2's REL overlay modules (gd/*.rel): the GameCube/Wii relocatable module format, kept by the Xbox 360 port.

Stage code is not in default.xex: the game loads these modules at runtime, relocates them against the main
executable (module 0) and calls their prolog. A static recompiler only sees default.xex, so every module's code has
to be recompiled as well, at a fixed load address. This tool shows what is in them.

Usage:
  python tools/rel_info.py assets/game_root/gd                summary of every module
  python tools/rel_info.py assets/game_root/gd/stg13D.rel -v  sections, imports and relocation types of one module
"""
import argparse
import collections
import glob
import os
import struct

# REL header (big-endian), version 1..3. Offsets in bytes.
HDR = struct.Struct('>IIIIIIIIIIIIBBBBIII')
HDR_FIELDS = ('id', 'next', 'prev', 'num_sections', 'section_info_offset', 'name_offset', 'name_size', 'version',
              'bss_size', 'rel_offset', 'imp_offset', 'imp_size', 'prolog_section', 'epilog_section',
              'unresolved_section', 'bss_section', 'prolog', 'epilog', 'unresolved')

RELOC_NAMES = {
    0: 'R_PPC_NONE', 1: 'R_PPC_ADDR32', 2: 'R_PPC_ADDR24', 3: 'R_PPC_ADDR16', 4: 'R_PPC_ADDR16_LO',
    5: 'R_PPC_ADDR16_HI', 6: 'R_PPC_ADDR16_HA', 7: 'R_PPC_ADDR14', 8: 'R_PPC_ADDR14_BRTAKEN',
    9: 'R_PPC_ADDR14_BRNTAKEN', 10: 'R_PPC_REL24', 11: 'R_PPC_REL14', 12: 'R_PPC_REL14_BRTAKEN',
    13: 'R_PPC_REL14_BRNTAKEN', 201: 'R_DOLPHIN_NOP', 202: 'R_DOLPHIN_SECTION', 203: 'R_DOLPHIN_END',
    204: 'R_DOLPHIN_MRKREF',
}


class Rel:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            self.data = f.read()
        self.h = dict(zip(HDR_FIELDS, HDR.unpack_from(self.data, 0)))
        if self.h['version'] >= 2:
            self.h['align'], self.h['bss_align'] = struct.unpack_from('>II', self.data, 0x40)
        if self.h['version'] >= 3:
            self.h['fix_size'] = struct.unpack_from('>I', self.data, 0x48)[0]
        self.sections = []
        for i in range(self.h['num_sections']):
            off, size = struct.unpack_from('>II', self.data, self.h['section_info_offset'] + 8 * i)
            self.sections.append({'index': i, 'offset': off & ~1, 'exec': bool(off & 1), 'size': size})
        self.imports = []
        for i in range(self.h['imp_size'] // 8):
            mod, off = struct.unpack_from('>II', self.data, self.h['imp_offset'] + 8 * i)
            self.imports.append({'module': mod, 'offset': off})

    def relocations(self, imp):
        """Yields (section, offset_in_section, type, target_section, addend) for one import table entry."""
        pos = imp['offset']
        section = None
        offset = 0
        while True:
            delta, rtype, tsec, addend = struct.unpack_from('>HBBI', self.data, pos)
            pos += 8
            if rtype == 203:  # R_DOLPHIN_END
                return
            offset += delta
            if rtype == 202:  # R_DOLPHIN_SECTION
                section = tsec
                offset = 0
                continue
            if rtype == 201:  # R_DOLPHIN_NOP: only advances the offset
                continue
            yield section, offset, rtype, tsec, addend

    def code_bytes(self):
        return sum(s['size'] for s in self.sections if s['exec'] and s['offset'])


def summary(paths):
    total_code = total_file = 0
    types = collections.Counter()
    import_modules = collections.Counter()
    versions = collections.Counter()
    print(f"{'module':<26}{'id':>4}{'ver':>4}{'file':>10}{'code':>10}{'bss':>9}  imports")
    for p in paths:
        r = Rel(p)
        versions[r.h['version']] += 1
        code = r.code_bytes()
        total_code += code
        total_file += len(r.data)
        mods = sorted({i['module'] for i in r.imports})
        for m in mods:
            import_modules[m] += 1
        for imp in r.imports:
            for _, _, t, _, _ in r.relocations(imp):
                types[RELOC_NAMES.get(t, t)] += 1
        print(f"{os.path.basename(p):<26}{r.h['id']:>4}{r.h['version']:>4}{len(r.data):>10}{code:>10}"
              f"{r.h['bss_size']:>9}  {mods}")
    print(f'\n{len(paths)} modules, {total_file / 2**20:.1f} MiB of files, {total_code / 2**20:.1f} MiB of code')
    print('versions:', dict(versions))
    print('modules imported from (module id: count of importers):', dict(sorted(import_modules.items())))
    print('relocation types:', dict(types.most_common()))


def verbose(path):
    r = Rel(path)
    for k, v in r.h.items():
        print(f'  {k:<20} {v:#x}' if isinstance(v, int) else f'  {k:<20} {v}')
    print('sections:')
    for s in r.sections:
        if s['size']:
            print(f"  [{s['index']:2}] off {s['offset']:#09x} size {s['size']:#09x} {'EXEC' if s['exec'] else ''}"
                  f"{' (bss)' if not s['offset'] else ''}")
    print('imports:')
    for imp in r.imports:
        types = collections.Counter(RELOC_NAMES.get(t, t) for _, _, t, _, _ in r.relocations(imp))
        print(f"  module {imp['module']}: {dict(types)}")
        if imp['module'] == 0:
            sample = [a for _, _, _, _, a in r.relocations(imp)][:8]
            print('    first targets:', ' '.join(f'{a:08X}' for a in sample))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('-v', action='store_true')
    a = ap.parse_args()
    if os.path.isdir(a.path):
        summary(sorted(glob.glob(os.path.join(a.path, '*.rel'))))
    elif a.v:
        verbose(a.path)
    else:
        summary([a.path])


if __name__ == '__main__':
    main()
