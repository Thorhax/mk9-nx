#!/usr/bin/env python3
"""Keeps only the scanned shader containers (out/tools/ue3_shader_scan) whose microcode no runtime container has.

The game rewrites a container's header when it loads it, so the copy in the Unreal package and the one the game
created can share their microcode but differ in the header (vertex element table, interpolators). With both in the
package, identification by microcode can pick the package copy for a draw: in MK9 (2026-10-20) whole stages came
out as exploded colored triangles. usf4-nx made the same rule for its EMZ shaders (extract_game_shaders.py --skip).

  python3 tools/filter_scanned_shaders.py --scan out/ue3scan --runtime out/pc/shaders <missing folders...> \
      --out out/ue3scan_filtered
"""
import argparse
import glob
import hashlib
import os
import shutil
import struct


def key(path):
    data = open(path, 'rb').read()
    virtual, physical = struct.unpack_from('>II', data, 4)
    return os.path.basename(path)[:2], hashlib.md5(data[virtual:virtual + physical]).digest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True, help='folder of scanned containers')
    ap.add_argument('--runtime', nargs='+', required=True, help='folders of containers the game created or bound')
    ap.add_argument('--out', required=True, help='output folder (emptied first)')
    a = ap.parse_args()
    have = {key(p) for d in a.runtime for p in glob.glob(os.path.join(d, '*.bin'))}
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    kept = skipped = 0
    for p in sorted(glob.glob(os.path.join(a.scan, '*.bin'))):
        if key(p) in have:
            skipped += 1
            continue
        shutil.copy(p, a.out)
        kept += 1
    print(f'kept {kept}, skipped {skipped} (their microcode is in a runtime container)')


if __name__ == '__main__':
    main()
