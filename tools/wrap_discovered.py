#!/usr/bin/env python3
"""Wraps the raw ring microcode the native renderer discovered (SA2_SHADER_DISCOVERY: <stage>_<hash>.ucode)
into register-linked containers for the shader library, but ONLY the shaders no real container covers.

Why the filter matters: a raw discovery is often a vertex shader Direct3D rewrote for a vertex declaration, whose
real container also exists (the dump from CreateVertexShader, a variant repaired by repair_vertex_variant_declarations,
or the rewritten container SA2_SHADER_MISSING keeps). Packed together, the library's code index prefers the
register-linked wrapper when two entries share their microcode, and a register-linked vertex shader paired with the
game's semantic-linked pixel shader passes its interpolators by the wrong location: SA2's health warning, SEGA and Sonic
Team screens came out black (the texture coordinates arrived as zero). Only Direct3D's own shaders, which have no
container anywhere, are wrapped.

  python tools/wrap_discovered.py --raw out/pc/shaders_discovery out/pc/shaders_discovery2 \
      --containers out/pc/shaders out/pc/shaders_missing out/pc/shaders_repaired --out out/pc/shaders_wrapped
"""
import argparse
import glob
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def physical_part(container):
    if len(container) < 12:
        return b''
    virtual = struct.unpack_from('>I', container, 4)[0]
    physical = struct.unpack_from('>I', container, 8)[0]
    return container[virtual:virtual + physical]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', nargs='+', required=True, help='folders with <stage>_<hash>.ucode files')
    ap.add_argument('--containers', nargs='+', required=True, help='folders with real containers (*.bin)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    covered = []
    for d in a.containers:
        for p in glob.glob(os.path.join(d, '*.bin')):
            covered.append((os.path.basename(p), physical_part(open(p, 'rb').read())))
    templates = {s: sorted(glob.glob(os.path.join(a.containers[0], f'{s}_*.bin'))) for s in ('vs', 'ps')}
    os.makedirs(a.out, exist_ok=True)
    for old in glob.glob(os.path.join(a.out, '*.bin')):
        os.remove(old)  # the folder is regenerated from scratch, so a skipped shader cannot linger
    wrapped = skipped = 0
    seen = set()
    for d in a.raw:
        for p in sorted(glob.glob(os.path.join(d, '*.ucode'))):
            name = os.path.basename(p)[:-len('.ucode')]
            if name in seen:
                continue
            seen.add(name)
            raw = open(p, 'rb').read()
            owner = next((n for n, phys in covered if raw and raw in phys), None)
            if owner:
                print(f'skip {name}: its microcode is in the real container {owner}')
                skipped += 1
                continue
            stage = name.split('_', 1)[0]
            if not templates.get(stage):
                sys.exit(f'no {stage} template container in {a.containers[0]}')
            subprocess.run([sys.executable, os.path.join(ROOT, 'shaders', 'tools', 'wrap_raw_shader.py'), p,
                            templates[stage][0], os.path.join(a.out, name + '.bin'), '--stage', stage], check=True)
            print(f'wrap {name}')
            wrapped += 1
    print(f'{wrapped} wrapped (Direct3D\'s own shaders), {skipped} skipped (a real container has them)')


if __name__ == '__main__':
    main()
