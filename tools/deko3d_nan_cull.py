#!/usr/bin/env python3
"""mk9-nx (2026-10-23, test): a copy of the deko3d GLSL whose vertex shaders replace a non-finite gl_Position with one
the clipper always discards (beyond the far plane), as the Xbox 360 (and the PC's driver) discard primitives with NaN
corners. For the fights' huge flat-coloured triangles: MK9's dead particles are zero-filled, and their vertices come
out as NaN.

  python3 tools/deko3d_nan_cull.py --in out/deko3d_shaders/glsl_opt --out out/deko3d_shaders/glsl_nancull
"""
import argparse
import glob
import os
import re
import shutil

HELPER = """vec4 dkNanCull(vec4 p)
{
    return (any(isnan(p)) || any(isinf(p))) ? vec4(0.0, 0.0, 2.0, 1.0) : p;
}
"""
ASSIGN = re.compile(r'^(\s*)gl_Position = ([^;]+);\s*$', re.M)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='src', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(a.out)
    done = 0
    for path in sorted(glob.glob(os.path.join(a.src, '*.glsl'))):
        name = os.path.basename(path)
        text = open(path).read()
        if name.startswith('vs_') and 'void main()' in text and ASSIGN.search(text):
            text = text.replace('void main()', HELPER + '\nvoid main()', 1)
            text = ASSIGN.sub(lambda m: f'{m.group(1)}gl_Position = dkNanCull({m.group(2)});', text)
            done += 1
        open(os.path.join(a.out, name), 'w').write(text)
    print(f'deko3d_nan_cull: {done} vertex shaders')


if __name__ == '__main__':
    main()
