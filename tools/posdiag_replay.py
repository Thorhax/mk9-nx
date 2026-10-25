#!/usr/bin/env python3
"""mk9-nx (2026-10-23): runs a vertex shader again on the PC with what a position-diagnostics capture recorded.

    python3 tools/posdiag_replay.py posdiag_cap_NN_<HASH>.bin [--glsl out/deko3d_shaders/glsl_opt] [--icd nvidia|lvp]

The capture (posdiag_cap_*.bin, written by the [posdiag] report of a test build, tools/deko3d_diag_positions.py) holds the
first bad vertex of a window shader: its vertex and instance IDs, the bound element types, the words it read from the
address window, and all its vertex and shared constants. The shader's deko3d GLSL becomes a compute shader (one
invocation, the vertex's IDs as constants, gl_Position into a storage buffer) that tools/replay/vkrun runs through the
PC's Vulkan driver: a finite position there, where the console's uam-compiled shader gave NaN, is a uam miscompile.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLSLANG = os.path.join(ROOT, 'out', 'glslang', 'StandAlone', 'glslangValidator')
VKRUN = os.path.join(ROOT, 'out', 'replay', 'vkrun')
CAPTURE, READLOG, VCONST, SCONST = 1028, 1356, 1552, 2580
ICDS = {'nvidia': '/usr/share/vulkan/icd.d/nvidia_icd.json', 'lvp': '/usr/share/vulkan/icd.d/lvp_icd.json'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capture')
    ap.add_argument('--glsl', default=os.path.join(ROOT, 'out', 'deko3d_shaders', 'glsl_opt'))
    ap.add_argument('--icd', default='nvidia', choices=sorted(ICDS))
    ap.add_argument('--keep', help='folder to keep the compute GLSL and buffers in')
    a = ap.parse_args()
    data = open(a.capture, 'rb').read()
    w = struct.unpack(f'<{len(data) // 4}I', data)
    c = w[CAPTURE:]
    vs_hash = (c[2] << 32) | c[1]
    pos = struct.unpack('<4f', struct.pack('<4I', *c[4:8]))
    vid, iid = c[74], w[READLOG + 3]
    elements = [c[76 + e * 10:76 + e * 10 + 10] for e in range(16)]
    n = min(w[READLOG], 96)
    reads = [(w[READLOG + 4 + r * 2], w[READLOG + 5 + r * 2]) for r in range(n)]
    print(f'vs {vs_hash:016X}: captured position {pos}, vertex id {vid}, instance {iid}, {w[READLOG]} window reads'
          + (' (log full: later reads replay as 0)' if w[READLOG] >= 96 else ''))

    text = open(os.path.join(a.glsl, f'vs_{vs_hash:016X}.glsl')).read()
    defines = ['layout(local_size_x = 1) in;']
    for i, el in enumerate(elements):
        if el[0] | el[1] | el[2] | el[3]:
            defines.append(f'#define SPIRV_CROSS_CONSTANT_ID_{1300 + i} {el[3]}u')
    text = text.replace('#version 460\n', '#version 460\n' + '\n'.join(defines) + '\n', 1)
    text = re.sub(r'^layout\(location = \d+\) out (\w+) (\w+);', r'\1 \2;', text, flags=re.M)
    text = text.replace('gl_VertexID', f'int({vid}u)').replace('gl_InstanceID', f'int({iid}u)')
    text = text.replace('gl_Position', 'g_replayPos').replace('gl_PointSize', 'g_replayPointSize')
    text = text.replace('void main()', 'vec4 g_replayPos;\nfloat g_replayPointSize;\n'
                        'layout(binding = 20, std430) buffer ReplayOut { vec4 p; } g_replayOut;\nvoid vs_main()', 1)
    text += '\nvoid main()\n{\n    vs_main();\n    g_replayOut.p = g_replayPos;\n}\n'

    tmp = a.keep or tempfile.mkdtemp(prefix='posdiag_replay_')
    os.makedirs(tmp, exist_ok=True)
    src, spv = os.path.join(tmp, 'replay.comp'), os.path.join(tmp, 'replay.spv')
    open(src, 'w').write(text)
    r = subprocess.run([GLSLANG, '-V', '--target-env', 'vulkan1.2', '-S', 'comp', '-o', spv, src],
                       capture_output=True, text=True)
    if r.returncode:
        sys.exit('glslang: ' + (r.stdout + r.stderr)[-2000:])
    window = [0] * (max([i for i, _ in reads], default=0) + 4)
    for i, v in reads:
        window[i] = v
    files = {'vconst': struct.pack(f'<{257 * 4}I', *w[VCONST:VCONST + 257 * 4]),
             'shared': struct.pack(f'<{85 * 4}I', *w[SCONST:SCONST + 85 * 4]),
             'window': struct.pack(f'<{len(window)}I', *window), 'out': bytes(16)}
    for name, blob in files.items():
        open(os.path.join(tmp, name + '.bin'), 'wb').write(blob)
    env = dict(os.environ, VK_ICD_FILENAMES=ICDS[a.icd])
    r = subprocess.run([VKRUN, spv, f'0:ubo:{tmp}/vconst.bin', f'2:ubo:{tmp}/shared.bin',
                        f'15:ssbo:{tmp}/window.bin', f'20:ssbo:{tmp}/out.bin', '-o', f'20:{tmp}/out.bin'],
                       capture_output=True, text=True, env=env)
    if r.returncode:
        sys.exit('vkrun: ' + r.stderr[-2000:])
    out = struct.unpack('<4f', open(os.path.join(tmp, 'out.bin'), 'rb').read()[:16])
    print(f'{r.stderr.strip()}: position {out}')


if __name__ == '__main__':
    main()
