#!/usr/bin/env python3
"""Translate every container to HLSL in its own translator process (8 in parallel), so one
container that crashes XenosRecomp does not stop the rest. Crashes are listed.
  shaders/tools/translate_all.py <containers> <hlsl out>"""
import concurrent.futures as cf, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
TOOL = os.path.join(ROOT, "out/tools/xenos_hlsl")  # built by shaders/tools/build_xenos_hlsl.sh
COMMON = os.path.join(ROOT, "shaders/XenosRecomp/shader_common.h")

def one(src, out):
    name = os.path.basename(src)[:-4]
    if os.path.exists(os.path.join(out, name + ".hlsl")):
        return name, 0
    with tempfile.TemporaryDirectory() as t:
        os.symlink(os.path.abspath(src), os.path.join(t, name + ".bin"))
        r = subprocess.run([TOOL, t, os.path.join(t, "o"), COMMON], capture_output=True)
        h = os.path.join(t, "o", name + ".hlsl")
        if r.returncode == 0 and os.path.exists(h):
            shutil.move(h, os.path.join(out, name + ".hlsl"))
            return name, 0
        return name, r.returncode or 1

def main():
    src, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    files = sorted(os.path.join(src, f) for f in os.listdir(src) if f.endswith(".bin"))
    bad = []
    with cf.ThreadPoolExecutor(8) as ex:
        for name, rc in ex.map(lambda f: one(f, out), files):
            if rc:
                bad.append((name, rc))
    with open(os.path.join(out, "..", "translate_failed.txt"), "w") as f:
        for n, rc in bad:
            f.write(f"{n} {rc}\n")
    print(f"{len(files) - len(bad)} translated, {len(bad)} failed")

main()
