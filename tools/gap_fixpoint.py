#!/usr/bin/env python3
"""Repeat the structural gap scan until no newly exposed gap qualifies.

Each pass: find_gaps.py on the current generated code -> gaps of 8..1024 bytes not examined before -> scratch
codegen with them declared -> gap_classify.py checks (tier 1 + no stack/LR + no undefined-at-entry registers, and
tier 2) -> append the accepted entries to app/overrides.toml -> real codegen (tools/codegen.sh), so the next pass
sees the code the new entries exposed. Evidence goes to <scratch>/gapscan-accepted.txt. Stops when a pass accepts
nothing.

  python3 tools/gap_fixpoint.py SCRATCH_DIR [examined.txt ...]

SCRATCH_DIR  work folder (one subfolder per pass; it is not cleaned).
examined.txt files whose lines start with an address that must not be examined again (selected.txt of earlier
             runs).

Environment: REXGLUE = path of the rexglue host tool (default sdk/out/host/rexglue, see tools/build_host.sh).
Needs the game in assets/game_root and an existing generated tree (tools/codegen.sh).
"""
import os, re, subprocess, sys

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = os.path.join(root, "app")
if len(sys.argv) < 2:
    sys.exit(__doc__)
scratch = os.path.abspath(sys.argv[1])
rexglue = os.path.abspath(os.environ.get("REXGLUE", os.path.join(root, "sdk", "out", "linux-amd64", "rexglue")))
overrides_path = os.path.join(app, "overrides.toml")
manifest_path = os.path.join(app, "mk9_manifest.toml")
gen = os.path.join(app, "generated", "default")
evidence_path = os.path.join(scratch, "gapscan-accepted.txt")
tools = os.path.join(root, "tools")

ns = {"__name__": "gc"}
exec(compile(open(os.path.join(tools, "gap_classify.py")).read(), "gc", "exec"), ns)
examined = set()
for p in sys.argv[2:]:
    examined |= {l.split()[0] for l in open(p)}
os.makedirs(scratch, exist_ok=True)

for n in range(1, 20):
    d = os.path.join(scratch, f"pass{n}")
    os.makedirs(d, exist_ok=True)
    subprocess.run([sys.executable, os.path.join(tools, "find_gaps.py"), "--gen", gen,
                    "--report", f"{d}/h.txt", "--toml", f"{d}/c.toml"], capture_output=True)
    sel = [(m.group(1), int(m.group(2))) for l in open(f"{d}/c.toml")
           for m in [re.match(r'"(0x[0-9A-F]+)" = \{ \}\s*#\s*(\d+) bytes', l)]
           if m and 8 <= int(m.group(2)) <= 1024 and m.group(1) not in examined]
    examined |= {a for a, _ in sel}
    if not sel:
        print(f"pass {n}: no new gaps; fixpoint")
        break
    ov = open(overrides_path).read()
    m = re.search(r"^\[functions\]\s*$", ov, re.M)
    scratch_overrides = f"{d}/overrides.toml"
    open(scratch_overrides, "w").write(ov[:m.end()] + "\n" + "".join(f'"{a}" = {{ }}\n' for a, _ in sel) + ov[m.end():])
    open(f"{d}/selected.txt", "w").write("".join(f"{a} {s}\n" for a, s in sel))
    # Scratch manifest: same game, output in the pass folder, and the scratch overrides instead of the real ones.
    man = open(manifest_path).read()
    man = re.sub(r'^game_root = .*$', f'game_root = "{root}/assets/game_root"', man, flags=re.M)
    man = re.sub(r'^file_path = .*$', f'file_path = "{root}/assets/game_root/default.xex"', man, flags=re.M)
    man = re.sub(r'^out_directory_path = .*$', f'out_directory_path = "{d}/generated"', man, flags=re.M)
    man = re.sub(r'^includes = .*$', f'includes = ["{scratch_overrides}"]', man, flags=re.M)
    open(f"{d}/manifest.toml", "w").write(man)
    r = subprocess.run([rexglue, "codegen", "manifest.toml"], cwd=d, capture_output=True, text=True)
    open(f"{d}/codegen.log", "w").write(r.stdout + r.stderr)
    if r.returncode:
        print(f"pass {n}: scratch codegen failed (see {d}/codegen.log)")
        break
    subprocess.run([sys.executable, os.path.join(tools, "gap_classify.py"), f"{d}/generated", f"{d}/selected.txt",
                    f"{d}/codegen.log", f"{d}/acc.txt"], capture_output=True)
    subprocess.run([sys.executable, os.path.join(tools, "gap_classify.py"), "--tier2", f"{d}/generated",
                    f"{d}/selected.txt", f"{d}/codegen.log", overrides_path, f"{d}/t2.txt"], capture_output=True)
    have = set(re.findall(r'"(0x[0-9A-Fa-f]+)"\s*=', ov))
    accepted = []
    for l in open(f"{d}/acc.txt").readlines() + open(f"{d}/t2.txt").readlines():
        a, s, rest = l.split(" ", 2)
        ins = [i.strip() for i in rest.lstrip("| ").strip().split(" ; ")]
        ops = [i.split()[0] for i in ins]
        if (a in have or any(re.search(r"\br1\b", i) for i in ins)
                or any(o in ("mtlr", "mflr", "ld", "std", "stwu", "lfd", "stfd") for o in ops)
                or ns["reads_undefined_at_entry"](ins)):
            continue
        accepted.append(l)
        have.add(a)
    print(f"pass {n}: {len(sel)} new gaps, {len(accepted)} accepted")
    if not accepted:
        print("fixpoint")
        break
    with open(evidence_path, "a") as ev_file:
        ev_file.writelines(accepted)
    block = f"\n# Structural gap scan, fixpoint pass {n} (tools/gap_fixpoint.py).\n" + "".join(
        f'"{l.split()[0]}" = {{ }}\n' for l in accepted)
    open(overrides_path, "w").write(ov[:m.end()] + block + ov[m.end():])
    b = subprocess.run(["bash", os.path.join(tools, "codegen.sh")], capture_output=True, text=True,
                       env={**os.environ, "REXGLUE": rexglue})
    if b.returncode:
        print("codegen failed")
        print((b.stdout + b.stderr)[-2000:])
        break
