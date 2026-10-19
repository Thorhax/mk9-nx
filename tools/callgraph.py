#!/usr/bin/env python3
"""Call graph of the generated code, for finding the game's Direct3D (finding where it is called).

  tools/callgraph.py build                 # index generated/default -> out/callgraph.json
  tools/callgraph.py callers <name>        # functions that call <name> (sub_XXXXXXXX or an import)
  tools/callgraph.py callees <name>
  tools/callgraph.py grep <regex>          # functions whose PPC listing matches <regex>
"""
import glob, json, os, re, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "app", "generated", "default")
IDX = os.path.join(ROOT, "out", "callgraph.json")
DEF = re.compile(r"^(?:DEFINE_REX_FUNC|PPC_FUNC_IMPL)\((?:__imp__)?(\w+)\)")
CALL = re.compile(r"\b(?:__imp__)?(sub_[0-9A-F]{8}|[A-Z][A-Za-z0-9_]+)\(ctx, base\)")


def build():
    funcs = {}
    for path in sorted(glob.glob(os.path.join(GEN, "*_recomp.*.cpp"))):
        cur = None
        with open(path, errors="replace") as f:
            for n, line in enumerate(f, 1):
                m = DEF.match(line)
                if m:
                    cur = m.group(1)
                    funcs[cur] = {"file": os.path.basename(path), "line": n, "calls": []}
                    continue
                if cur:
                    for c in CALL.findall(line):
                        if c != cur and c not in funcs[cur]["calls"]:
                            funcs[cur]["calls"].append(c)
    os.makedirs(os.path.dirname(IDX), exist_ok=True)
    json.dump(funcs, open(IDX, "w"))
    print(f"{len(funcs)} functions -> {IDX}")


def load():
    return json.load(open(IDX))


def listing(fn, info):
    lines = open(os.path.join(GEN, info["file"]), errors="replace").read().splitlines()
    out = []
    for line in lines[info["line"] - 1:]:
        out.append(line)
        if line.startswith("}"):
            break
    return out


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        return build()
    funcs = load()
    arg = sys.argv[2]
    if cmd == "callers":
        for k, v in sorted(funcs.items()):
            if arg in v["calls"]:
                print(k, v["file"], v["line"])
    elif cmd == "callees":
        print(" ".join(funcs[arg]["calls"]))
    elif cmd == "grep":
        rx = re.compile(arg)
        for k, v in sorted(funcs.items()):
            hits = [l.strip() for l in listing(k, v) if rx.search(l)]
            if hits:
                print(k, len(hits), hits[0][:100])


if __name__ == "__main__":
    main()
