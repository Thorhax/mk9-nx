"""rigenv.py: the rig's settings from rig/rig.env (plain KEY=value lines, the file the shell scripts source; copy
rig.env.example). Variables set in the environment override the file."""
import os
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rig.env")
def load():
    if not os.path.exists(PATH):
        raise SystemExit("rig: copy rig/rig.env.example to rig/rig.env and fill it in")
    env = {}
    for l in open(PATH).read().splitlines():
        l = l.strip()
        if not l or l.startswith("#") or "=" not in l: continue
        k, v = l.split("=", 1); env[k.strip()] = v.strip().strip("'\"")
    for k in env:
        if k in os.environ: env[k] = os.environ[k]
    return env
E = load()
