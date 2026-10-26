#!/usr/bin/env python3
"""fight.py <seconds> <shotprefix>: random attacks/specials over sys-botbase, screenshot every ~3 s"""
import sys, time, random, socket, binascii, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sb
dur, pref = float(sys.argv[1]), sys.argv[2]
random.seed(int(os.environ.get("SEED", "1")))  # the same inputs every run (fixed-scene A/B)
S = socket.create_connection(sb.HOST, timeout=15)
def c(x): S.sendall(x.encode() + b"\r\n")
def tap(b, h=0.08): c("press " + b); time.sleep(h); c("release " + b); time.sleep(0.05)
attacks = ["Y", "X", "B", "A"]  # positional: Xbox X,Y,A,B (never PLUS/MINUS/HOME)
# SAFE=1 (soaks): no Switch A (the Xbox layout's B: back out of a menu) and no down/up, so a continue screen or a menu
# reached by the mashing only ever gets confirm (Retry) and left/right.
SAFE = os.environ.get("SAFE") == "1"
if SAFE: attacks = ["Y", "X", "B"]
def special():
    seq = random.choice([["DLEFT", "DRIGHT"], ["DLEFT", "DLEFT"], ["DRIGHT", "DRIGHT"]] if SAFE else
                        [["DLEFT", "DRIGHT"], ["DDOWN", "DLEFT"], ["DDOWN", "DRIGHT"], ["DLEFT", "DLEFT"]])
    for d in seq: tap(d, 0.06)
    tap(random.choice(attacks))
t0 = time.time(); nshot = 0; last = 0
while time.time() - t0 < dur:
    r = random.random()
    if r < 0.4: special()
    elif r < 0.85: tap(random.choice(attacks))
    else:
        d = random.choice(["DRIGHT", "DLEFT"]); c("press " + d); time.sleep(0.3); c("release " + d)
    if time.time() - last > 3:
        last = time.time(); nshot += 1
        sb.shot(f"{pref}_{nshot:02d}.jpg")
    time.sleep(0.05)
S.close()
