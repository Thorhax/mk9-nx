#!/usr/bin/env python3
"""sys-botbase helper: sb.py cmd "click A" ... | sb.py shot out.jpg"""
import socket, sys, binascii, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigenv import E
HOST = (E["SWITCH_IP"], int(E.get("SB_PORT", 6000)))
def conn():
    return socket.create_connection(HOST, timeout=15)
def cmd(c):
    s = conn(); s.sendall(c.encode() + b"\r\n"); s.close()
def query(c):
    s = conn(); s.sendall(c.encode() + b"\r\n"); buf = b""
    while not buf.endswith(b"\n"):
        d = s.recv(1 << 20)
        if not d: break
        buf += d
    s.close(); return buf.strip()
def shot(path):
    open(path, "wb").write(binascii.unhexlify(query("pixelPeek")))
if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "shot": shot(a[1])
    elif a[0] == "title": print(query("getTitleID").decode())
    else:
        for c in a:
            if c.startswith("sleep "): time.sleep(float(c[6:]))
            elif c.startswith("click "):
                b = c[6:]; cmd("press " + b); time.sleep(0.15); cmd("release " + b); time.sleep(0.35)
            else: cmd(c); time.sleep(0.4)
