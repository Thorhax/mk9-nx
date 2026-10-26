#!/usr/bin/env python3
"""Threaded log listener: each connection on its own thread, keepalive, appended to <dir>/net_<n>.log"""
import socket, sys, threading, os, time
port, outdir = int(sys.argv[1]), sys.argv[2]
os.makedirs(outdir, exist_ok=True)
srv = socket.socket(); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("0.0.0.0", port)); srv.listen(8)
n = 0
def serve(c, path):
    c.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    c.settimeout(None)
    with open(path, "ab", buffering=0) as f:
        while True:
            try: d = c.recv(65536)
            except OSError: break
            if not d: break
            f.write(d)
    c.close()
while True:
    c, a = srv.accept(); n += 1
    p = os.path.join(outdir, f"net_{int(time.time())}_{n}.log")
    print("conn", a, p, flush=True)
    threading.Thread(target=serve, args=(c, p), daemon=True).start()
