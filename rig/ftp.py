#!/usr/bin/env python3
"""ftp.py put local remote | get remote local | getdir remote local | ls remote | rm remote"""
import ftplib, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rigenv import E
f = ftplib.FTP(); f.connect(E["SWITCH_IP"], int(E.get("FTP_PORT", 5000)), timeout=30); f.login(E["FTP_USER"], E["FTP_PASS"])
op, a = sys.argv[1], sys.argv[2:]
def getdir(r, l):
    os.makedirs(l, exist_ok=True)
    ents = []
    f.retrlines("LIST " + r, ents.append)
    for e in ents:
        parts = e.split(None, 8); name = parts[8]
        if e.startswith("d"): getdir(r + "/" + name, os.path.join(l, name))
        else:
            with open(os.path.join(l, name), "wb") as o: f.retrbinary("RETR " + r + "/" + name, o.write)
if op == "put":
    with open(a[0], "rb") as i: f.storbinary("STOR " + a[1], i)
elif op == "get":
    with open(a[1], "wb") as o: f.retrbinary("RETR " + a[0], o.write)
elif op == "getdir": getdir(a[0], a[1])
elif op == "ls": f.retrlines("LIST " + a[0])
elif op == "rm": f.delete(a[0])
f.quit()
