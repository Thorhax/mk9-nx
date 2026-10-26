#!/usr/bin/env python3
# soakstat.py <dir> [png]: from <dir>/net.log: per minute fps (min/avg), longest frame, heap in use, GPU textures and
# render targets; plus every 5-s window with a frame over 100 ms. Optional png: memory and fps over time.
import re, sys, datetime as dt
d = sys.argv[1]
L = open(f"{d}/net.log", errors="replace").read().splitlines()
def ts(l):
    m = re.match(r"\[(\d+-\d+-\d+ \d+:\d+:\d+\.\d+)\]", l)
    return dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f") if m else None
t0 = next(ts(l) for l in L if ts(l))
fps, heap, tex, slow = [], [], [], []
for l in L:
    t = ts(l)
    if not t: continue
    s = (t - t0).total_seconds()
    m = re.search(r"\[fps\] ([\d.]+) fps over [\d.]+ s \(\d+ frames\), longest frame ([\d.]+) ms", l)
    if m:
        fps.append((s, float(m[1]), float(m[2])))
        if float(m[2]) > 100: slow.append((s, float(m[1]), float(m[2])))
    m = re.search(r"Memory: heap (\d+) MB in use of (\d+) MB \((\d+) MB taken from it so far, (\d+) MB of that free in (\d+) pieces", l)
    if m: heap.append((s, int(m[1]), int(m[3]), int(m[4]), int(m[5])))
    m = re.search(r"Memory: GPU: (\d+) render targets (\d+) MB.*?, (\d+) textures (\d+) MB", l)
    if m: tex.append((s, int(m[2]), int(m[3]), int(m[4])))
print(f"log span {fps[-1][0]/60:.1f} min" if fps else "no fps lines")
print("min  fps(avg/min)  longest  heap-used  heap-taken  frag(MB/pieces)  RT MB  textures(n/MB)")
for minute in range(int(fps[-1][0] // 60) + 1 if fps else 0):
    F = [f for f in fps if minute * 60 <= f[0] < minute * 60 + 60]
    H = [h for h in heap if h[0] < minute * 60 + 60]
    T = [x for x in tex if x[0] < minute * 60 + 60]
    if not F: continue
    h = H[-1] if H else (0, 0, 0, 0, 0); x = T[-1] if T else (0, 0, 0, 0)
    print(f"{minute:3d}  {sum(f[1] for f in F)/len(F):5.1f}/{min(f[1] for f in F):5.1f}  {max(f[2] for f in F):7.1f}  "
          f"{h[1]:6d}  {h[2]:6d}  {h[3]:4d}/{h[4]:6d}  {x[1]:4d}  {x[2]:4d}/{x[3]:4d}")
print(f"\n5-s windows with a frame over 100 ms: {len(slow)}")
for s, f, lf in slow: print(f"  {s/60:6.2f} min  {f:5.1f} fps  longest {lf:.0f} ms")
if len(sys.argv) > 2 and heap:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot([h[0]/60 for h in heap], [h[1] for h in heap], label="heap in use MB")
    ax[0].plot([h[0]/60 for h in heap], [h[2] for h in heap], label="heap taken MB")
    if tex: ax[0].plot([x[0]/60 for x in tex], [x[3] for x in tex], label="GPU textures MB")
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot([f[0]/60 for f in fps], [f[1] for f in fps], lw=.8, label="fps (5 s)")
    ax[1].set_ylim(0, 65); ax[1].set_xlabel("minutes"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(sys.argv[2], dpi=90)
