"""Compares lockstep captures (out/pc/captures/<name>.frames + .hash: a presented-swap label and the XXH3 of the
full-resolution frame, uint64 each). Two runs number their swaps from different starting points, so the second is
aligned on the label offset most of its hashes agree at, then compared frame by frame.
  python tools/d3d/hashcmp.py r2f_r13 r2f_tmpl [r2f_tmploff ...]   (first = reference)"""
import os
import struct
import sys

CAP = os.path.join(os.path.dirname(__file__), '..', '..', 'out', 'pc', 'captures')


def load(name):
    frames = open(f'{CAP}/{name}.frames', 'rb').read()
    hashes = open(f'{CAP}/{name}.hash', 'rb').read()
    n = min(len(frames), len(hashes)) // 8
    labels = struct.unpack(f'<{n}Q', frames[:n * 8])
    values = struct.unpack(f'<{n}Q', hashes[:n * 8])
    return {l: h for l, h in zip(labels, values) if l != 0xFFFFFFFFFFFFFFFF}


ref_name = sys.argv[1]
ref = load(ref_name)
where = {}
for label, h in ref.items():
    where.setdefault(h, label)
for name in sys.argv[2:]:
    other = load(name)
    votes = {}
    for label, h in other.items():
        if h in where:
            votes[label - where[h]] = votes.get(label - where[h], 0) + 1
    offset = max(votes, key=votes.get) if votes else 0
    common = sorted(l for l in other if l - offset in ref)
    diff = [l for l in common if other[l] != ref[l - offset]]
    runs = []
    for l in diff:
        if runs and l <= runs[-1][1] + 2:
            runs[-1][1] = l
        else:
            runs.append([l, l])
    print(f'{name} vs {ref_name} (label offset {offset}): {len(common)} common frames, {len(common) - len(diff)} '
          f'identical, {len(diff)} differ' +
          (f'; differing ({name} labels): {", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs[:16])}'
           if runs else ''))
