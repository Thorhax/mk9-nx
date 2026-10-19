#!/usr/bin/env python3
# Targeted counterpart of tools/direct_calls.py for NEW hooks on an already post-processed generated tree.
#
# direct_calls.py rewrites `sub_X(ctx, base);` into `__imp__sub_X(ctx, base);` for every function that has no hook
# (the address appearing in the sources decides). A hook added afterwards is bypassed by those direct calls. This
# reverts only the given addresses, so only the files that really call them are touched (and rebuilt); a later full
# run of direct_calls.py (tools/codegen.sh) keeps them as hooks because the addresses are in the sources.
#
#   usage: restore_hook_calls.py [--gen <generated dir>] 82AA53C0 82B4D580 ...
import os
import re
import sys

args = sys.argv[1:]
gen = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'generated', 'default')
if '--gen' in args:
    i = args.index('--gen')
    gen = args[i + 1]
    del args[i:i + 2]
addrs = [a.upper() for a in args]
if not addrs:
    sys.exit(__doc__)
pat = re.compile(r'__imp__sub_(%s)\(ctx, base\);' % '|'.join(addrs))
total = files = 0
for f in sorted(os.listdir(gen)):
    if not (f.startswith('sa2_recomp.') and f.endswith('.cpp')):
        continue
    p = os.path.join(gen, f)
    t = open(p, encoding='utf-8').read()
    n = pat.sub(lambda m: 'sub_%s(ctx, base);' % m.group(1), t)
    if n != t:
        open(p, 'w', encoding='utf-8', newline='\n').write(n)
        files += 1
        total += len(pat.findall(t))
print('reverted %d direct calls in %d files for %s' % (total, files, ' '.join(addrs)))
