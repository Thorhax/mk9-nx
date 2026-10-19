"""Print the PPC listing of recompiled functions (from the generated code's comments), compactly.
   python ppcfn.py 82565C60 [82563AB0 ...] [--max N]"""
import glob, os, re, sys
GEN = os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'generated', 'default')
want = [a.upper().replace('0X', '') for a in sys.argv[1:] if not a.startswith('--')]
mx = 400
for a in sys.argv[1:]:
    if a.startswith('--max='): mx = int(a.split('=')[1])
DEF = re.compile(r'^DEFINE_REX_FUNC\(sub_([0-9A-F]{8})\) \{$')
found = {}
for f in sorted(glob.glob(GEN + '/sa2_recomp.*.cpp')):
    cur = None
    for ln in open(f, encoding='utf-8', errors='ignore'):
        m = DEF.match(ln)
        if m:
            cur = m.group(1) if m.group(1) in want else None
            if cur: found[cur] = []
            continue
        if cur is None: continue
        if ln.startswith('}'):
            cur = None; continue
        s = ln.strip()
        if s.startswith('// '):
            found[cur].append('    ' + s[3:])
        elif s.startswith('loc_'):
            found[cur].append(s)
        elif 'sub_' in s and '(ctx, base)' in s:
            pass  # the call is in the comment already (bl)
for a in want:
    lines = found.get(a)
    if lines is None:
        print(f'== sub_{a}: not found'); continue
    print(f'== sub_{a} ({len(lines)} lines)')
    for l in lines[:mx]: print(l)
    if len(lines) > mx: print(f'    ... {len(lines) - mx} more')
