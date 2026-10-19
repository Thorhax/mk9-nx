#!/usr/bin/env python3
"""Flow-sensitive "read before write" check for non_volatile_as_local (flow-sensitive; run it after every codegen).

With non_volatile_as_local / cr_as_local / ctr_as_local / xer_as_local, every recompiled function keeps
r14-r31, f14-f31, v14-v31/v64-v127, cr0-cr7, ctr and xer in C++ locals that start at zero. A function
that reads one of them on some path before writing it gets 0 instead of the value its parent set up:
that is a split piece (or funclet) that must be marked `share_registers` in app/perf_overrides.toml.

A simple linear scan of each function would report rotated loops (a `goto` forward to the loop condition,
the body textually first) as reads before writes. Marking such a normal
function share_registers is harmful: its __savegprlr/__restgprlr calls are elided (global gate), it
then clobbers ctx.r14-r31, and a `bl` caller copies those clobbered values back into its locals
(emit_call_sharing_registers copyBack). That corrupted sub_82978560's registers after its call to
sub_82977800 (startup crash, first attempt).

This version builds the control-flow graph from the generated labels/gotos/switches and runs a
must-be-written dataflow, so a read is reported only if some path from entry reaches it with the
register unwritten.

Calls to functions that never return are path ends: --noreturn= seeds them (Mass Effect: 0x82ACA550, the
guest longjmp; the D3DX HLSL compiler's error functions longjmp out), and every function whose `return;`
is then unreachable is added to a fixpoint. Without it, ~20 compiler functions show reads after an error
call that never comes back.

Plain stores of an unwritten register to memory (hand-written prologue saves such as stvx v127) are listed
separately on stderr and ignored: with locals they store a dead value, which is harmless.

Run it on code generated WITHOUT share_registers marks (they turn the marked functions back to ctx.*,
which hides them). Usage:
  tools/read_before_write.py app/generated/default --noreturn=82ACA550            # report
  tools/read_before_write.py app/generated/default --noreturn=82ACA550 --toml     # perf_overrides.toml lines
  ... --why=sub_827F6558       # print every offending line of that function with its label (stderr)

Volatile mode (non_argument_as_local): the registers are r0, r2, r11, r12, f0, v32-v63 and the
lwarx reservation (reserved_as_local).
  --volatile       code generated WITH the option: reads of the local of a function that does not share its
                   registers (share_registers callees, non_argument_as_local_keep functions and the save/restore
                   helpers read ctx.rN, which is not tracked)
  --volatile-ctx   code generated WITHOUT the option (ctx.rN spelling): every function that reads one of them
                   before writing it, i.e. every callee that takes an argument in r0/r11/r12/f0
  --kill-calls     (with either) forget the written set at every call: lists reads of a value the callee
                   would have to return in the register; the offending callee is shown after `<=`
In volatile mode a store of an unwritten register counts as a read: these are argument registers, and
__savegprlr_N storing the r12 its caller set up is exactly such a use. tools/non_argument_audit.py is the
whole-program version (callee write sets, values passed through, the generated hand-overs).
"""
import glob
import re
import sys

REG = (r'r(?:1[4-9]|2[0-9]|3[01])|f(?:1[4-9]|2[0-9]|3[01])|'
       r'v(?:1[4-9]|2[0-9]|3[01]|6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])|cr[0-7]|ctr|xer')
tok = re.compile(r'(?<![.\w])(' + REG + r')\b')
# After tools/args_in_registers.py (or the codegen option args_in_registers) the body of a function that has direct
# callers lives in `extern "C" ... __fast_sub_X(...) {`; the DEFINE_REX_FUNC of such a function is only a wrapper.
# Named functions too (banjo-nx: 29 CRT __finally funclets named rex_LN* need share_registers, and the sub_-only
# pattern never looked at them). The SDK's __save*/__rest* helpers are named as well: their reads are the point.
fn_re = re.compile(r'^(?:DEFINE_REX_FUNC\(([A-Za-z_][A-Za-z0-9_]*)\)|extern "C" (?:uint64_t|void) __fast_(sub_[0-9A-F]+)\(.*\) \{)')
lab_re = re.compile(r'^(loc_[0-9A-F]+):$')
goto_re = re.compile(r'^goto (loc_[0-9A-F]+);$')
ifgoto_re = re.compile(r'^if \(.*\) goto (loc_[0-9A-F]+);$')
asg = re.compile(r'^(' + REG + r')(?:\.[A-Za-z0-9_\[\]]+)*(?:\[[^\]]*\])?\s*(?:[-+|&^]?=)(?!=)')
call_re = re.compile(r'\b(sub_[0-9A-F]+|__imp__sub_[0-9A-F]+|__fast_sub_[0-9A-F]+)\(ctx, base(?:, [\w.]+)*\);')
decl_re = re.compile(r'^(?:PPCRegister|PPCXERRegister|PPCCRRegister|PPCVRegister|PPCFPRegister|'
                     r'uint32_t|uint64_t|PPCContext)\b')
share_copy = re.compile(r'^const auto s_(' + REG + r') = ctx\.\1; ctx\.\1 = \1;$')
share_back = re.compile(r'^(?:(' + REG + r') = ctx\.\1;|ctx\.(' + REG + r') = s_\2;)$')
stack_save = re.compile(r'^REX_STORE_U64\(ctx\.r1\.u32 \+ -?\d+, (?:r|f)\d+\.(?:u64|f64)\);$')

idx = {}
names = [f'r{i}' for i in range(14, 32)] + [f'f{i}' for i in range(14, 32)] + \
        [f'v{i}' for i in list(range(14, 32)) + list(range(64, 128))] + [f'cr{i}' for i in range(8)] + ['ctr', 'xer']
for i, n in enumerate(names):
    idx[n] = i
ALL = (1 << len(names)) - 1

# --volatile mode (non_argument_as_local): the registers are r0, r2, r11, r12, f0 and v32-v63. They are
# read in the generated code either as locals (code generated with the option) or as ctx.X (without it);
# both spellings are normalised to the plain name. With --kill-calls every call (bl, bctrl, import)
# forgets what was written: a read that is not preceded, on every path since the last call, by a write
# is a value the callee returned in that register (the ABI returns only r3, f1, v1).
VOL = False
VOL_CTX = False
KILL_CALLS = False
VOL_REG = r'r(?:0|2|11|12)|f0|v(?:3[2-9]|4[0-9]|5[0-9]|6[0-3])|reserved'
vol_ctx = re.compile(r'\bctx\.(' + VOL_REG + r')\b')
call_any = re.compile(r'\(ctx, base\);|REX_CALL_INDIRECT_FUNC|\bppc_setjmp\b|\bppc_longjmp\b')


def setup_volatile():
    global REG, tok, asg, names, idx, ALL, VOL, share_copy, share_back
    VOL = True
    REG = VOL_REG
    tok = re.compile(r'(?<![.\w])(' + REG + r')\b')
    asg = re.compile(r'^(' + REG + r')(?:\.[A-Za-z0-9_\[\]]+)*(?:\[[^\]]*\])?\s*(?:[-+|&^]?=)(?!=)')
    share_copy = re.compile(r'^const auto s_(' + REG + r') = ctx\.\1; ctx\.\1 = \1;$')
    share_back = re.compile(r'^(?:(' + REG + r') = ctx\.\1;|ctx\.(' + REG + r') = s_\2;)$')
    names = ['r0', 'r2', 'r11', 'r12', 'f0'] + [f'v{i}' for i in range(32, 64)] + ['reserved']
    idx = {n: i for i, n in enumerate(names)}
    ALL = (1 << len(names)) - 1


def uses_defs(s):
    """Registers read and written by one statement (reads happen before writes)."""
    if share_copy.match(s) or share_back.match(s):
        # copies done around a sharing call: not semantic reads
        m = re.match(r'^(' + REG + r') = ctx\.', s)
        return 0, (1 << idx[m.group(1)]) if m else 0
    if not VOL and stack_save.match(s):
        return 0, 0  # prologue save of the caller's value: harmless with locals
    s2 = re.sub(r'compare<[^>]*>\(([^;]*),\s*xer\)', r'compare(\1)', s)
    s2 = re.sub(r'cr\d\.so = xer\.so;', '', s2)
    defs = 0
    m = re.match(r'^(cr[0-7])\.(?:compare|setFromMask)', s2)
    if m and m.group(1) in idx:
        defs |= 1 << idx[m.group(1)]
        s2 = s2[len(m.group(1)):]
    ms = re.match(r'^simde_mm_store\w*\(\(?(?:simde__m128i?\*\))?(' + REG + r')\.\w+,(.*)$', s2)
    if ms:
        defs |= 1 << idx[ms.group(1)]
        s2 = ms.group(2)
    else:
        m2 = asg.match(s2)
        if m2:
            lhs = m2.group(1)
            defs |= 1 << idx[lhs]  # a field write (xer.ca = ...) counts as a definition too
            compound = s2[m2.end() - 2] in '-+|&^'
            s2 = (lhs + ' ' if compound else '') + s2[m2.end():]
    uses = 0
    for r in tok.findall(s2):
        uses |= 1 << idx[r]
    return uses, defs


def build_cfg(body):
    lines = []
    for ln in body:
        s = ln.strip()
        if VOL_CTX:
            s = vol_ctx.sub(r'\1', s)
        if not s or s.startswith('//') or s.startswith('REX_FUNC_PROLOGUE'):
            continue
        if decl_re.match(s) and s.endswith('{};'):
            continue
        lines.append(s)
    n = len(lines)
    labels = {}
    for i, s in enumerate(lines):
        m = lab_re.match(s)
        if m:
            labels[m.group(1)] = i
    succ = [[] for _ in range(n)]
    stack = []  # open braces: (line, kind)
    match_close = {}
    for i, s in enumerate(lines):
        opens = s.endswith('{') and not s.startswith('}')
        if s.startswith('{') and s.endswith('}'):
            opens = False
        if opens:
            stack.append(i)
        elif s == '}' or s.startswith('}'):
            if stack:
                match_close[stack.pop()] = i
    switch_cases = {}
    cur_switch = []
    for i, s in enumerate(lines):
        if s.startswith('switch ('):
            cur_switch.append(i)
            switch_cases[i] = []
        elif (s.startswith('case ') or s == 'default:') and cur_switch:
            switch_cases[cur_switch[-1]].append(i)
        if i in match_close.values() and cur_switch and match_close.get(cur_switch[-1]) == i:
            cur_switch.pop()
    for i, s in enumerate(lines):
        nxt = [i + 1] if i + 1 < n else []
        m = goto_re.match(s)
        if m:
            succ[i] = [labels[m.group(1)]] if m.group(1) in labels else []
            continue
        m = ifgoto_re.match(s)
        if m:
            succ[i] = nxt + ([labels[m.group(1)]] if m.group(1) in labels else [])
            continue
        if (s == 'return;' or s == 'return r3.u64;' or s.startswith('__builtin_trap()') or s.startswith('REX_FATAL(') or
                'ppc_longjmp(' in s):
            succ[i] = []
            continue
        if s.startswith('switch ('):
            succ[i] = switch_cases[i]
            continue
        if s.startswith('if (') and s.endswith('{') and i in match_close:
            succ[i] = nxt + ([match_close[i] + 1] if match_close[i] + 1 < n else [])
            continue
        succ[i] = nxt
    calls = {}
    for i, s in enumerate(lines):
        m = call_re.search(s)
        if m and not s.startswith('if ('):
            calls[i] = m.group(1).replace('__imp__', '').replace('__fast_', '')
    return lines, succ, calls


def returns(cfg, noreturn):
    """True if a `return;` is reachable from entry when calls to `noreturn` never come back."""
    lines, succ, calls = cfg
    if not lines:
        return True
    seen = {0}
    work = [0]
    while work:
        i = work.pop()
        if lines[i] in ('return;', 'return r3.u64;') or lines[i].endswith(' return;') or lines[i].endswith(' return r3.u64;'):
            return True
        if i in calls and calls[i] in noreturn:
            continue
        if i == len(lines) - 1 and succ[i] == [] and not lines[i].startswith(('goto ', '__builtin_trap', 'REX_FATAL')):
            return True  # falls off the end of the C++ function: returns
        for j in succ[i]:
            if j not in seen:
                seen.add(j)
                work.append(j)
    return False


WHY = False


def analyze(cfg, noreturn):
    lines, succ0, calls = cfg
    n = len(lines)
    succ = [([] if (i in calls and calls[i] in noreturn) else succ0[i]) for i in range(n)]
    ud = [uses_defs(s) for s in lines]
    # must-written dataflow: IN[i] = AND of OUT[preds]; entry IN = 0
    preds = [[] for _ in range(n)]
    for i in range(n):
        for j in succ[i]:
            preds[j].append(i)
    IN = [ALL] * n
    if n:
        IN[0] = 0
    OUT = [ALL] * n
    work = list(range(n))
    inwork = [True] * n
    while work:
        i = work.pop()
        inwork[i] = False
        if i == 0:
            new_in = 0
            for p in preds[i]:
                new_in &= OUT[p]
        else:
            new_in = ALL
            for p in preds[i]:
                new_in &= OUT[p]
            if not preds[i]:
                new_in = ALL  # unreachable
        IN[i] = new_in
        o = new_in | ud[i][1]
        if KILL_CALLS and call_any.search(lines[i]):
            o = 0
        if o != OUT[i]:
            OUT[i] = o
            for j in succ[i]:
                if not inwork[j]:
                    inwork[j] = True
                    work.append(j)
    bad = {}
    stores = {}
    for i in range(n):
        u = ud[i][0] & ~IN[i]
        if u:
            # a plain store of the value to memory (a hand-written save, e.g. stvx v127 in a
            # prologue) is harmless with locals; anything else uses the value. Every vector instruction is emitted as
            # simde_mm_store*(<destination register>, ...), so only a simde store whose target is guest memory is a
            # store (banjo-nx: the old startswith test took every vector read for one and hid v64-v77 globals).
            ln = lines[i]
            head = ln.split(',', 1)[0]
            is_store = ln.startswith('REX_STORE_') or (
                ln.startswith('simde_mm_store') and ('REX_RAW_ADDR(' in head or 'base +' in head))
            # volatile mode: a stored argument register is a use (__savegprlr_N stores its caller's r12)
            tgt = stores if is_store and not VOL else bad
            if WHY:
                lab = next((lines[j] for j in range(i, -1, -1) if lab_re.match(lines[j])), 'entry')
                print('  why', lab, i, lines[i][:110], file=sys.stderr)
            for k, nm in enumerate(names):
                if u >> k & 1 and nm not in tgt:
                    tgt[nm] = lines[i][:100]
                    if KILL_CALLS and tgt is bad:
                        tgt[nm] += ' <= ' + sources(lines, preds, ud, i, k)
    return bad, stores


def sources(lines, preds, ud, i, k):
    """Where the unwritten value of register k read at line i can come from: function entry or calls."""
    seen = {i}
    work = [i]
    out = set()
    first = True
    while work:
        j = work.pop()
        if not first:
            if ud[j][1] >> k & 1:
                continue
            if call_any.search(lines[j]):
                out.add(lines[j][:60])
                continue
        first = False
        if not preds[j] and j == 0:
            out.add('ENTRY')
        for p in preds[j]:
            if p not in seen:
                seen.add(p)
                work.append(p)
        if j == 0:
            out.add('ENTRY')
    return ' ; '.join(sorted(out))


def main():
    gen = sys.argv[1]
    want_toml = '--toml' in sys.argv
    global KILL_CALLS
    global VOL_CTX
    if '--volatile' in sys.argv or '--volatile-ctx' in sys.argv:
        setup_volatile()
        VOL_CTX = '--volatile-ctx' in sys.argv
    KILL_CALLS = '--kill-calls' in sys.argv
    noreturn = set()
    for a in sys.argv[2:]:
        if a.startswith('--noreturn='):
            noreturn |= {x if x.startswith('sub_') else 'sub_' + x.upper().replace('0X', '')
                         for x in a.split('=', 1)[1].split(',') if x}
    cfgs = {}
    for f in sorted(glob.glob(gen + '/*_recomp.*.cpp')):
        cur = None
        body = []
        for ln in open(f, encoding='utf-8', errors='ignore'):
            m = fn_re.match(ln)
            if m:
                cur = m.group(1) or m.group(2)
                body = []
                continue
            if cur is None:
                continue
            if ln.startswith('}'):
                cfgs[cur] = build_cfg(body)
                cur = None
                continue
            body.append(ln)
    # functions that never return (they longjmp/abort on every path), to a fixpoint
    seeds = set(noreturn)
    changed = True
    while changed:
        changed = False
        for nm, cfg in cfgs.items():
            if nm not in noreturn and not returns(cfg, noreturn):
                noreturn.add(nm)
                changed = True
    print('# noreturn:', len(noreturn), 'functions (seeds', ' '.join(sorted(seeds)) + ')', file=sys.stderr)
    if '--list-noreturn' in sys.argv:
        print('# noreturn list:', ' '.join(sorted(noreturn)), file=sys.stderr)
    res = {}
    only_stores = {}
    global WHY
    why = {a.split('=', 1)[1] for a in sys.argv[2:] if a.startswith('--why=')}
    for nm, cfg in cfgs.items():
        WHY = nm in why
        bad, stores = analyze(cfg, noreturn)
        if bad:
            res[nm] = bad
        elif stores:
            only_stores[nm] = stores
    for nm in sorted(res):
        regs = sorted(res[nm])
        if want_toml:
            print(f'"0x{nm[4:]}" = {{ share_registers = true }}  # {" ".join(regs)}')
        else:
            print(nm, ' '.join(regs), '|', list(res[nm].values())[0])
    for nm in sorted(only_stores):
        print('# store-only (ignored):', nm, ' '.join(sorted(only_stores[nm])), file=sys.stderr)
    print('TOTAL', len(res), file=sys.stderr)


if __name__ == '__main__':
    main()
