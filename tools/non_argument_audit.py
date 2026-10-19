#!/usr/bin/env python3
"""Whole-program audit for the codegen option non_argument_as_local (r0, r2, r11, r12, f0 and v32-v63 as C++ locals).

With the option a recompiled function keeps those registers in C++ locals that start at zero, so a value can no longer
cross a call in them. This tool finds every place where one does in the original program, and checks that the
generated code hands it over.

1. Code generated WITHOUT the option (every use spelled ctx.X), the default mode:
     ENTRY  registers a function may read before writing them on some path: an argument a caller passes there. A call
            uses what its callee reads on entry (transitively), so a value passed through a function that never touches
            it is found too.
     RET    registers read after a call before being written, where the value may come from the callee (directly or
            through its own callees): a value returned there. Calls whose callee never writes the register are
            transparent: MSVC's interprocedural register allocation keeps values live across them, and a local does too.
   Stores count as uses (a stored argument is used), and so do partial writes (r11.u32 = ..., v63.u16[4] = ...), which
   keep the other bytes; runs of element writes that cover a whole vector register (vlogefp128, vupkd3d128) count as
   one full write. Unknown callees (indirect calls, imports, other modules) may write everything; the bug check and
   the --noreturn functions never return (their fallthrough is dead).
     python tools/non_argument_audit.py app/generated/default --noreturn=827F2E48

   Expected for SA2 (2026-10-07): ENTRY = the SDK's __save*/__rest* helpers (r12), the MSVC EH funclets (r12, all
   share_registers), sub_827F5144 (the stack probe, r12) and the three audio resamplers sub_828A0DB0 / sub_828A2A60 /
   sub_828A3220 (v57-v63, shifted out: tools/vec_shiftout.py proves it); RET = sub_82805400 (r11 from its __finally
   funclet, taken back by the generated copy-back) and sub_8275B300 (`subfe r10,r11,r11`, independent of r11).

2. Code generated WITH the option (--with-option): every call from a function that localizes these registers to one
   that keeps them in ctx (share_registers, non_argument_as_local_keep, the helpers) must copy in what the callee reads
   on entry and copy back what the caller reads afterwards that the callee may write; and no localized register may be
   read before it is written (except the resamplers above).
     python tools/non_argument_audit.py app/generated/default --with-option --noreturn=827F2E48
"""
import glob
import os
import re
import sys

VOL_REG = r'r(?:0|2|11|12)|f0|v(?:3[2-9]|4[0-9]|5[0-9]|6[0-3])'
NAMES = ['r0', 'r2', 'r11', 'r12', 'f0'] + [f'v{i}' for i in range(32, 64)]
IDX = {n: i for i, n in enumerate(NAMES)}
ALL = (1 << len(NAMES)) - 1
vol_ctx = re.compile(r'\bctx\.(' + VOL_REG + r')\b')
tok = re.compile(r'(?<![.\w])(' + VOL_REG + r')\b')
fn_re = re.compile(r'^DEFINE_REX_FUNC\(([A-Za-z_][A-Za-z0-9_]*)\) \{$')
lab_re = re.compile(r'^(loc_[0-9A-F]+):$')
goto_re = re.compile(r'^goto (loc_[0-9A-F]+);$')
ifgoto_re = re.compile(r'^if \(.*\) goto (loc_[0-9A-F]+);$')
decl_re = re.compile(r'^(?:PPCRegister|PPCXERRegister|PPCCRRegister|PPCVRegister|PPCFPRegister|uint32_t|uint64_t|'
                     r'PPCContext)\b')
local_decl = re.compile(r'^\t(?:PPCRegister|PPCVRegister) (' + VOL_REG + r')\{\};$', re.M)
full_w = re.compile(r'^(' + VOL_REG + r')(?:\.(?:u64|s64|f64))?\s*=(?!=)')
part_w = re.compile(r'^(' + VOL_REG + r')\.(?:u32|s32|u16|s16|u8|s8|f32)(?:\[[^\]]*\])?\s*[-+|&^]?=(?!=)')
comp_w = re.compile(r'^(' + VOL_REG + r')\.\w+(?:\[[^\]]*\])?\s*[-+|&^]=(?!=)')
vec_w = re.compile(r'^simde_mm_store\w*\(\(?(?:simde__m128i?\*\))?(' + VOL_REG + r')\.\w+,(.*)$')
elem_w = re.compile(r'^(v(?:3[2-9]|4[0-9]|5[0-9]|6[0-3]))\.(u32|s32|f32|u16|s16|u8|s8)\[(\d+)\]\s*([-+|&^]?)=(?!=)(.*)$')
ESIZE = {'u32': 4, 's32': 4, 'f32': 4, 'u16': 2, 's16': 2, 'u8': 1, 's8': 1}
call_re = re.compile(r'\b((?:__imp__)?(?:sub_[0-9A-F]+|__imp__\w+|__\w+_\d+))\(ctx, base\);')
indirect_re = re.compile(r'REX_CALL_INDIRECT_FUNC|\bppc_setjmp\b')
# the generated hand-over around a call (emit_call_sharing_registers / emit_call_handing_non_arguments)
sync_in = re.compile(r'^const auto s_(\w+) = ctx\.\1; ctx\.\1 = \1;$')
sync_back = re.compile(r'^(\w+) = ctx\.\1;$')
sync_restore = re.compile(r'^ctx\.(\w+) = s_\1;$')

WITH_OPTION = '--with-option' in sys.argv


def norm(s, ctx_mode):
    """register spelling: in code without the option (or a ctx-mode function) ctx.X is the register X; with the
    option a localizing function's register is the local X and ctx.X is only the hand-over copy."""
    return vol_ctx.sub(r'\1', s) if ctx_mode else s


def uses_defs(s):
    defs = uses = 0
    rest = s
    m = vec_w.match(s)
    if m and m.group(1) in IDX:
        defs |= 1 << IDX[m.group(1)]
        rest = m.group(2)
    else:
        m = full_w.match(s)
        if m:
            defs |= 1 << IDX[m.group(1)]
            rest = s[m.end():]
        else:
            m = part_w.match(s) or comp_w.match(s)
            if m:
                b = 1 << IDX[m.group(1)]
                defs |= b
                uses |= b  # the bytes not written keep the old value
                rest = s[m.end():]
    for r in tok.findall(rest):
        uses |= 1 << IDX[r]
    return uses, defs


def compute_ud(lines):
    ud = [uses_defs(s) for s in lines]
    i, n = 0, len(lines)
    while i < n:
        m = elem_w.match(lines[i])
        if not m or m.group(4):
            i += 1
            continue
        reg = m.group(1)
        mentions = re.compile(r'(?<![.\w])' + reg + r'\b')
        covered, j, ok = 0, i, True
        while j < n:
            mj = elem_w.match(lines[j])
            if not mj or mj.group(1) != reg:
                s = lines[j]
                # statements that do not touch the register may sit between the element writes (vupkd3d128)
                if not mentions.search(s) and not s.startswith(('loc_', 'goto ', 'return', '{', '}', 'switch',
                                                                'case ', 'default:')) \
                        and 'goto ' not in s and '(ctx, base)' not in s and 'REX_CALL_INDIRECT_FUNC' not in s:
                    j += 1
                    continue
                break
            size = ESIZE[mj.group(2)]
            bits = ((1 << size) - 1) << (int(mj.group(3)) * size)
            if mj.group(4) and (bits & ~covered):
                break
            if mentions.search(mj.group(5)):
                ok = False
                break
            covered |= bits
            j += 1
        if ok and covered == 0xFFFF:
            b = 1 << IDX[reg]
            for t in range(i, j):
                u, d = ud[t]
                ud[t] = (u & ~b, d | b)
            i = j
        else:
            i += 1
    return ud


class Func:
    def __init__(self, name, body):
        self.name = name
        text = ''.join(body)
        self.localizes = WITH_OPTION and bool(local_decl.search(text))
        ctx_mode = not self.localizes
        raw = []
        for ln in body:
            s = ln.strip()
            if not s or s.startswith('//') or s.startswith('REX_FUNC_PROLOGUE'):
                continue
            if decl_re.match(s) and s.endswith('{};'):
                continue
            raw.append(s)
        self.raw = raw
        self.lines = [norm(s, ctx_mode) for s in raw]
        n = len(raw)
        labels = {m.group(1): i for i, s in enumerate(raw) for m in [lab_re.match(s)] if m}
        stack, match_close = [], {}
        for i, s in enumerate(raw):
            opens = s.endswith('{') and not s.startswith('}')
            if s.startswith('{') and s.endswith('}'):
                opens = False
            if opens:
                stack.append(i)
            elif s.startswith('}'):
                if stack:
                    match_close[stack.pop()] = i
        switch_cases, cur = {}, []
        for i, s in enumerate(raw):
            if s.startswith('switch ('):
                cur.append(i)
                switch_cases[i] = []
            elif (s.startswith('case ') or s == 'default:') and cur:
                switch_cases[cur[-1]].append(i)
            if cur and match_close.get(cur[-1]) == i:
                cur.pop()
        succ = []
        for i, s in enumerate(raw):
            nxt = [i + 1] if i + 1 < n else []
            m = goto_re.match(s)
            if m:
                succ.append([labels[m.group(1)]] if m.group(1) in labels else [])
                continue
            m = ifgoto_re.match(s)
            if m:
                succ.append(nxt + ([labels[m.group(1)]] if m.group(1) in labels else []))
                continue
            if s in ('return;', 'return r3.u64;') or s.startswith(('__builtin_trap()', 'REX_FATAL(')) or \
                    'ppc_longjmp(' in s:
                succ.append([])
                continue
            if s.startswith('switch ('):
                succ.append(switch_cases[i])
                continue
            if s.startswith('if (') and s.endswith('{') and i in match_close:
                succ.append(nxt + ([match_close[i] + 1] if match_close[i] + 1 < n else []))
                continue
            succ.append(nxt)
        self.succ = succ
        self.calls = {}
        for i, s in enumerate(raw):
            cs = [c[len('__imp__'):] if c.startswith('__imp__sub_') else c for c in call_re.findall(s)]
            ind = bool(indirect_re.search(s))
            if cs or ind:
                self.calls[i] = (cs, ind)
        # the hand-over statements around each call: copy-in, copy-back (with the option they are not semantic
        # reads or writes of the local, they are checked separately)
        self.sync = {}
        if self.localizes:
            for i in self.calls:
                j = i - 1
                cin = set()
                while j >= 0 and sync_in.match(raw[j]):
                    cin.add(sync_in.match(raw[j]).group(1))
                    j -= 1
                k = i + 1
                cback = set()
                while k < n and (sync_back.match(raw[k]) or sync_restore.match(raw[k])):
                    m = sync_back.match(raw[k])
                    if m:
                        cback.add(m.group(1))
                    k += 1
                self.sync[i] = (cin, cback)
        self.ud = compute_ud(self.lines)
        if self.localizes:
            # the copy-in/restore lines touch only ctx; a copy-back is a write of the local
            for i, s in enumerate(raw):
                if sync_in.match(s) or sync_restore.match(s):
                    self.ud[i] = (0, 0)
                elif sync_back.match(s):
                    r = sync_back.match(s).group(1)
                    self.ud[i] = (0, (1 << IDX[r]) if r in IDX else 0)


def load(gen):
    funcs = {}
    for f in sorted(glob.glob(os.path.join(gen, '*_recomp.*.cpp'))):
        cur, body = None, []
        for ln in open(f, encoding='utf-8', errors='replace'):
            m = fn_re.match(ln.rstrip('\n'))
            if m:
                cur, body = m.group(1), []
                continue
            if cur is None:
                continue
            if ln.startswith('}'):
                funcs[cur] = Func(cur, body)
                cur = None
                continue
            body.append(ln)
    return funcs


def main():
    gen = sys.argv[1]
    funcs = load(gen)
    noreturn = {'__imp__KeBugCheck', '__imp__KeBugCheckEx'}
    proven = set()
    for a in sys.argv[2:]:
        if a.startswith('--noreturn='):
            noreturn |= {'sub_' + x.upper() for x in a.split('=', 1)[1].split(',') if x}
        if a.startswith('--proven='):
            # functions whose v32-v63 entry reads are proven not to matter (tools/vec_shiftout.py): not passed on
            # to their callers
            proven |= {'sub_' + x.upper().replace('SUB_', '') for x in a.split('=', 1)[1].split(',') if x}
    vmask = sum(1 << IDX[f'v{i}'] for i in range(32, 64))

    def returns(fn):
        if not fn.raw:
            return True
        seen, work = {0}, [0]
        while work:
            i = work.pop()
            s = fn.raw[i]
            if s in ('return;', 'return r3.u64;') or s.endswith(' return;') or s.endswith(' return r3.u64;'):
                return True
            if i in fn.calls and any(x in noreturn for x in fn.calls[i][0]):
                continue
            if i == len(fn.raw) - 1 and not fn.succ[i] and not s.startswith(('goto ', '__builtin_trap', 'REX_FATAL')):
                return True
            for j in fn.succ[i]:
                if j not in seen:
                    seen.add(j)
                    work.append(j)
        return False

    changed = True
    while changed:
        changed = False
        for nm, fn in funcs.items():
            if nm not in noreturn and not returns(fn):
                noreturn.add(nm)
                changed = True

    # what a call can write in the caller's view: without the option every register write; with it only the
    # writes of functions that keep the registers in ctx (a localizing function's writes are its own locals, and its
    # hand-over restores ctx)
    own_w, callees, unknown = {}, {}, {}
    for nm, fn in funcs.items():
        w = 0
        if not fn.localizes:
            for u, d in fn.ud:
                w |= d
        own_w[nm] = w
        cs, unk = set(), False
        for i, (c, ind) in fn.calls.items():
            unk |= ind
            for x in c:
                if x in funcs:
                    cs.add(x)
                elif x.startswith(('__imp__', 'sub_')):
                    unk = True
        callees[nm], unknown[nm] = cs, unk
    mayw = {nm: (ALL if unknown[nm] and not funcs[nm].localizes else own_w[nm]) for nm in funcs}
    changed = True
    while changed:
        changed = False
        for nm in funcs:
            v = mayw[nm]
            if not funcs[nm].localizes:
                for c in callees[nm]:
                    v |= mayw[c]
            if v != mayw[nm]:
                mayw[nm] = v
                changed = True

    def callee_writes(fn, i):
        c, ind = fn.calls[i]
        if ind:
            return ALL
        w = 0
        for x in c:
            w |= mayw.get(x, ALL)
        return w

    entry_set = {nm: 0 for nm in funcs}

    def analyze(fn, report):
        lines, n = fn.lines, len(fn.lines)
        succ = [([] if (i in fn.calls and any(x in noreturn for x in fn.calls[i][0])) else fn.succ[i])
                for i in range(n)]
        ud = list(fn.ud)
        # A call reads what its callee reads on entry. In a localizing function only what it hands over is read
        # (from its locals); a register the callee reads that is not handed over is reported below.
        for i, (c, ind) in fn.calls.items():
            u, d = ud[i]
            need = 0
            for x in c:
                need |= entry_set.get(x, 0)
            if fn.localizes:
                cin = fn.sync.get(i, (set(), set()))[0]
                need &= sum(1 << IDX[r] for r in cin if r in IDX)
            ud[i] = (u | need, d)
        preds = [[] for _ in range(n)]
        for i in range(n):
            for j in succ[i]:
                preds[j].append(i)
        E_out, C_out = [0] * n, [0] * n
        E_in, C_in = [0] * n, [0] * n
        work = list(range(n))[::-1]
        inw = [True] * n
        while work:
            i = work.pop()
            inw[i] = False
            e = ALL if i == 0 else 0
            c = 0
            for p in preds[i]:
                e |= E_out[p]
                c |= C_out[p]
            E_in[i], C_in[i] = e, c
            u, d = ud[i]
            eo, co = e & ~d, c & ~d
            if i in fn.calls:
                co |= callee_writes(fn, i)
            if (eo, co) != (E_out[i], C_out[i]):
                E_out[i], C_out[i] = eo, co
                for j in succ[i]:
                    if not inw[j]:
                        inw[j] = True
                        work.append(j)
        entry = 0
        if not report:
            for i in range(n):
                entry |= ud[i][0] & E_in[i]
            return entry

        def ret_sources(i, k):
            seen, work2, out = {i}, list(preds[i]), set()
            while work2:
                j = work2.pop()
                if j in seen:
                    continue
                seen.add(j)
                if j in fn.calls and callee_writes(fn, j) >> k & 1:
                    out.add(','.join(fn.calls[j][0]) or 'INDIRECT')
                    continue
                if ud[j][1] >> k & 1:
                    continue
                work2.extend(preds[j])
            return ' '.join(sorted(out))

        ent, ret = {}, {}
        for i in range(n):
            u = ud[i][0]
            for k, r in enumerate(NAMES):
                if u >> k & 1:
                    if E_in[i] >> k & 1 and r not in ent:
                        ent[r] = lines[i][:110]
                    if C_in[i] >> k & 1 and r not in ret:
                        ret[r] = lines[i][:90] + ' <= ' + ret_sources(i, k)
        # The hand-over at each call of a localizing function: what the callee reads on entry must be copied in. (What
        # the caller reads afterwards and the callee may write must be copied back: a missing copy-back shows as RET,
        # since a copy-back is a write of the local.)
        bad = []
        if fn.localizes:
            for i, (c, ind) in fn.calls.items():
                cin = fn.sync.get(i, (set(), set()))[0]
                need_in = 0
                for x in c:
                    need_in |= entry_set.get(x, 0)
                for k, r in enumerate(NAMES):
                    if need_in >> k & 1 and r not in cin:
                        bad.append(f'call {",".join(c)} at line {i}: {r} not handed over')
        return ent, ret, bad

    # entry sets to a fixpoint (a call reads what its callee reads on entry)
    changed = True
    rounds = 0
    while changed:
        changed = False
        rounds += 1
        for nm, fn in funcs.items():
            e = analyze(fn, False)
            if nm in proven:
                e &= ~vmask
            if e != entry_set[nm]:
                entry_set[nm] = e
                changed = True
    print(f'# {len(funcs)} functions, {sum(1 for f in funcs.values() if f.localizes)} localize them, '
          f'noreturn {len(noreturn)}, entry fixpoint in {rounds} rounds', file=sys.stderr)
    counts = {'ENTRY': 0, 'RET': 0, 'HANDOVER': 0}
    for nm in sorted(funcs):
        ent, ret, bad = analyze(funcs[nm], True)
        if ent:
            counts['ENTRY'] += 1
            print('ENTRY', nm, ' '.join(sorted(ent, key=IDX.get)), '|', list(ent.values())[0])
        if ret:
            counts['RET'] += 1
            print('RET', nm, ' '.join(sorted(ret, key=IDX.get)), '|', list(ret.values())[0])
        for b in bad:
            counts['HANDOVER'] += 1
            print('HANDOVER', nm, b)
    print('# ' + ', '.join(f'{k} {v}' for k, v in counts.items()), file=sys.stderr)


if __name__ == '__main__':
    main()
