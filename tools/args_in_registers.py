#!/usr/bin/env python3
"""Arguments in registers (reference implementation of the codegen pass; post-codegen step).

Every recompiled function is `void sub_X(PPCContext& ctx, uint8_t* base)` and passes its arguments, return value
and every other register through the PPCContext in memory. The generated code is full of `ctx.r3 = ...; call;
r4 = ctx.r3` traffic. This tool rewrites the generated tree so that
direct calls between ordinary functions pass r3-r10, f1-f13, r0/r11/r12/f0 and lr as C++ arguments and return r3 as
the C++ return value. The same algorithm is implemented inside the codegen (SDK branch args-in-registers,
src/codegen/args_in_registers.cpp); the two produce byte-identical function bodies, this script stays as the
reference and as the fallback for a rexglue without the option.

  sub_X(ctx, base)            stable ABI entry: dispatch table, indirect calls, hooks, runtime. A thin wrapper:
                              unpack ctx -> __fast_sub_X -> store r3 back. A function that nothing calls directly
                              keeps ONE body (it loads what it reads from ctx into locals).
  __fast_sub_X(ctx, base,     fast entry, used by direct calls whose callee is not hooked. The body of the
               p_r3, ..)      function, with every access to a converted register turned into a C++ local.

Rules:
  * Par: registers read before they are written, by the function itself or by a fast callee, on some path from the
    entry, arrive by value (interprocedural liveness). Registers that are only handed on (to a callee, or stored to
    ctx) can instead be loaded from ctx at entry (Ld): the callers then store them (Fw).
  * An unknown callee (indirect call, hooked function, kernel import, ineligible function) may read every argument
    register and lr and write all of them. Before such a call the locals that differ from ctx are stored to ctx;
    after it the ones that are live are reloaded. A register that still holds its ENTRY value at such a call is
    forwarded untouched through ctx (Fw): a thunk that tail-calls through a vtable keeps its arguments in ctx.
  * r0/r11/r12/f0 are scratch: unknown callees are assumed not to read them (assumption A1), they are synchronised
    only when live across a call. A register a callee clobbers and nobody reads is junk (A2).
  * r3 is returned by value when the function may define it itself (Val); every other register a direct caller
    reads after the call (and f1/r3 for indirect callers) is exported through ctx at the function exits (Ex).
  * Functions with a hook (any 82xxxxxx address in the sources, as direct_calls.py), setjmp users and
    ppc_trap with a type other than 0 keep the old ABI: calls to them are unknown calls.

Usage (all on the tree written by the codegen, in place):
  python3 tools/args_in_registers.py --gen app/generated/default --apply [--instrument] [--poison] [--uninit]
      --instrument: ME_ARGS_COUNT call counters, --poison: ME_ARGS_POISON before fast calls, --uninit: locals start as
      0xDEADBEEF.... instead of 0 (a read of a never-assigned local becomes visible in traces and self-tests)
  python3 tools/args_in_registers.py --gen <dir>            statistics only
  python3 tools/args_in_registers.py --write-hooked FILE   list for the codegen option args_in_registers_exclude_file
  python3 tools/args_in_registers.py --fix-ordering FILE   add .text.__fast_sub_X after each .text.__imp__sub_X (ld script)
"""
import os
import re
import sys
import time
import collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, 'app', 'generated', 'default')

# ---------------------------------------------------------------------------------------------- registers
# ARG registers (r3-r10, f1-f13) are forwarded to unknown callees; TEMP registers (r0, r11, r12, f0) are scratch:
# an unknown callee is assumed not to read them (they are only synchronised when live across the call).
RN = ['r%d' % i for i in range(3, 11)] + ['f%d' % i for i in range(1, 14)] + ['r0', 'r11', 'r12', 'f0'] + ['lr']
IDX = {n: i for i, n in enumerate(RN)}
NR = len(RN)
ALL = (1 << NR) - 1
TEMP = (1 << 21) | (1 << 22) | (1 << 23) | (1 << 24)
FWD = ALL & ~TEMP          # r3-r10, f1-f13 and lr are forwarded to unknown callees
LR = 1 << IDX['lr']
R3 = 1 << IDX['r3']
F1 = 1 << IDX['f1']
GPR_MASK = (1 << 8) - 1
REG_ALT = r'r(?:[3-9]|1[0-2]|0)|f(?:[0-9]|1[0-3])'
REF = re.compile(r'\bctx\.(' + REG_ALT + r'|lr)\b')
FULL_DEF = re.compile(r'^ctx\.(?:(' + REG_ALT + r')\.(?:u64|s64|f64)|(lr))\s*=(?!=)')
ANY_DEF = re.compile(r'^ctx\.(' + REG_ALT + r')\.\w+\s*(?:[-+|&^]|<<|>>)?=(?!=)')

# ---------------------------------------------------------------------------------------------- statements
(K_PLAIN, K_LABEL, K_GOTO, K_IFGOTO, K_IFRET, K_RET, K_IFOPEN, K_CLOSE, K_BLOCK, K_SWITCH, K_CASE, K_DEFAULT,
 K_STOP, K_CALLF, K_CALLU) = range(15)

FN_DEF = re.compile(r'^DEFINE_REX_FUNC\((\w+)\) \{$')
DECL = re.compile(r'^(?:PPCRegister|PPCXERRegister|PPCCRRegister|PPCVRegister|PPCFPRegister|uint32_t|uint64_t|PPCContext)\b.*\{\};$')
LABEL = re.compile(r'^(loc_[0-9A-F]+):$')
GOTO = re.compile(r'^goto (loc_[0-9A-F]+);$')
IFGOTO = re.compile(r'^if \(.*\) goto (loc_[0-9A-F]+);$')
IFRET = re.compile(r'^if \(.*\) return;$')
IFOPEN = re.compile(r'^if \(.*\) \{$')
SWITCH = re.compile(r'^switch \(.*\) \{$')
CASE = re.compile(r'^case -?\d+:$')
CALL = re.compile(r'^(?:__imp__)?([A-Za-z_]\w*)\(ctx, base\);$')   # sub_X, __savevmx_N..., kernel imports
NONVOL = re.compile(r'\bctx\.(?:r(?:1[4-9]|2[0-9]|3[01])|f(?:1[4-9]|2[0-9]|3[01]))\b')


def strip_comment(s):
    i = s.find('//')
    return s[:i].rstrip() if i >= 0 else s


def stmt_masks(s):
    """(use, full-def, partial-def) masks of the tracked registers for one statement."""
    use = 0
    dfull = 0
    dpart = 0
    m = FULL_DEF.match(s)
    if m:
        dfull = 1 << IDX[m.group(1) or m.group(2)]
        rest = s[m.end():]
    else:
        m = ANY_DEF.match(s)
        if m:
            dpart = 1 << IDX[m.group(1)]
            use |= dpart  # read-modify-write
            rest = s[m.end():]
        else:
            rest = s
    for r in REF.findall(rest):
        use |= 1 << IDX[r]
    return use, dfull, dpart


class Fn:
    """Static facts about one generated function (no callee-dependent information)."""
    __slots__ = ('name', 'file', 'n', 'kind', 'use', 'dfull', 'dpart', 'succ', 'callee', 'reason', 'nstm')


def parse_body(body):
    """Statements of a function body. Returns (stmts, raw_index): stmt text (no comment) and its index in `body`."""
    stmts = []
    raw = []
    for i, ln in enumerate(body):
        s = ln.strip()
        if not s or s.startswith('//') or s.startswith('REX_FUNC_PROLOGUE'):
            continue
        s = strip_comment(s)
        if not s:
            continue
        if DECL.match(s) and not stmts:
            continue
        stmts.append(s)
        raw.append(i)
    return stmts, raw


def classify(s):
    if s == 'return;':
        return K_RET, None
    m = LABEL.match(s)
    if m:
        return K_LABEL, m.group(1)
    m = GOTO.match(s)
    if m:
        return K_GOTO, m.group(1)
    m = IFGOTO.match(s)
    if m:
        return K_IFGOTO, m.group(1)
    if IFRET.match(s):
        return K_IFRET, None
    if IFOPEN.match(s):
        return K_IFOPEN, None
    if SWITCH.match(s):
        return K_SWITCH, None
    if s == '}':
        return K_CLOSE, None
    if s == '{':
        return K_BLOCK, None
    if CASE.match(s):
        return K_CASE, None
    if s == 'default:':
        return K_DEFAULT, None
    if s.startswith('__builtin_trap()') or s.startswith('REX_FATAL(') or s.startswith('ppc_longjmp('):
        return K_STOP, None
    m = CALL.match(s)
    if m:
        return K_CALLF, m.group(1)     # a call; eligible only if the name is a function of the tree (not an import)
    if 'REX_CALL_INDIRECT_FUNC(' in s:
        return K_CALLU, None
    return K_PLAIN, None


def analyse_function(name, fname, body, share_ok):
    """Builds the static description of a function. Returns None (with reason) when it must keep the old ABI."""
    stmts, raw = parse_body(body)
    f = Fn()
    f.name = name
    f.file = fname
    f.reason = None
    n = len(stmts)
    kinds = [0] * n
    tgt = [None] * n
    for i, s in enumerate(stmts):
        kinds[i], tgt[i] = classify(s)
        if kinds[i] == K_PLAIN:
            if 'env' in s and ('env = ctx' in s or 'ctx = env' in s or s.startswith('env')):
                f.reason = 'setjmp'
            elif s.startswith('ppc_trap(ctx') and not s.startswith('ppc_trap(ctx, base, 0)'):
                f.reason = 'trap'
            elif s.count(';') > 1 and REF.search(s):
                f.reason = 'multi'
            elif 'ppc_setjmp' in s:
                f.reason = 'setjmp'
    # a function whose body can fall off its end gets a synthetic `return;` (statement index n, no raw line)
    if n == 0 or kinds[-1] not in (K_RET, K_GOTO, K_STOP):
        stmts.append(None)
        raw.append(None)
        kinds.append(K_RET)
        tgt.append(None)
        n += 1
    labels = {}
    for i in range(n):
        if kinds[i] == K_LABEL:
            labels[tgt[i]] = i
    # brace matching
    stack = []
    close = {}
    switch_cases = {}
    cur_sw = []
    for i in range(n):
        k = kinds[i]
        if k in (K_IFOPEN, K_BLOCK, K_SWITCH):
            stack.append(i)
            if k == K_SWITCH:
                cur_sw.append(i)
                switch_cases[i] = []
        elif k == K_CLOSE:
            if stack:
                o = stack.pop()
                close[o] = i
                if cur_sw and cur_sw[-1] == o:
                    cur_sw.pop()
        elif k in (K_CASE, K_DEFAULT) and cur_sw:
            switch_cases[cur_sw[-1]].append(i)
    # a trailing synthetic return when the body can fall off its end is added by the emitter; model it here
    succ = []
    for i in range(n):
        k = kinds[i]
        nxt = [i + 1] if i + 1 < n else [n]  # n = virtual exit
        if k == K_GOTO:
            succ.append([labels[tgt[i]]] if tgt[i] in labels else [])
        elif k == K_IFGOTO:
            succ.append(nxt + ([labels[tgt[i]]] if tgt[i] in labels else []))
        elif k == K_IFRET:
            succ.append(nxt + [n])
        elif k == K_RET:
            succ.append([n])
        elif k == K_STOP:
            succ.append([])
        elif k == K_SWITCH:
            succ.append(list(switch_cases[i]) or nxt)
        elif k == K_IFOPEN:
            succ.append(nxt + ([close[i] + 1 if close[i] + 1 < n else n] if i in close else []))
        else:
            succ.append(nxt)
    use = [0] * n
    dfull = [0] * n
    dpart = [0] * n
    for i, s in enumerate(stmts):
        k = kinds[i]
        if k in (K_PLAIN, K_IFGOTO, K_IFRET, K_IFOPEN, K_SWITCH, K_STOP):
            use[i], dfull[i], dpart[i] = stmt_masks(s)
    # nonvolatile registers in ctx: a share_registers piece; legal for us (only r3-r13/f1-f13 are touched)
    f.n = n
    f.kind = kinds
    f.use = use
    f.dfull = dfull
    f.dpart = dpart
    f.succ = succ
    f.callee = tgt
    f.nstm = (stmts, raw)
    return f


# ---------------------------------------------------------------------------------------------- tree
def hooked_functions():
    """Same heuristic as tools/direct_calls.py: every 82xxxxxx address in the sources counts as hooked."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('direct_calls', os.path.join(ROOT, 'tools', 'direct_calls.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.hooked()


def split_file(path):
    """Returns (segments, lines): segments = [('raw', [lines]) | ('fn', name, body_lines)] in file order."""
    lines = open(path, encoding='utf-8').read().split('\n')
    segs = []
    cur_raw = []
    i = 0
    n = len(lines)
    while i < n:
        m = FN_DEF.match(lines[i])
        if m:
            if cur_raw:
                segs.append(('raw', cur_raw))
                cur_raw = []
            j = i + 1
            while j < n and lines[j] != '}':
                j += 1
            segs.append(('fn', m.group(1), lines[i + 1:j]))
            i = j + 1
        else:
            cur_raw.append(lines[i])
            i += 1
    if cur_raw:
        segs.append(('raw', cur_raw))
    return segs


def generated_files(gen):
    return sorted((f for f in os.listdir(gen) if re.match(r'sa2_recomp\.\d+\.cpp$', f)),
                  key=lambda f: int(f.split('.')[1]))


def load_tree(gen, hooked, only=None):
    funcs = {}
    for fname in generated_files(gen):
        for seg in split_file(os.path.join(gen, fname)):
            if seg[0] != 'fn':
                continue
            name, body = seg[1], seg[2]
            if only is not None and name not in only:
                continue
            f = analyse_function(name, fname, body, True)
            if name in hooked:
                f.reason = f.reason or 'hooked'
            funcs[name] = f
    return funcs


LEGACY = {}   # ineligible (hooked...) function -> TEMP registers its legacy body reads anywhere (inputs it may expect)


class Summary:
    """Interprocedural facts of one eligible function (only ever grow during the fixpoints)."""
    __slots__ = ('dany', 'val', 'fw', 'par', 'ex', 'dem', 'calls', 'ein_unk')

    def __init__(self):
        self.dany = 0
        self.val = False
        self.fw = 0
        self.par = 0
        self.ex = 0
        self.dem = 0
        self.calls = []


def phase1(funcs, elig, S):
    """dany (registers the function may define, ALL if it has an unknown call) and val (r3 defined locally)."""
    own = {}
    callees = {}
    for name in elig:
        f = funcs[name]
        o = 0
        unk = False
        cs = set()
        for i in range(f.n):
            o |= f.dfull[i] | f.dpart[i]
            k = f.kind[i]
            if k == K_CALLU:
                unk = True
            elif k == K_CALLF:
                if f.callee[i] in elig:
                    cs.add(f.callee[i])
                else:
                    unk = True
        s = S[name]
        s.dany = ALL if unk else o
        s.val = bool(o & R3)
        own[name] = o
        callees[name] = cs
    changed = True
    while changed:
        changed = False
        for name in elig:
            s = S[name]
            d = s.dany
            v = s.val
            for g in callees[name]:
                d |= S[g].dany
                v = v or S[g].val
            if d != s.dany or v != s.val:
                s.dany = d
                s.val = v
                changed = True
    return callees




def entry_reach(f, S, elig):
    """Forward may-analysis: registers whose ENTRY value may still be the current value before each statement
    (killed by full definitions and, after a call, by what the callee may define). Returns the call sites as
    (stmt, callee or None, mask)."""
    n = f.n
    IN = [0] * (n + 1)
    IN[0] = ALL
    work = collections.deque([0])
    inq = [False] * (n + 1)
    inq[0] = True
    kind = f.kind
    while work:
        i = work.popleft()
        inq[i] = False
        if i >= n:
            continue
        k = kind[i]
        if k == K_CALLU:
            kill = ALL
        elif k == K_CALLF:
            g = f.callee[i]
            kill = S[g].dany if g in elig else ALL
        else:
            kill = f.dfull[i]
        out = IN[i] & ~kill
        for j in f.succ[i]:
            if j <= n and (IN[j] | out) != IN[j]:
                IN[j] |= out
                if not inq[j]:
                    inq[j] = True
                    work.append(j)
    sites = []
    for i in range(n):
        k = kind[i]
        if k == K_CALLU:
            sites.append((i, None, IN[i]))
        elif k == K_CALLF:
            g = f.callee[i]
            sites.append((i, g if g in elig else None, IN[i]))
        elif k == K_RET or k == K_IFRET:
            sites.append((i, '', IN[i]))      # exit: exported registers still holding their entry value
    return sites


def fweff(G):
    return G.fw & ~G.par


def evaluate(f, S, elig, detail=False, direct=True):
    """One evaluation of function f against the current summaries.
    Returns (newpar, newfw, demands{callee: mask}); with detail also the per-statement facts for the emitter."""
    s = S[f.name]
    n = f.n
    kind = f.kind
    succ = f.succ
    use = f.use
    dfull = f.dfull
    dpart = f.dpart
    gs = [None] * n
    for i in range(n):
        if kind[i] == K_CALLF and f.callee[i] in elig:
            gs[i] = S[f.callee[i]]
    val = s.val
    ex = s.ex
    # ---- forward: ctx already holds the current value (must analysis)
    cv = [ALL] * (n + 1)
    # a function without direct callers (`direct` False) is entered through the wrapper only: it loads what it
    # reads from ctx, so ctx holds every entry value and r3 can go back through ctx at the exit
    cv[0] = ALL & ~s.par if direct else ALL
    changed = True
    while changed:
        changed = False
        for i in range(n):
            k = kind[i]
            x = cv[i]
            if k == K_CALLU:
                out = ALL
            elif k == K_CALLF:
                G = gs[i]
                if G is None:
                    out = ALL
                else:
                    # stored for the callee (Fw), or clobbered by it: exported through ctx or junk nobody reads
                    out = x | (G.fw & ~G.par) | G.dany
                    if G.val:
                        out &= ~R3
            else:
                out = x & ~(dfull[i] | dpart[i])
            for j in succ[i]:
                if j < n:
                    nv = cv[j] & out
                    if nv != cv[j]:
                        cv[j] = nv
                        changed = True
    # ---- backward liveness of the locals, two levels: H = values the C++ code itself reads (they must be
    # parameters), A = H plus values only needed to be handed on (callee parameters, stores to ctx): those can be
    # satisfied from ctx at entry (Ld) instead of becoming parameters.
    H = [0] * (n + 1)
    A = [0] * (n + 1)
    changed = True
    exit_val = R3 if (val and direct) else 0
    exit_soft = (R3 if (val and not direct) else 0)
    while changed:
        changed = False
        for i in range(n - 1, -1, -1):
            oh = 0
            oa = 0
            for j in succ[i]:
                oh |= H[j]
                oa |= A[j]
            k = kind[i]
            uh = 0
            us = 0
            if k == K_CALLU:
                us = (FWD | oa) & ~cv[i]
                kill = ALL
            elif k == K_CALLF:
                G = gs[i]
                if G is None:
                    us = (FWD | LEGACY.get(f.callee[i], 0) | oa) & ~cv[i]
                    kill = ALL
                else:
                    us = G.par | ((G.fw & ~G.par) & ~cv[i])
                    kill = G.dany
            elif k == K_RET:
                uh = exit_val
                us = (ex | exit_soft) & ~cv[i]
                kill = 0
            elif k == K_IFRET:
                uh = use[i] | exit_val
                us = (ex | exit_soft) & ~cv[i]
                kill = 0
            else:
                uh = use[i]
                kill = dfull[i]
            nh = uh | (oh & ~kill)
            na = uh | us | (oa & ~kill)
            if nh != H[i] or na != A[i]:
                H[i] = nh
                A[i] = na
                changed = True
    newpar = H[0]
    newld = A[0] & ~H[0]
    newfw = newld
    for (i, g, ein) in s.calls:
        if g is None:
            newfw |= ein & FWD
        elif g == '':
            newfw |= ein & ex
        else:
            newfw |= ein & (S[g].fw & ~S[g].par)
    dem = {}
    for i in range(n):
        G = gs[i]
        if G is not None:
            la = 0
            for j in succ[i]:
                la |= A[j]
            m = la & G.dany
            if G.val:
                m &= ~R3
            if m:
                g = f.callee[i]
                dem[g] = dem.get(g, 0) | m
    if detail:
        return newpar, newfw, dem, cv, (H, A), gs, newld
    return newpar, newfw, dem


def phase3(funcs, elig, S):
    callers = collections.defaultdict(set)
    for name in elig:
        f = funcs[name]
        for i in range(f.n):
            if f.kind[i] == K_CALLF and f.callee[i] in elig:
                callers[f.callee[i]].add(name)
    direct_set = set(callers)    # functions with at least one direct (fast) caller; fixed before any lookup
    for name in elig:
        S[name].calls = entry_reach(funcs[name], S, elig)
        S[name].dem = 0
        S[name].ex = (R3 | F1) & S[name].dany & ~(R3 if S[name].val else 0)
    work = collections.deque(sorted(elig))
    inq = set(work)
    evals = 0
    while work:
        name = work.popleft()
        inq.discard(name)
        f = funcs[name]
        s = S[name]
        evals += 1
        newpar, newfw, dem = evaluate(f, S, elig, direct=(name in direct_set))
        ch_up = False  # facts the callers depend on
        if newpar & ~s.par:
            s.par |= newpar
            ch_up = True
        if newfw & ~s.fw:
            s.fw |= newfw
            ch_up = True
        for g, m in dem.items():
            G = S[g]
            if m & ~G.dem:
                G.dem |= m
                nex = (G.dem | R3 | F1) & G.dany & ~(R3 if G.val else 0)
                if nex != G.ex:
                    G.ex = nex
                    if g not in inq:
                        inq.add(g)
                        work.append(g)
        if ch_up:
            for c in sorted(callers[name]):
                if c not in inq:
                    inq.add(c)
                    work.append(c)
            if name not in inq:  # own par changed: cv entry changes
                inq.add(name)
                work.append(name)
    return evals


def popcount(x):
    return bin(x).count('1')



# ---------------------------------------------------------------------------------------------- emitter
MARK = '// args_in_registers: applied'
COUNT = collections.Counter()
INSTRUMENT = False
POISON = False
UNINIT = False


def reg_names(mask):
    return [RN[i] for i in range(NR) if mask >> i & 1]


def is_f(name):
    return name[0] == 'f'


def member(name):
    return 'f64' if name[0] == 'f' else 'u64'


def lv(name):
    """C++ local (or ctx field) access path of a register: r3.u64, f1.f64, lr."""
    return name if name == 'lr' else name + '.' + member(name)


def sig_params(G):
    """C++ parameter list (type, name) of the fast entry, in register order."""
    out = []
    for nm in reg_names(G.par):
        out.append(('double' if is_f(nm) else 'uint64_t', 'p_' + nm))
    return out


def prototype(name, G):
    ret = 'uint64_t' if G.val else 'void'
    ps = ''.join(', %s %s' % (t, n) for t, n in sig_params(G))
    return 'extern "C" %s __fast_%s(PPCContext& __restrict ctx, uint8_t* base%s)' % (ret, name, ps)


def store_lines(mask, ind):
    return ['%sctx.%s = %s;' % (ind, lv(nm), lv(nm)) for nm in reg_names(mask)]


def reload_lines(mask, ind):
    return ['%s%s = ctx.%s;' % (ind, lv(nm), lv(nm)) for nm in reg_names(mask)]


def emit_function(f, S, elig, body, direct):
    """Returns the C++ text (list of lines) of wrapper + fast entry for one eligible function."""
    s = S[f.name]
    newpar, newfw, dem, cv, (H, A), gs, ld = evaluate(f, S, elig, detail=True, direct=direct)
    if (newpar & ~s.par) or (newfw & ~s.fw):
        raise RuntimeError('summaries of %s are not a fixpoint' % f.name)
    stmts, raw = f.nstm
    n = f.n
    rawmap = {}
    for i in range(n):
        if raw[i] is not None:
            rawmap[raw[i]] = i
    first_stmt = min(rawmap) if rawmap else len(body)
    touched = s.par | ld
    for i in range(n):
        touched |= f.use[i] | f.dfull[i] | f.dpart[i]
    out_head = []
    out_body = []
    exit_val = s.val

    def ret_stmt(ind):
        return ind + ('return r3.u64;' if (exit_val and direct) else 'return;')

    def exit_code(i, ind):
        nonlocal touched
        st = (s.ex | (R3 if (exit_val and not direct) else 0)) & ~cv[i]
        COUNT['exit_store'] += popcount(st)
        COUNT['exit_sites'] += 1
        touched |= st
        if exit_val and direct:
            touched |= R3
        return store_lines(st, ind) + [ret_stmt(ind)]

    def stmt_lines(i, line):
        nonlocal touched
        k = f.kind[i]
        ind = line[:len(line) - len(line.lstrip())]
        if k == K_RET:
            return exit_code(i, ind)
        if k == K_IFRET:
            m = re.match(r'^(\s*)if \((.*)\) return;(.*)$', line)
            cond = REF.sub(lambda mm: mm.group(1), m.group(2))
            return [ind + 'if (%s) {' % cond] + exit_code(i, ind + '\t') + [ind + '}']
        if k == K_CALLU or (k == K_CALLF and gs[i] is None):
            la = 0
            for j in f.succ[i]:
                la |= A[j]
            stores = (FWD | (LEGACY.get(f.callee[i], 0) if k == K_CALLF else 0) | la) & ~cv[i]
            touched |= stores | la
            COUNT['unk_store'] += popcount(stores)
            COUNT['unk_reload'] += popcount(la)
            COUNT['unk_sites'] += 1
            return store_lines(stores, ind) + [line] + reload_lines(la, ind)
        if k == K_CALLF:
            G = gs[i]
            g = f.callee[i]
            stores = (G.fw & ~G.par) & ~cv[i]
            la = 0
            for j in f.succ[i]:
                la |= A[j]
            rl = la & G.dany & ~(R3 if G.val else 0)
            if rl & ~G.ex:
                raise RuntimeError('reload of %s after call to %s not exported' % (reg_names(rl & ~G.ex), g))
            args = []
            for nm in reg_names(G.par):
                args.append(lv(nm))
            touched |= stores | rl | G.par
            COUNT['fast_store'] += popcount(stores)
            COUNT['fast_reload'] += popcount(rl)
            COUNT['fast_args'] += popcount(G.par)
            COUNT['fast_sites'] += 1
            call = '__fast_%s(ctx, base%s)' % (g, ''.join(', ' + a for a in args))
            lines = store_lines(stores, ind)
            if POISON:
                # tripwire: the callee clobbers these anyway and nothing may read them (see --poison in the doc)
                pm = G.dany & ~(G.fw & ~G.par)
                lines.append('%sME_ARGS_POISON(0x%X);' % (ind, pm))
            if G.val and (la & R3):
                touched |= R3
                lines.append('%sr3.u64 = %s;' % (ind, call))
            else:
                lines.append('%s%s;' % (ind, call))
            lines += reload_lines(rl, ind)
            return lines
        return [REF.sub(lambda mm: mm.group(1), line)]

    for r, line in enumerate(body):
        if r < first_stmt:
            out_head.append(line)
            continue
        if r in rawmap:
            out_body += stmt_lines(rawmap[r], line)
        else:
            out_body.append(line)
    # synthetic return at the end of the body
    if raw[n - 1] is None:
        out_body += exit_code(n - 1, '\t')
    # lint: after the rewrite no statement may still address a tracked register through ctx, except the
    # explicit synchronisation copies (ctx.rN = rN / rN = ctx.rN)
    sync = re.compile(r'^\s*(?:ctx\.(\w+)(?:\.\w+)? = (\w+)(?:\.\w+)?;|(\w+)(?:\.\w+)? = ctx\.(\w+)(?:\.\w+)?;)$')
    for ln in out_body:
        if REF.search(ln):
            m = sync.match(ln)
            if not m or (m.group(1) or m.group(4)) != (m.group(2) or m.group(3)):
                raise RuntimeError('%s: statement still uses ctx: %s' % (f.name, ln.strip()))
    decls = []
    for nm in reg_names(touched):
        if UNINIT:    # validation: a read of a never-assigned local returns a pattern the runs/tests can see
            decls.append('\tuint64_t lr = 0xDEADBEEF5A5A5A5Aull;' if nm == 'lr' else
                         '\tPPCRegister %s{}; %s.u64 = 0xDEADBEEF%s5A5Aull;' % (nm, nm, '%04X' % (0xA000 + RN.index(nm))))
        else:
            decls.append('\tuint64_t lr{};' if nm == 'lr' else '\tPPCRegister %s{};' % nm)
    init = []
    if not direct:
        COUNT['combined'] += 1
        ld = ld | s.par
        s_par_params = 0
    else:
        s_par_params = s.par
    COUNT['entry_ld'] += popcount(ld)
    COUNT['entry_par'] += popcount(s.par)
    COUNT['functions'] += 1
    for nm in reg_names(s_par_params):
        init.append('\t%s = p_%s;' % (lv(nm), nm))
    for nm in reg_names(ld & ~s_par_params):
        init.append('\t%s = ctx.%s;' % (lv(nm), lv(nm)))
    if not direct:
        head = list(out_head)
        if INSTRUMENT:
            head.insert(1, '\tME_ARGS_COUNT("%s", 0);' % f.name)
        return ['DEFINE_REX_FUNC(%s) {' % f.name] + head + decls + init + out_body + ['}']
    # wrapper
    wargs = ''.join(', ctx.%s' % lv(nm) for nm in reg_names(s.par))
    wrapper = ['DEFINE_REX_FUNC(%s) {' % f.name, '\tREX_FUNC_PROLOGUE();']
    if INSTRUMENT:
        wrapper.append('\tME_ARGS_COUNT("%s", 1);' % f.name)
    if s.val:
        wrapper.append('\tctx.r3.u64 = __fast_%s(ctx, base%s);' % (f.name, wargs))
    else:
        wrapper.append('\t__fast_%s(ctx, base%s);' % (f.name, wargs))
    wrapper.append('}')
    head = list(out_head)
    if INSTRUMENT:
        head.insert(1, '\tME_ARGS_COUNT("%s", 2);' % f.name)
    # Compiler barrier: without it GCC (guest loads are not volatile on the Switch) infers that a fast function which
    # only reads guest memory is `pure` and hoists / merges calls to it out of polling loops (the old ABI wrote ctx,
    # which prevented that): the console build froze after the first frame.
    head.insert(1, '\tasm volatile("" ::: "memory");')
    fast = [prototype(f.name, s) + ' {'] + head + decls + init + out_body + ['}']
    return wrapper + [''] + fast


def apply_tree(gen, funcs, elig, S, hooked):
    nfiles = 0
    called = set()
    for n in elig:
        f = funcs[n]
        for i in range(f.n):
            if f.kind[i] == K_CALLF and f.callee[i] in elig:
                called.add(f.callee[i])
    for fname in generated_files(gen):
        path = os.path.join(gen, fname)
        segs = split_file(path)
        text0 = open(path, encoding='utf-8').read()
        if MARK in text0:
            raise RuntimeError('%s already processed' % fname)
        protos = []
        seen = set()

        def need(g):
            if g in elig and g not in seen:
                seen.add(g)
                protos.append(prototype(g, S[g]) + ';')
        out = []
        for seg in segs:
            if seg[0] == 'raw':
                out.append(('raw', seg[1]))
                continue
            name, body = seg[1], seg[2]
            if name not in elig:
                out.append(('fn', name, body, None))
                continue
            f = funcs[name]
            if name in called:
                need(name)
            for i in range(f.n):
                if f.kind[i] == K_CALLF and f.callee[i] in elig:
                    need(f.callee[i])
            out.append(('fn', name, body, emit_function(f, S, elig, body, name in called)))
        lines = []
        placed = False
        for item in out:
            if item[0] == 'raw':
                lines += item[1]
                if not placed and any(l.startswith('#include') for l in item[1]):
                    lines.append(MARK)
                    if INSTRUMENT or POISON:
                        lines += ['#ifndef ME_ARGS_COUNT', '#define ME_ARGS_COUNT(name, kind) ((void)0)', '#endif',
                                  '#ifndef ME_ARGS_POISON', '#define ME_ARGS_POISON(mask) ((void)0)', '#endif']
                    lines += protos
                    placed = True
            else:
                _, name, body, new = item
                if new is None:
                    lines.append('DEFINE_REX_FUNC(%s) {' % name)
                    lines += body
                    lines.append('}')
                else:
                    lines += new
        if not placed:
            raise RuntimeError('no include line in %s' % fname)
        open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(lines))
        nfiles += 1
    return nfiles


def stats(funcs, elig, S):
    print('functions', len(funcs), 'eligible', len(elig))
    reasons = collections.Counter(f.reason for f in funcs.values())
    print('ineligible reasons', dict(reasons))
    c_fw = collections.Counter(popcount(fweff(S[n])) for n in elig)
    print('|Fw_eff| histogram', sorted(c_fw.items()))
    c_par = collections.Counter(popcount(S[n].par) for n in elig)
    print('|Par| histogram', sorted(c_par.items()))
    print('val r3:', sum(1 for n in elig if S[n].val), 'dany=ALL:', sum(1 for n in elig if S[n].dany == ALL))
    tot = 0
    byval = 0
    fwst = 0
    for n in elig:
        f = funcs[n]
        for i in range(f.n):
            if f.kind[i] == K_CALLF and f.callee[i] in elig:
                G = S[f.callee[i]]
                tot += 1
                byval += popcount(G.par)
                fwst += popcount(G.fw & ~G.par)
    print('direct fast call sites', tot, 'args by value', byval, 'forwarded-through-ctx regs (upper bound of stores)', fwst)
    for r in ['r3', 'r4', 'r5', 'r6', 'f1']:
        b = 1 << IDX[r]
        print(' ', r, 'Par at', sum(1 for n in elig for i in range(funcs[n].n) if funcs[n].kind[i] == K_CALLF and funcs[n].callee[i] in elig and S[funcs[n].callee[i]].par & b),
              'sites; Fw_eff at', sum(1 for n in elig for i in range(funcs[n].n) if funcs[n].kind[i] == K_CALLF and funcs[n].callee[i] in elig and fweff(S[funcs[n].callee[i]]) & b))


def fix_ordering(path):
    """app/function_order.ld lists `*(.text.__imp__sub_X)` hot sections: the body of a function that has direct
    callers now lives in `.text.__fast_sub_X`. Adds those lines (idempotent)."""
    lines = open(path).read().split('\n')
    have = set(l.strip() for l in lines)
    out = []
    added = 0
    for l in lines:
        out.append(l)
        m = re.match(r'^(\s*)\*\(\.text\.__imp__(\w+)\)$', l)
        if m:
            nl = '%s*(.text.__fast_%s)' % (m.group(1), m.group(2))
            if nl.strip() not in have:
                out.append(nl)
                have.add(nl.strip())
                added += 1
    open(path, 'w').write('\n'.join(out))
    print('ordering: %d __fast_ sections added to %s' % (added, path))


def main():
    t0 = time.time()
    if '--fix-ordering' in sys.argv:
        fix_ordering(sys.argv[sys.argv.index('--fix-ordering') + 1])
        return
    if '--write-hooked' in sys.argv:
        # list for the codegen option args_in_registers_exclude_file: every function that may be hooked
        out = sys.argv[sys.argv.index('--write-hooked') + 1]
        names = sorted(hooked_functions())
        open(out, 'w').write('\n'.join(names) + '\n')
        print('wrote %d hooked/excluded function names to %s' % (len(names), out))
        return
    gen = GEN
    if '--gen' in sys.argv:
        gen = sys.argv[sys.argv.index('--gen') + 1]
    only = None
    hooked = hooked_functions()
    funcs = load_tree(gen, hooked, only)
    print('parsed', len(funcs), 'functions in %.1fs' % (time.time() - t0))
    elig = {n for n, f in funcs.items() if f.reason is None}
    for n, f in funcs.items():
        if f.reason is not None:
            m = 0
            for i in range(f.n):
                m |= f.use[i]
            LEGACY[n] = m & TEMP
    S = {n: Summary() for n in elig}
    phase1(funcs, elig, S)
    print('phase1 %.1fs' % (time.time() - t0))
    ev = phase3(funcs, elig, S)
    print('phase3 evaluations', ev, '%.1fs' % (time.time() - t0))
    stats(funcs, elig, S)
    global INSTRUMENT, POISON, UNINIT
    INSTRUMENT = '--instrument' in sys.argv
    POISON = '--poison' in sys.argv
    UNINIT = '--uninit' in sys.argv
    if '--apply' in sys.argv:
        nf = apply_tree(gen, funcs, elig, S, hooked)
        print('rewrote', nf, 'files in %.1fs' % (time.time() - t0))
        print(dict(COUNT))


if __name__ == '__main__':
    main()
