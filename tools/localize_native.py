#!/usr/bin/env python3
"""Writes a "localized copy" of a recompiled guest function: a native replacement made mechanically from the generated
C++ (app/generated/default), for app/src/cpu/hot/n_<ADDR>.h (UnleashedRecomp-NX's switch-localize.py idea).

The copy runs exactly the same operations as the recompiled function, with two differences that make it faster:
  - the guest registers it uses (r, f, v, cr0-cr7, ctr, xer, reserved) live in C++ locals instead of the PPCContext:
    they are loaded from ctx at entry, written back to ctx before every call and every return, and reloaded after
    every call, so callers and callees see exactly the state the recompiled function would leave;
  - guest memory is accessed without `volatile` (sa2_hot_guest.cpp redefines the REX_LOAD/STORE macros), so GCC can
    pair and schedule the loads and stores.
ctx.lr, ctx.fpscr and ctx.msr stay in the context. Every register is written back, so the guard of sa2_hot_guest.cpp
can compare ALL registers, not only the return registers.

Exactness caveat: floating-point code must not contain double-precision multiply/add pairs that GCC could contract
differently in the two copies. Every single-precision result of the generated code is rounded through float(), which
no contraction crosses; the tool refuses functions with a bare double multiply, add or subtract unless --allow-double.

Calls (round 3, "smart sync"): instead of writing every localized register back before each call and reloading all
of them after it, a call writes back and reloads only the localized registers its callee can touch: the registers the
callee's generated body mentions, together with everything its callees mention, transitively (every generated function
is indexed: out/localize_index.json). Registers the callee cannot reach stay in locals and are written back at the next
call that can reach them, or at the return. A callee whose reach is unknown gets the full sync: indirect calls, traps,
imports and any name that is not a generated function, and functions overridden by an app hook other than the natives
of sa2_hot_guest.cpp (a native touches no more than its original). --sync full restores the old behaviour.

  python tools/localize_native.py 82146ED8 [--what "njCalcPoint ..."] [--out app/src/cpu/hot/n_82146ED8.h]
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REG = re.compile(r'\bctx\.(r\d+|f\d+|v\d+|cr\d|ctr|xer|reserved)\b')
# Calls: recompiled functions, the save/restore helpers (__savegprlr_N and friends read and write the context too),
# indirect calls and traps.
CALLEE = r'(?:__imp__)?(?:sub_[0-9A-F]+|__(?:save|rest)[a-z]+_\d+)'
# Any call of the generated code's calling convention: a generated function, a helper, an import or a host function
# (group 1), an indirect call or a trap (no group 1).
CALL = re.compile(r'\b([A-Za-z_]\w*)\(ctx, base\);|\bREX_CALL_INDIRECT_FUNC\([^;]*\);|\bppc_trap\(ctx, base[^;]*\);')
ALL = None  # "reach unknown": full sync
INDEX = os.path.join(ROOT, 'out', 'localize_index.json')
DEF = re.compile(r'^DEFINE_REX_FUNC\((\w+)\) \{$')


def build_index(gen):
    """name -> [mentioned registers, callees, unknown-reach flag] for every generated function (cached by mtime)."""
    import json
    files = sorted(glob.glob(os.path.join(gen, '*_recomp.*.cpp')))
    stamp = [[os.path.basename(f), os.path.getmtime(f)] for f in files]
    if os.path.exists(INDEX):
        try:
            cached = json.load(open(INDEX, encoding='utf-8'))
            if cached.get('stamp') == stamp and cached.get('gen') == os.path.abspath(gen):
                return cached['funcs']
        except (OSError, ValueError):
            pass
    funcs = {}
    for path in files:
        name = None
        for line in open(path, encoding='utf-8', errors='replace'):
            m = DEF.match(line.rstrip('\n'))
            if m:
                name = m.group(1)
                funcs[name] = [set(), set(), False]
                continue
            if name is None:
                continue
            if line.startswith('}'):
                name = None
                continue
            if line.lstrip().startswith('//'):
                continue
            f = funcs[name]
            for m in REG.finditer(line):
                f[0].add(m.group(1))
            for m in CALL.finditer(line):
                if m.group(1) is None:  # indirect call or trap
                    f[2] = True
                else:
                    f[1].add(m.group(1))
    out = {k: [sorted(v[0]), sorted(v[1]), v[2]] for k, v in funcs.items()}
    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    json.dump({'gen': os.path.abspath(gen), 'stamp': stamp, 'funcs': out}, open(INDEX, 'w', encoding='utf-8'))
    return out


def native_hooks():
    """Addresses of the natives of sa2_hot_guest.cpp: a native touches no more than its original. The localized copies
    (SA2_HOT_HOOK_COPY) count too: a copy leaves exactly the registers its original leaves. SA2_COPY_FOR_HOOK does not:
    those functions are also hooked by the renderer (src/me/sa2_me_hooks.cpp), which reads argument registers itself."""
    text = open(os.path.join(ROOT, 'app', 'src', 'cpu', 'sa2_hot_guest.cpp'), encoding='utf-8').read()
    return set(re.findall(r'SA2_HOT_HOOK(?:_UNGUARDED|_COPY)?\(([0-9A-F]{8}),', text))


def other_hooks():
    """Every 82xxxxxx address the app or SDK sources' code mentions, minus the natives: an app hook may read any
    register. Comments and __imp__sub_X references are left out (2026-10-07, as in tools/direct_calls.py): a comment
    hooks nothing, and __imp__sub_X calls the original, which takes no register of the caller's copy."""
    names = set()
    for root in (os.path.join(ROOT, 'app', 'src'), os.path.join(ROOT, 'sdk', 'src'), os.path.join(ROOT, 'sdk', 'include')):
        for d, _, fs in os.walk(root):
            for fn in fs:
                if not fn.endswith(('.cpp', '.h', '.inc', '.hpp')):
                    continue
                try:
                    text = open(os.path.join(d, fn), encoding='utf-8', errors='ignore').read()
                except OSError:
                    continue
                text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
                text = re.sub(r'//[^\n]*', ' ', text)
                text = re.sub(r'__imp__sub_[0-9A-Fa-f]{8}', ' ', text)
                names.update(m.upper() for m in re.findall(r'(?<![0-9A-Fa-f])(82[0-9A-Fa-f]{6})(?![0-9A-Fa-f])', text))
    return names - native_hooks()


class Reach:
    """Transitive register reach of generated functions: the union of the registers mentioned by a function and by
    everything it calls (strongly connected components over the call graph, computed on demand). ALL = unknown."""

    def __init__(self, funcs, unsafe):
        self.funcs = funcs
        self.unsafe = unsafe
        self.memo = {}

    def own(self, name):
        imp = name.startswith('__imp__')
        base = name[len('__imp__'):] if imp else name
        if base.startswith('sub_') and base[4:] in self.unsafe and not imp:
            return ALL, ()  # overridden by an app hook that may read anything
        f = self.funcs.get(base)
        if f is None or f[2]:
            return ALL, ()  # not a generated function (import, host function), or it makes indirect calls
        return set(f[0]), list(f[1])

    def of(self, root):
        if root in self.memo:
            return self.memo[root]
        # Iterative Tarjan: frames of (node, list of callees, next position).
        index, low, on_stack, stack, owns = {}, {}, set(), [], {}
        counter = 0
        frames = []

        def push(n):
            nonlocal counter
            index[n] = low[n] = counter
            counter += 1
            stack.append(n)
            on_stack.add(n)
            regs, callees = self.own(n)
            owns[n] = (regs, callees)
            frames.append([n, callees, 0])

        push(root)
        while frames:
            frame = frames[-1]
            n, callees, pos = frame
            if pos < len(callees):
                frame[2] += 1
                c = callees[pos]
                if c in self.memo:
                    continue
                if c not in index:
                    push(c)
                elif c in on_stack:
                    low[n] = min(low[n], index[c])
                continue
            frames.pop()
            if frames:
                parent = frames[-1][0]
                low[parent] = min(low[parent], low[n])
            if low[n] != index[n]:
                continue
            members = []
            while True:
                w = stack.pop()
                on_stack.discard(w)
                members.append(w)
                if w == n:
                    break
            member_set = set(members)
            reach = set()
            for w in members:
                regs, callees = owns[w]
                if regs is ALL:
                    reach = ALL
                    break
                reach |= regs
                for c in callees:
                    if c in member_set:
                        continue
                    rc = self.memo.get(c, ALL)  # finished SCCs are memoized; anything else is unknown
                    if rc is ALL:
                        reach = ALL
                        break
                    reach |= rc
                if reach is ALL:
                    break
            for w in members:
                self.memo[w] = reach
        return self.memo[root]


def find_body(gen, name):
    for path in sorted(glob.glob(os.path.join(gen, '*_recomp.*.cpp'))):
        text = open(path, encoding='utf-8', errors='replace').read()
        m = re.search(r'^DEFINE_REX_FUNC\(' + name + r'\) \{\n(.*?)^\}\n', text, re.S | re.M)
        if m:
            return m.group(1), path
    return None, None


def type_of(reg):
    if reg.startswith('cr'):
        return 'PPCCRRegister'
    if reg == 'xer':
        return 'PPCXERRegister'
    if reg.startswith('v'):
        return 'PPCVRegister'
    return 'PPCRegister'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('address', nargs='?')
    ap.add_argument('--what', default='')
    ap.add_argument('--gen', default=os.path.join(ROOT, 'app', 'generated', 'default'))
    ap.add_argument('--out')
    ap.add_argument('--allow-double', action='store_true')
    ap.add_argument('--sync', choices=('smart', 'full'), default='smart',
                    help='smart: each call syncs only what its callee can reach (default); full: every register')
    ap.add_argument('--list', help='file of "<address> <what>" lines (app/src/cpu/hot/localized.txt): localize them all '
                                   'into app/generated/hot/')
    a = ap.parse_args()
    if a.list:
        for line in open(a.list, encoding='utf-8'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            addr, _, what = line.partition(' ')
            localize(addr, what.strip(), a.gen, os.path.join(ROOT, 'app', 'generated', 'hot', f'n_{addr.upper()}.h'),
                     a.allow_double, a.sync)
        return
    if not a.address:
        ap.error('an address or --list')
    localize(a.address, a.what, a.gen, a.out, a.allow_double, a.sync)


_REACH = {}


def reach_for(gen):
    if gen not in _REACH:
        _REACH[gen] = Reach(build_index(gen), other_hooks())
    return _REACH[gen]


def localize(address, what, gen, out, allow_double, sync='smart'):
    addr = address.upper().replace('SUB_', '').replace('0X', '')
    name = 'sub_' + addr
    body, path = find_body(gen, name)
    if body is None:
        sys.exit(f'{name} not found in {gen}')

    lines = body.splitlines()
    # Bare double-precision arithmetic (no float() rounding around it) could be contracted differently.
    risky = [l for l in lines if re.search(r'\.f64 = (?:ctx\.)?f\d+\.f64 [-+*] (?:ctx\.)?f\d+\.f64;', l)]
    if risky and not allow_double:
        sys.exit(f'{name}: double-precision arithmetic, check contraction by hand (--allow-double):\n' + '\n'.join(risky[:8]))

    regs = sorted({m.group(1) for l in lines for m in REG.finditer(l)},
                  key=lambda r: (re.sub(r'\d', '', r), int(re.sub(r'\D', '', r) or 0)))
    out_regs = '; '.join(f'ctx.{r} = l_{r}' for r in regs)
    in_regs = '; '.join(f'l_{r} = ctx.{r}' for r in regs)
    code = []
    calls = 0
    synced = []  # registers synced per call (statistics)
    reach = reach_for(gen) if sync == 'smart' else None

    def call_sync(m):
        callee = m.group(1)
        regset = ALL if (reach is None or callee is None) else reach.of(callee)
        if regset is ALL:
            synced.append(len(regs))
            return '{ SA2_SYNC_OUT(); ' + m.group(0) + ' SA2_SYNC_IN(); }'
        part = [r for r in regs if r in regset]
        synced.append(len(part))
        before = ''.join(f'ctx.{r} = l_{r}; ' for r in part)
        after = ''.join(f' l_{r} = ctx.{r};' for r in part)
        return '{ ' + before + m.group(0) + after + ' }'

    for l in lines:
        if l.strip().startswith('//'):
            continue  # the PPC disassembly comments
        l = REG.sub(lambda m: 'l_' + m.group(1), l)
        # Calls and returns get braces: the generated code also writes them as the body of a one-line `if`
        # ("if (ctx.cr6.eq) return;"), where a bare "SA2_SYNC_OUT(); return;" would return unconditionally.
        l, n = CALL.subn(call_sync, l)
        calls += n
        l = re.sub(r'\breturn;', '{ SA2_SYNC_OUT(); return; }', l)
        code.append(l)
    # Falling off the end of the function is a return too.
    code.append('\tSA2_SYNC_OUT();')

    decl = '\n'.join(f'  {type_of(r)} l_{r} = ctx.{r};' for r in regs)
    # Every direct callee needs a declaration: recompiled functions, the save/restore helpers and the kernel imports
    # (__imp__KeBugCheck, __imp__RtlEnterCriticalSection, ...: the generated files declare those in their own
    # sa2_funcs.N.h, which this header does not include). Indirect calls and traps have no name (no group 1).
    callees = sorted({m.group(1) for l in lines if not l.strip().startswith('//')
                      for m in CALL.finditer(l) if m.group(1)})
    externs = ''.join(f'REX_EXTERN({c});\n' for c in callees)
    rel = os.path.relpath(path, ROOT).replace('\\', '/')
    out = out or os.path.join(ROOT, 'app', 'generated', 'hot', f'n_{addr}.h')
    what = what or name
    sync_text = ('only those its callee can reach (smart sync; registers per call: ' +
                 ', '.join(str(x) for x in synced) + ')') if sync == 'smart' and calls else 'all of them'
    text = f'''// {name}: {what}.
// GENERATED by tools/localize_native.py from {rel} (do not edit; re-run the tool after a codegen change): the
// recompiled function's own operations, with its {len(regs)} guest registers in C++ locals and plain guest memory
// accesses: written back at every return and, around each of its {calls} call(s), {sync_text}. Exact by construction;
// the guard of sa2_hot_guest.cpp (where it applies) compares the registers and the written memory with the original.
#pragma once

{externs}
namespace sa2::hot::n_{addr} {{

inline void Native(PPCContext& ctx, uint8_t* base) {{
{decl}
#define SA2_SYNC_OUT() do {{ {out_regs}; }} while (0)
#define SA2_SYNC_IN() do {{ {in_regs}; }} while (0)
''' + '\n'.join(code) + '''
#undef SA2_SYNC_OUT
#undef SA2_SYNC_IN
}

}  // namespace sa2::hot::n_''' + addr + '\n'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, 'w', encoding='utf-8', newline='\n').write(text)
    try:
        shown = os.path.relpath(out, ROOT)
    except ValueError:  # another drive
        shown = out
    detail = f' (synced per call: {synced})' if calls and sync == 'smart' else ''
    print(f'{name}: {len(regs)} registers localized, {calls} calls{detail} -> {shown}')


if __name__ == '__main__':
    main()
