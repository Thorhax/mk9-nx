#!/usr/bin/env python3
# Direct calls to game functions that have no hook (post-codegen step, run after every codegen).
#
# (From masseffect-nx; sa2-nx: file names and hook sources adapted, run on generated/default and generated/data_dll.)
# With GCC every recompiled function sub_X is a weak alias of __imp__sub_X (DEFINE_REX_FUNC, <project>_pch.h),
# so that one of our REX_HOOK_RAW(sub_X) replaces it at link time. The codegen always calls sub_X, and the
# compiler cannot inline a weak function into another (a hook could replace it): not within the same file
# and not with LTO. This step changes `sub_X(ctx, base);` into `__imp__sub_X(ctx, base);` when sub_X has no
# hook, which is exactly the same function, but can now be inlined.
#
# Any 82xxxxxx address that appears in the code of the app or SDK sources (comments and __imp__sub_X references
# left out, see CodeOnly) or in the toml / txt files counts as hooked (the D3D trace builds its hooks with
# sub_##addr). That is slightly more than needed and safe.
#
# It does not touch sa2_init.cpp or sa2_register.cpp: the dispatch table for indirect calls still
# points to sub_X and sees the hooks. It has to be run again after every codegen (tools/codegen.sh does).
#
# Usage: python3 tools/direct_calls.py [--undo] [--dry-run] [--check] [--gen <generated folder>] [--hooked-list <file>]
#   --dry-run: count only, write nothing. --check: exit 1 if unhooked sub_X calls are still not direct, or if the
#   tree has no direct calls at all (a codegen that lost the step). Exit 1 also if no recomp files are found.
#   --undo: turn every __imp__sub_X call back into sub_X.
#   --redo: undo + convert in one pass, writing only files whose text really changes (post_codegen.py uses it).
#   --hooked-list: a file with the sub_X names to treat as hooked, instead of scanning the sources.
#   Environment: REXSDK_DIR = SDK folder to scan for hooks (default <repo>/sdk).
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
APP = os.path.join(ROOT, 'app')
GEN = os.path.join(APP, 'generated', 'default')
SDK = os.environ.get('REXSDK_DIR') or os.path.join(ROOT, 'sdk')
SOURCES = [os.path.join(APP, 'src'), os.path.join(SDK, 'src'), os.path.join(SDK, 'include'),
           os.path.join(APP, 'overrides.toml'), os.path.join(APP, 'perf_overrides.toml'),
           os.path.join(APP, 'mk9_manifest.toml'), os.path.join(APP, 'CMakeLists.txt')]
# Diagnostic builds (e.g. the Direct3D census) list hooks that live outside the scanned sources here, one sub_X per
# line; an empty or missing file adds nothing. Every run reads it, so a codegen re-run keeps them.
EXTRA_LIST = os.path.join(ROOT, 'out', 'extra_hooks.txt')


def hooked():
    names = set()
    if os.path.isfile(EXTRA_LIST):
        names |= set(open(EXTRA_LIST, encoding='utf-8').read().split())
    for root in SOURCES:
        if os.path.isfile(root):
            paths = [root]
        else:
            paths = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs
                     if not f.endswith(('.a', '.obj', '.o', '.png', '.jpg', '.bin', '_all.inc', '_all.inc'))]
        for path in paths:
            try:
                text = open(path, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            if path.endswith(CODE_SUFFIXES):
                text = CodeOnly(text)
            for m in re.findall(r'(?<![0-9A-Fa-f])(82[0-9A-Fa-f]{6})(?![0-9A-Fa-f])', text):
                names.add('sub_' + m.upper())
    return names


# In C/C++ files only the code counts (2026-10-07, after futari-nx found the same in its tool). Comments name
# addresses for the reader, and a native layer calls originals through __imp__sub_X: neither hooks sub_X. Before,
# both did: 1,021 of 1,136 "hooked" names came from them, and 651 calls stayed weak (not inlinable). The .toml and
# .txt files are still read whole, as before.
CODE_SUFFIXES = ('.cpp', '.cc', '.c', '.h', '.hpp', '.inc')


def CodeOnly(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    text = re.sub(r'//[^\n]*', ' ', text)
    return re.sub(r'__imp__sub_[0-9A-Fa-f]{8}', ' ', text)


def main():
    undo = '--undo' in sys.argv
    # --redo: --undo and then the conversion, in memory, writing only the files whose final text differs. A new hook
    # then rewrites only the files that call it (the build recompiles those, not all ~138: mtime-based).
    redo = '--redo' in sys.argv
    dry = '--dry-run' in sys.argv or '--check' in sys.argv
    check = '--check' in sys.argv
    gen = sys.argv[sys.argv.index('--gen') + 1] if '--gen' in sys.argv else GEN
    if '--hooked-list' in sys.argv:
        with_hook = set(open(sys.argv[sys.argv.index('--hooked-list') + 1], encoding='utf-8').read().split())
    else:
        with_hook = hooked()
    call = re.compile(r'(?<![\w])sub_([0-9A-F]{8})\(ctx, base\);')
    direct = re.compile(r'__imp__sub_([0-9A-F]{8})\(ctx, base\);')
    changed = 0
    files = 0
    seen = 0
    already_direct = 0
    for f in sorted(os.listdir(gen)):
        if not re.match(r'\w+_recomp\.\d+\.cpp$', f):  # sa2-nx: <project>_recomp.N.cpp
            continue
        path = os.path.join(gen, f)
        text = open(path, encoding='utf-8').read()
        seen += 1
        already_direct += len(direct.findall(text))
        if undo:
            new_value, n = direct.subn(lambda m: 'sub_%s(ctx, base);' % m.group(1), text)
        else:
            def change(m):
                return m.group(0) if 'sub_' + m.group(1) in with_hook else '__imp__sub_%s(ctx, base);' % m.group(1)
            source = direct.sub(lambda m: 'sub_%s(ctx, base);' % m.group(1), text) if redo else text
            new_value = call.sub(change, source)
            n = len(call.findall(source)) - len(call.findall(new_value))
        if new_value != text:
            files += 1
            if not dry:
                open(path, 'w', encoding='utf-8', newline='\n').write(new_value)
        changed += n
    if seen == 0:
        sys.exit('ERROR (post-codegen patch): direct_calls: no *_recomp.N.cpp in %s' % gen)
    if not undo and changed == 0 and already_direct == 0:
        sys.exit('ERROR (post-codegen patch): direct_calls: neither sub_X calls to convert nor __imp__ direct '
                 'calls in %s; the generated call shape changed' % gen)
    if check and not undo:
        print('direct_calls --check: %d direct calls present, %d still convertible' % (already_direct, changed))
        if changed or already_direct == 0:
            sys.exit(1)
        return
    print('%s: %d calls in %d files; %d hooked addresses respected' %
          ('undone' if undo else 'direct' + (' (dry-run)' if dry else ''), changed, files, len(with_hook)))


if __name__ == '__main__':
    main()
