#!/usr/bin/env python3
"""Imports Renut's native renderer (reNut's src/renut_engine/gpu) into banjo-nx as a second renderer.

banjo-nx started from Renut (github.com/masterspike52/reNut). Its upstream now renders N&B at the Direct3D level through
plume; banjo-nx keeps its own native renderer and chooses between the two at boot (`banjo_renderer`, see
app/src/renut_engine/gpu/banjo_renut_select.h). This tool copies the renderer from a Renut checkout and rewrites what
differs in banjo-nx:

  - Renut names its Direct3D hooks (`REX_HOOK(D3DDevice_Clear, ...)`, names from its config/renut_gpu_funcs.toml);
    banjo-nx's generated code names those functions sub_<address>. Every hook is rewritten to its address, and to a
    form that runs Renut's function only when Renut's renderer is chosen and the game's original otherwise:
    `BANJO_RENUT_HOOK(sub_X, fn)` / `BANJO_RENUT_HOOK_RAW(sub_X)`.
  - Addresses banjo-nx hooks itself (its own renderer's hooks, and the generated record-writer hooks) cannot be
    defined twice: there Renut's function becomes `renut_native_sub_X`, which banjo-nx's hook calls when Renut's
    renderer is chosen (`BANJO_RENUT_HOOK_SHARED`). The list goes to banjo_shared_hooks.h and to
    app/src/me/banjo_renut_shared.txt, which tools/d3d/d3d_record_hooks.py reads.
  - Renut's imports of game functions it names (XMemFree) are rewritten to banjo-nx's names.
  - perf/frame_stats.h is copied; banjo-nx has its own frame_stats.cpp (Renut's hooks the main loop, which banjo-nx's
    FPS code already hooks).

banjo-nx's own changes to Renut's files (the ARM work, the uniform-buffer constants, fixes) are kept as one patch,
tools/renut_patches/banjo.patch, which the import applies after the rewrite. After editing the imported files, save
them into the patch with --save-patch (it diffs the tree against a fresh rewrite of the same Renut checkout).

Usage: python tools/renut_gpu_import.py --renut "D:/Banjo kazooie port/reNut" [--dry-run | --save-patch | --no-patch]
Running it again over an imported tree gives the same result (the copy is replaced each time).
"""
import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_SRC = os.path.join(ROOT, 'app', 'src')
DEST = os.path.join(APP_SRC, 'renut_engine', 'gpu')
SHARED_LIST = os.path.join(APP_SRC, 'me', 'banjo_renut_shared.txt')
KEEP = {'banjo_renut_select.h', 'banjo_renut_select.cpp', 'banjo_shared_hooks.h', 'banjo_renut_capture.h',
        'banjo_renut_tuning.h', 'banjo_renut_tuning.cpp',
        'banjo_renut_checks.h', 'banjo_renut_checks.cpp',
        'banjo_renut_record.h', 'banjo_renut_record.cpp'}  # banjo-nx's own files in DEST
PATCH = os.path.join(ROOT, 'tools', 'renut_patches', 'banjo.patch')

HOOK_RE = re.compile(r'\bREX_HOOK\(\s*([A-Za-z0-9_]+)\s*,\s*([^;]+?)\s*\);')
RAW_RE = re.compile(r'\bREX_HOOK_RAW\(\s*(sub_[0-9A-Fa-f]{8})\s*\)')
IMPORT_RE = re.compile(r'\bREX_IMPORT\(\s*__imp__([A-Za-z0-9_]+)\s*,')
EXTERN_IMP_RE = re.compile(r'\bREX_EXTERN\(\s*__imp__(sub_[0-9A-Fa-f]{8})\s*\)')


def load_names(renut):
    names = {}
    for line in open(os.path.join(renut, 'config', 'renut_gpu_funcs.toml'), encoding='utf-8'):
        m = re.match(r'\s*(0x[0-9A-Fa-f]+)\s*=\s*\{\s*name\s*=\s*"([^"]+)"', line)
        if m:
            names[m.group(2)] = int(m.group(1), 16)
    return names


def banjo_generated_names():
    """Names banjo-nx's generated code gives functions (DEFINE_REX_FUNC(name)), other than sub_X."""
    found = set()
    gen = os.path.join(ROOT, 'app', 'generated')
    for f in os.listdir(gen):
        if f.endswith('.cpp'):
            for m in re.finditer(r'^DEFINE_REX_FUNC\(([A-Za-z_][A-Za-z0-9_]*)\)', open(os.path.join(gen, f),
                                 encoding='utf-8', errors='ignore').read(), re.M):
                if not m.group(1).startswith('sub_'):
                    found.add(m.group(1))
    return found


def banjo_hooked():
    """Addresses banjo-nx defines a hook for, outside the imported renderer, in the files it compiles."""
    hooked = set()
    cmake = open(os.path.join(ROOT, 'app', 'CMakeLists.txt'), encoding='utf-8').read()
    native = os.path.abspath(os.path.join(APP_SRC, 'native'))
    for base in (APP_SRC, os.path.join(ROOT, 'sdk', 'src')):
        for root, dirs, files in os.walk(base):
            if os.path.abspath(root).startswith(os.path.abspath(DEST)):
                continue
            for f in files:
                if not f.endswith(('.cpp', '.h', '.inc')):
                    continue
                path = os.path.join(root, f)
                # app/src/native/ is the old nfsmw renderer: only the files the app's CMakeLists names are built.
                if os.path.abspath(path).startswith(native) and f.endswith('.cpp'):
                    rel = os.path.relpath(path, os.path.join(ROOT, 'app')).replace(os.sep, '/')
                    if rel not in cmake:
                        continue
                text = open(path, encoding='utf-8', errors='ignore').read()
                for m in re.finditer(r'\b(?:REX_HOOK_RAW|REX_HOOK|REX_STUB)\(\s*sub_([0-9A-Fa-f]{8})', text):
                    hooked.add(int(m.group(1), 16))
                for m in re.finditer(r'\bBANJO_D3D_RECORD_HOOK(?:_SHARED)?\(([0-9A-Fa-f]{8})\)', text):
                    hooked.add(int(m.group(1), 16))
    return hooked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--renut', required=True)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--save-patch', action='store_true', help='write the changes made to the imported files to ' + PATCH)
    ap.add_argument('--no-patch', action='store_true', help='import without applying ' + PATCH)
    ap.add_argument('--check-into', metavar='DIR',
                    help='write a fresh import with the patch applied into DIR (outside the repository) and compare it '
                         'with the imported tree: the round trip a Renut update relies on')
    args = ap.parse_args()
    src = os.path.join(args.renut, 'src', 'renut_engine', 'gpu')
    if not os.path.isdir(src):
        sys.exit(f'renut_gpu_import: no {src}')
    names = load_names(args.renut)
    ours = banjo_generated_names()
    hooked = banjo_hooked()

    texts = {}
    for root, _, files in os.walk(src):
        for f in files:
            path = os.path.join(root, f)
            rel = os.path.relpath(path, src)
            texts[rel] = open(path, 'rb').read()

    def address(name, where):
        if name.startswith('sub_'):
            return int(name[4:], 16)
        if name in names:
            return names[name]
        sys.exit(f'renut_gpu_import: {where}: hook name {name} is not in renut_gpu_funcs.toml')

    shared = set()
    hooks = 0
    for rel, data in texts.items():
        if not rel.endswith(('.cpp', '.h')):
            continue
        text = data.decode('utf-8')
        for m in HOOK_RE.finditer(text):
            a = address(m.group(1), rel)
            hooks += 1
            if a in hooked:
                shared.add(a)
        for m in RAW_RE.finditer(text):
            hooks += 1
            if int(m.group(1)[4:], 16) in hooked:
                shared.add(int(m.group(1)[4:], 16))

    def rewrite(text, rel):
        def hook(m):
            a = address(m.group(1), rel)
            macro = 'BANJO_RENUT_HOOK_SHARED' if a in shared else 'BANJO_RENUT_HOOK'
            return f'{macro}(sub_{a:08X}, {m.group(2)});  // {m.group(1)}'
        text = HOOK_RE.sub(hook, text)
        text = RAW_RE.sub(lambda m: ('BANJO_RENUT_HOOK_RAW_SHARED' if int(m.group(1)[4:], 16) in shared
                                     else 'BANJO_RENUT_HOOK_RAW') + f'({m.group(1).upper().replace("SUB_", "sub_")})',
                          text)

        def imp(m):
            n = m.group(1)
            if n in ours or n.startswith('sub_'):
                return m.group(0)
            if n in names:
                return f'REX_IMPORT(__imp__sub_{names[n]:08X},'
            return m.group(0)  # a kernel import: the same name in banjo-nx
        text = IMPORT_RE.sub(imp, text)
        if ('BANJO_RENUT_HOOK' in text) and 'banjo_renut_select.h' not in text:
            text = text.replace('#include <rex/hook.h>', '#include <rex/hook.h>\n\n#include "renut_engine/gpu/banjo_renut_select.h"', 1)
            if 'banjo_renut_select.h' not in text:
                first = text.find('#include')
                text = text[:first] + '#include "renut_engine/gpu/banjo_renut_select.h"\n' + text[first:]
        return text

    out = {}
    for rel, data in texts.items():
        if rel.endswith(('.cpp', '.h')):
            out[rel] = rewrite(data.decode('utf-8'), rel).encode('utf-8')
        else:
            out[rel] = data
    # gpu.cmake builds Renut's XenosRecomp as an ExternalProject with only the compiler paths: banjo-nx's PC toolchain
    # (clang for the MSVC ABI) also needs the target, the linker and the flags (BANJO_RENUT_XENOS_RECOMP_ARGS, set in
    # app/cmake/banjo_renut_gpu.cmake), or clang builds it for MinGW and the DXC link fails.
    cm = out['gpu.cmake'].decode('utf-8')
    m = re.search(r'( *)-DXENOS_RECOMP_DXIL=\$\{recomp_dxil\}(\r?\n)', cm)
    if not m:
        sys.exit('renut_gpu_import: gpu.cmake: the ExternalProject arguments moved; update the BANJO_RENUT_XENOS_RECOMP_ARGS rewrite')
    cm = cm[:m.end()] + m.group(1) + '${BANJO_RENUT_XENOS_RECOMP_ARGS}' + m.group(2) + cm[m.end():]
    # The Switch is a cross build on a Windows host: the DXC that compiles the host shaders runs on the host
    # (CMAKE_HOST_WIN32, not WIN32), and the Switch has no X11 (REXGLUE_PLATFORM_SWITCH).
    for old, new in (('if(WIN32)\r\n    set(RENUT_DXC ', 'if(CMAKE_HOST_WIN32)\r\n    set(RENUT_DXC '),
                     ('if(WIN32)\n    set(RENUT_DXC ', 'if(CMAKE_HOST_WIN32)\n    set(RENUT_DXC '),
                     ('elseif(NOT APPLE)', 'elseif(NOT APPLE AND NOT REXGLUE_PLATFORM_SWITCH)')):
        cm = cm.replace(old, new)
    if 'if(CMAKE_HOST_WIN32)' not in cm or 'NOT REXGLUE_PLATFORM_SWITCH' not in cm:
        sys.exit('renut_gpu_import: gpu.cmake: the DXC or X11 branch moved; update the Switch rewrites')
    # banjo-nx's cmake/banjo_renut_gpu.cmake may give the shader cache from elsewhere (the Switch build takes the one a
    # host build generated): a hook before Renut's ExternalProject of its XenosRecomp.
    hook = 'function(renut_xenos_shader_cache TARGET)'
    if hook not in cm:
        sys.exit('renut_gpu_import: gpu.cmake: renut_xenos_shader_cache moved')
    cm = cm.replace(hook, hook + '\n    if(COMMAND banjo_renut_shader_cache_override)\n'
                    '        banjo_renut_shader_cache_override(${TARGET})\n        return()\n    endif()', 1)
    out['gpu.cmake'] = cm.encode('utf-8')

    shared_list = sorted(shared)
    header = ['// Generated by tools/renut_gpu_import.py: the addresses both renderers hook. banjo-nx\'s hook there calls',
              '// renut_native_sub_X when Renut\'s renderer is chosen (banjo_renut_select.h).',
              '#pragma once', '', '#include <rex/ppc.h>', '']
    header += [f'extern "C" void renut_native_sub_{a:08X}(PPCContext& ctx, uint8_t* base);' for a in shared_list]
    print(f'renut_gpu_import: {len(out)} files, {hooks} hooks, {len(shared_list)} shared with banjo-nx\'s own:',
          ' '.join(f'{a:08X}' for a in shared_list))
    if args.dry_run:
        return
    if args.save_patch:
        save_patch(out)
        return
    if args.check_into:
        check_round_trip(out, args.check_into)
        return
    if os.path.isdir(DEST):
        for root, dirs, files in os.walk(DEST, topdown=False):
            for f in files:
                if f not in KEEP:
                    os.remove(os.path.join(root, f))
            for d in dirs:
                p = os.path.join(root, d)
                if not os.listdir(p):
                    os.rmdir(p)
    for rel, data in out.items():
        path = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(data)
    open(os.path.join(DEST, 'banjo_shared_hooks.h'), 'w', encoding='utf-8').write('\n'.join(header) + '\n')
    open(SHARED_LIST, 'w', encoding='utf-8').write(''.join(f'{a:08X}\n' for a in shared_list))
    perf = os.path.join(APP_SRC, 'renut_engine', 'perf')
    os.makedirs(perf, exist_ok=True)
    shutil.copyfile(os.path.join(args.renut, 'src', 'renut_engine', 'perf', 'frame_stats.h'),
                    os.path.join(perf, 'frame_stats.h'))
    if os.path.isfile(PATCH) and not args.no_patch:
        # Paths in the patch are relative to DEST. Run from the repository's root with --directory: inside a work
        # tree, git apply skips (silently) the paths it does not find relative to the root. -v lists every file.
        rel = os.path.relpath(DEST, ROOT).replace(os.sep, '/')
        r = subprocess.run(['git', 'apply', '-v', '--whitespace=nowarn', '--reject', '-p1', f'--directory={rel}', PATCH],
                           cwd=ROOT)
        if r.returncode:
            sys.exit(f'renut_gpu_import: {PATCH} did not apply cleanly (see the .rej files in {DEST}); '
                     'merge the rejected hunks, then run --save-patch')
        print(f'renut_gpu_import: applied {os.path.relpath(PATCH, ROOT)}')


def check_round_trip(pristine, into):
    """A fresh rewrite of the checkout into 'into' with PATCH applied must equal the imported tree."""
    into = os.path.abspath(into)
    inside = os.path.splitdrive(into)[0].lower() == os.path.splitdrive(ROOT)[0].lower() and \
        os.path.commonpath([into, ROOT]) == ROOT
    if inside:
        sys.exit('renut_gpu_import: --check-into needs a folder outside the repository (git apply skips paths there)')
    if os.path.exists(into) and os.listdir(into):
        sys.exit(f'renut_gpu_import: {into} is not empty')
    for rel, data in pristine.items():
        path = os.path.join(into, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'wb').write(data)
    if os.path.isfile(PATCH):
        r = subprocess.run(['git', 'apply', '--whitespace=nowarn', '-p1', PATCH], cwd=into)
        if r.returncode:
            sys.exit('renut_gpu_import: the patch does not apply to a fresh import')
    differ = []
    for root, _, files in os.walk(DEST):
        for f in files:
            if f in KEEP or f.endswith(('.rej', '.orig')):
                continue
            rel = os.path.relpath(os.path.join(root, f), DEST)
            other = os.path.join(into, rel)
            if not os.path.isfile(other) or open(other, 'rb').read() != open(os.path.join(root, f), 'rb').read():
                differ.append(rel)
    print(f'renut_gpu_import: round trip {"OK" if not differ else "DIFFERS"}: '
          f'{len(differ)} files differ from the imported tree' + (': ' + ', '.join(differ[:10]) if differ else ''))
    if differ:
        sys.exit(1)


def save_patch(pristine):
    """banjo-nx's changes: the imported tree against the rewrite of the same Renut checkout ('pristine')."""
    current = {}
    for root, _, files in os.walk(DEST):
        for f in files:
            rel = os.path.relpath(os.path.join(root, f), DEST).replace(os.sep, '/')
            if f not in KEEP and not f.endswith(('.rej', '.orig')):
                current[rel] = open(os.path.join(root, f), 'rb').read()
    pristine = {rel.replace(os.sep, '/'): data for rel, data in pristine.items()}

    def lines(data):
        if data is None:
            return []
        out = re.findall(r'[^\n]*\n|[^\n]+$', data.decode('utf-8'))  # not splitlines: only \n ends a line in a patch
        if out and not out[-1].endswith('\n'):
            out[-1] += '\n\\ No newline at end of file\n'
        return out

    chunks = []
    for rel in sorted(set(current) | set(pristine)):
        a, b = pristine.get(rel), current.get(rel)
        if a == b:
            continue
        if b is None:
            sys.exit(f'renut_gpu_import: {rel} was removed from the imported tree; a patch does not remove files')
        head = f'diff --git a/{rel} b/{rel}\n'
        if a is None:
            head += 'new file mode 100644\n'
        diff = list(difflib.unified_diff(lines(a), lines(b), '/dev/null' if a is None else f'a/{rel}', f'b/{rel}', n=3))
        chunks.append(head + ''.join(diff))
    os.makedirs(os.path.dirname(PATCH), exist_ok=True)
    with open(PATCH, 'w', encoding='utf-8', newline='') as f:
        f.write(''.join(chunks))
    print(f'renut_gpu_import: {len(chunks)} changed files saved to {os.path.relpath(PATCH, ROOT)}')


if __name__ == '__main__':
    main()
