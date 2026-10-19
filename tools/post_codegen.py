#!/usr/bin/env python3
"""The post-codegen steps of the codegen flavour, in one place (docs/enhancements.md, "Build-time").

Run after EVERY codegen, by tools/codegen.ps1 and by the build itself: app/CMakeLists.txt appends this script to the
generated sa2_codegen command, because the PC build re-runs the generator on its own whenever an input changes (the
manifest, a toml, the xex, the generator binary) and that would otherwise drop the steps silently.

  - the flavour is read from app/codegen_flavor.toml (written by tools/codegen.ps1 -Flavor): "optimised" when it sets
    any codegen key, else "classic";
  - optimised: tools/pch_no_volatile.py + tools/loop_barriers.py (required together) and tools/direct_calls.py on both
    generated trees;
  - both: the localized native copies (tools/localize_native.py --list app/src/cpu/hot/localized.txt) and
    app/generated/codegen_flavor.txt (the build's name in the first "[enh]" log line).

  python tools/post_codegen.py
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, 'app')
GEN = os.path.join(APP, 'generated')


def run(*args):
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(f'post_codegen: {" ".join(os.path.basename(a) for a in args[:1])} failed ({result.returncode})')


def main():
    flavor_toml = open(os.path.join(APP, 'codegen_flavor.toml'), encoding='utf-8').read()
    optimised = re.search(r'^\s*[a-z_]+\s*=\s*true', flavor_toml, re.M) is not None
    flavor = 'optimised' if optimised else 'classic'
    if optimised:
        # usf4-nx: one module (Default.xex), so one generated tree.
        run(os.path.join(ROOT, 'tools', 'pch_no_volatile.py'), os.path.join(GEN, 'default', 'mk9_pch.h'))
        # REQUIRED with the line above: without barriers GCC turns guest polling loops into infinite loops.
        run(os.path.join(ROOT, 'tools', 'loop_barriers.py'), os.path.join(GEN, 'default'))
        os.environ.setdefault('REXSDK_DIR', os.path.join(ROOT, 'sdk'))
        for tree in ('default',):
            # Undo and redo in one pass: a function hooked since the last run (a new native) must be called through
            # its hook again. --redo writes only the files that change, so the build does not recompile the others.
            run(os.path.join(ROOT, 'tools', 'direct_calls.py'), '--redo', '--gen', os.path.join(GEN, tree))
    localized = os.path.join(APP, 'src', 'cpu', 'hot', 'localized.txt')
    if os.path.isfile(localized):  # usf4-nx: no localized native copies (yet)
        run(os.path.join(ROOT, 'tools', 'localize_native.py'), '--list', localized)
    # Round 7: the options added since round 6 are named too, so a console log tells the builds apart.
    extras = [name for key, name in (('non_volatile_as_local', 'non-volatile locals'), ('elide_lr_stores', 'no lr stores'),
                                     ('fp_moves_modeless', 'modeless fp moves'),
                                     ('non_argument_as_local', 'non-argument locals'))
              if re.search(r'^\s*' + key + r'\s*=\s*true', flavor_toml, re.M)]
    # Written only when it changes: the file is a CMake configure dependency (app/CMakeLists.txt), and rewriting it
    # made every build re-run CMake, rebuild the generator and run the codegen again (2026-10-07).
    text = f'{flavor} codegen' + (f' ({", ".join(extras)})' if extras else '')
    path = os.path.join(GEN, 'codegen_flavor.txt')
    try:
        old_text = open(path, encoding='utf-8').read()
    except OSError:
        old_text = None
    if old_text != text:
        with open(path, 'w', encoding='utf-8', newline='\n') as f:
            f.write(text)
    print(f'post_codegen: {flavor} codegen steps done')


if __name__ == '__main__':
    main()
