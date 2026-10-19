"""Fetches the third-party sources of the SDK into sdk/thirdparty.

sdk/thirdparty only carries the files this fork changed (see THIRD_PARTY_NOTICES.md if present). Everything else
comes from the ReXGlue SDK release the fork is based on (v0.10.0) and its submodules. This script gets that release
with its submodules and copies every file that sdk/thirdparty does not have yet; files already there are kept.

Usage:
    python tools/fetch_thirdparty.py                  clone the release from GitHub into a temporary folder (needs git)
    python tools/fetch_thirdparty.py --source DIR     use a local rexglue-sdk checkout (submodules initialised) and
                                                      do not touch the network
    python tools/fetch_thirdparty.py --target DIR     fill DIR instead of sdk/thirdparty
"""
import argparse
import os
import shutil
import stat
import subprocess
import sys
import tempfile

UPSTREAM = 'https://github.com/rexglue/rexglue-sdk.git'
COMMIT = 'c94f5ebdcb3c9d1a460ca48e04f9758448f8d518'  # v0.10.0


def run(*args, cwd=None):
    print('>', ' '.join(args))
    subprocess.run(args, cwd=cwd, check=True)


def remove_readonly(func, path, _):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def copy_missing(source, target):
    """Copies every file under source that target does not have. Returns (copied, kept)."""
    copied = kept = 0
    for dirpath, dirnames, filenames in os.walk(source):
        dirnames[:] = [d for d in dirnames if d != '.git']
        # Symbolic links to directories are listed with the directories; os.walk does not follow them.
        entries = list(filenames) + [d for d in dirnames if os.path.islink(os.path.join(dirpath, d))]
        for name in entries:
            if name == '.git':
                continue
            src = os.path.join(dirpath, name)
            dst = os.path.join(target, os.path.relpath(src, source))
            if os.path.lexists(dst):
                kept += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.islink(src):
                try:
                    os.symlink(os.readlink(src), dst)
                except (OSError, NotImplementedError):
                    continue  # no symlink support (Windows without privileges): only used by unused vendored demos
            else:
                shutil.copy2(src, dst)
            copied += 1
    return copied, kept


def resolve_symlink_stubs(target):
    """Git for Windows without symlink support checks a symbolic link out as a small text file holding the link's
    target path (libmspack's cabextract/mspack/*.c, for example), which then fails to compile. Replaces each such
    stub that points to an existing file with that file's content. Returns the number of files replaced."""
    fixed = 0
    for dirpath, dirnames, filenames in os.walk(target):
        dirnames[:] = [d for d in dirnames if d != '.git']
        for name in filenames:
            path = os.path.join(dirpath, name)
            if os.path.islink(path) or not 0 < os.path.getsize(path) <= 300:
                continue
            with open(path, 'rb') as f:
                data = f.read().strip()
            if b'\n' in data or b' ' in data or b'/' not in data:
                continue
            linked = os.path.normpath(os.path.join(dirpath, data.decode('ascii', 'replace')))
            if os.path.isfile(linked) and os.path.realpath(linked) != os.path.realpath(path):
                shutil.copyfile(linked, path)
                fixed += 1
    return fixed


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', help='local rexglue-sdk checkout (submodules initialised) to copy from')
    parser.add_argument('--target', help='folder to fill (default: sdk/thirdparty of this repository)')
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target = os.path.abspath(args.target) if args.target else os.path.join(root, 'sdk', 'thirdparty')
    os.makedirs(target, exist_ok=True)

    if args.source:
        source = os.path.join(os.path.abspath(args.source), 'thirdparty')
        if not os.path.isdir(source):
            sys.exit(f'{source} not found: --source must be a rexglue-sdk checkout')
        copied, kept = copy_missing(source, target)
    else:
        work = tempfile.mkdtemp(prefix='rexglue-sdk-')
        try:
            run('git', 'init', '-q', work)
            run('git', 'remote', 'add', 'origin', UPSTREAM, cwd=work)
            run('git', 'fetch', '-q', '--depth', '1', 'origin', COMMIT, cwd=work)
            run('git', 'checkout', '-q', 'FETCH_HEAD', cwd=work)
            run('git', 'submodule', 'update', '--init', '--recursive', cwd=work)
            copied, kept = copy_missing(os.path.join(work, 'thirdparty'), target)
        finally:
            if sys.version_info >= (3, 12):
                shutil.rmtree(work, onexc=remove_readonly)
            else:
                shutil.rmtree(work, onerror=remove_readonly)
    print(f'{copied} files copied into {target}, {kept} files of this fork kept')
    stubs = resolve_symlink_stubs(target)
    if stubs:
        print(f'{stubs} symbolic link stubs replaced by the files they point to')


if __name__ == '__main__':
    main()
