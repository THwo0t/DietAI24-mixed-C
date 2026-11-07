"""Install the hash-pinned, original mixed-C runtime. No model clients."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / 'runtime/release.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def specification():
    return json.loads(SPEC.read_text())


def members(archive, expected):
    names = archive.namelist()
    if len(names) != len(set(names)) or set(names) != set(expected):
        raise ValueError('Runtime archive membership differs from the published manifest')
    if sum(i.file_size for i in archive.infolist()) > 600 * 1024 * 1024:
        raise ValueError('Runtime archive is too large')
    for info in archive.infolist():
        p = PurePosixPath(info.filename)
        if (p.is_absolute() or '..' in p.parts or '\\' in info.filename
                or info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError('Unsafe runtime archive path')
        dest = ROOT / p
        if dest.resolve() != dest or ROOT not in dest.parents:
            raise ValueError('Runtime destination contains a symlink')
    return names


def verify():
    spec = specification()
    for name, expected in spec['files'].items():
        path = ROOT / name
        if not path.is_file() or path.resolve() != path or sha(path) != expected:
            raise ValueError('Runtime missing or changed: ' + name)
    return {'status': 'passed', 'files': len(spec['files']), 'new_api_calls': 0,
            'index': 'original Chroma graph, no rebuilding, original search parameters'}


def install(archive=None):
    spec = specification()
    complete = True
    for name, expected in spec['files'].items():
        target = ROOT / name
        if target.resolve() != target:
            raise ValueError('Runtime destination contains a symlink')
        if target.exists() and (not target.is_file() or sha(target) != expected):
            raise ValueError('Existing runtime file differs; preserve it before installing: ' + name)
        complete = complete and target.is_file()
    if complete:
        return verify()
    with tempfile.TemporaryDirectory(prefix='mixed-c-runtime-') as tmp:
        if archive is None:
            archive = Path(tmp) / 'runtime.zip'
            request = urllib.request.Request(spec['url'], headers={'User-Agent': 'DietAI24-mixed-C'})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open('wb') as output:
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > spec['size_bytes']:
                        raise ValueError('Download exceeds pinned size')
                    output.write(block)
        archive = Path(archive).resolve()
        if archive.stat().st_size != spec['size_bytes'] or sha(archive) != spec['sha256']:
            raise ValueError('Runtime archive size or SHA256 mismatch')
        with zipfile.ZipFile(archive) as z:
            names = members(z, spec['files'])
            stage = Path(tmp) / 'verified'
            for name in names:
                target = ROOT / name
                if target.exists() and sha(target) != spec['files'][name]:
                    raise ValueError('Existing runtime file differs; preserve it before installing: ' + name)
                copy = stage / name
                copy.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, copy.open('wb') as dest:
                    shutil.copyfileobj(source, dest)
                if sha(copy) != spec['files'][name]:
                    raise ValueError('Runtime member hash mismatch: ' + name)
            for name in names:
                target = ROOT / name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    # Publish atomically on the target filesystem. Interrupted copies
                    # never become the final runtime member. The temporary prefix is
                    # outside the indexed directory and can be safely kept for diagnosis.
                    pending = ROOT / '.runtime-downloads'
                    if pending.resolve() != pending:
                        raise ValueError('Runtime staging directory contains a symlink')
                    pending.mkdir(exist_ok=True)
                    with tempfile.NamedTemporaryFile(dir=pending, delete=False) as f:
                        partial = Path(f.name)
                        with (stage / name).open('rb') as source:
                            shutil.copyfileobj(source, f)
                        f.flush()
                        os.fsync(f.fileno())
                    os.link(partial, target)  # Atomic, and refuses to overwrite a newly created target.
                    partial.unlink()
    return verify()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['verify', 'install'])
    parser.add_argument('--archive', type=Path, help='Use the identical archive obtained separately')
    args = parser.parse_args()
    print(json.dumps(install(args.archive) if args.action == 'install' else verify(), ensure_ascii=False))


if __name__ == '__main__':
    main()
