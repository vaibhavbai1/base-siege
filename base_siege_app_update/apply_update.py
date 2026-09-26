"""Install/restore the browser-local smooth manual-play interface update."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

BUNDLE = Path(__file__).resolve().parent
ROOT = Path.cwd().resolve()
BACKUP = ROOT / 'artifacts/app-backups/manual-smooth-ui-v3'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def no_symlinks(path):
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise SystemExit(f'Symbolic link at update or backup path: {path}')
    if path.exists() and not path.is_file():
        raise SystemExit(f'Expected a file at: {path}')


def protected_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for folder in ('env', 'agents', 'train', 'experiments')
            for p in (ROOT / folder).rglob('*.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore', action='store_true')
    args = parser.parse_args()
    if not (ROOT / 'pyproject.toml').is_file() or not (ROOT / 'env/siege_env.py').is_file():
        raise SystemExit('Run this command from your Base Siege project root.')
    manifest = json.loads((BUNDLE / 'update.json').read_text())
    protected = protected_hashes()
    actual = {}
    # Complete the preflight before changing any application file.
    for name, item in manifest.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts
                or relative.parts[0] not in ('ui', 'tools', 'tests')):
            raise SystemExit(f'Unexpected update path: {name}')
        for path in (ROOT / relative, BACKUP / relative, BUNDLE / 'payload' / relative):
            no_symlinks(path)
        if digest(BUNDLE / 'payload' / relative) != item['new_sha256']:
            raise SystemExit(f'Corrupted update payload: {name}')
        actual[name] = digest(ROOT / relative)
        if actual[name] not in [*item['accepted_previous_sha256'], item['new_sha256']]:
            raise SystemExit(f'{name} has other local edits. No application files changed. Share this message for review.')
    record_path = BACKUP / 'update-record.json'
    no_symlinks(record_path)
    record = json.loads(record_path.read_text()) if record_path.exists() else None
    if record is not None and record['new_hashes'] != {k: v['new_sha256'] for k,v in manifest.items()}:
        raise SystemExit('The existing backup belongs to a different update. No application files changed.')
    if args.restore:
        if record is None:
            raise SystemExit('No installed v3 interface backup is available.')
        for name, old_hash in record['previous_hashes'].items():
            if actual[name] not in (old_hash, manifest[name]['new_sha256']):
                raise SystemExit(f'{name} changed after installation. No application files changed.')
            if old_hash is not None and digest(BACKUP / name) != old_hash:
                raise SystemExit(f'Backup is missing or changed: {name}')
        for name, old_hash in record['previous_hashes'].items():
            if old_hash is None:
                (ROOT / name).unlink(missing_ok=True)
            else:
                atomic(ROOT / name, (BACKUP / name).read_bytes())
        action = 'Restored the interface that was present before this update'
    else:
        if all(actual[k] == v['new_sha256'] for k,v in manifest.items()):
            print('This interface update is already installed. No files changed.')
            return
        if record is None:
            record = {'installed_utc': datetime.now(timezone.utc).isoformat(),
                      'previous_hashes': actual,
                      'new_hashes': {k: v['new_sha256'] for k,v in manifest.items()},
                      'registered_sources_before': protected}
            for name, old_hash in actual.items():
                if old_hash is not None:
                    existing = digest(BACKUP / name)
                    if existing not in (None, old_hash):
                        raise SystemExit(f'Existing backup differs: {name}')
            for name, old_hash in actual.items():
                if old_hash is not None:
                    atomic(BACKUP / name, (ROOT / name).read_bytes())
            atomic(record_path, json.dumps(record, indent=2).encode())
        else:
            for name, old_hash in record['previous_hashes'].items():
                if actual[name] not in (old_hash, manifest[name]['new_sha256']):
                    raise SystemExit(f'{name} differs from this installation backup. No application files changed.')
                if old_hash is not None and digest(BACKUP / name) != old_hash:
                    raise SystemExit(f'Backup is missing or changed: {name}')
        for name in manifest:
            atomic(ROOT / name, (BUNDLE / 'payload' / name).read_bytes())
        action = 'Installed smooth browser-local manual play, all four layouts, and the custom-base editor'
    if protected_hashes() != protected:
        raise RuntimeError('Registered sources changed unexpectedly. Stop and investigate.')
    print(action + '.')
    print('Simulator/training source hashes are unchanged; checkpoints and evaluation files were not modified.')
    print('Start: uv run streamlit run ui/app.py --server.address 0.0.0.0 --server.port 8501')


if __name__ == '__main__':
    main()
