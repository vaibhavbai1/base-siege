"""Retain three verified V2 recovery generations and milestone models."""
import shutil
from train.checkpoints import verify_checkpoint, atomic_bytes, atomic_json
from experiments.provenance import sha256


def retain_and_export(output, committed, contract):
    if contract.get('experiment') != 'version2-seed-replication' or contract['keep_checkpoints'] < 3:
        raise ValueError('Retention only operates on this V2 experiment with at least three generations')
    root = output/'checkpoints'
    if root.is_symlink() or committed.is_symlink() or committed.parent != root:
        raise ValueError('Unexpected checkpoint location')
    metadata = verify_checkpoint(committed)
    if metadata['contract_sha256'] != sha256(output/'contract.json'):
        raise ValueError('Wrong checkpoint contract')
    if metadata['actual_steps'] in contract['milestones']:
        dest = output/'milestones'/f"step-{metadata['actual_steps']:012d}.zip"
        atomic_bytes(dest, (committed/'model.zip').read_bytes())
        atomic_json(dest.with_suffix('.json'), {'actual_steps': metadata['actual_steps'],
            'phase': metadata['phase'], 'sha256': sha256(dest),
            'contract_sha256': metadata['contract_sha256'], 'purpose': 'inference and development comparisons'})
    valid = []
    for path in root.glob('checkpoint-*'):
        if path.is_symlink() or not path.is_dir():
            continue
        try:
            item = verify_checkpoint(path)
            if item['contract_sha256'] == metadata['contract_sha256']:
                valid.append((item['actual_steps'], (path/'checkpoint.json').stat().st_mtime_ns, path))
        except (ValueError, OSError, KeyError):
            continue
    valid.sort(reverse=True)
    keep = {item[2] for item in valid[:contract['keep_checkpoints']]}
    keep.add(committed)
    # Corrupt/unrecognized generations remain available for inspection.
    if len(keep) >= contract['keep_checkpoints']:
        for _, _, path in valid:
            if path not in keep:
                shutil.rmtree(path)
