"""Application-only catalogue of the twenty approved V2 final checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=Path(__file__).with_name('v2_checkpoints.json')


def read_registry():
    return json.loads(REGISTRY.read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def label_v2(manifest, target, registry):
    result=dict(manifest)
    result.update(application_version='V2',recommended_action_mode='DS',actual_steps=target['transitions'],
                  config=registry['config'],source_hashes=registry['source_hashes'],
                  evidence_scope='Version 2 final checkpoint; individual replays are illustrations')
    result['settings']={**manifest['settings'],'condition':target['condition'],'seed':target['learner_seed']}
    return result


def application_catalog(root, paths, *, public=False):
    """Public mode reads published assets only; local mode adds pinned V2 milestones."""
    from ui.policy_playback import model_catalog
    root=Path(root);registry=read_registry();catalog=[];issues=[]
    known={m['sha256']:m for m in registry['models']}
    for path,manifest in model_catalog(paths):
        # V2 local runs are added once, below, using their approved endpoint path.
        if not public and any(path.parent==root/Path(m['manifest_path']).parent for m in registry['models']):continue
        source=manifest.get('source_model_sha256',manifest.get('model_sha256'))
        target=known.get(source)
        if target:
            if (manifest.get('actual_steps')!=target['transitions'] or
                manifest['settings'].get('condition')!=target['condition'] or
                manifest['settings'].get('seed')!=target['learner_seed']):
                issues.append(f'{path.name}: V2 export identity differs from the approved checkpoint');continue
            manifest=label_v2(manifest,target,registry)
        else:
            manifest={**manifest,'application_version':'Earlier models'}
        catalog.append((path,manifest))
    if not public:
        for target in registry['models']:
            path=root/target['path'];manifest_path=root/target['manifest_path']
            if not path.is_file():continue
            try:
                if digest(manifest_path)!=target['manifest_sha256']:raise ValueError('training manifest checksum differs')
                manifest=json.loads(manifest_path.read_text())
                if manifest['status']!='completed':raise ValueError('training run is incomplete')
                manifest=label_v2(manifest,target,registry)
                # A milestone's checksum is supplied by the frozen registry.
                manifest['model_sha256']=target['sha256']
                catalog.append((path,manifest))
            except (OSError,ValueError,KeyError,TypeError) as exc:
                issues.append(f'{target["run"]}: {exc}')
    catalog.sort(key=lambda x:(x[1]['application_version']!='V2',x[1]['settings']['condition'],
                              x[1]['settings']['seed'],x[1]['settings'].get('mode')!='full',str(x[0])))
    return catalog,issues


def check(root=ROOT):
    """Validate actual SageMaker files, without training or writing research data."""
    from experiments.provenance import source_hashes
    from ui.policy_playback import validate_compatibility
    root=Path(root);registry=read_registry();problems=[];verified=[]
    if source_hashes()!=registry['source_hashes']:problems.append('Scientific source differs from the frozen V2 source')
    for target in registry['models']:
        try:
            if digest(root/target['path'])!=target['sha256']:raise ValueError('checkpoint checksum differs')
            if digest(root/target['manifest_path'])!=target['manifest_sha256']:raise ValueError('manifest checksum differs')
            m=json.loads((root/target['manifest_path']).read_text())
            if m['status']!='completed' or m['actual_steps']!=4194304:raise ValueError('run is incomplete')
            if m['settings']['condition']!=target['condition'] or m['settings']['seed']!=target['learner_seed']:raise ValueError('learner identity differs')
            validate_compatibility(label_v2(m,target,registry));verified.append(target['run'])
        except (OSError,ValueError,KeyError) as exc:problems.append(target['run']+': '+str(exc))
    report={'status':'ready' if not problems and len(verified)==20 else 'needs_attention',
            'verified_v2_models':len(verified),'expected_models':20,'default_action_selection':'DS','issues':problems}
    print(json.dumps(report,indent=2))
    return not problems and len(verified)==20


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['check'])
    parser.parse_args()
    raise SystemExit(0 if check() else 1)
