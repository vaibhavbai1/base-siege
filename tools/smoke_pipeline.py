"""Bounded end-to-end integration check; never starts a full experiment."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from env import config as C
from env.siege_env import SiegeEnv
from agents.heuristic import HeuristicCommander
from experiments.pools import build_pool, load_pool
from experiments.provenance import write_json, runtime_manifest
from experiments.rollout import play_episode, save_replay
from train.train_ppo import TrainingSettings, train
from train.evaluate import evaluate_study, validate_matched_runs, paired_summary, study_summary


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train_path, dev_path = output/"pools/train.json", output/"pools/dev.json"
    build_pool(0, 19, C.TRAIN_MOTIFS, train_path)
    build_pool(10200, 10204, C.DEV_MOTIFS, dev_path)
    manifests = []
    bases = load_pool(dev_path, required_split="dev")
    for condition in ("P", "PA"):
        settings = TrainingSettings(condition=condition, n_envs=2)
        manifest = train(settings, output/condition, train_pool=train_path, dev_pool=dev_path)
        manifests.append(manifest)
        print(condition, manifest["actual_steps"], "steps; reload passed", flush=True)
    validate_matched_runs(manifests)
    frame = evaluate_study(bases, [output/"P", output/"PA"], output/"evaluation")
    repeated = evaluate_study(bases, [output/"P", output/"PA"], output/"evaluation", resume=True)
    pd.testing.assert_frame_equal(frame, repeated)
    frame["evidence_scope"] = "software smoke test"
    frame.to_csv(output/"episodes.csv", index=False)
    write_json(output/"paired_summary.json", paired_summary(frame))
    write_json(output/"study_summary.json", study_summary(frame))
    replay = play_episode(SiegeEnv(base_seed=42, record=True), HeuristicCommander())
    save_replay(replay, output/"heuristic-replay.json")
    write_json(output/"verification.json", {"status": "passed", "scope": "software integration, not policy performance",
        "two_worker_training": True, "journal_resume_idempotent": True,
        "conditions": ["P", "PA"], "steps_per_condition": 256,
        "dev_layouts": len(bases), "evaluation_rows": len(frame), "views": 8,
        "final_seeds_accessed": False, **runtime_manifest()})
    print("Smoke pipeline passed:", output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run(parser.parse_args().out)
