"""Development-only evaluation and a recorded replay compatible with Saved replay."""
import csv
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from env.siege_env import SiegeEnv
from env.symmetry import SymmetryWrapper
from experiments.design import development_panel
from experiments.journal import open_journal
from experiments.provenance import sha256
from experiments.rollout import play_episode, save_replay
from experiments.storage import atomic_bytes, publish_json
from ddqn_experiment.core import QNetwork, choose_action
from ddqn_experiment.runner import checked_pool, configure, sources, runtime


def load_policy(run, *, steps=None, device="cpu"):
    run = Path(run)
    contract = json.loads((run / "contract.json").read_text())
    if contract["sources"] != sources():
        raise ValueError("Source changed since training; restore the matching code")
    if contract["runtime"] != runtime():
        raise ValueError("Runtime changed since training; use the recorded environment")
    if steps is None:
        steps = json.loads((run / "LATEST.json").read_text())["actual_steps"]
    path = run / "policies" / f"step-{steps:012d}.pt"
    metadata = json.loads(path.with_suffix(".json").read_text())
    if sha256(path) != metadata["sha256"] or metadata["contract_sha256"] != sha256(run / "contract.json"):
        raise ValueError("Policy checksum or training contract mismatch")
    state = torch.load(path, map_location=device, weights_only=True)
    if state["algorithm"] != "masked-double-dqn" or state["steps"] != steps:
        raise ValueError("Wrong policy type or step")
    network = QNetwork(SiegeEnv().observation_space).to(device)
    network.load_state_dict(state["online"])
    network.eval()
    return network, {"algorithm": "masked-double-dqn", "learner_seed": contract["seed"],
                     "condition": "no-augmentation", "training_steps": steps,
                     "training_mode": contract["mode"], "policy_sha256": metadata["sha256"],
                     "training_contract_sha256": sha256(run / "contract.json"),
                     "acting": "greedy-legal-lowest-index-tie"}


class GreedyPolicy:
    def __init__(self, network, device="cpu"):
        self.network, self.device = network, device
        self.rng = np.random.default_rng(0)  # Unused by greedy acting.

    def act(self, observation, mask):
        return choose_action(self.network, observation, mask, self.rng, 0.0, self.device)


def evaluate(run, dev_pool, output, *, steps=None, views=(0, 1, 4, 5), panel=False,
             resume=False, device="cpu"):
    configure(0, device)
    if not views or any(v not in range(8) for v in views) or len(set(views)) != len(views):
        raise ValueError("Views must be unique integers in 0–7")
    bases = checked_pool(dev_pool, "dev")
    training_contract = json.loads((Path(run) / "contract.json").read_text())
    if sha256(dev_pool) != training_contract["pools"].get("dev", {}).get("sha256"):
        raise ValueError("Development pool differs from the training contract")
    if panel:
        bases = development_panel(bases)
    bases = sorted(bases, key=lambda b: (b.motif, b.seed))
    network, identity = load_policy(run, steps=steps, device=device)
    output = Path(output)
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
    elif not output.is_dir():
        raise FileNotFoundError(output)
    contract = {"scope": "development only; not final evaluation", "policy": identity,
                "pool_sha256": sha256(dev_pool), "views": list(views), "device": device,
                "layouts": [json.loads(b.to_json()) for b in bases],
                "sources": sources(), "runtime": runtime()}
    publish_json(output / "contract.json", contract)
    rows = []
    policy = GreedyPolicy(network, device)
    with open_journal(output / "progress.sqlite3", contract, resume=resume) as journal:
        for base in bases:
            for view in views:
                key = f"{base.motif}:{base.seed}:{view}"
                row = journal.get(key)
                if row is None:
                    env = SymmetryWrapper(SiegeEnv(), transform=view)
                    started = perf_counter()
                    try:
                        replay = play_episode(env, policy, seed=0, options={"spec": base})
                    finally:
                        env.close()
                    row = {"base_seed": base.seed, "motif": base.motif, "view": view,
                           **replay["summary"], "measured_episode_seconds": perf_counter()-started}
                    journal.put(key, row)
                    print(f"Development {len(rows)+1}/{len(bases)*len(views)} "
                          f"destruction={row['destruction']:.3f}", flush=True)
                rows.append(row)
    summary = {"scope": "development only", "policy": identity, "layouts": len(bases),
               "views": list(views), "episodes": len(rows),
               "mean_destruction_percent": 100*float(np.mean([r["destruction"] for r in rows])),
               "full_clear_percent": 100*float(np.mean([r["full_clear"] for r in rows])),
               "town_hall_percent": 100*float(np.mean([r["town_hall"] for r in rows])),
               "mean_return": float(np.mean([r["episode_return"] for r in rows])),
               "mean_spent": float(np.mean([r["spent"] for r in rows])),
               "measured_episode_seconds_sum": sum(r["measured_episode_seconds"] for r in rows),
               "limitation": "one learner; no across-learner uncertainty estimate"}
    fields = ("base_seed", "motif", "view", "destruction", "full_clear", "town_hall",
              "episode_return", "spent", "tick", "decisions", "finish_reason", "measured_episode_seconds")
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(output / "episodes.csv", buffer.getvalue().encode())
    publish_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


def record_replay(run, dev_pool, output, *, steps=None, index=0, view=0, device="cpu"):
    configure(0, device)
    bases = sorted(checked_pool(dev_pool, "dev"), key=lambda b: (b.motif, b.seed))
    if not 0 <= index < len(bases) or view not in range(8):
        raise ValueError("Invalid development layout index or view")
    contract = json.loads((Path(run) / "contract.json").read_text())
    if sha256(dev_pool) != contract["pools"].get("dev", {}).get("sha256"):
        raise ValueError("Development pool differs from training")
    network, identity = load_policy(run, steps=steps, device=device)
    env = SymmetryWrapper(SiegeEnv(record=True, record_ticks=True), transform=view)
    try:
        replay = play_episode(env, GreedyPolicy(network, device), seed=0, options={"spec": bases[index]})
        replay["policy"] = identity
        replay["scope"] = "development demonstration"
        save_replay(replay, output)
    finally:
        env.close()
    return replay["summary"]
