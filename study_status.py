"""Live status of the TRPO study, safe to run from a second terminal.

    python3 study_status.py                      # all runs under runs/study-v3
    python3 study_status.py runs/trpo-pilot-100   # one run
    watch -n 60 python3 study_status.py           # refresh every minute

Reads only committed files, so it never interferes with the trainer.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

TARGET_KL = 0.01


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def utc(text):
    try:
        return datetime.fromisoformat(text)
    except (TypeError, ValueError):
        return None


def report(run):
    man = load(run / "manifest.json")
    if man is None:
        print(f"{run.name:<10} no manifest yet")
        return None
    contract = load(run / "contract.json") or {}
    total = (contract.get("settings") or {}).get("total_steps") or 0
    steps = man.get("actual_steps") or 0
    status = man.get("status", "?")
    phase = man.get("current_phase", "-")
    pct = (100 * steps / total) if total else 0

    started = utc((man.get("attempts") or [{}])[0].get("started_utc"))
    finished = utc(man.get("completed_utc"))
    now = datetime.now(timezone.utc)
    elapsed = ((finished or now) - started).total_seconds() if started else 0
    rate = steps / elapsed if elapsed > 0 and steps else 0
    eta = ""
    if status == "running" and rate and total > steps:
        eta = f"  eta {(total - steps) / rate / 3600:4.1f}h"

    line = (f"{run.name:<10} {status:<10} {phase:<9} {steps:>10,} / {total:<10,} "
            f"{pct:5.1f}%  {elapsed / 3600:5.2f}h{eta}")
    if (run / "STOP_REQUESTED.json").exists():
        line += "  [pause requested]"
    print(line)

    dev = load(run / "development.json") or {}
    trpo = dev.get("trpo") or []
    if trpo:
        kls = [t["kl_divergence"] for t in trpo if t.get("kl_divergence") is not None]
        acc = [t["line_search_success"] for t in trpo if t.get("line_search_success") is not None]
        over = sum(1 for k in kls if k > TARGET_KL + 1e-9)
        flag = "  <-- KL OVER TARGET" if over else ""
        print(f"           {len(trpo):>4} updates   max KL {max(kls):.5f}"
              f"   line search {sum(acc) / len(acc):.0%} accepted{flag}")
    sampled = [r for r in (dev.get("records") or []) if r.get("mode") == "sampled"]
    if sampled:
        last = sampled[-1]
        print(f"           sampled destruction {last['mean_destruction']:.3f} "
              f"at {last['timesteps']:,}  ({len(sampled)} checks)")
    if man.get("last_error"):
        print(f"           ERROR {man['last_error']}")
    return status


def main():
    expected = None
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        runs = [Path(sys.argv[1])]
    else:
        root = Path("runs/study-v3")
        # The study creates each run directory only when its seed starts, so a
        # bare glob undercounts. Show the full planned set and mark the rest
        # pending, otherwise "5 of 6" reads as though the study were 6 learners.
        expected = [root / f"T-{s}" for s in range(10)]
        runs = [r for r in expected if r.is_dir()]
        if not runs:
            print("No runs under runs/study-v3 yet. Pass a run directory explicitly.")
            return 0

    print(f"{'run':<10} {'status':<10} {'phase':<9} {'transitions':>23}  "
          f"{'done':>5}  {'elapsed':>7}")
    print("-" * 86)
    seen = [report(r) for r in runs]
    pending = [r for r in (expected or []) if not r.is_dir()]
    if pending:
        print(f"{'':<10} not started: {', '.join(r.name for r in pending)}")
    print("-" * 86)
    total = len(expected) if expected else len(runs)
    done = sum(1 for s in seen if s == "completed")
    print(f"{done} of {total} learners completed, "
          f"{sum(1 for s in seen if s == 'running')} running, "
          f"{sum(1 for s in seen if s == 'paused')} paused, "
          f"{sum(1 for s in seen if s == 'failed')} failed, "
          f"{len(pending)} not started")
    return 0


if __name__ == "__main__":
    sys.exit(main())
