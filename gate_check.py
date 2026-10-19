"""Pilot gate check for the TRPO arm.

    python3 gate_check.py runs/trpo-pilot-100

Reads the run's own files only. Compares against PPO's published development
curve at the same transition count when that run is on disk, matching acting
mode to acting mode -- deterministic against deterministic, sampled against
sampled. Mixing the two makes TRPO look far worse than it is.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

TARGET_KL = 0.01
PPO_REFERENCE = Path("runs/version2-longer-control/P-0/development.json")


def load(path):
    return json.loads(Path(path).read_text())


def curve(records, mode):
    """(timesteps, destruction) for one acting mode, oldest first.

    V1-era runs wrote one record per interval with no 'mode' key; treat those
    as deterministic, which is what play_episode does with a bare model.
    """
    rows = [r for r in records if r.get("mode", "deterministic") == mode]
    return [(r["timesteps"], r["mean_destruction"]) for r in rows]


def main():
    run = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/trpo-pilot-100")
    dev = load(run / "development.json")
    man = load(run / "manifest.json")
    trpo = dev.get("trpo", [])
    records = dev.get("records", [])

    print(f"run              {run}")
    print(f"status           {man.get('status')}  ({man.get('actual_steps', 0):,} transitions)")
    print(f"policy updates   {len(trpo)}")

    ok = True

    if trpo:
        kls = [t["kl_divergence"] for t in trpo if t.get("kl_divergence") is not None]
        accepted = [t["line_search_success"] for t in trpo if t.get("line_search_success") is not None]
        over = [k for k in kls if k > TARGET_KL + 1e-9]
        rate = sum(accepted) / len(accepted) if accepted else 0.0
        print(f"max KL           {max(kls):.5f}  (target {TARGET_KL})")
        print(f"KL over target   {len(over)} of {len(kls)} updates")
        print(f"line search      {rate:.0%} accepted")
        if over:
            print("  FAIL  KL exceeded target: the constraint may be measured unmasked")
            ok = False
        if rate == 0:
            print("  FAIL  every step rejected: step size or damping is wrong")
            ok = False
    else:
        print("  NOTE  no TRPO diagnostics recorded")

    modes = sorted({r.get("mode", "deterministic") for r in records})
    print(f"development      {len(records)} records, modes: {', '.join(modes) or 'none'}")

    reference = load(PPO_REFERENCE) if PPO_REFERENCE.is_file() else None
    for mode in ("sampled", "deterministic"):
        points = curve(records, mode)
        if not points:
            continue
        first, last = points[0], points[-1]
        print(f"  {mode:<14} {first[1]:.3f} at {first[0]:,}  ->  {last[1]:.3f} at {last[0]:,}")
        # Only the sampled curve gates. The frozen evaluation acts with DS
        # (deterministic purchasing, sampled deployment), so sampled is the
        # closer proxy; pure-deterministic acting is a known weak mode in this
        # project (V1: 43.5% deterministic against 86.0% sampled for PPO) and
        # is reported here as a diagnostic, not a gate.
        gating = mode == "sampled"
        if len(points) >= 3 and last[1] <= first[1]:
            print(f"    {'FAIL' if gating else 'NOTE'}  no improvement in {mode} acting")
            if gating:
                ok = False
        if reference:
            ppo = dict(curve(reference.get("records", reference), mode))
            match = ppo.get(last[0])
            if match is not None:
                gap = last[1] - match
                verdict = "ahead of" if gap > 0 else "behind"
                print(f"    PPO P-0 at {last[0]:,}: {match:.3f}  ->  TRPO is {abs(gap):.3f} {verdict} PPO")
                if gap < -0.10:
                    print(f"    {'REVIEW' if gating else 'NOTE '}  {abs(gap):.3f} behind PPO at a matched budget")
                    if gating:
                        ok = False
            else:
                have = sorted(ppo)
                near = min(have, key=lambda t: abs(t - last[0])) if have else None
                if near is not None:
                    print(f"    PPO P-0 has no record at {last[0]:,}; nearest is {near:,} ({ppo[near]:.3f})")

    attempts = man.get("attempts") or []
    if attempts and attempts[0].get("started_utc") and man.get("completed_utc"):
        start = datetime.fromisoformat(attempts[0]["started_utc"])
        end = datetime.fromisoformat(man["completed_utc"])
        seconds = (end - start).total_seconds()
        per = seconds / max(man.get("actual_steps", 1), 1)
        print(f"runtime          {seconds / 60:.1f} min  ({per * 1000:.2f} ms/transition)")
        print(f"  4,194,304 would take ~{per * 4194304 / 3600:.1f} h per learner, "
              f"~{per * 4194304 * 10 / 3600:.0f} h for ten")
        print("  PPO measured 1.70 ms/transition on a T4 for comparison")

    print("\nGATES:", "PASS" if ok else "REVIEW")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
