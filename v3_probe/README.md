# Acting-mode probe (exploratory)

**This is not part of the frozen V3 protocol.** It was written after seeing the V3
result and must be reported as a post-hoc analysis. It reads
`artifacts/version3-final-evaluation/` and never writes to it.

## The question

The V3 final evaluation ran one acting mode, **DS** — deterministic purchasing,
sampled deployment — and produced:

| Condition | Destruction | Full clear | Army eliminated |
| --- | --- | --- | --- |
| P | 92.9% | 74.2% | 9.5% |
| T | 56.1% | 7.5% | 72% |
| heuristic | 55.1% | 3.5% | — |

But development, which sampled both phases, had T at 92.5% against P's 92.3%. T
also spends 199.7 of its 200 resources under DS, so it is not underbuying — it
buys a full-value army deterministically and that army is destroyed.

Both results can only hold if the deficit is in the purchasing phase.

## The test

```
DS = (purchasing deterministic, deployment sampled)   the frozen protocol
SS = (purchasing sampled,       deployment sampled)   this probe
```

Deployment is sampled in both, so **purchasing is the only variable**. No third
mode is needed; `test_probe_varies_purchasing_only` enforces this.

The probe reuses `final.acting_seed`, so each SS episode runs under the same RNG
seed as its DS counterpart. The comparison is paired per learner per layout per
view, and a difference cannot be a different draw.

Condition P is included as the control. If P moves under the mode switch too,
the effect is not specific to TRPO and the interpretation changes.

## Expected outcomes

| Result | Reading |
| --- | --- |
| T recovers to ~92%, P unchanged | The V3 deficit is acting-mode sensitivity, not policy quality. TRPO's purchasing head never sharpened, so its argmax is degenerate while its samples are fine. |
| T stays near 56% | The deficit is not about purchasing determinism. The V3 result stands as measured and needs another explanation. |
| P also moves substantially | The mode switch is not TRPO-specific; re-examine the comparison itself. |

Record whichever happens. The second and third outcomes are results, not failures.

## Running

```bash
python -m pytest v3_probe/tests -q
python -m v3_probe run --max-new 200        # smoke test, a few minutes
python -m v3_probe run --resume             # the full probe
python -m v3_probe export
```

Full size is `conditions x 10 learners x 241 layouts x 8 views x 3 sample seeds`
— 57,840 attacks per condition, so 115,680 for the default `P T`. At the measured
~12 attacks/sec that is roughly 2.7 hours on CPU. `--conditions T` alone halves it,
at the cost of losing the control.

Journalled per attack, so `--resume` after any interruption costs one episode.

## Outputs

Written to `artifacts/version3-mode-probe/`:

| File | Contents |
| --- | --- |
| `probe_contract.json` | What was probed, against which freeze and pool, with hashes |
| `episodes.csv` | One row per attack, with `purchasing` and `deployment` columns |
| `mode_by_learner.csv` | DS and SS destruction per learner, and the difference |
| `mode_summary.csv` | Per condition: mean DS, mean SS, mean change, learners improved |
| `complete.json` | Completion record and output hashes |

## What it refuses to do

It will not run against a V3 evaluation that has not completed (there would be no
DS numbers to compare against), will not run if `protocol.json` changed since the
freeze, and will not reuse an output directory that holds a probe of a different
mode, pool or condition set. It verifies `episodes.csv` against the hash in the
final evaluation's `complete.json` before reading the DS baseline from it.
