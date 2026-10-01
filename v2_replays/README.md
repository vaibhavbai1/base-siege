# V2 replay inspection

This is the next diagnostic step after the completed 121,056-attack full development evaluation. It reproduces selected attacks from the six completed 4,194,304-transition models. It does not train, change rewards or movement, add reachable-target fallback, or access final-test layouts.

## What the package produces

There are **48 replay cases arranged into 24 comparisons**. For each learner (P-0, P-1, P-2, PA-0, PA-1, PA-2), the package selects four sampled attacks:

1. The lowest-destruction attack ending with the army eliminated.
2. The lowest-destruction attack ending at the battle time limit.
3. The middle-ranked non-clear attack, ordered by destruction.
4. A full clear with a middle-ranked completion tick.

Each sampled attack is paired with the recorded deterministic attack from the same learner, layout and coordinate view. Ties follow a fixed evaluation-key order. The lower median is used for an even number of candidates. `selection_plan.json` contains the exact cases and metrics before any replay is run.

This deliberately selected collection includes failures. **Do not average its results to estimate agent performance.** Use the full-development summaries for that. Replaying these attacks also does not add 48 new independent observations to the completed evaluation.

The existing general-purpose demo interface derives sampling seeds differently. This tool uses the original V2 evaluation seed derivation and the stored action-seed label, then verifies destruction, return, finish reason, Town Hall/full-clear flags, spending, tick, decision count, base, motif and view. A mismatch stops execution. The evaluation CSV did not retain full action sequences, so matching scalar metrics is not a direct comparison against an archived original action trace.

## 1. Upload and install

Upload `Base_Siege_V2_Replay_Inspection.zip` into your existing SageMaker project folder. Open a **terminal**, not a notebook cell:

```bash
cd ~/Base_Siege_Smooth_Demo/base-siege-assignment3
unzip -n Base_Siege_V2_Replay_Inspection.zip
```

If you moved the project, adjust the first path. The correct folder contains `version2/`, `v2_development/`, `env/`, `runs/`, `artifacts/`, `pyproject.toml` and `uv.lock`.

The ZIP adds only `v2_replays/`. Keep the completed training and evaluation folders as they are. Do not change dependency versions. The offline player is a frozen copy of the existing project player, stored inside this additional package, so the application files are not replaced.

## 2. Verify the inputs

```bash
uv run --frozen python -m v2_replays preflight
uv run --frozen python -m pytest v2_replays/tests -q
```

Expected preflight output includes `status: ready`, `cases: 48`, `pairs: 24` and `models: 6`. It checks the reviewed result hashes, recomputes the case selection, and uses the existing full-development preflight to verify the frozen scientific source, evaluation code, registered model milestones and recorded software environment. It checks all 24 milestones required by that earlier evaluator, although the replays use only the six final milestones.

The seven software tests use the existing original `runs/study-v2/P-0/model.zip`. They do not train. The replay run uses the six V2 checkpoints instead. If a check fails, share the complete error. Do not edit the reference hashes or expected scores.

## 3. Generate the replays and download package

```bash
uv run --frozen python -m v2_replays run
uv run --frozen python -m v2_replays pack
```

This reproduces only 48 attacks, so the simulation workload is much smaller than the full evaluation. It also writes per-tick frames and HTML files, so the evaluation throughput is not an exact runtime estimate. Progress prints after each verified case.

Files are saved under:

```text
artifacts/version2-replay-inspection/
```

The `pack` command verifies the completed outputs and creates:

```text
artifacts/Base_Siege_V2_Replay_Inspection_Results.zip
```

In the SageMaker file browser, open `artifacts`, refresh, right-click that ZIP and choose **Download**. Upload the ZIP here for failure analysis.

## 4. Watch the attacks on your computer

On Windows, right-click the downloaded results ZIP and choose **Extract All**. Open the extracted `version2-replay-inspection` folder and double-click **index.html**. Keep the `cases` folder beside it. Open the extracted index rather than opening files directly inside the ZIP.

The index shows the sampled and deterministic versions side by side as links. Select **Watch**, then press **Play**. Use the speed menu, timeline, Ranges and Paths controls. No Python installation, SageMaker connection or web server is required for these offline HTML replays. The displayed battlefield uses the original base orientation; the agent received the selected rotated/reflected view.

The **Army and decisions** link shows:

- Troops purchased, deployed and left in reserve.
- Every purchasing/deployment decision and the simulator tick at which it happened.
- Remaining non-wall buildings and surviving troops.
- A passive diagnostic count described below.

Suggested cases to examine first:

| Replay | Reason |
|---|---|
| `case-007` | P-0 sampled full clear |
| `case-001` and `case-002` | P-0's worst sampled army-elimination attack and its deterministic counterpart |
| `case-003` and `case-004` | P-0's worst sampled time-limit attack and its counterpart |
| `case-033` and `case-034` | PA-1's zero-destruction sampled attack and its counterpart |

The sampled/deterministic pair can differ in both purchases and deployments. Watching one pair does not isolate which phase caused the difference.

## Diagnostic limits

The package adds annotations to copies of recorded presentation frames. It does not change the policy observations, legal-action masks, combat state or simulator functions. The software tests check that this recording preserves actions, outcomes and Torch's random-number progression.

For each surviving troop at the end of a tick, it records its target, whether the target is still alive/in range, and the number of approach goals returned by the existing pure geometry helper. It performs no extra path search and never switches targets. Duplicate frames at the same tick are excluded from the count.

An out-of-range troop with no approach goal is a **post-tick diagnostic observation**, not proof that it caused the attack to fail. It does not measure all movement failures and must not be reported as the prevalence of inaccessible-target failures across all 121,056 attacks. This feature only makes the existing mechanics easier to inspect. No reachable-target fallback was added.

Likewise, `army_eliminated` and `tick_limit` describe how the battle ended, not why the strategy was poor. Inspect the timeline and playback before assigning causes.

## Interruptions and repeat commands

Each case has a checksum receipt after its replay, diagnostics and HTML exports are complete. If the process stops, use the same `run` command again. Completed cases are verified and reused. An interrupted case can be replayed; original evaluation results are never overwritten. Do not run two copies at once.

For an intentionally short batch:

```bash
uv run --frozen python -m v2_replays run --max-new-cases 4
```

Then resume without `--max-new-cases`. Packaging requires all cases to be complete. Keep the package code and inputs unchanged between attempts. Repeating `pack` with identical outputs is safe.

## What happens after this inspection

Use the generated evidence to decide whether failures are mainly associated with troop purchases, deployment timing/positions, surviving structures, or existing simulator limitations. A targeted observation or reward experiment would need a separate version and a fair comparison; visual correlation alone does not prove that a proposed change will help.

If the current setup is retained, the next experimental extension can add learner seeds 3–9 for each condition at the selected 4,194,304-transition budget. That is fourteen new independent training runs, not continuation of the existing six. Record the extension and fresh final-test protocol before starting. This replay package does not launch that training or create a final test.

## Files

`inspection_contract.json` identifies inputs/code; `selection_plan.json` explains selection; `diagnostics.csv` summarizes the selected cases; `cases/` contains per-case recorded frames, diagnostics, HTML playback, decision details and checksum receipts; `complete.json` identifies the completed outputs. Keep the full results ZIP as the diagnostic record.
