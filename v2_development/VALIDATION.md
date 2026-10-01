# Software validation

Seven tests passed locally after the final evaluator change:

```text
.......                                                                  [100%]
7 passed in 3.25s
```

The tests cover:

1. A small real-policy and heuristic evaluation, comparing uninterrupted execution against pause/resume. Raw CSVs, summaries and completion receipts agree; rerunning completed work performs no attacks.
2. Rejection of a changed evaluation contract.
3. A pause request that finishes and saves the current attack before stopping.
4. Rejection of a corrupted episode checksum in the SQLite progress store.
5. Model-file hash and timestep-counter checks.
6. The full design's 121,056 unique evaluation units, without running those attacks.
7. Aggregation by learner and budget, including exclusion of the twelve monitored layouts from the secondary reporting scope.

The small execution tests use an existing original P-0 checkpoint. No model was trained for these tests. They are software checks, not new scientific performance results.

Local test environment: Python 3.12.14, PyTorch 2.13.0, NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0 and SB3 Contrib 2.9.0; CPU inference. Production preflight separately requires the scientific versions recorded in the supplied V2 run records.

The actual V2 checkpoint ZIPs were absent from the review upload. The full production preflight and 121,056-attack evaluation have therefore **not** been run here. Run the supplied preflight on SageMaker, where those checkpoints are stored, before beginning evaluation. No duration estimate is claimed until the first real evaluation batch has measured throughput.
