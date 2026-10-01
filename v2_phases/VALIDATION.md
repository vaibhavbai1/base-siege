# Software validation

The evaluator passed thirteen local software tests:

```text
.............                                                            [100%]
13 passed in 11.32s
```

Checks cover all four phase-flag combinations, Finish shopping behavior, exact SS/DD action-sequence and metric agreement with the original acting policy, unchanged Torch random-number progression for those uniform modes, matching purchasing prefixes in the mixed modes, pause/resume parity, idempotent completed exports, packaging and output-hash checks, safe pause after saving the current attack, changed-contract rejection, corrupt-progress rejection, DD baseline counting, duplicate-unit rejection, the complete 44,928-unit design and equal-weight learner comparisons.

The actual reviewed baseline CSV was imported separately: 29,952 rows, comprising 22,464 SS and 7,488 DD. Recomputed final-checkpoint destruction means matched the reviewed full-development figures. The original scientific source and restored full-development evaluator match their recorded hashes. They were not edited.

Tests use an existing original P-0 checkpoint. No model was trained. Local runtime: Python 3.12.14, Torch 2.13.0, NumPy 2.5.2, Gymnasium 1.3.0, Stable-Baselines3 2.9.0 and SB3 Contrib 2.9.0, with CPU inference.

The six actual final V2 model ZIPs are stored on SageMaker and were not included in the uploaded results. Consequently, the production preflight and full 44,928-attack mixed-mode evaluation have not been run here. No mixed-mode performance result or measured production duration is claimed. SageMaker preflight must verify the real checkpoint files before evaluation starts.
