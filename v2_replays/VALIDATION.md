# Validation record

The final replay code passed seven local software tests:

```text
.......                                                                  [100%]
7 passed in 7.35s
```

Tests use the existing original P-0 checkpoint, not a newly trained model. They cover sampled and deterministic action/metric parity with the ordinary simulator, unchanged Torch random-number progression, inverse-transformed deployment coordinates, army accounting, rejection of wrong expected metrics/seeds, identical uninterrupted versus resumed exports, repeat packaging, corruption rejection, selected-case identities and reviewed-result hash guards.

The full selected case plan was generated from the uploaded 121,056-row `episodes.csv`: 48 unique cases in 24 matched layout/view pairs. The PA-1 zero-destruction attack is included. Outcome selection is deliberate and limits this collection to qualitative diagnosis.

The recorded scientific source hashes and the existing full-development evaluator hashes were checked against the completed evaluation contract and remain unchanged. The replay renderer and its six frontend assets were copied from the existing offline exporter into this new package. No application or scientific source file was overwritten.

Generated test HTML contained all required inline scripts/styles, and all four embedded JavaScript blocks passed Node syntax checks. A live browser playback check could not be completed because the browser executable was unavailable and its download failed. No new browser-rendering verification is claimed.

Local runtime: Python 3.12.14; PyTorch 2.13.0; NumPy 2.5.2; Gymnasium 1.3.0; Stable-Baselines3 2.9.0; SB3 Contrib 2.9.0; CPU inference. Production preflight verifies the saved evaluation runtime before generating V2 replays.

The uploaded results archive did not include the six actual V2 checkpoint ZIPs. Therefore the 48 production replays have not been generated locally, and no cause of a V2 failure is claimed yet. Running this package on SageMaker will reproduce and verify those attacks, then export their recorded frames for inspection.
