# Completed V2 records review

Reviewed archive: `Base_Siege_V2_Review_20260927_214254.zip`.

## Evidence and limits

The archive contains the longer-control registration and six run manifests, contracts, diagnostic development results and milestone metadata. It does **not** contain the actual model ZIPs, training episode journals or checkpoint state. Metadata consistency is verified; the package's SageMaker preflight must verify the actual 24 selected model files.

All six manifests report `completed`, 4,194,304 transitions and a successful save/reload check. Their scientific-source, V2-source and registration identities agree. Contract hashes match the manifests. Each run records six milestones: 32,768; 262,144; 524,288; 1,048,576; 2,097,152; 4,194,304. The final model hash agrees with the last milestone metadata. The supplied registered V2 code and original scientific source match their recorded hashes.

All 204 rows of the four-million diagnostic summary agree with their underlying episode records. These cover six learners, seventeen budgets and two action-selection modes, comprising 19,584 diagnostic evaluation attacks. Diagnostic acting seeds and unique evaluation units were checked.

The supplied status report totals 640,807 completed training episodes across six runs, and the completed budgets total 25,165,824 transitions. Episode journals were absent from this archive, so the episode counts were not independently recounted. These are three independently trained learners per condition, not ten.

## Timing correction

| Run | Sum of recorded attempt seconds | Interpretation |
|---|---:|---|
| P-0 | 7,126.068294 | All recorded attempts have durations |
| P-1 | 7,234.666133 | Includes a failed attempt followed by successful recovery |
| P-2 | 7,143.137338 | All recorded attempts have durations |
| PA-0 | 7,233.780879 | All recorded attempts have durations |
| PA-1 | 4,210.058734 | Incomplete: attempt 3 has only a start time |
| PA-2 | 7,058.259937 | All recorded attempts have durations |

The recorded sum is 40,005.971315 seconds, approximately **11 hours 6 minutes 46 seconds**. This includes work such as diagnostics and checkpoint saving during attempts. It excludes the missing duration of PA-1 attempt 3, so it is not a complete total or a pure training-compute measurement.

PA-1 attempt 3 started at `2026-09-27T14:03:55.389357+00:00`; no finish time, status or duration was retained. Attempt 4 started at `2026-09-27T20:30:33.626829+00:00` and completed successfully. The gap between those starts must not be substituted for active computation time. PA-1 cannot be described as faster on this evidence. The records were not edited to fill the gap.

All six V2 run records identify a **Tesla T4**. That evidence applies to these V2 runs only; the GPU used for the original study remains unestablished.

## Diagnostic interpretation

At 4,194,304 transitions, mean sampled destruction on the small monitored development panel is 92.27% for P and 93.26% for PA. Deterministic means are 87.99% and 89.81%, respectively. Destruction measures building value destroyed. The sampled PA-minus-P difference is approximately +0.99 percentage points across only three learner seeds; it is not proof of a general symmetry benefit.

These monitoring results are not directly comparable with the original final-set scores. The accompanying evaluator broadens the development comparison using four retained budgets, all 156 development layouts and eight views. Its plan was specified after the four-million diagnostic results. No new final-set result or full-development result is claimed here.

No movement or targeting rule was changed. The known inaccessible-target limitation and the original scientific results remain unchanged.
