# Complete TRPO experiment import

This branch adds the complete project payload from the supplied `base-siege.zip`
to the existing Base Siege main revision `c268ada`. It preserves the current PPO,
Double DQN and selected-TRPO application integration.

## Included material

- TRPO training runner, checkpoint-retention helper and algorithm tests.
- V3 final-evaluation and exploratory mode-probe packages, protocols and tests.
- All supplied T-0 through T-9 final and placement models, training contracts,
  manifests, development histories, episode logs, TensorBoard files, milestones,
  recovery generations and previous-attempt records.
- TRPO pilot and smoke records, final/probe progress databases, lock remnants,
  terminal logs, helper scripts, handoff documents and alternate V3 UI files.
- The supplied nested experiment ZIPs, including duplicate archive copies. No
  experiment-history pruning was performed for this import.
- The supplied original PPO and combined V1/V2 handbooks. The DDQN/TRPO handbook
  was already present in main.

The existing LFS rules cover binary models, databases, event files and archives.
Scoped additional rules put the new TRPO JSONL histories and cumulative episode
snapshots in LFS without migrating existing PPO files.

## Source identity and completeness

`reference/trpo-import/IMPORT_MANIFEST.json` records the source archive SHA-256,
all 6,098 project-file members, original member hashes, import destinations and
the explicit exclusions. It distinguishes the 1,911 source LFS pointers from
their already-hydrated equivalents retained from main.

Five shared source files differ from current main: `.gitattributes`, `README.md`,
`tests/test_v2_application.py`, `ui/app.py` and `ui/policy_playback.py`. Their exact
supplied versions are preserved under
`reference/trpo-import/source-variants/`. The original attributes file has a
`.txt` suffix there so it does not install nested Git attributes. Current main's
application, replay implementation, DDQN adapters and scientific source remain
the active versions. The older `ui/app_v3.py` launch path now delegates to the
current app without monkey-patching its PPO catalogue; its original source is
retained under `source-variants/ui/app_v3.py`.

Focused test corrections accompany the import. The main UI test now checks
that the removed Experiment results mode stays absent. The V3 missing-learner
test uses a temporary incomplete registry instead of assuming the supplied
research archive is incomplete; its exact original source is also retained in
`source-variants/`. A regression checks that the older V3 launch path preserves
the current catalogue and all three algorithm selectors. CI retrieves the
inference checkpoints through LFS and runs
the imported test suites alongside the application regressions. The complete
research history remains available without downloading it all for each CI run.

Machine-specific `.venv`, nested `.git`, and generated Python, pytest and
notebook caches are excluded. They are not experiment models or research
history. The original supplied ZIP remains unchanged.

## Recorded limits

- All ten main TRPO endpoint weights match their manifests and the V3 freeze.
  The thirty V3 P/PA/T endpoint identities were checked against the supplied
  archive. These checks do not rerun training or the final evaluation.
- T-1 through T-6 do not include the recovery generations referenced by their
  `LATEST.json` files. Their endpoint models and logs are included; omitted
  recovery state cannot be reconstructed by this import.
- The supplied current TRPO runner differs from the runner digest recorded in
  historical training contracts. Algorithm and configuration files match.
- Some supplied handoff text is historical. The frozen V3 final acting mode is
  DS; the subsequent SS probe is exploratory. The completed TRPO SS result is
  90.8475%, and the evidence does not establish equivalence to PPO.
- The archive does not supply the missing full DDQN recovery buffers/states.
- Public-deployment preparation remains separate work. This import does not
  deploy an application or change the current application implementation.

Use a full LFS clone to retrieve the recorded history. GitHub's source ZIP may
contain LFS pointers rather than usable models:

```bash
git lfs install --local
git lfs pull
```
