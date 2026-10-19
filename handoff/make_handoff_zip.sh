#!/usr/bin/env bash
# Assemble the TRPO handoff ZIP for the team.
#
#   bash handoff/make_handoff_zip.sh                 # T-1 as the selected checkpoint
#   bash handoff/make_handoff_zip.sh --learner 5
#   bash handoff/make_handoff_zip.sh --out ~/trpo-handoff.zip
#
# Run from the project root. Produces ~/trpo-handoff.zip.
#
# Deliberately EXCLUDED, per their message and for size:
#   .venv, package caches                 not wanted
#   intermediate checkpoints, milestones  not wanted; one selected checkpoint is enough
#   evaluation.progress.sqlite3 (108 MB)  the journal; episodes.csv is the artifact
#                                         of record and is hashed in complete.json
#   episodes.jsonl per learner            raw training episodes; development.json
#                                         carries the learning curve they asked for

set -euo pipefail

LEARNER=1
OUT="$HOME/trpo-handoff.zip"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --learner) LEARNER="$2"; shift 2 ;;
    --out)     OUT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

[[ -f handoff/EXPERIMENT_DETAILS.md ]] || {
  echo "Run this from the project root (handoff/ not found)." >&2; exit 1; }

RUN="runs/study-v3/T-${LEARNER}"
[[ -f "$RUN/model.zip" ]] || { echo "No checkpoint at $RUN/model.zip" >&2; exit 1; }

echo "Selected checkpoint: $RUN/model.zip"
rm -f "$OUT"

# --- 1. code: implementation, evaluation, acting procedure, UI integration ----
zip -qr "$OUT" \
  trpo/ v3_final/ v3_probe/ handoff/ \
  v2_phases/runner.py \
  ui/v3_models.py ui/app_v3.py ui/v3_checkpoints.json \
  source_check.py metrics.py study_status.py gate_check.py \
  pyproject.toml requirements.txt uv.lock \
  -x '*__pycache__*' '*.pyc'

# The simulator, unmodified. All four directories go in whole rather than
# file-by-file, for two reasons: "any changed environment files" is best answered
# by handing over every environment file, and the 28 files hashed in
# ui/v2_checkpoints.json live across all four. A partial copy would make
# `python -m ui.v2_models check` fail on a missing file and look like a change.
# Together they are under 700 KB.
zip -qr "$OUT" env/ train/ agents/ experiments/ \
  ui/v2_models.py ui/v2_checkpoints.json \
  -x '*__pycache__*' '*.pyc'

# --- 2. the selected checkpoint and its provenance ---------------------------
zip -qr "$OUT" "$RUN/model.zip" "$RUN/contract.json" "$RUN/manifest.json"

# --- 3. training logs for every learner --------------------------------------
zip -qr "$OUT" \
  runs/study-v3/T-*/contract.json \
  runs/study-v3/T-*/manifest.json \
  runs/study-v3/T-*/development.json \
  runs/study-v3/T-*/recovery/ \
  2>/dev/null || true

# the pilot, which validated the implementation before the study
zip -qr "$OUT" \
  runs/trpo-pilot-100/contract.json \
  runs/trpo-pilot-100/manifest.json \
  runs/trpo-pilot-100/development.json \
  2>/dev/null || true

# --- 4. evaluation results and the exact layouts -----------------------------
zip -qr "$OUT" \
  artifacts/version3-final-evaluation/episodes.csv \
  artifacts/version3-final-evaluation/model_summary.csv \
  artifacts/version3-final-evaluation/condition_summary.csv \
  artifacts/version3-final-evaluation/comparisons.csv \
  artifacts/version3-final-evaluation/comparisons_by_learner.csv \
  artifacts/version3-final-evaluation/final_pool.json \
  artifacts/version3-final-evaluation/freeze.json \
  artifacts/version3-final-evaluation/timing_summary.json \
  artifacts/version3-final-evaluation/complete.json

# the acting-mode probe, if it has been run and exported
if [[ -f artifacts/version3-mode-probe/complete.json ]]; then
  zip -qr "$OUT" \
    artifacts/version3-mode-probe/episodes.csv \
    artifacts/version3-mode-probe/mode_by_learner.csv \
    artifacts/version3-mode-probe/mode_summary.csv \
    artifacts/version3-mode-probe/probe_contract.json \
    artifacts/version3-mode-probe/complete.json
  echo "Included the acting-mode probe."
else
  echo "! The acting-mode probe has not been exported yet."
  echo "  The DS numbers alone understate TRPO substantially."
  echo "  Run it, then re-run this script, or say so in your message."
fi

# --- 5. the pools the models trained and were checked against ----------------
zip -qr "$OUT" bases/train-v2.json bases/dev-v2.json 2>/dev/null || true

echo
echo "Wrote $OUT"
du -h "$OUT"
echo
echo "Check it opens and the checkpoint is inside:"
echo "  unzip -l $OUT | grep -E 'model.zip|EXPERIMENT_DETAILS|predict_example'"
