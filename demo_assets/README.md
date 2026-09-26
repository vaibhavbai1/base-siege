# Public demo assets

This directory is an explicit publishing boundary. The public app never scans `runs/`
or `artifacts/`. It starts with manual play and two baseline agents; no trained-policy
claim is made when there is no exported model.

Use `python tools/export_demo_model.py --run runs/P-0 --name P-0` after reviewing a
trusted, completed run. The exporter saves CPU inference weights and a small public
manifest under `models/P-0/`. It does not copy recovery state or private run metadata.

Reviewed replay JSON files can go in `replays/`, and reviewed development evaluation
CSVs can go in `evaluations/`. Do not add credentials, raw notebooks, training logs,
protected evaluation layouts, private paths or recovery checkpoints. Never load an
untrusted model: SB3 model files can contain executable Python serialisation.

See `docs/DEPLOYMENT.md` for the full workflow.
