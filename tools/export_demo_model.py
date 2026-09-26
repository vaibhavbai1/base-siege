"""Export a trusted completed run for inference, without its recovery state."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def export_model(run, destination):
    from sb3_contrib import MaskablePPO
    from env.siege_env import SiegeEnv
    from train.features import SiegeFeatures

    run, destination = Path(run), Path(destination)
    if destination.exists():
        raise FileExistsError("Choose a new export folder; existing assets are not overwritten.")
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("status") != "completed":
        raise ValueError("Export a completed run, not a partial recovery checkpoint.")
    settings = manifest["settings"]
    if settings["mode"] not in ("smoke", "full") or settings["condition"] not in ("P", "PA"):
        raise ValueError("Unrecognised training condition or evidence mode.")
    model_path = run / "model.zip"
    digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest != manifest.get("model_sha256"):
        raise ValueError("Run model checksum does not match its manifest.")
    # Load only the team's trusted artifacts: SB3 uses Python serialisation.
    source = MaskablePPO.load(model_path, device="cpu")
    if source.num_timesteps != manifest["actual_steps"]:
        raise ValueError("Model step count differs from the run manifest.")
    env = SiegeEnv(base_seed=42, motifs=("closed_ring",))
    try:
        policy_kwargs = {"features_extractor_class": SiegeFeatures, "normalize_images": False,
                         "net_arch": source.policy_kwargs["net_arch"]}
        exported = MaskablePPO("MultiInputPolicy", env, device="cpu", n_steps=2,
                              batch_size=2, policy_kwargs=policy_kwargs, verbose=0)
        exported.policy.load_state_dict(source.policy.state_dict(), strict=True)
        exported.num_timesteps = source.num_timesteps
        obs, _ = env.reset(seed=0)
        mask = env.action_masks()
        if int(source.predict(obs, action_masks=mask, deterministic=True)[0]) != int(
                exported.predict(obs, action_masks=mask, deterministic=True)[0]):
            raise ValueError("Exported policy failed the prediction parity check.")
        destination.mkdir(parents=True, exist_ok=False)
        # The new optimiser has no learned state; SB3 retains its empty configuration.
        exported.save(destination / "model.zip")
    finally:
        env.close()
    public_manifest = {
        "schema_version": 1, "purpose": "inference only; not a resumable training checkpoint",
        "status": "completed", "actual_steps": int(manifest["actual_steps"]),
        "settings": {key: settings[key] for key in ("condition", "mode", "seed")},
        "evidence_scope": ("software smoke test" if settings["mode"] == "smoke"
                           else "trained policy; performance claims require held-out evaluation"),
        "source_model_sha256": digest,
        "model_sha256": hashlib.sha256((destination / "model.zip").read_bytes()).hexdigest(),
    }
    (destination / "manifest.json").write_text(json.dumps(public_manifest, indent=2) + "\n")
    return public_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Trusted, completed training run")
    parser.add_argument("--name", required=True, help="New public model label, for example P-0")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.name):
        parser.error("Use a short label containing only letters, numbers, hyphens or underscores.")
    result = export_model(args.run, ROOT / "demo_assets/models" / args.name)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
