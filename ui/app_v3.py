"""Compatibility entrypoint for the current PPO, DDQN and TRPO application.

Keep older SageMaker launch commands working without patching the PPO catalogue
or routing TRPO checkpoints through the PPO loader.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
