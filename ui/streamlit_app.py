"""Public Community Cloud entrypoint; training remains in SageMaker Studio."""
import os
from pathlib import Path
import runpy

# Set before NumPy or PyTorch imports. A demo should not claim every CPU thread.
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

runpy.run_path(str(Path(__file__).with_name("app.py")),
               init_globals={"PUBLIC_DEMO": True}, run_name="__main__")
