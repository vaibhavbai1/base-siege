"""
pytest configuration.

Its main job is simply to exist: pytest treats the directory containing the
topmost conftest.py as the rootdir. The explicit sys.path insert below makes
`from env import config` work no matter which directory you invoke pytest
from, on any OS, which saves the "works on my machine" conversation.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
