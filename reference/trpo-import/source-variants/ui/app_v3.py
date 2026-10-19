"""V3 entrypoint: the V2 application, plus the TRPO condition in the catalog.

    streamlit run ui/app_v3.py

Follows the same convention as ui/streamlit_app.py -- run ui/app.py through
runpy -- so that **no V2 file is modified**. The V2 application, its registry
and its twenty approved checkpoints stay byte-identical; this only adds the
separate V3 registry to the catalog the app builds at startup.

Switching algorithm in the interface: Policy replay -> Model version, where V2
gives the PPO conditions P and PA and V3 gives the TRPO condition T. Each
version reads its own registry, so a problem with the TRPO arm can never
invalidate the V2 models.
"""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ui.v2_models as v2_models
from ui.v3_models import application_catalog as v3_application_catalog

_v2_application_catalog = v2_models.application_catalog


def _combined_catalog(root, paths, *, public=False):
    """V2 catalog first, then V3. Local run folders stay out of the public demo."""
    catalog, issues = _v2_application_catalog(root, paths, public=public)
    if public:
        return catalog, issues
    v3_catalog, v3_issues = v3_application_catalog(root)
    return catalog + v3_catalog, issues + v3_issues


# app.py does `from ui.v2_models import application_catalog` when runpy executes
# it below, so rebinding the attribute here is what reaches it. Patching before
# the run is the whole mechanism; nothing in V2 is edited on disk.
v2_models.application_catalog = _combined_catalog

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
