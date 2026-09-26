"""Execute checked-in notebooks with this project's Python kernel."""
from pathlib import Path
import argparse
import json
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

ROOT = Path(__file__).resolve().parents[1]
def execute(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        manager = KernelManager(kernel_name="python3")
        manager.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
        client = NotebookClient(notebook, timeout=120, km=manager, resources={"metadata": {"path": str(ROOT)}})
        try:
            client.execute()
        finally:
            if manager.has_kernel:
                manager.shutdown_kernel(now=True)
        nbformat.write(notebook, output / path.name)
        results.append({"notebook": path.name, "code_cells": sum(c.cell_type == "code" for c in notebook.cells), "status": "passed"})
        print(path.name, "passed", flush=True)
    (output / "execution.json").write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output folder; previous evidence is preserved")
    execute(parser.parse_args().out)
