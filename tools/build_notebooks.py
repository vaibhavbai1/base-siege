"""Build readable, executable notebooks from versioned study cells."""
from copy import deepcopy
import inspect
from pathlib import Path
import textwrap
import nbformat as nbf
from notebook_content import NOTEBOOKS, SETUP

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    # Unlike dedent, cleandoc handles a heading on the opening-quote line.
    return nbf.v4.new_markdown_cell(inspect.cleandoc(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def build(output=None):
    output = Path(output or ROOT / "notebooks")
    output.mkdir(parents=True, exist_ok=True)
    for name, sections in NOTEBOOKS.items():
        cells = [md(sections[0]), code(SETUP)]
        for kind, source in sections[1:]:
            cells.append(md(source) if kind == "md" else code(source))
        notebook = nbf.v4.new_notebook(cells=cells)
        notebook.metadata.kernelspec = {"display_name": "Python 3 (Base Siege)", "language": "python", "name": "python3"}
        notebook.metadata.language_info = {"name": "python", "version": "3.12"}
        for i, cell in enumerate(notebook.cells):
            cell.id = f"cell-{i:03d}"
        nbf.validate(notebook)
        nbf.write(notebook, output / name)
        print(f"{name}: {len(cells)} cells")


if __name__ == "__main__":
    build()
