import ast
from pathlib import Path
import re

import nbformat
from nbconvert.filters import markdown2html

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_valid_readable_and_bounded():
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(paths) == 7
    for path in paths:
        nb = nbformat.read(path, as_version=4)
        nbformat.validate(nb)
        assert len(nb.cells) >= 12
        assert sum(c.cell_type == "markdown" and c.source.startswith("# ") for c in nb.cells) == 1
        for cell in nb.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
                assert cell.execution_count is None
                assert cell.outputs == []
            else:
                # Prose after a first-line title must not become an indented code block.
                unfenced = re.sub(r"(?ms)^~~~.*?^~~~\s*$", "", cell.source)
                assert not re.search(r"(?m)^(?: {4}|\t)\S", unfenced), (path.name, cell.id)
                html = markdown2html(unfenced)
                assert "<pre>" not in html, (path.name, cell.id)
    training = nbformat.read(ROOT / "notebooks/03_maskable_ppo_training.ipynb", as_version=4)
    source = "\n".join(c.source for c in training.cells if c.cell_type == "code")
    assert "RUN_SOFTWARE_SMOKE = False" in source
    assert "subprocess.run" not in source


def test_markdown_first_line_heading_regression():
    import inspect
    source = "# Heading\n\n    Normal paragraph with **bold** text."
    html = markdown2html(inspect.cleandoc(source))
    assert "<p>Normal paragraph with <strong>bold</strong>" in html
    assert "<pre>" not in html
