"""
Render the four wall motifs as figures.

This is a PREVIEW TOOL, not part of the environment. It exists so the wall
geometry can be inspected before generator.py is written, and so the report
figure is generated from config.py rather than drawn by hand — a hand-drawn
figure of a 16x16 grid will disagree with the code within a fortnight.

Every coordinate comes from env/config.py. No geometry is hardcoded here.

Stage 3 landed: all geometry now comes from env.generator, so there is
exactly one implementation of the motif templates in the repository. This
file draws them and nothing else.

Usage:
    python tools/preview_motifs.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import config as C  # noqa: E402
from env import generator as G  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# Cell categories, in colour order.
EMPTY, APPROACH, DEPLOY, WALL, INTERIOR, GATE, CORE = range(7)

PALETTE = ListedColormap([
    "#f4f4f2",   # EMPTY     — unused
    "#f4f4f2",   # APPROACH  — open ground between the deploy ring and the walls
    "#a8c4dc",   # DEPLOY    — the 60 perimeter tiles troops can be dropped on
    "#33414f",   # WALL      — 300 HP, passable at a cost, worth 0 points
    "#e8dfc8",   # INTERIOR  — where buildings are placed
    "#d97a56",   # GATE      — wall tile removed by the motif
    "#a8cbbd",   # CORE      — inner enclosed region (layered core only)
])

LEGEND = [
    Patch(facecolor="#a8c4dc", edgecolor="#8fa8bd", label="Deployment tiles (60)"),
    Patch(facecolor="#f4f4f2", edgecolor="#d5d5d0", label="Open approach"),
    Patch(facecolor="#33414f", edgecolor="#33414f", label="Wall (300 HP, 0 pts)"),
    Patch(facecolor="#e8dfc8", edgecolor="#cfc4a8", label="Building interior"),
    Patch(facecolor="#d97a56", edgecolor="#c26440", label="Gate (wall removed)"),
    Patch(facecolor="#a8cbbd", edgecolor="#8db3a3", label="Inner core (Town Hall)"),
]


def _gate_tiles(motif):
    """
    The wall tiles a gate motif removes, derived by difference from the
    closed ring. Not a second implementation of the gate rule — there is
    only one, in env/generator.py, and this asks it what it did.
    """
    if motif in ("ns_gates", "ew_gates"):
        return G.wall_tiles("closed_ring") - G.wall_tiles(motif)
    return set()


def build_grid(motif):
    """Return (category grid, wall tile set, gate tile set) for one motif."""
    walls = G.wall_tiles(motif)
    gates = _gate_tiles(motif)

    compartments = G.placement_compartments(motif)
    if motif == C.TRANSFER_MOTIF:
        core, interior = compartments[0], compartments[1]
    else:
        core, interior = set(), compartments[0]

    deploy = set(C.DEPLOY_TILES)

    grid = np.full((C.GRID_SIZE, C.GRID_SIZE), APPROACH, dtype=int)
    for r, c in interior:
        grid[r, c] = INTERIOR
    for r, c in core:
        grid[r, c] = CORE
    for r, c in walls:
        grid[r, c] = WALL
    for r, c in gates:
        grid[r, c] = GATE
    for r, c in deploy:
        grid[r, c] = DEPLOY

    return grid, walls, gates


def draw(ax, motif, show_ticks=True):
    grid, walls, gates = build_grid(motif)

    ax.imshow(grid, cmap=PALETTE, vmin=0, vmax=len(PALETTE.colors) - 1,
              interpolation="nearest")

    ax.set_xticks(np.arange(-0.5, C.GRID_SIZE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, C.GRID_SIZE, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.8)
    ax.tick_params(which="minor", length=0)

    if show_ticks:
        ax.set_xticks(range(0, C.GRID_SIZE, 3))
        ax.set_yticks(range(0, C.GRID_SIZE, 3))
        ax.tick_params(labelsize=7, colors="#666666", length=2)
        ax.set_xlabel("column", fontsize=7, color="#666666")
        ax.set_ylabel("row", fontsize=7, color="#666666")
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")

    trained = "excluded from training" if motif == C.TRANSFER_MOTIF else "in training set"
    ax.set_title(
        f"{motif}\n{len(walls)} walls"
        + (f", {len(gates)} gate tiles" if gates else "")
        + f"  ·  {trained}",
        fontsize=9, pad=8,
    )
    return len(walls), len(gates)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    written = []

    # Combined 2x2 figure — the one for the report.
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 10))
    for ax, motif in zip(axes.ravel(), C.MOTIFS):
        draw(ax, motif)
    fig.suptitle(
        "Base Siege wall motifs · 16×16 grid · rendered from env/config.py",
        fontsize=12, y=0.975,
    )
    fig.legend(handles=LEGEND, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.055, 1, 0.955))
    path = OUT_DIR / "motifs_all.png"
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)
    written.append(path)

    # One file per motif — for slides.
    for motif in C.MOTIFS:
        fig, ax = plt.subplots(figsize=(5.2, 5.6))
        draw(ax, motif)
        fig.legend(handles=LEGEND, loc="lower center", ncol=2, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.0))
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        path = OUT_DIR / f"motif_{motif}.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        written.append(path)

    for p in written:
        print(f"wrote {p.relative_to(OUT_DIR.parent)}")

    # Console summary — cheap sanity check on the geometry.
    print()
    print(f"{'motif':<14}{'walls':>7}{'gates':>7}{'placeable':>10}  overlaps deploy ring?")
    deploy = set(C.DEPLOY_TILES)
    for motif in C.MOTIFS:
        grid, walls, gates = build_grid(motif)
        placeable = int(((grid == INTERIOR) | (grid == CORE)).sum())
        clash = bool(walls & deploy)
        print(f"{motif:<14}{len(walls):>7}{len(gates):>7}{placeable:>10}"
              f"  {'YES — BUG' if clash else 'no'}")


if __name__ == "__main__":
    main()
