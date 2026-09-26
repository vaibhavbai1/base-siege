"""
Render the example snapshot in every style and overlay combination.

Run this to check env/render.py works on your machine before combat.py exists:

    python tools/render_demo.py

Writes to figures/. Delete once real episodes are available — at that point
use draw_episode_strip on recorded snapshots instead.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import render as R  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"


def main():
    OUT.mkdir(exist_ok=True)
    snap = R.example_snapshot()
    written = []

    # 1. The two styles side by side. This is the comparison that decides
    #    which one you use while debugging.
    fig, axes = plt.subplots(1, 2, figsize=(11, 6))
    for ax, style in zip(axes, ("debug", "presentation")):
        R.draw_snapshot(snap, ax, style=style)
        ax.set_title(style, fontsize=11, pad=10)
    fig.suptitle("Same snapshot, both styles", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT / "render_styles.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    written.append(path)

    # 2. Presentation style with every overlay on. The range circles are the
    #    demo money shot: the mortar's blind spot, drawn rather than explained.
    fig, ax = plt.subplots(figsize=(7.5, 8.6))
    R.draw_snapshot(snap, ax, style="presentation",
                    show_ranges=True, show_paths=True, show_damage=True)
    ax.set_title("All overlays — ranges, blind spot, paths, damage",
                 fontsize=11, pad=10)
    fig.legend(handles=R.legend_handles(), loc="lower center", ncol=5,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    path = OUT / "render_overlays.png"
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)
    written.append(path)

    # 3. An episode strip. This is the Stage 5 "watch one episode" tool,
    #    faked here with one repeated frame.
    frames = []
    for tick in range(0, 120, 20):
        frame = dict(snap)
        frame["tick"] = tick
        frames.append(frame)
    fig = R.draw_episode_strip(frames, ncols=3, style="debug")
    path = OUT / "render_episode_strip.png"
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    written.append(path)

    for p in written:
        print(f"wrote {p.relative_to(OUT.parent)}")


if __name__ == "__main__":
    main()
