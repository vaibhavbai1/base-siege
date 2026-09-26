"""
Watch one episode. The Stage 5 "After" instruction, and the reason the
renderer was pulled forward from Stage 14.

    python tools/watch_episode.py

Renders every 20th tick into a grid of subplots plus one full-size debug
frame. Look for, in this order:

    troops stuck against walls          pathfinding or movement bug
    attack lines longer than the range  the wrong distance metric
    tanks walking past defences         target rule wired to the wrong pool
    a mortar firing inside distance 2   min_range off by one
    two troops on one tile              expected, not a bug

Twenty minutes here is worth a day in a debugger.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import combat as B          # noqa: E402
from env import config as C          # noqa: E402
from env import generator as G       # noqa: E402
from env.entities import Troop       # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "figures"
OUT.mkdir(exist_ok=True)

SEED, MOTIF = 42, "closed_ring"
ARMY = [
    ("breaker", (0, 7)), ("breaker", (0, 8)),
    ("tank", (0, 6)), ("tank", (0, 9)),
    ("melee", (0, 5)), ("melee", (0, 10)), ("melee", (1, 0)),
    ("ranged", (0, 4)), ("ranged", (0, 11)),
]


def main() -> None:
    spec = G.generate_base(SEED, MOTIF)
    state = B.BattleState.from_spec(spec)
    for troop_type, tile in ARMY:
        state.troops.append(Troop.spawn(troop_type, tile, state.next_entity_id))
        state.next_entity_id += 1
        state.troops_total += 1
    state.spent = sum(C.TROOP_COSTS[t] for t, _ in ARMY)

    from env import render as R

    frames = [B.record_snapshot(state)]
    while not state.is_finished:
        B.step_tick(state)
        if state.tick % 20 == 0:
            frames.append(B.record_snapshot(state))
    frames.append(B.record_snapshot(state))

    print(f"seed {SEED} · {MOTIF}")
    print(f"  ended at tick {state.tick} ({state.finish_reason})")
    print(f"  destroyed {state.destroyed_value}/{state.total_base_value} "
          f"= {state.destroyed_fraction:.1%}")
    print(f"  troops alive {state.troops_alive}/{state.troops_total}")
    print(f"  milestones {state.milestones}")
    print(f"  path cache hit rate {state.path_cache.hit_rate:.1%}")

    cols = 5
    rows = (len(frames) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.2 * rows))
    for ax, frame in zip(axes.ravel(), frames):
        R.draw_snapshot(frame, ax, style="debug", show_ranges=False,
                        show_paths=True, show_damage=True, show_hud=False)
        ax.set_title(f"tick {frame['tick']}  ·  {frame['destroyed_fraction']:.0%}",
                     fontsize=9)
    for ax in axes.ravel()[len(frames):]:
        ax.axis("off")
    fig.suptitle(f"Base Siege · episode walkthrough · seed {SEED} · {MOTIF}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "episode_walkthrough.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote {OUT / 'episode_walkthrough.png'}")

    mid = frames[len(frames) // 3]
    fig, ax = plt.subplots(figsize=(8, 8))
    R.draw_snapshot(mid, ax, style="debug", show_ranges=True, show_paths=True,
                    show_damage=True, show_hud=True)
    fig.tight_layout()
    fig.savefig(OUT / "debug_frame_example.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT / 'debug_frame_example.png'}")


if __name__ == "__main__":
    main()
