"""
Rewrite the prototype's constants block from env/config.py.

    python tools/sync_prototype.py            apply
    python tools/sync_prototype.py --check    report drift, change nothing

WHY THIS EXISTS

The browser prototype is an independent implementation of the same game, so
it holds a second copy of every number. That copy drifted during the Stage 5
defence rebalance — mortar range, fire rate, splash falloff and the layered
core contents all moved in Python and not in JavaScript — and a balance tool
that plays a different game from the simulator is worse than no balance tool
at all, because it is confidently wrong.

`config.py` is the single source of truth. This script pushes it into the
region marked BEGIN/END GENERATED in the HTML, and
`tests/test_prototype_parity.py` fails if anyone edits that region by hand.

WHAT IS AND IS NOT GENERATED

Numbers come from config. PRESENTATION does not: colours and the flavour tail
of each troop description are prototype-only and are preserved below, because
the prototype draws stone-textured walls in a different colour from the
report renderer and neither is wrong.

The numeric part of each description IS regenerated. A balance tool that
prints stale stats in its own buy panel while you play is exactly the failure
this file exists to prevent.
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from env import config as C  # noqa: E402

HTML = REPO / "prototype" / "base_siege_game.html"
BEGIN = "// ==== BEGIN GENERATED FROM env/config.py — python tools/sync_prototype.py ====\n"
END = "// ==== END GENERATED ====\n"

# Prototype-only presentation. Not derivable from config, so kept here.
COLOURS = {
    "town_hall": "#6C63C8", "cannon": "#C74E77", "mortar": "#CE5A2E",
    "storage": "#7A8B8E", "wall": "#6b6560",
    "tank": "#1E8F6E", "melee": "#3179C6", "ranged": "#5E8F22",
    "breaker": "#D99425",
}
FLAVOUR = {
    "tank": "slow",
    "melee": "",
    "ranged": "shoots over walls",
    "breaker": "one wall, then dies · fast",
}
TARGET_LABEL = {
    "nearest_defence": "defence",
    "nearest_building": "building",
    "nearest_wall": "wall",
}


def _num(value) -> str:
    """Render a float that is really an integer without a trailing .0."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_block() -> str:
    lines = [BEGIN]
    lines.append(
        f"const GRID={C.GRID_SIZE}, BUDGET={C.RESOURCE_BUDGET}, "
        f"MAX_ARMY={C.MAX_ARMY_SIZE}, TICK_LIMIT={C.BATTLE_TICK_LIMIT};\n\n"
    )

    lines.append("const TROOPS={\n")
    width = max(len(t) for t in C.TROOP_TYPES)
    for troop_type in C.TROOP_TYPES:
        stats = C.TROOP_STATS[troop_type]
        desc = (
            f"{stats['hp']} hp · {stats['damage_per_tick']} dmg "
            f"· rng {_num(stats['range'])}"
        )
        if FLAVOUR.get(troop_type):
            desc += f" · {FLAVOUR[troop_type]}"
        lines.append(
            f"  {troop_type:<{width}} :{{hp:{stats['hp']},"
            f"dmg:{stats['damage_per_tick']},rng:{_num(stats['range'])},"
            f"spd:{stats['speed']},cost:{stats['cost']},"
            f"tgt:'{TARGET_LABEL[stats['target_rule']]}',"
            f"col:'{COLOURS[troop_type]}',\n"
            f"  {'':<{width}}  desc:'{desc}'}},\n"
        )
    lines.append("};\n\n")

    lines.append("// `dmg` is damage per SHOT; `cd` is ticks between shots (1 = every tick).\n")
    lines.append("const BUILDINGS={\n")
    width = max(len(b) for b in C.BUILDING_TYPES)
    for btype in C.BUILDING_TYPES:
        stats = C.BUILDING_STATS[btype]
        splash = (
            f",splash:{_num(stats['splash_radius'])}"
            if stats["splash_radius"] > 0 else ""
        )
        lines.append(
            f"  {btype:<{width}} :{{hp:{stats['hp']},"
            f"dmg:{stats['damage_per_shot']},rng:{_num(stats['range'])},"
            f"min:{_num(stats['min_range'])},cd:{stats['attack_cooldown']},"
            f"val:{stats['value']},col:'{COLOURS[btype]}'{splash}}},\n"
        )
    lines.append("};\n\n")

    counts = ",".join(f"{k}:{v}" for k, v in C.BUILDING_COUNTS.items())
    lines.append(f"const COUNTS={{{counts}}};\n")
    lines.append(f"const SPLASH_FALLOFF={C.SPLASH_FALLOFF};\n")
    lines.append(
        "// Derived, never typed: a hardcoded total silently mis-scales the\n"
        "// destruction percentage the moment a value or count changes.\n"
    )
    lines.append(
        "const TOTAL_VALUE=Object.entries(COUNTS)\n"
        "  .reduce((s,[k,n])=>s+BUILDINGS[k].val*n,0);\n"
    )

    ring = ",".join(str(v) for v in C.CENTRED_RING_BOX)
    core = ",".join(str(v) for v in C.LAYERED_CORE_BOX)
    lines.append(f"const RING=[{ring}], CORE=[{core}];\n")
    contents = ",".join(f"'{b}'" for b in C.LAYERED_CORE_CONTENTS)
    lines.append(f"const CORE_CONTENTS=[{contents}];\n")
    lines.append(END)
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="report drift without writing")
    args = parser.parse_args()

    source = HTML.read_text()
    if BEGIN not in source or END not in source:
        print("ERROR: the generated markers are missing from the prototype.",
              file=sys.stderr)
        return 2

    head, rest = source.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    new_source = head + build_block() + tail

    if new_source == source:
        print("prototype is already in sync with env/config.py")
        return 0

    if args.check:
        print("DRIFT: the prototype does not match env/config.py")
        print("       run: python tools/sync_prototype.py")
        return 1

    HTML.write_text(new_source)
    print(f"synced {HTML.relative_to(REPO)} from env/config.py")
    print("  mortar:", C.BUILDING_STATS["mortar"])
    print("  core:  ", C.LAYERED_CORE_CONTENTS)
    print("\nnow run: pytest -q")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
