"""
The prototype is a second implementation of the same game, in JavaScript, and
a second implementation means a second copy of every number.

That copy has already drifted once. During the Stage 5 defence rebalance the
Python stats moved — mortar range, fire rate, splash falloff, the layered core
contents — and the prototype was left on the old values for long enough that
it would have been balanced against a game the simulator no longer played.

These tests read the constants straight out of the HTML and compare them with
config.py. They do not test behaviour: the two implementations are allowed to
differ in animation, input and rendering. They test that the NUMBERS agree,
which is the only thing a balance tool has to get right.

If one of these fails, run:  python tools/sync_prototype.py
"""

import json
import re
from pathlib import Path

import pytest

from env import config as C

PROTOTYPE = Path(__file__).resolve().parent.parent / "prototype" / "base_siege_game.html"


def _js_block(name: str) -> str:
    """Pull `const NAME={...};` out of the HTML."""
    source = PROTOTYPE.read_text()
    match = re.search(rf"const {name}=\{{(.*?)\n\}};", source, re.S)
    assert match, f"could not find const {name} in the prototype"
    return match.group(1)


def _js_object(name: str) -> dict:
    """
    Parse a flat JS object literal of the form `key:{a:1,b:'x'},...`.

    Deliberately a small hand-rolled parser rather than anything clever: the
    block it reads is a fixed table maintained by sync_prototype.py, and a
    parser that silently copes with arbitrary JS would also silently cope with
    a table that has quietly changed shape.
    """
    body = _js_block(name)
    body = re.sub(r"//[^\n]*", "", body)
    entries = {}
    for key, fields in re.findall(r"(\w+)\s*:\s*\{([^}]*)\}", body):
        record = {}
        for field, value in re.findall(r"(\w+)\s*:\s*('[^']*'|[\d.]+)", fields):
            record[field] = value.strip("'") if value.startswith("'") else float(value)
        entries[key] = record
    assert entries, f"parsed nothing out of const {name}"
    return entries


@pytest.fixture(scope="module")
def prototype_buildings() -> dict:
    return _js_object("BUILDINGS")


@pytest.fixture(scope="module")
def prototype_troops() -> dict:
    return _js_object("TROOPS")


def test_the_prototype_exists():
    assert PROTOTYPE.exists(), f"{PROTOTYPE} is missing"


def test_building_stats_match_config(prototype_buildings):
    """hp, damage per shot, range, minimum range, cooldown and score value."""
    mapping = {
        "hp": "hp",
        "dmg": "damage_per_shot",
        "rng": "range",
        "min": "min_range",
        "cd": "attack_cooldown",
        "val": "value",
    }
    assert set(prototype_buildings) == set(C.BUILDING_TYPES)

    for btype, fields in prototype_buildings.items():
        for js_key, py_key in mapping.items():
            assert fields[js_key] == pytest.approx(
                C.BUILDING_STATS[btype][py_key]
            ), f"{btype}.{js_key}: prototype {fields[js_key]}, config {C.BUILDING_STATS[btype][py_key]}"


def test_splash_radius_matches_config(prototype_buildings):
    assert prototype_buildings["mortar"]["splash"] == pytest.approx(
        C.BUILDING_STATS["mortar"]["splash_radius"]
    )
    for btype, fields in prototype_buildings.items():
        if btype != "mortar":
            assert "splash" not in fields, f"{btype} has splash in the prototype only"


def test_troop_stats_match_config(prototype_troops):
    mapping = {
        "hp": "hp",
        "dmg": "damage_per_tick",
        "rng": "range",
        "spd": "speed",
        "cost": "cost",
    }
    assert set(prototype_troops) == set(C.TROOP_TYPES)

    for troop_type, fields in prototype_troops.items():
        for js_key, py_key in mapping.items():
            assert fields[js_key] == pytest.approx(
                C.TROOP_STATS[troop_type][py_key]
            ), f"{troop_type}.{js_key}"


def test_scalar_constants_match_config():
    source = PROTOTYPE.read_text()

    def scalar(name):
        match = re.search(rf"const {name}=([\d.]+)", source)
        assert match, f"const {name} not found"
        return float(match.group(1))

    assert scalar("SPLASH_FALLOFF") == pytest.approx(C.SPLASH_FALLOFF)

    grid = re.search(r"const GRID=(\d+), BUDGET=(\d+), MAX_ARMY=(\d+), TICK_LIMIT=(\d+)", source)
    assert grid, "the top-line constants block has changed shape"
    assert int(grid.group(1)) == C.GRID_SIZE
    assert int(grid.group(2)) == C.RESOURCE_BUDGET
    assert int(grid.group(3)) == C.MAX_ARMY_SIZE
    assert int(grid.group(4)) == C.BATTLE_TICK_LIMIT


def test_building_counts_and_core_contents_match_config():
    source = PROTOTYPE.read_text()

    counts = re.search(r"const COUNTS=\{([^}]*)\}", source).group(1)
    parsed = {k: int(v) for k, v in re.findall(r"(\w+)\s*:\s*(\d+)", counts)}
    assert parsed == C.BUILDING_COUNTS

    core = re.search(r"const CORE_CONTENTS=\[([^\]]*)\]", source).group(1)
    assert [x.strip().strip("'\"") for x in core.split(",")] == list(
        C.LAYERED_CORE_CONTENTS
    )


def test_geometry_matches_config():
    source = re.sub(r"//[^\n]*", "", PROTOTYPE.read_text())
    ring = re.search(r"const RING=\[([^\]]*)\]", source).group(1)
    core = re.search(r"CORE=\[([^\]]*)\]", source).group(1)

    assert [int(x) for x in ring.split(",")] == list(C.CENTRED_RING_BOX)
    assert [int(x) for x in core.split(",")] == list(C.LAYERED_CORE_BOX)


def test_total_value_is_derived_not_hardcoded():
    """
    A literal 155 silently mis-scales the destruction percentage the moment a
    building value or count changes. It must be computed from the table.
    """
    source = PROTOTYPE.read_text()
    assert not re.search(r"const TOTAL_VALUE\s*=\s*\d+\s*;", source), (
        "TOTAL_VALUE is hardcoded in the prototype"
    )
    assert "reduce" in re.search(
        r"const TOTAL_VALUE=(.*?);", source, re.S
    ).group(1), "TOTAL_VALUE is not derived from COUNTS and BUILDINGS"


def test_the_prototype_implements_the_cooldown_and_falloff_rules():
    """
    Numbers agreeing is not enough if the prototype ignores them. These two
    mechanics were added to Python during the Stage 5 rebalance; the prototype
    has to honour them or a mortar there is still a continuous beam.
    """
    source = PROTOTYPE.read_text()
    assert "b.cool" in source, "the prototype has no cooldown state"
    assert re.search(r"if\(b\.cool>0\)", source), "cooldown is never checked before firing"
    assert "SPLASH_FALLOFF" in source.split("const SPLASH_FALLOFF")[1], (
        "SPLASH_FALLOFF is declared but never used"
    )
