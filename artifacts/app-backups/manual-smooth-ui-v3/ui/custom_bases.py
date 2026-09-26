"""Validated, portable demo layouts. Never writes an experimental base pool."""
from collections import Counter
import hashlib
import json
from env import config as C
from env.generator import BaseSpec

SCHEMA = 'base-siege-custom-base-v1'
MAX_BYTES = 32_768


def validate_layout(data):
    if not isinstance(data, dict) or data.get('schema') != SCHEMA:
        raise ValueError('Choose a Base Siege custom-base JSON export.')
    name = data.get('name', 'My custom base')
    if not isinstance(name, str) or not name.strip() or len(name) > 80:
        raise ValueError('Give the base a name of 1–80 characters.')
    buildings, walls = data.get('buildings'), data.get('walls')
    if not isinstance(buildings, list) or not isinstance(walls, list):
        raise ValueError('Buildings and walls must be lists.')
    if len(buildings) != C.N_SCORING_BUILDINGS or len(walls) > (C.GRID_SIZE-2)**2 - C.N_SCORING_BUILDINGS:
        raise ValueError('Place all 10 buildings and keep walls on free interior tiles.')
    occupied, counts, clean_buildings, clean_walls = set(), Counter(), [], []
    def position(values):
        if len(values) != 2 or any(type(v) is not int or not 1 <= v < C.GRID_SIZE-1 for v in values):
            raise ValueError('Place buildings and walls inside the grass perimeter (rows/columns 1–14).')
        tile = tuple(values)
        if tile in occupied:
            raise ValueError('Buildings and walls cannot overlap.')
        occupied.add(tile)
        return tile
    for item in buildings:
        if (not isinstance(item, (list, tuple)) or len(item) != 3
                or not isinstance(item[0], str) or item[0] not in C.BUILDING_COUNTS):
            raise ValueError('Unknown building type or malformed building.')
        kind = item[0]
        tile = position(item[1:])
        counts[kind] += 1
        clean_buildings.append((kind, *tile))
    if counts != Counter(C.BUILDING_COUNTS):
        raise ValueError('Use 1 Town Hall, 3 cannons, 2 mortars and 4 storages.')
    for item in walls:
        if not isinstance(item, (list, tuple)):
            raise ValueError('Malformed wall position.')
        clean_walls.append(position(item))
    return {'schema': SCHEMA, 'name': name.strip(),
            'buildings': [list(b) for b in sorted(clean_buildings)],
            'walls': [list(w) for w in sorted(clean_walls)]}


def parse_layout(raw):
    if len(raw) > MAX_BYTES:
        raise ValueError('The custom-base file is too large (maximum 32 KB).')
    try:
        return validate_layout(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError('The file is not valid JSON.') from exc


def layout_from_spec(spec, name='My custom base'):
    return validate_layout({'schema': SCHEMA, 'name': name,
                            'buildings': list(spec.buildings), 'walls': list(spec.walls)})


def layout_id(layout):
    clean = validate_layout(layout)
    geometry = {k: clean[k] for k in ('buildings', 'walls')}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def custom_spec(layout):
    clean = validate_layout(layout)
    # Seed 0 is a demo placeholder; motif + content digest identify the map.
    return BaseSpec(seed=0, motif='custom',
                    buildings=tuple(tuple(b) for b in clean['buildings']),
                    walls=tuple(tuple(w) for w in clean['walls']))


def validate_custom_spec(spec):
    if spec.seed != 0 or spec.motif != 'custom':
        raise ValueError('Custom demo maps must use the custom structure and seed 0.')
    return custom_spec(layout_from_spec(spec))
