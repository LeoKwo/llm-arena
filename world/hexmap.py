"""Hex-grid layout helpers for the LLM Arena world.

The board is generated as a rectangular grid of offset coordinates
(``col`` increases east, ``row`` increases south) and converted to axial
``(q, r)`` coordinates for adjacency/distance maths. Hexes are drawn pointy-top,
so the board as a whole looks like a regular rectangle.

Pointy-top hexes have no direct north/south neighbour; the six directions are
``E, SE, SW, W, NW, NE``.
"""

from __future__ import annotations

# Direction name -> axial delta (pointy-top layout).
DIRECTIONS = {
    "E": (1, 0),
    "SE": (0, 1),
    "SW": (-1, 1),
    "W": (-1, 0),
    "NW": (0, -1),
    "NE": (1, -1),
}

# Cities on an offset grid (col = east, row = south) roughly matching Europe.
CITY_OFFSET = {
    "Lisbon": (0, 10),
    "Madrid": (3, 10),
    "Barcelona": (5, 10),
    "Bordeaux": (2, 7),
    "Paris": (5, 6),
    "Brussels": (6, 5),
    "Amsterdam": (6, 3),
    "London": (3, 4),
    "Dublin": (1, 3),
    "Oslo": (8, 0),
    "Stockholm": (11, 0),
    "Copenhagen": (10, 2),
    "Berlin": (10, 4),
    "Munich": (9, 6),
    "Milan": (8, 8),
    "Rome": (10, 9),
    "Warsaw": (13, 4),
    "Vienna": (12, 6),
    "Budapest": (14, 7),
    "Belgrade": (14, 9),
    "Bucharest": (16, 8),
    "Athens": (14, 11),
    "Istanbul": (18, 10),
    "Kiev": (16, 6),
    "Moscow": (18, 3),
    "Leningrad": (16, 1),
}

# Faction -> capital city.
CAPITALS = {
    "Germany": "Berlin",
    "France": "Paris",
    "United Kingdom": "London",
    "Soviet Union": "Moscow",
}

CAPITAL_RESOURCES = 100.0
CITY_RESOURCE_RANGE = (30.0, 80.0)
TILE_RESOURCE_RANGE = (5.0, 25.0)


def offset_to_axial(col: int, row: int) -> tuple[int, int]:
    """Convert odd-r offset coordinates to axial coordinates."""
    q = col - (row - (row & 1)) // 2
    return q, row


def axial_to_offset(q: int, r: int) -> tuple[int, int]:
    col = q + (r - (r & 1)) // 2
    return col, r


CITY_COORDS = {
    name: offset_to_axial(col, row) for name, (col, row) in CITY_OFFSET.items()
}


def neighbors(q: int, r: int) -> dict[str, tuple[int, int]]:
    return {name: (q + dq, r + dr) for name, (dq, dr) in DIRECTIONS.items()}


def to_cube(q: int, r: int) -> tuple[int, int, int]:
    return q, -q - r, r


def hex_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    ax, ay, az = to_cube(*a)
    bx, by, bz = to_cube(*b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def direction_toward(src: tuple[int, int], dst: tuple[int, int]) -> str:
    """Return the hex direction that gets closest to ``dst`` from ``src``."""
    best = "E"
    best_distance = None
    for name, (dq, dr) in DIRECTIONS.items():
        candidate = (src[0] + dq, src[1] + dr)
        distance = hex_distance(candidate, dst)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = name
    return best


def board_coords() -> set[tuple[int, int]]:
    """Rectangle of offset coordinates (padded by 1) converted to axial."""
    cols = [col for col, _ in CITY_OFFSET.values()]
    rows = [row for _, row in CITY_OFFSET.values()]
    coords: set[tuple[int, int]] = set()
    # Pad rows a little more than columns so the board reads as a balanced
    # rectangle rather than a very wide strip.
    for row in range(min(rows) - 2, max(rows) + 3):
        for col in range(min(cols) - 1, max(cols) + 2):
            coords.add(offset_to_axial(col, row))
    return coords
