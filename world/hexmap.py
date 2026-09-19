"""Hex-grid layout helpers for the LLM Arena world.

The board uses axial coordinates ``(q, r)`` with six neighbours. North is
``(0, -1)`` and rows increase towards the south-west, so the manually placed
cities below roughly match the geography of Europe.
"""

from __future__ import annotations

# Direction name -> axial delta.
DIRECTIONS = {
    "N": (0, -1),
    "NE": (1, -1),
    "SE": (1, 0),
    "S": (0, 1),
    "SW": (-1, 1),
    "NW": (-1, 0),
}

# Approximate European city positions on the board (axial q, r).
CITY_COORDS = {
    "Stockholm": (4, -3),
    "Moscow": (10, -3),
    "London": (-3, -1),
    "Amsterdam": (0, -1),
    "Berlin": (2, -2),
    "Warsaw": (5, -2),
    "Paris": (-1, 1),
    "Vienna": (3, 1),
    "Madrid": (-4, 4),
    "Lisbon": (-6, 4),
    "Rome": (1, 4),
    "Athens": (3, 6),
}

# Faction -> capital city.
CAPITALS = {
    "Germany": "Berlin",
    "France": "Paris",
    "United Kingdom": "London",
}

CAPITAL_RESOURCES = 100.0
CITY_RESOURCE_RANGE = (30.0, 80.0)
TILE_RESOURCE_RANGE = (5.0, 25.0)


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
    best = "N"
    best_distance = None
    for name, (dq, dr) in DIRECTIONS.items():
        candidate = (src[0] + dq, src[1] + dr)
        distance = hex_distance(candidate, dst)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = name
    return best


def board_coords() -> set[tuple[int, int]]:
    """All hex coordinates that make up the board (a rectangle around cities)."""
    qs = [q for q, _ in CITY_COORDS.values()]
    rs = [r for _, r in CITY_COORDS.values()]
    coords: set[tuple[int, int]] = set()
    for q in range(min(qs) - 1, max(qs) + 2):
        for r in range(min(rs) - 1, max(rs) + 2):
            coords.add((q, r))
    return coords
