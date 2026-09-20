import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Deterministic tile resources across the whole suite.
os.environ.setdefault("MAP_SEED", "test-seed")

from world.environment import build_default_world  # noqa: E402


@pytest.fixture
def world():
    return build_default_world(seed="test-seed")


@pytest.fixture
def save_dir(tmp_path):
    directory = tmp_path / "saves"
    directory.mkdir()
    return directory
