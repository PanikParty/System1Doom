import json
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_json():
    def load(name):
        return json.loads((FIX / name).read_text())
    return load
