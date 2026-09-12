"""
Pytest configuration — wires up an ephemeral SQLite database and disables the
rate limiter for deterministic testing.  Every session starts with a fresh DB
so tests never collide and are safe to run in parallel.
"""

import os
import sys
import pathlib
import tempfile
import pytest

# ── Make sure the project root is on sys.path and is the working directory ──
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Set env vars BEFORE any project imports (these are read at module level)
_tmp = os.path.join(tempfile.mkdtemp(prefix="ids_test_"), "test.db")
os.environ["IDS_DB_PATH"]       = _tmp
os.environ["RATE_LIMIT_PER_MIN"] = "0"        # disable for tests
os.environ["CONFIDENCE_THRESHOLD"] = "55.0"
os.environ["IDS_SENSOR_TOKEN"]  = ""          # auth disabled by default in tests
os.environ["IDS_RETENTION_DAYS"] = "0"        # no pruning by default in tests


@pytest.fixture(scope="session")
def app_module():
    """Import the Flask app (triggers model loading once per test session)."""
    import importlib
    return importlib.import_module("app")


@pytest.fixture(scope="session")
def client(app_module):
    """A reusable Flask test client."""
    return app_module.app.test_client()


@pytest.fixture(autouse=True)
def _reset_rate_limiter(app_module):
    """Ensure the rate limiter starts fresh every test (no 429 bleed)."""
    import time
    app_module.rate_limiter._hits.clear()


@pytest.fixture
def sample_feature_vector():
    """One real feature vector extracted from the dataset (BENIGN class)."""
    import json
    with open(os.path.join("templates", "attack_samples.json")) as f:
        samples = json.load(f)
    return samples["BENIGN"]
