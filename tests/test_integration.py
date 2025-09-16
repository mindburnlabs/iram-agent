"""
Integration tests for IRAM components.
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.mcp_server import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_readiness_probe(client):
    """Test the readiness probe endpoint."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert "ready" in response.json()


# Add more integration tests for your application's endpoints here
