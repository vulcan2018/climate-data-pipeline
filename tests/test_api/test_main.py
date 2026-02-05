"""Tests for FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from climate_pipeline.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test that root returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_readiness_check(self, client):
        """Test readiness check."""
        response = client.get("/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "checks" in data

    def test_liveness_check(self, client):
        """Test liveness check."""
        response = client.get("/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"


class TestDataEndpoints:
    """Tests for data access endpoints."""

    def test_list_datasets(self, client):
        """Test listing datasets."""
        response = client.get("/api/v1/data/datasets")

        assert response.status_code == 200
        data = response.json()
        assert "datasets" in data
        assert "count" in data
        assert len(data["datasets"]) > 0

    def test_get_dataset(self, client):
        """Test getting specific dataset."""
        response = client.get("/api/v1/data/datasets/era5-t2m")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "era5-t2m"
        assert "available_metrics" in data

    def test_get_nonexistent_dataset(self, client):
        """Test 404 for nonexistent dataset."""
        response = client.get("/api/v1/data/datasets/nonexistent")

        assert response.status_code == 404

    def test_get_point_data(self, client):
        """Test point data extraction."""
        response = client.get(
            "/api/v1/data/datasets/era5-t2m/point",
            params={"lat": 51.5, "lon": -0.1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["location"]["lat"] == 51.5
        assert "data" in data

    def test_get_region_data(self, client):
        """Test region data extraction."""
        response = client.get(
            "/api/v1/data/datasets/era5-t2m/region",
            params={"west": -10, "south": 35, "east": 30, "north": 60},
        )

        assert response.status_code == 200
        data = response.json()
        assert "grid" in data


class TestMetricsEndpoints:
    """Tests for metrics endpoints."""

    def test_temporal_metrics(self, client):
        """Test temporal metrics computation."""
        response = client.get(
            "/api/v1/metrics/temporal/era5-t2m",
            params={"metric": "monthly", "lat": 51.5, "lon": -0.1},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "monthly_mean"
        assert "values" in data

    def test_percentiles(self, client):
        """Test percentile computation."""
        response = client.get(
            "/api/v1/metrics/percentiles/era5-t2m",
            params={"lat": 51.5, "lon": -0.1},
        )

        assert response.status_code == 200
        data = response.json()
        assert "percentiles" in data
        assert "values" in data

    def test_trend(self, client):
        """Test trend computation."""
        response = client.get(
            "/api/v1/metrics/trend/era5-t2m",
            params={"lat": 51.5, "lon": -0.1},
        )

        assert response.status_code == 200
        data = response.json()
        assert "trend" in data
        assert "slope" in data["trend"]

    def test_anomaly(self, client):
        """Test anomaly computation."""
        response = client.get(
            "/api/v1/metrics/anomaly/era5-t2m",
            params={"lat": 51.5, "lon": -0.1, "time": "2023-07-15"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "anomaly" in data


class TestApiInfo:
    """Tests for API info endpoint."""

    def test_api_info(self, client):
        """Test API info endpoint."""
        response = client.get("/api/v1/info")

        assert response.status_code == 200
        data = response.json()
        assert "capabilities" in data
        assert "processing" in data
        assert "limits" in data
