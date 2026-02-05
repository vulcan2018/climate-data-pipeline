"""Tests for anomalies module."""

import numpy as np
import pytest

from climate_pipeline.metrics.anomalies import (
    classify_anomaly_severity,
    compute_anomaly,
    compute_standardized_anomaly,
)


class TestComputeAnomaly:
    """Tests for compute_anomaly function."""

    def test_basic_anomaly(self, sample_dataset):
        """Test basic anomaly computation."""
        result = compute_anomaly(sample_dataset)

        assert "temperature" in result.data_vars
        assert result.time.size == sample_dataset.time.size

    def test_anomaly_mean_near_zero(self, sample_dataset):
        """Test that anomaly mean is near zero."""
        result = compute_anomaly(sample_dataset)

        # Mean anomaly should be close to zero
        mean_anomaly = float(result["temperature"].mean())
        assert abs(mean_anomaly) < 1.0  # Within 1K of zero

    def test_anomaly_with_reference_period(self, sample_dataset):
        """Test anomaly with specific reference period."""
        result = compute_anomaly(
            sample_dataset,
            reference_period=("2020-01-01", "2020-12-31"),
        )

        assert result is not None
        assert "temperature" in result.data_vars


class TestComputeStandardizedAnomaly:
    """Tests for compute_standardized_anomaly function."""

    def test_basic_standardized_anomaly(self, sample_dataset):
        """Test basic standardized anomaly computation."""
        result = compute_standardized_anomaly(sample_dataset)

        assert "temperature" in result.data_vars

    def test_standardized_anomaly_properties(self, sample_dataset):
        """Test that standardized anomaly has expected properties."""
        result = compute_standardized_anomaly(sample_dataset)

        # Mean should be near zero
        mean_z = float(result["temperature"].mean())
        assert abs(mean_z) < 0.5

        # Standard deviation should be near 1
        std_z = float(result["temperature"].std())
        assert 0.5 < std_z < 1.5


class TestClassifyAnomalySeverity:
    """Tests for classify_anomaly_severity function."""

    def test_classification_range(self, sample_dataset):
        """Test that classification values are in expected range."""
        std_anomaly = compute_standardized_anomaly(sample_dataset)
        result = classify_anomaly_severity(std_anomaly)

        severity_var = f"temperature_severity"
        assert severity_var in result.data_vars

        # Values should be in [-3, 3]
        assert int(result[severity_var].min()) >= -3
        assert int(result[severity_var].max()) <= 3

    def test_classification_symmetry(self, sample_dataset):
        """Test that positive and negative extremes are classified."""
        std_anomaly = compute_standardized_anomaly(sample_dataset)
        result = classify_anomaly_severity(std_anomaly)

        severity_var = f"temperature_severity"
        unique_values = np.unique(result[severity_var].values)

        # Should have both positive and negative classifications
        assert any(v < 0 for v in unique_values)
        assert any(v > 0 for v in unique_values)
