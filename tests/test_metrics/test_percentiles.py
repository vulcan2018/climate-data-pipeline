"""Tests for percentiles module."""

import numpy as np
import pytest

from climate_pipeline.metrics.percentiles import (
    compute_climatological_percentiles,
    compute_exceedance_frequency,
    compute_return_periods,
)


class TestComputeClimatologicalPercentiles:
    """Tests for compute_climatological_percentiles function."""

    def test_default_percentiles(self, sample_dataset):
        """Test computation with default percentiles."""
        result = compute_climatological_percentiles(sample_dataset)

        assert "percentile" in result.dims
        assert list(result.percentile.values) == [10, 25, 50, 75, 90, 95, 99]

    def test_custom_percentiles(self, sample_dataset):
        """Test computation with custom percentiles."""
        result = compute_climatological_percentiles(
            sample_dataset,
            percentiles=[5, 50, 95],
        )

        assert list(result.percentile.values) == [5, 50, 95]

    def test_monthly_groupby(self, sample_dataset):
        """Test percentiles grouped by month."""
        result = compute_climatological_percentiles(
            sample_dataset,
            groupby="month",
        )

        assert "month" in result.dims
        assert result.month.size == 12

    def test_no_groupby(self, sample_dataset):
        """Test percentiles without grouping."""
        result = compute_climatological_percentiles(
            sample_dataset,
            groupby=None,
        )

        # Should not have month dimension
        assert "month" not in result.dims

    def test_percentile_ordering(self, sample_dataset):
        """Test that percentile values are properly ordered."""
        result = compute_climatological_percentiles(
            sample_dataset,
            percentiles=[10, 50, 90],
        )

        # P10 should be less than P50, which should be less than P90
        p10 = float(result["temperature"].sel(percentile=10).mean())
        p50 = float(result["temperature"].sel(percentile=50).mean())
        p90 = float(result["temperature"].sel(percentile=90).mean())

        assert p10 < p50 < p90


class TestComputeExceedanceFrequency:
    """Tests for compute_exceedance_frequency function."""

    def test_exceedance_frequency(self, sample_dataset):
        """Test exceedance frequency computation."""
        thresholds = compute_climatological_percentiles(sample_dataset)
        result = compute_exceedance_frequency(
            sample_dataset,
            thresholds,
            percentile=90,
        )

        # Frequency should be between 0 and 1
        assert float(result["temperature"].min()) >= 0
        assert float(result["temperature"].max()) <= 1

        # For 90th percentile, expect ~10% exceedance
        mean_freq = float(result["temperature"].mean())
        assert 0.05 < mean_freq < 0.20


class TestComputeReturnPeriods:
    """Tests for compute_return_periods function."""

    def test_default_return_periods(self, sample_dataset):
        """Test return periods with default values."""
        result = compute_return_periods(sample_dataset)

        assert "return_period" in result.dims
        assert list(result.return_period.values) == [2, 5, 10, 25, 50, 100]

    def test_return_period_ordering(self, sample_dataset):
        """Test that return period values increase with period."""
        result = compute_return_periods(sample_dataset)

        # Higher return period = more extreme value
        rp2 = float(result["temperature"].sel(return_period=2).mean())
        rp100 = float(result["temperature"].sel(return_period=100).mean())

        assert rp100 > rp2
