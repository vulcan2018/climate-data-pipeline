"""Tests for temporal metrics module."""

import numpy as np
import pytest

from climate_pipeline.metrics.temporal import (
    compute_annual_mean,
    compute_climatology,
    compute_monthly_mean,
    compute_rolling_mean,
    compute_seasonal_mean,
)


class TestComputeMonthlyMean:
    """Tests for compute_monthly_mean function."""

    def test_basic_monthly_mean(self, sample_dataset):
        """Test basic monthly mean computation."""
        result = compute_monthly_mean(sample_dataset)

        # Should have 24 months (2 years of data)
        assert result.time.size == 24
        assert "temperature" in result.data_vars

    def test_monthly_mean_preserves_spatial(self, sample_dataset):
        """Test that spatial dimensions are preserved."""
        result = compute_monthly_mean(sample_dataset)

        assert result.lat.size == sample_dataset.lat.size
        assert result.lon.size == sample_dataset.lon.size

    def test_monthly_mean_reduces_variance(self, sample_dataset):
        """Test that monthly mean reduces variance."""
        result = compute_monthly_mean(sample_dataset)

        # Monthly mean should have lower variance than daily data
        daily_std = float(sample_dataset["temperature"].std())
        monthly_std = float(result["temperature"].std())
        assert monthly_std < daily_std


class TestComputeSeasonalMean:
    """Tests for compute_seasonal_mean function."""

    def test_basic_seasonal_mean(self, sample_dataset):
        """Test basic seasonal mean computation."""
        result = compute_seasonal_mean(sample_dataset)

        # Should have 8 seasons (2 years)
        assert result.time.size == 8
        assert "temperature" in result.data_vars


class TestComputeAnnualMean:
    """Tests for compute_annual_mean function."""

    def test_basic_annual_mean(self, sample_dataset):
        """Test basic annual mean computation."""
        result = compute_annual_mean(sample_dataset)

        # Should have 2 years
        assert result.time.size == 2
        assert "temperature" in result.data_vars

    def test_annual_mean_values(self, sample_dataset):
        """Test that annual mean values are reasonable."""
        result = compute_annual_mean(sample_dataset)

        # Global mean temperature should be around 280K
        global_mean = float(result["temperature"].mean())
        assert 260 < global_mean < 300


class TestComputeClimatology:
    """Tests for compute_climatology function."""

    def test_monthly_climatology(self, sample_dataset):
        """Test monthly climatology computation."""
        result = compute_climatology(sample_dataset, groupby="month")

        assert result.month.size == 12
        assert "temperature" in result.data_vars

    def test_seasonal_climatology(self, sample_dataset):
        """Test seasonal climatology computation."""
        result = compute_climatology(sample_dataset, groupby="season")

        assert result.season.size == 4

    def test_climatology_reference_period(self, sample_dataset):
        """Test climatology with reference period."""
        result = compute_climatology(
            sample_dataset,
            groupby="month",
            reference_period=("2020-01-01", "2020-12-31"),
        )

        # Should still have 12 months
        assert result.month.size == 12


class TestComputeRollingMean:
    """Tests for compute_rolling_mean function."""

    def test_basic_rolling_mean(self, sample_dataset):
        """Test basic rolling mean computation."""
        result = compute_rolling_mean(sample_dataset, window=30)

        # Should have same time dimension size
        assert result.time.size == sample_dataset.time.size

    def test_rolling_mean_smooths(self, small_dataset):
        """Test that rolling mean smooths the data."""
        result = compute_rolling_mean(small_dataset, window=3)

        # Rolling mean should reduce variance
        original_std = float(small_dataset["temperature"].std())
        smoothed_std = float(result["temperature"].std(skipna=True))
        assert smoothed_std <= original_std
