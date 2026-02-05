"""Tests for chunking module."""

import pytest

from climate_pipeline.processing.chunking import (
    determine_optimal_chunks,
    estimate_chunk_memory,
    rechunk_dataset,
    suggest_chunks_for_workflow,
    validate_chunks,
)


class TestDetermineOptimalChunks:
    """Tests for determine_optimal_chunks function."""

    def test_balanced_pattern(self, sample_dataset):
        """Test balanced chunking pattern."""
        chunks = determine_optimal_chunks(
            sample_dataset,
            access_pattern="balanced",
        )

        assert all(v > 0 for v in chunks.values())
        assert "time" in chunks
        assert "lat" in chunks
        assert "lon" in chunks

    def test_timeseries_pattern(self, sample_dataset):
        """Test timeseries chunking pattern."""
        chunks = determine_optimal_chunks(
            sample_dataset,
            access_pattern="timeseries",
        )

        # Should favor larger time chunks
        assert chunks["time"] >= chunks.get("lat", 1)

    def test_spatial_pattern(self, sample_dataset):
        """Test spatial chunking pattern."""
        chunks = determine_optimal_chunks(
            sample_dataset,
            access_pattern="spatial",
        )

        # Should have time=1 for single timestep access
        assert chunks["time"] == 1

    def test_respects_target_size(self, sample_dataset):
        """Test that chunks respect target size."""
        chunks = determine_optimal_chunks(
            sample_dataset,
            target_mb=1.0,
        )

        estimates = estimate_chunk_memory(sample_dataset, chunks)
        # Should be reasonably close to target
        assert estimates["total_per_chunk"] < 5.0  # Some tolerance


class TestRechunkDataset:
    """Tests for rechunk_dataset function."""

    def test_basic_rechunk(self, sample_dataset):
        """Test basic rechunking."""
        chunks = {"time": 100, "lat": 10, "lon": 10}
        result = rechunk_dataset(sample_dataset, chunks)

        assert result["temperature"].chunks is not None


class TestEstimateChunkMemory:
    """Tests for estimate_chunk_memory function."""

    def test_estimate_structure(self, sample_dataset):
        """Test that estimates have expected structure."""
        chunks = {"time": 100, "lat": 10, "lon": 10}
        estimates = estimate_chunk_memory(sample_dataset, chunks)

        assert "temperature" in estimates
        assert "total_per_chunk" in estimates
        assert "total_chunks" in estimates
        assert "total_data_mb" in estimates

    def test_estimate_values(self, sample_dataset):
        """Test that estimates are reasonable."""
        chunks = {"time": 100, "lat": 10, "lon": 10}
        estimates = estimate_chunk_memory(sample_dataset, chunks)

        assert estimates["total_per_chunk"] > 0
        assert estimates["total_data_mb"] > 0


class TestSuggestChunksForWorkflow:
    """Tests for suggest_chunks_for_workflow function."""

    def test_temporal_workflow(self, sample_dataset):
        """Test suggestions for temporal workflow."""
        workflow = ["monthly_mean", "annual_trend", "temporal_aggregation"]
        chunks = suggest_chunks_for_workflow(sample_dataset, workflow)

        # Should favor time-optimized chunks
        assert chunks["time"] >= 1

    def test_spatial_workflow(self, sample_dataset):
        """Test suggestions for spatial workflow."""
        workflow = ["spatial_map", "region_extract", "bbox_slice"]
        chunks = suggest_chunks_for_workflow(sample_dataset, workflow)

        # Should favor spatial-optimized chunks
        assert "lat" in chunks
        assert "lon" in chunks


class TestValidateChunks:
    """Tests for validate_chunks function."""

    def test_valid_chunks(self, sample_dataset):
        """Test validation of valid chunks."""
        chunks = {"time": 100, "lat": 10, "lon": 10}
        warnings = validate_chunks(sample_dataset, chunks)

        # Should have no warnings for reasonable chunks
        assert len(warnings) == 0

    def test_oversized_chunks(self, sample_dataset):
        """Test warning for oversized chunks."""
        chunks = {"time": 10000, "lat": 1000, "lon": 1000}
        warnings = validate_chunks(sample_dataset, chunks)

        # Should warn about large chunks or exceeding dimensions
        assert len(warnings) > 0

    def test_unknown_dimension(self, sample_dataset):
        """Test warning for unknown dimension."""
        chunks = {"time": 100, "unknown_dim": 10}
        warnings = validate_chunks(sample_dataset, chunks)

        assert any("not in dataset" in w for w in warnings)
