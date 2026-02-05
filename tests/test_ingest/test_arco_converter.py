"""Tests for ARCO converter module."""

import pytest

from climate_pipeline.ingest.arco_converter import (
    convert_to_zarr,
    open_zarr,
    optimize_chunks,
)


class TestOptimizeChunks:
    """Tests for optimize_chunks function."""

    def test_balanced_chunks(self, sample_dataset):
        """Test balanced chunking strategy."""
        chunks = optimize_chunks(sample_dataset, access_pattern="balanced")

        assert "time" in chunks
        assert "lat" in chunks
        assert "lon" in chunks
        assert all(v > 0 for v in chunks.values())

    def test_timeseries_chunks(self, sample_dataset):
        """Test timeseries-optimized chunking."""
        chunks = optimize_chunks(sample_dataset, access_pattern="timeseries")

        # Timeseries pattern should have larger time chunks
        assert chunks["time"] > chunks.get("lat", 1)

    def test_spatial_chunks(self, sample_dataset):
        """Test spatial-optimized chunking."""
        chunks = optimize_chunks(sample_dataset, access_pattern="spatial")

        # Spatial pattern should have time=1
        assert chunks["time"] == 1


class TestConvertToZarr:
    """Tests for convert_to_zarr function."""

    def test_basic_conversion(self, sample_dataset, tmp_path):
        """Test basic NetCDF to Zarr conversion."""
        output_path = tmp_path / "output.zarr"

        result_path = convert_to_zarr(sample_dataset, output_path)

        assert result_path.exists()
        assert (result_path / ".zmetadata").exists()

    def test_conversion_with_compression(self, small_dataset, tmp_path):
        """Test conversion with compression."""
        output_path = tmp_path / "compressed.zarr"

        convert_to_zarr(
            small_dataset,
            output_path,
            compression="gzip",
            compression_level=5,
        )

        assert output_path.exists()

    def test_roundtrip(self, small_dataset, tmp_path):
        """Test that data survives roundtrip conversion."""
        output_path = tmp_path / "roundtrip.zarr"

        convert_to_zarr(small_dataset, output_path)
        loaded = open_zarr(output_path)

        # Check data integrity
        assert "temperature" in loaded.data_vars
        assert loaded["temperature"].shape == small_dataset["temperature"].shape

    def test_overwrite_protection(self, small_dataset, tmp_path):
        """Test that overwrite=False prevents overwriting."""
        output_path = tmp_path / "no_overwrite.zarr"

        convert_to_zarr(small_dataset, output_path)

        with pytest.raises(FileExistsError):
            convert_to_zarr(small_dataset, output_path, overwrite=False)

    def test_overwrite_allowed(self, small_dataset, tmp_path):
        """Test that overwrite=True allows overwriting."""
        output_path = tmp_path / "overwrite.zarr"

        convert_to_zarr(small_dataset, output_path)
        convert_to_zarr(small_dataset, output_path, overwrite=True)

        assert output_path.exists()
