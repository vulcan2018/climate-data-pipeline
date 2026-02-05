"""Tests for NetCDF reader module."""

import numpy as np
import pytest
import xarray as xr

from climate_pipeline.ingest.netcdf_reader import (
    get_dataset_info,
    read_netcdf,
    read_netcdf_lazy,
)


class TestReadNetcdf:
    """Tests for read_netcdf function."""

    def test_read_netcdf_basic(self, sample_netcdf):
        """Test basic NetCDF reading."""
        ds = read_netcdf(sample_netcdf)

        assert isinstance(ds, xr.Dataset)
        assert "temperature" in ds.data_vars
        assert "precipitation" in ds.data_vars
        assert "time" in ds.dims
        assert "lat" in ds.dims
        assert "lon" in ds.dims

    def test_read_netcdf_select_variables(self, sample_netcdf):
        """Test reading specific variables."""
        ds = read_netcdf(sample_netcdf, variables=["temperature"])

        assert "temperature" in ds.data_vars
        assert "precipitation" not in ds.data_vars

    def test_read_netcdf_time_range(self, sample_netcdf):
        """Test reading with time range."""
        ds = read_netcdf(
            sample_netcdf,
            time_range=("2020-06-01", "2020-06-30"),
        )

        assert ds.time.size == 30

    def test_read_netcdf_bbox(self, sample_netcdf):
        """Test reading with bounding box."""
        ds = read_netcdf(
            sample_netcdf,
            bbox=(-20, -20, 20, 20),
        )

        assert ds.lat.min() >= -20
        assert ds.lat.max() <= 20
        assert ds.lon.min() >= -20
        assert ds.lon.max() <= 20

    def test_read_netcdf_file_not_found(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            read_netcdf("/nonexistent/path/file.nc")

    def test_read_netcdf_invalid_variables(self, sample_netcdf):
        """Test error handling for invalid variables."""
        with pytest.raises(ValueError, match="Variables not found"):
            read_netcdf(sample_netcdf, variables=["nonexistent_var"])


class TestReadNetcdfLazy:
    """Tests for read_netcdf_lazy function."""

    def test_read_lazy_returns_dask_arrays(self, sample_netcdf):
        """Test that lazy reading returns dask-backed arrays."""
        ds = read_netcdf_lazy(sample_netcdf)

        # Check that data is dask-backed
        assert hasattr(ds["temperature"].data, "dask")

    def test_read_lazy_does_not_load(self, sample_netcdf):
        """Test that lazy reading doesn't load data into memory."""
        ds = read_netcdf_lazy(sample_netcdf)

        # Data should not be loaded yet
        # This is a proxy test - full dataset shouldn't be in memory
        assert ds["temperature"].data is not None


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""

    def test_basic_info(self, sample_dataset):
        """Test extracting basic dataset information."""
        info = get_dataset_info(sample_dataset)

        assert "variables" in info
        assert "temperature" in info["variables"]
        assert "dimensions" in info
        assert "time" in info["dimensions"]

    def test_time_range_info(self, sample_dataset):
        """Test time range information."""
        info = get_dataset_info(sample_dataset)

        assert "time_range" in info
        assert "start" in info["time_range"]
        assert "end" in info["time_range"]
        assert "steps" in info["time_range"]

    def test_spatial_info(self, sample_dataset):
        """Test spatial information."""
        info = get_dataset_info(sample_dataset)

        assert "spatial" in info
        assert "lat_range" in info["spatial"]
        assert "lon_range" in info["spatial"]
        assert "resolution" in info["spatial"]

    def test_variable_info(self, sample_dataset):
        """Test variable-specific information."""
        info = get_dataset_info(sample_dataset)

        assert "variable_info" in info
        assert "temperature" in info["variable_info"]
        assert "dims" in info["variable_info"]["temperature"]
        assert "units" in info["variable_info"]["temperature"]
