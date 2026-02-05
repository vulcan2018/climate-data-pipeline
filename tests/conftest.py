"""Pytest configuration and fixtures."""

import numpy as np
import pytest
import xarray as xr


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    """Create a sample climate dataset for testing."""
    # Create time coordinate (daily data for 2 years)
    times = np.arange("2020-01-01", "2022-01-01", dtype="datetime64[D]")

    # Create spatial coordinates
    lats = np.arange(-90, 91, 10.0)  # 19 points
    lons = np.arange(-180, 180, 10.0)  # 36 points

    # Create temperature data with realistic patterns
    nt, ny, nx = len(times), len(lats), len(lons)

    # Base temperature: warmer at equator, colder at poles
    lat_effect = 288 - 40 * np.abs(lats) / 90
    lat_grid = np.broadcast_to(lat_effect[:, np.newaxis], (ny, nx))

    # Seasonal cycle
    day_of_year = np.arange(nt) % 365
    seasonal = 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)

    # Combine effects
    temp_data = np.zeros((nt, ny, nx), dtype=np.float32)
    for t in range(nt):
        temp_data[t] = lat_grid + seasonal[t] + np.random.randn(ny, nx) * 2

    # Create dataset
    ds = xr.Dataset(
        {
            "temperature": (["time", "lat", "lon"], temp_data),
            "precipitation": (
                ["time", "lat", "lon"],
                np.random.exponential(0.001, (nt, ny, nx)).astype(np.float32),
            ),
        },
        coords={
            "time": times,
            "lat": lats,
            "lon": lons,
        },
        attrs={
            "title": "Sample Climate Dataset",
            "source": "Generated for testing",
        },
    )

    # Add variable attributes
    ds["temperature"].attrs = {
        "long_name": "2m Temperature",
        "units": "K",
        "standard_name": "air_temperature",
    }
    ds["precipitation"].attrs = {
        "long_name": "Total Precipitation",
        "units": "m",
        "standard_name": "precipitation_amount",
    }

    return ds


@pytest.fixture
def sample_netcdf(tmp_path, sample_dataset) -> str:
    """Create a sample NetCDF file for testing."""
    filepath = tmp_path / "sample_data.nc"
    sample_dataset.to_netcdf(filepath)
    return str(filepath)


@pytest.fixture
def small_dataset() -> xr.Dataset:
    """Create a small dataset for quick tests."""
    times = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")
    lats = np.array([-10.0, 0.0, 10.0])
    lons = np.array([-10.0, 0.0, 10.0])

    temp = 280 + np.random.randn(10, 3, 3) * 5

    return xr.Dataset(
        {"temperature": (["time", "lat", "lon"], temp)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
