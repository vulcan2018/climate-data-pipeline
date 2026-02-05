"""NetCDF file reading with Xarray.

Provides both eager and lazy loading options for climate data files.
Supports CF conventions and automatic coordinate detection.
"""

from pathlib import Path
from typing import Any

import xarray as xr


def read_netcdf(
    path: str | Path,
    variables: list[str] | None = None,
    time_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Read NetCDF file into memory.

    Args:
        path: Path to NetCDF file
        variables: List of variable names to load (None = all)
        time_range: Optional (start, end) time strings for slicing
        bbox: Optional (west, south, east, north) bounding box

    Returns:
        Xarray Dataset loaded into memory
    """
    ds = read_netcdf_lazy(path, variables, time_range, bbox)
    return ds.load()


def read_netcdf_lazy(
    path: str | Path,
    variables: list[str] | None = None,
    time_range: tuple[str, str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> xr.Dataset:
    """Read NetCDF file lazily (dask-backed).

    Args:
        path: Path to NetCDF file
        variables: List of variable names to load (None = all)
        time_range: Optional (start, end) time strings for slicing
        bbox: Optional (west, south, east, north) bounding box

    Returns:
        Xarray Dataset with dask arrays (lazy loading)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {path}")

    # Open with dask chunks for lazy loading
    ds = xr.open_dataset(path, chunks="auto", engine="netcdf4")

    # Select specific variables if requested
    if variables:
        available = list(ds.data_vars)
        missing = [v for v in variables if v not in available]
        if missing:
            raise ValueError(f"Variables not found in dataset: {missing}. Available: {available}")
        ds = ds[variables]

    # Apply time slice if requested
    if time_range:
        start, end = time_range
        time_dim = _detect_time_dim(ds)
        if time_dim:
            ds = ds.sel({time_dim: slice(start, end)})

    # Apply spatial bounding box if requested
    if bbox:
        ds = _apply_bbox(ds, bbox)

    return ds


def _detect_time_dim(ds: xr.Dataset) -> str | None:
    """Detect the time dimension name following CF conventions."""
    time_names = ["time", "t", "Time", "TIME", "date", "datetime"]
    for name in time_names:
        if name in ds.dims:
            return name
    # Check for dimensions with datetime dtype
    for dim in ds.dims:
        if ds[dim].dtype.kind == "M":  # datetime64
            return dim
    return None


def _detect_lat_lon_dims(ds: xr.Dataset) -> tuple[str | None, str | None]:
    """Detect latitude and longitude dimension names."""
    lat_names = ["lat", "latitude", "Lat", "Latitude", "LAT", "LATITUDE", "y"]
    lon_names = ["lon", "longitude", "Lon", "Longitude", "LON", "LONGITUDE", "x"]

    lat_dim = None
    lon_dim = None

    for name in lat_names:
        if name in ds.dims or name in ds.coords:
            lat_dim = name
            break

    for name in lon_names:
        if name in ds.dims or name in ds.coords:
            lon_dim = name
            break

    return lat_dim, lon_dim


def _apply_bbox(
    ds: xr.Dataset, bbox: tuple[float, float, float, float]
) -> xr.Dataset:
    """Apply bounding box selection to dataset.

    Args:
        ds: Input dataset
        bbox: (west, south, east, north) in degrees

    Returns:
        Sliced dataset
    """
    west, south, east, north = bbox
    lat_dim, lon_dim = _detect_lat_lon_dims(ds)

    if lat_dim is None or lon_dim is None:
        raise ValueError("Could not detect lat/lon dimensions for bbox selection")

    # Handle longitude wrapping (0-360 vs -180-180)
    lon_vals = ds[lon_dim].values
    if lon_vals.min() >= 0 and west < 0:
        # Data is 0-360, query is -180-180
        west = west % 360
        east = east % 360

    # Select region
    if lat_dim in ds.dims:
        # Check if latitude is ascending or descending
        if ds[lat_dim].values[0] > ds[lat_dim].values[-1]:
            ds = ds.sel({lat_dim: slice(north, south)})
        else:
            ds = ds.sel({lat_dim: slice(south, north)})

    if lon_dim in ds.dims:
        if east > west:
            ds = ds.sel({lon_dim: slice(west, east)})
        else:
            # Crosses antimeridian
            ds1 = ds.sel({lon_dim: slice(west, None)})
            ds2 = ds.sel({lon_dim: slice(None, east)})
            ds = xr.concat([ds1, ds2], dim=lon_dim)

    return ds


def get_dataset_info(ds: xr.Dataset) -> dict[str, Any]:
    """Extract metadata and summary information from dataset.

    Args:
        ds: Xarray Dataset

    Returns:
        Dictionary with dataset information
    """
    time_dim = _detect_time_dim(ds)
    lat_dim, lon_dim = _detect_lat_lon_dims(ds)

    info: dict[str, Any] = {
        "variables": list(ds.data_vars),
        "dimensions": dict(ds.dims),
        "coordinates": list(ds.coords),
        "attributes": dict(ds.attrs),
    }

    if time_dim and time_dim in ds.coords:
        time_vals = ds[time_dim].values
        info["time_range"] = {
            "start": str(time_vals[0]),
            "end": str(time_vals[-1]),
            "steps": len(time_vals),
        }

    if lat_dim and lon_dim:
        info["spatial"] = {
            "lat_range": [float(ds[lat_dim].min()), float(ds[lat_dim].max())],
            "lon_range": [float(ds[lon_dim].min()), float(ds[lon_dim].max())],
            "resolution": {
                "lat": float(abs(ds[lat_dim].diff(lat_dim).mean())),
                "lon": float(abs(ds[lon_dim].diff(lon_dim).mean())),
            },
        }

    # Variable-specific info
    info["variable_info"] = {}
    for var in ds.data_vars:
        var_info: dict[str, Any] = {
            "dims": list(ds[var].dims),
            "shape": list(ds[var].shape),
            "dtype": str(ds[var].dtype),
        }
        if "units" in ds[var].attrs:
            var_info["units"] = ds[var].attrs["units"]
        if "long_name" in ds[var].attrs:
            var_info["long_name"] = ds[var].attrs["long_name"]
        info["variable_info"][var] = var_info

    return info
