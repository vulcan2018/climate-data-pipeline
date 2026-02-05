"""Optimal chunking strategies for climate data.

Provides utilities for determining and applying optimal chunk sizes
based on data characteristics and intended access patterns.
"""

from typing import Literal

import numpy as np
import xarray as xr


# Target chunk size in bytes (optimal range: 1-10 MB)
DEFAULT_TARGET_MB = 4


def determine_optimal_chunks(
    ds: xr.Dataset,
    target_mb: float = DEFAULT_TARGET_MB,
    access_pattern: Literal["timeseries", "spatial", "balanced"] = "balanced",
    max_chunks_per_dim: int = 1000,
) -> dict[str, int]:
    """Determine optimal chunk sizes for a dataset.

    Args:
        ds: Input Xarray dataset
        target_mb: Target chunk size in megabytes
        access_pattern:
            - "timeseries": Optimize for extracting time series at points
            - "spatial": Optimize for extracting spatial maps at times
            - "balanced": Balance between access patterns
        max_chunks_per_dim: Maximum number of chunks per dimension

    Returns:
        Dictionary mapping dimension names to chunk sizes
    """
    target_bytes = target_mb * 1024 * 1024

    # Identify dimension types
    dims = _identify_dimensions(ds)
    time_dim = dims.get("time")
    lat_dim = dims.get("lat")
    lon_dim = dims.get("lon")

    # Get representative dtype size
    sample_var = list(ds.data_vars)[0]
    dtype_bytes = ds[sample_var].dtype.itemsize

    chunks: dict[str, int] = {}

    if access_pattern == "timeseries":
        # Prioritize time continuity, small spatial chunks
        if time_dim:
            chunks[time_dim] = min(
                ds.dims[time_dim],
                max(1, int(target_bytes / (dtype_bytes * 100))),  # ~100 spatial points
            )
        if lat_dim:
            chunks[lat_dim] = min(ds.dims[lat_dim], 10)
        if lon_dim:
            chunks[lon_dim] = min(ds.dims[lon_dim], 10)

    elif access_pattern == "spatial":
        # Prioritize spatial continuity, single time steps
        if time_dim:
            chunks[time_dim] = 1
        spatial_budget = target_bytes // dtype_bytes
        spatial_side = int(np.sqrt(spatial_budget))
        if lat_dim:
            chunks[lat_dim] = min(ds.dims[lat_dim], spatial_side)
        if lon_dim:
            chunks[lon_dim] = min(ds.dims[lon_dim], spatial_side)

    else:  # balanced
        # Distribute chunk size across dimensions
        n_dims = len([d for d in [time_dim, lat_dim, lon_dim] if d])
        if n_dims > 0:
            elements_per_chunk = target_bytes // dtype_bytes
            elements_per_dim = int(elements_per_chunk ** (1 / n_dims))

            if time_dim:
                chunks[time_dim] = min(ds.dims[time_dim], max(1, elements_per_dim))
            if lat_dim:
                chunks[lat_dim] = min(ds.dims[lat_dim], max(1, elements_per_dim))
            if lon_dim:
                chunks[lon_dim] = min(ds.dims[lon_dim], max(1, elements_per_dim))

    # Handle other dimensions
    for dim in ds.dims:
        if dim not in chunks:
            # Default: keep dimension whole if small, otherwise chunk
            if ds.dims[dim] <= 100:
                chunks[dim] = ds.dims[dim]
            else:
                chunks[dim] = min(ds.dims[dim], 100)

    # Ensure we don't exceed max chunks per dimension
    for dim, chunk_size in chunks.items():
        n_chunks = ds.dims[dim] // chunk_size
        if n_chunks > max_chunks_per_dim:
            chunks[dim] = max(1, ds.dims[dim] // max_chunks_per_dim)

    return chunks


def _identify_dimensions(ds: xr.Dataset) -> dict[str, str | None]:
    """Identify time, lat, lon dimensions by name patterns."""
    result: dict[str, str | None] = {"time": None, "lat": None, "lon": None}

    time_patterns = ["time", "t", "datetime", "date"]
    lat_patterns = ["lat", "latitude", "y", "nlat"]
    lon_patterns = ["lon", "longitude", "x", "nlon"]

    for dim in ds.dims:
        dim_lower = dim.lower()
        if any(p in dim_lower for p in time_patterns):
            result["time"] = dim
        elif any(p in dim_lower for p in lat_patterns):
            result["lat"] = dim
        elif any(p in dim_lower for p in lon_patterns):
            result["lon"] = dim

    return result


def rechunk_dataset(
    ds: xr.Dataset,
    chunks: dict[str, int],
    parallel: bool = True,
) -> xr.Dataset:
    """Rechunk dataset to specified chunk sizes.

    Args:
        ds: Input dataset
        chunks: Target chunk sizes
        parallel: Whether to use dask for parallel rechunking

    Returns:
        Rechunked dataset
    """
    return ds.chunk(chunks)


def estimate_chunk_memory(
    ds: xr.Dataset,
    chunks: dict[str, int],
) -> dict[str, float]:
    """Estimate memory usage for given chunk configuration.

    Args:
        ds: Input dataset
        chunks: Chunk sizes to evaluate

    Returns:
        Dictionary with memory estimates in MB
    """
    estimates = {}

    for var in ds.data_vars:
        dtype_bytes = ds[var].dtype.itemsize

        # Calculate chunk size
        chunk_elements = 1
        for dim in ds[var].dims:
            chunk_elements *= chunks.get(dim, ds.dims[dim])

        chunk_mb = (chunk_elements * dtype_bytes) / (1024 * 1024)
        estimates[var] = chunk_mb

    estimates["total_per_chunk"] = sum(estimates.values())

    # Calculate number of chunks
    total_chunks = 1
    for dim, size in ds.dims.items():
        chunk_size = chunks.get(dim, size)
        total_chunks *= (size + chunk_size - 1) // chunk_size

    estimates["total_chunks"] = total_chunks
    estimates["total_data_mb"] = sum(
        ds[var].nbytes / (1024 * 1024) for var in ds.data_vars
    )

    return estimates


def suggest_chunks_for_workflow(
    ds: xr.Dataset,
    workflow: list[str],
    target_mb: float = DEFAULT_TARGET_MB,
) -> dict[str, int]:
    """Suggest chunk sizes based on planned workflow operations.

    Args:
        ds: Input dataset
        workflow: List of operation types, e.g.:
            ["temporal_mean", "spatial_slice", "timeseries_extract"]
        target_mb: Target chunk size in MB

    Returns:
        Recommended chunk sizes
    """
    # Score access patterns based on workflow
    time_priority = 0
    spatial_priority = 0

    for op in workflow:
        op_lower = op.lower()
        if any(w in op_lower for w in ["temporal", "time", "annual", "monthly", "trend"]):
            time_priority += 1
        if any(w in op_lower for w in ["spatial", "map", "region", "bbox"]):
            spatial_priority += 1
        if any(w in op_lower for w in ["point", "timeseries", "extract"]):
            time_priority += 2  # Timeseries extraction strongly favors time continuity

    # Determine access pattern
    if time_priority > spatial_priority * 1.5:
        pattern = "timeseries"
    elif spatial_priority > time_priority * 1.5:
        pattern = "spatial"
    else:
        pattern = "balanced"

    return determine_optimal_chunks(ds, target_mb, pattern)


def validate_chunks(
    ds: xr.Dataset,
    chunks: dict[str, int],
) -> list[str]:
    """Validate chunk configuration and return any warnings.

    Args:
        ds: Input dataset
        chunks: Chunk sizes to validate

    Returns:
        List of warning messages (empty if valid)
    """
    warnings = []

    for dim, chunk_size in chunks.items():
        if dim not in ds.dims:
            warnings.append(f"Dimension '{dim}' not in dataset")
            continue

        dim_size = ds.dims[dim]

        if chunk_size > dim_size:
            warnings.append(
                f"Chunk size {chunk_size} for '{dim}' exceeds dimension size {dim_size}"
            )

        if chunk_size < 1:
            warnings.append(f"Chunk size for '{dim}' must be at least 1")

    # Check total chunk size
    estimates = estimate_chunk_memory(ds, chunks)
    if estimates["total_per_chunk"] > 100:
        warnings.append(
            f"Chunk size {estimates['total_per_chunk']:.1f} MB is very large (>100 MB)"
        )
    elif estimates["total_per_chunk"] < 0.1:
        warnings.append(
            f"Chunk size {estimates['total_per_chunk']:.3f} MB is very small (<0.1 MB)"
        )

    return warnings
