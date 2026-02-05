"""NetCDF to ARCO (Analysis-Ready Cloud-Optimised) Zarr conversion.

Provides tools for converting NetCDF files to Zarr format with optimal
chunking strategies for different access patterns.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
import zarr


# Target chunk sizes in bytes (aim for 1-10 MB chunks)
TARGET_CHUNK_SIZE_MB = 4
TARGET_CHUNK_BYTES = TARGET_CHUNK_SIZE_MB * 1024 * 1024


def optimize_chunks(
    ds: xr.Dataset,
    access_pattern: Literal["timeseries", "spatial", "balanced"] = "balanced",
) -> dict[str, int]:
    """Determine optimal chunk sizes for a dataset.

    Args:
        ds: Input Xarray dataset
        access_pattern:
            - "timeseries": Optimize for point time series extraction
            - "spatial": Optimize for spatial map extraction at single times
            - "balanced": Balance between both access patterns

    Returns:
        Dictionary of dimension names to chunk sizes
    """
    # Detect dimensions
    time_dim = None
    lat_dim = None
    lon_dim = None

    for dim in ds.dims:
        dim_lower = dim.lower()
        if "time" in dim_lower or dim_lower == "t":
            time_dim = dim
        elif "lat" in dim_lower or dim_lower == "y":
            lat_dim = dim
        elif "lon" in dim_lower or dim_lower == "x":
            lon_dim = dim

    chunks: dict[str, int] = {}

    # Get a representative variable for dtype size calculation
    sample_var = list(ds.data_vars)[0]
    dtype_size = ds[sample_var].dtype.itemsize

    if access_pattern == "timeseries":
        # Small spatial chunks, large time chunks
        # Ideal for extracting full time series at specific points
        if time_dim:
            chunks[time_dim] = min(ds.dims[time_dim], 8760)  # Up to 1 year hourly
        if lat_dim:
            chunks[lat_dim] = min(ds.dims[lat_dim], 10)
        if lon_dim:
            chunks[lon_dim] = min(ds.dims[lon_dim], 10)

    elif access_pattern == "spatial":
        # Large spatial chunks, small time chunks
        # Ideal for extracting spatial maps at specific times
        if time_dim:
            chunks[time_dim] = 1
        if lat_dim:
            chunks[lat_dim] = min(ds.dims[lat_dim], 500)
        if lon_dim:
            chunks[lon_dim] = min(ds.dims[lon_dim], 500)

    else:  # balanced
        # Balance chunk sizes to achieve target chunk size
        total_elements = TARGET_CHUNK_BYTES // dtype_size

        if time_dim and lat_dim and lon_dim:
            # 3D data: aim for roughly cubic chunks in normalized space
            nt = ds.dims[time_dim]
            ny = ds.dims[lat_dim]
            nx = ds.dims[lon_dim]

            # Start with cube root
            target_per_dim = int(total_elements ** (1 / 3))

            # Adjust based on actual dimension sizes
            chunks[time_dim] = min(nt, max(1, target_per_dim))
            chunks[lat_dim] = min(ny, max(1, target_per_dim))
            chunks[lon_dim] = min(nx, max(1, target_per_dim))

        elif lat_dim and lon_dim:
            # 2D spatial data
            ny = ds.dims[lat_dim]
            nx = ds.dims[lon_dim]
            target_per_dim = int(total_elements ** 0.5)
            chunks[lat_dim] = min(ny, max(1, target_per_dim))
            chunks[lon_dim] = min(nx, max(1, target_per_dim))

    # Handle any remaining dimensions
    for dim in ds.dims:
        if dim not in chunks:
            chunks[dim] = min(ds.dims[dim], 100)

    return chunks


def convert_to_zarr(
    ds: xr.Dataset,
    output_path: str | Path,
    chunks: dict[str, int] | None = None,
    access_pattern: Literal["timeseries", "spatial", "balanced"] = "balanced",
    compression: str = "zstd",
    compression_level: int = 3,
    overwrite: bool = False,
) -> Path:
    """Convert Xarray Dataset to Zarr format with optimal chunking.

    Args:
        ds: Input Xarray dataset
        output_path: Output path for Zarr store
        chunks: Optional explicit chunk sizes (auto-computed if None)
        access_pattern: Chunking optimization target if chunks not specified
        compression: Compression codec ("zstd", "lz4", "gzip", or None)
        compression_level: Compression level (higher = smaller but slower)
        overwrite: Whether to overwrite existing store

    Returns:
        Path to created Zarr store
    """
    output_path = Path(output_path)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Zarr store already exists: {output_path}")

    # Determine chunks
    if chunks is None:
        chunks = optimize_chunks(ds, access_pattern)

    # Rechunk dataset
    ds_chunked = ds.chunk(chunks)

    # Set up compression
    encoding = {}
    if compression:
        compressor = _get_compressor(compression, compression_level)
        for var in ds.data_vars:
            encoding[var] = {"compressor": compressor}

    # Write to Zarr
    ds_chunked.to_zarr(
        output_path,
        mode="w" if overwrite else "w-",
        encoding=encoding,
        consolidated=True,
    )

    return output_path


def _get_compressor(codec: str, level: int) -> zarr.codecs.Codec:
    """Get Zarr compressor for the specified codec."""
    if codec == "zstd":
        try:
            from numcodecs import Zstd
            return Zstd(level=level)
        except ImportError:
            from numcodecs import Blosc
            return Blosc(cname="zstd", clevel=level)
    elif codec == "lz4":
        from numcodecs import Blosc
        return Blosc(cname="lz4", clevel=level)
    elif codec == "gzip":
        from numcodecs import GZip
        return GZip(level=level)
    else:
        raise ValueError(f"Unknown compression codec: {codec}")


def rechunk_dataset(
    ds: xr.Dataset,
    chunks: dict[str, int],
) -> xr.Dataset:
    """Rechunk dataset to new chunk sizes.

    Args:
        ds: Input dataset
        chunks: New chunk sizes

    Returns:
        Rechunked dataset
    """
    return ds.chunk(chunks)


def open_zarr(path: str | Path) -> xr.Dataset:
    """Open a Zarr store as an Xarray dataset.

    Args:
        path: Path to Zarr store

    Returns:
        Xarray Dataset with dask arrays
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Zarr store not found: {path}")

    return xr.open_zarr(path, consolidated=True)


def get_zarr_info(path: str | Path) -> dict:
    """Get information about a Zarr store.

    Args:
        path: Path to Zarr store

    Returns:
        Dictionary with store information
    """
    path = Path(path)
    store = zarr.open(path, mode="r")

    info = {
        "path": str(path),
        "arrays": {},
        "total_size_bytes": 0,
    }

    for name, arr in store.arrays():
        arr_info = {
            "shape": arr.shape,
            "chunks": arr.chunks,
            "dtype": str(arr.dtype),
            "compressor": str(arr.compressor) if arr.compressor else None,
            "nbytes": arr.nbytes,
            "nbytes_stored": arr.nbytes_stored if hasattr(arr, "nbytes_stored") else None,
        }
        info["arrays"][name] = arr_info
        info["total_size_bytes"] += arr.nbytes

    info["total_size_mb"] = info["total_size_bytes"] / (1024 * 1024)

    return info
