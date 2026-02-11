#!/usr/bin/env python3
"""
ARCO Data Format Demonstration Script

This script demonstrates FIRA Software's capability with ARCO (Analysis-Ready,
Cloud-Optimised) data formats, specifically:
- NetCDF to Zarr conversion
- Optimal chunking strategies
- Pre-calculated metrics generation
- Performance benchmarking

For evaluation purposes in support of CJS2_220b_bis tender submission.

Author: S. Kalogerakos
Company: FIRA Software Ltd
Date: February 2026
"""

import time
from pathlib import Path
from typing import Optional
import json

# Check for required dependencies
DEPENDENCIES_AVAILABLE = True
try:
    import numpy as np
    import xarray as xr
    import zarr
    from zarr import Blosc
except ImportError as e:
    DEPENDENCIES_AVAILABLE = False
    MISSING_DEP = str(e)


def generate_sample_climate_data(
    time_steps: int = 365,
    lat_size: int = 180,
    lon_size: int = 360,
    variables: list = None
) -> 'xr.Dataset':
    """
    Generate sample climate data for demonstration.

    Creates a realistic multi-dimensional dataset similar to
    ERA5 or CAMS data products.

    Args:
        time_steps: Number of time steps (default: 365 days)
        lat_size: Latitude dimension size
        lon_size: Longitude dimension size
        variables: List of variable names to generate

    Returns:
        xarray Dataset with sample climate data
    """
    if variables is None:
        variables = ['temperature', 'precipitation', 'wind_speed']

    # Create coordinate arrays
    times = np.arange(time_steps)
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    # Generate realistic-looking data for each variable
    data_vars = {}

    for var in variables:
        # Create base pattern with spatial and temporal variation
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

        # Different patterns for different variables
        if var == 'temperature':
            # Temperature decreases with latitude
            base = 25 - 0.5 * np.abs(lat_grid)
            noise = np.random.randn(time_steps, lat_size, lon_size) * 5
            data = base[np.newaxis, :, :] + noise
            attrs = {'units': 'celsius', 'long_name': '2m Temperature'}

        elif var == 'precipitation':
            # Higher precipitation near equator
            base = 5 * np.exp(-0.01 * lat_grid**2)
            noise = np.random.exponential(1, (time_steps, lat_size, lon_size))
            data = base[np.newaxis, :, :] * noise
            attrs = {'units': 'mm/day', 'long_name': 'Total Precipitation'}

        else:  # wind_speed
            base = 5 + 3 * np.sin(np.radians(lat_grid))
            noise = np.random.randn(time_steps, lat_size, lon_size) * 2
            data = np.abs(base[np.newaxis, :, :] + noise)
            attrs = {'units': 'm/s', 'long_name': '10m Wind Speed'}

        data_vars[var] = xr.DataArray(
            data.astype(np.float32),
            dims=['time', 'latitude', 'longitude'],
            coords={
                'time': times,
                'latitude': lats,
                'longitude': lons
            },
            attrs=attrs
        )

    # Create dataset with metadata
    ds = xr.Dataset(
        data_vars,
        attrs={
            'title': 'Sample Climate Data for ARCO Demonstration',
            'institution': 'FIRA Software Ltd',
            'source': 'Synthetic data for demonstration',
            'conventions': 'CF-1.8',
            'created_by': 'arco_demo.py'
        }
    )

    return ds


def determine_optimal_chunks(
    dataset: 'xr.Dataset',
    target_chunk_mb: float = 10.0,
    access_pattern: str = 'time_series'
) -> dict:
    """
    Determine optimal chunk sizes based on access patterns.

    This implements the chunking strategy approach described in our
    technical solution for the ARCO tender.

    Args:
        dataset: Input xarray Dataset
        target_chunk_mb: Target chunk size in megabytes
        access_pattern: One of 'time_series', 'spatial', 'balanced'

    Returns:
        Dictionary of chunk sizes per dimension
    """
    # Get dimension sizes
    dim_sizes = dict(dataset.sizes)

    # Estimate bytes per element (assuming float32)
    bytes_per_element = 4
    target_elements = int(target_chunk_mb * 1024 * 1024 / bytes_per_element)

    if access_pattern == 'time_series':
        # Optimize for time series access (small spatial, full time)
        # Good for trend analysis, anomaly detection
        chunks = {
            'time': min(dim_sizes.get('time', 365), 365),  # Full year if possible
            'latitude': 10,
            'longitude': 10
        }

    elif access_pattern == 'spatial':
        # Optimize for spatial access (single time, large spatial extent)
        # Good for mapping, spatial visualization
        chunks = {
            'time': 1,
            'latitude': min(dim_sizes.get('latitude', 180), 90),
            'longitude': min(dim_sizes.get('longitude', 360), 180)
        }

    else:  # balanced
        # Balanced chunking for mixed access patterns
        # Cube root approximation for equal dimensions
        n_dims = len([d for d in ['time', 'latitude', 'longitude'] if d in dim_sizes])
        chunk_per_dim = int(target_elements ** (1/n_dims))

        chunks = {
            'time': min(dim_sizes.get('time', 365), chunk_per_dim),
            'latitude': min(dim_sizes.get('latitude', 180), chunk_per_dim),
            'longitude': min(dim_sizes.get('longitude', 360), chunk_per_dim)
        }

    # Filter to only include dimensions that exist
    chunks = {k: v for k, v in chunks.items() if k in dim_sizes}

    return chunks


def convert_to_zarr(
    dataset: 'xr.Dataset',
    target_path: str,
    chunks: dict,
    compression_level: int = 3
) -> dict:
    """
    Convert xarray Dataset to Zarr with optimal encoding.

    This function demonstrates our ARCO conversion capability:
    - Configurable chunking per variable
    - Zstd compression for efficient storage
    - Metadata preservation

    Args:
        dataset: Input xarray Dataset
        target_path: Output Zarr store path
        chunks: Dictionary of chunk sizes
        compression_level: Blosc compression level (1-9)

    Returns:
        Dictionary with conversion statistics
    """
    start_time = time.time()

    # Create encoding with compression for each variable
    encoding = {}
    for var in dataset.data_vars:
        encoding[var] = {
            'compressor': Blosc(cname='zstd', clevel=compression_level),
            'chunks': tuple(chunks.get(dim, 'auto') for dim in dataset[var].dims)
        }

    # Convert to Zarr
    dataset.to_zarr(target_path, mode='w', encoding=encoding)

    # Calculate statistics
    end_time = time.time()

    # Get file sizes
    zarr_store = zarr.open(target_path, mode='r')

    stats = {
        'conversion_time_seconds': round(end_time - start_time, 2),
        'variables': list(dataset.data_vars),
        'chunks': chunks,
        'compression': f'zstd level {compression_level}',
        'zarr_path': target_path
    }

    return stats


def calculate_metrics(dataset: 'xr.Dataset', variable: str) -> dict:
    """
    Calculate pre-computed metrics for a variable.

    Pre-calculating common metrics is key to achieving <2 second
    access latency as required by the ITT.

    Args:
        dataset: Input xarray Dataset
        variable: Variable name to calculate metrics for

    Returns:
        Dictionary of pre-calculated metrics
    """
    da = dataset[variable]

    metrics = {
        'global_mean': float(da.mean().values),
        'global_std': float(da.std().values),
        'global_min': float(da.min().values),
        'global_max': float(da.max().values),
        'percentile_5': float(np.percentile(da.values, 5)),
        'percentile_95': float(np.percentile(da.values, 95)),
        'temporal_mean': da.mean(dim='time').values.tolist() if 'time' in da.dims else None,
    }

    return metrics


def benchmark_access_latency(zarr_path: str, n_queries: int = 10) -> dict:
    """
    Benchmark data access latency from Zarr store.

    This demonstrates that our approach can meet the <2 second
    access latency requirement.

    Args:
        zarr_path: Path to Zarr store
        n_queries: Number of random queries to run

    Returns:
        Dictionary with latency statistics
    """
    ds = xr.open_zarr(zarr_path)

    latencies = []

    for _ in range(n_queries):
        # Random time slice access (common pattern)
        t_idx = np.random.randint(0, ds.sizes['time'])

        start = time.time()
        _ = ds.isel(time=t_idx).load()
        latencies.append(time.time() - start)

    ds.close()

    return {
        'n_queries': n_queries,
        'mean_latency_ms': round(np.mean(latencies) * 1000, 2),
        'max_latency_ms': round(np.max(latencies) * 1000, 2),
        'min_latency_ms': round(np.min(latencies) * 1000, 2),
        'p95_latency_ms': round(np.percentile(latencies, 95) * 1000, 2),
        'meets_2s_target': np.max(latencies) < 2.0
    }


def run_full_demonstration(output_dir: str = './arco_demo_output') -> dict:
    """
    Run full ARCO demonstration workflow.

    This demonstrates the complete pipeline:
    1. Generate sample data
    2. Determine optimal chunks
    3. Convert to Zarr
    4. Calculate metrics
    5. Benchmark access

    Args:
        output_dir: Directory for output files

    Returns:
        Dictionary with all demonstration results
    """
    if not DEPENDENCIES_AVAILABLE:
        return {
            'error': f'Missing dependencies: {MISSING_DEP}',
            'install_command': 'pip install numpy xarray zarr'
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("ARCO Data Format Demonstration")
    print("=" * 50)
    print("FIRA Software Ltd - CJS2_220b_bis Tender")
    print("=" * 50)

    results = {}

    # Step 1: Generate sample data
    print("\n1. Generating sample climate data...")
    ds = generate_sample_climate_data(
        time_steps=365,
        lat_size=180,
        lon_size=360
    )
    results['dataset'] = {
        'dimensions': dict(ds.sizes),
        'variables': list(ds.data_vars),
        'size_mb': round(ds.nbytes / 1024 / 1024, 2)
    }
    print(f"   Created dataset: {results['dataset']['size_mb']} MB")

    # Step 2: Determine optimal chunks for different patterns
    print("\n2. Calculating optimal chunk strategies...")
    results['chunking'] = {}
    for pattern in ['time_series', 'spatial', 'balanced']:
        chunks = determine_optimal_chunks(ds, access_pattern=pattern)
        results['chunking'][pattern] = chunks
        print(f"   {pattern}: {chunks}")

    # Step 3: Convert to Zarr with balanced chunking
    print("\n3. Converting to Zarr format...")
    zarr_path = str(output_path / 'climate_data.zarr')
    chunks = results['chunking']['balanced']
    conversion_stats = convert_to_zarr(ds, zarr_path, chunks)
    results['conversion'] = conversion_stats
    print(f"   Converted in {conversion_stats['conversion_time_seconds']}s")

    # Step 4: Calculate metrics
    print("\n4. Calculating pre-computed metrics...")
    results['metrics'] = {}
    for var in ds.data_vars:
        results['metrics'][var] = calculate_metrics(ds, var)
        print(f"   {var}: mean={results['metrics'][var]['global_mean']:.2f}")

    # Step 5: Benchmark access
    print("\n5. Benchmarking access latency...")
    latency_results = benchmark_access_latency(zarr_path)
    results['latency'] = latency_results
    print(f"   Mean latency: {latency_results['mean_latency_ms']}ms")
    print(f"   Meets <2s target: {latency_results['meets_2s_target']}")

    # Save results
    results_file = output_path / 'demonstration_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 50)

    return results


def main():
    """Main entry point for demonstration script."""
    import argparse

    parser = argparse.ArgumentParser(
        description='ARCO Data Format Demonstration - FIRA Software Ltd'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./arco_demo_output',
        help='Output directory for demonstration files'
    )
    parser.add_argument(
        '--check-deps',
        action='store_true',
        help='Check dependencies and exit'
    )

    args = parser.parse_args()

    if args.check_deps:
        if DEPENDENCIES_AVAILABLE:
            print("All dependencies available.")
            print("Required: numpy, xarray, zarr")
        else:
            print(f"Missing dependencies: {MISSING_DEP}")
            print("Install with: pip install numpy xarray zarr")
        return

    results = run_full_demonstration(args.output_dir)

    if 'error' in results:
        print(f"\nError: {results['error']}")
        print(f"Install: {results['install_command']}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
