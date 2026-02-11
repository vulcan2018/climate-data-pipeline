#!/usr/bin/env python3
"""
ARCO Performance Benchmark Suite

Comprehensive benchmarking tool for ARCO data access patterns,
demonstrating performance capability for <2 second latency target.

Author: S. Kalogerakos
Company: FIRA Software Ltd
Date: February 2026
"""

import json
import time
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
from datetime import datetime

try:
    import numpy as np
    import xarray as xr
    import zarr
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    description: str
    n_iterations: int
    mean_ms: float
    median_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float
    meets_target: bool
    target_ms: float = 2000.0


class ARCOBenchmarkSuite:
    """
    Benchmark suite for ARCO data access patterns.

    Tests various access patterns to validate <2 second
    latency requirement.
    """

    def __init__(self, zarr_path: str, target_latency_ms: float = 2000.0):
        """
        Initialize benchmark suite.

        Args:
            zarr_path: Path to Zarr store
            target_latency_ms: Target latency in milliseconds
        """
        self.zarr_path = zarr_path
        self.target_latency_ms = target_latency_ms
        self.results = []

    def _run_benchmark(
        self,
        name: str,
        description: str,
        func,
        n_iterations: int = 20,
        warmup: int = 3
    ) -> BenchmarkResult:
        """
        Run a single benchmark.

        Args:
            name: Benchmark name
            description: What is being tested
            func: Function to benchmark (takes dataset, returns data)
            n_iterations: Number of test iterations
            warmup: Number of warmup iterations

        Returns:
            BenchmarkResult with timing statistics
        """
        ds = xr.open_zarr(self.zarr_path)

        # Warmup
        for _ in range(warmup):
            _ = func(ds)

        # Timed iterations
        latencies_ms = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            _ = func(ds)
            elapsed_ms = (time.perf_counter() - start) * 1000
            latencies_ms.append(elapsed_ms)

        ds.close()

        result = BenchmarkResult(
            name=name,
            description=description,
            n_iterations=n_iterations,
            mean_ms=round(statistics.mean(latencies_ms), 2),
            median_ms=round(statistics.median(latencies_ms), 2),
            std_ms=round(statistics.stdev(latencies_ms), 2) if len(latencies_ms) > 1 else 0,
            min_ms=round(min(latencies_ms), 2),
            max_ms=round(max(latencies_ms), 2),
            p95_ms=round(np.percentile(latencies_ms, 95), 2),
            p99_ms=round(np.percentile(latencies_ms, 99), 2),
            meets_target=max(latencies_ms) < self.target_latency_ms,
            target_ms=self.target_latency_ms
        )

        self.results.append(result)
        return result

    def benchmark_single_timestep(self, n_iterations: int = 20) -> BenchmarkResult:
        """Benchmark single timestep access (common pattern)."""
        def access_func(ds):
            t_idx = np.random.randint(0, ds.sizes['time'])
            return ds.isel(time=t_idx).load()

        return self._run_benchmark(
            name="single_timestep",
            description="Access single time slice (full spatial extent)",
            func=access_func,
            n_iterations=n_iterations
        )

    def benchmark_spatial_subset(self, n_iterations: int = 20) -> BenchmarkResult:
        """Benchmark spatial subset access."""
        def access_func(ds):
            t_idx = np.random.randint(0, ds.sizes['time'])
            lat_start = np.random.randint(0, ds.sizes['latitude'] - 20)
            lon_start = np.random.randint(0, ds.sizes['longitude'] - 20)
            return ds.isel(
                time=t_idx,
                latitude=slice(lat_start, lat_start + 20),
                longitude=slice(lon_start, lon_start + 20)
            ).load()

        return self._run_benchmark(
            name="spatial_subset",
            description="Access spatial subset (20x20 grid, single time)",
            func=access_func,
            n_iterations=n_iterations
        )

    def benchmark_time_series_point(self, n_iterations: int = 20) -> BenchmarkResult:
        """Benchmark time series at single point."""
        def access_func(ds):
            lat_idx = np.random.randint(0, ds.sizes['latitude'])
            lon_idx = np.random.randint(0, ds.sizes['longitude'])
            return ds.isel(
                latitude=lat_idx,
                longitude=lon_idx
            ).load()

        return self._run_benchmark(
            name="time_series_point",
            description="Access full time series at single point",
            func=access_func,
            n_iterations=n_iterations
        )

    def benchmark_monthly_mean(self, n_iterations: int = 20) -> BenchmarkResult:
        """Benchmark monthly mean calculation."""
        def access_func(ds):
            # Get first month (30 days)
            return ds.isel(time=slice(0, 30)).mean(dim='time').load()

        return self._run_benchmark(
            name="monthly_mean",
            description="Calculate monthly mean (30 timesteps)",
            func=access_func,
            n_iterations=n_iterations
        )

    def benchmark_single_variable(self, variable: str, n_iterations: int = 20) -> BenchmarkResult:
        """Benchmark single variable access."""
        def access_func(ds):
            t_idx = np.random.randint(0, ds.sizes['time'])
            return ds[variable].isel(time=t_idx).load()

        return self._run_benchmark(
            name=f"single_variable_{variable}",
            description=f"Access single variable ({variable}) at one timestep",
            func=access_func,
            n_iterations=n_iterations
        )

    def run_all_benchmarks(self, n_iterations: int = 20) -> list:
        """
        Run complete benchmark suite.

        Args:
            n_iterations: Number of iterations per benchmark

        Returns:
            List of BenchmarkResult objects
        """
        print("ARCO Performance Benchmark Suite")
        print("=" * 60)
        print(f"Target latency: {self.target_latency_ms}ms")
        print(f"Iterations per benchmark: {n_iterations}")
        print("=" * 60)

        # Get available variables
        ds = xr.open_zarr(self.zarr_path)
        variables = list(ds.data_vars)
        ds.close()

        benchmarks = [
            ("Single Timestep", self.benchmark_single_timestep),
            ("Spatial Subset", self.benchmark_spatial_subset),
            ("Time Series Point", self.benchmark_time_series_point),
            ("Monthly Mean", self.benchmark_monthly_mean),
        ]

        # Add per-variable benchmarks
        for var in variables[:3]:  # First 3 variables
            benchmarks.append((f"Variable: {var}", lambda n, v=var: self.benchmark_single_variable(v, n)))

        for name, func in benchmarks:
            print(f"\n{name}...")
            result = func(n_iterations)
            status = "PASS" if result.meets_target else "FAIL"
            print(f"  Mean: {result.mean_ms}ms | P95: {result.p95_ms}ms | Max: {result.max_ms}ms | [{status}]")

        return self.results

    def generate_report(self, output_path: Optional[str] = None) -> dict:
        """
        Generate benchmark report.

        Args:
            output_path: Optional path to save JSON report

        Returns:
            Dictionary with full report
        """
        report = {
            "benchmark_suite": "ARCO Performance Benchmark",
            "organization": "FIRA Software Ltd",
            "timestamp": datetime.now().isoformat(),
            "target_latency_ms": self.target_latency_ms,
            "zarr_path": self.zarr_path,
            "results": [asdict(r) for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "passed": sum(1 for r in self.results if r.meets_target),
                "failed": sum(1 for r in self.results if not r.meets_target),
                "overall_pass": all(r.meets_target for r in self.results)
            }
        }

        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)

        return report


def create_test_data(output_path: str, size: str = 'medium') -> str:
    """
    Create test Zarr data for benchmarking.

    Args:
        output_path: Output directory
        size: 'small', 'medium', or 'large'

    Returns:
        Path to created Zarr store
    """
    sizes = {
        'small': (100, 90, 180),    # ~6MB
        'medium': (365, 180, 360),  # ~90MB
        'large': (730, 360, 720)    # ~720MB
    }

    time_steps, lat_size, lon_size = sizes.get(size, sizes['medium'])

    print(f"Creating {size} test dataset...")

    # Generate data
    times = np.arange(time_steps)
    lats = np.linspace(-90, 90, lat_size)
    lons = np.linspace(-180, 180, lon_size)

    data_vars = {}
    for var in ['temperature', 'precipitation', 'wind_speed']:
        data = np.random.randn(time_steps, lat_size, lon_size).astype(np.float32)
        data_vars[var] = xr.DataArray(
            data,
            dims=['time', 'latitude', 'longitude'],
            coords={'time': times, 'latitude': lats, 'longitude': lons}
        )

    ds = xr.Dataset(data_vars)

    # Determine chunks
    chunk_size = 50
    chunks = {
        'time': min(time_steps, chunk_size),
        'latitude': min(lat_size, chunk_size),
        'longitude': min(lon_size, chunk_size)
    }

    # Save to Zarr
    zarr_path = str(Path(output_path) / 'benchmark_data.zarr')
    encoding = {
        var: {
            'compressor': zarr.Blosc(cname='zstd', clevel=3),
            'chunks': tuple(chunks.get(dim, 'auto') for dim in ds[var].dims)
        }
        for var in ds.data_vars
    }

    ds.to_zarr(zarr_path, mode='w', encoding=encoding)
    print(f"Created: {zarr_path}")

    return zarr_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='ARCO Performance Benchmark Suite - FIRA Software Ltd'
    )
    parser.add_argument(
        '--zarr-path', '-z',
        help='Path to existing Zarr store to benchmark'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./benchmark_output',
        help='Output directory for results'
    )
    parser.add_argument(
        '--create-data',
        choices=['small', 'medium', 'large'],
        help='Create test data of specified size'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=20,
        help='Number of iterations per benchmark'
    )
    parser.add_argument(
        '--target-ms', '-t',
        type=float,
        default=2000.0,
        help='Target latency in milliseconds'
    )

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        print("Error: Required dependencies not available")
        print("Install with: pip install numpy xarray zarr")
        return 1

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create or use existing data
    if args.create_data:
        zarr_path = create_test_data(str(output_path), args.create_data)
    elif args.zarr_path:
        zarr_path = args.zarr_path
    else:
        # Create medium test data by default
        zarr_path = create_test_data(str(output_path), 'medium')

    # Run benchmarks
    suite = ARCOBenchmarkSuite(zarr_path, target_latency_ms=args.target_ms)
    suite.run_all_benchmarks(n_iterations=args.iterations)

    # Generate report
    report_path = output_path / 'benchmark_report.json'
    report = suite.generate_report(str(report_path))

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total: {report['summary']['total_benchmarks']}")
    print(f"Passed: {report['summary']['passed']}")
    print(f"Failed: {report['summary']['failed']}")
    print(f"Overall: {'PASS' if report['summary']['overall_pass'] else 'FAIL'}")
    print(f"\nReport saved to: {report_path}")

    return 0 if report['summary']['overall_pass'] else 1


if __name__ == '__main__':
    exit(main())
