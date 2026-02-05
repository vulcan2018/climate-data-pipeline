# Chunking Strategy for Climate Data

This document describes the chunking strategies used for optimal data access patterns.

## Overview

Chunking determines how multi-dimensional arrays are split into smaller blocks for storage and parallel processing. The optimal chunk size depends on:

1. **Access pattern**: How data will be read (time series vs spatial maps)
2. **Memory constraints**: Available RAM per worker
3. **I/O characteristics**: Network/disk bandwidth

## Access Patterns

### Timeseries Pattern

Optimized for extracting full time series at specific geographic points.

```python
chunks = optimize_chunks(ds, access_pattern="timeseries")
# Result: {"time": 8760, "lat": 10, "lon": 10}
```

- Large time chunks (up to 1 year of hourly data)
- Small spatial chunks (10x10 grid cells)
- Ideal for: point extraction, trend analysis, anomaly computation

### Spatial Pattern

Optimized for extracting 2D maps at specific time steps.

```python
chunks = optimize_chunks(ds, access_pattern="spatial")
# Result: {"time": 1, "lat": 500, "lon": 500}
```

- Single time step per chunk
- Large spatial chunks
- Ideal for: map visualization, spatial statistics

### Balanced Pattern

Compromise between both access patterns.

```python
chunks = optimize_chunks(ds, access_pattern="balanced")
# Result: {"time": 100, "lat": 100, "lon": 100}
```

- Moderate chunks in all dimensions
- Good for mixed workloads

## Target Chunk Size

We target 1-10 MB per chunk:

- Too small: excessive overhead from chunk metadata
- Too large: memory issues, poor parallelization
- 4 MB default provides good balance

## Compression

All Zarr output uses Zstd compression by default:

- Level 3 provides good compression/speed tradeoff
- ~50% size reduction typical for climate data
- Negligible decompression overhead
