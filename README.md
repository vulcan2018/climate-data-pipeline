# Climate Data Pipeline

Python data processing pipeline for climate variables with NetCDF/Zarr support, pre-calculated metrics, and FastAPI endpoints.

## Live Demo

**[https://climate-data-pipeline.vercel.app](https://climate-data-pipeline.vercel.app)** - API documentation and endpoints

See also:
- [ARCO Demo](https://arco-demo.vercel.app) - Interactive demonstration
- [Climate Viz Frontend](https://climate-viz-frontend.vercel.app) - Data visualisation

## Features

- **Data Ingestion**: Read NetCDF files with Xarray, lazy loading with Dask
- **ARCO Conversion**: Convert to Analysis-Ready Cloud-Optimised Zarr format with optimal chunking
- **Pre-calculated Metrics**:
  - Temporal averages (monthly, seasonal, annual)
  - Climatological percentiles (10th, 25th, 50th, 75th, 90th, 95th, 99th)
  - Linear trend computation with significance testing
  - Absolute and standardized anomalies
- **Parallel Processing**: Dask-based distributed computation
- **REST API**: FastAPI endpoints with Redis caching
- **Testing**: Pytest suite with >60% coverage

## Installation

```bash
pip install -e .
```

Or with development dependencies:

```bash
pip install -e ".[dev]"
```

## Quick Start

### Data Processing

```python
from climate_pipeline.ingest import read_netcdf_lazy
from climate_pipeline.ingest.arco_converter import convert_to_zarr
from climate_pipeline.metrics import compute_monthly_mean, compute_anomaly

# Load data
ds = read_netcdf_lazy("era5_temperature.nc")

# Convert to Zarr for efficient access
convert_to_zarr(ds, "output.zarr", access_pattern="timeseries")

# Compute metrics
monthly = compute_monthly_mean(ds)
anomaly = compute_anomaly(ds, reference_period=("1991-01-01", "2020-12-31"))
```

### Running the API

```bash
uvicorn climate_pipeline.api.main:app --reload
```

API documentation available at `http://localhost:8000/docs`

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /api/v1/data/datasets` | List available datasets |
| `GET /api/v1/data/datasets/{id}/point` | Extract point time series |
| `GET /api/v1/data/datasets/{id}/region` | Extract regional data |
| `GET /api/v1/metrics/temporal/{id}` | Compute temporal averages |
| `GET /api/v1/metrics/percentiles/{id}` | Compute percentile thresholds |
| `GET /api/v1/metrics/trend/{id}` | Compute linear trends |
| `GET /api/v1/metrics/anomaly/{id}` | Compute anomalies |

## Project Structure

```
climate-data-pipeline/
├── src/climate_pipeline/
│   ├── ingest/           # Data ingestion (NetCDF, Zarr)
│   ├── metrics/          # Climate metrics computation
│   ├── processing/       # Parallel processing pipeline
│   ├── api/              # FastAPI application
│   └── tasks/            # Celery background tasks
├── tests/                # Pytest test suite
├── docs/                 # Documentation
└── Dockerfile            # Container build
```

## Technologies

| Component | Technology |
|-----------|------------|
| Data Arrays | Xarray 2024.x |
| Parallel Computing | Dask 2024.x |
| Cloud Storage | Zarr 2.x |
| REST API | FastAPI 0.109+ |
| Caching | Redis 7.x |
| Background Tasks | Celery 5.x |
| Testing | Pytest 8.x |

## Running Tests

```bash
pytest --cov=src/climate_pipeline
```

## Docker

Build and run:

```bash
docker build -t climate-pipeline .
docker run -p 8000:8000 climate-pipeline
```

## License

MIT License - see LICENSE file.

## Author

S. Kalogerakos - FIRA Software Ltd
