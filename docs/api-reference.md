# API Reference

Complete reference for the Climate Data Pipeline REST API.

## Base URL

- Development: `http://localhost:8000`
- Production: `https://climate-data-pipeline.vercel.app`

## Authentication

Currently no authentication required. Future versions may add API key authentication.

## Endpoints

### Health

#### GET /health

Basic health check.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T12:00:00Z",
  "service": "climate-data-pipeline"
}
```

#### GET /health/ready

Readiness check with component status.

#### GET /health/live

Liveness probe for Kubernetes.

### Data Access

#### GET /api/v1/data/datasets

List all available datasets.

**Response:**
```json
{
  "datasets": [
    {
      "id": "era5-t2m",
      "name": "ERA5 2m Temperature",
      "variable": "2m_temperature",
      "temporal_range": "1940-present",
      "resolution": "0.25deg"
    }
  ],
  "count": 3
}
```

#### GET /api/v1/data/datasets/{id}

Get detailed dataset information.

**Parameters:**
- `id` (path): Dataset identifier

#### GET /api/v1/data/datasets/{id}/point

Extract time series at a point.

**Parameters:**
- `id` (path): Dataset identifier
- `lat` (query, required): Latitude (-90 to 90)
- `lon` (query, required): Longitude (-180 to 180)
- `start_date` (query): Start date (YYYY-MM-DD)
- `end_date` (query): End date (YYYY-MM-DD)

#### GET /api/v1/data/datasets/{id}/region

Extract spatial data for a region.

**Parameters:**
- `west`, `south`, `east`, `north` (query, required): Bounding box
- `time` (query): Time slice (YYYY-MM-DD)

### Metrics

#### GET /api/v1/metrics/temporal/{id}

Compute temporal averages.

**Parameters:**
- `metric` (query, required): "monthly", "seasonal", or "annual"
- `lat`, `lon` (query, required): Location
- `start_year`, `end_year` (query): Year range

#### GET /api/v1/metrics/percentiles/{id}

Compute climatological percentiles.

**Parameters:**
- `lat`, `lon` (query, required): Location
- `percentiles` (query): Comma-separated list (default: "10,25,50,75,90,95,99")
- `reference_start`, `reference_end` (query): Reference period years

#### GET /api/v1/metrics/trend/{id}

Compute linear trend with significance.

**Parameters:**
- `lat`, `lon` (query, required): Location
- `start_year`, `end_year` (query): Period for trend
- `confidence` (query): Confidence level (default: 0.95)

**Response:**
```json
{
  "trend": {
    "slope": 0.023,
    "slope_units": "K per year",
    "p_value": 0.0012,
    "significant": true,
    "confidence_interval": {
      "lower": 0.018,
      "upper": 0.028
    }
  }
}
```

#### GET /api/v1/metrics/anomaly/{id}

Compute anomaly for a specific time.

**Parameters:**
- `lat`, `lon` (query, required): Location
- `time` (query, required): Date (YYYY-MM-DD)
- `anomaly_type` (query): "absolute" or "standardized"
- `reference_start`, `reference_end` (query): Reference period

## Error Responses

All errors return JSON with this structure:

```json
{
  "detail": "Error message",
  "timestamp": "2024-01-15T12:00:00Z"
}
```

Status codes:
- `400`: Bad request (invalid parameters)
- `404`: Resource not found
- `500`: Internal server error
