"""Celery background tasks for data processing.

Long-running computations are handled asynchronously through Celery,
with results cached in Redis for fast retrieval.
"""

import os
from typing import Any

from celery import Celery

# Celery configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "climate_pipeline",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1 hour max
    worker_prefetch_multiplier=1,
)


@celery_app.task(bind=True, name="process_dataset")
def process_dataset(
    self,
    input_path: str,
    output_path: str,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Process a dataset with specified operations.

    Args:
        input_path: Path to input NetCDF/Zarr file
        output_path: Path for output file
        operations: List of operations to apply

    Returns:
        Processing result metadata
    """
    from climate_pipeline.ingest import read_netcdf_lazy
    from climate_pipeline.ingest.arco_converter import convert_to_zarr

    self.update_state(state="PROCESSING", meta={"step": "loading"})

    try:
        # Load dataset
        ds = read_netcdf_lazy(input_path)

        # Apply operations
        for i, op in enumerate(operations):
            self.update_state(
                state="PROCESSING",
                meta={"step": op.get("name", f"operation_{i}")},
            )
            ds = _apply_operation(ds, op)

        # Save result
        self.update_state(state="PROCESSING", meta={"step": "saving"})
        convert_to_zarr(ds, output_path, overwrite=True)

        return {
            "status": "completed",
            "output_path": output_path,
            "operations_applied": len(operations),
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(bind=True, name="compute_metrics_task")
def compute_metrics_task(
    self,
    dataset_path: str,
    metric_type: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Compute metrics for a dataset.

    Args:
        dataset_path: Path to dataset
        metric_type: Type of metric to compute
        params: Metric-specific parameters

    Returns:
        Computed metric results
    """
    from climate_pipeline.ingest import read_netcdf_lazy
    from climate_pipeline.metrics import (
        compute_annual_mean,
        compute_anomaly,
        compute_climatological_percentiles,
        compute_linear_trend,
        compute_monthly_mean,
        compute_seasonal_mean,
    )

    self.update_state(state="PROCESSING", meta={"step": "loading"})

    try:
        ds = read_netcdf_lazy(dataset_path)

        self.update_state(state="PROCESSING", meta={"step": f"computing_{metric_type}"})

        if metric_type == "monthly_mean":
            result = compute_monthly_mean(ds)
        elif metric_type == "seasonal_mean":
            result = compute_seasonal_mean(ds)
        elif metric_type == "annual_mean":
            result = compute_annual_mean(ds)
        elif metric_type == "percentiles":
            result = compute_climatological_percentiles(
                ds,
                percentiles=params.get("percentiles", [10, 25, 50, 75, 90, 95, 99]),
                reference_period=params.get("reference_period"),
            )
        elif metric_type == "trend":
            result = compute_linear_trend(
                ds,
                reference_period=params.get("reference_period"),
            )
        elif metric_type == "anomaly":
            result = compute_anomaly(
                ds,
                reference_period=params.get("reference_period"),
            )
        else:
            return {"status": "failed", "error": f"Unknown metric type: {metric_type}"}

        # Convert to serializable format
        self.update_state(state="PROCESSING", meta={"step": "serializing"})

        return {
            "status": "completed",
            "metric_type": metric_type,
            "variables": list(result.data_vars),
            "shape": {var: list(result[var].shape) for var in result.data_vars},
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


@celery_app.task(bind=True, name="convert_to_zarr_task")
def convert_to_zarr_task(
    self,
    input_path: str,
    output_path: str,
    access_pattern: str = "balanced",
) -> dict[str, Any]:
    """Convert NetCDF to Zarr format.

    Args:
        input_path: Path to input NetCDF file
        output_path: Path for output Zarr store
        access_pattern: Chunking optimization target

    Returns:
        Conversion result metadata
    """
    from climate_pipeline.ingest import read_netcdf_lazy
    from climate_pipeline.ingest.arco_converter import convert_to_zarr, get_zarr_info

    self.update_state(state="PROCESSING", meta={"step": "loading"})

    try:
        ds = read_netcdf_lazy(input_path)

        self.update_state(state="PROCESSING", meta={"step": "converting"})
        convert_to_zarr(
            ds,
            output_path,
            access_pattern=access_pattern,  # type: ignore
            overwrite=True,
        )

        self.update_state(state="PROCESSING", meta={"step": "verifying"})
        info = get_zarr_info(output_path)

        return {
            "status": "completed",
            "output_path": output_path,
            "total_size_mb": info["total_size_mb"],
            "arrays": list(info["arrays"].keys()),
        }

    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
        }


def _apply_operation(ds: Any, op: dict[str, Any]) -> Any:
    """Apply a single operation to a dataset."""
    op_type = op.get("type")

    if op_type == "slice_time":
        return ds.sel(time=slice(op["start"], op["end"]))

    elif op_type == "slice_region":
        return ds.sel(
            lat=slice(op["south"], op["north"]),
            lon=slice(op["west"], op["east"]),
        )

    elif op_type == "resample":
        return ds.resample(time=op.get("freq", "ME")).mean()

    elif op_type == "compute_mean":
        dim = op.get("dim", "time")
        return ds.mean(dim=dim)

    else:
        raise ValueError(f"Unknown operation type: {op_type}")
