"""Health check endpoints."""

from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint.

    Returns:
        Health status and timestamp
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "climate-data-pipeline",
    }


@router.get("/health/ready")
async def readiness_check() -> dict:
    """Readiness check - verifies service can handle requests.

    Returns:
        Readiness status with component checks
    """
    checks = {
        "api": True,
        "processing": True,
    }

    # Try to import core modules to verify they're available
    try:
        from climate_pipeline.ingest import read_netcdf_lazy
        from climate_pipeline.metrics import compute_monthly_mean
        checks["modules"] = True
    except ImportError:
        checks["modules"] = False

    all_ready = all(checks.values())

    return {
        "status": "ready" if all_ready else "not_ready",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
    }


@router.get("/health/live")
async def liveness_check() -> dict:
    """Liveness check - verifies service is running.

    Returns:
        Liveness status
    """
    return {
        "status": "alive",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
