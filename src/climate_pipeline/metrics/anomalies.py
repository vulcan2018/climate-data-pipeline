"""Anomaly computation for climate data.

Computes absolute and standardized anomalies relative to a climatological
reference period.
"""

import numpy as np
import xarray as xr

from .temporal import compute_climatology


def compute_anomaly(
    ds: xr.Dataset,
    time_dim: str = "time",
    reference_period: tuple[str, str] | None = None,
    groupby: str = "month",
    climatology: xr.Dataset | None = None,
) -> xr.Dataset:
    """Compute absolute anomalies relative to climatology.

    Anomaly = Value - Climatological Mean

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        reference_period: Optional (start, end) dates for climatology
        groupby: How to group climatology ("month", "dayofyear", or "season")
        climatology: Pre-computed climatology (if None, computed from data)

    Returns:
        Dataset with anomaly values
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Compute or use provided climatology
    if climatology is None:
        if reference_period:
            ref_ds = ds.sel({time_dim: slice(reference_period[0], reference_period[1])})
        else:
            ref_ds = ds
        climatology = compute_climatology(ref_ds, time_dim, groupby)

    # Compute anomaly
    if groupby == "month":
        anomaly = ds.groupby(f"{time_dim}.month") - climatology
    elif groupby == "dayofyear":
        anomaly = ds.groupby(f"{time_dim}.dayofyear") - climatology
    elif groupby == "season":
        anomaly = ds.groupby(f"{time_dim}.season") - climatology
    else:
        raise ValueError(f"Unknown groupby value: {groupby}")

    # Add attributes
    for var in anomaly.data_vars:
        original_attrs = ds[var].attrs.copy()
        anomaly[var].attrs = original_attrs
        anomaly[var].attrs["long_name"] = f"{original_attrs.get('long_name', var)} anomaly"
        anomaly[var].attrs["standard_name"] = f"{var}_anomaly"
        if reference_period:
            anomaly[var].attrs["reference_period"] = f"{reference_period[0]} to {reference_period[1]}"

    return anomaly


def compute_standardized_anomaly(
    ds: xr.Dataset,
    time_dim: str = "time",
    reference_period: tuple[str, str] | None = None,
    groupby: str = "month",
    climatology: xr.Dataset | None = None,
    climatology_std: xr.Dataset | None = None,
) -> xr.Dataset:
    """Compute standardized anomalies (z-scores) relative to climatology.

    Standardized Anomaly = (Value - Mean) / Standard Deviation

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        reference_period: Optional (start, end) dates for climatology
        groupby: How to group climatology ("month", "dayofyear", or "season")
        climatology: Pre-computed climatological mean (if None, computed from data)
        climatology_std: Pre-computed climatological std (if None, computed from data)

    Returns:
        Dataset with standardized anomaly values (dimensionless)
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Get reference data
    if reference_period:
        ref_ds = ds.sel({time_dim: slice(reference_period[0], reference_period[1])})
    else:
        ref_ds = ds

    # Compute climatology mean if not provided
    if climatology is None:
        climatology = compute_climatology(ref_ds, time_dim, groupby)

    # Compute climatology std if not provided
    if climatology_std is None:
        if groupby == "month":
            climatology_std = ref_ds.groupby(f"{time_dim}.month").std(dim=time_dim)
        elif groupby == "dayofyear":
            climatology_std = ref_ds.groupby(f"{time_dim}.dayofyear").std(dim=time_dim)
        elif groupby == "season":
            climatology_std = ref_ds.groupby(f"{time_dim}.season").std(dim=time_dim)
        else:
            raise ValueError(f"Unknown groupby value: {groupby}")

    # Compute anomaly
    if groupby == "month":
        anomaly = ds.groupby(f"{time_dim}.month") - climatology
        std_anomaly = anomaly.groupby(f"{time_dim}.month") / climatology_std
    elif groupby == "dayofyear":
        anomaly = ds.groupby(f"{time_dim}.dayofyear") - climatology
        std_anomaly = anomaly.groupby(f"{time_dim}.dayofyear") / climatology_std
    elif groupby == "season":
        anomaly = ds.groupby(f"{time_dim}.season") - climatology
        std_anomaly = anomaly.groupby(f"{time_dim}.season") / climatology_std
    else:
        raise ValueError(f"Unknown groupby value: {groupby}")

    # Add attributes
    for var in std_anomaly.data_vars:
        std_anomaly[var].attrs["long_name"] = f"Standardized {var} anomaly"
        std_anomaly[var].attrs["standard_name"] = f"{var}_standardized_anomaly"
        std_anomaly[var].attrs["units"] = "1"  # Dimensionless
        if reference_period:
            std_anomaly[var].attrs["reference_period"] = f"{reference_period[0]} to {reference_period[1]}"

    return std_anomaly


def compute_percentile_anomaly(
    ds: xr.Dataset,
    percentiles: xr.Dataset,
    time_dim: str = "time",
    groupby: str = "month",
) -> xr.Dataset:
    """Compute anomaly relative to percentile values.

    Returns the percentile rank of each value within the historical distribution.

    Args:
        ds: Input dataset with time dimension
        percentiles: Pre-computed percentile thresholds
        time_dim: Name of time dimension
        groupby: How percentiles were grouped

    Returns:
        Dataset with percentile rank values (0-100)
    """
    if "percentile" not in percentiles.dims:
        raise ValueError("Percentiles dataset must have 'percentile' dimension")

    pct_values = percentiles.percentile.values

    def find_percentile_rank(value: float, thresholds: np.ndarray, pcts: np.ndarray) -> float:
        """Find percentile rank by interpolation."""
        if np.isnan(value):
            return np.nan
        if value <= thresholds[0]:
            return pcts[0]
        if value >= thresholds[-1]:
            return pcts[-1]
        return float(np.interp(value, thresholds, pcts))

    result_vars = {}

    for var in ds.data_vars:
        if var not in percentiles:
            continue

        if groupby == "month":
            result = xr.apply_ufunc(
                find_percentile_rank,
                ds[var].groupby(f"{time_dim}.month"),
                percentiles[var],
                pct_values,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )
        else:
            # Simplified version without groupby
            result = xr.apply_ufunc(
                find_percentile_rank,
                ds[var],
                percentiles[var].mean(dim="month" if "month" in percentiles.dims else []),
                pct_values,
                vectorize=True,
                dask="parallelized",
                output_dtypes=[float],
            )

        result_vars[f"{var}_percentile_rank"] = result
        result_vars[f"{var}_percentile_rank"].attrs["long_name"] = f"Percentile rank of {var}"
        result_vars[f"{var}_percentile_rank"].attrs["units"] = "%"

    return xr.Dataset(result_vars)


def classify_anomaly_severity(
    standardized_anomaly: xr.Dataset,
) -> xr.Dataset:
    """Classify standardized anomalies into severity categories.

    Categories:
        -3: Extremely below normal (z < -2)
        -2: Severely below normal (-2 <= z < -1.5)
        -1: Moderately below normal (-1.5 <= z < -1)
         0: Near normal (-1 <= z <= 1)
         1: Moderately above normal (1 < z <= 1.5)
         2: Severely above normal (1.5 < z <= 2)
         3: Extremely above normal (z > 2)

    Args:
        standardized_anomaly: Dataset with standardized anomaly values

    Returns:
        Dataset with severity classification (-3 to 3)
    """
    result_vars = {}

    for var in standardized_anomaly.data_vars:
        z = standardized_anomaly[var]

        severity = xr.where(z < -2, -3,
                   xr.where(z < -1.5, -2,
                   xr.where(z < -1, -1,
                   xr.where(z <= 1, 0,
                   xr.where(z <= 1.5, 1,
                   xr.where(z <= 2, 2, 3))))))

        result_vars[f"{var}_severity"] = severity.astype(np.int8)
        result_vars[f"{var}_severity"].attrs["long_name"] = f"Anomaly severity of {var}"
        result_vars[f"{var}_severity"].attrs["flag_values"] = [-3, -2, -1, 0, 1, 2, 3]
        result_vars[f"{var}_severity"].attrs["flag_meanings"] = (
            "extremely_below severely_below moderately_below "
            "near_normal moderately_above severely_above extremely_above"
        )

    return xr.Dataset(result_vars)
