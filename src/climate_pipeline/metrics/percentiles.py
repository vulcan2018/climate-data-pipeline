"""Climatological percentile computations.

Calculates percentile thresholds from historical data for anomaly detection
and classification of extreme events.
"""

import numpy as np
import xarray as xr


DEFAULT_PERCENTILES = [10, 25, 50, 75, 90, 95, 99]


def compute_climatological_percentiles(
    ds: xr.Dataset,
    percentiles: list[int] | None = None,
    time_dim: str = "time",
    reference_period: tuple[str, str] | None = None,
    groupby: str | None = "month",
) -> xr.Dataset:
    """Compute climatological percentiles from time series data.

    Args:
        ds: Input dataset with time dimension
        percentiles: List of percentile values to compute (default: [10,25,50,75,90,95,99])
        time_dim: Name of time dimension
        reference_period: Optional (start, end) dates for reference period
        groupby: How to group data before computing percentiles
                 - "month": compute percentiles for each month separately
                 - "dayofyear": compute percentiles for each day of year
                 - None: compute percentiles over entire time series

    Returns:
        Dataset with percentile values for each variable
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    if percentiles is None:
        percentiles = DEFAULT_PERCENTILES

    # Validate percentiles
    for p in percentiles:
        if not 0 <= p <= 100:
            raise ValueError(f"Percentile must be between 0 and 100, got {p}")

    # Slice to reference period if specified
    if reference_period:
        start, end = reference_period
        ds = ds.sel({time_dim: slice(start, end)})

    # Compute percentiles
    if groupby == "month":
        result = ds.groupby(f"{time_dim}.month").quantile(
            [p / 100 for p in percentiles], dim=time_dim
        )
    elif groupby == "dayofyear":
        result = ds.groupby(f"{time_dim}.dayofyear").quantile(
            [p / 100 for p in percentiles], dim=time_dim
        )
    elif groupby is None:
        result = ds.quantile([p / 100 for p in percentiles], dim=time_dim)
    else:
        raise ValueError(f"Unknown groupby value: {groupby}")

    # Rename quantile dimension
    result = result.rename({"quantile": "percentile"})
    result = result.assign_coords(percentile=percentiles)

    # Add attributes
    for var in result.data_vars:
        result[var].attrs["percentiles"] = percentiles
        result[var].attrs["groupby"] = groupby if groupby else "all"
        if reference_period:
            result[var].attrs["reference_period"] = f"{reference_period[0]} to {reference_period[1]}"

    return result


def compute_exceedance_frequency(
    ds: xr.Dataset,
    thresholds: xr.Dataset,
    time_dim: str = "time",
    percentile: int = 90,
) -> xr.Dataset:
    """Compute frequency of exceedance above percentile thresholds.

    Args:
        ds: Input dataset with time dimension
        thresholds: Dataset with percentile thresholds (from compute_climatological_percentiles)
        time_dim: Name of time dimension
        percentile: Which percentile threshold to use

    Returns:
        Dataset with exceedance frequency (fraction of time exceeding threshold)
    """
    # Select the specific percentile threshold
    if "percentile" in thresholds.dims:
        threshold = thresholds.sel(percentile=percentile)
    else:
        threshold = thresholds

    # Count exceedances
    exceedances = ds > threshold
    frequency = exceedances.mean(dim=time_dim)

    # Add attributes
    for var in frequency.data_vars:
        frequency[var].attrs["long_name"] = f"Exceedance frequency above {percentile}th percentile"
        frequency[var].attrs["units"] = "1"

    return frequency


def classify_by_percentile(
    ds: xr.Dataset,
    thresholds: xr.Dataset,
    time_dim: str = "time",
) -> xr.Dataset:
    """Classify data into percentile bins.

    Args:
        ds: Input dataset with time dimension
        thresholds: Dataset with percentile thresholds

    Returns:
        Dataset with percentile bin classification (0-100)
    """
    if "percentile" not in thresholds.dims:
        raise ValueError("Thresholds must have a 'percentile' dimension")

    percentiles = thresholds.percentile.values

    # Initialize result
    result = xr.zeros_like(ds, dtype=np.int8)

    # Classify each value into percentile bins
    for var in ds.data_vars:
        var_thresholds = thresholds[var]
        for i, p in enumerate(percentiles):
            mask = ds[var] >= var_thresholds.sel(percentile=p)
            result[var] = result[var].where(~mask, p)

    # Add attributes
    for var in result.data_vars:
        result[var].attrs["long_name"] = "Percentile classification"
        result[var].attrs["percentile_bins"] = list(percentiles)

    return result


def compute_return_periods(
    ds: xr.Dataset,
    time_dim: str = "time",
    return_periods: list[int] | None = None,
) -> xr.Dataset:
    """Compute values corresponding to return periods.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        return_periods: List of return periods in years (default: [2, 5, 10, 25, 50, 100])

    Returns:
        Dataset with values for each return period
    """
    if return_periods is None:
        return_periods = [2, 5, 10, 25, 50, 100]

    # Convert return periods to exceedance probabilities
    # Return period T years = probability 1/T of exceedance per year
    probabilities = [1 - 1 / t for t in return_periods]

    # Compute annual maxima
    annual_max = ds.resample({time_dim: "YE"}).max()

    # Compute quantiles of annual maxima
    result = annual_max.quantile(probabilities, dim=time_dim)
    result = result.rename({"quantile": "return_period"})
    result = result.assign_coords(return_period=return_periods)

    # Add attributes
    for var in result.data_vars:
        result[var].attrs["long_name"] = "Return period values"
        result[var].attrs["return_periods_years"] = return_periods

    return result
