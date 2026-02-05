"""Temporal averaging functions for climate data.

Computes monthly, seasonal, and annual means from time series data.
"""

from typing import Literal

import numpy as np
import xarray as xr


# Season definitions (Northern Hemisphere convention)
SEASONS = {
    "DJF": [12, 1, 2],    # Winter
    "MAM": [3, 4, 5],     # Spring
    "JJA": [6, 7, 8],     # Summer
    "SON": [9, 10, 11],   # Autumn
}


def compute_monthly_mean(
    ds: xr.Dataset,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute monthly means from time series data.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension

    Returns:
        Dataset with monthly mean values
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Group by year-month and compute mean
    monthly = ds.resample({time_dim: "ME"}).mean()

    # Add attributes
    for var in monthly.data_vars:
        monthly[var].attrs["cell_methods"] = f"{time_dim}: mean (monthly)"

    return monthly


def compute_seasonal_mean(
    ds: xr.Dataset,
    time_dim: str = "time",
    seasons: dict[str, list[int]] | None = None,
) -> xr.Dataset:
    """Compute seasonal means from time series data.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        seasons: Optional custom season definitions (month lists)

    Returns:
        Dataset with seasonal mean values
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    if seasons is None:
        seasons = SEASONS

    # Group by season and compute mean
    # Note: xarray's built-in season grouping uses standard DJF/MAM/JJA/SON
    seasonal = ds.resample({time_dim: "QE-NOV"}).mean()

    # Add season coordinate
    season_labels = []
    for t in seasonal[time_dim].values:
        month = np.datetime64(t, "M").astype("datetime64[M]").astype(int) % 12 + 1
        for season_name, months in seasons.items():
            if month in months:
                season_labels.append(season_name)
                break
        else:
            season_labels.append("UNK")

    seasonal = seasonal.assign_coords(season=(time_dim, season_labels))

    for var in seasonal.data_vars:
        seasonal[var].attrs["cell_methods"] = f"{time_dim}: mean (seasonal)"

    return seasonal


def compute_annual_mean(
    ds: xr.Dataset,
    time_dim: str = "time",
) -> xr.Dataset:
    """Compute annual means from time series data.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension

    Returns:
        Dataset with annual mean values
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Group by year and compute mean
    annual = ds.resample({time_dim: "YE"}).mean()

    # Add year coordinate
    years = annual[time_dim].dt.year.values
    annual = annual.assign_coords(year=(time_dim, years))

    for var in annual.data_vars:
        annual[var].attrs["cell_methods"] = f"{time_dim}: mean (annual)"

    return annual


def compute_climatology(
    ds: xr.Dataset,
    time_dim: str = "time",
    groupby: Literal["month", "dayofyear", "season"] = "month",
    reference_period: tuple[str, str] | None = None,
) -> xr.Dataset:
    """Compute climatological mean (long-term average by month/day/season).

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        groupby: How to group the data ("month", "dayofyear", or "season")
        reference_period: Optional (start, end) dates for reference period

    Returns:
        Climatology dataset
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Slice to reference period if specified
    if reference_period:
        start, end = reference_period
        ds = ds.sel({time_dim: slice(start, end)})

    # Group and compute mean
    if groupby == "month":
        clim = ds.groupby(f"{time_dim}.month").mean(dim=time_dim)
    elif groupby == "dayofyear":
        clim = ds.groupby(f"{time_dim}.dayofyear").mean(dim=time_dim)
    elif groupby == "season":
        clim = ds.groupby(f"{time_dim}.season").mean(dim=time_dim)
    else:
        raise ValueError(f"Unknown groupby value: {groupby}")

    for var in clim.data_vars:
        clim[var].attrs["cell_methods"] = f"{time_dim}: mean over years"
        clim[var].attrs["climatology_groupby"] = groupby

    return clim


def compute_rolling_mean(
    ds: xr.Dataset,
    window: int,
    time_dim: str = "time",
    center: bool = True,
    min_periods: int | None = None,
) -> xr.Dataset:
    """Compute rolling/moving average along time dimension.

    Args:
        ds: Input dataset with time dimension
        window: Window size in time steps
        time_dim: Name of time dimension
        center: Whether to center the window
        min_periods: Minimum number of observations required

    Returns:
        Dataset with rolling mean values
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    if min_periods is None:
        min_periods = window // 2

    rolled = ds.rolling({time_dim: window}, center=center, min_periods=min_periods).mean()

    for var in rolled.data_vars:
        rolled[var].attrs["cell_methods"] = f"{time_dim}: mean (rolling window={window})"

    return rolled
