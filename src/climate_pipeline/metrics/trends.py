"""Linear trend computation for climate time series.

Computes trends, trend significance, and confidence intervals using
ordinary least squares regression.
"""

import numpy as np
import xarray as xr
from scipy import stats


def compute_linear_trend(
    ds: xr.Dataset,
    time_dim: str = "time",
    reference_period: tuple[str, str] | None = None,
) -> xr.Dataset:
    """Compute linear trend (slope) for each grid point.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        reference_period: Optional (start, end) dates for trend calculation

    Returns:
        Dataset with trend values (units per year)
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Slice to reference period if specified
    if reference_period:
        start, end = reference_period
        ds = ds.sel({time_dim: slice(start, end)})

    # Convert time to numeric (years since start)
    time_vals = ds[time_dim].values
    t0 = np.datetime64(time_vals[0], "ns")
    time_numeric = (time_vals.astype("datetime64[ns]") - t0).astype(float)
    time_years = time_numeric / (365.25 * 24 * 3600 * 1e9)  # Convert ns to years

    # Create time coordinate for regression
    ds = ds.assign_coords(_time_years=(time_dim, time_years))

    # Compute trend using polyfit
    trend = ds.polyfit(dim=time_dim, deg=1)

    # Extract slope (coefficient of degree 1)
    result_vars = {}
    for var in ds.data_vars:
        if var == "_time_years":
            continue
        coef_var = f"{var}_polyfit_coefficients"
        if coef_var in trend:
            # Select the slope (degree 1 coefficient)
            slope = trend[coef_var].sel(degree=1)
            result_vars[var] = slope
            result_vars[var].attrs["long_name"] = f"Linear trend of {var}"
            result_vars[var].attrs["units"] = f"{ds[var].attrs.get('units', 'units')} per year"

    result = xr.Dataset(result_vars)
    result.attrs["trend_method"] = "ordinary least squares"
    if reference_period:
        result.attrs["trend_period"] = f"{reference_period[0]} to {reference_period[1]}"

    return result


def compute_trend_significance(
    ds: xr.Dataset,
    time_dim: str = "time",
    alpha: float = 0.05,
) -> xr.Dataset:
    """Compute trend significance using Mann-Kendall test.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        alpha: Significance level (default 0.05 = 95% confidence)

    Returns:
        Dataset with:
        - trend: Sen's slope (robust trend estimator)
        - p_value: Two-tailed p-value
        - significant: Boolean mask where trend is significant
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    def mann_kendall_trend(data: np.ndarray) -> tuple[float, float, bool]:
        """Compute Mann-Kendall trend test for a single time series."""
        n = len(data)
        if n < 4:
            return np.nan, np.nan, False

        # Remove NaN values
        valid = ~np.isnan(data)
        if valid.sum() < 4:
            return np.nan, np.nan, False

        x = data[valid]
        n = len(x)

        # Compute S statistic
        s = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                s += np.sign(x[j] - x[i])

        # Compute variance of S
        unique, counts = np.unique(x, return_counts=True)
        ties = counts[counts > 1]
        var_s = (n * (n - 1) * (2 * n + 5)) / 18
        if len(ties) > 0:
            for t in ties:
                var_s -= t * (t - 1) * (2 * t + 5) / 18

        # Compute Z statistic
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0

        # Two-tailed p-value
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))

        # Sen's slope
        slopes = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                slopes.append((x[j] - x[i]) / (j - i))
        sen_slope = np.median(slopes) if slopes else np.nan

        return sen_slope, p_value, p_value < alpha

    # Apply Mann-Kendall to each grid point
    result_vars = {}

    for var in ds.data_vars:
        da = ds[var]

        # Apply along time dimension
        trend, p_value, significant = xr.apply_ufunc(
            mann_kendall_trend,
            da,
            input_core_dims=[[time_dim]],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, bool],
        )

        result_vars[f"{var}_trend"] = trend
        result_vars[f"{var}_trend"].attrs["long_name"] = f"Sen's slope of {var}"

        result_vars[f"{var}_p_value"] = p_value
        result_vars[f"{var}_p_value"].attrs["long_name"] = "Mann-Kendall p-value"

        result_vars[f"{var}_significant"] = significant
        result_vars[f"{var}_significant"].attrs["long_name"] = f"Significant trend at alpha={alpha}"

    result = xr.Dataset(result_vars)
    result.attrs["test"] = "Mann-Kendall"
    result.attrs["alpha"] = alpha

    return result


def compute_trend_with_confidence(
    ds: xr.Dataset,
    time_dim: str = "time",
    confidence: float = 0.95,
) -> xr.Dataset:
    """Compute linear trend with confidence intervals.

    Args:
        ds: Input dataset with time dimension
        time_dim: Name of time dimension
        confidence: Confidence level (default 0.95 = 95%)

    Returns:
        Dataset with trend, lower bound, and upper bound
    """
    if time_dim not in ds.dims:
        raise ValueError(f"Time dimension '{time_dim}' not found in dataset")

    # Convert time to numeric (years)
    time_vals = ds[time_dim].values
    t0 = np.datetime64(time_vals[0], "ns")
    time_numeric = (time_vals.astype("datetime64[ns]") - t0).astype(float)
    time_years = time_numeric / (365.25 * 24 * 3600 * 1e9)

    def linear_regression_ci(
        y: np.ndarray, x: np.ndarray, conf: float
    ) -> tuple[float, float, float]:
        """Compute linear regression with confidence interval."""
        valid = ~np.isnan(y)
        if valid.sum() < 3:
            return np.nan, np.nan, np.nan

        x_valid = x[valid]
        y_valid = y[valid]

        n = len(y_valid)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)

        # Confidence interval for slope
        t_crit = stats.t.ppf((1 + conf) / 2, n - 2)
        margin = t_crit * std_err

        return slope, slope - margin, slope + margin

    result_vars = {}

    for var in ds.data_vars:
        da = ds[var]

        trend, lower, upper = xr.apply_ufunc(
            linear_regression_ci,
            da,
            time_years,
            confidence,
            input_core_dims=[[time_dim], [time_dim], []],
            output_core_dims=[[], [], []],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float, float, float],
        )

        result_vars[f"{var}_trend"] = trend
        result_vars[f"{var}_trend"].attrs["long_name"] = f"Linear trend of {var}"
        result_vars[f"{var}_trend"].attrs["units"] = f"{da.attrs.get('units', 'units')} per year"

        result_vars[f"{var}_trend_lower"] = lower
        result_vars[f"{var}_trend_lower"].attrs["long_name"] = f"Trend lower bound ({confidence*100:.0f}%)"

        result_vars[f"{var}_trend_upper"] = upper
        result_vars[f"{var}_trend_upper"].attrs["long_name"] = f"Trend upper bound ({confidence*100:.0f}%)"

    result = xr.Dataset(result_vars)
    result.attrs["confidence_level"] = confidence

    return result
