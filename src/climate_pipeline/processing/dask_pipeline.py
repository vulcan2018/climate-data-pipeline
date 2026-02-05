"""Dask-based parallel processing pipeline for climate data.

Provides utilities for building and executing processing pipelines
that leverage Dask for parallel and distributed computation.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import dask
import xarray as xr
from dask.diagnostics import ProgressBar


@dataclass
class PipelineStep:
    """A single step in the processing pipeline."""

    name: str
    func: Callable[[xr.Dataset], xr.Dataset]
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Pipeline:
    """A sequence of processing steps to apply to a dataset."""

    steps: list[PipelineStep] = field(default_factory=list)
    name: str = "pipeline"

    def add_step(
        self,
        name: str,
        func: Callable[[xr.Dataset], xr.Dataset],
        **kwargs: Any,
    ) -> "Pipeline":
        """Add a processing step to the pipeline.

        Args:
            name: Step name for logging/tracking
            func: Function that takes and returns an xr.Dataset
            **kwargs: Additional arguments passed to the function

        Returns:
            Self for method chaining
        """
        self.steps.append(PipelineStep(name=name, func=func, kwargs=kwargs))
        return self

    def execute(
        self,
        ds: xr.Dataset,
        progress: bool = True,
    ) -> xr.Dataset:
        """Execute the pipeline on a dataset.

        Args:
            ds: Input dataset
            progress: Whether to show progress bar

        Returns:
            Processed dataset
        """
        result = ds

        for step in self.steps:
            # Apply the step function with kwargs
            if step.kwargs:
                result = step.func(result, **step.kwargs)
            else:
                result = step.func(result)

        # Compute if dask-backed
        if hasattr(result, "compute"):
            if progress:
                with ProgressBar():
                    result = result.compute()
            else:
                result = result.compute()

        return result


def create_pipeline(name: str = "pipeline") -> Pipeline:
    """Create a new processing pipeline.

    Args:
        name: Pipeline name for logging

    Returns:
        Empty Pipeline instance
    """
    return Pipeline(name=name)


def execute_parallel(
    datasets: list[xr.Dataset],
    func: Callable[[xr.Dataset], xr.Dataset],
    scheduler: str = "threads",
    n_workers: int | None = None,
    progress: bool = True,
) -> list[xr.Dataset]:
    """Execute a function on multiple datasets in parallel.

    Args:
        datasets: List of input datasets
        func: Function to apply to each dataset
        scheduler: Dask scheduler ("threads", "processes", or "synchronous")
        n_workers: Number of workers (None = auto)
        progress: Whether to show progress bar

    Returns:
        List of processed datasets
    """
    # Create delayed tasks
    delayed_results = [dask.delayed(func)(ds) for ds in datasets]

    # Configure scheduler
    scheduler_kwargs: dict[str, Any] = {}
    if n_workers:
        scheduler_kwargs["num_workers"] = n_workers

    # Execute
    if progress:
        with ProgressBar():
            results = dask.compute(*delayed_results, scheduler=scheduler, **scheduler_kwargs)
    else:
        results = dask.compute(*delayed_results, scheduler=scheduler, **scheduler_kwargs)

    return list(results)


def map_blocks(
    ds: xr.Dataset,
    func: Callable[[xr.DataArray], xr.DataArray],
    variables: list[str] | None = None,
    template: xr.Dataset | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """Apply a function to each chunk of the dataset.

    Args:
        ds: Input dataset
        func: Function to apply to each block
        variables: Variables to process (None = all)
        template: Template for output structure
        **kwargs: Additional arguments for map_blocks

    Returns:
        Processed dataset
    """
    if variables is None:
        variables = list(ds.data_vars)

    result_vars = {}
    for var in variables:
        if var in ds.data_vars:
            result_vars[var] = ds[var].map_blocks(func, template=template, **kwargs)

    return xr.Dataset(result_vars, coords=ds.coords, attrs=ds.attrs)


def reduce_along_dim(
    ds: xr.Dataset,
    dim: str,
    func: Callable[[xr.DataArray], xr.DataArray],
    variables: list[str] | None = None,
) -> xr.Dataset:
    """Apply a reduction function along a dimension.

    Args:
        ds: Input dataset
        dim: Dimension to reduce
        func: Reduction function
        variables: Variables to process (None = all)

    Returns:
        Reduced dataset
    """
    if variables is None:
        variables = list(ds.data_vars)

    result_vars = {}
    for var in variables:
        if var in ds.data_vars and dim in ds[var].dims:
            result_vars[var] = func(ds[var], dim=dim)
        elif var in ds.data_vars:
            # Variable doesn't have this dimension, keep as-is
            result_vars[var] = ds[var]

    return xr.Dataset(result_vars, attrs=ds.attrs)


def apply_to_groups(
    ds: xr.Dataset,
    group_dim: str,
    groupby: str,
    func: Callable[[xr.Dataset], xr.Dataset],
) -> xr.Dataset:
    """Apply a function to groups within the dataset.

    Args:
        ds: Input dataset
        group_dim: Dimension to group by
        groupby: Grouping specification (e.g., "time.month")
        func: Function to apply to each group

    Returns:
        Dataset with function applied to each group
    """
    return ds.groupby(groupby).map(func)


class LazyPipeline:
    """A pipeline that builds computation graphs without executing.

    Useful for building complex workflows that are executed once at the end.
    """

    def __init__(self, ds: xr.Dataset):
        """Initialize with input dataset.

        Args:
            ds: Input dataset (should be dask-backed for lazy evaluation)
        """
        self.ds = ds
        self.history: list[str] = []

    def apply(
        self,
        func: Callable[[xr.Dataset], xr.Dataset],
        name: str = "transform",
        **kwargs: Any,
    ) -> "LazyPipeline":
        """Apply a transformation lazily.

        Args:
            func: Transformation function
            name: Step name for history
            **kwargs: Arguments for the function

        Returns:
            Self for method chaining
        """
        if kwargs:
            self.ds = func(self.ds, **kwargs)
        else:
            self.ds = func(self.ds)
        self.history.append(name)
        return self

    def compute(self, progress: bool = True) -> xr.Dataset:
        """Execute the built computation graph.

        Args:
            progress: Whether to show progress bar

        Returns:
            Computed dataset
        """
        if progress:
            with ProgressBar():
                return self.ds.compute()
        return self.ds.compute()

    def persist(self) -> "LazyPipeline":
        """Persist intermediate results to cluster memory.

        Useful for caching intermediate results that will be reused.

        Returns:
            Self with persisted data
        """
        self.ds = self.ds.persist()
        return self

    def get_graph_info(self) -> dict[str, Any]:
        """Get information about the computation graph.

        Returns:
            Dictionary with graph statistics
        """
        # Get dask graph info
        if hasattr(self.ds, "__dask_graph__"):
            graph = self.ds.__dask_graph__()
            return {
                "n_tasks": len(graph),
                "history": self.history,
                "variables": list(self.ds.data_vars),
            }
        return {
            "n_tasks": 0,
            "history": self.history,
            "variables": list(self.ds.data_vars),
            "note": "Dataset is not dask-backed",
        }
