"""Data ingestion modules for NetCDF reading and ARCO conversion."""

from .netcdf_reader import read_netcdf, read_netcdf_lazy
from .arco_converter import convert_to_zarr, optimize_chunks

__all__ = ["read_netcdf", "read_netcdf_lazy", "convert_to_zarr", "optimize_chunks"]
