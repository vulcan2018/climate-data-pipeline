#!/usr/bin/env python3
"""
STAC API Demonstration Script

This script demonstrates FIRA Software's capability with STAC
(SpatioTemporal Asset Catalog) API implementation, as required
for the CJS2_220b_bis ARCO tender.

Demonstrates:
- STAC Catalog creation
- STAC Collection definition
- STAC Item generation
- Search/query capabilities

Author: S. Kalogerakos
Company: FIRA Software Ltd
Date: February 2026
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import hashlib

# Check for pystac dependency
PYSTAC_AVAILABLE = True
try:
    import pystac
    from pystac import Catalog, Collection, Item, Asset, MediaType
    from pystac.extensions.eo import EOExtension
except ImportError:
    PYSTAC_AVAILABLE = False


def create_arco_catalog(
    catalog_id: str = "ecmwf-arco-demo",
    title: str = "ECMWF ARCO Data Catalog Demo",
    description: str = "Demonstration STAC catalog for ARCO datasets"
) -> 'Catalog':
    """
    Create a STAC Catalog for ARCO datasets.

    This demonstrates catalog structure for organising
    climate data collections.

    Args:
        catalog_id: Unique catalog identifier
        title: Human-readable title
        description: Catalog description

    Returns:
        pystac Catalog object
    """
    catalog = Catalog(
        id=catalog_id,
        title=title,
        description=description,
        extra_fields={
            "stac_extensions": [],
            "keywords": ["ARCO", "climate", "Zarr", "Copernicus", "ECMWF"],
            "providers": [
                {
                    "name": "FIRA Software Ltd",
                    "description": "ARCO data processing demonstration",
                    "roles": ["processor"],
                    "url": "https://firasoftware.com"
                },
                {
                    "name": "ECMWF",
                    "description": "European Centre for Medium-Range Weather Forecasts",
                    "roles": ["host", "producer"],
                    "url": "https://www.ecmwf.int"
                }
            ]
        }
    )

    return catalog


def create_era5_collection(
    collection_id: str = "era5-arco",
    start_date: str = "1979-01-01",
    end_date: str = "2024-12-31"
) -> 'Collection':
    """
    Create a STAC Collection for ERA5-like ARCO data.

    Demonstrates collection metadata structure for
    climate reanalysis datasets.

    Args:
        collection_id: Unique collection identifier
        start_date: Temporal extent start
        end_date: Temporal extent end

    Returns:
        pystac Collection object
    """
    spatial_extent = pystac.SpatialExtent(bboxes=[[-180, -90, 180, 90]])

    temporal_extent = pystac.TemporalExtent(
        intervals=[[
            datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc),
            datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
        ]]
    )

    extent = pystac.Extent(spatial=spatial_extent, temporal=temporal_extent)

    collection = Collection(
        id=collection_id,
        title="ERA5 Reanalysis - ARCO Format",
        description="""
        ERA5 is the fifth generation ECMWF atmospheric reanalysis of the
        global climate. This collection provides ERA5 data in Analysis-Ready,
        Cloud-Optimised (ARCO) Zarr format for efficient cloud-native access.

        Key features:
        - Hourly data from 1979 to present
        - 0.25° horizontal resolution
        - 137 pressure levels
        - Optimised chunking for time-series and spatial access
        """,
        extent=extent,
        license="proprietary",
        keywords=["ERA5", "reanalysis", "climate", "ARCO", "Zarr"],
        providers=[
            pystac.Provider(
                name="ECMWF",
                roles=["producer", "licensor"],
                url="https://www.ecmwf.int"
            ),
            pystac.Provider(
                name="Copernicus Climate Change Service",
                roles=["host"],
                url="https://cds.climate.copernicus.eu"
            )
        ],
        extra_fields={
            "cube:dimensions": {
                "time": {
                    "type": "temporal",
                    "extent": [start_date, end_date],
                    "step": "P1H"
                },
                "latitude": {
                    "type": "spatial",
                    "axis": "y",
                    "extent": [-90, 90],
                    "step": 0.25,
                    "reference_system": "EPSG:4326"
                },
                "longitude": {
                    "type": "spatial",
                    "axis": "x",
                    "extent": [-180, 180],
                    "step": 0.25,
                    "reference_system": "EPSG:4326"
                }
            },
            "cube:variables": {
                "2m_temperature": {
                    "type": "data",
                    "dimensions": ["time", "latitude", "longitude"],
                    "unit": "K"
                },
                "10m_u_wind": {
                    "type": "data",
                    "dimensions": ["time", "latitude", "longitude"],
                    "unit": "m/s"
                },
                "10m_v_wind": {
                    "type": "data",
                    "dimensions": ["time", "latitude", "longitude"],
                    "unit": "m/s"
                },
                "total_precipitation": {
                    "type": "data",
                    "dimensions": ["time", "latitude", "longitude"],
                    "unit": "m"
                }
            },
            "arco:format": "zarr",
            "arco:chunk_scheme": "balanced",
            "arco:access_latency_target_ms": 2000
        }
    )

    return collection


def create_zarr_item(
    item_id: str,
    zarr_path: str,
    bbox: list = [-180, -90, 180, 90],
    datetime_str: str = "2024-01-01T00:00:00Z",
    collection: Optional['Collection'] = None
) -> 'Item':
    """
    Create a STAC Item for a Zarr store.

    Demonstrates item-level metadata for individual
    ARCO datasets.

    Args:
        item_id: Unique item identifier
        zarr_path: Path/URL to Zarr store
        bbox: Bounding box [west, south, east, north]
        datetime_str: Item datetime
        collection: Parent collection

    Returns:
        pystac Item object
    """
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
        ]]
    }

    item = Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime.fromisoformat(datetime_str.replace('Z', '+00:00')),
        properties={
            "title": f"ARCO Dataset: {item_id}",
            "description": "ARCO Zarr store with climate data",
            "arco:chunk_size_mb": 10,
            "arco:compression": "zstd",
            "arco:compression_level": 3
        }
    )

    # Add Zarr asset
    item.add_asset(
        "zarr",
        Asset(
            href=zarr_path,
            title="Zarr Store",
            description="Cloud-optimised Zarr data store",
            media_type="application/vnd+zarr",
            roles=["data"],
            extra_fields={
                "xarray:open_kwargs": {
                    "engine": "zarr",
                    "chunks": "auto"
                }
            }
        )
    )

    # Add metadata asset
    item.add_asset(
        "metadata",
        Asset(
            href=zarr_path + "/.zattrs",
            title="Zarr Attributes",
            description="Root-level Zarr attributes",
            media_type="application/json",
            roles=["metadata"]
        )
    )

    if collection:
        item.collection_id = collection.id

    return item


def generate_stac_api_spec() -> dict:
    """
    Generate example STAC API specification.

    Shows the API structure we would implement for
    ARCO data discovery.

    Returns:
        Dictionary with API specification
    """
    return {
        "openapi": "3.0.3",
        "info": {
            "title": "ARCO STAC API",
            "description": "SpatioTemporal Asset Catalog API for ARCO datasets",
            "version": "1.0.0",
            "contact": {
                "name": "FIRA Software Ltd",
                "url": "https://firasoftware.com"
            }
        },
        "servers": [
            {
                "url": "https://arco-stac.example.ecmwf.int",
                "description": "ARCO STAC API Server"
            }
        ],
        "paths": {
            "/": {
                "get": {
                    "summary": "Landing page",
                    "description": "Returns the root STAC Catalog"
                }
            },
            "/collections": {
                "get": {
                    "summary": "List collections",
                    "description": "Returns all available ARCO collections"
                }
            },
            "/collections/{collection_id}": {
                "get": {
                    "summary": "Get collection",
                    "description": "Returns a specific collection by ID"
                }
            },
            "/collections/{collection_id}/items": {
                "get": {
                    "summary": "List items",
                    "description": "Returns items in a collection"
                }
            },
            "/search": {
                "get": {
                    "summary": "Search items",
                    "description": "Search across all collections"
                },
                "post": {
                    "summary": "Search items (POST)",
                    "description": "Search with complex filters"
                }
            }
        },
        "conformsTo": [
            "https://api.stacspec.org/v1.0.0/core",
            "https://api.stacspec.org/v1.0.0/collections",
            "https://api.stacspec.org/v1.0.0/item-search"
        ]
    }


def run_stac_demonstration(output_dir: str = './stac_demo_output') -> dict:
    """
    Run full STAC demonstration.

    Creates a complete STAC catalog structure demonstrating
    our capability for the ARCO tender.

    Args:
        output_dir: Output directory for STAC files

    Returns:
        Dictionary with demonstration results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("STAC API Demonstration")
    print("=" * 50)
    print("FIRA Software Ltd - CJS2_220b_bis Tender")
    print("=" * 50)

    results = {}

    if not PYSTAC_AVAILABLE:
        # Generate static examples without pystac
        print("\npystac not installed - generating static examples")

        results['catalog'] = {
            "type": "Catalog",
            "stac_version": "1.0.0",
            "id": "ecmwf-arco-demo",
            "title": "ECMWF ARCO Data Catalog Demo",
            "description": "Demonstration STAC catalog for ARCO datasets"
        }

        results['collection'] = {
            "type": "Collection",
            "stac_version": "1.0.0",
            "id": "era5-arco",
            "title": "ERA5 Reanalysis - ARCO Format",
            "extent": {
                "spatial": {"bbox": [[-180, -90, 180, 90]]},
                "temporal": {"interval": [["1979-01-01T00:00:00Z", "2024-12-31T23:59:59Z"]]}
            }
        }

        results['item'] = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "era5-2024-01",
            "geometry": {"type": "Polygon", "coordinates": [[[-180,-90],[180,-90],[180,90],[-180,90],[-180,-90]]]},
            "properties": {"datetime": "2024-01-01T00:00:00Z"}
        }

    else:
        # Full demonstration with pystac
        print("\n1. Creating STAC Catalog...")
        catalog = create_arco_catalog()
        results['catalog'] = catalog.to_dict()
        print(f"   Catalog ID: {catalog.id}")

        print("\n2. Creating ERA5 Collection...")
        collection = create_era5_collection()
        catalog.add_child(collection)
        results['collection'] = collection.to_dict()
        print(f"   Collection ID: {collection.id}")

        print("\n3. Creating sample Items...")
        items = []
        for month in range(1, 4):
            item_id = f"era5-2024-{month:02d}"
            item = create_zarr_item(
                item_id=item_id,
                zarr_path=f"s3://arco-data-lake/era5/2024/{month:02d}.zarr",
                datetime_str=f"2024-{month:02d}-01T00:00:00Z",
                collection=collection
            )
            collection.add_item(item)
            items.append(item.to_dict())
            print(f"   Created item: {item_id}")

        results['items'] = items

        # Save catalog
        catalog_path = output_path / 'catalog'
        catalog.normalize_and_save(str(catalog_path), pystac.CatalogType.SELF_CONTAINED)
        print(f"\n   Saved catalog to: {catalog_path}")

    # Generate API spec
    print("\n4. Generating STAC API specification...")
    api_spec = generate_stac_api_spec()
    results['api_spec'] = api_spec

    api_spec_path = output_path / 'stac-api-spec.json'
    with open(api_spec_path, 'w') as f:
        json.dump(api_spec, f, indent=2)
    print(f"   Saved API spec to: {api_spec_path}")

    # Save results
    results_path = output_path / 'stac_demo_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n✓ Results saved to: {results_path}")
    print("=" * 50)

    return results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='STAC API Demonstration - FIRA Software Ltd'
    )
    parser.add_argument(
        '--output-dir', '-o',
        default='./stac_demo_output',
        help='Output directory for STAC files'
    )

    args = parser.parse_args()
    run_stac_demonstration(args.output_dir)


if __name__ == '__main__':
    main()
