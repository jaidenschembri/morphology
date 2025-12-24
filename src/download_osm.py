"""Download city street graphs from OpenStreetMap using OSMnx."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import osmnx as ox

from . import city_registry, config


def raw_graph_path(city: city_registry.City, base_dir: Optional[Path] = None) -> Path:
    """Return the path for a city's raw GraphML artifact."""

    root = Path(base_dir) if base_dir is not None else config.DATA_DIR
    return root / "raw" / f"{city.slug}_raw.graphml"


def download_city_graph(city: city_registry.City, overwrite: bool = False, base_dir: Optional[Path] = None):
    """Download and cache a city's street network graph.

    If a cached GraphML exists and ``overwrite`` is False, the graph is loaded
    from disk instead of hitting the network.
    """

    target = raw_graph_path(city, base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        return ox.load_graphml(target)

    graph = ox.graph_from_place(city.query, network_type=config.DEFAULT_NETWORK_TYPE)
    ox.save_graphml(graph, target)
    return graph


def download_all_missing(overwrite: bool = False, base_dir: Optional[Path] = None, log_file: Optional[Path] = None) -> None:
    """Download graphs for all cities, skipping cached files unless overwritten."""

    config.ensure_output_dirs(base_dir)
    log_path = log_file or (config.LOG_DIR if base_dir is None else Path(base_dir) / "logs") / "download.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log:
        for city in city_registry.iter_cities(base_dir):
            path = raw_graph_path(city, base_dir)
            if path.exists() and not overwrite:
                log.write(f"SKIP {city.slug} cached at {path}\n")
                continue
            try:
                graph = download_city_graph(city, overwrite=overwrite, base_dir=base_dir)
                log.write(f"OK   {city.slug} saved to {path}\n")
                log.flush()
            except Exception as exc:  # noqa: BLE001
                log.write(f"FAIL {city.slug} error: {exc}\n")
                log.flush()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download street networks from OSM")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", help="City slug as defined in data/cities_list.txt")
    group.add_argument("--all", action="store_true", help="Download all cities from registry")
    parser.add_argument("--overwrite", action="store_true", help="Redownload even if cached")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.all:
        download_all_missing(overwrite=args.overwrite)
        print("Completed batch download (see output/logs/download.log)")
    else:
        try:
            city = city_registry.get_city(args.city)
        except KeyError as exc:
            raise SystemExit(f"Unknown city slug: {args.city}") from exc

        graph = download_city_graph(city, overwrite=args.overwrite)
        path = raw_graph_path(city)
        print(f"Saved {city.name} graph to {path}")


if __name__ == "__main__":
    main()
