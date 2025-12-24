"""Construct and clean street-network graphs from raw OSM downloads.

This module loads raw GraphMLs, projects them to a metric CRS, simplifies,
annotates edges/nodes, validates basic assumptions, and writes processed
artifacts for downstream metrics and visuals.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional

import osmnx as ox

from . import city_registry, config


def raw_graph_path(city: city_registry.City, base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir) if base_dir is not None else config.DATA_DIR
    return root / "raw" / f"{city.slug}_raw.graphml"


def processed_graph_path(city: city_registry.City, base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir) if base_dir is not None else config.DATA_DIR
    return root / "processed" / f"{city.slug}_processed.graphml"


def load_raw_graph(city: city_registry.City, base_dir: Optional[Path] = None):
    path = raw_graph_path(city, base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Raw graph missing for {city.slug}: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Raw graph file is empty for {city.slug}: {path}")
    return ox.load_graphml(path)


def project_graph_to_meters(G):
    """Project to a local metric CRS (UTM chosen by OSMnx)."""

    return ox.project_graph(G)


def simplify_graph(G):
    """Simplify graph topology while preserving geometry."""

    if G.graph.get("simplified"):
        return G
    return ox.simplify_graph(G, remove_rings=True)


def annotate_bearings(G):
    """Add edge bearings to an unprojected graph (in-place)."""

    if any("bearing" in d for _, _, d in G.edges(data=True)):
        return G
    G = ox.bearing.add_edge_bearings(G)
    return G


def annotate_edge_lengths(G):
    """Add edge lengths in meters (in-place)."""

    G = ox.distance.add_edge_lengths(G)
    return G


def drop_zero_length_edges(G, min_length: float = 1e-3):
    """Remove edges with non-positive or tiny lengths."""

    to_remove = [(u, v, k) for u, v, k, d in G.edges(keys=True, data=True) if d.get("length", 0) <= min_length]
    G.remove_edges_from(to_remove)
    return G


def annotate_nodes(G):
    """Add degree and street counts (in-place)."""

    street_counts = ox.stats.count_streets_per_node(G)
    for node, degree in G.degree():
        G.nodes[node]["degree"] = degree
        if node in street_counts:
            G.nodes[node]["street_count"] = street_counts[node]
    return G


def validate_graph(G):
    errors = []

    if len(G) == 0 or G.number_of_edges() == 0:
        errors.append("Graph is empty after processing")

    crs = G.graph.get("crs")
    if crs is None:
        errors.append("Graph CRS missing")
    elif not getattr(crs, "is_projected", False):
        errors.append(f"Graph CRS not projected: {crs}")

    zero_length = [d for _, _, d in G.edges(data=True) if d.get("length", 0) <= 0]
    if zero_length:
        errors.append(f"Found {len(zero_length)} edges with non-positive length")

    nan_lengths = [d for _, _, d in G.edges(data=True) if not math.isfinite(d.get("length", 0))]
    if nan_lengths:
        errors.append(f"Found {len(nan_lengths)} edges with non-finite length")

    bad_geom = []
    for _, _, data in G.edges(data=True):
        geom = data.get("geometry")
        if geom is not None and not geom.is_valid:
            bad_geom.append(geom)
    if bad_geom:
        errors.append(f"Found {len(bad_geom)} invalid edge geometries")

    if errors:
        raise ValueError("; ".join(errors))


def process_city(city: city_registry.City, overwrite: bool = False, base_dir: Optional[Path] = None) -> Path:
    G_raw = load_raw_graph(city, base_dir)
    G_bear = annotate_bearings(G_raw)
    G_proj = project_graph_to_meters(G_bear)
    G_simple = simplify_graph(G_proj)
    G_edges = annotate_edge_lengths(G_simple)
    G_edges = drop_zero_length_edges(G_edges)
    G_nodes = annotate_nodes(G_edges)
    validate_graph(G_nodes)

    processed_path = processed_graph_path(city, base_dir)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G_nodes, processed_path)

    _write_metadata(city, G_nodes, base_dir)
    return processed_path


def _write_metadata(city: city_registry.City, G, base_dir: Optional[Path]) -> None:
    config.ensure_output_dirs(base_dir)
    meta_path = config.output_path("reports", f"{city.slug}_build_metadata.json", base_dir=base_dir)
    stats = {
        "city": city.slug,
        "nodes": len(G),
        "edges": G.number_of_edges(),
        "crs": str(G.graph.get("crs")),
        "total_edge_length_m": sum(d.get("length", 0) for _, _, d in G.edges(data=True)),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/clean processed street graphs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", help="City slug as defined in data/cities_list.txt")
    group.add_argument("--all", action="store_true", help="Process all cities")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild even if cached")
    return parser.parse_args()


def _log(message: str, base_dir: Optional[Path] = None) -> None:
    config.ensure_output_dirs(base_dir)
    log_path = config.output_path("logs", "build.log", base_dir=base_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(message + "\n")


def main() -> None:
    args = _parse_args()
    if args.all:
        for city in city_registry.iter_cities():
            dest = processed_graph_path(city)
            if dest.exists() and not args.overwrite:
                _log(f"SKIP {city.slug} cached at {dest}")
                continue
            try:
                path = process_city(city, overwrite=args.overwrite)
                _log(f"OK   {city.slug} -> {path}")
            except Exception as exc:  # noqa: BLE001
                _log(f"FAIL {city.slug} error: {exc}")
                print(f"Failed {city.slug}: {exc}")
                break
        print("Completed build; see output/logs/build.log")
    else:
        try:
            city = city_registry.get_city(args.city)
        except KeyError as exc:
            raise SystemExit(f"Unknown city slug: {args.city}") from exc

        dest = processed_graph_path(city)
        if dest.exists() and not args.overwrite:
            print(f"Processed graph exists, skipping: {dest}")
            return

        path = process_city(city, overwrite=args.overwrite)
        _log(f"OK   {city.slug} -> {path}")
        print(f"Saved processed graph to {path}")


if __name__ == "__main__":
    main()
