"""Compute per-city graph metrics from cleaned street networks."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional

import networkx as nx
import numpy as np

from . import config


def load_processed_graph(path: Path) -> nx.MultiDiGraph:
    if not path.exists():
        raise FileNotFoundError(f"Processed graph missing: {path}")
    G = nx.read_graphml(path)
    _coerce_numeric_attrs(G)
    return G


def _coerce_numeric_attrs(G: nx.MultiDiGraph) -> None:
    """Ensure numeric edge/node attrs are floats; drop edges with invalid lengths."""

    to_remove = []
    if G.is_multigraph():
        edge_iter = G.edges(keys=True, data=True)
    else:
        edge_iter = ((u, v, None, d) for u, v, d in G.edges(data=True))

    for u, v, k, data in edge_iter:
        if "length" in data:
            try:
                data["length"] = float(data["length"])
            except (TypeError, ValueError):
                data["length"] = float("nan")
        if "bearing" in data:
            try:
                data["bearing"] = float(data["bearing"])
            except (TypeError, ValueError):
                data.pop("bearing", None)
        if not np.isfinite(data.get("length", np.nan)):
            to_remove.append((u, v, k))

    if to_remove:
        if G.is_multigraph():
            G.remove_edges_from(to_remove)
        else:
            G.remove_edges_from([(u, v) for u, v, _ in to_remove])

    for node, data in G.nodes(data=True):
        for key in ("x", "y"):
            if key in data:
                try:
                    data[key] = float(data[key])
                except (TypeError, ValueError):
                    data.pop(key, None)


def _largest_component(G: nx.Graph) -> nx.Graph:
    if nx.is_empty(G):
        raise ValueError("Graph is empty")
    if G.is_directed():
        components = nx.weakly_connected_components(G)
        largest_nodes = max(components, key=len)
        return G.subgraph(largest_nodes).copy()
    components = nx.connected_components(G)
    largest_nodes = max(components, key=len)
    return G.subgraph(largest_nodes).copy()


def _to_simple_graph_min_length(G: nx.Graph) -> nx.Graph:
    """Collapse multiedges by keeping the edge with the smallest length."""

    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return G.copy()

    simple = nx.Graph()
    simple.add_nodes_from(G.nodes(data=True))
    for u, v, data in G.edges(data=True):
        length = data.get("length")
        if simple.has_edge(u, v):
            existing = simple.edges[u, v].get("length")
            if existing is not None and length is not None and length >= existing:
                continue
        simple.add_edge(u, v, **{k: v for k, v in data.items() if k != "key"})
    return simple


def degree_stats(G: nx.Graph) -> Dict[str, float]:
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    return {
        "degree_mean": float(np.mean(degrees)),
        "degree_median": float(np.median(degrees)),
        "degree_std": float(np.std(degrees)),
    }


def intersection_fractions(G: nx.Graph) -> Dict[str, float]:
    counts = np.bincount([d for _, d in G.degree()], minlength=5)
    total = counts.sum() if counts.sum() else 1
    return {
        "fraction_dead_end": float(counts[1] / total),
        "fraction_3_way": float(counts[3] / total),
        "fraction_4_way": float(counts[4] / total),
    }


def clustering_metrics(G: nx.Graph) -> Dict[str, float]:
    clustering = nx.clustering(G.to_undirected())
    values = np.array(list(clustering.values()), dtype=float)
    return {"clustering_mean": float(np.mean(values))}


def betweenness_metrics(G: nx.Graph, max_exact_nodes: int = 8000, sample_k: int = 150) -> Dict[str, float]:
    weight = "length" if any("length" in d for _, _, d in G.edges(data=True)) else None
    n = G.number_of_nodes()
    if n > max_exact_nodes:
        # Heavy graphs: small sample to bound runtime
        k = min(sample_k, n)
        _log_msg(f"betweenness sampling k={k} of {n} nodes (weighted={bool(weight)})")
        btwn = nx.betweenness_centrality(
            G,
            weight=weight,
            normalized=True,
            endpoints=False,
            k=k,
            seed=config.DEFAULT_SEED,
        )
    else:
        _log_msg(f"betweenness exact on {n} nodes (weighted={bool(weight)})")
        btwn = nx.betweenness_centrality(G, weight=weight, normalized=True, endpoints=False)
    values = np.array(list(btwn.values()), dtype=float)
    return {
        "betweenness_mean": float(np.mean(values)),
        "betweenness_median": float(np.median(values)),
    }


def closeness_metrics(G: nx.Graph) -> Dict[str, float]:
    weight = "length" if any("length" in d for _, _, d in G.edges(data=True)) else None
    n = G.number_of_nodes()
    if n > 20000:
        _log_msg(f"closeness skipped (n={n} > 20000)")
        return {"closeness_mean": float("nan"), "closeness_median": float("nan")}
    _log_msg(f"closeness exact on {n} nodes (weighted={bool(weight)})")
    close = nx.closeness_centrality(G, distance=weight)
    values = np.array(list(close.values()), dtype=float)
    return {
        "closeness_mean": float(np.mean(values)),
        "closeness_median": float(np.median(values)),
    }


def edge_length_stats(G: nx.Graph) -> Dict[str, float]:
    lengths = np.array([d.get("length", np.nan) for _, _, d in G.edges(data=True)], dtype=float)
    lengths = lengths[np.isfinite(lengths)]
    return {
        "edge_length_mean": float(np.mean(lengths)),
        "edge_length_median": float(np.median(lengths)),
        "edge_length_std": float(np.std(lengths)),
    }


def edge_betweenness_metrics(G: nx.Graph, max_exact_nodes: int = 8000, sample_k: int = 150) -> Dict[str, float]:
    weight = "length" if any("length" in d for _, _, d in G.edges(data=True)) else None
    n = G.number_of_nodes()
    if n > max_exact_nodes:
        k = min(sample_k, n)
        btwn = nx.edge_betweenness_centrality(G, weight=weight, normalized=True, k=k, seed=config.DEFAULT_SEED)
    else:
        btwn = nx.edge_betweenness_centrality(G, weight=weight, normalized=True)
    values = np.array(list(btwn.values()), dtype=float)
    return {
        "edge_betweenness_mean": float(np.mean(values)),
        "edge_betweenness_median": float(np.median(values)),
    }


def orientation_histogram(G: nx.Graph, bins: int = 18) -> Dict[str, float]:
    bearings = np.array([d.get("bearing") for _, _, d in G.edges(data=True) if "bearing" in d], dtype=float)
    bearings = bearings[np.isfinite(bearings)]
    if len(bearings) == 0:
        return {f"orientation_bin_{i}": 0.0 for i in range(bins)}

    # Fold to [0, 180)
    bearings = np.mod(bearings, 180.0)
    counts, bin_edges = np.histogram(bearings, bins=bins, range=(0.0, 180.0), density=True)
    hist = {f"orientation_bin_{i}": float(counts[i]) for i in range(bins)}
    hist["orientation_bin_edges"] = [float(x) for x in bin_edges]
    return hist


def intersection_density(G: nx.Graph) -> Dict[str, float]:
    # Estimate area via convex hull of nodes in projected CRS
    xs = [data.get("x") for _, data in G.nodes(data=True) if "x" in data]
    ys = [data.get("y") for _, data in G.nodes(data=True) if "y" in data]
    if not xs or not ys:
        return {"intersection_density_per_km2": float("nan")}
    import shapely.geometry as geom  # local import to avoid heavy global dep cost

    points = [geom.Point(x, y) for x, y in zip(xs, ys)]
    hull = geom.MultiPoint(points).convex_hull
    area_m2 = hull.area if hull.is_valid else float("nan")
    if not np.isfinite(area_m2) or area_m2 == 0:
        return {"intersection_density_per_km2": float("nan")}
    density = len(G) / (area_m2 / 1_000_000.0)
    return {"intersection_density_per_km2": float(density)}


def compute_all_metrics(G: nx.Graph) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    t0 = time.time()

    # Use undirected for many metrics to avoid direction skew
    G_u = G.to_undirected()
    G_largest = _largest_component(G_u)
    G_simple = _to_simple_graph_min_length(G_largest)
    _log_step("prepare_graph", t0)

    t = time.time()

    metrics.update(degree_stats(G_simple))
    metrics.update(intersection_fractions(G_simple))
    metrics.update(clustering_metrics(G_simple))
    _log_step("degree/clustering", t)

    t = time.time()
    metrics.update(betweenness_metrics(G_simple))
    _log_step("betweenness", t)

    t = time.time()
    metrics.update(closeness_metrics(G_simple))
    _log_step("closeness", t)

    t = time.time()
    metrics.update(edge_betweenness_metrics(G_simple))
    _log_step("edge_betweenness", t)

    t = time.time()
    metrics.update(edge_length_stats(G_simple))
    metrics.update(orientation_histogram(G))  # use original directed with bearings
    metrics.update(intersection_density(G_simple))
    _log_step("lengths/orientation/density", t)

    metrics["nodes"] = G_simple.number_of_nodes()
    metrics["edges"] = G_simple.number_of_edges()
    return metrics


def _metrics_path(city_slug: str, base_dir: Optional[Path] = None) -> Path:
    config.ensure_output_dirs(base_dir)
    return config.output_path("reports", f"{city_slug}_metrics.json", base_dir=base_dir)


def compute_city_metrics(city_slug: str, base_dir: Optional[Path] = None) -> Dict[str, float]:
    processed_path = config.data_path("processed", f"{city_slug}_processed.graphml", base_dir=base_dir)
    _log_msg(f"load {processed_path}")
    G = load_processed_graph(processed_path)
    metrics = compute_all_metrics(G)
    metrics["city"] = city_slug
    metrics["crs"] = str(G.graph.get("crs"))
    return metrics


def save_metrics(city_slug: str, metrics: Dict[str, float], base_dir: Optional[Path] = None) -> Path:
    path = _metrics_path(city_slug, base_dir)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute street-network metrics")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--city", help="City slug to process")
    group.add_argument("--all", action="store_true", help="Process all cities")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if cached")
    return parser.parse_args()


def _log_msg(msg: str) -> None:
    print(f"[metrics] {msg}", flush=True)


def _log_step(name: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    print(f"[metrics] step {name} {elapsed:.1f}s", flush=True)


def main() -> None:
    args = _parse_args()
    if args.all:
        for city in (p.stem.replace("_processed", "") for p in Path(config.DATA_DIR / "processed").glob("*_processed.graphml")):
            out_path = _metrics_path(city)
            if out_path.exists() and not args.overwrite:
                print(f"SKIP {city} cached at {out_path}")
                continue
            try:
                print(f"[metrics] start {city}")
                metrics = compute_city_metrics(city)
                save_metrics(city, metrics)
                print(f"[metrics] OK {city} -> {out_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[metrics] FAIL {city}: {exc}")
                break
    else:
        city = args.city
        out_path = _metrics_path(city)
        if out_path.exists() and not args.overwrite:
            print(f"Metrics exist, skipping: {out_path}")
            return
        print(f"[metrics] start {city}")
        metrics = compute_city_metrics(city)
        save_metrics(city, metrics)
        print(f"[metrics] OK {city} -> {out_path}")


if __name__ == "__main__":
    main()
