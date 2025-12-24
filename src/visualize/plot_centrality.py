"""Visualize node betweenness centrality across a city street network."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely import wkt
from shapely.geometry import LineString

from .. import config


def load_graph(city_slug: str, base_dir: Optional[Path] = None) -> nx.Graph:
    path = config.data_path("processed", f"{city_slug}_processed.graphml", base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Processed graph missing: {path}")
    G = nx.read_graphml(path)
    _coerce_numeric_attrs(G)
    G = _largest_component(G)
    return G


def _coerce_numeric_attrs(G: nx.MultiGraph) -> None:
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
        if "geometry" in data and isinstance(data["geometry"], str):
            try:
                geom = wkt.loads(data["geometry"])
                if isinstance(geom, LineString):
                    data["geometry"] = geom
            except Exception:
                data.pop("geometry", None)
        if not np.isfinite(data.get("length", np.nan)):
            to_remove.append((u, v, k))

    if to_remove:
        if G.is_multigraph():
            G.remove_edges_from(to_remove)
        else:
            G.remove_edges_from([(u, v) for u, v, _ in to_remove])

    remove_nodes = []
    for n, d in G.nodes(data=True):
        for key in ("x", "y"):
            if key in d:
                try:
                    d[key] = float(d[key])
                except (TypeError, ValueError):
                    d.pop(key, None)
            else:
                remove_nodes.append(n)
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)

    remove_nodes = [n for n, d in G.nodes(data=True) if "x" not in d or "y" not in d]
    if remove_nodes:
        G.remove_nodes_from(remove_nodes)


def _to_simple_graph(G: nx.Graph) -> nx.Graph:
    if not isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        return G.to_undirected()

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


def _largest_component(G: nx.Graph) -> nx.Graph:
    if G.is_directed():
        comps = nx.weakly_connected_components(G)
    else:
        comps = nx.connected_components(G)
    largest = max(comps, key=len)
    return G.subgraph(largest).copy()


def _edge_geom(u: str, v: str, data: dict, G: nx.Graph) -> LineString:
    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            geom = wkt.loads(geom)
        except Exception:
            geom = None
    if isinstance(geom, LineString):
        return geom
    return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])] )


def compute_betweenness(G: nx.Graph, k: Optional[int] = 500) -> dict:
    weight = "length" if any("length" in d for _, _, d in G.edges(data=True)) else None
    G_simple = _to_simple_graph(G)
    n = G_simple.number_of_nodes()
    if k is None or k >= n:
        return nx.betweenness_centrality(G_simple, weight=weight, normalized=True)
    return nx.betweenness_centrality(G_simple, weight=weight, normalized=True, k=k, seed=config.DEFAULT_SEED)


def calculate_node_size(num_nodes: int) -> float:
    """
    Calculate appropriate node size based on graph size.
    Larger cities need smaller nodes to avoid overcrowding.
    """
    if num_nodes < 5000:
        return 5.0
    elif num_nodes < 20000:
        return 3.0
    elif num_nodes < 100000:
        return 2.0
    else:
        return 1.5


def plot_node_centrality(city_slug: str, k: Optional[int] = 500, base_dir: Optional[Path] = None) -> Path:
    G = load_graph(city_slug, base_dir)
    print(f"[centrality] computing betweenness for {city_slug} (k={k})", flush=True)
    bc = compute_betweenness(G, k=k)
    values = np.array([bc.get(node, 0.0) for node in G.nodes()], dtype=float)
    if len(values) == 0:
        raise ValueError("No nodes to plot")

    out_dir = config.output_path("visuals", "single_city", base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_slug}_centrality.png"

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.axis("off")

    xs_nodes = np.array([G.nodes[n]["x"] for n in G.nodes()], dtype=float)
    ys_nodes = np.array([G.nodes[n]["y"] for n in G.nodes()], dtype=float)
    xpad = (xs_nodes.max() - xs_nodes.min()) * 0.02 if len(xs_nodes) else 0
    ypad = (ys_nodes.max() - ys_nodes.min()) * 0.02 if len(ys_nodes) else 0

    # Draw edges lightly
    for u, v, data in G.edges(data=True):
        geom = _edge_geom(u, v, data, G)
        xs, ys = geom.xy
        ax.plot(xs, ys, linewidth=0.2, color="#444444", alpha=0.6)

    # Draw nodes colored by betweenness
    num_nodes = G.number_of_nodes()
    node_size = calculate_node_size(num_nodes)
    vmax = np.percentile(values, 99) if len(values) else values.max()
    vmin = values.min()
    sc = ax.scatter(
        xs_nodes,
        ys_nodes,
        c=values,
        cmap="plasma",
        s=node_size,
        linewidths=0,
        alpha=0.9,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("equal")
    ax.set_xlim(xs_nodes.min() - xpad, xs_nodes.max() + xpad)
    ax.set_ylim(ys_nodes.min() - ypad, ys_nodes.max() + ypad)
    title = f"{city_slug.replace('_', ' ').title()} — Node betweenness"
    ax.set_title(title, color="white", fontsize=10, pad=6)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6)
    cbar.ax.tick_params(labelcolor="white")
    cbar.outline.set_edgecolor("white")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot node betweenness centrality for a city")
    parser.add_argument("--city", required=True, help="City slug to plot")
    parser.add_argument("--k", type=int, default=500, help="Sample size for betweenness (set 0 for exact)")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    k = None if args.k == 0 else args.k
    out_path = config.output_path("visuals", "single_city", f"{args.city}_centrality.png")
    if out_path.exists() and not args.overwrite:
        print(f"Plot exists, skipping: {out_path}")
        return
    path = plot_node_centrality(args.city, k=k)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
