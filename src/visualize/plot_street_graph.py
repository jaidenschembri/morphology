"""Plot city street graphs with simple skeleton or styled layers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
from shapely import wkt
from shapely.geometry import LineString

from .. import config


def load_processed_graph(city_slug: str, base_dir: Optional[Path] = None) -> nx.Graph:
    path = config.data_path("processed", f"{city_slug}_processed.graphml", base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Processed graph missing: {path}")
    return nx.read_graphml(path)


def _ensure_output(city_slug: str, base_dir: Optional[Path] = None) -> Path:
    config.ensure_output_dirs(base_dir)
    out_dir = config.output_path("visuals", "single_city", base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{city_slug}_street.png"


def plot_city_graph(city_slug: str, color_by_highway: bool = False, base_dir: Optional[Path] = None) -> Path:
    G = load_processed_graph(city_slug, base_dir)
    out_path = _ensure_output(city_slug, base_dir)

    fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
    ax.set_facecolor("black")
    ax.axis("off")

    for u, v, data in G.edges(data=True):
        geom = _edge_geom(u, v, data, G)
        xs, ys = geom.xy
        color = "white"
        if color_by_highway:
            color = _highway_color(data.get("highway"))
        ax.plot(xs, ys, linewidth=0.5 if color_by_highway else 0.3, color=color, alpha=0.9)
    ax.set_aspect("equal")
    ax.margins(0)
    title = f"{city_slug.replace('_', ' ').title()}"
    ax.set_title(title, color="white", fontsize=10, pad=6)

    plt.tight_layout(pad=0)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _highway_color(highway: Optional[str]) -> str:
    if isinstance(highway, list):
        highway = highway[0]
    palette = {
        "motorway": "#ff7f50",
        "trunk": "#ffa500",
        "primary": "#ffd700",
        "secondary": "#adff2f",
        "tertiary": "#87cefa",
        "residential": "#f5f5f5",
        "unclassified": "#d3d3d3",
    }
    return palette.get(highway, "#cccccc")


def _edge_xy(u: str, v: str, data: dict, G: nx.Graph) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    geom = data.get("geometry")
    if geom is not None:
        if isinstance(geom, str):
            try:
                geom = wkt.loads(geom)
            except Exception:
                geom = None
        if isinstance(geom, LineString):
            xs, ys = geom.xy
            return xs, ys
    # fallback to straight line between nodes
    return (G.nodes[u]["x"], G.nodes[v]["x"]), (G.nodes[u]["y"], G.nodes[v]["y"])


def _edge_geom(u: str, v: str, data: dict, G: nx.Graph) -> LineString:
    geom = data.get("geometry")
    if isinstance(geom, str):
        try:
            geom = wkt.loads(geom)
        except Exception:
            geom = None
    if isinstance(geom, LineString):
        return geom
    # fallback straight line
    return LineString([(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])] )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a city's street graph")
    parser.add_argument("--city", required=True, help="City slug to plot")
    parser.add_argument("--color-by-highway", action="store_true", help="Color edges by highway type")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render even if file exists")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = config.output_path("visuals", "single_city", f"{args.city}_street.png")
    if out_path.exists() and not args.overwrite:
        print(f"Plot exists, skipping: {out_path}")
        return
    path = plot_city_graph(args.city, color_by_highway=args.color_by_highway)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
