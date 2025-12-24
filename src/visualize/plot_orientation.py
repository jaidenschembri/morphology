"""Generate rose diagrams of street orientation for each city."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .. import config


def load_bearings(city_slug: str, base_dir: Optional[Path] = None) -> np.ndarray:
    path = config.data_path("processed", f"{city_slug}_processed.graphml", base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Processed graph missing: {path}")
    G = nx.read_graphml(path)
    bearings = np.array([d.get("bearing") for _, _, d in G.edges(data=True) if "bearing" in d], dtype=float)
    bearings = bearings[np.isfinite(bearings)]
    return bearings


def plot_orientation_rose(city_slug: str, bins: int = 18, base_dir: Optional[Path] = None) -> Path:
    bearings = load_bearings(city_slug, base_dir)
    out_dir = config.output_path("visuals", "single_city", base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_slug}_orientation_rose.png"

    if len(bearings) == 0:
        raise ValueError(f"No bearings found for {city_slug}")

    folded = np.deg2rad(np.mod(bearings, 180.0))
    counts, edges = np.histogram(folded, bins=bins, range=(0, np.pi))
    widths = np.diff(edges)

    fig = plt.figure(figsize=(6, 6), facecolor="black")
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor("black")
    bars = ax.bar(edges[:-1], counts, width=widths, bottom=0.0, color="cyan", edgecolor="cyan", alpha=0.8)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.grid(color="#444444")
    ax.set_yticklabels([])
    ax.set_xticklabels(["N", "NE", "E", "SE", "S", "SW", "W", "NW"], color="white")
    ax.set_title(f"{city_slug.replace('_', ' ').title()} — Street orientations", color="white", pad=10)
    for spine in ax.spines.values():
        spine.set_color("white")

    plt.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot street orientation rose diagram")
    parser.add_argument("--city", required=True, help="City slug to plot")
    parser.add_argument("--bins", type=int, default=18, help="Number of bins over 180 degrees")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = config.output_path("visuals", "single_city", f"{args.city}_orientation_rose.png")
    if out_path.exists() and not args.overwrite:
        print(f"Plot exists, skipping: {out_path}")
        return
    path = plot_orientation_rose(args.city, bins=args.bins)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
