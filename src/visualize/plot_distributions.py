"""Plot distributions such as node degree and edge lengths per city."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from .. import config


def load_graph(city_slug: str, base_dir: Optional[Path] = None) -> nx.Graph:
    path = config.data_path("processed", f"{city_slug}_processed.graphml", base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Processed graph missing: {path}")
    return nx.read_graphml(path)


def plot_distributions(city_slug: str, base_dir: Optional[Path] = None) -> Path:
    G = load_graph(city_slug, base_dir)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    lengths = np.array([data.get("length", np.nan) for _, _, data in G.edges(data=True)], dtype=float)
    lengths = lengths[np.isfinite(lengths)]

    out_dir = config.output_path("visuals", "single_city", base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{city_slug}_distributions.png"

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    fig.patch.set_facecolor("#0b0c10")
    ax_deg, ax_len = axes

    # Style both axes with dark theme
    for ax in axes:
        ax.set_facecolor("#0b0c10")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, alpha=0.3, color="#444444", linewidth=0.5)

    ax_deg.hist(degrees, bins=range(1, 9), color="#3182bd", edgecolor="white", alpha=0.9)
    ax_deg.set_title("Node degree", color="white")
    ax_deg.set_xlabel("Degree", color="white")
    ax_deg.set_ylabel("Count", color="white")

    ax_len.hist(lengths, bins=40, color="#9e9ac8", edgecolor="white", alpha=0.9)
    ax_len.set_title("Edge length (m)", color="white")
    ax_len.set_xlabel("Meters", color="white")

    fig.suptitle(city_slug.replace("_", " ").title(), color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#0b0c10")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot node degree and edge length distributions")
    parser.add_argument("--city", required=True, help="City slug")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = config.output_path("visuals", "single_city", f"{args.city}_distributions.png")
    if out_path.exists() and not args.overwrite:
        print(f"Plot exists, skipping: {out_path}")
        return
    path = plot_distributions(args.city)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
