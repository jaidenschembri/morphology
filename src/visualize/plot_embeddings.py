"""Render PCA/UMAP embeddings and comparative visuals for cities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .. import config, city_registry


def load_embeddings(name: str = "pca", base_dir: Optional[Path] = None) -> pd.DataFrame:
    path = config.output_path("embeddings", f"city_{name}.csv", base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"Embedding CSV missing: {path}")
    return pd.read_csv(path)


def _region_map() -> dict:
    return {c.slug: c.region for c in city_registry.get_all_cities()}


def format_region_name(region: str) -> str:
    """Convert region slug to display name."""
    region_map = {
        "north_america": "North America",
        "latin_america": "Latin America",
        "europe": "Europe",
        "africa_middle_east": "Africa & Middle East",
        "asia_pacific": "Asia-Pacific"
    }
    return region_map.get(region, region.replace("_", " ").title())


def plot_embedding(df: pd.DataFrame, name: str, color_by: str = "region", base_dir: Optional[Path] = None) -> Path:
    out_dir = config.output_path("visuals", "comparisons", base_dir=base_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"city_{name}_{color_by}.png"

    region_lookup = _region_map()
    df["region"] = df["city"].map(region_lookup)

    x_col, y_col = df.columns[1], df.columns[2]

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    fig.patch.set_facecolor("#0b0c10")
    ax.set_facecolor("#0b0c10")

    if color_by == "cluster" and any(col.startswith("cluster_") for col in df.columns):
        # pick the cluster column for this embedding if present
        cluster_col = f"cluster_{name}" if f"cluster_{name}" in df.columns else [c for c in df.columns if c.startswith("cluster_")][0]
        df[color_by] = df[cluster_col].astype(int)
        groups = sorted(df[color_by].unique())
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
        labels = {g: f"Cluster {g}" for g in groups}
    else:
        df[color_by] = df["region"]
        groups = sorted(df[color_by].dropna().unique())
        cmap = plt.get_cmap("tab10")
        colors = {g: cmap(i % 10) for i, g in enumerate(groups)}
        labels = {g: format_region_name(g) for g in groups}

    for _, row in df.iterrows():
        group = row[color_by]
        ax.scatter(row[x_col], row[y_col], color=colors.get(group, "gray"), alpha=0.85, s=50, label=labels.get(group, group))

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[g]) for g in groups]
    legend_labels = [labels[g] for g in groups]
    legend = ax.legend(handles, legend_labels, title=color_by.title(), loc="best",
                      facecolor="#1a1a1a", edgecolor="white", framealpha=0.9)
    legend.get_title().set_color("white")
    for text in legend.get_texts():
        text.set_color("white")

    # Dark theme styling
    ax.tick_params(colors="white")
    ax.spines["bottom"].set_color("white")
    ax.spines["left"].set_color("white")
    ax.spines["top"].set_color("white")
    ax.spines["right"].set_color("white")
    ax.grid(True, alpha=0.2, color="#444444", linewidth=0.5)

    ax.set_xlabel(x_col, color="white")
    ax.set_ylabel(y_col, color="white")
    ax.set_title(f"City embedding ({name.upper()}) colored by {color_by}", color="white")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="#0b0c10")
    plt.close(fig)
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot city embeddings (PCA/UMAP)")
    parser.add_argument("--name", default="pca", help="Embedding name, e.g., pca or umap")
    parser.add_argument("--overwrite", action="store_true", help="Force re-render")
    parser.add_argument("--color-by", choices=["region", "cluster"], default="region", help="Color points by region or cluster")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    out_path = config.output_path("visuals", "comparisons", f"city_{args.name}_{args.color_by}.png")
    if out_path.exists() and not args.overwrite:
        print(f"Plot exists, skipping: {out_path}")
        return
    df = load_embeddings(args.name)
    path = plot_embedding(df, args.name, color_by=args.color_by)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
