"""Feature aggregation and embeddings for cross-city morphology analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import umap

from . import config


def load_metrics_reports(base_dir: Optional[Path] = None) -> List[Dict]:
    reports_dir = config.output_path("reports", base_dir=base_dir)
    reports = []
    for path in Path(reports_dir).glob("*_metrics.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        reports.append(data)
    return sorted(reports, key=lambda d: d.get("city", ""))


def reports_to_dataframe(reports: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(reports)
    # Drop non-feature columns
    non_feature_cols = {"city", "crs", "orientation_bin_edges"}
    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    df = df[["city"] + feature_cols]

    # Impute NaNs with column means (per feature) to allow PCA/UMAP
    for col in feature_cols:
        if df[col].isna().any():
            mean_val = df[col].mean(skipna=True)
            df[col] = df[col].fillna(mean_val)
    return df


def save_features_csv(df: pd.DataFrame, base_dir: Optional[Path] = None) -> Path:
    config.ensure_output_dirs(base_dir)
    path = config.output_path("embeddings", "city_features.csv", base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def compute_embeddings(df: pd.DataFrame, n_components: int = 2) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "city"]
    X = df[feature_cols].values
    pipeline = Pipeline([("scale", StandardScaler()), ("pca", PCA(n_components=n_components))])
    pca_vals = pipeline.fit_transform(X)
    pca_cols = [f"pca_{i+1}" for i in range(pca_vals.shape[1])]
    pca_df = pd.DataFrame(pca_vals, columns=pca_cols)
    pca_df.insert(0, "city", df["city"].values)
    return pca_df


def compute_umap(
    df: pd.DataFrame,
    n_components: int = 2,
    random_state: int = config.DEFAULT_SEED,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "city"]
    X = df[feature_cols].values
    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "umap",
                umap.UMAP(
                    n_components=n_components,
                    random_state=random_state,
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                ),
            ),
        ]
    )
    umap_vals = pipeline.fit_transform(X)
    umap_cols = [f"umap_{i+1}" for i in range(umap_vals.shape[1])]
    umap_df = pd.DataFrame(umap_vals, columns=umap_cols)
    umap_df.insert(0, "city", df["city"].values)
    return umap_df


def cluster_features(df: pd.DataFrame, k: int = 5, embedding: str = "pca") -> pd.DataFrame:
    """Run k-means on an embedding and attach labels."""

    feature_cols = [c for c in df.columns if c != "city"]
    X = df[feature_cols].values
    km = KMeans(n_clusters=k, n_init=10, random_state=config.DEFAULT_SEED)
    labels = km.fit_predict(X)
    out = df.copy()
    out[f"cluster_{embedding}"] = labels
    return out


def save_embeddings(df: pd.DataFrame, name: str, base_dir: Optional[Path] = None) -> Path:
    config.ensure_output_dirs(base_dir)
    path = config.output_path("embeddings", f"city_{name}.csv", base_dir=base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate metrics and compute embeddings")
    parser.add_argument("--umap", action="store_true", help="Also compute UMAP embeddings")
    parser.add_argument("--pca-components", type=int, default=2, help="Number of PCA components (default 2)")
    parser.add_argument("--umap-components", type=int, default=2, help="Number of UMAP components (default 2)")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="UMAP n_neighbors (default 15)")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="UMAP min_dist (default 0.1)")
    parser.add_argument("--kmeans-k", type=int, default=5, help="Number of k-means clusters")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = load_metrics_reports()
    if not reports:
        raise SystemExit("No metrics reports found. Run compute_metrics first.")
    df = reports_to_dataframe(reports)
    features_path = save_features_csv(df)
    print(f"Saved features CSV to {features_path}")

    pca_df = compute_embeddings(df, n_components=args.pca_components)
    pca_clustered = cluster_features(pca_df, k=args.kmeans_k, embedding="pca")
    pca_path = save_embeddings(pca_clustered, "pca")
    print(f"Saved PCA embedding to {pca_path}")

    if args.umap:
        umap_df = compute_umap(
            df,
            n_components=args.umap_components,
            n_neighbors=args.umap_n_neighbors,
            min_dist=args.umap_min_dist,
        )
        umap_clustered = cluster_features(umap_df, k=args.kmeans_k, embedding="umap")
        umap_path = save_embeddings(umap_clustered, "umap")
        print(f"Saved UMAP embedding to {umap_path}")


if __name__ == "__main__":
    main()
