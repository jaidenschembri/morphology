# Urban Constellations

A graph-theoretic atlas of city street networks. This project pulls street networks from OpenStreetMap, builds city graphs, computes morphology features, and generates visuals and embeddings for cross-city comparison.

## Setup
1. Create a virtual environment (Python 3.10+ recommended):
   - `python -m venv .venv`
   - `source .venv/bin/activate`
2. Install dependencies:
   - `pip install -r requirements.txt`

## Quickstart
- Configure `data/cities_list.txt` (see `plan.md` for format).
- Download one city graph (once implemented):
  - `python -m src.download_osm --city <slug>`

## Repository Layout
- `src/`: pipeline modules for registry, download, cleaning, metrics, and visualization.
- `data/`: input lists and raw artifacts.
- `output/`: generated logs, reports, embeddings, and visuals (treat as build artifacts).
- `blog/`: draft writeups and assets for the Urban Constellations post.

## End-to-end pipeline (bash commands)
Run from repo root with the venv active (`source .venv/bin/activate`).

1) Download OSM graphs (cached in `data/raw/`):
   - Single city: `python -m src.download_osm --city <slug> [--overwrite]`
   - All cities: `python -m src.download_osm --all [--overwrite]`

2) Build/clean graphs (writes `data/processed/`, metadata to `output/reports/`):
   - `python -m src.build_graph --all --overwrite`
   - Per-city: `python -m src.build_graph --city <slug> [--overwrite]`

3) Compute metrics (writes `output/reports/<slug>_metrics.json`; logs per-step):
   - `python -m src.compute_metrics --all --overwrite | tee output/logs/metrics_run.log`
   - Per-city: `python -m src.compute_metrics --city <slug> [--overwrite]`
   - Note: node betweenness is sampled on large graphs; closeness is skipped above ~20k nodes to avoid stalls.

4) Aggregate features and embeddings (writes `output/embeddings/`):
   - `python -m src.features --umap`
   - Outputs: `city_features.csv`, `city_pca.csv`, `city_umap.csv`.

5) Visualizations (writes `output/visuals/`):
   - Street skeleton: `python -m src.visualize.plot_street_graph --city <slug> [--color-by-highway] --overwrite`
   - Centrality heatmap (betweenness sampling via `k`; use small `k` like 50–200 for large cities): `python -m src.visualize.plot_centrality --city <slug> --k 100 --overwrite`
   - Orientation rose: `python -m src.visualize.plot_orientation --city <slug> --overwrite`
   - Degree/length histograms: `python -m src.visualize.plot_distributions --city <slug> --overwrite`
   - Embedding plots: `python -m src.visualize.plot_embeddings --name pca` (or `umap`) `--overwrite`

### Batch loops for visuals
Visual scripts take `--city` only. To render everything:

```bash
for c in $(cut -d, -f1 data/cities_list.txt); do
  python -m src.visualize.plot_street_graph --city "$c" --overwrite
done

for c in $(cut -d, -f1 data/cities_list.txt); do
  python -m src.visualize.plot_orientation --city "$c" --overwrite
done

for c in $(cut -d, -f1 data/cities_list.txt); do
  python -m src.visualize.plot_distributions --city "$c" --overwrite
done

# Centrality: use small k for big cities to keep runtime reasonable
for c in $(cut -d, -f1 data/cities_list.txt); do
  python -m src.visualize.plot_centrality --city "$c" --k 100 --overwrite
done

# Cross-city embeddings (after metrics/features)
python -m src.visualize.plot_embeddings --name pca --overwrite
python -m src.visualize.plot_embeddings --name umap --overwrite
```

### Rebuilding ignored artifacts
Large artifacts produced by the pipeline (`data/raw/`, `data/processed/`, `output/`, and any experiment `cache/` or `models/` directories) are intentionally excluded from version control to keep the repo lightweight. All visuals (PNGs/SVGs) are emitted under `output/visuals/` and can be reproduced by rerunning the visualize commands above. When cloning, follow the setup and pipeline steps to regenerate raw/processed GraphML files, metrics, embeddings, and publish-ready figures before running analyses.

## How the code fits together
- `src/city_registry.py` loads the city list into a `City` dataclass for consistent querying.
- `src/config.py` centralizes paths, defaults, and output helpers.
- `src/download_osm.py` pulls each city’s street network from OSMnx and caches GraphML in `data/raw/`.
- `src/build_graph.py` loads raw graphs, adds bearings, projects to metric CRS, simplifies, adds lengths/degree, validates, and writes `data/processed/` plus build metadata.
- `src/compute_metrics.py` loads processed graphs, uses the largest component, computes node degree/clustering/betweenness/closeness (skips closeness on huge graphs), edge lengths, orientation histograms, and density; emits per-city JSON.
- `src/features.py` stitches JSON metrics into a features CSV, imputes NaNs, and produces PCA/UMAP embeddings.
- `src/visualize/` modules render single-city skeletons, centrality heatmaps, orientation roses, distributions, and cross-city embedding scatters.

### What are nodes and edges?
- **Nodes** represent street intersections (OSM graph nodes). Metrics derived from nodes capture junction complexity: degree distribution (number of incident streets per intersection), clustering (how interconnected neighbors are), betweenness/closeness (how often intersections sit on shortest paths or how close they are to all others), and intersection density.
- **Edges** represent street segments between intersections. Edge metrics capture morphology: length distribution, orientation histograms (bearings folded into 0–180°), and derived density measures. When building visuals, edges carry geometry (`LineString`) and highway tags for styling; lengths feed distance-weighted centrality.
- Metrics are computed on the largest connected component to avoid tiny islands. Betweenness is sampled on large graphs, and closeness is skipped when node counts exceed ~20k to keep runtimes feasible.

## Goals
- Produce a clean, reproducible pipeline from OSM download through processed graphs, metrics, embeddings, and visuals.
- Enable quick per-city or batch runs with clear logging.
- Generate portfolio-ready images and embeddings for comparing street-network morphology across ~35 global cities.
