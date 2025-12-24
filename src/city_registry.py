"""City registry definitions and helpers for managing OSM queries.

This module provides a lightweight registry for cities to drive downloads
and analysis. City metadata lives in ``data/cities_list.txt`` with the
following CSV fields (no header)::

    slug,display_name,country,region,osm_query_string

The loader returns ``City`` instances and helper accessors make it easy to
filter by region or fetch a single city.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class City:
    """City metadata for OSM queries and reporting."""

    slug: str
    name: str
    country: str
    region: str
    query: str


def _registry_path(base_dir: Optional[Path] = None) -> Path:
    """Return path to the city registry file."""

    root = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parents[1]
    return root / "data" / "cities_list.txt"


def load_cities(base_dir: Optional[Path] = None) -> List[City]:
    """Load all cities from the registry file.

    Parameters
    ----------
    base_dir: Optional[Path]
        Override project root (useful for testing).
    """

    path = _registry_path(base_dir)
    cities: List[City] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            slug, name, country, region, query = _parse_line(stripped)
            cities.append(City(slug=slug, name=name, country=country, region=region, query=query))
    return cities


def _parse_line(line: str) -> tuple[str, str, str, str, str]:
    """Parse a registry line, expecting five comma-separated fields."""

    parts = _split_csv_line(line)
    if len(parts) != 5:
        raise ValueError(f"Expected 5 fields per line, got {len(parts)}: {line}")
    slug, name, country, region, query = parts
    return slug, name, country, region, query


def _split_csv_line(line: str) -> List[str]:
    """Split a CSV line without full parser overhead (quotes allowed on query).

    The format is simple enough that we can split on commas, but we respect
    quoted query strings that may contain commas.
    """

    fields: List[str] = []
    current: List[str] = []
    in_quotes = False

    for char in line:
        if char == '"':
            in_quotes = not in_quotes
            continue
        if char == "," and not in_quotes:
            fields.append("".join(current).strip())
            current = []
        else:
            current.append(char)

    if current:
        fields.append("".join(current).strip())

    return fields


def get_all_cities(base_dir: Optional[Path] = None) -> List[City]:
    """Return all cities in the registry."""

    return load_cities(base_dir)


def get_cities_by_region(region: str, base_dir: Optional[Path] = None) -> List[City]:
    """Return cities matching a region (case-insensitive)."""

    region_lower = region.lower()
    return [city for city in load_cities(base_dir) if city.region.lower() == region_lower]


def get_city(slug: str, base_dir: Optional[Path] = None) -> City:
    """Return a city by slug or raise ``KeyError`` if not found."""

    for city in load_cities(base_dir):
        if city.slug == slug:
            return city
    raise KeyError(f"City not found: {slug}")


def iter_cities(base_dir: Optional[Path] = None) -> Iterable[City]:
    """Yield cities one by one (memory-friendly)."""

    yield from load_cities(base_dir)
