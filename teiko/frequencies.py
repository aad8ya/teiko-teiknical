"""Compute and write the Part 2 relative frequency summary table."""

import os
from pathlib import Path

import pandas as pd

from teiko.config import POPULATIONS, TABLES_DIR
from teiko.db import get_connection


def compute_frequencies(conn) -> pd.DataFrame:
    """
    Query cell_counts and return per-sample population frequencies.

    Returns a DataFrame with columns: sample, total_count, population, count, percentage.
    Sorted by sample ascending, then population in POPULATIONS order.
    """
    df = pd.read_sql_query(
        "SELECT sample_id AS sample, population, count FROM cell_counts",
        conn,
    )

    df["total_count"] = df.groupby("sample")["count"].transform("sum")
    df["percentage"] = df["count"] / df["total_count"] * 100

    # Guard: zero-total samples get 0.0 percentage (avoids divide-by-zero NaN)
    df.loc[df["total_count"] == 0, "percentage"] = 0.0

    df = df[["sample", "total_count", "population", "count", "percentage"]]

    # Deterministic sort: sample ascending, then population in canonical POPULATIONS order
    pop_order = {pop: i for i, pop in enumerate(POPULATIONS)}
    df["_pop_rank"] = df["population"].map(pop_order)
    df = df.sort_values(["sample", "_pop_rank"]).drop(columns="_pop_rank")
    df = df.reset_index(drop=True)

    return df


def write_frequencies(df: pd.DataFrame, path: Path = None) -> None:
    if path is None:
        path = TABLES_DIR / "cell_frequencies.csv"
    df = df.copy()
    df["total_count"] = df["total_count"].astype(int)
    df["count"] = df["count"].astype(int)
    # percentage stays at full float precision — no rounding here
    df.to_csv(path, index=False)


def run(conn=None) -> pd.DataFrame:
    """Compute frequencies, write cell_frequencies.csv, and return the DataFrame."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    close_after = conn is None
    if conn is None:
        conn = get_connection()
    try:
        df = compute_frequencies(conn)
        write_frequencies(df)
        return df
    finally:
        if close_after:
            conn.close()
