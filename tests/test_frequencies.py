"""Tests for teiko/frequencies.py — oracle-gated acceptance criteria."""

import math
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from teiko.config import POPULATIONS, TABLES_DIR
from teiko.db import build_database, get_connection
from teiko.frequencies import compute_frequencies, run, write_frequencies


@pytest.fixture(scope="module")
def conn(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    build_database(db_path=db_path)
    c = get_connection(db_path)
    yield c
    c.close()


@pytest.fixture(scope="module")
def freq_df(conn):
    return compute_frequencies(conn)


class TestSchema:
    def test_column_order(self, freq_df):
        assert list(freq_df.columns) == ["sample", "total_count", "population", "count", "percentage"]

    def test_row_count(self, freq_df):
        assert len(freq_df) == 52_500

    def test_no_missing_values(self, freq_df):
        assert not freq_df.isnull().any().any()


class TestPercentages:
    def test_per_sample_sums_to_100(self, freq_df):
        sums = freq_df.groupby("sample")["percentage"].sum()
        assert (sums - 100.0).abs().max() < 1e-6

    def test_no_zero_total(self, freq_df):
        assert (freq_df["total_count"] == 0).sum() == 0

    def test_total_count_range(self, freq_df):
        assert freq_df["total_count"].min() == 84_247
        assert freq_df["total_count"].max() == 122_788

    def test_percentage_formula(self, freq_df):
        computed = freq_df["count"] / freq_df["total_count"] * 100
        assert (freq_df["percentage"] - computed).abs().max() < 1e-10


class TestSample00000:
    """Spot-check oracle values for sample00000."""

    def test_total_count(self, freq_df):
        sub = freq_df[freq_df["sample"] == "sample00000"]
        assert sub["total_count"].iloc[0] == 93_214

    def test_percentages(self, freq_df):
        sub = freq_df[freq_df["sample"] == "sample00000"].set_index("population")
        for _, row in sub.iterrows():
            expected = row["count"] / 93_214 * 100
            assert abs(row["percentage"] - expected) < 1e-10


class TestSort:
    def test_population_order_within_sample(self, freq_df):
        for _, group in freq_df.groupby("sample", sort=False):
            pops = list(group["population"])
            assert pops == POPULATIONS

    def test_samples_ascending(self, freq_df):
        samples = freq_df["sample"].drop_duplicates().tolist()
        assert samples == sorted(samples)


class TestWriteAndReread:
    def test_csv_reproduces_rows(self, freq_df, tmp_path):
        csv_path = tmp_path / "cell_frequencies.csv"
        write_frequencies(freq_df, path=csv_path)
        reread = pd.read_csv(csv_path)
        assert len(reread) == 52_500
        assert list(reread.columns) == ["sample", "total_count", "population", "count", "percentage"]

    def test_total_count_dtype_int(self, freq_df, tmp_path):
        csv_path = tmp_path / "cell_frequencies.csv"
        write_frequencies(freq_df, path=csv_path)
        reread = pd.read_csv(csv_path)
        # CSV-read integers come back as int64
        assert reread["total_count"].dtype == int
        assert reread["count"].dtype == int

    def test_percentage_full_precision(self, freq_df, tmp_path):
        """Percentage should not be rounded to 2 dp in the CSV."""
        csv_path = tmp_path / "cell_frequencies.csv"
        write_frequencies(freq_df, path=csv_path)
        reread = pd.read_csv(csv_path)
        # At least some percentage values should have more than 2 decimal places
        has_precision = (reread["percentage"] * 100).apply(lambda x: x != round(x, 2)).any()
        assert has_precision


class TestRunFunction:
    def test_run_creates_csv(self, conn):
        """run() should write the CSV and return a DataFrame."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            from teiko import config as cfg
            original = cfg.TABLES_DIR
            cfg.TABLES_DIR = Path(tmpdir)
            try:
                df = run(conn=conn)
            finally:
                cfg.TABLES_DIR = original

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 52_500

    def test_run_default_path(self, conn, tmp_path, monkeypatch):
        """run() writes to TABLES_DIR/cell_frequencies.csv by default."""
        from teiko import frequencies as freq_mod
        monkeypatch.setattr(freq_mod, "TABLES_DIR", tmp_path)

        df = run(conn=conn)
        csv_out = tmp_path / "cell_frequencies.csv"
        assert csv_out.exists()
        assert len(df) == 52_500
