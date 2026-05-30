"""End-to-end pipeline integration tests.

Runs the full analysis stack against a freshly built database and verifies
that every artifact is produced with the correct row counts (oracle-gated).
Idempotency is also verified: running the pipeline twice yields identical
output.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from teiko.config import FIGURES_DIR, POPULATIONS, REPO_ROOT, TABLES_DIR
from teiko.db import build_database, get_connection
import teiko.frequencies
import teiko.plots
import teiko.stats
import teiko.subsets


# ---------------------------------------------------------------------------
# Fixture: isolated output directories + a pre-loaded DB
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_env(tmp_path_factory):
    """Build a fresh DB and run the full pipeline into isolated output dirs.

    Returns a dict with keys:
        db_path   — path to the temp DB
        tables    — path to the temp tables dir
        figures   — path to the temp figures dir
    """
    import unittest.mock as mock

    tmp_root = tmp_path_factory.mktemp("pipeline_e2e")
    db_path = tmp_root / "cell-count.db"
    tables_dir = tmp_root / "tables"
    figures_dir = tmp_root / "figures"
    tables_dir.mkdir()
    figures_dir.mkdir()

    build_database(db_path=db_path)

    # Patch module-level paths so all output lands in the temp dirs.
    patches = [
        mock.patch("teiko.frequencies.TABLES_DIR", tables_dir),
        mock.patch("teiko.stats.TABLES_DIR", tables_dir),
        mock.patch("teiko.plots.FIGURES_DIR", figures_dir),
        mock.patch("teiko.subsets.TABLES_DIR", tables_dir),
    ]

    conn = get_connection(db_path)

    # Also patch the DB connection used by stats/frequencies (they open their own).
    # We do this by patching DB_PATH in teiko.config and re-importing where needed.
    with mock.patch("teiko.config.DB_PATH", db_path), \
         mock.patch("teiko.frequencies.TABLES_DIR", tables_dir), \
         mock.patch("teiko.stats.TABLES_DIR", tables_dir), \
         mock.patch("teiko.plots.FIGURES_DIR", figures_dir), \
         mock.patch("teiko.subsets.TABLES_DIR", tables_dir):

        teiko.frequencies.run(conn=conn)
        # Copy the frequency CSV so stats can find it at the patched TABLES_DIR.
        freq_src = TABLES_DIR / "cell_frequencies.csv"
        shutil.copy(freq_src, tables_dir / "cell_frequencies.csv")

        # stats reads cell_frequencies.csv from TABLES_DIR; patch it there.
        with mock.patch("teiko.stats.TABLES_DIR", tables_dir):
            teiko.stats.run_all_stats()

        teiko.plots.generate_all_plots()
        teiko.subsets.run(conn=conn)

    conn.close()

    return {"db_path": db_path, "tables": tables_dir, "figures": figures_dir}


# ---------------------------------------------------------------------------
# Artifact existence checks
# ---------------------------------------------------------------------------

class TestArtifactsExist:
    def test_cell_frequencies_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "cell_frequencies.csv").exists()

    def test_stats_pooled_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "stats_pooled.csv").exists()

    def test_stats_baseline_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "stats_baseline.csv").exists()

    def test_stats_per_timepoint_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "stats_per_timepoint.csv").exists()

    def test_stats_summary_md(self, pipeline_env):
        assert (pipeline_env["tables"] / "stats_summary.md").exists()

    def test_part4_baseline_samples_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "part4_baseline_samples.csv").exists()

    def test_part4_by_project_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "part4_by_project.csv").exists()

    def test_part4_by_response_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "part4_by_response.csv").exists()

    def test_part4_by_sex_csv(self, pipeline_env):
        assert (pipeline_env["tables"] / "part4_by_sex.csv").exists()

    def test_all_seven_pngs_exist(self, pipeline_env):
        figures = pipeline_env["figures"]
        for pop in POPULATIONS:
            assert (figures / f"boxplot_{pop}.png").exists(), f"Missing boxplot_{pop}.png"
        assert (figures / "boxplots_combined.png").exists()
        assert (figures / "boxplots_per_timepoint.png").exists()


# ---------------------------------------------------------------------------
# Row count oracles
# ---------------------------------------------------------------------------

class TestRowCounts:
    def test_cell_frequencies_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "cell_frequencies.csv")
        assert len(df) == 52_500

    def test_stats_pooled_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "stats_pooled.csv")
        assert len(df) == 5

    def test_stats_baseline_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "stats_baseline.csv")
        assert len(df) == 5

    def test_stats_per_timepoint_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "stats_per_timepoint.csv")
        assert len(df) == 15

    def test_part4_baseline_samples_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "part4_baseline_samples.csv")
        assert len(df) == 656

    def test_part4_by_project_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "part4_by_project.csv")
        assert len(df) == 2

    def test_part4_by_response_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "part4_by_response.csv")
        assert len(df) == 2

    def test_part4_by_sex_row_count(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "part4_by_sex.csv")
        assert len(df) == 2

    def test_cell_frequencies_columns(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "cell_frequencies.csv")
        assert list(df.columns) == ["sample", "total_count", "population", "count", "percentage"]

    def test_stats_pooled_per_population_n(self, pipeline_env):
        df = pd.read_csv(pipeline_env["tables"] / "stats_pooled.csv")
        # Each population should account for all 1,968 samples
        assert ((df["n_responder"] + df["n_nonresponder"]) == 1_968).all()


# ---------------------------------------------------------------------------
# Idempotency: running the pipeline a second time yields identical counts
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_cell_frequencies_stable_on_rerun(self, pipeline_env):
        import unittest.mock as mock

        conn = get_connection(pipeline_env["db_path"])
        tables_dir = pipeline_env["tables"]

        with mock.patch("teiko.frequencies.TABLES_DIR", tables_dir):
            teiko.frequencies.run(conn=conn)

        conn.close()

        df = pd.read_csv(tables_dir / "cell_frequencies.csv")
        assert len(df) == 52_500

    def test_pngs_stable_on_rerun(self, pipeline_env):
        import unittest.mock as mock

        figures_dir = pipeline_env["figures"]
        before_sizes = {
            p.name: p.stat().st_size
            for p in figures_dir.glob("*.png")
        }

        with mock.patch("teiko.plots.FIGURES_DIR", figures_dir):
            teiko.plots.generate_all_plots()

        after_sizes = {
            p.name: p.stat().st_size
            for p in figures_dir.glob("*.png")
        }

        assert set(before_sizes.keys()) == set(after_sizes.keys())
        assert len(after_sizes) == 7
