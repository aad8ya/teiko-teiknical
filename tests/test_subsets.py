"""Oracle-gated tests for teiko/subsets.py (Part 4 baseline subset queries)."""

import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from teiko.config import DB_PATH, TABLES_DIR
from teiko.db import build_database, get_connection
from teiko.subsets import baseline_subset, by_project, by_response, by_sex, run


# ---------------------------------------------------------------------------
# Shared fixture — real DB via a temp copy so we don't clobber cell-count.db
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def conn():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        tmp_path = Path(f.name)
    build_database(db_path=tmp_path)
    c = get_connection(tmp_path)
    yield c
    c.close()
    tmp_path.unlink(missing_ok=True)


@pytest.fixture(scope="module")
def subset(conn):
    return baseline_subset(conn)


# ---------------------------------------------------------------------------
# baseline_subset
# ---------------------------------------------------------------------------

class TestBaselineSubset:
    def test_row_count(self, subset):
        assert len(subset) == 656

    def test_distinct_subjects(self, subset):
        assert subset["subject"].nunique() == 656

    def test_all_melanoma(self, subset):
        assert (subset["condition"] == "melanoma").all()

    def test_all_miraclib(self, subset):
        assert (subset["treatment"] == "miraclib").all()

    def test_all_pbmc(self, subset):
        assert (subset["sample_type"] == "PBMC").all()

    def test_all_baseline(self, subset):
        assert (subset["time_from_treatment_start"] == 0).all()

    def test_sorted_by_sample(self, subset):
        assert list(subset["sample"]) == sorted(subset["sample"].tolist())

    def test_columns_present(self, subset):
        expected = {
            "sample", "subject", "project", "condition", "treatment",
            "sample_type", "time_from_treatment_start", "response", "sex",
        }
        assert expected.issubset(set(subset.columns))


# ---------------------------------------------------------------------------
# by_project (sample-level)
# ---------------------------------------------------------------------------

class TestByProject:
    def test_oracle_values(self, subset):
        result = by_project(subset)
        mapping = dict(zip(result["project"], result["n_samples"]))
        assert mapping.get("prj1") == 384
        assert mapping.get("prj3") == 272
        assert "prj2" not in mapping

    def test_sum_matches_total(self, subset):
        result = by_project(subset)
        assert result["n_samples"].sum() == 656

    def test_columns(self, subset):
        result = by_project(subset)
        assert list(result.columns) == ["project", "n_samples"]


# ---------------------------------------------------------------------------
# by_response (subject-level)
# ---------------------------------------------------------------------------

class TestByResponse:
    def test_oracle_values(self, subset):
        result = by_response(subset)
        mapping = dict(zip(result["response"], result["n_subjects"]))
        assert mapping.get("no") == 325
        assert mapping.get("yes") == 331

    def test_sum_matches_subjects(self, subset):
        result = by_response(subset)
        assert result["n_subjects"].sum() == 656

    def test_no_null_response(self, subset):
        result = by_response(subset)
        assert result["response"].isna().sum() == 0

    def test_columns(self, subset):
        result = by_response(subset)
        assert list(result.columns) == ["response", "n_subjects"]


# ---------------------------------------------------------------------------
# by_sex (subject-level)
# ---------------------------------------------------------------------------

class TestBySex:
    def test_oracle_values(self, subset):
        result = by_sex(subset)
        mapping = dict(zip(result["sex"], result["n_subjects"]))
        assert mapping.get("F") == 312
        assert mapping.get("M") == 344

    def test_sum_matches_subjects(self, subset):
        result = by_sex(subset)
        assert result["n_subjects"].sum() == 656

    def test_columns(self, subset):
        result = by_sex(subset)
        assert list(result.columns) == ["sex", "n_subjects"]


# ---------------------------------------------------------------------------
# Empty-subset guard
# ---------------------------------------------------------------------------

class TestEmptySubsetGuard:
    def _make_empty(self):
        return pd.DataFrame(
            columns=[
                "sample", "subject", "project", "condition", "treatment",
                "sample_type", "time_from_treatment_start", "response", "sex",
            ]
        )

    def test_by_project_empty(self):
        result = by_project(self._make_empty())
        assert list(result.columns) == ["project", "n_samples"]
        assert len(result) == 0

    def test_by_response_empty(self):
        result = by_response(self._make_empty())
        assert list(result.columns) == ["response", "n_subjects"]
        assert len(result) == 0

    def test_by_sex_empty(self):
        result = by_sex(self._make_empty())
        assert list(result.columns) == ["sex", "n_subjects"]
        assert len(result) == 0


# ---------------------------------------------------------------------------
# CSV output via run()
# ---------------------------------------------------------------------------

_RUN_OUTPUTS: dict = {}


@pytest.fixture(scope="module", autouse=False)
def run_outputs(conn, tmp_path_factory):
    """Run subsets.run() once, redirecting TABLES_DIR to a temp dir."""
    import unittest.mock
    import teiko.subsets as subsets_mod

    tmp_tables = tmp_path_factory.mktemp("part4_tables")
    mp = unittest.mock.patch.object(subsets_mod, "TABLES_DIR", tmp_tables)
    mp.start()
    try:
        results = run(conn=conn)
        _RUN_OUTPUTS["tmp_tables"] = tmp_tables
        _RUN_OUTPUTS["results"] = results
    finally:
        mp.stop()
    yield _RUN_OUTPUTS


class TestRunOutputs:
    def test_returns_four_items(self, run_outputs):
        assert len(run_outputs["results"]) == 4

    def test_baseline_csv_exists(self, run_outputs):
        assert (run_outputs["tmp_tables"] / "part4_baseline_samples.csv").exists()

    def test_baseline_csv_row_count(self, run_outputs):
        df = pd.read_csv(run_outputs["tmp_tables"] / "part4_baseline_samples.csv")
        assert len(df) == 656

    def test_by_project_csv_exists(self, run_outputs):
        assert (run_outputs["tmp_tables"] / "part4_by_project.csv").exists()

    def test_by_response_csv_exists(self, run_outputs):
        assert (run_outputs["tmp_tables"] / "part4_by_response.csv").exists()

    def test_by_sex_csv_exists(self, run_outputs):
        assert (run_outputs["tmp_tables"] / "part4_by_sex.csv").exists()

    def test_baseline_csv_columns(self, run_outputs):
        df = pd.read_csv(run_outputs["tmp_tables"] / "part4_baseline_samples.csv")
        expected = [
            "sample", "subject", "project", "condition", "treatment",
            "sample_type", "time_from_treatment_start", "response", "sex",
        ]
        assert list(df.columns) == expected
