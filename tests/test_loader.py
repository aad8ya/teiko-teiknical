"""Tests for data loading and load_sample_metadata."""

import tempfile
from pathlib import Path

import pytest

import pandas as pd

from teiko.config import CSV_PATH, POPULATIONS
from teiko.db import _validate_csv, build_database, get_connection, load_rows, load_sample_metadata


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def loaded_db(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("db") / "test_loader.db"
    build_database(db_path=db_path)
    return db_path


@pytest.fixture(scope="module")
def conn(loaded_db):
    c = get_connection(loaded_db)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Row count oracles
# ---------------------------------------------------------------------------

def test_sample_count(conn):
    count = conn.execute("SELECT COUNT(*) FROM samples;").fetchone()[0]
    assert count == 10_500


def test_subject_count(conn):
    count = conn.execute("SELECT COUNT(*) FROM subjects;").fetchone()[0]
    assert count == 3_500


def test_project_count(conn):
    count = conn.execute("SELECT COUNT(*) FROM projects;").fetchone()[0]
    assert count == 3


def test_cell_counts_row_count(conn):
    count = conn.execute("SELECT COUNT(*) FROM cell_counts;").fetchone()[0]
    assert count == 52_500


# ---------------------------------------------------------------------------
# cell_counts structural checks
# ---------------------------------------------------------------------------

def test_every_sample_has_five_populations(conn):
    bad = conn.execute(
        "SELECT sample_id, COUNT(*) AS n FROM cell_counts"
        " GROUP BY sample_id HAVING n != 5;"
    ).fetchall()
    assert bad == [], f"Samples with != 5 populations: {bad[:5]}"


def test_distinct_populations_match_config(conn):
    rows = conn.execute(
        "SELECT DISTINCT population FROM cell_counts ORDER BY population;"
    ).fetchall()
    actual = [r[0] for r in rows]
    assert set(actual) == set(POPULATIONS)


# ---------------------------------------------------------------------------
# Vocabulary / value checks (data-reality table from PLAN.md)
# ---------------------------------------------------------------------------

def test_condition_vocab(conn):
    rows = conn.execute(
        "SELECT condition, COUNT(*) FROM subjects GROUP BY condition ORDER BY condition;"
    ).fetchall()
    vocab = {r[0]: r[1] for r in rows}
    # 3500 subjects: melanoma=1725, carcinoma=1301, healthy=474  (subject-level)
    assert set(vocab.keys()) == {"melanoma", "carcinoma", "healthy"}


def test_sample_type_vocab(conn):
    rows = conn.execute(
        "SELECT sample_type, COUNT(*) FROM samples GROUP BY sample_type;"
    ).fetchall()
    vocab = {r[0]: r[1] for r in rows}
    assert set(vocab.keys()) == {"PBMC", "WB"}
    # PLAN.md: sample-level PBMC=7500, WB=3000
    assert vocab["PBMC"] == 7_500
    assert vocab["WB"] == 3_000


def test_time_vocab(conn):
    rows = conn.execute(
        "SELECT DISTINCT time_from_treatment_start FROM samples ORDER BY time_from_treatment_start;"
    ).fetchall()
    assert [r[0] for r in rows] == [0, 7, 14]


def test_sex_vocab(conn):
    rows = conn.execute(
        "SELECT sex, COUNT(*) FROM subjects GROUP BY sex ORDER BY sex;"
    ).fetchall()
    vocab = {r[0]: r[1] for r in rows}
    assert set(vocab.keys()) == {"F", "M"}


def test_project_vocab(conn):
    rows = conn.execute(
        "SELECT project_id FROM projects ORDER BY project_id;"
    ).fetchall()
    assert [r[0] for r in rows] == ["prj1", "prj2", "prj3"]


# ---------------------------------------------------------------------------
# NULL response — healthy subjects must have NULL, not empty string
# ---------------------------------------------------------------------------

def test_null_response_count_positive(conn):
    null_count = conn.execute(
        "SELECT COUNT(*) FROM subjects WHERE response IS NULL;"
    ).fetchone()[0]
    assert null_count > 0


def test_no_empty_string_response(conn):
    empty_count = conn.execute(
        "SELECT COUNT(*) FROM subjects WHERE response = '';"
    ).fetchone()[0]
    assert empty_count == 0


def test_null_response_matches_healthy_subjects(conn):
    """Healthy subjects (treatment='none') should have NULL response."""
    null_count = conn.execute(
        "SELECT COUNT(*) FROM subjects WHERE response IS NULL;"
    ).fetchone()[0]
    none_treatment_count = conn.execute(
        "SELECT COUNT(*) FROM subjects WHERE treatment = 'none';"
    ).fetchone()[0]
    assert null_count == none_treatment_count


# ---------------------------------------------------------------------------
# Subject-invariance guard
# (each subject must have exactly one distinct value per metadata field)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["project_id", "condition", "sex", "treatment", "age", "response"])
def test_subject_metadata_invariant(conn, field):
    bad = conn.execute(
        f"SELECT subject_id, COUNT(DISTINCT {field}) AS n"
        f" FROM subjects GROUP BY subject_id HAVING n > 1;"
    ).fetchall()
    assert bad == [], f"Field '{field}' varies within a subject: {bad[:3]}"


# ---------------------------------------------------------------------------
# No orphan foreign keys
# ---------------------------------------------------------------------------

def test_no_orphan_cell_counts(conn):
    orphans = conn.execute(
        "SELECT COUNT(*) FROM cell_counts cc"
        " LEFT JOIN samples s ON cc.sample_id = s.sample_id"
        " WHERE s.sample_id IS NULL;"
    ).fetchone()[0]
    assert orphans == 0


def test_no_orphan_samples(conn):
    orphans = conn.execute(
        "SELECT COUNT(*) FROM samples s"
        " LEFT JOIN subjects su ON s.subject_id = su.subject_id"
        " WHERE su.subject_id IS NULL;"
    ).fetchone()[0]
    assert orphans == 0


def test_no_orphan_subjects(conn):
    orphans = conn.execute(
        "SELECT COUNT(*) FROM subjects su"
        " LEFT JOIN projects p ON su.project_id = p.project_id"
        " WHERE p.project_id IS NULL;"
    ).fetchone()[0]
    assert orphans == 0


# ---------------------------------------------------------------------------
# load_sample_metadata
# ---------------------------------------------------------------------------

def test_load_sample_metadata_row_count(loaded_db):
    c = get_connection(loaded_db)
    df = load_sample_metadata(c)
    c.close()
    assert len(df) == 10_500


def test_load_sample_metadata_columns(loaded_db):
    c = get_connection(loaded_db)
    df = load_sample_metadata(c)
    c.close()
    expected = [
        "sample", "subject", "project", "condition", "age",
        "sex", "treatment", "response", "sample_type",
        "time_from_treatment_start",
    ]
    assert list(df.columns) == expected


def test_load_sample_metadata_no_duplicate_samples(loaded_db):
    c = get_connection(loaded_db)
    df = load_sample_metadata(c)
    c.close()
    assert df["sample"].nunique() == 10_500


# ---------------------------------------------------------------------------
# FileNotFoundError on missing CSV
# ---------------------------------------------------------------------------

def test_load_rows_missing_csv_raises(tmp_path):
    db_path = tmp_path / "empty.db"
    from teiko.db import init_schema
    conn = get_connection(db_path)
    init_schema(conn)
    with pytest.raises(FileNotFoundError, match="CSV not found"):
        load_rows(conn, csv_path=tmp_path / "nonexistent.csv")
    conn.close()


# ---------------------------------------------------------------------------
# Input validation (_validate_csv)
# ---------------------------------------------------------------------------

class TestLoaderValidation:
    _BASE = {
        "project": "prj1",
        "subject": "sub1",
        "condition": "melanoma",
        "age": 30,
        "sex": "M",
        "treatment": "miraclib",
        "response": "yes",
        "sample": "smp1",
        "sample_type": "PBMC",
        "time_from_treatment_start": 0,
        "b_cell": 100,
        "cd8_t_cell": 200,
        "cd4_t_cell": 150,
        "nk_cell": 50,
        "monocyte": 80,
    }

    def _df(self, **overrides):
        return pd.DataFrame([{**self._BASE, **overrides}])

    def test_missing_column_raises(self):
        df = pd.DataFrame([{k: v for k, v in self._BASE.items() if k != "age"}])
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        msg = str(exc.value)
        assert "missing" in msg
        assert "age" in msg

    def test_extra_column_raises(self):
        df = self._df()
        df["extra_col"] = "oops"
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        msg = str(exc.value)
        assert "unexpected" in msg
        assert "extra_col" in msg

    def test_duplicate_sample_raises(self):
        row2 = {**self._BASE, "subject": "sub2"}
        df = pd.DataFrame([self._BASE, row2])
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        assert "duplicate sample" in str(exc.value)

    def test_negative_count_raises(self):
        df = self._df(b_cell=-1)
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        assert "negative" in str(exc.value)

    def test_zero_total_raises(self):
        df = self._df(b_cell=0, cd8_t_cell=0, cd4_t_cell=0, nk_cell=0, monocyte=0)
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        assert "zero total" in str(exc.value)

    def test_subject_drift_raises(self):
        row2 = {**self._BASE, "sample": "smp2", "condition": "carcinoma"}
        df = pd.DataFrame([self._BASE, row2])
        with pytest.raises(ValueError) as exc:
            _validate_csv(df)
        msg = str(exc.value)
        assert "sub1" in msg
        assert "condition" in msg


# ---------------------------------------------------------------------------
# Idempotency: build_database twice → identical counts, no IntegrityError
# ---------------------------------------------------------------------------

def test_build_database_idempotent(tmp_path):
    db_path = tmp_path / "idem.db"
    build_database(db_path=db_path)

    counts_first = {}
    c = get_connection(db_path)
    for table in ("projects", "subjects", "samples", "cell_counts"):
        counts_first[table] = c.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
    c.close()

    # Second run must not raise and must produce identical counts.
    build_database(db_path=db_path)

    c = get_connection(db_path)
    for table in ("projects", "subjects", "samples", "cell_counts"):
        count = c.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        assert count == counts_first[table], f"{table} count changed on second run"
    c.close()
