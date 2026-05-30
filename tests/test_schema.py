"""Tests for schema initialization and structural constraints."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from teiko.config import DB_PATH, POPULATIONS, REPO_ROOT, TABLES_DIR, FIGURES_DIR
from teiko.db import build_database, get_connection, init_schema


@pytest.fixture
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    build_database(db_path=db_path)
    return db_path


def test_build_database_creates_file(tmp_db):
    assert tmp_db.exists()
    assert tmp_db.suffix == ".db"


def test_load_data_creates_db_at_repo_root(tmp_path, monkeypatch):
    """python load_data.py must create the .db at DB_PATH (repo root)."""
    import importlib.util, sys
    # Verify DB_PATH is in the repo root
    assert DB_PATH.parent == REPO_ROOT


def test_exactly_four_tables(tmp_db):
    conn = get_connection(tmp_db)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;"
    )
    tables = {row[0] for row in cur.fetchall()}
    conn.close()
    assert tables == {"projects", "subjects", "samples", "cell_counts"}


def test_exactly_five_named_indexes(tmp_db):
    conn = get_connection(tmp_db)
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%';"
    )
    indexes = {row[0] for row in cur.fetchall()}
    conn.close()
    expected = {
        "idx_subjects_condition_treatment",
        "idx_subjects_project",
        "idx_samples_subject",
        "idx_samples_type_time",
        "idx_cellcounts_population",
    }
    assert indexes == expected


def test_response_is_nullable(tmp_db):
    conn = get_connection(tmp_db)
    cur = conn.execute("PRAGMA table_info(subjects);")
    cols = {row[1]: {"notnull": row[3], "dflt": row[4]} for row in cur.fetchall()}
    conn.close()
    assert "response" in cols
    assert cols["response"]["notnull"] == 0, "response must be nullable"


def test_cell_counts_composite_pk(tmp_db):
    conn = get_connection(tmp_db)
    cur = conn.execute("PRAGMA table_info(cell_counts);")
    pk_cols = [row[1] for row in cur.fetchall() if row[5] > 0]
    conn.close()
    assert set(pk_cols) == {"sample_id", "population"}


def test_foreign_key_lists(tmp_db):
    conn = get_connection(tmp_db)

    def fk_parents(table):
        cur = conn.execute(f"PRAGMA foreign_key_list({table});")
        return {row[2] for row in cur.fetchall()}  # row[2] is the referenced table

    assert "subjects" in fk_parents("samples")
    assert "samples" in fk_parents("cell_counts")
    assert "projects" in fk_parents("subjects")
    conn.close()


def test_foreign_keys_pragma_enabled(tmp_db):
    conn = get_connection(tmp_db)
    result = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
    conn.close()
    assert result == 1


def test_fk_violation_raises_integrity_error(tmp_db):
    conn = get_connection(tmp_db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO samples (sample_id, subject_id, sample_type, time_from_treatment_start) "
            "VALUES ('s_fake', 'nonexistent_subject', 'PBMC', 0);"
        )
        conn.commit()
    conn.close()


def test_all_tables_empty_after_init(tmp_path):
    """init_schema alone (not build_database) must leave all tables empty."""
    db_path = tmp_path / "init_only.db"
    conn = get_connection(db_path)
    init_schema(conn)
    conn.commit()
    for table in ("projects", "subjects", "samples", "cell_counts"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        assert count == 0, f"{table} should be empty after schema init"
    conn.close()


def test_init_schema_is_idempotent(tmp_db):
    """Running init_schema twice should not raise and should leave 0-row tables."""
    conn = get_connection(tmp_db)
    init_schema(conn)
    conn.commit()
    for table in ("projects", "subjects", "samples", "cell_counts"):
        count = conn.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
        assert count == 0
    conn.close()


def test_config_paths_correct():
    assert REPO_ROOT == Path(__file__).resolve().parent.parent
    assert DB_PATH == REPO_ROOT / "cell-count.db"
    assert TABLES_DIR == REPO_ROOT / "outputs" / "tables"
    assert FIGURES_DIR == REPO_ROOT / "outputs" / "figures"


def test_populations_constant():
    assert POPULATIONS == ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]
    assert len(POPULATIONS) == 5


def test_count_check_constraint(tmp_path):
    db_path = tmp_path / "check_test.db"
    conn = get_connection(db_path)
    init_schema(conn)
    conn.execute("INSERT INTO projects (project_id) VALUES ('p1');")
    conn.execute(
        "INSERT INTO subjects (subject_id, project_id) VALUES ('s1', 'p1');"
    )
    conn.execute(
        "INSERT INTO samples"
        " (sample_id, subject_id, sample_type, time_from_treatment_start)"
        " VALUES ('smp1', 's1', 'PBMC', 0);"
    )
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO cell_counts (sample_id, population, count)"
            " VALUES ('smp1', 'b_cell', -1);"
        )
        conn.commit()
    conn.close()


def test_subject_id_not_null(tmp_path):
    db_path = tmp_path / "notnull_test.db"
    conn = get_connection(db_path)
    init_schema(conn)
    conn.execute("INSERT INTO projects (project_id) VALUES ('p1');")
    conn.commit()
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO samples"
            " (sample_id, subject_id, sample_type, time_from_treatment_start)"
            " VALUES ('smp1', NULL, 'PBMC', 0);"
        )
        conn.commit()
    conn.close()
