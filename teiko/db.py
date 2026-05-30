"""
SQLite database access: schema initialization, data loading, and metadata queries.

Schema rationale
----------------
Four normalized tables avoid repeating subject-level metadata (condition, sex, treatment,
response, age) on every sample row. Since each subject appears at three timepoints, storing
those fields once in `subjects` cuts redundancy by 3× and makes a typo or update to one
timepoint impossible to propagate inconsistently.

`cell_counts` is long-format (one row per sample × population). That makes the five
populations a data concern rather than a schema concern — adding a sixth population later
requires no DDL change, only new rows. It also makes population-level aggregations and
filters natural SQL operations (WHERE population = ?).

Scalability notes (hundreds of projects, thousands of samples):
- All FK join paths are indexed: subjects(project_id), samples(subject_id),
  cell_counts(population), samples(sample_type, time_from_treatment_start),
  subjects(condition, treatment).
- Long-format cell_counts scales linearly: N samples → 5N rows; no schema change needed
  for new populations or analytes.
- The subjects table cleanly separates subject-invariant metadata from sample-level events,
  which maps well to a future migration to Postgres or a columnar store.

The loader validates the CSV against the expected column contract (exact set, no duplicates,
non-negative counts, non-zero per-row totals, subject-metadata invariance) before any insert.
"""

import sqlite3
from pathlib import Path

import pandas as pd

from teiko.config import CSV_PATH, DB_PATH, POPULATIONS


SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT NOT NULL PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id  TEXT NOT NULL PRIMARY KEY,
    project_id  TEXT NOT NULL REFERENCES projects(project_id),
    condition   TEXT,
    age         INTEGER,
    sex         TEXT,
    treatment   TEXT,
    response    TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id                   TEXT NOT NULL PRIMARY KEY,
    subject_id                  TEXT NOT NULL REFERENCES subjects(subject_id),
    sample_type                 TEXT,
    time_from_treatment_start   INTEGER
);

CREATE TABLE IF NOT EXISTS cell_counts (
    sample_id  TEXT NOT NULL REFERENCES samples(sample_id),
    population TEXT NOT NULL,
    count      INTEGER NOT NULL CHECK (count >= 0),
    PRIMARY KEY (sample_id, population)
);

CREATE INDEX IF NOT EXISTS idx_subjects_condition_treatment
    ON subjects(condition, treatment);

CREATE INDEX IF NOT EXISTS idx_subjects_project
    ON subjects(project_id);

CREATE INDEX IF NOT EXISTS idx_samples_subject
    ON samples(subject_id);

CREATE INDEX IF NOT EXISTS idx_samples_type_time
    ON samples(sample_type, time_from_treatment_start);

CREATE INDEX IF NOT EXISTS idx_cellcounts_population
    ON cell_counts(population);
"""


_REQUIRED_COLUMNS = [
    "project", "subject", "condition", "age", "sex", "treatment", "response",
    "sample", "sample_type", "time_from_treatment_start",
] + POPULATIONS  # POPULATIONS appended so the list is the canonical contract


def _validate_csv(df: pd.DataFrame) -> None:
    """Raise ValueError (message starting with 'load_data:') on first contract violation."""
    required = set(_REQUIRED_COLUMNS)
    actual = set(df.columns)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing columns: {missing}")
        if unexpected:
            parts.append(f"unexpected columns: {unexpected}")
        raise ValueError(f"load_data: {'; '.join(parts)}")

    dupes = df.loc[df["sample"].duplicated(), "sample"]
    if not dupes.empty:
        raise ValueError(f"load_data: duplicate sample id: {dupes.iloc[0]!r}")

    for pop in POPULATIONS:
        bad = df[df[pop] < 0]
        if not bad.empty:
            row = bad.iloc[0]
            raise ValueError(
                f"load_data: negative count for population {pop!r}"
                f" in sample {row['sample']!r}: {row[pop]}"
            )

    zero_rows = df[df[POPULATIONS].sum(axis=1) == 0]
    if not zero_rows.empty:
        raise ValueError(
            f"load_data: zero total cell count for sample {zero_rows.iloc[0]['sample']!r}"
        )

    inv_cols = ["project", "condition", "sex", "treatment", "response", "age"]
    check = df.copy()
    check["response"] = check["response"].fillna("__BLANK__")
    nunique = check.groupby("subject")[inv_cols].nunique()
    drifted = nunique[nunique.gt(1).any(axis=1)]
    if not drifted.empty:
        subj = drifted.index[0]
        col = nunique.columns[(nunique.loc[subj] > 1).values][0]
        raise ValueError(
            f"load_data: subject {subj!r} has inconsistent values for column {col!r}"
        )


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute("DROP TABLE IF EXISTS cell_counts;")
    conn.execute("DROP TABLE IF EXISTS samples;")
    conn.execute("DROP TABLE IF EXISTS subjects;")
    conn.execute("DROP TABLE IF EXISTS projects;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_DDL)


def load_rows(conn: sqlite3.Connection, csv_path: Path = CSV_PATH) -> None:
    """Read cell-count.csv and insert all rows into the normalized schema."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Blank response cells (healthy subjects) become NULL, not empty string.
    df["response"] = df["response"].where(df["response"].notna(), other=None)

    _validate_csv(df)

    for col in POPULATIONS + ["age", "time_from_treatment_start"]:
        df[col] = df[col].astype(int)

    project_ids = df["project"].unique().tolist()

    subj_df = df[
        ["subject", "project", "condition", "age", "sex", "treatment", "response"]
    ].drop_duplicates("subject")

    sample_df = df[["sample", "subject", "sample_type", "time_from_treatment_start"]]

    counts_df = df[["sample"] + POPULATIONS].melt(
        id_vars="sample",
        value_vars=POPULATIONS,
        var_name="population",
        value_name="count",
    )

    with conn:
        conn.executemany(
            "INSERT INTO projects (project_id) VALUES (?);",
            [(pid,) for pid in project_ids],
        )
        conn.executemany(
            "INSERT INTO subjects"
            " (subject_id, project_id, condition, age, sex, treatment, response)"
            " VALUES (?, ?, ?, ?, ?, ?, ?);",
            [
                (r.subject, r.project, r.condition, int(r.age), r.sex, r.treatment, r.response)
                for r in subj_df.itertuples(index=False)
            ],
        )
        conn.executemany(
            "INSERT INTO samples"
            " (sample_id, subject_id, sample_type, time_from_treatment_start)"
            " VALUES (?, ?, ?, ?);",
            [
                (r.sample, r.subject, r.sample_type, int(r.time_from_treatment_start))
                for r in sample_df.itertuples(index=False)
            ],
        )
        conn.executemany(
            "INSERT INTO cell_counts (sample_id, population, count) VALUES (?, ?, ?);",
            [
                (r.sample, r.population, int(r.count))
                for r in counts_df.itertuples(index=False)
            ],
        )


def load_sample_metadata(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return one row per sample with all subject and project metadata joined in."""
    sql = """
        SELECT
            sa.sample_id                    AS sample,
            su.subject_id                   AS subject,
            p.project_id                    AS project,
            su.condition,
            su.age,
            su.sex,
            su.treatment,
            su.response,
            sa.sample_type,
            sa.time_from_treatment_start
        FROM samples sa
        JOIN subjects su ON sa.subject_id = su.subject_id
        JOIN projects  p ON su.project_id = p.project_id
    """
    return pd.read_sql_query(sql, conn)


def build_database(db_path: Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    try:
        init_schema(conn)
        load_rows(conn)
        conn.commit()
    finally:
        conn.close()
