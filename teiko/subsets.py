"""Part 4 baseline subset queries and breakdowns."""

import os

import pandas as pd

from teiko.config import DB_PATH, TABLES_DIR
from teiko.db import get_connection


def baseline_subset(conn) -> pd.DataFrame:
    """Return melanoma+miraclib+PBMC samples at baseline (time_from_treatment_start=0)."""
    sql = """
        SELECT
            sa.sample_id                    AS sample,
            su.subject_id                   AS subject,
            p.project_id                    AS project,
            su.condition,
            su.treatment,
            sa.sample_type,
            sa.time_from_treatment_start,
            su.response,
            su.sex
        FROM samples sa
        JOIN subjects su ON sa.subject_id = su.subject_id
        JOIN projects  p ON su.project_id = p.project_id
        WHERE su.condition = 'melanoma'
          AND su.treatment = 'miraclib'
          AND sa.sample_type = 'PBMC'
          AND sa.time_from_treatment_start = 0
        ORDER BY sa.sample_id
    """
    return pd.read_sql_query(sql, conn)


def by_project(df: pd.DataFrame) -> pd.DataFrame:
    """Sample-level count by project. Projects present in data only — never hardcoded."""
    if df.empty:
        return pd.DataFrame(columns=["project", "n_samples"])
    result = df.groupby("project").size().reset_index(name="n_samples")
    return result.sort_values("project").reset_index(drop=True)


def by_response(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-level count by response (deduplicated on subject)."""
    if df.empty:
        return pd.DataFrame(columns=["response", "n_subjects"])
    subjects = df.drop_duplicates("subject")
    result = subjects.groupby("response").size().reset_index(name="n_subjects")
    return result.sort_values("response").reset_index(drop=True)


def by_sex(df: pd.DataFrame) -> pd.DataFrame:
    """Subject-level count by sex (deduplicated on subject)."""
    if df.empty:
        return pd.DataFrame(columns=["sex", "n_subjects"])
    subjects = df.drop_duplicates("subject")
    result = subjects.groupby("sex").size().reset_index(name="n_subjects")
    return result.sort_values("sex").reset_index(drop=True)


def write_outputs(subset_df, project_df, response_df, sex_df) -> None:
    os.makedirs(TABLES_DIR, exist_ok=True)

    subset_df[
        [
            "sample",
            "subject",
            "project",
            "condition",
            "treatment",
            "sample_type",
            "time_from_treatment_start",
            "response",
            "sex",
        ]
    ].to_csv(TABLES_DIR / "part4_baseline_samples.csv", index=False)

    project_df.to_csv(TABLES_DIR / "part4_by_project.csv", index=False)
    response_df.to_csv(TABLES_DIR / "part4_by_response.csv", index=False)
    sex_df.to_csv(TABLES_DIR / "part4_by_sex.csv", index=False)


def run(conn=None):
    """Query the DB, compute the three breakdowns, write four CSVs, and return them."""
    close_after = conn is None
    if conn is None:
        conn = get_connection(DB_PATH)
    try:
        subset_df = baseline_subset(conn)
        project_df = by_project(subset_df)
        response_df = by_response(subset_df)
        sex_df = by_sex(subset_df)
        write_outputs(subset_df, project_df, response_df, sex_df)
        return subset_df, project_df, response_df, sex_df
    finally:
        if close_after:
            conn.close()
