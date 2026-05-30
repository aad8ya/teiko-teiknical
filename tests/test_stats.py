"""Tests for teiko/stats.py — oracle-gated + unit tests."""

import math
import shutil

import numpy as np
import pandas as pd
import pytest

from teiko.config import POPULATIONS, TABLES_DIR
from teiko.db import build_database, get_connection
from teiko.stats import (
    STATS_COLS,
    compare_population,
    load_frequency_with_metadata,
    part3_subset,
    run_all_stats,
    run_scope,
)


@pytest.fixture(scope="module")
def full_df():
    """Load the merged frequency+metadata DataFrame once for the whole module."""
    return load_frequency_with_metadata()


@pytest.fixture(scope="module")
def pooled(full_df):
    return part3_subset(full_df)


class TestPart3Subset:
    def test_sample_count(self, pooled):
        assert pooled["sample"].nunique() == 1_968

    def test_response_split(self, pooled):
        counts = pooled.drop_duplicates("sample")["response"].value_counts().to_dict()
        assert counts["no"] == 975
        assert counts["yes"] == 993

    def test_subject_count(self, pooled):
        assert pooled["subject"].nunique() == 656

    def test_only_melanoma_miraclib_pbmc(self, pooled):
        assert (pooled["condition"] == "melanoma").all()
        assert (pooled["treatment"] == "miraclib").all()
        assert (pooled["sample_type"] == "PBMC").all()

    def test_no_blank_response(self, pooled):
        assert pooled["response"].isin(["yes", "no"]).all()


class TestBaselineFilter:
    def test_sample_count(self, pooled):
        baseline = pooled[pooled["time_from_treatment_start"] == 0]
        assert baseline["sample"].nunique() == 656

    def test_one_sample_per_subject(self, pooled):
        baseline = pooled[pooled["time_from_treatment_start"] == 0]
        assert baseline["subject"].nunique() == baseline["sample"].nunique()

    def test_response_split(self, pooled):
        baseline = pooled[pooled["time_from_treatment_start"] == 0]
        counts = baseline.drop_duplicates("subject")["response"].value_counts().to_dict()
        assert counts["no"] == 325
        assert counts["yes"] == 331


class TestComparePopulationUnit:
    """Synthetic unit tests that don't depend on real data."""

    def test_identical_groups_high_p(self):
        vals = pd.Series([10.0, 11.0, 10.5, 9.8, 10.2] * 20)
        result = compare_population(vals, vals)
        assert result["t_p_raw"] == pytest.approx(1.0)
        assert result["mwu_p_raw"] > 0.9

    def test_clearly_shifted_groups_small_p(self):
        rng = np.random.default_rng(42)
        a = pd.Series(rng.normal(loc=10, scale=1, size=100))
        b = pd.Series(rng.normal(loc=30, scale=1, size=100))
        result = compare_population(a, b)
        assert result["t_p_raw"] < 1e-10
        assert result["mwu_p_raw"] < 1e-10

    def test_returns_expected_keys(self):
        vals = pd.Series([1.0, 2.0, 3.0])
        result = compare_population(vals, vals)
        expected = {
            "n_responder", "n_nonresponder", "median_responder", "median_nonresponder",
            "mean_responder", "mean_nonresponder", "mean_difference", "median_difference",
            "rank_biserial", "t_stat", "t_p_raw", "mwu_stat", "mwu_p_raw",
        }
        assert expected <= set(result.keys())

    def test_effect_sizes_responders_dominate(self):
        # When every responder value exceeds every non-responder value, rb == 1.0
        resp = pd.Series([10.0, 11.0, 12.0])
        nonresp = pd.Series([1.0, 2.0, 3.0])
        result = compare_population(resp, nonresp)
        assert result["rank_biserial"] == pytest.approx(1.0)
        assert result["mean_difference"] > 0

    def test_effect_sizes_identical_groups(self):
        vals = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = compare_population(vals, vals)
        assert result["rank_biserial"] == pytest.approx(0.0)
        assert result["mean_difference"] == pytest.approx(0.0)

    def test_n_counts(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([4.0, 5.0])
        result = compare_population(a, b)
        assert result["n_responder"] == 3
        assert result["n_nonresponder"] == 2


class TestRunScope:
    def test_bh_never_below_raw_t(self, pooled):
        result = run_scope(pooled, "pooled")
        assert (result["t_p_bh"] >= result["t_p_raw"] - 1e-12).all()

    def test_bh_never_below_raw_mwu(self, pooled):
        result = run_scope(pooled, "pooled")
        assert (result["mwu_p_bh"] >= result["mwu_p_raw"] - 1e-12).all()

    def test_column_order(self, pooled):
        result = run_scope(pooled, "pooled")
        assert list(result.columns) == STATS_COLS

    def test_significant_primary_is_mwu_bh(self, pooled):
        result = run_scope(pooled, "pooled")
        expected = result["mwu_p_bh"] < 0.05
        pd.testing.assert_series_equal(result["significant_primary"], expected, check_names=False)

    def test_significant_high_confidence_is_both_tests(self, pooled):
        result = run_scope(pooled, "pooled")
        expected = (result["t_p_bh"] < 0.05) & (result["mwu_p_bh"] < 0.05)
        pd.testing.assert_series_equal(result["significant_high_confidence"], expected, check_names=False)

    def test_effect_size_columns_present(self, pooled):
        result = run_scope(pooled, "pooled")
        for col in ("mean_difference", "median_difference", "rank_biserial"):
            assert col in result.columns

    def test_five_rows_one_per_population(self, pooled):
        result = run_scope(pooled, "pooled")
        assert len(result) == 5
        assert list(result["population"]) == POPULATIONS


class TestStatsPooledCsv:
    @pytest.fixture(scope="class")
    def pooled_csv(self, tmp_path_factory, pooled):
        out = tmp_path_factory.mktemp("stats") / "stats_pooled.csv"
        result = run_scope(pooled, "pooled")
        result.to_csv(out, index=False)
        return pd.read_csv(out)

    def test_column_count_and_order(self, pooled_csv):
        assert list(pooled_csv.columns) == STATS_COLS

    def test_no_significant_bh_column(self, pooled_csv):
        assert "significant_bh" not in pooled_csv.columns

    def test_n_counts_sum_to_1968_per_pop(self, pooled_csv):
        for _, row in pooled_csv.iterrows():
            assert row["n_responder"] + row["n_nonresponder"] == 1_968


class TestPerTimepointCsv:
    @pytest.fixture(scope="class")
    def per_tp(self, pooled):
        frames = []
        for tp in [0, 7, 14]:
            tp_df = pooled[pooled["time_from_treatment_start"] == tp].copy()
            tp_stats = run_scope(tp_df, f"tp_{tp}")
            tp_stats.insert(0, "timepoint", tp)
            frames.append(tp_stats)
        return pd.concat(frames, ignore_index=True)

    def test_row_count(self, per_tp):
        assert len(per_tp) == 15

    def test_each_timepoint_has_five_rows(self, per_tp):
        for tp in [0, 7, 14]:
            assert len(per_tp[per_tp["timepoint"] == tp]) == 5

    def test_each_timepoint_n_sums_to_656(self, per_tp):
        for tp in [0, 7, 14]:
            tp_df = per_tp[per_tp["timepoint"] == tp]
            for _, row in tp_df.iterrows():
                assert row["n_responder"] + row["n_nonresponder"] == 656


class TestNoStaleColumnNames:
    """Guard: code must use actual CSV column names, not the stale spec names."""

    def test_no_indication_column(self, full_df):
        assert "indication" not in full_df.columns

    def test_no_gender_column(self, full_df):
        assert "gender" not in full_df.columns

    def test_no_sample_id_column(self, full_df):
        assert "sample_id" not in full_df.columns

    def test_uses_condition(self, full_df):
        assert "condition" in full_df.columns

    def test_uses_sex(self, full_df):
        assert "sex" in full_df.columns

    def test_uses_sample(self, full_df):
        assert "sample" in full_df.columns


class TestRunAllStats:
    """Integration test: run_all_stats writes all four output files correctly."""

    @pytest.fixture(scope="class")
    def stats_outdir(self, tmp_path_factory):
        """
        Run run_all_stats once with TABLES_DIR redirected to a temp dir.
        The committed cell_frequencies.csv is copied in so load_frequency_with_metadata
        can still read it from the same patched TABLES_DIR.
        """
        from teiko.config import TABLES_DIR as REAL_TABLES_DIR
        import teiko.stats as stats_mod
        import teiko.config as cfg

        out = tmp_path_factory.mktemp("stats_out")
        shutil.copy(REAL_TABLES_DIR / "cell_frequencies.csv", out / "cell_frequencies.csv")

        mp = pytest.MonkeyPatch()
        mp.setattr(stats_mod, "TABLES_DIR", out)
        mp.setattr(cfg, "TABLES_DIR", out)
        try:
            run_all_stats()
        finally:
            mp.undo()
        return out

    def test_artifacts_created(self, stats_outdir):
        for fname in [
            "stats_pooled.csv",
            "stats_baseline.csv",
            "stats_per_timepoint.csv",
            "stats_summary.md",
        ]:
            assert (stats_outdir / fname).exists(), f"{fname} not created"

    def test_pooled_row_count(self, stats_outdir):
        df = pd.read_csv(stats_outdir / "stats_pooled.csv")
        assert len(df) == 5

    def test_baseline_row_count(self, stats_outdir):
        df = pd.read_csv(stats_outdir / "stats_baseline.csv")
        assert len(df) == 5

    def test_per_timepoint_row_count(self, stats_outdir):
        df = pd.read_csv(stats_outdir / "stats_per_timepoint.csv")
        assert len(df) == 15

    def test_summary_md_mentions_scopes(self, stats_outdir):
        text = (stats_outdir / "stats_summary.md").read_text()
        assert "pooled" in text.lower()
        assert "baseline" in text.lower()
        assert "timepoint" in text.lower()

    def test_summary_md_baseline_first(self, stats_outdir):
        text = (stats_outdir / "stats_summary.md").read_text()
        # Baseline section must appear before the pooled section
        assert text.index("## Primary analysis") < text.index("## Sensitivity")

    def test_summary_md_has_high_confidence_section(self, stats_outdir):
        text = (stats_outdir / "stats_summary.md").read_text()
        assert "High-confidence" in text
