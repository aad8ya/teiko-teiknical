"""Part 3 statistical analysis: responder vs non-responder comparisons with BH-FDR correction."""

import os
from pathlib import Path

import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

from teiko.config import POPULATIONS, TABLES_DIR
from teiko.db import get_connection, load_sample_metadata


STATS_COLS = [
    "population",
    "n_responder",
    "n_nonresponder",
    "median_responder",
    "median_nonresponder",
    "mean_responder",
    "mean_nonresponder",
    "mean_difference",
    "median_difference",
    "rank_biserial",
    "t_stat",
    "t_p_raw",
    "t_p_bh",
    "mwu_stat",
    "mwu_p_raw",
    "mwu_p_bh",
    "significant_primary",
    "significant_high_confidence",
]


def load_frequency_with_metadata() -> pd.DataFrame:
    """
    Read the Part 2 summary table and join it with per-sample metadata from the DB.

    Returns one row per (sample, population) with columns:
    sample, population, percentage, condition, treatment, sample_type, response,
    time_from_treatment_start, subject (plus other metadata columns from the DB join).
    """
    freq_path = TABLES_DIR / "cell_frequencies.csv"
    freq = pd.read_csv(freq_path)

    conn = get_connection()
    try:
        meta = load_sample_metadata(conn)
    finally:
        conn.close()

    merged = freq.merge(meta, on="sample", how="inner")
    assert len(merged) == 52_500, f"Expected 52500 merged rows, got {len(merged)}"
    return merged


def part3_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to melanoma + miraclib + PBMC samples with a known response (yes or no)."""
    mask = (
        (df["condition"] == "melanoma")
        & (df["treatment"] == "miraclib")
        & (df["sample_type"] == "PBMC")
        & (df["response"].isin(["yes", "no"]))
    )
    return df[mask].copy()


def compare_population(responder_vals: pd.Series, nonresponder_vals: pd.Series) -> dict:
    """
    Run Welch's t-test and Mann-Whitney U on two groups; return raw statistics.

    BH-FDR adjustment is applied by run_scope across populations, not here.
    """
    r = responder_vals.values
    nr = nonresponder_vals.values

    t_stat, t_p_raw = scipy_stats.ttest_ind(r, nr, equal_var=False)
    mwu_stat, mwu_p_raw = scipy_stats.mannwhitneyu(r, nr, alternative="two-sided")

    median_r = float(pd.Series(r).median())
    median_nr = float(pd.Series(nr).median())
    mean_r = float(pd.Series(r).mean())
    mean_nr = float(pd.Series(nr).mean())

    return {
        "n_responder": len(r),
        "n_nonresponder": len(nr),
        "median_responder": median_r,
        "median_nonresponder": median_nr,
        "mean_responder": mean_r,
        "mean_nonresponder": mean_nr,
        "mean_difference": mean_r - mean_nr,
        "median_difference": median_r - median_nr,
        "rank_biserial": (2 * float(mwu_stat)) / (len(r) * len(nr)) - 1,
        "t_stat": float(t_stat),
        "t_p_raw": float(t_p_raw),
        "mwu_stat": float(mwu_stat),
        "mwu_p_raw": float(mwu_p_raw),
    }


def run_scope(long_df: pd.DataFrame, scope_label: str) -> pd.DataFrame:  # noqa: ARG001
    """
    Compare responders vs non-responders for each population in long_df.

    Applies BH-FDR across the 5 populations separately for t-test and MWU raw p-values.
    A population is flagged significant only when *both* BH-adjusted p-values < 0.05.
    """
    rows = []
    for pop in POPULATIONS:
        pop_df = long_df[long_df["population"] == pop]
        resp = pop_df[pop_df["response"] == "yes"]["percentage"]
        nonresp = pop_df[pop_df["response"] == "no"]["percentage"]
        row = {"population": pop}
        row.update(compare_population(resp, nonresp))
        rows.append(row)

    result = pd.DataFrame(rows)

    _, t_p_bh, _, _ = multipletests(result["t_p_raw"].values, method="fdr_bh")
    _, mwu_p_bh, _, _ = multipletests(result["mwu_p_raw"].values, method="fdr_bh")

    result["t_p_bh"] = t_p_bh
    result["mwu_p_bh"] = mwu_p_bh
    result["significant_primary"] = result["mwu_p_bh"] < 0.05
    result["significant_high_confidence"] = (result["t_p_bh"] < 0.05) & (result["mwu_p_bh"] < 0.05)

    return result[STATS_COLS]


def write_significance_summary(
    pooled: pd.DataFrame,
    baseline: pd.DataFrame,
    per_tp: pd.DataFrame,
    path: Path = None,
) -> None:
    """Write a human-readable markdown summary of significant findings per scope."""
    if path is None:
        path = TABLES_DIR / "stats_summary.md"

    def primary_sig_pops(df: pd.DataFrame) -> list[str]:
        return df[df["significant_primary"]]["population"].tolist()

    def high_conf_pops(df: pd.DataFrame) -> list[str]:
        return df[df["significant_high_confidence"]]["population"].tolist()

    def fmt_baseline_pop(pop: str) -> str:
        row = baseline[baseline["population"] == pop].iloc[0]
        return (
            f"{pop}: mwu_p_bh={row['mwu_p_bh']:.4g}, "
            f"rank_biserial={row['rank_biserial']:.3f}, "
            f"median_difference={row['median_difference']:.3f}"
        )

    def fmt_pooled_pop(pop: str) -> str:
        row = pooled[pooled["population"] == pop].iloc[0]
        return f"{pop}: mwu_p_bh={row['mwu_p_bh']:.4g}, rank_biserial={row['rank_biserial']:.3f}"

    lines = [
        "## Primary analysis — baseline (n=656 subjects, one PBMC sample per subject at time=0)",
        "",
        "Comparison of relative cell-population frequencies between responders and non-responders "
        "among melanoma patients treated with miraclib, restricted to a single pre-treatment "
        "PBMC sample per subject (time_from_treatment_start = 0).",
        "",
    ]

    baseline_sig = primary_sig_pops(baseline)
    if baseline_sig:
        for pop in baseline_sig:
            lines.append(fmt_baseline_pop(pop) + ".")
    else:
        lines.append(
            "No population reached the primary significance threshold (`mwu_p_bh < 0.05`) at baseline."
        )

    lines += [
        "",
        "---",
        "",
        "## Sensitivity — pooled (n=1968 samples, 656 subjects × 3 timepoints)",
        "",
        "Pooling all three timepoints per subject violates statistical independence because each "
        "subject contributes three samples. This analysis is reported here as a higher-power "
        "exploratory cross-check, not the primary result.",
        "",
    ]

    pooled_sig = primary_sig_pops(pooled)
    if pooled_sig:
        for pop in pooled_sig:
            lines.append(fmt_pooled_pop(pop) + ".")
    else:
        lines.append(
            "No population reached the primary significance threshold (`mwu_p_bh < 0.05`) in the pooled analysis."
        )

    lines += [
        "",
        "---",
        "",
        "## Exploratory — per-timepoint (n=656 each at t=0, 7, 14)",
        "",
        "Three independent within-timepoint analyses, each BH-corrected over the 5 populations. "
        "Useful for tracking whether responder-versus-non-responder differences change with time on treatment.",
        "",
    ]

    tp_sig_rows = per_tp[per_tp["significant_primary"]]
    if tp_sig_rows.empty:
        lines.append("No population reached significance at any individual timepoint.")
    else:
        for _, row in tp_sig_rows.iterrows():
            lines.append(
                f"(t={int(row['timepoint'])}) {row['population']}: "
                f"mwu_p_bh={row['mwu_p_bh']:.4g}, rb={row['rank_biserial']:.3f}"
            )

    lines += [
        "",
        "---",
        "",
        "## High-confidence cross-check",
        "",
    ]

    hc = high_conf_pops(baseline)
    if hc:
        pop_str = ", ".join(hc)
        lines.append(
            f"Baseline populations satisfying both tests at adjusted p < 0.05: {pop_str}."
        )
    else:
        lines.append("No baseline population satisfied both tests at adjusted p < 0.05.")

    path.write_text("\n".join(lines))


def run_all_stats() -> None:
    """Run the full Part 3 statistical analysis and write all output files."""
    os.makedirs(TABLES_DIR, exist_ok=True)

    df = load_frequency_with_metadata()
    pooled_df = part3_subset(df)

    # Pooled (primary): all timepoints combined
    pooled_stats = run_scope(pooled_df, "pooled")
    pooled_stats.to_csv(TABLES_DIR / "stats_pooled.csv", index=False)

    # Baseline (sensitivity): time == 0 only, one sample per subject
    baseline_df = pooled_df[pooled_df["time_from_treatment_start"] == 0].copy()
    baseline_stats = run_scope(baseline_df, "baseline")
    baseline_stats.to_csv(TABLES_DIR / "stats_baseline.csv", index=False)

    # Per-timepoint: separate BH family for each of 0, 7, 14
    timepoint_frames = []
    for tp in [0, 7, 14]:
        tp_df = pooled_df[pooled_df["time_from_treatment_start"] == tp].copy()
        tp_stats = run_scope(tp_df, f"timepoint_{tp}")
        tp_stats.insert(0, "timepoint", tp)
        timepoint_frames.append(tp_stats)

    per_tp_stats = pd.concat(timepoint_frames, ignore_index=True)
    per_tp_stats.to_csv(TABLES_DIR / "stats_per_timepoint.csv", index=False)

    write_significance_summary(pooled_stats, baseline_stats, per_tp_stats)
