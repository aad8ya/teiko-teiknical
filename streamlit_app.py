"""Interactive dashboard for the immune cell-count clinical trial analysis."""

import pandas as pd
import streamlit as st

from teiko.config import DB_PATH, FIGURES_DIR, POPULATIONS, TABLES_DIR
from teiko.db import get_connection

st.set_page_config(layout="wide", page_title="Immune Cell Analysis — Teiko")

# ---------------------------------------------------------------------------
# Cached artifact loaders
# ---------------------------------------------------------------------------

@st.cache_data
def load_frequencies() -> pd.DataFrame | None:
    path = TABLES_DIR / "cell_frequencies.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_stats(scope: str) -> pd.DataFrame | None:
    path = TABLES_DIR / f"stats_{scope}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data
def load_summary() -> str | None:
    path = TABLES_DIR / "stats_summary.md"
    if not path.exists():
        return None
    return path.read_text()


@st.cache_data
def load_subsets() -> dict[str, pd.DataFrame | None]:
    names = ["baseline_samples", "by_project", "by_response", "by_sex"]
    out = {}
    for name in names:
        path = TABLES_DIR / f"part4_{name}.csv"
        out[name] = pd.read_csv(path) if path.exists() else None
    return out


@st.cache_data
def load_overview_counts() -> dict | None:
    if not DB_PATH.exists():
        return None
    conn = get_connection(DB_PATH)
    try:
        n_samples = conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0]
        n_subjects = conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
        n_projects = conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0]
        conditions = [r[0] for r in conn.execute(
            "SELECT DISTINCT condition FROM subjects ORDER BY condition"
        ).fetchall()]
        sample_types = [r[0] for r in conn.execute(
            "SELECT DISTINCT sample_type FROM samples ORDER BY sample_type"
        ).fetchall()]
        treatments = [r[0] for r in conn.execute(
            "SELECT DISTINCT treatment FROM subjects ORDER BY treatment"
        ).fetchall()]
        return {
            "n_samples": n_samples,
            "n_subjects": n_subjects,
            "n_projects": n_projects,
            "n_populations": len(POPULATIONS),
            "conditions": conditions,
            "sample_types": sample_types,
            "treatments": treatments,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

section = st.sidebar.radio(
    "Navigate",
    ["Overview", "Part 2: Cell Frequencies", "Part 3: Responder Analysis", "Part 4: Baseline Subset"],
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

if section == "Overview":
    st.title("Immune Cell-Count Clinical Trial Analysis")
    st.write(
        "This dashboard summarizes a clinical trial where immune cell populations were profiled "
        "across patient samples. Use the sidebar to explore the frequency table (Part 2), "
        "responder vs. non-responder statistics (Part 3), and baseline subset breakdowns (Part 4)."
    )

    counts = load_overview_counts()
    if counts is None:
        st.warning("Database not found. Run `make pipeline` to initialize it.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Samples", f"{counts['n_samples']:,}")
        c2.metric("Subjects", f"{counts['n_subjects']:,}")
        c3.metric("Projects", counts["n_projects"])
        c4.metric("Cell populations", counts["n_populations"])

        st.divider()
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("Conditions")
            for c in counts["conditions"]:
                st.write(f"- {c}")
        with col_b:
            st.subheader("Sample types")
            for s in counts["sample_types"]:
                st.write(f"- {s}")
        with col_c:
            st.subheader("Treatments")
            for t in counts["treatments"]:
                st.write(f"- {t}")

    st.divider()
    st.subheader("Database schema")
    st.markdown(
        """
        The data is stored in a normalized **SQLite** database with four tables:

        | Table | Rows | Description |
        |---|---|---|
        | `projects` | 3 | One row per project (prj1 / prj2 / prj3) |
        | `subjects` | 3,500 | Subject-level metadata: condition, age, sex, treatment, response |
        | `samples` | 10,500 | One row per biological sample; linked to subject and project |
        | `cell_counts` | 52,500 | Long-format: one row per (sample, population), with raw count |

        **Design rationale:** Each subject appears at three timepoints. Storing condition, sex,
        treatment, and response once in `subjects` avoids 3× repetition and prevents inconsistent
        updates. Long-format `cell_counts` means adding a new cell population requires only new
        rows, not schema changes.

        **Indexed on:** `subjects(condition, treatment)`, `subjects(project_id)`,
        `samples(subject_id)`, `samples(sample_type, time_from_treatment_start)`,
        `cell_counts(population)` — covering the most common join and filter paths.
        """
    )

# ---------------------------------------------------------------------------
# Part 2: Cell Frequencies
# ---------------------------------------------------------------------------

elif section == "Part 2: Cell Frequencies":
    st.title("Part 2 — Relative Frequency of Each Cell Population")
    st.write(
        "For each sample the total cell count is the sum across all five populations. "
        "The percentage is each population's share of that total."
    )

    df = load_frequencies()
    if df is None:
        st.warning("Cell frequency table not found. Run `make pipeline` to generate it.")
        st.stop()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        pop_filter = st.multiselect(
            "Filter by population",
            options=POPULATIONS,
            default=POPULATIONS,
        )
    with col2:
        sample_search = st.text_input("Filter by sample ID (substring match)", value="")

    filtered = df[df["population"].isin(pop_filter)]
    if sample_search:
        filtered = filtered[filtered["sample"].str.contains(sample_search, case=False, na=False)]

    display = filtered.copy()
    display["percentage"] = display["percentage"].round(2)

    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption(f"{len(display):,} rows shown")

    csv_bytes = filtered.to_csv(index=False).encode()
    st.download_button(
        label="Download filtered table as CSV",
        data=csv_bytes,
        file_name="cell_frequencies_filtered.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------------
# Part 3: Responder Analysis
# ---------------------------------------------------------------------------

elif section == "Part 3: Responder Analysis":
    st.title("Part 3 — Responders vs. Non-Responders (melanoma, miraclib, PBMC)")
    st.write(
        "Comparing relative cell-population frequencies between patients who responded "
        "to miraclib treatment and those who did not. Only PBMC samples from melanoma patients "
        "on miraclib are included."
    )

    # Pull the summary once at the top — used for both the info callout and the expander below.
    summary_md = load_summary()
    if summary_md:
        chunks = summary_md.split("\n\n")
        first_para = ""
        found_heading = False
        for chunk in chunks:
            if chunk.strip().startswith("## "):
                found_heading = True
                continue
            if found_heading and chunk.strip():
                first_para = chunk.strip()
                break
        st.info(first_para if first_para else summary_md)

    # --- Baseline primary results ---
    st.subheader("Primary analysis — baseline (n = 656 subjects, time = 0)")
    st.write(
        "One pre-treatment PBMC sample per subject. This scope satisfies the independence "
        "assumption and matches the specification's aim of predicting response before treatment begins."
    )

    baseline_df = load_stats("baseline")
    if baseline_df is not None:
        styled = baseline_df.style.format({
            "mwu_p_bh": "{:.4f}",
            "t_p_bh": "{:.4f}",
            "mwu_p_raw": "{:.4f}",
            "t_p_raw": "{:.4f}",
            "rank_biserial": "{:.3f}",
            "mean_difference": "{:.3f}",
            "median_difference": "{:.3f}",
        })
        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.warning("stats_baseline.csv not found. Run `make pipeline`.")

    # --- Boxplots (annotations from baseline) ---
    st.divider()
    st.subheader("Boxplots (annotations from baseline analysis)")

    pop_labels = {
        "b_cell": "B Cell",
        "cd8_t_cell": "CD8 T Cell",
        "cd4_t_cell": "CD4 T Cell",
        "nk_cell": "NK Cell",
        "monocyte": "Monocyte",
    }
    pop_cols = st.columns(len(POPULATIONS))
    for col, pop in zip(pop_cols, POPULATIONS):
        img_path = FIGURES_DIR / f"boxplot_{pop}.png"
        if img_path.exists():
            col.image(str(img_path), caption=pop_labels[pop], use_container_width=True)
        else:
            col.warning(f"{pop} plot missing")

    combined_path = FIGURES_DIR / "boxplots_combined.png"
    if combined_path.exists():
        st.image(str(combined_path), caption="All populations — combined view", use_container_width=True)
    else:
        st.warning("Combined boxplot not found. Run `make pipeline`.")

    # --- Full statistical summary ---
    if summary_md:
        with st.expander("Full statistical summary"):
            st.markdown(summary_md)
    else:
        st.warning("Stats summary not found. Run `make pipeline` to generate it.")

    # --- Pooled sensitivity in expander ---
    with st.expander("Sensitivity — pooled analysis (n=1968, repeated measures)"):
        st.write(
            "Pooling all three timepoints per subject violates statistical independence — "
            "each of the 656 subjects contributes three samples. This is reported as a "
            "higher-power exploratory cross-check, not the primary result."
        )
        pooled_df = load_stats("pooled")
        if pooled_df is not None:
            st.dataframe(pooled_df, use_container_width=True, hide_index=True)
        else:
            st.warning("stats_pooled.csv not found. Run `make pipeline`.")

    # --- Per-timepoint expander ---
    with st.expander("Exploratory — per-timepoint analysis"):
        st.write(
            "Three independent analyses at timepoints 0, 7, and 14, each BH-corrected over "
            "the 5 populations. Useful for tracking how responder vs. non-responder differences "
            "evolve with time on treatment."
        )
        pt_df = load_stats("per_timepoint")
        if pt_df is not None:
            st.dataframe(pt_df, use_container_width=True, hide_index=True)
        else:
            st.warning("stats_per_timepoint.csv not found. Run `make pipeline`.")

        pt_img = FIGURES_DIR / "boxplots_per_timepoint.png"
        if pt_img.exists():
            st.image(str(pt_img), caption="Per-timepoint boxplots", use_container_width=True)

# ---------------------------------------------------------------------------
# Part 4: Baseline Subset
# ---------------------------------------------------------------------------

elif section == "Part 4: Baseline Subset":
    st.title("Part 4 — Baseline Subset: melanoma, miraclib, PBMC, time = 0")
    st.write(
        "This subset contains **656 samples** from melanoma patients treated with miraclib, "
        "collected at baseline (time from treatment start = 0), PBMC sample type only. "
        "Because all samples are at the same timepoint, each sample corresponds to a unique subject."
    )

    subsets = load_subsets()

    if all(v is None for v in subsets.values()):
        st.warning("Part 4 outputs not found. Run `make pipeline` to generate them.")
        st.stop()

    by_response = subsets["by_response"]
    if by_response is not None:
        st.metric("Baseline subjects", int(by_response["n_subjects"].sum()))

    # --- By project ---
    st.subheader("Samples by project")
    proj_df = subsets["by_project"]
    if proj_df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(proj_df, use_container_width=True, hide_index=True)
        with col2:
            st.bar_chart(proj_df.set_index("project")["n_samples"])
    else:
        st.warning("part4_by_project.csv not found.")

    st.divider()

    # --- By response ---
    st.subheader("Subjects by response (responder / non-responder)")
    resp_df = subsets["by_response"]
    if resp_df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(resp_df, use_container_width=True, hide_index=True)
        with col2:
            st.bar_chart(resp_df.set_index("response")["n_subjects"])
    else:
        st.warning("part4_by_response.csv not found.")

    st.divider()

    # --- By sex ---
    st.subheader("Subjects by sex")
    sex_df = subsets["by_sex"]
    if sex_df is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(sex_df, use_container_width=True, hide_index=True)
        with col2:
            st.bar_chart(sex_df.set_index("sex")["n_subjects"])
    else:
        st.warning("part4_by_sex.csv not found.")

    st.divider()

    # --- Full baseline sample list ---
    with st.expander("Full baseline sample list (656 rows)"):
        baseline = subsets["baseline_samples"]
        if baseline is not None:
            st.dataframe(baseline, use_container_width=True, hide_index=True)
            csv_bytes = baseline.to_csv(index=False).encode()
            st.download_button(
                "Download baseline samples CSV",
                data=csv_bytes,
                file_name="part4_baseline_samples.csv",
                mime="text/csv",
            )
        else:
            st.warning("part4_baseline_samples.csv not found.")
