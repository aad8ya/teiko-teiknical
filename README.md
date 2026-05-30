# Teiko Labs Take-Home Assignment

Live Dashboard: [https://teiko-teiknical-adithya.streamlit.app/](https://teiko-teiknical-adithya.streamlit.app/)

A Python data pipeline and interactive dashboard analyzing immune cell-count data from a clinical
trial. The source data (`data/cell-count.csv`) measures five immune cell populations
(`b_cell`, `cd8_t_cell`, `cd4_t_cell`, `nk_cell`, `monocyte`) across 10,500 biological samples
collected from 3,500 subjects at three timepoints. The pipeline loads the data into a normalized
SQLite database, computes per-sample relative frequencies, runs a responder vs non-responder
statistical comparison on the melanoma + miraclib + PBMC subset, queries a baseline subset with
three demographic breakdowns, and renders all results in a Streamlit dashboard.

## Table of contents

1. [Run and reproduce](#1-run-and-reproduce)
2. [Database schema and scalability](#2-database-schema-and-scalability)
3. [Code structure](#3-code-structure)
4. [Deployed dashboard](#4-deployed-dashboard)
5. [Data notes](#data-notes)
6. [Analysis methodology](#analysis-methodology)

---

## 1. Run and reproduce

The grader is expected to run this project in GitHub Codespaces. The repository ships a
`.devcontainer/devcontainer.json` pinned to `mcr.microsoft.com/devcontainers/python:3.12` with
`postCreateCommand: "make setup"` and port `8501` forwarded. Opening the repository in Codespaces
installs the dependencies on container creation; subsequent commands are the three Make targets
required by the assignment.

### Three commands

```bash
make setup
make pipeline
make dashboard
```

- `make setup` runs `pip install -r requirements.txt`. The eight pinned dependencies are
  `pandas==2.2.3`, `numpy==2.1.3`, `scipy==1.14.1`, `statsmodels==0.14.4`, `matplotlib==3.9.2`,
  `seaborn==0.13.2`, `streamlit==1.40.2`, and `pytest==8.3.4`. `sqlite3` is in the Python standard
  library and is therefore not pinned.
- `make pipeline` runs `python load_data.py && python run_pipeline.py`. The first command
  initializes the SQLite schema and loads all 10,500 sample rows into `cell-count.db` at the
  repository root. The second command runs the four analysis stages in dependency order: Part 2
  frequencies, Part 3 statistics, Part 3 boxplot figures, and Part 4 baseline subset queries.
  All artifacts are written to `outputs/tables/` and `outputs/figures/`.
- `make dashboard` runs `streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port 8501`.
  Binding to `0.0.0.0` allows the Codespaces port-forward to surface the dashboard in the browser.

After `make pipeline` completes the following artifacts exist:

| Path | Rows | Description |
|------|------|-------------|
| `cell-count.db` | (binary) | Normalized SQLite database |
| `outputs/tables/cell_frequencies.csv` | 52,500 | Part 2 summary table |
| `outputs/tables/stats_pooled.csv` | 5 | Part 3 pooled analysis |
| `outputs/tables/stats_baseline.csv` | 5 | Part 3 baseline-only sensitivity analysis |
| `outputs/tables/stats_per_timepoint.csv` | 15 | Part 3 per-timepoint analysis |
| `outputs/tables/stats_summary.md` | (prose) | Plain-language significance summary |
| `outputs/tables/part4_baseline_samples.csv` | 656 | Part 4 baseline subset |
| `outputs/tables/part4_by_project.csv` | 2 | Part 4 sample counts by project |
| `outputs/tables/part4_by_response.csv` | 2 | Part 4 subject counts by response |
| `outputs/tables/part4_by_sex.csv` | 2 | Part 4 subject counts by sex |
| `outputs/figures/boxplot_<population>.png` | (image) | One figure per population |
| `outputs/figures/boxplots_combined.png` | (image) | All five populations side by side |
| `outputs/figures/boxplots_per_timepoint.png` | (image) | Per-timepoint hue-by-response figure |

The `outputs/` directory and `cell-count.db` are tracked in version control. This is intentional:
the deployed dashboard on Streamlit Community Cloud reads these pre-generated artifacts and
therefore does not need to run the pipeline at boot time.

### Running tests

The repository ships 161 tests covering schema, loader, frequencies, statistics, plots, subsets,
end-to-end pipeline, and a headless dashboard smoke test. To run them:

```bash
pytest
```

The `pyproject.toml` configures `pythonpath = ["."]` and `testpaths = ["tests"]` so `pytest` picks
up the suite without any flags.

---

## 2. Database schema and scalability

### Schema

The schema lives in `teiko/db.py` as `SCHEMA_DDL`. Four normalized tables and five indexes:

```sql
CREATE TABLE IF NOT EXISTS projects (
    project_id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS subjects (
    subject_id  TEXT PRIMARY KEY,
    project_id  TEXT REFERENCES projects(project_id),
    condition   TEXT,
    age         INTEGER,
    sex         TEXT,
    treatment   TEXT,
    response    TEXT
);

CREATE TABLE IF NOT EXISTS samples (
    sample_id                   TEXT PRIMARY KEY,
    subject_id                  TEXT REFERENCES subjects(subject_id),
    sample_type                 TEXT,
    time_from_treatment_start   INTEGER
);

CREATE TABLE IF NOT EXISTS cell_counts (
    sample_id  TEXT REFERENCES samples(sample_id),
    population TEXT,
    count      INTEGER,
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
```

Connections opened through `teiko.db.get_connection` set `PRAGMA foreign_keys = ON` so referential
integrity is enforced on every insert. The `subjects.response` column is nullable; healthy
subjects in the source data have no recorded response and the loader writes `NULL` rather than an
empty string.

### Normalization rationale

Profiling the source CSV confirmed that each subject appears at three timepoints (timepoints 0,
7, and 14) with identical values across `project`, `condition`, `sex`, `treatment`, `response`,
and `age`. Storing these subject-invariant fields once in `subjects` rather than three times in a
wide sample table reduces the row width of the high-cardinality table (`samples`) and prevents
the class of bug where a subject's metadata diverges between timepoints. The `subjects` table
also provides a natural target for queries that operate at the subject level, such as the Part 4
response and sex breakdowns.

The `cell_counts` table is stored in long format with a composite primary key
`(sample_id, population)`. This decision is the most consequential of the four. A wide layout
(`samples` with five population columns) would require schema changes to add a sixth analyte and
would force every population-level filter to be written as a `UNION ALL` or as five separate
column references. In long format, `WHERE population = ?` and
`GROUP BY population` become first-class operations and adding a new population requires only new
data rows. The cost is a 5x row multiplier on `cell_counts`, which is acceptable for the
expected data volumes and is addressed by the population index below.

### Indexing

Five named indexes cover the actual query patterns used by the analysis modules:

| Index | Supports |
|-------|----------|
| `idx_subjects_condition_treatment` | The Part 3 and Part 4 filter `condition='melanoma' AND treatment='miraclib'`. |
| `idx_subjects_project` | The Part 4 "samples per project" breakdown and any future per-project rollups. |
| `idx_samples_subject` | The `samples`-to-`subjects` join used by `load_sample_metadata` and every Part 3 / Part 4 query. |
| `idx_samples_type_time` | The composite filter `sample_type='PBMC' AND time_from_treatment_start=0` used by Part 4 baseline queries. |
| `idx_cellcounts_population` | Population-level scans used by Part 2 aggregation and any analysis filtering on a specific cell type. |

### Scaling to hundreds of projects and thousands of samples

The schema is designed to extend without structural changes. Three concrete dimensions:

**Indexing.** The five existing indexes already cover the most expensive filter and join paths.
At the scale of hundreds of projects and tens of thousands of samples, the composite
`idx_samples_type_time` index keeps the Part 4 baseline filter selective without a full table
scan, and `idx_subjects_condition_treatment` keeps the Part 3 filter selective. Additional
indexes can be added as new analyses surface new filter columns. SQLite supports up to
2,000-column indexes and per-column statistics through `ANALYZE`, which the loader can run after
each rebuild to keep the query planner accurate.

**Normalization.** Because subject-invariant metadata is stored exactly once, growing the dataset
to thousands of subjects multiplies row counts in `subjects` linearly but does not duplicate
metadata into `samples` or `cell_counts`. A new project adds one row to `projects`; new subjects
add one row each to `subjects`; new samples add one row each to `samples` and five rows each to
`cell_counts`. None of these changes require schema migrations.

**Query patterns.** The long-format `cell_counts` table makes population-level aggregations
expressible as ordinary `GROUP BY` queries. Common analytics rollups (per-population frequency,
per-population mean by condition, per-project sample counts) reduce to a single
`SELECT ... GROUP BY ...` and execute against the existing indexes. The `load_sample_metadata`
helper provides a one-row-per-sample DataFrame that the analysis modules merge with frequency
data in pandas; this pattern moves the join cost out of every analytical query while keeping the
storage layer normalized. For a larger deployment the same schema migrates cleanly to PostgreSQL
or a columnar store: the table boundaries and primary keys are unchanged, and the indexes map
one-to-one.

For workloads beyond a single SQLite file (write concurrency, multi-user analytics), the
recommended next steps are: (1) switch to PostgreSQL using the same DDL, (2) add a materialized
view of `cell_frequencies` if the per-sample percentage is read more often than the raw counts,
and (3) partition `cell_counts` by `population` if a single analyte dominates query traffic.
None of these require touching the analysis code; the analysis modules call
`teiko.db.get_connection` and consume the result through SQL or pandas.

---

## 3. Code structure

### Repository layout

```
teiko-teiknical/
├── data/cell-count.csv               # Source data (10,500 samples)
├── cell-count.db                     # SQLite database (committed)
├── load_data.py                      # Entry point: initialize and load the database
├── run_pipeline.py                   # Entry point: run all four analysis stages
├── streamlit_app.py                  # Entry point: interactive dashboard
├── teiko/                            # Analysis package
│   ├── __init__.py
│   ├── config.py                     # Shared path constants and POPULATIONS list
│   ├── db.py                         # Schema DDL, connection helpers, loader, metadata join
│   ├── frequencies.py                # Part 2: per-sample relative frequency table
│   ├── stats.py                      # Part 3: Welch + Mann-Whitney U + BH-FDR
│   ├── plots.py                      # Part 3: boxplot figures
│   └── subsets.py                    # Part 4: baseline subset and breakdowns
├── outputs/
│   ├── tables/                       # CSVs and the significance summary markdown
│   └── figures/                      # Seven PNG figures
├── tests/                            # 161 tests across nine files
├── .devcontainer/devcontainer.json   # Python 3.12 container for Codespaces
├── .streamlit/config.toml            # Headless server config
├── .gitignore
├── Makefile                          # setup, pipeline, dashboard targets
├── pyproject.toml                    # pytest and ruff configuration
└── requirements.txt                  # Eight pinned dependencies
```

### Entry points at the repository root

Three Python files sit at the repository root. Each is an entry point that can be invoked
directly with `python <file>.py`:

- **`load_data.py`** initializes the database and loads the CSV. The file is two lines:
  `from teiko.db import build_database` and `if __name__ == "__main__": build_database()`. The
  assignment requires this file to be at the root and runnable without arguments or
  `python -m`; both constraints are satisfied.
- **`run_pipeline.py`** orchestrates the four analysis stages. It does not touch the database
  schema. It calls `teiko.frequencies.run()`, `teiko.stats.run_all_stats()`,
  `teiko.plots.generate_all_plots()`, and `teiko.subsets.run()` in order, printing one progress
  line per stage to stdout. The `make pipeline` target runs `load_data.py` first to enforce that
  the database exists before the analysis stages query it.
- **`streamlit_app.py`** is the Streamlit dashboard. It reads committed artifacts from
  `outputs/` and the committed `cell-count.db`. Cached loaders return `None` if an artifact is
  missing rather than raising, so the app boots cleanly even before `make pipeline` has run.

### The `teiko/` analysis package

The package separates concerns by analysis stage. Each module exports a small, named API and is
covered by a dedicated `tests/test_<module>.py` file:

- **`config.py`** holds the path constants (`REPO_ROOT`, `DB_PATH`, `CSV_PATH`, `OUTPUT_DIR`,
  `TABLES_DIR`, `FIGURES_DIR`) and the canonical population order (`POPULATIONS`). All paths are
  derived from `__file__` so they resolve regardless of the working directory the entry point is
  invoked from.
- **`db.py`** owns the schema DDL, the connection helper (`get_connection`), the schema
  initializer (`init_schema`), the loader (`load_rows`), the top-level rebuild
  (`build_database`), and the one-row-per-sample metadata join (`load_sample_metadata`). Every
  other analysis module reads data through this single entry point.
- **`frequencies.py`** implements Part 2. The public `run` function reads `cell_counts` through
  SQL, computes `total_count` and `percentage` in pandas, and writes
  `outputs/tables/cell_frequencies.csv` at full float precision. Rounding is a display concern
  handled by the dashboard; the stored artifact preserves precision because the Part 3 statistics
  consume the same file.
- **`stats.py`** implements Part 3. The public `run_all_stats` function runs three scopes
  (baseline primary, pooled sensitivity, per-timepoint exploratory), each through a shared
  `run_scope` helper. Each scope applies Mann-Whitney U (primary) and Welch's t-test per
  population, then corrects across the five populations with the Benjamini-Hochberg FDR. Two
  significance flags are produced: `significant_primary` (MWU BH p < 0.05) and
  `significant_high_confidence` (both tests agree). Effect sizes (`mean_difference`,
  `median_difference`, `rank_biserial`) are included in every stats CSV. A plain-language
  summary is written to `stats_summary.md`.
- **`plots.py`** implements the Part 3 figures. The module sets the matplotlib `Agg` backend
  before any other matplotlib import so headless rendering works under CI and Streamlit Cloud.
  It produces five per-population boxplots, one combined 1x5 figure, and one per-timepoint
  figure hued by response. Each per-population figure is annotated with the corresponding
  BH-adjusted p-values from `stats_baseline.csv` (the primary scope).
- **`subsets.py`** implements Part 4. The public `run` function executes the baseline filter
  (`condition='melanoma' AND treatment='miraclib' AND sample_type='PBMC' AND
  time_from_treatment_start=0`), then writes the full baseline list plus three breakdowns: by
  project (sample-level), by response (subject-level after deduplication), and by sex
  (subject-level after deduplication).

### Why `outputs/` is committed

The grading workflow runs `make setup && make pipeline && make dashboard` from a clean checkout,
so the pipeline regenerates every artifact. The deployed Streamlit Cloud instance, by contrast,
needs the artifacts present at boot. Committing the directories (with `.gitkeep` placeholders to
keep the empty directories tracked) gives the deployed dashboard a working state without
requiring the Cloud worker to install scipy / statsmodels / matplotlib and run the full pipeline
each cold start. The pipeline overwrites these files deterministically on each run, so the
committed state stays current as long as `make pipeline` is run before a commit.

### Tests

The `tests/` directory contains 161 tests across nine files:

| File | Tests | Scope |
|------|-------|-------|
| `tests/conftest.py` | (fixtures) | Session-scoped database fixtures |
| `tests/test_schema.py` | 13 | DDL, indexes, foreign keys, idempotency |
| `tests/test_loader.py` | 28 | Row counts, vocab, FK integrity, subject invariance |
| `tests/test_frequencies.py` | 16 | Per-sample sums, sort order, oracle spot-checks |
| `tests/test_stats.py` | 33 | Subset filters, BH ordering, unit tests on synthetic data |
| `tests/test_plots.py` | 11 | Backend, file existence, PNG magic-byte validation |
| `tests/test_subsets.py` | 28 | Oracle row counts and breakdown totals |
| `tests/test_pipeline_e2e.py` | 22 | End-to-end run into an isolated temp directory |
| `tests/test_dashboard.py` | 10 | Streamlit `AppTest` headless smoke test |

The session-scoped fixtures in `tests/conftest.py` build a fresh database from the real CSV into
a temporary path so unit tests do not touch the committed `cell-count.db`. The end-to-end tests
run the full pipeline into a patched output directory and assert the row-count oracles. The
dashboard smoke tests use `streamlit.testing.v1.AppTest.from_file("streamlit_app.py").run()` so
they execute without a browser, server, or display.

---

## 4. Deployed dashboard

Dashboard: [https://teiko-teiknical-adithya.streamlit.app/](https://teiko-teiknical-adithya.streamlit.app/)

The dashboard reads the committed `outputs/` artifacts and `cell-count.db`, so the deployed instance is functional
immediately after the build finishes.

The dashboard has four sections selected through a sidebar radio:

- **Overview.** Headline counts (samples, subjects, projects, populations) recomputed live from
  the database, plus the value vocabularies for `condition`, `sample_type`, and `treatment`.
- **Part 2.** The 52,500-row frequency table with a population filter, a sample-ID substring
  filter, and a CSV download of the filtered result.
- **Part 3.** The auto-generated significance narrative, the five per-population boxplots, the
  combined figure, the pooled statistics table, plus expanders for the baseline sensitivity
  analysis and the per-timepoint analysis.
- **Part 4.** Headline subject count, three breakdown tables paired with bar charts, and the
  full 656-row baseline sample list in an expander with a CSV download.

---

## Data notes

### Column name mapping

The assignment specification names the columns `sample_id`, `indication`, and `gender`. The actual CSV header at `data/cell-count.csv` uses different names, and the actual
file contains four additional columns the specification does not mention. The pipeline code uses
the actual column names everywhere; this section documents the drift.

| Specification name | Actual CSV name | Notes |
|--------------------|-----------------|-------|
| `sample_id` | `sample` | Sample identifier (e.g. `sample00000`). |
| `indication` | `condition` | Clinical condition (melanoma, carcinoma, healthy). |
| `gender` | `sex` | Subject sex (M, F). |
| `treatment` | `treatment` | Same name. Vocab: miraclib, phauximab, none. |
| `response` | `response` | Same name. Blank for healthy subjects. |
| `time_from_treatment_start` | `time_from_treatment_start` | Same name. Values: 0, 7, 14. |
| (not in spec) | `project` | New: project identifier (prj1, prj2, prj3). Required by Part 4. |
| (not in spec) | `subject` | New: subject identifier. Each subject appears at three timepoints. |
| (not in spec) | `age` | New: subject age in years. |
| (not in spec) | `sample_type` | New: sample type (PBMC, WB). Required by Part 3 and Part 4. |

The five population columns (`b_cell`, `cd8_t_cell`, `cd4_t_cell`, `nk_cell`, `monocyte`) match
the specification exactly.

### Data shape and quality

Profiling the source CSV confirmed the following invariants, which the loader tests assert:

- 10,500 sample rows, 3,500 subjects, 3 projects.
- Each subject has exactly three timepoints (0, 7, 14).
- Subject-level metadata (project, condition, sex, treatment, response, age) is invariant
  across a subject's three timepoints. The loader deduplicates on `subject` and uses the first
  occurrence; the test suite asserts that this is lossless.
- All cell counts are non-negative integers. Per-sample totals range from 84,247 to 122,788; no
  sample has a total of zero.
- The only blank fields are `response` on the 1,422 healthy / treatment-none rows. The loader
  writes these as SQL `NULL`, not as the empty string.
- Value vocabularies (sample counts):
  - `condition`: melanoma 5,175; carcinoma 3,903; healthy 1,422.
  - `treatment`: miraclib 4,695; phauximab 4,383; none 1,422.
  - `sample_type`: PBMC 7,500; WB 3,000.
  - `response`: yes 4,611; no 4,467; blank 1,422.
  - `sex`: M 5,430; F 5,070.
  - `project`: prj1 4,500; prj2 3,000; prj3 3,000.

---

## Analysis methodology

### Part 2: per-sample relative frequency

For each of the 10,500 samples, the per-sample `total_count` is the sum of all five population
counts, and `percentage` is `100 * count / total_count`. The result is one row per
(sample, population), so the artifact has 52,500 rows. Percentages are stored at full float
precision because the Part 3 statistics consume the same file; the dashboard rounds to two
decimal places for display. The CSV is sorted by `sample` ascending then by `population` in the
canonical `POPULATIONS` order.

### Part 3: responder vs non-responder comparison

The Part 3 subset is melanoma patients receiving miraclib whose samples are PBMC. After applying
this filter the analysis covers 1,968 samples (993 responders, 975 non-responders) drawn from
656 subjects across the three timepoints. The aim is to identify cell populations whose relative
frequency differs between responders and non-responders.

**Two statistical tests, applied to each population:**

- **Mann-Whitney U** (`scipy.stats.mannwhitneyu(alternative="two-sided")`). The primary test. A
  non-parametric rank-based test appropriate for bounded frequency data (percentages are
  constrained to [0, 100]) that may not be normally distributed.
- **Welch's t-test** (`scipy.stats.ttest_ind(equal_var=False)`). A parametric two-sample test
  that does not assume equal variances; retained as a secondary check.

**Multiple comparison correction.** Each test produces five raw p-values (one per population).
These are corrected for multiple comparisons using the Benjamini-Hochberg false discovery rate
(`statsmodels.stats.multitest.multipletests(method="fdr_bh")`). Each test family is corrected
independently: the five t-test p-values are corrected together, and the five Mann-Whitney
p-values are corrected together.

**Effect sizes.** Each population also reports `mean_difference` (mean_responder −
mean_nonresponder), `median_difference` (median_responder − median_nonresponder), and
`rank_biserial` (1 − 2U / (n_r × n_nr), where positive values indicate responders tend to have
higher frequencies).

**Significance flags.** Two flags are included in every stats CSV:

- `significant_primary`: `mwu_p_bh < 0.05`. The primary significance call, driven by
  Mann-Whitney alone after FDR correction.
- `significant_high_confidence`: both `mwu_p_bh < 0.05` and `t_p_bh < 0.05`. Agreement between
  the parametric and non-parametric tests is a stronger claim and more likely to convince a
  reviewer.

**Three scopes, each clearly labeled.** Pooling all 1,968 samples violates statistical
independence: each subject contributes up to three samples. Rather than choose silently between
the literal specification (pooled) and the statistically clean variant (baseline only), the
pipeline runs all three and labels them:

- **Baseline** (primary): 656 samples at `time_from_treatment_start = 0`, one per subject.
  Independence is satisfied; this scope directly addresses the specification's aim of predicting
  response from a pre-treatment state.
- **Pooled** (sensitivity): 1,968 samples across all three timepoints. Reported as a
  higher-power exploratory cross-check with the independence violation stated in plain language.
- **Per-timepoint** (exploratory): three independent 656-sample analyses at timepoints 0, 7, and
  14, each with its own BH correction over the five populations. Useful for tracking whether
  responder-versus-non-responder differences change with time on treatment.

**Actual results.** The auto-generated `outputs/tables/stats_summary.md` is the source of truth;
this README does not hardcode population-level p-values.

### Part 4: baseline subset and demographic breakdowns

The Part 4 subset adds `time_from_treatment_start = 0` to the Part 3 filter. The result is 656
samples, which equals 656 subjects (each subject contributes one baseline sample). The three
breakdowns are:

- **By project** (sample-level count): prj1 = 384, prj3 = 272. Note that prj2 is absent from this
  subset; the code derives the project list from the data rather than hardcoding three rows.
- **By response** (subject-level count after deduplication): no = 325, yes = 331.
- **By sex** (subject-level count after deduplication): F = 312, M = 344.

Project counts are sample-level by specification. Response and sex counts are subject-level
because the question is about subjects, not samples; at baseline these are equivalent because
each subject contributes exactly one sample, but the dedup is left in place defensively.

### Verification oracles

The following counts are asserted by the test suite (`tests/test_loader.py`,
`tests/test_frequencies.py`, `tests/test_stats.py`, `tests/test_subsets.py`,
`tests/test_pipeline_e2e.py`):

| Quantity | Value |
|----------|-------|
| Samples loaded | 10,500 |
| Subjects loaded | 3,500 |
| Projects loaded | 3 |
| `cell_counts` rows loaded | 52,500 |
| Part 2 summary rows | 52,500 |
| Per-sample percentage sum | 100.000 (tolerance 1e-6) |
| Per-sample total range | 84,247 to 122,788 |
| Part 3 subset samples | 1,968 |
| Part 3 subset by response | no = 975, yes = 993 |
| Part 3 subset subjects | 656 |
| Baseline subset samples | 656 |
| Baseline subset by response | no = 325, yes = 331 |
| Baseline subset by sex | F = 312, M = 344 |
| Baseline subset by project | prj1 = 384, prj3 = 272 (prj2 absent) |

These values are baked into the regression tests so a future data refresh that breaks any of
them fails CI rather than silently producing different numbers.
