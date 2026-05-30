import os

import teiko.frequencies
import teiko.plots
import teiko.stats
import teiko.subsets
from teiko.config import FIGURES_DIR, TABLES_DIR


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Step 1/4: Computing cell frequency summary table...")
    teiko.frequencies.run()
    print(f"  -> outputs/tables/cell_frequencies.csv written")

    print("Step 2/4: Running statistical analysis (pooled, baseline, per-timepoint)...")
    teiko.stats.run_all_stats()
    print(f"  -> stats_pooled.csv, stats_baseline.csv, stats_per_timepoint.csv, stats_summary.md written")

    print("Step 3/4: Generating boxplot figures...")
    teiko.plots.generate_all_plots()
    print(f"  -> 7 PNG figures written to outputs/figures/")

    print("Step 4/4: Running baseline subset queries...")
    teiko.subsets.run()
    print(f"  -> part4_baseline_samples.csv, part4_by_project.csv, part4_by_response.csv, part4_by_sex.csv written")

    print("\nPipeline complete. All artifacts are in outputs/.")


if __name__ == "__main__":
    main()
