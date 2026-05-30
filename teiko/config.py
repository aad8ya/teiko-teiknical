"""Shared path constants and configuration used across all teiko modules."""

from pathlib import Path

POPULATIONS = ["b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte"]

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "cell-count.db"
CSV_PATH = REPO_ROOT / "data" / "cell-count.csv"
OUTPUT_DIR = REPO_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
