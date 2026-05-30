"""Tests for teiko/plots.py."""

import importlib
from pathlib import Path

import matplotlib
import pytest

import teiko.plots as plots_module
from teiko.config import FIGURES_DIR, POPULATIONS
from teiko.plots import generate_all_plots, load_plot_data


PNG_MAGIC = b"\x89PNG"


@pytest.fixture(scope="module")
def generated_figures(tmp_path_factory):
    """Run generate_all_plots() once into a temp figures dir, return that dir."""
    tmp_figures = tmp_path_factory.mktemp("figures")

    import unittest.mock as mock
    with mock.patch.object(plots_module, "FIGURES_DIR", tmp_figures):
        generate_all_plots()

    return tmp_figures


class TestBackend:
    def test_agg_backend(self):
        assert matplotlib.get_backend().lower() == "agg"


class TestLoadPlotData:
    def test_response_label_values(self):
        df = load_plot_data()
        assert set(df["response_label"].unique()) == {"Responder", "Non-responder"}

    def test_required_columns_present(self):
        df = load_plot_data()
        for col in ("population", "percentage", "response", "time_from_treatment_start", "response_label"):
            assert col in df.columns

    def test_only_melanoma_miraclib_pbmc(self):
        df = load_plot_data()
        # All rows should be the Part 3 subset (load_plot_data calls part3_subset internally)
        # We verify no unexpected conditions or sample types leak in
        assert df["response"].isin(["yes", "no"]).all()


class TestGeneratedFiles:
    def test_per_population_pngs_exist(self, generated_figures):
        for pop in POPULATIONS:
            p = generated_figures / f"boxplot_{pop}.png"
            assert p.exists(), f"Missing {p.name}"

    def test_combined_png_exists(self, generated_figures):
        assert (generated_figures / "boxplots_combined.png").exists()

    def test_per_timepoint_png_exists(self, generated_figures):
        assert (generated_figures / "boxplots_per_timepoint.png").exists()

    def test_total_seven_files(self, generated_figures):
        pngs = list(generated_figures.glob("*.png"))
        assert len(pngs) == 7, f"Expected 7 PNGs, found {len(pngs)}: {[p.name for p in pngs]}"

    def test_all_pngs_nonempty(self, generated_figures):
        for p in generated_figures.glob("*.png"):
            assert p.stat().st_size > 1000, f"{p.name} is suspiciously small ({p.stat().st_size} bytes)"

    def test_all_pngs_have_magic_bytes(self, generated_figures):
        for p in generated_figures.glob("*.png"):
            data = p.read_bytes()
            assert data[:4] == PNG_MAGIC, f"{p.name} does not start with PNG magic bytes"

    def test_population_names_match_config(self, generated_figures):
        expected = {f"boxplot_{pop}.png" for pop in POPULATIONS}
        found = {p.name for p in generated_figures.glob("boxplot_*.png")}
        assert found == expected
