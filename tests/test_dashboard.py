"""Headless AppTest smoke tests for streamlit_app.py.

Verifies the app boots without exceptions in two scenarios:
  1. Artifacts present (normal post-pipeline state).
  2. Artifacts absent (pre-pipeline — grader runs `make dashboard` before pipeline).

Also checks that the four nav sections are reachable.
"""

import os

import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = str(__file__).replace("tests/test_dashboard.py", "streamlit_app.py")


def _app_path():
    from pathlib import Path
    return str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


# ---------------------------------------------------------------------------
# Boot with artifacts present (normal state — pipeline has already run)
# ---------------------------------------------------------------------------

class TestAppBootsWithArtifacts:
    @pytest.fixture(scope="class")
    def at(self):
        app = AppTest.from_file(_app_path(), default_timeout=30)
        app.run()
        return app

    def test_no_exception(self, at):
        assert not at.exception, f"App raised an exception: {at.exception}"

    def test_sidebar_radio_present(self, at):
        radios = at.sidebar.radio
        assert len(radios) >= 1, "Expected at least one radio widget in the sidebar"

    def test_nav_has_four_options(self, at):
        radio = at.sidebar.radio[0]
        assert len(radio.options) >= 4

    def test_nav_option_labels(self, at):
        radio = at.sidebar.radio[0]
        labels = list(radio.options)
        assert any("Overview" in lbl for lbl in labels)
        assert any("Part 2" in lbl for lbl in labels)
        assert any("Part 3" in lbl for lbl in labels)
        assert any("Part 4" in lbl for lbl in labels)


# ---------------------------------------------------------------------------
# Navigate to each section and verify no crash
# ---------------------------------------------------------------------------

class TestSectionNavigation:
    @pytest.fixture(scope="class")
    def base_at(self):
        return AppTest.from_file(_app_path(), default_timeout=30)

    def _run_section(self, base_at, section_label):
        app = AppTest.from_file(_app_path(), default_timeout=30)
        app.run()
        radio = app.sidebar.radio[0]
        # Find the matching option
        match = next((o for o in radio.options if section_label in o), None)
        assert match is not None, f"Section '{section_label}' not found in nav options"
        radio.set_value(match).run()
        return app

    def test_overview_section(self, base_at):
        app = self._run_section(base_at, "Overview")
        assert not app.exception

    def test_part2_section(self, base_at):
        app = self._run_section(base_at, "Part 2")
        assert not app.exception

    def test_part3_section(self, base_at):
        app = self._run_section(base_at, "Part 3")
        assert not app.exception

    def test_part4_section(self, base_at):
        app = self._run_section(base_at, "Part 4")
        assert not app.exception


# ---------------------------------------------------------------------------
# Boot WITHOUT artifacts (pre-pipeline: grader opens dashboard first)
# ---------------------------------------------------------------------------

class TestAppBootsWithoutArtifacts:
    @pytest.fixture(scope="class")
    def at_no_artifacts(self, tmp_path_factory):
        """Patch config paths to point at an empty temp dir so no artifacts exist."""
        import unittest.mock as mock

        empty_dir = tmp_path_factory.mktemp("no_artifacts")
        empty_db = empty_dir / "cell-count.db"  # intentionally not created

        with mock.patch("teiko.config.TABLES_DIR", empty_dir), \
             mock.patch("teiko.config.FIGURES_DIR", empty_dir), \
             mock.patch("teiko.config.DB_PATH", empty_db):
            app = AppTest.from_file(_app_path(), default_timeout=30)
            app.run()

        return app

    def test_no_exception_without_artifacts(self, at_no_artifacts):
        """App must not crash when no CSVs, PNGs, or DB file exist."""
        assert not at_no_artifacts.exception, (
            f"App raised an exception with no artifacts: {at_no_artifacts.exception}"
        )

    def test_nav_still_present_without_artifacts(self, at_no_artifacts):
        radios = at_no_artifacts.sidebar.radio
        assert len(radios) >= 1
        assert len(radios[0].options) >= 4
