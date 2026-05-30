"""Shared pytest fixtures for the test suite.

Session-scoped DB fixture builds a full database from the real CSV once per
test session, isolating unit tests from whatever state the committed
cell-count.db happens to be in.
"""

import pytest

from teiko.db import build_database, get_connection


@pytest.fixture(scope="session")
def session_db(tmp_path_factory):
    """Build a fresh DB from the real CSV into a temp path.

    Yields an open connection; the DB file is torn down automatically by
    pytest's tmp_path_factory at session end.
    """
    db_path = tmp_path_factory.mktemp("session_db") / "cell-count.db"
    build_database(db_path=db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def session_db_path(tmp_path_factory):
    """Build a fresh DB and return the path (for tests that need to open their own connections)."""
    db_path = tmp_path_factory.mktemp("session_db_path") / "cell-count.db"
    build_database(db_path=db_path)
    return db_path
