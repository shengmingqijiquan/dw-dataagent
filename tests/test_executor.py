"""DuckDB Executor 测试。"""
import pytest
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TABLES


@pytest.fixture
def executor(tmp_path):
    ex = DuckDBExecutor(str(tmp_path / "test.duckdb"), TABLES)
    ex.setup()
    yield ex
    ex.close()


def test_setup_creates_all_tables(executor):
    import duckdb
    con = duckdb.connect(executor.path)
    names = {r[0] for r in con.execute(
        "SELECT table_name FROM information_schema.tables").fetchall()}
    assert set(TABLES.keys()) == names
    con.close()


def test_execute_returns_rows(executor):
    import duckdb
    con = duckdb.connect(executor.path)
    con.execute("INSERT INTO dim_category_info VALUES (1, '服饰', 0)")
    con.close()
    rows = executor.execute(
        "SELECT category_id, category_name FROM dim_category_info")
    assert rows == [(1, "服饰")]


def test_bad_sql_raises_query_error(executor):
    with pytest.raises(QueryError):
        executor.execute("SELECT * FROM nonexistent_table")


def test_ddl_restricted(executor):
    with pytest.raises(QueryError):
        executor.execute("DROP TABLE dim_category_info")
