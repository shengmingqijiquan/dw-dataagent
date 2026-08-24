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


def test_multi_statement_rejected(executor):
    # M3: 分号多语句绕过关键字前缀检查（SELECT 1; DROP …），必须被拦
    with pytest.raises(QueryError):
        executor.execute("SELECT 1; DROP TABLE dim_category_info")
    with pytest.raises(QueryError):
        executor.execute("SELECT 1; SELECT 2")


def test_trailing_semicolon_still_allowed(executor):
    # 单条尾分号经 rstrip 归一后照常执行（不误伤）
    rows = executor.execute("SELECT 1;")
    assert rows == [(1,)]
