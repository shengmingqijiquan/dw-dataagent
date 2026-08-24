"""StarRocks Executor 无连接测试（只读护栏与未初始化路径，无需真实服务）。"""
import pytest
from dataagent.executor.base import QueryError
from dataagent.executor.starrocks_executor import StarRocksExecutor
from dataagent.warehouse.schema import TABLES


def _executor():
    """不调用 setup()，不建立任何连接。"""
    return StarRocksExecutor("localhost", 9030, "root", "", TABLES)


@pytest.mark.parametrize("sql", [
    "DROP TABLE dim_category_info",
    "INSERT INTO dim_category_info VALUES (1, 'x', 0)",
    "DELETE FROM dim_category_info",
    "UPDATE dim_category_info SET status='off'",
    "CREATE TABLE x (a INT)",
    "ALTER TABLE dim_category_info ADD COLUMN x INT",
    "TRUNCATE TABLE dim_category_info",
])
def test_ddl_dml_blocked_without_connection(sql):
    # 关键字前缀拦截在触碰连接之前触发（纵深防御层，见 executor 注释）
    with pytest.raises(QueryError):
        _executor().execute(sql)


def test_select_without_setup_raises_query_error():
    # 未 setup 时 _con 为 None，SELECT 路径统一抛 QueryError 而非裸 AttributeError
    with pytest.raises(QueryError):
        _executor().execute("SELECT 1")


def test_multi_statement_blocked_without_connection():
    # M3: 分号多语句（SELECT 1; DROP …）在触碰连接之前拦截（纵深防御层）
    with pytest.raises(QueryError):
        _executor().execute("SELECT 1; DROP TABLE dim_category_info")
