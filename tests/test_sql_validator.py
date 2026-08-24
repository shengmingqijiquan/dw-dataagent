"""SQL 规则校验测试。"""
from dataagent.guardrails.sql_validator import validate_sql


def test_valid_select_passes():
    result = validate_sql(
        "SELECT dt, SUM(gmv_amount) FROM dws_order_summary_di "
        "WHERE dt >= '2026-07-01' GROUP BY dt", "data_analyst")
    assert result.passed, result.errors


def test_syntax_error_rejected():
    result = validate_sql("SELEC a FROM b", "data_analyst")
    assert not result.passed
    assert any("语法" in e for e in result.errors)


def test_unknown_table_rejected():
    result = validate_sql("SELECT * FROM nonexistent_table", "data_analyst")
    assert not result.passed
    assert any("不存在" in e for e in result.errors)


def test_unauthorized_table_rejected():
    result = validate_sql("SELECT * FROM dws_payment_summary_di", "data_analyst")
    assert not result.passed
    assert any("无权限" in e for e in result.errors)


def test_non_select_rejected():
    for sql in ["DROP TABLE dim_category_info",
                "DELETE FROM dim_category_info",
                "UPDATE dim_category_info SET status='off'",
                "INSERT INTO dim_category_info VALUES (1,'x',0)"]:
        result = validate_sql(sql, "admin")
        assert not result.passed, sql
        assert any("只读" in e for e in result.errors)


def test_fact_table_requires_partition_filter():
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di", "data_analyst")
    assert not result.passed
    assert any("分区" in e for e in result.errors)


def test_partition_filter_passes():
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di WHERE dt >= '2026-07-01'",
        "data_analyst")
    assert result.passed, result.errors


def test_partition_like_column_name_rejected():
    # dt_modified 含 dt 子串但不能替代分区字段 dt（精确列名匹配，R11 回归）
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di WHERE dt_modified >= '2026-07-01'",
        "data_analyst")
    assert not result.passed
    assert any("分区" in e for e in result.errors)


def test_union_branch_without_partition_rejected():
    # UNION 第二分支不带分区过滤，不允许逃逸（全树 WHERE 覆盖，R11 回归）
    result = validate_sql(
        "SELECT COUNT(*) FROM dwd_order_detail_di WHERE dt >= '2026-07-01' "
        "UNION ALL SELECT COUNT(*) FROM dwd_order_detail_di",
        "data_analyst")
    assert not result.passed
    assert any("分区" in e for e in result.errors)


def test_subquery_with_partition_passes():
    # 外层聚合套子查询：DWD 表在子查询内已带分区过滤，
    # 校验只看表直接所在的最内层作用域，外层不得误报（R11a 回归）
    result = validate_sql(
        "SELECT SUM(cnt) FROM (SELECT COUNT(*) cnt FROM dwd_order_detail_di "
        "WHERE dt >= '2026-07-01') t", "data_analyst")
    assert result.passed, result.errors
