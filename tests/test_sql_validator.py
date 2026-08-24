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
