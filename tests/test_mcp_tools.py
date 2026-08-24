"""MCP 元数据查询函数测试（含权限过滤）。"""
import pytest
from dataagent.mcp_server.metadata import (
    query_table_list, query_table_schema, query_lineage,
    query_metric_definition, TableNotFoundError,
)


def test_table_list_filtered_by_role():
    tables = query_table_list("data_analyst", None)
    domains = {t["domain"] for t in tables}
    assert domains == {"订单域", "用户域", "商品域"}


def test_table_list_by_domain():
    tables = query_table_list("data_analyst", "订单域")
    assert all(t["domain"] == "订单域" for t in tables)
    assert len(tables) == 6


def test_table_schema_returns_columns():
    schema = query_table_schema("data_analyst", "dws_order_summary_di")
    assert schema["table_name"] == "dws_order_summary_di"
    cols = {c["name"] for c in schema["columns"]}
    assert {"gmv_amount", "order_cnt", "dt"} <= cols


def test_schema_denied_for_unauthorized_table():
    # data_analyst 无支付域权限 → 表不存在（源头阻断，不泄露存在性）
    with pytest.raises(TableNotFoundError):
        query_table_schema("data_analyst", "dws_payment_summary_di")


def test_schema_denied_returns_not_found_for_unknown_table():
    with pytest.raises(TableNotFoundError):
        query_table_schema("admin", "no_such_table")


def test_finance_can_access_payment():
    schema = query_table_schema("finance_analyst", "dws_payment_summary_di")
    assert schema["table_name"] == "dws_payment_summary_di"


def test_lineage_returns_upstream_and_downstream():
    result = query_lineage("data_analyst", "dws_order_summary_di")
    upstream = {e["source_table"] for e in result if e["direction"] == "upstream"}
    downstream = {e["target_table"] for e in result if e["direction"] == "downstream"}
    assert "dwd_order_detail_di" in upstream
    assert "ads_order_daily_report_di" in downstream


def test_metric_definition():
    m = query_metric_definition("GMV")
    assert m["metric_name"] == "GMV"
    assert "支付成功" in m["definition"]
    assert m["formula"]
