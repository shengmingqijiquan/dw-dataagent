"""案例库加载测试。"""
from dataagent.rag.cases import load_cases


def test_loads_50_cases():
    cases = load_cases()
    assert len(cases) == 50


def test_case_fields_complete():
    for c in load_cases():
        assert c.id and c.question and c.sql
        assert c.domain in {"订单域", "用户域", "商品域", "支付域", "物流域"}
        assert c.tables, c.id


def test_case_sql_references_registered_tables():
    from dataagent.warehouse.schema import TABLES
    for c in load_cases():
        for t in c.tables:
            assert t in TABLES, f"{c.id}: {t} 不在注册表"
