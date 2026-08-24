"""RBAC 权限过滤测试。"""
from dataagent.permissions import ROLES, filter_tables_by_role, resolve_role
from dataagent.warehouse.schema import TABLES


def test_analyst_sees_three_domains():
    visible = filter_tables_by_role("data_analyst", TABLES)
    domains = {t.domain for t in visible.values()}
    assert domains == {"订单域", "用户域", "商品域"}
    # 支付域、物流域完全不可见（源头阻断）
    assert not any(t.domain in ("支付域", "物流域") for t in visible.values())


def test_finance_sees_payment():
    visible = filter_tables_by_role("finance_analyst", TABLES)
    domains = {t.domain for t in visible.values()}
    assert domains == {"订单域", "支付域"}


def test_admin_sees_all():
    visible = filter_tables_by_role("admin", TABLES)
    assert len(visible) == len(TABLES)


def test_unknown_role_falls_back_to_analyst():
    visible = filter_tables_by_role("nonexistent_role", TABLES)
    assert len(visible) == len(filter_tables_by_role("data_analyst", TABLES))


def test_resolve_role_default():
    assert resolve_role("any_user") == "data_analyst"
    assert resolve_role("finance_wang") == "finance_analyst"
    assert resolve_role("admin_li") == "admin"
