"""RBAC 表级权限：角色 → 主题域白名单。

生产对标：对应大厂数据权限中心（如 DataWorks 的 RBAC 模型）。
核心原则：无权限的表在元数据检索层完全不可见——从源头消除越权取数可能。
"""
from nl2insight.warehouse.schema import TableSpec

ROLES: dict[str, list[str]] = {
    "data_analyst": ["订单域", "用户域", "商品域"],
    "finance_analyst": ["订单域", "支付域"],
    "ops_analyst": ["物流域"],
    "admin": ["*"],
}

# 用户 → 角色映射（演示用；生产对应 LDAP/权限中心）
_USER_ROLE_MAP: dict[str, str] = {
    "finance_wang": "finance_analyst",
    "ops_zhang": "ops_analyst",
    "admin_li": "admin",
}

DEFAULT_ROLE = "data_analyst"


def resolve_role(user: str) -> str:
    # 角色名直通——MCP x-user 头已传角色名（data_analyst 等）时不做用户映射，
    # 避免把角色名当成新用户名而回落默认角色
    if user in ROLES:
        return user
    return _USER_ROLE_MAP.get(user, DEFAULT_ROLE)


def filter_tables_by_role(role: str, tables: dict[str, TableSpec]) -> dict[str, TableSpec]:
    allowed_domains = ROLES.get(role, ROLES[DEFAULT_ROLE])
    if allowed_domains == ["*"]:
        return dict(tables)
    return {
        name: spec
        for name, spec in tables.items()
        if spec.domain in allowed_domains
    }
