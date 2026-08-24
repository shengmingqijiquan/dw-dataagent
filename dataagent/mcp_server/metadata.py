"""元数据查询核心：MCP 工具的业务逻辑层（协议无关，可单测）。

生产对标：此模块对应元数据中心（DataHub/Atlas/自研）的 API 封装；
MCP Server 只是它的协议暴露层，对接真实元数据中心时此层签名不变。
"""
from dataagent.warehouse.schema import TABLES, LINEAGE, METRICS
from dataagent.permissions import filter_tables_by_role


class TableNotFoundError(Exception):
    """表不存在或无权限（两者不可区分，避免侧信道泄露）。"""


def query_table_list(role: str, domain: str | None = None) -> list[dict]:
    visible = filter_tables_by_role(role, TABLES)
    result = [
        {
            "table_name": t.name, "domain": t.domain, "layer": t.layer,
            "granularity": t.grain, "description": t.description,
            "partition_col": t.partition_col,
        }
        for t in visible.values()
        if domain is None or t.domain == domain
    ]
    return sorted(result, key=lambda r: r["table_name"])


def query_table_schema(role: str, table_name: str) -> dict:
    visible = filter_tables_by_role(role, TABLES)
    spec = visible.get(table_name)
    if spec is None:
        raise TableNotFoundError(table_name)
    return {
        "table_name": spec.name,
        "domain": spec.domain,
        "layer": spec.layer,
        "granularity": spec.grain,
        "description": spec.description,
        "partition_col": spec.partition_col,
        "columns": [
            {"name": c.name, "data_type": c.data_type, "comment": c.comment}
            for c in spec.columns
        ],
    }


def query_lineage(role: str, table_name: str) -> list[dict]:
    visible = filter_tables_by_role(role, TABLES)
    if table_name not in visible:
        raise TableNotFoundError(table_name)
    result = []
    for e in LINEAGE:
        if e.source == table_name:
            result.append({"direction": "downstream",
                           "source_table": e.source,
                           "target_table": e.target,
                           "relation": e.relation})
        if e.target == table_name:
            result.append({"direction": "upstream",
                           "source_table": e.source,
                           "target_table": e.target,
                           "relation": e.relation})
    return result


def query_metric_definition(metric_name: str) -> dict:
    m = METRICS.get(metric_name)
    if m is None:
        return {"metric_name": metric_name,
                "definition": "未收录该指标", "formula": ""}
    return {"metric_name": m.name, "definition": m.definition,
            "formula": m.formula}
