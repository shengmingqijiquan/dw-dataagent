"""SQL 规则校验器：确定性护栏第一层（不依赖 LLM）。

生产对标：规则引擎 100% 拦截非法 SQL；LLM Critic 只做语义审查——
确定性检查与语义检查分层，是护栏体系的核心设计。
"""
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from dataagent.permissions import filter_tables_by_role
from dataagent.warehouse.schema import TABLES


@dataclass
class ValidationResult:
    passed: bool
    errors: list[str] = field(default_factory=list)


def validate_sql(sql: str, role: str) -> ValidationResult:
    errors: list[str] = []

    # 1. 语法解析
    try:
        parsed = sqlglot.parse_one(sql)
    except Exception:
        return ValidationResult(False, ["语法错误: 无法解析 SQL"])

    # 2. 只读检查
    if not isinstance(parsed, (exp.Select, exp.Union)):
        errors.append("只读限制: 仅允许 SELECT 查询")

    # 3. 表存在性 + 权限
    # CTE 别名不是真实表——收集 exp.CTE 别名并排除，避免 WITH 查询被判「表不存在」；
    # 分区检查（第 4 步）仍按表节点遍历，CTE 内层 DWD 表照常受作用域检查，不弱化
    cte_names = {c.alias.lower() for c in parsed.find_all(exp.CTE)}
    tables = {t.name for t in parsed.find_all(exp.Table)
              if t.name.lower() not in cte_names}
    visible = filter_tables_by_role(role, TABLES)
    for t in tables:
        if t not in TABLES:
            errors.append(f"表不存在: {t}")
        elif t not in visible:
            errors.append(f"无权限: {t}")

    # 4. 分区检查：DWD 明细大表必须带分区过滤
    # 按表节点判定：取每个 DWD 表直接所在的最内层 SELECT 作用域（find_ancestor），
    # 精确列名匹配（避免 dt_modified 等子串逃逸）；UNION 各分支/子查询各查各的 WHERE
    reported: set[str] = set()
    for node in parsed.find_all(exp.Table):
        t = node.name
        if t not in visible:
            continue
        spec = TABLES[t]
        if spec.layer != "DWD":
            continue
        sel = node.find_ancestor(exp.Select)
        where = sel.args.get("where") if sel is not None else None
        cols = {c.name.lower() for c in where.find_all(exp.Column)} if where is not None else set()
        if spec.partition_col.lower() not in cols and t not in reported:
            reported.add(t)
            errors.append(
                f"分区检查: {t} 为 DWD 明细表，WHERE 必须包含分区字段 {spec.partition_col}")

    return ValidationResult(len(errors) == 0, errors)
