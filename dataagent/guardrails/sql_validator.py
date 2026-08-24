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
    tables = {t.name for t in parsed.find_all(exp.Table)}
    visible = filter_tables_by_role(role, TABLES)
    for t in tables:
        if t not in TABLES:
            errors.append(f"表不存在: {t}")
        elif t not in visible:
            errors.append(f"无权限: {t}")

    # 4. 分区检查：DWD 明细大表必须带分区过滤（每个 SELECT 作用域都要有）
    # 精确列名匹配（避免 dt_modified 等子串逃逸）；find_all 覆盖 UNION 各分支/子查询
    for t in tables:
        if t not in visible:
            continue
        spec = TABLES[t]
        if spec.layer != "DWD":
            continue
        for sel in parsed.find_all(exp.Select):
            if t not in {x.name for x in sel.find_all(exp.Table)}:
                continue
            where = sel.args.get("where")
            cols = set()
            if where is not None:
                cols = {c.name.lower() for c in where.find_all(exp.Column)}
            if spec.partition_col.lower() not in cols:
                errors.append(
                    f"分区检查: {t} 为 DWD 明细表，WHERE 必须包含分区字段 {spec.partition_col}")
                break

    return ValidationResult(len(errors) == 0, errors)
