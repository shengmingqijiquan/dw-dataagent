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
        return ValidationResult(False, [f"语法错误: 无法解析 SQL"])

    # 2. 只读检查
    if not isinstance(parsed, (exp.Select, exp.Union)):
        errors.append(f"只读限制: 仅允许 SELECT 查询")

    # 3. 表存在性 + 权限
    tables = {t.name for t in parsed.find_all(exp.Table)}
    visible = filter_tables_by_role(role, TABLES)
    for t in tables:
        if t not in TABLES:
            errors.append(f"表不存在: {t}")
        elif t not in visible:
            errors.append(f"无权限: {t}")

    # 4. 分区检查：DWD 明细大表必须带分区过滤
    for t in tables:
        if t not in visible:
            continue
        spec = TABLES[t]
        if spec.layer == "DWD":
            where = parsed.find(exp.Where)
            has_partition = (
                where is not None
                and spec.partition_col in
                "".join(c.sql() for c in where.find_all(exp.Column))
            )
            if not has_partition:
                errors.append(
                    f"分区检查: {t} 为 DWD 明细表，WHERE 必须包含分区字段 {spec.partition_col}")

    return ValidationResult(len(errors) == 0, errors)
