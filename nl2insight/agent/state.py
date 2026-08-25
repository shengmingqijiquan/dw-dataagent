"""LangGraph 状态定义。"""
from typing import TypedDict


class DWState(TypedDict, total=False):
    question: str          # 用户自然语言需求
    role: str              # 用户角色（权限过滤用）
    parsed: dict           # 需求解析结果 {domain, metrics, grain, time_range}
    context: str           # 已收集的上下文（元数据 + 案例）
    sql: str               # 生成的 SQL
    validation_errors: list  # 规则校验错误
    critic_feedback: str   # Critic 审查反馈（空 = 通过）
    result: str            # 执行结果（文本）
    insight: str           # 数据洞察（自然语言解读）
    generate_attempts: int  # SQL 生成重试次数
    execute_attempts: int   # 执行重试次数
