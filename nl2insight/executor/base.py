"""QueryExecutor 抽象：隔离执行引擎差异。

生产对标：开发环境 DuckDB 零成本模拟，生产环境 StarRocks 对齐；
Agent 代码只依赖本接口，引擎切换零改动。
"""
from typing import Protocol


class QueryError(Exception):
    """SQL 执行失败统一异常。message 保留原始引擎错误供 Agent 修正。"""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class QueryExecutor(Protocol):
    def setup(self) -> None:
        """建表/初始化连接。"""
        ...

    def execute(self, sql: str) -> list[tuple]:
        """执行 SELECT 返回行；DML/DDL 按引擎支持执行；失败抛 QueryError。"""
        ...

    def close(self) -> None:
        """释放连接。"""
        ...
