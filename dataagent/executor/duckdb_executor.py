"""DuckDB 执行器（开发兜底，零内存开销）。"""
import duckdb
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TableSpec


class DuckDBExecutor:
    def __init__(self, path: str, tables: dict[str, TableSpec]):
        self.path = path
        self.tables = tables
        self._con = None

    def setup(self) -> None:
        import os
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._con = duckdb.connect(self.path)
        for spec in self.tables.values():
            cols = ", ".join(
                f"{c.name} {c.data_type}" for c in spec.columns)
            self._con.execute(
                f"CREATE TABLE IF NOT EXISTS {spec.name} ({cols})")
        # 禁止读写库外文件（DDL/DML 由 execute() 关键字拦截，见下）
        self._con.execute("PRAGMA enable_external_access=false")

    def execute(self, sql: str) -> list[tuple]:
        stripped = sql.strip().rstrip(";").upper()
        if stripped.startswith(("DROP", "DELETE", "UPDATE", "CREATE", "ALTER", "TRUNCATE", "INSERT")):
            raise QueryError(f"只读限制：不允许执行 {stripped.split()[0]} 语句")
        try:
            return self._con.execute(sql).fetchall()
        except Exception as e:  # duckdb.Error 及一切执行异常
            raise QueryError(str(e)) from e

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
