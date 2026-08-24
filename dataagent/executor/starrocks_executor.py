"""StarRocks 执行器（生产对齐；按需启动容器）。

连接方式：PyMySQL（MySQL 协议）直连 StarRocks FE（默认 9030 端口）。
StarRocks FE 兼容 MySQL 协议，PyMySQL 是生产环境标准客户端之一
（安装的 `starrocks` pip 包为 SQLAlchemy dialect，无 `starrocks.connect`
快捷函数，故用 PyMySQL 承载同一 MySQL 协议连接）。

只读设计（对齐生产叙事）：生产环境对 Agent 的只读控制以**引擎侧权限为
权威**——StarRocks 侧创建仅授 SELECT 的只读账户/角色，Agent 以此身份连接，
DML/DDL 在引擎层即被拒绝；代码侧的关键字前缀拦截只是纵深防御
（best-effort，可被 `-- 注释` 前缀或 CTE 绕过），不构成安全边界，
真正的 SQL 语义防线在上游 SQLGlot 规则校验。
"""
import pymysql
from dataagent.executor.base import QueryError
from dataagent.warehouse.schema import TableSpec


class StarRocksExecutor:
    """与 DuckDBExecutor 同接口：setup/execute/close，Agent 层无感知切换。"""

    def __init__(self, host: str, port: int, user: str, password: str,
                 tables: dict[str, TableSpec], database: str = "demo"):
        self.host, self.port = host, port
        self.user, self.password = user, password
        self.tables = tables
        self.database = database
        self._con = None

    def setup(self) -> None:
        """建库建表（30 张：27 张含 dt 日分区列 + 3 张 DIM 全量快照无 dt）。"""
        self._con = pymysql.connect(
            host=self.host, port=self.port,
            user=self.user, password=self.password,
            autocommit=True)   # 关键：PyMySQL 默认 autocommit=False，
                               # 指向真实 MySQL 协议服务器时 close() 会静默回滚全部写入
        cur = self._con.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
        cur.execute(f"USE {self.database}")
        for spec in self.tables.values():
            cols = ", ".join(
                f"{c.name} {c.data_type}" for c in spec.columns)
            cur.execute(
                f"CREATE TABLE IF NOT EXISTS {spec.name} ({cols}) "
                f"ENGINE=OLAP DUPLICATE KEY({spec.columns[0].name}) "
                f"DISTRIBUTED BY HASH({spec.columns[0].name}) BUCKETS 1")
        self._con.commit()
        cur.close()

    def execute(self, sql: str) -> list[tuple]:
        # 纵深防御（非权威防线）：关键字前缀拦截，可被 `-- 注释`/CTE/分号多语句
        # （SELECT 1; DROP …）绕过，此处一并快速失败；只读的权威控制是引擎侧账户
        # 授权（仅授 SELECT），上游另有 SQLGlot 规则校验。
        stripped = sql.strip().rstrip(";").upper()
        if ";" in stripped:
            raise QueryError("只读限制：不允许执行多语句 SQL（分号分隔）")
        if stripped.startswith(("DROP", "DELETE", "UPDATE", "CREATE",
                                "ALTER", "TRUNCATE", "INSERT")):
            raise QueryError(f"只读限制：不允许执行 {stripped.split()[0]} 语句")
        try:
            cur = self._con.cursor()
            cur.execute(sql)
            rows = cur.fetchall()
            cur.close()
            return rows
        except Exception as e:
            raise QueryError(str(e)) from e

    def close(self) -> None:
        if self._con is not None:
            self._con.close()
            self._con = None
