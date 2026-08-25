"""历史取数案例库加载。案例 = 需求 + SQL 配对，RAG 检索素材。"""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    question: str
    sql: str
    tables: list[str]
    metrics: list[str]

    def text(self) -> str:
        """入库/检索用文本：需求描述为核心语义载体。"""
        parts = [f"需求: {self.question}", f"SQL: {self.sql}"]
        if self.metrics:
            parts.append(f"指标: {', '.join(self.metrics)}")
        return "\n".join(parts)


def load_cases(path: str = "data/cases/cases.yaml") -> list[Case]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"案例文件不存在: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return [Case(**item) for item in raw]
