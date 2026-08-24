"""Golden Set 评测执行器。

指标：
- 执行成功率 = 通过校验且 DuckDB 执行成功的比例
- 要素准确率 = SQL 包含全部预期表与关键字且执行成功的比例（主指标）
- 失败原因分类：校验失败 / 执行失败 / 要素缺失
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import yaml

from dataagent.agent.graph import run_agent
from dataagent.executor.duckdb_executor import DuckDBExecutor
from dataagent.guardrails.sql_validator import validate_sql
from dataagent.warehouse.schema import TABLES


def load_golden_set(path: str = "evals/golden_set.yaml") -> list[dict]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def judge_sql(sql: str, expected: dict) -> dict:
    checks = []
    for t in expected.get("tables", []):
        checks.append({"check": f"表 {t}", "ok": t in sql,
                       "desc": f"SQL 使用了表 {t}"})
    for kw in expected.get("keywords", []):
        checks.append({"check": f"关键字 {kw}", "ok": kw.lower() in sql.lower(),
                       "desc": f"SQL 包含 {kw}"})
    passed = all(c["ok"] for c in checks)
    return {"passed": passed, "checks": checks}


def check_execution(sql: str, role: str) -> tuple[bool, str]:
    """校验 + DuckDB 真实执行。"""
    validation = validate_sql(sql, role)
    if not validation.passed:
        return False, "校验失败: " + "; ".join(validation.errors)
    executor = DuckDBExecutor("data/warehouse.duckdb", TABLES)
    executor.setup()
    try:
        executor.execute(sql)
        return True, ""
    except Exception as e:
        return False, f"执行失败: {e}"
    finally:
        executor.close()


def run_all(golden_set_path: str = "evals/golden_set.yaml",
            report_path: str = "evals/report.yaml") -> dict:
    cases = load_golden_set(golden_set_path)
    results = []
    for case in cases:
        t0 = time.time()
        try:
            agent_result = run_agent(case["question"], case["role"])
            sql = agent_result["sql"]
            exec_ok, exec_msg = check_execution(sql, case["role"])
            judge = judge_sql(sql, case["expected"])
            passed = exec_ok and judge["passed"]
            failure = "" if passed else (
                exec_msg or "要素缺失: " + "; ".join(
                    c["check"] for c in judge["checks"] if not c["ok"]))
        except Exception as e:
            sql, exec_ok, judge, passed, failure = "", False, {"passed": False, "checks": []}, False, f"Agent 异常: {e}"
        results.append({
            "id": case["id"], "question": case["question"],
            "difficulty": case["difficulty"], "passed": passed,
            "sql": sql, "failure": failure,
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[{case['id']}] {'PASS' if passed else 'FAIL'} "
              f"({case['difficulty']}) {failure[:80]}")

    total = len(results)
    passed_n = sum(1 for r in results if r["passed"])
    accuracy = passed_n / total if total else 0
    exec_ok_n = sum(1 for r in results if r["passed"] or "要素缺失" in r["failure"])

    by_difficulty = {}
    for diff in ("simple", "join", "metric", "complex"):
        sub = [r for r in results if r["difficulty"] == diff]
        by_difficulty[diff] = (sum(1 for r in sub if r["passed"]) / len(sub)) if sub else None

    failures = [r for r in results if not r["passed"]]
    report = {
        "total": total, "passed": passed_n, "accuracy": round(accuracy, 4),
        "exec_success_rate": round(exec_ok_n / total, 4) if total else 0,
        "by_difficulty": by_difficulty,
        "failures": [
            {"id": r["id"], "question": r["question"], "failure": r["failure"],
             "sql": r["sql"]}
            for r in failures
        ],
    }
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(
        yaml.safe_dump(report, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
    print(f"\n准确率: {accuracy:.1%} ({passed_n}/{total})")
    print(f"报告: {report_path}")
    return report


if __name__ == "__main__":
    run_all()
