"""评测判定函数测试。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.eval_runner import judge_sql, check_execution


def test_judge_sql_passes_when_all_present():
    sql = "SELECT category_id, SUM(gmv_amount) AS gmv FROM dws_category_order_di WHERE dt >= '2026-07-01' GROUP BY category_id"
    expected = {"tables": ["dws_category_order_di"], "keywords": ["SUM", "gmv_amount", "GROUP BY"]}
    result = judge_sql(sql, expected)
    assert result["passed"], result


def test_judge_sql_fails_on_missing_table():
    sql = "SELECT * FROM dws_order_summary_di"
    expected = {"tables": ["dws_category_order_di"], "keywords": []}
    result = judge_sql(sql, expected)
    assert not result["passed"]
    assert any("表" in c["desc"] for c in result["checks"] if not c["ok"])


def test_judge_sql_fails_on_missing_keyword():
    sql = "SELECT * FROM dws_category_order_di"
    expected = {"tables": ["dws_category_order_di"], "keywords": ["SUM"]}
    result = judge_sql(sql, expected)
    assert not result["passed"]
