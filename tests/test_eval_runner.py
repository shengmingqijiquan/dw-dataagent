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


def test_judge_sql_keyword_not_matched_as_substring():
    # M5(a): pay_order_cnt 不得满足预期关键字 order_cnt（词边界，T15-③ 回归）
    sql = "SELECT pay_order_cnt FROM dws_order_summary_di WHERE dt >= '2026-07-01'"
    expected = {"tables": ["dws_order_summary_di"], "keywords": ["order_cnt"]}
    result = judge_sql(sql, expected)
    assert not result["passed"]


def test_judge_sql_keyword_matches_across_newlines():
    # M5(b): CASE\nWHEN 换行书写仍满足关键字 CASE WHEN（空白归一）
    sql = ("SELECT CASE\nWHEN cnt > 10 THEN 'high' ELSE 'low' END "
           "FROM dws_order_summary_di WHERE dt >= '2026-07-01'")
    expected = {"tables": ["dws_order_summary_di"], "keywords": ["CASE WHEN"]}
    result = judge_sql(sql, expected)
    assert result["passed"], result


def test_judge_sql_table_matches_different_case():
    # M5(c): 表名大小写不同仍匹配（大小写不敏感）
    sql = "SELECT * FROM DWS_CATEGORY_ORDER_DI"
    expected = {"tables": ["dws_category_order_di"], "keywords": []}
    result = judge_sql(sql, expected)
    assert result["passed"], result
