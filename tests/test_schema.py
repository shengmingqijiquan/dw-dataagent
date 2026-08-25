"""Schema 注册表测试。"""
from nl2insight.warehouse.schema import TABLES, LINEAGE, METRICS, TableSpec


def test_exactly_30_tables():
    assert len(TABLES) == 30


def test_all_domains_covered():
    domains = {t.domain for t in TABLES.values()}
    assert domains == {"订单域", "用户域", "商品域", "支付域", "物流域"}


def test_every_table_has_partition_col():
    for t in TABLES.values():
        if t.layer == "DIM":
            continue  # 全量维度表无分区
        assert t.partition_col in {c.name for c in t.columns}, t.name


def test_layer_naming_convention():
    for name, t in TABLES.items():
        prefix = name.split("_")[0]
        assert prefix in {"dwd", "dws", "ads", "dim"}, name


def test_lineage_references_existing_tables():
    for edge in LINEAGE:
        assert edge.source in TABLES, edge.source
        assert edge.target in TABLES, edge.target


def test_metrics_have_formula():
    for m in METRICS.values():
        assert m.definition and m.formula
