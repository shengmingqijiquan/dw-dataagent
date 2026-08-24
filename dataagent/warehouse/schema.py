"""数仓 Schema 注册表：30 张表的单一事实源。

生产对标：真实环境中此注册表对应元数据中心（DataHub/Atlas）的 API 返回；
MCP 工具只依赖本模块的查询函数，对接真实元数据中心时接口不变。
"""
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    data_type: str
    comment: str


@dataclass(frozen=True)
class TableSpec:
    name: str
    domain: str            # 主题域
    layer: str             # DWD/DWS/ADS/DIM
    grain: str             # 粒度：日/全量
    description: str
    columns: tuple[ColumnSpec, ...]
    partition_col: str = "dt"


def C(name, dtype, comment=""):
    return ColumnSpec(name, dtype, comment)


def T(name, domain, layer, grain, desc, cols):
    return TableSpec(name, domain, layer, grain, desc, tuple(cols))


def _common_order_cols():
    return [
        C("order_id", "BIGINT", "订单ID"),
        C("user_id", "BIGINT", "用户ID"),
        C("product_id", "BIGINT", "商品ID"),
        C("dt", "DATE", "分区日期"),
    ]


TABLES: dict[str, TableSpec] = {}

def _reg(t: TableSpec):
    TABLES[t.name] = t


# ===== 订单域（6 表） =====
_reg(T("dwd_order_detail_di", "订单域", "DWD", "日", "订单明细日表（含支付状态）", [
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("category_id", "BIGINT", "品类ID"),
    C("platform", "VARCHAR", "平台(iOS/Android/Web)"),
    C("pay_status", "VARCHAR", "支付状态(paid/unpaid/refunded)"),
    C("order_amount", "DECIMAL(20,2)", "订单金额"),
    C("item_count", "INT", "商品件数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_order_created_di", "订单域", "DWD", "日", "订单创建日志日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("order_status", "VARCHAR", "订单状态(created/paid/shipped/signed/cancelled)"),
    C("create_time", "DATETIME", "创建时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_order_summary_di", "订单域", "DWS", "日", "订单汇总日表", [
    C("order_cnt", "BIGINT", "下单数"),
    C("order_user_cnt", "BIGINT", "下单用户数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额（GMV，支付成功口径）"),
    C("pay_order_cnt", "BIGINT", "支付成功订单数"),
    C("pay_rate", "DECIMAL(10,4)", "支付率=pay_order_cnt/order_cnt"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_category_order_di", "订单域", "DWS", "日", "品类订单汇总日表", [
    C("category_id", "BIGINT", "品类ID"),
    C("order_cnt", "BIGINT", "下单数"),
    C("pay_order_cnt", "BIGINT", "支付成功订单数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_platform_order_di", "订单域", "DWS", "日", "平台订单汇总日表", [
    C("platform", "VARCHAR", "平台"),
    C("order_cnt", "BIGINT", "下单数"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_order_daily_report_di", "订单域", "ADS", "日", "订单日报表", [
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("order_cnt", "BIGINT", "下单数"),
    C("avg_order_amount", "DECIMAL(20,2)", "客单价=GMV/支付订单数"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 用户域（6 表） =====
_reg(T("dwd_user_behavior_di", "用户域", "DWD", "日", "用户行为明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("behavior_type", "VARCHAR", "行为类型(view/cart/buy)"),
    C("product_id", "BIGINT", "商品ID"),
    C("behavior_time", "DATETIME", "行为时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_user_register_di", "用户域", "DWD", "日", "用户注册明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("register_channel", "VARCHAR", "注册渠道(organic/ad/wechat/appstore)"),
    C("register_time", "DATETIME", "注册时间"),
    C("city", "VARCHAR", "城市"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_active_di", "用户域", "DWS", "日", "用户活跃汇总日表", [
    C("active_user_cnt", "BIGINT", "活跃用户数（当日有行为）"),
    C("new_user_cnt", "BIGINT", "新增用户数"),
    C("dau", "BIGINT", "日活跃用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_behavior_summary_di", "用户域", "DWS", "日", "用户行为汇总日表", [
    C("behavior_type", "VARCHAR", "行为类型"),
    C("user_cnt", "BIGINT", "行为用户数"),
    C("behavior_cnt", "BIGINT", "行为次数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_user_retention_di", "用户域", "DWS", "日", "用户留存日表", [
    C("register_dt", "DATE", "注册日期"),
    C("retain_d1", "DECIMAL(10,4)", "次日留存率"),
    C("retain_d7", "DECIMAL(10,4)", "7日留存率"),
    C("retain_d30", "DECIMAL(10,4)", "30日留存率"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_user_growth_report_di", "用户域", "ADS", "日", "用户增长日报表", [
    C("new_user_cnt", "BIGINT", "新增用户数"),
    C("active_user_cnt", "BIGINT", "活跃用户数"),
    C("retention_rate", "DECIMAL(10,4)", "留存率（次日）"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 商品域（6 表） =====
_reg(T("dim_product_info", "商品域", "DIM", "全量", "商品维度表", [
    C("product_id", "BIGINT", "商品ID"),
    C("product_name", "VARCHAR", "商品名称"),
    C("category_id", "BIGINT", "品类ID"),
    C("brand", "VARCHAR", "品牌"),
    C("price", "DECIMAL(20,2)", "标价"),
    C("status", "VARCHAR", "状态(on/off)"),
]))
_reg(T("dim_category_info", "商品域", "DIM", "全量", "品类维度表", [
    C("category_id", "BIGINT", "品类ID"),
    C("category_name", "VARCHAR", "品类名称"),
    C("parent_category_id", "BIGINT", "父品类ID"),
]))
_reg(T("dwd_product_view_di", "商品域", "DWD", "日", "商品浏览明细日表", [
    C("user_id", "BIGINT", "用户ID"),
    C("product_id", "BIGINT", "商品ID"),
    C("view_time", "DATETIME", "浏览时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_product_gmv_di", "商品域", "DWS", "日", "商品GMV汇总日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("gmv_amount", "DECIMAL(20,2)", "成交总额"),
    C("order_cnt", "BIGINT", "支付订单数"),
    C("pay_user_cnt", "BIGINT", "支付用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_product_view_di", "商品域", "DWS", "日", "商品浏览汇总日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("view_cnt", "BIGINT", "浏览量"),
    C("view_user_cnt", "BIGINT", "浏览用户数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_product_ranking_di", "商品域", "ADS", "日", "商品排行榜日表", [
    C("product_id", "BIGINT", "商品ID"),
    C("gmv_rank", "INT", "GMV排名"),
    C("view_rank", "INT", "浏览量排名"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 支付域（6 表） =====
_reg(T("dwd_payment_detail_di", "支付域", "DWD", "日", "支付明细日表", [
    C("payment_id", "BIGINT", "支付ID"),
    C("order_id", "BIGINT", "订单ID"),
    C("user_id", "BIGINT", "用户ID"),
    C("pay_channel", "VARCHAR", "支付渠道(alipay/wechat/card)"),
    C("pay_amount", "DECIMAL(20,2)", "支付金额"),
    C("pay_time", "DATETIME", "支付时间"),
    C("pay_status", "VARCHAR", "支付状态(success/failed)"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_payment_summary_di", "支付域", "DWS", "日", "支付汇总日表", [
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("pay_order_cnt", "BIGINT", "支付订单数"),
    C("pay_user_cnt", "BIGINT", "支付用户数"),
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_payment_channel_di", "支付域", "DWS", "日", "支付渠道汇总日表", [
    C("pay_channel", "VARCHAR", "支付渠道"),
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("pay_order_cnt", "BIGINT", "支付订单数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_refund_detail_di", "支付域", "DWD", "日", "退款明细日表", [
    C("refund_id", "BIGINT", "退款ID"),
    C("order_id", "BIGINT", "订单ID"),
    C("refund_amount", "DECIMAL(20,2)", "退款金额"),
    C("refund_time", "DATETIME", "退款时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_refund_summary_di", "支付域", "DWS", "日", "退款汇总日表", [
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("refund_order_cnt", "BIGINT", "退款订单数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("ads_payment_daily_report_di", "支付域", "ADS", "日", "支付日报表", [
    C("pay_amount", "DECIMAL(20,2)", "支付总额"),
    C("refund_amount", "DECIMAL(20,2)", "退款总额"),
    C("net_amount", "DECIMAL(20,2)", "净额=支付-退款"),
    C("dt", "DATE", "分区日期"),
]))

# ===== 物流域（6 表） =====
_reg(T("dwd_logistics_tracking_di", "物流域", "DWD", "日", "物流轨迹日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("logistics_company", "VARCHAR", "物流公司(sf/jd/zt)"),
    C("status", "VARCHAR", "状态(shipped/transit/signed)"),
    C("ship_time", "DATETIME", "发货时间"),
    C("sign_time", "DATETIME", "签收时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dwd_logistics_shipped_di", "物流域", "DWD", "日", "发货明细日表", [
    C("order_id", "BIGINT", "订单ID"),
    C("warehouse_id", "BIGINT", "仓库ID"),
    C("ship_time", "DATETIME", "发货时间"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_logistics_summary_di", "物流域", "DWS", "日", "物流汇总日表", [
    C("ship_order_cnt", "BIGINT", "发货订单数"),
    C("sign_order_cnt", "BIGINT", "签收订单数"),
    C("avg_delivery_days", "DECIMAL(10,2)", "平均配送时长（天）"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dws_logistics_company_di", "物流域", "DWS", "日", "物流公司汇总日表", [
    C("logistics_company", "VARCHAR", "物流公司"),
    C("ship_cnt", "BIGINT", "发货数"),
    C("sign_cnt", "BIGINT", "签收数"),
    C("dt", "DATE", "分区日期"),
]))
_reg(T("dim_warehouse_info", "物流域", "DIM", "全量", "仓库维度表", [
    C("warehouse_id", "BIGINT", "仓库ID"),
    C("warehouse_name", "VARCHAR", "仓库名称"),
    C("city", "VARCHAR", "城市"),
]))
_reg(T("ads_logistics_daily_report_di", "物流域", "ADS", "日", "物流日报表", [
    C("ship_cnt", "BIGINT", "发货数"),
    C("sign_cnt", "BIGINT", "签收数"),
    C("on_time_rate", "DECIMAL(10,4)", "准时率"),
    C("dt", "DATE", "分区日期"),
]))


@dataclass(frozen=True)
class LineageEdge:
    source: str
    target: str
    relation: str


LINEAGE: list[LineageEdge] = [
    LineageEdge("dwd_order_detail_di", "dws_order_summary_di", "ETL_AGG"),
    LineageEdge("dws_order_summary_di", "ads_order_daily_report_di", "ETL_JOIN"),
    LineageEdge("dwd_order_detail_di", "dws_category_order_di", "ETL_AGG"),
    LineageEdge("dwd_order_detail_di", "dws_platform_order_di", "ETL_AGG"),
    LineageEdge("dwd_user_behavior_di", "dws_user_active_di", "ETL_AGG"),
    LineageEdge("dwd_user_register_di", "dws_user_active_di", "ETL_AGG"),
    LineageEdge("dwd_user_behavior_di", "dws_user_behavior_summary_di", "ETL_AGG"),
    LineageEdge("dwd_user_register_di", "dws_user_retention_di", "ETL_JOIN"),
    LineageEdge("dws_user_active_di", "ads_user_growth_report_di", "ETL_JOIN"),
    LineageEdge("dwd_order_detail_di", "dws_product_gmv_di", "ETL_AGG"),
    LineageEdge("dwd_product_view_di", "dws_product_view_di", "ETL_AGG"),
    LineageEdge("dws_product_gmv_di", "ads_product_ranking_di", "ETL_JOIN"),
    LineageEdge("dws_product_view_di", "ads_product_ranking_di", "ETL_JOIN"),
    LineageEdge("dwd_payment_detail_di", "dws_payment_summary_di", "ETL_AGG"),
    LineageEdge("dwd_payment_detail_di", "dws_payment_channel_di", "ETL_AGG"),
    LineageEdge("dwd_refund_detail_di", "dws_refund_summary_di", "ETL_AGG"),
    LineageEdge("dws_payment_summary_di", "ads_payment_daily_report_di", "ETL_JOIN"),
    LineageEdge("dws_refund_summary_di", "ads_payment_daily_report_di", "ETL_JOIN"),
    LineageEdge("dwd_logistics_tracking_di", "dws_logistics_summary_di", "ETL_AGG"),
    LineageEdge("dwd_logistics_tracking_di", "dws_logistics_company_di", "ETL_AGG"),
    LineageEdge("dwd_logistics_shipped_di", "dws_logistics_summary_di", "ETL_AGG"),
    LineageEdge("dws_logistics_summary_di", "ads_logistics_daily_report_di", "ETL_JOIN"),
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    definition: str
    formula: str


METRICS: dict[str, MetricSpec] = {
    "GMV": MetricSpec(
        name="GMV",
        definition="支付成功订单的成交总额",
        formula="SUM(gmv_amount) WHERE pay_status='paid'",
    ),
    "支付率": MetricSpec(
        name="支付率",
        definition="支付成功订单数占下单数的比例",
        formula="pay_order_cnt / order_cnt",
    ),
    "客单价": MetricSpec(
        name="客单价",
        definition="平均每个支付订单的成交金额",
        formula="GMV / pay_order_cnt",
    ),
    "DAU": MetricSpec(
        name="DAU",
        definition="当日有任意行为的独立用户数",
        formula="COUNT(DISTINCT user_id) FROM dwd_user_behavior_di",
    ),
    "新增用户": MetricSpec(
        name="新增用户",
        definition="当日注册的独立用户数",
        formula="COUNT(DISTINCT user_id) FROM dwd_user_register_di",
    ),
    "次日留存率": MetricSpec(
        name="次日留存率",
        definition="注册次日仍活跃用户占注册用户比例",
        formula="retain_d1 FROM dws_user_retention_di",
    ),
    "退款率": MetricSpec(
        name="退款率",
        definition="退款订单数占支付订单数的比例",
        formula="refund_order_cnt / pay_order_cnt",
    ),
    "履约准时率": MetricSpec(
        name="履约准时率",
        definition="准时签收订单占发货订单的比例",
        formula="on_time_rate FROM ads_logistics_daily_report_di",
    ),
}


def generate_metadata_yaml(out_dir: str | Path) -> None:
    """从注册表生成元数据 YAML 文件（tables/columns/lineage/metrics）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = [
        {
            "table_name": t.name, "domain": t.domain, "layer": t.layer,
            "granularity": t.grain, "description": t.description,
            "partition_col": t.partition_col,
        }
        for t in TABLES.values()
    ]
    (out / "tables.yaml").write_text(
        yaml.safe_dump(tables, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    columns = [
        {"table_name": t.name, "column_name": c.name,
         "data_type": c.data_type, "comment": c.comment}
        for t in TABLES.values() for c in t.columns
    ]
    (out / "columns.yaml").write_text(
        yaml.safe_dump(columns, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    lineage = [
        {"source_table": e.source, "target_table": e.target,
         "relation": e.relation}
        for e in LINEAGE
    ]
    (out / "lineage.yaml").write_text(
        yaml.safe_dump(lineage, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    metrics = [
        {"metric_name": m.name, "definition": m.definition, "formula": m.formula}
        for m in METRICS.values()
    ]
    (out / "metrics.yaml").write_text(
        yaml.safe_dump(metrics, allow_unicode=True, sort_keys=False),
        encoding="utf-8")
