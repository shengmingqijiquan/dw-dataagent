"""模拟数仓初始化：建表 + 灌数 + 元数据 YAML。

用法:
  python scripts/init_warehouse.py --engine duckdb   # 默认，开发兜底
  python scripts/init_warehouse.py --engine starrocks  # 需容器已启动（按需）

确定性：主随机流 rng = random.Random(42)，90 天数据每次重跑产物一致；
补齐的 DWD 明细使用独立随机流 detail_rng（seed 4242），不扰动主 rng，
因此 brief 规定的口径数字（订单量/支付量/GMV 等）与种子 42 完全一致。

口径自洽：DWS/ADS 汇总由当日明细聚合得出（商品域 GMV/浏览榜亦由明细聚合）。
"""
import argparse
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nl2insight.config import load_config
from nl2insight.executor.duckdb_executor import DuckDBExecutor
from nl2insight.executor.starrocks_executor import StarRocksExecutor
from nl2insight.warehouse.schema import TABLES, generate_metadata_yaml

DAYS = 90
END_DATE = date(2026, 7, 31)
rng = random.Random(42)
# 辅助 DWD 明细的独立随机流：确定性由相同种子保证，且不改变主 rng 的抽取序列
# （主 rng 只服务 brief 规定的口径数字，保证与计划文档逐位一致）。
detail_rng = random.Random(4242)

CATEGORIES = [(1, "服饰"), (2, "数码"), (3, "食品"), (4, "家居"), (5, "美妆")]
PLATFORMS = ["iOS", "Android", "Web"]
CHANNELS = ["alipay", "wechat", "card"]
LOGISTICS = ["sf", "jd", "zt"]
WAREHOUSES = [(1, "华东仓", "上海"), (2, "华南仓", "广州"), (3, "华北仓", "北京")]
USERS, PRODUCTS = 50_000, 5_000
REGISTER_CHANNELS = ["organic", "ad", "wechat", "appstore"]
CITIES = ["上海", "北京", "广州", "深圳", "杭州"]


def dates():
    for i in range(DAYS):
        yield END_DATE - timedelta(days=DAYS - 1 - i)


def build_dim(executor):
    """维度表灌数（全量快照，无 dt 列：INSERT 值数 = 真实列数）。"""
    for cid, cname in CATEGORIES:
        _insert(executor, "dim_category_info",
                [(cid, cname, 0)])
    for pid in range(1, PRODUCTS + 1):
        cid = CATEGORIES[pid % 5][0]
        _insert(executor, "dim_product_info",
                [(pid, f"商品{pid}", cid,
                  f"品牌{pid % 20}", round(rng.uniform(9.9, 999), 2),
                  "on")])
    for wid, wname, city in WAREHOUSES:
        _insert(executor, "dim_warehouse_info", [(wid, wname, city)])


def _fmt(v):
    """值 → SQL 字面量。数据全部由本脚本确定性生成（无外部输入），内联安全。"""
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return f"'{v}'"
    return str(v)


def _insert(executor, table: str, rows: list[tuple]):
    """通过引擎内部连接灌数（初始化专用，绕过只读护栏）。

    INSERT 值数 = len(TABLES[table].columns)：DIM 全量快照表无 dt 列，
    其余表含真实 dt 列——统一以注册表为准，杜绝列数不匹配。
    duckdb 的 executemany 为 Python 层逐行绑定（约 20s/6.5 万行），
    改为分批内联多 VALUES 语句（2000 行/批），提速约 11 倍。
    """
    if not rows:
        return
    con = executor._con
    if isinstance(executor, DuckDBExecutor):
        batch = 2000
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            sql = ",".join(
                "(" + ",".join(_fmt(v) for v in row) + ")"
                for row in chunk)
            con.execute(f"INSERT INTO {table} VALUES {sql}")
    else:
        cur = con.cursor()
        for row in rows:
            vals = ",".join(_fmt(v) for v in row)
            cur.execute(f"INSERT INTO {table} VALUES ({vals})")
        cur.close()


def _time(d: date, r: random.Random) -> str:
    """由日期与随机源生成 DATETIME 字符串。"""
    return (f"{d.isoformat()} "
            f"{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:"
            f"{r.randint(0, 59):02d}")


def build_fact_daily(executor):
    """日增量事实/DWS/ADS 表灌数。"""
    for d in dates():
        dt = d.isoformat()
        active_users = rng.randint(20_000, 30_000)
        orders = rng.randint(50_000, 80_000)
        pay_orders = int(orders * rng.uniform(0.55, 0.7))

        # dwd_order_detail_di：按日抽样订单明细
        order_rows, pay_rows, refund_rows = [], [], []
        for i in range(orders):
            uid = rng.randint(1, USERS)
            pid = rng.randint(1, PRODUCTS)
            cid = CATEGORIES[pid % 5][0]
            plat = rng.choice(PLATFORMS)
            status = rng.choices(
                ["paid", "unpaid", "refunded"], weights=[55, 35, 10])[0]
            amount = round(rng.uniform(20, 500), 2)
            oid = i + 1
            order_rows.append((oid, uid, pid, cid, plat, status,
                               amount, rng.randint(1, 3), dt))
            if status in ("paid", "refunded"):
                ch = rng.choice(CHANNELS)
                pay_rows.append((oid, oid, uid, ch, amount, dt, "success", dt))
            if status == "refunded":
                # 退款金额按 2 位小数落库：保证明细 DECIMAL 列与 DWS 聚合口径一致
                refund_rows.append((oid, oid, round(amount * 0.6, 2), dt, dt))

        # DWD 明细落库：订单/支付/退款明细（DWS/ADS 由其聚合，口径自洽）
        _insert(executor, "dwd_order_detail_di", order_rows)
        _insert(executor, "dwd_payment_detail_di", pay_rows)
        _insert(executor, "dwd_refund_detail_di", refund_rows)

        # DWS 汇总（由明细聚合口径算出，保证口径自洽）
        paid_amount = sum(r[6] for r in order_rows
                          if r[5] in ("paid", "refunded"))
        pay_cnt = len(pay_rows)
        _insert(executor, "dws_order_summary_di",
                [(orders, len({r[1] for r in order_rows}), round(paid_amount, 2),
                  pay_cnt, round(pay_cnt / orders, 4), dt)])

        cat_agg = {}
        for r in order_rows:
            cid = r[3]
            cat_agg.setdefault(cid, [0, 0, 0.0])
            cat_agg[cid][0] += 1
            if r[5] in ("paid", "refunded"):
                cat_agg[cid][1] += 1
                cat_agg[cid][2] += r[6]
        for cid, (cnt, pcnt, amt) in cat_agg.items():
            _insert(executor, "dws_category_order_di",
                    [(cid, cnt, pcnt, round(amt, 2), dt)])

        for plat in PLATFORMS:
            plat_rows = [r for r in order_rows if r[4] == plat]
            amt = sum(r[6] for r in plat_rows
                      if r[5] in ("paid", "refunded"))
            _insert(executor, "dws_platform_order_di",
                    [(plat, len(plat_rows), round(amt, 2), dt)])

        _insert(executor, "ads_order_daily_report_di",
                [(round(paid_amount, 2), orders,
                  round(paid_amount / pay_cnt, 2) if pay_cnt else 0, dt)])

        # dwd_order_created_di：下单日志（由订单明细派生，独立随机流）
        created_rows = []
        for r in order_rows:
            oid, uid, pid, status = r[0], r[1], r[2], r[5]
            order_status = ("cancelled" if status != "paid"
                            else detail_rng.choice(["paid", "shipped", "signed"]))
            created_rows.append(
                (oid, uid, pid, order_status, _time(d, detail_rng), dt))
        _insert(executor, "dwd_order_created_di", created_rows)

        # 用户域
        new_users = rng.randint(800, 2000)

        # dwd_user_behavior_di：行为明细（view/cart/buy），先于 DWS 汇总生成，
        # 供用户行为/活跃汇总真实聚合（评审 Fix 1：口径自洽）
        behavior_rows = []
        for _ in range(detail_rng.randint(20_000, 40_000)):
            behavior_rows.append((
                detail_rng.randint(1, USERS),
                detail_rng.choice(["view", "cart", "buy"]),
                detail_rng.randint(1, PRODUCTS),
                _time(d, detail_rng),
                dt))
        _insert(executor, "dwd_user_behavior_di", behavior_rows)

        # 用户行为/活跃 DWS 改为由 dwd_user_behavior_di 真实聚合推导
        # （评审 Fix 1：原独立 randint 摘要与明细差 10 倍以上）。
        # 保留原 6 次/日 rng 抽取（值弃用）以维持主随机流逐位不变，
        # 订单/商品/支付/物流域已验证数字不受影响；纯聚合不消耗随机流。
        for _ in range(3):
            rng.randint(5_000, active_users)
            rng.randint(10_000, 200_000)

        behavior_agg = {}   # behavior_type -> [cnt, set(users)]
        for r in behavior_rows:
            b = behavior_agg.setdefault(r[1], [0, set()])
            b[0] += 1
            b[1].add(r[0])
        # 全组合补 0：3 type × 90 天 = 270 行不变
        for bt in ("view", "cart", "buy"):
            cnt, users = behavior_agg.get(bt, [0, set()])
            _insert(executor, "dws_user_behavior_summary_di",
                    [(bt, len(users), cnt, dt)])

        # dws_user_active_di：dau = 当日行为明细 distinct 用户数
        dau = len({r[0] for r in behavior_rows})
        _insert(executor, "dws_user_active_di",
                [(active_users, new_users, dau, dt)])

        _insert(executor, "dws_user_retention_di",
                [((d - timedelta(days=1)).isoformat(),
                  round(rng.uniform(0.3, 0.5), 4),
                  round(rng.uniform(0.15, 0.3), 4),
                  round(rng.uniform(0.05, 0.15), 4), dt)])
        _insert(executor, "ads_user_growth_report_di",
                [(new_users, active_users,
                  round(rng.uniform(0.3, 0.5), 4), dt)])

        # dwd_user_register_di：注册明细（当日新增用户数）
        register_rows = []
        for _ in range(new_users):
            register_rows.append((
                detail_rng.randint(1, USERS),
                detail_rng.choice(REGISTER_CHANNELS),
                _time(d, detail_rng),
                detail_rng.choice(CITIES),
                dt))
        _insert(executor, "dwd_user_register_di", register_rows)

        # 商品域：DWD 浏览明细 + DWS/ADS（由明细聚合，口径自洽）
        view_rows = []
        for _ in range(detail_rng.randint(20_000, 40_000)):
            view_rows.append((
                detail_rng.randint(1, USERS),
                detail_rng.randint(1, PRODUCTS),
                _time(d, detail_rng),
                dt))
        _insert(executor, "dwd_product_view_di", view_rows)

        prod_gmv = {}   # pid -> [order_cnt, pay_order_cnt, gmv, pay_users]
        for r in order_rows:
            pid = r[2]
            b = prod_gmv.setdefault(pid, [0, 0, 0.0, set()])
            b[0] += 1
            if r[5] in ("paid", "refunded"):
                b[1] += 1
                b[2] += r[6]
                b[3].add(r[1])
        _insert(executor, "dws_product_gmv_di",
                [(pid, round(b[2], 2), b[1], len(b[3]), dt)
                 for pid, b in prod_gmv.items()])

        view_agg = {}   # pid -> [view_cnt, view_users]
        for r in view_rows:
            pid = r[1]
            b = view_agg.setdefault(pid, [0, set()])
            b[0] += 1
            b[1].add(r[0])
        _insert(executor, "dws_product_view_di",
                [(pid, b[0], len(b[1]), dt) for pid, b in view_agg.items()])

        # ads_product_ranking_di：GMV / 浏览量双榜 Top100（榜外记 0）
        top_gmv = sorted(prod_gmv.items(),
                         key=lambda kv: kv[1][2], reverse=True)[:100]
        top_view = sorted(view_agg.items(),
                          key=lambda kv: kv[1][0], reverse=True)[:100]
        gmv_rank = {pid: i + 1 for i, (pid, _) in enumerate(top_gmv)}
        view_rank = {pid: i + 1 for i, (pid, _) in enumerate(top_view)}
        _insert(executor, "ads_product_ranking_di",
                [(pid, gmv_rank.get(pid, 0), view_rank.get(pid, 0), dt)
                 for pid in sorted(set(gmv_rank) | set(view_rank))])

        # 支付域
        total_pay = sum(r[4] for r in pay_rows)
        _insert(executor, "dws_payment_summary_di",
                [(round(total_pay, 2), len(pay_rows),
                  len({r[2] for r in pay_rows}),
                  round(sum(r[2] for r in refund_rows), 2), dt)])
        for ch in CHANNELS:
            ch_rows = [r for r in pay_rows if r[3] == ch]
            _insert(executor, "dws_payment_channel_di",
                    [(ch, round(sum(r[4] for r in ch_rows), 2),
                      len(ch_rows), dt)])
        refund_amt = sum(r[2] for r in refund_rows)
        _insert(executor, "dws_refund_summary_di",
                [(round(refund_amt, 2), len(refund_rows), dt)])
        _insert(executor, "ads_payment_daily_report_di",
                [(round(total_pay, 2), round(refund_amt, 2),
                  round(total_pay - refund_amt, 2), dt)])

        # 物流域
        ship_cnt = pay_cnt
        sign_cnt = int(ship_cnt * rng.uniform(0.7, 0.9))
        _insert(executor, "dws_logistics_summary_di",
                [(ship_cnt, sign_cnt, round(rng.uniform(1.5, 3.5), 2), dt)])
        for lg in LOGISTICS:
            _insert(executor, "dws_logistics_company_di",
                    [(lg, ship_cnt // 3, sign_cnt // 3, dt)])
        _insert(executor, "ads_logistics_daily_report_di",
                [(ship_cnt, sign_cnt, round(rng.uniform(0.85, 0.97), 4), dt)])

        # dwd_logistics_tracking_di：物流轨迹（支付订单发货/签收）
        tracking_rows = []
        for r in pay_rows:
            oid = r[1]
            company = detail_rng.choice(LOGISTICS)
            status = detail_rng.choices(
                ["shipped", "transit", "signed"], weights=[10, 10, 80])[0]
            ship_time = _time(d, detail_rng)
            sign_time = None
            if status == "signed":
                sign_time = _time(d + timedelta(days=detail_rng.randint(1, 3)),
                                  detail_rng)
            tracking_rows.append((oid, company, status, ship_time, sign_time, dt))
        _insert(executor, "dwd_logistics_tracking_di", tracking_rows)

        # dwd_logistics_shipped_di：发货明细（支付订单全部发货）
        shipped_rows = []
        for r in pay_rows:
            shipped_rows.append((
                r[1], detail_rng.randint(1, 3), _time(d, detail_rng), dt))
        _insert(executor, "dwd_logistics_shipped_di", shipped_rows)

        print(f"[{dt}] orders={orders} pay={pay_cnt} new_users={new_users}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="duckdb",
                        choices=["duckdb", "starrocks"])
    args = parser.parse_args()

    settings = load_config()
    if args.engine == "duckdb":
        db_path = Path(settings.warehouse_path)
        if db_path.exists():
            db_path.unlink()   # 初始化即重建：保证 seed 42 重跑产物一致（无残留叠加）
        executor = DuckDBExecutor(settings.warehouse_path, TABLES)
    else:
        # 注意：StarRocks 重跑会叠加重复数据，需先手动 DROP 库表；
        # 本环境无 Docker，StarRocks 路径待有条件环境验证（见 README）。
        sr = settings.executor.starrocks
        executor = StarRocksExecutor(
            sr.host, sr.port, sr.user, sr.password, TABLES)

    print(f"[init] 建表 {len(TABLES)} 张 ({args.engine})...")
    executor.setup()
    print("[init] 灌维度表...")
    build_dim(executor)
    print("[init] 灌日增量表（90 天）...")
    build_fact_daily(executor)
    print("[init] 生成元数据 YAML...")
    generate_metadata_yaml(settings.metadata_dir)
    executor.close()
    print("[init] 完成")


if __name__ == "__main__":
    main()
