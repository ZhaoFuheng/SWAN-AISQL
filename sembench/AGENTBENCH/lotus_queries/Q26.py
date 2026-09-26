# Q26: Marketable parts, dependable suppliers, important orders (all semantic) joined via
# partsupp bridge, cross-joined with business-like customers (semantic).
import pandas as pd


def run(load, lm):
    part = load("part")
    supplier = load("supplier")
    nation = load("nation")
    region = load("region")
    lineitem = load("lineitem")
    orders = load("orders")
    partsupp = load("partsupp")
    customer = load("customer")

    customer_sample = customer[
        (customer["c_acctbal"].astype(float) > 500)
        & customer["c_mktsegment"].isin(["AUTOMOBILE", "BUILDING", "MACHINERY"])
    ]

    a = part
    if len(a) > 0:
        a = a.sem_filter(
            """You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: {p_name}
Part Type: {p_type}
Brand: {p_brand}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    a_candidates = a[["p_partkey", "p_name", "p_type", "p_brand"]]

    s = supplier.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
    s = s.merge(region, left_on="n_regionkey", right_on="r_regionkey")
    s = s[s["s_acctbal"].astype(float) > 0]
    if len(s) > 0:
        s = s.sem_filter(
            """You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look commercially dependable?
Supplier Name: {s_name}
Account Balance: {s_acctbal}
Comment: {s_comment}
Nation: {n_name}
Region: {r_name}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    s_candidates = s.rename(columns={"n_name": "supplier_nation", "r_name": "supplier_region"})[[
        "s_suppkey", "s_name", "s_acctbal", "s_comment", "supplier_nation", "supplier_region",
    ]]

    b = lineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    ship = pd.to_datetime(b["l_shipdate"])
    b = b[
        (ship >= pd.Timestamp("1995-01-01"))
        & (ship < pd.Timestamp("1997-01-01"))
        & b["l_discount"].astype(float).between(0.05, 0.08)
        & b["l_quantity"].astype(float).between(16, 24)
        & b["l_shipmode"].isin(["AIR", "REG AIR"])
        & (b["l_shipinstruct"] == "DELIVER IN PERSON")
        & (b["o_orderstatus"] == "O")
        & b["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])
        & (b["o_totalprice"].astype(float) > 90000)
    ]
    if len(b) > 0:
        b = b.sem_filter(
            """You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally important?
Order Priority: {o_orderpriority}
Total Price: {o_totalprice}
Order Comment: {o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    b_base = b[[
        "l_orderkey", "l_linenumber", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice",
        "l_discount", "l_tax", "l_shipmode", "l_shipinstruct",
    ]]

    ps = partsupp.merge(a_candidates[["p_partkey"]], left_on="ps_partkey", right_on="p_partkey")
    ps = ps.merge(s_candidates[["s_suppkey"]], left_on="ps_suppkey", right_on="s_suppkey")
    ps = ps[
        ps["ps_supplycost"].astype(float).between(80, 900)
        & (ps["ps_availqty"] >= 50)
    ]
    ps_bridge = ps[["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]]

    cc = customer_sample
    if len(cc) > 0:
        cc = cc.sem_filter(
            """You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer look business-like and operationally relevant?
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Comment: {c_comment}
Return a single YES or NO, with no other words. If uncertain, return NO."""
        )

    out = b_base.merge(ps_bridge, left_on=["l_partkey", "l_suppkey"], right_on=["ps_partkey", "ps_suppkey"])
    out = out.merge(a_candidates, left_on="l_partkey", right_on="p_partkey")
    out = out.merge(s_candidates, left_on="l_suppkey", right_on="s_suppkey")
    out = out.assign(_ck=1).merge(cc.assign(_ck=1), on="_ck").drop(columns="_ck")
    out = out.rename(columns={"c_custkey": "context_custkey"})
    return out[[
        "l_orderkey", "l_linenumber", "l_quantity", "l_extendedprice", "l_discount", "l_tax",
        "l_shipmode", "l_shipinstruct",
        "p_name", "p_type", "p_brand",
        "s_name", "supplier_nation", "supplier_region",
        "ps_supplycost", "ps_availqty",
        "context_custkey", "c_name", "c_mktsegment", "c_acctbal",
    ]]
