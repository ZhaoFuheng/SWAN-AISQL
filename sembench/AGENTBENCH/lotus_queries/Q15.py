# Q15: Marketable parts (semantic) x air-shipment lineitems in high-priority open orders,
# cross-joined with business-like customers (semantic) from a 100-row customer sample.
import pandas as pd


def run(load, lm):
    part = load("part")
    lineitem = load("lineitem")
    orders = load("orders")
    customer = load("customer")

    customer_sample = customer.head(100)

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

    b = lineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    ship = pd.to_datetime(b["l_shipdate"])
    b = b[
        (ship >= pd.Timestamp("1995-01-01"))
        & (ship < pd.Timestamp("1997-01-01"))
        & b["l_discount"].astype(float).between(0.05, 0.07)
        & b["l_quantity"].astype(float).between(18, 22)
        & b["l_shipmode"].isin(["AIR", "REG AIR"])
        & (b["l_shipinstruct"] == "DELIVER IN PERSON")
        & (b["o_orderstatus"] == "O")
        & b["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])
        & (b["o_totalprice"].astype(float) > 100000)
    ]
    b_base = b[[
        "l_orderkey", "l_linenumber", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice",
        "l_discount", "l_shipmode", "l_shipinstruct", "o_orderdate", "o_orderpriority", "o_totalprice",
    ]]

    ab = a_candidates.merge(b_base, left_on="p_partkey", right_on="l_partkey")
    ab_join = ab[[
        "p_partkey", "p_name", "p_type", "p_brand",
        "l_orderkey", "l_linenumber", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount",
        "l_shipmode", "l_shipinstruct", "o_orderdate", "o_orderpriority", "o_totalprice",
    ]]

    c = customer_sample
    if len(c) > 0:
        c = c.sem_filter(
            """You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look more like a business customer likely to place larger operational orders?
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return NO."""
        )

    out = ab_join.assign(_ck=1).merge(c.assign(_ck=1), on="_ck").drop(columns="_ck")
    return out
