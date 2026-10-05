# Q28: Operationally significant orders (semantic) x lineitems, equi-joined with strategic
# customers (semantic) plus a SEMANTIC join condition -> equi-merge then semantic filter.
import pandas as pd


def run(load, lm):
    orders = load("orders")
    lineitem = load("lineitem")
    customer = load("customer")

    od = pd.to_datetime(orders["o_orderdate"])
    ob = orders[
        (od >= pd.Timestamp("1995-01-01"))
        & (od < pd.Timestamp("1996-07-01"))
        & orders["o_totalprice"].astype(float).between(100000, 220000)
        & (orders["o_orderstatus"] == "O")
        & orders["o_orderpriority"].isin(["1-URGENT", "2-HIGH"])
    ]
    orders_base = ob[["o_orderkey", "o_orderpriority", "o_totalprice", "o_comment", "o_orderdate", "o_custkey"]]

    of = orders_base
    if len(of) > 0:
        of = of.sem_filter(
            """Does this order look operationally significant based on priority label, total price, and comment content?
Order Priority: {o_orderpriority}
Total Price: {o_totalprice}
Comment: {o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )

    ship = pd.to_datetime(lineitem["l_shipdate"])
    lf = lineitem[
        (ship >= pd.Timestamp("1995-01-01"))
        & (ship < pd.Timestamp("1996-07-01"))
        & lineitem["l_quantity"].astype(float).between(18, 26)
        & lineitem["l_discount"].astype(float).between(0.04, 0.07)
        & lineitem["l_shipmode"].isin(["AIR", "REG AIR"])
    ]
    lineitem_filtered = lf[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice",
        "l_discount", "l_shipmode", "l_shipdate",
    ]]

    joined = of.merge(lineitem_filtered, left_on="o_orderkey", right_on="l_orderkey")
    joined = joined[[
        "o_orderkey", "o_orderpriority", "o_totalprice", "o_comment", "o_orderdate", "o_custkey",
        "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode",
    ]]

    # customer_context: DISTINCT o_custkey LIMIT 60 (no ORDER BY -> first 60 in current order)
    keys = joined[["o_custkey"]].drop_duplicates().head(60)
    cc = customer.merge(keys, left_on="c_custkey", right_on="o_custkey").drop(columns=["o_custkey"])
    if len(cc) > 0:
        cc = cc.sem_filter(
            """Does this customer look operationally strategic based on name style, market segment, and account balance?
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    cc = cc[["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]]

    out = joined.merge(cc, left_on="o_custkey", right_on="c_custkey")
    if len(out) > 0:
        out = out.sem_filter(
            """Could this customer account be strategically important for order fulfillment?
Order Priority: {o_orderpriority}
Order Total Price: {o_totalprice}
Lineitem Ship Mode: {l_shipmode}
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    return out
