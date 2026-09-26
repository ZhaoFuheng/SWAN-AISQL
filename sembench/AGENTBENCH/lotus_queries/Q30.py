# Q30: Distribution-suitable parts (semantic) x lineitem/order/customer activity, equi-joined
# with qualified suppliers (semantic) plus a SEMANTIC join condition -> merge then filter.
import pandas as pd


def run(load, lm):
    lineitem = load("lineitem")
    orders = load("orders")
    customer = load("customer")
    part = load("part")
    supplier = load("supplier")

    olc = lineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    olc = olc.merge(customer, left_on="o_custkey", right_on="c_custkey")
    od = pd.to_datetime(olc["o_orderdate"])
    olc = olc[
        olc["l_shipmode"].isin(["AIR", "REG AIR", "TRUCK"])
        & olc["l_quantity"].astype(float).between(12, 30)
        & olc["l_discount"].astype(float).between(0.04, 0.07)
        & (od >= pd.Timestamp("1996-01-01"))
        & (od < pd.Timestamp("1998-01-01"))
        & olc["o_totalprice"].astype(float).between(60000, 170000)
        & olc["c_mktsegment"].isin(["FURNITURE", "HOUSEHOLD", "MACHINERY", "BUILDING"])
    ]
    orders_lineitem_customer = olc[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount",
        "l_shipmode", "o_orderdate", "o_orderpriority", "o_totalprice",
        "c_custkey", "c_name", "c_mktsegment", "c_acctbal",
    ]]

    # part_base: parts among the first 200 distinct l_partkey (no ORDER BY -> current order)
    pkeys = orders_lineitem_customer[["l_partkey"]].drop_duplicates().head(200)
    pb = part.merge(pkeys, left_on="p_partkey", right_on="l_partkey").drop(columns=["l_partkey"])
    pb = pb[
        pb["p_size"].astype(int).between(5, 28)
        & pb["p_container"].isin(["SM BOX", "MED BOX", "LG BOX", "SM CASE"])
    ]
    part_base = pb[["p_partkey", "p_name", "p_type", "p_container", "p_size"]]

    pf = part_base
    if len(pf) > 0:
        pf = pf.sem_filter(
            """Does this part look suitable for customer-facing distribution based on name, type, and packaging?
Part Name: {p_name}
Part Type: {p_type}
Container: {p_container}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )

    joined = pf.merge(orders_lineitem_customer, left_on="p_partkey", right_on="l_partkey")
    joined = joined[[
        "p_partkey", "p_name", "p_type", "p_container",
        "l_orderkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode",
        "o_orderdate", "o_orderpriority", "o_totalprice",
        "c_custkey", "c_name", "c_mktsegment", "c_acctbal",
    ]]

    # supplier_context: DISTINCT l_suppkey LIMIT 40 (no ORDER BY -> first 40 in current order)
    skeys = joined[["l_suppkey"]].drop_duplicates().head(40)
    sc = supplier.merge(skeys, left_on="s_suppkey", right_on="l_suppkey").drop(columns=["l_suppkey"])
    if len(sc) > 0:
        sc = sc.sem_filter(
            """Does this supplier appear qualified for consumer-facing distribution partnerships?
Supplier Name: {s_name}
Account Balance: {s_acctbal}
Comment: {s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    sc = sc[["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]]

    out = joined.merge(sc, left_on="l_suppkey", right_on="s_suppkey")
    if len(out) > 0:
        out = out.sem_filter(
            """Would this supplier be suitable for retail distribution of this product?
Part Type: {p_type}
Ship Mode: {l_shipmode}
Customer Segment: {c_mktsegment}
Supplier Name: {s_name}
Account Balance: {s_acctbal}
Comment: {s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    return out
