# Q17: Important orders (semantic) x lineitems x part/supplier, joined ON SEMANTIC with
# prioritized customers (semantic) from a 120-row sample -> cross join + semantic filter.
import pandas as pd


def run(load, lm):
    orders = load("orders")
    lineitem = load("lineitem")
    part = load("part")
    partsupp = load("partsupp")
    supplier = load("supplier")
    nation = load("nation")
    customer = load("customer")

    customer_sample = customer.head(120)

    od = pd.to_datetime(orders["o_orderdate"])
    o = orders[
        (od >= pd.Timestamp("1993-01-01"))
        & (od < pd.Timestamp("1997-01-01"))
        & orders["o_orderstatus"].isin(["O", "F"])
        & (orders["o_totalprice"].astype(float) > 50000)
    ]
    if len(o) > 0:
        o = o.sem_filter(
            """You are triaging enterprise orders for operations monitoring. From the priority label, total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager?
Order Priority: {o_orderpriority}
Total Price: {o_totalprice}
Comment: {o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    order_focus = o[["o_orderkey", "o_custkey", "o_orderdate", "o_orderpriority", "o_totalprice", "o_comment"]]

    ship = pd.to_datetime(lineitem["l_shipdate"])
    lf = lineitem[
        (ship >= pd.Timestamp("1993-01-01"))
        & (ship < pd.Timestamp("1998-01-01"))
        & lineitem["l_quantity"].astype(float).between(5, 40)
        & lineitem["l_discount"].astype(float).between(0.01, 0.10)
        & lineitem["l_shipmode"].isin(["AIR", "REG AIR", "TRUCK"])
    ]
    lineitem_focus = lf[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_shipmode", "l_shipinstruct",
        "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag",
    ]]

    ps = part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
    ps = ps.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    ps = ps.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
    ps = ps[
        ps["p_size"].astype(int).between(5, 35)
        & ps["ps_supplycost"].astype(float).between(10, 1000)
    ]
    part_supplier = ps.rename(columns={"n_name": "supplier_nation"})[[
        "p_partkey", "p_name", "p_type", "p_brand", "s_suppkey", "s_name", "supplier_nation",
    ]]

    joined = order_focus.merge(lineitem_focus, left_on="o_orderkey", right_on="l_orderkey")
    joined = joined.merge(part_supplier, left_on=["l_partkey", "l_suppkey"], right_on=["p_partkey", "s_suppkey"])
    joined = joined[[
        "o_orderkey", "o_custkey", "o_orderdate", "o_orderpriority", "o_totalprice",
        "l_partkey", "l_suppkey", "l_shipmode", "l_shipinstruct", "l_quantity", "l_extendedprice",
        "l_discount", "l_tax", "l_returnflag",
        "p_name", "p_type", "p_brand", "s_name", "supplier_nation",
    ]]

    c = customer_sample
    if len(c) > 0:
        c = c.sem_filter(
            """You are identifying high-value B2B customer profiles. From the customer name format, market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics?
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Comment: {c_comment}
Return a single YES or NO, with no other words. If uncertain, return NO."""
        )

    # JOIN ... ON SEMANTIC(...) where the prompt reads ONLY customer-side columns: the predicate
    # commutes with the cross product, so filter that side FIRST (what any practitioner writes;
    # the literal pair-wise form is 481k calls here for identical results). The trailing ')'
    # after {c_comment} is present in the SQL prompt text and kept verbatim.
    if len(c) > 0:
        c = c.sem_filter(
            """You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up. The customer has the following profile:
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Comment: {c_comment})"""
        )
    out = joined.assign(_ck=1).merge(c.assign(_ck=1), on="_ck").drop(columns="_ck")
    return out
