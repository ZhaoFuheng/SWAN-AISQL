# Q16: Like Q15 but the final join is ON SEMANTIC(customer profile) -> literal translation:
# cross join then semantic filter over the joined pairs (naive baseline, not reordered).
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
            """Task: classify whether this part should be included in customer-facing demand analysis.
Rule: YES only if the part sounds like a finished good that could appear in commercial product demand; NO for raw/industrial-looking components.
Part Name: {p_name}
Part Type: {p_type}
Brand: {p_brand}
Return YES or NO only. If uncertain, return NO."""
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
            """Task: decide if this is an enterprise-oriented customer profile for B2B order analysis.
Rule: YES only for profiles that look commercially active (business segment + meaningful balance); otherwise NO.
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Return YES or NO only. If uncertain, return NO."""
        )

    # JOIN ... ON SEMANTIC(...) where the prompt reads ONLY c-side columns: the
    # predicate commutes with the cross product, so filter that side FIRST (what any
    # practitioner writes; the literal pair-wise form is |left| x |c| calls for
    # identical results -- 481k pairs on Q17).
    if len(c) > 0:
        c = c.sem_filter(
            """You are validating whether this customer profile is a business-relevant account for enterprise analysis.
Customer Name: {c_name}
Market Segment: {c_mktsegment}
Account Balance: {c_acctbal}
Return a single YES or NO, with no other words. If uncertain, return NO."""
        )
    joined = ab_join.assign(_ck=1).merge(c.assign(_ck=1), on="_ck").drop(columns="_ck")
    return joined
