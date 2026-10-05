# Q18: Distribution-oriented parts (semantic) x lineitem/orders x customer geography,
# cross-joined with semantically relevant nations from a 25-row nation sample.
import pandas as pd


def run(load, lm):
    part = load("part")
    lineitem = load("lineitem")
    orders = load("orders")
    customer = load("customer")
    nation = load("nation")
    region = load("region")

    nation_sample = nation.head(25)

    p = part[
        part["p_size"].astype(int).between(3, 30)
        & part["p_container"].isin(["SM BOX", "SM CASE", "MED BOX", "MED BAG", "LG BOX"])
    ]
    if len(p) > 0:
        p = p.sem_filter(
            """You are preparing a logistics performance dashboard for retail distribution. Based on product name, type, packaging container, and size, does this part look like something that would realistically move through customer-facing distribution channels?
Part Name: {p_name}
Part Type: {p_type}
Container: {p_container}
Size: {p_size}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    part_candidates = p[["p_partkey", "p_name", "p_type", "p_container", "p_size"]]

    lo = lineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    od = pd.to_datetime(lo["o_orderdate"])
    lo = lo[
        lo["l_shipmode"].isin(["AIR", "REG AIR", "TRUCK", "MAIL"])
        & lo["l_shipinstruct"].isin(["DELIVER IN PERSON", "TAKE BACK RETURN"])
        & lo["l_quantity"].astype(float).between(10, 45)
        & (od >= pd.Timestamp("1994-01-01"))
        & (od < pd.Timestamp("1998-01-01"))
        & (lo["o_totalprice"].astype(float) > 30000)
    ]
    lineitem_orders = lo[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_shipmode", "l_shipinstruct",
        "l_quantity", "l_extendedprice", "l_discount",
        "o_custkey", "o_orderdate", "o_orderpriority", "o_totalprice",
    ]]

    cg = customer.merge(nation, left_on="c_nationkey", right_on="n_nationkey")
    cg = cg.merge(region, left_on="n_regionkey", right_on="r_regionkey")
    cg = cg[cg["r_name"].isin(["AMERICA", "EUROPE", "ASIA"])]
    customer_geo = cg.rename(columns={"n_name": "customer_nation", "r_name": "customer_region"})[[
        "c_custkey", "c_name", "c_nationkey", "c_mktsegment", "c_acctbal", "customer_nation", "customer_region",
    ]]

    joined = part_candidates.merge(lineitem_orders, left_on="p_partkey", right_on="l_partkey")
    joined = joined.merge(customer_geo, left_on="o_custkey", right_on="c_custkey")
    joined = joined[[
        "p_partkey", "p_name", "p_type", "p_container",
        "l_orderkey", "l_shipmode", "l_shipinstruct", "l_quantity", "l_extendedprice", "l_discount",
        "o_orderdate", "o_orderpriority", "o_totalprice",
        "c_name", "c_mktsegment", "c_acctbal", "customer_nation", "customer_region",
    ]]

    nc = nation_sample
    if len(nc) > 0:
        nc = nc.sem_filter(
            """You are selecting countries to include in a cross-border freight and trade analysis view. Is this nation likely meaningful enough to keep in a global logistics discussion dataset?
Nation: {n_name}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )

    out = joined.assign(_ck=1).merge(nc.assign(_ck=1), on="_ck").drop(columns="_ck")
    return out
