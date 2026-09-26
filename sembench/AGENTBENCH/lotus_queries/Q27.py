# Q27: Differentiated parts (semantic) x supply base x orders, equi-joined with dependable
# suppliers (semantic) plus a SEMANTIC join condition -> equi-merge then semantic filter.
# SELECT * duplicates s_name/s_acctbal/s_comment in SQL; the joined-side copies get an
# _j suffix here so the supplier_context copies keep the names the prompt references.
import pandas as pd


def run(load, lm):
    part = load("part")
    partsupp = load("partsupp")
    supplier = load("supplier")
    nation = load("nation")
    region = load("region")
    lineitem = load("lineitem")
    orders = load("orders")

    pf = part
    if len(pf) > 0:
        pf = pf.sem_filter(
            """You are screening parts for strategic sourcing analysis. Looking at part name, type, brand, and size, does this look like a differentiated component where supplier quality and pricing strategy would matter?
Part Name: {p_name}
Part Type: {p_type}
Brand: {p_brand}
Size: {p_size}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    part_focus = pf[["p_partkey", "p_name", "p_type", "p_brand", "p_size"]]

    sb = partsupp.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    sb = sb.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
    sb = sb.merge(region, left_on="n_regionkey", right_on="r_regionkey")
    sb = sb[
        sb["ps_availqty"].between(10, 600)
        & sb["ps_supplycost"].astype(float).between(20, 900)
        & sb["r_name"].isin(["EUROPE", "ASIA", "AMERICA"])
    ]
    supply_base = sb.rename(columns={"n_name": "nation_name", "r_name": "region_name"})[[
        "ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost",
        "s_name", "s_acctbal", "s_comment", "s_nationkey", "nation_name", "region_name",
    ]]

    ol = lineitem.merge(orders, left_on="l_orderkey", right_on="o_orderkey")
    od = pd.to_datetime(ol["o_orderdate"])
    ol = ol[
        (od >= pd.Timestamp("1994-01-01"))
        & (od < pd.Timestamp("1997-01-01"))
        & ol["l_quantity"].astype(float).between(8, 35)
        & ol["l_discount"].astype(float).between(0.00, 0.08)
    ]
    orders_lineitem = ol[[
        "l_partkey", "l_suppkey", "l_orderkey", "l_quantity", "l_extendedprice", "l_discount",
        "o_orderdate", "o_orderpriority", "o_totalprice",
    ]]

    joined = part_focus.merge(supply_base, left_on="p_partkey", right_on="ps_partkey")
    joined = joined.merge(orders_lineitem, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
    joined = joined[[
        "p_partkey", "p_name", "p_type", "p_brand",
        "ps_suppkey", "s_name", "s_acctbal", "s_comment", "nation_name", "region_name",
        "ps_availqty", "ps_supplycost",
        "l_orderkey", "l_quantity", "l_extendedprice",
        "o_orderdate", "o_orderpriority", "o_totalprice",
    ]]

    # supplier_context: DISTINCT ps_suppkey LIMIT 80 (no ORDER BY -> first 80 in current order)
    keys = joined[["ps_suppkey"]].drop_duplicates().head(80)
    sc = supplier.merge(keys, left_on="s_suppkey", right_on="ps_suppkey").drop(columns=["ps_suppkey"])
    if len(sc) > 0:
        sc = sc.sem_filter(
            """You are shortlisting suppliers for long-term procurement partnerships. Based on supplier identity style, account balance, and profile comment tone, does this supplier appear commercially dependable enough for preferred-vendor consideration?
Supplier Name: {s_name}
Account Balance: {s_acctbal}
Comment: {s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    sc = sc[["s_suppkey", "s_name", "s_address", "s_nationkey", "s_phone", "s_acctbal", "s_comment"]]

    out = joined.merge(sc, left_on="ps_suppkey", right_on="s_suppkey", suffixes=("_j", ""))
    if len(out) > 0:
        out = out.sem_filter(
            """You are validating whether this supplier profile is suitable for preferred-vendor sourcing.
Part Type: {p_type}
Region: {region_name}
Supplier Name: {s_name}
Account Balance: {s_acctbal}
Comment: {s_comment}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )
    return out
