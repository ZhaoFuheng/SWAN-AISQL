# Q29: Reliable suppliers with qualifying partsupp activity, grouped candidates, then a
# SEMANTIC join condition against nation/region context -> equi-merge then semantic filter.
# Note: despite its name, semantic_candidates has no semantic op in the SQL (pure GROUP BY).
import pandas as pd


def run(load, lm):
    partsupp = load("partsupp")
    supplier = load("supplier")
    nation = load("nation")
    region = load("region")

    pb = partsupp[
        partsupp["ps_availqty"].between(1600, 3200)
        & partsupp["ps_supplycost"].astype(float).between(200, 420)
    ]
    partsupp_base = pb[["ps_partkey", "ps_suppkey", "ps_availqty", "ps_supplycost"]]

    # DISTINCT ps_suppkey LIMIT 70 (no ORDER BY -> first 70 in current order)
    keys = partsupp_base[["ps_suppkey"]].drop_duplicates().head(70)
    sb = supplier.merge(nation, left_on="s_nationkey", right_on="n_nationkey")
    sb = sb.merge(region, left_on="n_regionkey", right_on="r_regionkey")
    sb = sb.merge(keys, left_on="s_suppkey", right_on="ps_suppkey").drop(columns=["ps_suppkey"])
    sb = sb[
        sb["s_acctbal"].astype(float).between(2000, 8000)
        & sb["r_name"].isin(["EUROPE", "ASIA", "AMERICA"])
    ]
    supplier_base = sb.rename(columns={"n_name": "supplier_nation_name", "r_name": "supplier_region_name"})[[
        "s_suppkey", "s_name", "s_address", "s_acctbal", "s_comment", "s_nationkey",
        "supplier_nation_name", "supplier_region_name",
    ]]

    joined = supplier_base.merge(partsupp_base, left_on="s_suppkey", right_on="ps_suppkey")
    joined = joined[[
        "s_suppkey", "s_name", "s_address", "s_acctbal", "s_comment", "s_nationkey",
        "supplier_nation_name", "supplier_region_name",
        "ps_partkey", "ps_availqty", "ps_supplycost",
    ]]

    # semantic_candidates: GROUP BY s_suppkey, s_nationkey with MIN aggregates, LIMIT 50
    scand = (
        joined.groupby(["s_suppkey", "s_nationkey"], sort=False, as_index=False)
        .agg(
            s_acctbal=("s_acctbal", "min"),
            ps_supplycost=("ps_supplycost", "min"),
            supplier_region_name=("supplier_region_name", "min"),
        )
        .head(50)
    )

    # nation_context: nations of the first 5 distinct candidate nation keys
    nk = scand[["s_nationkey"]].drop_duplicates().head(5)
    nc = nation.merge(region, left_on="n_regionkey", right_on="r_regionkey")
    nc = nc.merge(nk, left_on="n_nationkey", right_on="s_nationkey").drop(columns=["s_nationkey"])
    nc = nc[["n_nationkey", "n_name", "n_regionkey", "n_comment", "r_regionkey", "r_name", "r_comment"]]

    # joined x semantic_candidates (aggregated copies keep the plain names for the prompt;
    # the joined-side duplicates get an _j suffix) x nation_context, then the SEMANTIC condition.
    out = joined.merge(scand, on=["s_suppkey", "s_nationkey"], suffixes=("_j", ""))
    out = out.merge(nc, left_on="s_nationkey", right_on="n_nationkey")
    if len(out) > 0:
        out = out.sem_filter(
            """Does this sourcing region align with procurement priorities?
Supplier Account Balance: {s_acctbal}
Supply Cost: {ps_supplycost}
Supplier Region: {supplier_region_name}
Nation Name: {n_name}
Region Name: {r_name}
Return a single YES or NO, with no other words. If uncertain, return YES."""
        )

    out = out[[
        "s_suppkey", "s_name", "s_address", "s_acctbal_j", "s_comment", "s_nationkey",
        "supplier_nation_name", "supplier_region_name_j",
        "ps_partkey", "ps_availqty", "ps_supplycost_j",
        "n_nationkey", "n_name", "n_regionkey", "r_regionkey", "r_name", "n_comment", "r_comment",
    ]].rename(columns={
        "s_acctbal_j": "s_acctbal",
        "supplier_region_name_j": "supplier_region_name",
        "ps_supplycost_j": "ps_supplycost",
        "n_nationkey": "context_nationkey",
        "n_name": "context_nation_name",
        "n_regionkey": "context_regionkey",
        "r_regionkey": "context_region_id",
        "r_name": "context_region_name",
        "n_comment": "context_nation_comment",
        "r_comment": "context_region_comment",
    })
    return out
