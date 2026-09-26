# Q19: Problematic returned lineitems (semantic) x order/customer x part/supplier,
# cross-joined with higher-risk customer profiles (semantic) from a 180-row sample.
# The SQL prompts contain literal backslash-n sequences (standard SQL strings do not
# interpret escapes), so they are kept as literal \n characters here.
import pandas as pd


def run(load, lm):
    lineitem = load("lineitem")
    orders = load("orders")
    customer = load("customer")
    part = load("part")
    partsupp = load("partsupp")
    supplier = load("supplier")

    customer_sample = customer.head(180)

    ship = pd.to_datetime(lineitem["l_shipdate"])
    lr = lineitem[
        (ship >= pd.Timestamp("1994-01-01"))
        & (ship < pd.Timestamp("1998-01-01"))
        & lineitem["l_returnflag"].isin(["R", "A", "N"])
        & lineitem["l_quantity"].astype(float).between(3, 38)
        & lineitem["l_discount"].astype(float).between(0.00, 0.10)
    ]
    if len(lr) > 0:
        lr = lr.sem_filter(
            "Lineitem shipping details:\\nMode: {l_shipmode}\\nInstruction: {l_shipinstruct}\\nComment: {l_comment}\\nIs this likely a potentially problematic fulfillment case worth audit attention? Answer YES or NO. If unsure, answer YES."
        )
    lineitem_returns = lr[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount",
        "l_tax", "l_returnflag", "l_linestatus", "l_shipmode", "l_shipinstruct", "l_comment",
    ]]

    oc = orders.merge(customer, left_on="o_custkey", right_on="c_custkey")
    od = pd.to_datetime(oc["o_orderdate"])
    oc = oc[
        (od >= pd.Timestamp("1994-01-01"))
        & (od < pd.Timestamp("1998-01-01"))
        & oc["o_orderstatus"].isin(["O", "F"])
        & (oc["o_totalprice"].astype(float) > 20000)
    ]
    order_customer = oc[[
        "o_orderkey", "o_custkey", "o_orderdate", "o_orderpriority", "o_orderstatus", "o_totalprice",
        "c_name", "c_mktsegment", "c_acctbal", "c_comment",
    ]]

    ps = part.merge(partsupp, left_on="p_partkey", right_on="ps_partkey")
    ps = ps.merge(supplier, left_on="ps_suppkey", right_on="s_suppkey")
    ps = ps[
        ps["p_size"].astype(int).between(1, 40)
        & ps["ps_supplycost"].astype(float).between(5, 1200)
    ]
    part_supplier = ps[["p_partkey", "p_name", "p_type", "p_brand", "s_suppkey", "s_name", "s_acctbal"]]

    joined = lineitem_returns.merge(order_customer, left_on="l_orderkey", right_on="o_orderkey")
    joined = joined.merge(part_supplier, left_on=["l_partkey", "l_suppkey"], right_on=["p_partkey", "s_suppkey"])
    joined = joined[[
        "l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount",
        "l_tax", "l_returnflag", "l_linestatus", "l_shipmode", "l_shipinstruct",
        "o_orderdate", "o_orderpriority", "o_orderstatus", "o_totalprice",
        "c_name", "c_mktsegment", "c_acctbal",
        "p_name", "p_type", "p_brand", "s_name", "s_acctbal",
    ]]

    cc = customer_sample
    if len(cc) > 0:
        cc = cc.sem_filter(
            "Customer profile check:\\nName: {c_name}\\nSegment: {c_mktsegment}\\nBalance: {c_acctbal}\\nComment: {c_comment}\\nAnswer YES only if this customer appears likely to have higher complaint/escalation risk in commercial operations. Otherwise answer NO. If unsure, answer NO."
        )

    # SELECT * over CROSS JOIN duplicates c_name/c_mktsegment/c_acctbal column names in SQL;
    # the customer_context copies are disambiguated with an _cc suffix here.
    out = joined.assign(_ck=1).merge(cc.assign(_ck=1), on="_ck", suffixes=("", "_cc")).drop(columns="_ck")
    return out
