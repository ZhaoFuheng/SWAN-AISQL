-- setup (DuckDB backend)
/* File: synth_tpch_q5.sql */ /* Label: Q19 */ /* Question: Which potentially problematic returned lineitems (1994-1997), with related order/customer and part/supplier context, should be analyzed alongside semantically higher-risk customer profiles? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
CREATE TEMPORARY TABLE customer_sample AS SELECT * FROM customer LIMIT 180;
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    l.l_orderkey AS l__l_orderkey,
    l.l_partkey AS l__l_partkey,
    l.l_suppkey AS l__l_suppkey,
    l.l_quantity AS l__l_quantity,
    l.l_extendedprice AS l__l_extendedprice,
    l.l_discount AS l__l_discount,
    l.l_tax AS l__l_tax,
    l.l_returnflag AS l__l_returnflag,
    l.l_linestatus AS l__l_linestatus,
    l.l_shipmode AS l__l_shipmode,
    l.l_shipinstruct AS l__l_shipinstruct,
    l.l_comment AS l__l_comment,
    l.l_shipdate AS l__l_shipdate,
    CONCAT(
      CAST(l.l_shipmode AS TEXT),
      '\nInstruction: ',
      CAST(l.l_shipinstruct AS TEXT),
      '\nComment: ',
      CAST(l.l_comment AS TEXT),
      '\nIs this likely a potentially problematic fulfillment case worth audit attention? Answer YES or NO. If unsure, answer YES.'
    ) AS __p0
  FROM lineitem AS l
  WHERE
    l.l_shipdate >= CAST('1994-01-01' AS DATE)
    AND l.l_shipdate < CAST('1998-01-01' AS DATE)
    AND l.l_returnflag IN ('R', 'A', 'N')
    AND l.l_quantity BETWEEN 3 AND 38
    AND l.l_discount BETWEEN 0.00 AND 0.10
), lineitem_returns AS (
  SELECT
    __b1.l__l_orderkey AS l_orderkey,
    __b1.l__l_partkey AS l_partkey,
    __b1.l__l_suppkey AS l_suppkey,
    __b1.l__l_quantity AS l_quantity,
    __b1.l__l_extendedprice AS l_extendedprice,
    __b1.l__l_discount AS l_discount,
    __b1.l__l_tax AS l_tax,
    __b1.l__l_returnflag AS l_returnflag,
    __b1.l__l_linestatus AS l_linestatus,
    __b1.l__l_shipmode AS l_shipmode,
    __b1.l__l_shipinstruct AS l_shipinstruct,
    __b1.l__l_comment AS l_comment
  FROM __b1
  WHERE
    {{LLMMap('Lineitem shipping details:\nMode:', __b1.__p0)}} = TRUE
), order_customer AS (
  SELECT
    o.o_orderkey,
    o.o_custkey,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_orderstatus,
    o.o_totalprice,
    c.c_name,
    c.c_mktsegment,
    c.c_acctbal,
    c.c_comment
  FROM orders AS o
  JOIN customer AS c
    ON c.c_custkey = o.o_custkey
  WHERE
    o.o_orderdate >= CAST('1994-01-01' AS DATE)
    AND o.o_orderdate < CAST('1998-01-01' AS DATE)
    AND o.o_orderstatus IN ('O', 'F')
    AND o.o_totalprice > 20000
), part_supplier AS (
  SELECT
    p.p_partkey,
    p.p_name,
    p.p_type,
    p.p_brand,
    s.s_suppkey,
    s.s_name,
    s.s_acctbal
  FROM part AS p
  JOIN partsupp AS ps
    ON ps.ps_partkey = p.p_partkey
  JOIN supplier AS s
    ON s.s_suppkey = ps.ps_suppkey
  WHERE
    p.p_size BETWEEN 1 AND 40 AND ps.ps_supplycost BETWEEN 5 AND 1200
), joined AS (
  SELECT
    lr.l_orderkey,
    lr.l_partkey,
    lr.l_suppkey,
    lr.l_quantity,
    lr.l_extendedprice,
    lr.l_discount,
    lr.l_tax,
    lr.l_returnflag,
    lr.l_linestatus,
    lr.l_shipmode,
    lr.l_shipinstruct,
    oc.o_orderdate,
    oc.o_orderpriority,
    oc.o_orderstatus,
    oc.o_totalprice,
    oc.c_name,
    oc.c_mktsegment,
    oc.c_acctbal,
    ps.p_name,
    ps.p_type,
    ps.p_brand,
    ps.s_name,
    ps.s_acctbal
  FROM lineitem_returns AS lr
  JOIN order_customer AS oc
    ON oc.o_orderkey = lr.l_orderkey
  JOIN part_supplier AS ps
    ON ps.p_partkey = lr.l_partkey AND ps.s_suppkey = lr.l_suppkey
), __b2 AS (
  SELECT
    *,
    CONCAT(
      CAST(c.c_name AS TEXT),
      '\nSegment: ',
      CAST(c.c_mktsegment AS TEXT),
      '\nBalance: ',
      CAST(c.c_acctbal AS TEXT),
      '\nComment: ',
      CAST(c.c_comment AS TEXT),
      '\nAnswer YES only if this customer appears likely to have higher complaint/escalation risk in commercial operations. Otherwise answer NO. If unsure, answer NO.'
    ) AS __p1
  FROM customer_sample AS c
), customer_context AS (
  SELECT
    *
    EXCLUDE (__p1)
  FROM __b2
  WHERE
    {{LLMMap('Customer profile check:\nName:', __b2.__p1)}} = TRUE
)
SELECT
  *
FROM joined AS j
CROSS JOIN customer_context AS cc
