-- setup (DuckDB backend)
/* File: synth_tpch_q3.sql */ /* Label: Q17 */ /* Question: Which important orders and lineitems (1993-1997) with matching part-supplier info are associated with semantically prioritized business customers? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
CREATE TEMPORARY TABLE customer_sample AS SELECT * FROM customer LIMIT 120;
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    o.o_orderkey AS o__o_orderkey,
    o.o_custkey AS o__o_custkey,
    o.o_orderdate AS o__o_orderdate,
    o.o_orderpriority AS o__o_orderpriority,
    o.o_totalprice AS o__o_totalprice,
    o.o_comment AS o__o_comment,
    o.o_orderstatus AS o__o_orderstatus,
    CONCAT(
      'Order Priority: ',
      CAST(o.o_orderpriority AS TEXT),
      '
Total Price: ',
      CAST(o.o_totalprice AS TEXT),
      '
Comment: ',
      CAST(o.o_comment AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM orders AS o
  WHERE
    o.o_orderdate >= CAST('1993-01-01' AS DATE)
    AND o.o_orderdate < CAST('1997-01-01' AS DATE)
    AND o.o_orderstatus IN ('O', 'F')
    AND o.o_totalprice > 50000
), order_focus AS (
  SELECT
    __b1.o__o_orderkey AS o_orderkey,
    __b1.o__o_custkey AS o_custkey,
    __b1.o__o_orderdate AS o_orderdate,
    __b1.o__o_orderpriority AS o_orderpriority,
    __b1.o__o_totalprice AS o_totalprice,
    __b1.o__o_comment AS o_comment
  FROM __b1
  WHERE
    {{LLMMap('You are triaging enterprise orders for operations monitoring. From the priority label, total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager?', __b1.__p0)}} = TRUE
), lineitem_focus AS (
  SELECT
    l.l_orderkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_shipmode,
    l.l_shipinstruct,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_tax,
    l.l_returnflag
  FROM lineitem AS l
  WHERE
    l.l_shipdate >= CAST('1993-01-01' AS DATE)
    AND l.l_shipdate < CAST('1998-01-01' AS DATE)
    AND l.l_quantity BETWEEN 5 AND 40
    AND l.l_discount BETWEEN 0.01 AND 0.10
    AND l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
), part_supplier AS (
  SELECT
    p.p_partkey,
    p.p_name,
    p.p_type,
    p.p_brand,
    s.s_suppkey,
    s.s_name,
    n.n_name AS supplier_nation
  FROM part AS p
  JOIN partsupp AS ps
    ON ps.ps_partkey = p.p_partkey
  JOIN supplier AS s
    ON s.s_suppkey = ps.ps_suppkey
  JOIN nation AS n
    ON n.n_nationkey = s.s_nationkey
  WHERE
    p.p_size BETWEEN 5 AND 35 AND ps.ps_supplycost BETWEEN 10 AND 1000
), joined AS (
  SELECT
    of.o_orderkey,
    of.o_custkey,
    of.o_orderdate,
    of.o_orderpriority,
    of.o_totalprice,
    lf.l_partkey,
    lf.l_suppkey,
    lf.l_shipmode,
    lf.l_shipinstruct,
    lf.l_quantity,
    lf.l_extendedprice,
    lf.l_discount,
    lf.l_tax,
    lf.l_returnflag,
    ps.p_name,
    ps.p_type,
    ps.p_brand,
    ps.s_name,
    ps.supplier_nation
  FROM order_focus AS of
  JOIN lineitem_focus AS lf
    ON lf.l_orderkey = of.o_orderkey
  JOIN part_supplier AS ps
    ON ps.p_partkey = lf.l_partkey AND ps.s_suppkey = lf.l_suppkey
), __b2 AS (
  SELECT
    *,
    CONCAT(
      'Customer Name: ',
      CAST(c.c_name AS TEXT),
      '
Market Segment: ',
      CAST(c.c_mktsegment AS TEXT),
      '
Account Balance: ',
      CAST(c.c_acctbal AS TEXT),
      '
Comment: ',
      CAST(c.c_comment AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return NO.'
    ) AS __p1
  FROM customer_sample AS c
), customer_context AS (
  SELECT
    *
    EXCLUDE (__p1)
  FROM __b2
  WHERE
    {{LLMMap('You are identifying high-value B2B customer profiles. From the customer name format, market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics?', __b2.__p1)}} = TRUE
), __b3 AS (
  SELECT
    *,
    CONCAT(
      'Customer Name: ',
      CAST(c.c_name AS TEXT),
      '
Market Segment: ',
      CAST(c.c_mktsegment AS TEXT),
      '
Account Balance: ',
      CAST(c.c_acctbal AS TEXT),
      '
Comment: ',
      CAST(c.c_comment AS TEXT),
      ')'
    ) AS __p2
  FROM joined AS j, customer_context AS c
)
SELECT
  *
  EXCLUDE (__p2)
FROM __b3
WHERE
  {{LLMMap('You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up. The customer has the following profile:', __b3.__p2)}} = TRUE
