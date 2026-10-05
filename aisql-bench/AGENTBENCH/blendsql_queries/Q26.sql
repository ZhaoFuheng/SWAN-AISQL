-- setup (DuckDB backend)
/* File: synth_tpch_q26_multi_join.sql */ /* Label: Q26 */ /* Question: Which air-shipment lineitems in high-priority open orders (1995-1996) map to semantically marketable parts, dependable suppliers, and business-like customers across supplier geography? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
CREATE TEMPORARY TABLE customer_sample AS SELECT * FROM customer WHERE c_acctbal > 500 AND c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY');
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    p.p_partkey AS p__p_partkey,
    p.p_name AS p__p_name,
    p.p_type AS p__p_type,
    p.p_brand AS p__p_brand,
    CONCAT(
      'Part Name: ',
      CAST(p.p_name AS TEXT),
      '
Part Type: ',
      CAST(p.p_type AS TEXT),
      '
Brand: ',
      CAST(p.p_brand AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM part AS p
), a_candidates AS (
  SELECT
    __b1.p__p_partkey AS p_partkey,
    __b1.p__p_name AS p_name,
    __b1.p__p_type AS p_type,
    __b1.p__p_brand AS p_brand
  FROM __b1
  WHERE
    {{LLMMap('You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?', __b1.__p0)}} = TRUE
), __b2 AS (
  SELECT
    s.s_suppkey AS s__s_suppkey,
    s.s_name AS s__s_name,
    s.s_acctbal AS s__s_acctbal,
    s.s_comment AS s__s_comment,
    n.n_name AS n__n_name,
    r.r_name AS r__r_name,
    n.n_nationkey AS n__n_nationkey,
    s.s_nationkey AS s__s_nationkey,
    r.r_regionkey AS r__r_regionkey,
    n.n_regionkey AS n__n_regionkey,
    CONCAT(
      'Supplier Name: ',
      CAST(s.s_name AS TEXT),
      '
Account Balance: ',
      CAST(s.s_acctbal AS TEXT),
      '
Comment: ',
      CAST(s.s_comment AS TEXT),
      '
Nation: ',
      CAST(n.n_name AS TEXT),
      '
Region: ',
      CAST(r.r_name AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p1
  FROM supplier AS s
  JOIN nation AS n
    ON n.n_nationkey = s.s_nationkey
  JOIN region AS r
    ON r.r_regionkey = n.n_regionkey
  WHERE
    s.s_acctbal > 0
), s_candidates AS (
  SELECT
    __b2.s__s_suppkey AS s_suppkey,
    __b2.s__s_name AS s_name,
    __b2.s__s_acctbal AS s_acctbal,
    __b2.s__s_comment AS s_comment,
    __b2.n__n_name AS supplier_nation,
    __b2.r__r_name AS supplier_region
  FROM __b2
  WHERE
    {{LLMMap('You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look commercially dependable?', __b2.__p1)}} = TRUE
), __b3 AS (
  SELECT
    l.l_orderkey AS l__l_orderkey,
    l.l_linenumber AS l__l_linenumber,
    l.l_partkey AS l__l_partkey,
    l.l_suppkey AS l__l_suppkey,
    l.l_quantity AS l__l_quantity,
    l.l_extendedprice AS l__l_extendedprice,
    l.l_discount AS l__l_discount,
    l.l_tax AS l__l_tax,
    l.l_shipmode AS l__l_shipmode,
    l.l_shipinstruct AS l__l_shipinstruct,
    o.o_orderkey AS o__o_orderkey,
    o.o_totalprice AS o__o_totalprice,
    o.o_orderpriority AS o__o_orderpriority,
    o.o_orderstatus AS o__o_orderstatus,
    l.l_shipdate AS l__l_shipdate,
    CONCAT(
      'Order Priority: ',
      CAST(o.o_orderpriority AS TEXT),
      '
Total Price: ',
      CAST(o.o_totalprice AS TEXT),
      '
Order Comment: ',
      CAST(o.o_comment AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p2
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  WHERE
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1997-01-01' AS DATE)
    AND l.l_discount BETWEEN 0.05 AND 0.08
    AND l.l_quantity BETWEEN 16 AND 24
    AND l.l_shipmode IN ('AIR', 'REG AIR')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    AND o.o_totalprice > 90000
), b_base AS (
  SELECT
    __b3.l__l_orderkey AS l_orderkey,
    __b3.l__l_linenumber AS l_linenumber,
    __b3.l__l_partkey AS l_partkey,
    __b3.l__l_suppkey AS l_suppkey,
    __b3.l__l_quantity AS l_quantity,
    __b3.l__l_extendedprice AS l_extendedprice,
    __b3.l__l_discount AS l_discount,
    __b3.l__l_tax AS l_tax,
    __b3.l__l_shipmode AS l_shipmode,
    __b3.l__l_shipinstruct AS l_shipinstruct
  FROM __b3
  WHERE
    {{LLMMap('You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally important?', __b3.__p2)}} = TRUE
), ps_bridge AS (
  SELECT
    ps.ps_partkey,
    ps.ps_suppkey,
    ps.ps_supplycost,
    ps.ps_availqty
  FROM partsupp AS ps
  JOIN a_candidates AS a
    ON a.p_partkey = ps.ps_partkey
  JOIN s_candidates AS s
    ON s.s_suppkey = ps.ps_suppkey
  WHERE
    ps.ps_supplycost BETWEEN 80 AND 900 AND ps.ps_availqty >= 50
), __b4 AS (
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
    ) AS __p3
  FROM customer_sample AS c
), c_context AS (
  SELECT
    *
    EXCLUDE (__p3)
  FROM __b4
  WHERE
    {{LLMMap('You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer look business-like and operationally relevant?', __b4.__p3)}} = TRUE
)
SELECT
  b.l_orderkey,
  b.l_linenumber,
  b.l_quantity,
  b.l_extendedprice,
  b.l_discount,
  b.l_tax,
  b.l_shipmode,
  b.l_shipinstruct,
  a.p_name,
  a.p_type,
  a.p_brand,
  s.s_name,
  s.supplier_nation,
  s.supplier_region,
  ps.ps_supplycost,
  ps.ps_availqty,
  c.c_custkey AS context_custkey,
  c.c_name,
  c.c_mktsegment,
  c.c_acctbal
FROM b_base AS b
JOIN ps_bridge AS ps
  ON ps.ps_partkey = b.l_partkey AND ps.ps_suppkey = b.l_suppkey
JOIN a_candidates AS a
  ON a.p_partkey = b.l_partkey
JOIN s_candidates AS s
  ON s.s_suppkey = b.l_suppkey
CROSS JOIN c_context AS c
