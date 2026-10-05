-- setup (DuckDB backend)
/* File: synth_tpch_q1.sql */ /* Label: Q15 */ /* Question: Which lineitems in high-priority open orders (1995-1996) match air-shipment discount/quantity filters for semantically marketable parts, paired with semantically business-like customers? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
CREATE TEMPORARY TABLE customer_sample AS SELECT * FROM customer LIMIT 100;
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
), b_base AS (
  SELECT
    l.l_orderkey,
    l.l_linenumber,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_shipmode,
    l.l_shipinstruct,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_totalprice
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  WHERE
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1997-01-01' AS DATE)
    AND l.l_discount BETWEEN 0.05 AND 0.07
    AND l.l_quantity BETWEEN 18 AND 22
    AND l.l_shipmode IN ('AIR', 'REG AIR')
    AND l.l_shipinstruct = 'DELIVER IN PERSON'
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
    AND o.o_totalprice > 100000
), ab_join AS (
  SELECT
    a.p_partkey,
    a.p_name,
    a.p_type,
    a.p_brand,
    b.l_orderkey,
    b.l_linenumber,
    b.l_suppkey,
    b.l_quantity,
    b.l_extendedprice,
    b.l_discount,
    b.l_shipmode,
    b.l_shipinstruct,
    b.o_orderdate,
    b.o_orderpriority,
    b.o_totalprice
  FROM a_candidates AS a
  JOIN b_base AS b
    ON b.l_partkey = a.p_partkey
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
Return a single YES or NO, with no other words. If uncertain, return NO.'
    ) AS __p1
  FROM customer_sample AS c
), c_context AS (
  SELECT
    *
    EXCLUDE (__p1)
  FROM __b2
  WHERE
    {{LLMMap('You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look more like a business customer likely to place larger operational orders?', __b2.__p1)}} = TRUE
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
Return a single YES or NO, with no other words. If uncertain, return NO.'
    ) AS __p2
  FROM ab_join AS ab, c_context AS c
)
SELECT
  *
  EXCLUDE (__p2)
FROM __b3
WHERE
  {{LLMMap('You are validating whether this customer profile is a business-relevant account for enterprise analysis.', __b3.__p2)}} = TRUE
