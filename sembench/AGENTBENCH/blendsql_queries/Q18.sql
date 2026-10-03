-- setup (DuckDB backend)
/* File: synth_tpch_q4.sql */ /* Label: Q18 */ /* Question: Which distribution-oriented parts and shipment/order records (1994-1997) joined with customer geography should be analyzed together with semantically relevant nations for cross-border logistics? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
CREATE TEMPORARY TABLE nation_sample AS SELECT * FROM nation LIMIT 25;
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    p.p_partkey AS p__p_partkey,
    p.p_name AS p__p_name,
    p.p_type AS p__p_type,
    p.p_container AS p__p_container,
    p.p_size AS p__p_size,
    CONCAT(
      'Part Name: ',
      CAST(p.p_name AS TEXT),
      '
Part Type: ',
      CAST(p.p_type AS TEXT),
      '
Container: ',
      CAST(p.p_container AS TEXT),
      '
Size: ',
      CAST(p.p_size AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM part AS p
  WHERE
    p.p_size BETWEEN 3 AND 30
    AND p.p_container IN ('SM BOX', 'SM CASE', 'MED BOX', 'MED BAG', 'LG BOX')
), part_candidates AS (
  SELECT
    __b1.p__p_partkey AS p_partkey,
    __b1.p__p_name AS p_name,
    __b1.p__p_type AS p_type,
    __b1.p__p_container AS p_container,
    __b1.p__p_size AS p_size
  FROM __b1
  WHERE
    {{LLMMap('You are preparing a logistics performance dashboard for retail distribution. Based on product name, type, packaging container, and size, does this part look like something that would realistically move through customer-facing distribution channels?', __b1.__p0)}} = TRUE
), lineitem_orders AS (
  SELECT
    l.l_orderkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_shipmode,
    l.l_shipinstruct,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    o.o_custkey,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_totalprice
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  WHERE
    l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK', 'MAIL')
    AND l.l_shipinstruct IN ('DELIVER IN PERSON', 'TAKE BACK RETURN')
    AND l.l_quantity BETWEEN 10 AND 45
    AND o.o_orderdate >= CAST('1994-01-01' AS DATE)
    AND o.o_orderdate < CAST('1998-01-01' AS DATE)
    AND o.o_totalprice > 30000
), customer_geo AS (
  SELECT
    c.c_custkey,
    c.c_name,
    c.c_nationkey,
    c.c_mktsegment,
    c.c_acctbal,
    n.n_name AS customer_nation,
    r.r_name AS customer_region
  FROM customer AS c
  JOIN nation AS n
    ON n.n_nationkey = c.c_nationkey
  JOIN region AS r
    ON r.r_regionkey = n.n_regionkey
  WHERE
    r.r_name IN ('AMERICA', 'EUROPE', 'ASIA')
), joined AS (
  SELECT
    pc.p_partkey,
    pc.p_name,
    pc.p_type,
    pc.p_container,
    lo.l_orderkey,
    lo.l_shipmode,
    lo.l_shipinstruct,
    lo.l_quantity,
    lo.l_extendedprice,
    lo.l_discount,
    lo.o_orderdate,
    lo.o_orderpriority,
    lo.o_totalprice,
    cg.c_name,
    cg.c_mktsegment,
    cg.c_acctbal,
    cg.customer_nation,
    cg.customer_region
  FROM part_candidates AS pc
  JOIN lineitem_orders AS lo
    ON lo.l_partkey = pc.p_partkey
  JOIN customer_geo AS cg
    ON cg.c_custkey = lo.o_custkey
), __b2 AS (
  SELECT
    *,
    CONCAT(
      'Nation: ',
      CAST(n.n_name AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p1
  FROM nation_sample AS n
), nation_context AS (
  SELECT
    *
    EXCLUDE (__p1)
  FROM __b2
  WHERE
    {{LLMMap('You are selecting countries to include in a cross-border freight and trade analysis view. Is this nation likely meaningful enough to keep in a global logistics discussion dataset?', __b2.__p1)}} = TRUE
)
SELECT
  *
FROM joined AS j
CROSS JOIN nation_context AS nc
