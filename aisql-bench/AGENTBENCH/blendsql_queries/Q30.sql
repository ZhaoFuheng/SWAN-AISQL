-- setup (DuckDB backend)
/* File: synth_tpch_q5.sql */ /* Label: Q30 */ /* Question: Which parts suitable for distribution with qualifying lineitem/order/customer activity (retail focus) pass supplier validation for vendor partnerships? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
-- query (BlendSQL)
WITH orders_lineitem_customer AS (
  SELECT
    l.l_orderkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_shipmode,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_totalprice,
    c.c_custkey,
    c.c_name,
    c.c_mktsegment,
    c.c_acctbal
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  JOIN customer AS c
    ON c.c_custkey = o.o_custkey
  WHERE
    l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
    AND l.l_quantity BETWEEN 12 AND 30
    AND l.l_discount BETWEEN 0.04 AND 0.07
    AND o.o_orderdate >= CAST('1996-01-01' AS DATE)
    AND o.o_orderdate < CAST('1998-01-01' AS DATE)
    AND o.o_totalprice BETWEEN 60000 AND 170000
    AND c.c_mktsegment IN ('FURNITURE', 'HOUSEHOLD', 'MACHINERY', 'BUILDING')
), part_base AS (
  SELECT
    p.p_partkey,
    p.p_name,
    p.p_type,
    p.p_container,
    p.p_size
  FROM part AS p
  JOIN (
    SELECT DISTINCT
      l_partkey
    FROM orders_lineitem_customer
    LIMIT 200
  ) AS k
    ON k.l_partkey = p.p_partkey
  WHERE
    p.p_size BETWEEN 5 AND 28
    AND p.p_container IN ('SM BOX', 'MED BOX', 'LG BOX', 'SM CASE')
), __b1 AS (
  SELECT
    pb.p_partkey AS pb__p_partkey,
    pb.p_name AS pb__p_name,
    pb.p_type AS pb__p_type,
    pb.p_container AS pb__p_container,
    pb.p_size AS pb__p_size,
    CONCAT(
      'Part Name: ',
      CAST(pb.p_name AS TEXT),
      '
Part Type: ',
      CAST(pb.p_type AS TEXT),
      '
Container: ',
      CAST(pb.p_container AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM part_base AS pb
), part_focus AS (
  SELECT
    __b1.pb__p_partkey AS p_partkey,
    __b1.pb__p_name AS p_name,
    __b1.pb__p_type AS p_type,
    __b1.pb__p_container AS p_container,
    __b1.pb__p_size AS p_size
  FROM __b1
  WHERE
    {{LLMMap('Does this part look suitable for customer-facing distribution based on name, type, and packaging?', __b1.__p0)}} = TRUE
), joined AS (
  SELECT
    pf.p_partkey,
    pf.p_name,
    pf.p_type,
    pf.p_container,
    olc.l_orderkey,
    olc.l_suppkey,
    olc.l_quantity,
    olc.l_extendedprice,
    olc.l_discount,
    olc.l_shipmode,
    olc.o_orderdate,
    olc.o_orderpriority,
    olc.o_totalprice,
    olc.c_custkey,
    olc.c_name,
    olc.c_mktsegment,
    olc.c_acctbal
  FROM part_focus AS pf
  JOIN orders_lineitem_customer AS olc
    ON olc.l_partkey = pf.p_partkey
), __b2 AS (
  SELECT
    s.s_suppkey AS s__s_suppkey,
    s.s_name AS s__s_name,
    s.s_address AS s__s_address,
    s.s_nationkey AS s__s_nationkey,
    s.s_phone AS s__s_phone,
    s.s_acctbal AS s__s_acctbal,
    s.s_comment AS s__s_comment,
    k.l_suppkey AS k__l_suppkey,
    l_suppkey AS c__l_suppkey,
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
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p1
  FROM supplier AS s
  JOIN (
    SELECT DISTINCT
      l_suppkey
    FROM joined
    LIMIT 40
  ) AS k
    ON k.l_suppkey = s.s_suppkey
), supplier_context AS (
  SELECT
    __b2.s__s_suppkey AS s_suppkey,
    __b2.s__s_name AS s_name,
    __b2.s__s_address AS s_address,
    __b2.s__s_nationkey AS s_nationkey,
    __b2.s__s_phone AS s_phone,
    __b2.s__s_acctbal AS s_acctbal,
    __b2.s__s_comment AS s_comment
  FROM __b2
  WHERE
    {{LLMMap('Does this supplier appear qualified for consumer-facing distribution partnerships?', __b2.__p1)}} = TRUE
), __b3 AS (
  SELECT
    *,
    CONCAT(
      'Part Type: ',
      CAST(j.p_type AS TEXT),
      '
Ship Mode: ',
      CAST(j.l_shipmode AS TEXT),
      '
Customer Segment: ',
      CAST(j.c_mktsegment AS TEXT),
      '
Supplier Name: ',
      CAST(sc.s_name AS TEXT),
      '
Account Balance: ',
      CAST(sc.s_acctbal AS TEXT),
      '
Comment: ',
      CAST(sc.s_comment AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p2
  FROM joined AS j
  JOIN supplier_context AS sc
    ON sc.s_suppkey = j.l_suppkey
)
SELECT
  *
  EXCLUDE (__p2)
FROM __b3
WHERE
  {{LLMMap('Would this supplier be suitable for retail distribution of this product?', __b3.__p2)}} = TRUE
