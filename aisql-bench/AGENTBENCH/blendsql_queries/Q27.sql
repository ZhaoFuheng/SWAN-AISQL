-- setup (DuckDB backend)
/* File: synth_tpch_q2_ext.sql */ /* Label: Q27 */ /* Question: Which semantically differentiated parts with qualifying supply and order activity (1994-1996) pass supplier validation for preferred-vendor sourcing? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
-- query (BlendSQL)
WITH __b1 AS (
  SELECT
    p.p_partkey AS p__p_partkey,
    p.p_name AS p__p_name,
    p.p_type AS p__p_type,
    p.p_brand AS p__p_brand,
    p.p_size AS p__p_size,
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
Size: ',
      CAST(p.p_size AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM part AS p
), part_focus AS (
  SELECT
    __b1.p__p_partkey AS p_partkey,
    __b1.p__p_name AS p_name,
    __b1.p__p_type AS p_type,
    __b1.p__p_brand AS p_brand,
    __b1.p__p_size AS p_size
  FROM __b1
  WHERE
    {{LLMMap('You are screening parts for strategic sourcing analysis. Looking at part name, type, brand, and size, does this look like a differentiated component where supplier quality and pricing strategy would matter?', __b1.__p0)}} = TRUE
), supply_base AS (
  SELECT
    ps.ps_partkey,
    ps.ps_suppkey,
    ps.ps_availqty,
    ps.ps_supplycost,
    s.s_name,
    s.s_acctbal,
    s.s_comment,
    s.s_nationkey,
    n.n_name AS nation_name,
    r.r_name AS region_name
  FROM partsupp AS ps
  JOIN supplier AS s
    ON s.s_suppkey = ps.ps_suppkey
  JOIN nation AS n
    ON n.n_nationkey = s.s_nationkey
  JOIN region AS r
    ON r.r_regionkey = n.n_regionkey
  WHERE
    ps.ps_availqty BETWEEN 10 AND 600
    AND ps.ps_supplycost BETWEEN 20 AND 900
    AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA')
), orders_lineitem AS (
  SELECT
    l.l_partkey,
    l.l_suppkey,
    l.l_orderkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    o.o_orderdate,
    o.o_orderpriority,
    o.o_totalprice
  FROM lineitem AS l
  JOIN orders AS o
    ON o.o_orderkey = l.l_orderkey
  WHERE
    o.o_orderdate >= CAST('1994-01-01' AS DATE)
    AND o.o_orderdate < CAST('1997-01-01' AS DATE)
    AND l.l_quantity BETWEEN 8 AND 35
    AND l.l_discount BETWEEN 0.00 AND 0.08
), joined AS (
  SELECT
    pf.p_partkey,
    pf.p_name,
    pf.p_type,
    pf.p_brand,
    sb.ps_suppkey,
    sb.s_name,
    sb.s_acctbal,
    sb.s_comment,
    sb.nation_name,
    sb.region_name,
    sb.ps_availqty,
    sb.ps_supplycost,
    ol.l_orderkey,
    ol.l_quantity,
    ol.l_extendedprice,
    ol.o_orderdate,
    ol.o_orderpriority,
    ol.o_totalprice
  FROM part_focus AS pf
  JOIN supply_base AS sb
    ON sb.ps_partkey = pf.p_partkey
  JOIN orders_lineitem AS ol
    ON ol.l_partkey = sb.ps_partkey AND ol.l_suppkey = sb.ps_suppkey
), __b2 AS (
  SELECT
    s.s_suppkey AS s__s_suppkey,
    s.s_name AS s__s_name,
    s.s_address AS s__s_address,
    s.s_nationkey AS s__s_nationkey,
    s.s_phone AS s__s_phone,
    s.s_acctbal AS s__s_acctbal,
    s.s_comment AS s__s_comment,
    k.ps_suppkey AS k__ps_suppkey,
    ps_suppkey AS c__ps_suppkey,
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
      ps_suppkey
    FROM joined
    LIMIT 80
  ) AS k
    ON k.ps_suppkey = s.s_suppkey
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
    {{LLMMap('You are shortlisting suppliers for long-term procurement partnerships. Based on supplier identity style, account balance, and profile comment tone, does this supplier appear commercially dependable enough for preferred-vendor consideration?', __b2.__p1)}} = TRUE
), __b3 AS (
  SELECT
    *,
    CONCAT(
      'Part Type: ',
      CAST(j.p_type AS TEXT),
      '
Region: ',
      CAST(j.region_name AS TEXT),
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
    ON sc.s_suppkey = j.ps_suppkey
)
SELECT
  *
  EXCLUDE (__p2)
FROM __b3
WHERE
  {{LLMMap('You are validating whether this supplier profile is suitable for preferred-vendor sourcing.', __b3.__p2)}} = TRUE
