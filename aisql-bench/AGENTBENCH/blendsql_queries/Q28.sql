-- setup (DuckDB backend)
/* File: synth_tpch_q3.sql */ /* Label: Q28 */ /* Question: Which operationally significant orders with qualifying lineitem activity (1995-1996) pass customer validation for strategic fulfillment? */ /* tpch sf-0.005 from the benchmark's committed parquet files */ CREATE VIEW IF NOT EXISTS part AS SELECT * FROM "./dataset/tpch/part.parquet";
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM "./dataset/tpch/supplier.parquet";
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM "./dataset/tpch/customer.parquet";
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM "./dataset/tpch/lineitem.parquet";
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM "./dataset/tpch/orders.parquet";
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM "./dataset/tpch/partsupp.parquet";
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM "./dataset/tpch/nation.parquet";
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM "./dataset/tpch/region.parquet";
-- query (BlendSQL)
WITH orders_base AS (
  SELECT
    o.o_orderkey,
    o.o_orderpriority,
    o.o_totalprice,
    o.o_comment,
    o.o_orderdate,
    o.o_custkey
  FROM orders AS o
  WHERE
    o.o_orderdate >= CAST('1995-01-01' AS DATE)
    AND o.o_orderdate < CAST('1996-07-01' AS DATE)
    AND o.o_totalprice BETWEEN 100000 AND 220000
    AND o.o_orderstatus = 'O'
    AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
), __b1 AS (
  SELECT
    ob.o_orderkey AS ob__o_orderkey,
    ob.o_orderpriority AS ob__o_orderpriority,
    ob.o_totalprice AS ob__o_totalprice,
    ob.o_comment AS ob__o_comment,
    ob.o_orderdate AS ob__o_orderdate,
    ob.o_custkey AS ob__o_custkey,
    CONCAT(
      'Order Priority: ',
      CAST(ob.o_orderpriority AS TEXT),
      '
Total Price: ',
      CAST(ob.o_totalprice AS TEXT),
      '
Comment: ',
      CAST(ob.o_comment AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p0
  FROM orders_base AS ob
), order_focus AS (
  SELECT
    __b1.ob__o_orderkey AS o_orderkey,
    __b1.ob__o_orderpriority AS o_orderpriority,
    __b1.ob__o_totalprice AS o_totalprice,
    __b1.ob__o_comment AS o_comment,
    __b1.ob__o_orderdate AS o_orderdate,
    __b1.ob__o_custkey AS o_custkey
  FROM __b1
  WHERE
    {{LLMMap('Does this order look operationally significant based on priority label, total price, and comment content?', __b1.__p0)}} = TRUE
), lineitem_filtered AS (
  SELECT
    l.l_orderkey,
    l.l_partkey,
    l.l_suppkey,
    l.l_quantity,
    l.l_extendedprice,
    l.l_discount,
    l.l_shipmode,
    l.l_shipdate
  FROM lineitem AS l
  WHERE
    l.l_shipdate >= CAST('1995-01-01' AS DATE)
    AND l.l_shipdate < CAST('1996-07-01' AS DATE)
    AND l.l_quantity BETWEEN 18 AND 26
    AND l.l_discount BETWEEN 0.04 AND 0.07
    AND l.l_shipmode IN ('AIR', 'REG AIR')
), joined AS (
  SELECT
    of.o_orderkey,
    of.o_orderpriority,
    of.o_totalprice,
    of.o_comment,
    of.o_orderdate,
    of.o_custkey,
    lf.l_partkey,
    lf.l_suppkey,
    lf.l_quantity,
    lf.l_extendedprice,
    lf.l_discount,
    lf.l_shipmode
  FROM order_focus AS of
  JOIN lineitem_filtered AS lf
    ON lf.l_orderkey = of.o_orderkey
), __b2 AS (
  SELECT
    c.c_custkey AS c__c_custkey,
    c.c_name AS c__c_name,
    c.c_address AS c__c_address,
    c.c_nationkey AS c__c_nationkey,
    c.c_phone AS c__c_phone,
    c.c_acctbal AS c__c_acctbal,
    c.c_mktsegment AS c__c_mktsegment,
    c.c_comment AS c__c_comment,
    k.o_custkey AS k__o_custkey,
    o_custkey AS c__o_custkey,
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
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p1
  FROM customer AS c
  JOIN (
    SELECT DISTINCT
      o_custkey
    FROM joined
    LIMIT 60
  ) AS k
    ON k.o_custkey = c.c_custkey
), customer_context AS (
  SELECT
    __b2.c__c_custkey AS c_custkey,
    __b2.c__c_name AS c_name,
    __b2.c__c_address AS c_address,
    __b2.c__c_nationkey AS c_nationkey,
    __b2.c__c_phone AS c_phone,
    __b2.c__c_acctbal AS c_acctbal,
    __b2.c__c_mktsegment AS c_mktsegment,
    __b2.c__c_comment AS c_comment
  FROM __b2
  WHERE
    {{LLMMap('Does this customer look operationally strategic based on name style, market segment, and account balance?', __b2.__p1)}} = TRUE
), __b3 AS (
  SELECT
    *,
    CONCAT(
      'Order Priority: ',
      CAST(j.o_orderpriority AS TEXT),
      '
Order Total Price: ',
      CAST(j.o_totalprice AS TEXT),
      '
Lineitem Ship Mode: ',
      CAST(j.l_shipmode AS TEXT),
      '
Customer Name: ',
      CAST(cc.c_name AS TEXT),
      '
Market Segment: ',
      CAST(cc.c_mktsegment AS TEXT),
      '
Account Balance: ',
      CAST(cc.c_acctbal AS TEXT),
      '
Return a single YES or NO, with no other words. If uncertain, return YES.'
    ) AS __p2
  FROM joined AS j
  JOIN customer_context AS cc
    ON cc.c_custkey = j.o_custkey
)
SELECT
  *
  EXCLUDE (__p2)
FROM __b3
WHERE
  {{LLMMap('Could this customer account be strategically important for order fulfillment?', __b3.__p2)}} = TRUE
