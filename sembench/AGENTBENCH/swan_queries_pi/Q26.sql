-- File: synth_tpch_q26_multi_join.sql
-- Label: Q26
-- Question: Which air-shipment lineitems in high-priority open orders (1995-1996) map to semantically marketable parts, dependable suppliers, and business-like customers across supplier geography?

-- tpch sf-0.005 from the benchmark's committed parquet files
CREATE VIEW IF NOT EXISTS part     AS SELECT * FROM './dataset/tpch/part.parquet';
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM './dataset/tpch/supplier.parquet';
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM './dataset/tpch/customer.parquet';
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM './dataset/tpch/lineitem.parquet';
CREATE VIEW IF NOT EXISTS orders   AS SELECT * FROM './dataset/tpch/orders.parquet';
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM './dataset/tpch/partsupp.parquet';
CREATE VIEW IF NOT EXISTS nation   AS SELECT * FROM './dataset/tpch/nation.parquet';
CREATE VIEW IF NOT EXISTS region   AS SELECT * FROM './dataset/tpch/region.parquet';

CREATE TEMP TABLE customer_sample AS
SELECT *
FROM customer
WHERE c_acctbal > 500
  AND c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY');

WITH
a_candidates AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_brand
	FROM part p
	WHERE (lower(ai_complete('You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: ' || p.p_name || '
Part Type: ' || p.p_type || '
Brand: ' || p.p_brand || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
s_candidates AS (
	SELECT
		s.s_suppkey,
		s.s_name,
		s.s_acctbal,
		s.s_comment,
		n.n_name AS supplier_nation,
		r.r_name AS supplier_region
	FROM supplier s
	JOIN nation n ON n.n_nationkey = s.s_nationkey
	JOIN region r ON r.r_regionkey = n.n_regionkey
	WHERE s.s_acctbal > 0
	  AND (lower(ai_complete('You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look commercially dependable?
Supplier Name: ' || s.s_name || '
Account Balance: ' || s.s_acctbal || '
Comment: ' || s.s_comment || '
Nation: ' || n.n_name || '
Region: ' || r.r_name || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
b_base AS (
	SELECT
		l.l_orderkey,
		l.l_linenumber,
		l.l_partkey,
		l.l_suppkey,
		l.l_quantity,
		l.l_extendedprice,
		l.l_discount,
		l.l_tax,
		l.l_shipmode,
		l.l_shipinstruct
	FROM lineitem l
	JOIN orders o ON o.o_orderkey = l.l_orderkey
	WHERE l.l_shipdate >= DATE '1995-01-01'
	  AND l.l_shipdate < DATE '1997-01-01'
	  AND l.l_discount BETWEEN 0.05 AND 0.08
	  AND l.l_quantity BETWEEN 16 AND 24
	  AND l.l_shipmode IN ('AIR', 'REG AIR')
	  AND l.l_shipinstruct = 'DELIVER IN PERSON'
	  AND o.o_orderstatus = 'O'
	  AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
	  AND o.o_totalprice > 90000
	  AND (lower(ai_complete('You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally important?
Order Priority: ' || o.o_orderpriority || '
Total Price: ' || o.o_totalprice || '
Order Comment: ' || o.o_comment || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
ps_bridge AS (
	SELECT
		ps.ps_partkey,
		ps.ps_suppkey,
		ps.ps_supplycost,
		ps.ps_availqty
	FROM partsupp ps
	JOIN a_candidates a ON a.p_partkey = ps.ps_partkey
	JOIN s_candidates s ON s.s_suppkey = ps.ps_suppkey
	WHERE ps.ps_supplycost BETWEEN 80 AND 900
	  AND ps.ps_availqty >= 50
),
c_context AS (
	SELECT *
	FROM customer_sample c
	WHERE (lower(ai_complete('You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer look business-like and operationally relevant?
Customer Name: ' || c.c_name || '
Market Segment: ' || c.c_mktsegment || '
Account Balance: ' || c.c_acctbal || '
Comment: ' || c.c_comment || '
Return a single YES or NO, with no other words. If uncertain, return NO.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
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
FROM b_base b
JOIN ps_bridge ps
  ON ps.ps_partkey = b.l_partkey
 AND ps.ps_suppkey = b.l_suppkey
JOIN a_candidates a ON a.p_partkey = b.l_partkey
JOIN s_candidates s ON s.s_suppkey = b.l_suppkey
CROSS JOIN c_context c;
