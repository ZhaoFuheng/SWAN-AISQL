-- File: synth_tpch_q1.sql
-- Label: Q15
-- Question: Which lineitems in high-priority open orders (1995-1996) match air-shipment discount/quantity filters for semantically marketable parts, paired with semantically business-like customers?

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
LIMIT 100;

WITH
a_candidates AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_brand
	FROM part p
	WHERE ai_filter('You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a marketable finished good rather than an upstream industrial input?
Part Name: ' || p.p_name || '
Part Type: ' || p.p_type || '
Brand: ' || p.p_brand || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')
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
		l.l_shipmode,
		l.l_shipinstruct,
		o.o_orderdate,
		o.o_orderpriority,
		o.o_totalprice
	FROM lineitem l
	JOIN orders o
		ON o.o_orderkey = l.l_orderkey
	WHERE l.l_shipdate >= DATE '1995-01-01'
		AND l.l_shipdate < DATE '1997-01-01'
		AND l.l_discount BETWEEN 0.05 AND 0.07
		AND l.l_quantity BETWEEN 18 AND 22
		AND l.l_shipmode IN ('AIR', 'REG AIR')
		AND l.l_shipinstruct = 'DELIVER IN PERSON'
		AND o.o_orderstatus = 'O'
		AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
		AND o.o_totalprice > 100000
),

ab_join AS (
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
	FROM a_candidates a
	JOIN b_base b
		ON b.l_partkey = a.p_partkey
),
c_context AS (
	SELECT * FROM customer_sample AS c
	WHERE ai_filter('You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look more like a business customer likely to place larger operational orders?
Customer Name: ' || c.c_name || '
Market Segment: ' || c.c_mktsegment || '
Account Balance: ' || c.c_acctbal || '
Return a single YES or NO, with no other words. If uncertain, return NO.' || e'\n Return a single yes or no, do not contain any other words.')
)

SELECT *
FROM ab_join ab
CROSS JOIN c_context c;
