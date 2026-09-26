-- File: synth_tpch_q16.sql
-- Label: Q16
-- Question: Which air-shipment lineitems in high-priority 1995-1996 orders involve semantically market-ready parts and semantically enterprise-like customers?

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
	WHERE (lower(ai_complete('Task: classify whether this part should be included in customer-facing demand analysis.
Rule: YES only if the part sounds like a finished good that could appear in commercial product demand; NO for raw/industrial-looking components.
Part Name: ' || p.p_name || '
Part Type: ' || p.p_type || '
Brand: ' || p.p_brand || '
Return YES or NO only. If uncertain, return NO.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
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
	WHERE (lower(ai_complete('Task: decide if this is an enterprise-oriented customer profile for B2B order analysis.
Rule: YES only for profiles that look commercially active (business segment + meaningful balance); otherwise NO.
Customer Name: ' || c.c_name || '
Market Segment: ' || c.c_mktsegment || '
Account Balance: ' || c.c_acctbal || '
Return YES or NO only. If uncertain, return NO.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
)

SELECT *
FROM ab_join ab
JOIN c_context c
	ON (lower(ai_complete('You are validating whether this customer profile is a business-relevant account for enterprise analysis.
Customer Name: ' || c.c_name || '
Market Segment: ' || c.c_mktsegment || '
Account Balance: ' || c.c_acctbal || '
Return a single YES or NO, with no other words. If uncertain, return NO.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'));
