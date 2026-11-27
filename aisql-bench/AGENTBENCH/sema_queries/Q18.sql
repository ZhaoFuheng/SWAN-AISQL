-- File: synth_tpch_q4.sql
-- Label: Q18
-- Question: Which distribution-oriented parts and shipment/order records (1994-1997) joined with customer geography should be analyzed together with semantically relevant nations for cross-border logistics?

-- tpch sf-0.005: the benchmark's parquet files, written into a Sema database by sema_agentbench.py
ATTACH IF NOT EXISTS './dataset/tpch_sema.duckdb' AS tpch_sema (READ_ONLY);
CREATE VIEW IF NOT EXISTS part AS SELECT * FROM tpch_sema.part;
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM tpch_sema.supplier;
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM tpch_sema.customer;
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM tpch_sema.lineitem;
CREATE VIEW IF NOT EXISTS orders AS SELECT * FROM tpch_sema.orders;
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM tpch_sema.partsupp;
CREATE VIEW IF NOT EXISTS nation AS SELECT * FROM tpch_sema.nation;
CREATE VIEW IF NOT EXISTS region AS SELECT * FROM tpch_sema.region;

CREATE TEMP TABLE nation_sample AS
SELECT *
FROM nation
ORDER BY n_nationkey
LIMIT 25;

WITH
part_candidates AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_container,
		p.p_size
	FROM part p
	WHERE p.p_size BETWEEN 3 AND 30
		AND p.p_container IN ('SM BOX', 'SM CASE', 'MED BOX', 'MED BAG', 'LG BOX')
		AND s'You are preparing a logistics performance dashboard for retail distribution. Based on product name, type, packaging container, and size, does this part look like something that would realistically move through customer-facing distribution channels?
Part Name: {p.p_name}
Part Type: {p.p_type}
Container: {p.p_container}
Size: {p.p_size}
Return a single YES or NO, with no other words. If uncertain, return YES.'
),
lineitem_orders AS (
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
	FROM lineitem l
	JOIN orders o
		ON o.o_orderkey = l.l_orderkey
	WHERE l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK', 'MAIL')
		AND l.l_shipinstruct IN ('DELIVER IN PERSON', 'TAKE BACK RETURN')
		AND l.l_quantity BETWEEN 10 AND 45
		AND o.o_orderdate >= DATE '1994-01-01'
		AND o.o_orderdate < DATE '1998-01-01'
		AND o.o_totalprice > 30000
),
customer_geo AS (
	SELECT
		c.c_custkey,
		c.c_name,
		c.c_nationkey,
		c.c_mktsegment,
		c.c_acctbal,
		n.n_name AS customer_nation,
		r.r_name AS customer_region
	FROM customer c
	JOIN nation n ON n.n_nationkey = c.c_nationkey
	JOIN region r ON r.r_regionkey = n.n_regionkey
	WHERE r.r_name IN ('AMERICA', 'EUROPE', 'ASIA')
),
joined AS (
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
	FROM part_candidates pc
	JOIN lineitem_orders lo ON lo.l_partkey = pc.p_partkey
	JOIN customer_geo cg ON cg.c_custkey = lo.o_custkey
),
nation_context AS (
	SELECT *
	FROM nation_sample n
	WHERE s'You are selecting countries to include in a cross-border freight and trade analysis view. Is this nation likely meaningful enough to keep in a global logistics discussion dataset?
Nation: {n.n_name}
Return a single YES or NO, with no other words. If uncertain, return YES.'
)

SELECT *
FROM joined j
CROSS JOIN nation_context nc;
