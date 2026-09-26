-- File: synth_tpch_q2_ext.sql
-- Label: Q27
-- Question: Which semantically differentiated parts with qualifying supply and order activity (1994-1996) pass supplier validation for preferred-vendor sourcing?

-- tpch sf-0.005 from the benchmark's committed parquet files
CREATE VIEW IF NOT EXISTS part     AS SELECT * FROM './dataset/tpch/part.parquet';
CREATE VIEW IF NOT EXISTS supplier AS SELECT * FROM './dataset/tpch/supplier.parquet';
CREATE VIEW IF NOT EXISTS customer AS SELECT * FROM './dataset/tpch/customer.parquet';
CREATE VIEW IF NOT EXISTS lineitem AS SELECT * FROM './dataset/tpch/lineitem.parquet';
CREATE VIEW IF NOT EXISTS orders   AS SELECT * FROM './dataset/tpch/orders.parquet';
CREATE VIEW IF NOT EXISTS partsupp AS SELECT * FROM './dataset/tpch/partsupp.parquet';
CREATE VIEW IF NOT EXISTS nation   AS SELECT * FROM './dataset/tpch/nation.parquet';
CREATE VIEW IF NOT EXISTS region   AS SELECT * FROM './dataset/tpch/region.parquet';

WITH
part_focus AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_brand,
		p.p_size
	FROM part p
	WHERE (lower(ai_complete('You are screening parts for strategic sourcing analysis. Looking at part name, type, brand, and size, does this look like a differentiated component where supplier quality and pricing strategy would matter?
Part Name: ' || p.p_name || '
Part Type: ' || p.p_type || '
Brand: ' || p.p_brand || '
Size: ' || p.p_size || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
supply_base AS (
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
	FROM partsupp ps
	JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
	JOIN nation n ON n.n_nationkey = s.s_nationkey
	JOIN region r ON r.r_regionkey = n.n_regionkey
	WHERE ps.ps_availqty BETWEEN 10 AND 600
		AND ps.ps_supplycost BETWEEN 20 AND 900
		AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA')
),
orders_lineitem AS (
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
	FROM lineitem l
	JOIN orders o ON o.o_orderkey = l.l_orderkey
	WHERE o.o_orderdate >= DATE '1994-01-01'
		AND o.o_orderdate < DATE '1997-01-01'
		AND l.l_quantity BETWEEN 8 AND 35
		AND l.l_discount BETWEEN 0.00 AND 0.08
),
joined AS (
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
	FROM part_focus pf
	JOIN supply_base sb
		ON sb.ps_partkey = pf.p_partkey
	JOIN orders_lineitem ol
		ON ol.l_partkey = sb.ps_partkey
		AND ol.l_suppkey = sb.ps_suppkey
),
supplier_context AS (
	SELECT
		s.s_suppkey,
		s.s_name,
		s.s_address,
		s.s_nationkey,
		s.s_phone,
		s.s_acctbal,
		s.s_comment
	FROM supplier s
	JOIN (
		SELECT DISTINCT ps_suppkey
		FROM joined
		LIMIT 80
	) k ON k.ps_suppkey = s.s_suppkey
	WHERE (lower(ai_complete('You are shortlisting suppliers for long-term procurement partnerships. Based on supplier identity style, account balance, and profile comment tone, does this supplier appear commercially dependable enough for preferred-vendor consideration?
Supplier Name: ' || s.s_name || '
Account Balance: ' || s.s_acctbal || '
Comment: ' || s.s_comment || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
)

SELECT *
FROM joined j
JOIN supplier_context sc
	ON sc.s_suppkey = j.ps_suppkey
	AND (lower(ai_complete('You are validating whether this supplier profile is suitable for preferred-vendor sourcing.
Part Type: ' || j.p_type || '
Region: ' || j.region_name || '
Supplier Name: ' || sc.s_name || '
Account Balance: ' || sc.s_acctbal || '
Comment: ' || sc.s_comment || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'));
