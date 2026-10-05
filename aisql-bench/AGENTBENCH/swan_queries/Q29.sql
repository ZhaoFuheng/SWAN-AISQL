-- File: synth_tpch_q4.sql
-- Label: Q29
-- Question: Which reliable suppliers with qualifying partsupp activity (availability/cost) pass regional validation for global sourcing strategy?

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
partsupp_base AS (
	SELECT
		ps.ps_partkey,
		ps.ps_suppkey,
		ps.ps_availqty,
		ps.ps_supplycost
	FROM partsupp ps
	WHERE ps.ps_availqty BETWEEN 1600 AND 3200
		AND ps.ps_supplycost BETWEEN 200 AND 420
),
supplier_base AS (
	SELECT
		s.s_suppkey,
		s.s_name,
		s.s_address,
		s.s_acctbal,
		s.s_comment,
		s.s_nationkey,
		n.n_name AS supplier_nation_name,
		r.r_name AS supplier_region_name
	FROM supplier s
	JOIN nation n ON n.n_nationkey = s.s_nationkey
	JOIN region r ON r.r_regionkey = n.n_regionkey
	JOIN (
		SELECT DISTINCT ps_suppkey
		FROM partsupp_base
		LIMIT 70
	) k ON k.ps_suppkey = s.s_suppkey
	WHERE s.s_acctbal BETWEEN 2000 AND 8000
		AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA')
),
joined AS (
	SELECT
		sb.s_suppkey,
		sb.s_name,
		sb.s_address,
		sb.s_acctbal,
		sb.s_comment,
		sb.s_nationkey,
		sb.supplier_nation_name,
		sb.supplier_region_name,
		pb.ps_partkey,
		pb.ps_availqty,
		pb.ps_supplycost
	FROM supplier_base sb
	JOIN partsupp_base pb
		ON pb.ps_suppkey = sb.s_suppkey
),
semantic_candidates AS (
	SELECT
		j.s_suppkey,
		j.s_nationkey,
		MIN(j.s_acctbal) AS s_acctbal,
		MIN(j.ps_supplycost) AS ps_supplycost,
		MIN(j.supplier_region_name) AS supplier_region_name
	FROM joined j
	GROUP BY
		j.s_suppkey,
		j.s_nationkey
	LIMIT 50
),
nation_context AS (
	SELECT
		ns.n_nationkey,
		ns.n_name,
		ns.n_regionkey,
		ns.n_comment,
		r.r_regionkey,
		r.r_name,
		r.r_comment
	FROM nation ns
	JOIN region r ON r.r_regionkey = ns.n_regionkey
	JOIN (
		SELECT DISTINCT s_nationkey
		FROM semantic_candidates
		ORDER BY s_nationkey
		LIMIT 5
	) k ON k.s_nationkey = ns.n_nationkey
)

SELECT
	j.s_suppkey,
	j.s_name,
	j.s_address,
	j.s_acctbal,
	j.s_comment,
	j.s_nationkey,
	j.supplier_nation_name,
	j.supplier_region_name,
	j.ps_partkey,
	j.ps_availqty,
	j.ps_supplycost,
	nc.n_nationkey AS context_nationkey,
	nc.n_name AS context_nation_name,
	nc.n_regionkey AS context_regionkey,
	nc.r_regionkey AS context_region_id,
	nc.r_name AS context_region_name,
	nc.n_comment AS context_nation_comment,
	nc.r_comment AS context_region_comment
FROM joined j
JOIN semantic_candidates sc
	ON sc.s_suppkey = j.s_suppkey
	AND sc.s_nationkey = j.s_nationkey
JOIN nation_context nc
	ON nc.n_nationkey = sc.s_nationkey
	AND ai_filter('Does this sourcing region align with procurement priorities?
Supplier Account Balance: ' || sc.s_acctbal || '
Supply Cost: ' || sc.ps_supplycost || '
Supplier Region: ' || sc.supplier_region_name || '
Nation Name: ' || nc.n_name || '
Region Name: ' || nc.r_name || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.');
