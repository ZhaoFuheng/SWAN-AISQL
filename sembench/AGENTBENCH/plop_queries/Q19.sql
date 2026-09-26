-- File: synth_tpch_q5.sql
-- Label: Q19
-- Question: Which potentially problematic returned lineitems (1994-1997), with related order/customer and part/supplier context, should be analyzed alongside semantically higher-risk customer profiles?

LOAD 'build/release/extension/tpch/tpch.duckdb_extension';
CALL dbgen(sf=0.005);

CREATE TEMP TABLE customer_sample AS
SELECT *
FROM customer
LIMIT 180;

WITH
lineitem_returns AS (
	SELECT
		l.l_orderkey,
		l.l_partkey,
		l.l_suppkey,
		l.l_quantity,
		l.l_extendedprice,
		l.l_discount,
		l.l_tax,
		l.l_returnflag,
		l.l_linestatus,
		l.l_shipmode,
		l.l_shipinstruct,
		l.l_comment
	FROM lineitem l
	WHERE l.l_shipdate >= DATE '1994-01-01'
		AND l.l_shipdate < DATE '1998-01-01'
		AND l.l_returnflag IN ('R', 'A', 'N')
		AND l.l_quantity BETWEEN 3 AND 38
		AND l.l_discount BETWEEN 0.00 AND 0.10
		AND SEMANTIC('Lineitem shipping details:\nMode: {l.l_shipmode}\nInstruction: {l.l_shipinstruct}\nComment: {l.l_comment}\nIs this likely a potentially problematic fulfillment case worth audit attention? Answer YES or NO. If unsure, answer YES.')
),
order_customer AS (
	SELECT
		o.o_orderkey,
		o.o_custkey,
		o.o_orderdate,
		o.o_orderpriority,
		o.o_orderstatus,
		o.o_totalprice,
		c.c_name,
		c.c_mktsegment,
		c.c_acctbal,
		c.c_comment
	FROM orders o
	JOIN customer c ON c.c_custkey = o.o_custkey
	WHERE o.o_orderdate >= DATE '1994-01-01'
		AND o.o_orderdate < DATE '1998-01-01'
		AND o.o_orderstatus IN ('O', 'F')
		AND o.o_totalprice > 20000
),
part_supplier AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_brand,
		s.s_suppkey,
		s.s_name,
		s.s_acctbal
	FROM part p
	JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
	JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
	WHERE p.p_size BETWEEN 1 AND 40
		AND ps.ps_supplycost BETWEEN 5 AND 1200
),
joined AS (
	SELECT
		lr.l_orderkey,
		lr.l_partkey,
		lr.l_suppkey,
		lr.l_quantity,
		lr.l_extendedprice,
		lr.l_discount,
		lr.l_tax,
		lr.l_returnflag,
		lr.l_linestatus,
		lr.l_shipmode,
		lr.l_shipinstruct,
		oc.o_orderdate,
		oc.o_orderpriority,
		oc.o_orderstatus,
		oc.o_totalprice,
		oc.c_name,
		oc.c_mktsegment,
		oc.c_acctbal,
		ps.p_name,
		ps.p_type,
		ps.p_brand,
		ps.s_name,
		ps.s_acctbal
	FROM lineitem_returns lr
	JOIN order_customer oc ON oc.o_orderkey = lr.l_orderkey
	JOIN part_supplier ps
		ON ps.p_partkey = lr.l_partkey
		AND ps.s_suppkey = lr.l_suppkey
),
customer_context AS (
	SELECT *
	FROM customer_sample c
	WHERE SEMANTIC('Customer profile check:\nName: {c.c_name}\nSegment: {c.c_mktsegment}\nBalance: {c.c_acctbal}\nComment: {c.c_comment}\nAnswer YES only if this customer appears likely to have higher complaint/escalation risk in commercial operations. Otherwise answer NO. If unsure, answer NO.')
)

SELECT *
FROM joined j
CROSS JOIN customer_context cc;
