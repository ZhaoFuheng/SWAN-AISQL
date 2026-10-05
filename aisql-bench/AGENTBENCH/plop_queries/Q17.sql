-- File: synth_tpch_q3.sql
-- Label: Q17
-- Question: Which important orders and lineitems (1993-1997) with matching part-supplier info are associated with semantically prioritized business customers?

LOAD 'build/release/extension/tpch/tpch.duckdb_extension';
CALL dbgen(sf=0.005);

CREATE TEMP TABLE customer_sample AS
SELECT *
FROM customer
LIMIT 120;

WITH
order_focus AS (
	SELECT
		o.o_orderkey,
		o.o_custkey,
		o.o_orderdate,
		o.o_orderpriority,
		o.o_totalprice,
		o.o_comment
	FROM orders o
	WHERE o.o_orderdate >= DATE '1993-01-01'
		AND o.o_orderdate < DATE '1997-01-01'
		AND o.o_orderstatus IN ('O', 'F')
		AND o.o_totalprice > 50000
		AND SEMANTIC('You are triaging enterprise orders for operations monitoring. From the priority label, total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager?
Order Priority: {o.o_orderpriority}
Total Price: {o.o_totalprice}
Comment: {o.o_comment}
Return a single YES or NO, with no other words. If uncertain, return YES.')
),
lineitem_focus AS (
	SELECT
		l.l_orderkey,
		l.l_partkey,
		l.l_suppkey,
		l.l_shipmode,
		l.l_shipinstruct,
		l.l_quantity,
		l.l_extendedprice,
		l.l_discount,
		l.l_tax,
		l.l_returnflag
	FROM lineitem l
	WHERE l.l_shipdate >= DATE '1993-01-01'
		AND l.l_shipdate < DATE '1998-01-01'
		AND l.l_quantity BETWEEN 5 AND 40
		AND l.l_discount BETWEEN 0.01 AND 0.10
		AND l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK')
),
part_supplier AS (
	SELECT
		p.p_partkey,
		p.p_name,
		p.p_type,
		p.p_brand,
		s.s_suppkey,
		s.s_name,
		n.n_name AS supplier_nation
	FROM part p
	JOIN partsupp ps ON ps.ps_partkey = p.p_partkey
	JOIN supplier s ON s.s_suppkey = ps.ps_suppkey
	JOIN nation n ON n.n_nationkey = s.s_nationkey
	WHERE p.p_size BETWEEN 5 AND 35
		AND ps.ps_supplycost BETWEEN 10 AND 1000
),
joined AS (
	SELECT
		of.o_orderkey,
		of.o_custkey,
		of.o_orderdate,
		of.o_orderpriority,
		of.o_totalprice,
		lf.l_partkey,
		lf.l_suppkey,
		lf.l_shipmode,
		lf.l_shipinstruct,
		lf.l_quantity,
		lf.l_extendedprice,
		lf.l_discount,
		lf.l_tax,
		lf.l_returnflag,
		ps.p_name,
		ps.p_type,
		ps.p_brand,
		ps.s_name,
		ps.supplier_nation
	FROM order_focus of
	JOIN lineitem_focus lf ON lf.l_orderkey = of.o_orderkey
	JOIN part_supplier ps
		ON ps.p_partkey = lf.l_partkey
		AND ps.s_suppkey = lf.l_suppkey
),
customer_context AS (
	SELECT *
	FROM customer_sample c
	WHERE SEMANTIC('You are identifying high-value B2B customer profiles. From the customer name format, market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics?
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Comment: {c.c_comment}
Return a single YES or NO, with no other words. If uncertain, return NO.')
)

SELECT *
FROM joined j
JOIN customer_context c
	ON SEMANTIC('You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up. The customer has the following profile:
Customer Name: {c.c_name}
Market Segment: {c.c_mktsegment}
Account Balance: {c.c_acctbal}
Comment: {c.c_comment})');
	
