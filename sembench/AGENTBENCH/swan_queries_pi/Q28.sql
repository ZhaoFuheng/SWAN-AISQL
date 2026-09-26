-- File: synth_tpch_q3.sql
-- Label: Q28
-- Question: Which operationally significant orders with qualifying lineitem activity (1995-1996) pass customer validation for strategic fulfillment?

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
orders_base AS (
	SELECT
		o.o_orderkey,
		o.o_orderpriority,
		o.o_totalprice,
		o.o_comment,
		o.o_orderdate,
		o.o_custkey
	FROM orders o
	WHERE o.o_orderdate >= DATE '1995-01-01'
		AND o.o_orderdate < DATE '1996-07-01'
		AND o.o_totalprice BETWEEN 100000 AND 220000
		AND o.o_orderstatus = 'O'
		AND o.o_orderpriority IN ('1-URGENT', '2-HIGH')
),
order_focus AS (
	SELECT
		ob.o_orderkey,
		ob.o_orderpriority,
		ob.o_totalprice,
		ob.o_comment,
		ob.o_orderdate,
		ob.o_custkey
	FROM orders_base ob
	WHERE (lower(ai_complete('Does this order look operationally significant based on priority label, total price, and comment content?
Order Priority: ' || ob.o_orderpriority || '
Total Price: ' || ob.o_totalprice || '
Comment: ' || ob.o_comment || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
),
lineitem_filtered AS (
	SELECT
		l.l_orderkey,
		l.l_partkey,
		l.l_suppkey,
		l.l_quantity,
		l.l_extendedprice,
		l.l_discount,
		l.l_shipmode,
		l.l_shipdate
	FROM lineitem l
	WHERE l.l_shipdate >= DATE '1995-01-01'
		AND l.l_shipdate < DATE '1996-07-01'
		AND l.l_quantity BETWEEN 18 AND 26
		AND l.l_discount BETWEEN 0.04 AND 0.07
		AND l.l_shipmode IN ('AIR', 'REG AIR')
),
joined AS (
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
	FROM order_focus of
	JOIN lineitem_filtered lf
		ON lf.l_orderkey = of.o_orderkey
),
customer_context AS (
	SELECT
		c.c_custkey,
		c.c_name,
		c.c_address,
		c.c_nationkey,
		c.c_phone,
		c.c_acctbal,
		c.c_mktsegment,
		c.c_comment
	FROM customer c
	JOIN (
		SELECT DISTINCT o_custkey
		FROM joined
		LIMIT 60
	) k ON k.o_custkey = c.c_custkey
	WHERE (lower(ai_complete('Does this customer look operationally strategic based on name style, market segment, and account balance?
Customer Name: ' || c.c_name || '
Market Segment: ' || c.c_mktsegment || '
Account Balance: ' || c.c_acctbal || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'))
)

SELECT *
FROM joined j
JOIN customer_context cc
	ON cc.c_custkey = j.o_custkey
	AND (lower(ai_complete('Could this customer account be strategically important for order fulfillment?
Order Priority: ' || j.o_orderpriority || '
Order Total Price: ' || j.o_totalprice || '
Lineitem Ship Mode: ' || j.l_shipmode || '
Customer Name: ' || cc.c_name || '
Market Segment: ' || cc.c_mktsegment || '
Account Balance: ' || cc.c_acctbal || '
Return a single YES or NO, with no other words. If uncertain, return YES.' || e'\n Return a single yes or no, do not contain any other words.')) IN ('yes', 'true'));
