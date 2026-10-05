#!/usr/bin/env python3
"""Writes thalamusdb_queries/QN.sql: our ThalamusDB formulation of the hybrid bench's 30 queries (the PLOP
authors wrote none, unlike SemBench). The rule, the one SemBench itself used for its ThalamusDB queries:

* ThalamusDB's `NLfilter(table.column, condition)` takes ONE column of a base table, so every semantic
  predicate becomes a filter over a materialised `item` column that carries the SWAN prompt's labelled
  fields verbatim (`Title: ...\\nSubtitle: ...`), with NULL fields rendered empty as the other systems
  render them; the condition is the prompt's instruction, without the answer-format line (ThalamusDB adds
  its own). The table that carries the column keeps the query's plain SQL predicates on it, which is what
  ThalamusDB prunes by.
* A predicate over two tables (the `JOIN ... ON ai_filter(...)` of Q27-Q30) is a filter over the
  equi-joined rows, so the join is materialised with a combined item column and filtered: the same work the
  other systems do per joined row, where ThalamusDB's `NLjoin` would judge every pair of the two inputs.
* ThalamusDB knows no CTEs; a CTE that only reshapes data is a table built in the setup part, and a CTE whose
  rows a later LIMIT subquery depends on becomes a stage (`CREATE TABLE ... AS <query with NLfilter>`, run
  through ThalamusDB by thalamusdb_exec.py before the rest), as does each single-side predicate feeding a
  cross product (Q17, Q19, Q26), so the product is formed over survivors only.
* Q1-Q3 classify each row into one of k categories with `ai_complete`. ThalamusDB has no projection
  operator, but it accepts `NLfilter` inside a `CASE WHEN` in the SELECT list (SemBench's own MOVIE q8 does
  this), so the classification becomes a cascade of boolean predicates. Each condition names the full option
  set and asks for one class; the specific classes are asked first and the broadest class is the `ELSE`
  (a generic test asked first would swallow the specific ones: "Non-Fiction" matches every biography). The
  runner then keeps the first k rows of what
  ThalamusDB returns (`--apply-limit`), since ThalamusDB strips a LIMIT and returns every certain row.

Data loading follows swan_queries/QN.sql (same files, same filters, same LIMIT samples).
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thalamusdb_queries")

TPCH = "".join(f"CREATE TABLE {t} AS SELECT * FROM read_parquet('./dataset/tpch/{t}.parquet');\n"
               for t in ("part", "supplier", "customer", "lineitem", "orders", "partsupp", "nation", "region"))
BOOKS = ("CREATE TABLE books_info AS SELECT * FROM read_csv_auto('./dataset/book_review/books_info.csv', maximum_line_size=1048576);\n"
         "CREATE TABLE reviews AS SELECT * FROM read_csv_auto('./dataset/book_review/reviews.csv', maximum_line_size=1048576);\n")
GOOGLE = ("CREATE TABLE business_description AS SELECT * FROM read_csv_auto('./dataset/googlelocal/business_description.csv');\n"
          "CREATE TABLE reviews AS SELECT * FROM read_csv_auto('./dataset/googlelocal/review.csv');\n")
YELP = ("CREATE TABLE yelp_business AS SELECT * FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);\n"
        "CREATE TABLE yelp_review AS SELECT * FROM read_csv_auto('./dataset/yelp/review.csv', maximum_line_size=1048576);\n"
        "CREATE TABLE yelp_tip AS SELECT * FROM read_csv_auto('./dataset/yelp/tip.csv', maximum_line_size=1048576);\n")
YELP_LINKS = ("CREATE TABLE reviews_link AS SELECT 'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id, r.rating, r.date AS review_date FROM yelp_review r WHERE r.business_ref LIKE 'businessref_%';\n"
              "CREATE TABLE tips_link AS SELECT 'businessid_' || regexp_extract(t.business_ref, '([0-9]+)$') AS business_id, t.date AS tip_date FROM yelp_tip t WHERE t.business_ref LIKE 'businessref_%';\n")
CATEGORY_RE = ("'(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|"
               "in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\\s+(.+?)(?:\\.|$)'")
CATEGORY_LEAD = ("'^(?:services(?: and products)? in|services including|services, including|including|destination for|seeking|specializes in|such as|"
                 "in the fields of|selection of|range of|array of|variety of|mix of|collection of|menu featuring|options for|dedicated to|catering to)\\s+'")
TS_FORMATS = ["%Y-%m-%d %H:%M:%S", "%d %b %Y, %H:%M", "%d %b %Y %H:%M", "%d %b %Y, %I:%M %p", "%d %B %Y, %H:%M", "%d %B %Y, %I:%M %p",
              "%B %d, %Y at %I:%M %p", "%B %d, %Y %H:%M %p", "%d %B %Y at %I:%M %p", "%d %b %Y at %I:%M %p"]


def item(*fields, header=None, sep="\\n"):
    """The SWAN prompt's labelled field block as one expression (NULL fields empty)."""
    parts = [f"'{label}: ' || coalesce(CAST({expr} AS VARCHAR), '')" for label, expr in fields]
    glue = " || E'\\n' || " if sep == "\\n" else f" || '{sep}' || "
    body = glue.join(parts)
    return f"'{header}' || E'\\n' || {body}" if header else body


def ts(col):
    return "COALESCE(" + ", ".join(f"try_strptime({col}, '{f}')" for f in TS_FORMATS) + ")"


def category_tokens(src):
    return (f"CREATE TABLE category_tokens AS SELECT n.business_id, lower(trim(tokens.raw_token)) AS category FROM (SELECT s.business_id, "
            f"regexp_replace(coalesce(regexp_extract(s.description, {CATEGORY_RE}, 1), ''), {CATEGORY_LEAD}, '') AS raw_categories FROM {src} s) n, "
            "LATERAL UNNEST(string_split(replace(replace(replace(n.raw_categories, ', and ', ', '), ' and ', ', '), '  ', ' '), ',')) AS tokens(raw_token) "
            "WHERE trim(tokens.raw_token) <> '';\n")


BOOK_ITEM = item(("Title", "b.title"), ("Subtitle", "b.subtitle"), ("Author", "b.author"), ("Categories", "b.categories"),
                 ("Description", "b.description"), ("Features", "b.features"), ("Details", "b.details"))


def book_candidates(cond, review_pred):
    return (BOOKS +
            f"CREATE TABLE candidates_src AS SELECT CAST(SPLIT_PART(b.book_id, '_', 2) AS INTEGER) AS book_idx, b.book_id, b.title, {BOOK_ITEM} AS item FROM books_info b WHERE b.title IS NOT NULL;\n"
            "CREATE TABLE reviews_filtered AS SELECT CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INTEGER) AS purchase_idx, r.purchase_id, r.review_time, "
            f"CAST(r.rating AS DOUBLE) AS rating, r.text, r.helpful_vote, r.verified_purchase FROM reviews r WHERE {review_pred};\n",
            "SELECT candidates_src.book_id, candidates_src.title, rf.review_time, rf.rating, rf.helpful_vote, rf.purchase_id, rf.text "
            "FROM candidates_src JOIN reviews_filtered rf ON rf.purchase_idx = candidates_src.book_idx "
            f"WHERE NLfilter(candidates_src.item, '{cond}') ORDER BY rf.review_time DESC")


def google(src_pred, cond, review_pred, select, order):
    return (GOOGLE +
            f"CREATE TABLE biz_src AS SELECT bd.*, {item(('Name', 'bd.name'), ('Description', 'bd.description'), ('Misc', 'bd.misc'))} AS item "
            f"FROM business_description bd WHERE {src_pred};\n",
            f"SELECT {select} FROM biz_src JOIN reviews r ON r.gmap_id = biz_src.gmap_id WHERE {review_pred} AND NLfilter(biz_src.item, '{cond}') ORDER BY {order}")


CUSTOMER_COLS = ["c_custkey", "c_name", "c_address", "c_nationkey", "c_phone", "c_acctbal", "c_mktsegment", "c_comment"]
AB_COLS = ["l_orderkey", "l_linenumber", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode", "l_shipinstruct", "o_orderdate", "o_orderpriority", "o_totalprice"]
B_BASE = ("CREATE TABLE b_base AS SELECT l.l_orderkey, l.l_linenumber, l.l_partkey, l.l_suppkey, l.l_quantity, l.l_extendedprice, l.l_discount, l.l_shipmode, l.l_shipinstruct, "
          "o.o_orderdate, o.o_orderpriority, o.o_totalprice FROM lineitem l JOIN orders o ON o.o_orderkey = l.l_orderkey WHERE l.l_shipdate >= DATE '1995-01-01' AND l.l_shipdate < DATE '1997-01-01' "
          "AND l.l_discount BETWEEN 0.05 AND 0.07 AND l.l_quantity BETWEEN 18 AND 22 AND l.l_shipmode IN ('AIR', 'REG AIR') AND l.l_shipinstruct = 'DELIVER IN PERSON' "
          "AND o.o_orderstatus = 'O' AND o.o_orderpriority IN ('1-URGENT', '2-HIGH') AND o.o_totalprice > 100000;\n")
PART_ITEM3 = item(("Part Name", "p_name"), ("Part Type", "p_type"), ("Brand", "p_brand"))
CUST_ITEM3 = item(("Customer Name", "c_name"), ("Market Segment", "c_mktsegment"), ("Account Balance", "c_acctbal"))
CUST_ITEM4 = item(("Customer Name", "c_name"), ("Market Segment", "c_mktsegment"), ("Account Balance", "c_acctbal"), ("Comment", "c_comment"))
A_COND = ("You are helping a retail analytics team pick products for customer demand analysis. From the name, type, and brand, does this item read like a "
          "marketable finished good rather than an upstream industrial input? If uncertain, answer YES.")
C_COND = ("You are tagging accounts for B2B revenue analysis. Based on this customer name pattern, market segment, and balance level, does this profile look "
          "more like a business customer likely to place larger operational orders? If uncertain, answer NO.")
C_VALID = "You are validating whether this customer profile is a business-relevant account for enterprise analysis. If uncertain, answer NO."


def tpch_ab(a_cond, c_conds, customer_limit=100):
    setup = (TPCH + B_BASE +
             f"CREATE TABLE part_src AS SELECT p_partkey, p_name, p_type, p_brand, {PART_ITEM3} AS item FROM part;\n"
             f"CREATE TABLE customer_src AS SELECT *, {CUST_ITEM3} AS item FROM (SELECT * FROM customer LIMIT {customer_limit});\n")
    cols = ", ".join(["part_src.p_partkey", "part_src.p_name", "part_src.p_type", "part_src.p_brand"] + [f"b.{c}" for c in AB_COLS] + [f"customer_src.{c}" for c in CUSTOMER_COLS])
    preds = " AND ".join([f"NLfilter(part_src.item, '{a_cond}')"] + [f"NLfilter(customer_src.item, '{c}')" for c in c_conds])
    return setup, f"SELECT {cols} FROM part_src JOIN b_base b ON b.l_partkey = part_src.p_partkey, customer_src WHERE {preds}"


Q = {}


def cascade(col, cases, fallback):
    """CASE WHEN NLfilter(col, c1) THEN l1 WHEN ... ELSE fallback END."""
    whens = " ".join(f"WHEN NLfilter({col}, '{cond}') THEN '{label}'" for label, cond in cases)
    return f"CASE {whens} ELSE '{fallback}' END"


BOOK_ITEM1 = "coalesce(CAST(title AS VARCHAR), '') || ' by ' || coalesce(CAST(author AS VARCHAR), '')"
Q["Q1"] = ("CREATE TABLE books_src AS SELECT title, " + BOOK_ITEM1 + " AS item FROM read_csv_auto('./dataset/book_review/books_info.csv');\n"
           "CREATE TABLE decades AS SELECT * FROM read_csv_auto('./dataset/book_review/decades.csv');\n",
           "SELECT books_src.title, decades.decade, "
           + cascade("books_src.item", [(c, f"Which category best fits this book, among Fiction, Non-Fiction, Memoir, Biography and Self-Help? It is best described as {a}")
                                        for c, a in [("Memoir", "a Memoir"), ("Biography", "a Biography"), ("Self-Help", "Self-Help"), ("Fiction", "Fiction")]], "Non-Fiction")
           + " AS category FROM books_src CROSS JOIN decades LIMIT 5")
BIZ_ITEM2 = "coalesce(CAST(name AS VARCHAR), '') || ' - ' || coalesce(CAST(description AS VARCHAR), '')"
Q["Q2"] = ("CREATE TABLE biz_src AS SELECT name, description, " + BIZ_ITEM2 + " AS item FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);\n"
           "CREATE TABLE expansion AS SELECT * FROM (VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9), (10)) AS t(exp_id);\n",
           "SELECT biz_src.name, expansion.exp_id, "
           + cascade("biz_src.item", [(c, f"Classify business quality as Poor, Fair, Good or Excellent: the quality of this business is best described as {c}")
                                      for c in ["Poor", "Fair", "Excellent"]], "Good")
           + " AS quality_tier FROM biz_src CROSS JOIN expansion LIMIT 3")
LOC_ITEM3 = "coalesce(CAST(name AS VARCHAR), '') || ' in ' || coalesce(CAST(state AS VARCHAR), '') || '. Description: ' || coalesce(CAST(description AS VARCHAR), '')"
Q["Q3"] = ("CREATE TABLE loc_src AS SELECT name, state, description, " + LOC_ITEM3 + " AS item FROM read_csv_auto('./dataset/googlelocal/business_description.csv');\n"
           "CREATE TABLE exp AS SELECT * FROM generate_series(1, 20) AS t(e);\n",
           "SELECT loc_src.name, exp.e, "
           + cascade("loc_src.item", [(c, f"Classify the business location as Urban, Suburban or Rural: it is best described as {c}")
                                      for c in ["Urban", "Rural"]], "Suburban")
           + " AS location_type FROM loc_src CROSS JOIN exp LIMIT 4")
Q["Q4"] = (BOOKS +
           "CREATE TABLE sem_popular_src AS SELECT DISTINCT b.book_id, b.title, "
           + item(("Title", "b.title"), ("Subtitle", "b.subtitle"), ("Author", "b.author"), ("Categories", "b.categories"), ("Description", "b.description"))
           + " AS item FROM books_info b JOIN (SELECT DISTINCT 'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '') AS book_id FROM reviews r WHERE EXTRACT(YEAR FROM r.review_time) >= 2016) p "
           "ON p.book_id = b.book_id WHERE b.title IS NOT NULL;\n",
           "SELECT r.review_time, r.rating, r.purchase_id, sem_popular_src.book_id, sem_popular_src.title FROM reviews r "
           "JOIN sem_popular_src ON sem_popular_src.book_id = 'bookid_' || REPLACE(r.purchase_id, 'purchaseid_', '') "
           "WHERE EXTRACT(YEAR FROM r.review_time) >= 2016 AND CAST(r.rating AS INTEGER) >= 4 "
           "AND NLfilter(sem_popular_src.item, 'Is this book recently popular since 2016?') ORDER BY r.review_time DESC")
Q["Q5"] = (BOOKS +
           "CREATE TABLE narrowed_books AS SELECT b.book_id, b.title, substring(b.description, 1, 600) AS description_snippet, b.categories, "
           + item(("Title", "b.title"), ("Categories", "b.categories"), ("Description Sample", "substring(b.description, 1, 600)"))
           + " AS item FROM books_info b WHERE b.title IS NOT NULL AND lower(b.title) LIKE '%cleo porter%' AND lower(coalesce(b.categories, '')) ILIKE '%science fiction%';\n"
           "CREATE TABLE qualified_reviews AS SELECT CAST(SPLIT_PART(r.purchase_id, '_', 2) AS INTEGER) AS purchase_idx, r.purchase_id, r.review_time, CAST(r.rating AS DOUBLE) AS rating, "
           "r.helpful_vote, r.text FROM reviews r WHERE r.verified_purchase = 1 AND CAST(r.rating AS DOUBLE) = 5 AND r.helpful_vote >= 1 "
           "AND r.review_time BETWEEN TIMESTAMP '2020-01-01 00:00:00' AND TIMESTAMP '2020-12-31 23:59:59';\n",
           "SELECT narrowed_books.book_id, narrowed_books.title, qr.review_time, qr.rating, qr.helpful_vote, qr.purchase_id, qr.text FROM narrowed_books "
           "JOIN qualified_reviews qr ON qr.purchase_idx = CAST(SPLIT_PART(narrowed_books.book_id, '_', 2) AS INTEGER) "
           "WHERE NLfilter(narrowed_books.item, 'Return true if this book is a middle-grade science fiction adventure set during a pandemic lockdown in which Cleo Porter "
           "leaves a sealed apartment to deliver life-saving medicine, clearly positioned for young readers.') ORDER BY qr.helpful_vote DESC, qr.review_time DESC")
BUDDY = ("Confirm this is the Buddy the Soldier Bear children''s picture book that follows a stuffed bear from a toy store to a battlefield, emphasizes supporting "
         "soldiers and veteran families, and is categorized under children''s literature.")
R2020 = "r.verified_purchase = 1 AND CAST(r.rating AS DOUBLE) >= 5 AND r.helpful_vote >= 2 AND r.review_time >= TIMESTAMP '2020-01-01' AND r.review_time < TIMESTAMP '2021-01-01'"
Q["Q6"] = book_candidates(BUDDY, R2020)
setup6, query6 = Q["Q6"]
Q["Q7"] = (setup6 + "CREATE TABLE rc AS SELECT * FROM read_csv_auto('./dataset/book_review/review_context.csv');\n"
           "CREATE TABLE rc_src AS SELECT rc.*, " + item(("Context ID", "rc.context_id"), ("Name", "rc.context_name"), ("Audience", "rc.audience"),
                                                          ("Focus", "rc.message_focus"), ("Rule", "rc.quality_rule"), sep="; ") + " AS item FROM rc;\n",
           "SELECT candidates_src.book_id, candidates_src.title, rf.review_time, rf.rating, rf.helpful_vote, rf.purchase_id, rf.text "
           "FROM candidates_src JOIN reviews_filtered rf ON rf.purchase_idx = candidates_src.book_idx, rc_src "
           f"WHERE NLfilter(candidates_src.item, '{BUDDY}') AND NLfilter(rc_src.item, 'Does this context match reviews about Buddy the Soldier Bear as toddler bedtime "
           "comfort + military/veteran support in 2020 with verified 5-star and >=2 helpful votes? If unsure, answer NO.') ORDER BY rf.review_time DESC")
Q["Q8"] = book_candidates("Confirm this is the second edition of Make: Electronics, the hands-on beginner''s electronics guide from the Make magazine team that highlights "
                          "colorful diagrams and step-by-step experiments.",
                          "r.verified_purchase = 1 AND CAST(r.rating AS DOUBLE) >= 5 AND r.helpful_vote >= 50 AND r.review_time >= TIMESTAMP '2017-01-01' AND r.review_time < TIMESTAMP '2018-01-01'")
Q["Q9"] = book_candidates("Confirm this is French Illusions, the travel memoir where Linda Kovic-Skow recounts her au pair year with the demanding Dubois family in "
                          "France''s Loire Valley.",
                          "r.verified_purchase = 1 AND CAST(r.rating AS DOUBLE) >= 5 AND r.helpful_vote >= 20 AND r.review_time >= TIMESTAMP '2013-01-01' AND r.review_time < TIMESTAMP '2014-01-01'")
Q["Q10"] = (GOOGLE +
            "CREATE TABLE biz_src AS SELECT bd.*, " + item(("Name", "bd.name"), ("Description", "bd.description"), ("Misc", "bd.misc"), ("Number of Reviews", "bd.num_of_reviews"))
            + " AS item FROM business_description bd WHERE bd.num_of_reviews IS NOT NULL AND CAST(bd.num_of_reviews AS INTEGER) >= 15 AND bd.state LIKE 'Open%';\n",
            "SELECT biz_src.gmap_id, biz_src.name, CAST(biz_src.num_of_reviews AS INTEGER) AS business_review_count, biz_src.state, r.time AS review_time, CAST(r.rating AS DOUBLE) AS rating, r.text "
            "FROM biz_src JOIN reviews r ON r.gmap_id = biz_src.gmap_id WHERE CAST(r.rating AS DOUBLE) >= 4.8 AND r.text IS NOT NULL AND LENGTH(r.text) >= 150 "
            "AND (r.time LIKE '%2021%' OR r.time LIKE '%2022%' OR r.time LIKE '%2023%' OR r.time LIKE '%2024%') "
            "AND NLfilter(biz_src.item, 'Return true if this business primarily offers massage therapy or spa treatments. Otherwise return false.') ORDER BY rating DESC, review_time DESC")
SEL_G = "biz_src.gmap_id, biz_src.name, CAST(biz_src.num_of_reviews AS INTEGER) AS num_reviews, biz_src.state, r.time AS review_time, CAST(r.rating AS DOUBLE) AS rating, r.text"
Q["Q11"] = google("TRUE", "Return true only if this business is primarily a cannabis dispensary, delivery service, or weed retailer. Otherwise return false.",
                  "r.text IS NOT NULL AND LENGTH(r.text) >= 160 AND CAST(r.rating AS DOUBLE) >= 4.8 AND r.time ILIKE '%2021%' AND LOWER(r.text) LIKE '%staff%' AND "
                  "(LOWER(r.text) LIKE '%helpful%' OR LOWER(r.text) LIKE '%friendly%' OR LOWER(r.text) LIKE '%knowledgeable%' OR LOWER(r.text) LIKE '%budtender%' OR LOWER(r.text) LIKE '%bud tender%')",
                  SEL_G, "rating DESC, review_time DESC, LENGTH(text) DESC")
Q["Q12"] = google("bd.num_of_reviews IS NOT NULL AND CAST(bd.num_of_reviews AS INTEGER) >= 10 AND bd.state ILIKE 'Open%'",
                  "Return true only if this business is primarily an auto repair, mechanic, brake, tire, or vehicle maintenance shop. Otherwise return false.",
                  "r.text IS NOT NULL AND LENGTH(r.text) >= 120 AND CAST(r.rating AS DOUBLE) >= 4.5 AND r.time ILIKE '%2021%' AND "
                  "(LOWER(r.text) LIKE '%honest%' OR LOWER(r.text) LIKE '%quick%' OR LOWER(r.text) LIKE '%fast%')",
                  SEL_G + ", LENGTH(r.text) AS text_length", "rating DESC, text_length DESC")
Q["Q13"] = google("bd.num_of_reviews IS NOT NULL AND CAST(bd.num_of_reviews AS INTEGER) >= 3",
                  "Return true only if this business primarily offers nail salon, manicure, pedicure, brow, or spa services. Otherwise return false.",
                  "r.text IS NOT NULL AND LENGTH(r.text) >= 120 AND CAST(r.rating AS DOUBLE) >= 4.5 AND r.time ILIKE '%2021%' AND "
                  "(LOWER(r.text) LIKE '%nail%' OR LOWER(r.text) LIKE '%mani%' OR LOWER(r.text) LIKE '%pedi%' OR LOWER(r.text) LIKE '%brow%' OR LOWER(r.text) LIKE '%dip%')",
                  SEL_G, "rating DESC, review_time DESC")
Q["Q14"] = google("bd.num_of_reviews IS NOT NULL AND CAST(bd.num_of_reviews AS INTEGER) >= 5",
                  "Return true only if this place is a campground, RV park, off-road recreation area, or outdoor trailhead. Otherwise return false.",
                  "r.text IS NOT NULL AND LENGTH(r.text) >= 80 AND CAST(r.rating AS DOUBLE) >= 4 AND r.time ILIKE '%2021%' AND "
                  "(LOWER(r.text) LIKE '%dirt bike%' OR LOWER(r.text) LIKE '%trail%' OR LOWER(r.text) LIKE '%off-road%' OR LOWER(r.text) LIKE '%camp%')",
                  SEL_G, "rating DESC, review_time DESC")
Q["Q15"] = tpch_ab(A_COND, [C_COND])
Q["Q16"] = tpch_ab("Task: classify whether this part should be included in customer-facing demand analysis. Rule: YES only if the part sounds like a finished good that "
                   "could appear in commercial product demand; NO for raw/industrial-looking components. If uncertain, answer NO.",
                   ["Task: decide if this is an enterprise-oriented customer profile for B2B order analysis. Rule: YES only for profiles that look commercially active "
                    "(business segment + meaningful balance); otherwise NO. If uncertain, answer NO.", C_VALID])
Q["Q25"] = tpch_ab(A_COND, [C_COND, C_VALID])
JOINED17 = ["o_orderkey", "o_custkey", "o_orderdate", "o_orderpriority", "o_totalprice"]
Q["Q17"] = (TPCH +
            "CREATE TABLE orders_src AS SELECT o.*, " + item(("Order Priority", "o.o_orderpriority"), ("Total Price", "o.o_totalprice"), ("Comment", "o.o_comment"))
            + " AS item FROM orders o WHERE o.o_orderdate >= DATE '1993-01-01' AND o.o_orderdate < DATE '1997-01-01' AND o.o_orderstatus IN ('O', 'F') AND o.o_totalprice > 50000;\n"
            "CREATE TABLE lineitem_focus AS SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_shipmode, l.l_shipinstruct, l.l_quantity, l.l_extendedprice, l.l_discount, l.l_tax, l.l_returnflag "
            "FROM lineitem l WHERE l.l_shipdate >= DATE '1993-01-01' AND l.l_shipdate < DATE '1998-01-01' AND l.l_quantity BETWEEN 5 AND 40 AND l.l_discount BETWEEN 0.01 AND 0.10 AND l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK');\n"
            "CREATE TABLE part_supplier AS SELECT p.p_partkey, p.p_name, p.p_type, p.p_brand, s.s_suppkey, s.s_name, n.n_name AS supplier_nation FROM part p "
            "JOIN partsupp ps ON ps.ps_partkey = p.p_partkey JOIN supplier s ON s.s_suppkey = ps.ps_suppkey JOIN nation n ON n.n_nationkey = s.s_nationkey "
            "WHERE p.p_size BETWEEN 5 AND 35 AND ps.ps_supplycost BETWEEN 10 AND 1000;\n"
            f"CREATE TABLE customer_src AS SELECT *, {CUST_ITEM4} AS item FROM (SELECT * FROM customer LIMIT 120);\n"
            # the three predicates each read one side of the final cross product, so each side is a stage and the
            # product (the query's answer shape) is formed only over the survivors: ThalamusDB's own SQL
            # re-evaluations over the unfiltered product exhaust memory.
            "CREATE TABLE orders_sem AS SELECT * FROM orders_src WHERE NLfilter(orders_src.item, 'You are triaging enterprise orders for operations monitoring. From the priority label, "
            "total price, and order comment, does this order look important enough to deserve active follow-up by an operations manager? If uncertain, answer YES.');\n"
            "CREATE TABLE customer_sem AS SELECT * FROM customer_src WHERE NLfilter(customer_src.item, 'You are identifying high-value B2B customer profiles. From the customer name format, "
            "market segment, balance, and account comment, does this record look like a customer worth prioritizing in business account analytics? If uncertain, answer NO.') "
            "AND NLfilter(customer_src.item, 'You are an operations manager assessing whether a high-value customer is associated with an important order that requires active follow-up.');\n",
            "SELECT " + ", ".join([f"orders_sem.{c}" for c in JOINED17] + ["lf.l_partkey", "lf.l_suppkey", "lf.l_shipmode", "lf.l_shipinstruct", "lf.l_quantity", "lf.l_extendedprice",
                                                                          "lf.l_discount", "lf.l_tax", "lf.l_returnflag", "ps.p_name", "ps.p_type", "ps.p_brand", "ps.s_name", "ps.supplier_nation"]
                                  + [f"customer_sem.{c}" for c in CUSTOMER_COLS]) +
            " FROM orders_sem JOIN lineitem_focus lf ON lf.l_orderkey = orders_sem.o_orderkey JOIN part_supplier ps ON ps.p_partkey = lf.l_partkey AND ps.s_suppkey = lf.l_suppkey, customer_sem")
Q["Q18"] = (TPCH +
            "CREATE TABLE part_src AS SELECT p_partkey, p_name, p_type, p_container, p_size, " + item(("Part Name", "p_name"), ("Part Type", "p_type"), ("Container", "p_container"), ("Size", "p_size"))
            + " AS item FROM part WHERE p_size BETWEEN 3 AND 30 AND p_container IN ('SM BOX', 'SM CASE', 'MED BOX', 'MED BAG', 'LG BOX');\n"
            "CREATE TABLE lineitem_orders AS SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_shipmode, l.l_shipinstruct, l.l_quantity, l.l_extendedprice, l.l_discount, "
            "o.o_custkey, o.o_orderdate, o.o_orderpriority, o.o_totalprice FROM lineitem l JOIN orders o ON o.o_orderkey = l.l_orderkey WHERE l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK', 'MAIL') "
            "AND l.l_shipinstruct IN ('DELIVER IN PERSON', 'TAKE BACK RETURN') AND l.l_quantity BETWEEN 10 AND 45 AND o.o_orderdate >= DATE '1994-01-01' AND o.o_orderdate < DATE '1998-01-01' AND o.o_totalprice > 30000;\n"
            "CREATE TABLE customer_geo AS SELECT c.c_custkey, c.c_name, c.c_nationkey, c.c_mktsegment, c.c_acctbal, n.n_name AS customer_nation, r.r_name AS customer_region FROM customer c "
            "JOIN nation n ON n.n_nationkey = c.c_nationkey JOIN region r ON r.r_regionkey = n.n_regionkey WHERE r.r_name IN ('AMERICA', 'EUROPE', 'ASIA');\n"
            "CREATE TABLE nation_src AS SELECT *, " + item(("Nation", "n_name")) + " AS item FROM (SELECT * FROM nation LIMIT 25);\n",
            "SELECT part_src.p_partkey, part_src.p_name, part_src.p_type, part_src.p_container, lo.l_orderkey, lo.l_shipmode, lo.l_shipinstruct, lo.l_quantity, lo.l_extendedprice, lo.l_discount, "
            "lo.o_orderdate, lo.o_orderpriority, lo.o_totalprice, cg.c_name, cg.c_mktsegment, cg.c_acctbal, cg.customer_nation, cg.customer_region, nation_src.n_nationkey, nation_src.n_name, "
            "nation_src.n_regionkey, nation_src.n_comment FROM part_src JOIN lineitem_orders lo ON lo.l_partkey = part_src.p_partkey JOIN customer_geo cg ON cg.c_custkey = lo.o_custkey, nation_src "
            "WHERE NLfilter(part_src.item, 'You are preparing a logistics performance dashboard for retail distribution. Based on product name, type, packaging container, and size, does this part "
            "look like something that would realistically move through customer-facing distribution channels? If uncertain, answer YES.') "
            "AND NLfilter(nation_src.item, 'You are selecting countries to include in a cross-border freight and trade analysis view. Is this nation likely meaningful enough to keep in a global "
            "logistics discussion dataset? If uncertain, answer YES.')")
L19 = ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_tax", "l_returnflag", "l_linestatus", "l_shipmode", "l_shipinstruct"]
Q["Q19"] = (TPCH +
            "CREATE TABLE lineitem_src AS SELECT l.*, " + item(("Mode", "l.l_shipmode"), ("Instruction", "l.l_shipinstruct"), ("Comment", "l.l_comment"), header="Lineitem shipping details:")
            + " AS item FROM lineitem l WHERE l.l_shipdate >= DATE '1994-01-01' AND l.l_shipdate < DATE '1998-01-01' AND l.l_returnflag IN ('R', 'A', 'N') AND l.l_quantity BETWEEN 3 AND 38 AND l.l_discount BETWEEN 0.00 AND 0.10;\n"
            "CREATE TABLE order_customer AS SELECT o.o_orderkey, o.o_custkey, o.o_orderdate, o.o_orderpriority, o.o_orderstatus, o.o_totalprice, c.c_name, c.c_mktsegment, c.c_acctbal, c.c_comment "
            "FROM orders o JOIN customer c ON c.c_custkey = o.o_custkey WHERE o.o_orderdate >= DATE '1994-01-01' AND o.o_orderdate < DATE '1998-01-01' AND o.o_orderstatus IN ('O', 'F') AND o.o_totalprice > 20000;\n"
            "CREATE TABLE part_supplier AS SELECT p.p_partkey, p.p_name, p.p_type, p.p_brand, s.s_suppkey, s.s_name, s.s_acctbal FROM part p JOIN partsupp ps ON ps.ps_partkey = p.p_partkey "
            "JOIN supplier s ON s.s_suppkey = ps.ps_suppkey WHERE p.p_size BETWEEN 1 AND 40 AND ps.ps_supplycost BETWEEN 5 AND 1200;\n"
            "CREATE TABLE customer_src AS SELECT *, " + item(("Name", "c_name"), ("Segment", "c_mktsegment"), ("Balance", "c_acctbal"), ("Comment", "c_comment"), header="Customer profile check:")
            + " AS item FROM (SELECT * FROM customer LIMIT 180);\n"
            # single-side predicates over a lineitem-scale cross product (98k answer rows): staged, as in Q17
            "CREATE TABLE lineitem_sem AS SELECT * FROM lineitem_src WHERE NLfilter(lineitem_src.item, 'Is this likely a potentially problematic fulfillment case worth audit attention? If unsure, answer YES.');\n"
            "CREATE TABLE customer_sem AS SELECT * FROM customer_src WHERE NLfilter(customer_src.item, 'Answer YES only if this customer appears likely to have higher complaint/escalation risk in commercial operations. Otherwise answer NO. If unsure, answer NO.');\n",
            "SELECT " + ", ".join([f"lineitem_sem.{c}" for c in L19] + ["oc.o_orderdate", "oc.o_orderpriority", "oc.o_orderstatus", "oc.o_totalprice", "oc.c_name", "oc.c_mktsegment", "oc.c_acctbal",
                                                                         "ps.p_name", "ps.p_type", "ps.p_brand", "ps.s_name", "ps.s_acctbal", "customer_sem.c_custkey", "customer_sem.c_name AS cc_name",
                                                                         "customer_sem.c_address", "customer_sem.c_nationkey", "customer_sem.c_phone", "customer_sem.c_acctbal AS cc_acctbal",
                                                                         "customer_sem.c_mktsegment AS cc_mktsegment", "customer_sem.c_comment"]) +
            " FROM lineitem_sem JOIN order_customer oc ON oc.o_orderkey = lineitem_sem.l_orderkey JOIN part_supplier ps ON ps.p_partkey = lineitem_sem.l_partkey AND ps.s_suppkey = lineitem_sem.l_suppkey, customer_sem")
YELP_ITEM = item(("Business ID", "yb.business_id"), ("Attributes", "yb.attributes"), ("Description", "yb.description"))
Q["Q20"] = (YELP + YELP_LINKS + f"CREATE TABLE yelp_src AS SELECT yb.*, {YELP_ITEM} AS item FROM yelp_business yb;\n",
            "SELECT yelp_src.business_id, rl.review_date, rl.rating FROM yelp_src JOIN reviews_link rl ON rl.business_id = yelp_src.business_id "
            "LEFT JOIN tips_link tl ON tl.business_id = yelp_src.business_id WHERE rl.review_date ILIKE '%2018%' AND rl.rating >= 4 AND yelp_src.attributes ILIKE '%\"BikeParking\": \"True\"%' "
            "AND tl.tip_date ILIKE '%2018%' AND yelp_src.description ILIKE '%Philadelphia%' "
            "AND NLfilter(yelp_src.item, 'Return true if the business offers customer parking (lot, garage, street, or valet) or any form of bike parking. Use the provided attributes and description to decide.') "
            "ORDER BY rl.rating DESC, rl.review_date DESC")
Q["Q21"] = (YELP + YELP_LINKS + f"CREATE TABLE yelp_src AS SELECT yb.*, {YELP_ITEM} AS item FROM yelp_business yb;\n" + category_tokens("yelp_src"),
            "SELECT ct.category, yelp_src.business_id, rl.review_date, rl.rating FROM category_tokens ct JOIN yelp_src ON yelp_src.business_id = ct.business_id "
            "JOIN reviews_link rl ON rl.business_id = ct.business_id JOIN tips_link tl ON tl.business_id = ct.business_id "
            "WHERE yelp_src.attributes ILIKE '%\"BusinessAcceptsCreditCards\": \"True\"%' AND rl.review_date ILIKE '%2019%' AND tl.tip_date ILIKE '%2019%' "
            "AND NLfilter(yelp_src.item, 'Return true if the business clearly accepts credit card payments. Prefer explicit evidence from attributes or the narrative.') "
            "ORDER BY rl.rating DESC, rl.review_date DESC")
STATES = "'AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'"
Q["Q22"] = (YELP + YELP_LINKS + f"CREATE TABLE yelp_src AS SELECT yb.*, {YELP_ITEM} AS item FROM yelp_business yb;\n"
            f"CREATE TABLE valid_states AS SELECT business_id, state_code FROM (SELECT business_id, regexp_extract(description, ',\\s*([A-Z]{{2}})\\b', 1) AS state_code FROM yelp_src) s WHERE state_code IN ({STATES});\n",
            "SELECT vs.state_code AS state, yelp_src.business_id, rl.review_date, rl.rating FROM valid_states vs JOIN yelp_src ON yelp_src.business_id = vs.business_id "
            "JOIN reviews_link rl ON rl.business_id = vs.business_id JOIN tips_link tl ON tl.business_id = vs.business_id "
            "WHERE yelp_src.attributes ILIKE '%\"WiFi\":%' AND rl.review_date ILIKE '%2018%' AND tl.tip_date ILIKE '%2018%' "
            "AND NLfilter(yelp_src.item, 'Return true if the business offers public WiFi access for its customers. Use both the attributes and description for evidence.') "
            "ORDER BY rl.rating DESC, rl.review_date DESC")
Q["Q23"] = (YELP + "CREATE TABLE yelp_src AS SELECT yb.*, " + item(("Business ID", "yb.business_id"), ("Name", "yb.name"), ("Attributes", "yb.attributes"), ("Description", "yb.description"))
            + " AS item FROM yelp_business yb;\n" + category_tokens("yelp_src") +
            f"CREATE TABLE reviews_link AS SELECT 'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id, r.rating, {ts('r.date')} AS review_ts FROM yelp_review r WHERE r.business_ref LIKE 'businessref_%';\n",
            "SELECT yelp_src.name AS business_name, ct.category, rl.review_ts AS review_time, rl.rating FROM yelp_src JOIN reviews_link rl ON rl.business_id = yelp_src.business_id "
            "JOIN category_tokens ct ON ct.business_id = yelp_src.business_id WHERE rl.review_ts BETWEEN TIMESTAMP '2016-01-01 00:00:00' AND TIMESTAMP '2016-06-30 23:59:59' AND rl.review_ts IS NOT NULL "
            "AND NLfilter(yelp_src.item, 'Return true if the description makes the primary business category explicit (for example, Italian restaurant, dental clinic, yoga studio). "
            "Favor entries where you can clearly read at least one category phrase.') ORDER BY rl.rating DESC, rl.review_ts DESC")
Q["Q24"] = ("CREATE TABLE yelp_business AS SELECT * FROM read_csv_auto('./dataset/yelp/yelp_business_csv/business.csv', maximum_line_size=1048576);\n"
            "CREATE TABLE yelp_review AS SELECT * FROM read_csv_auto('./dataset/yelp/review.csv', maximum_line_size=1048576);\n"
            "CREATE TABLE yelp_user AS SELECT * FROM read_csv_auto('./dataset/yelp/user.csv', maximum_line_size=1048576);\n"
            f"CREATE TABLE qualified_users AS SELECT user_id FROM (SELECT u.user_id, {ts('u.yelping_since')} AS joined_ts FROM yelp_user u) j WHERE joined_ts BETWEEN TIMESTAMP '2016-01-01 00:00:00' AND TIMESTAMP '2016-12-31 23:59:59';\n"
            f"CREATE TABLE filtered_reviews AS SELECT pr.business_id, pr.user_id, pr.review_ts FROM (SELECT 'businessid_' || regexp_extract(r.business_ref, '([0-9]+)$') AS business_id, r.user_id, {ts('r.date')} AS review_ts "
            "FROM yelp_review r WHERE r.business_ref LIKE 'businessref_%') pr JOIN qualified_users qu ON qu.user_id = pr.user_id WHERE pr.review_ts >= TIMESTAMP '2016-01-01 00:00:00' AND pr.review_ts IS NOT NULL;\n"
            f"CREATE TABLE yelp_src AS SELECT yb.*, {YELP_ITEM} AS item FROM yelp_business yb JOIN (SELECT DISTINCT business_id FROM filtered_reviews) ab ON ab.business_id = yb.business_id;\n"
            + category_tokens("yelp_src"),
            "SELECT ct.category, fr.user_id, fr.review_ts FROM category_tokens ct JOIN filtered_reviews fr ON fr.business_id = ct.business_id JOIN yelp_src ON yelp_src.business_id = ct.business_id "
            "WHERE NLfilter(yelp_src.item, 'Return true if the business description clearly lists multiple categories or service types that can be tokenized. Focus on entries with explicit category phrases.') "
            "ORDER BY fr.review_ts DESC")
Q["Q26"] = (TPCH +
            f"CREATE TABLE customer_src AS SELECT *, {CUST_ITEM4} AS item FROM customer WHERE c_acctbal > 500 AND c_mktsegment IN ('AUTOMOBILE', 'BUILDING', 'MACHINERY');\n"
            f"CREATE TABLE part_src AS SELECT p_partkey, p_name, p_type, p_brand, {PART_ITEM3} AS item FROM part;\n"
            "CREATE TABLE supplier_src AS SELECT s.s_suppkey, s.s_name, s.s_acctbal, s.s_comment, n.n_name AS supplier_nation, r.r_name AS supplier_region, "
            + item(("Supplier Name", "s.s_name"), ("Account Balance", "s.s_acctbal"), ("Comment", "s.s_comment"), ("Nation", "n.n_name"), ("Region", "r.r_name"))
            + " AS item FROM supplier s JOIN nation n ON n.n_nationkey = s.s_nationkey JOIN region r ON r.r_regionkey = n.n_regionkey WHERE s.s_acctbal > 0;\n"
            "CREATE TABLE orders_src AS SELECT o.*, " + item(("Order Priority", "o.o_orderpriority"), ("Total Price", "o.o_totalprice"), ("Order Comment", "o.o_comment"))
            + " AS item FROM orders o WHERE o.o_orderstatus = 'O' AND o.o_orderpriority IN ('1-URGENT', '2-HIGH') AND o.o_totalprice > 90000;\n"
            # four single-side predicates over a cross product with customer_src: staged (as Q17/Q19), the product
            # is then plain SQL over survivors
            f"CREATE TABLE part_sem AS SELECT * FROM part_src WHERE NLfilter(part_src.item, '{A_COND}');\n"
            "CREATE TABLE supplier_sem AS SELECT * FROM supplier_src WHERE NLfilter(supplier_src.item, 'You are shortlisting suppliers for stable operations. Based on supplier name style, account balance, comment, nation, and region, does this supplier look "
            "commercially dependable? If uncertain, answer YES.');\n"
            "CREATE TABLE orders_sem AS SELECT * FROM orders_src WHERE NLfilter(orders_src.item, 'You are triaging enterprise orders for operations monitoring. From priority, total price, and comment, should this order be considered operationally "
            "important? If uncertain, answer YES.');\n"
            "CREATE TABLE customer_sem AS SELECT * FROM customer_src WHERE NLfilter(customer_src.item, 'You are tagging accounts for B2B revenue analysis. Based on customer name, market segment, account balance, and profile comment, does this customer "
            "look business-like and operationally relevant? If uncertain, answer NO.');\n",
            "SELECT l.l_orderkey, l.l_linenumber, l.l_quantity, l.l_extendedprice, l.l_discount, l.l_tax, l.l_shipmode, l.l_shipinstruct, part_sem.p_name, part_sem.p_type, part_sem.p_brand, "
            "supplier_sem.s_name, supplier_sem.supplier_nation, supplier_sem.supplier_region, ps.ps_supplycost, ps.ps_availqty, customer_sem.c_custkey AS context_custkey, customer_sem.c_name, "
            "customer_sem.c_mktsegment, customer_sem.c_acctbal FROM lineitem l JOIN orders_sem ON orders_sem.o_orderkey = l.l_orderkey JOIN partsupp ps ON ps.ps_partkey = l.l_partkey AND ps.ps_suppkey = l.l_suppkey "
            "JOIN part_sem ON part_sem.p_partkey = l.l_partkey JOIN supplier_sem ON supplier_sem.s_suppkey = l.l_suppkey, customer_sem "
            "WHERE l.l_shipdate >= DATE '1995-01-01' AND l.l_shipdate < DATE '1997-01-01' AND l.l_discount BETWEEN 0.05 AND 0.08 AND l.l_quantity BETWEEN 16 AND 24 AND l.l_shipmode IN ('AIR', 'REG AIR') "
            "AND l.l_shipinstruct = 'DELIVER IN PERSON' AND ps.ps_supplycost BETWEEN 80 AND 900 AND ps.ps_availqty >= 50")
J27 = ["p_partkey", "p_name", "p_type", "p_brand", "ps_suppkey", "s_name", "s_acctbal", "s_comment", "nation_name", "region_name", "ps_availqty", "ps_supplycost", "l_orderkey", "l_quantity",
       "l_extendedprice", "o_orderdate", "o_orderpriority", "o_totalprice"]
Q["Q27"] = (TPCH +
            "CREATE TABLE part_src AS SELECT p_partkey, p_name, p_type, p_brand, p_size, " + item(("Part Name", "p_name"), ("Part Type", "p_type"), ("Brand", "p_brand"), ("Size", "p_size")) + " AS item FROM part;\n"
            "CREATE TABLE supply_base AS SELECT ps.ps_partkey, ps.ps_suppkey, ps.ps_availqty, ps.ps_supplycost, s.s_name, s.s_acctbal, s.s_comment, s.s_nationkey, n.n_name AS nation_name, r.r_name AS region_name "
            "FROM partsupp ps JOIN supplier s ON s.s_suppkey = ps.ps_suppkey JOIN nation n ON n.n_nationkey = s.s_nationkey JOIN region r ON r.r_regionkey = n.n_regionkey "
            "WHERE ps.ps_availqty BETWEEN 10 AND 600 AND ps.ps_supplycost BETWEEN 20 AND 900 AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA');\n"
            "CREATE TABLE orders_lineitem AS SELECT l.l_partkey, l.l_suppkey, l.l_orderkey, l.l_quantity, l.l_extendedprice, l.l_discount, o.o_orderdate, o.o_orderpriority, o.o_totalprice "
            "FROM lineitem l JOIN orders o ON o.o_orderkey = l.l_orderkey WHERE o.o_orderdate >= DATE '1994-01-01' AND o.o_orderdate < DATE '1997-01-01' AND l.l_quantity BETWEEN 8 AND 35 AND l.l_discount BETWEEN 0.00 AND 0.08;\n"
            "CREATE TABLE part_focus AS SELECT part_src.p_partkey, part_src.p_name, part_src.p_type, part_src.p_brand, part_src.p_size FROM part_src "
            "WHERE NLfilter(part_src.item, 'You are screening parts for strategic sourcing analysis. Looking at part name, type, brand, and size, does this look like a differentiated component "
            "where supplier quality and pricing strategy would matter? If uncertain, answer YES.');\n"
            "CREATE TABLE joined AS SELECT pf.p_partkey, pf.p_name, pf.p_type, pf.p_brand, sb.ps_suppkey, sb.s_name, sb.s_acctbal, sb.s_comment, sb.nation_name, sb.region_name, sb.ps_availqty, sb.ps_supplycost, "
            "ol.l_orderkey, ol.l_quantity, ol.l_extendedprice, ol.o_orderdate, ol.o_orderpriority, ol.o_totalprice FROM part_focus pf JOIN supply_base sb ON sb.ps_partkey = pf.p_partkey "
            "JOIN orders_lineitem ol ON ol.l_partkey = sb.ps_partkey AND ol.l_suppkey = sb.ps_suppkey;\n"
            "CREATE TABLE supplier_src AS SELECT s.s_suppkey, s.s_name, s.s_address, s.s_nationkey, s.s_phone, s.s_acctbal, s.s_comment, "
            + item(("Supplier Name", "s.s_name"), ("Account Balance", "s.s_acctbal"), ("Comment", "s.s_comment"))
            + " AS item FROM supplier s JOIN (SELECT DISTINCT ps_suppkey FROM joined LIMIT 80) k ON k.ps_suppkey = s.s_suppkey;\n"
            "CREATE TABLE pairs AS SELECT j.*, " + item(("Part Type", "j.p_type"), ("Region", "j.region_name"), ("Supplier Name", "sc.s_name"), ("Account Balance", "sc.s_acctbal"), ("Comment", "sc.s_comment"))
            + " AS vitem FROM joined j JOIN supplier_src sc ON sc.s_suppkey = j.ps_suppkey;\n",
            "SELECT " + ", ".join(f"pairs.{c}" for c in J27) + ", supplier_src.s_suppkey, supplier_src.s_address, supplier_src.s_nationkey, supplier_src.s_phone "
            "FROM pairs JOIN supplier_src ON supplier_src.s_suppkey = pairs.ps_suppkey "
            "WHERE NLfilter(supplier_src.item, 'You are shortlisting suppliers for long-term procurement partnerships. Based on supplier identity style, account balance, and profile comment tone, "
            "does this supplier appear commercially dependable enough for preferred-vendor consideration? If uncertain, answer YES.') "
            "AND NLfilter(pairs.vitem, 'You are validating whether this supplier profile is suitable for preferred-vendor sourcing. If uncertain, answer YES.')")
J28 = ["o_orderkey", "o_orderpriority", "o_totalprice", "o_comment", "o_orderdate", "o_custkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode"]
Q["Q28"] = (TPCH +
            "CREATE TABLE orders_src AS SELECT o.o_orderkey, o.o_orderpriority, o.o_totalprice, o.o_comment, o.o_orderdate, o.o_custkey, "
            + item(("Order Priority", "o.o_orderpriority"), ("Total Price", "o.o_totalprice"), ("Comment", "o.o_comment"))
            + " AS item FROM orders o WHERE o.o_orderdate >= DATE '1995-01-01' AND o.o_orderdate < DATE '1996-07-01' AND o.o_totalprice BETWEEN 100000 AND 220000 AND o.o_orderstatus = 'O' AND o.o_orderpriority IN ('1-URGENT', '2-HIGH');\n"
            "CREATE TABLE order_focus AS SELECT orders_src.o_orderkey, orders_src.o_orderpriority, orders_src.o_totalprice, orders_src.o_comment, orders_src.o_orderdate, orders_src.o_custkey FROM orders_src "
            "WHERE NLfilter(orders_src.item, 'Does this order look operationally significant based on priority label, total price, and comment content? If uncertain, answer YES.');\n"
            "CREATE TABLE lineitem_filtered AS SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_quantity, l.l_extendedprice, l.l_discount, l.l_shipmode, l.l_shipdate FROM lineitem l "
            "WHERE l.l_shipdate >= DATE '1995-01-01' AND l.l_shipdate < DATE '1996-07-01' AND l.l_quantity BETWEEN 18 AND 26 AND l.l_discount BETWEEN 0.04 AND 0.07 AND l.l_shipmode IN ('AIR', 'REG AIR');\n"
            "CREATE TABLE joined AS SELECT of.o_orderkey, of.o_orderpriority, of.o_totalprice, of.o_comment, of.o_orderdate, of.o_custkey, lf.l_partkey, lf.l_suppkey, lf.l_quantity, lf.l_extendedprice, "
            "lf.l_discount, lf.l_shipmode FROM order_focus of JOIN lineitem_filtered lf ON lf.l_orderkey = of.o_orderkey;\n"
            f"CREATE TABLE customer_src AS SELECT c.c_custkey, c.c_name, c.c_address, c.c_nationkey, c.c_phone, c.c_acctbal, c.c_mktsegment, c.c_comment, {item(('Customer Name', 'c.c_name'), ('Market Segment', 'c.c_mktsegment'), ('Account Balance', 'c.c_acctbal'))} AS item "
            "FROM customer c JOIN (SELECT DISTINCT o_custkey FROM joined LIMIT 60) k ON k.o_custkey = c.c_custkey;\n"
            "CREATE TABLE pairs AS SELECT j.*, " + item(("Order Priority", "j.o_orderpriority"), ("Order Total Price", "j.o_totalprice"), ("Lineitem Ship Mode", "j.l_shipmode"), ("Customer Name", "cc.c_name"),
                                                        ("Market Segment", "cc.c_mktsegment"), ("Account Balance", "cc.c_acctbal"))
            + " AS vitem FROM joined j JOIN customer_src cc ON cc.c_custkey = j.o_custkey;\n",
            "SELECT " + ", ".join(f"pairs.{c}" for c in J28) + ", " + ", ".join(f"customer_src.{c}" for c in CUSTOMER_COLS) +
            " FROM pairs JOIN customer_src ON customer_src.c_custkey = pairs.o_custkey "
            "WHERE NLfilter(customer_src.item, 'Does this customer look operationally strategic based on name style, market segment, and account balance? If uncertain, answer YES.') "
            "AND NLfilter(pairs.vitem, 'Could this customer account be strategically important for order fulfillment? If uncertain, answer YES.')")
OUT29 = ["s_suppkey", "s_name", "s_address", "s_acctbal", "s_comment", "s_nationkey", "supplier_nation_name", "supplier_region_name", "ps_partkey", "ps_availqty", "ps_supplycost",
         "context_nationkey", "context_nation_name", "context_regionkey", "context_region_id", "context_region_name", "context_nation_comment", "context_region_comment"]
Q["Q29"] = (TPCH +
            "CREATE TABLE partsupp_base AS SELECT ps.ps_partkey, ps.ps_suppkey, ps.ps_availqty, ps.ps_supplycost FROM partsupp ps WHERE ps.ps_availqty BETWEEN 1600 AND 3200 AND ps.ps_supplycost BETWEEN 200 AND 420;\n"
            "CREATE TABLE supplier_base AS SELECT s.s_suppkey, s.s_name, s.s_address, s.s_acctbal, s.s_comment, s.s_nationkey, n.n_name AS supplier_nation_name, r.r_name AS supplier_region_name FROM supplier s "
            "JOIN nation n ON n.n_nationkey = s.s_nationkey JOIN region r ON r.r_regionkey = n.n_regionkey JOIN (SELECT DISTINCT ps_suppkey FROM partsupp_base LIMIT 70) k ON k.ps_suppkey = s.s_suppkey "
            "WHERE s.s_acctbal BETWEEN 2000 AND 8000 AND r.r_name IN ('EUROPE', 'ASIA', 'AMERICA');\n"
            "CREATE TABLE joined AS SELECT sb.s_suppkey, sb.s_name, sb.s_address, sb.s_acctbal, sb.s_comment, sb.s_nationkey, sb.supplier_nation_name, sb.supplier_region_name, pb.ps_partkey, pb.ps_availqty, pb.ps_supplycost "
            "FROM supplier_base sb JOIN partsupp_base pb ON pb.ps_suppkey = sb.s_suppkey;\n"
            "CREATE TABLE semantic_candidates AS SELECT j.s_suppkey, j.s_nationkey, MIN(j.s_acctbal) AS s_acctbal, MIN(j.ps_supplycost) AS ps_supplycost, MIN(j.supplier_region_name) AS supplier_region_name "
            "FROM joined j GROUP BY j.s_suppkey, j.s_nationkey LIMIT 50;\n"
            "CREATE TABLE nation_context AS SELECT ns.n_nationkey, ns.n_name, ns.n_regionkey, ns.n_comment, r.r_regionkey, r.r_name, r.r_comment FROM nation ns JOIN region r ON r.r_regionkey = ns.n_regionkey "
            "JOIN (SELECT DISTINCT s_nationkey FROM semantic_candidates ORDER BY s_nationkey LIMIT 5) k ON k.s_nationkey = ns.n_nationkey;\n"
            "CREATE TABLE pairs AS SELECT j.s_suppkey, j.s_name, j.s_address, j.s_acctbal, j.s_comment, j.s_nationkey, j.supplier_nation_name, j.supplier_region_name, j.ps_partkey, j.ps_availqty, j.ps_supplycost, "
            "nc.n_nationkey AS context_nationkey, nc.n_name AS context_nation_name, nc.n_regionkey AS context_regionkey, nc.r_regionkey AS context_region_id, nc.r_name AS context_region_name, "
            "nc.n_comment AS context_nation_comment, nc.r_comment AS context_region_comment, "
            + item(("Supplier Account Balance", "sc.s_acctbal"), ("Supply Cost", "sc.ps_supplycost"), ("Supplier Region", "sc.supplier_region_name"), ("Nation Name", "nc.n_name"), ("Region Name", "nc.r_name"))
            + " AS vitem FROM joined j JOIN semantic_candidates sc ON sc.s_suppkey = j.s_suppkey AND sc.s_nationkey = j.s_nationkey JOIN nation_context nc ON nc.n_nationkey = sc.s_nationkey;\n",
            "SELECT " + ", ".join(f"pairs.{c}" for c in OUT29) + " FROM pairs WHERE NLfilter(pairs.vitem, 'Does this sourcing region align with procurement priorities? If uncertain, answer YES.')")
J30 = ["p_partkey", "p_name", "p_type", "p_container", "l_orderkey", "l_suppkey", "l_quantity", "l_extendedprice", "l_discount", "l_shipmode", "o_orderdate", "o_orderpriority", "o_totalprice",
       "c_custkey", "c_name", "c_mktsegment", "c_acctbal"]
Q["Q30"] = (TPCH +
            "CREATE TABLE olc AS SELECT l.l_orderkey, l.l_partkey, l.l_suppkey, l.l_quantity, l.l_extendedprice, l.l_discount, l.l_shipmode, o.o_orderdate, o.o_orderpriority, o.o_totalprice, "
            "c.c_custkey, c.c_name, c.c_mktsegment, c.c_acctbal FROM lineitem l JOIN orders o ON o.o_orderkey = l.l_orderkey JOIN customer c ON c.c_custkey = o.o_custkey "
            "WHERE l.l_shipmode IN ('AIR', 'REG AIR', 'TRUCK') AND l.l_quantity BETWEEN 12 AND 30 AND l.l_discount BETWEEN 0.04 AND 0.07 AND o.o_orderdate >= DATE '1996-01-01' AND o.o_orderdate < DATE '1998-01-01' "
            "AND o.o_totalprice BETWEEN 60000 AND 170000 AND c.c_mktsegment IN ('FURNITURE', 'HOUSEHOLD', 'MACHINERY', 'BUILDING');\n"
            "CREATE TABLE part_src AS SELECT p.p_partkey, p.p_name, p.p_type, p.p_container, p.p_size, " + item(("Part Name", "p.p_name"), ("Part Type", "p.p_type"), ("Container", "p.p_container"))
            + " AS item FROM part p JOIN (SELECT DISTINCT l_partkey FROM olc LIMIT 200) k ON k.l_partkey = p.p_partkey WHERE p.p_size BETWEEN 5 AND 28 AND p.p_container IN ('SM BOX', 'MED BOX', 'LG BOX', 'SM CASE');\n"
            "CREATE TABLE part_focus AS SELECT part_src.p_partkey, part_src.p_name, part_src.p_type, part_src.p_container, part_src.p_size FROM part_src "
            "WHERE NLfilter(part_src.item, 'Does this part look suitable for customer-facing distribution based on name, type, and packaging? If uncertain, answer YES.');\n"
            "CREATE TABLE joined AS SELECT pf.p_partkey, pf.p_name, pf.p_type, pf.p_container, olc.l_orderkey, olc.l_suppkey, olc.l_quantity, olc.l_extendedprice, olc.l_discount, olc.l_shipmode, "
            "olc.o_orderdate, olc.o_orderpriority, olc.o_totalprice, olc.c_custkey, olc.c_name, olc.c_mktsegment, olc.c_acctbal FROM part_focus pf JOIN olc ON olc.l_partkey = pf.p_partkey;\n"
            "CREATE TABLE supplier_src AS SELECT s.s_suppkey, s.s_name, s.s_address, s.s_nationkey, s.s_phone, s.s_acctbal, s.s_comment, "
            + item(("Supplier Name", "s.s_name"), ("Account Balance", "s.s_acctbal"), ("Comment", "s.s_comment"))
            + " AS item FROM supplier s JOIN (SELECT DISTINCT l_suppkey FROM joined LIMIT 40) k ON k.l_suppkey = s.s_suppkey;\n"
            "CREATE TABLE pairs AS SELECT j.*, " + item(("Part Type", "j.p_type"), ("Ship Mode", "j.l_shipmode"), ("Customer Segment", "j.c_mktsegment"), ("Supplier Name", "sc.s_name"),
                                                        ("Account Balance", "sc.s_acctbal"), ("Comment", "sc.s_comment"))
            + " AS vitem FROM joined j JOIN supplier_src sc ON sc.s_suppkey = j.l_suppkey;\n",
            "SELECT " + ", ".join(f"pairs.{c}" for c in J30) + ", supplier_src.s_suppkey, supplier_src.s_name, supplier_src.s_address, supplier_src.s_nationkey, supplier_src.s_phone, "
            "supplier_src.s_acctbal, supplier_src.s_comment FROM pairs JOIN supplier_src ON supplier_src.s_suppkey = pairs.l_suppkey "
            "WHERE NLfilter(supplier_src.item, 'Does this supplier appear qualified for consumer-facing distribution partnerships? If uncertain, answer YES.') "
            "AND NLfilter(pairs.vitem, 'Would this supplier be suitable for retail distribution of this product? If uncertain, answer YES.')")


def unique_item_names(sql: str) -> str:
    """`item` -> `item_<table>` per table: ThalamusDB's rewriter references the filtered column unqualified, so two
    filtered tables in one query must not share a column name."""
    import re
    # statement by statement: an item text may itself contain ';' (the "; "-separated items), so the rename
    # must not stop at the first semicolon
    for stmt in re.split(r";[ \t]*\n", sql):
        m = re.match(r"\s*CREATE TABLE (\w+) AS SELECT.*? AS item FROM", stmt, re.S)
        if not m:
            continue
        table = m.group(1)
        new = re.sub(r"(CREATE TABLE \w+ AS SELECT.*?) AS item FROM", rf"\1 AS item_{table} FROM", stmt, count=1, flags=re.S)
        sql = sql.replace(stmt, new, 1)
        sql = sql.replace(f"NLfilter({table}.item,", f"NLfilter({table}.item_{table},")
    return sql


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, (setup, query) in Q.items():
        setup, query = unique_item_names(setup + "\x00" + query).split("\x00")
        header = (f"-- {name}: our ThalamusDB formulation of swan_queries/{name}.sql (see make_thalamusdb_queries.py for the rule).\n"
                  "-- Statements end with ';' at a line end; thalamusdb_exec.py runs the plain ones as setup, a CREATE TABLE ... AS <select with\n"
                  "-- NLfilter> through ThalamusDB as a stage, and the last statement as the query.\n")
        with open(os.path.join(OUT, name + ".sql"), "w") as f:
            f.write(header + setup + query + ";\n")
    print(f"wrote {len(Q)} queries to {OUT}")


if __name__ == "__main__":
    main()
