#!/bin/bash
# Pattern parity matrix: runs a family of CTE/join plan shapes (GROUP BY, LIMIT, LEFT JOIN,
# dual AI CTEs, DISTINCT, aggregates, UNION, nested CTEs, EXISTS, AI join conditions, 3-way,
# OFFSET) under the FULL default optimizer stack vs everything disabled, asserting no errors
# and byte-identical results under the deterministic mock. Bug-hunting tool for the
# hoist/push/binding machinery -- the shapes that caught the pullup projection-routing and
# push-below positional-map bugs (2026-09-21). Run from anywhere; ~1 min.
cd "$(dirname "$0")/.."
BIN=${DUCKDB_BIN:-build/release/duckdb}
SETUP="CALL ai_mock_start();
CREATE TABLE books AS SELECT 'b_'||range::VARCHAR AS book_id, 'title '||range::VARCHAR AS title,
 'sub '||range::VARCHAR AS subtitle, 'auth '||(range%3)::VARCHAR AS author,
 'cat '||(range%4)::VARCHAR AS categories FROM range(30);
CREATE TABLE reviews AS SELECT 'p_'||(range%30)::VARCHAR AS purchase_id, range AS rid,
 range % 5 AS rating, 'txt '||range::VARCHAR AS text, range % 2 AS verified FROM range(120);
"
OFF="SET ai_pullup=false; SET ai_factorize='off'; SET ai_semi_reduce=false; SET ai_reorder=false; SET ai_join_factorize='off'; SET ai_limit=false;"
CAND="WITH cand AS (SELECT CAST(SPLIT_PART(book_id,'_',2) AS INT) AS bidx, book_id, title FROM books
 WHERE ai_filter('[p=60] ok? '||title||' '||subtitle||' '||author||' '||categories)),
 rf AS (SELECT CAST(SPLIT_PART(purchase_id,'_',2) AS INT) AS pidx, rid, rating, text FROM reviews WHERE verified=1)"
NAMES=(groupby limit_above leftjoin two_ai_ctes distinct_consumer agg_consumer union_ctes nested_cte exists_sub ai_join_cond threeway offset_order)
QUERIES=(
"$CAND SELECT c.title, count(*) AS n FROM cand c JOIN rf ON rf.pidx=c.bidx GROUP BY c.title ORDER BY n DESC, c.title;"
"$CAND SELECT c.book_id, rf.rid FROM cand c JOIN rf ON rf.pidx=c.bidx ORDER BY rf.rid LIMIT 7;"
"$CAND SELECT c.book_id, rf.rid FROM cand c LEFT JOIN rf ON rf.pidx=c.bidx ORDER BY c.book_id, rf.rid;"
"WITH a AS (SELECT CAST(SPLIT_PART(book_id,'_',2) AS INT) AS bidx, title FROM books WHERE ai_filter('[p=70] x? '||title||' '||subtitle)),
 b AS (SELECT CAST(SPLIT_PART(purchase_id,'_',2) AS INT) AS pidx, text FROM reviews WHERE ai_filter('[p=70] y? '||text))
 SELECT a.title, b.text FROM a JOIN b ON b.pidx=a.bidx ORDER BY a.title, b.text;"
"$CAND SELECT DISTINCT c.title FROM cand c JOIN rf ON rf.pidx=c.bidx ORDER BY c.title;"
"$CAND SELECT sum(rf.rating) FROM cand c JOIN rf ON rf.pidx=c.bidx;"
"WITH a AS (SELECT title AS v FROM books WHERE ai_filter('[p=50] u? '||title||' '||subtitle)),
 b AS (SELECT text AS v FROM reviews WHERE ai_filter('[p=50] w? '||text))
 SELECT v FROM a UNION SELECT v FROM b ORDER BY v;"
"WITH inner_c AS (SELECT CAST(SPLIT_PART(book_id,'_',2) AS INT) AS bidx, title, author FROM books
   WHERE ai_filter('[p=60] n1? '||title||' '||subtitle)),
 outer_c AS (SELECT bidx, upper(title) AS ut FROM inner_c WHERE author LIKE 'auth%')
 SELECT o.ut, rf.rid FROM outer_c o JOIN (SELECT CAST(SPLIT_PART(purchase_id,'_',2) AS INT) AS pidx, rid FROM reviews WHERE verified=1) rf ON rf.pidx=o.bidx ORDER BY o.ut, rf.rid;"
"$CAND SELECT c.title FROM cand c WHERE EXISTS (SELECT 1 FROM rf WHERE rf.pidx=c.bidx AND rf.rating>=2) ORDER BY c.title;"
"SELECT b.title, r.rid FROM books b, reviews r
 WHERE ai_filter('[p=30] rel? '||b.title||' :: '||r.text) AND r.verified=1 AND b.author='auth 1' ORDER BY b.title, r.rid;"
"$CAND, agg AS (SELECT pidx, max(rating) AS mr FROM rf GROUP BY pidx)
 SELECT c.title, a.mr, u.k FROM cand c JOIN agg a ON a.pidx=c.bidx JOIN (SELECT range AS k FROM range(5)) u ON u.k = a.mr ORDER BY c.title, u.k;"
"$CAND SELECT c.book_id, rf.rid FROM cand c JOIN rf ON rf.pidx=c.bidx ORDER BY rf.rid DESC OFFSET 3 LIMIT 5;"
)
for i in "${!NAMES[@]}"; do
  Q="${QUERIES[$i]}"
  ON_OUT=$($BIN -csv -noheader -c "$SETUP $Q" 2>&1 | grep -vE "^[0-9]{4,5}$"); ON_ERR=$(echo "$ON_OUT" | grep -cE "INTERNAL|Error")
  OFF_OUT=$($BIN -csv -noheader -c "$SETUP $OFF $Q" 2>&1 | grep -v '^true$' | grep -vE "^[0-9]{4,5}$"); OFF_ERR=$(echo "$OFF_OUT" | grep -cE "INTERNAL|Error")
  if [ "$ON_ERR" != "0" ]; then echo "${NAMES[$i]}: ERROR(defaults): $(echo "$ON_OUT" | grep -E "Error" | head -1 | cut -c1-110)"
  elif [ "$OFF_ERR" != "0" ]; then echo "${NAMES[$i]}: ERROR(off): $(echo "$OFF_OUT" | grep -E "Error" | head -1 | cut -c1-110)"
  elif [ "$ON_OUT" != "$OFF_OUT" ]; then echo "${NAMES[$i]}: RESULT MISMATCH"
  else echo "${NAMES[$i]}: OK"
  fi
done
