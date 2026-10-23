"""Fill cache.canon_key (the envelope-independent request key, see canon_for in ai_cache_server.py) for every
row recorded before the column existed. Run with the proxy STOPPED:  .venv/bin/python serve/migrate_canon_key.py
[path to .llm_cache.duckdb]. Idempotent: only rows with a NULL canon_key are touched."""
import hashlib, json, os, re, sys, tempfile
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
DB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, ".llm_cache.duckdb")
# the key functions, loaded from the server source so the two can never drift (importing the module would open
# the live store)
SRC = open(os.path.join(HERE, "ai_cache_server.py")).read()
NS = {"hashlib": hashlib, "json": json}
for name in ("_hash_data_url", "_merge_text_parts", "_carries_cache_fields", "_canon_user", "canon_for"):
    exec(re.search(r"^def %s\(.*?(?=^def |^[A-Z_]+ *=|^class )" % re.escape(name), SRC, re.S | re.M).group(0), NS)
canon_for = NS["canon_for"]

con = duckdb.connect(DB)
con.execute("ALTER TABLE cache ADD COLUMN IF NOT EXISTS canon_key VARCHAR")
todo = con.execute("SELECT count(*) FROM cache WHERE canon_key IS NULL").fetchone()[0]
print(f"rows without canon_key: {todo:,}")
if todo:
    tmp = os.path.join(tempfile.gettempdir(), "canon_keys.csv")
    n = done = 0
    with open(tmp, "w") as f:
        f.write("key,canon_key\n")
        cur = con.execute("SELECT key, request FROM cache WHERE canon_key IS NULL")
        while True:
            rows = cur.fetchmany(20000)
            if not rows:
                break
            for k, req in rows:
                n += 1
                try:
                    c = canon_for(json.loads(req))
                except Exception:
                    c = None
                if c:
                    done += 1
                    f.write(f"{k},{c}\n")
            if n % 200000 == 0:
                print(f"  {n:,} scanned, {done:,} keyed", flush=True)
    con.execute(f"CREATE TEMP TABLE ck AS SELECT * FROM read_csv('{tmp}', header=true, columns={{'key': 'VARCHAR', 'canon_key': 'VARCHAR'}})")
    con.execute("UPDATE cache SET canon_key = ck.canon_key FROM ck WHERE cache.key = ck.key")
    con.execute("DROP TABLE ck")
    os.remove(tmp)
    print(f"keyed {done:,} of {n:,} scanned (the rest are embeddings / System One / malformed)")
con.execute("CREATE INDEX IF NOT EXISTS cache_canon_idx ON cache(canon_key)")
print("distinct canon keys:", con.execute("SELECT count(DISTINCT canon_key) FROM cache").fetchone()[0], "| rows with one:", con.execute("SELECT count(*) FROM cache WHERE canon_key IS NOT NULL").fetchone()[0])
con.execute("CHECKPOINT"); con.close()
