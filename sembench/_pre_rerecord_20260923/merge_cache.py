"""Merge the back-to-back re-record into the main cache: migrate recorded_at, then UPSERT every
re-recorded key (fresh row replaces the old one). Keys the re-record never touched (agent_bench,
PLOP, gpt-5-mini runs, ...) are left exactly as they were, with recorded_at NULL = unknown."""
import duckdb, sys
main_db, fresh_db = sys.argv[1], sys.argv[2]
c = duckdb.connect(main_db)
c.execute("ALTER TABLE cache ADD COLUMN IF NOT EXISTS recorded_at TIMESTAMP")
c.execute(f"ATTACH '{fresh_db}' AS rr (READ_ONLY)")
before = c.execute("SELECT count(*) FROM cache").fetchone()[0]
fresh = c.execute("SELECT count(*) FROM rr.cache").fetchone()[0]
overlap = c.execute("SELECT count(*) FROM rr.cache f JOIN cache m USING (key)").fetchone()[0]
c.execute("""INSERT INTO cache (key, request, output, status, headers, latency, cost, recorded_at)
             SELECT key, request, output, status, headers, latency, cost, recorded_at FROM rr.cache
             ON CONFLICT (key) DO UPDATE SET request = excluded.request, output = excluded.output,
               status = excluded.status, headers = excluded.headers, latency = excluded.latency,
               cost = excluded.cost, recorded_at = excluded.recorded_at""")
after = c.execute("SELECT count(*) FROM cache").fetchone()[0]
# every re-recorded row must now be byte-identical in the main cache
mismatch = c.execute("""SELECT count(*) FROM rr.cache f JOIN cache m USING (key)
    WHERE m.output IS DISTINCT FROM f.output OR m.latency IS DISTINCT FROM f.latency
       OR m.recorded_at IS DISTINCT FROM f.recorded_at""").fetchone()[0]
missing = c.execute("SELECT count(*) FROM rr.cache f ANTI JOIN cache m USING (key)").fetchone()[0]
stamped = c.execute("SELECT count(*) FROM cache WHERE recorded_at IS NOT NULL").fetchone()[0]
c.execute("DETACH rr"); c.execute("CHECKPOINT"); c.close()
print(f"main before={before:,}  fresh rows={fresh:,}  replaced={overlap:,}  new={fresh-overlap:,}  main after={after:,}")
print(f"verify: mismatched={mismatch}  missing={missing}  rows with recorded_at={stamped:,} (must equal fresh rows)")
ok = mismatch == 0 and missing == 0 and after == before + (fresh - overlap) and stamped == fresh
print("MERGE OK" if ok else "MERGE CHECK FAILED"); sys.exit(0 if ok else 1)
