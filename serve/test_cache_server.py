#!/usr/bin/env python3
"""Self-test for ai_cache_server.py: schema migration, recording with recorded_at, and replay.

Starts a fake upstream and a proxy on a throwaway store created with the PRE-recorded_at schema
(the shape of a real cache that predates the column), then checks that
  1. the store is migrated in place and its legacy row keeps recorded_at NULL (unknown, never invented);
  2. a fresh call is recorded with a recorded_at close to now (UTC);
  3. the identical call replays from the store (no second upstream hit, no duplicate row).
No network beyond localhost, no real LLM.   run:  python3 serve/test_cache_server.py
"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
UP_PORT, PROXY_PORT = 4098, 4097
upstream_hits = []

class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        upstream_hits.append(self.path)
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        body = json.dumps({"choices": [{"message": {"content": "OK"}}],
                           "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("x-litellm-response-cost", "0.000123")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

def main():
    tmp = tempfile.mkdtemp(); db = os.path.join(tmp, "legacy.duckdb")
    con = duckdb.connect(db)
    con.execute("""CREATE TABLE cache(key VARCHAR PRIMARY KEY, request VARCHAR, output VARCHAR, status INTEGER,
                   headers VARCHAR, latency DOUBLE, cost DOUBLE)""")
    con.execute("INSERT INTO cache VALUES ('legacy','{}','{}',200,'{}',1.5,0.001)"); con.close()

    up = ThreadingHTTPServer(("127.0.0.1", UP_PORT), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    env = {**os.environ, "CACHE_PROXY_PORT": str(PROXY_PORT), "CACHE_DB": db,
           "CACHE_UPSTREAM": f"http://127.0.0.1:{UP_PORT}", "CACHE_SIMULATE_LATENCY": "0"}
    proxy = subprocess.Popen([sys.executable, os.path.join(HERE, "ai_cache_server.py")], env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for _ in range(50):
            try: urllib.request.urlopen(f"http://127.0.0.1:{PROXY_PORT}/cache/stats", timeout=1); break
            except Exception: time.sleep(0.1)
        req = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}]}).encode()
        def call():
            r = urllib.request.Request(f"http://127.0.0.1:{PROXY_PORT}/v1/chat/completions", data=req,
                                       headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(r, timeout=10))["choices"][0]["message"]["content"]
        before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
        assert call() == "OK" and call() == "OK"
        assert len(upstream_hits) == 1, f"replay hit the upstream: {upstream_hits}"
    finally:
        proxy.terminate(); proxy.wait(timeout=10); up.shutdown()
    c = duckdb.connect(db, read_only=True)
    cols = [r[0] for r in c.execute("SELECT column_name FROM information_schema.columns WHERE table_name='cache' ORDER BY ordinal_position").fetchall()]
    assert cols[-1] == "recorded_at", cols
    rows = {k: (rec, lat) for k, rec, lat in c.execute("SELECT key, recorded_at, latency FROM cache").fetchall()}
    assert len(rows) == 2, rows                                  # legacy + exactly one new row
    assert rows["legacy"][0] is None, "legacy row must keep recorded_at NULL"
    (new_rec, _), = [v for k, v in rows.items() if k != "legacy"]
    assert new_rec is not None and abs((new_rec - before).total_seconds()) < 60, new_rec
    print("cache server self-test: migration, recorded_at stamping, replay -- OK")

if __name__ == "__main__":
    main()
