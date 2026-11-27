#!/usr/bin/env python3
"""The hybrid bench's SWAN and BlendSQL runners on stand-ins (no engine, no BlendSQL, no model): each records, per
query, how many of its requests the cache proxy had to send on to the provider -- llm_calls_fresh, the proxy's
miss-counter delta while the query ran, as every other runner records it -- and no count (null) when the endpoint has
no such counter, so that a run which did not go through the cache proxy never reads as "0 fresh".

Run: python3 -m pytest aisql-bench/AGENTBENCH/test_runners.py   (or python3 test_runners.py). Needs pandas, like the
BlendSQL runner. Each test copies the runner into a temporary directory, so nothing under results/ is read or written.
"""
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))

# stands in for the engine's shell: swallows the script, prints one result row and the usage row the runner asks for
FAKE_ENGINE = ("#!/bin/sh\ncat > /dev/null\n"
               "printf 'id\\n7\\n===USAGE===\\nchat,hits,tok,cost,hedged\\n3,1,40,0.00125,0\\n'\n")

# stands in for the blendsql package: one answer row, and the counters the runner reads from the result
FAKE_BLENDSQL = {
    "__init__.py": ("import pandas as pd\n"
                    "class _Meta:\n    num_generation_calls, prompt_tokens, completion_tokens = 4, 1000, 100\n"
                    "class _Smoothie:\n    df, meta = pd.DataFrame({'id': [7]}), _Meta()\n"
                    "class BlendSQL:\n"
                    "    def __init__(self, db, model=None, ingredients=None):\n        pass\n"
                    "    def execute(self, query):\n        return _Smoothie()\n"),
    "db.py": ("class _Con:\n    def sql(self, stmt):\n        pass\n"
              "class DuckDB:\n    con = _Con()\n"
              "    @classmethod\n    def from_pandas(cls, frames):\n        return cls()\n"),
    "ingredients.py": "class LLMMap:\n    pass\nclass LLMQA:\n    pass\n",
    "models.py": "class OpenAI:\n    def __init__(self, model, api_key=None, base_url=None):\n        pass\n",
}
BLENDSQL_QUERY = "-- setup (DuckDB backend)\nCREATE TABLE t AS SELECT 7 AS id;\n-- query (BlendSQL)\nSELECT id FROM t\n"


class Proxy:
    """A stand-in for the cache proxy's /cache/stats: each read answers the next miss count of `counts`."""

    def __init__(self, counts):
        counts = list(counts)

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                known = self.path == "/cache/stats" and bool(counts)
                body = json.dumps({"hits": 0, "misses": counts.pop(0)}).encode() if known else b"{}"
                self.send_response(200 if known else 404)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


def closed_port():
    """An address nothing listens on: an endpoint that is not the cache proxy."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"http://127.0.0.1:{s.getsockname()[1]}"


def sandbox(tmp, runner, query_dir, text):
    """A scratch aisql-bench/AGENTBENCH holding the runner and the two queries Q1 and Q2."""
    root = os.path.join(tmp, "aisql-bench", "AGENTBENCH")
    os.makedirs(os.path.join(root, query_dir))
    shutil.copy(os.path.join(HERE, runner), root)
    for q in ("Q1", "Q2"):
        with open(os.path.join(root, query_dir, q + ".sql"), "w") as f:
            f.write(text)
    return root


def records(root, runner, **env):
    """(the per-query records of the run tagged `t`, what the runner printed)."""
    base = {k: v for k, v in os.environ.items() if not k.startswith(("SWAN_", "AI_", "BLENDSQL_", "DUCKDB_"))}
    p = subprocess.run([sys.executable, runner, "Q1", "Q2"], cwd=root, capture_output=True, text=True,
                       env={**base, **env})
    assert p.returncode == 0, p.stderr
    with open(os.path.join(root, "results", "t_agentbench_results.json")) as f:
        return json.load(f)["per_query"], p.stdout


def swan_run(tmp, proxy_url):
    root = sandbox(tmp, "swan_agentbench.py", "swan_queries", "SELECT 7 AS id;\n")
    engine = os.path.join(tmp, "duckdb")
    with open(engine, "w") as f:
        f.write(FAKE_ENGINE)
    os.chmod(engine, 0o755)
    return records(root, "swan_agentbench.py", DUCKDB_BIN=engine, AI_PROXY_URL=proxy_url, SWAN_TAG="t")


def blendsql_run(tmp, proxy_url):
    root = sandbox(tmp, "blendsql_agentbench.py", "blendsql_queries", BLENDSQL_QUERY)
    package = os.path.join(tmp, "fake", "blendsql")
    os.makedirs(package)
    for name, text in FAKE_BLENDSQL.items():
        with open(os.path.join(package, name), "w") as f:
            f.write(text)
    return records(root, "blendsql_agentbench.py", PYTHONPATH=os.path.dirname(package), AI_PROXY_URL=proxy_url,
                   BLENDSQL_TAG="t")


def test_the_swan_runner_records_the_calls_new_to_the_cache():
    # the proxy's miss counter read before and after each query: 5 -> 7 while Q1 ran, 7 -> 7 while Q2 ran
    with tempfile.TemporaryDirectory() as tmp, Proxy([5, 7, 7, 7]) as proxy:
        recs, out = swan_run(tmp, proxy.url)
        assert recs["Q1"]["llm_calls_fresh"] == 2 and recs["Q2"]["llm_calls_fresh"] == 0, recs
        assert recs["Q1"]["llm_calls"] == 3 and recs["Q1"]["rows"] == 1 and recs["Q1"]["error"] is None, recs
        assert "fresh=     2" in out and "fresh=     0" in out, out


def test_the_swan_runner_records_no_count_without_the_proxys_counter():
    with tempfile.TemporaryDirectory() as tmp:
        recs, out = swan_run(tmp, closed_port())
        assert recs["Q1"]["llm_calls_fresh"] is None and recs["Q2"]["llm_calls_fresh"] is None, recs
        assert recs["Q1"]["llm_calls"] == 3 and recs["Q1"]["error"] is None and "fresh=  None" in out, (recs, out)


def test_the_blendsql_runner_records_the_calls_new_to_the_cache():
    with tempfile.TemporaryDirectory() as tmp, Proxy([10, 13, 13, 13]) as proxy:
        recs, out = blendsql_run(tmp, proxy.url)
        assert recs["Q1"]["llm_calls_fresh"] == 3 and recs["Q2"]["llm_calls_fresh"] == 0, recs
        assert recs["Q1"]["llm_calls"] == 4 and recs["Q1"]["rows"] == 1 and recs["Q1"]["error"] is None, recs
        assert "fresh=     3" in out and "fresh=     0" in out, out


def test_the_blendsql_runner_records_no_count_without_the_proxys_counter():
    with tempfile.TemporaryDirectory() as tmp:
        recs, _ = blendsql_run(tmp, closed_port())
        assert recs["Q1"]["llm_calls_fresh"] is None and recs["Q1"]["llm_calls"] == 4 and recs["Q1"]["error"] is None, recs


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
