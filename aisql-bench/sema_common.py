#!/usr/bin/env python3
"""What the two Sema runners (MOVIE/sema_movie.py, AGENTBENCH/sema_agentbench.py) share.

A run is Sema as its authors ship it, on the build setup_sema.sh makes, and a results file says so:
  - every shell starts without ~/.duckdbrc (-init /dev/null) and gets SET threads=1 and nothing else of Sema's;
  - `check_build` stops a run whose binary lacks sema_fork.patch (it would ignore the request cap and timeout the
    run records) or whose semantic settings are not the shipped defaults;
  - `configuration` is what a results file records and `incompatible` why a saved one cannot take this run's
    queries: a file holds one configuration, and one that does not say is not assumed to match;
  - each query's record carries the hash of the query text it ran, so rows of an older text are told apart.
Sema puts false or an empty string where a model answer is missing and still exits 0, so a query during which a
model request failed is recorded as failed (`query_error`): a request that failed may have been retried and answered,
but when two rows send the same text a call that failed for good cannot be told from a retried one.
"""
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))  # the folder that holds this checkout and Sema's
sys.path.insert(0, HERE)
from swanbench_meter import Meter  # noqa: E402,F401 -- SWANBench's meter (stdlib only); the runners take it from here

BIN = os.environ.get("SEMA_BIN", os.path.join(ROOT, "Sema", "build", "release", "duckdb"))
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
CONC = os.environ.get("AI_MAX_CONCURRENCY", "20")
TIMEOUT_S = os.environ.get("SEMA_LLM_TIMEOUT_S", "1800")
QUERY_TIMEOUT_S = int(os.environ.get("SEMA_QUERY_TIMEOUT_S", "43200"))
ENV = {**os.environ, "DUCKDB_AGENT_MODE": "0", "SEMA_LLM_MAX_CONNECTIONS": CONC, "SEMA_LLM_TIMEOUT_S": TIMEOUT_S}
NO_RC = ("-init", os.devnull)  # no ~/.duckdbrc: the only settings a shell gets are the ones a runner writes
THREADS = "SET threads=1;\n"  # the patch caps the requests of one thread, so one thread makes the cap the run's

#: Sema's semantic settings as shipped (src/include/duckdb/main/config.hpp at the pinned commit), as its shell prints them
DEFAULTS = {"enable_semantic_filter_multiplexer": "false", "semantic_batch_size": "1", "semantic_voting_rounds": "1",
            "semantic_filter_batch_size": "4", "semantic_filter_voting_rounds": "1",
            "semantic_filter_accuracy_threshold": "0.95", "semantic_filter_latency_first": "true"}
#: what a session reports when it is set up as a query's is: the defaults, on one thread
SETTINGS = {**DEFAULTS, "threads": "1"}
PATCH_MARKERS = (b"SEMA_LLM_MAX_CONNECTIONS", b"SEMA_LLM_TIMEOUT_S")  # the two variables sema_fork.patch reads


def shell(script, *args, cwd, binary=None, timeout=None):
    """One run of a DuckDB shell on `script`: Sema's, unless `binary` names another."""
    return subprocess.run([binary or BIN, *NO_RC, *args], input=script, text=True, capture_output=True, env=ENV,
                          cwd=cwd, timeout=timeout)


def session(meter_url):
    """The lines every query starts with: where the model is, and one thread."""
    model = MODEL.replace("'", "''")
    return (".bail on\n"
            f"SET llm_url='{meter_url}/v1/chat/completions';\nSET llm_model='{model}';\nSET llm_api_key='sk-test';\n"
            + THREADS)


def reported_settings(cwd):
    """The semantic settings (and thread count) Sema reports in a session set up the way a query's is."""
    p = shell(".bail on\n" + THREADS + "SELECT name, value FROM duckdb_settings() "
              "WHERE name LIKE '%semantic%' OR name = 'threads' ORDER BY name;\n", "-csv", "-noheader", cwd=cwd)
    rows = {r[0]: r[1] for r in csv.reader(io.StringIO(p.stdout)) if len(r) == 2}
    if p.returncode != 0 or not rows:
        raise SystemExit("could not read Sema's settings: " + (p.stderr.strip()[-300:] or "no output"))
    return rows


def check_build(cwd):
    """The settings of a run on this binary; stops unless it is the patched build at Sema's shipped defaults."""
    if not os.path.isfile(BIN):
        raise SystemExit(f"no Sema shell at {BIN}: run aisql-bench/setup_sema.sh, or set SEMA_BIN")
    with open(BIN, "rb") as f:
        binary = f.read()
    if not all(marker in binary for marker in PATCH_MARKERS):
        raise SystemExit(f"{BIN} was built without aisql-bench/sema_fork.patch: it would keep Sema's own 50 requests "
                         f"in flight and 30 s request timeout, not the {CONC} and {TIMEOUT_S} s this run records")
    got = reported_settings(cwd)
    if got != SETTINGS:
        other = ", ".join(f"{k} = {got.get(k, 'missing')} (shipped: {SETTINGS.get(k, 'no such setting')})"
                          for k in sorted(set(got) | set(SETTINGS)) if got.get(k) != SETTINGS.get(k))
        raise SystemExit(f"Sema is compared at its shipped default settings; {BIN} reports {other}")
    return got


def configuration(settings=None):
    """What a results file records about the run that wrote it (the settings once the build is checked)."""
    config = {"model": MODEL, "concurrency": CONC, "request_timeout_s": TIMEOUT_S}
    return config if settings is None else {**config, "settings": settings}


def incompatible(saved, config):
    """Why a saved results file cannot take this run's queries, or None. Every part of `config` must be recorded
    there and equal; a file with no query yet takes any run."""
    if not saved.get("per_query"):
        return None
    for key, now in config.items():
        if key not in saved:
            return f"it does not record its {key} (an earlier version of the runner wrote it)"
        if saved[key] != now:
            return f"its {key} is {saved[key]}, this run's is {now}"
    return None


def proxy_replays_latency():
    """Whether the cache proxy returns recorded answers after their recorded response time (its /cache/stats
    `simulate_latency`); None when the endpoint does not say (not the cache proxy)."""
    try:
        with urllib.request.urlopen(PROXY.rstrip("/") + "/cache/stats", timeout=10) as r:
            return json.load(r).get("simulate_latency")
    except Exception:
        return None


def timing(fresh, replays_latency):
    """How a query's latency was measured: "live" when some of its calls were new to the cache (asked of the
    provider during the run), "replay" when the cache answered every call after its recorded response time."""
    if fresh is None:
        return "not through the cache proxy"
    if fresh:
        return "live"
    return {True: "replay", False: "replay without recorded latency"}.get(replays_latency, "replay, latency mode unknown")


def run_query(script, *args, cwd):
    """(the finished shell, or None when the query ran past SEMA_QUERY_TIMEOUT_S; seconds)."""
    start = time.time()
    try:
        p = shell(script, *args, cwd=cwd, timeout=QUERY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        p = None
    return p, time.time() - start


def query_error(p, usage):
    """Why a query's output may not be Sema's answer, or None: it timed out, the shell failed, or a model request
    failed (Sema puts false or an empty string where an answer is missing and exits 0). Any failed request counts:
    the meter's `unanswered` sees only the texts no request got an answer for, and a failed call whose text another
    row also sent, and got answered, is not among them."""
    if p is None:
        return f"timeout after {QUERY_TIMEOUT_S} s"
    if p.returncode != 0 or "Error" in p.stderr:
        return "sql: " + ((p.stderr.strip().splitlines() or ["nonzero exit"])[-1])[:300]
    if usage["errors"]:
        return f"{usage['errors']} model request(s) failed ({usage['unanswered']} request text(s) never got an answer)"
    return None


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write(path, text):
    """Replace `path` in one step: an interrupted run leaves the earlier file or the new one, never half of one."""
    part = f"{path}.tmp{os.getpid()}"
    with open(part, "w") as f:
        f.write(text)
    os.replace(part, path)
