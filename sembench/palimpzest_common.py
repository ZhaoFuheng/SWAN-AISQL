"""Shared pieces of the Palimpzest runners (MOVIE/palimpzest_movie.py, ECOMM/palimpzest_ecomm.py,
MMQA/palimpzest_mmqa.py): where Palimpzest's interpreter is, one query execution through palimpzest_exec.py,
and the accounting.

Palimpzest (PyPI `palimpzest`, MIT DSG; the Abacus cost-based optimizer is its query optimizer) runs in its own
interpreter (`setup_palimpzest.sh` creates `sembench/.venv-palimpzest`; `PALIMPZEST_PYTHON` names another).
It is run with Abacus ON (the current release's optimizer; SemBench ran 0.8.2 with it off) under SemBench's
other settings (MaxQuality, parallel execution, 20 workers). The benchmark model goes in as a self-hosted
model through the cache proxy, which Palimpzest prices at zero, so calls and tokens are the proxy's counter
deltas during the query (run the suites one at a time through a proxy nothing else is using) and cost is
estimated from those tokens at the gpt-5.6 list price, as for PLOP, BlendSQL and ThalamusDB. Abacus's own
sampling calls are part of the query's calls, as its optimization is part of answering the query.
"""
import json
import os
import subprocess
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXEC = os.path.join(HERE, "palimpzest_exec.py")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
DOP = int(os.environ.get("AI_MAX_CONCURRENCY", "20"))
POLICY = os.environ.get("PALIMPZEST_POLICY", "MaxQuality")
OPTIMIZER = os.environ.get("PALIMPZEST_OPTIMIZER", "pareto")  # Abacus on; "none" = SemBench's setting
P_IN, P_OUT = 0.25, 2.00  # gpt-5.6 list price per 1M tokens (input / output); estimate only


def python() -> str:
    for c in (os.environ.get("PALIMPZEST_PYTHON"), os.path.join(HERE, ".venv-palimpzest", "bin", "python")):
        if c and os.path.isfile(c):
            return c
    raise SystemExit("Palimpzest's interpreter is missing: run sembench/setup_palimpzest.sh or set PALIMPZEST_PYTHON")


def proxy_counters():
    """(requests, misses, prompt tokens, completion tokens) from the proxy's running counters."""
    try:
        s = json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))
        return s["hits"] + s["misses"], s["misses"], s.get("prompt_tokens", 0), s.get("completion_tokens", 0)
    except Exception:
        return 0, -1, 0, 0


def run_query(module_path: str, function: str, data_dir: str, timeout: int = 7200) -> dict:
    """One Palimpzest execution; returns the exec script's JSON (its `seconds` is the query's own time inside
    the interpreter, after the ~10 s import of palimpzest/torch) plus `error`, `llm_calls`, `llm_calls_fresh`,
    `tokens` and `cost_usd` (proxy deltas; estimate)."""
    r0, m0, p0, c0 = proxy_counters()
    cmd = [python(), EXEC, module_path, function, data_dir, MODEL, PROXY, str(DOP), "--policy", POLICY, "--optimizer", OPTIMIZER]
    try:
        p = subprocess.run(cmd, input="", text=True, capture_output=True, timeout=timeout, cwd=HERE)
    except subprocess.TimeoutExpired:  # Palimpzest has no cap of its own; record the overrun, keep the suite going
        p = None
    r1, m1, p1, c1 = proxy_counters()
    if p is None:
        out = {"error": f"timeout after {timeout} s", "rows": [], "columns": [], "stats": {}, "seconds": float(timeout)}
    elif p.returncode != 0 or not p.stdout.strip():
        tail = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
        out = {"error": (tail[-1] if tail else f"exit {p.returncode}")[:300], "rows": [], "columns": [], "stats": {}, "seconds": 0.0}
    else:
        out = json.loads(p.stdout)
        out["error"] = None
    out["llm_calls"] = r1 - r0
    out["llm_calls_fresh"] = m1 - m0
    out["tokens"] = (p1 - p0) + (c1 - c0)
    out["cost_usd"] = round((p1 - p0) / 1e6 * P_IN + (c1 - c0) / 1e6 * P_OUT, 5)
    return out


def cell(v, integral_as_int: bool = True) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and integral_as_int and v.is_integer():
        return str(int(v))
    return str(v)
