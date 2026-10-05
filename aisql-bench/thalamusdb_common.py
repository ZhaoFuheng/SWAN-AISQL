"""Shared pieces of the ThalamusDB runners (MOVIE/thalamusdb_movie.py, ECOMM/thalamusdb_ecomm.py,
MMQA/thalamusdb_mmqa.py, AGENTBENCH/thalamusdb_agentbench.py): where ThalamusDB's interpreter is, the model configuration that routes it through
the cache proxy, a copy of each suite's database that ThalamusDB's DuckDB can open, and one query execution.

ThalamusDB (PyPI `thalamusdb`, the version SemBench evaluated) pins its dependencies and its DuckDB cannot
open this repository's database files, so it runs in its own interpreter (`setup_thalamusdb.sh` creates
`aisql-bench/.venv-thalamusdb`; `THALAMUSDB_PYTHON` names another) on a copy of each suite's tables built from a
parquet export (`<suite>/thalamusdb.duckdb`, rebuilt when missing). Calls and tokens come from ThalamusDB's
own counters; cost is estimated from the tokens at the gpt-5.6 list price, as for PLOP and BlendSQL (its
requests carry no provider cost header through litellm). Fresh calls are the proxy's miss delta.
"""
import json
import os
import re
import subprocess
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
EXEC = os.path.join(HERE, "thalamusdb_exec.py")
PROXY = os.environ.get("AI_PROXY_URL", "http://localhost:4001")
MODEL = os.environ.get("AI_MODEL", "gpt-5.6-luna")
DOP = int(os.environ.get("AI_MAX_CONCURRENCY", "20"))
MAX_SECONDS = int(os.environ.get("THALAMUSDB_MAX_SECONDS", "6000"))  # SemBench's per-query cap
P_IN, P_OUT = 0.25, 2.00  # gpt-5.6 list price per 1M tokens (input / output); estimate only
_BUILDS = [os.path.abspath(os.path.join(HERE, "..", "build", b, "duckdb")) for b in ("release", "reldebug")]
DUCKDB_BIN = os.environ.get("DUCKDB_BIN", next((p for p in _BUILDS if os.path.exists(p)), _BUILDS[0]))


def python() -> str:
    """ThalamusDB's interpreter: THALAMUSDB_PYTHON, else aisql-bench/.venv-thalamusdb, else SWAN_bench's."""
    candidates = [os.environ.get("THALAMUSDB_PYTHON"),
                  os.path.join(HERE, ".venv-thalamusdb", "bin", "python"),
                  os.path.join(HERE, "..", "..", "SWAN_bench", ".venv-thalamusdb", "bin", "python")]
    for c in candidates:
        if c and os.path.isfile(c):
            return c
    raise SystemExit("ThalamusDB's interpreter is missing: run aisql-bench/setup_thalamusdb.sh or set THALAMUSDB_PYTHON")


def models_json() -> str:
    """ThalamusDB's model configuration: the benchmark model through the proxy, temperature 0, for text and images."""
    lm_model = MODEL if "/" in MODEL else f"openai/{MODEL}"
    # drop_params: litellm's client-side check rejects temperature=0 for gpt-5-family model names; ThalamusDB's
    # filter path sets litellm.drop_params itself, its join path does not, so the configuration carries it
    call = {"model": lm_model, "api_base": PROXY.rstrip("/") + "/v1", "api_key": "sk-test", "temperature": 0, "drop_params": True}
    path = os.path.join(tempfile.mkdtemp(prefix="thalamusdb_"), "models.json")
    with open(path, "w") as f:
        json.dump({"models": [{"modalities": ["text", "image"], "priority": 10,
                               "kwargs": {"filter": call, "join": call}}]}, f, indent=1)
    return path


def build_db(suite_dir: str, source_db: str, tables: list[str], create_sql: str, out_name: str = "thalamusdb.duckdb") -> str:
    """A DuckDB file ThalamusDB can open: `tables` of `source_db` exported to parquet with this repository's
    binary, then created by `create_sql` (which reads `<table>.parquet` from the export directory, available
    as the SQL variable `$export`) in ThalamusDB's interpreter. Built once; delete the file to rebuild."""
    out = os.path.join(suite_dir, out_name)
    if os.path.exists(out):
        return out
    export = os.path.join(suite_dir, "thalamusdb_export")
    os.makedirs(export, exist_ok=True)
    script = "\n".join(f"COPY (SELECT * FROM \"{t}\") TO '{os.path.join(export, t + '.parquet')}' (FORMAT PARQUET);" for t in tables)
    subprocess.run([DUCKDB_BIN, "-readonly", "-bail", source_db], input=script, text=True, capture_output=True, check=True)
    create = create_sql.replace("$export", export.replace("'", "''"))
    prog = ("import duckdb, sys\ncon = duckdb.connect(sys.argv[1])\n"
            "for stmt in sys.argv[2].split(';\\n'):\n    stmt = stmt.strip()\n    con.execute(stmt) if stmt else None\ncon.close()\n")
    subprocess.run([python(), "-c", prog, out, create], check=True, capture_output=True, text=True)
    return out


def proxy_stats():
    try:
        s = json.load(urllib.request.urlopen(PROXY + "/cache/stats", timeout=5))
        return s["hits"] + s["misses"], s["misses"]
    except Exception:
        return 0, -1


def run_query(db_path: str, config_path: str, sql: str, plan: bool = False, timeout: int | None = None,
              extra_args: tuple = ()) -> dict:
    """Run one script through the exec script; returns its JSON plus `error` (None on success).

    MAX_SECONDS is ThalamusDB's cap per execution, and a multi-statement script runs one execution per
    statement with a semantic predicate (each stage, then the query), so the wrapper's timeout is sized
    from that count; a script that still overruns is recorded as an error, not raised."""
    n_exec = sum(1 for stmt in re.split(r";[ \t]*\n", sql) if re.search(r"\bNL(?:filter|join)\s*\(", stmt, re.I))
    if timeout is None:
        timeout = (n_exec + 1) * MAX_SECONDS + 300
    cmd = [python(), EXEC, db_path, config_path, str(DOP), str(MAX_SECONDS)] + (["--plan"] if plan else []) + list(extra_args)
    try:
        p = subprocess.run(cmd, input=sql, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout} s", "rows": [], "columns": [],
                "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": float(timeout)}
    if p.returncode != 0 or not p.stdout.strip():
        tail = [ln for ln in p.stderr.strip().splitlines() if ln.strip()]
        return {"error": (tail[-1] if tail else f"exit {p.returncode}")[:300], "rows": [], "columns": [],
                "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0.0}
    out = json.loads(p.stdout)
    out["error"] = None
    return out


def cost_estimate(out: dict) -> float:
    return round(out.get("input_tokens", 0) / 1e6 * P_IN + out.get("output_tokens", 0) / 1e6 * P_OUT, 5)


def cell(v, integral_as_int: bool = True) -> str:
    """A result cell as the CSV-style string the suite scorers compare (ThalamusDB returns pandas values)."""
    if v is None:
        return ""
    if isinstance(v, float) and integral_as_int and v.is_integer():
        return str(int(v))
    return str(v)


def load_queries(query_dir: str) -> dict:
    """{name: sql} for every qN.sql in the directory, in query order."""
    names = sorted((f[:-4] for f in os.listdir(query_dir) if f.endswith(".sql")),
                   key=lambda n: (int("".join(ch for ch in n if ch.isdigit()) or 0), n))
    return {n: open(os.path.join(query_dir, n + ".sql")).read() for n in names}
