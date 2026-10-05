"""Run one ThalamusDB script and print the result as JSON. Runs in ThalamusDB's own interpreter
(setup_thalamusdb.sh), so it imports nothing from this repository.

    python thalamusdb_exec.py DB_FILE MODELS_JSON DOP MAX_SECONDS [--plan] [--csv PATH] [--apply-limit] < script.sql

The script is one or more statements separated by `;` at a line end. A statement without a semantic
predicate runs as plain SQL (setup: loading tables, materialising derived columns). `CREATE TABLE t AS
<select with NLfilter/NLjoin>` runs the select through ThalamusDB and materialises its result as `t` (a
stage whose output a later statement reads). The last statement is the query whose rows are returned; with
a semantic predicate it runs through ThalamusDB, otherwise as plain SQL.

Each ThalamusDB execution is SemBench's `execute_thalamusdb_query` (src/runner/generic_thalamusdb_runner):
`Query` on the database, `ExecutionEngine.run` with the call and token caps lifted and SemBench's time cap
(6,000 s per execution: each stage of a multi-statement script is one), the error bound left at ThalamusDB's default of 0, and the returned frame taken as the answer,
LIMIT included in whatever way ThalamusDB handles it. ThalamusDB's progress output goes to stderr; stdout
carries one JSON object: {"columns", "rows", "llm_calls", "input_tokens", "output_tokens", "seconds"}.
Rows come from ThalamusDB's own rewritten SQL (`QueryRewriter.pure_sql` over the working tables its run
filled, with no unevaluated row counted as passing), evaluated by DuckDB: that keeps the query's column types
(decimals, dates) and duplicate rows, where ThalamusDB's returned frame is a pandas set; after a complete run
the two hold the same rows. `--csv PATH` also writes that result through DuckDB's COPY, so its formatting
matches the other systems' result files. ThalamusDB strips a top-level `LIMIT k` to drive its progress loop (it stops once k rows are
certain) and returns every certain row; `--apply-limit` keeps the first k of them, which is what the query
asked for (used on the hybrid bench, whose scorer requires exactly k rows; SemBench's suites are run as
SemBench ran them, without it). With --plan nothing is sent to the model: the output gives each operator's
pending work after ThalamusDB's own SQL pruning (a stage is then not materialised, so later statements are
not planned).
"""
import json
import math
import re
import sys
import time

_NL = re.compile(r"\bNL(?:filter|join)\s*\(", re.I)
_STAGE = re.compile(r"^\s*CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+(\"?[A-Za-z_][A-Za-z0-9_]*\"?)\s+AS\s+(.*)$", re.I | re.S)


def _py(value):
    if hasattr(value, "item"):  # numpy scalar
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is not None and type(value).__name__ in ("Timestamp", "NaTType", "Timedelta", "date", "datetime"):
        return None if str(value) == "NaT" else str(value)
    return value


def _statements(script: str) -> list[str]:
    parts = [p.strip() for p in re.split(r";[ \t]*\n", script)]
    return [p.rstrip(";").strip() for p in parts if p.strip() and p.strip() != ";"]


def main() -> None:
    db_path, config_path, dop, max_seconds = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    flags = sys.argv[5:]
    plan_only = "--plan" in flags
    csv_path = flags[flags.index("--csv") + 1] if "--csv" in flags else None
    apply_limit = "--apply-limit" in flags
    statements = _statements(sys.stdin.read())
    real_stdout = sys.stdout
    sys.stdout = sys.stderr

    from tdb.data.relational import Database
    from tdb.execution.constraints import Constraints
    from tdb.execution.engine import ExecutionEngine
    from tdb.queries.query import Query

    db = Database(db_path)
    engine = ExecutionEngine(db, dop, config_path)
    constraints = Constraints(max_calls=100000000000, max_seconds=max_seconds,
                              max_tokens=10000000000000000000000)  # SemBench's runner settings
    out = {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "pending_per_operator": [], "predicates": [],
           "columns": [], "rows": []}
    final_limit = [float("inf")]  # the query's LIMIT, applied under --apply-limit (ThalamusDB strips it)

    from tdb.queries.rewriter import QueryRewriter

    def semantic(sql, stage=False):
        """One ThalamusDB execution; returns the pure SQL of its result (None under --plan)."""
        query = Query(db, sql)
        if stage and query.limit != float("inf"):
            raise RuntimeError("a stage carries a LIMIT, which ThalamusDB would drop while materialising every certain row")
        if not stage:
            final_limit[0] = query.limit
        if not query.semantic_predicates:
            return sql
        out["predicates"] += [p.sql for p in query.semantic_predicates]
        if plan_only:
            operators = engine._create_operators(query)
            for op in operators:
                op.prepare()
            out["pending_per_operator"] += [op.counters.unprocessed_tasks for op in operators]
            return None
        _, counters = engine.run(query, constraints)
        out["llm_calls"] += counters.total_LLM_calls()
        out["input_tokens"] += counters.total_input_tokens()
        out["output_tokens"] += counters.total_output_tokens()
        # the operators the run created filled ThalamusDB_<operator id> working tables in this connection; the
        # same ids come back from _create_operators, so the rewriter can address them (no prepare(): that would
        # empty them). Unevaluated rows (a LIMIT stop) count as not passing, which is ThalamusDB's certain set.
        operators = engine._create_operators(query)
        return QueryRewriter(db, query).pure_sql({op: False for op in operators})

    t0 = time.time()
    result_sql = None
    try:
        for i, stmt in enumerate(statements):
            last = i == len(statements) - 1
            if not _NL.search(stmt):
                if last:
                    result_sql = None if plan_only else stmt
                else:
                    db.execute2list(stmt)
                continue
            m = _STAGE.match(stmt)
            if m and not last:
                stage_sql = semantic(m.group(2), stage=True)
                if stage_sql is None:
                    raise RuntimeError("plan: stage not materialised")
                db.execute2list(f"CREATE TABLE {m.group(1)} AS {stage_sql}")
            else:
                result_sql = semantic(stmt)
    except RuntimeError as ex:
        if not (plan_only and "plan:" in str(ex)):
            raise
    out["seconds"] = round(time.time() - t0, 2)
    if result_sql is not None:
        if apply_limit and final_limit[0] != float("inf"):
            result_sql = f"SELECT * FROM ({result_sql}) AS _tdb_limited LIMIT {int(final_limit[0])}"
        df = db.execute2df(result_sql)
        out["columns"] = [str(c) for c in df.columns]
        out["rows"] = [[_py(v) for v in row] for row in df.itertuples(index=False, name=None)]
        if csv_path:
            db.execute2list(f"COPY ({result_sql}) TO '{csv_path}' (HEADER, DELIMITER ',')")
    json.dump(out, real_stdout, default=str)
    real_stdout.write("\n")


if __name__ == "__main__":
    main()
