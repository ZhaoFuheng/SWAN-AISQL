#!/usr/bin/env python3
"""The Sema port (sema_common.py, MOVIE/sema_movie.py, AGENTBENCH/sema_agentbench.py,
AGENTBENCH/translate_to_sema.py), checked without Sema, a proxy or a model:

  - the runners apply and offer nothing of Sema's: a run can only be Sema at its shipped default settings, on the
    patched build, and stops before it writes anything otherwise
  - a results file is one configuration: it takes more queries only from a run that it records as its own
  - a query whose shell failed, that timed out, or during which a model request failed is a failed one
  - how a query's latency was measured follows from the calls that were new to the cache and the proxy's mode
  - AGENTBENCH/sema_queries/ is what the translator writes from plop_queries/: every prompt carried over, a NULL
    prompt column rendered as PLOP renders it, an integer answer asked for and read as PLOP does

Run: python3 -m pytest aisql-bench/test_sema.py   (or python3 test_sema.py). Standard library only; the check of
the translator's table list against the data needs the duckdb module and is skipped without it.
"""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
HYBRID = os.path.join(HERE, "AGENTBENCH", "sema_agentbench.py")
MOVIE = os.path.join(HERE, "MOVIE", "sema_movie.py")
COMMON = os.path.join(HERE, "sema_common.py")
TRANSLATOR = os.path.join(HERE, "AGENTBENCH", "translate_to_sema.py")


def load(path):
    """A script as a module (its main() is not run); its own folder is importable, as when it is run."""
    sys.path.insert(0, os.path.dirname(path))
    try:
        spec = importlib.util.spec_from_file_location(os.path.basename(path)[:-3] + "_under_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(os.path.dirname(path))


def fake_shell(folder, settings, patched=True):
    """A stand-in for Sema's shell that prints `settings` as the real one prints its own."""
    path = os.path.join(folder, "duckdb")
    marker = "# SEMA_LLM_MAX_CONNECTIONS SEMA_LLM_TIMEOUT_S\n" if patched else ""
    with open(path, "w") as f:
        f.write("#!/usr/bin/env python3\n" + marker + f"import sys\nsys.stdin.read()\nprint({settings!r})\n")
    os.chmod(path, 0o755)
    return path


def as_csv(settings):
    return "\n".join(f"{k},{v}" for k, v in sorted(settings.items()))


def stops(call, *said):
    try:
        call()
    except SystemExit as e:
        assert all(text in str(e) for text in said), str(e)
        return
    raise AssertionError("the run went on")


def test_the_runners_set_nothing_of_semas():
    for path in (COMMON, HYBRID, MOVIE):
        src = open(path).read()
        assert set(re.findall(r"\bSET (\w+)\s*=", src)) <= {"llm_url", "llm_model", "llm_api_key", "threads"}, path
        assert not re.search(r"SEMA_BATCH|SEMA_EXTRA_SET|semantic_\w+\s*=", src), path
    # every shell goes through sema_common.shell, which starts it without ~/.duckdbrc (that file could set anything)
    assert '[binary or BIN, *NO_RC, *args]' in open(COMMON).read() and 'NO_RC = ("-init", os.devnull)' in open(COMMON).read()
    for path in (HYBRID, MOVIE):
        assert "subprocess" not in open(path).read(), path


def test_a_run_needs_the_patched_build_at_semas_shipped_settings():
    S = load(COMMON)
    with tempfile.TemporaryDirectory() as tmp:
        S.BIN = os.path.join(tmp, "nowhere")
        stops(lambda: S.check_build(tmp), "no Sema shell")
        S.BIN = fake_shell(tmp, as_csv(S.SETTINGS))
        assert S.check_build(tmp) == S.SETTINGS
        S.BIN = fake_shell(tmp, as_csv(S.SETTINGS), patched=False)  # it would ignore the request cap and timeout
        stops(lambda: S.check_build(tmp), "built without", "sema_fork.patch")
        for changed in ({"semantic_batch_size": "4"}, {"enable_semantic_filter_multiplexer": "true"}, {"threads": "8"},
                        {"semantic_something_new": "1"}):
            S.BIN = fake_shell(tmp, as_csv({**S.SETTINGS, **changed}))
            stops(lambda: S.check_build(tmp), "shipped default settings", next(iter(changed)))
    assert set(S.SETTINGS) == set(S.DEFAULTS) | {"threads"} and len(S.DEFAULTS) == 7


def test_a_results_file_is_one_configuration():
    S = load(COMMON)
    config = S.configuration(dict(S.SETTINGS))
    assert set(config) == {"model", "concurrency", "request_timeout_s", "settings"}
    saved = {**config, "per_query": {"Q1": {}}}
    assert S.incompatible(saved, config) is None
    assert S.incompatible({"model": "another", "per_query": {}}, config) is None  # no query yet: any run may start it
    for key in config:
        other = {**saved, key: "another"}
        assert key in S.incompatible(other, config), key
        missing = {k: v for k, v in saved.items() if k != key}  # what a file does not say is not assumed to match
        assert "does not record its " + key in S.incompatible(missing, config), key
    earlier = {"model": config["model"], "workers": "20", "batch": "4", "per_query": {"Q1": {}}}
    assert "does not record its concurrency" in S.incompatible(earlier, S.configuration())


def _refused(runner, folder, saved_name, saved, argv):
    """Run a copy of `runner` next to a saved results file of another configuration: (process, files unchanged)."""
    with tempfile.TemporaryDirectory() as tmp:
        bench = os.path.join(tmp, "aisql-bench", folder)
        os.makedirs(os.path.dirname(os.path.join(bench, saved_name)))
        for f in (COMMON, os.path.join(HERE, "swanbench_meter.py")):
            shutil.copy(f, os.path.join(tmp, "aisql-bench"))
        shutil.copy(runner, bench)
        if folder == "MOVIE":
            shutil.copy(os.path.join(HERE, "MOVIE", "movie_common.py"), bench)
        with open(os.path.join(bench, saved_name), "w") as f:
            json.dump(saved, f)
        before = {root: sorted(files) for root, _, files in os.walk(bench)}
        text = open(os.path.join(bench, saved_name)).read()
        env = {k: v for k, v in os.environ.items() if k not in ("AI_MODEL", "AI_MAX_CONCURRENCY", "SEMA_TAG", "SWAN_TAG")}
        # no Sema, no SWAN-AISQL and nothing listening: the run must stop before it needs any of them
        env.update(SEMA_BIN="/nonexistent/sema", DUCKDB_BIN="/nonexistent/duckdb", AI_PROXY_URL="http://127.0.0.1:9")
        p = subprocess.run([sys.executable, os.path.basename(runner), *argv], cwd=bench, env=env,
                           capture_output=True, text=True)
        same = (text == open(os.path.join(bench, saved_name)).read()
                and before == {root: sorted(f for f in files if not f.endswith(".pyc")) for root, _, files in os.walk(bench)
                               if "__pycache__" not in root})
        return p, same


def test_a_run_of_another_configuration_stops_before_it_writes_anything():
    for runner, folder, name, argv in ((HYBRID, "AGENTBENCH", "results/sema_agentbench_results.json", ["Q1", "Q2"]),
                                       (MOVIE, "MOVIE", "sema_movie_results.json", ["q1"])):
        for saved, said in (({"model": "another-model", "concurrency": "20", "request_timeout_s": "1800"}, "another-model"),
                            ({"model": "gpt-5.6-luna", "workers": "20"}, "does not record its concurrency")):
            p, same = _refused(runner, folder, name, {**saved, "per_query": {argv[0]: {"llm_calls": 3}}}, argv)
            assert p.returncode != 0 and "another configuration" in p.stderr and said in p.stderr, (folder, p.stderr)
            assert same, folder


def test_what_makes_a_query_a_failed_one():
    S = load(COMMON)
    ok, none = SimpleNamespace(returncode=0, stderr=""), {"errors": 0, "unanswered": 0}
    assert S.query_error(ok, none) is None
    assert "timeout" in S.query_error(None, none)  # it ran past SEMA_QUERY_TIMEOUT_S
    assert "no such table" in S.query_error(SimpleNamespace(returncode=1, stderr="Catalog Error: no such table\n"), none)
    assert "sql:" in S.query_error(SimpleNamespace(returncode=0, stderr="Binder Error: x"), none)
    # Sema answers a request that failed with false or an empty string and exits 0
    assert "2 model request(s) failed" in S.query_error(ok, {"errors": 2, "unanswered": 2})
    # one row's request failed while another row's request with the same text was answered: no text is left without
    # an answer, yet a row is
    assert "1 model request(s) failed (0 request text(s) never got an answer)" in S.query_error(ok, {"errors": 1, "unanswered": 0})


def test_how_a_latency_was_measured():
    S = load(COMMON)
    assert S.timing(3, True) == "live" and S.timing(3, False) == "live"  # calls new to the cache went to the provider
    assert S.timing(0, True) == "replay"
    assert S.timing(0, False) == "replay without recorded latency"
    assert S.timing(0, None) == "replay, latency mode unknown"
    assert S.timing(None, True) == "not through the cache proxy"


def test_the_committed_queries_are_what_the_translator_writes():
    t = load(TRANSLATOR)
    names = sorted(f for f in os.listdir(t.SRC) if f.endswith(".sql"))
    assert len(names) == 30 and names == sorted(os.listdir(t.DST))
    for name in names:
        assert t.translate(open(os.path.join(t.SRC, name)).read()) == open(os.path.join(t.DST, name)).read(), name


def test_a_prompt_carries_over():
    t = load(TRANSLATOR)
    out = t.translate("SELECT 1 FROM r WHERE SEMANTIC('Is {r.text} about it''s {b.name}?') AND r.x > 1;")
    assert out == "SELECT 1 FROM r WHERE s'Is {r.text} about it’s {b.name}?' AND r.x > 1;"  # '' would reach the model as two characters
    out = t.translate("SELECT semantic_int('Rate (1) to (4). Only number: ' || name || ' - ' || b.description) FROM b;")
    assert out == ("SELECT COALESCE(TRY_CAST(regexp_extract(s'Rate (1) to (4). Only number: {name} - {b.description}"
                   "\n Return a single integer, do not contain any other words.', "
                   r"'^\s*([-+]?[0-9]+)', 1) AS INTEGER), 0) FROM b;")
    for bad in ("SELECT 1 FROM r WHERE SEMANTIC('a brace { that is no column');",
                "SELECT semantic_string('x ' || c) FROM r;"):
        try:
            t.translate(bad)
        except ValueError:
            continue
        raise AssertionError(f"translated {bad!r}")


def test_a_null_prompt_column_is_rendered_as_plop_renders_it():
    t = load(TRANSLATOR)
    out = t.translate("SELECT b.id FROM books_info b JOIN p ON p.id = b.id WHERE b.title IS NOT NULL AND "
                      "SEMANTIC('T: {b.title} A: {b.author} T again: {b.title}') AND b.author IS NULL;")
    assert out == ("SELECT b.id FROM books_info b CROSS JOIN LATERAL (SELECT COALESCE(CAST(b.title AS VARCHAR), '') AS "
                   "title, COALESCE(CAST(b.author AS VARCHAR), '') AS author) b_text JOIN p ON p.id = b.id WHERE "
                   "b.title IS NOT NULL AND s'T: {b_text.title} A: {b_text.author} T again: {b_text.title}' AND b.author IS NULL;")
    # a table without a NULL in a prompt column, and semantic_int (whose prompt PLOP builds with ||: NULL asks nothing)
    plain = "SELECT 1 FROM part p WHERE SEMANTIC('N: {p.p_name}');"
    assert t.translate(plain) == "SELECT 1 FROM part p WHERE s'N: {p.p_name}';"
    assert "LATERAL" not in t.translate("SELECT semantic_int('R: ' || title || ' by ' || author) FROM books_info;")


def test_the_tables_with_a_null_prompt_column_are_the_translators_list():
    try:
        import duckdb
    except ImportError:
        print("   (skipped: needs the duckdb module)")
        return
    t = load(TRANSLATOR)
    name, con, nulls = t.NAME, duckdb.connect(), set()
    for f in sorted(os.listdir(t.SRC)):
        sql = open(os.path.join(t.SRC, f)).read().replace("./test/semantic/agent_bench/dataset/",
                                                          os.path.join(HERE, "AGENTBENCH", "dataset") + "/")
        files = {m.group(1).lower(): m.group(2) for m in
                 re.finditer(rf"CREATE TABLE ({name}) AS\s+SELECT \*\s+FROM (read_csv_auto\([^)]*\))", sql, re.I)}
        for table in t.TPCH_TABLES:
            files.setdefault(table, f"read_parquet('{os.path.join(HERE, 'AGENTBENCH', 'dataset', 'tpch', table)}.parquet')")
        for m in re.finditer(r"\bSEMANTIC\s*\(\s*", sql, re.I):
            body, _ = t.split_literal(sql, m.end())
            for alias, column in set(re.findall(rf"\{{({name})\.({name})\}}", body)):
                declared = list(re.finditer(rf"\b(?:FROM|JOIN)\s+({name})\s+(?:AS\s+)?{re.escape(alias)}\b", sql[:m.start()], re.I))
                table = declared[-1].group(1).lower() if declared else None
                if table in files and con.execute(f'SELECT count(*) FROM {files[table]} WHERE "{column}" IS NULL').fetchone()[0]:
                    nulls.add(table)
    assert nulls == set(t.NULL_AS_EMPTY_TEXT), nulls


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
