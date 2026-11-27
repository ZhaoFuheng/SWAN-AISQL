#!/usr/bin/env python3
"""The hybrid bench's scoring commands on made-up results: score_agentbench.py, eval_agentbench.py and
report_three_way.py must agree on what a run is.

  - a query a system did not run is n/a in all three and outside its macro, which says how many queries it covers
  - a query that ran and failed scores 0 in all three; the report calls it "failed" and marks its sums
  - the report is of one model and one number of requests in flight: it stops, writing nothing, when the selected
    runs (or the ground truth's model) differ, and its title names the model it read
  - the report marks a run whose records show it was timed live or replayed without the recorded latency
  - the report shows Sema at its default settings only, from the run SEMA_TAG names: that run must record Sema's
    shipped settings and the query texts it ran

Run: python3 -m pytest aisql-bench/AGENTBENCH/test_scoring.py   (or python3 test_scoring.py). Needs pandas, like the
scorer. Each test copies the scripts into a temporary directory with a one-row ground truth per query, so nothing
under results/ is read or written.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import sema_common  # noqa: E402 -- the settings a Sema summary records

QUERIES = [f"Q{i}" for i in range(1, 31)]
EMPTY_GT = "Q13"  # a query whose ground truth is no row, as four are in the benchmark
TABLE = os.path.join("results", "agentbench_comparison_three_way.md")


class Bench:
    """A scratch copy of the scoring scripts with a made-up ground truth: the answer to QN is the one row `QN`, and
    to EMPTY_GT no row. LOTUS, PLOP and SWAN (the systems the report always shows) start with a full correct run."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.join(self.tmp.name, "aisql-bench", "AGENTBENCH")
        os.makedirs(os.path.join(self.root, "plop_gt_queries"))
        for f in ("sema_common.py", "swanbench_meter.py"):
            shutil.copy(os.path.join(os.path.dirname(HERE), f), os.path.dirname(self.root))
        for f in ("score_agentbench.py", "eval_agentbench.py", "report_three_way.py"):
            shutil.copy(os.path.join(HERE, f), self.root)
        shutil.copy(os.path.join(HERE, "plop_gt_queries", "output_limits.json"),
                    os.path.join(self.root, "plop_gt_queries"))
        shutil.copytree(os.path.join(HERE, "sema_queries"), os.path.join(self.root, "sema_queries"))
        self.run("plop_gt", workers="20", mode="none")
        for tag in ("lotus", "plop", "swan"):
            self.run(tag)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.tmp.cleanup()

    def run(self, tag, queries=QUERIES, wrong=(), failed=(), other_columns=(), no_rows_file=(), summary=True,
            model="model-a", record=None, sema=False, **top):
        """results/<tag>/QN.csv for `queries`: the right answer, another row for `wrong`, the empty file a failed
        query leaves for `failed`, a row under another column name for `other_columns`, no file for `no_rows_file`.
        Unless summary=False, also the runner's summary, with `top` as its file-level fields and `record` added to
        every query's record; `sema` writes it the way sema_agentbench.py does."""
        out = os.path.join(self.root, "results", tag)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        per_query = {}
        for q in queries:
            rows = [] if q in failed or (q == EMPTY_GT and q not in wrong) else ["nope" if q in wrong else q]
            if q not in no_rows_file:
                with open(os.path.join(out, q + ".csv"), "w") as f:
                    f.write(("other\n" if q in other_columns else "id\n") + "".join(r + "\n" for r in rows) if rows else "\n")
            per_query[q] = {"latency_s": 1.5, "llm_calls": 2, "cost_usd": 0.01, "llm_calls_fresh": 0, "rows": len(rows),
                            "error": "boom" if q in failed else None, "ran_at": "2026-01-02 03:04:05 UTC"}
            if sema:
                text = open(os.path.join(self.root, "sema_queries", q + ".sql"), "rb").read()
                per_query[q].update(timing="replay", query_sha256=hashlib.sha256(text).hexdigest())
            per_query[q].update(record or {})
        path = os.path.join(self.root, "results", tag + "_agentbench_results.json")
        if os.path.exists(path):
            os.remove(path)
        if summary:
            if sema:
                top = {"request_timeout_s": "1800", "settings": dict(sema_common.SETTINGS), **top}
            if "concurrency" not in top and "workers" not in top:
                top["concurrency"] = "20"
            with open(path, "w") as f:
                json.dump({"model": model, **top, "per_query": per_query}, f)

    def cmd(self, script, *args, **env):
        base = {k: v for k, v in os.environ.items() if k not in ("GT_TAG", "GT_SYSTEM", "SWAN_TAG", "SEMA_TAG")}
        return subprocess.run([sys.executable, script, *args], cwd=self.root, capture_output=True, text=True,
                              env={**base, "SWAN_TAG": "swan", **env})

    def scores(self, script, *systems):
        """{system: {query or 'macro': cell}} and the text one of the two scorers printed."""
        p = self.cmd(script, *systems)
        assert p.returncode == 0, p.stderr
        cells = {s: {} for s in systems}
        for line in p.stdout.splitlines():
            m = re.match(r"(Q\d+|macro|MACRO)\s+(.*)", line)
            if m:
                for s, cell in zip(systems, m.group(2).split()):
                    cells[s][m.group(1).lower() if m.group(1) in ("macro", "MACRO") else m.group(1)] = cell
        return cells, p.stdout

    def report(self, **env):
        """(the process, {system: {query or 'macro': quality cell}}, the table's text) of report_three_way.py."""
        p = self.cmd("report_three_way.py", **env)
        path = os.path.join(self.root, TABLE)
        if p.returncode != 0 or not os.path.exists(path):
            return p, {}, ""
        text = open(path).read()
        rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in text.splitlines()
                if line.startswith("| ")]
        names = [c for c in rows[0][1:] if not c.endswith((" calls", " lat (s)", " $"))]
        cells = {n: {} for n in names}
        for r in rows[1:]:
            key = "macro" if "macro" in r[0] else r[0]
            for n, cell in zip(names, r[1:]):
                cells[n][key] = cell
        return p, cells, text

    def refused(self, *said, **env):
        p, _, _ = self.report(**env)
        assert p.returncode != 0 and all(text in p.stderr for text in said), p.stderr
        assert not os.path.exists(os.path.join(self.root, TABLE))


def test_an_unrun_query_is_na_in_every_command():
    with Bench() as b:
        b.run("sema", queries=["Q1", "Q2"], wrong=["Q2"], sema=True)
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, out = b.scores(script, "sema", "lotus")
            assert cells["sema"]["Q1"] == "1.000" and cells["sema"]["Q2"] == "0.000", (script, cells["sema"])
            assert all(cells["sema"][q] == "n/a" for q in QUERIES[2:]), (script, cells["sema"])
            assert cells["sema"]["macro"] == "0.500" and cells["lotus"]["macro"] == "1.000", (script, out)
            assert "sema: 2 of 30 queries scored" in out and not re.search(r"lotus: \d+ of 30", out), out
        p, cells, text = b.report()
        assert p.returncode == 0, p.stderr
        assert cells["Sema"]["Q1"] == "1.000" and cells["Sema"]["Q2"] == "0.000" and cells["Sema"]["Q3"] == "n/a"
        assert cells["Sema"]["macro"] == "**0.500** (2q)" and cells["LOTUS"]["macro"] == "**1.000**", cells
        assert "Sema 2026-01-02 (2 of 30 queries)" in text and "one run of the 30 queries" in text
        assert "4 (2q)" in text and "n/a marks a query a system did not run" in text


def test_a_failed_query_scores_zero_in_every_command():
    with Bench() as b:
        # a failed query leaves an empty file: on EMPTY_GT that would read as the right answer
        b.run("sema", failed=["Q5", EMPTY_GT], sema=True)
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, out = b.scores(script, "sema")
            assert cells["sema"]["Q5"] == "0.000" and cells["sema"][EMPTY_GT] == "0.000", (script, cells["sema"])
            assert cells["sema"]["macro"] == "0.933" and "queries scored" not in out, (script, out)
        p, cells, text = b.report()
        assert p.returncode == 0, p.stderr
        assert cells["Sema"]["Q5"] == "failed" and cells["Sema"][EMPTY_GT] == "failed" and cells["Sema"]["Q4"] == "1.000"
        assert cells["Sema"]["macro"] == "**0.933**" and "q)" not in text and "one full run" in text, text[:900]
        assert "Sema 2026-01-02 (2 failed)" in text and "60 (2 failed)" in text and "failed marks a query" in text


def test_the_empty_ground_truth_is_matched_by_an_empty_answer_only():
    with Bench() as b:
        b.run("sema", wrong=[EMPTY_GT], sema=True)
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, _ = b.scores(script, "sema", "lotus")
            assert cells["sema"][EMPTY_GT] == "0.000" and cells["lotus"][EMPTY_GT] == "1.000", (script, cells)


def test_an_answer_with_none_of_the_ground_truths_columns_scores_zero():
    with Bench() as b:
        b.run("sema", other_columns=["Q2", "Q4"], no_rows_file=["Q7"], sema=True)  # Q2 is a LIMIT query, Q4 is not
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, out = b.scores(script, "sema")
            assert cells["sema"]["Q2"] == cells["sema"]["Q4"] == cells["sema"]["Q7"] == "0.000", (script, cells["sema"])
            assert cells["sema"]["macro"] == "0.900", (script, out)
        assert "sema: no readable rows file for Q7" in b.scores("eval_agentbench.py", "sema")[1]


def test_outputs_without_a_summary_count_the_files_present():
    with Bench() as b:
        b.run("byhand", queries=["Q1", "Q4"], summary=False)
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, out = b.scores(script, "byhand")
            assert cells["byhand"]["Q1"] == cells["byhand"]["Q4"] == "1.000" and cells["byhand"]["Q2"] == "n/a"
            assert cells["byhand"]["macro"] == "1.000" and "byhand: 2 of 30 queries scored" in out, (script, out)


def test_a_tag_with_no_results_scores_nothing():
    with Bench() as b:
        for script in ("eval_agentbench.py", "score_agentbench.py"):
            cells, out = b.scores(script, "nosuch")
            assert set(cells["nosuch"].values()) == {"n/a"} and "nosuch: no query scored" in out, (script, out)


def test_the_report_is_of_one_model():
    with Bench() as b:
        b.run("sema", model="model-b", sema=True)
        b.refused("Sema model-b", "SWAN model-a")
        b.run("sema", sema=True)
        b.run("plop_gt", model="model-b", workers="20", mode="none")
        b.refused("the ground truth (plop_gt) model-b")


def test_the_report_is_of_one_number_of_requests_in_flight():
    with Bench() as b:
        b.run("sema", sema=True, concurrency="10")
        b.refused("Sema 10", "LOTUS 20")


def test_the_report_reads_its_model_and_its_sema_run_from_the_files():
    with Bench() as b:
        p, cells, text = b.report()  # no Sema run: no Sema column
        assert p.returncode == 0 and "Sema" not in text, p.stderr
        assert text.startswith("# agent_bench Q1-Q30: LOTUS vs PLOP-DP vs SWAN (model-a) -- "), text[:120]
        b.run("mysema", sema=True)
        p, cells, text = b.report(SEMA_TAG="mysema")
        assert p.returncode == 0, p.stderr
        assert text.startswith("# agent_bench Q1-Q30: LOTUS vs Sema vs PLOP-DP vs SWAN (model-a) -- "), text[:120]
        assert "Sema run: results/mysema." in text and "with 20 requests in flight" in text
        assert "one full run of the 30 queries" in text and "n/a" not in text and "q)" not in text
        assert cells["Sema"]["macro"] == "**1.000**" and "except as marked" not in text


def test_the_report_stops_on_a_tag_that_has_no_run():
    with Bench() as b:
        b.refused("no Sema run at results/nosuch_agentbench_results.json", SEMA_TAG="nosuch")
        b.refused("no SWAN run at results/nosuch_agentbench_results.json", SWAN_TAG="nosuch")


def test_the_report_shows_sema_at_its_default_settings_only():
    with Bench() as b:
        b.run("sema", workers="20", batch="4")  # a run of the earlier runner: no settings on record
        b.refused("does not record Sema's settings", "default settings only")
        b.run("sema", sema=True, settings={**sema_common.SETTINGS, "semantic_batch_size": "4"})
        b.refused("other than Sema's shipped defaults", "semantic_batch_size")
        b.run("sema", sema=True)
        with open(os.path.join(b.root, "sema_queries", "Q3.sql"), "a") as f:  # the query was translated anew since
            f.write("-- another text\n")
        b.refused("Q3 ran another text than sema_queries/ now holds")


def test_the_report_marks_a_run_its_records_show_was_not_a_replay_at_the_recorded_pace():
    with Bench() as b:
        b.run("sema", sema=True, record={"llm_calls_fresh": 2, "timing": "live"})
        p, _, text = b.report()
        assert p.returncode == 0, p.stderr
        assert "Sema 2026-01-02 (live)" in text and "(except as marked for Sema)" in text, text[:900]
        b.run("sema", sema=True, record={"timing": "replay without recorded latency"})
        b.run("lotus", record={"llm_calls_fresh": 3})    # a runner that writes the count alone: some calls were fresh
        b.run("swan", record={"llm_calls_fresh": None})  # no count on record: nothing to mark
        p, _, text = b.report()
        assert p.returncode == 0, p.stderr
        assert "LOTUS 2026-01-02 (live)" in text and "Sema 2026-01-02 (replay without recorded latency)" in text
        assert "(except as marked for LOTUS, Sema)" in text and "SWAN 2026-01-02)" in text, text[:900]
        assert "not recorded" not in text


if __name__ == "__main__":
    for n, f in sorted(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
