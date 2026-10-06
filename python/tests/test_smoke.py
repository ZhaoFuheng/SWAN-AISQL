"""Smoke test for an installed swan-aisql wheel: no network, no key (the extension's built-in mock LLM).

    pip install python/dist/swan_aisql-*.whl && python -m pytest python/tests   (or: python python/tests/test_smoke.py)
"""

import swan_aisql


def test_connect_and_query():
    con = swan_aisql.connect(ai_concurrency=4)
    con.execute("CALL ai_mock_start()")  # in-process deterministic model
    rows = con.execute(
        "SELECT ai_filter('Is ' || x || ' an even number?') FROM range(6) t(x)"
    ).fetchall()
    assert len(rows) == 6 and all(isinstance(r[0], bool) for r in rows)
    usage = con.execute("SELECT sum(llm_calls) FROM ai_usage()").fetchone()[0]
    assert usage >= 1
    con.execute("CALL ai_mock_stop()")


def test_settings_passthrough():
    con = swan_aisql.connect(model="gpt-5.6-luna", ai_concurrency=7, ai_limit=False)
    value = con.execute("SELECT current_setting('ai_concurrency')").fetchone()[0]
    assert int(value) == 7
    assert con.execute("SELECT current_setting('ai_model')").fetchone()[0] == "gpt-5.6-luna"


if __name__ == "__main__":
    test_connect_and_query()
    test_settings_passthrough()
    print("swan_aisql smoke test ok")
