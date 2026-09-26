# Testing this extension

`test/sql/` holds the extension's [sqllogictests](https://duckdb.org/dev/sqllogictest/intro.html), grouped
by subsystem:

    functions/       scalar/table AI functions, local cache, NULL handling, TypeSafe backend
    region/          AI region operator: waves, overlap, LIMIT early-stop, per-leaf factorization
    factor_join/     factorized semantic joins and factor graphs
    optimizer/       predicate rewrite, reorder, CASE laziness, plan shapes
    pullup/          semantic filter pull-up through joins and CTEs
    semi_reduce/     Yannakakis semi-join reduction (computed keys, CTE boundaries, AI projections)
    limit/           LIMIT push-down into AI evaluation
    lotus_envelope/  never-more-calls-than-flat-evaluation invariants

Every test starts with `require aisql` and `CALL ai_mock_start()`: the in-process mock answers
deterministically from a hash of the prompt (`[p=NN]` in a prompt fixes its pass probability), so the
suite needs no network, keys or provider. Count-sensitive tests reset usage with `SELECT ai_usage_reset()`
and clear the local cache with `SELECT ai_local_cache_clear()` between cases.

Run everything:

```bash
make test_reldebug          # or make test (release) / make test_debug
```

Run one file or group directly (tests are registered by absolute path):

```bash
build/reldebug/test/unittest "*test/sql/semi_reduce/*"
build/reldebug/test/unittest "$PWD/test/sql/functions/typesafe_backend.test"
```
