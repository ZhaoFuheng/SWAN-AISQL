# SWAN 2.0 queries

A copy of the SWAN 2.0 benchmark's queries, so that this repository holds every query behind its results
tables. The benchmark itself (harness, scaled databases, scoring, the five systems' adapters, results) is
[github.com/ZhaoFuheng/SWANBench](https://github.com/ZhaoFuheng/SWANBench); this copy was taken from its commit
`ef70911` (2026-10-03) and is refreshed with `sync.sh` from a sibling checkout. The benchmark's copy is the
one its harness runs; edit there, then sync.

| directory | what |
|---|---|
| `aisql/<qid>.sql` | the question's query in DuckDB SQL with AI functions (`ai_filter`, `ai_classify`, `ai_complete`, `ai_agg`); every system runs this one query, SWAN-AISQL as written and BlendSQL, LOTUS, PLOP, ThalamusDB and Palimpzest through the benchmark's mechanical translations or its written-order executor |
| `oracle/<qid>.sql` | the same query with each AI call replaced by the true hidden value; it returns the gold answer on the original database |
| `questions/<db>.csv` | the 120 questions: id, database, question text, evidence, difficulty, query shape and BIRD's gold SQL |

The queries run on the benchmark's masked, scaled databases (`swan-bench prepare` builds them from BIRD), so
they are not runnable from this folder alone: install the benchmark, then `swan-bench run --system aisql`
(its `scripts/run_swan_aisql.sh` builds this repository and starts the serving stack). The rules the queries
follow are in the benchmark's `docs/SWAN2_AUTHORING.md`.
