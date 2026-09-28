# SWAN AI-SQL — design

The organising question: **a semantic predicate costs an LLM call; everything else is cheap.**
So the optimizer's job is to make each AI predicate see the fewest possible *distinct* inputs,
and never to change the answer while doing it.

That yields one cost rule and one correctness rule, and every stage below is an instance of them:

* **Cost.** Minimise `distinct(prompts)` evaluated, not rows scanned. Relational work is free by
  comparison; spending it to avoid a call is always right.
* **Correctness.** Every stage is result-preserving *on its own*. Each has a parity test that runs
  the same query with the stage disabled and asserts an identical result set. A stage that
  cannot preserve results is not merged, however large the win.

---

## 1. Two currencies

Relational operators move **rows**. AI evaluation consumes **distinct values**.

The AI region operator folds its input into a dictionary of distinct inputs plus multiplicities,
fires the LLM once per dictionary entry, and fans the results back out by count. Switching to
distinct-value currency as early and as widely as possible is the single biggest lever in the
system: a join fan-out that repeats a row 50 times would otherwise repeat its prompt 50 times.

Three consequences follow.

* **Batching is denominated in prompts, not rows.** The region evaluates in *waves* of new distinct
  inputs and keeps several waves in flight, so the request pool is never idle behind the
  operator's own bookkeeping, and it never speculates on a call the query would not have made.
* **Dictionaries are kept per predicate.** A folded node with several AI predicates keeps one
  dictionary per predicate on that predicate's own columns. Where the predicates read different
  join sides the dictionary is the *sum* of the sides, not their *product*. Ordering is still
  decided per row, over the predicates' predicted pass rates and costs; a later predicate is asked
  exactly for the inputs that still need it.
* **A semantic join never materialises its cross product.** The factor graph keeps one dictionary
  per side and enumerates only surviving pairs, so intermediate size is bounded by the pair domain
  that is still live, not by `|A| × |B|`.

---

## 2. Pipeline

Ordered so that **all relational pruning happens before any semantic decision**, and semantic
placement happens before physical factorisation.

```
  pre-optimize hook   inline AI-bearing CTEs            (boundary removal)
  --- DuckDB built-ins run here: filter pushdown, join order, column pruning ---
  1  semi-join reduce  prune base tables by join keys; the SEMI lands below the leaf's AI stage
                       (an ai_filter or an AI projection), so the LLM sees reduced rows   (relational)
  2  pull-up           lift AI filters above the joins                             (semantic placement)
     cleanup           re-run column pruning / build side after 1-2 reshaped the plan
  3  reorder           order the AI predicates by learned selectivity, fold them into one node
  4  region            row currency -> distinct-value currency (per-predicate dictionaries)
  5  join factorize    factor graph for AI-condition joins (pair domain, never the cross product)
  6  limit / exists    consumer-aware early termination
```

**Why this order.** 1 and 2 decide *how many distinct inputs exist*; 3 decides *in what order* the
predicates run; 4 and 5 decide *how* they are evaluated; 6 decides *when to stop*. Reversing any
adjacent pair loses information the later stage needs — reordering before reduction, for
instance, would cost-model against un-pruned cardinalities.

**A leaf is any boolean expression with one AI call in it.** `ai_filter(p)`, `ai_classify(x) = 'a'`,
`ai_score(x) * 10 > 5`, `lower(ai_complete(x)) IN ('yes', 'true')`, a `CASE` over the same call
twice: each is one leaf, asked once, whose wrapper is applied to the answer. Every stage sees the
same leaf, so a predicate spelled around the call is lifted, reordered, factorised and pushed
exactly like a bare one. The only inputs a leaf may have besides its call are constants.

**The folded node gets a filter of its own.** When a WHERE clause mixes AI predicates with
relational ones, the folded AI node is installed in its own filter *above* the relational
remainder. DuckDB evaluates a conjunction through an adaptive filter that permutes conjunct order,
so leaving the two side by side could run the LLM node first over every row; a separate filter
makes the cheap-first order structural.

**Selectivity is learned, not assumed.** The reorder's predictions come from a small model trained
in-process on the verdicts the query itself produces. Its feature for a predicate is the
embedding of the predicate text, the embedding of the row's input, and their cosine; text inputs
use a sentence encoder, image inputs a CLIP dual encoder so that an `ai_image` predicate is
compared with its image in one joint space. A query starts cold — the first rows are ordered by
cost alone — and warms as its own verdicts land: the region ingests in slices, and after the first
slice it waits until every predicate has a batch of verdicts before ordering the rest. The model is
process-global and keeps training across queries; the predicate text being part of the feature is
what lets it generalise across predicates rather than forget.

---

## 3. Boundary rule: relational pruning must cross CTE boundaries

A materialised CTE is an optimisation barrier. DuckDB materialises any CTE referenced more than
once, and a semantic filter sealed inside one runs on its **full base table** no matter how
selective the outer query is.

Three distinct mechanisms must cross that boundary, and they are not interchangeable:

| mechanism | what it needs to cross | how |
|---|---|---|
| **relational predicate pushdown** — a predicate on the *same* relation as the AI filter | the predicate and the filter must be in one scope | inlining |
| **semi-join reduction** — pruning by a *neighbour's* keys | the reducer's cluster walk must see the join tree inside the CTE | the reducer descends into the main query, and below an AI *projection* whose key passes through it |
| **AI filter pull-up** — lifting the predicate above the joins | the pull-up walk must descend past the CTE node | pull-up descends into a CTE's main query |

**Inlining is the boundary removal, not an optimisation in itself.** AI-bearing CTEs are marked
never-materialise in the pre-optimize hook so DuckDB's own inliner runs them at its natural
position. Re-running the inliner *after* the built-ins does not work: by then the plan is
specialised around the CTE shape.

**Pull-up may descend into a CTE's main query, never its body.** Lifting a predicate out of a body
that other references also read would silently drop the filter for those references.

---

## 4. Duplication safety

Plan duplication is routine — CTE inlining copies a body per reference, and the semi-join reducer
deep-copies a neighbour as its build side. `LogicalOperator::Copy` serialises and re-binds, so:

* **Binding must be idempotent.** Every bind function must be a no-op the second time; a bind that
  appends columns on each call shifts the argument layout the scalar functions infer from it.
* **Never duplicate an LLM predicate to serve a relational purpose.** The reducer needs only the
  neighbour's *keys*, so its build copy keeps the relational skeleton and neutralises AI
  predicates to `TRUE`. Semi-joining against the resulting superset is still result-preserving.
* **Positional maps are remapped by binding identity** whenever a rewrite reorders an operator's
  output, and the map may sit several operators above the one that was reshaped.
* **A mock that is more forgiving than the provider hides bugs.** Mock fidelity is a correctness
  feature; the in-process mock rejects malformed requests exactly as the provider would.

---

## 5. Caching and accounting

Three layers, each with one job:

* **Local cache** (in-process, query-scoped by default) — serves fan-out duplicates and repeated
  prompts within a query, which is what allows a duplicated subtree to agree with its original.
* **In-flight registry** — batches that *start* the same prompt concurrently share one request:
  a batch either owns a key or waits on its owner, issuing everything it owns before waiting on
  anything, which keeps it deadlock-free. Value-level dedup is the only layer that works here:
  DuckDB's common-subplan pass cannot merge volatile expressions, and every AI function is volatile.
* **Provider prefix cache** — explicit prompt caching at the provider. Operators only declare
  structure (`prefix`, expected reuse); the client owns the economics: when to spend a cache write,
  parking same-prefix calls until the write lands, re-priming after the provider's TTL, failing
  open. Caching changes cost, never answers, replay hits or tests.
* **Recording proxy** (bench only) — replays recorded cost and latency so experiments are free and
  reproducible. Replayed latency reproduces the provider's load *at recording time*, so a
  cross-system latency claim requires both systems recorded back-to-back, per query.

**Shared-verdict protocol (cross-engine benches).** Two engines agree on a verdict only if they key
the same recorded sample, so compared engines send byte-identical prompts, including any
answer-format instruction the other engine appends. `ai_debug_prompt_variant='plain'` is the
matching engine side; every other bench keeps the full envelope, which is what makes a verdict
reliable.

**TypeSafe System One backend (optional).** `ai_filter`, `ai_classify` and a bounded integer
`ai_score` can be answered by a TypeSafe Noul / Choice / Score instead of a chat completion (a Score
takes the rubric's levels, low to high, and answers the probability-weighted level, mapped back onto
the function's scale). The function only annotates its request; the
client decides the route per request, re-encodes the typed answer as the chat envelope, and keys the
cache with a distinct marker, so parsers, the reorder node and the selectivity model are untouched.

**Accounting rule:** report chat calls only (embeddings are latency, not the metric under study),
count true requests for any system being compared, and keep caches query-scoped so one query's
work can never subsidise the next.

---

## 6. Measured state (2026-09-26)

All numbers are from the recorded runs in `sembench/` (replayed answers, latency and cost; chat
model `gpt-5.6-luna`; SWAN and LOTUS recorded back-to-back; the Jev column routes ai_filter, ai_classify
and, on MOVIE, the bounded ai_score of q9/q10 to System One).

| benchmark | SWAN | SWAN + Jev (ai_filter/ai_classify on TypeSafe) | LOTUS / PLOP |
|---|---|---|---|
| SemBench MOVIE (10q) | **0.818**, 18.8k calls, $1.53, 1,139s | 0.795, 18.8k, **$0.39**, 511s | LOTUS 0.780, 201k, $14.45, 10,654s |
| SemBench ECOMM (14q) | **0.699**, 16.6k, $8.02, 1,036s | 0.688, 16.6k, $7.39, 826s | LOTUS 0.637, 17.8k, $6.40, 1,990s |
| SemBench MMQA (11q) | **0.636**, 16.1k, $1.84, 1,058s | 0.603, 14.1k, $1.53, 884s | LOTUS 0.449, 19.0k, $2.43, 1,720s |
| agent_bench Q1–Q30 | **1.000**, 11,172 calls, $0.29, 811s | — | PLOP-DP 1.000, 13,602, $0.53, 1,080s; LOTUS 0.620*, 25,738, $2.84, 3,297s |

Quality is each suite's own metric (F1 / ARI / row-multiset F1; agent_bench uses the deterministic
LIMIT-free PLOP ground truth). *LOTUS's agent_bench quality is not like-for-like: SWAN and PLOP
consume the same recorded verdict per shared prompt, LOTUS's verdicts are independent samples, and
where an answer hinges on a few judgments any disagreement scores 0; its calls, latency and cost are
comparable. Per-query tables: `sembench/AGENTBENCH/results/agentbench_comparison_three_way.md`,
`sembench/typesafe_comparison_20260925.md`, and `sembench/compare.py SUITE` for the SemBench suites.

---

## 7. What "clean" means here

A stage belongs in this pipeline only if it can state:

1. **its purpose** in terms of distinct prompts avoided;
2. **its invariant** — why it cannot change the result;
3. **its boundary behaviour** — what it does at a CTE, a projection, an aggregate;
4. **its test** — a parity assertion against the disabled stage, plus a call-count assertion that
   actually discriminates (a test that passes with the feature reverted guards nothing).
