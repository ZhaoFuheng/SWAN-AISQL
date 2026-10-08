# SWAN AI-SQL — design

The organising question: **a semantic predicate costs an LLM call; everything else is cheap.**
So the optimizer's job is to make each AI predicate see the fewest possible *distinct* inputs,
and never to change the answer while doing it.

That yields one cost rule and one correctness rule, and every stage below is an instance of them:

* **Cost.** Minimise `distinct(prompts)` evaluated, not rows scanned. Relational work is free by
  comparison; spending it to avoid a call is always right.
* **Correctness.** Every stage is result-preserving *on its own*. Each has a parity test that runs
  the same query with the stage disabled and asserts an identical result set. A stage that
  cannot preserve results is not merged, however large the win. An error is a result too: a
  query that fails without a stage fails with the same error with it.

---

## 1. Vocabulary

* **AI call** — `ai_filter`, `ai_classify`, `ai_score` or `ai_complete`: one LLM request per
  distinct input.
* **Leaf** — a boolean predicate built around exactly one AI call: `ai_filter(p)`,
  `ai_classify(x) = 'a'`, `ai_score(x) * 10 > 5`, `lower(ai_complete(x)) IN ('yes', 'true')`, a
  `CASE` over the same call twice. The call is asked once and the surrounding expression is
  applied to its answer. Besides the call, a leaf may only read constants.
* **Node** — a WHERE clause's AND/OR/NOT tree over its leaves, folded into one operator argument.
  A node carries the tree and its leaves; it does **not** fix the order they are evaluated in.
* **Region** — the operator that switches currency (§2): rows in, one dictionary of distinct
  inputs per leaf, verdicts out, fanned back to the rows.
* **Factor graph** — the region's form for a semantic join: one dictionary per join side, and the
  node's leaves split into per-side checks and per-pair checks.

---

## 2. Two currencies

Relational operators move **rows**. AI evaluation consumes **distinct values**.

The region folds its input into a dictionary of distinct inputs plus multiplicities, fires the LLM
once per dictionary entry, and fans the results back out by count. Switching to distinct-value
currency as early and as widely as possible is the single biggest lever in the system: a join
fan-out that repeats a row 50 times would otherwise repeat its prompt 50 times.

Three consequences follow.

* **Calls are dispatched per distinct input, not per row or per batch.** The region hands each new
  distinct input to a request pool the moment it is known, and applies each answer the moment it
  lands (asynchronous iteration), so no call waits for a batch to fill or to finish. It never
  speculates on a call the query would not have made. Under a pushed LIMIT it evaluates inline
  batches instead, so it can stop as soon as k rows are confirmed.
* **Dictionaries are kept per leaf.** A node with several leaves keeps one dictionary per leaf on
  that leaf's own columns. Where the leaves read different join sides the dictionary is the *sum*
  of the sides, not their *product*. A later leaf is asked exactly for the inputs that still need
  it, in the order decided per row at run time (§4).
* **A semantic join never materialises its cross product.** The factor graph keeps one dictionary
  per side and enumerates only surviving pairs, so intermediate size is bounded by the pair domain
  that is still live, not by `|A| × |B|`. The join predicate is a conjunction of *factors*, each a
  Boolean tree (AND, OR, NOT) over leaves that read one side or the same two sides; a factor is
  evaluated per member or per pair as one unit that folds its tree three-valued and stops as soon
  as no open leaf could still make it true, so `a OR b` asks `b` only where `a` was false. The scheduler's own state follows the calls made, not the domain:
  per member it keeps counters and the partners it was asked with, and the pairs still to ask are enumerated
  from a cursor, never stored, so a join of two 100k-row sides is scheduled from a few megabytes. Only a
  cyclic edge graph, which takes the staged schedule and lists each edge's live pairs, is capped
  (`ai_factor_pair_limit`) and left to the region over the cross product beyond it.

---

## 3. Plan-time pipeline

These are plan rewrites, in the order they run. They decide *how many* distinct inputs exist and
*what* each predicate is evaluated on. None of them decides the evaluation order — that is a
run-time decision (§4). The stages are ordered so that **all relational pruning happens before any
semantic decision**, and semantic placement happens before physical factorisation.

```
  pre-optimize hook   inline AI-bearing CTEs                                        (boundary removal)
  --- DuckDB built-ins run here: filter pushdown, join order, column pruning ---
  0  top-N first       ORDER BY plain keys LIMIT k: keep the k rows, then run the AI select list on them
  1  semi-join reduce  prune each base table by its join neighbours' keys, below the leaf's
                       AI stage (an ai_filter or an AI projection), so the LLM sees reduced rows
  2  pull-up           lift each AI predicate through every operator it commutes with, settle it
                       above the highest row-removing one, leave a speculative pre-filter at the leaf
                                                                               (semantic placement)
     cleanup           re-run column pruning / build-side choice after 1-2 reshaped the plan
  3  fold              group a filter's AI predicates into one node (tree + leaves)
  4  region            place the node, or a lone AI call, in a region: row -> distinct-value currency
  5  join factorize    a node over a cross product becomes a factor graph (pair domain, never the product)
  6  limit / exists    push a LIMIT k or an EXISTS into the region or graph: stop once k survivors are known
                       (through a plain DISTINCT too: then k *distinct* survivors over the DISTINCT's columns)
```

**Why this order.** 1 and 2 decide *how many distinct inputs exist*; 3 decides *which predicates
travel together*; 4 and 5 decide *what they are evaluated on*; 6 decides *when to stop*. Each stage
matches the shape the previous one produced: the region wants one node in a filter of its own, the
factor graph wants a node over a cross product, the limit push-down wants a region or graph whose
result feeds nothing but the LIMIT. Reversing any adjacent pair loses that shape.

**Top-N first (stage 0).** For `SELECT ai_x(...) ... ORDER BY <plain column> LIMIT k`, DuckDB computes the
SELECT list, AI calls included, for every row and then keeps k. Stage 0 keeps the k rows first and calls the
LLM for those only (one SWAN 2.0 question: 745 calls -> 5). The answer is unchanged because the SELECT list
is computed row by row. It is skipped when the sort reads an AI result.

**Every spelling of a predicate is the same leaf.** Because a leaf is *any* boolean expression around
one AI call, a predicate written as `lower(ai_complete(x)) IN ('yes', 'true')` is lifted, folded,
factorised and pushed exactly like a bare `ai_filter(p)`; the wrapper travels with the leaf and is
applied to the answer.

**The folded node gets a filter of its own.** When a WHERE clause mixes AI predicates with
relational ones, the node is installed in its own filter *above* the relational remainder. DuckDB
evaluates a conjunction through an adaptive filter that permutes conjunct order, so leaving the two
side by side could run the LLM node first over every row; a separate filter makes the cheap-first
order structural.

**Pull-up is the inverse of push-down.** DuckDB pushes a cheap predicate as low as it legally can;
an AI predicate costs a call per distinct input, so stage 2 lifts it as high as it legally can. The
walk is bottom-up and knows nothing about query shapes, only one commutation law per operator: a
filter on the columns of one input moves above a filter, a projection (routing its columns through,
appending a passthrough column when the consumers above tolerate it), a cross product, the
preserved side of a join (both sides of INNER; the left of SEMI, ANTI, LEFT, SINGLE, MARK; the right
of RIGHT, RIGHT_SEMI, RIGHT_ANTI), and a CTE node's main query; it stops at aggregates, limits,
sorts, set operations, windows and a CTE body. Lifting through a preserving operator never adds a
call (the output projected onto the predicate's columns is a subset of its input), so the only
question is where the lift stops paying: the predicate settles directly above the highest operator
it crossed that can remove rows of its input (inner, semi and anti joins, a filter with relational
conjuncts), and goes back exactly where it was if it crossed none. Predicates settling at one site
share one filter, so the fold stage sees them together. This is what makes `IN (SELECT ...)` and
`EXISTS` work: DuckDB plans them as semi joins above the scan and pushes the AI predicate under
them; the law for a semi join lifts it back above (SWAN 2.0 football-16: 296 calls -> 22).

**Speculative pre-filter.** When the pull-up lifts a predicate above a join it leaves a copy at the
leaf that only *prunes*: a row the selectivity model expects to fail is evaluated there and dropped
before the join; every other row passes through unevaluated to the lifted predicate, which is
complete on its own. The semi-join reduction has already pruned the leaf relationally, so this spends
calls only on rows the join would keep. `ai_speculative` switches it off for parity runs.

---

## 4. Run time: the order is decided per row

Nothing in the plan fixes which predicate runs first. The node carries the tree; the operator that
holds it chooses, per input, which leaf to ask next, so the expensive calls short-circuit as early as
the data allows.

* **In a region**, each row picks its next leaf by an exact minimum-expected-cost search over the
  tree, using the leaf's predicted pass rate and its prompt cost. An input whose answer is already
  on its way costs the row nothing, so the search prefers waiting for it over opening another leaf.
  A leaf is asked only for the distinct inputs that some still-undecided row needs; each input is
  sent on its own, and the verdict re-decides every row that shares it the moment it lands. Where
  every row of a join fan-out fails the first leaf, the second leaf is never asked.
* **In a factor graph**, the scheduler picks the next member to check from the pass rates it has
  observed so far in the query, preferring checks whose failure cancels the most pairs. A per-side
  check tends to go first: one failed member removes every pair it would have formed. Pairs are asked
  from the chosen parent member's cursor over its live, confirmed partners (each member's cursor starts
  at a different partner, so a batch spreads over the child side), and a member's child edges open one
  at a time, most lethal first. Within a pair, the leaves run in the node's order and stop at the first
  false answer. Selection costs O(members) per batch, so the domain's size never shows in the scheduler.
* **A LIMIT stops the operator, not the scan.** Once the region or graph has confirmed enough
  passing rows for the pushed `LIMIT k` (counted by fan-out, so k *output* rows), it stops asking.
  Under a LIMIT the region asks in waves of max(k, `ai_concurrency`) inputs and re-checks after each,
  so a small k is settled by one wave, not by a hundred. A LIMIT above a plain DISTINCT reaches the
  region as "k distinct passing tuples" over the DISTINCT's columns (mapped down through the
  projections), so `SELECT DISTINCT name ... WHERE ai_filter(...) LIMIT 5` stops after five names.
* **Rows leave the region as they are decided.** Without a LIMIT the region does not wait for its
  whole input to be decided: once every input is known it puts every remaining call in flight and
  hands rows out in input order as their verdicts land. A consumer that stops early -- a LIMIT the
  push-down cannot place, say above an inner join or a UNION -- ends the query with the calls still
  queued behind the pool never made.
* **The request pool is the bound.** With the calls fixed, a query's latency is roughly its calls
  times the mean call latency divided by `ai_concurrency`. Sending each input as soon as it is known
  only matters where the pool would otherwise sit idle; the largest benchmark queries keep it full.
* **Errors travel like verdicts.** An error raised while evaluating an input — a wrapper that cannot
  read the answer, a prompt expression that fails — is carried back to the query and fails it, exactly
  as the per-row evaluation would. No worker thread drops an error, and none lets one escape.
* **A missing answer is `NULL`, never `false`.** A call that fails after its retries yields a NULL verdict
  on every path — scalar, region and factor graph — so a filter drops the row, `NOT` never passes it, and
  a wrapper such as `IS NULL` is evaluated over the missing answer. `ai_usage()` counts it in `failed_calls`.

**Selectivity is learned, not assumed.** A tree with one leaf has nothing to order, so it embeds
nothing and records no training example; the model is consulted and trained only where a prediction
can change behaviour (two or more leaves, or a speculative gate). The region's predictions come from a
small model trained in-process on the verdicts the query itself produces: one full-batch gradient step over the most
recent 256 verdicts every 3 × `ai_concurrency` calls. The cadence is what decides how early the order
follows the model: stepping more often orders rows on too few labels, stepping less often learns only
after most rows are decided. Its feature for a leaf is the embedding of the
predicate text, the embedding of the row's input, and their cosine; text inputs use a sentence
encoder, image inputs a CLIP dual encoder so that an `ai_image` predicate is compared with its image
in one joint space. Every training step makes the stored predictions stale; before the rows a landed
verdict re-decides are decided, their open inputs are re-predicted in one blocked forward pass per leaf
rather than one pass each, and since a leaf's inputs share the predicate half of the feature, the first
layer's product with that half is formed once per pass. The arithmetic per input is the same sequence as a
lone pass, so the predictions are identical; only the cost changes (on the heaviest benchmark queries, half
the time it took). A query starts cold — the first rows are ordered by cost alone — and warms as its
own verdicts land: the region ingests in slices, and while it is cold it sends a predicate's inputs
only once a concurrency-wide batch of them is pending, and before deciding the next slice it waits for
the calls already out. Once every leaf has a batch of verdicts, everything is sent as soon as it is
known. The model is process-global and keeps training
across queries; the predicate text being part of the feature is what lets it generalise across
predicates rather than forget.

**A worked example.**

```sql
SELECT r.id FROM reviews r JOIN products p USING (product_id)
WHERE p.category = 'camera'
  AND ai_filter('mentions battery life: ' || r.text)
  AND ai_classify(r.text, ['positive', 'negative']) = 'negative'
LIMIT 10;
```

1. *Semi-join reduce*: `reviews` is reduced to the reviews of camera products before any call.
2. *Pull-up*: both AI predicates are lifted above the join; a speculative copy stays at `reviews`.
3. *Fold*: the two predicates become one node with two leaves — the `ai_filter`, and the
   `ai_classify` comparison whose wrapper (`= 'negative'`) is applied to the answer.
4. *Region*: the node is placed in a region above the join, with one dictionary per leaf keyed on
   `r.text`, so a review joined to several products is asked once.
5. *Limit*: the `LIMIT 10` is pushed into the region.
6. *Run time*: per review, the search asks the cheaper or more selective leaf first (cold: by cost;
   warm: by the model's pass rates); the other leaf is asked only for reviews that survived; the
   region stops once ten reviews have passed both.

---

## 5. Boundary rule: relational pruning must cross CTE boundaries

A materialised CTE is an optimisation barrier. DuckDB materialises any CTE referenced more than
once, and a semantic filter sealed inside one runs on its **full base table** no matter how
selective the outer query is.

Three distinct mechanisms must cross that boundary, and they are not interchangeable:

| mechanism | what it needs to cross | how |
|---|---|---|
| **relational predicate pushdown** — a predicate on the *same* relation as the AI filter | the predicate and the filter must be in one scope | inlining |
| **semi-join reduction** — pruning by a *neighbour's* keys | the reducer's cluster walk must see the join tree inside the CTE | the reducer descends into the main query, and below an AI *projection* whose key passes through it |
| **AI filter pull-up** — lifting the predicate above the row-removing operators | the pull-up walk must descend past the CTE node | the commutation law admits a CTE node's main query |

**Inlining is the boundary removal, not an optimisation in itself.** AI-bearing CTEs are marked
never-materialise in the pre-optimize hook so DuckDB's own inliner runs them at its natural
position. Re-running the inliner *after* the built-ins does not work: by then the plan is
specialised around the CTE shape.

**Pull-up may descend into a CTE's main query, never its body.** Lifting a predicate out of a body
that other references also read would silently drop the filter for those references.

---

## 6. Duplication safety

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

## 7. Caching and accounting

Three layers, each with one job:

* **Local cache** (in-process, query-scoped by default) — serves fan-out duplicates and repeated
  prompts within a query, which is what allows a duplicated subtree to agree with its original. Its key
  carries the endpoint with the model and prompt, so `SET ai_endpoint` never serves the previous
  endpoint's answers.
* **In-flight registry** — batches that *start* the same prompt concurrently share one request:
  a batch either owns a key or waits on its owner, issuing everything it owns before waiting on
  anything, which keeps it deadlock-free. Value-level dedup is the only layer that works here:
  DuckDB's common-subplan pass cannot merge volatile expressions, and every AI function is volatile.
* **Provider prefix cache** — explicit prompt caching at the provider. Operators only declare
  structure (`prefix`, expected reuse); the client owns the economics: when to spend a cache write,
  parking same-prefix calls until the write lands, re-priming after the provider's TTL, failing
  open. Caching changes cost, never answers, replay hits or tests.
* **Recording proxy** (bench only) — replays recorded answers, cost and latency so experiments are free
  and reproducible. Model inference at temperature 0 is treated as deterministic, so one cache is the
  single source of answers and latencies for every system and every reported number is a replay of it;
  asking the provider afresh is a different experiment. Replayed latency reproduces the provider's load
  *at recording time*; the cleanest cross-system latency comparison records both systems back to back,
  per query, and the result tables state the recording dates where the systems were recorded on different
  days.

**Shared-verdict protocol (cross-engine benches).** Two engines agree on a verdict only if they key
the same recorded sample, so compared engines send byte-identical prompts, including any
answer-format instruction the other engine appends. `ai_debug_prompt_variant='plain'` is the
matching engine side; every other bench keeps the full envelope, which is what makes a verdict
reliable.

**TypeSafe System One backend (optional).** `ai_filter`, `ai_classify` and a bounded integer
`ai_score` can be answered by a TypeSafe Noul / Choice / Score instead of a chat completion (a Score
takes the rubric's levels, low to high, and answers the probability-weighted level, mapped back onto
the function's scale). The function only annotates its request; the client decides the route per
request, re-encodes the typed answer as the chat envelope, and keys the cache with a distinct
marker, so parsers, the folded node and the selectivity model are untouched.

**Accounting rule:** report chat calls only (embeddings are latency, not the metric under study),
count true requests for any system being compared, and keep caches query-scoped so one query's
work can never subsidise the next.

---

## 8. What is configurable

Each plan stage has one setting: `ai_inline_ai_ctes` (the pre-optimize hook), `ai_semi_reduce` (1),
`ai_pullup` and `ai_speculative` (2), `ai_reorder` (3, the fold and the run-time ordering it enables),
`ai_factorize` (4), `ai_join_factorize` (5) and `ai_limit` (0 and 6). Their purpose is the parity tests and
A/B runs; the defaults are the measured composition. Everything that was
measured to be right — the ingest slice, the warm gate, the training cadence, the speculative gate,
connection reuse — is a constant in the code, not a knob. The remaining settings name the
endpoints, models, keys and concurrency of the LLM, embedding and TypeSafe backends. Where a
connection's requests go (endpoint, model, key, retries, prices, TLS) and every optimizer flag are per
connection, with `SET GLOBAL` for the process default; the request pool, the embedding client and the
local cache are process-wide. Two debug
settings stay: `ai_debug_log` (diagnostics per subsystem) and `ai_debug_prompt_variant` (the
cross-engine protocol above). The full list, with defaults and environment variables, is
[SETTINGS.md](SETTINGS.md).

---

## 9. Measured state (2026-10-03)

All numbers are from the recorded runs in `aisql-bench/` (chat model `gpt-5.6-luna`; the Jev column routes
ai_filter, ai_classify and, on MOVIE, the bounded ai_score of q9/q10 to System One). †The SWAN column was
recorded fresh on 2026-10-01, after `ai_filter` gained its reasoning sentence; the LOTUS and Jev recordings
are from 2026-09-23/25 and PLOP's and BlendSQL's from 2026-10-02 (agent_bench LOTUS and BlendSQL: 2026-10-02/03),
so the SemBench and agent_bench latencies compare recordings from different days. The SWAN 2.0 row is one
back-to-back session. Quality, calls and cost are comparable throughout.

| benchmark | SWAN | SWAN + Jev (ai_filter/ai_classify on TypeSafe) | LOTUS / PLOP / BlendSQL |
|---|---|---|---|
| SemBench MOVIE (10q) | **0.832**, 18.8k calls, $2.28, 1,686s† | 0.795, 18.8k, **$0.39**, 511s | LOTUS 0.780, 201k, $14.45, 10,654s; BlendSQL 0.763, 18.7k, $1.44, 1,188s; Palimpzest 1.5.3 (Abacus optimizer) 0.776, 68.5k, $17.89, 4,865s; ThalamusDB 0.841 on 8 of 10, 1.0k, $0.52, 3,661s; PLOP-DP 0.726, 19.1k, $1.69, 1,665s |
| SemBench ECOMM (14q) | **0.724**, 16.4k, $8.32, 2,882s† | 0.688, 16.6k, $7.39, 826s | LOTUS 0.637, 17.8k, $6.40, 1,990s; Palimpzest 0.680 on q1–q13 (SWAN 0.703 / LOTUS 0.648 there), 17.7k, $16.33, 1,494s; ThalamusDB 0.596 on 5 of 14, 2.8k, $3.45, 12,305s |
| SemBench MMQA (11q) | **0.692**, 15.9k, $2.31, 1,314s† | 0.603, 14.1k, $1.53, 884s | LOTUS 0.449, 19.0k, $2.43, 1,720s; Palimpzest 0.687, 46.3k, $13.79, 3,284s; ThalamusDB 0.327 on 7 of 11, 1.3k, $0.52, 2,191s |
| agent_bench Q1–Q30 | **1.000**, 11,172 calls, $0.29, 799s | — | LOTUS 0.607*, 25,749, $2.90, 1,802s†; BlendSQL 0.758*, 24,631, $2.76, 1,960s†; Palimpzest 1.5.3 (Abacus optimizer) 0.710*, 25,785, $6.82, 2,088s†; ThalamusDB 0.661*, 19,512, $2.09, 8,542s† (Q19 at the 6,000 s cap); PLOP-DP 1.000, 13,602, $0.53, 1,080s |
| SWAN 2.0 (120q, 4 BIRD databases) | 0.755, **19,695** calls, **$2.13**, **2,055s** | — | LOTUS 0.760, 69,204, $4.04, 4,101s; BlendSQL 0.761, 59,620, $4.95, 4,608s; Palimpzest (Abacus optimizer) 0.766, 70,933, $9.85, 2,991s; ThalamusDB 0.396 (0.689 on the 69 questions its filters express), 158,290, $12.73, 36,945s; PLOP-DP 0.690, 25,604, $1.72, 10,431s |

Quality is each suite's own metric (F1 / ARI / row-multiset F1; agent_bench uses the deterministic
LIMIT-free PLOP ground truth). PLOP and BlendSQL do not support images, so ECOMM and MMQA are not run for
them; their costs are computed from their token counts at list price, as is ThalamusDB's, which runs SemBench's own ThalamusDB queries
(MOVIE 8 of 10, ECOMM 5 of 14, MMQA 7 of 11) in exact mode under SemBench's runner settings, and on
agent_bench our NLfilter formulation of the 30 queries. Palimpzest's latency is the time its programs report
(interpreter start-up excluded). *LOTUS's, BlendSQL's, ThalamusDB's and Palimpzest's
agent_bench quality is not like-for-like: SWAN and PLOP
consume the same recorded verdict per shared prompt, the others' verdicts are independent samples, and
where an answer hinges on a few judgments any disagreement scores 0; its calls, latency and cost are
comparable. Per-query tables: `aisql-bench/AGENTBENCH/results/agentbench_comparison_three_way.md`,
`aisql-bench/typesafe_comparison_20260925.md`, and `aisql-bench/compare.py SUITE` for the SemBench suites.
SWAN 2.0 is the benchmark in github.com/ZhaoFuheng/SWANBench: one AISQL query per question that every system
plans itself, with scaled and duplicated databases; its results folder holds the four systems' answers.
Its row is the replay, from the published cache, of each system's fresh recording through an empty cache
(LOTUS and PLOP from the session of 2026-10-03; BlendSQL and Palimpzest re-recorded on 2026-10-05, SWAN on 2026-10-06), so
the numbers reproduce; SWAN, BlendSQL, LOTUS and Palimpzest are at parity on quality (the model's verdicts
bound it: four fresh SWAN recordings scored 0.775 / 0.757 / 0.763 / 0.755, BlendSQL's two 0.773 / 0.761,
Palimpzest's two 0.769 / 0.766) and calls, cost and latency separate them.

---

## 10. What "clean" means here

A stage belongs in this pipeline only if it can state:

1. **its purpose** in terms of distinct prompts avoided;
2. **its invariant** — why it cannot change the result;
3. **its boundary behaviour** — what it does at a CTE, a projection, an aggregate;
4. **its test** — a parity assertion against the disabled stage, plus a call-count assertion that
   actually discriminates (a test that passes with the feature reverted guards nothing).
