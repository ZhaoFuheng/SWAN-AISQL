# SWAN AI-SQL — design

The organising question: **a semantic predicate costs an LLM call; everything else is cheap.**
So the optimizer's job is to make each AI predicate see the fewest possible *distinct* inputs,
and never to change the answer while doing it.

That yields one cost rule and one correctness rule, and every stage below is an instance of them:

* **Cost.** Minimise `distinct(prompts)` evaluated, not rows scanned. Relational work is free by
  comparison; spending it to avoid a call is always right.
* **Correctness.** Every stage is result-preserving *on its own*. Each has a parity test that runs
  the same query with the whole stack disabled and asserts an identical result set. A stage that
  cannot preserve results is not merged, however large the win.

---

## 1. Two currencies

Relational operators move **rows**. AI evaluation consumes **distinct values**.

`AISQLMapChunk` (unary) and the factor representation (n-ary) hold a dictionary of distinct
inputs plus multiplicities. The AI call fires once per dictionary entry; results fan back out by
count. Switching to distinct-value currency as early and as widely as possible is the single
biggest lever in the system, because a join fan-out that repeats a row 50 times repeats its
prompt 50 times otherwise.

**Consequence for the scheduler:** the region batches work in *waves* of new distinct reps, and a
wave blocks `Sink` until its LLM batch returns. But concurrency is spent on distinct **prompts**, and
above a join that repeats one side's prompt across a whole chunk a wave of a thousand reps can carry
a single new prompt (agent_bench Q17: 240 waves, 1.4 calls each, 19 of 20 workers idle). Every unit of
batching must therefore be denominated in prompts in flight, not reps -- the fix is to keep up to
`ai_debug_wave_overlap` waves in flight at once (Q17 295s -> 88s, Q19 867s -> 686s, results and call
counts unchanged), never to speculate on a call the tree would not have made. Pull-up is what creates
the mismatch (it collapses 246,960 pair-reps into 324 prompts) and it stays: on Q17 it is a 12.5x call
saving; disabling it made the query only 17% faster at 12x the cost.

**Consequence for the region's dictionary (per-leaf factorization).** A folded node's reps are kept
**per leaf** -- one dictionary per leaf on that leaf's own columns -- never on the union of all leaf
columns. A per-leaf dictionary is never larger than the tuple one (each leaf's distinct values are a
projection of the tuples), and where the leaves live on different join sides it is the *sum* instead
of the *product*: agent_bench Q19 holds 9,357 + 180 reps instead of 1.68M, and the region's CPU
fell from 55s to 20s. Ordering is still decided **per row**: a row is a vector of rep ids, each rep
carries the model's P(true) and its cost, and the same DP the per-row batch uses picks the row's next
leaf over those numbers -- arithmetic over rep ids, nothing materialised. The cost of asking a leaf
for a row is the leaf's cost amortized over the rows its rep serves (above a join the small side's rep
is shared by every row of the big side, so it is the cheap question even when the model is cold; on a
single table the fan-out is 1 and it is plain cost, the batch's own choice). A leaf's pending reps are
those some undecided row waits on, so a wave is in prompt currency by construction and a later leaf is
asked exactly for the reps still needed: Q17's customers all fail and its order leaf is never asked.
A rep's prediction is stamped with the model's training step and re-predicted the next time a row
decides over it after the model has trained -- the per-row batch's just-in-time learning, per rep. The
rep is embedded ONCE when staged (one batched request per chunk, cold or warm) and keeps its feature, so
a re-prediction is a forward pass and never a request: the earlier "embed only when warm" rule made a
cold-staged rep's first refresh its own 2-text embed request on the Sink thread (Q19: 9,357 of them,
~100s during which landed verdicts were not applied and no wave could launch). Likewise a landed
verdict on a shared rep re-decides every row behind it (Q19: 1.78M rows behind 180 customer reps), so
`ApplyWave` launches the next leaf's wave as soon as a full batch is pending rather than after the last
waiter -- the Sink thread's bookkeeping never idles the pool. And `Fire` first reaps every landed
wave, then, when NOTHING is in flight, fires at `min(floor, concurrency)` instead of the floor: above a
join whose big side trickles the small leaf's reps, waiting for a floor-sized first batch idled the
pool for the first ~1M ingested rows (Q19's first wave moved from deep in the ingest to 6.5s; wall
655s -> 579s, PLOP 590s, LLM floor ~553s; results identical). The `[leaf-region]` log line
(`ai_debug_log='region'`) prints where the Sink thread's time went (embed / refresh / decide / drain). Measured on govreport with real verdicts: conj_q44 1,198 calls vs the
recorded JIT DP's 1,195, conj_q0 1,268 vs 1,270; on the mock the whole selected set is within -0.9%
with identical id sets. Evaluation reuses the per-row batch machinery one leaf at a time (`AILeafEvaluate`: a one-leaf tree
through the same worker pool, single-flight, speculative gate and MLP training); predictions come from
one batched embed per leaf per chunk, which is also what took Q19's embedding requests from 12,548 to
2,423. There is no mode and no plan-time criterion: `exec/ai_leaf_region.cpp` is the node path; the
tuple map remains for plain scalar calls and for the blocking path, which is the parity reference.

**Consequence for joins:** a semantic join must never materialise its cross product. The factor
graph keeps one dictionary per side and enumerates only surviving pairs, so intermediate size is
bounded by the *pair domain that is still live*, not by `|A| x |B|`.

---

## 2. Pipeline

Ordered so that **all relational pruning happens before any semantic decision**, and semantic
placement happens before physical factorisation.

```
  pre-optimize hook   inline AI-bearing CTEs            (boundary removal)
  --- DuckDB built-ins run here: filter pushdown, join order, column pruning ---
  1  semi-join reduce  prune base tables by join keys; the SEMI lands BELOW the leaf's AI stage
                       (an ai_filter or an AI projection), so the LLM sees reduced rows   (relational)
  2  pull-up           lift AI filters above the joins                             (semantic placement)
     cleanup           re-run column pruning / build side after 1-2 reshaped the plan
  3  reorder           DP-order the AI leaves, fold into one node, speculative gate
  4  region            row currency -> distinct-value currency (per-leaf dictionaries, |A|+|B| reps)
  5  join factorize    factor graph for AI-condition joins (pair domain, never the cross product)
  6  limit / exists    consumer-aware early termination
```

Execution (§1): the region folds rows into reps, fires LLM *waves* over pending reps with up to 8
in flight, learns selectivity just-in-time from landed verdicts, and never idles the request pool
behind its own bookkeeping. The LLM backend is a chat model (default `gpt-5.6-luna`) or, per
function, TypeSafe System One (§5).

**Why this order.** 1 and 2 decide *how many distinct inputs exist*; 3 decides *in what order* the
predicates run; 4 and 5 decide *how* they are evaluated; 6 decides *when to stop*. Reversing any
adjacent pair loses information the later stage needs — e.g. reordering before reduction would
cost-model against un-pruned cardinalities.

---

## 3. Boundary rule: relational pruning must cross CTE boundaries

A materialised CTE is an optimisation barrier. DuckDB materialises any CTE referenced more than
once, and a semantic filter sealed inside one runs on its **full base table** no matter how
selective the outer query is. Measured on agent_bench: Q22 evaluated 89 businesses where 10
survive; Q26 evaluated 1,000 parts where 5 do.

Three distinct mechanisms must cross that boundary, and they are not interchangeable:

| mechanism | what it needs to cross | status |
|---|---|---|
| **relational predicate pushdown** — a predicate on the *same* relation as the AI filter (`attributes ILIKE '%WiFi%'`) | the predicate and the filter must be in one scope | via inlining |
| **semi-join reduction** — pruning by a *neighbour's* keys (2018 reviews/tips) | the reducer's cluster walk must see the join tree inside/below the CTE | done (main query only) |
| **semi-join reduction below an AI projection** — MMQA q1's `ai_complete` in a CTE, joined on its pass-through key | the reducer descends below AI *filters*; an AI projection is an LLM stage too, so once the key translates through it the SEMI belongs underneath (above it prunes rows already paid for); leaf AI is visible regardless of `ai_pullup` | done (test/sql/semi_reduce/leaf_projection_reduction.test) |
| **AI filter pull-up** — lifting the predicate above the joins | the pull-up walk must descend past the CTE node | done |

**Inlining is the boundary removal, not an optimisation in itself.** `ai_inline_ai_ctes` marks
AI-bearing CTEs `CTE_MATERIALIZE_NEVER` in the pre-optimize hook so DuckDB's own inliner runs
them at its natural position. Re-running the inliner *after* the built-ins does not work: by then
the plan is specialised around the CTE shape.

**Pull-up may descend into a CTE's MAIN QUERY, never its BODY.** Lifting a predicate out of a body
that other references also read would silently drop the filter for those references.

---

## 4. Duplication safety

Plan duplication is routine — CTE inlining copies a body per reference, and the semi-join reducer
deep-copies a neighbour as its build side. `LogicalOperator::Copy` **serialises and re-binds**, so:

* **Binding must be idempotent.** `AIKeyBind` appends one history-key column per column-ref in the
  prompt and records `key_start` = the user-argument count. Appending again on a re-bind shifts
  `key_start`, and the scalar functions infer their optional arguments from it
  (`ai_complete: has_schema = key_start >= 2`). The result was a key column being sent as a JSON
  schema, a rejected request, a NULL predicate, and rows vanishing with no error. *Any* new bind
  function must be a no-op the second time.
* **Never duplicate an LLM predicate to serve a relational purpose.** The reducer needs only the
  neighbour's *keys*, so its build copy keeps the **relational skeleton**: AI predicates are
  stripped (neutralised to `TRUE`, not spliced out — removing the node drops its `projection_map`
  and changes output bindings). Semi-joining against the resulting superset is still
  result-preserving.
* **A positional map is not necessarily on the reshaped operator's direct parent.** Pushing a
  region below a join reorders the join's output; a `Filter`/`ORDER BY` projection_map built for the
  old order must be remapped by binding identity -- and the map can sit above an intervening region
  (which appends its result to its child's order) or a map-less filter. `ai_reorder=false` produces
  exactly that stack (two unfolded regions, the filter testing both on top); the remap walks up until
  it reaches a map. Symptom when missed: the binding resolver rejects the LAST column of one side.
* **A mock that is more forgiving than the provider hides bugs.** The mock answered malformed
  bodies with `"ok"`, which is exactly why the binding bug survived the test suite. Mock fidelity
  is a correctness feature.

---

## 5. Caching and accounting

Three layers, each with one job:

* **Local cache** (in-process, query-scoped by default) — serves fan-out duplicates and repeated
  prompts within a query; makes a predicate *consistent within a query*, which is what allows a
  duplicated subtree to agree with its original.
* **In-flight registry** — the cache is written when a response *lands*, so batches that START the
  same prompt concurrently all miss and all pay. Inlining guarantees that shape: each CTE
  reference becomes its own copy of the same scan, and the copies run as separate pipelines over
  the same rows. A batch either owns a key or waits on its owner. It issues and publishes
  everything it owns *before* waiting on anything owned by others, which is what makes it
  deadlock-free; and the cache is written before a registry entry retires, so no instant is
  uncovered. Structural re-merging is NOT an alternative here: DuckDB's `common_subplan` pass
  sets `can_materialize = false` on any volatile expression, and every AI function is volatile —
  and after pushdown the copies are usually no longer identical anyway, which is the point of
  inlining. Value-level dedup is the only layer that works, and the right one.
* **Provider prefix cache** — operators declare `(prefix, expected_reuse)`; the client owns the
  write/read policy, parking and TTL. Operators never decide economics.
* **Recording proxy** (bench only) — replays recorded cost and latency so experiments are free and
  reproducible. Never a substitute for the above. Replayed latency reproduces the provider's load
  **at recording time**, so it is only comparable between entries recorded under the same
  conditions: every row carries `recorded_at` (UTC; NULL on rows that predate the column), and a
  cross-system latency claim requires both systems recorded **back-to-back, per query** (LOTUS
  then SWAN, into one fresh store, then merged). Cost is token-determined and does not drift;
  quality moves with fresh sampling and must be re-certified together with latency. Measured on
  ECOMM q8: LOTUS's earlier recording was 2.3x inflated, and SWAN's 5.3x speedup was really 3.9x.

**Shared-verdict protocol (cross-engine benches).** Two engines agree on a verdict only if they
key the same recorded sample, and the alias keys on the **user text** alone. So the compared
engines' prompts must be byte-identical *including any answer-format instruction the other engine
appends inside the prompt* -- agent_bench's PLOP appends `"\n Return a single yes or no, ..."`, and
dropping it from the translation cost 0.20 macro-F1 that looked like a placement difference.
`ai_debug_prompt_variant='plain'` is the matching engine side: ai_filter sends the prompt bare (no
system message, no schema, no prefix-cache splitting -- a split body sends content as an ARRAY,
which the alias cannot key) and reads the verdict from the answer text. Comparison scaffolding
only; every other bench keeps the envelope, which is what makes a verdict reliable.

**What makes each comparison fair.** agent_bench vs PLOP: prompt-identical (`plain`) + the
shared-verdict alias, so both engines consume the same recorded sample *and the same recorded
latency* for every shared prompt (~98% of SWAN's calls); ground truth is PLOP's un-rewritten UDF
execution with output-truncating LIMITs removed (`plop_gt`), and limited queries are scored on
soundness + cardinality. SemBench vs LOTUS: LOTUS unmodified, SWAN with its own envelope, same
model and effort, query-scoped caches, recorded back-to-back on 2026-09-23. Row order of unordered
results and call counts of short-circuit-dominated queries vary run to run and are never asserted.

**TypeSafe System One backend (optional, `SET ai_typesafe='filter,classify'`).** ai_filter can be
answered by a TypeSafe Noul and ai_classify by a Choice (Jev, `POST /v1/systemone`) instead of a chat
completion. The function only *annotates* its request (`AIRequest::question`, the Choice options,
the classified input as `state`); the chat prompt, system message and schema stay set, and the
CLIENT decides the route per request (`AIUsesTypeSafe`): the setting must name the function, and an
image-bearing prompt always stays on the chat model because Jev is text-only. A Noul judges the whole
ai_filter prompt as its state (byte-identical to the chat prompt, prefix included) under an
instruction that restates the system prompt's job AND quotes the predicate text -- the prompt's
constant part (the AISplitPrompt split the selectivity features already use; the scalar path
computes it at bind time, the factor graph folds the leaf's constant column). Jev reads literally:
over one undifferentiated blob it compressed every ECOMM q1 answer into 0.1-0.48 (perfect ranking,
zero rows); naming the claim separates the modes (0.95+ vs <0.5) while the whole-prompt state keeps
pair prompts' labels, which a state reduced to the row-varying text would lose. The probability
becomes the verdict at `ai_typesafe_threshold` (0.5). The answer is re-encoded as the chat envelope's `{"result": ...}`, so
every parser, the reorder node (its classify leaf recovers the option list from the baked
`Categories:` frame) and the selectivity model are untouched. Cache identity gains a typesafe marker
(model + instructions + options): a System One answer is a different sample from a chat answer to
the same prompt and must never be served from its entry. Cost is input tokens x
`typesafe_price_input` (output is free); the recording proxy caches `/v1/systemone` like chat and
forwards it to `CACHE_TYPESAFE_UPSTREAM` (https://api.typesafe.ai) -- it also terminates TLS, since
the engine's bundled httplib speaks plain http, so `ai_typesafe_endpoint` defaults to the proxy. Comparison caveat: a chat run replays recorded latency
while a System One run is fresh -- compare quality, calls and cost first, latency with that in mind.

**Accounting rule:** report chat calls only (embeddings are latency, not the metric under study),
count true requests (hits + misses) for any system being compared, and keep caches query-scoped so
one query's work can never subsidise the next.

---

## 6. Measured state (2026-09-25)

All numbers are from the recorded runs in `sembench/` (replayed answers, latency and cost; chat
model `gpt-5.6-luna`; SWAN and LOTUS recorded back-to-back on 2026-09-23, Jev fresh on 2026-09-25).

| benchmark | SWAN | SWAN + Jev (ai_filter/ai_classify on TypeSafe) | LOTUS / PLOP |
|---|---|---|---|
| SemBench MOVIE (10q) | **0.818**, 18.8k calls, $1.53, 1,082s | 0.797, 18.8k, **$0.53**, 585s | LOTUS 0.780, 201k, $14.45, 10,654s |
| SemBench ECOMM (14q) | **0.699**, 16.6k, $8.02, 1,029s | 0.688, 16.6k, $7.39, 826s | LOTUS 0.637, 17.8k, $6.40, 1,990s |
| SemBench MMQA (11q) | **0.636**, 16.1k, $1.84, 1,059s | 0.603, 14.1k, $1.53, 884s | LOTUS 0.449, 19.0k, $2.43, 1,720s |
| agent_bench Q1–Q30 | **1.000**, 11,253 calls, $0.30, 824s | — | PLOP-DP 1.000, 13,602, $0.53, 1,080s |

Quality is each suite's own metric (F1 / ARI / row-multiset F1; agent_bench uses the deterministic
LIMIT-free PLOP ground truth). The tables per query are `sembench/typesafe_comparison_20260925.md`
and `sembench/AGENTBENCH/results/agentbench_comparison_swan_leaf2.md`; the extension-template build
reproduces them (`sembench/port_regression_20260925.md`).

---

## 7. What "clean" means here

A stage belongs in this pipeline only if it can state:

1. **its purpose** in terms of distinct prompts avoided;
2. **its invariant** — why it cannot change the result;
3. **its boundary behaviour** — what it does at a CTE, a projection, an aggregate;
4. **its test** — a parity assertion against the disabled stack, plus a call-count assertion that
   actually discriminates (a test that passes with the feature reverted guards nothing, and must
   say so in its header until it does).
