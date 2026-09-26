# govreport reorder benchmark (gpt-5-mini, 973 docs)

## Headline progression (cold single-pass, 9/9 correct each, concurrency 20)

| config | chat calls OFF→ON | cost OFF→ON |
|--------|-------------------|-------------|
| v1  frozen p_true (up-front), single-embed feature, SGD  | 19101→18475 (+3.3%)  | $9.10→$8.97 (+1.5%)  |
| v2  factored [pred,input,cos] feature, still frozen, SGD | 19101→19964 (-4.5%)  | $9.10→$9.76 (-7.3%)  |
| v3  + JIT re-predict per batch (within-query learning), SGD | 19101→16512 (+13.6%) | $9.10→$8.01 (+12.0%) |
| **v4  + full-batch training + Adam optimizer**           | **19101→15927 (+16.6%)** | **$9.10→$7.53 (+17.2%)** |

The decisive change is **v2→v3**: re-predicting `p_true` per batch against the live (training) model
instead of a frozen up-front pass, so the model steers the very query that trains it. **v3→v4**
(full-batch gradient + Adam) then removes the SGD regressions (conj/q0, conj/q22) and halves the
worst one (disj/q44 -28%→-16%). Snapshots saved as `bench_results_v{1,2,3,4}.json`.

## MLP feature A/B: factored `[pred,input,cos]` vs single `emb(prompt)` (9 queries, replay, conc 20)

Env-selectable via `AI_MLP_FEATURE=factored|prompt`. Run with latency replay ON so online training
happens at a faithful cadence. Both feature choices are 9/9 exact-match.

| feature | calls OFF->ON | cost OFF->ON | latency OFF->ON |
|---------|---------------|--------------|-----------------|
| factored (769-d)      | -17.7% | -17.2% | 2136s->1849s (**-13.5%, ON faster**) |
| single emb(prompt) (384-d) | -17.5% | -16.8% | 2139s->2748s (**+28.5%, ON slower**) |

Calls/cost/accuracy tie (the JIT + Adam + DP drive the savings, not the feature). Latency is the
decisive difference: single-emb(prompt) re-embeds the *document per leaf* (`973 x n` encodings) vs
factored embedding it *once, shared across leaves* (`973 + n`). On 10-leaf queries prompt is ~1.7-2.3x
the latency of factored (conj/q44 159s vs 362s; mix/q44 325s vs 561s), turning a call saving into a
latency loss. **Decision: keep the factored feature** -- equal quality, strictly better latency.

---


Comparison of `DUCKDB_AI_REORDER` **OFF** (filters in the query's written order, DuckDB's normal
conjunction execution) vs **ON** (one per-row-adaptive `ai_predicate` that orders each row's leaves
by the exact min-expected-cost DP). Chat calls flow through a persistent global cache proxy so each
unique prompt is paid once and re-runs are $0; the cache replays original latency, so ON (which
reuses OFF's prompts) is not artificially fast. CPU sentence-transformers embeddings are $0.

Metric sources: chat calls + prompt/completion tokens from the cache proxy (chat-only, embeddings
bypass it); cost_usd from `ai_usage()` (chat-only, embeddings free); latency is the wall-clock of the
filtering statement (includes ON's embed CPU time).

## Results (v1 MLP: input = full-prompt embedding only, cold/untrained)

| query | leaves | match | rows | calls OFF→ON | Δcalls | cost OFF→ON | latency OFF→ON |
|-------|--------|-------|------|-------------|--------|-------------|----------------|
| conj/q0  | 2  | yes | 41  | 1186→1245 | -5%  | $0.5376→$0.5699 | 55→125s |
| conj/q22 | 6  | yes | 2   | 1241→1496 | -21% | $0.5825→$0.7158 | 72→347s |
| conj/q44 | 10 | yes | 1   | 2631→1269 | +52% | $1.3019→$0.6117 | 143→460s |
| disj/q0  | 2  | yes | 474 | 1724→1674 | +3%  | $0.7840→$0.7698 | 74→137s |
| disj/q22 | 6  | yes | 969 | 2648→1642 | +38% | $1.2910→$0.8722 | 135→334s |
| disj/q44 | 10 | yes | 962 | 1524→2571 | -69% | $0.7346→$1.2548 | 94→538s |
| mix/q0   | 2  | yes | 474 | 1724→1674 | +3%  | $0.7840→$0.7698 | 75→142s |
| mix/q22  | 6  | yes | 408 | 3631→3562 | +2%  | $1.6768→$1.6666 | 193→423s |
| mix/q44  | 10 | yes | 568 | 2792→3342 | -20% | $1.4061→$1.7356 | 157→592s |
| **TOTAL**| | **9/9** | | **19101→18475** | **+3.3%** | **$9.0983→$8.9661** | **998→3098s** |

Actual OpenAI spend for the whole run: **$6.87** (13,763 unique prompts paid once; 24,884 cache hits
served free).

## Findings

1. **Correctness: exact.** All 9 result id-sets are identical OFF vs ON — the boolean tree + per-row
   ordering is faithful; only *which* LLM calls happen changes.
2. **Cold-model calls/cost: net-neutral (-3.3% calls, -1.5% cost), high variance.** conj/q44 saved
   52% of calls and disj/q22 38%, but disj/q44 was 69% worse. With an untrained selectivity MLP over
   a single data chunk, per-row `P(true)` is essentially random, so per-row ordering is a coin-flip
   against the written order; wins and losses cancel. The mechanism works; it has no signal yet.
3. **Latency: ON ~3.1x slower — this is `ai_embed` CPU, not LLM work or training.** The v1 rewrite
   embeds each leaf's full `question || summary` prompt, re-encoding the same document `n_leaves`
   times: `973 x n_leaves` long-text CPU encodings serialized before the filter. It's $0 (CPU) but
   wall-clock. The delta scales linearly with n_leaves (conj/q0 2L: +71s; conj/q22 6L: +275s). One
   16-sample MLP training step is ~0.85 ms and fully hidden behind the ~2.5 s network call, so
   training contributes ~0.

## Next: factored MLP input `[predicate_emb, input_emb, cos_sim]`

Splitting each prompt into predicate (question) and input (document) and feeding the MLP
`[emb(predicate), emb(input), cosine(emb(predicate), emb(input))]`:
- gives a real selectivity prior (question<->document cosine) even before training, and
- fixes the latency: the shared document embeds dedup to **once per doc** (973 + n_leaves embeds
  instead of 973 x n_leaves).

---

## Speculative pre-filter (MLP-gated partial pushdown) — `bench_speculative.py`

`speculative_ai_filter_with_embed` sits at the pushed-down (leaf) position when `DUCKDB_SEMANTIC_PULLUP`
lifts a single-table `ai_filter` above a join. Per row it MLP-estimates `P(pass)`; likely-to-fail rows
are pre-evaluated (LLM) and pruned **before** the join, likely-to-pass rows pass through for the
pulled-up filter to re-check (a cache hit). A warm-up budget guarantees the MLP always sees labels.

Two Morrila-style queries (a semantic filter over a small relation, then an **expansive join**),
adapted as a govreport **self-join on org** (50 docs → 2450 join rows, 49× fan-out), gpt-5-mini, replay
ON. Three result-preserving plans: **baseline** (no pull-up, filter pushed to the base scan),
**pull-up only** (`AI_SPECULATIVE_THRESHOLD=0` → the leaf node passes every row through == the old
placeholder; filter runs above the join on 2450 rows), **speculative** (default threshold 0.5).

| query | plan | chat calls | latency |
| --- | --- | --- | --- |
| **Q1** both filters on `a` → one combined `A(L0,L1)` node | baseline | 68 | 17.7s |
| | pull-up only | 68 | 28.6s |
| | **speculative** | **65** | **12.8s** |
| **Q2** health on `a`, security on `b` → one node per leaf | baseline | 100 | 17.7s |
| | pull-up only | 100 | 28.2s |
| | **speculative** | 100 | 18.4s |

All id-sets identical across the three plans (result-preserving, verified by md5).

### Findings

1. **Pull-up above an expansive join is the slow plan (~28s):** the filter re-evaluates on all 2450
   join rows. Prompts dedup so LLM *call count* is unchanged, but the join materialization + per-row
   filter work is pure overhead.
2. **The speculative pre-filter recovers pushdown efficiency:** it prunes likely-to-fail rows at the
   leaf before the 49× fan-out — landing at/below baseline (Q1 −55% vs pull-up-only, −28% vs baseline;
   Q2 −35% vs pull-up-only, ~even vs baseline).
3. **Combined node can cut calls too (Q1 68→65):** pre-evaluating a likely-fail row lets the `AND`
   short-circuit the second leaf. Q2's per-side single-leaf nodes have no cross-leaf short-circuit, so
   calls stay flat.
4. Single-run wall-clock has variance; the robust signal is the ordering **pull-up-only ≫ speculative
   ≈ baseline**. `AI_SPECULATIVE_THRESHOLD` tunes aggressiveness (0 = never prune == placeholder,
   1 = always evaluate == full pushdown).

---

## Where should the AI filter run? use-case suite — `bench_speculative.py` (v2)

Reorder is ON in every config (it almost always helps, so it's a constant); the three configs are
purely *where the filter runs*: **push_down** (base scans), **pull_up** (lifted above the join, leaf
node pass-through, `AI_SPECULATIVE_THRESHOLD=0`), **pull_up+speculative** (leaf gate 0.5). govreport
self-joins over 50 docs sweep the join's cardinality effect; cached health/security predicates, replay
ON. LLM calls = successful proxy calls (OpenAI or global cache).

| use case | shape | push_down | pull_up | speculative |
| --- | --- | --- | --- | --- |
| UC1 | single filter + **selective** join (10/50 survive) | 50 / 9.3s | **10 / 9.6s** | 50 / 10.0s |
| UC2 | single filter + **1:1** join | 50 / 9.2s | 50 / 12.9s | 50 / 10.2s |
| UC3 | two filters, **same** relation + expansive join | 65 / 12.9s | 67 / 23.6s | 65 / 12.7s |
| UC4a | two filters **split** + expansive **cross** join | 100 / 17.7s | 100 / 24.0s | 100 / 18.8s |
| UC4b | two filters **split** + **sparse** join | 100 / 17.7s | **66 / 14.0s** | 100 / 18.1s |

All id-sets identical across configs (result-preserving).

### Findings

1. **push_down is the robust default** — best or near-best latency everywhere, never catastrophic.
2. **pull_up cuts LLM calls exactly when the join REDUCES** (UC1: 10 vs 50, the Yannakakis win —
   requires pull-up; Yannakakis alone leaves the filter below the join) **or enables cross-filter
   short-circuit** (UC4b: the two filters land in one node above the join, 66 vs 100; a cross-product
   join is the worst case, UC4a, where every b has a passing partner). Elsewhere it ties.
3. **The speculative gate does not win on its own** — it ties push_down and HURTS on reducing joins
   (pre-evaluates at the leaf, below the join, spending calls the join would discard). Its only niche
   is a filter forced above an expansive join. Recommendation: keep it off by default;
   push down by default; pull up only when a cardinality-reducing op / same-key short-circuit is
   present; make the gate join-expansion-aware (bypass when a reducing op sits above it).

### Concurrency fix uncovered by the suite (`ai_functions.cpp`)

`ai_filter_with_embed`'s per-row worker pool re-issued duplicate prompts on a duplicated (join-output)
input, because concurrent workers all missed the query cache before any wrote it back — UC3 pull_up
was **1320 calls / 181s**, UC4a **833 / 92s**. Fixed with (a) row-tuple dedup (one representative per
distinct prompt tuple, fan out) and (b) **non-blocking per-leaf-prompt single-flight**: one LLM call
per distinct prompt; a worker hitting an in-flight prompt parks its (row, leaf) and moves on, and the
fetcher routes all parked rows when the result lands. Result: UC3 pull_up 1320→65 calls, UC4a 833→100,
and latency 181s→24s / 92s→24s. Deadlock-free; 47/47 optimizer tests pass (x3 stable).
