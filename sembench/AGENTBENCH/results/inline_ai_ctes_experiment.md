# ai_inline_ai_ctes experiment (agent_bench Q1-Q30) — NEGATIVE RESULT, default stays OFF

Config: `SET ai_inline_ai_ctes=true` (mark AI-bearing materialized CTEs NEVER-materialize,
re-run DuckDB's CTEInlining), everything else at SWAN defaults, shared-verdict protocol.

| config | macro-F1 vs GT | calls | wall (s) | cost |
|---|---|---|---|---|
| SWAN default (materialized) | **0.972** | 15,174 | 1043 | $0.447 |
| SWAN + inlining | 0.740 | 14,317 | 965 | $0.391 |
| PLOP-DP | 0.978 | 13,591 | 1079 | $0.525 |

Seven queries break (F1 1.000 -> 0.000); all return 0 rows:

| q | rows (inline) | rows (default) | calls (inline) | calls (default) |
|---|---|---|---|---|
| Q21 | 0 | 2,712 | **0** | 91 |
| Q22 | 0 | 270 | **0** | 89 |
| Q23 | 0 | 625 | **0** | 91 |
| Q24 | 0 | 790 | **0** | 56 |
| Q26 | 0 | 125 | 397 | 1,431 |
| Q27 | 0 | 83 | 50 | 238 |
| Q28 | 0 | 10 | **787** | 62 |

Diagnosis notes:
- Q21-Q24 make **zero LLM calls** under inlining: the AI predicate is not merely mis-scoped,
  it is never evaluated -- the inlined plan loses the filter (or short-circuits to empty).
  This is a plan-construction failure, not a verdict-agreement problem.
- Q28 goes the other way (787 calls vs 62) and still returns nothing.
- CTE reference counts do NOT separate broken from working queries: Q15 (max_refs 2) and Q18
  (max_refs 3) are fine, while Q21 (2) and Q26 (3) break. So a "single-reference only" gate
  would not make this safe.
- Q26 alone was previously shown to lose rows with `ai_inline_ai_ctes=true` and EVERY other
  SWAN pass disabled, so DuckDB's inlining transform is implicated there independently.

## ROOT CAUSE (found 2026-09-21)

Marking a CTE `CTE_MATERIALIZE_NEVER` takes the FORCED branch of DuckDB's inliner:

```cpp
if (ref_count > 1) {
    if (cte.materialize == CTE_MATERIALIZE_NEVER) {
        Inline(..., true);   // "we have to inline it if possible" -- no safety check
        return;
    }
    PreventInlining prevent_inlining;      // <-- SKIPPED by the branch above
    ...  if (expr->IsVolatile()) prevent_inlining = true;
```

`PreventInlining` exists precisely to stop duplicating VOLATILE expressions -- and our
`ai_filter` is VOLATILE. So the marking does not ask DuckDB to inline safely; it *overrides the
guard that protects against duplicating our own functions*, and the duplicated-then-rebound AI
subtree silently yields nothing on complex shapes.

Evidence that it is duplication-of-a-bound-subtree, not inlining per se:
- A SIMPLE AI CTE referenced twice inlines correctly either way (26 rows = 26 rows).
- Q21 with DuckDB's own `AS NOT MATERIALIZED` returns the correct 2,712 rows, because the
  BINDER inlines before the plan exists -- each reference is bound separately with fresh
  bindings. Our flag flips an already-bound materialized CTE and asks the optimizer to
  duplicate + rebind it, which is the unsafe path.
- Moving the marking to the `pre_optimize_function` hook (before RunBuiltInOptimizers, the
  correct pipeline position) does NOT help: Q21 still gives 0 rows. Timing was not the issue.
- Same family as the semi-join reducer's refusal to reduce BY an AI-bearing neighbour, where
  the deep copy fails with "Failed to bind column reference". Both are: **AI expressions do not
  survive logical-plan duplication + rebinding.**

Next step is therefore NOT a narrower CTE gate but making AI expressions duplication-safe
(bind data + AIKeyBind's appended key columns must survive a rebind). That single fix would
unlock both this CTE win and reduction-by-AI-neighbour.

Conclusion: keep `ai_inline_ai_ctes` default OFF. The prize is real (Q26: 1,431 -> 397 calls,
matching PLOP's 389), but the pass as built is not result-preserving and the zero-call cases
show the failure is in plan construction, which must be understood before any narrower gate is
worth trying.

## FIX ATTEMPT 2 (2026-09-22): bind-time vs optimizer-time — NOT fixable from our hooks

Four hypotheses tested and eliminated:

1. **Wrong pipeline position.** Moved the marking from the post-built-ins optimizer slot to
   `pre_optimize_function` (runs immediately before `RunBuiltInOptimizers`, so DuckDB's own
   `cte_inlining` does the transform at its natural point). Q21 still 0 rows. Timing is not it.
2. **Bind data lost in the copy.** `LogicalOperator::Copy` serializes, and no AI `FunctionData`
   implements `Serialize` -- but a probe test (CTE with an always-true AI filter plus an `md5`
   probe column, referenced twice) shows rows, bindings and probe values all intact under
   inlining: 20 rows, 0 NULL probes, 0 mismatches. Bindings survive.
3. **Partial inlining of a CTE chain.** Marking only the AI-bearing CTE while a CTE that
   references it stays materialized. Marking the WHOLE chain instead: Q21 and Q26 both still 0.
4. **Nested-call or LATERAL shapes.** Synthetic CTE chains, `LATERAL UNNEST`, and nested
   `lower(ai_complete(..)) IN (..)` all inline correctly in isolation.

The decisive asymmetry: DuckDB's own `AS NOT MATERIALIZED` on Q21 returns the correct 2,712
rows, while setting `CTE_MATERIALIZE_NEVER` on the same query from any optimizer hook returns 0.
`NOT MATERIALIZED` is a PARSER attribute (set in transform_select.cpp), so it reaches the BINDER
and the plan is built inlined; our flag flips an already-bound `LogicalMaterializedCTE` and asks
the optimizer to duplicate it. Only the first path is safe here, and DuckDB exposes no
bind-time hook to an extension.

Conclusion: this approach cannot be fixed from the extension API. `ai_inline_ai_ctes` stays OFF.
The remaining viable route to the same win is to teach the semantic pull-up and the Yannakakis
reducer to SEE THROUGH a materialized-CTE operator (reach the predicate where it sits, never
duplicate it) -- which also fixes the multi-hop reduction gap and reduction-by-AI-neighbour.

## FIX ATTEMPT 3 (2026-09-22): "port PLOP's old bind-time logic" — would NOT deliver the win

PLOP's base (MorrilaPLOP/PLOP) has **no cte_inlining optimizer pass at all**. Its binder
(src/planner/binder.cpp:270-320) decides materialization up front with a narrow rule: a
multiply-referenced CTE is materialized ONLY if it ends in an aggregation, a DISTINCT modifier,
or an aggregate select item. Every agent_bench AI CTE therefore stays inlined, and PLOP's DP
sees one flat join tree (Q26 plan: 12 SEQ_SCAN, 10 HASH_JOIN, 0 CTE nodes).

Proposal tested: give AI-bearing CTEs that old rule, keep the new logic for the rest. Emulated
exactly by putting `AS NOT MATERIALIZED` on the CTEs at PARSER level (the same signal PLOP's
binder would produce) and running the full SWAN stack:

| query | rows (inlined) | rows (expected) | calls (inlined) | calls (default) |
|---|---|---|---|---|
| Q21 | 2,712 ✅ | 2,712 | **91 — no saving** | 91 |
| Q26 | **0** ❌ | 125 | 398 | 1,431 |

Both halves fail the purpose:
- where bind-time inlining is CORRECT (Q21) it saves NOTHING -- SWAN's passes already reach
  that predicate, so the materialized CTE was never the bottleneck there;
- where it SAVES (Q26, 398 vs 1,431 calls) it returns the wrong answer, and a fully-cached
  rerun reproduces 0 rows exactly -- structural, not verdict resampling.

So the missing piece is NOT the old materialization rule. PLOP reaches 125 rows AND 389 calls
on an inlined plan, so a correct inlined execution exists; ours is wrong on that shape. The
next concrete step is to diff our inlined Q26 plan against PLOP's flat plan operator by
operator, rather than to change materialization policy.

## WHERE THE CALLS ACTUALLY GO (Q26, measured per filter)

PLOP's own log, counted by prompt text, for one costmodel run of Q26:

| filter | PLOP calls | SWAN calls (materialized) | distinct values in the 125-row output |
|---|---|---|---|
| A = part | **5** | 1,000 | 5 |
| S = supplier | **6** | 45 | 4 |
| B = orders | **5** | 11 | -- |
| C = customer (CROSS JOIN, irreducible) | 373 | 373 | 25 |
| total | **389** | ~1,431 | |

PLOP evaluates each filter on exactly the handful of rows that survive the joins. This is NOT a
placement algorithm SWAN lacks: the semantic pull-up does the same thing, and with the CTEs
inlined SWAN issues **398 calls vs PLOP's 389**. The entire gap is that two filters are sealed
inside materialized CTEs where the pull-up cannot reach them.

## THE REMAINING BLOCKER, PINNED

In the inlined Q26 the individual CTEs are all fine (a_candidates 671 rows, s_candidates 29,
b_base 11, ps_bridge 1,259, c_context 25) and `b_base JOIN ps_bridge` correctly yields 5. But:

```sql
SELECT count(*) FROM a_candidates a1, a_candidates a2;   -- 0   (671 x 671 expected)
```

A CROSS PRODUCT of two non-empty relations returns ZERO rows. That is a hard correctness
failure in the inlined plan, not a verdict/semantics question -- which also retires the
"copies disagree on LLM verdicts" theory entirely.

Ruled out as the trigger (each reproduces correctly in isolation): bare ai_filter CTEs, nested
`lower(ai_complete(..)) IN (..)`, CTE chains, LATERAL UNNEST, 2 and 3 references, a reference
from another CTE, function stability (VOLATILE vs CONSISTENT_WITHIN_QUERY), and plain non-AI
CTEs of the same shape.

Next session's handle: bisect Q26 itself down to the empty cross product (it is deterministic
and needs no LLM calls once cached), rather than building synthetic cases upward.
