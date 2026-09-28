//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/ai_dedup.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/types/value.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/shared_ptr.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {

class ClientContext;
class ColumnDataCollection;
class DataChunk;
class BoundFunctionExpression;
class Expression;
struct AIFilterTreeNode;

//! If `call` is an ai_function_with_embed node, set its LIMIT (stop evaluating once this many rows pass);
//! a no-op otherwise. Used by the AILimitPushdown optimizer pass to push a LIMIT k into the filter node.
void AISetFilterLimit(Expression &call, int64_t limit);
//! Mark a speculative leaf node to stand down (a LIMIT was pushed into the evaluation above it).
void AISetSpeculativeStandDown(Expression &call);

//===--------------------------------------------------------------------===//
// Streaming-dedup seams (used by the streaming PhysicalAIRegion to fire AI in waves as distinct keys arrive)
//===--------------------------------------------------------------------===//

//! The child-output columns `eval_call` reads (its dedup key).
vector<idx_t> AIDedupKeyCols(const BoundFunctionExpression &eval_call);

//! True iff `eval_call` is an ai_function_with_embed / speculative_ai_function_with_embed node (evaluated via
//! the reorder worker pool + count-weighted LIMIT early-stop), as opposed to a plain scalar AI function.
bool AIDedupIsFilterNode(const BoundFunctionExpression &eval_call);

//! Evaluate ONE wave of distinct representative rows for an ai_function_with_embed NODE. `wave_reps` holds the
//! rep rows (the operator's child types), scanned in ordinal order. Fills out_result[i]/out_valid[i] (0/1) for
//! the i-th rep. `remaining_limit` = k minus the OUTPUT rows already passed (-1 = no early-stop);
//! `wave_weights[i]` = rep i's fan-out count, so the pool stops once the passing reps' broadcast rows sum to
//! `remaining_limit`. Reuses the same worker pool as the scalar node (short-circuit + DP reorder).
void AIDedupFireWave(ClientContext &context, const BoundFunctionExpression &eval_call, ColumnDataCollection &wave_reps,
                     const string &query_text, int64_t remaining_limit, const vector<idx_t> &wave_weights,
                     vector<char> &out_result, vector<char> &out_valid);

//! Evaluate ONE wave of distinct representative rows for a plain SCALAR AI function (ai_classify / ai_score /
//! ai_complete / ai_filter): runs the scalar function itself over the rep rows. Fills out_values[i] with the
//! scalar's return-type Value (NULL for a NULL input). Overlap-only -- scalars are never under a LIMIT here.
void AIDedupFireWaveScalar(ClientContext &context, const BoundFunctionExpression &eval_call,
                           ColumnDataCollection &wave_reps, vector<Value> &out_values);

//===--------------------------------------------------------------------===//
// Factorized (AISQLMapData) eval seams -- the AI node consumes ONE factorized DataChunk directly: columns
// [0, ncol) are the distinct rep rows (the AI call's key/input columns), the trailing column is the BIGINT
// `__count` multiplicity. This makes AISQLMapData the explicit input to the AI function (two-currency model):
// the arg expressions read the rep-row columns; `__count` is the fan-out weight for the count-weighted stop.
//===--------------------------------------------------------------------===//

//! Node path: evaluate the ai_function_with_embed tree over a factorized chunk. `remaining_limit` = k minus
//! output rows already passed (-1 = none). Weights are read from the trailing `__count` column.
void AIDedupFireWaveFactorized(ClientContext &context, const BoundFunctionExpression &eval_call, DataChunk &factorized,
                               const string &query_text, int64_t remaining_limit, vector<char> &out_result,
                               vector<char> &out_valid);

//! Scalar path: run the scalar AI function over the factorized chunk's rep rows (`__count` ignored).
void AIDedupFireWaveScalarFactorized(ClientContext &context, const BoundFunctionExpression &eval_call,
                                     DataChunk &factorized, vector<Value> &out_values);

//===--------------------------------------------------------------------===//
// Factor-graph seams (used by PhysicalAIFactorGraph: unary leaves evaluate over side dictionaries,
// binary leaves over surviving pair domains -- the cross product is never materialized)
//===--------------------------------------------------------------------===//

//! One leaf of a factor-decomposed conjunctive node: the side ids its argument expressions read.
struct AIFactorLeaf {
	vector<idx_t> sides; //!< ascending; size 1 = unary (side predicate), size 2 = binary (edge)
};

//! Classify each leaf of a RESOLVED conjunctive folded node by the sides its argument columns
//! belong to (`side_of[col]` = owning side per combined-layout column index). Returns false when
//! the node's tree is not a pure conjunction of leaves or any leaf reads zero or >2 sides.
bool AIFactorDecompose(const BoundFunctionExpression &node, const vector<idx_t> &side_of,
                       vector<AIFactorLeaf> &out_leaves);

//! Flatten a CALL-PROMPT expression into its ordered concat operands (a non-concat expression
//! yields itself). Used to split pair prompts into a cacheable left-member prefix and a
//! right-member suffix for explicit provider prompt caching.
void AIFactorPromptOperands(const Expression &call_prompt, vector<const Expression *> &ops);

//! Evaluate ONE factor-graph unit (a member or pair) of a conjunctive sub-node from its baked
//! leaf prompts: leaves run in order with conjunction short-circuit; each leaf call goes through
//! the client (query cache, single-flight, hedging, the global concurrency gate). `valid` false
//! on a failed call. Backs the streaming graph executor's worker tasks.
bool AIFactorEvalUnit(ClientContext &context, const BoundFunctionExpression &sub_node,
                      const vector<string> &leaf_prompts, const string &query_text, bool &valid,
                      const vector<string> *leaf_prefixes = nullptr, idx_t expected_reuse = 0);

//! Clone a conjunctive folded node restricted to `leaf_ids`, remapping every column reference
//! through `index_map` (combined layout -> evaluation-chunk layout). The clone is a plain
//! (non-speculative, unlimited) node evaluable via AIDedupFireWaveFactorized.
unique_ptr<Expression> AIFactorSubNode(const BoundFunctionExpression &node, const vector<idx_t> &leaf_ids,
                                       const vector<idx_t> &index_map);

//===--------------------------------------------------------------------===//
// Per-leaf seams -- the leaf-factorized region. A node's reps are kept PER LEAF (one dictionary on that
// leaf's own columns: |A| + |B| reps, never |A| x |B|), a leaf is evaluated on its own over its distinct
// reps, and the order in which a ROW's leaves are asked is decided per row over rep-level predictions.
// Each seam is one slice of the per-row batch machinery, applied to a single leaf.
//===--------------------------------------------------------------------===//
//! Texts of one leaf over a set of reps (parallel vectors; `valid[i]` = 0 when the prompt was NULL).
struct AILeafTexts {
	vector<string> prompt, pred_text, input_text;
	vector<double> cost;
	vector<char> valid;
	idx_t Size() const {
		return prompt.size();
	}
};
//! Number of leaves of the folded node.
idx_t AILeafCount(const BoundFunctionExpression &eval_call);
//! The node's boolean tree (leaf indices match the seams below).
shared_ptr<AIFilterTreeNode> AILeafTree(const BoundFunctionExpression &eval_call);
//! The child-output columns leaf `leaf` reads -- its own dedup key.
vector<idx_t> AILeafKeyCols(const BoundFunctionExpression &eval_call, idx_t leaf);
//! Materialize leaf `leaf`'s prompt / predicate text / input text over `rows` (child types), appending to `out`.
void AILeafBuildTexts(ClientContext &context, const BoundFunctionExpression &eval_call, idx_t leaf, DataChunk &rows,
                      AILeafTexts &out);
//! P(true) per rep from the live selectivity model (one batched embed for the whole set). Neutral 0.5 while
//! the model is cold, for image leaves, and for invalid reps -- the same rules the per-row batch applies.
void AILeafPredict(const BoundFunctionExpression &eval_call, const AILeafTexts &texts, const string &query_text,
                   vector<double> &p_out);
//! The selectivity feature per rep (ONE batched embed for the whole set, whether or not the model is warm),
//! empty for image leaves and invalid reps. Kept on the rep so a later re-prediction is a forward pass only.
void AILeafFeatures(const BoundFunctionExpression &eval_call, const AILeafTexts &texts, const string &query_text,
                    vector<vector<float>> &feats_out);
//! P(true) from a stored feature: the live model's forward pass (0.5 for an empty feature or a cold model).
double AILeafPredictFeature(const vector<float> &feat);
//! The selectivity model's training-step counter, so a stored prediction can be recognised as stale.
uint64_t AISelectivityTrainSteps();
//! Evaluate leaf `leaf` over `texts`: the LLM verdict per rep, through the same worker pool, single-flight,
//! speculative gate and MLP training as the per-row batch. out_valid[i] = 0 -> no verdict (NULL).
void AILeafEvaluate(ClientContext &context, const BoundFunctionExpression &eval_call, idx_t leaf,
                    const AILeafTexts &texts, const string &query_text, vector<char> &out_result,
                    vector<char> &out_valid);

} // namespace duckdb
