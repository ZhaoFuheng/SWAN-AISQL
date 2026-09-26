//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/ai_filter_tree_build.hpp
//
// Shared helpers for turning a boolean tree of ai_filter leaves into an AIFilterTreeNode + the
// per-leaf prompt/(predicate,input) text. Used by both the reorder rewrite (ai_predicate_rewrite)
// and the semantic-filter pull-up's speculative pre-filter (semantic_filter_pullup).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "filter_tree_order.hpp"

#include <utility>

namespace duckdb {
class Expression;
class ClientContext;

//! Is `expr` a call to ai_filter?
bool AIIsFilterLeaf(const Expression &expr);
//! Is `expr` a pure boolean tree (AND/OR/NOT) over ai_filter leaves and nothing else?
bool AIIsPureFilterBoolean(const Expression &expr);
//! Number of ai_filter leaves in `expr`.
idx_t AICountFilterLeaves(const Expression &expr);
//! Build the boolean tree for a pure-ai-boolean `expr`, appending each leaf's prompt (a copy) to
//! `leaf_prompts` so leaf indices line up with argument positions. Same-type nodes are flattened
//! to keep the tree n-ary.
unique_ptr<AIFilterTreeNode> AIBuildFilterTree(const Expression &expr, vector<unique_ptr<Expression>> &leaf_prompts);
//! Look past AIKeyBind's all-columns-NULL guard to the prompt underneath.
const Expression &AIUnwrapNullGuard(const Expression &expr);
//! Split a prompt into (predicate text, input text): foldable/constant parts (the question, shared
//! across rows) vs row-varying parts (the document). Falls back to (prompt, prompt) if no clean
//! split. Needs `context` to bind `concat` when a side has multiple parts.
std::pair<unique_ptr<Expression>, unique_ptr<Expression>> AISplitPrompt(ClientContext &context,
                                                                        const Expression &prompt);

//===----------------------------------------------------------------------===//
// Generalized MIXED AI-comparison leaves: ai_filter(p), or a comparison (=,!=,>,<,>=,<=) between an
// ai_classify / ai_score / ai_complete call and a constant. Shared by the reorder rewrite and the
// speculative pull-up so both can lift/reorder a boolean tree mixing all four AI functions.
//===----------------------------------------------------------------------===//

//! One leaf of a mixed AI-comparison boolean tree.
struct AIMixedLeaf {
	char kind = 'F';                     //!< F ai_filter / C classify / S score / M complete
	char op = 0;                         //!< op char: e/n/g/l/G/L single compare; i IN / I NOT IN (set); 0 for F
	const Expression *ai_call = nullptr; //!< the ai_* call
	const Expression *cst = nullptr;     //!< the constant compared against (single compare; null for F/set)
	vector<const Expression *> cst_set;  //!< the set members, for IN ('i') / NOT IN ('I') on classify/complete
};

//! Detect an AI leaf (ai_filter or a supported comparison of classify/score/complete vs a constant);
//! fills `out` and returns true if `expr` is such a leaf.
bool AIDetectMixedLeaf(const Expression &expr, AIMixedLeaf &out);
//! Is `expr` a boolean tree (AND/OR/NOT) whose leaves are all AI leaves?
bool AIIsMixedBoolean(const Expression &expr);
//! Number of AI leaves in a mixed boolean tree.
idx_t AICountMixedLeaves(const Expression &expr);
//! Does `expr` contain any non-ai_filter (classify/score/complete) leaf?
bool AIAnyNonFilterLeaf(const Expression &expr);
//! Build the boolean tree (flattening same-type nodes) and append leaves in tree order to `leaves`.
unique_ptr<AIFilterTreeNode> AIBuildMixedTree(const Expression &expr, vector<AIMixedLeaf> &leaves);
//! Per-leaf node arguments matching the (speculative_)ai_function_with_embed signature: the baked,
//! scalar-exact call prompt; the MLP-feature (predicate, input) split carrying the comparison; and the
//! joined `meta_str` (per-leaf `F` or `<K>:<op>:<val>` tokens, ';'-separated). Returns false if any leaf
//! is not bakeable (classify with an instruction arg, complete with a schema arg, unevaluable RHS, ...),
//! in which case the caller should leave the plan untouched.
bool AIBuildMixedLeafArgs(ClientContext &context, const vector<AIMixedLeaf> &leaves,
                          vector<unique_ptr<Expression>> &call_prompts, vector<unique_ptr<Expression>> &feat_pred,
                          vector<unique_ptr<Expression>> &feat_input, string &meta_str);

} // namespace duckdb
