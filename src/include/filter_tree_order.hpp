//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/filter_tree_order.hpp
//
// Boolean tree over ai_filter leaves (AND / OR / NOT / LEAF) plus a per-row order planner used by
// ai_predicate: given each leaf's P(true), pick which leaf to evaluate next so the expensive LLM
// calls short-circuit as early as possible. ai_filter is assumed strictly true/false; evaluation is
// 2-valued with UNKNOWN standing for "not yet evaluated".
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"
#include "duckdb/common/string.hpp"

namespace duckdb {

enum class AIFilterTreeType : uint8_t { LEAF, NOT_OP, AND_OP, OR_OP };
enum class AITriState : uint8_t { TRI_FALSE = 0, TRI_TRUE = 1, TRI_UNKNOWN = 2 };

struct AIFilterTreeNode {
	AIFilterTreeType type;
	idx_t leaf_index = 0;                             // LEAF: index into the prompt/embedding arrays
	vector<unique_ptr<AIFilterTreeNode>> children;    // AND/OR: >=1; NOT: exactly 1

	static unique_ptr<AIFilterTreeNode> Leaf(idx_t index);
	static unique_ptr<AIFilterTreeNode> Op(AIFilterTreeType type, vector<unique_ptr<AIFilterTreeNode>> children);
	//! Highest leaf_index + 1 (== number of leaves for a well-formed 0..n-1 tree).
	idx_t LeafCount() const;
};

//! Compact prefix serialization, e.g. `A(L0,O(L1,N(L2)))` for `L0 AND (L1 OR NOT L2)`.
string AIFilterTreeSerialize(const AIFilterTreeNode &node);

//! Boolean normalization: NOT pushed to the leaves (De Morgan), same-type nodes flattened,
//! children canonically ordered, and duplicate/absorbed/common-factored subtrees collapsed. Every
//! rule is a KLEENE equivalence, since an AI leaf's verdict may be UNKNOWN. Its purpose is to make
//! the DP orderer's leaf-independence assumption true: two occurrences of one predicate are
//! perfectly correlated, and left un-collapsed they mis-cost the tree.
unique_ptr<AIFilterTreeNode> AINormalizeFilterTree(unique_ptr<AIFilterTreeNode> node);
//! Parse the serialization; returns nullptr on malformed input.
unique_ptr<AIFilterTreeNode> AIFilterTreeParse(const string &text);

//! Evaluate the tree from per-leaf tri-state values (2-valued short-circuit; UNKNOWN propagates).
AITriState AIFilterTreeEval(const AIFilterTreeNode &node, const vector<AITriState> &leaf_values);

//! Pick the next unevaluated leaf to evaluate so the boolean tree resolves at minimum expected LLM
//! cost. Uses the exact minimum-expected-cost DP for sequential boolean evaluation under leaf
//! independence (optimal for arbitrary AND/OR/NOT); falls back to a greedy cost/selectivity rank
//! when the tree has too many undetermined leaves. `p_true[i]` and `cost[i]` are this row's per-leaf
//! selectivity and LLM cost. Precondition: the tree currently evaluates to TRI_UNKNOWN.
idx_t AIFilterTreeChooseNextLeaf(const AIFilterTreeNode &node, const vector<AITriState> &leaf_values,
                                 const vector<double> &p_true, const vector<double> &cost);

//! Estimate P(the whole tree evaluates true) from per-leaf P(true), assuming leaf independence
//! (AND -> product, OR -> 1 - product of complements, NOT -> complement). Used by the speculative
//! pre-filter to decide, per row, whether the node is selective enough to be worth evaluating early.
double AIFilterTreeEstimateSelectivity(const AIFilterTreeNode &node, const vector<double> &p_true);

} // namespace duckdb
