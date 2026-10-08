#include "filter_tree_order.hpp"

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace duckdb {

unique_ptr<AIFilterTreeNode> AIFilterTreeNode::Leaf(idx_t index) {
	auto node = make_uniq<AIFilterTreeNode>();
	node->type = AIFilterTreeType::LEAF;
	node->leaf_index = index;
	return node;
}

unique_ptr<AIFilterTreeNode> AIFilterTreeNode::Op(AIFilterTreeType type,
                                                  vector<unique_ptr<AIFilterTreeNode>> children) {
	auto node = make_uniq<AIFilterTreeNode>();
	node->type = type;
	node->children = std::move(children);
	return node;
}

idx_t AIFilterTreeNode::LeafCount() const {
	if (type == AIFilterTreeType::LEAF) {
		return leaf_index + 1;
	}
	idx_t n = 0;
	for (auto &child : children) {
		n = std::max(n, child->LeafCount());
	}
	return n;
}

//===--------------------------------------------------------------------===//
// Serialization
//===--------------------------------------------------------------------===//
static char TypeTag(AIFilterTreeType type) {
	switch (type) {
	case AIFilterTreeType::NOT_OP:
		return 'N';
	case AIFilterTreeType::AND_OP:
		return 'A';
	case AIFilterTreeType::OR_OP:
		return 'O';
	default:
		return 'L';
	}
}

string AIFilterTreeSerialize(const AIFilterTreeNode &node) {
	if (node.type == AIFilterTreeType::LEAF) {
		return "L" + std::to_string(node.leaf_index);
	}
	string out(1, TypeTag(node.type));
	out += "(";
	for (idx_t i = 0; i < node.children.size(); i++) {
		if (i > 0) {
			out += ",";
		}
		out += AIFilterTreeSerialize(*node.children[i]);
	}
	out += ")";
	return out;
}

//===--------------------------------------------------------------------===//
// Normalization
//
// Every rule here is a Kleene (three-valued) equivalence, which is the logic an AI leaf actually
// lives in: its verdict can be UNKNOWN, and AIFilterTreeEval propagates that. De Morgan,
// flattening, idempotence, absorption and distribution all hold in Kleene. COMPLEMENT does not
// (`a AND NOT a` is UNKNOWN, not FALSE, when a is UNKNOWN) and is deliberately absent -- there is
// no constant node to fold it into either.
//
// The point is not to save calls directly: a repeated leaf already costs one call, since the two
// occurrences share a prompt and the local cache serves the second. The point is that the DP
// orderer costs the tree assuming leaves are INDEPENDENT. Two occurrences of the same predicate
// are perfectly correlated, so an un-normalized tree is mis-costed and can be ordered badly.
// Collapsing them to one leaf makes the model's assumption true.
//===--------------------------------------------------------------------===//

//! Canonical child order, so structurally equal subtrees compare equal as strings regardless of
//! how the SQL was written. AND/OR are commutative, so sorting changes no meaning.
static void SortChildren(vector<unique_ptr<AIFilterTreeNode>> &children) {
	std::sort(children.begin(), children.end(),
	          [](const unique_ptr<AIFilterTreeNode> &a, const unique_ptr<AIFilterTreeNode> &b) {
		          return AIFilterTreeSerialize(*a) < AIFilterTreeSerialize(*b);
	          });
}

//! Does `haystack` (an AND/OR node) have a child equal to `needle`?
static bool HasChild(const AIFilterTreeNode &haystack, const string &needle) {
	for (auto &child : haystack.children) {
		if (AIFilterTreeSerialize(*child) == needle) {
			return true;
		}
	}
	return false;
}

static unique_ptr<AIFilterTreeNode> Normalize(unique_ptr<AIFilterTreeNode> node);

//! NOT pushed to the leaves (De Morgan), double negation removed.
static unique_ptr<AIFilterTreeNode> PushNot(unique_ptr<AIFilterTreeNode> child) {
	if (child->type == AIFilterTreeType::NOT_OP) {
		return Normalize(std::move(child->children[0])); // NOT NOT x == x
	}
	if (child->type == AIFilterTreeType::AND_OP || child->type == AIFilterTreeType::OR_OP) {
		const auto flipped =
		    child->type == AIFilterTreeType::AND_OP ? AIFilterTreeType::OR_OP : AIFilterTreeType::AND_OP;
		vector<unique_ptr<AIFilterTreeNode>> negated;
		for (auto &grandchild : child->children) {
			negated.push_back(PushNot(std::move(grandchild)));
		}
		return Normalize(AIFilterTreeNode::Op(flipped, std::move(negated)));
	}
	vector<unique_ptr<AIFilterTreeNode>> one;
	one.push_back(Normalize(std::move(child)));
	return AIFilterTreeNode::Op(AIFilterTreeType::NOT_OP, std::move(one));
}

//! Factor the literals common to EVERY branch of a disjunction out of it:
//! `(a AND b) OR (a AND c)` -> `a AND (b OR c)`. Distribution is a Kleene equivalence, and the
//! factored form lets the orderer see one `a` instead of one per branch.
static unique_ptr<AIFilterTreeNode> FactorCommon(unique_ptr<AIFilterTreeNode> node) {
	if (node->type != AIFilterTreeType::OR_OP || node->children.size() < 2) {
		return node;
	}
	for (auto &child : node->children) {
		if (child->type != AIFilterTreeType::AND_OP) {
			return node; // every branch must be a conjunction to factor across them
		}
	}
	// common = the conjuncts present in all branches
	vector<string> common;
	for (auto &candidate : node->children[0]->children) {
		const string key = AIFilterTreeSerialize(*candidate);
		bool in_all = true;
		for (idx_t i = 1; i < node->children.size() && in_all; i++) {
			in_all = HasChild(*node->children[i], key);
		}
		if (in_all) {
			common.push_back(key);
		}
	}
	if (common.empty()) {
		return node;
	}
	vector<unique_ptr<AIFilterTreeNode>> factored;
	vector<unique_ptr<AIFilterTreeNode>> remainders;
	for (auto &branch : node->children) {
		vector<unique_ptr<AIFilterTreeNode>> rest;
		for (auto &conjunct : branch->children) {
			const string key = AIFilterTreeSerialize(*conjunct);
			if (std::find(common.begin(), common.end(), key) == common.end()) {
				rest.push_back(std::move(conjunct));
			} else if (factored.size() < common.size() &&
			           std::find_if(factored.begin(), factored.end(), [&key](const unique_ptr<AIFilterTreeNode> &f) {
				           return AIFilterTreeSerialize(*f) == key;
			           }) == factored.end()) {
				factored.push_back(std::move(conjunct));
			}
		}
		if (rest.empty()) {
			// this branch was exactly the common part, so the disjunction reduces to it
			return Normalize(AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(factored)));
		}
		remainders.push_back(rest.size() == 1 ? std::move(rest[0])
		                                      : AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(rest)));
	}
	factored.push_back(AIFilterTreeNode::Op(AIFilterTreeType::OR_OP, std::move(remainders)));
	return Normalize(AIFilterTreeNode::Op(AIFilterTreeType::AND_OP, std::move(factored)));
}

static unique_ptr<AIFilterTreeNode> Normalize(unique_ptr<AIFilterTreeNode> node) {
	if (node->type == AIFilterTreeType::LEAF) {
		return node;
	}
	if (node->type == AIFilterTreeType::NOT_OP) {
		return PushNot(std::move(node->children[0]));
	}
	// AND / OR: normalize children, splice same-type children in (associativity)
	vector<unique_ptr<AIFilterTreeNode>> children;
	for (auto &child : node->children) {
		auto sub = Normalize(std::move(child));
		if (sub->type == node->type) {
			for (auto &grandchild : sub->children) {
				children.push_back(std::move(grandchild));
			}
		} else {
			children.push_back(std::move(sub));
		}
	}
	SortChildren(children);
	// idempotence: x AND x == x, x OR x == x
	vector<unique_ptr<AIFilterTreeNode>> unique_children;
	string previous;
	for (auto &child : children) {
		string key = AIFilterTreeSerialize(*child);
		if (key == previous) {
			continue;
		}
		previous = std::move(key);
		unique_children.push_back(std::move(child));
	}
	// absorption: x AND (x OR y) == x, and dually x OR (x AND y) == x
	const auto inner = node->type == AIFilterTreeType::AND_OP ? AIFilterTreeType::OR_OP : AIFilterTreeType::AND_OP;
	vector<unique_ptr<AIFilterTreeNode>> kept;
	for (auto &child : unique_children) {
		bool absorbed = false;
		if (child->type == inner) {
			for (auto &other : unique_children) {
				if (other.get() == child.get() || !other) {
					continue;
				}
				if (HasChild(*child, AIFilterTreeSerialize(*other))) {
					absorbed = true;
					break;
				}
			}
		}
		if (!absorbed) {
			kept.push_back(std::move(child));
		}
	}
	if (kept.size() == 1) {
		return std::move(kept[0]);
	}
	return FactorCommon(AIFilterTreeNode::Op(node->type, std::move(kept)));
}

unique_ptr<AIFilterTreeNode> AINormalizeFilterTree(unique_ptr<AIFilterTreeNode> node) {
	return Normalize(std::move(node));
}

static unique_ptr<AIFilterTreeNode> ParseNode(const string &s, idx_t &pos);

static bool ParseChildren(const string &s, idx_t &pos, vector<unique_ptr<AIFilterTreeNode>> &out) {
	if (pos >= s.size() || s[pos] != '(') {
		return false;
	}
	pos++; // consume '('
	while (true) {
		auto child = ParseNode(s, pos);
		if (!child) {
			return false;
		}
		out.push_back(std::move(child));
		if (pos >= s.size()) {
			return false;
		}
		if (s[pos] == ',') {
			pos++;
			continue;
		}
		if (s[pos] == ')') {
			pos++;
			return true;
		}
		return false;
	}
}

static unique_ptr<AIFilterTreeNode> ParseNode(const string &s, idx_t &pos) {
	if (pos >= s.size()) {
		return nullptr;
	}
	const char tag = s[pos++];
	if (tag == 'L') {
		idx_t start = pos;
		while (pos < s.size() && s[pos] >= '0' && s[pos] <= '9') {
			pos++;
		}
		if (pos == start) {
			return nullptr;
		}
		return AIFilterTreeNode::Leaf(static_cast<idx_t>(std::stoull(s.substr(start, pos - start))));
	}
	AIFilterTreeType type;
	if (tag == 'N') {
		type = AIFilterTreeType::NOT_OP;
	} else if (tag == 'A') {
		type = AIFilterTreeType::AND_OP;
	} else if (tag == 'O') {
		type = AIFilterTreeType::OR_OP;
	} else {
		return nullptr;
	}
	vector<unique_ptr<AIFilterTreeNode>> children;
	if (!ParseChildren(s, pos, children) || children.empty()) {
		return nullptr;
	}
	if (type == AIFilterTreeType::NOT_OP && children.size() != 1) {
		return nullptr;
	}
	return AIFilterTreeNode::Op(type, std::move(children));
}

unique_ptr<AIFilterTreeNode> AIFilterTreeParse(const string &text) {
	idx_t pos = 0;
	auto node = ParseNode(text, pos);
	if (!node || pos != text.size()) {
		return nullptr;
	}
	return node;
}

bool AIFilterTreeIsConjunction(const AIFilterTreeNode &node) {
	if (node.type == AIFilterTreeType::LEAF) {
		return true;
	}
	if (node.type != AIFilterTreeType::AND_OP) {
		return false;
	}
	for (auto &child : node.children) {
		if (!AIFilterTreeIsConjunction(*child)) {
			return false;
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Evaluation (2-valued with UNKNOWN)
//===--------------------------------------------------------------------===//
AITriState AIFilterTreeEval(const AIFilterTreeNode &node, const vector<AITriState> &leaf_values) {
	switch (node.type) {
	case AIFilterTreeType::LEAF:
		return leaf_values[node.leaf_index];
	case AIFilterTreeType::NOT_OP: {
		auto v = AIFilterTreeEval(*node.children[0], leaf_values);
		if (v == AITriState::TRI_UNKNOWN || v == AITriState::TRI_NULL) {
			return v; // not yet known stays open; no answer stays NULL
		}
		return v == AITriState::TRI_TRUE ? AITriState::TRI_FALSE : AITriState::TRI_TRUE;
	}
	case AIFilterTreeType::AND_OP: {
		// A false decides; an unevaluated child keeps the fold open (it may still turn false); otherwise a
		// NULL child makes the conjunction NULL (SQL three-valued logic).
		bool any_unknown = false, any_null = false;
		for (auto &child : node.children) {
			auto v = AIFilterTreeEval(*child, leaf_values);
			if (v == AITriState::TRI_FALSE) {
				return AITriState::TRI_FALSE; // short-circuit
			}
			any_unknown = any_unknown || v == AITriState::TRI_UNKNOWN;
			any_null = any_null || v == AITriState::TRI_NULL;
		}
		return any_unknown ? AITriState::TRI_UNKNOWN : any_null ? AITriState::TRI_NULL : AITriState::TRI_TRUE;
	}
	case AIFilterTreeType::OR_OP: {
		bool any_unknown = false, any_null = false;
		for (auto &child : node.children) {
			auto v = AIFilterTreeEval(*child, leaf_values);
			if (v == AITriState::TRI_TRUE) {
				return AITriState::TRI_TRUE; // short-circuit
			}
			any_unknown = any_unknown || v == AITriState::TRI_UNKNOWN;
			any_null = any_null || v == AITriState::TRI_NULL;
		}
		return any_unknown ? AITriState::TRI_UNKNOWN : any_null ? AITriState::TRI_NULL : AITriState::TRI_FALSE;
	}
	}
	return AITriState::TRI_UNKNOWN;
}

//===--------------------------------------------------------------------===//
// Order planner
//
// Primary: the exact minimum-expected-cost DP for sequentially evaluating a boolean function under
// leaf independence. Fallback (very wide trees): a greedy per-node rank descent.
//===--------------------------------------------------------------------===//
static double Clamp01(double p) {
	const double eps = 1e-4;
	return p < eps ? eps : (p > 1.0 - eps ? 1.0 - eps : p);
}

//===--------------------------------------------------------------------===//
// Greedy fallback: effective (P(true), cost) per subtree, then a rank-based descent
//===--------------------------------------------------------------------===//
struct SubtreeStats {
	double p_true;
	double cost;
};

static SubtreeStats ComputeStats(const AIFilterTreeNode &node, const vector<double> &p, const vector<double> &cost) {
	switch (node.type) {
	case AIFilterTreeType::LEAF:
		return {Clamp01(p[node.leaf_index]), cost[node.leaf_index]};
	case AIFilterTreeType::NOT_OP: {
		auto s = ComputeStats(*node.children[0], p, cost);
		return {1.0 - s.p_true, s.cost};
	}
	case AIFilterTreeType::AND_OP: {
		vector<SubtreeStats> cs;
		for (auto &child : node.children) {
			cs.push_back(ComputeStats(*child, p, cost));
		}
		// optimal order for AND: ascending cost / P(false)
		std::sort(cs.begin(), cs.end(), [](const SubtreeStats &a, const SubtreeStats &b) {
			return a.cost * (1.0 - b.p_true) < b.cost * (1.0 - a.p_true);
		});
		double reach = 1.0, c = 0.0, prod = 1.0;
		for (auto &s : cs) {
			c += reach * s.cost;
			reach *= s.p_true;
			prod *= s.p_true;
		}
		return {prod, c};
	}
	case AIFilterTreeType::OR_OP: {
		vector<SubtreeStats> cs;
		for (auto &child : node.children) {
			cs.push_back(ComputeStats(*child, p, cost));
		}
		// optimal order for OR: ascending cost / P(true)
		std::sort(cs.begin(), cs.end(),
		          [](const SubtreeStats &a, const SubtreeStats &b) { return a.cost * b.p_true < b.cost * a.p_true; });
		double reach = 1.0, c = 0.0, prod_false = 1.0;
		for (auto &s : cs) {
			c += reach * s.cost;
			reach *= (1.0 - s.p_true);
			prod_false *= (1.0 - s.p_true);
		}
		return {1.0 - prod_false, c};
	}
	}
	return {0.5, 1.0};
}

static idx_t ChooseNextLeafGreedy(const AIFilterTreeNode &node, const vector<AITriState> &leaf_values,
                                  const vector<double> &p_true, const vector<double> &cost) {
	switch (node.type) {
	case AIFilterTreeType::LEAF:
		return node.leaf_index;
	case AIFilterTreeType::NOT_OP:
		return ChooseNextLeafGreedy(*node.children[0], leaf_values, p_true, cost);
	case AIFilterTreeType::AND_OP:
	case AIFilterTreeType::OR_OP: {
		const bool is_and = node.type == AIFilterTreeType::AND_OP;
		const AIFilterTreeNode *best = nullptr;
		double best_rank = 0.0;
		for (auto &child : node.children) {
			if (AIFilterTreeEval(*child, leaf_values) != AITriState::TRI_UNKNOWN) {
				continue; // already resolved -> not worth (re)visiting
			}
			auto s = ComputeStats(*child, p_true, cost);
			// AND: rank by cost / P(false); OR: rank by cost / P(true). Lower rank first.
			double rank = is_and ? s.cost / Clamp01(1.0 - s.p_true) : s.cost / Clamp01(s.p_true);
			if (!best || rank < best_rank) {
				best = child.get();
				best_rank = rank;
			}
		}
		if (!best) {
			best = node.children[0].get(); // defensive: should not happen when node is UNKNOWN
		}
		return ChooseNextLeafGreedy(*best, leaf_values, p_true, cost);
	}
	}
	return 0;
}

//===--------------------------------------------------------------------===//
// Exact minimum-expected-cost DP
//===--------------------------------------------------------------------===//
//! Exact DP is used up to this many leaves (3^12 == 531441 memo states); above it, greedy fallback.
static constexpr idx_t kExactMaxLeaves = 12;

static void CollectLeaves(const AIFilterTreeNode &node, vector<idx_t> &out) {
	if (node.type == AIFilterTreeType::LEAF) {
		out.push_back(node.leaf_index);
		return;
	}
	for (auto &child : node.children) {
		CollectLeaves(*child, out);
	}
}

//! Base-4 encoding of the partial assignment (TRI_FALSE=0, TRI_TRUE=1, TRI_UNKNOWN=2, TRI_NULL=3) -> memo key.
static uint64_t AssignKey(const vector<AITriState> &assign) {
	uint64_t key = 0;
	for (auto v : assign) {
		key = key * 4 + static_cast<uint64_t>(v);
	}
	return key;
}

//! Minimum expected cost to fully resolve `node` from partial assignment `assign` (mutated in place
//! then restored), under per-leaf selectivity `p` and cost `cost`. Memoized on the assignment.
static double MinExpectedCost(const AIFilterTreeNode &node, vector<AITriState> &assign, const vector<double> &p,
                              const vector<double> &cost, const vector<idx_t> &leaves,
                              std::unordered_map<uint64_t, double> &cache) {
	if (AIFilterTreeEval(node, assign) != AITriState::TRI_UNKNOWN) {
		return 0.0; // already resolved to a constant -> no further cost
	}
	const uint64_t key = AssignKey(assign);
	auto it = cache.find(key);
	if (it != cache.end()) {
		return it->second;
	}
	double best = std::numeric_limits<double>::infinity();
	for (idx_t q : leaves) {
		if (assign[q] != AITriState::TRI_UNKNOWN) {
			continue;
		}
		assign[q] = AITriState::TRI_TRUE;
		const double cost_pass = MinExpectedCost(node, assign, p, cost, leaves, cache);
		assign[q] = AITriState::TRI_FALSE;
		const double cost_fail = MinExpectedCost(node, assign, p, cost, leaves, cache);
		assign[q] = AITriState::TRI_UNKNOWN;
		const double pq = Clamp01(p[q]);
		const double expected = cost[q] + pq * cost_pass + (1.0 - pq) * cost_fail;
		if (expected < best) {
			best = expected;
		}
	}
	cache[key] = best;
	return best;
}

idx_t AIFilterTreeChooseNextLeaf(const AIFilterTreeNode &node, const vector<AITriState> &leaf_values,
                                 const vector<double> &p_true, const vector<double> &cost) {
	vector<idx_t> leaves;
	CollectLeaves(node, leaves);
	if (leaves.size() > kExactMaxLeaves) {
		return ChooseNextLeafGreedy(node, leaf_values, p_true, cost); // too wide for exact DP
	}
	// Pick the first leaf achieving the minimum total expected cost (exact DP, memoized per call
	// since p/cost are this row's). Ties resolve to the earliest leaf in tree order.
	vector<AITriState> assign = leaf_values;
	std::unordered_map<uint64_t, double> cache;
	idx_t best_leaf = 0;
	bool found = false;
	double best_cost = std::numeric_limits<double>::infinity();
	for (idx_t q : leaves) {
		if (assign[q] != AITriState::TRI_UNKNOWN) {
			continue;
		}
		assign[q] = AITriState::TRI_TRUE;
		const double cost_pass = MinExpectedCost(node, assign, p_true, cost, leaves, cache);
		assign[q] = AITriState::TRI_FALSE;
		const double cost_fail = MinExpectedCost(node, assign, p_true, cost, leaves, cache);
		assign[q] = AITriState::TRI_UNKNOWN;
		const double pq = Clamp01(p_true[q]);
		const double expected = cost[q] + pq * cost_pass + (1.0 - pq) * cost_fail;
		if (!found || expected < best_cost) {
			found = true;
			best_cost = expected;
			best_leaf = q;
		}
	}
	if (!found) {
		return leaves.empty() ? 0 : leaves[0]; // defensive: tree should have an UNKNOWN leaf
	}
	return best_leaf;
}

double AIFilterTreeEstimateSelectivity(const AIFilterTreeNode &node, const vector<double> &p_true) {
	// Reuse the independence roll-up; cost is irrelevant for a pure selectivity estimate.
	vector<double> cost(p_true.size(), 1.0);
	return ComputeStats(node, p_true, cost).p_true;
}

} // namespace duckdb
