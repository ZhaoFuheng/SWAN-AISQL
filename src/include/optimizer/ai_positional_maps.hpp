//===----------------------------------------------------------------------===//
// ai_positional_maps.hpp -- keep DuckDB's positional projection maps valid across a plan rewrite
//
// FILTER, ORDER_BY and every join carry projection maps that select their output columns by POSITION in
// their children's outputs (ColumnLifetimeAnalyzer builds them). A SWAN-AISQL pass that inserts an AI region
// (whose result column is appended to its child's columns) or narrows a join (a SEMI conversion) changes
// those positions under an ancestor, and a stale map then selects the wrong column: "INTERNAL Error: Failed to
// bind column reference". Binding identities survive such rewrites, only their sequence changes, so a pass
// snapshots every map with its child's binding order before it runs and remaps each map by identity after.
//===----------------------------------------------------------------------===//

#pragma once

#include <unordered_set>

#include "duckdb/planner/logical_operator.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_join.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

namespace duckdb {

class PositionalMapSnapshot {
public:
	//! `joins_only`: snapshot only join maps (for a pass that already fixes its filter/order maps itself; a
	//! second identity remap of an already-fixed map would corrupt it).
	explicit PositionalMapSnapshot(LogicalOperator &root, bool joins_only = false) : joins_only(joins_only) {
		Take(root);
	}

	//! Remap every snapshotted map of an operator still in `root`'s tree by binding identity. A map entry
	//! whose binding is gone keeps its position (a pass that removes a column must fix its own consumers).
	void Remap(LogicalOperator &root) const {
		std::unordered_set<const LogicalOperator *> live;
		Collect(root, live);
		for (auto &entry : entries) {
			if (!live.count(entry.op) || entry.op->children.size() <= entry.child) {
				continue;
			}
			auto *map = MapOf(*entry.op, entry.child);
			if (!map) {
				continue;
			}
			const auto now = entry.op->children[entry.child]->GetColumnBindings();
			for (auto &m : *map) {
				if (m.GetIndex() >= entry.before.size()) {
					continue;
				}
				const auto binding = entry.before[m.GetIndex()];
				for (idx_t j = 0; j < now.size(); j++) {
					if (now[j] == binding) {
						m = ProjectionIndex(j);
						break;
					}
				}
			}
		}
	}

private:
	struct Entry {
		LogicalOperator *op;
		idx_t child;
		vector<ColumnBinding> before;
	};
	vector<Entry> entries;
	bool joins_only;

	static bool IsJoin(const LogicalOperator &op) {
		switch (op.type) {
		case LogicalOperatorType::LOGICAL_COMPARISON_JOIN:
		case LogicalOperatorType::LOGICAL_ANY_JOIN:
		case LogicalOperatorType::LOGICAL_ASOF_JOIN:
		case LogicalOperatorType::LOGICAL_DELIM_JOIN:
			return true;
		default:
			return false;
		}
	}

	static vector<ProjectionIndex> *MapOf(LogicalOperator &op, idx_t child) {
		if (op.type == LogicalOperatorType::LOGICAL_FILTER && child == 0) {
			return &op.Cast<LogicalFilter>().projection_map;
		}
		if (op.type == LogicalOperatorType::LOGICAL_ORDER_BY && child == 0) {
			return &op.Cast<LogicalOrder>().projection_map;
		}
		if (IsJoin(op) && child < 2) {
			auto &join = op.Cast<LogicalJoin>();
			return child == 0 ? &join.left_projection_map : &join.right_projection_map;
		}
		return nullptr;
	}

	void Take(LogicalOperator &op) {
		for (idx_t c = 0; c < op.children.size(); c++) {
			auto *map = joins_only && !IsJoin(op) ? nullptr : MapOf(op, c);
			if (map && !map->empty()) {
				entries.push_back({&op, c, op.children[c]->GetColumnBindings()});
			}
		}
		for (auto &child : op.children) {
			Take(*child);
		}
	}

	static void Collect(const LogicalOperator &op, std::unordered_set<const LogicalOperator *> &live) {
		live.insert(&op);
		for (auto &child : op.children) {
			Collect(*child, live);
		}
	}
};

} // namespace duckdb
