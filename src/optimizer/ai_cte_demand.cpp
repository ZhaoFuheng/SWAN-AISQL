#include "optimizer/ai_cte_demand.hpp"

#include "optimizer/semi_join_reducer.hpp"
#include "ai_client.hpp"

#include <cstdio>

#include "duckdb/common/projection_index.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "duckdb/planner/logical_operator_deep_copy.hpp"
#include "duckdb/planner/operator/logical_comparison_join.hpp"
#include "duckdb/planner/operator/logical_cteref.hpp"
#include "duckdb/planner/operator/logical_filter.hpp"
#include "duckdb/planner/operator/logical_materialized_cte.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/operator/logical_set_operation.hpp"

#include <functional>

namespace duckdb {

static bool IsAIFunctionName(const Identifier &name) {
	return name == "ai_filter" || name == "ai_classify" || name == "ai_score" || name == "ai_complete" ||
	       name == "ai_function_with_embed";
}

static bool ExpressionHasAIFunction(const Expression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
	    IsAIFunctionName(expr.Cast<BoundFunctionExpression>().Function().GetName())) {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(expr, [&](const Expression &child) {
		if (!found && ExpressionHasAIFunction(child)) {
			found = true;
		}
	});
	return found;
}

//! The body's single AI-bearing filter, or null when there is none or more than one.
static optional_ptr<LogicalFilter> FindSoleAIFilter(LogicalOperator &op) {
	optional_ptr<LogicalFilter> found;
	bool ambiguous = false;
	std::function<void(LogicalOperator &)> walk = [&](LogicalOperator &node) {
		if (node.type == LogicalOperatorType::LOGICAL_FILTER) {
			for (auto &expr : node.expressions) {
				if (ExpressionHasAIFunction(*expr)) {
					if (found) {
						ambiguous = true;
					}
					found = &node.Cast<LogicalFilter>();
					break;
				}
			}
		}
		for (auto &child : node.children) {
			walk(*child);
		}
	};
	walk(op);
	return ambiguous ? optional_ptr<LogicalFilter>() : found;
}

//! What one reference can consume: the CTE column it joins on, plus the subplan supplying the keys.
struct ReferenceDemand {
	idx_t cte_column = 0;
	const Expression *neighbour_key = nullptr;
	unique_ptr<LogicalOperator> *neighbour = nullptr;
};

//! Every reference must contribute, and on the same column. A reference that consumes the CTE
//! unrestricted makes the union universal, and pruning by a universal demand is not pruning -- so
//! the whole reduction is abandoned rather than applied to a subset of the references, which would
//! drop rows the unrestricted one still needs.
static bool GatherDemands(unique_ptr<LogicalOperator> &main, TableIndex cte_index, vector<ReferenceDemand> &demands) {
	vector<TableIndex> reference_tables;
	std::function<void(LogicalOperator &)> find_refs = [&](LogicalOperator &node) {
		if (node.type == LogicalOperatorType::LOGICAL_CTE_REF && node.Cast<LogicalCTERef>().cte_index == cte_index) {
			reference_tables.push_back(node.Cast<LogicalCTERef>().table_index);
		}
		for (auto &child : node.children) {
			find_refs(*child);
		}
	};
	find_refs(*main);
	if (reference_tables.empty()) {
		return false;
	}
	// Walks nested CTE bodies too: a reference inside ANOTHER CTE's body counts, and collecting
	// only the main query's references loses the rows the nested one would have admitted.
	vector<bool> satisfied(reference_tables.size(), false);
	std::function<void(unique_ptr<LogicalOperator> &)> find_joins = [&](unique_ptr<LogicalOperator> &node) {
		if (node->type == LogicalOperatorType::LOGICAL_COMPARISON_JOIN) {
			auto &join = node->Cast<LogicalComparisonJoin>();
			for (auto &cond : join.conditions) {
				if (!cond.IsComparison() || cond.GetComparisonType() != ExpressionType::COMPARE_EQUAL) {
					continue;
				}
				const Expression *sides[2] = {&cond.GetLHS(), &cond.GetRHS()};
				for (idx_t s = 0; s < 2; s++) {
					if (sides[s]->GetExpressionClass() != ExpressionClass::BOUND_COLUMN_REF) {
						continue;
					}
					const auto binding = sides[s]->Cast<BoundColumnRefExpression>().Binding();
					for (idx_t r = 0; r < reference_tables.size(); r++) {
						if (binding.table_index != reference_tables[r] || satisfied[r]) {
							continue;
						}
						ReferenceDemand demand;
						demand.cte_column = binding.column_index.GetIndex();
						demand.neighbour_key = sides[1 - s];
						demand.neighbour = &join.children[s == 0 ? 1 : 0];
						demands.push_back(demand);
						satisfied[r] = true;
					}
				}
			}
		}
		for (auto &child : node->children) {
			find_joins(child);
		}
	};
	find_joins(main);
	for (idx_t r = 0; r < reference_tables.size(); r++) {
		if (!satisfied[r]) {
			return false;
		}
	}
	for (auto &demand : demands) {
		if (demand.cte_column != demands[0].cte_column) {
			return false; // no single key to reduce on
		}
	}
	return true;
}

static bool Decline(const char *reason);

//! Every CTE definition in the plan, by index, so a reference inside a copied key source can be
//! replaced with the thing it names.
static void CollectCTEBodies(LogicalOperator &op, unordered_map<idx_t, LogicalOperator *> &bodies) {
	if (op.type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE && op.children.size() == 2) {
		bodies[op.Cast<LogicalMaterializedCTE>().table_index.index] = op.children[0].get();
	}
	for (auto &child : op.children) {
		CollectCTEBodies(*child, bodies);
	}
}

//! A key source gets COPIED into the CTE body it prunes, where any CTE it references is NOT in
//! scope -- the reference would sit above its own definition ("Unable to find materialized CTE
//! stats"). So each reference is replaced by a copy of the CTE's BODY, wrapped in a projection
//! that re-exposes the columns AT THE REFERENCE'S OWN table index. Keeping the output bindings
//! identical to the reference's is what lets everything above it stay untouched.
static bool ResolveCTERefs(Binder &binder, unique_ptr<LogicalOperator> &op,
                           const unordered_map<idx_t, LogicalOperator *> &bodies, TableIndex reducing,
                           idx_t depth = 0) {
	if (depth > 8) {
		return Decline("resolve: nesting deeper than 8 CTE levels");
	}
	if (op->type == LogicalOperatorType::LOGICAL_CTE_REF) {
		auto &ref = op->Cast<LogicalCTERef>();
		// A key source that reads the CTE being reduced is NOT circular once stripped: what gets
		// substituted is that CTE's body with its AI predicate neutralised, i.e. its relational
		// scan, which references nothing recursive. Reducing A by (A_relational JOIN partsupp)
		// still prunes -- by partsupp -- and terminates in one substitution. This is Q26's
		// arrangement: ps_bridge reads a_candidates, and the main query joins a_candidates
		// against ps_bridge, so refusing it declined the whole reduction.
		auto entry = bodies.find(ref.cte_index.index);
		if (entry == bodies.end()) {
			return Decline("resolve: no definition found for the referenced CTE");
		}
		unique_ptr<LogicalOperator> body_copy;
		{
			LogicalOperatorDeepCopy deep_copy(binder, nullptr);
			unique_ptr<LogicalOperator> source = entry->second->Copy(binder.context);
			try {
				body_copy = deep_copy.DeepCopy(source);
			} catch (...) {
				return Decline("resolve: deep copy of the referenced CTE's body threw");
			}
		}
		if (!ResolveCTERefs(binder, body_copy, bodies, reducing, depth + 1)) {
			return false;
		}
		// the body may carry its own AI predicate; a relational key source must not evaluate it
		if (!AIStripAIPredicates(body_copy)) {
			return Decline("resolve: referenced CTE's body has an unstrippable AI call");
		}
		body_copy->ResolveOperatorTypes();
		auto body_bindings = body_copy->GetColumnBindings();
		if (body_bindings.size() < ref.chunk_types.size()) {
			return Decline("resolve: referenced CTE's body exposes fewer columns than the reference");
		}
		vector<unique_ptr<Expression>> exprs;
		for (idx_t i = 0; i < ref.chunk_types.size(); i++) {
			exprs.push_back(make_uniq<BoundColumnRefExpression>(body_copy->types[i], body_bindings[i]));
		}
		auto projection = make_uniq<LogicalProjection>(ref.table_index, std::move(exprs));
		projection->children.push_back(std::move(body_copy));
		projection->ResolveOperatorTypes();
		op = std::move(projection);
		return true;
	}
	for (auto &child : op->children) {
		if (!ResolveCTERefs(binder, child, bodies, reducing, depth)) {
			return false;
		}
	}
	return true;
}

//! UNION ALL of every reference's key source, each kept to its relational skeleton.
static unique_ptr<LogicalOperator> BuildDemandUnion(Binder &binder, const vector<ReferenceDemand> &demands,
                                                    const unordered_map<idx_t, LogicalOperator *> &bodies,
                                                    TableIndex reducing) {
	vector<unique_ptr<LogicalOperator>> sources;
	for (auto &demand : demands) {
		LogicalOperatorDeepCopy deep_copy(binder, nullptr);
		unique_ptr<LogicalOperator> copy;
		try {
			copy = deep_copy.DeepCopy(*demand.neighbour);
		} catch (...) {
			Decline("union: deep copy of the key source threw");
			return nullptr;
		}
		// resolve any CTE the key source reads into that CTE's own body, so the copy is valid
		// where it is about to be placed
		if (!ResolveCTERefs(binder, copy, bodies, reducing)) {
			Decline("union: a CTE inside the key source could not be resolved to its body");
			return nullptr;
		}
		// never duplicate an LLM predicate to serve a relational purpose
		if (!AIStripAIPredicates(copy)) {
			Decline("union: key source has an AI call outside a filter (join condition/projection)");
			return nullptr;
		}
		copy->ResolveOperatorTypes();
		auto original = (*demand.neighbour)->GetColumnBindings();
		auto copied = copy->GetColumnBindings();
		if (original.size() != copied.size()) {
			Decline("union: the copy's column count differs from the original");
			return nullptr;
		}
		auto key = demand.neighbour_key->Copy();
		bool ok = true;
		std::function<void(Expression &)> repoint = [&](Expression &expr) {
			if (expr.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
				auto &colref = expr.Cast<BoundColumnRefExpression>();
				for (idx_t b = 0; b < original.size(); b++) {
					if (original[b] == colref.Binding()) {
						colref.BindingMutable() = copied[b];
						return;
					}
				}
				ok = false;
				return;
			}
			ExpressionIterator::EnumerateChildren(expr, [&](Expression &child) { repoint(child); });
		};
		repoint(*key);
		if (!ok) {
			Decline("union: key expression references a column the copy does not expose");
			return nullptr;
		}
		vector<unique_ptr<Expression>> proj;
		proj.push_back(std::move(key));
		auto projection = make_uniq<LogicalProjection>(binder.GenerateTableIndex(), std::move(proj));
		projection->children.push_back(std::move(copy));
		projection->ResolveOperatorTypes();
		sources.push_back(std::move(projection));
	}
	if (sources.empty()) {
		return nullptr;
	}
	auto keys = std::move(sources[0]);
	for (idx_t i = 1; i < sources.size(); i++) {
		auto setop =
		    make_uniq<LogicalSetOperation>(binder.GenerateTableIndex(), 1, std::move(keys), std::move(sources[i]),
		                                   LogicalOperatorType::LOGICAL_UNION, true, false);
		setop->ResolveOperatorTypes();
		keys = std::move(setop);
	}
	return keys;
}

//! Is every column `expr` reads available in `bindings`?
static bool KeyAvailable(const Expression &expr, const vector<ColumnBinding> &bindings) {
	bool all = true;
	std::function<void(const Expression &)> walk = [&](const Expression &e) {
		if (e.GetExpressionClass() == ExpressionClass::BOUND_COLUMN_REF) {
			if (std::find(bindings.begin(), bindings.end(), e.Cast<BoundColumnRefExpression>().Binding()) ==
			    bindings.end()) {
				all = false;
			}
			return;
		}
		ExpressionIterator::EnumerateChildren(e, [&](const Expression &c) { walk(c); });
	};
	walk(expr);
	return all;
}

//! Why a CTE was not reduced. Visible with ai_debug_log containing 'demand'; the pass declines
//! for many legitimate reasons and the count alone never says which.
static bool Decline(const char *reason) {
	if (AIConfig::Get().debug_log.find("demand") != string::npos) {
		fprintf(stderr, "[demand] declined: %s\n", reason);
	}
	return false;
}

static bool ReduceOne(Binder &binder, LogicalMaterializedCTE &cte,
                      const unordered_map<idx_t, LogicalOperator *> &bodies) {
	if (cte.children.size() != 2) {
		return false;
	}
	auto ai_filter = FindSoleAIFilter(*cte.children[0]);
	if (!ai_filter || ai_filter->children.size() != 1) {
		return Decline("no single AI-bearing filter in the body");
	}
	if (cte.children[0]->type != LogicalOperatorType::LOGICAL_PROJECTION) {
		return Decline("body root is not a projection");
	}
	vector<ReferenceDemand> demands;
	if (!GatherDemands(cte.children[1], cte.table_index, demands)) {
		return Decline("a reference consumes the CTE unrestricted, or they disagree on the key column");
	}
	auto &root = cte.children[0]->Cast<LogicalProjection>();
	if (demands[0].cte_column >= root.expressions.size()) {
		return Decline("demanded column is past the body's output list");
	}
	// The body-side key is the EXPRESSION that produces the demanded output column, copied -- not a
	// column reference to it. A CTE's join key is routinely computed ('k_'||id, CAST(SPLIT_PART(..))),
	// and requiring a plain column reference here skipped exactly the queries that needed reducing.
	auto body_key = root.expressions[demands[0].cte_column]->Copy();
	if (ExpressionHasAIFunction(*body_key)) {
		return Decline("the demanded column is produced by an AI call");
	}
	auto &target = ai_filter->children[0];
	target->ResolveOperatorTypes();
	if (!KeyAvailable(*body_key, target->GetColumnBindings())) {
		return Decline("key columns are not visible below the predicate");
	}
	auto keys = BuildDemandUnion(binder, demands, bodies, cte.table_index);
	if (!keys) {
		return Decline("could not build the key union (copy, CTE resolution or AI stripping failed)");
	}
	auto key_bindings = keys->GetColumnBindings();
	if (key_bindings.size() != 1) {
		return false;
	}
	// A SEMI join emits only its left side's columns, so everything above keeps its bindings.
	auto reducer = make_uniq<LogicalComparisonJoin>(JoinType::SEMI);
	reducer->conditions.push_back(JoinCondition(std::move(body_key),
	                                            make_uniq<BoundColumnRefExpression>(keys->types[0], key_bindings[0]),
	                                            ExpressionType::COMPARE_EQUAL));
	reducer->children.push_back(std::move(target));
	reducer->children.push_back(std::move(keys));
	reducer->ResolveOperatorTypes();
	target = std::move(reducer);
	return true;
}

static bool ReduceTree(Binder &binder, unique_ptr<LogicalOperator> &plan,
                       const unordered_map<idx_t, LogicalOperator *> &bodies) {
	bool changed = false;
	if (plan->type == LogicalOperatorType::LOGICAL_MATERIALIZED_CTE) {
		if (ReduceOne(binder, plan->Cast<LogicalMaterializedCTE>(), bodies)) {
			changed = true;
		}
	}
	for (auto &child : plan->children) {
		if (ReduceTree(binder, child, bodies)) {
			changed = true;
		}
	}
	return changed;
}

bool AIReduceCTEsByDemand(Binder &binder, unique_ptr<LogicalOperator> &plan) {
	unordered_map<idx_t, LogicalOperator *> bodies;
	CollectCTEBodies(*plan, bodies);
	return ReduceTree(binder, plan, bodies);
}

} // namespace duckdb
