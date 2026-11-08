#include "exec/physical_ai_factor_graph.hpp"

#include "ai_client.hpp"
#include "filter_tree_order.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "ai_prompt_cost.hpp"
#include "ai_settings.hpp"

#include "duckdb/common/operator/add.hpp"
#include "duckdb/common/operator/multiply.hpp"
#include "duckdb/common/value_operations/value_operations.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <chrono>
#include <deque>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <set>

namespace duckdb {

PhysicalAIFactorGraph::PhysicalAIFactorGraph(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                                             unique_ptr<Expression> node_p, vector<idx_t> side_widths_p,
                                             vector<AIExactEdge> exact_edges_p, int64_t limit_p,
                                             idx_t existential_side_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types_p), estimated_cardinality),
      node(std::move(node_p)), side_widths(std::move(side_widths_p)), exact_edges(std::move(exact_edges_p)),
      limit(limit_p), existential_side(existential_side_p) {
	side_offsets.resize(side_widths.size());
	idx_t offset = 0;
	vector<idx_t> side_of;
	for (idx_t s = 0; s < side_widths.size(); s++) {
		side_offsets[s] = offset;
		side_of.insert(side_of.end(), side_widths[s], s);
		offset += side_widths[s];
	}
	auto &fn = node->Cast<BoundFunctionExpression>();
	vector<vector<AIFactor>> term_factors;
	if (!AIFactorDecompose(fn, side_of, /*max_terms=*/1u << 16, term_factors)) {
		throw InternalException("AIFactorGraph: node is not factor-decomposable (rewrite gate should have caught it)");
	}
	// Per term, merge the factors of one side (or one pair of sides) into that side's (edge's) leaf list and
	// tree: the AND of the factors' trees, each renumbered to its leaves' positions in the merged list.
	for (auto &factors : term_factors) {
		TermPlan plan;
		plan.factors = factors;
		plan.unary_leaves.resize(side_widths.size());
		plan.unary_trees.resize(side_widths.size());
		plan.unary_factors.resize(side_widths.size());
		vector<vector<string>> unary_parts(side_widths.size());
		vector<vector<string>> edge_parts;
		auto merge = [&](vector<idx_t> &leaf_ids, vector<string> &parts, const AIFactor &factor) {
			vector<idx_t> leaf_map(factor.leaf_ids.size());
			for (idx_t i = 0; i < factor.leaf_ids.size(); i++) {
				leaf_map[i] = leaf_ids.size();
				leaf_ids.push_back(factor.leaf_ids[i]);
			}
			auto tree = AIFilterTreeParse(factor.tree);
			if (!tree) {
				throw InternalException("AIFactorGraph: unparseable factor tree '%s'", factor.tree);
			}
			parts.push_back(AIFilterTreeSerialize(*AIFilterTreeRenumber(*tree, leaf_map)));
		};
		for (idx_t f = 0; f < plan.factors.size(); f++) {
			auto &factor = plan.factors[f];
			if (factor.sides.size() == 1) {
				merge(plan.unary_leaves[factor.sides[0]], unary_parts[factor.sides[0]], factor);
				plan.unary_factors[factor.sides[0]].push_back(f);
				continue;
			}
			auto it = std::find_if(plan.edges.begin(), plan.edges.end(),
			                       [&](const Edge &e) { return e.s == factor.sides[0] && e.t == factor.sides[1]; });
			if (it == plan.edges.end()) {
				plan.edges.push_back(Edge {factor.sides[0], factor.sides[1], {}, ""});
				edge_parts.emplace_back();
				it = plan.edges.end() - 1;
			}
			merge(it->leaf_ids, edge_parts[NumericCast<idx_t>(it - plan.edges.begin())], factor);
		}
		auto conjoin = [](const vector<string> &parts) {
			if (parts.size() == 1) {
				return parts[0];
			}
			string tree = "A(";
			for (idx_t i = 0; i < parts.size(); i++) {
				tree += (i ? "," : "") + parts[i];
			}
			return tree + ")";
		};
		for (idx_t s = 0; s < side_widths.size(); s++) {
			if (!unary_parts[s].empty()) {
				plan.unary_trees[s] = conjoin(unary_parts[s]);
			}
		}
		for (idx_t e = 0; e < plan.edges.size(); e++) {
			plan.edges[e].tree = conjoin(edge_parts[e]);
		}
		// The absorbed joins hold in every term: an equality between two sides keys the semantic edge already
		// connecting them, or stands alone as an exact edge (no leaves: its bucket pairs are true without a call).
		for (auto &exact : exact_edges) {
			auto it = std::find_if(plan.edges.begin(), plan.edges.end(),
			                       [&](const Edge &e) { return e.s == exact.s && e.t == exact.t; });
			if (it == plan.edges.end()) {
				plan.edges.push_back(Edge {exact.s, exact.t, {}, "", exact.s_cols, exact.t_cols});
			} else {
				it->key_s = exact.s_cols;
				it->key_t = exact.t_cols;
			}
		}
		terms.push_back(std::move(plan));
	}
}

InsertionOrderPreservingMap<string> PhysicalAIFactorGraph::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Sides"] = std::to_string(side_widths.size());
	// per term: its edges, and each factor's tree in the node's leaf numbering (its own numbering is positional)
	string edge_str, factor_str;
	for (auto &term : terms) {
		string edges_of_term, factors_of_term;
		for (auto &e : term.edges) {
			edges_of_term +=
			    (edges_of_term.empty() ? "" : ", ") + ("S" + std::to_string(e.s) + "-S" + std::to_string(e.t));
		}
		for (auto &f : term.factors) {
			auto tree = AIFilterTreeParse(f.tree);
			factors_of_term += (factors_of_term.empty() ? "" : ", ") +
			                   (tree ? AIFilterTreeSerialize(*AIFilterTreeRenumber(*tree, f.leaf_ids)) : f.tree);
		}
		edge_str += (edge_str.empty() ? "" : " | ") + edges_of_term;
		factor_str += (factor_str.empty() ? "" : " | ") + factors_of_term;
	}
	if (terms.size() > 1) {
		result["Terms"] = std::to_string(terms.size());
	}
	result["Edges"] = edge_str;
	result["Factors"] = factor_str;
	if (!exact_edges.empty()) {
		string keys;
		for (auto &exact : exact_edges) {
			for (idx_t c = 0; c < exact.s_cols.size(); c++) {
				keys += (keys.empty() ? "" : ", ") +
				        ("S" + std::to_string(exact.s) + "#" + std::to_string(exact.s_cols[c]) + " = S" +
				         std::to_string(exact.t) + "#" + std::to_string(exact.t_cols[c]));
			}
		}
		result["Join Keys"] = keys;
	}
	result["Mode"] = exact_edges.empty()
	                     ? "factor graph (member/pair domains, exact backward pruning)"
	                     : "factor graph over an equi-join (pair domains confined to equal-key buckets)";
	return result;
}

//===--------------------------------------------------------------------===//
// Sink: one dictionary of DISTINCT rows + multiplicities per side
//===--------------------------------------------------------------------===//
class FactorGraphSinkState : public GlobalSinkState {
public:
	explicit FactorGraphSinkState(idx_t k) : sides(k) {
	}

	struct Side {
		vector<vector<Value>> reps;
		vector<idx_t> counts;
		std::unordered_map<string, idx_t> key_to_rep;
	};

	mutex lock;
	vector<Side> sides;
};

class FactorGraphLocalSinkState : public LocalSinkState {
public:
	explicit FactorGraphLocalSinkState(idx_t side) : side(side) {
	}
	idx_t side;
};

unique_ptr<GlobalSinkState> PhysicalAIFactorGraph::GetGlobalSinkState(ClientContext &) const {
	return make_uniq<FactorGraphSinkState>(children.size());
}

unique_ptr<LocalSinkState> PhysicalAIFactorGraph::GetLocalSinkState(ExecutionContext &context) const {
	if (!context.pipeline || !context.pipeline->GetSource()) {
		throw InternalException("AIFactorGraph: build pipeline without a source");
	}
	auto it = source_to_side.find(context.pipeline->GetSource().get());
	if (it == source_to_side.end()) {
		throw InternalException("AIFactorGraph: build pipeline source does not belong to any side");
	}
	return make_uniq<FactorGraphLocalSinkState>(it->second);
}

// A tuple's multiplicity is the product of its members' fan-outs, and a term's confirmed rows the sum of those:
// both can pass 2^63 (eight aliases of 256 identical rows: 256^8), where wrapping would count nothing. Saturate.
static idx_t SaturatingMul(idx_t a, idx_t b) {
	idx_t result;
	return TryMultiplyOperator::Operation<idx_t, idx_t, idx_t>(a, b, result) ? result : NumericLimits<idx_t>::Maximum();
}

static int64_t SaturatingAdd(int64_t a, int64_t b) {
	int64_t result;
	return TryAddOperator::Operation<int64_t, int64_t, int64_t>(a, b, result) ? result
	                                                                          : NumericLimits<int64_t>::Maximum();
}

//! The multiplicity of the tuple `assignment` (one representative per side), as a signed count for LIMIT arithmetic.
static int64_t TupleMultiplicity(const FactorGraphSinkState &sink, const vector<idx_t> &assignment) {
	idx_t product = 1;
	for (idx_t s = 0; s < assignment.size(); s++) {
		product = SaturatingMul(product, sink.sides[s].counts[assignment[s]]);
	}
	return product > static_cast<idx_t>(NumericLimits<int64_t>::Maximum()) ? NumericLimits<int64_t>::Maximum()
	                                                                       : static_cast<int64_t>(product);
}

static string FactorRowKey(DataChunk &chunk, idx_t row) {
	string key; // every column, NULL-flagged and length-prefixed (AIAppendKeyPart): no value makes two members collide
	for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
		AIAppendKeyPart(key, chunk.data[col].GetValue(row));
	}
	return key;
}

SinkResultType PhysicalAIFactorGraph::Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<FactorGraphSinkState>();
	auto &lstate = input.local_state.Cast<FactorGraphLocalSinkState>();
	lock_guard<mutex> guard(gstate.lock);
	auto &side = gstate.sides[lstate.side];
	for (idx_t row = 0; row < chunk.size(); row++) {
		auto key = FactorRowKey(chunk, row);
		auto it = side.key_to_rep.find(key);
		if (it != side.key_to_rep.end()) {
			side.counts[it->second]++;
			continue;
		}
		vector<Value> rep;
		rep.reserve(chunk.ColumnCount());
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			rep.push_back(chunk.data[col].GetValue(row));
		}
		side.key_to_rep.emplace(std::move(key), side.reps.size());
		side.reps.push_back(std::move(rep));
		side.counts.push_back(1);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalAIFactorGraph::Combine(ExecutionContext &, OperatorSinkCombineInput &) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalAIFactorGraph::Finalize(Pipeline &, Event &, ClientContext &,
                                                 OperatorSinkFinalizeInput &) const {
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source: factorized evaluation (unary pre-pass, edges with cascade), then survivor enumeration
//===--------------------------------------------------------------------===//
class FactorGraphSourceState : public GlobalSourceState {
public:
	bool evaluated = false;
	//! Surviving tuples: one rep id per side.
	vector<vector<idx_t>> assignments;
	//! Pagination: assignment index + duplicate index within its count product.
	idx_t cur_assignment = 0;
	idx_t cur_dup = 0;
};

unique_ptr<GlobalSourceState> PhysicalAIFactorGraph::GetGlobalSourceState(ClientContext &) const {
	return make_uniq<FactorGraphSourceState>();
}

//! Evaluate `sub_node` over `rows` of side reps laid out by `fill` into `chunk_types` chunks;
//! appends one 0/1 result per row to `out_pass` (invalid/NULL counts as false — filter semantics).
template <class FILL>
static void EvaluateDomain(ClientContext &context, const Expression &sub_node, const vector<LogicalType> &chunk_types,
                           idx_t row_count, const string &query_text, FILL fill, vector<char> &out_pass) {
	auto &fn = sub_node.Cast<BoundFunctionExpression>();
	DataChunk chunk;
	chunk.Initialize(BufferAllocator::Get(context), chunk_types);
	idx_t done = 0;
	while (done < row_count) {
		const idx_t batch = MinValue<idx_t>(STANDARD_VECTOR_SIZE, row_count - done);
		chunk.Reset();
		chunk.SetChildCardinality(batch);
		for (idx_t j = 0; j < batch; j++) {
			fill(chunk, j, done + j);
		}
		vector<char> result, valid;
		AIDedupFireWaveFactorized(context, fn, chunk, query_text, /*remaining_limit=*/-1, result, valid);
		for (idx_t j = 0; j < batch; j++) {
			out_pass.push_back(j < result.size() && (j >= valid.size() || valid[j]) && result[j] ? 1 : 0);
		}
		done += batch;
	}
}

//===--------------------------------------------------------------------===//
// Lazy (need-driven streaming) evaluation
//
// The graph is the single source of truth: results write back to MEMBERS and PAIRS, deletions
// prune the dispatch frontier, and tuples are never materialized (they are enumerated from the
// surviving graph at the end, call-free). No stage barriers: a unary prompt is dispatchable
// while its member is live; a pair the moment both endpoints are unary-confirmed and its
// parent-side member carries a confirmed-true upstream pair (strict need -- every dispatched
// pair lies on a path of confirmed predicates, matching the staged schedule's call envelope
// while edges stream). Frontier micro-batches are sized to the LLM concurrency, so the pool
// stays full across what used to be stage boundaries.
//===--------------------------------------------------------------------===//

//! Orient the edge graph as a forest (BFS per component). parent_edge_of_side[s] = the edge
//! connecting s to its parent (INVALID for roots); side_level[s] = BFS depth. False on a cycle.
static bool OrientForest(const PhysicalAIFactorGraph &op, const PhysicalAIFactorGraph::TermPlan &plan,
                         const FactorGraphSinkState &sink, vector<idx_t> &parent_edge_of_side,
                         vector<idx_t> &side_level) {
	const idx_t k = op.side_widths.size();
	parent_edge_of_side.assign(k, DConstants::INVALID_INDEX);
	side_level.assign(k, 0);
	vector<vector<idx_t>> adj(k);
	for (idx_t e = 0; e < plan.edges.size(); e++) {
		adj[plan.edges[e].s].push_back(e);
		adj[plan.edges[e].t].push_back(e);
	}
	// Root each component where the pruning starts: a unary-pruned side (most unary leaves,
	// then fewest members), so strict-need support flows outward from the smallest confirmed
	// domains -- the streaming analogue of the staged schedule's smallest-domain-first order.
	// Side indices carry NO join-order meaning (the join optimizer reorders the cross chain).
	vector<idx_t> root_order(k);
	for (idx_t s = 0; s < k; s++) {
		root_order[s] = s;
	}
	std::sort(root_order.begin(), root_order.end(), [&](idx_t x, idx_t y) {
		// an existentially-consumed side is ALWAYS the root: satisfaction gates flow outward from it
		if ((x == op.existential_side) != (y == op.existential_side)) {
			return x == op.existential_side;
		}
		if (plan.unary_leaves[x].size() != plan.unary_leaves[y].size()) {
			return plan.unary_leaves[x].size() > plan.unary_leaves[y].size();
		}
		return sink.sides[x].reps.size() < sink.sides[y].reps.size();
	});
	vector<char> visited(k, 0);
	vector<char> edge_used(plan.edges.size(), 0);
	for (const auto root : root_order) {
		if (visited[root]) {
			continue;
		}
		visited[root] = 1;
		vector<idx_t> queue {root};
		while (!queue.empty()) {
			const idx_t s = queue.back();
			queue.pop_back();
			for (const auto e : adj[s]) {
				if (edge_used[e]) {
					continue;
				}
				edge_used[e] = 1;
				const idx_t other = plan.edges[e].s == s ? plan.edges[e].t : plan.edges[e].s;
				if (visited[other]) {
					return false; // cycle: fall back to staged evaluation
				}
				visited[other] = 1;
				parent_edge_of_side[other] = e;
				side_level[other] = side_level[s] + 1;
				queue.push_back(other);
			}
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Streaming unit pool: C persistent workers for one graph evaluation. The driver submits units
// (a member or pair with its baked leaf prompts) and applies completions as they land -- no
// per-round thread churn and no drain barriers; a straggler delays only its own slot. The
// process-global chat gate in the client independently caps total in-flight HTTP at
// ai_concurrency, so this pool can never stack past the cap even alongside other operators.
//===--------------------------------------------------------------------===//
struct GraphStreamPool {
	struct Task {
		const Expression *sub_node;
		vector<string> prompts;
		vector<string> prefixes; // per-leaf provider-cache prefixes; empty = plain prompts
		vector<char> nulls;      // per-leaf: the prompt is NULL (decided by its wrapper, no call)
		string key;
		bool is_unary;
		idx_t domain;
		idx_t a;
		idx_t b;
		idx_t expected_reuse = 0; // fan-out hint for the client's prefix-cache write policy
	};
	struct Done {
		string key;
		bool is_unary;
		idx_t domain;
		idx_t a;
		idx_t b;
		bool value;
	};

	std::mutex m;
	std::condition_variable task_cv;
	std::condition_variable done_cv;
	std::deque<Task> tasks;
	vector<Done> done;
	idx_t unapplied = 0; // submitted and not yet applied by the driver
	bool stop = false;
	vector<std::thread> threads;
	string query_text;
	optional_ptr<ClientContext> context;
	//! The first exception a unit raised (a wrapper that cannot read an answer, a failed prompt expression);
	//! rethrown to the driver by Drain, so the query fails exactly as the scalar evaluation would.
	ErrorData error;

	~GraphStreamPool() {
		Shutdown(); // an exception unwinding the driver must still join the workers
	}

	void Start(ClientContext &ctx, idx_t n, string qt) {
		context = &ctx;
		query_text = std::move(qt);
		for (idx_t w = 0; w < n; w++) {
			threads.emplace_back([this]() {
				for (;;) {
					Task task;
					{
						std::unique_lock<std::mutex> lock(m);
						task_cv.wait(lock, [&]() { return stop || !tasks.empty(); });
						if (stop && tasks.empty()) {
							return;
						}
						task = std::move(tasks.front());
						tasks.pop_front();
					}
					bool valid = false;
					bool value = false;
					ErrorData failure;
					try {
						value = AIFactorEvalUnit(*context, task.sub_node->Cast<BoundFunctionExpression>(), task.prompts,
						                         query_text, valid, task.prefixes.empty() ? nullptr : &task.prefixes,
						                         static_cast<double>(task.expected_reuse), &task.nulls) &&
						        valid;
					} catch (std::exception &ex) {
						failure = ErrorData(ex);
					}
					{
						std::lock_guard<std::mutex> lock(m);
						if (failure.HasError() && !error.HasError()) {
							error = std::move(failure);
						}
						done.push_back(Done {std::move(task.key), task.is_unary, task.domain, task.a, task.b, value});
					}
					done_cv.notify_all();
				}
			});
		}
	}
	void Submit(Task task) {
		{
			std::lock_guard<std::mutex> lock(m);
			tasks.push_back(std::move(task));
			unapplied++;
		}
		task_cv.notify_one();
	}
	//! Harvest completions; blocking=true waits until at least one lands (or nothing is pending).
	//! Rethrows the first error a unit raised.
	void Drain(vector<Done> &out, bool blocking) {
		ErrorData raised;
		{
			std::unique_lock<std::mutex> lock(m);
			if (blocking) {
				done_cv.wait(lock, [&]() { return !done.empty() || unapplied == 0 || error.HasError(); });
			}
			if (error.HasError()) {
				raised = error;
			}
			out.insert(out.end(), std::make_move_iterator(done.begin()), std::make_move_iterator(done.end()));
			done.clear();
		}
		if (raised.HasError()) {
			raised.Throw();
		}
	}
	void MarkApplied(idx_t n) {
		std::lock_guard<std::mutex> lock(m);
		unapplied -= MinValue<idx_t>(n, unapplied);
	}
	idx_t Pending() {
		std::lock_guard<std::mutex> lock(m);
		return unapplied;
	}
	void Shutdown() {
		{
			std::lock_guard<std::mutex> lock(m);
			stop = true;
			tasks.clear();
		}
		task_cv.notify_all();
		for (auto &thread : threads) {
			if (thread.joinable()) {
				thread.join();
			}
		}
	}
};

//===--------------------------------------------------------------------===//
// Unit prompts: per-domain sub-nodes and executors that bake one member's or one pair's leaf
// prompts (vectorized expression evaluation over a one-row chunk), plus the prefix-cache split
// for pair prompts. Shared by the dense and the sparse scheduler below.
//===--------------------------------------------------------------------===//
struct GraphUnitBaker {
	ClientContext &context;
	const PhysicalAIFactorGraph &op;
	const PhysicalAIFactorGraph::TermPlan &plan;
	FactorGraphSinkState &sink;
	vector<unique_ptr<Expression>> side_sub, edge_sub;
	vector<string> side_sig, edge_sig; // tree + leaf meta: what a verdict depends on besides the prompts
	vector<vector<LogicalType>> side_types, edge_types;
	vector<unique_ptr<ExpressionExecutor>> side_exec, edge_exec, edge_ops_exec;
	// A side's unary factors as units of their own (plan.unary_factors[s][f]): the sparse scheduler asks a member
	// one conjunct at a time; the dense scheduler asks the whole side (side_sub) in one unit.
	vector<vector<unique_ptr<Expression>>> factor_sub;
	vector<vector<string>> factor_sig;
	vector<vector<unique_ptr<ExpressionExecutor>>> factor_exec;
	vector<unique_ptr<DataChunk>> side_row, edge_row;
	// Explicit provider prompt caching for pair prompts (ai_prefix_cache): split each single-leaf
	// edge's call prompt at the first operand that reads the other side. The graph only DECLARES
	// structure: every eligible pair call carries (prefix, suffix) plus an expected_reuse hint
	// (the member's remaining live pairs). Size gate, write policy, and prime/park sequencing all
	// live in the client's prefix lifecycle (ai_client.cpp), shared by every operator.
	vector<char> cache_enabled;
	vector<vector<const Expression *>> cache_ops;
	vector<idx_t> cache_split;
	vector<char> cache_group_s; // group axis: 1 = edge.s owns the prefix, 0 = edge.t

	GraphUnitBaker(ClientContext &context_p, const PhysicalAIFactorGraph &op_p,
	               const PhysicalAIFactorGraph::TermPlan &plan_p, FactorGraphSinkState &sink_p)
	    : context(context_p), op(op_p), plan(plan_p), sink(sink_p) {
		const idx_t k1 = op.side_widths.size();
		const idx_t ne = plan.edges.size();
		const idx_t total_width = op.side_offsets.back() + op.side_widths.back();
		auto &fn = op.node->Cast<BoundFunctionExpression>();
		side_sub.resize(k1);
		side_sig.resize(k1);
		side_types.resize(k1);
		side_exec.resize(k1);
		side_row.resize(k1);
		factor_sub.resize(k1);
		factor_sig.resize(k1);
		factor_exec.resize(k1);
		for (idx_t s = 0; s < k1; s++) {
			if (plan.unary_leaves[s].empty()) {
				continue;
			}
			vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
			for (idx_t c = 0; c < op.side_widths[s]; c++) {
				index_map[op.side_offsets[s] + c] = c;
			}
			side_sub[s] = AIFactorSubNode(fn, plan.unary_leaves[s], index_map, plan.unary_trees[s]);
			side_sig[s] = AIFactorSubNodeSignature(*side_sub[s]);
			for (const auto f : plan.unary_factors[s]) {
				auto &factor = plan.factors[f];
				factor_sub[s].push_back(AIFactorSubNode(fn, factor.leaf_ids, index_map, factor.tree));
				factor_sig[s].push_back(AIFactorSubNodeSignature(*factor_sub[s].back()));
			}
			factor_exec[s].resize(factor_sub[s].size());
			side_types[s].assign(op.types.begin() + NumericCast<int64_t>(op.side_offsets[s]),
			                     op.types.begin() + NumericCast<int64_t>(op.side_offsets[s] + op.side_widths[s]));
			side_types[s].push_back(LogicalType::BIGINT);
		}
		edge_sub.resize(ne);
		edge_sig.resize(ne);
		edge_types.resize(ne);
		edge_exec.resize(ne);
		edge_ops_exec.resize(ne);
		edge_row.resize(ne);
		for (idx_t e = 0; e < ne; e++) {
			auto &edge = plan.edges[e];
			if (!edge.Semantic()) {
				continue; // an exact edge asks nothing: no sub-node, no unit is ever baked for it
			}
			vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
			for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
				index_map[op.side_offsets[edge.s] + c] = c;
			}
			for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
				index_map[op.side_offsets[edge.t] + c] = op.side_widths[edge.s] + c;
			}
			edge_sub[e] = AIFactorSubNode(fn, edge.leaf_ids, index_map, edge.tree);
			edge_sig[e] = AIFactorSubNodeSignature(*edge_sub[e]);
			edge_types[e].assign(op.types.begin() + NumericCast<int64_t>(op.side_offsets[edge.s]),
			                     op.types.begin() +
			                         NumericCast<int64_t>(op.side_offsets[edge.s] + op.side_widths[edge.s]));
			for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
				edge_types[e].push_back(op.types[op.side_offsets[edge.t] + c]);
			}
			edge_types[e].push_back(LogicalType::BIGINT);
		}
		cache_enabled.assign(ne, 0);
		cache_ops.resize(ne);
		cache_split.assign(ne, 0);
		cache_group_s.assign(ne, 1);
		if (!AIConfig::Get().prefix_cache) {
			return;
		}
		for (idx_t e = 0; e < ne; e++) {
			if (!edge_sub[e]) {
				continue; // exact edge: no prompt to split
			}
			auto &fn_sub = edge_sub[e]->Cast<BoundFunctionExpression>();
			if ((fn_sub.GetChildren().size() - 1) / 3 != 1) {
				continue; // v1: single-leaf edges only
			}
			vector<const Expression *> ops;
			// look past the all-columns-NULL guard: the prefix lives in the concatenation under it
			AIFactorPromptOperands(AIUnwrapNullGuard(*fn_sub.GetChildren()[1]), ops);
			auto op_side = [&](const Expression &expr) {
				vector<idx_t> refs;
				std::function<void(const Expression &)> walk = [&](const Expression &node) {
					if (node.GetExpressionClass() == ExpressionClass::BOUND_REF) {
						refs.push_back(node.Cast<BoundReferenceExpression>().Index());
						return;
					}
					ExpressionIterator::EnumerateChildren(node, [&](const Expression &child) { walk(child); });
				};
				walk(expr);
				bool s_axis = false, t_axis = false;
				for (const auto col : refs) {
					(col < op.side_widths[plan.edges[e].s] ? s_axis : t_axis) = true;
				}
				return s_axis && t_axis ? 3 : (t_axis ? 2 : (s_axis ? 1 : 0));
			};
			idx_t first_ref = ops.size();
			for (idx_t o = 0; o < ops.size(); o++) {
				const auto side = op_side(*ops[o]);
				if (side == 1 || side == 2) {
					first_ref = o;
					break;
				}
				if (side == 3) {
					break; // an operand reads both sides before any single-side operand: no clean split
				}
			}
			if (first_ref == ops.size()) {
				continue;
			}
			const auto group_side = op_side(*ops[first_ref]);
			idx_t split = ops.size();
			for (idx_t o = first_ref + 1; o < ops.size(); o++) {
				const auto side = op_side(*ops[o]);
				if (side == 3 || (side != 0 && side != group_side)) {
					split = o;
					break;
				}
			}
			if (split == ops.size()) {
				continue; // never touches the other side: not a pair prompt
			}
			cache_enabled[e] = 1;
			cache_ops[e] = std::move(ops);
			cache_split[e] = split;
			cache_group_s[e] = static_cast<char>(group_side == 1);
		}
	}

	//! The member that owns a pair's cached prefix, and its side.
	idx_t CacheGroupMember(idx_t e, idx_t a, idx_t b) const {
		return cache_group_s[e] ? a : b;
	}
	idx_t CacheGroupSide(idx_t e) const {
		return cache_group_s[e] ? plan.edges[e].s : plan.edges[e].t;
	}
	//! The unit's sub-node: an edge's, a side's whole unary tree, or (factor < the side's unary factor count) one
	//! of the side's unary factors.
	const Expression *SubNode(bool is_unary, idx_t domain, idx_t factor) const {
		if (!is_unary) {
			return edge_sub[domain].get();
		}
		return factor < factor_sub[domain].size() ? factor_sub[domain][factor].get() : side_sub[domain].get();
	}

	void FillPairRow(DataChunk &row, idx_t e, idx_t a, idx_t b) {
		auto &edge = plan.edges[e];
		row.Reset();
		row.SetChildCardinality(1);
		for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
			row.data[c].SetValue(0, sink.sides[edge.s].reps[a][c]);
		}
		for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
			row.data[op.side_widths[edge.s] + c].SetValue(0, sink.sides[edge.t].reps[b][c]);
		}
		row.data[op.side_widths[edge.s] + op.side_widths[edge.t]].SetValue(0, Value::BIGINT(1));
	}

	void BakeSplit(idx_t e, idx_t a, idx_t b, string &prefix, string &suffix) {
		auto &exec = edge_ops_exec[e];
		auto &row = edge_row[e];
		if (!exec) {
			exec = make_uniq<ExpressionExecutor>(context);
			for (const auto *operand : cache_ops[e]) {
				exec->AddExpression(*operand);
			}
			if (!row) {
				row = make_uniq<DataChunk>();
				row->Initialize(BufferAllocator::Get(context), edge_types[e]);
			}
		}
		FillPairRow(*row, e, a, b);
		DataChunk out;
		vector<LogicalType> out_types(cache_ops[e].size(), LogicalType::VARCHAR);
		out.Initialize(BufferAllocator::Get(context), out_types);
		exec->Execute(*row, out);
		prefix.clear();
		suffix.clear();
		for (idx_t o = 0; o < cache_ops[e].size(); o++) {
			const Value v = out.data[o].GetValue(0);
			(o < cache_split[e] ? prefix : suffix) += v.IsNull() ? string() : StringValue::Get(v);
		}
	}

	//! `nulls[l]` = 1 when leaf l's prompt is NULL (its answer is NULL; the unit decides it without a call).
	vector<string> Bake(bool is_unary, idx_t domain, idx_t a, idx_t b, idx_t factor, vector<char> &nulls) {
		const bool one_factor = is_unary && factor < factor_sub[domain].size();
		auto &fn_sub = SubNode(is_unary, domain, factor)->Cast<BoundFunctionExpression>();
		const idx_t nleaf = (fn_sub.GetChildren().size() - 1) / 3; // tree + 3n (+meta)
		auto &exec = one_factor ? factor_exec[domain][factor] : is_unary ? side_exec[domain] : edge_exec[domain];
		auto &row = is_unary ? side_row[domain] : edge_row[domain]; // one row buffer per side, shared by its units
		if (!exec) {
			exec = make_uniq<ExpressionExecutor>(context); // call prompts are children [1 .. n]
			for (idx_t l = 0; l < nleaf; l++) {
				exec->AddExpression(*fn_sub.GetChildren()[1 + l]);
			}
		}
		if (!row) {
			row = make_uniq<DataChunk>();
			row->Initialize(BufferAllocator::Get(context), is_unary ? side_types[domain] : edge_types[domain]);
		}
		if (is_unary) {
			row->Reset();
			row->SetChildCardinality(1);
			auto &side = sink.sides[domain];
			for (idx_t c = 0; c < op.side_widths[domain]; c++) {
				row->data[c].SetValue(0, side.reps[a][c]);
			}
			row->data[op.side_widths[domain]].SetValue(0, Value::BIGINT(1));
		} else {
			FillPairRow(*row, domain, a, b);
		}
		DataChunk out;
		vector<LogicalType> out_types(nleaf, LogicalType::VARCHAR);
		out.Initialize(BufferAllocator::Get(context), out_types);
		exec->Execute(*row, out);
		vector<string> prompts(nleaf);
		nulls.assign(nleaf, 0);
		for (idx_t l = 0; l < nleaf; l++) {
			const Value v = out.data[l].GetValue(0);
			nulls[l] = v.IsNull() ? 1 : 0;
			prompts[l] = v.IsNull() ? string() : StringValue::Get(v);
		}
		return prompts;
	}

	//! One unit's prompts (+ prefixes when the pair prompt is cache-split) and its single-flight key:
	//! prefix+prompt concatenated, so cached and plain forms of the same pair share one flight. The key
	//! ends with the factor's signature (tree + leaf meta): the flight shares the factor's VERDICT, so two
	//! factors whose prompts coincide (sides of identical text) but differ in NOT, OR/AND or a wrapper
	//! must not share it.
	//! `factor`: for a unary unit, the side's unary factor to ask (index into plan.unary_factors[domain]), or any
	//! value >= their count for the side's whole unary tree; ignored for a pair unit.
	void Unit(bool is_unary, idx_t domain, idx_t a, idx_t b, idx_t factor, vector<string> &prompts,
	          vector<string> &prefixes, vector<char> &nulls, string &key) {
		prefixes.clear();
		// The full prompts decide NULL-ness (the all-columns-NULL guard lives in the prompt expression, not in
		// its split operands); a cache-split pair prompt is then re-baked as prefix + suffix when it is not NULL.
		prompts = Bake(is_unary, domain, a, b, factor, nulls);
		if (!is_unary && cache_enabled[domain] && prompts.size() == 1 && !nulls[0]) {
			string prefix, suffix;
			BakeSplit(domain, a, b, prefix, suffix);
			prompts[0] = std::move(suffix);
			prefixes.push_back(std::move(prefix));
		}
		key.clear();
		for (idx_t l = 0; l < prompts.size(); l++) {
			// prefix and prompt each NULL-flagged and length-prefixed (AIAppendKeyPart): a NULL leaf is not an
			// empty prompt, and no prefix/prompt split or separator byte can make two units collide
			AIAppendKeyPart(key, l < prefixes.size() ? prefixes[l] : string(), nulls[l] != 0);
			AIAppendKeyPart(key, prompts[l], nulls[l] != 0);
		}
		key += is_unary ? "u" : "e";
		key += !is_unary                            ? edge_sig[domain]
		       : factor < factor_sig[domain].size() ? factor_sig[domain][factor]
		                                            : side_sig[domain];
	}
};

// The root of the forest component holding `side` (follow parent edges up).
static idx_t ComponentRoot(const PhysicalAIFactorGraph::TermPlan &plan, const vector<idx_t> &parent_edge_of_side,
                           idx_t side) {
	while (parent_edge_of_side[side] != DConstants::INVALID_INDEX) {
		auto &edge = plan.edges[parent_edge_of_side[side]];
		side = edge.s == side ? edge.t : edge.s;
	}
	return side;
}

// Does the component rooted at `root` hold one tuple (a live, unary-true representative per side) whose every
// edge is confirmed TRUE? The same anchored walk the existential satisfaction check runs, restricted to that
// component's sides and without an anchor.
static bool ComponentHasConfirmedTuple(const FactorGraphSinkState &sink, const PhysicalAIFactorGraph::TermPlan &plan,
                                       const vector<idx_t> &parent_edge_of_side, idx_t root,
                                       const vector<vector<char>> &live, const vector<vector<char>> &uval,
                                       const std::function<bool(idx_t, idx_t, idx_t)> &pair_true) {
	const idx_t k1 = live.size();
	vector<idx_t> sides;
	for (idx_t s = 0; s < k1; s++) {
		if (ComponentRoot(plan, parent_edge_of_side, s) == root) {
			sides.push_back(s);
		}
	}
	vector<idx_t> assignment(k1, DConstants::INVALID_INDEX);
	std::function<bool(idx_t)> walk = [&](idx_t pos) {
		if (pos >= sides.size()) {
			return true;
		}
		const idx_t side = sides[pos];
		for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
			if (!live[side][r] || (!plan.unary_leaves[side].empty() && uval[side][r] != 1)) {
				continue;
			}
			bool ok = true;
			for (idx_t e = 0; e < plan.edges.size() && ok; e++) {
				auto &ed = plan.edges[e];
				if (ed.s != side && ed.t != side) {
					continue;
				}
				const idx_t other = ed.s == side ? ed.t : ed.s;
				if (assignment[other] == DConstants::INVALID_INDEX) {
					continue; // checked when the other endpoint is assigned
				}
				ok = ed.s == side ? pair_true(e, r, assignment[other]) : pair_true(e, assignment[other], r);
			}
			if (!ok) {
				continue;
			}
			assignment[side] = r;
			if (walk(pos + 1)) {
				return true;
			}
			assignment[side] = DConstants::INVALID_INDEX;
		}
		return false;
	};
	return walk(0);
}

// EXISTENTIAL usefulness seed, shared by both schedulers: a member of the existential side is useful while it
// is live, unsatisfied and (with unary leaves) confirmed. The edge graph is a forest, not always a tree:
// EXISTS (... FROM tb, tc, td WHERE f(ta.x, tb.y) AND g(tc.z, td.w)) has a second component no parent edge
// reaches from the existential side. Its pairs would never be asked and no member could ever be satisfied.
// Such a component is independent of every member: ONE confirmed tuple in it satisfies the existential for
// all of them, so its root is useful in full while an unsatisfied member remains AND the component holds no
// confirmed tuple yet; the caller then lets children inherit across their parent edge in level order.
static void SeedExistentialUsefulness(const FactorGraphSinkState &sink, const PhysicalAIFactorGraph::TermPlan &plan,
                                      idx_t es, const vector<idx_t> &parent_edge_of_side,
                                      const vector<vector<char>> &live, const vector<vector<char>> &uval,
                                      const vector<char> &satisfied,
                                      const std::function<bool(idx_t, idx_t, idx_t)> &pair_true,
                                      vector<vector<char>> &useful) {
	const idx_t k1 = useful.size();
	for (idx_t s = 0; s < k1; s++) {
		useful[s].assign(sink.sides[s].reps.size(), 0);
	}
	for (idx_t m = 0; m < satisfied.size(); m++) {
		useful[es][m] =
		    static_cast<char>(live[es][m] && !satisfied[m] && (plan.unary_leaves[es].empty() || uval[es][m] == 1));
	}
	bool any_open = false;
	for (idx_t m = 0; m < satisfied.size() && !any_open; m++) {
		any_open = useful[es][m] != 0;
	}
	const idx_t es_root = ComponentRoot(plan, parent_edge_of_side, es);
	for (idx_t s = 0; s < k1; s++) {
		if (s == es_root || parent_edge_of_side[s] != DConstants::INVALID_INDEX) {
			continue;
		}
		const bool open =
		    any_open && !ComponentHasConfirmedTuple(sink, plan, parent_edge_of_side, s, live, uval, pair_true);
		for (idx_t m = 0; m < sink.sides[s].reps.size(); m++) {
			useful[s][m] = static_cast<char>(open && live[s][m] && (plan.unary_leaves[s].empty() || uval[s][m] == 1));
		}
	}
}

static void LazyFactorGraphEvaluation(ClientContext &context, const PhysicalAIFactorGraph &op,
                                      const PhysicalAIFactorGraph::TermPlan &plan, int64_t term_limit,
                                      FactorGraphSinkState &sink, const string &query_text,
                                      const vector<idx_t> &parent_edge_of_side, const vector<idx_t> &side_level,
                                      vector<vector<char>> &live, vector<vector<std::pair<idx_t, idx_t>>> &edge_pass,
                                      const vector<vector<std::pair<idx_t, idx_t>>> &pre_true) {
	const idx_t k1 = op.side_widths.size();
	const idx_t ne = plan.edges.size();
	const idx_t total_width = op.side_offsets.back() + op.side_widths.back();
	auto &fn = op.node->Cast<BoundFunctionExpression>();
	const idx_t batch_cap = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);

	// Edge orientation: the child side is the one whose parent edge is this edge.
	vector<idx_t> child_of(ne);
	vector<idx_t> parent_of(ne);
	for (idx_t e = 0; e < ne; e++) {
		const bool s_is_child = parent_edge_of_side[plan.edges[e].s] == e;
		child_of[e] = s_is_child ? plan.edges[e].s : plan.edges[e].t;
		parent_of[e] = s_is_child ? plan.edges[e].t : plan.edges[e].s;
	}
	// Edge dispatch order: shallower child first (upstream support arrives first).
	vector<idx_t> edge_order(ne);
	for (idx_t e = 0; e < ne; e++) {
		edge_order[e] = e;
	}
	std::sort(edge_order.begin(), edge_order.end(),
	          [&](idx_t a, idx_t b) { return side_level[child_of[a]] < side_level[child_of[b]]; });

	// Member unary state: 0 pending, 1 confirmed-true, 2 false. Sides without unary leaves confirm.
	vector<vector<char>> uval(k1);
	for (idx_t s = 0; s < k1; s++) {
		uval[s].assign(sink.sides[s].reps.size(), plan.unary_leaves[s].empty() ? 1 : 0);
	}
	// Pair state per edge (row-major over edge.s x edge.t): 0 unknown, 1 true, 2 false.
	vector<vector<char>> pval(ne);
	for (idx_t e = 0; e < ne; e++) {
		pval[e].assign(sink.sides[plan.edges[e].s].reps.size() * sink.sides[plan.edges[e].t].reps.size(), 0);
	}
	auto pkey = [&](idx_t e, idx_t si, idx_t tj) {
		return si * sink.sides[plan.edges[e].t].reps.size() + tj;
	};

	// A parent-side member supports its child edge once it has a confirmed-true pair on ITS
	// parent edge (roots support unconditionally).
	auto supported = [&](idx_t s, idx_t m) {
		const idx_t pe = parent_edge_of_side[s];
		if (pe == DConstants::INVALID_INDEX) {
			return true;
		}
		const bool m_on_s_axis = plan.edges[pe].s == s;
		const idx_t nother = sink.sides[m_on_s_axis ? plan.edges[pe].t : plan.edges[pe].s].reps.size();
		for (idx_t q = 0; q < nother; q++) {
			const idx_t key = m_on_s_axis ? pkey(pe, m, q) : pkey(pe, q, m);
			if (pval[pe][key] == 1) {
				return true;
			}
		}
		return false;
	};

	// Deletion cascade: a member dies on unary-false, or when an incident edge holds no true and
	// no still-possible pair with a live partner. Recomputed to fixed point after every batch.
	auto cascade = [&]() {
		bool changed = true;
		while (changed) {
			changed = false;
			for (idx_t e = 0; e < ne; e++) {
				auto &edge = plan.edges[e];
				const idx_t ns = sink.sides[edge.s].reps.size();
				const idx_t nt = sink.sides[edge.t].reps.size();
				for (idx_t i = 0; i < ns; i++) {
					if (!live[edge.s][i]) {
						continue;
					}
					bool ok = false;
					for (idx_t j = 0; j < nt && !ok; j++) {
						ok = live[edge.t][j] && pval[e][pkey(e, i, j)] != 2;
					}
					if (!ok) {
						live[edge.s][i] = 0;
						changed = true;
					}
				}
				for (idx_t j = 0; j < nt; j++) {
					if (!live[edge.t][j]) {
						continue;
					}
					bool ok = false;
					for (idx_t i = 0; i < ns && !ok; i++) {
						ok = live[edge.s][i] && pval[e][pkey(e, i, j)] != 2;
					}
					if (!ok) {
						live[edge.t][j] = 0;
						changed = true;
					}
				}
			}
			for (idx_t s = 0; s < k1; s++) {
				for (idx_t m = 0; m < uval[s].size(); m++) {
					if (live[s][m] && uval[s][m] == 2) {
						live[s][m] = 0;
						changed = true;
					}
				}
			}
		}
	};

	// EXISTENTIAL mode: the consumer (a MARK join) needs only a boolean per member of
	// op.existential_side. A member is SATISFIED once one full tuple through it is confirmed;
	// satisfied members stop generating work, and downstream units stay dispatchable only while
	// some UNSATISFIED member still routes through them (the `useful` flags, recomputed with
	// every cascade). Emission is unchanged: confirmed tuples only, so the mark sees >=1 row per
	// existing member and none per non-existing.
	const bool existential = op.existential_side != DConstants::INVALID_INDEX;
	vector<char> satisfied;
	if (existential) {
		satisfied.assign(sink.sides[op.existential_side].reps.size(), 0);
	}
	vector<vector<char>> useful(k1);
	auto refresh_existential = [&]() {
		if (!existential) {
			return;
		}
		// TRUE-pair adjacency once per refresh
		vector<vector<std::unordered_map<idx_t, vector<idx_t>>>> adj(ne);
		for (idx_t e = 0; e < ne; e++) {
			auto &edge = plan.edges[e];
			adj[e].resize(2);
			const idx_t ns = sink.sides[edge.s].reps.size();
			const idx_t nt = sink.sides[edge.t].reps.size();
			for (idx_t i = 0; i < ns; i++) {
				for (idx_t j = 0; j < nt; j++) {
					if (pval[e][pkey(e, i, j)] == 1 && live[edge.s][i] && live[edge.t][j]) {
						adj[e][0][i].push_back(j);
						adj[e][1][j].push_back(i);
					}
				}
			}
		}
		// satisfaction: anchored walk from each unsatisfied member over confirmed pairs
		const idx_t es = op.existential_side;
		for (idx_t m = 0; m < satisfied.size(); m++) {
			if (satisfied[m] || !live[es][m]) {
				continue;
			}
			vector<idx_t> assignment(k1, DConstants::INVALID_INDEX);
			assignment[es] = m;
			std::function<bool(idx_t)> walk = [&](idx_t side) {
				while (side < k1 && side == es) {
					side++;
				}
				if (side >= k1) {
					return true;
				}
				for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
					if (!live[side][r] || (!plan.unary_leaves[side].empty() && uval[side][r] != 1)) {
						continue;
					}
					bool ok = true;
					for (idx_t e = 0; e < ne && ok; e++) {
						auto &ed = plan.edges[e];
						idx_t other = DConstants::INVALID_INDEX;
						bool on_s = false;
						if (ed.s == side) {
							other = ed.t;
							on_s = true;
						} else if (ed.t == side) {
							other = ed.s;
						} else {
							continue;
						}
						if (assignment[other] == DConstants::INVALID_INDEX) {
							continue; // other endpoint not assigned yet; checked when it is
						}
						const idx_t key = on_s ? pkey(e, r, assignment[other]) : pkey(e, assignment[other], r);
						ok = pval[e][key] == 1;
					}
					if (!ok) {
						continue;
					}
					assignment[side] = r;
					if (walk(side + 1)) {
						return true;
					}
					assignment[side] = DConstants::INVALID_INDEX;
				}
				return false;
			};
			if (walk(0)) {
				satisfied[m] = 1;
			}
		}
		SeedExistentialUsefulness(
		    sink, plan, es, parent_edge_of_side, live, uval, satisfied,
		    [&](idx_t e, idx_t i, idx_t j) { return pval[e][pkey(e, i, j)] == 1; }, useful);
		// sides in increasing level order inherit usefulness across their parent edge
		vector<idx_t> order(k1);
		for (idx_t s = 0; s < k1; s++) {
			order[s] = s;
		}
		std::sort(order.begin(), order.end(), [&](idx_t x, idx_t y) { return side_level[x] < side_level[y]; });
		for (const auto s : order) {
			const idx_t pe = parent_edge_of_side[s];
			if (pe == DConstants::INVALID_INDEX) {
				continue;
			}
			auto &edge = plan.edges[pe];
			const bool s_is_s = edge.s == s;
			const idx_t ps = s_is_s ? edge.t : edge.s;
			for (idx_t m = 0; m < sink.sides[s].reps.size(); m++) {
				if (!live[s][m] || (!plan.unary_leaves[s].empty() && uval[s][m] != 1)) {
					continue;
				}
				const idx_t nother = sink.sides[ps].reps.size();
				for (idx_t q = 0; q < nother && !useful[s][m]; q++) {
					const idx_t key = s_is_s ? pkey(pe, m, q) : pkey(pe, q, m);
					useful[s][m] = static_cast<char>(useful[ps][q] && pval[pe][key] == 1);
				}
			}
		}
	};
	auto parent_ok = [&](idx_t e, idx_t pm) {
		if (existential) {
			return useful[parent_of[e]][pm] != 0;
		}
		return supported(parent_of[e], pm);
	};
	refresh_existential();

	// Sub-nodes, chunk layouts and the prefix-cache split, built once per domain.
	GraphUnitBaker baker(context, op, plan, sink);
	auto &side_sub = baker.side_sub;
	auto &edge_sub = baker.edge_sub;

	if (AIConfig::Get().debug_log.find("graph") != string::npos) {
		for (idx_t e = 0; e < ne; e++) {
			fprintf(stderr, "[graph] edge%llu sides %llu-%llu parent=%llu child=%llu childlevel=%llu\n",
			        (unsigned long long)e, (unsigned long long)plan.edges[e].s, (unsigned long long)plan.edges[e].t,
			        (unsigned long long)parent_of[e], (unsigned long long)child_of[e],
			        (unsigned long long)side_level[child_of[e]]);
		}
		for (idx_t s = 0; s < k1; s++) {
			fprintf(stderr, "[graph] side%llu reps=%llu unary_leaves=%llu\n", (unsigned long long)s,
			        (unsigned long long)sink.sides[s].reps.size(), (unsigned long long)plan.unary_leaves[s].size());
		}
	}
	// Adaptive mode: observed pass-rate statistics drive frontier priorities -- the graph-level
	// analogue of per-tuple filter reordering. Estimates are Laplace-smoothed observed rates,
	// per domain and per member (falling back to the domain rate while a member is unsampled).
	// A unit's score is its expected pruning payoff: P(fail) x the unknown pairs its member's
	// death would cancel (spread over the member's remaining pairs on that edge). Selection
	// stays strictly behind the same need gate, so adaptivity can only REDUCE calls, never add.
	const bool adaptive = true;
	vector<idx_t> un_evals(k1, 0), un_passes(k1, 0);
	vector<idx_t> ed_evals(ne, 0), ed_passes(ne, 0);
	vector<vector<idx_t>> me_evals_s(ne), me_passes_s(ne), me_evals_t(ne), me_passes_t(ne);
	for (idx_t e = 0; e < ne; e++) {
		me_evals_s[e].assign(sink.sides[plan.edges[e].s].reps.size(), 0);
		me_passes_s[e].assign(sink.sides[plan.edges[e].s].reps.size(), 0);
		me_evals_t[e].assign(sink.sides[plan.edges[e].t].reps.size(), 0);
		me_passes_t[e].assign(sink.sides[plan.edges[e].t].reps.size(), 0);
	}
	auto est = [](idx_t passes, idx_t evals) {
		return (1.0 + static_cast<double>(passes)) / (2.0 + static_cast<double>(evals));
	};
	auto member_est = [&](idx_t e, bool s_axis, idx_t m) {
		const idx_t ev = s_axis ? me_evals_s[e][m] : me_evals_t[e][m];
		const idx_t ps = s_axis ? me_passes_s[e][m] : me_passes_t[e][m];
		return ev > 0 ? est(ps, ev) : est(ed_passes[e], ed_evals[e]);
	};
	// unknown pairs of member m (on side s) across incident edges, with live partners
	auto unknown_pairs = [&](idx_t s, idx_t m, idx_t skip_edge, idx_t only_edge) {
		idx_t n = 0;
		for (idx_t e = 0; e < ne; e++) {
			if (e == skip_edge || (only_edge != DConstants::INVALID_INDEX && e != only_edge)) {
				continue;
			}
			auto &edge = plan.edges[e];
			if (edge.s == s) {
				const idx_t nt = sink.sides[edge.t].reps.size();
				for (idx_t j = 0; j < nt; j++) {
					n += live[edge.t][j] && pval[e][pkey(e, m, j)] == 0;
				}
			} else if (edge.t == s) {
				const idx_t ns = sink.sides[edge.s].reps.size();
				for (idx_t i = 0; i < ns; i++) {
					n += live[edge.s][i] && pval[e][pkey(e, i, m)] == 0;
				}
			}
		}
		return n;
	};

	// Per-member sibling sequencing (adaptive mode): a member's CHILD edges evaluate one at a
	// time, in the member's own most-lethal-first order (observed estimates). Sequencing keeps
	// the call envelope -- finishing one sibling can kill the member and cancel the others --
	// while the per-member ORDER is where adaptivity earns its call savings.
	auto edge_incomplete_for = [&](idx_t e, idx_t s, idx_t m) {
		auto &edge = plan.edges[e];
		if (edge.s == s) {
			const idx_t nt = sink.sides[edge.t].reps.size();
			for (idx_t j = 0; j < nt; j++) {
				if (live[edge.t][j] && pval[e][pkey(e, m, j)] == 0) {
					return true;
				}
			}
		} else {
			const idx_t ns = sink.sides[edge.s].reps.size();
			for (idx_t i = 0; i < ns; i++) {
				if (live[edge.s][i] && pval[e][pkey(e, i, m)] == 0) {
					return true;
				}
			}
		}
		return false;
	};
	auto open_child_edge = [&](idx_t s, idx_t m) {
		idx_t open = DConstants::INVALID_INDEX;
		double open_est = 2.0;
		for (idx_t e = 0; e < ne; e++) {
			if (parent_of[e] != s || !edge_incomplete_for(e, s, m)) {
				continue;
			}
			const double p_pass = member_est(e, plan.edges[e].s == s, m);
			if (p_pass < open_est) {
				open_est = p_pass;
				open = e;
			}
		}
		return open;
	};

	// Kill credit of endpoint m for a pair on edge e: only a member with NO true on e can still
	// die there (a confirmed true makes death-via-e impossible); credit = the unknown pairs its
	// death would cancel elsewhere, discounted by P(all its remaining e-pairs fail). A small
	// base term keeps raw lethality (p_fail) driving the ordering when no kill is imminent --
	// the earlier kill-share formula rewarded near-COMPLETE members instead of near-DEAD ones,
	// which pinned the scheduler on finishing the current edge and prevented the pivot.
	auto kill_credit = [&](idx_t e, idx_t s, bool s_axis, idx_t m) {
		const idx_t passes = s_axis ? me_passes_s[e][m] : me_passes_t[e][m];
		if (passes > 0) {
			return 0.0;
		}
		const idx_t rem = unknown_pairs(s, m, DConstants::INVALID_INDEX, e);
		const double fail = 1.0 - member_est(e, s_axis, m);
		double all_fail = 1.0;
		for (idx_t r = 1; r < MinValue<idx_t>(rem, 12); r++) {
			all_fail *= fail;
		}
		return static_cast<double>(unknown_pairs(s, m, e, DConstants::INVALID_INDEX)) * all_fail;
	};

	// LIMIT early-stop: count CONFIRMED output rows (all predicates TRUE, count products
	// included) with an early bail at k1. Only confirmed tuples are ever emitted, so stopping at
	// k1 confirmed rows is sound and complete for the LIMIT above.
	auto confirmed_at_least = [&](int64_t want) {
		vector<vector<std::unordered_map<idx_t, vector<idx_t>>>> adj(ne);
		for (idx_t e = 0; e < ne; e++) {
			auto &edge = plan.edges[e];
			adj[e].resize(2);
			const idx_t ns = sink.sides[edge.s].reps.size();
			const idx_t nt = sink.sides[edge.t].reps.size();
			for (idx_t i = 0; i < ns; i++) {
				for (idx_t j = 0; j < nt; j++) {
					if (pval[e][pkey(e, i, j)] == 1 && live[edge.s][i] && live[edge.t][j]) {
						adj[e][0][i].push_back(j);
						adj[e][1][j].push_back(i);
					}
				}
			}
		}
		int64_t confirmed = 0;
		vector<idx_t> assignment(k1);
		std::function<bool(idx_t)> walk = [&](idx_t side) {
			if (side == k1) {
				confirmed = SaturatingAdd(confirmed, TupleMultiplicity(sink, assignment));
				return confirmed >= want;
			}
			for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
				if (!live[side][r] || (!plan.unary_leaves[side].empty() && uval[side][r] != 1)) {
					continue;
				}
				bool ok = true;
				for (idx_t e = 0; e < ne && ok; e++) {
					auto &ed = plan.edges[e];
					if (ed.s == side && ed.t < side) {
						auto it = adj[e][0].find(r);
						ok = it != adj[e][0].end() &&
						     std::binary_search(it->second.begin(), it->second.end(), assignment[ed.t]);
					} else if (ed.t == side && ed.s < side) {
						auto it = adj[e][1].find(r);
						ok = it != adj[e][1].end() &&
						     std::binary_search(it->second.begin(), it->second.end(), assignment[ed.s]);
					} else if ((ed.s == side && ed.t > side) || (ed.t == side && ed.s > side)) {
						// later side: this member needs at least one confirmed partner to extend
						auto &m = ed.s == side ? adj[e][0] : adj[e][1];
						ok = m.find(r) != m.end();
					}
				}
				if (!ok) {
					continue;
				}
				assignment[side] = r;
				if (walk(side + 1)) {
					return true;
				}
			}
			return false;
		};
		return walk(0);
	};

	// Streaming driver: persistent pool + single-flight + dispatch marking (state 3). Prompts
	// bake per unit through per-domain executors (vectorized expression evaluation over a
	// one-row chunk); identical prompt vectors share one call via the single-flight table.
	GraphStreamPool pool;
	pool.Start(context, batch_cap, query_text);
	std::unordered_map<string, vector<GraphStreamPool::Done>> parked; // key -> units awaiting the winner
	auto submit_unit = [&](bool is_unary, idx_t domain, idx_t a, idx_t b) {
		idx_t expected_reuse = 0;
		if (!is_unary && baker.cache_enabled[domain]) {
			// Declaration only: the client's prefix lifecycle decides whether a breakpoint is emitted;
			// the hint is this member's remaining live pairs, this call included.
			expected_reuse = unknown_pairs(baker.CacheGroupSide(domain), baker.CacheGroupMember(domain, a, b),
			                               DConstants::INVALID_INDEX, domain);
			expected_reuse = expected_reuse > 0 ? expected_reuse - 1 : 0; // the later reads: this call excluded
		}
		vector<string> prompts, prefixes;
		vector<char> nulls;
		string key;
		baker.Unit(is_unary, domain, a, b, DConstants::INVALID_INDEX, prompts, prefixes, nulls, key);
		if (is_unary) {
			uval[domain][a] = 3; // dispatched
		} else {
			pval[domain][pkey(domain, a, b)] = 3;
		}
		auto it = parked.find(key);
		if (it != parked.end()) {
			it->second.push_back(GraphStreamPool::Done {key, is_unary, domain, a, b, false});
			return; // identical prompt already in flight: park, the winner's value routes here
		}
		parked.emplace(key, vector<GraphStreamPool::Done> {});
		pool.Submit(GraphStreamPool::Task {is_unary ? side_sub[domain].get() : edge_sub[domain].get(),
		                                   std::move(prompts), std::move(prefixes), std::move(nulls), std::move(key),
		                                   is_unary, domain, a, b, expected_reuse});
	};
	auto apply_value = [&](bool is_unary, idx_t domain, idx_t a, idx_t b, bool value) {
		if (is_unary) {
			uval[domain][a] = static_cast<char>(value ? 1 : 2);
			un_evals[domain]++;
			un_passes[domain] += value ? 1 : 0;
		} else {
			pval[domain][pkey(domain, a, b)] = static_cast<char>(value ? 1 : 2);
			ed_evals[domain]++;
			ed_passes[domain] += value ? 1 : 0;
			me_evals_s[domain][a]++;
			me_passes_s[domain][a] += value ? 1 : 0;
			me_evals_t[domain][b]++;
			me_passes_t[domain][b] += value ? 1 : 0;
		}
	};
	// pairs an earlier term already proved for every tuple through them: TRUE without a call (they also
	// enter the pass-rate statistics, which only biases the order a little toward asking them later)
	for (idx_t e = 0; e < ne; e++) {
		for (const auto &pr : pre_true[e]) {
			apply_value(false, e, pr.first, pr.second, true);
		}
	}
	cascade();
	refresh_existential();
	auto harvest = [&](bool blocking) {
		vector<GraphStreamPool::Done> dones;
		pool.Drain(dones, blocking);
		for (auto &done : dones) {
			apply_value(done.is_unary, done.domain, done.a, done.b, done.value);
			auto it = parked.find(done.key);
			if (it != parked.end()) {
				for (auto &waiter : it->second) {
					apply_value(waiter.is_unary, waiter.domain, waiter.a, waiter.b, done.value);
				}
				parked.erase(it);
			}
		}
		if (!dones.empty()) {
			pool.MarkApplied(dones.size());
			cascade();
			refresh_existential();
		}
		return !dones.empty();
	};

	// Frontier loop: submit up to the free slots from the best domain, then harvest completions
	// as they land -- continuous refill, no drain barriers.
	for (;;) {
		if (term_limit >= 0 && confirmed_at_least(term_limit)) {
			break; // k1 output rows confirmed: the LIMIT above needs nothing more
		}
		harvest(false); // apply anything that landed while selecting
		const idx_t pending_now = pool.Pending();
		if (pending_now >= batch_cap) {
			harvest(true); // pool full: wait for a slot to open
			continue;
		}
		const idx_t slots = batch_cap - pending_now;
		if (adaptive) {
			// score every ready unit; evaluate a batch from the best-scoring unit's domain
			double best_unary_score = -1;
			idx_t best_unary_side = k1;
			for (idx_t s = 0; s < k1; s++) {
				if (plan.unary_leaves[s].empty()) {
					continue;
				}
				for (idx_t m = 0; m < uval[s].size(); m++) {
					if (!live[s][m] || uval[s][m] != 0) {
						continue;
					}
					const double score = (1.0 - est(un_passes[s], un_evals[s])) *
					                         static_cast<double>(unknown_pairs(s, m, DConstants::INVALID_INDEX,
					                                                           DConstants::INVALID_INDEX)) +
					                     1.0;
					if (score > best_unary_score) {
						best_unary_score = score;
						best_unary_side = s;
					}
				}
			}
			double best_pair_score = -1;
			idx_t best_pair_edge = ne;
			for (idx_t e = 0; e < ne; e++) {
				auto &edge = plan.edges[e];
				const idx_t ns = sink.sides[edge.s].reps.size();
				const idx_t nt = sink.sides[edge.t].reps.size();
				for (idx_t i = 0; i < ns; i++) {
					if (!live[edge.s][i] || uval[edge.s][i] != 1) {
						continue;
					}
					for (idx_t j = 0; j < nt; j++) {
						if (!live[edge.t][j] || uval[edge.t][j] != 1 || pval[e][pkey(e, i, j)] != 0) {
							continue;
						}
						const idx_t pm = parent_of[e] == edge.s ? i : j;
						if (!parent_ok(e, pm) || open_child_edge(parent_of[e], pm) != e) {
							continue;
						}
						const double p_fail = 1.0 - 0.5 * (member_est(e, true, i) + member_est(e, false, j));
						const double score =
						    p_fail * (0.1 + kill_credit(e, edge.s, true, i) + kill_credit(e, edge.t, false, j));
						if (score > best_pair_score) {
							best_pair_score = score;
							best_pair_edge = e;
						}
					}
				}
			}
			if (best_unary_side < k1 && (best_pair_edge == ne || best_unary_score >= best_pair_score)) {
				// batch = highest-impact ready members of the chosen side
				vector<std::pair<double, idx_t>> scored;
				for (idx_t m = 0; m < uval[best_unary_side].size(); m++) {
					if (live[best_unary_side][m] && uval[best_unary_side][m] == 0) {
						scored.emplace_back(
						    -static_cast<double>(unknown_pairs(best_unary_side, m, DConstants::INVALID_INDEX,
						                                       DConstants::INVALID_INDEX)),
						    m);
					}
				}
				std::sort(scored.begin(), scored.end());
				idx_t submitted = 0;
				for (idx_t u = 0; u < scored.size() && submitted < slots; u++) {
					submit_unit(true, best_unary_side, scored[u].second, 0);
					submitted++;
				}
				harvest(false);
				continue;
			}
			if (best_pair_edge == ne) {
				if (pool.Pending() > 0) {
					harvest(true); // nothing selectable until in-flight results land
					continue;
				}
				break;
			}
			// batch = top-scoring ready pairs of the chosen edge
			auto &edge = plan.edges[best_pair_edge];
			const idx_t e = best_pair_edge;
			const idx_t ns = sink.sides[edge.s].reps.size();
			const idx_t nt = sink.sides[edge.t].reps.size();
			vector<std::pair<double, std::pair<idx_t, idx_t>>> scored;
			for (idx_t i = 0; i < ns; i++) {
				if (!live[edge.s][i] || uval[edge.s][i] != 1) {
					continue;
				}
				for (idx_t j = 0; j < nt; j++) {
					if (!live[edge.t][j] || uval[edge.t][j] != 1 || pval[e][pkey(e, i, j)] != 0) {
						continue;
					}
					const idx_t pm = parent_of[e] == edge.s ? i : j;
					if (!parent_ok(e, pm) || open_child_edge(parent_of[e], pm) != e) {
						continue;
					}
					const double p_fail = 1.0 - 0.5 * (member_est(e, true, i) + member_est(e, false, j));
					const double score =
					    p_fail * (0.1 + kill_credit(e, edge.s, true, i) + kill_credit(e, edge.t, false, j));
					scored.emplace_back(-score, std::make_pair(i, j));
				}
			}
			std::sort(scored.begin(), scored.end());
			// Diversity fill: spread submissions across MEMBERS (at most 2 per endpoint until
			// candidates run out) -- keeps future frontiers and the pool full.
			idx_t submitted = 0;
			std::unordered_map<idx_t, idx_t> used_s, used_t;
			vector<std::pair<idx_t, idx_t>> taken;
			for (idx_t u = 0; u < scored.size() && submitted < slots; u++) {
				if (used_s[scored[u].second.first] >= 2 || used_t[scored[u].second.second] >= 2) {
					continue;
				}
				used_s[scored[u].second.first]++;
				used_t[scored[u].second.second]++;
				taken.push_back(scored[u].second);
				submit_unit(false, e, scored[u].second.first, scored[u].second.second);
				submitted++;
			}
			for (idx_t u = 0; u < scored.size() && submitted < slots; u++) {
				if (std::find(taken.begin(), taken.end(), scored[u].second) == taken.end()) {
					submit_unit(false, e, scored[u].second.first, scored[u].second.second);
					submitted++;
				}
			}
			harvest(false);
			continue;
		}
		// unary first (cheapest pruning per call), sides in order
		idx_t pick_side = k1;
		vector<idx_t> members;
		for (idx_t s = 0; s < k1 && members.empty(); s++) {
			if (plan.unary_leaves[s].empty()) {
				continue;
			}
			for (idx_t m = 0; m < uval[s].size() && members.size() < slots; m++) {
				if (live[s][m] && uval[s][m] == 0) {
					members.push_back(m);
				}
			}
			if (!members.empty()) {
				pick_side = s;
			}
		}
		if (pick_side < k1) {
			for (const auto m : members) {
				submit_unit(true, pick_side, m, 0);
			}
			harvest(false);
			continue;
		}
		// then pairs: shallowest child edge with strict-need-ready units
		idx_t pick_edge = ne;
		vector<std::pair<idx_t, idx_t>> pairs;
		for (const auto e : edge_order) {
			auto &edge = plan.edges[e];
			const idx_t ns = sink.sides[edge.s].reps.size();
			const idx_t nt = sink.sides[edge.t].reps.size();
			for (idx_t i = 0; i < ns && pairs.size() < slots; i++) {
				if (!live[edge.s][i] || uval[edge.s][i] != 1) {
					continue;
				}
				for (idx_t j = 0; j < nt && pairs.size() < slots; j++) {
					if (!live[edge.t][j] || uval[edge.t][j] != 1 || pval[e][pkey(e, i, j)] != 0) {
						continue;
					}
					const idx_t pm = parent_of[e] == edge.s ? i : j;
					if (!parent_ok(e, pm)) {
						continue;
					}
					pairs.emplace_back(i, j);
				}
			}
			if (!pairs.empty()) {
				pick_edge = e;
				break;
			}
		}
		if (pick_edge == ne) {
			if (pool.Pending() > 0) {
				harvest(true); // in-flight results may unlock more work
				continue;
			}
			break; // no dispatchable work left: every needed prompt is resolved
		}
		for (auto &pr : pairs) {
			submit_unit(false, pick_edge, pr.first, pr.second);
		}
		harvest(false);
	}
	pool.Shutdown(); // limit/existential early exits may leave stragglers: join before touching state

	for (idx_t s = 0; s < k1; s++) {
		if (plan.unary_leaves[s].empty()) {
			continue;
		}
		for (idx_t m = 0; m < uval[s].size(); m++) {
			if (uval[s][m] != 1) {
				live[s][m] = 0; // unevaluated or false unary: never emit (matters under LIMIT early-stop)
			}
		}
	}
	for (idx_t e = 0; e < ne; e++) {
		auto &edge = plan.edges[e];
		const idx_t ns = sink.sides[edge.s].reps.size();
		const idx_t nt = sink.sides[edge.t].reps.size();
		for (idx_t i = 0; i < ns; i++) {
			for (idx_t j = 0; j < nt; j++) {
				if (pval[e][pkey(e, i, j)] == 1 && live[edge.s][i] && live[edge.t][j]) {
					edge_pass[e].emplace_back(i, j);
				}
			}
		}
	}
}

//===--------------------------------------------------------------------===//
// Key buckets of the keyed edges (an absorbed equi-join). A member's partner domain on such an edge is its
// bucket on the other axis, not the whole side: per axis and member its bucket (INVALID = a NULL key, which
// joins nothing), per bucket its members on each axis and how many of them are live. Bucket identity is SQL
// equality of the key columns: DuckDB's value hash (which already folds -0.0 into +0.0 and normalizes
// intervals) narrows the candidates and DuckDB's value equality confirms -- never the values' text. An edge
// with keys and no leaves is EXACT: every bucket pair is true without a call, so it is never dispatched and
// never stored; bucket membership answers every question about its pairs, in the scheduler and in the final
// enumeration alike.
//===--------------------------------------------------------------------===//
struct GraphKeyBuckets {
	struct EdgeBuckets {
		vector<idx_t> bucket_of[2];
		vector<vector<idx_t>> members[2];
		vector<idx_t> live_in[2];
	};
	vector<EdgeBuckets> edges; // per edge of the term; empty for an edge without keys

	GraphKeyBuckets(const PhysicalAIFactorGraph::TermPlan &plan, const FactorGraphSinkState &sink,
	                const vector<vector<char>> &live) {
		edges.resize(plan.edges.size());
		for (idx_t e = 0; e < plan.edges.size(); e++) {
			auto &edge = plan.edges[e];
			if (!edge.Keyed()) {
				continue;
			}
			auto &b = edges[e];
			// a bucket's representative key, for the equality check behind a hash match
			struct Rep {
				idx_t side;
				idx_t member;
				const vector<idx_t> *cols;
			};
			vector<Rep> reps;
			std::unordered_map<hash_t, vector<idx_t>> by_hash;
			for (int x = 0; x < 2; x++) {
				const idx_t side = x ? edge.t : edge.s;
				const auto &cols = x ? edge.key_t : edge.key_s;
				b.bucket_of[x].assign(sink.sides[side].reps.size(), DConstants::INVALID_INDEX);
				for (idx_t m = 0; m < sink.sides[side].reps.size(); m++) {
					const auto &row = sink.sides[side].reps[m];
					hash_t h = 0;
					bool null_key = false;
					for (const auto c : cols) {
						if (row[c].IsNull()) {
							null_key = true;
							break;
						}
						h = CombineHash(h, row[c].Hash());
					}
					if (null_key) {
						continue;
					}
					idx_t id = DConstants::INVALID_INDEX;
					auto &candidates = by_hash[h];
					for (const auto cand : candidates) {
						const auto &rep = reps[cand];
						const auto &rep_row = sink.sides[rep.side].reps[rep.member];
						bool equal = true;
						for (idx_t c = 0; c < cols.size() && equal; c++) {
							equal = ValueOperations::Equals(row[cols[c]], rep_row[(*rep.cols)[c]]);
						}
						if (equal) {
							id = cand;
							break;
						}
					}
					if (id == DConstants::INVALID_INDEX) {
						id = reps.size();
						reps.push_back(Rep {side, m, &cols});
						candidates.push_back(id);
						for (int y = 0; y < 2; y++) {
							b.members[y].emplace_back();
							b.live_in[y].push_back(0);
						}
					}
					b.bucket_of[x][m] = id;
					b.members[x][id].push_back(m);
					b.live_in[x][id] += live[side][m] ? 1 : 0;
				}
			}
		}
	}
	//! the members of m's (axis x) bucket on the other axis; none for a NULL key
	const vector<idx_t> &Partners(idx_t e, int x, idx_t m) const {
		static const vector<idx_t> none;
		const idx_t b = edges[e].bucket_of[x][m];
		return b == DConstants::INVALID_INDEX ? none : edges[e].members[1 - x][b];
	}
	//! the live members of m's (axis x) bucket on the other axis
	idx_t LivePartners(idx_t e, int x, idx_t m) const {
		const idx_t b = edges[e].bucket_of[x][m];
		return b == DConstants::INVALID_INDEX ? 0 : edges[e].live_in[1 - x][b];
	}
	//! is the pair (i on axis 0, j on axis 1) one bucket, i.e. true on an exact edge?
	bool SameBucket(idx_t e, idx_t i, idx_t j) const {
		const idx_t b = edges[e].bucket_of[0][i];
		return b != DConstants::INVALID_INDEX && edges[e].bucket_of[1][j] == b;
	}
	//! member m (axis x) died: its bucket's partners lose a live partner
	void OnKill(idx_t e, int x, idx_t m) {
		const idx_t b = edges[e].bucket_of[x][m];
		if (b != DConstants::INVALID_INDEX) {
			edges[e].live_in[x][b]--;
		}
	}
};

//===--------------------------------------------------------------------===//
// Sparse lazy evaluation: the dense scheduler's need gate, cascade, existential mode, LIMIT stop
// and output, with state proportional to the MEMBERS and the CALLS MADE instead of the pair
// domain. Per member and incident edge it keeps counters (true and false pairs with a live
// partner, true pairs with any partner, in flight) and the partners it was evaluated with; the
// pairs still to ask are enumerated from a cursor over the partner side (rotated per member, so
// parents start on different children) and never stored. Selection scores MEMBERS, not pairs:
// O(members) per batch where the dense scheduler scans the pair domain.
//===--------------------------------------------------------------------===//
static void SparseLazyFactorGraphEvaluation(
    ClientContext &context, const PhysicalAIFactorGraph &op, const PhysicalAIFactorGraph::TermPlan &plan,
    int64_t term_limit, FactorGraphSinkState &sink, const string &query_text, const vector<idx_t> &parent_edge_of_side,
    const vector<idx_t> &side_level, vector<vector<char>> &live, vector<vector<std::pair<idx_t, idx_t>>> &edge_pass,
    const vector<vector<std::pair<idx_t, idx_t>>> &pre_true, GraphKeyBuckets &buckets) {
	const idx_t k1 = op.side_widths.size();
	const idx_t ne = plan.edges.size();
	const idx_t batch_cap = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);

	vector<idx_t> child_of(ne), parent_of(ne);
	for (idx_t e = 0; e < ne; e++) {
		const bool s_is_child = parent_edge_of_side[plan.edges[e].s] == e;
		child_of[e] = s_is_child ? plan.edges[e].s : plan.edges[e].t;
		parent_of[e] = s_is_child ? plan.edges[e].t : plan.edges[e].s;
	}
	vector<vector<idx_t>> incident(k1);
	for (idx_t e = 0; e < ne; e++) {
		incident[plan.edges[e].s].push_back(e);
		incident[plan.edges[e].t].push_back(e);
	}
	auto axis_of = [&](idx_t e, idx_t s) {
		return plan.edges[e].s == s ? 0 : 1;
	};
	auto side_of_axis = [&](idx_t e, int x) {
		return x ? plan.edges[e].t : plan.edges[e].s;
	};

	// Member unary state: 0 pending, 1 confirmed-true, 2 false, 3 dispatched.
	vector<vector<char>> uval(k1);
	vector<idx_t> live_count(k1, 0);
	for (idx_t s = 0; s < k1; s++) {
		uval[s].assign(sink.sides[s].reps.size(), plan.unary_leaves[s].empty() ? 1 : 0);
		for (const auto f : live[s]) {
			live_count[s] += f;
		}
	}

	// The keyed edges' buckets (GraphKeyBuckets): partner domains, exact-edge truth and bucket liveness.
	auto exact_only = [&](idx_t e) {
		return plan.edges[e].Keyed() && !plan.edges[e].Semantic();
	};
	// the live partners of member m (axis x) on edge e: its bucket on the other axis, or the whole other side
	auto partner_live = [&](idx_t e, int x, idx_t m) -> idx_t {
		return plan.edges[e].Keyed() ? buckets.LivePartners(e, x, m) : live_count[side_of_axis(e, 1 - x)];
	};
	// the partner domain of member m (axis x) on a keyed edge: the member ids of its bucket on the other axis
	auto bucket_partners = [&](idx_t e, int x, idx_t m) -> const vector<idx_t> & {
		return buckets.Partners(e, x, m);
	};
	// an exact edge's pair (i on axis 0, j on axis 1) is true iff the two share a bucket
	auto exact_pair = [&](idx_t e, idx_t i, idx_t j) {
		return buckets.SameBucket(e, i, j);
	};
	// does member m (axis x) have a live, unary-confirmed partner on exact edge e?
	auto exact_confirmed_partner = [&](idx_t e, int x, idx_t m) {
		const idx_t o = side_of_axis(e, 1 - x);
		for (const auto q : bucket_partners(e, x, m)) {
			if (live[o][q] && uval[o][q] == 1) {
				return true;
			}
		}
		return false;
	};
	// Per edge and axis (0 = members of edge.s, 1 = members of edge.t), per member: what the dense
	// pair array answered by scanning a row. tl/fl count true/false pairs whose partner is still
	// live; pass_any counts true pairs with any partner (what supports a child edge); ptrue/pfalse
	// list the partners (both endpoints live when the pair landed).
	struct AxisState {
		vector<uint32_t> tl, fl, pass_any, evals, inflight;
		vector<vector<uint32_t>> ptrue, pfalse;
	};
	vector<AxisState> axis_state[2];
	for (int x = 0; x < 2; x++) {
		axis_state[x].resize(ne);
		for (idx_t e = 0; e < ne; e++) {
			const idx_t n = sink.sides[side_of_axis(e, x)].reps.size();
			auto &a = axis_state[x][e];
			a.tl.assign(n, 0);
			a.fl.assign(n, 0);
			a.pass_any.assign(n, 0);
			a.evals.assign(n, 0);
			a.inflight.assign(n, 0);
			a.ptrue.resize(n);
			a.pfalse.resize(n);
		}
	}
	auto A = [&](idx_t e, int x) -> AxisState & {
		return axis_state[x][e];
	};
	vector<idx_t> ed_evals(ne, 0), ed_passes(ne, 0);
	// Unary verdicts are asked ONE FACTOR (side-local conjunct) at a time: fval[s][m * nf[s] + f] is the member's
	// state on factor f (0 pending, 1 true, 2 false, 3 in flight), uval[s][m] the member's summary (1 once every
	// factor is true, 2 at the first false, 3 while a factor is in flight, else 0). Each factor keeps its own
	// pass-rate statistics, so the scheduler orders the conjuncts of ALL sides by selectivity, as a per-row
	// filter orders its conjuncts, instead of asking a side's whole conjunction before the next side's.
	vector<idx_t> nf(k1, 0);
	vector<vector<char>> fval(k1);
	vector<vector<idx_t>> fac_evals(k1), fac_passes(k1);
	for (idx_t s = 0; s < k1; s++) {
		nf[s] = plan.unary_factors[s].size();
		fval[s].assign(sink.sides[s].reps.size() * nf[s], 0);
		fac_evals[s].assign(nf[s], 0);
		fac_passes[s].assign(nf[s], 0);
	}
	// Exploration shares: one slot-round of calls split evenly over the unary conjuncts, each conjunct's share spent
	// on members asked while it had no verdict yet (see factor_fail below).
	idx_t unary_factor_count = 0;
	for (idx_t s = 0; s < k1; s++) {
		unary_factor_count += nf[s];
	}
	const idx_t explore_quota = unary_factor_count ? (batch_cap + unary_factor_count - 1) / unary_factor_count : 0;
	vector<vector<idx_t>> explore_used(k1); // per factor: members asked it while it had no verdict yet
	for (idx_t s = 0; s < k1; s++) {
		explore_used[s].assign(nf[s], 0);
	}
	// Pair enumeration: per edge and parent-side member, a cursor over the child side (rotated per
	// member). A parent whose cursor meets a child with a pending unary verdict WAITS there and asks for
	// that verdict (`wanted`), rather than skipping ahead and remembering the child: remembering would
	// store every (parent, pending child) pair -- the pair domain again -- before a single pair call.
	vector<vector<uint32_t>> cursor(ne);
	for (idx_t e = 0; e < ne; e++) {
		cursor[e].assign(sink.sides[parent_of[e]].reps.size(), 0);
	}
	vector<vector<idx_t>> wanted(k1); // members whose unary verdict a waiting parent asked for
	vector<vector<char>> wanted_flag(k1);
	for (idx_t s = 0; s < k1; s++) {
		wanted_flag[s].assign(sink.sides[s].reps.size(), 0);
	}
	// pairs an earlier term proved (keyed s_rep * |t| + t_rep): landed as TRUE below, and the cursor steps
	// over them so they are never asked again
	vector<std::unordered_set<idx_t>> pre_known(ne);
	for (idx_t e = 0; e < ne; e++) {
		const idx_t nt = sink.sides[plan.edges[e].t].reps.size();
		for (const auto &pr : pre_true[e]) {
			pre_known[e].insert(pr.first * nt + pr.second);
		}
	}

	auto unknown = [&](idx_t e, int x, idx_t m) -> idx_t {
		if (exact_only(e)) {
			return 0; // nothing to ask: the key decides every pair
		}
		auto &a = A(e, x);
		const idx_t lp = partner_live(e, x, m);
		const idx_t known = a.tl[m] + a.fl[m] + a.inflight[m];
		return lp > known ? lp - known : 0;
	};
	// A member stays alive on an edge while some live partner is not a confirmed false.
	auto alive_on = [&](idx_t e, int x, idx_t m) {
		return partner_live(e, x, m) > A(e, x).fl[m];
	};
	auto unknown_pairs = [&](idx_t s, idx_t m, idx_t skip_edge, idx_t only_edge) {
		idx_t n = 0;
		for (const auto e : incident[s]) {
			if (e == skip_edge || (only_edge != DConstants::INVALID_INDEX && e != only_edge)) {
				continue;
			}
			n += unknown(e, axis_of(e, s), m);
		}
		return n;
	};
	// A parent-side member supports its child edge once it has a confirmed-true pair on ITS
	// parent edge (any partner, as the dense scheduler counts it); roots support unconditionally.
	auto supported = [&](idx_t s, idx_t m) {
		const idx_t pe = parent_edge_of_side[s];
		if (pe == DConstants::INVALID_INDEX) {
			return true;
		}
		if (exact_only(pe)) {
			return partner_live(pe, axis_of(pe, s), m) > 0; // a live bucket partner is a true pair
		}
		return A(pe, axis_of(pe, s)).pass_any[m] > 0;
	};

	// Deletion cascade. A death is only QUEUED (kill); the cascade drains the queue, taking each dead
	// member's pairs out of its partners' live counters, and only then judges liveness: the members a
	// false pair touched since the last cascade, and every member facing a side that shrank. The
	// counters are consistent only between a drained queue and the next kill, so each sweep collects
	// the dead first and kills them afterwards (a death can only make others dead, never alive).
	// Deciding inside apply_value, with a death queued but its partners' counters not yet adjusted,
	// killed a member that had exactly one live non-false partner left.
	std::deque<std::pair<idx_t, idx_t>> dying;
	vector<std::pair<idx_t, idx_t>> touched; // (side, member) whose false-pair count rose
	vector<char> side_shrunk(k1, 0);
	auto kill = [&](idx_t s, idx_t m) {
		if (!live[s][m]) {
			return;
		}
		live[s][m] = 0;
		live_count[s]--;
		for (const auto e : incident[s]) {
			if (plan.edges[e].Keyed()) {
				buckets.OnKill(e, axis_of(e, s), m); // its bucket partners lose a live partner
			}
		}
		dying.emplace_back(s, m);
	};
	auto cascade = [&]() {
		for (;;) {
			while (!dying.empty()) {
				const auto sm = dying.front();
				dying.pop_front();
				const idx_t s = sm.first, m = sm.second;
				for (const auto e : incident[s]) {
					const int x = axis_of(e, s);
					const idx_t o = side_of_axis(e, 1 - x);
					auto &mine = A(e, x);
					auto &theirs = A(e, 1 - x);
					for (const auto q : mine.ptrue[m]) {
						if (live[o][q] && theirs.tl[q] > 0) {
							theirs.tl[q]--;
						}
					}
					for (const auto q : mine.pfalse[m]) {
						if (live[o][q] && theirs.fl[q] > 0) {
							theirs.fl[q]--;
						}
					}
					side_shrunk[s] = 1;
				}
			}
			// consistent state: collect the dead, then kill
			vector<std::pair<idx_t, idx_t>> dead;
			auto dead_on_any_edge = [&](idx_t s, idx_t m) {
				for (const auto e : incident[s]) {
					if (!alive_on(e, axis_of(e, s), m)) {
						return true;
					}
				}
				return false;
			};
			for (const auto &sm : touched) {
				if (live[sm.first][sm.second] && dead_on_any_edge(sm.first, sm.second)) {
					dead.push_back(sm);
				}
			}
			touched.clear();
			for (idx_t s = 0; s < k1; s++) {
				if (!side_shrunk[s]) {
					continue;
				}
				side_shrunk[s] = 0;
				for (const auto e : incident[s]) {
					const int xo = 1 - axis_of(e, s);
					const idx_t o = side_of_axis(e, xo);
					for (idx_t q = 0; q < live[o].size(); q++) {
						if (live[o][q] && !alive_on(e, xo, q)) {
							dead.emplace_back(o, q);
						}
					}
				}
			}
			if (dead.empty()) {
				break;
			}
			for (const auto &sm : dead) {
				kill(sm.first, sm.second);
			}
		}
	};

	// EXISTENTIAL mode (see the dense scheduler): satisfied members stop generating work; downstream
	// units stay dispatchable while an unsatisfied member still routes through them.
	const bool existential = op.existential_side != DConstants::INVALID_INDEX;
	vector<char> satisfied;
	if (existential) {
		satisfied.assign(sink.sides[op.existential_side].reps.size(), 0);
	}
	vector<vector<char>> useful(k1);
	auto true_adjacency = [&]() {
		// [e][0]: live s_rep -> live t_reps with a confirmed pair, [1]: the transpose; lists sorted for lookup
		vector<vector<std::unordered_map<idx_t, vector<idx_t>>>> adj(ne);
		for (idx_t e = 0; e < ne; e++) {
			auto &edge = plan.edges[e];
			adj[e].resize(2);
			auto &as = A(e, 0);
			for (idx_t i = 0; i < as.ptrue.size(); i++) {
				if (!live[edge.s][i]) {
					continue;
				}
				for (const auto j : as.ptrue[i]) {
					if (live[edge.t][j]) {
						adj[e][0][i].push_back(j);
						adj[e][1][j].push_back(i);
					}
				}
			}
			for (int x = 0; x < 2; x++) {
				for (auto &entry : adj[e][x]) {
					std::sort(entry.second.begin(), entry.second.end());
				}
			}
		}
		return adj;
	};
	auto refresh_existential = [&]() {
		if (!existential) {
			return;
		}
		auto adj = true_adjacency();
		auto has_pair = [&](idx_t e, idx_t i, idx_t j) {
			if (exact_only(e)) {
				return live[plan.edges[e].s][i] && live[plan.edges[e].t][j] && exact_pair(e, i, j);
			}
			auto it = adj[e][0].find(i);
			return it != adj[e][0].end() && std::binary_search(it->second.begin(), it->second.end(), j);
		};
		const idx_t es = op.existential_side;
		for (idx_t m = 0; m < satisfied.size(); m++) {
			if (satisfied[m] || !live[es][m]) {
				continue;
			}
			vector<idx_t> assignment(k1, DConstants::INVALID_INDEX);
			assignment[es] = m;
			std::function<bool(idx_t)> walk = [&](idx_t side) {
				while (side < k1 && side == es) {
					side++;
				}
				if (side >= k1) {
					return true;
				}
				for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
					if (!live[side][r] || (!plan.unary_leaves[side].empty() && uval[side][r] != 1)) {
						continue;
					}
					bool ok = true;
					for (const auto e : incident[side]) {
						auto &ed = plan.edges[e];
						const idx_t other = ed.s == side ? ed.t : ed.s;
						if (assignment[other] == DConstants::INVALID_INDEX) {
							continue;
						}
						ok = ed.s == side ? has_pair(e, r, assignment[other]) : has_pair(e, assignment[other], r);
						if (!ok) {
							break;
						}
					}
					if (!ok) {
						continue;
					}
					assignment[side] = r;
					if (walk(side + 1)) {
						return true;
					}
					assignment[side] = DConstants::INVALID_INDEX;
				}
				return false;
			};
			if (walk(0)) {
				satisfied[m] = 1;
			}
		}
		SeedExistentialUsefulness(sink, plan, es, parent_edge_of_side, live, uval, satisfied, has_pair, useful);
		vector<idx_t> order(k1);
		for (idx_t s = 0; s < k1; s++) {
			order[s] = s;
		}
		std::sort(order.begin(), order.end(), [&](idx_t x, idx_t y) { return side_level[x] < side_level[y]; });
		for (const auto s : order) {
			const idx_t pe = parent_edge_of_side[s];
			if (pe == DConstants::INVALID_INDEX) {
				continue;
			}
			const int x = axis_of(pe, s);
			const idx_t ps = side_of_axis(pe, 1 - x);
			auto &a = A(pe, x);
			for (idx_t m = 0; m < sink.sides[s].reps.size(); m++) {
				if (!live[s][m] || (!plan.unary_leaves[s].empty() && uval[s][m] != 1)) {
					continue;
				}
				// useful once a parent it has a true pair with is: its bucket on an exact edge, else its ptrue list
				auto inherit = [&](idx_t q) {
					if (live[ps][q] && useful[ps][q]) {
						useful[s][m] = 1;
					}
				};
				if (exact_only(pe)) {
					for (const auto q : bucket_partners(pe, x, m)) {
						if (useful[s][m]) {
							break;
						}
						inherit(q);
					}
				} else {
					for (const auto q : a.ptrue[m]) {
						if (useful[s][m]) {
							break;
						}
						inherit(q);
					}
				}
			}
		}
	};
	auto parent_ok = [&](idx_t e, idx_t pm) {
		if (existential) {
			return useful[parent_of[e]][pm] != 0;
		}
		return supported(parent_of[e], pm);
	};
	refresh_existential();

	// Observed pass rates, Laplace-smoothed, per edge and per member (dense scheduler's estimates).
	auto est = [](idx_t passes, idx_t evals) {
		return (1.0 + static_cast<double>(passes)) / (2.0 + static_cast<double>(evals));
	};
	auto member_est = [&](idx_t e, int x, idx_t m) {
		auto &a = A(e, x);
		return a.evals[m] > 0 ? est(a.pass_any[m], a.evals[m]) : est(ed_passes[e], ed_evals[e]);
	};
	auto edge_incomplete_for = [&](idx_t e, idx_t s, idx_t m) {
		return unknown(e, axis_of(e, s), m) > 0;
	};
	auto open_child_edge = [&](idx_t s, idx_t m) {
		idx_t open = DConstants::INVALID_INDEX;
		double open_est = 2.0;
		for (const auto e : incident[s]) {
			if (parent_of[e] != s || !edge_incomplete_for(e, s, m)) {
				continue;
			}
			const double p_pass = member_est(e, axis_of(e, s), m);
			if (p_pass < open_est) {
				open_est = p_pass;
				open = e;
			}
		}
		return open;
	};
	auto kill_credit = [&](idx_t e, idx_t s, int x, idx_t m) {
		if (A(e, x).pass_any[m] > 0) {
			return 0.0;
		}
		const idx_t rem = unknown(e, x, m);
		const double fail = 1.0 - member_est(e, x, m);
		double all_fail = 1.0;
		for (idx_t r = 1; r < MinValue<idx_t>(rem, 12); r++) {
			all_fail *= fail;
		}
		return static_cast<double>(unknown_pairs(s, m, e, DConstants::INVALID_INDEX)) * all_fail;
	};

	// LIMIT early-stop: confirmed output rows (count products included) with an early bail at k1.
	auto confirmed_at_least = [&](int64_t want) {
		auto adj = true_adjacency();
		int64_t confirmed = 0;
		vector<idx_t> assignment(k1);
		std::function<bool(idx_t)> walk = [&](idx_t side) {
			if (side == k1) {
				confirmed = SaturatingAdd(confirmed, TupleMultiplicity(sink, assignment));
				return confirmed >= want;
			}
			for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
				if (!live[side][r] || (!plan.unary_leaves[side].empty() && uval[side][r] != 1)) {
					continue;
				}
				bool ok = true;
				for (idx_t e = 0; e < ne && ok; e++) {
					auto &ed = plan.edges[e];
					if (exact_only(e)) {
						// the key decides: an assigned partner shares the bucket; a later side holds a confirmed one
						if (ed.s == side && ed.t < side) {
							ok = exact_pair(e, r, assignment[ed.t]);
						} else if (ed.t == side && ed.s < side) {
							ok = exact_pair(e, assignment[ed.s], r);
						} else if (ed.s == side || ed.t == side) {
							ok = exact_confirmed_partner(e, axis_of(e, side), r);
						}
						continue;
					}
					if (ed.s == side && ed.t < side) {
						auto it = adj[e][0].find(r);
						ok = it != adj[e][0].end() &&
						     std::binary_search(it->second.begin(), it->second.end(), assignment[ed.t]);
					} else if (ed.t == side && ed.s < side) {
						auto it = adj[e][1].find(r);
						ok = it != adj[e][1].end() &&
						     std::binary_search(it->second.begin(), it->second.end(), assignment[ed.s]);
					} else if ((ed.s == side && ed.t > side) || (ed.t == side && ed.s > side)) {
						auto &m = ed.s == side ? adj[e][0] : adj[e][1];
						ok = m.find(r) != m.end();
					}
				}
				if (!ok) {
					continue;
				}
				assignment[side] = r;
				if (walk(side + 1)) {
					return true;
				}
			}
			return false;
		};
		return walk(0);
	};

	// What a unary verdict on member m puts at stake: the pairs still unknown on its semantic edges, plus the
	// unary calls its death would free through the exact edges. A death empties its bucket on an exact edge
	// only together with the bucket's other live members (so it carries 1 / live members of that bucket), and
	// an emptied bucket kills every partner in it: each pending partner is a call saved, and a partner of any
	// state carries on, through its own other exact edges, to the partners it would take with it (a link table
	// between two AI-filtered sides asks nothing itself but relays the deaths). The edge graph is a forest, so
	// the walk never revisits an edge. Memoized per (edge, axis, bucket) within one frontier round: every member
	// of a bucket shares the value, which keeps a round's scoring at O(members).
	// memo slots: per edge, a block of 2 x its bucket count (axis-major), so no two (edge, axis, bucket) collide
	vector<idx_t> memo_base(ne + 1, 0);
	for (idx_t e = 0; e < ne; e++) {
		memo_base[e + 1] = memo_base[e] + 2 * buckets.edges[e].live_in[0].size();
	}
	std::unordered_map<idx_t, double> bucket_stake_memo;
	std::function<double(idx_t, int, idx_t)> bucket_stake = [&](idx_t e, int x, idx_t b) -> double {
		const idx_t memo_key = memo_base[e] + static_cast<idx_t>(x) * buckets.edges[e].live_in[0].size() + b;
		auto it = bucket_stake_memo.find(memo_key);
		if (it != bucket_stake_memo.end()) {
			return it->second;
		}
		double stake = 0;
		const auto &eb = buckets.edges[e];
		const idx_t mine = eb.live_in[x][b];
		if (mine > 0) {
			const idx_t o = side_of_axis(e, 1 - x);
			for (const auto q : eb.members[1 - x][b]) {
				if (!live[o][q]) {
					continue;
				}
				stake += uval[o][q] == 0 ? 1.0 : 0.0;
				for (const auto e2 : incident[o]) {
					if (e2 == e || !exact_only(e2)) {
						continue;
					}
					const int x2 = axis_of(e2, o);
					const idx_t b2 = buckets.edges[e2].bucket_of[x2][q];
					if (b2 != DConstants::INVALID_INDEX) {
						stake += bucket_stake(e2, x2, b2);
					}
				}
			}
			stake /= static_cast<double>(mine);
		}
		bucket_stake_memo.emplace(memo_key, stake);
		return stake;
	};
	auto unary_stake = [&](idx_t s, idx_t m) {
		double stake = static_cast<double>(unknown_pairs(s, m, DConstants::INVALID_INDEX, DConstants::INVALID_INDEX));
		for (const auto e : incident[s]) {
			if (!exact_only(e)) {
				continue;
			}
			const int x = axis_of(e, s);
			const idx_t b = buckets.edges[e].bucket_of[x][m];
			if (b != DConstants::INVALID_INDEX) {
				stake += bucket_stake(e, x, b);
			}
		}
		return stake;
	};
	// Sides joined through keyed edges reduce one another: a unary death empties buckets, and a pending member
	// may still die from a verdict in flight somewhere in its bucket chain -- a partner of its own bucket, or a
	// member a link side relays from further away. Such a member WAITS for those verdicts rather than being
	// asked while the cascade is about to spare it (the semantic counterpart of running the semi-join reduction
	// before the predicate). The wait is per member: buckets with nothing in flight keep flowing, so one slow
	// request holds back only the members its outcome can reach. Memoized per (edge, axis, bucket) per round,
	// the same slots as the stake.
	std::unordered_map<idx_t, char> bucket_exposed_memo;
	std::function<bool(idx_t, int, idx_t)> bucket_exposed = [&](idx_t e, int x, idx_t b) -> bool {
		const idx_t memo_key = memo_base[e] + static_cast<idx_t>(x) * buckets.edges[e].live_in[0].size() + b;
		auto it = bucket_exposed_memo.find(memo_key);
		if (it != bucket_exposed_memo.end()) {
			return it->second != 0;
		}
		bucket_exposed_memo.emplace(memo_key, 0); // provisional: the forest has no cycle to come back through
		bool exposed = false;
		const idx_t o = side_of_axis(e, 1 - x);
		for (const auto q : buckets.edges[e].members[1 - x][b]) {
			if (!live[o][q]) {
				continue;
			}
			if (uval[o][q] == 3) {
				exposed = true;
				break;
			}
			for (const auto e2 : incident[o]) {
				if (e2 == e || !plan.edges[e2].Keyed()) {
					continue;
				}
				const int x2 = axis_of(e2, o);
				const idx_t b2 = buckets.edges[e2].bucket_of[x2][q];
				if (b2 != DConstants::INVALID_INDEX && bucket_exposed(e2, x2, b2)) {
					exposed = true;
					break;
				}
			}
			if (exposed) {
				break;
			}
		}
		bucket_exposed_memo[memo_key] = exposed ? 1 : 0;
		return exposed;
	};
	auto member_exposed = [&](idx_t s, idx_t m) {
		for (const auto e : incident[s]) {
			if (!plan.edges[e].Keyed()) {
				continue;
			}
			const int x = axis_of(e, s);
			const idx_t b = buckets.edges[e].bucket_of[x][m];
			if (b != DConstants::INVALID_INDEX && bucket_exposed(e, x, b)) {
				return true;
			}
		}
		return false;
	};

	// Streaming driver (see the dense scheduler): persistent pool, single-flight, dispatch marking.
	GraphUnitBaker baker(context, op, plan, sink);
	GraphStreamPool pool;
	pool.Start(context, batch_cap, query_text);
	std::unordered_map<string, vector<GraphStreamPool::Done>> parked;
	idx_t debug_submitted_before = 0; // units submitted so far (the round trace)
	const bool debug_graph = AIConfig::Get().debug_log.find("graph") != string::npos;
	std::unordered_map<string, std::chrono::steady_clock::time_point> debug_submitted_at;
	std::unordered_map<string, string> debug_prompt_head;
	auto submit_unit = [&](bool is_unary, idx_t domain, idx_t a, idx_t b) {
		idx_t expected_reuse = 0;
		if (!is_unary && baker.cache_enabled[domain]) {
			const idx_t gs = baker.CacheGroupSide(domain);
			expected_reuse = unknown(domain, axis_of(domain, gs), baker.CacheGroupMember(domain, a, b));
			expected_reuse = expected_reuse > 0 ? expected_reuse - 1 : 0; // the later reads: this call excluded
		}
		vector<string> prompts, prefixes;
		vector<char> nulls;
		string key;
		// a unary unit's `b` is the factor asked (see fval)
		baker.Unit(is_unary, domain, a, b, is_unary ? b : DConstants::INVALID_INDEX, prompts, prefixes, nulls, key);
		debug_submitted_before++;
		if (is_unary) {
			D_ASSERT(b < nf[domain] && fval[domain][a * nf[domain] + b] == 0);
			fval[domain][a * nf[domain] + b] = 3;
			uval[domain][a] = 3;
			if (fac_evals[domain][b] == 0) {
				explore_used[domain][b]++;
			}
		} else {
			A(domain, 0).inflight[a]++;
			A(domain, 1).inflight[b]++;
		}
		auto it = parked.find(key);
		if (it != parked.end()) {
			it->second.push_back(GraphStreamPool::Done {key, is_unary, domain, a, b, false});
			return;
		}
		parked.emplace(key, vector<GraphStreamPool::Done> {});
		if (debug_graph) {
			debug_submitted_at[key] = std::chrono::steady_clock::now();
			debug_prompt_head[key] = prompts.empty() ? string() : prompts[0].substr(0, 160);
		}
		pool.Submit(GraphStreamPool::Task {baker.SubNode(is_unary, domain, b), std::move(prompts), std::move(prefixes),
		                                   std::move(nulls), std::move(key), is_unary, domain, a, b, expected_reuse});
	};
	auto apply_value = [&](bool is_unary, idx_t domain, idx_t a, idx_t b, bool value) {
		if (is_unary) {
			fval[domain][a * nf[domain] + b] = static_cast<char>(value ? 1 : 2);
			fac_evals[domain][b]++;
			fac_passes[domain][b] += value ? 1 : 0;
			if (!value) {
				uval[domain][a] = 2; // one false conjunct: the member fails, its other conjuncts are never asked
				kill(domain, a);
				return;
			}
			bool all_true = true;
			for (idx_t f = 0; f < nf[domain] && all_true; f++) {
				all_true = fval[domain][a * nf[domain] + f] == 1;
			}
			uval[domain][a] = all_true ? 1 : 0; // every conjunct true, or back to pending for the next one
			return;
		}
		auto &as = A(domain, 0);
		auto &at = A(domain, 1);
		if (as.inflight[a] > 0) {
			as.inflight[a]--;
		}
		if (at.inflight[b] > 0) {
			at.inflight[b]--;
		}
		ed_evals[domain]++;
		ed_passes[domain] += value ? 1 : 0;
		as.evals[a]++;
		at.evals[b]++;
		if (value) {
			as.pass_any[a]++;
			at.pass_any[b]++;
		}
		const idx_t s = plan.edges[domain].s, t = plan.edges[domain].t;
		if (!live[s][a] || !live[t][b]) {
			return; // a dead endpoint: the pair can neither keep anyone alive nor be emitted
		}
		if (value) {
			as.tl[a]++;
			at.tl[b]++;
			as.ptrue[a].push_back(NumericCast<uint32_t>(b));
			at.ptrue[b].push_back(NumericCast<uint32_t>(a));
		} else {
			as.fl[a]++;
			at.fl[b]++;
			as.pfalse[a].push_back(NumericCast<uint32_t>(b));
			at.pfalse[b].push_back(NumericCast<uint32_t>(a));
			touched.emplace_back(s, a); // judged by the cascade, once every queued death has settled
			touched.emplace_back(t, b);
		}
	};
	// pairs an earlier term already proved for every tuple through them: TRUE without a call (they also
	// enter the pass-rate statistics, which only biases the order a little toward asking them later)
	for (idx_t e = 0; e < ne; e++) {
		for (const auto &pr : pre_true[e]) {
			apply_value(false, e, pr.first, pr.second, true);
		}
	}
	// Judge every member once before any dispatch: a member with no live partner on some edge -- a key no
	// member of the other side carries (a side the relational reduction below did not shrink to the join's
	// keys), or a NULL key -- is in no tuple and is never asked. The cascade judges members only when an
	// event touches them, so every side is marked shrunk here.
	side_shrunk.assign(k1, 1);
	cascade();
	refresh_existential();
	auto harvest = [&](bool blocking) {
		vector<GraphStreamPool::Done> dones;
		pool.Drain(dones, blocking);
		for (auto &done : dones) {
			if (debug_graph) {
				auto at = debug_submitted_at.find(done.key);
				if (at != debug_submitted_at.end()) {
					const double secs =
					    std::chrono::duration<double>(std::chrono::steady_clock::now() - at->second).count();
					if (secs > 10.0) {
						string text = debug_prompt_head[done.key];
						for (auto &c : text) {
							c = c == '\n' ? ' ' : c;
						}
						fprintf(stderr, "[graph] slow call %.1fs side=%llu factor=%llu value=%d prompt=\"%s\"\n", secs,
						        (unsigned long long)done.domain, (unsigned long long)done.b, (int)done.value,
						        text.c_str());
					}
					debug_submitted_at.erase(at);
				}
			}
			apply_value(done.is_unary, done.domain, done.a, done.b, done.value);
			auto it = parked.find(done.key);
			if (it != parked.end()) {
				for (auto &waiter : it->second) {
					apply_value(waiter.is_unary, waiter.domain, waiter.a, waiter.b, done.value);
				}
				parked.erase(it);
			}
		}
		if (!dones.empty()) {
			pool.MarkApplied(dones.size());
			cascade();
			refresh_existential();
		}
		return !dones.empty();
	};

	// The next child partner of parent member pm on edge e that is live and unary-confirmed, or
	// false when the cursor stands at a child whose verdict is still pending (asked for) or when
	// the child side is exhausted. Dead children are stepped over.
	auto next_partner = [&](idx_t e, idx_t pm, idx_t &j_out) {
		const idx_t cs = child_of[e];
		const bool s_is_parent = parent_of[e] == plan.edges[e].s;
		const idx_t nt_edge = sink.sides[plan.edges[e].t].reps.size();
		// the parent's partner domain: its key bucket on the child side when the edge is keyed, else the side
		const vector<idx_t> *domain =
		    plan.edges[e].Keyed() ? &bucket_partners(e, axis_of(e, parent_of[e]), pm) : nullptr;
		const idx_t nt = domain ? domain->size() : sink.sides[cs].reps.size();
		auto &cur = cursor[e][pm];
		const idx_t start = nt == 0 ? 0 : (pm * 7919) % nt;
		while (cur < nt) {
			const idx_t j = domain ? (*domain)[(start + cur) % nt] : (start + cur) % nt;
			if (!live[cs][j]) {
				cur++;
				continue;
			}
			if (!pre_known[e].empty() && pre_known[e].count(s_is_parent ? pm * nt_edge + j : j * nt_edge + pm)) {
				cur++; // proved by an earlier term
				continue;
			}
			if (uval[cs][j] != 1) {
				if (uval[cs][j] == 0 && !wanted_flag[cs][j]) {
					wanted_flag[cs][j] = 1;
					wanted[cs].push_back(j);
				}
				return false; // wait for this child's verdict; the cursor stays on it
			}
			cur++;
			j_out = j;
			return true;
		}
		return false;
	};

	// Cold-phase exploration: a conjunct nobody has sampled yet scores a failure probability of 1.0 so that it is asked
	// before the conjuncts are compared -- with the plain Laplace prior an untouched conjunct loses to any observed
	// rate below one half and the first one would be asked to exhaustion before the next is tried (a one-to-one join
	// of a 28%-pass side and a 5%-pass side: 470 calls that way, 351 with the selective side first). The sample is an
	// even share of ONE slot-round (the cold slice a per-row filter pays) per conjunct: once a conjunct's share is in
	// flight it waits for its verdicts at the prior instead of drawing more blind (a cold round per conjunct would
	// cost a slice per conjunct). A per-graph pool drawn highest stake first was measured neutral on SWAN 2.0
	// (+0.3% calls) and dropped.
	auto factor_fail = [&](idx_t s, idx_t f) {
		if (fac_evals[s][f] > 0) {
			return 1.0 - est(fac_passes[s][f], fac_evals[s][f]);
		}
		return explore_used[s][f] < explore_quota ? 1.0 : 1.0 - est(0, 0);
	};
	// An unanswered factor whose share is spent: its members wait for the verdicts in flight instead of being asked
	// blind (never while nothing is in flight, so the loop cannot stall).
	auto factor_waits = [&](idx_t s, idx_t f) {
		return fac_evals[s][f] == 0 && explore_used[s][f] >= explore_quota && pool.Pending() > 0;
	};
	// The conjunct to ask a pending member next: its most selective pending factor that is not waiting for its
	// exploration samples (first by index on a tie); a waiting factor only when every pending factor waits, and then
	// the member itself waits (member_waits) -- it is neither scored nor submitted, so a side whose runnable work is
	// exhausted cannot win the selection and idle the slots another side could use.
	auto next_factor = [&](idx_t s, idx_t m) {
		idx_t best = DConstants::INVALID_INDEX;
		double best_fail = -1;
		bool best_waits = true;
		for (idx_t f = 0; f < nf[s]; f++) {
			if (fval[s][m * nf[s] + f] != 0) {
				continue;
			}
			const bool waits = factor_waits(s, f);
			const double fail = factor_fail(s, f);
			if (best == DConstants::INVALID_INDEX || (best_waits && !waits) ||
			    (waits == best_waits && fail > best_fail)) {
				best = f;
				best_fail = fail;
				best_waits = waits;
			}
		}
		D_ASSERT(best != DConstants::INVALID_INDEX); // a pending member (uval 0) has a pending factor
		return best;
	};
	auto member_waits = [&](idx_t s, idx_t m) {
		return factor_waits(s, next_factor(s, m));
	};
	// A pending member's unary score: P(its next conjunct fails) x the work its death would spare, + 1.
	auto member_score = [&](idx_t s, idx_t m) {
		const idx_t f = next_factor(s, m);
		return factor_fail(s, f) * unary_stake(s, m) + 1.0;
	};

	// Frontier loop: score members, submit up to the free slots from the best domain, harvest.
	const auto loop_start = std::chrono::steady_clock::now();
	idx_t debug_round = 0;
	for (;;) {
		if (term_limit >= 0 && confirmed_at_least(term_limit)) {
			break;
		}
		harvest(false);
		const idx_t pending_now = pool.Pending();
		if (debug_graph) {
			const auto ms =
			    std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - loop_start)
			        .count();
			fprintf(stderr, "[graph] round %llu t=%lldms pending=%llu submitted_so_far=%llu\n",
			        (unsigned long long)debug_round++, (long long)ms, (unsigned long long)pending_now,
			        (unsigned long long)debug_submitted_before);
		}
		if (pending_now >= batch_cap) {
			harvest(true);
			continue;
		}
		const idx_t slots = batch_cap - pending_now;
		bucket_stake_memo.clear(); // the stakes and exposures change with every verdict applied
		bucket_exposed_memo.clear();
		// unary candidates: expected pruning payoff of the side's pending members, each at its next conjunct.
		// Over keyed edges the sides prune one another, so the order of the conjuncts across the sides is the
		// order of the conjuncts of a per-row filter: the most selective first spares the others their calls.
		double best_unary_score = -1;
		idx_t best_unary_side = k1;
		for (idx_t s = 0; s < k1; s++) {
			if (plan.unary_leaves[s].empty()) {
				continue;
			}
			for (idx_t m = 0; m < uval[s].size(); m++) {
				if (!live[s][m] || uval[s][m] != 0 || member_exposed(s, m) || member_waits(s, m)) {
					continue; // pending, not waiting on a verdict that could spare it, not waiting for samples
				}
				const double score = member_score(s, m);
				if (score > best_unary_score) {
					best_unary_score = score;
					best_unary_side = s;
				}
			}
		}
		// pair candidates: per edge, the ready parent members (supported, this edge open for them,
		// unknown pairs left); the child's share of the score is the edge's mean child credit
		vector<double> edge_best(ne, -1.0);
		vector<double> child_credit(ne, 0.0);
		vector<vector<std::pair<double, idx_t>>> ready(ne);
		for (idx_t e = 0; e < ne; e++) {
			const idx_t ps = parent_of[e], cs = child_of[e];
			const int px = axis_of(e, ps), cx = 1 - px;
			double credit = 0;
			idx_t nc = 0;
			for (idx_t j = 0; j < live[cs].size(); j++) {
				if (live[cs][j] && uval[cs][j] == 1 && unknown(e, cx, j) > 0) {
					credit += kill_credit(e, cs, cx, j);
					nc++;
				}
			}
			child_credit[e] = nc ? credit / static_cast<double>(nc) : 0.0;
			const double child_est = est(ed_passes[e], ed_evals[e]);
			for (idx_t pm = 0; pm < live[ps].size(); pm++) {
				if (!live[ps][pm] || uval[ps][pm] != 1 || unknown(e, px, pm) == 0) {
					continue;
				}
				if (!parent_ok(e, pm) || open_child_edge(ps, pm) != e) {
					continue;
				}
				const double p_fail = 1.0 - 0.5 * (member_est(e, px, pm) + child_est);
				const double score = p_fail * (0.1 + kill_credit(e, ps, px, pm) + child_credit[e]);
				ready[e].emplace_back(-score, pm);
				edge_best[e] = MaxValue<double>(edge_best[e], score);
			}
		}
		idx_t best_pair_edge = ne;
		for (idx_t e = 0; e < ne; e++) {
			if (edge_best[e] >= 0 && (best_pair_edge == ne || edge_best[e] > edge_best[best_pair_edge])) {
				best_pair_edge = e;
			}
		}
		// The verdicts waiting parents asked for: submitted up to `room` of them, the rest stay queued.
		auto dispatch_wanted = [&](idx_t side, idx_t room) {
			idx_t n = 0;
			vector<idx_t> keep;
			for (const auto m : wanted[side]) {
				if (!live[side][m] || uval[side][m] != 0) {
					wanted_flag[side][m] = 0; // dead, or asked by another path meanwhile
					continue;
				}
				if (n < room) {
					wanted_flag[side][m] = 0;
					submit_unit(true, side, m, next_factor(side, m));
					n++;
				} else {
					keep.push_back(m);
				}
			}
			wanted[side] = std::move(keep);
			return n;
		};
		auto dispatch_unary = [&](idx_t side) {
			// members a waiting parent asked for go first, then the highest-impact pending members
			idx_t submitted = dispatch_wanted(side, slots);
			vector<std::pair<double, idx_t>> scored;
			for (idx_t m = 0; m < uval[side].size(); m++) {
				if (live[side][m] && uval[side][m] == 0 && !member_exposed(side, m) && !member_waits(side, m)) {
					scored.emplace_back(-member_score(side, m), m);
				}
			}
			std::sort(scored.begin(), scored.end());
			for (idx_t u = 0; u < scored.size() && submitted < slots; u++) {
				// re-chosen per submission: the pool draw a submission spends can move the next member on to
				// another conjunct, or make it wait
				const idx_t f = next_factor(side, scored[u].second);
				if (factor_waits(side, f)) {
					continue;
				}
				submit_unit(true, side, scored[u].second, f);
				submitted++;
			}
			return submitted;
		};
		if (best_unary_side < k1 && (best_pair_edge == ne || best_unary_score >= edge_best[best_pair_edge])) {
			if (dispatch_unary(best_unary_side) == 0 && pool.Pending() > 0) {
				harvest(true); // every candidate waits on a verdict in flight: let it land
				continue;
			}
			harvest(false);
			continue;
		}
		// verdicts that waiting parents asked for take the first slots of every round, so a parent
		// stalled on a pending child is unblocked while other edges keep the pool busy
		idx_t submitted = 0;
		for (idx_t s = 0; s < k1 && submitted < slots; s++) {
			if (!wanted[s].empty()) {
				submitted += dispatch_wanted(s, slots - submitted);
			}
		}
		// pairs: edges by best score; within an edge, members by score, at most 2 pairs each in the
		// first pass (keeps future frontiers and the pool full), then fill
		vector<idx_t> edges_by_score;
		for (idx_t e = 0; e < ne; e++) {
			if (edge_best[e] >= 0) {
				edges_by_score.push_back(e);
			}
		}
		std::sort(edges_by_score.begin(), edges_by_score.end(),
		          [&](idx_t a, idx_t b) { return edge_best[a] > edge_best[b]; });
		for (const auto e : edges_by_score) {
			if (submitted >= slots) {
				break;
			}
			auto &members = ready[e];
			std::sort(members.begin(), members.end());
			for (idx_t pass = 0; pass < 2 && submitted < slots; pass++) {
				const idx_t per_member = pass == 0 ? 2 : slots;
				for (auto &cand : members) {
					if (submitted >= slots) {
						break;
					}
					const idx_t pm = cand.second;
					for (idx_t n = 0; n < per_member && submitted < slots; n++) {
						idx_t j;
						if (!next_partner(e, pm, j)) {
							break;
						}
						const bool s_is_parent = parent_of[e] == plan.edges[e].s;
						submit_unit(false, e, s_is_parent ? pm : j, s_is_parent ? j : pm);
						submitted++;
					}
				}
			}
		}
		if (submitted == 0 && best_unary_side < k1) {
			// A ready parent counts its unconfirmed partners as unknown, so an edge can win the score
			// contest with nothing dispatchable yet: ask the pending unary members instead of stalling.
			submitted = dispatch_unary(best_unary_side);
		}
		if (submitted == 0) {
			if (pool.Pending() > 0) {
				harvest(true); // nothing selectable until in-flight results land
				continue;
			}
			break; // no dispatchable work left: every needed prompt is resolved
		}
		harvest(false);
	}
	if (AIConfig::Get().debug_log.find("graph") != string::npos) {
		for (idx_t s = 0; s < k1; s++) {
			idx_t u0 = 0, u1 = 0, u2 = 0, u3 = 0;
			for (idx_t m = 0; m < uval[s].size(); m++) {
				(uval[s][m] == 0 ? u0 : uval[s][m] == 1 ? u1 : uval[s][m] == 2 ? u2 : u3)++;
			}
			fprintf(stderr,
			        "[graph] exit side%llu live=%llu/%llu uval pending=%llu true=%llu false=%llu inflight=%llu\n",
			        (unsigned long long)s, (unsigned long long)live_count[s], (unsigned long long)live[s].size(),
			        (unsigned long long)u0, (unsigned long long)u1, (unsigned long long)u2, (unsigned long long)u3);
		}
		for (idx_t e = 0; e < ne; e++) {
			idx_t tl = 0, fl = 0, inf = 0, unk = 0, pany = 0, ready_n = 0;
			const idx_t ps = parent_of[e];
			const int px = axis_of(e, ps);
			for (idx_t m = 0; m < live[ps].size(); m++) {
				tl += A(e, px).tl[m];
				fl += A(e, px).fl[m];
				inf += A(e, px).inflight[m];
				pany += A(e, px).pass_any[m];
				if (live[ps][m] && uval[ps][m] == 1) {
					unk += unknown(e, px, m);
					ready_n += parent_ok(e, m) && open_child_edge(ps, m) == e && unknown(e, px, m) > 0;
				}
			}
			fprintf(stderr,
			        "[graph] exit edge%llu parent=side%llu tl=%llu fl=%llu inflight=%llu pass_any=%llu unknown=%llu "
			        "ready=%llu pending_pool=%llu\n",
			        (unsigned long long)e, (unsigned long long)ps, (unsigned long long)tl, (unsigned long long)fl,
			        (unsigned long long)inf, (unsigned long long)pany, (unsigned long long)unk,
			        (unsigned long long)ready_n, (unsigned long long)pool.Pending());
		}
	}
	pool.Shutdown();

	for (idx_t s = 0; s < k1; s++) {
		if (plan.unary_leaves[s].empty()) {
			continue;
		}
		for (idx_t m = 0; m < uval[s].size(); m++) {
			if (uval[s][m] != 1) {
				live[s][m] = 0; // unevaluated or false unary: never emit (matters under LIMIT early-stop)
			}
		}
	}
	for (idx_t e = 0; e < ne; e++) {
		auto &edge = plan.edges[e];
		if (exact_only(e)) {
			continue; // its pairs are never listed: the enumeration reads them off the key buckets
		}
		auto &as = A(e, 0);
		for (idx_t i = 0; i < as.ptrue.size(); i++) {
			if (!live[edge.s][i]) {
				continue;
			}
			for (const auto j : as.ptrue[i]) {
				if (live[edge.t][j]) {
					edge_pass[e].emplace_back(i, j);
				}
			}
		}
	}
}

//! One term: a conjunctive factor graph over the sides, from the members still `live` (an earlier term may
//! have proved some entirely) and the `pre_true` pairs per edge (likewise). Appends its surviving tuples.
static void RunTerm(ClientContext &context, const PhysicalAIFactorGraph &op,
                    const PhysicalAIFactorGraph::TermPlan &plan, int64_t term_limit, FactorGraphSinkState &sink,
                    const string &query_text, vector<vector<char>> &live,
                    const vector<vector<std::pair<idx_t, idx_t>>> &pre_true, vector<vector<idx_t>> &assignments) {
	const idx_t k = op.side_widths.size();
	const idx_t total_width = op.side_offsets.back() + op.side_widths.back();
	auto &fn = op.node->Cast<BoundFunctionExpression>();

	auto live_count = [&](idx_t s) {
		idx_t n = 0;
		for (const auto f : live[s]) {
			n += f;
		}
		return n;
	};
	const idx_t ne = plan.edges.size();
	vector<vector<std::pair<idx_t, idx_t>>> edge_pass(ne);
	bool lazy_done = false;
	// Keyed edges (an absorbed equi-join) exist only in the sparse scheduler's bucket form; the rewrite admits them
	// under that scheduler and rejects cycles, so the dense and the staged schedules never see one.
	const bool keyed = !op.exact_edges.empty();
	GraphKeyBuckets buckets(plan, sink, live);
	{
		vector<idx_t> parent_edge_of_side, side_level;
		if (OrientForest(op, plan, sink, parent_edge_of_side, side_level)) {
			if (keyed || AIVarcharSetting(context, "ai_factor_state", "dense") == "sparse") {
				SparseLazyFactorGraphEvaluation(context, op, plan, term_limit, sink, query_text, parent_edge_of_side,
				                                side_level, live, edge_pass, pre_true, buckets);
			} else {
				LazyFactorGraphEvaluation(context, op, plan, term_limit, sink, query_text, parent_edge_of_side,
				                          side_level, live, edge_pass, pre_true);
			}
			lazy_done = true;
		}
	}
	if (!lazy_done && keyed) {
		throw InternalException(
		    "AIFactorGraph: a keyed graph has no staged schedule (rewrite gate should have caught it)");
	}
	if (!lazy_done) {
		// 1. Unary pre-pass: side predicates over member domains; a false member dies before any
		//    tuple containing it exists.
		for (idx_t s = 0; s < k; s++) {
			if (plan.unary_leaves[s].empty() || sink.sides[s].reps.empty()) {
				continue;
			}
			vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
			for (idx_t c = 0; c < op.side_widths[s]; c++) {
				index_map[op.side_offsets[s] + c] = c;
			}
			auto sub_node = AIFactorSubNode(fn, plan.unary_leaves[s], index_map, plan.unary_trees[s]);
			vector<LogicalType> chunk_types(op.types.begin() + NumericCast<int64_t>(op.side_offsets[s]),
			                                op.types.begin() +
			                                    NumericCast<int64_t>(op.side_offsets[s] + op.side_widths[s]));
			chunk_types.push_back(LogicalType::BIGINT); // trailing __count
			auto &side = sink.sides[s];
			vector<char> pass;
			EvaluateDomain(
			    context, *sub_node, chunk_types, side.reps.size(), query_text,
			    [&](DataChunk &chunk, idx_t j, idx_t row) {
				    for (idx_t c = 0; c < op.side_widths[s]; c++) {
					    chunk.data[c].SetValue(j, side.reps[row][c]);
				    }
				    chunk.data[op.side_widths[s]].SetValue(j, Value::BIGINT(NumericCast<int64_t>(side.counts[row])));
			    },
			    pass);
			for (idx_t r = 0; r < side.reps.size(); r++) {
				live[s][r] = pass[r];
			}
		}

		// 2. Edges over surviving pair domains, smallest live domain first; cascade deletions to
		//    fixed point after each edge so later domains shrink further.
		vector<char> edge_done(ne, 0);
		for (idx_t round = 0; round < ne; round++) {
			idx_t best = ne;
			idx_t best_size = 0;
			for (idx_t e = 0; e < ne; e++) {
				if (edge_done[e]) {
					continue;
				}
				const idx_t size = live_count(plan.edges[e].s) * live_count(plan.edges[e].t);
				if (best == ne || size < best_size) {
					best = e;
					best_size = size;
				}
			}
			auto &edge = plan.edges[best];
			edge_done[best] = 1;
			std::unordered_set<idx_t> known_true;
			const idx_t nt_edge = sink.sides[edge.t].reps.size();
			for (const auto &pr : pre_true[best]) {
				known_true.insert(pr.first * nt_edge + pr.second);
				edge_pass[best].push_back(pr);
			}
			vector<std::pair<idx_t, idx_t>> pairs;
			for (idx_t i = 0; i < sink.sides[edge.s].reps.size(); i++) {
				if (!live[edge.s][i]) {
					continue;
				}
				for (idx_t j = 0; j < sink.sides[edge.t].reps.size(); j++) {
					if (live[edge.t][j] && !known_true.count(i * nt_edge + j)) {
						pairs.emplace_back(i, j);
					}
				}
			}
			vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
			for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
				index_map[op.side_offsets[edge.s] + c] = c;
			}
			for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
				index_map[op.side_offsets[edge.t] + c] = op.side_widths[edge.s] + c;
			}
			auto sub_node = AIFactorSubNode(fn, edge.leaf_ids, index_map, edge.tree);
			vector<LogicalType> chunk_types(op.types.begin() + NumericCast<int64_t>(op.side_offsets[edge.s]),
			                                op.types.begin() +
			                                    NumericCast<int64_t>(op.side_offsets[edge.s] + op.side_widths[edge.s]));
			for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
				chunk_types.push_back(op.types[op.side_offsets[edge.t] + c]);
			}
			chunk_types.push_back(LogicalType::BIGINT);
			auto &s_side = sink.sides[edge.s];
			auto &t_side = sink.sides[edge.t];
			vector<char> pass;
			EvaluateDomain(
			    context, *sub_node, chunk_types, pairs.size(), query_text,
			    [&](DataChunk &chunk, idx_t j, idx_t row) {
				    auto &pair = pairs[row];
				    for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
					    chunk.data[c].SetValue(j, s_side.reps[pair.first][c]);
				    }
				    for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
					    chunk.data[op.side_widths[edge.s] + c].SetValue(j, t_side.reps[pair.second][c]);
				    }
				    chunk.data[op.side_widths[edge.s] + op.side_widths[edge.t]].SetValue(j, Value::BIGINT(1));
			    },
			    pass);
			for (idx_t p = 0; p < pairs.size(); p++) {
				if (pass[p]) {
					edge_pass[best].push_back(pairs[p]);
				}
			}
			// Backward cascade: a live member with no passing live partner on an evaluated incident
			// edge cannot appear in any surviving tuple -- delete it; repeat to fixed point.
			bool changed = true;
			while (changed) {
				changed = false;
				for (idx_t e = 0; e < ne; e++) {
					if (!edge_done[e]) {
						continue;
					}
					auto &ed = plan.edges[e];
					vector<char> has_s(sink.sides[ed.s].reps.size(), 0);
					vector<char> has_t(sink.sides[ed.t].reps.size(), 0);
					for (auto &pair : edge_pass[e]) {
						if (live[ed.s][pair.first] && live[ed.t][pair.second]) {
							has_s[pair.first] = 1;
							has_t[pair.second] = 1;
						}
					}
					for (idx_t r = 0; r < has_s.size(); r++) {
						if (live[ed.s][r] && !has_s[r]) {
							live[ed.s][r] = 0;
							changed = true;
						}
					}
					for (idx_t r = 0; r < has_t.size(); r++) {
						if (live[ed.t][r] && !has_t[r]) {
							live[ed.t][r] = 0;
							changed = true;
						}
					}
				}
			}
		}
	}

	// 3. Enumerate surviving tuples: DFS in side order. A side's candidates come from one already-assigned
	//    neighbor's partner list -- its key bucket on an exact edge (those pairs were never listed), its
	//    confirmed-true partners on a semantic edge -- and the other assigned neighbors are checked by lookup,
	//    so the walk is linear in what it emits rather than in the sides' sizes; only a side with no assigned
	//    neighbor (the first side of a component) is scanned. A pushed LIMIT ends the walk once the confirmed
	//    multiplicity covers it: the LIMIT above takes k rows and stops the pipeline, so nothing beyond k rows'
	//    worth of tuples is ever built, however wide the join.
	// [e][0]: s_rep -> t_reps, [1]: t_rep -> s_reps; every list sorted, so membership is a binary search
	vector<vector<std::unordered_map<idx_t, vector<idx_t>>>> adj(ne);
	for (idx_t e = 0; e < ne; e++) {
		adj[e].resize(2);
		for (auto &pair : edge_pass[e]) {
			if (live[plan.edges[e].s][pair.first] && live[plan.edges[e].t][pair.second]) {
				adj[e][0][pair.first].push_back(pair.second);
				adj[e][1][pair.second].push_back(pair.first);
			}
		}
		for (int x = 0; x < 2; x++) {
			for (auto &entry : adj[e][x]) {
				std::sort(entry.second.begin(), entry.second.end());
			}
		}
	}
	auto adjacent = [&](idx_t e, int x, idx_t from, idx_t to) {
		auto it = adj[e][x].find(from);
		return it != adj[e][x].end() && std::binary_search(it->second.begin(), it->second.end(), to);
	};
	auto exact_only = [&](idx_t e) {
		return plan.edges[e].Keyed() && !plan.edges[e].Semantic();
	};
	// Side order of the walk: each side after the first of its component is joined to an already-assigned one,
	// so its candidates always come from a partner list (in input order a bushy forest can place a side before
	// any neighbor it is joined to, and that side would be scanned once per partial assignment). A component
	// starts at its fewest live members and extends to the adjacent side with the fewest.
	vector<idx_t> order;
	vector<char> visited(k, 0);
	while (order.size() < k) {
		idx_t pick = k;
		bool pick_adjacent = false;
		for (idx_t s = 0; s < k; s++) {
			if (visited[s]) {
				continue;
			}
			bool adjacent = false;
			for (idx_t e = 0; e < ne && !adjacent; e++) {
				auto &ed = plan.edges[e];
				adjacent = (ed.s == s && visited[ed.t]) || (ed.t == s && visited[ed.s]);
			}
			if (pick == k || (adjacent && !pick_adjacent) ||
			    (adjacent == pick_adjacent && live_count(s) < live_count(pick))) {
				pick = s;
				pick_adjacent = adjacent;
			}
		}
		visited[pick] = 1;
		order.push_back(pick);
	}
	vector<idx_t> pos_of(k);
	for (idx_t i = 0; i < k; i++) {
		pos_of[order[i]] = i;
	}
	vector<idx_t> assignment(k);
	int64_t enumerated_rows = 0;
	bool enough = false;
	std::function<void(idx_t)> enumerate = [&](idx_t pos) {
		if (pos == k) {
			assignments.push_back(assignment);
			if (term_limit >= 0) {
				enumerated_rows = SaturatingAdd(enumerated_rows, TupleMultiplicity(sink, assignment));
				enough = enumerated_rows >= term_limit;
			}
			return;
		}
		// candidates: an assigned neighbor's partner list (exact edge: its bucket; semantic edge: its confirmed
		// partners, none when it has no entry), else every member of the side
		const idx_t side = order[pos];
		auto assigned = [&](idx_t other) {
			return pos_of[other] < pos;
		};
		static const vector<idx_t> no_candidates;
		const vector<idx_t> *domain = nullptr;
		idx_t domain_edge = ne; // the edge the candidates came from: already satisfied, not checked again
		for (idx_t e = 0; e < ne && !domain; e++) {
			auto &ed = plan.edges[e];
			const bool s_assigned = ed.t == side && assigned(ed.s);
			const bool t_assigned = ed.s == side && assigned(ed.t);
			if (!s_assigned && !t_assigned) {
				continue;
			}
			domain_edge = e;
			if (exact_only(e)) {
				domain =
				    s_assigned ? &buckets.Partners(e, 0, assignment[ed.s]) : &buckets.Partners(e, 1, assignment[ed.t]);
			} else {
				auto &by_member = s_assigned ? adj[e][0] : adj[e][1];
				auto it = by_member.find(s_assigned ? assignment[ed.s] : assignment[ed.t]);
				domain = it == by_member.end() ? &no_candidates : &it->second;
			}
		}
		const idx_t n = domain ? domain->size() : sink.sides[side].reps.size();
		for (idx_t ri = 0; ri < n && !enough; ri++) {
			const idx_t r = domain ? (*domain)[ri] : ri;
			if (!live[side][r]) {
				continue;
			}
			bool ok = true;
			for (idx_t e = 0; e < ne && ok; e++) {
				if (e == domain_edge) {
					continue;
				}
				auto &ed = plan.edges[e];
				// the other edges whose far endpoint is already assigned: bucket or sorted-list lookup
				if (exact_only(e)) {
					if (ed.s == side && assigned(ed.t)) {
						ok = buckets.SameBucket(e, r, assignment[ed.t]);
					} else if (ed.t == side && assigned(ed.s)) {
						ok = buckets.SameBucket(e, assignment[ed.s], r);
					}
				} else if (ed.s == side && assigned(ed.t)) {
					ok = adjacent(e, 0, r, assignment[ed.t]);
				} else if (ed.t == side && assigned(ed.s)) {
					ok = adjacent(e, 1, r, assignment[ed.s]);
				}
			}
			if (!ok) {
				continue;
			}
			assignment[side] = r;
			enumerate(pos + 1);
		}
	};
	bool any_empty = false;
	for (idx_t s = 0; s < k; s++) {
		any_empty = any_empty || live_count(s) == 0;
	}
	if (!any_empty) {
		enumerate(0);
	}
}

//! The node's terms, their results united. Before a term runs, what the union already proves is taken out
//! of its domains: a member is COVERED when every tuple through it is in the union (then nothing asked about
//! it can change a result), a pair likewise; in existential mode a member of the existential side with one
//! confirmed tuple is done. The next term to run is the one with the fewest estimated calls over what is
//! still live (a side factor: its live members; an edge factor: its live pairs, an upper bound), re-estimated
//! after every term, so a cheap term's coverage shrinks the expensive ones before they run. With several
//! terms a LIMIT stops between terms only (a term's own early stop could count tuples the union already holds).
void RunFactorGraphEvaluation(ClientContext &context, const PhysicalAIFactorGraph &op, FactorGraphSinkState &sink,
                              FactorGraphSourceState &state) {
	const string query_text = context.GetCurrentQuery();
	const idx_t k = op.side_widths.size();
	const bool multi = op.terms.size() > 1;
	std::set<vector<idx_t>> union_set;
	auto product_except = [&](const vector<idx_t> &skip) {
		idx_t n = 1;
		for (idx_t s = 0; s < k; s++) {
			if (std::find(skip.begin(), skip.end(), s) == skip.end()) {
				n = SaturatingMul(n, sink.sides[s].reps.size());
			}
		}
		return n;
	};
	auto union_rows = [&]() {
		int64_t rows = 0;
		for (auto &t : union_set) {
			rows = SaturatingAdd(rows, TupleMultiplicity(sink, t));
		}
		return rows;
	};
	// a term's domains after coverage, and its call estimate over them
	auto prepare = [&](const PhysicalAIFactorGraph::TermPlan &plan, bool covered, vector<vector<char>> &live,
	                   vector<vector<std::pair<idx_t, idx_t>>> &pre_true) {
		live.assign(k, {});
		for (idx_t s = 0; s < k; s++) {
			live[s].assign(sink.sides[s].reps.size(), 1);
		}
		pre_true.assign(plan.edges.size(), {});
		if (covered) {
			// member coverage: tuples through m already in the union == all tuples through m
			for (idx_t s = 0; s < k; s++) {
				vector<idx_t> through(sink.sides[s].reps.size(), 0);
				for (auto &t : union_set) {
					through[t[s]]++;
				}
				const idx_t all = product_except({s});
				for (idx_t m = 0; m < through.size(); m++) {
					if (through[m] == all || (s == op.existential_side && through[m] > 0)) {
						live[s][m] = 0;
					}
				}
			}
			// pair coverage per edge of this term
			for (idx_t e = 0; e < plan.edges.size(); e++) {
				auto &edge = plan.edges[e];
				const idx_t nt = sink.sides[edge.t].reps.size();
				std::unordered_map<idx_t, idx_t> through;
				for (auto &t : union_set) {
					through[t[edge.s] * nt + t[edge.t]]++;
				}
				const idx_t all = product_except({edge.s, edge.t});
				for (auto &kv : through) {
					const idx_t i = kv.first / nt, j = kv.first % nt;
					if (kv.second == all && live[edge.s][i] && live[edge.t][j]) {
						pre_true[e].emplace_back(i, j);
					}
				}
			}
		}
		vector<idx_t> live_count(k, 0);
		for (idx_t s = 0; s < k; s++) {
			for (const auto f : live[s]) {
				live_count[s] += f;
			}
		}
		double calls = 0;
		for (idx_t s = 0; s < k; s++) {
			calls += plan.unary_leaves[s].empty() ? 0.0 : static_cast<double>(live_count[s]);
		}
		for (idx_t e = 0; e < plan.edges.size(); e++) {
			if (!plan.edges[e].Semantic()) {
				continue; // an exact edge asks nothing
			}
			calls +=
			    static_cast<double>(live_count[plan.edges[e].s]) * static_cast<double>(live_count[plan.edges[e].t]) -
			    static_cast<double>(pre_true[e].size());
		}
		return calls;
	};
	vector<char> done(op.terms.size(), 0);
	for (idx_t round = 0; round < op.terms.size(); round++) {
		if (multi && op.limit >= 0 && union_rows() >= op.limit) {
			break;
		}
		// the cheapest remaining term over what is still live (ties: the planner's order)
		idx_t pick = op.terms.size();
		double pick_calls = 0;
		vector<vector<char>> live, cand_live;
		vector<vector<std::pair<idx_t, idx_t>>> pre_true, cand_pre;
		for (idx_t ti = 0; ti < op.terms.size(); ti++) {
			if (done[ti]) {
				continue;
			}
			const double calls = prepare(op.terms[ti], round > 0, cand_live, cand_pre);
			if (pick == op.terms.size() || calls < pick_calls) {
				pick = ti;
				pick_calls = calls;
				live.swap(cand_live);
				pre_true.swap(cand_pre);
			}
		}
		done[pick] = 1;
		auto &plan = op.terms[pick];
		vector<vector<idx_t>> assignments;
		RunTerm(context, op, plan, multi ? -1 : op.limit, sink, query_text, live, pre_true, assignments);
		if (!multi) {
			state.assignments = std::move(assignments);
			return;
		}
		for (auto &a : assignments) {
			union_set.insert(a);
		}
	}
	state.assignments.assign(union_set.begin(), union_set.end());
}

SourceResultType PhysicalAIFactorGraph::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                        OperatorSourceInput &input) const {
	auto &state = input.global_state.Cast<FactorGraphSourceState>();
	auto &sink = sink_state->Cast<FactorGraphSinkState>();
	if (!state.evaluated) {
		RunFactorGraphEvaluation(context.client, *this, sink, state);
		state.evaluated = true;
	}
	const idx_t k = side_widths.size();
	const idx_t result_col = types.size() - 1;
	chunk.SetChildCardinality(STANDARD_VECTOR_SIZE);
	idx_t out = 0;
	while (state.cur_assignment < state.assignments.size() && out < STANDARD_VECTOR_SIZE) {
		auto &assignment = state.assignments[state.cur_assignment];
		// saturated (TupleMultiplicity): a wrap to 0 would emit nothing where a LIMIT expects rows
		const idx_t dup_total = static_cast<idx_t>(TupleMultiplicity(sink, assignment));
		if (state.cur_dup >= dup_total) {
			state.cur_assignment++;
			state.cur_dup = 0;
			continue;
		}
		for (idx_t s = 0; s < k; s++) {
			auto &rep = sink.sides[s].reps[assignment[s]];
			for (idx_t c = 0; c < side_widths[s]; c++) {
				chunk.data[side_offsets[s] + c].SetValue(out, rep[c]);
			}
		}
		chunk.data[result_col].SetValue(out, Value::BOOLEAN(true));
		state.cur_dup++;
		out++;
	}
	chunk.SetChildCardinality(out);
	return state.cur_assignment < state.assignments.size() ? SourceResultType::HAVE_MORE_OUTPUT
	                                                       : SourceResultType::FINISHED;
}

//===--------------------------------------------------------------------===//
// Pipeline construction: k sequential build pipelines into one sink, then source (IEJoin pattern)
//===--------------------------------------------------------------------===//
void PhysicalAIFactorGraph::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();
	auto &state = meta_pipeline.GetState();
	// becomes a source after every child fully sinks its side
	state.SetPipelineSource(current, *this);
	source_to_side.clear();
	for (idx_t i = 0; i < children.size(); i++) {
		for (auto &source : children[i].get().GetSources()) {
			source_to_side[&source.get()] = i;
		}
	}
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this);
	auto base_pipeline = child_meta_pipeline.GetBasePipeline();
	children[children.size() - 1].get().BuildPipelines(*base_pipeline, child_meta_pipeline);
	for (idx_t i = children.size() - 1; i-- > 0;) {
		auto &pipeline = child_meta_pipeline.CreatePipeline();
		children[i].get().BuildPipelines(pipeline, child_meta_pipeline);
		child_meta_pipeline.AddFinishEvent(pipeline);
	}
}

} // namespace duckdb
