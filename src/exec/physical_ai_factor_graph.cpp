#include "exec/physical_ai_factor_graph.hpp"

#include "ai_client.hpp"
#include "optimizer/ai_filter_tree_build.hpp"
#include "ai_prompt_cost.hpp"

#include "duckdb/main/client_context.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <deque>
#include <thread>
#include <unordered_map>

namespace duckdb {

PhysicalAIFactorGraph::PhysicalAIFactorGraph(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                                             unique_ptr<Expression> node_p, vector<idx_t> side_widths_p,
                                             int64_t limit_p, idx_t existential_side_p, idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types_p), estimated_cardinality),
      node(std::move(node_p)), side_widths(std::move(side_widths_p)), limit(limit_p),
      existential_side(existential_side_p) {
	side_offsets.resize(side_widths.size());
	idx_t offset = 0;
	vector<idx_t> side_of;
	for (idx_t s = 0; s < side_widths.size(); s++) {
		side_offsets[s] = offset;
		side_of.insert(side_of.end(), side_widths[s], s);
		offset += side_widths[s];
	}
	auto &fn = node->Cast<BoundFunctionExpression>();
	if (!AIFactorDecompose(fn, side_of, leaves)) {
		throw InternalException("AIFactorGraph: node is not factor-decomposable (rewrite gate should have caught it)");
	}
	unary_leaves.resize(side_widths.size());
	for (idx_t l = 0; l < leaves.size(); l++) {
		auto &sides = leaves[l].sides;
		if (sides.size() == 1) {
			unary_leaves[sides[0]].push_back(l);
			continue;
		}
		auto it =
		    std::find_if(edges.begin(), edges.end(), [&](const Edge &e) { return e.s == sides[0] && e.t == sides[1]; });
		if (it == edges.end()) {
			edges.push_back(Edge {sides[0], sides[1], {l}});
		} else {
			it->leaf_ids.push_back(l);
		}
	}
}

InsertionOrderPreservingMap<string> PhysicalAIFactorGraph::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Sides"] = std::to_string(side_widths.size());
	string edge_str;
	for (auto &e : edges) {
		if (!edge_str.empty()) {
			edge_str += ", ";
		}
		edge_str += "S" + std::to_string(e.s) + "-S" + std::to_string(e.t);
	}
	result["Edges"] = edge_str;
	result["Mode"] = "factor graph (member/pair domains, exact backward pruning)";
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

static string FactorRowKey(DataChunk &chunk, idx_t row) {
	string key;
	for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
		const Value v = chunk.data[col].GetValue(row);
		key.push_back(v.IsNull() ? '\x00' : '\x01');
		if (!v.IsNull()) {
			key += v.ToString();
		}
		key.push_back('\x1f');
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
// Lazy (need-driven streaming) evaluation -- ai_debug_graph_eval='lazy'
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
static bool OrientForest(const PhysicalAIFactorGraph &op, const FactorGraphSinkState &sink,
                         vector<idx_t> &parent_edge_of_side, vector<idx_t> &side_level) {
	const idx_t k = op.side_widths.size();
	parent_edge_of_side.assign(k, DConstants::INVALID_INDEX);
	side_level.assign(k, 0);
	vector<vector<idx_t>> adj(k);
	for (idx_t e = 0; e < op.edges.size(); e++) {
		adj[op.edges[e].s].push_back(e);
		adj[op.edges[e].t].push_back(e);
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
		if (op.unary_leaves[x].size() != op.unary_leaves[y].size()) {
			return op.unary_leaves[x].size() > op.unary_leaves[y].size();
		}
		return sink.sides[x].reps.size() < sink.sides[y].reps.size();
	});
	vector<char> visited(k, 0);
	vector<char> edge_used(op.edges.size(), 0);
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
				const idx_t other = op.edges[e].s == s ? op.edges[e].t : op.edges[e].s;
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
					const bool value =
					    AIFactorEvalUnit(*context, task.sub_node->Cast<BoundFunctionExpression>(), task.prompts,
					                     query_text, valid, task.prefixes.empty() ? nullptr : &task.prefixes,
					                     task.expected_reuse) &&
					    valid;
					{
						std::lock_guard<std::mutex> lock(m);
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
	void Drain(vector<Done> &out, bool blocking) {
		std::unique_lock<std::mutex> lock(m);
		if (blocking) {
			done_cv.wait(lock, [&]() { return !done.empty() || unapplied == 0; });
		}
		out.insert(out.end(), std::make_move_iterator(done.begin()), std::make_move_iterator(done.end()));
		done.clear();
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
			thread.join();
		}
	}
};

static void LazyFactorGraphEvaluation(ClientContext &context, const PhysicalAIFactorGraph &op,
                                      FactorGraphSinkState &sink, const string &query_text,
                                      const vector<idx_t> &parent_edge_of_side, const vector<idx_t> &side_level,
                                      vector<vector<char>> &live, vector<vector<std::pair<idx_t, idx_t>>> &edge_pass) {
	const idx_t k1 = op.side_widths.size();
	const idx_t ne = op.edges.size();
	const idx_t total_width = op.side_offsets.back() + op.side_widths.back();
	auto &fn = op.node->Cast<BoundFunctionExpression>();
	const idx_t batch_cap = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);

	// Edge orientation: the child side is the one whose parent edge is this edge.
	vector<idx_t> child_of(ne);
	vector<idx_t> parent_of(ne);
	for (idx_t e = 0; e < ne; e++) {
		const bool s_is_child = parent_edge_of_side[op.edges[e].s] == e;
		child_of[e] = s_is_child ? op.edges[e].s : op.edges[e].t;
		parent_of[e] = s_is_child ? op.edges[e].t : op.edges[e].s;
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
		uval[s].assign(sink.sides[s].reps.size(), op.unary_leaves[s].empty() ? 1 : 0);
	}
	// Pair state per edge (row-major over edge.s x edge.t): 0 unknown, 1 true, 2 false.
	vector<vector<char>> pval(ne);
	for (idx_t e = 0; e < ne; e++) {
		pval[e].assign(sink.sides[op.edges[e].s].reps.size() * sink.sides[op.edges[e].t].reps.size(), 0);
	}
	auto pkey = [&](idx_t e, idx_t si, idx_t tj) {
		return si * sink.sides[op.edges[e].t].reps.size() + tj;
	};

	// A parent-side member supports its child edge once it has a confirmed-true pair on ITS
	// parent edge (roots support unconditionally).
	auto supported = [&](idx_t s, idx_t m) {
		const idx_t pe = parent_edge_of_side[s];
		if (pe == DConstants::INVALID_INDEX) {
			return true;
		}
		const bool m_on_s_axis = op.edges[pe].s == s;
		const idx_t nother = sink.sides[m_on_s_axis ? op.edges[pe].t : op.edges[pe].s].reps.size();
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
				auto &edge = op.edges[e];
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
			auto &edge = op.edges[e];
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
					if (!live[side][r] || (!op.unary_leaves[side].empty() && uval[side][r] != 1)) {
						continue;
					}
					bool ok = true;
					for (idx_t e = 0; e < ne && ok; e++) {
						auto &ed = op.edges[e];
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
		// usefulness: root = live unsatisfied confirmed members; downstream = confirmed-true
		// pair from a useful parent member
		for (idx_t s = 0; s < k1; s++) {
			useful[s].assign(sink.sides[s].reps.size(), 0);
		}
		for (idx_t m = 0; m < satisfied.size(); m++) {
			useful[es][m] =
			    static_cast<char>(live[es][m] && !satisfied[m] && (op.unary_leaves[es].empty() || uval[es][m] == 1));
		}
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
			auto &edge = op.edges[pe];
			const bool s_is_s = edge.s == s;
			const idx_t ps = s_is_s ? edge.t : edge.s;
			for (idx_t m = 0; m < sink.sides[s].reps.size(); m++) {
				if (!live[s][m] || (!op.unary_leaves[s].empty() && uval[s][m] != 1)) {
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

	// Sub-nodes and chunk layouts, built once per domain.
	vector<unique_ptr<Expression>> side_sub(k1);
	vector<vector<LogicalType>> side_types(k1);
	for (idx_t s = 0; s < k1; s++) {
		if (op.unary_leaves[s].empty()) {
			continue;
		}
		vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
		for (idx_t c = 0; c < op.side_widths[s]; c++) {
			index_map[op.side_offsets[s] + c] = c;
		}
		side_sub[s] = AIFactorSubNode(fn, op.unary_leaves[s], index_map);
		side_types[s].assign(op.types.begin() + NumericCast<int64_t>(op.side_offsets[s]),
		                     op.types.begin() + NumericCast<int64_t>(op.side_offsets[s] + op.side_widths[s]));
		side_types[s].push_back(LogicalType::BIGINT);
	}
	vector<unique_ptr<Expression>> edge_sub(ne);
	vector<vector<LogicalType>> edge_types(ne);
	for (idx_t e = 0; e < ne; e++) {
		auto &edge = op.edges[e];
		vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
		for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
			index_map[op.side_offsets[edge.s] + c] = c;
		}
		for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
			index_map[op.side_offsets[edge.t] + c] = op.side_widths[edge.s] + c;
		}
		edge_sub[e] = AIFactorSubNode(fn, edge.leaf_ids, index_map);
		edge_types[e].assign(op.types.begin() + NumericCast<int64_t>(op.side_offsets[edge.s]),
		                     op.types.begin() + NumericCast<int64_t>(op.side_offsets[edge.s] + op.side_widths[edge.s]));
		for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
			edge_types[e].push_back(op.types[op.side_offsets[edge.t] + c]);
		}
		edge_types[e].push_back(LogicalType::BIGINT);
	}

	if (AIConfig::Get().debug_log.find("graph") != string::npos) {
		for (idx_t e = 0; e < ne; e++) {
			fprintf(stderr, "[graph] edge%llu sides %llu-%llu parent=%llu child=%llu childlevel=%llu\n",
			        (unsigned long long)e, (unsigned long long)op.edges[e].s, (unsigned long long)op.edges[e].t,
			        (unsigned long long)parent_of[e], (unsigned long long)child_of[e],
			        (unsigned long long)side_level[child_of[e]]);
		}
		for (idx_t s = 0; s < k1; s++) {
			fprintf(stderr, "[graph] side%llu reps=%llu unary_leaves=%llu\n", (unsigned long long)s,
			        (unsigned long long)sink.sides[s].reps.size(), (unsigned long long)op.unary_leaves[s].size());
		}
	}
	// Adaptive mode: observed pass-rate statistics drive frontier priorities -- the graph-level
	// analogue of per-tuple filter reordering. Estimates are Laplace-smoothed observed rates,
	// per domain and per member (falling back to the domain rate while a member is unsampled).
	// A unit's score is its expected pruning payoff: P(fail) x the unknown pairs its member's
	// death would cancel (spread over the member's remaining pairs on that edge). Selection
	// stays strictly behind the same need gate, so adaptivity can only REDUCE calls, never add.
	const bool adaptive = AIConfig::Get().graph_eval_adaptive;
	vector<idx_t> un_evals(k1, 0), un_passes(k1, 0);
	vector<idx_t> ed_evals(ne, 0), ed_passes(ne, 0);
	vector<vector<idx_t>> me_evals_s(ne), me_passes_s(ne), me_evals_t(ne), me_passes_t(ne);
	for (idx_t e = 0; e < ne; e++) {
		me_evals_s[e].assign(sink.sides[op.edges[e].s].reps.size(), 0);
		me_passes_s[e].assign(sink.sides[op.edges[e].s].reps.size(), 0);
		me_evals_t[e].assign(sink.sides[op.edges[e].t].reps.size(), 0);
		me_passes_t[e].assign(sink.sides[op.edges[e].t].reps.size(), 0);
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
			auto &edge = op.edges[e];
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
		auto &edge = op.edges[e];
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
			const double p_pass = member_est(e, op.edges[e].s == s, m);
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
			auto &edge = op.edges[e];
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
				int64_t product = 1;
				for (idx_t s = 0; s < k1; s++) {
					product *= NumericCast<int64_t>(sink.sides[s].counts[assignment[s]]);
				}
				confirmed += product;
				return confirmed >= want;
			}
			for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
				if (!live[side][r] || (!op.unary_leaves[side].empty() && uval[side][r] != 1)) {
					continue;
				}
				bool ok = true;
				for (idx_t e = 0; e < ne && ok; e++) {
					auto &ed = op.edges[e];
					if (ed.s == side && ed.t < side) {
						auto it = adj[e][0].find(r);
						ok = it != adj[e][0].end() &&
						     std::find(it->second.begin(), it->second.end(), assignment[ed.t]) != it->second.end();
					} else if (ed.t == side && ed.s < side) {
						auto it = adj[e][1].find(r);
						ok = it != adj[e][1].end() &&
						     std::find(it->second.begin(), it->second.end(), assignment[ed.s]) != it->second.end();
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
	vector<unique_ptr<ExpressionExecutor>> side_exec(k1), edge_exec(ne);
	vector<unique_ptr<DataChunk>> side_row(k1), edge_row(ne);
	std::unordered_map<string, vector<GraphStreamPool::Done>> parked; // key -> units awaiting the winner
	// Explicit provider prompt caching for pair prompts (ai_prefix_cache): split each single-leaf
	// edge's call prompt at the first operand that reads the other side. The graph only DECLARES
	// structure: every eligible pair call carries (prefix, suffix) plus an expected_reuse hint
	// (the member's remaining live pairs). Size gate, write policy, and prime/park sequencing all
	// live in the client's prefix lifecycle (ai_client.cpp), shared by every operator.
	const bool prefix_caching = AIConfig::Get().prefix_cache;
	vector<char> cache_enabled(ne, 0);
	vector<vector<const Expression *>> cache_ops(ne);
	vector<idx_t> cache_split(ne, 0);
	vector<char> cache_group_s(ne, 1); // group axis: 1 = edge.s owns the prefix, 0 = edge.t
	if (prefix_caching) {
		for (idx_t e = 0; e < ne; e++) {
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
					(col < op.side_widths[op.edges[e].s] ? s_axis : t_axis) = true;
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
	auto cache_group_member = [&](idx_t e, idx_t a, idx_t b) {
		return cache_group_s[e] ? a : b;
	};
	vector<unique_ptr<ExpressionExecutor>> edge_ops_exec(ne);
	auto bake_split = [&](idx_t e, idx_t a, idx_t b, string &prefix, string &suffix) {
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
		auto &edge = op.edges[e];
		row->Reset();
		row->SetChildCardinality(1);
		for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
			row->data[c].SetValue(0, sink.sides[edge.s].reps[a][c]);
		}
		for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
			row->data[op.side_widths[edge.s] + c].SetValue(0, sink.sides[edge.t].reps[b][c]);
		}
		row->data[op.side_widths[edge.s] + op.side_widths[edge.t]].SetValue(0, Value::BIGINT(1));
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
	};
	auto bake = [&](bool is_unary, idx_t domain, idx_t a, idx_t b) {
		auto &sub = is_unary ? side_sub[domain] : edge_sub[domain];
		auto &fn_sub = sub->Cast<BoundFunctionExpression>();
		const idx_t nleaf = (fn_sub.GetChildren().size() - 1) / 3; // tree + 3n (+meta)
		auto &exec = is_unary ? side_exec[domain] : edge_exec[domain];
		auto &row = is_unary ? side_row[domain] : edge_row[domain];
		if (!exec) {
			vector<unique_ptr<Expression>> copies; // call prompts are children [1 .. n]
			exec = make_uniq<ExpressionExecutor>(context);
			for (idx_t l = 0; l < nleaf; l++) {
				exec->AddExpression(*fn_sub.GetChildren()[1 + l]);
			}
			row = make_uniq<DataChunk>();
			row->Initialize(BufferAllocator::Get(context), is_unary ? side_types[domain] : edge_types[domain]);
		}
		row->Reset();
		row->SetChildCardinality(1);
		if (is_unary) {
			auto &side = sink.sides[domain];
			for (idx_t c = 0; c < op.side_widths[domain]; c++) {
				row->data[c].SetValue(0, side.reps[a][c]);
			}
			row->data[op.side_widths[domain]].SetValue(0, Value::BIGINT(1));
		} else {
			auto &edge = op.edges[domain];
			for (idx_t c = 0; c < op.side_widths[edge.s]; c++) {
				row->data[c].SetValue(0, sink.sides[edge.s].reps[a][c]);
			}
			for (idx_t c = 0; c < op.side_widths[edge.t]; c++) {
				row->data[op.side_widths[edge.s] + c].SetValue(0, sink.sides[edge.t].reps[b][c]);
			}
			row->data[op.side_widths[edge.s] + op.side_widths[edge.t]].SetValue(0, Value::BIGINT(1));
		}
		DataChunk out;
		vector<LogicalType> out_types(nleaf, LogicalType::VARCHAR);
		out.Initialize(BufferAllocator::Get(context), out_types);
		exec->Execute(*row, out);
		vector<string> prompts(nleaf);
		for (idx_t l = 0; l < nleaf; l++) {
			const Value v = out.data[l].GetValue(0);
			prompts[l] = v.IsNull() ? string() : StringValue::Get(v);
		}
		return prompts;
	};
	auto submit_unit = [&](bool is_unary, idx_t domain, idx_t a, idx_t b) {
		vector<string> prompts;
		vector<string> prefixes;
		idx_t expected_reuse = 0;
		if (!is_unary && cache_enabled[domain]) {
			// Declaration only: split at the structural boundary and hand the client the known
			// fan-out (this member's remaining live pairs, this call included). Whether a
			// breakpoint is emitted -- and the write-before-reads sequencing -- is the client
			// prefix lifecycle's decision.
			const idx_t g = cache_group_member(domain, a, b);
			string prefix, suffix;
			bake_split(domain, a, b, prefix, suffix);
			expected_reuse = unknown_pairs(cache_group_s[domain] ? op.edges[domain].s : op.edges[domain].t, g,
			                               DConstants::INVALID_INDEX, domain);
			prompts.push_back(std::move(suffix));
			prefixes.push_back(std::move(prefix));
		}
		if (prompts.empty()) {
			prompts = bake(is_unary, domain, a, b);
		}
		string key;
		for (idx_t l = 0; l < prompts.size(); l++) {
			// local identity is prefix+prompt concatenated: split-invariant, so cached and plain
			// forms of the same pair share one flight and one cache entry
			if (l < prefixes.size()) {
				key += prefixes[l];
			}
			key += prompts[l];
			key += '\x1f';
		}
		key += is_unary ? "u" : "e"; // unary and edge sub-nodes may share prompt text but not meta
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
		                                   std::move(prompts), std::move(prefixes), std::move(key), is_unary, domain, a,
		                                   b, expected_reuse});
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
		if (op.limit >= 0 && confirmed_at_least(op.limit)) {
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
				if (op.unary_leaves[s].empty()) {
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
				auto &edge = op.edges[e];
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
			auto &edge = op.edges[best_pair_edge];
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
			if (op.unary_leaves[s].empty()) {
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
			auto &edge = op.edges[e];
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
		if (op.unary_leaves[s].empty()) {
			continue;
		}
		for (idx_t m = 0; m < uval[s].size(); m++) {
			if (uval[s][m] != 1) {
				live[s][m] = 0; // unevaluated or false unary: never emit (matters under LIMIT early-stop)
			}
		}
	}
	for (idx_t e = 0; e < ne; e++) {
		auto &edge = op.edges[e];
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

void RunFactorGraphEvaluation(ClientContext &context, const PhysicalAIFactorGraph &op, FactorGraphSinkState &sink,
                              FactorGraphSourceState &state) {
	const string query_text = context.GetCurrentQuery();
	const idx_t k = op.side_widths.size();
	const idx_t total_width = op.side_offsets.back() + op.side_widths.back();
	auto &fn = op.node->Cast<BoundFunctionExpression>();

	vector<vector<char>> live(k);
	for (idx_t s = 0; s < k; s++) {
		live[s].assign(sink.sides[s].reps.size(), 1);
	}

	auto live_count = [&](idx_t s) {
		idx_t n = 0;
		for (const auto f : live[s]) {
			n += f;
		}
		return n;
	};
	const idx_t ne = op.edges.size();
	vector<vector<std::pair<idx_t, idx_t>>> edge_pass(ne);
	bool lazy_done = false;
	{
		vector<idx_t> parent_edge_of_side, side_level;
		if (AIConfig::Get().graph_eval_lazy && OrientForest(op, sink, parent_edge_of_side, side_level)) {
			LazyFactorGraphEvaluation(context, op, sink, query_text, parent_edge_of_side, side_level, live, edge_pass);
			lazy_done = true;
		}
	}
	if (!lazy_done) {
		// 1. Unary pre-pass: side predicates over member domains; a false member dies before any
		//    tuple containing it exists.
		for (idx_t s = 0; s < k; s++) {
			if (op.unary_leaves[s].empty() || sink.sides[s].reps.empty()) {
				continue;
			}
			vector<idx_t> index_map(total_width, DConstants::INVALID_INDEX);
			for (idx_t c = 0; c < op.side_widths[s]; c++) {
				index_map[op.side_offsets[s] + c] = c;
			}
			auto sub_node = AIFactorSubNode(fn, op.unary_leaves[s], index_map);
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
				const idx_t size = live_count(op.edges[e].s) * live_count(op.edges[e].t);
				if (best == ne || size < best_size) {
					best = e;
					best_size = size;
				}
			}
			auto &edge = op.edges[best];
			edge_done[best] = 1;
			vector<std::pair<idx_t, idx_t>> pairs;
			for (idx_t i = 0; i < sink.sides[edge.s].reps.size(); i++) {
				if (!live[edge.s][i]) {
					continue;
				}
				for (idx_t j = 0; j < sink.sides[edge.t].reps.size(); j++) {
					if (live[edge.t][j]) {
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
			auto sub_node = AIFactorSubNode(fn, edge.leaf_ids, index_map);
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
					auto &ed = op.edges[e];
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

	// 3. Enumerate surviving tuples: DFS in side order; each new side's candidates are its live
	//    members intersected with the pass-partners of every already-assigned neighbor.
	vector<vector<std::unordered_map<idx_t, vector<idx_t>>>> adj(ne); // [e][0]: s_rep -> t_reps, [1]: t_rep -> s_reps
	for (idx_t e = 0; e < ne; e++) {
		adj[e].resize(2);
		for (auto &pair : edge_pass[e]) {
			if (live[op.edges[e].s][pair.first] && live[op.edges[e].t][pair.second]) {
				adj[e][0][pair.first].push_back(pair.second);
				adj[e][1][pair.second].push_back(pair.first);
			}
		}
	}
	vector<idx_t> assignment(k);
	std::function<void(idx_t)> enumerate = [&](idx_t side) {
		if (side == k) {
			state.assignments.push_back(assignment);
			return;
		}
		for (idx_t r = 0; r < sink.sides[side].reps.size(); r++) {
			if (!live[side][r]) {
				continue;
			}
			bool ok = true;
			for (idx_t e = 0; e < ne && ok; e++) {
				auto &ed = op.edges[e];
				// check edges whose OTHER endpoint is already assigned (< side)
				if (ed.s == side && ed.t < side) {
					auto it = adj[e][0].find(r);
					ok = it != adj[e][0].end() &&
					     std::find(it->second.begin(), it->second.end(), assignment[ed.t]) != it->second.end();
				} else if (ed.t == side && ed.s < side) {
					auto it = adj[e][1].find(r);
					ok = it != adj[e][1].end() &&
					     std::find(it->second.begin(), it->second.end(), assignment[ed.s]) != it->second.end();
				}
			}
			if (!ok) {
				continue;
			}
			assignment[side] = r;
			enumerate(side + 1);
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
		idx_t dup_total = 1;
		for (idx_t s = 0; s < k; s++) {
			dup_total *= sink.sides[s].counts[assignment[s]];
		}
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
