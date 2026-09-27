#include "exec/physical_ai_factor_join.hpp"

#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/execution/expression_executor.hpp"
#include "duckdb/parallel/meta_pipeline.hpp"
#include "duckdb/parallel/pipeline.hpp"
#include "duckdb/planner/expression/bound_reference_expression.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <unordered_map>

namespace duckdb {

//! Collect the cross-layout column indices `expr` references below `left_width` (the probe key).
static void CollectLeftKeyCols(const Expression &expr, idx_t left_width, vector<idx_t> &out) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_REF) {
		const auto idx = expr.Cast<BoundReferenceExpression>().Index();
		if (idx < left_width && std::find(out.begin(), out.end(), idx) == out.end()) {
			out.push_back(idx);
		}
		return;
	}
	ExpressionIterator::EnumerateChildren(expr,
	                                      [&](const Expression &child) { CollectLeftKeyCols(child, left_width, out); });
}

PhysicalAIFactorJoin::PhysicalAIFactorJoin(PhysicalPlan &physical_plan, vector<LogicalType> types_p,
                                           unique_ptr<Expression> predicate_p, idx_t left_width_p,
                                           idx_t estimated_cardinality)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types_p), estimated_cardinality),
      predicate(std::move(predicate_p)), left_width(left_width_p) {
	CollectLeftKeyCols(*predicate, left_width, left_key_cols);
}

InsertionOrderPreservingMap<string> PhysicalAIFactorJoin::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["Predicate"] = predicate->ToString();
	result["Mode"] = "factorized (pair domain, no cross materialization)";
	return result;
}

//===--------------------------------------------------------------------===//
// Sink: build-side dictionary of DISTINCT full rows + multiplicities
//===--------------------------------------------------------------------===//
class FactorJoinBuildState : public GlobalSinkState {
public:
	mutex lock;
	//! Distinct build rows (full row Values, emission-ready) + multiplicity per distinct row.
	vector<vector<Value>> reps;
	vector<idx_t> counts;
	std::unordered_map<string, idx_t> key_to_rep;
};

class FactorJoinBuildLocalState : public LocalSinkState {};

unique_ptr<GlobalSinkState> PhysicalAIFactorJoin::GetGlobalSinkState(ClientContext &) const {
	return make_uniq<FactorJoinBuildState>();
}

unique_ptr<LocalSinkState> PhysicalAIFactorJoin::GetLocalSinkState(ExecutionContext &) const {
	return make_uniq<FactorJoinBuildLocalState>();
}

static string RowKey(DataChunk &chunk, idx_t row) {
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

SinkResultType PhysicalAIFactorJoin::Sink(ExecutionContext &, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &gstate = input.global_state.Cast<FactorJoinBuildState>();
	lock_guard<mutex> guard(gstate.lock);
	for (idx_t row = 0; row < chunk.size(); row++) {
		auto key = RowKey(chunk, row);
		auto it = gstate.key_to_rep.find(key);
		if (it != gstate.key_to_rep.end()) {
			gstate.counts[it->second]++;
			continue;
		}
		vector<Value> rep;
		rep.reserve(chunk.ColumnCount());
		for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
			rep.push_back(chunk.data[col].GetValue(row));
		}
		gstate.key_to_rep.emplace(std::move(key), gstate.reps.size());
		gstate.reps.push_back(std::move(rep));
		gstate.counts.push_back(1);
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalAIFactorJoin::Combine(ExecutionContext &, OperatorSinkCombineInput &) const {
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalAIFactorJoin::Finalize(Pipeline &, Event &, ClientContext &,
                                                OperatorSinkFinalizeInput &) const {
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Operator: probe side streams; new probe reps evaluate against all build reps
//===--------------------------------------------------------------------===//
class FactorJoinProbeState : public OperatorState {
public:
	FactorJoinProbeState(ClientContext &context, const PhysicalAIFactorJoin &op) : executor(context, *op.predicate) {
		// pair-evaluation chunk in the cross layout: probe columns then build columns
		vector<LogicalType> cross_types(op.types.begin(), op.types.end() - 1);
		pair_chunk.Initialize(Allocator::Get(context), cross_types);
	}

	ExpressionExecutor executor;
	DataChunk pair_chunk;
	//! probe predicate-key -> passset id
	std::unordered_map<string, idx_t> key_to_passset;
	//! per passset: the build rep ids whose pair passed
	vector<vector<idx_t>> passsets;
	//! current input chunk's per-row passset ids (aligned with the input chunk)
	vector<idx_t> row_passset;
	//! pagination cursor over (input row, passset entry, duplicate)
	idx_t cur_row = 0;
	idx_t cur_entry = 0;
	idx_t cur_dup = 0;
	bool chunk_prepared = false;
};

unique_ptr<OperatorState> PhysicalAIFactorJoin::GetOperatorState(ExecutionContext &context) const {
	return make_uniq<FactorJoinProbeState>(context.client, *this);
}

static string ProbeKey(DataChunk &input, const vector<idx_t> &key_cols, idx_t row) {
	string key;
	for (const auto col : key_cols) {
		const Value v = input.data[col].GetValue(row);
		key.push_back(v.IsNull() ? '\x00' : '\x01');
		if (!v.IsNull()) {
			key += v.ToString();
		}
		key.push_back('\x1f');
	}
	return key;
}

//! Evaluate MANY new probe reps against every build rep in as few pair batches as possible, so a
//! chunk's whole new pair domain saturates LLM concurrency in one pass (per-rep rounds serialized
//! at |build| pairs a round -- the ablation's q7 40s-vs-11s finding).
static void EvaluateProbeReps(FactorJoinProbeState &state, const PhysicalAIFactorJoin &op, DataChunk &input,
                              const vector<idx_t> &rows, const FactorJoinBuildState &build,
                              vector<vector<idx_t>> &out_passsets) {
	const idx_t right_width = op.types.size() - 1 - op.left_width;
	const idx_t nb = build.reps.size();
	out_passsets.assign(rows.size(), {});
	if (nb == 0 || rows.empty()) {
		return;
	}
	const idx_t total = rows.size() * nb;
	idx_t done = 0;
	while (done < total) {
		const idx_t batch = MinValue<idx_t>(STANDARD_VECTOR_SIZE, total - done);
		auto &pair_chunk = state.pair_chunk;
		pair_chunk.Reset();
		pair_chunk.SetChildCardinality(batch);
		for (idx_t j = 0; j < batch; j++) {
			const idx_t pair = done + j;
			const idx_t row = rows[pair / nb];
			auto &rep = build.reps[pair % nb];
			for (idx_t c = 0; c < op.left_width; c++) {
				pair_chunk.data[c].SetValue(j, input.data[c].GetValue(row));
			}
			for (idx_t c = 0; c < right_width; c++) {
				pair_chunk.data[op.left_width + c].SetValue(j, rep[c]);
			}
		}
		Vector result(LogicalType::BOOLEAN);
		state.executor.ExecuteExpression(pair_chunk, result);
		for (idx_t j = 0; j < batch; j++) {
			const Value v = result.GetValue(j);
			if (!v.IsNull() && BooleanValue::Get(v)) {
				const idx_t pair = done + j;
				out_passsets[pair / nb].push_back(pair % nb);
			}
		}
		done += batch;
	}
}

OperatorResultType PhysicalAIFactorJoin::Execute(ExecutionContext &context, DataChunk &input, DataChunk &chunk,
                                                 GlobalOperatorState &, OperatorState &state_p) const {
	auto &state = state_p.Cast<FactorJoinProbeState>();
	auto &build = sink_state->Cast<FactorJoinBuildState>();

	if (!state.chunk_prepared) {
		// Resolve every row's passset, evaluating unseen probe reps (batched; the client dedups
		// repeated prompts, so calls = new distinct pairs).
		state.row_passset.assign(input.size(), 0);
		vector<idx_t> new_rows; // one representative row per unseen probe key, this chunk
		vector<string> new_keys;
		for (idx_t row = 0; row < input.size(); row++) {
			auto key = ProbeKey(input, left_key_cols, row);
			auto it = state.key_to_passset.find(key);
			if (it == state.key_to_passset.end()) {
				it = state.key_to_passset.emplace(key, state.passsets.size()).first;
				state.passsets.emplace_back();
				new_rows.push_back(row);
				new_keys.push_back(std::move(key));
			}
			state.row_passset[row] = it->second;
		}
		if (!new_rows.empty()) {
			vector<vector<idx_t>> passsets;
			EvaluateProbeReps(state, *this, input, new_rows, build, passsets);
			for (idx_t u = 0; u < new_rows.size(); u++) {
				state.passsets[state.key_to_passset[new_keys[u]]] = std::move(passsets[u]);
			}
		}
		state.cur_row = 0;
		state.cur_entry = 0;
		state.cur_dup = 0;
		state.chunk_prepared = true;
	}

	// Emit survivors: (input row) x (passing build rep) x (build multiplicity), paginated.
	const idx_t right_width = types.size() - 1 - left_width;
	const idx_t result_col = types.size() - 1;
	chunk.SetChildCardinality(STANDARD_VECTOR_SIZE);
	idx_t out = 0;
	while (state.cur_row < input.size() && out < STANDARD_VECTOR_SIZE) {
		auto &passset = state.passsets[state.row_passset[state.cur_row]];
		if (state.cur_entry >= passset.size()) {
			state.cur_row++;
			state.cur_entry = 0;
			state.cur_dup = 0;
			continue;
		}
		const idx_t rep_id = passset[state.cur_entry];
		if (state.cur_dup >= build.counts[rep_id]) {
			state.cur_entry++;
			state.cur_dup = 0;
			continue;
		}
		for (idx_t c = 0; c < left_width; c++) {
			chunk.data[c].SetValue(out, input.data[c].GetValue(state.cur_row));
		}
		auto &rep = build.reps[rep_id];
		for (idx_t c = 0; c < right_width; c++) {
			chunk.data[left_width + c].SetValue(out, rep[c]);
		}
		chunk.data[result_col].SetValue(out, Value::BOOLEAN(true));
		state.cur_dup++;
		out++;
	}
	chunk.SetChildCardinality(out);

	if (state.cur_row < input.size()) {
		return OperatorResultType::HAVE_MORE_OUTPUT;
	}
	state.chunk_prepared = false;
	return OperatorResultType::NEED_MORE_INPUT;
}

vector<const_reference<PhysicalOperator>> PhysicalAIFactorJoin::GetSources() const {
	// Like a join: the probe side is this pipeline's source.
	return children[0].get().GetSources();
}

void PhysicalAIFactorJoin::BuildPipelines(Pipeline &current, MetaPipeline &meta_pipeline) {
	op_state.reset();
	sink_state.reset();
	// probe pipeline: this operator streams; build side becomes a child meta pipeline sinking here
	auto &state = meta_pipeline.GetState();
	state.AddPipelineOperator(current, *this);
	auto &child_meta_pipeline = meta_pipeline.CreateChildMetaPipeline(current, *this, MetaPipelineType::JOIN_BUILD);
	child_meta_pipeline.Build(children[1]);
	children[0].get().BuildPipelines(current, meta_pipeline);
}

} // namespace duckdb
