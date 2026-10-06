#include "exec/ai_leaf_region.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "filter_tree_order.hpp"
#include "ai_client.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <algorithm>
#include <chrono>

namespace duckdb {

AILeafRegionState::AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
                                     const vector<LogicalType> &child_types_p, int64_t limit,
                                     vector<idx_t> limit_distinct_cols)
    : context(context), call(eval_call), tree(AILeafTree(eval_call)), n(AILeafCount(eval_call)), limit(limit),
      wave_cap(limit >= 0
                   ? MaxValue<idx_t>(MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1), NumericCast<idx_t>(limit))
                   : STANDARD_VECTOR_SIZE),
      distinct_cols(std::move(limit_distinct_cols)), query_text(context.GetCurrentQuery()), child_types(child_types_p),
      leaves(n), landed(n, 0), evaluator(make_uniq<AILeafUnitEvaluator>(eval_call)) {
	for (idx_t l = 0; l < n; l++) {
		leaves[l].key_cols = AILeafKeyCols(eval_call, l);
		leaves[l].stage.Initialize(Allocator::Get(context), child_types);
	}
}

AILeafRegionState::~AILeafRegionState() = default;

vector<idx_t> AILeafRegionState::DistinctPerLeaf() const {
	vector<idx_t> out;
	for (auto &leaf : leaves) {
		out.push_back(leaf.reps.size());
	}
	return out;
}

// Texts + predictions for the staged new reps of leaf `l` (one batched embed), then clear the stage.
void AILeafRegionState::FlushStage(idx_t l) {
	auto &leaf = leaves[l];
	if (leaf.stage.size() == 0) {
		return;
	}
	AILeafTexts fresh;
	AILeafBuildTexts(context, call, l, leaf.stage, fresh);
	if (AIConfig::Get().debug_log.find("leaftexts") != string::npos) {
		for (idx_t i = 0; i < MinValue<idx_t>(fresh.Size(), 4); i++) {
			fprintf(stderr, "[leaf %llu rep %u] valid=%d prompt=%s\n", (unsigned long long)l, leaf.stage_reps[i],
			        (int)fresh.valid[i], fresh.prompt[i].substr(0, 60).c_str());
		}
	}
	// Embed once per rep (batched, cold or warm): the feature stays on the rep, so every later
	// re-prediction is a forward pass and never a request. A single-leaf tree has nothing to order, so its
	// reps carry no feature (neutral p, no training example): MOVIE's LIMIT queries spent more time in these
	// per-slice embed requests than in the model calls they never influenced.
	vector<vector<float>> feats;
	const uint64_t step = AISelectivityTrainSteps();
	const auto t0 = std::chrono::steady_clock::now();
	if (n > 1) {
		AILeafFeatures(call, fresh, query_text, feats);
	} else {
		feats.assign(fresh.Size(), vector<float>());
	}
	t_embed += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	for (idx_t i = 0; i < fresh.Size(); i++) {
		auto &rep = leaf.reps[leaf.stage_reps[i]];
		rep.feat = std::move(feats[i]);
		rep.p = AILeafPredictFeature(rep.feat);
		rep.p_step = step;
		rep.cost = fresh.cost[i];
		rep.valid = fresh.valid[i] != 0;
		D_ASSERT(leaf.texts.Size() == leaf.stage_reps[i]);
		leaf.texts.prompt.push_back(std::move(fresh.prompt[i]));
		leaf.texts.pred_text.push_back(std::move(fresh.pred_text[i]));
		leaf.texts.input_text.push_back(std::move(fresh.input_text[i]));
		leaf.texts.cost.push_back(fresh.cost[i]);
		leaf.texts.valid.push_back(fresh.valid[i]);
	}
	leaf.stage.Reset();
	leaf.stage_reps.clear();
}

idx_t AILeafRegionState::Append(DataChunk &chunk, idx_t limit_floor, bool &limit_met) {
	const idx_t count = chunk.size();
	const idx_t rows_before = row_result.size();
	limit_met = false;
	// Rows ingested per slice before the region embeds, decides and dispatches. Slicing lets the first
	// calls start after the first slice's features instead of the chunk's (500 CLIP image embeds are ~80 s
	// of ingest time), and lets this query's own verdicts order the later rows: text regions too (agent_bench
	// Q17 237 -> 156 calls for +4-7 s).
	constexpr idx_t kSliceRows = 100;
	for (idx_t begin = 0; begin < count;) {
		const idx_t slice = MinValue<idx_t>(kSliceRows, count - begin);
		if (begin > 0 && limit < 0 && dispatcher && !Warm()) {
			// Warm gate: before deciding more rows, wait for the calls already out -- verdicts are coming, and the
			// model trains on them -- so the next slice is ordered by a model that knows this query's pass
			// rates. Nothing new is dispatched while waiting, so the wait is bounded by one round of calls, and
			// it costs nothing when they landed before the next slice was embedded.
			const auto t0 = std::chrono::steady_clock::now();
			while (!Warm() && dispatcher->Outstanding() > 0) {
				WaitAndReap();
			}
			t_warm_gate += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		}
		if (AppendSlice(chunk, begin, begin + slice)) {
			limit_met = true;
			return row_result.size() - rows_before;
		}
		begin += slice;
		if (begin < count && Pump(limit_floor)) {
			limit_met = true;
			return row_result.size() - rows_before;
		}
	}
	return count;
}

bool AILeafRegionState::AppendSlice(DataChunk &chunk, idx_t begin, idx_t end) {
	const idx_t count = end - begin;
	const uint32_t first_row = NumericCast<uint32_t>(row_result.size());
	row_reps.resize(row_reps.size() + count * n);
	row_result.resize(row_result.size() + count, -2);
	if (!distinct_cols.empty()) {
		for (idx_t row = begin; row < end; row++) {
			row_dkey.push_back(DistinctKeyOf(chunk, row));
		}
	}
	// 1. Per-leaf dedup of every row; new reps are staged so their texts can be built in one pass.
	for (idx_t l = 0; l < n; l++) {
		auto &leaf = leaves[l];
		for (idx_t row = begin; row < end; row++) {
			string key;
			for (const idx_t kc : leaf.key_cols) {
				const Value v = chunk.data[kc].GetValue(row);
				key.push_back(v.IsNull() ? '\x00' : '\x01');
				if (!v.IsNull()) {
					key += v.ToString();
				}
				key.push_back('\x1f');
			}
			auto it = leaf.dict.find(key);
			uint32_t rep;
			if (it != leaf.dict.end()) {
				rep = it->second;
			} else {
				rep = NumericCast<uint32_t>(leaf.reps.size());
				leaf.dict.emplace(std::move(key), rep);
				leaf.reps.emplace_back();
				if (leaf.stage.size() == STANDARD_VECTOR_SIZE) {
					FlushStage(l);
				}
				// vectors carry their own size: size the children BEFORE the writes (as AISQLMapChunk::StageRow does)
				const idx_t at = leaf.stage.size();
				leaf.stage.SetChildCardinality(at + 1);
				for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
					leaf.stage.data[col].SetValue(at, chunk.data[col].GetValue(row));
				}
				leaf.stage_reps.push_back(rep);
			}
			row_reps[(first_row + row - begin) * n + l] = rep;
		}
	}
	// 2. Texts + predictions for this slice's new reps (one batched embed per leaf).
	for (idx_t l = 0; l < n; l++) {
		FlushStage(l);
	}
	// 3. Each new row chooses its first leaf (or is already decided by verdicts landed earlier).
	for (idx_t row = 0; row < count; row++) {
		DecideRow(first_row + NumericCast<uint32_t>(row));
		if (LimitMet()) {
			return true;
		}
	}
	return false;
}

void AILeafRegionState::RefreshPrediction(Leaf &leaf, uint32_t rep_id) {
	auto &rep = leaf.reps[rep_id];
	const uint64_t step = AISelectivityTrainSteps();
	if (rep.p_step == step || !rep.valid) {
		return;
	}
	const auto t0 = std::chrono::steady_clock::now();
	rep.p = AILeafPredictFeature(rep.feat);
	t_refresh += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	refreshes++;
	rep.p_step = step;
}

// Tree-evaluate a row over its reps' verdicts; finish it, or register it on its next leaf's rep.
void AILeafRegionState::DecideRow(uint32_t row) {
	const auto t0 = std::chrono::steady_clock::now();
	struct Timer { // inclusive of the JIT refreshes it triggers
		double &acc;
		std::chrono::steady_clock::time_point t0;
		~Timer() {
			acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		}
	} timer {t_decide, t0};
	vector<AITriState> values(n, AITriState::TRI_UNKNOWN);
	vector<double> p(n), cost(n);
	bool any_open = false;
	for (idx_t l = 0; l < n; l++) {
		auto &rep = leaves[l].reps[row_reps[row * n + l]];
		if (rep.state != 3 && rep.valid) {
			RefreshPrediction(leaves[l], row_reps[row * n + l]); // JIT: the model may have learned since
		}
		if (rep.state == 3 && rep.valid) {
			values[l] = rep.verdict ? AITriState::TRI_TRUE : AITriState::TRI_FALSE;
		}
		p[l] = rep.p;
		// A leaf already asked, or one that can never answer (NULL prompt), is not a candidate. The cost of
		// asking a leaf FOR THIS ROW is its call cost amortized over the rows its rep serves: above a join
		// the small side's rep is shared by every row of the big side, so it is the cheap question even when
		// the model is cold (fan-out 1 on a single table: plain cost, exactly the per-row batch's choice).
		const bool open = rep.state != 3 && rep.valid;
		const double fanout = MaxValue<double>(1.0, double(row_result.size()) / double(leaves[l].reps.size()));
		cost[l] = open ? rep.cost / fanout : 1e30;
		// A rep already pending or in flight will be answered anyway: waiting on it costs this row no call.
		// Verdicts land one by one, so a row is often re-decided while its other leaf is still in flight;
		// without this it could open a third leaf that the in-flight answer would have made unnecessary.
		if (open && (rep.state == 1 || rep.state == 2)) {
			cost[l] *= 1e-6;
		}
		any_open |= open;
	}
	const AITriState v = AIFilterTreeEval(*tree, values);
	if (v != AITriState::TRI_UNKNOWN || !any_open) {
		row_result[row] = static_cast<int8_t>(v == AITriState::TRI_TRUE ? 1 : (v == AITriState::TRI_FALSE ? 0 : -1));
		if (v == AITriState::TRI_TRUE) {
			passed++;
			if (!distinct_cols.empty()) {
				passed_keys.insert(row_dkey[row]);
			}
		}
		return;
	}
	idx_t next = AIFilterTreeChooseNextLeaf(*tree, values, p, cost);
	if (next >= n || cost[next] >= 1e29) {
		next = n; // the DP declined: take the cheapest open leaf
		for (idx_t l = 0; l < n; l++) {
			if (cost[l] < 1e29 && (next == n || cost[l] < cost[next])) {
				next = l;
			}
		}
	}
	auto &leaf = leaves[next];
	const uint32_t rep_id = row_reps[row * n + next];
	auto &rep = leaf.reps[rep_id];
	rep.waiters.push_back(row);
	if (rep.state == 0) {
		rep.state = 1;
		leaf.pending.push_back(rep_id);
	}
}

uint32_t AILeafRegionState::DistinctKeyOf(DataChunk &chunk, idx_t row) {
	string key;
	for (const idx_t c : distinct_cols) {
		const Value v = chunk.data[c].GetValue(row);
		key.push_back(v.IsNull() ? '\x00' : '\x01');
		if (!v.IsNull()) {
			key += v.ToString();
		}
		key.push_back('\x1f');
	}
	auto it = dkey_dict.find(key);
	if (it != dkey_dict.end()) {
		return it->second;
	}
	const auto id = NumericCast<uint32_t>(dkeys.size());
	dkeys.push_back(key);
	dkey_dict.emplace(std::move(key), id);
	return id;
}

void AILeafRegionState::MergePassedKeys(std::unordered_set<string> &out) const {
	for (const uint32_t id : passed_keys) {
		out.insert(dkeys[id]);
	}
}

void AILeafRegionState::ApplyVerdict(idx_t l, uint32_t rep_id, bool value, bool valid) {
	auto &rep = leaves[l].reps[rep_id];
	rep.state = 3;
	rep.valid = rep.valid && valid;
	rep.verdict = value ? 1 : 0;
	landed[l]++;
}

bool AILeafRegionState::Warm() const {
	if (n < 2) {
		return true; // nothing to order
	}
	if (AISelectivityTrainSteps() == 0) {
		return false;
	}
	for (idx_t l = 0; l < n; l++) {
		if (landed[l] < GATE_LABELS) {
			return false;
		}
	}
	return true;
}

void AILeafRegionState::DispatchPending(bool flush) {
	if (limit >= 0) {
		return;
	}
	// Warm: every pending rep goes at once. Cold: a leaf's reps go once a concurrency-wide batch is pending, so
	// the first verdicts train the model before most rows are decided (agent_bench Q17 156 calls, not 237).
	const idx_t floor = (flush || Warm()) ? 1 : MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
	vector<char> ready(n, 0);
	for (idx_t l = 0; l < n; l++) {
		ready[l] = (!leaves[l].pending.empty() && leaves[l].pending.size() >= floor) ? 1 : 0;
		if (ready[l] && AIConfig::Get().debug_log.find("dispatch") != string::npos) {
			fprintf(stderr, "[dispatch] t=%.2f leaf=%llu n=%llu rows=%llu floor=%llu flush=%d out=%llu\n",
			        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count(),
			        (unsigned long long)l, (unsigned long long)leaves[l].pending.size(),
			        (unsigned long long)row_result.size(), (unsigned long long)floor, (int)flush,
			        (unsigned long long)(dispatcher ? dispatcher->Outstanding() : 0));
		}
	}
	// Round-robin over the leaves, so a long backlog on one leaf does not queue another leaf's calls behind it.
	for (idx_t i = 0;; i++) {
		bool any = false;
		for (idx_t l = 0; l < n; l++) {
			auto &leaf = leaves[l];
			if (!ready[l] || i >= leaf.pending.size()) {
				continue;
			}
			any = true;
			const uint32_t rep_id = leaf.pending[i];
			auto &rep = leaf.reps[rep_id];
			rep.state = 2;
			if (!dispatcher) {
				dispatcher = make_uniq<AIAsyncDispatcher>(AIConfig::Get().max_concurrency);
			}
			if (dispatched == 0) {
				t_first_call = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
				rows_at_first_call = row_result.size();
			}
			dispatched++;
			const AILeafUnitEvaluator &eval = *evaluator;
			ClientContext &ctx = context;
			const string &qt = query_text;
			dispatcher->Submit(
			    (static_cast<idx_t>(l) << 32) | rep_id,
			    [&eval, &ctx, &qt, l, prompt = leaf.texts.prompt[rep_id], pred = leaf.texts.pred_text[rep_id],
			     feat = rep.feat]() { return Value::BOOLEAN(eval.Evaluate(ctx, l, prompt, pred, feat, qt)); });
		}
		if (!any) {
			break;
		}
	}
	for (idx_t l = 0; l < n; l++) {
		if (ready[l]) {
			leaves[l].pending.clear();
		}
	}
}

void AILeafRegionState::ReapLanded() {
	if (!dispatcher) {
		return;
	}
	vector<AIAsyncDispatcher::Landed> got;
	dispatcher->Reap(got);
	if (got.empty()) {
		return;
	}
	// Record every landed verdict first, then re-decide the rows behind them: a row whose other leaf also
	// landed in this batch resolves without being queued on it.
	vector<uint32_t> waiters;
	for (auto &g : got) {
		const idx_t l = g.key >> 32;
		const auto rep_id = static_cast<uint32_t>(g.key & 0xFFFFFFFFULL);
		ApplyVerdict(l, rep_id, BooleanValue::Get(g.value), true);
		auto &rep = leaves[l].reps[rep_id];
		waiters.insert(waiters.end(), rep.waiters.begin(), rep.waiters.end());
		rep.waiters.clear();
		rep.waiters.shrink_to_fit();
	}
	for (idx_t i = 0; i < waiters.size(); i++) {
		DecideRow(waiters[i]);
		// A verdict on a shared rep re-decides every row behind it (above a join: hundreds of thousands).
		// Start the next leaf's calls while the rest are still being re-decided.
		if ((i & 4095) == 4095) {
			DispatchPending();
		}
	}
}

void AILeafRegionState::WaitAndReap() {
	t_wait += dispatcher->WaitAny();
	waits++;
	ReapLanded();
}

void AILeafRegionState::RunInlineWave(idx_t l) {
	auto &leaf = leaves[l];
	AILeafWave wave;
	wave.leaf = l;
	const idx_t take = MinValue<idx_t>(leaf.pending.size(), wave_cap);
	for (idx_t i = 0; i < take; i++) {
		const uint32_t rep_id = leaf.pending[i];
		leaf.reps[rep_id].state = 2;
		wave.reps.push_back(rep_id);
		wave.texts.prompt.push_back(leaf.texts.prompt[rep_id]);
		wave.texts.pred_text.push_back(leaf.texts.pred_text[rep_id]);
		wave.texts.input_text.push_back(leaf.texts.input_text[rep_id]);
		wave.texts.cost.push_back(leaf.texts.cost[rep_id]);
		wave.texts.valid.push_back(leaf.texts.valid[rep_id]);
	}
	leaf.pending.erase(leaf.pending.begin(), leaf.pending.begin() + NumericCast<int64_t>(take));
	if (dispatched == 0) {
		t_first_call = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
		rows_at_first_call = row_result.size();
	}
	dispatched += take;
	AILeafEvaluate(context, call, l, wave.texts, query_text, wave.out_result, wave.out_valid);
	vector<uint32_t> waiters;
	for (idx_t i = 0; i < wave.reps.size(); i++) {
		ApplyVerdict(l, wave.reps[i], wave.out_result[i] != 0, wave.out_valid[i] != 0);
		auto &rep = leaf.reps[wave.reps[i]];
		waiters.insert(waiters.end(), rep.waiters.begin(), rep.waiters.end());
		rep.waiters.clear();
		rep.waiters.shrink_to_fit();
	}
	for (const auto row : waiters) {
		DecideRow(row);
	}
}

bool AILeafRegionState::Pump(idx_t limit_floor) {
	if (limit < 0) {
		ReapLanded();
		DispatchPending();
		return false;
	}
	// Under a LIMIT the floor's job is to stop early: evaluate inline and re-check after every wave.
	for (;;) {
		idx_t ready = n;
		for (idx_t l = 0; l < n; l++) {
			if (!leaves[l].pending.empty() && leaves[l].pending.size() >= limit_floor) {
				ready = l;
				break;
			}
		}
		if (ready == n) {
			return false;
		}
		RunInlineWave(ready);
		if (LimitMet()) {
			return true;
		}
	}
}

void AILeafRegionState::Drain() {
	while (dispatcher && dispatcher->Outstanding() > 0) {
		WaitAndReap();
	}
}

void AILeafRegionState::Finish() {
	if (LimitMet()) {
		return;
	}
	if (limit >= 0) {
		Pump(1);
		return;
	}
	// Invariant: every undecided row waits on a rep that is pending or outstanding, so when both are empty
	// every row is decided.
	for (;;) {
		ReapLanded();
		DispatchPending(/*flush=*/true);
		if (!dispatcher || dispatcher->Outstanding() == 0) {
			return;
		}
		WaitAndReap();
	}
}

void AILeafRegionState::FlushDispatch() {
	ReapLanded();
	DispatchPending(/*flush=*/true);
}

void AILeafRegionState::DecideThrough(idx_t row) {
	for (;;) {
		FlushDispatch();
		if (row_result[row] != -2) {
			return;
		}
		// Invariant: an undecided row waits on a rep that is pending or in flight, and FlushDispatch just
		// dispatched every pending rep, so something is outstanding; wait for a verdict and re-decide.
		if (!dispatcher || dispatcher->Outstanding() == 0) {
			throw InternalException("AI region: row %llu undecided with nothing in flight", row);
		}
		WaitAndReap();
	}
}

idx_t AILeafRegionState::DecidedPrefix(idx_t from, idx_t to) const {
	idx_t n_ready = 0;
	for (idx_t r = from; r < to && row_result[r] != -2; r++) {
		n_ready++;
	}
	return n_ready;
}

Value AILeafRegionState::RowValue(idx_t row) const {
	const int8_t r = row_result[row];
	if (r == 1) {
		return Value::BOOLEAN(true);
	}
	if (r == -1) {
		return Value(LogicalType::BOOLEAN);
	}
	return Value::BOOLEAN(false);
}

string AILeafRegionState::TimingSummary() const {
	return StringUtil::Format("embed=%.1fs refresh=%.1fs(%llu) decide=%.1fs warm_gate=%.1fs wait=%.1fs(%llu) "
	                          "calls=%llu first_call=%.1fs@%llu rows",
	                          t_embed, t_refresh, refreshes, t_decide, t_warm_gate, t_wait, waits, dispatched,
	                          t_first_call, rows_at_first_call);
}

void AILeafRegionState::Results(vector<Value> &out) const {
	for (const int8_t r : row_result) {
		if (r == 1) {
			out.push_back(Value::BOOLEAN(true));
		} else if (r == -1) {
			out.push_back(Value(LogicalType::BOOLEAN));
		} else {
			out.push_back(Value::BOOLEAN(false)); // decided false, or left undecided by an early stop
		}
	}
}

} // namespace duckdb
