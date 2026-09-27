#include "exec/ai_leaf_region.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "filter_tree_order.hpp"
#include "ai_client.hpp"
#include "duckdb/planner/expression_iterator.hpp"

#include <algorithm>
#include <chrono>
#include <thread>

namespace duckdb {

// Does the node's expression evaluate ai_image() anywhere? (The call sits inside the region's own leaf
// expressions, so the child chunk carries plain paths, never the sentinel.)
static bool ExpressionHasAIImage(const Expression &expr) {
	if (expr.GetExpressionClass() == ExpressionClass::BOUND_FUNCTION &&
	    expr.Cast<BoundFunctionExpression>().Function().GetName() == "ai_image") {
		return true;
	}
	bool found = false;
	ExpressionIterator::EnumerateChildren(
	    expr, [&](const Expression &child) { found = found || ExpressionHasAIImage(child); });
	return found;
}

AILeafRegionState::AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
                                     const vector<LogicalType> &child_types_p, int64_t limit)
    : context(context), call(eval_call), tree(AILeafTree(eval_call)), n(AILeafCount(eval_call)), limit(limit),
      query_text(context.GetCurrentQuery()), child_types(child_types_p), leaves(n),
      has_image_leaf(ExpressionHasAIImage(eval_call)), landed(n, 0) {
	for (idx_t l = 0; l < n; l++) {
		leaves[l].key_cols = AILeafKeyCols(eval_call, l);
		leaves[l].stage.Initialize(Allocator::Get(context), child_types);
	}
}

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
	// re-prediction is a forward pass and never a request.
	vector<vector<float>> feats;
	const uint64_t step = AISelectivityTrainSteps();
	const auto t0 = std::chrono::steady_clock::now();
	AILeafFeatures(call, fresh, query_text, feats);
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

bool AILeafRegionState::Append(DataChunk &chunk, idx_t fire_floor, idx_t fire_overlap) {
	const idx_t count = chunk.size();
	const auto &cfg = AIConfig::Get();
	const idx_t configured = cfg.embed_slice;
	constexpr idx_t kGateLabels = 20; // one trainer batch per leaf
	for (idx_t begin = 0; begin < count;) {
		// Slice regions with an image leaf (slow embeds; and the slices let this query's own verdicts order
		// the later rows), or any region when asked; otherwise a text-only region takes the rest of the
		// chunk in one go (one batched embed per leaf, one fire).
		const bool slicing = configured > 0 && (has_image_leaf || cfg.embed_slice_text);
		const idx_t slice = slicing ? MinValue<idx_t>(configured, count - begin) : count - begin;
		if (begin > 0 && n >= 2 && cfg.warm_gate && fire_floor > 0) {
			// Warm gate: the first slice was decided cold and fired; before deciding more rows, wait until
			// every leaf has a batch of verdicts (the rows that survived the first leaf have visited the
			// second) and the model has trained on them. Waits only while a wave is in flight -- verdicts
			// are coming -- so it is bounded by the first waves, and it costs nothing when they landed
			// before the next slice was embedded.
			auto warm = [&]() {
				if (AISelectivityTrainSteps() == 0) {
					return false;
				}
				for (idx_t l = 0; l < n; l++) {
					if (landed[l] < kGateLabels) {
						return false;
					}
				}
				return true;
			};
			const auto t0 = std::chrono::steady_clock::now();
			while (!warm() && !inflight.empty()) {
				ReapLanded();
				if (!warm() && !inflight.empty()) {
					std::this_thread::sleep_for(std::chrono::milliseconds(20));
				}
			}
			t_warm_gate += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		}
		if (AppendSlice(chunk, begin, begin + slice)) {
			return true;
		}
		begin += slice;
		if (fire_floor > 0 && begin < count && Fire(fire_floor, fire_overlap)) {
			return true;
		}
	}
	return false;
}

bool AILeafRegionState::AppendSlice(DataChunk &chunk, idx_t begin, idx_t end) {
	const idx_t count = end - begin;
	const uint32_t first_row = NumericCast<uint32_t>(row_result.size());
	row_reps.resize(row_reps.size() + count * n);
	row_result.resize(row_result.size() + count, -2);
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
		any_open |= open;
	}
	const AITriState v = AIFilterTreeEval(*tree, values);
	if (v != AITriState::TRI_UNKNOWN || !any_open) {
		row_result[row] = static_cast<int8_t>(v == AITriState::TRI_TRUE ? 1 : (v == AITriState::TRI_FALSE ? 0 : -1));
		if (v == AITriState::TRI_TRUE) {
			passed++;
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

unique_ptr<AILeafWave> AILeafRegionState::PrepareWave(idx_t l) {
	auto &leaf = leaves[l];
	auto wave = make_uniq<AILeafWave>();
	wave->leaf = l;
	const idx_t take = MinValue<idx_t>(leaf.pending.size(), STANDARD_VECTOR_SIZE);
	for (idx_t i = 0; i < take; i++) {
		const uint32_t rep_id = leaf.pending[i];
		leaf.reps[rep_id].state = 2;
		wave->reps.push_back(rep_id);
		wave->texts.prompt.push_back(leaf.texts.prompt[rep_id]);
		wave->texts.pred_text.push_back(leaf.texts.pred_text[rep_id]);
		wave->texts.input_text.push_back(leaf.texts.input_text[rep_id]);
		wave->texts.cost.push_back(leaf.texts.cost[rep_id]);
		wave->texts.valid.push_back(leaf.texts.valid[rep_id]);
	}
	leaf.pending.erase(leaf.pending.begin(), leaf.pending.begin() + NumericCast<int64_t>(take));
	if (waves == 0) {
		t_first_wave = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
		rows_at_first_wave = row_result.size();
	}
	waves++;
	return wave;
}

void AILeafRegionState::RunWave(AILeafWave &wave) {
	AILeafEvaluate(context, call, wave.leaf, wave.texts, query_text, wave.out_result, wave.out_valid);
}

void AILeafRegionState::ApplyWave(AILeafWave &wave) {
	auto &leaf = leaves[wave.leaf];
	landed[wave.leaf] += wave.reps.size();
	vector<uint32_t> waiters;
	for (idx_t i = 0; i < wave.reps.size(); i++) {
		auto &rep = leaf.reps[wave.reps[i]];
		rep.state = 3;
		rep.valid = rep.valid && wave.out_valid[i] != 0;
		rep.verdict = wave.out_result[i] ? 1 : 0;
		waiters.insert(waiters.end(), rep.waiters.begin(), rep.waiters.end());
		rep.waiters.clear();
		rep.waiters.shrink_to_fit();
	}
	for (idx_t i = 0; i < waiters.size(); i++) {
		DecideRow(waiters[i]);
		// A verdict on a shared rep re-decides every row behind it (above a join: hundreds of thousands).
		// Start the next leaf's calls as soon as a full batch is pending instead of after the last waiter.
		if ((i & 4095) == 4095 && limit < 0) {
			LaunchReady(STANDARD_VECTOR_SIZE);
		}
	}
}

void AILeafRegionState::LaunchReady(idx_t min_pending) {
	for (idx_t l = 0; l < n && inflight.size() < MaxValue<idx_t>(overlap, 1); l++) {
		while (leaves[l].pending.size() >= min_pending && inflight.size() < MaxValue<idx_t>(overlap, 1)) {
			auto wave = PrepareWave(l);
			auto *raw = wave.get();
			raw->done = std::async(std::launch::async, [this, raw]() { RunWave(*raw); });
			inflight.push_back(std::move(wave));
		}
	}
}

void AILeafRegionState::DrainOldest() {
	auto wave = std::move(inflight.front());
	inflight.erase(inflight.begin());
	const auto t0 = std::chrono::steady_clock::now();
	if (wave->done.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
		drains_blocked++;
	}
	wave->done.get(); // rethrows the wave's exception on this thread
	t_drain += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	ApplyWave(*wave);
}

void AILeafRegionState::ReapLanded() {
	for (idx_t i = 0; i < inflight.size();) {
		if (inflight[i]->done.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
			auto wave = std::move(inflight[i]);
			inflight.erase(inflight.begin() + NumericCast<int64_t>(i));
			wave->done.get();
			ApplyWave(*wave);
			i = 0; // ApplyWave may have launched new waves: rescan
			continue;
		}
		i++;
	}
}

void AILeafRegionState::Drain() {
	while (!inflight.empty()) {
		DrainOldest();
	}
}

bool AILeafRegionState::Fire(idx_t floor, idx_t overlap_p) {
	overlap = overlap_p;
	if (limit < 0) {
		ReapLanded(); // landed verdicts move rows onto their next leaf before we look at what is pending
	}
	for (;;) {
		// An idle pool is the one thing worse than a small wave: with nothing in flight, fire what a
		// concurrency-wide batch can use. Above a join whose big side trickles the small leaf's reps
		// (Q19: 180 customers behind 1.78M rows), waiting for a floor-sized batch of the first leaf
		// idled the pool for the first ~1M rows of ingest.
		const idx_t floor_now = (limit < 0 && inflight.empty())
		                            ? MinValue<idx_t>(floor, MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1))
		                            : floor;
		idx_t ready = n;
		for (idx_t l = 0; l < n; l++) {
			if (leaves[l].pending.size() >= floor_now) {
				ready = l;
				break;
			}
		}
		if (ready == n) {
			return false;
		}
		auto wave = PrepareWave(ready);
		if (limit >= 0) {
			// Under a LIMIT the floor's job is to stop early: evaluate inline and re-check after every wave.
			RunWave(*wave);
			ApplyWave(*wave);
			if (LimitMet()) {
				return true;
			}
			continue;
		}
		auto *raw = wave.get();
		raw->done = std::async(std::launch::async, [this, raw]() { RunWave(*raw); });
		inflight.push_back(std::move(wave));
		while (inflight.size() >= MaxValue<idx_t>(overlap, 1)) {
			DrainOldest();
		}
	}
}

void AILeafRegionState::Finish(idx_t overlap_p) {
	overlap = overlap_p;
	Drain();
	if (LimitMet()) {
		return;
	}
	// Invariant: every undecided row waits on a rep that is pending or in flight, so when both are empty
	// every row is decided.
	for (;;) {
		if (Fire(1, overlap)) {
			return;
		}
		if (inflight.empty()) {
			return;
		}
		DrainOldest();
	}
}

string AILeafRegionState::TimingSummary() const {
	return StringUtil::Format(
	    "embed=%.1fs refresh=%.1fs(%llu) decide=%.1fs warm_gate=%.1fs drain_wait=%.1fs(blocked %llu/%llu) "
	    "first_wave=%.1fs@%llu rows",
	    t_embed, t_refresh, refreshes, t_decide, t_warm_gate, t_drain, drains_blocked, waves, t_first_wave,
	    rows_at_first_wave);
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
