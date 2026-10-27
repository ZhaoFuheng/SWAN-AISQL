#include "exec/ai_leaf_region.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/common/string_util.hpp"
#include "duckdb/main/client_context.hpp"
#include "filter_tree_order.hpp"
#include "ai_client.hpp"
#include "duckdb/planner/expression_iterator.hpp"
#include "optimizer/ai_filter_tree_build.hpp"

#include <algorithm>
#include <chrono>

namespace duckdb {

//! Weight of the model's prediction against observed verdicts in the prefix-cache write decision (answers).
static constexpr double PASS_RATE_PRIOR = 4.0;

AILeafRegionState::AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
                                     const vector<LogicalType> &child_types_p, int64_t limit,
                                     vector<idx_t> limit_distinct_cols)
    : context(context), call(eval_call), tree(AILeafTree(eval_call)), n(AILeafCount(eval_call)), limit(limit),
      wave_cap(limit >= 0
                   ? MaxValue<idx_t>(MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1), NumericCast<idx_t>(limit))
                   : STANDARD_VECTOR_SIZE),
      distinct_cols(std::move(limit_distinct_cols)), query_text(context.GetCurrentQuery()), child_types(child_types_p),
      leaves(n), landed(n, 0), observed_n(n, 0), observed_true(n, 0), async_inflight(n, 0),
      evaluator(make_uniq<AILeafUnitEvaluator>(eval_call)) {
	for (idx_t l = 0; l < n; l++) {
		leaves[l].key_cols = AILeafKeyCols(eval_call, l);
		leaves[l].stage.Initialize(Allocator::Get(context), child_types);
	}
	// the operand each leaf's prompt leads with (the convention: the shared input first). Only ai_filter
	// leaves share a span: another function's request has its own system prompt and schema, so the same
	// input under it is another span, and its reads must not be counted on.
	lead_key.assign(n, string());
	sum_logit_pask.assign(n, 0.0);
	for (idx_t l = 0; l < n; l++) {
		lead_key[l] = evaluator->LeafKind(l) == 'F' ? AILeafLeadKey(eval_call, l) : string();
	}
	has_partner.assign(n, 0);
	for (idx_t l = 0; l < n; l++) {
		for (idx_t k = 0; k < n && !has_partner[l]; k++) {
			has_partner[l] = k != l && !lead_key[l].empty() && lead_key[l] == lead_key[k] ? 1 : 0;
		}
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
	// One blocked forward pass for the slice's new reps (same values as one pass per rep).
	vector<const vector<float> *> feat_ptrs(feats.size());
	for (idx_t i = 0; i < feats.size(); i++) {
		feat_ptrs[i] = &feats[i];
	}
	vector<double> p_fresh;
	AILeafPredictFeatures(feat_ptrs, p_fresh);
	for (idx_t i = 0; i < fresh.Size(); i++) {
		auto &rep = leaf.reps[leaf.stage_reps[i]];
		rep.feat = std::move(feats[i]);
		rep.p = p_fresh[i];
		rep.p_step = step;
		rep.cost = fresh.cost[i];
		rep.valid = fresh.valid[i] != 0;
		if (!rep.valid) {
			// A NULL prompt is decided without a call: the leaf's answer is NULL, and its wrapper may turn that
			// into a verdict (`ai_filter(a) IS NULL`, coalesce); a bare call stays a NULL (unknown) leaf.
			const AITriState v = evaluator->NullOutcome(context, l);
			rep.state = 3;
			if (v != AITriState::TRI_NULL) {
				rep.valid = true;
				rep.verdict = v == AITriState::TRI_TRUE ? 1 : 0;
			}
		}
		D_ASSERT(leaf.texts.Size() == leaf.stage_reps[i]);
		leaf.texts.prompt.push_back(std::move(fresh.prompt[i]));
		leaf.texts.pred_text.push_back(std::move(fresh.pred_text[i]));
		leaf.texts.input_text.push_back(std::move(fresh.input_text[i]));
		leaf.texts.cost.push_back(fresh.cost[i]);
		leaf.texts.valid.push_back(fresh.valid[i]);
		leaf.texts.lead_span.push_back(i < fresh.lead_span.size() ? std::move(fresh.lead_span[i]) : string());
		leaf.texts.head = fresh.head;
		AIDebugSelectivityKnob(leaf.texts.prompt.back(), rep.p); // test knob: a pinned prediction
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
	// SET ai_leaf_slice_rows (default 100). A smaller slice commits fewer rows to their first leaf while cold.
	const idx_t kSliceRows = MaxValue<idx_t>(AIConfig::Get().leaf_slice_rows, 1);
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
		if (limit < 0) {
			slices_ingested++;
		}
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
			string key; // the leaf's key columns, each NULL-flagged and length-prefixed (AIAppendKeyPart)
			for (const idx_t kc : leaf.key_cols) {
				AIAppendKeyPart(key, chunk.data[kc].GetValue(row));
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
	{
		vector<uint32_t> new_rows(count);
		for (idx_t row = 0; row < count; row++) {
			new_rows[row] = first_row + NumericCast<uint32_t>(row);
		}
		RefreshStalePredictions(new_rows.data(), new_rows.size());
	}
	for (idx_t row = 0; row < count; row++) {
		DecideRow(first_row + NumericCast<uint32_t>(row));
		if (LimitMet()) {
			return true;
		}
	}
	return false;
}

double AILeafRegionState::ReuseHintFor(uint32_t row, idx_t next) {
	if (lead_key[next].empty()) {
		return -1.0;
	}
	bool sibling = false;
	vector<char> shares(n, 0);
	vector<AITriState> values(n, AITriState::TRI_UNKNOWN);
	vector<double> p_hint(n), cost(n);
	for (idx_t l = 0; l < n; l++) {
		auto &leaf_l = leaves[l];
		const uint32_t rid = row_reps[row * n + l];
		auto &rep = leaf_l.reps[rid];
		if (rep.state == 3 && rep.valid) {
			values[l] = rep.verdict ? AITriState::TRI_TRUE : AITriState::TRI_FALSE;
		}
		cost[l] = (rep.state == 3 || !rep.valid || l == next) ? 1e30 : rep.cost;
		sibling |= l != next && lead_key[l] == lead_key[next];
		shares[l] = l != next && lead_key[l] == lead_key[next] && values[l] == AITriState::TRI_UNKNOWN ? 1 : 0;
		// this query's observed pass rate, shrunk toward the prediction; a test knob pins it
		double pinned = 0.0;
		const bool knob = rid < leaf_l.texts.prompt.size() && AIDebugSelectivityKnob(leaf_l.texts.prompt[rid], pinned);
		p_hint[l] = knob ? pinned
		                 : (static_cast<double>(observed_true[l]) + PASS_RATE_PRIOR * rep.p) /
		                       (static_cast<double>(observed_n[l]) + PASS_RATE_PRIOR);
	}
	if (!sibling) {
		return -1.0; // no other leaf of this node leads with the span: the reuse, if any, is elsewhere
	}
	if (values[next] != AITriState::TRI_UNKNOWN) {
		return 0.0;
	}
	return AIFilterTreeExpectedSharedReads(*tree, values, p_hint, cost, next, shares);
}

void AILeafRegionState::RefreshPrediction(Leaf &leaf, uint32_t rep_id) {
	auto &rep = leaf.reps[rep_id];
	const uint64_t step = AISelectivityTrainSteps();
	if (rep.p_step == step || !rep.valid) {
		return;
	}
	const auto t0 = std::chrono::steady_clock::now();
	rep.p = AILeafPredictFeature(rep.feat);
	AIDebugSelectivityKnob(leaf.texts.prompt[rep_id], rep.p); // test knob: a pinned prediction
	t_refresh += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	refreshes++;
	rep.p_step = step;
}

void AILeafRegionState::RefreshStalePredictions(const uint32_t *rows, idx_t count) {
	if (n <= 1 || count == 0) {
		return;
	}
	const uint64_t step = AISelectivityTrainSteps();
	vector<const vector<float> *> feats;
	vector<std::pair<idx_t, uint32_t>> which;
	for (idx_t r = 0; r < count; r++) {
		const uint32_t row = rows[r];
		if (row_result[row] != -2) {
			continue;
		}
		for (idx_t l = 0; l < n; l++) {
			const uint32_t rep_id = row_reps[row * n + l];
			auto &rep = leaves[l].reps[rep_id];
			if (rep.state == 3 || !rep.valid || rep.feat.empty() || rep.p_step == step) {
				continue;
			}
			rep.p_step = step; // claimed: listed once even if several of the rows share it
			which.emplace_back(l, rep_id);
			feats.push_back(&rep.feat);
		}
	}
	if (which.empty()) {
		return;
	}
	const auto t0 = std::chrono::steady_clock::now();
	// One pass per leaf: a leaf's reps share the predicate half of their feature (see AILeafPredictFeatures).
	for (idx_t l = 0; l < n; l++) {
		vector<const vector<float> *> leaf_feats;
		vector<idx_t> slots;
		for (idx_t k = 0; k < which.size(); k++) {
			if (which[k].first == l) {
				leaf_feats.push_back(feats[k]);
				slots.push_back(k);
			}
		}
		if (leaf_feats.empty()) {
			continue;
		}
		vector<double> p;
		AILeafPredictFeatures(leaf_feats, p);
		for (idx_t k = 0; k < slots.size(); k++) {
			leaves[l].reps[which[slots[k]].second].p = p[k];
		}
	}
	refreshes += which.size();
	t_refresh += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
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
		p[l] = OrderingP(l, rep.p);
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
	// prefix-cache write hint: remembered by row, computed at dispatch (see ReuseHintFor)
	if (has_partner[next] && rep.state == 0 && !rep.has_hint_row) {
		rep.hint_row = row;
		rep.has_hint_row = true;
	}
	if (rep.state == 0) {
		rep.state = 1;
		leaf.pending.push_back(rep_id);
	}
}

uint32_t AILeafRegionState::DistinctKeyOf(DataChunk &chunk, idx_t row) {
	// The tuple's identity under SQL equality, as the DISTINCT above will see it: each column's DuckDB hash
	// (equal values hash equal: -0.0 and 0.0, '30 days' and '1 month'), not its text, which told them apart
	// and over-counted the distinct passers -- the early stop then left the DISTINCT short of k rows. A hash
	// collision can only merge two tuples, which under-counts and asks more: never fewer rows out.
	string key;
	for (const idx_t c : distinct_cols) {
		const Value v = chunk.data[c].GetValue(row);
		key.push_back(v.IsNull() ? '\x00' : '\x01');
		if (!v.IsNull()) {
			const hash_t h = v.Hash();
			key.append(const_char_ptr_cast(&h), sizeof(h));
		}
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

double AILeafRegionState::OrderingP(idx_t l, double p_mlp) const {
	return AILeafOrderingP(p_mlp, observed_true[l], observed_n[l], sum_logit_pask[l]);
}

void AILeafRegionState::ApplyVerdict(idx_t l, uint32_t rep_id, bool value, bool valid) {
	auto &rep = leaves[l].reps[rep_id];
	rep.state = 3;
	rep.valid = rep.valid && valid;
	rep.verdict = value ? 1 : 0;
	landed[l]++;
	if (rep.async_inflight) {
		rep.async_inflight = false;
		async_inflight[l]--;
	}
	if (valid) {
		observed_n[l]++;
		observed_true[l] += value ? 1 : 0;
		sum_logit_pask[l] += AILeafLogit(rep.p);
	}
}

bool AILeafRegionState::Warm() const {
	if (n < 2) {
		return true; // nothing to order
	}
	const idx_t cold_slices = AIConfig::Get().leaf_cold_slices;
	if (cold_slices > 0 && !under_limit && slices_ingested >= cold_slices) {
		return true; // bounded cold phase: a rarely asked leaf would otherwise gate every slice
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

void AILeafRegionState::SubmitRep(idx_t l, uint32_t rep_id) {
	auto &leaf = leaves[l];
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
	rep.async_inflight = true;
	async_inflight[l]++;
	if (rep.has_hint_row) {
		rep.reuse_hint = ReuseHintFor(rep.hint_row, l); // on the verdicts observed by now
	}
	const AILeafUnitEvaluator &eval = *evaluator;
	ClientContext &ctx = context;
	const string &qt = query_text;
	dispatcher->Submit((static_cast<idx_t>(l) << 32) | rep_id,
	                   [&eval, &ctx, &qt, l, prompt = leaf.texts.prompt[rep_id], pred = leaf.texts.pred_text[rep_id],
	                    feat = rep.feat, hint = rep.reuse_hint, lead = leaf.texts.lead_span[rep_id],
	                    head = leaf.texts.head, head_reads = static_cast<double>(leaf.reps.size()) - 1.0]() {
		                   return eval.Evaluate(ctx, l, prompt, pred, feat, qt, hint, &lead, &head, head_reads);
	                   });
}

// SET ai_leaf_lookahead = k: keep at most k x ai_concurrency calls outstanding, and re-decide the rows behind a
// pending rep just before it is submitted, so a row's next leaf is chosen with the estimates of the moment it
// is asked, not of the moment it was queued. Round-robin over the leaves as below. Afterwards either something
// is outstanding or nothing is pending (Finish and DecideThrough rely on it). A rep is re-decided at most once
// per call, so the loop cannot cycle.
void AILeafRegionState::DispatchBounded(idx_t cap) {
	const uint32_t gen = ++dispatch_gen;
	vector<idx_t> head(n, 0);
	auto outstanding = [&]() {
		return dispatcher ? dispatcher->Outstanding() : idx_t(0);
	};
	// As in DispatchPending: a leaf whose reps carry a prefix-cache write hint keeps one concurrency-wide batch
	// in flight until it has that many verdicts, so its hints are computed on observed pass rates.
	const idx_t hint_batch = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
	auto hint_capped = [&](idx_t l) {
		return has_partner[l] && observed_n[l] < hint_batch && async_inflight[l] >= hint_batch;
	};
	for (;;) {
		bool any = false;
		for (idx_t l = 0; l < n && outstanding() < cap; l++) {
			auto &leaf = leaves[l];
			while (head[l] < leaf.pending.size() && !hint_capped(l)) {
				const uint32_t rep_id = leaf.pending[head[l]++];
				auto &rep = leaf.reps[rep_id];
				if (rep.state != 1) {
					continue; // a stale entry: re-decided away, or already submitted
				}
				any = true;
				if (rep.redecided != gen) {
					rep.redecided = gen;
					vector<uint32_t> rows = std::move(rep.waiters);
					rep.waiters.clear();
					rep.state = 0;
					RefreshStalePredictions(rows.data(), rows.size());
					for (const auto row : rows) {
						DecideRow(row); // may finish the row, move it, or choose this rep again (re-queued)
					}
					if (rep.state != 1) {
						break; // nobody chose it now
					}
				}
				SubmitRep(l, rep_id);
				break;
			}
		}
		if (!any || outstanding() >= cap) {
			break;
		}
	}
	for (idx_t l = 0; l < n; l++) {
		auto &pending = leaves[l].pending;
		vector<uint32_t> keep;
		for (idx_t i = head[l]; i < pending.size(); i++) {
			if (leaves[l].reps[pending[i]].state == 1) {
				keep.push_back(pending[i]);
			}
		}
		pending.swap(keep);
	}
}

void AILeafRegionState::DispatchPending(bool flush) {
	if (limit >= 0) {
		return;
	}
	// A region of one leaf has nothing to order, so it dispatches as before.
	const idx_t lookahead = AIConfig::Get().leaf_lookahead;
	if (lookahead > 0 && n >= 2 && (flush || Warm())) {
		DispatchBounded(lookahead * MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1));
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
	// A leaf whose reps carry a prefix-cache write hint goes out one concurrency-wide batch per pass until
	// it has that many verdicts: the hint is computed at submission from the pass rate observed so far, and
	// a whole backlog submitted cold would write every row on the model's neutral guess (measured: 80 rows,
	// a 10% leaf, 80 writes). The pool is that wide anyway, so the first batches lose nothing.
	const idx_t hint_batch = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
	vector<idx_t> cap(n, DConstants::INVALID_INDEX);
	for (idx_t l = 0; l < n; l++) {
		if (ready[l] && has_partner[l] && observed_n[l] < hint_batch) {
			cap[l] = hint_batch > async_inflight[l] ? hint_batch - async_inflight[l] : 0; // the window left
		}
	}
	// Round-robin over the leaves, so a long backlog on one leaf does not queue another leaf's calls behind it.
	for (idx_t i = 0;; i++) {
		bool any = false;
		for (idx_t l = 0; l < n; l++) {
			auto &leaf = leaves[l];
			if (!ready[l] || i >= leaf.pending.size() || i >= cap[l]) {
				continue;
			}
			any = true;
			SubmitRep(l, leaf.pending[i]);
		}
		if (!any) {
			break;
		}
	}
	for (idx_t l = 0; l < n; l++) {
		if (!ready[l]) {
			continue;
		}
		auto &pending = leaves[l].pending;
		if (cap[l] < pending.size()) {
			pending.erase(pending.begin(), pending.begin() + NumericCast<int64_t>(cap[l])); // the rest waits
		} else {
			pending.clear();
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
		// A NULL value is a call that got no answer: the rep is decided but invalid, exactly as the inline
		// wave path records it, so the row's tree evaluates with an unknown there.
		const bool valid = !g.value.IsNull();
		ApplyVerdict(l, rep_id, valid && BooleanValue::Get(g.value), valid);
		auto &rep = leaves[l].reps[rep_id];
		waiters.insert(waiters.end(), rep.waiters.begin(), rep.waiters.end());
		rep.waiters.clear();
		rep.waiters.shrink_to_fit();
	}
	RefreshStalePredictions(waiters.data(), waiters.size());
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
		if (leaf.reps[rep_id].has_hint_row) {
			leaf.reps[rep_id].reuse_hint = ReuseHintFor(leaf.reps[rep_id].hint_row, l);
		}
		leaf.reps[rep_id].state = 2;
		wave.reps.push_back(rep_id);
		wave.texts.prompt.push_back(leaf.texts.prompt[rep_id]);
		wave.texts.pred_text.push_back(leaf.texts.pred_text[rep_id]);
		wave.texts.input_text.push_back(leaf.texts.input_text[rep_id]);
		wave.texts.cost.push_back(leaf.texts.cost[rep_id]);
		wave.texts.valid.push_back(leaf.texts.valid[rep_id]);
		wave.texts.reuse_hint.push_back(leaf.reps[rep_id].reuse_hint);
		wave.texts.lead_span.push_back(leaf.texts.lead_span[rep_id]);
		wave.texts.head = leaf.texts.head;
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
	RefreshStalePredictions(waiters.data(), waiters.size());
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
