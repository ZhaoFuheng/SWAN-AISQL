//===----------------------------------------------------------------------===//
// exec/ai_leaf_region.hpp -- the leaf-factorized state of an AI region over a folded node
//
// Representation: one dictionary PER LEAF on that leaf's own columns, so the region holds |A| + |B| reps
// where the tuple dictionary would hold |A| x |B| (agent_bench Q19: 9,537 vs 1.68M). A row is a vector of
// rep ids, one per leaf; a rep carries its texts, the model's P(true) and, once asked, its verdict.
// Evaluation: per ROW, the next leaf to ask is chosen by the same DP the per-row batch uses, over the
// row's rep-level predictions and costs; the row then waits on that leaf's rep. A rep that some undecided
// row waits on becomes `pending`. Without a LIMIT, pending reps are dispatched one at a time to an
// asynchronous pool (exec/ai_async_dispatcher.hpp) as soon as they are known, and each verdict re-decides
// the rows waiting on it the moment it lands -- no batch to fill, no batch to finish. Under a LIMIT the
// region evaluates floor-sized waves inline instead, so it can stop as soon as k rows have passed. So a
// leaf is asked exactly for the reps some row still needed: Q17's customers all fail and its order leaf
// is never asked. Single-owner: every method runs on the thread feeding the region; dispatched units
// touch only their own copies.
//===----------------------------------------------------------------------===//
#pragma once

#include "ai_dedup.hpp"
#include "exec/ai_async_dispatcher.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace duckdb {

class ClientContext;

//! One inline leaf batch (LIMIT path only).
struct AILeafWave {
	idx_t leaf = 0;
	vector<uint32_t> reps;
	AILeafTexts texts;
	vector<char> out_result, out_valid;
};

class AILeafRegionState {
public:
	//! `limit_distinct_cols`: with a LIMIT that reached the region through a plain DISTINCT, the child columns
	//! the DISTINCT keys on -- the early stop then counts distinct passing tuples over them, not passing rows.
	AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
	                  const vector<LogicalType> &child_types, int64_t limit, vector<idx_t> limit_distinct_cols = {});
	~AILeafRegionState();

	//! Fold a child chunk in slices of 100 rows: each slice is deduped, its new reps get texts + predictions
	//! (one batched embed per leaf), and each new row picks its first leaf. Between slices, landed verdicts
	//! are applied and pending reps dispatched (or, under a LIMIT, floor-sized waves are evaluated inline), so
	//! the first calls start after the first slice's features and later slices are decided by a model already
	//! trained on earlier verdicts. Returns the number of the chunk's rows ingested (whole slices); fewer than
	//! the chunk holds only when the LIMIT was met part-way (`limit_met`), after which the caller stops
	//! feeding and must buffer only the ingested prefix, so the buffered rows and the results stay aligned.
	idx_t Append(DataChunk &chunk, idx_t limit_floor, bool &limit_met);
	//! Apply every landed verdict and dispatch every pending rep, without blocking; under a LIMIT, evaluate
	//! inline waves of at least `limit_floor` reps instead. Returns true once the LIMIT is met.
	bool Pump(idx_t limit_floor);
	//! Settle every dispatched unit (before the state changes owner). Pending reps stay pending.
	void Drain();
	//! Finish: evaluate whatever is still needed until every row is decided (or the LIMIT is met).
	void Finish();
	//! Streaming output (no LIMIT): apply landed verdicts and dispatch every pending rep, without waiting.
	void FlushDispatch();
	//! Streaming output: wait until `row` is decided (verdicts keep landing from the pool meanwhile).
	void DecideThrough(idx_t row);
	//! Streaming output: how many rows from `from` (exclusive end `to`) are decided, without waiting.
	idx_t DecidedPrefix(idx_t from, idx_t to) const;
	//! The result of one decided row (see Results).
	Value RowValue(idx_t row) const;
	//! One result per appended row, in append order: BOOLEAN, NULL (a needed leaf had no verdict), or
	//! FALSE for rows left undecided by a LIMIT early-stop.
	void Results(vector<Value> &out) const;

	idx_t RowCount() const {
		return row_result.size();
	}
	int64_t Passed() const {
		return passed;
	}
	//! The LIMIT is met: k passing rows, or k distinct passing tuples in distinct mode.
	bool LimitMet() const {
		if (limit < 0) {
			return false;
		}
		return distinct_cols.empty() ? passed >= limit : passed_keys.size() >= NumericCast<idx_t>(limit);
	}
	//! Distinct mode: add this segment's passing tuples (their key strings) to `out`.
	void MergePassedKeys(std::unordered_set<string> &out) const;
	//! Calls dispatched (or, under a LIMIT, reps evaluated in inline waves).
	idx_t Dispatched() const {
		return dispatched;
	}
	vector<idx_t> DistinctPerLeaf() const;
	//! Where the owning thread's time went (seconds): batched embeds in FlushStage, JIT re-predictions,
	//! row decisions, the warm gate, and waiting for verdicts. Diagnostic for the [leaf-region] log line.
	string TimingSummary() const;

private:
	struct Rep {
		double p = 0.5;
		uint64_t p_step = 0; //! model training step the prediction was made at (stale once it advances)
		vector<float> feat;  //! selectivity feature, embedded once at staging; a refresh is a forward pass
		double cost = 1.0;
		bool valid = true;
		int8_t verdict = -1;      //! -1 unknown, 0 false, 1 true (meaningful once state == DONE)
		uint8_t state = 0;        //! 0 idle, 1 pending, 2 in flight, 3 done
		vector<uint32_t> waiters; //! rows waiting on this rep's verdict
	};
	struct Leaf {
		vector<idx_t> key_cols;
		std::unordered_map<string, uint32_t> dict;
		vector<Rep> reps;
		AILeafTexts texts;        //! per rep (index-aligned with `reps`)
		vector<uint32_t> pending; //! reps some undecided row waits on, not yet dispatched
		DataChunk stage;          //! rows of reps whose texts are not built yet
		vector<uint32_t> stage_reps;
	};

	bool AppendSlice(DataChunk &chunk, idx_t begin, idx_t end);
	void FlushStage(idx_t leaf);
	//! Re-predict a rep's P(true) if the model has trained since it was predicted: a forward pass over the
	//! rep's stored feature, never a request; bounded by reps x training steps, never rows. The fallback
	//! behind RefreshStalePredictions for a rep that one missed (a step landing between the two).
	void RefreshPrediction(Leaf &leaf, uint32_t rep_id);
	//! Before deciding `rows`, re-predict their open reps whose prediction predates the model's latest
	//! training step, all in ONE blocked forward pass (the same values as one pass per rep). Exactly the reps
	//! the lazy per-rep refresh would have touched, without a lock, allocation and weight-matrix walk each.
	void RefreshStalePredictions(const uint32_t *rows, idx_t count);
	void DecideRow(uint32_t row);
	//! A verdict landed on (leaf, rep): record it, then re-decide every row waiting on it.
	void ApplyVerdict(idx_t leaf, uint32_t rep_id, bool value, bool valid);
	//! Hand pending reps to the dispatcher (round-robin over the leaves): all of them once the model is warm
	//! (or on `flush`), else only leaves with a concurrency-wide batch pending.
	void DispatchPending(bool flush = false);
	//! Every leaf has a batch of verdicts and the model has trained on them (always true for one leaf).
	bool Warm() const;
	static constexpr idx_t GATE_LABELS = 20; //! verdicts per leaf before the model counts as warm
	//! Apply every verdict that has landed, without blocking.
	void ReapLanded();
	//! Wait for at least one verdict, then apply what landed.
	void WaitAndReap();
	//! LIMIT path: evaluate one inline wave of up to `wave_cap` of leaf `l`'s pending reps.
	void RunInlineWave(idx_t l);
	//! Distinct mode: the row's tuple over `distinct_cols`, interned to an id (one string per distinct tuple).
	uint32_t DistinctKeyOf(DataChunk &chunk, idx_t row);

	ClientContext &context;
	const BoundFunctionExpression &call;
	shared_ptr<AIFilterTreeNode> tree;
	const idx_t n;
	const int64_t limit;
	//! LIMIT path: reps per inline wave, max(k, ai_concurrency) (see AIRegionWaveSize): a wave fills the
	//! pool once, and the next wave is asked only if the passers so far are fewer than k.
	const idx_t wave_cap;
	const vector<idx_t> distinct_cols;
	std::unordered_map<string, uint32_t> dkey_dict; //! distinct tuple -> id
	vector<string> dkeys;                           //! id -> distinct tuple
	vector<uint32_t> row_dkey;                      //! per row, its distinct tuple id (distinct mode only)
	std::unordered_set<uint32_t> passed_keys;       //! distinct tuples of the rows decided TRUE
	const string query_text;
	vector<LogicalType> child_types;
	vector<Leaf> leaves;
	vector<uint32_t> row_reps; //! n per row
	vector<int8_t> row_result; //! -2 undecided, -1 NULL, 0 false, 1 true
	int64_t passed = 0;
	idx_t dispatched = 0;
	vector<idx_t> landed; //! verdicts landed per leaf (the warm gate waits for a batch on every leaf)
	//! Owning-thread time accounting (seconds), see TimingSummary.
	double t_embed = 0, t_refresh = 0, t_decide = 0, t_wait = 0, t_warm_gate = 0;
	idx_t refreshes = 0, waits = 0;
	std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
	double t_first_call = -1; //! seconds from construction to the first dispatch
	idx_t rows_at_first_call = 0;
	unique_ptr<AILeafUnitEvaluator> evaluator;
	//! MUST STAY LAST: destroyed first, so its workers are joined before anything they read is freed.
	unique_ptr<AIAsyncDispatcher> dispatcher;
};

} // namespace duckdb
