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
	AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
	                  const vector<LogicalType> &child_types, int64_t limit);
	~AILeafRegionState();

	//! Fold a child chunk in slices of 100 rows: each slice is deduped, its new reps get texts + predictions
	//! (one batched embed per leaf), and each new row picks its first leaf. Between slices, landed verdicts
	//! are applied and pending reps dispatched (or, under a LIMIT, floor-sized waves are evaluated inline), so
	//! the first calls start after the first slice's features and later slices are decided by a model already
	//! trained on earlier verdicts. Returns true once the LIMIT is met (the caller stops feeding).
	bool Append(DataChunk &chunk, idx_t limit_floor);
	//! Apply every landed verdict and dispatch every pending rep, without blocking; under a LIMIT, evaluate
	//! inline waves of at least `limit_floor` reps instead. Returns true once the LIMIT is met.
	bool Pump(idx_t limit_floor);
	//! Settle every dispatched unit (before the state changes owner). Pending reps stay pending.
	void Drain();
	//! Finish: evaluate whatever is still needed until every row is decided (or the LIMIT is met).
	void Finish();
	//! One result per appended row, in append order: BOOLEAN, NULL (a needed leaf had no verdict), or
	//! FALSE for rows left undecided by a LIMIT early-stop.
	void Results(vector<Value> &out) const;

	idx_t RowCount() const {
		return row_result.size();
	}
	int64_t Passed() const {
		return passed;
	}
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
	//! rep's stored feature, never a request; bounded by reps x training steps, never rows.
	void RefreshPrediction(Leaf &leaf, uint32_t rep_id);
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
	//! LIMIT path: evaluate one inline wave of leaf `l`'s pending reps.
	void RunInlineWave(idx_t l);
	bool LimitMet() const {
		return limit >= 0 && passed >= limit;
	}

	ClientContext &context;
	const BoundFunctionExpression &call;
	shared_ptr<AIFilterTreeNode> tree;
	const idx_t n;
	const int64_t limit;
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
