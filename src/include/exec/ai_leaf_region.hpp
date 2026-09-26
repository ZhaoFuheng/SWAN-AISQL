//===----------------------------------------------------------------------===//
// exec/ai_leaf_region.hpp -- the leaf-factorized state of an AI region over a folded node
//
// Representation: one dictionary PER LEAF on that leaf's own columns, so the region holds |A| + |B| reps
// where the tuple dictionary would hold |A| x |B| (agent_bench Q19: 9,537 vs 1.68M). A row is a vector of
// rep ids, one per leaf; a rep carries its texts, the model's P(true) and, once asked, its verdict.
// Evaluation: per ROW, the next leaf to ask is chosen by the same DP the per-row batch uses, over the
// row's rep-level predictions and costs; the row then waits on that leaf's rep. Per LEAF, reps that some
// undecided row is waiting on accumulate as `pending`, and a wave fires once they reach the floor -- the
// floor is in prompt currency by construction. A verdict landing on a rep re-decides every row waiting on
// it (tree evaluation over the row's known leaf verdicts), which either finishes the row or moves it onto
// its next leaf. So a leaf is asked exactly for the reps some row still needed: Q17's customers all fail
// and its order leaf is never asked. Single-owner: every method runs on the Sink thread; a wave's LLM
// batch runs on a background thread and touches only its own AILeafWave.
//===----------------------------------------------------------------------===//
#pragma once

#include "ai_dedup.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/types/value.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <chrono>
#include <future>
#include <unordered_map>

namespace duckdb {

class ClientContext;

//! One leaf batch handed to a background thread. `done` MUST stay the last member: members destruct in
//! reverse order, so the future (whose destructor joins the worker) goes first.
struct AILeafWave {
	idx_t leaf = 0;
	vector<uint32_t> reps;
	AILeafTexts texts;
	vector<char> out_result, out_valid;
	std::future<void> done;
};

class AILeafRegionState {
public:
	AILeafRegionState(ClientContext &context, const BoundFunctionExpression &eval_call,
	                  const vector<LogicalType> &child_types, int64_t limit);

	//! Fold a child chunk: per-leaf dedup, texts + predictions for the new reps, and each new row's first
	//! leaf choice. Returns true once the LIMIT is met (the caller stops feeding).
	bool Append(DataChunk &chunk);
	//! Fire a wave for every leaf whose pending reps reach `floor`, keeping up to `overlap` waves in flight
	//! (inline under a LIMIT). Returns true once the LIMIT is met.
	bool Fire(idx_t floor, idx_t overlap);
	//! Settle every wave in flight (before the state changes owner).
	void Drain();
	//! Finish: fire whatever is pending at any size until every row is decided (or the LIMIT is met).
	void Finish(idx_t overlap);
	//! One result per appended row, in append order: BOOLEAN, NULL (a needed leaf had no verdict), or
	//! FALSE for rows left undecided by a LIMIT early-stop.
	void Results(vector<Value> &out) const;

	idx_t RowCount() const {
		return row_result.size();
	}
	int64_t Passed() const {
		return passed;
	}
	idx_t Waves() const {
		return waves;
	}
	vector<idx_t> DistinctPerLeaf() const;
	//! Where the Sink thread's time went (seconds): batched embeds in FlushStage, JIT re-predictions,
	//! row decisions, and blocking on a wave in Fire/Finish. Diagnostic for the [leaf-region] log line.
	string TimingSummary() const;

private:
	struct Rep {
		double p = 0.5;
		uint64_t p_step = 0;   //! model training step the prediction was made at (stale once it advances)
		vector<float> feat;    //! selectivity feature, embedded once at staging; a refresh is a forward pass
		double cost = 1.0;
		bool valid = true;
		int8_t verdict = -1;   //! -1 unknown, 0 false, 1 true (meaningful once state == DONE)
		uint8_t state = 0;     //! 0 idle, 1 pending, 2 in flight, 3 done
		vector<uint32_t> waiters; //! rows waiting on this rep's verdict
	};
	struct Leaf {
		vector<idx_t> key_cols;
		std::unordered_map<string, uint32_t> dict;
		vector<Rep> reps;
		AILeafTexts texts;         //! per rep (index-aligned with `reps`)
		vector<uint32_t> pending;  //! reps some undecided row waits on, not yet fired
		DataChunk stage;           //! rows of reps whose texts are not built yet
		vector<uint32_t> stage_reps;
	};

	void FlushStage(idx_t leaf);
	//! Re-predict a rep's P(true) if the model has trained since it was predicted: a forward pass over the
	//! rep's stored feature, never a request; bounded by reps x training steps, never rows. (Before the
	//! feature was stored, a cold-staged rep's first refresh was its own 2-text embed request on the Sink
	//! thread -- agent_bench Q19: 9,357 of them, ~100s during which no wave could launch.)
	void RefreshPrediction(Leaf &leaf, uint32_t rep_id);
	//! Launch waves for leaves whose pending reps reach `min_pending`, up to the overlap, WITHOUT blocking.
	//! Called while a landed wave's waiters are being re-decided, so the next leaf's calls start as soon as a
	//! batch fills rather than after every waiter is processed.
	void LaunchReady(idx_t min_pending);
	void DecideRow(uint32_t row);
	unique_ptr<AILeafWave> PrepareWave(idx_t leaf);
	void RunWave(AILeafWave &wave);
	void ApplyWave(AILeafWave &wave);
	void DrainOldest();
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
	vector<uint32_t> row_reps;   //! n per row
	vector<int8_t> row_result;   //! -2 undecided, -1 NULL, 0 false, 1 true
	int64_t passed = 0;
	idx_t waves = 0;
	vector<unique_ptr<AILeafWave>> inflight;
	idx_t overlap = 1; //! waves kept in flight (set by Fire/Finish; 1 = inline under a LIMIT)
	//! Sink-thread time accounting (seconds), see TimingSummary.
	double t_embed = 0, t_refresh = 0, t_decide = 0, t_drain = 0;
	idx_t refreshes = 0, drains_blocked = 0;
	std::chrono::steady_clock::time_point t_start = std::chrono::steady_clock::now();
	double t_first_wave = -1; //! seconds from construction to the first wave launch
	idx_t rows_at_first_wave = 0;
	//! Apply every wave that has already landed (no blocking), so their verdicts re-decide rows now.
	void ReapLanded();
};

} // namespace duckdb
