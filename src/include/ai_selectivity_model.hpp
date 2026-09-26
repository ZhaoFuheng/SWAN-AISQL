//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/ai_selectivity_model.hpp
//
// Online-trained selectivity model for ai_filter reordering. A small 3-layer MLP maps a prompt
// embedding to P(ai_filter returns true). Weights train online in mini-batches drawn at random from
// a fixed-size FIFO of self-labeled (embedding, label) pairs collected from real ai_filter calls; a
// training step is built to overlap the blocking remote LLM batch call (LaunchOverlappedTraining).
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <random>
#include <thread>

namespace duckdb {

//! Immutable weights for the 3-layer MLP
//!   Linear(d->H1) -> ReLU -> Linear(H1->H2) -> ReLU -> Linear(H2->1) -> sigmoid.
//! Held behind a shared_ptr and swapped copy-on-write so Forward() reads never block training.
struct AISelectivityParams {
	idx_t input_dim = 0;
	idx_t h1 = 0;
	idx_t h2 = 0;
	vector<float> w1, b1; // w1: [h1 x input_dim] row-major, b1: [h1]
	vector<float> w2, b2; // w2: [h2 x h1] row-major, b2: [h2]
	vector<float> w3, b3; // w3: [1 x h2], b3: [1]

	//! Forward pass -> P(true) in (0, 1). Returns 0.5 if `dim` does not match input_dim.
	double Forward(const float *embedding, idx_t dim) const;
};

//! Snapshot of the model's training counters, for ai_selectivity_stats().
struct AISelectivityStats {
	idx_t input_dim = 0;
	idx_t buffered = 0;
	uint64_t examples_seen = 0;
	uint64_t train_steps = 0;
	double first_loss = -1.0;
	double last_loss = -1.0;
};

//! Process-global online selectivity model. Thread-safe: Predict() reads the current weights
//! lock-free (copy-on-write), AddExample()/TrainMiniBatchStep() are internally synchronized, and at
//! most one background training step runs at a time.
class AISelectivityModel {
public:
	static AISelectivityModel &Global();

	//! P(ai_filter true) for a row's prompt embedding. Lazily inits weights on first use for `dim`.
	//! Returns 0.5 when `dim == 0`.
	double Predict(const float *embedding, idx_t dim);

	//! Push a (embedding, label) pair into the fixed-size FIFO (capacity 64; oldest evicted).
	void AddExample(const vector<float> &embedding, bool label);

	//! One Adam step on the full FIFO (full-batch gradient), BCE loss. No-op until a minimum number
	//! of examples are buffered.
	void TrainMiniBatchStep();

	//! If training is worthwhile (buffer full enough) and none is in flight, run one step on a
	//! background thread and return the joinable thread; the caller joins it after the overlapped
	//! LLM batch call. Otherwise returns a non-joinable std::thread{}. Bounds trainers to one.
	std::thread LaunchOverlappedTraining();

	//! Run one training step inline on the calling thread if no other trainer is in flight
	//! (single-flighted). Used by the worker pool to train periodically without a dedicated thread.
	void TrainInlineIfReady();

	AISelectivityStats Stats();
	//! Clear weights, FIFO, and counters. Returns the number of buffered examples cleared.
	uint64_t Reset();

private:
	AISelectivityModel() = default;

	std::shared_ptr<const AISelectivityParams> GetParams();
	//! Return params matching `dim`, (re)initializing random weights under params_mutex if needed.
	std::shared_ptr<const AISelectivityParams> EnsureParams(idx_t dim);

	std::mutex params_mutex;
	std::shared_ptr<const AISelectivityParams> params; // current weights (copy-on-write)
	std::mt19937 init_rng {0x5EED1234u};               // weight init (guarded by params_mutex)

	struct Sample {
		vector<float> emb;
		float label;
	};
	std::mutex buffer_mutex;
	std::deque<Sample> buffer;         // fixed-size FIFO of the most recent examples
	uint64_t examples_seen = 0;        // total AddExample calls
	uint64_t train_steps = 0;          // total completed mini-batch steps
	double first_loss = -1.0;          // BCE of the first step (-1 = none yet)
	double last_loss = -1.0;           // BCE of the most recent step

	std::atomic<bool> training_in_flight {false};

	// Adam optimizer state: per-weight first (m) and second (v) moments + step count. Persists across
	// steps; guarded by params_mutex (only the single in-flight trainer and Reset touch it).
	struct AdamState {
		vector<double> m_w1, v_w1, m_b1, v_b1;
		vector<double> m_w2, v_w2, m_b2, v_b2;
		vector<double> m_w3, v_w3, m_b3, v_b3;
		idx_t dim = 0, h1 = 0, h2 = 0;
		uint64_t t = 0;
	};
	AdamState adam;
};

} // namespace duckdb
