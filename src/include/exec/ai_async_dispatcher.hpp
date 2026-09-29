//===----------------------------------------------------------------------===//
// exec/ai_async_dispatcher.hpp -- asynchronous iteration for the AI region's LLM calls
//
// A persistent pool of workers that evaluates independent units (one distinct input each) and posts every
// result back as soon as it lands. The region's owning thread submits units and reaps results between the
// chunks it ingests, so the request pool is refilled the moment a slot frees and a verdict reaches the rows
// waiting on it without waiting for a batch to complete. Units must be self-contained: a worker touches
// only what its closure owns (copied texts, the bound expression, the client context), never the owner's
// state. The number of calls actually in flight is still capped process-wide by the client's chat gate.
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/error_data.hpp"
#include "duckdb/common/types/value.hpp"

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace duckdb {

class AIAsyncDispatcher {
public:
	using Work = std::function<Value()>;
	struct Landed {
		idx_t key;
		Value value;
	};

	explicit AIAsyncDispatcher(idx_t workers);
	~AIAsyncDispatcher();
	AIAsyncDispatcher(const AIAsyncDispatcher &) = delete;
	AIAsyncDispatcher &operator=(const AIAsyncDispatcher &) = delete;

	//! Queue one unit; returns immediately. Workers start on the first submission.
	void Submit(idx_t key, Work work);
	//! Units submitted and not yet reaped (queued, running, or landed but not reaped).
	idx_t Outstanding() const {
		return outstanding;
	}
	//! Units queued or running, i.e. not yet landed.
	idx_t Unlanded();
	//! Move every landed result into `out` without blocking. Rethrows the first error a unit raised.
	void Reap(vector<Landed> &out);
	//! Block until a result is ready to reap (or nothing is outstanding); returns the seconds waited.
	double WaitAny();

private:
	void WorkerLoop();

	const idx_t worker_count;
	vector<std::thread> threads;
	std::mutex mutex;
	std::condition_variable work_cv;
	std::condition_variable landed_cv;
	std::deque<std::pair<idx_t, Work>> queue;
	vector<Landed> landed;
	idx_t running = 0;
	bool stop = false;
	ErrorData error;
	//! Owner-side count; only the owning thread reads or writes it.
	idx_t outstanding = 0;
};

} // namespace duckdb
