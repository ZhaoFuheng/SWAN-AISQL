#include "exec/ai_async_dispatcher.hpp"

#include <chrono>

namespace duckdb {

AIAsyncDispatcher::AIAsyncDispatcher(idx_t workers) : worker_count(MaxValue<idx_t>(workers, 1)) {
}

AIAsyncDispatcher::~AIAsyncDispatcher() {
	{
		std::lock_guard<std::mutex> lock(mutex);
		stop = true;
		queue.clear(); // units not yet started are dropped; running ones finish their call
	}
	work_cv.notify_all();
	for (auto &t : threads) {
		t.join();
	}
}

void AIAsyncDispatcher::Submit(idx_t key, Work work) {
	{
		std::lock_guard<std::mutex> lock(mutex);
		queue.emplace_back(key, std::move(work));
		if (threads.size() < worker_count && threads.size() < queue.size() + running) {
			threads.emplace_back([this]() { WorkerLoop(); });
		}
	}
	outstanding++;
	work_cv.notify_one();
}

idx_t AIAsyncDispatcher::Unlanded() {
	std::lock_guard<std::mutex> lock(mutex);
	return queue.size() + running;
}

void AIAsyncDispatcher::WorkerLoop() {
	for (;;) {
		std::pair<idx_t, Work> unit;
		{
			std::unique_lock<std::mutex> lock(mutex);
			work_cv.wait(lock, [&]() { return stop || !queue.empty(); });
			if (stop) {
				return;
			}
			unit = std::move(queue.front());
			queue.pop_front();
			running++;
		}
		Value value;
		ErrorData failure;
		try {
			value = unit.second();
		} catch (std::exception &ex) {
			failure = ErrorData(ex);
		}
		{
			std::lock_guard<std::mutex> lock(mutex);
			running--;
			if (failure.HasError() && !error.HasError()) {
				error = std::move(failure);
			}
			landed.push_back(Landed {unit.first, std::move(value)});
		}
		landed_cv.notify_all();
	}
}

void AIAsyncDispatcher::Reap(vector<Landed> &out) {
	ErrorData raised;
	{
		std::lock_guard<std::mutex> lock(mutex);
		if (error.HasError()) {
			raised = error;
		}
		for (auto &l : landed) {
			out.push_back(std::move(l));
		}
		outstanding -= landed.size();
		landed.clear();
	}
	if (raised.HasError()) {
		raised.Throw();
	}
}

double AIAsyncDispatcher::WaitAny() {
	const auto t0 = std::chrono::steady_clock::now();
	std::unique_lock<std::mutex> lock(mutex);
	landed_cv.wait(lock, [&]() { return !landed.empty() || error.HasError() || outstanding == 0; });
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

} // namespace duckdb
