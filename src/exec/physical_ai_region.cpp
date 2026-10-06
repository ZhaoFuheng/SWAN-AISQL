#include "exec/physical_ai_region.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/exception.hpp"
#include "rep/ai_sql_map_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "ai_dedup.hpp"
#include "exec/ai_async_dispatcher.hpp"
#include "exec/ai_leaf_region.hpp"
#include "ai_client.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>

namespace duckdb {

PhysicalAIRegion::PhysicalAIRegion(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                   unique_ptr<Expression> eval_call, idx_t estimated_cardinality, int64_t limit,
                                   vector<idx_t> limit_distinct_cols)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      eval_call(std::move(eval_call)), limit(limit), limit_distinct_cols(std::move(limit_distinct_cols)) {
}

PhysicalAIRegion::~PhysicalAIRegion() {
	// Members of this class (eval_call) are destroyed after this body, base members (sink_state) later still:
	// release the sink state here so every dispatched call has returned before eval_call is freed.
	sink_state.reset();
}

InsertionOrderPreservingMap<string> PhysicalAIRegion::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["__expression__"] = eval_call->ToString();
	if (limit >= 0) {
		result["Limit"] = to_string(limit);
		if (!limit_distinct_cols.empty()) {
			string cols;
			for (const idx_t c : limit_distinct_cols) {
				cols += (cols.empty() ? "#" : ", #") + to_string(c);
			}
			result["Distinct On"] = cols;
		}
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

// Under a pushed LIMIT k the region evaluates inline waves of max(k, ai_concurrency) new distinct reps,
// re-checking the LIMIT after each: a wave fills the request pool once, and the next wave is asked only if
// the passers so far are fewer than k (a 5 x concurrency wave would evaluate 100 reps to find 5 passers).
// Without a LIMIT the scalar path folds 5 x concurrency reps per wave and the node path has no floor:
// every distinct input is dispatched as soon as it is known.
static idx_t AIRegionWaveSize(int64_t limit) {
	const idx_t concurrency = MaxValue<idx_t>(AIConfig::Get().max_concurrency, 1);
	if (limit >= 0) {
		return MaxValue<idx_t>(concurrency, NumericCast<idx_t>(limit));
	}
	return 5 * concurrency;
}

// Streaming dedup state (single-owner). Lives in the local sink state during Sink; moved to the global sink
// state in Combine.
struct AIRegionStreamState {
	//! Node (ai_function_with_embed) path: per-leaf dictionaries + per-row leaf order (see ai_leaf_region.hpp).
	unique_ptr<AILeafRegionState> leaf;
	//! Scalar path: one dictionary on the call's key columns.
	unique_ptr<AISQLMapChunk> map;
	vector<Value> rep_result; //! per rep: result Value (BOOLEAN on the node path, return-type on the scalar path)
	vector<char> rep_decided; //! per rep: has it been evaluated?
	bool is_node = false;     //! folded node (-> `leaf`) vs plain scalar AI call (-> `map`)
	// Diagnostics (ai_debug_log='region').
	idx_t calls = 0;
	idx_t rows_at_first_call = 0;
	//! Scalar path: each distinct input is evaluated on its own, asynchronously (asynchronous iteration). Held
	//! by pointer so moving the state into the global sink state never moves the pool its workers run in.
	unique_ptr<AIAsyncDispatcher> dispatcher;
};

//===--------------------------------------------------------------------===//
// Sink
//===--------------------------------------------------------------------===//
// The operator's own `types` are the child columns plus the one appended result column, so the buffered
// child rows carry every type except the last.
class AIRegionGlobalSinkState : public GlobalSinkState {
public:
	AIRegionGlobalSinkState(ClientContext &context, vector<LogicalType> child_types)
	    : buffer(BufferAllocator::Get(context), std::move(child_types)) {
	}
	mutex lock;
	std::atomic<bool> finalized {false};
	//! No LIMIT: the source emits rows as they are decided (Finalize leaves the calls in flight); with a
	//! LIMIT every row is decided in Finalize and `results` holds the broadcast.
	bool streaming = false;
	ColumnDataCollection buffer;  //! all child rows (child types only), emission order
	vector<Value> results;        //! one broadcast result per buffered row (filled in Finalize), scan order
	vector<idx_t> segment_starts; //! per segment, the offset of its first row in `buffer`
	//! One segment per local sink state that received rows, in Combine order -- the SAME order their
	//! buffers were combined into `buffer`, so per-segment broadcasts concatenate row-aligned. New
	//! DuckDB runs even a ParallelSink()==false pipeline as multiple tasks with their own local
	//! states, so any number of segments can carry rows (the fork's single-owner assumption is dead).
	vector<AIRegionStreamState> segments;
};

class AIRegionLocalSinkState : public LocalSinkState {
public:
	AIRegionLocalSinkState(ClientContext &context, const PhysicalAIRegion &op, const vector<LogicalType> &child_types)
	    : buffer(BufferAllocator::Get(context), child_types) {
		buffer.InitializeAppend(append_state);
		auto &fn = op.eval_call->Cast<BoundFunctionExpression>();
		stream.is_node = AIDedupIsFilterNode(fn);
		if (stream.is_node) {
			stream.leaf = make_uniq<AILeafRegionState>(context, fn, child_types, op.limit, op.limit_distinct_cols);
		} else {
			stream.map = make_uniq<AISQLMapChunk>(context, child_types, AIDedupKeyCols(fn));
		}
	}
	ColumnDataCollection buffer;
	ColumnDataAppendState append_state;
	AIRegionStreamState stream;
};

unique_ptr<GlobalSinkState> PhysicalAIRegion::GetGlobalSinkState(ClientContext &context) const {
	vector<LogicalType> child_types(types.begin(), types.end() - 1);
	return make_uniq<AIRegionGlobalSinkState>(context, std::move(child_types));
}

unique_ptr<LocalSinkState> PhysicalAIRegion::GetLocalSinkState(ExecutionContext &context) const {
	vector<LogicalType> child_types(types.begin(), types.end() - 1);
	return make_uniq<AIRegionLocalSinkState>(context.client, *this, std::move(child_types));
}

// Emit the map's unfired representatives as factorized chunks (rep rows + trailing `__count`) and hand each
// one to `consume(chunk, base)`, where `base` is the rep ordinal of the chunk's first row.
template <class F>
static void AIRegionEmitUnfired(ClientContext &context, AIRegionStreamState &st, F &&consume) {
	auto &map = *st.map;
	while (map.NewDistinctSince() > 0) {
		DataChunk factorized;
		factorized.Initialize(Allocator::Get(context), map.FactorizedTypes());
		const idx_t base = map.FiredCount();
		const idx_t m = map.EmitFactorized(factorized);
		if (m == 0) {
			return;
		}
		if (st.calls == 0) {
			st.rows_at_first_call = map.RowCount();
		}
		st.calls += m;
		consume(factorized, base);
	}
}

// LIMIT path: evaluate every unfired rep inline, one factorized chunk at a time.
static void AIRegionEvaluateInline(const PhysicalAIRegion &op, ClientContext &context, AIRegionStreamState &st) {
	auto &fn = op.eval_call->Cast<BoundFunctionExpression>();
	AIRegionEmitUnfired(context, st, [&](DataChunk &factorized, idx_t base) {
		vector<Value> values;
		AIDedupFireWaveScalarFactorized(context, fn, factorized, values);
		for (idx_t j = 0; j < values.size(); j++) {
			st.rep_decided[base + j] = 1;
			st.rep_result[base + j] = values[j];
		}
	});
}

// Dispatch every unfired rep on its own: the scalar call over a one-row factorized chunk, the same prompt and
// answer the batch would produce, but asked as soon as the input is known.
static void AIRegionDispatch(const PhysicalAIRegion &op, ClientContext &context, AIRegionStreamState &st) {
	if (!st.dispatcher) {
		st.dispatcher = make_uniq<AIAsyncDispatcher>(AIConfig::Get().max_concurrency);
	}
	auto &fn = op.eval_call->Cast<BoundFunctionExpression>();
	// Not const: the closure below copies it, and a const member would make the closure's move constructor
	// copy (which may throw) where the dispatcher's queue expects a nothrow move.
	auto types = st.map->FactorizedTypes();
	AIRegionEmitUnfired(context, st, [&](DataChunk &factorized, idx_t base) {
		for (idx_t j = 0; j < factorized.size(); j++) {
			vector<Value> row;
			row.reserve(factorized.ColumnCount());
			for (idx_t c = 0; c < factorized.ColumnCount(); c++) {
				row.push_back(factorized.data[c].GetValue(j));
			}
			st.dispatcher->Submit(base + j, [&fn, &context, types, row]() {
				DataChunk one;
				one.Initialize(Allocator::DefaultAllocator(), types);
				one.SetChildCardinality(1);
				for (idx_t c = 0; c < row.size(); c++) {
					one.data[c].SetValue(0, row[c]);
				}
				vector<Value> values;
				AIDedupFireWaveScalarFactorized(context, fn, one, values);
				return values[0];
			});
		}
	});
}

// Apply every result that has landed (without blocking, or after waiting for at least one).
static void AIRegionReap(AIRegionStreamState &st, bool wait) {
	if (!st.dispatcher) {
		return;
	}
	if (wait) {
		st.dispatcher->WaitAny();
	}
	vector<AIAsyncDispatcher::Landed> got;
	st.dispatcher->Reap(got);
	for (auto &g : got) {
		st.rep_decided[g.key] = 1;
		st.rep_result[g.key] = std::move(g.value);
	}
}

static void AIRegionDrainAll(AIRegionStreamState &st) {
	while (st.dispatcher && st.dispatcher->Outstanding() > 0) {
		AIRegionReap(st, /*wait=*/true);
	}
}

SinkResultType PhysicalAIRegion::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<AIRegionLocalSinkState>();
	auto &st = lstate.stream;
	if (st.leaf) {
		// Fold first, then buffer exactly the rows folded: a LIMIT met part-way through the chunk (rows landing
		// on decided-TRUE reps) stops the ingest at a slice boundary, and the rows after it must not be
		// buffered, or the emitted rows and the broadcast results would fall out of alignment.
		bool limit_met = false;
		const idx_t ingested = st.leaf->Append(chunk, AIRegionWaveSize(limit), limit_met);
		if (ingested < chunk.size()) {
			chunk.SetCardinalityUnsafe(ingested); // logical count only (the vectors are the pipeline's); FINISHED below
		}
		lstate.buffer.Append(lstate.append_state, chunk);
		if (limit_met) {
			return SinkResultType::FINISHED;
		}
		if (st.leaf->Pump(AIRegionWaveSize(limit))) {
			return SinkResultType::FINISHED;
		}
		return SinkResultType::NEED_MORE_INPUT;
	}
	// Buffer every row for emission, then fold it.
	lstate.buffer.Append(lstate.append_state, chunk);
	st.map->Append(chunk);
	const idx_t distinct = st.map->DistinctCount();
	if (st.rep_result.size() < distinct) {
		st.rep_result.resize(distinct);
		st.rep_decided.resize(distinct, 0);
	}
	if (limit >= 0) {
		if (st.map->NewDistinctSince() >= AIRegionWaveSize(limit)) {
			AIRegionEvaluateInline(*this, context.client, st);
		}
		return SinkResultType::NEED_MORE_INPUT;
	}
	// Asynchronous iteration: apply what has landed, then ask for every new distinct input right away. The
	// client's chat gate keeps ai_concurrency calls in flight; nothing waits for a batch to fill or finish.
	AIRegionReap(st, /*wait=*/false);
	AIRegionDispatch(*this, context.client, st);
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalAIRegion::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<AIRegionGlobalSinkState>();
	auto &lstate = input.local_state.Cast<AIRegionLocalSinkState>();
	// Calls in flight keep running across the change of owner: the pool is held by pointer and its units
	// touch only their own copies, and the state itself is next touched by Finalize / the source, strictly
	// after every Combine. (Under a LIMIT the waves are inline, so nothing is in flight anyway.)
	lock_guard<mutex> guard(gstate.lock);
	const idx_t start = gstate.buffer.Count();
	gstate.buffer.Combine(lstate.buffer);
	// Keep EVERY local state that received rows as its own segment, in this Combine order -- the same
	// order its rows just entered `buffer`. Empty locals contribute nothing.
	const bool has_rows = lstate.stream.leaf ? lstate.stream.leaf->RowCount() > 0
	                                         : (lstate.stream.map && lstate.stream.map->RowCount() > 0);
	if (has_rows) {
		gstate.segments.push_back(std::move(lstate.stream));
		gstate.segment_starts.push_back(start);
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalAIRegion::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                            OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<AIRegionGlobalSinkState>();
	if (limit < 0) {
		// Streaming output: every input is known now, so put every remaining call in flight and let the source
		// hand out rows as their verdicts land, in input order; a consumer that stops early (a LIMIT above a
		// join) ends the query with the queued calls never made.
		for (auto &st : gstate.segments) {
			if (st.leaf) {
				st.leaf->FlushDispatch();
			} else {
				AIRegionDispatch(*this, context, st);
			}
		}
		gstate.streaming = true;
		gstate.finalized.store(true);
		return SinkFinalizeType::READY;
	}
	gstate.results.reserve(gstate.buffer.Count());
	int64_t passed_total = 0;
	std::unordered_set<string> passed_keys; // distinct mode: the union of passing tuples over the segments
	for (auto &st : gstate.segments) {
		if (st.leaf) {
			auto &leaf = *st.leaf;
			// Segments are decided one after the other; a later one is skipped once the LIMIT is met by the
			// earlier ones -- in distinct mode by the UNION of their passing tuples, never their sum.
			const bool met_before =
			    limit_distinct_cols.empty() ? passed_total >= limit : passed_keys.size() >= NumericCast<idx_t>(limit);
			const bool done = limit >= 0 && (leaf.LimitMet() || met_before);
			if (!done) {
				leaf.Finish();
			}
			leaf.Results(gstate.results);
			passed_total += leaf.Passed();
			leaf.MergePassedKeys(passed_keys);
			if (AIConfig::Get().debug_log.find("region") != string::npos) {
				string distinct;
				for (const idx_t d : leaf.DistinctPerLeaf()) {
					distinct += (distinct.empty() ? "" : ",") + std::to_string(d);
				}
				fprintf(stderr, "[leaf-region] leaves=%llu distinct=[%s] rows=%llu passed=%lld limit=%lld %s\n",
				        (unsigned long long)leaf.DistinctPerLeaf().size(), distinct.c_str(),
				        (unsigned long long)leaf.RowCount(), (long long)leaf.Passed(), (long long)limit,
				        leaf.TimingSummary().c_str());
			}
			continue;
		}
		auto &map = *st.map;
		// Scalar path: evaluate every remaining distinct input so its buffered rows get a real result.
		if (limit >= 0) {
			AIRegionEvaluateInline(*this, context, st);
		} else {
			AIRegionDispatch(*this, context, st);
			AIRegionDrainAll(st);
		}
		// Broadcast each rep's answer to this segment's buffered rows (undecided reps only occur on a
		// node+limit early-stop -> false; the LIMIT above already has its k passers).
		const idx_t rows = map.RowCount();
		for (idx_t i = 0; i < rows; i++) {
			const idx_t rep = map.RepOfRow(i);
			gstate.results.push_back(st.rep_decided[rep] ? st.rep_result[rep] : Value::BOOLEAN(false));
		}
		if (AIConfig::Get().debug_log.find("region") != string::npos) {
			fprintf(stderr, "[stream-dedup] distinct=%llu rows=%llu calls=%llu first_call_at_rows=%llu segments=%zu\n",
			        (unsigned long long)map.DistinctCount(), (unsigned long long)rows, (unsigned long long)st.calls,
			        (unsigned long long)st.rows_at_first_call, gstate.segments.size());
		}
	}
	D_ASSERT(gstate.results.size() == gstate.buffer.Count());
	gstate.finalized.store(true);
	return SinkFinalizeType::READY;
}

//===--------------------------------------------------------------------===//
// Source
//===--------------------------------------------------------------------===//
// Single-threaded scan (ParallelSource() == false) so the running row offset stays aligned with the
// broadcast-result order produced by Finalize.
class AIDedupGlobalSourceState : public GlobalSourceState {
public:
	explicit AIDedupGlobalSourceState(const ColumnDataCollection &buffer) {
		buffer.InitializeScan(scan_state);
	}
	idx_t MaxThreads() override {
		return 1;
	}
	ColumnDataScanState scan_state;
	idx_t offset = 0; //! rows emitted so far (= the global row index of the next row to emit)
	// Streaming: the scanned chunk being handed out, and how much of it has gone.
	DataChunk pending;
	bool has_pending = false;
	idx_t pending_offset = 0;
	idx_t segment = 0; //! the segment the next row belongs to (rows go out in segment order)
};

// The debug lines the non-streaming Finalize prints, for the streaming source once everything is out.
static void AIRegionLogSegments(const AIRegionGlobalSinkState &gsink, int64_t limit) {
	if (AIConfig::Get().debug_log.find("region") == string::npos) {
		return;
	}
	for (auto &st : gsink.segments) {
		if (st.leaf) {
			auto &leaf = *st.leaf;
			string distinct;
			for (const idx_t d : leaf.DistinctPerLeaf()) {
				distinct += (distinct.empty() ? "" : ",") + std::to_string(d);
			}
			fprintf(stderr, "[leaf-region] leaves=%llu distinct=[%s] rows=%llu passed=%lld limit=%lld %s\n",
			        (unsigned long long)leaf.DistinctPerLeaf().size(), distinct.c_str(),
			        (unsigned long long)leaf.RowCount(), (long long)leaf.Passed(), (long long)limit,
			        leaf.TimingSummary().c_str());
		} else {
			fprintf(stderr, "[stream-dedup] distinct=%llu rows=%llu calls=%llu first_call_at_rows=%llu segments=%zu\n",
			        (unsigned long long)st.map->DistinctCount(), (unsigned long long)st.map->RowCount(),
			        (unsigned long long)st.calls, (unsigned long long)st.rows_at_first_call, gsink.segments.size());
		}
	}
}

// Streaming: the rows of `st` from local row `from` that are decided, waiting for the first one if needed.
static idx_t AIRegionDecidedPrefix(AIRegionStreamState &st, idx_t from, idx_t to) {
	if (st.leaf) {
		st.leaf->DecideThrough(from);
		return st.leaf->DecidedPrefix(from, to);
	}
	auto &map = *st.map;
	while (!st.rep_decided[map.RepOfRow(from)]) {
		// Finalize dispatched every unfired rep, so an undecided rep is in flight; wait for a verdict.
		if (!st.dispatcher || st.dispatcher->Outstanding() == 0) {
			throw InternalException("AI region: row %llu undecided with nothing in flight", from);
		}
		AIRegionReap(st, /*wait=*/true);
	}
	idx_t n_ready = 0;
	for (idx_t r = from; r < to && st.rep_decided[map.RepOfRow(r)]; r++) {
		n_ready++;
	}
	return n_ready;
}

static Value AIRegionRowValue(const AIRegionStreamState &st, idx_t row) {
	if (st.leaf) {
		return st.leaf->RowValue(row);
	}
	return st.rep_result[st.map->RepOfRow(row)];
}

unique_ptr<GlobalSourceState> PhysicalAIRegion::GetGlobalSourceState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<AIRegionGlobalSinkState>();
	return make_uniq<AIDedupGlobalSourceState>(gsink.buffer);
}

SourceResultType PhysicalAIRegion::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                   OperatorSourceInput &input) const {
	auto &gsink = sink_state->Cast<AIRegionGlobalSinkState>();
	auto &gstate = input.global_state.Cast<AIDedupGlobalSourceState>();

	if (gsink.streaming) {
		// Hand out the decided prefix of the current buffered chunk (input order), scanning the next chunk when
		// the current one is out; wait only when the very next row is still undecided.
		if (!gstate.has_pending) {
			gstate.pending.Destroy();
			gstate.pending.Initialize(Allocator::Get(context.client), gsink.buffer.Types());
			if (!gsink.buffer.Scan(gstate.scan_state, gstate.pending)) {
				AIRegionLogSegments(gsink, limit);
				return SourceResultType::FINISHED;
			}
			gstate.has_pending = true;
			gstate.pending_offset = 0;
		}
		const idx_t global = gstate.offset;
		auto &starts = gsink.segment_starts;
		while (gstate.segment + 1 < starts.size() && starts[gstate.segment + 1] <= global) {
			gstate.segment++;
		}
		const idx_t seg_end = gstate.segment + 1 < starts.size() ? starts[gstate.segment + 1] : gsink.buffer.Count();
		const idx_t local = global - starts[gstate.segment];
		const idx_t avail = MinValue<idx_t>(gstate.pending.size() - gstate.pending_offset, seg_end - global);
		auto &st = gsink.segments[gstate.segment];
		const idx_t ready = AIRegionDecidedPrefix(st, local, local + avail);
		D_ASSERT(ready > 0);
		const idx_t child_cols = gstate.pending.ColumnCount();
		chunk.SetChildCardinality(ready);
		for (idx_t c = 0; c < child_cols; c++) {
			VectorOperations::Copy(gstate.pending.data[c], chunk.data[c], gstate.pending_offset + ready,
			                       gstate.pending_offset, 0);
		}
		auto &result_vec = chunk.data[child_cols];
		for (idx_t i = 0; i < ready; i++) {
			result_vec.SetValue(i, AIRegionRowValue(st, local + i));
		}
		chunk.Verify();
		gstate.pending_offset += ready;
		gstate.offset += ready;
		if (gstate.pending_offset == gstate.pending.size()) {
			gstate.has_pending = false;
		}
		return SourceResultType::HAVE_MORE_OUTPUT;
	}

	// Scan the next buffered chunk of child columns, then append this chunk's broadcast result column.
	DataChunk child_chunk;
	child_chunk.Initialize(Allocator::Get(context.client), gsink.buffer.Types());
	if (!gsink.buffer.Scan(gstate.scan_state, child_chunk)) {
		return SourceResultType::FINISHED;
	}
	const idx_t count = child_chunk.size();
	const idx_t child_cols = child_chunk.ColumnCount();
	// NEW-TREE CONTRACT: vectors carry their own size; size the children BEFORE writing them.
	chunk.SetChildCardinality(count);
	for (idx_t c = 0; c < child_cols; c++) {
		VectorOperations::Copy(child_chunk.data[c], chunk.data[c], count, 0, 0);
	}
	auto &result_vec = chunk.data[child_cols];
	for (idx_t i = 0; i < count; i++) {
		result_vec.SetValue(i, gsink.results[gstate.offset + i]);
	}
	chunk.Verify();
	gstate.offset += count;
	return SourceResultType::HAVE_MORE_OUTPUT;
}

} // namespace duckdb
