#include "exec/physical_ai_region.hpp"

#include "duckdb/common/allocator.hpp"
#include "rep/ai_sql_map_chunk.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/value.hpp"
#include "ai_dedup.hpp"
#include "exec/ai_leaf_region.hpp"
#include "ai_client.hpp"
#include "duckdb/common/vector_operations/vector_operations.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/planner/expression/bound_function_expression.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <string>

namespace duckdb {

PhysicalAIRegion::PhysicalAIRegion(PhysicalPlan &physical_plan, vector<LogicalType> types,
                                 unique_ptr<Expression> eval_call, idx_t estimated_cardinality, int64_t limit)
    : PhysicalOperator(physical_plan, PhysicalOperatorType::EXTENSION, std::move(types), estimated_cardinality),
      eval_call(std::move(eval_call)), limit(limit) {
}

InsertionOrderPreservingMap<string> PhysicalAIRegion::ParamsToString() const {
	InsertionOrderPreservingMap<string> result;
	result["__expression__"] = eval_call->ToString();
	if (limit >= 0) {
		result["Limit"] = to_string(limit);
	}
	SetEstimatedCardinality(result, estimated_cardinality);
	return result;
}

// Streaming is the default execution model; SET ai_debug_region_blocking=true falls back to blocking.
static bool AIRegionStreamingEnabled() {
	return AIConfig::Get().region_streaming;
}

// The AISQLMapData cardinality floor: a factorized batch is fired once this many NEW distinct reps have
// accumulated, so every LLM batch saturates concurrency. Default = 5 * AI_MAX_CONCURRENCY (the LLM concurrency,
// default 20 -> floor 100). Overridable via DUCKDB_AI_STREAM_WAVE.
static idx_t AIRegionWaveSize() {
	const auto &config = AIConfig::Get();
	return config.wave_size > 0 ? config.wave_size : 5 * config.max_concurrency;
}

// Waves the region may keep in flight at once (ai_debug_wave_overlap; 1 = synchronous).
static idx_t AIRegionWaveOverlap() {
	return MaxValue<idx_t>(AIConfig::Get().wave_overlap, 1);
}

// One wave's work. PREPARED on the Sink thread (the map is single-owner), EVALUATED anywhere, APPLIED back
// on the Sink thread -- so the only thing that crosses threads is this self-contained unit, and the waves
// that run concurrently own disjoint rep ranges.
struct AIRegionWave {
	DataChunk factorized;   //! the reps this wave owns (rep rows + trailing `__count`)
	idx_t base = 0;         //! rep ordinal of factorized row 0
	idx_t m = 0;            //! reps in this wave
	vector<Value> values;    //! the scalar call's results, one per rep
	string query_text;       //! captured on the Sink thread: ClientContext is not ours to read from a worker
	//! Only set when the wave runs in the background; carries its exception to whoever get()s it.
	//! MUST STAY LAST: members destruct in reverse declaration order, so this future -- whose destructor
	//! waits for the worker -- is destroyed FIRST, before the buffers that worker is still writing into.
	//! That is what makes dropping an un-drained wave (an exception unwinding Sink) safe.
	std::future<void> done;
};

// Streaming dedup state (single-owner). Lives in the local sink state during Sink; moved to the global sink
// state in Combine (ParallelSink()==false guarantees exactly one local state, so the move is not a merge).
struct AIRegionStreamState {
	//! Node (ai_function_with_embed) path: per-leaf dictionaries + per-row leaf order (see ai_leaf_region.hpp).
	unique_ptr<AILeafRegionState> leaf;
	//! Scalar path: one dictionary on the call's key columns.
	unique_ptr<AISQLMapChunk> map;
	vector<Value> rep_result;  //! per rep: result Value (BOOLEAN on the node path, return-type on the scalar path)
	vector<char> rep_decided;  //! per rep: has a wave evaluated it?
	bool is_node = false;      //! folded node (-> `leaf`) vs plain scalar AI call (-> `map`)
	bool active = false;       //! streaming enabled (else the blocking Finalize runs)
	// Diagnostics (ai_debug_log='region'): wave shape + early-teardown signals.
	idx_t waves = 0;
	idx_t rows_at_first_wave = 0;
	// Cardinality-floor validation: every floor-triggered (Sink) wave must carry >= floor distinct reps; only the
	// final Finalize flush (the leftover tail) may be smaller. min_full_wave = smallest Sink wave (0 = none yet).
	idx_t min_full_wave = 0;
	idx_t last_flush_reps = 0;
	//! Waves handed to background threads and not yet applied, oldest first. Drained in Combine, so a
	//! segment never carries a live thread into the global state.
	vector<unique_ptr<AIRegionWave>> inflight;
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
	ColumnDataCollection buffer; //! all child rows (child types only), emission order
	vector<Value> results;       //! one broadcast result per buffered row (filled in Finalize), scan order
	//! One segment per local sink state that received rows, in Combine order -- the SAME order their
	//! buffers were combined into `buffer`, so per-segment broadcasts concatenate row-aligned. New
	//! DuckDB runs even a ParallelSink()==false pipeline as multiple tasks with their own local
	//! states, so any number of segments can carry rows (the fork's single-owner assumption is dead).
	vector<AIRegionStreamState> segments;
};

class AIRegionLocalSinkState : public LocalSinkState {
public:
	AIRegionLocalSinkState(ClientContext &context, const PhysicalAIRegion &op, vector<LogicalType> child_types)
	    : buffer(BufferAllocator::Get(context), child_types) {
		buffer.InitializeAppend(append_state);
		if (AIRegionStreamingEnabled()) {
			auto &fn = op.eval_call->Cast<BoundFunctionExpression>();
			stream.active = true;
			stream.is_node = AIDedupIsFilterNode(fn);
			if (stream.is_node) {
				stream.leaf = make_uniq<AILeafRegionState>(context, fn, child_types, op.limit);
			} else {
				stream.map = make_uniq<AISQLMapChunk>(context, child_types, AIDedupKeyCols(fn));
			}
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

// Take ownership of the map's currently-unfired representatives as one wave. Sink-thread only: it mutates the
// map and the per-segment diagnostics. Returns nullptr when there is nothing to fire.
// `sink_wave` = fired from Sink because >= floor new reps accumulated (must carry >= floor reps); false = the
// Finalize flush of the leftover tail (may be < floor). Split so the floor invariant is observable/verifiable.
static unique_ptr<AIRegionWave> AIRegionPrepareWave(ClientContext &context, AIRegionStreamState &st,
                                                    bool sink_wave) {
	auto &map = *st.map;
	auto wave = make_uniq<AIRegionWave>();
	wave->base = map.FiredCount();
	// Emit the unfired reps as an explicit factorized chunk (rep rows + trailing `__count`) -- the AISQLMapData
	// currency the AI function consumes. EmitFactorized() advances FiredCount() by what it emitted.
	wave->factorized.Initialize(Allocator::Get(context), map.FactorizedTypes());
	wave->m = map.EmitFactorized(wave->factorized);
	if (wave->m == 0) {
		return nullptr;
	}
	if (st.waves == 0) {
		st.rows_at_first_wave = map.RowCount();
	}
	st.waves++;
	if (sink_wave) {
		st.min_full_wave = st.min_full_wave == 0 ? wave->m : std::min(st.min_full_wave, wave->m);
	} else {
		st.last_flush_reps = wave->m;
	}
	wave->query_text = context.GetCurrentQuery();
	return wave;
}

// Evaluate a prepared wave. Runs on the Sink thread or a background thread; touches only `wave`.
static void AIRegionRunWave(const PhysicalAIRegion &op, ClientContext &context, AIRegionWave &wave) {
	auto &fn = op.eval_call->Cast<BoundFunctionExpression>();
	AIDedupFireWaveScalarFactorized(context, fn, wave.factorized, wave.values);
}

// Fold a finished wave's results back into the segment. Sink-thread only.
static void AIRegionApplyWave(AIRegionStreamState &st, AIRegionWave &wave) {
	for (idx_t j = 0; j < wave.m; j++) {
		st.rep_decided[wave.base + j] = 1;
		st.rep_result[wave.base + j] = wave.values[j];
	}
}

// Prepare + evaluate + apply, all inline. The LIMIT path and the Finalize tail use this.
static void AIRegionFireOneWave(const PhysicalAIRegion &op, ClientContext &context, AIRegionStreamState &st,
                                bool sink_wave) {
	auto wave = AIRegionPrepareWave(context, st, sink_wave);
	if (!wave) {
		return;
	}
	AIRegionRunWave(op, context, *wave);
	AIRegionApplyWave(st, *wave);
}

// Wait for the oldest in-flight wave and apply it. get() rethrows whatever the wave threw, on this thread.
static void AIRegionDrainOldest(const PhysicalAIRegion &op, AIRegionStreamState &st) {
	auto wave = std::move(st.inflight.front());
	st.inflight.erase(st.inflight.begin());
	wave->done.get();
	AIRegionApplyWave(st, *wave);
}

static void AIRegionDrainAll(const PhysicalAIRegion &op, AIRegionStreamState &st) {
	while (!st.inflight.empty()) {
		AIRegionDrainOldest(op, st);
	}
}

SinkResultType PhysicalAIRegion::Sink(ExecutionContext &context, DataChunk &chunk, OperatorSinkInput &input) const {
	auto &lstate = input.local_state.Cast<AIRegionLocalSinkState>();
	auto &st = lstate.stream;
	if (!st.active) {
		lstate.buffer.Append(lstate.append_state, chunk); // blocking path (kill-switch)
		return SinkResultType::NEED_MORE_INPUT;
	}
	// Buffer every row for emission, then fold it.
	lstate.buffer.Append(lstate.append_state, chunk);
	if (st.leaf) {
		if (st.leaf->Append(chunk, AIRegionWaveSize(), limit >= 0 ? 1 : AIRegionWaveOverlap())) {
			return SinkResultType::FINISHED; // LIMIT met while folding (rows landing on decided-TRUE reps)
		}
		if (st.leaf->Fire(AIRegionWaveSize(), limit >= 0 ? 1 : AIRegionWaveOverlap())) {
			return SinkResultType::FINISHED;
		}
		return SinkResultType::NEED_MORE_INPUT;
	}
	const idx_t prev_rows = st.map->RowCount();
	st.map->Append(chunk);
	const idx_t distinct = st.map->DistinctCount();
	if (st.rep_result.size() < distinct) {
		st.rep_result.resize(distinct);
		st.rep_decided.resize(distinct, 0);
	}
	// Fire a wave whenever enough new distinct reps have accumulated; early-stop once the LIMIT is met.
	const idx_t wave = AIRegionWaveSize();
	// A wave blocks Sink, and its LLM batch is only as wide as the DISTINCT PROMPTS its reps carry -- which,
	// above a join that repeats one side across a chunk, can be one. Overlapping waves keeps the request pool
	// fed from the reps already in hand instead of speculating on calls the tree may never need. Under a
	// pushed LIMIT the floor's job is the opposite one (stop early), so that path stays inline.
	const idx_t overlap = (limit >= 0) ? 1 : AIRegionWaveOverlap();
	while (st.map->NewDistinctSince() >= wave) {
		if (overlap == 1) {
			AIRegionFireOneWave(*this, context.client, st, /*sink_wave=*/true);
			continue;
		}
		auto in_flight = AIRegionPrepareWave(context.client, st, /*sink_wave=*/true);
		if (!in_flight) {
			break;
		}
		auto &op = *this;
		auto &client = context.client;
		auto *raw = in_flight.get();
		raw->done = std::async(std::launch::async, [&op, &client, raw]() { AIRegionRunWave(op, client, *raw); });
		st.inflight.push_back(std::move(in_flight));
		while (st.inflight.size() >= overlap) {
			AIRegionDrainOldest(*this, st);
		}
	}
	return SinkResultType::NEED_MORE_INPUT;
}

SinkCombineResultType PhysicalAIRegion::Combine(ExecutionContext &context, OperatorSinkCombineInput &input) const {
	auto &gstate = input.global_state.Cast<AIRegionGlobalSinkState>();
	auto &lstate = input.local_state.Cast<AIRegionLocalSinkState>();
	// Settle every background wave before the segment leaves this thread's ownership.
	if (lstate.stream.leaf) {
		lstate.stream.leaf->Drain();
	} else {
		AIRegionDrainAll(*this, lstate.stream);
	}
	lock_guard<mutex> guard(gstate.lock);
	gstate.buffer.Combine(lstate.buffer);
	// Keep EVERY local state that received rows as its own segment, in this Combine order -- the same
	// order its rows just entered `buffer`. Empty locals contribute nothing.
	const bool has_rows = lstate.stream.leaf ? lstate.stream.leaf->RowCount() > 0
	                                         : (lstate.stream.map && lstate.stream.map->RowCount() > 0);
	if (lstate.stream.active && has_rows) {
		gstate.segments.push_back(std::move(lstate.stream));
	}
	return SinkCombineResultType::FINISHED;
}

SinkFinalizeType PhysicalAIRegion::Finalize(Pipeline &pipeline, Event &event, ClientContext &context,
                                           OperatorSinkFinalizeInput &input) const {
	auto &gstate = input.global_state.Cast<AIRegionGlobalSinkState>();
	if (gstate.segments.empty()) {
		// Blocking path: one global, deduplicated, fully-concurrent evaluation over every buffered row.
		AIDedupEvaluate(context, eval_call->Cast<BoundFunctionExpression>(), gstate.buffer, gstate.results, limit);
		gstate.finalized.store(true);
		return SinkFinalizeType::READY;
	}
	gstate.results.reserve(gstate.buffer.Count());
	int64_t passed_total = 0;
	for (auto &st : gstate.segments) {
		if (st.leaf) {
			auto &leaf = *st.leaf;
			const bool done = limit >= 0 && (leaf.Passed() >= limit || passed_total >= limit);
			if (!done) {
				leaf.Finish(limit >= 0 ? 1 : AIRegionWaveOverlap());
			}
			leaf.Results(gstate.results);
			passed_total += leaf.Passed();
			if (AIConfig::Get().debug_log.find("region") != string::npos) {
				string distinct;
				for (const idx_t d : leaf.DistinctPerLeaf()) {
					distinct += (distinct.empty() ? "" : ",") + std::to_string(d);
				}
				fprintf(stderr,
				        "[leaf-region] leaves=%llu distinct=[%s] rows=%llu waves=%llu passed=%lld limit=%lld %s\n",
				        (unsigned long long)leaf.DistinctPerLeaf().size(), distinct.c_str(),
				        (unsigned long long)leaf.RowCount(), (unsigned long long)leaf.Waves(),
				        (long long)leaf.Passed(), (long long)limit, leaf.TimingSummary().c_str());
			}
			continue;
		}
		auto &map = *st.map;
		// Scalar path: evaluate every remaining distinct input so its buffered rows get a real result.
		while (map.NewDistinctSince() > 0) {
			AIRegionFireOneWave(*this, context, st, /*sink_wave=*/false);
		}
		// Broadcast each rep's answer to this segment's buffered rows (undecided reps only occur on a
		// node+limit early-stop -> false; the LIMIT above already has its k passers).
		const idx_t rows = map.RowCount();
		for (idx_t i = 0; i < rows; i++) {
			const idx_t rep = map.RepOfRow(i);
			gstate.results.push_back(st.rep_decided[rep] ? st.rep_result[rep] : Value::BOOLEAN(false));
		}
		if (AIConfig::Get().debug_log.find("region") != string::npos) {
			fprintf(stderr,
			        "[stream-dedup] distinct=%llu rows=%llu waves=%llu first_wave_at_rows=%llu "
			        "floor=%llu min_full_wave=%llu last_flush=%llu segments=%zu\n",
			        (unsigned long long)map.DistinctCount(), (unsigned long long)rows,
			        (unsigned long long)st.waves, (unsigned long long)st.rows_at_first_wave,
			        (unsigned long long)AIRegionWaveSize(), (unsigned long long)st.min_full_wave,
			        (unsigned long long)st.last_flush_reps, gstate.segments.size());
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
	idx_t offset = 0;
};

unique_ptr<GlobalSourceState> PhysicalAIRegion::GetGlobalSourceState(ClientContext &context) const {
	auto &gsink = sink_state->Cast<AIRegionGlobalSinkState>();
	return make_uniq<AIDedupGlobalSourceState>(gsink.buffer);
}

SourceResultType PhysicalAIRegion::GetDataInternal(ExecutionContext &context, DataChunk &chunk,
                                                  OperatorSourceInput &input) const {
	auto &gsink = sink_state->Cast<AIRegionGlobalSinkState>();
	auto &gstate = input.global_state.Cast<AIDedupGlobalSourceState>();

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
