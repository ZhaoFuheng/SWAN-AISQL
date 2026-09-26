//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/common/types/ai_sql_map_chunk.hpp
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/allocator.hpp"
#include "duckdb/common/types/column/column_data_collection.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/common/vector.hpp"

#include <unordered_map>

namespace duckdb {
class ClientContext;

//! AISQLMapChunk incrementally deduplicates streamed rows by a fixed set of key columns (the columns an AI
//! call reads), modeling DuckDB's DataChunk/ColumnDataCollection. Append() folds each incoming row onto a
//! distinct "representative" (first-seen) ordinal, bumps that rep's fan-out count, and records the row's rep
//! for later broadcast. Only the still-UNFIRED representative rows are retained (staged, spillable); once a
//! wave of AI calls evaluates them they are dropped -- the answer then lives in the operator's per-rep result
//! -- so memory stays O(unfired reps) + O(distinct keys) rather than O(all rows). Single-owner (no locking):
//! the streaming AI-dedup operator drives it under ParallelSink()==false.
class AISQLMapChunk {
public:
	AISQLMapChunk(ClientContext &context, vector<LogicalType> row_types, vector<idx_t> key_cols);

	//! Dedup every row of `chunk` by the key columns: assign each a rep ordinal (staging first-seen rows),
	//! bump the rep's fan-out count, and append the row's rep to the broadcast map.
	void Append(DataChunk &chunk);

	//! Number of distinct representatives seen so far.
	idx_t DistinctCount() const {
		return key_to_rep.size();
	}
	//! Number of rows appended so far (== buffered emission rows).
	idx_t RowCount() const {
		return row_rep.size();
	}
	//! Representatives already handed to a wave (reps [0, FiredCount()) are decided by the operator).
	idx_t FiredCount() const {
		return fired;
	}
	//! New distinct reps since the last MarkFired() -- the wave trigger.
	idx_t NewDistinctSince() const {
		return DistinctCount() - fired;
	}
	//! The representative ordinal of buffered row `row` (for broadcast at emit time).
	idx_t RepOfRow(idx_t row) const {
		return row_rep[row];
	}
	//! The fan-out count of representative `rep` (its output-row weight, snapshot at call time).
	idx_t CountOf(idx_t rep) const {
		return counts[rep];
	}
	//! Flush staging and return the unfired representative rows (reps [FiredCount(), DistinctCount())) in
	//! ordinal order, for one wave of AI evaluation.
	ColumnDataCollection &UnfiredReps();
	//! Mark the currently-staged reps as fired and drop their rows (advance FiredCount to DistinctCount).
	void MarkFired();

	//===------ factorized currency (AISQLMapData) ------===//
	//! The distinct-rep row schema (the columns an AI call reads + carries).
	const vector<LogicalType> &RowTypes() const {
		return row_types;
	}
	//! The factorized-chunk schema: the rep row types plus a trailing BIGINT `__count` multiplicity column.
	vector<LogicalType> FactorizedTypes() const;
	//! True once the unfired distinct reps reach `floor` -- the AI currency's cardinality floor, so each
	//! emitted factorized chunk saturates LLM concurrency (floor == 5 * AI_MAX_CONCURRENCY).
	bool AtFloor(idx_t floor) const {
		return NewDistinctSince() >= floor;
	}
	//! Emit up to one DataChunk of the currently-unfired reps (rep row columns + `__count`), in ordinal order,
	//! advancing FiredCount() by the number emitted. Returns that count, which is < NewDistinctSince() when the
	//! unfired set exceeds a chunk -- call again (or let the next wave) to drain the rest. `out` must have
	//! FactorizedTypes()
	//! (true when firing at the floor, which is << SVS). Returns the number of reps emitted.
	idx_t EmitFactorized(DataChunk &out);

private:
	void StageRow(DataChunk &chunk, idx_t row);

private:
	Allocator &allocator;
	vector<LogicalType> row_types;
	vector<idx_t> key_cols;
	std::unordered_map<string, idx_t> key_to_rep; //! distinct key -> rep ordinal (first-seen)
	vector<idx_t> counts;                          //! per rep: fan-out count
	vector<idx_t> row_rep;                         //! per appended row: rep ordinal (broadcast map)
	ColumnDataCollection staged;                   //! unfired rep rows (spillable), dropped each MarkFired()
	ColumnDataAppendState staged_append;
	DataChunk stage_chunk; //! staging buffer, flushed to `staged` at STANDARD_VECTOR_SIZE
	idx_t fired = 0;
	idx_t staged_emitted = 0; //! reps of `staged` already emitted, when a wave could not fit in one chunk
};

} // namespace duckdb
