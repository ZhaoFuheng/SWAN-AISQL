#include "rep/ai_sql_map_chunk.hpp"

#include "duckdb/common/allocator.hpp"
#include "duckdb/main/client_context.hpp"

namespace duckdb {

AISQLMapChunk::AISQLMapChunk(ClientContext &context, vector<LogicalType> row_types_p, vector<idx_t> key_cols_p)
    : allocator(Allocator::Get(context)), row_types(std::move(row_types_p)), key_cols(std::move(key_cols_p)),
      staged(BufferAllocator::Get(context), row_types) {
	stage_chunk.Initialize(allocator, row_types);
	staged.InitializeAppend(staged_append);
}

vector<LogicalType> AISQLMapChunk::FactorizedTypes() const {
	vector<LogicalType> types = row_types;
	types.push_back(LogicalType::BIGINT); // trailing `__count` multiplicity column
	return types;
}

// Materialize unfired reps as one factorized chunk: the rep row columns followed by a BIGINT `__count`.
// The staged scan order is append order == rep-ordinal order, so the j-th emitted row is rep (fired + j) with
// weight counts[fired + j]. At most one chunk's worth is emitted per call -- the unfired set can exceed a
// chunk, since one input chunk contributes up to STANDARD_VECTOR_SIZE new reps on top of whatever sub-floor
// remainder the last wave left. The surplus stays staged and the caller's next wave drains it.
idx_t AISQLMapChunk::EmitFactorized(DataChunk &out) {
	if (stage_chunk.size() > 0) {
		staged.Append(staged_append, stage_chunk);
		stage_chunk.Reset();
	}
	const idx_t base = fired;
	const idx_t ncol = row_types.size();
	const idx_t emit = MinValue<idx_t>(staged.Count() - staged_emitted, STANDARD_VECTOR_SIZE);
	out.Reset();
	out.SetChildCardinality(emit); // sized BEFORE writes (vectors carry size)
	DataChunk rep_chunk;
	rep_chunk.Initialize(allocator, row_types);
	ColumnDataScanState scan;
	staged.InitializeScan(scan);
	idx_t skip = staged_emitted; // reps already emitted by an earlier wave from this same staging buffer
	idx_t filled = 0;
	while (filled < emit && staged.Scan(scan, rep_chunk)) {
		for (idx_t r = 0; r < rep_chunk.size() && filled < emit; r++) {
			if (skip > 0) {
				skip--;
				continue;
			}
			for (idx_t col = 0; col < ncol; col++) {
				out.data[col].SetValue(filled, rep_chunk.data[col].GetValue(r));
			}
			out.data[ncol].SetValue(filled, Value::BIGINT(static_cast<int64_t>(counts[base + filled])));
			filled++;
		}
		rep_chunk.Reset();
	}
	D_ASSERT(filled == emit);
	fired += emit;
	staged_emitted += emit;
	if (staged_emitted == staged.Count()) {
		MarkFired();
	}
	return filled;
}

// Copy `chunk`'s row `row` into the staging buffer; flush to the spillable `staged` collection at capacity so
// a burst of new distinct keys in one Append never overflows a single chunk.
void AISQLMapChunk::StageRow(DataChunk &chunk, idx_t row) {
	const idx_t rc = stage_chunk.size();
	stage_chunk.SetChildCardinality(rc + 1); // sized BEFORE the write (vectors carry size)
	for (idx_t col = 0; col < chunk.ColumnCount(); col++) {
		stage_chunk.data[col].SetValue(rc, chunk.data[col].GetValue(row));
	}
	if (stage_chunk.size() == STANDARD_VECTOR_SIZE) {
		staged.Append(staged_append, stage_chunk);
		stage_chunk.Reset();
	}
}

void AISQLMapChunk::Append(DataChunk &chunk) {
	for (idx_t row = 0; row < chunk.size(); row++) {
		// Serialize the key columns: NULL-flag byte (keeps NULL distinct from "") + value + 0x1f separator.
		string key;
		for (const idx_t kc : key_cols) {
			const Value v = chunk.data[kc].GetValue(row);
			key.push_back(v.IsNull() ? '\x00' : '\x01');
			if (!v.IsNull()) {
				key += v.ToString();
			}
			key.push_back('\x1f');
		}
		auto it = key_to_rep.find(key);
		if (it != key_to_rep.end()) {
			const idx_t rep = it->second; // a fanned-out duplicate: reuse its representative
			row_rep.push_back(rep);
			counts[rep]++;
			continue;
		}
		const idx_t rep = key_to_rep.size();
		key_to_rep.emplace(std::move(key), rep);
		row_rep.push_back(rep);
		counts.push_back(1);
		StageRow(chunk, row);
	}
}

ColumnDataCollection &AISQLMapChunk::UnfiredReps() {
	if (stage_chunk.size() > 0) {
		staged.Append(staged_append, stage_chunk);
		stage_chunk.Reset();
	}
	return staged;
}

void AISQLMapChunk::MarkFired() {
	fired = key_to_rep.size();
	staged.ResetForReuse();
	staged.InitializeAppend(staged_append);
	staged_emitted = 0;
}

} // namespace duckdb
