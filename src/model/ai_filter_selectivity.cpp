#include "ai_filter_selectivity.hpp"

#include "ai_selectivity_model.hpp"

namespace duckdb {

double AIEstimateFilterTrueProb(const float *embedding, idx_t dim) {
	return AISelectivityModel::Global().Predict(embedding, dim);
}

} // namespace duckdb
