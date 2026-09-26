#include "ai_prompt_cost.hpp"

#include <cmath>

namespace duckdb {

double AIEstimatePromptCost(const string &prompt) {
	// tiktoken stand-in: English text is ~4 bytes/token; +1 for the single yes/no output token.
	const double input_tokens = std::ceil(static_cast<double>(prompt.size()) / 4.0);
	const double tokens = (input_tokens < 1.0 ? 1.0 : input_tokens) + 1.0;
	return tokens;
}

} // namespace duckdb
