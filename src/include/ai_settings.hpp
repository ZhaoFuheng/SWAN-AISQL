//===----------------------------------------------------------------------===//
// ai_settings.hpp — typed accessors for the extension's settings
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb/main/client_context.hpp"

namespace duckdb {

inline bool AIBoolSetting(ClientContext &context, const char *name, bool def) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		return BooleanValue::Get(v.DefaultCastAs(LogicalType::BOOLEAN));
	}
	return def;
}

inline string AIVarcharSetting(ClientContext &context, const char *name, const string &def) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		return StringValue::Get(v.DefaultCastAs(LogicalType::VARCHAR));
	}
	return def;
}

inline double AIDoubleSetting(ClientContext &context, const char *name, double def) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		return DoubleValue::Get(v.DefaultCastAs(LogicalType::DOUBLE));
	}
	return def;
}

inline uint64_t AIUBigintSetting(ClientContext &context, const char *name, uint64_t def) {
	Value v;
	if (context.TryGetCurrentSetting(name, v) && !v.IsNull()) {
		return UBigIntValue::Get(v.DefaultCastAs(LogicalType::UBIGINT));
	}
	return def;
}

} // namespace duckdb
