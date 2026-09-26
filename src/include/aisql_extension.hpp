//===----------------------------------------------------------------------===//
// aisql_extension.hpp — extension declaration for SWAN AI-SQL v2
//===----------------------------------------------------------------------===//
#pragma once

#include "duckdb.hpp"

namespace duckdb {

class AisqlExtension : public Extension {
public:
	void Load(ExtensionLoader &loader) override;
	std::string Name() override;
	std::string Version() const override;
};

} // namespace duckdb
