#pragma once

#include "filesystem.hpp"

namespace glasssix
{
	namespace utils
	{
		bool safe_remove_file(const fs::path& path) noexcept;
	}
}
