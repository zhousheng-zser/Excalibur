#pragma once

#include "filesystem.hpp"

#include <string_view>

namespace glasssix
{
	namespace utils
	{
		bool safe_remove_file(const fs::path& path) noexcept;
		bool safe_create_directories(const fs::path& path) noexcept;
		fs::path path_from_string_view(std::string_view str) noexcept;
	}
}
