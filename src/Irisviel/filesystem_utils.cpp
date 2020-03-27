#include "filesystem_utils.hpp"

namespace glasssix
{
	namespace utils
	{
		namespace
		{
			thread_local std::error_code last_error_code;
		}

		bool safe_remove_file(const fs::path& path) noexcept
		{
			return fs::exists(path, last_error_code) ? fs::remove(path, last_error_code) : false;
		}

		bool safe_create_directories(const fs::path& path) noexcept
		{
			return fs::create_directories(path, last_error_code);
		}
	}
}
