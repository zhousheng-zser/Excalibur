#include "filesystem_utils.hpp"

namespace glasssix
{
	namespace utils
	{
		bool safe_remove_file(const fs::path& path) noexcept
		{
			std::error_code code;

			return fs::exists(path, code) ? fs::remove(path, code) : false;
		}
	}
}
